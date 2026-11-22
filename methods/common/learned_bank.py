"""M8 learned-method synthetic-bank infrastructure (E03 STDN, E05 PCGAN, E06b DSDG-NATIVE, E06c DSDG-BIN-IDFREE).

Scope. A new module, separate from the hash-pinned E01/E02 writer `methods/common/bank_writer.py`, which stays
byte-unchanged. It holds the framework-free parts of learned-bank generation under the owner-approved contract
`configs/amendments/learned_bank_generation_contract.yaml` (d0b4e4e): authority gates, seed derivations, population
construction, the TRAIN-only O_NOATIME face reader, the create-only bank writer, the bank reader and the bitwise
run comparison. The four framework-specific inference adapters live next to their methods:
`methods/stdn/bank_adapter.py` (TF1), `methods/pcgan/bank_adapter.py` and `methods/dsdg/bank_adapter.py` (PyTorch).
No adapter can train: none constructs an optimizer, computes a gradient or writes a checkpoint.

Python 3.8 compatible (the E03 TF1 environment). Imports only the standard library, NumPy and Pillow; never torch or
TensorFlow.

Modes, each with its own gate (no mode can stand in for another):
  FIXTURE     synthetic inputs and fixture (random-init) weights only; never a benchmark artifact. Needs only the
              approved generation contract.
  S7_TOY      the real toy acceptance (C-Q3): real checkpoints, the 186 toy pairs / stream indices 0..185. Needs the
              approved contract, the E04 seed-2026 completion audit PASS, the E04 family closure (or documented roadmap
              authority) and an owner-approved S7 launch authority record. None of the last three exists at d0b4e4e,
              so S7 refuses (C-Q5).
  PRODUCTION  the full 8838-row bank. Additionally needs a per-method owner production authority that binds a passed
              S7 acceptance record, and an activated, live-verified storage authority. None exists, so production
              refuses.

Bank layout (`run_dir`, created by the writer and never pre-existing):
  bank/<row_id>.png            canonical 256x256 RGB uint8, PNG (compress_level 6, optimize False), mode 0444
  bank_manifest.jsonl          one canonical JSON object per row, deterministic (no time, host or run id)
  bank_manifest.sha256         "<sha256>  bank_manifest.jsonl"
  bank_provenance.json         run-specific record (authorization, identities, environment, timings, accounting)
Two runs that satisfy C-Q2 have byte-identical `bank/` files and `bank_manifest.jsonl`.
"""
from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import os
import platform
import re
import stat
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 'gpat_learned_bank_v1'
METHODS = ('E03', 'E05', 'E06b', 'E06c')
PAIR_METHODS = ('E03', 'E05')
DSDG_METHODS = ('E06b', 'E06c')
SEEDS = (42, 1337, 2026)
N_SYN = 8838
TOY_N = 186
DSDG_BATCH = 100
DSDG_HDIM = 128
CANONICAL = 256
PNG_COMPRESS_LEVEL = 6
MODES = ('FIXTURE', 'S7_TOY', 'PRODUCTION')
POPULATION_KINDS = {'FIXTURE': 'SYNTHETIC_FIXTURE', 'S7_TOY': 'S7_TOY_POPULATION',
                    'PRODUCTION': 'PRODUCTION_FROZEN_TRAIN_POPULATION'}
PAIR_ID_RE = re.compile(r'^PTR\d{6}$')
SYNTHETIC_ID_RE = re.compile(r'^(E06b|E06c)_S(42|1337|2026)_(\d{6})$')
FIXTURE_SYNTHETIC_ID_RE = re.compile(r'^(E06b|E06c)_S(\d+)_(\d{6})$')

# ------------------------------------------------------------------ governance
CONTRACT_PATH = 'configs/amendments/learned_bank_generation_contract.yaml'
CONTRACT_SCHEMA = 'learned_bank_generation_contract_v1'
DRAFT_PATH = 'configs/amendments/learned_bank_generation_contract.DRAFT.yaml'
DRAFT_SHA256 = '217e5145a5a09349f3191e42f3990c7d3f5f7bd4931829c09615a3fb630c1aad'
APPROVED_DECISIONS = {
    'E03-Q1_inference_mode': 'A_OFFICIAL_TEST_MODE', 'E03-Q2_weight_source': 'A_RAW_VARIABLES',
    'E03-Q3_inference_augmentation': 'A_NONE', 'E03-Q5_auxiliary_outputs': 'A_NONE',
    'E05-Q1_stylegan_noise': 'A_PER_PAIR_SEEDED', 'DSDG-Q1_latent_stream': 'A_OFFICIAL_BATCHES_SEEDED',
    'DSDG-Q2_output_half': 'A_SPOOF_ONLY', 'DSDG-Q4_bank_identity': 'A_SYNTHETIC_ID', 'E06b-Q0_n_syn': 'A_POOLED_8838',
    'Q4_float_to_uint8': 'A_SOURCE_IMAGE_WRITER', 'C-Q1_encoding': 'A_PNG_LOSSLESS',
    'C-Q2_device_and_reproducibility': 'A_GPU_FP32_BITWISE_REPLAY',
    'C-Q3_s7_bank_stage_acceptance': 'A_TOY_TWO_RUN_BITWISE', 'C-Q5_gpu_timing': 'A_AFTER_E04_COMPLETES'}
S7_AUTHORITY_PATH = 'configs/amendments/learned_bank_s7_launch_authority.yaml'
S7_AUTHORITY_SCHEMA = 'learned_bank_s7_launch_authority_v1'
PRODUCTION_AUTHORITY_TEMPLATE = 'configs/amendments/learned_bank_production_{method}.yaml'
PRODUCTION_AUTHORITY_SCHEMA = 'learned_bank_production_authority_v1'
STORAGE_AUTHORITY_PATH = 'configs/amendments/learned_bank_storage_location.yaml'
STORAGE_AUTHORITY_SCHEMA = 'learned_bank_storage_location_v1'
STORAGE_PENDING_PATH = 'configs/amendments/learned_bank_storage_location.PENDING_ACTIVATION.yaml'
E04_RUN_ID = 'e7100eb9a5ab0537'
E04_COMPLETION_RECORD = 'outputs/audit/E04_SEED2026_COMPLETION.json'
E04_COMPLETION_STATUS = 'E04_SEED2026_COMPLETED_CLEANLY'
E04_FAMILY_CLOSURE_RECORD = 'outputs/audit/E04_THREE_SEED_TRAINING_CLOSURE.json'
E04_FAMILY_CLOSURE_STATUS = 'SCIENTIFIC_TRAINING_COMPLETED_3_OF_3_SEEDS'

STATUS_SCIENTIFIC = 'LEARNED_BANK_BLOCKED_BY_SCIENTIFIC_AUTHORITY'
STATUS_E04 = 'LEARNED_BANK_BLOCKED_BY_E04_COMPLETION'
STATUS_S7 = 'LEARNED_BANK_BLOCKED_BY_S7_LAUNCH_AUTHORITY'
STATUS_PRODUCTION = 'LEARNED_BANK_BLOCKED_BY_PRODUCTION_AUTHORITY'
STATUS_STORAGE = 'LEARNED_BANK_BLOCKED_BY_STORAGE_AUTHORITY'
STATUS_CHECKPOINT = 'LEARNED_BANK_BLOCKED_BY_CHECKPOINT_IDENTITY'

# ------------------------------------------------------------------ frozen inputs
PAIRS_PATH = 'manifests/pairs_train_v1.parquet'
PAIRS_SHA256 = 'a5e4fdaef236f15730c7e3885b537e08e995faffbe654167fc940f44bbc75243'
SPLIT_PATH = 'manifests/split_v1.parquet'
SPLIT_SHA256 = 'fb9aeb369a124fc96ba855ef2ce269236c4a743fe960e73ab739412c9cb5092d'
M2_ACCOUNTING_PATH = 'manifests/m2_sample_accounting.parquet'
M2_ACCOUNTING_SHA256 = '65bb83e69ea7cebcb37373971e5e8f5248609b7963fd9d32aa1c0ba4377c0662'
TOY_MANIFEST_PATH = 'outputs/audit/M8_SMOKE_EVIDENCE/run_a/select/toy_manifest.json'
TOY_MANIFEST_SHA256 = '50ac8fb8b9fc6938ba630d0853523d358250137c8fbbc62704d2a73830f7f690'
DATASETS = ('casia_fasd', 'msu_mfsd', 'siwmv2')

METHOD_CONFIGS = {
    'E03': ('configs/methods/e03_stdn.yaml', '08dde851bc9e8ac688c794a6a6999204a7e5b75fb244fcd97e46b22dcb302fe0'),
    'E05': ('configs/methods/e05_pcgan.yaml', '478756e150c427832315800acba954cedf778bac520eee3bde8cabf7359e71de'),
    'E06b': ('configs/methods/e06b_dsdg_native.yaml', '9d665dc2c909d421b8e54964407e27bb40d133f11bceec2e2cb29810f268417f'),
    'E06c': ('configs/methods/e06c_dsdg_bin_idfree.yaml',
             '7176dd4cd49007320ab0c44519e7efde60568098503077303218bd913b6b5002'),
}
ENVIRONMENTS = {
    'E03': {'name': 'gpat-m6-e03', 'python': '/home/student20261/miniconda3/envs/gpat-m6-e03/bin/python',
            'lock': ('environments/e03_stdn.lock.json',
                     'b2c6c2eadca1b93a0bdf6798990821cb889e199643a5bdb5368ed0b12fdc8c62')},
    'E05': {'name': 'gpat-m6-e05', 'python': '/home/student20261/miniconda3/envs/gpat-m6-e05/bin/python',
            'lock': ('environments/e05.lock.json', '849a100430e36048acde80d58801ae8908c15ae43719e97f4dcb90d77564ac50')},
    'E06b': {'name': 'gpat-m6-e06c', 'python': '/home/student20261/miniconda3/envs/gpat-m6-e06c/bin/python',
             'lock': ('environments/e06c.lock.json', '91416a20fef6eb4bbe550dc0ccdc703163f51d8df9168c1418f7a2de48e64e95')},
    'E06c': {'name': 'gpat-m6-e06c', 'python': '/home/student20261/miniconda3/envs/gpat-m6-e06c/bin/python',
             'lock': ('environments/e06c.lock.json', '91416a20fef6eb4bbe550dc0ccdc703163f51d8df9168c1418f7a2de48e64e95')},
}

# The twelve approved final checkpoints (outputs/audit/LEARNED_BANK_AUTHORITY_ASSESSMENT.json, 3433ad0; re-hashed on
# the host 2026-11-22T22:01Z). Full SHA256 only; nothing is inferred from a prefix.
CHECKPOINT_RULES = {'E03': 'OFFICIAL_LATEST_FINAL_CKPT_50', 'E05': 'BASELINE_FINAL_STATE_V1',
                    'E06b': 'OFFICIAL_GENERATOR_EPOCH_200', 'E06c': 'OFFICIAL_GENERATOR_EPOCH_200'}
CHECKPOINT_FILES = {
    'E03': {'data-00000-of-00001': 'checkpoints/ckpt-50.data-00000-of-00001', 'index': 'checkpoints/ckpt-50.index',
            'meta': 'checkpoints/ckpt-50.meta'},
    'E05': {'pth': 'checkpoints/iteration_4000.pth'},
    'E06b': {'pth': 'checkpoints/netG_model_epoch_200_iter_0.pth'},
    'E06c': {'pth': 'checkpoints/netG_model_epoch_200_iter_0.pth'},
}
CHECKPOINTS = {
    ('E03', 42): {'run_id': 'a565b76bc7349532', 'seed_closure_commit': 'b95c127e394d9b2ea644573d317b44405652039e',
                   'family_closure_commit': '374de40', 'sha256': {
        'data-00000-of-00001': '5b337a63db63e405bf35bceefdacc101060cb7d1625b5a4d2d389b963e311753',
        'index': 'b6b9a0c4cfd6470959453cdb861341bcacff7047b66fb414f8dff504452dc899',
        'meta': 'dbcf372027172396c616ba19f7d76475363d34e8bac0f438c0c7f5b97fae5f97',
    }},
    ('E03', 1337): {'run_id': 'e8c85cc378e778f5', 'seed_closure_commit': '5cdb65db0a75fcdcddb64b1b5af040364a057723',
                   'family_closure_commit': '374de40', 'sha256': {
        'data-00000-of-00001': '316c94c066c32314c6819bc105d0043e1ddc7b27329afd48ddcfb8bc8d181551',
        'index': '29b36079f28c557ab9a44daa8dedb85770ed9d32c5cd0b04d002f39ae386ab70',
        'meta': 'a6119bdc1f830fdf0b968054ea929dce6e6e5838050ffbe06826d8e499a40e1c',
    }},
    ('E03', 2026): {'run_id': '237887eebab18637', 'seed_closure_commit': '15bfc84dd4cf1d68427d85d867188b702a847b65',
                   'family_closure_commit': '374de40', 'sha256': {
        'data-00000-of-00001': '9dbd463cb3eb56c9ecc578dd275e0c2507a1e6d5f8b9b4876406d5011bd52cb3',
        'index': '42107c3d56f1ec2892def0840eea520e1950ced453ffa09981899ded11ed96dc',
        'meta': '504845f926c1940a4aa0ba647dada611dba79044cb247dc11410614464f75543',
    }},
    ('E05', 42): {'run_id': '18463bf5cd471e55', 'seed_closure_commit': '072ce7d68cbdb0fab7f6d6d85ebce5b4ac1f4cfa',
                   'family_closure_commit': '7677c3a', 'sha256': {
        'pth': '4ad80b35e759154f741d249db8c6e43622cab916e13dc785ad9273bd65113cb1',
    }},
    ('E05', 1337): {'run_id': '036c806bb2f339b7', 'seed_closure_commit': '023beb4d69c63a29e504f4497945ac1da1bb6985',
                   'family_closure_commit': '7677c3a', 'sha256': {
        'pth': 'b6a4cb2c61b1b582f1acd1eb3514f6a15ea49e544e1dac48415004c6ce4f7b34',
    }},
    ('E05', 2026): {'run_id': 'b739351daf9779bc', 'seed_closure_commit': '21d47c24fbe827d296c6b67db1f28393f819c9c9',
                   'family_closure_commit': '7677c3a', 'sha256': {
        'pth': 'b7f0575f1e061936175ac04d2ac333e1008577bdd3654566c009a2f72bf0b166',
    }},
    ('E06b', 42): {'run_id': 'aa96129289d080ba', 'seed_closure_commit': '1f546c9fab1e284efd9deaef18ab7d415ff2c2ee',
                   'family_closure_commit': '3b0d308', 'sha256': {
        'pth': 'd56122e0cf363a6c870e29b394998732f4bcdaf35a96a6110e937905b64ff8aa',
    }},
    ('E06b', 1337): {'run_id': '9ffda3ea3f6cc767', 'seed_closure_commit': 'a8a7b5dd9a1fd5cd8e82b870162732163b2859f9',
                   'family_closure_commit': '3b0d308', 'sha256': {
        'pth': '645538e866a6d835a1bc35240c2f15c27296742850df96d722fd4d38609b4f53',
    }},
    ('E06b', 2026): {'run_id': 'ec71778b28479de5', 'seed_closure_commit': '1729333197c5caab7a0d4d9302f1d3274ac90d98',
                   'family_closure_commit': '3b0d308', 'sha256': {
        'pth': 'e8d0d01cfdb59b92d0f24318ab4fec042188bf0a72f9c59054ea10204a326cad',
    }},
    ('E06c', 42): {'run_id': '27dcaf23aac18d7a', 'seed_closure_commit': 'cf5ed6cbd4fbc7f989136e33577ee298619fe6a4',
                   'family_closure_commit': '24f7049', 'sha256': {
        'pth': '665684fddf11ec28b95e2a98122c85bfc5352bb8ea69ecb60caac9418c0871d9',
    }},
    ('E06c', 1337): {'run_id': '49a179a4dca77f41', 'seed_closure_commit': '9281c79d1c1f8e2b34199ff5434a6731d6e8bc9a',
                   'family_closure_commit': '24f7049', 'sha256': {
        'pth': '43ffa30dc7eaa568bc0b573a34978f897f46156232ee1a7e71d7f994beb8f336',
    }},
    ('E06c', 2026): {'run_id': '86b631403e3c4ee1', 'seed_closure_commit': '81a8ef6d1e1dec97452b32955c17d4e8718abcb7',
                   'family_closure_commit': '24f7049', 'sha256': {
        'pth': 'dc6c54b4357a51546c3abab3b957dd4b19044ae0188dde417df03446fc951878',
    }},
}


def require(ok, message):
    if not ok:
        raise LearnedBankError(message)


class LearnedBankError(RuntimeError):
    """A learned-bank contract violation: the operation stops; nothing is retried or substituted."""


class LearnedBankBlocked(LearnedBankError):
    def __init__(self, message, status):
        super().__init__('%s: %s' % (status, message))
        self.status = status


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def pixel_sha256(arr):
    """Encoder-independent image identity, identical to methods.common.bank_writer.pixel_sha256."""
    a = np.ascontiguousarray(arr)
    return sha256_bytes(('%s|%s|' % (a.dtype.str, 'x'.join(map(str, a.shape)))).encode() + a.tobytes())


def canonical_json(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False)


def _is_within(path, root):
    try:
        Path(path).relative_to(root)
        return True
    except ValueError:
        return False


# ================================================================== seeds and identities (contract byte rules)
def _seed31(preimage):
    return int.from_bytes(hashlib.sha256(preimage).digest()[:8], 'big') % (2 ** 31 - 1)


def e05_pair_seed(seed, pair_id):
    """E05-Q1 A_PER_PAIR_SEEDED: UTF-8 of 'GPAT_TRANSFERBENCH|E05|BANK|<seed>|<pair_id>'."""
    require(isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0, 'seed must be a non-negative int')
    require(isinstance(pair_id, str) and PAIR_ID_RE.match(pair_id) is not None, 'pair_id must be PTRnnnnnn')
    return _seed31(('GPAT_TRANSFERBENCH|E05|BANK|%d|%s' % (seed, pair_id)).encode('utf-8'))


def dsdg_bank_seed(method_id, seed):
    """DSDG-Q1 A_OFFICIAL_BATCHES_SEEDED: UTF-8 of 'GPAT_TRANSFERBENCH|<E06b|E06c>|BANK|<seed>'."""
    require(method_id in DSDG_METHODS, 'DSDG bank seed is defined only for E06b and E06c')
    require(isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0, 'seed must be a non-negative int')
    return _seed31(('GPAT_TRANSFERBENCH|%s|BANK|%d' % (method_id, seed)).encode('utf-8'))


def synthetic_id(method_id, seed, index):
    """DSDG-Q4 A_SYNTHETIC_ID: '<method>_S<seed>_<index:06d>', zero-based stream index."""
    require(method_id in DSDG_METHODS, 'synthetic ids exist only for DSDG banks')
    require(isinstance(index, int) and 0 <= index <= 999999, 'stream index out of range')
    return '%s_S%d_%06d' % (method_id, seed, index)


def dsdg_stream_plan(n):
    """Official batches of 100; keep the first n samples in order; count the discarded tail of the final batch."""
    require(isinstance(n, int) and n > 0, 'n must be a positive int')
    batches = -(-n // DSDG_BATCH)
    return {'batch_size': DSDG_BATCH, 'samples_kept': n, 'batches': batches,
            'samples_drawn': batches * DSDG_BATCH, 'kept_in_final_batch': n - (batches - 1) * DSDG_BATCH,
            'discarded_final_batch_samples': batches * DSDG_BATCH - n,
            'draw_order_per_batch': ['noise = randn(100, 128, generator=g)', 'noise_s = randn(100, 128, generator=g)',
                                     'z = cat(noise_s, noise, noise)']}


def dsdg_location(index):
    return index // DSDG_BATCH, index % DSDG_BATCH


# ================================================================== reference conversions (cross-checks only)
# The adapters call each pinned source's own writer; these NumPy transcriptions exist so qualification can
# recompute every conversion independently. They are never used to produce bank pixels.
def reference_e03_uint8(synth_hwc_float32):
    x = np.asarray(synth_hwc_float32, dtype=np.float32)
    return (np.clip(x, np.float32(0.0), np.float32(1.0)) * np.float32(255)).astype(np.uint8)


def reference_e05_uint8(chw_float32):
    x = np.asarray(chw_float32, dtype=np.float32)
    x = (np.transpose(x, (1, 2, 0)) + 1) / 2.0 * 255.0
    return np.clip(x, 0, 255).astype(np.uint8)


def reference_dsdg_uint8(chw_float32):
    x = np.asarray(chw_float32, dtype=np.float32)
    return np.transpose((255 * x).astype('uint8'), (1, 2, 0))


# ================================================================== authority gates
@dataclasses.dataclass(frozen=True)
class Authorization:
    mode: str
    methods: tuple
    seeds: tuple
    contract_sha256: str
    label: str
    records: tuple = ()          # sorted (name, sha256) of every authority record this authorization rests on

    def record(self):
        d = dataclasses.asdict(self)
        d['methods'], d['seeds'], d['records'] = list(self.methods), list(self.seeds), dict(self.records)
        return d


def _read_record(rel, root=None):
    p = (Path(root) if root is not None else ROOT) / rel
    if not p.is_file() or p.is_symlink():
        return None, None
    raw = p.read_bytes()
    try:
        return json.loads(raw.decode('utf-8')), sha256_bytes(raw)
    except ValueError:
        return {'_unparseable': True}, sha256_bytes(raw)


def load_generation_contract(root=None):
    """The owner-approved contract with exactly the approved decisions; otherwise STATUS_SCIENTIFIC."""
    def blocked(why):
        raise LearnedBankBlocked(why, STATUS_SCIENTIFIC)
    rec, digest = _read_record(CONTRACT_PATH, root)
    if rec is None:
        blocked('no approved contract at %s' % CONTRACT_PATH)
    if rec.get('schema_version') != CONTRACT_SCHEMA or rec.get('status') != 'OWNER_APPROVED' \
            or rec.get('approved') is not True:
        blocked('contract is not OWNER_APPROVED')
    if rec.get('decisions') != APPROVED_DECISIONS:
        blocked('contract decisions differ from the approved owner rulings')
    draft = rec.get('approved_draft') or {}
    if draft.get('path') != DRAFT_PATH or draft.get('sha256') != DRAFT_SHA256:
        blocked('contract does not bind the approved DRAFT')
    d, dsha = _read_record(DRAFT_PATH, root)
    if d is None or dsha != DRAFT_SHA256:
        blocked('the DRAFT is missing or changed')
    auth = rec.get('authorizes') or {}
    if auth.get('production_bank_generation') is not False or auth.get('generator_retraining') is not False:
        blocked('contract authorizes more than the approved scope')
    return rec, digest


def e04_completion_state(root=None, s7_record=None):
    """C-Q5: the E04 seed-2026 completion audit PASS and the family closure (or documented roadmap authority)."""
    comp, comp_sha = _read_record(E04_COMPLETION_RECORD, root)
    fam, fam_sha = _read_record(E04_FAMILY_CLOSURE_RECORD, root)
    seed_ok = bool(comp) and comp.get('status') == E04_COMPLETION_STATUS and comp.get('failed_gates') == [] \
        and comp.get('run_id') == E04_RUN_ID and comp.get('seed') == 2026
    family_ok = bool(fam) and fam.get('status') == E04_FAMILY_CLOSURE_STATUS and fam.get('failed_gates') == []
    substitute = None
    if not family_ok and s7_record:
        alt = s7_record.get('e04_documented_roadmap_authority') or {}
        if alt.get('path'):
            alt_rec, alt_sha = _read_record(alt['path'], root)
            if alt_rec and alt_sha == alt.get('sha256') and alt_rec.get('status') == 'OWNER_APPROVED':
                substitute = {'path': alt['path'], 'sha256': alt_sha}
    return {'seed_2026_completion_record': E04_COMPLETION_RECORD, 'seed_2026_completion_sha256': comp_sha,
            'seed_2026_completed_and_audited': seed_ok, 'family_closure_record': E04_FAMILY_CLOSURE_RECORD,
            'family_closure_sha256': fam_sha, 'family_closed': family_ok, 'documented_roadmap_authority': substitute,
            'ok': seed_ok and (family_ok or substitute is not None)}


def load_s7_authority(method_id, root=None):
    """S7_TOY authorization for one method (all three seeds); refuses unless every C-Q5 / C-Q3 precondition holds."""
    require(method_id in METHODS, 'unknown learned method %r' % (method_id,))
    _, contract_sha = load_generation_contract(root)
    s7, s7_sha = _read_record(S7_AUTHORITY_PATH, root)
    e04 = e04_completion_state(root, s7)
    if not e04['ok']:
        raise LearnedBankBlocked('E04 seed 2026 completion audit PASS and E04 family closure (or documented roadmap '
                                 'authority) are required before any learned-method GPU inference: %s'
                                 % canonical_json({k: e04[k] for k in ('seed_2026_completed_and_audited',
                                                                        'family_closed')}), STATUS_E04)
    if s7 is None:
        raise LearnedBankBlocked('no owner S7 launch authority at %s' % S7_AUTHORITY_PATH, STATUS_S7)
    if s7.get('schema_version') != S7_AUTHORITY_SCHEMA or s7.get('status') != 'OWNER_APPROVED' \
            or s7.get('approved') is not True:
        raise LearnedBankBlocked('S7 launch authority is not OWNER_APPROVED', STATUS_S7)
    if method_id not in (s7.get('methods') or []) or s7.get('contract_sha256') != contract_sha \
            or s7.get('e04_seed_2026_completion_sha256') != e04['seed_2026_completion_sha256']:
        raise LearnedBankBlocked('S7 launch authority does not cover %s under this contract and E04 record'
                                 % method_id, STATUS_S7)
    if not (s7.get('resource_preflight') or {}).get('required') is True:
        raise LearnedBankBlocked('S7 launch authority must require a resource preflight', STATUS_S7)
    return Authorization(mode='S7_TOY', methods=(method_id,), seeds=SEEDS, contract_sha256=contract_sha,
                         label='s7_toy', records=tuple(sorted({
                             CONTRACT_PATH: contract_sha, S7_AUTHORITY_PATH: s7_sha,
                             E04_COMPLETION_RECORD: e04['seed_2026_completion_sha256']}.items())))


def load_production_authority(method_id, seed, root=None, *, live_storage=True):
    """PRODUCTION authorization for (method, seed); refuses unless contract, E04, S7 acceptance binding, per-family
    owner production authority and an activated, live-verified storage authority all hold."""
    require(method_id in METHODS and seed in SEEDS, 'unknown production bank %r seed %r' % (method_id, seed))
    _, contract_sha = load_generation_contract(root)
    rel = PRODUCTION_AUTHORITY_TEMPLATE.format(method=method_id)
    prod, prod_sha = _read_record(rel, root)
    e04 = e04_completion_state(root, prod)
    if not e04['ok']:
        raise LearnedBankBlocked('E04 completion authority missing', STATUS_E04)
    if prod is None:
        raise LearnedBankBlocked('no owner production authority at %s' % rel, STATUS_PRODUCTION)
    if prod.get('schema_version') != PRODUCTION_AUTHORITY_SCHEMA or prod.get('status') != 'OWNER_APPROVED' \
            or prod.get('approved') is not True or prod.get('method') != method_id \
            or seed not in (prod.get('seeds') or []) or prod.get('contract_sha256') != contract_sha:
        raise LearnedBankBlocked('production authority is not an owner approval of %s seed %d under this contract'
                                 % (method_id, seed), STATUS_PRODUCTION)
    s7 = prod.get('s7_acceptance_record') or {}
    s7_rec, s7_sha = _read_record(s7.get('path') or '', root) if s7.get('path') else (None, None)
    if not s7_rec or s7_sha != s7.get('sha256') or s7_rec.get('method') != method_id \
            or s7_rec.get('s7_acceptance_passed') is not True or s7_rec.get('failed_gates') != [] \
            or sorted(s7_rec.get('seeds') or []) != sorted(SEEDS):
        raise LearnedBankBlocked('production authority does not bind a passed S7 acceptance record for %s'
                                 % method_id, STATUS_PRODUCTION)
    storage = load_storage_authority(root)
    if live_storage:
        verify_storage_root(storage)
    return Authorization(mode='PRODUCTION', methods=(method_id,), seeds=(seed,), contract_sha256=contract_sha,
                         label='production', records=tuple(sorted({
                             CONTRACT_PATH: contract_sha, rel: prod_sha, s7['path']: s7_sha,
                             STORAGE_AUTHORITY_PATH: storage['_sha256'],
                             E04_COMPLETION_RECORD: e04['seed_2026_completion_sha256']}.items())))


def fixture_authorization(label, root=None):
    """Synthetic-fixture authorization; its outputs are never benchmark evidence."""
    require(isinstance(label, str) and re.match(r'^[a-z0-9_-]+$', label) is not None, 'label must be [a-z0-9_-]+')
    _, contract_sha = load_generation_contract(root)
    return Authorization(mode='FIXTURE', methods=METHODS, seeds=(), contract_sha256=contract_sha, label=label,
                         records=((CONTRACT_PATH, contract_sha),))


# ================================================================== storage (L1 B_EXTERNAL_HDD_ADDITIVE_ROOT)
HDD = {'bank_root': '/media/ext_data/student20261_gpat_m8_banks', 'mount_point': '/media/ext_data',
       'filesystem_uuid': '608647d2-9cb0-45b0-b999-73c080ba909b', 'fstype': 'ext4', 'owner': 'student20261',
       'mode': 0o700, 'forbidden_mount_prefix': '/media/student20261/'}


def filesystem_identity(path):
    """Device, mount point, fs type and UUID of `path` (stat + /proc/self/mountinfo + /dev/disk/by-uuid; read-only).
    Same algorithm as methods.common.bank_writer.filesystem_identity (longest mount-point prefix of the same device),
    so a path resolved through /media/ext_data never reports the udisks duplicate mount."""
    st = os.stat(path)
    dev = '%d:%d' % (os.major(st.st_dev), os.minor(st.st_dev))
    mount = source = fstype = options = None
    best = -1
    rp = os.path.realpath(path)
    super_options = None
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        left, right = line.split(' - ', 1)
        f = left.split()
        if f[2] != dev:
            continue
        mp = f[4].replace('\\040', ' ')
        if (rp == mp or rp.startswith(mp.rstrip('/') + '/')) and len(mp) > best:
            best, mount, options = len(mp), mp, f[5]
            fstype, source = right.split()[0], right.split()[1]
            super_options = right.split()[2] if len(right.split()) > 2 else None
    uuid = None
    by_uuid = Path('/dev/disk/by-uuid')
    if by_uuid.is_dir():
        for link in sorted(by_uuid.iterdir()):
            try:
                if os.stat(link).st_rdev == st.st_dev:
                    uuid = link.name
                    break
            except OSError:
                continue
    vfs = os.statvfs(path)
    return {'path': str(path), 'device': dev, 'source': source, 'fstype': fstype, 'mount_point': mount,
            'mount_options': options, 'super_options': super_options, 'filesystem_uuid': uuid,
            'free_bytes': vfs.f_bavail * vfs.f_frsize, 'free_inodes': vfs.f_favail}


def _read_write(options):
    opts = (options or '').split(',')
    return 'rw' in opts and 'ro' not in opts


def bank_root_contents_ok(root, owner_uid):
    """The private root holds only <method>/seed_<s>/ real directories owned by the bank user (empty at activation).
    Anything else (files, symlinks, foreign directories, other names) is unexpected and refuses the root."""
    bad = []
    seeds = {'seed_%d' % s for s in SEEDS}
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        st = os.lstat(p)
        if name not in METHODS or not stat.S_ISDIR(st.st_mode) or st.st_uid != owner_uid:
            bad.append(name)
            continue
        for sub in sorted(os.listdir(p)):
            sst = os.lstat(os.path.join(p, sub))
            if sub not in seeds or not stat.S_ISDIR(sst.st_mode) or sst.st_uid != owner_uid:
                bad.append(name + '/' + sub)
    return bad


def storage_preflight(bank_root=None, expected=None):
    """Read-only checks of the owner-created private bank root. Never creates, chmods or mounts anything."""
    exp = dict(HDD, **(expected or {}))
    root = bank_root or exp['bank_root']
    checks = {}
    try:
        lst = os.lstat(root)
    except FileNotFoundError:
        return {'bank_root': root, 'exists': False, 'checks': {'exists': False}, 'ok': False,
                'owner_action': 'the owner must create the private directory (see the storage pending record)'}
    import pwd
    checks['exists'] = True
    checks['not_symlink'] = not stat.S_ISLNK(lst.st_mode)
    checks['is_directory'] = stat.S_ISDIR(lst.st_mode)
    checks['realpath_equals_path'] = os.path.realpath(root) == root
    checks['owner'] = pwd.getpwuid(lst.st_uid).pw_name == exp['owner']
    checks['mode_0700_no_setgid'] = stat.S_IMODE(lst.st_mode) == exp['mode']
    checks['not_udisks_duplicate'] = not root.startswith(exp['forbidden_mount_prefix'])
    fs = filesystem_identity(root) if checks['is_directory'] else {}
    checks['mount_point'] = fs.get('mount_point') == exp['mount_point']
    checks['filesystem_uuid'] = fs.get('filesystem_uuid') == exp['filesystem_uuid']
    checks['fstype_ext4'] = fs.get('fstype') == exp['fstype']
    checks['mount_read_write'] = _read_write(fs.get('mount_options')) and _read_write(fs.get('super_options'))
    checks['user_rwx_access'] = bool(checks['is_directory'] and lst.st_uid == os.getuid()
                                     and os.access(root, os.R_OK | os.W_OK | os.X_OK))
    unexpected = bank_root_contents_ok(root, lst.st_uid) if checks['is_directory'] and checks['not_symlink'] else ['?']
    checks['no_unexpected_contents'] = unexpected == []
    return {'bank_root': root, 'exists': True, 'mode_octal': oct(stat.S_IMODE(lst.st_mode)), 'filesystem': fs,
            'unexpected_entries': unexpected, 'checks': checks, 'ok': all(checks.values())}


def load_storage_authority(root=None):
    """The ACTIVATED storage authority (OWNER_APPROVED, option B, the owner-created directory verified at activation).
    A pending-activation record is never accepted."""
    rec, digest = _read_record(STORAGE_AUTHORITY_PATH, root)
    if rec is None:
        raise LearnedBankBlocked('no activated storage authority at %s (a pending-activation record is not usable)'
                                 % STORAGE_AUTHORITY_PATH, STATUS_STORAGE)
    if rec.get('schema_version') != STORAGE_AUTHORITY_SCHEMA or rec.get('status') != 'OWNER_APPROVED' \
            or rec.get('approved') is not True or (rec.get('decisions') or {}).get('L1_bank_root') \
            != 'B_EXTERNAL_HDD_ADDITIVE_ROOT':
        raise LearnedBankBlocked('storage authority is not an activated L1 B approval', STATUS_STORAGE)
    reg = rec.get('bank_root_registry') or {}
    if reg.get('bank_root') != HDD['bank_root'] or reg.get('filesystem_uuid') != HDD['filesystem_uuid'] \
            or reg.get('mount_point') != HDD['mount_point'] or reg.get('fstype') != HDD['fstype'] \
            or reg.get('owner') != HDD['owner'] or reg.get('mode') != '0700' \
            or reg.get('layout') != BANK_ROOT_LAYOUT or sorted(reg.get('methods') or ()) != sorted(METHODS):
        raise LearnedBankBlocked('storage registry differs from the approved HDD root', STATUS_STORAGE)
    act = rec.get('activation_verification') or {}
    if act.get('ok') is not True:
        raise LearnedBankBlocked('storage activation verification did not pass', STATUS_STORAGE)
    out = dict(reg)
    out['_sha256'] = digest
    return out


def verify_storage_root(storage):
    pf = storage_preflight(storage['bank_root'])
    if not pf['ok']:
        raise LearnedBankBlocked('live storage preflight failed: %s' % canonical_json(pf['checks']), STATUS_STORAGE)
    return pf


BANK_ROOT_LAYOUT = '<bank_root>/<method>/seed_<seed>/'


def production_bank_dir(method_id, seed, storage):
    """The single registered location of a production bank: <bank_root>/<method>/seed_<seed>. Only the approved HDD
    root is accepted (never the udisks duplicate, never a relative or redirected root), and the result must stay
    inside it (no traversal: method and seed come from the closed METHODS / SEEDS sets)."""
    require(method_id in METHODS and seed in SEEDS and type(seed) is int, 'unknown production bank')
    root = storage['bank_root']
    require(root == HDD['bank_root'] and os.path.isabs(root) and os.path.normpath(root) == root,
            'bank root is not the registered HDD root')
    require(not root.startswith(HDD['forbidden_mount_prefix']), 'the udisks duplicate mount is never a bank path')
    out = Path(root) / method_id / ('seed_%d' % seed)
    require(os.path.normpath(str(out)) == str(out) and _is_within(out, root) and out.parent.parent == Path(root),
            'bank path escapes the registered root')
    return out


def resolve_registered_bank(method_id, seed, root=None):
    """Downstream readers resolve a learned production bank only through the activated registry (never a symlink)."""
    storage = load_storage_authority(root)
    return production_bank_dir(method_id, seed, storage)


# ================================================================== checkpoints and configs
def verify_config_identity(method_id, root=None):
    rel, expected = METHOD_CONFIGS[method_id]
    p = (Path(root) if root is not None else ROOT) / rel
    require(p.is_file() and not p.is_symlink(), 'missing frozen config %s' % rel)
    digest = sha256_file(p)
    if digest != expected:
        raise LearnedBankBlocked('%s changed (%s != %s)' % (rel, digest, expected), STATUS_SCIENTIFIC)
    return {'path': rel, 'sha256': digest}


def checkpoint_authority(method_id, seed):
    require((method_id, seed) in CHECKPOINTS, 'no approved checkpoint for %s seed %r' % (method_id, seed))
    return CHECKPOINTS[(method_id, seed)]


def checkpoint_identity_sha256(method_id, seed):
    """One digest naming the exact approved checkpoint file set (canonical JSON of {file key: sha256})."""
    return sha256_bytes(canonical_json(checkpoint_authority(method_id, seed)['sha256']).encode())


def verify_checkpoint(method_id, seed, run_root):
    """Hash every approved file of (method, seed) under <runtime_root>/runs/m6/<method>/seed_<seed>; any difference,
    absence or symlink blocks that method. Never opens a checkpoint for writing."""
    auth = checkpoint_authority(method_id, seed)
    run_root = Path(run_root)
    require(run_root.parts[-4:] == ('runs', 'm6', method_id, 'seed_%d' % seed),
            'run root %s is not runs/m6/%s/seed_%d' % (run_root, method_id, seed))
    out = {}
    for key, rel in CHECKPOINT_FILES[method_id].items():
        p = run_root / rel
        if not p.is_file() or p.is_symlink():
            raise LearnedBankBlocked('%s seed %d: %s missing or a symlink' % (method_id, seed, rel), STATUS_CHECKPOINT)
        digest = sha256_file(p)
        if digest != auth['sha256'][key]:
            raise LearnedBankBlocked('%s seed %d: %s sha256 %s != approved %s' % (method_id, seed, rel, digest,
                                                                                    auth['sha256'][key]),
                                     STATUS_CHECKPOINT)
        out[key] = {'path': str(p), 'sha256': digest, 'size': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns}
    summary = run_root / 'run_summary.json'
    if summary.is_file():
        rid = json.loads(summary.read_text()).get('run_id')
        if rid != auth['run_id']:
            raise LearnedBankBlocked('%s seed %d: run_id %s != approved %s' % (method_id, seed, rid, auth['run_id']),
                                     STATUS_CHECKPOINT)
    return {'method_id': method_id, 'seed': seed, 'run_id': auth['run_id'], 'rule': CHECKPOINT_RULES[method_id],
            'files': out, 'checkpoint_identity_sha256': checkpoint_identity_sha256(method_id, seed),
            'seed_closure_commit': auth['seed_closure_commit'], 'family_closure_commit': auth['family_closure_commit']}


SOURCES_PATH = 'methods/common/learned_bank_sources.json'


def verify_pinned_sources(family, repo_root=None):
    """SHA256 of every pinned upstream file a family's adapter executes or cites equals the committed table."""
    table = json.loads((ROOT / SOURCES_PATH).read_text())['families'][family]
    root = (Path(repo_root) if repo_root is not None else ROOT) / table['root']
    files = {}
    for rel, digest in sorted(table['files'].items()):
        p = root / rel
        require(p.is_file() and not p.is_symlink(), '%s pinned file missing: %s' % (family, rel))
        files[rel] = sha256_file(p)
        require(files[rel] == digest, '%s pinned file changed: %s' % (family, rel))
    return {'family': family, 'root': str(root), 'commit': table['commit'], 'files': files}


def train_face_allowlist_parquet(sample_datasets, root=None):
    """{sample_id: {split, dataset, face_png_sha256}} for the referenced samples only: split_v1 and the M2 accounting
    are hash-checked before parsing and read with a row filter on the referenced ids, so no other row (in particular no
    VAL or TEST row) reaches Python. Every referenced sample must be TRAIN and M2 COMPLETE."""
    import pyarrow.parquet as pq
    base = Path(root) if root is not None else ROOT
    used = sorted(sample_datasets)
    require(sha256_file(base / SPLIT_PATH) == SPLIT_SHA256, 'split_v1 sha256')
    require(sha256_file(base / M2_ACCOUNTING_PATH) == M2_ACCOUNTING_SHA256, 'M2 accounting sha256')
    sp = pq.read_table(base / SPLIT_PATH, columns=['sample_id', 'split', 'dataset'],
                       filters=[('sample_id', 'in', used)]).to_pydict()
    m2 = pq.read_table(base / M2_ACCOUNTING_PATH, columns=['sample_id', 'face_png_sha256', 'final_status'],
                       filters=[('sample_id', 'in', used)]).to_pydict()
    split = dict(zip(sp['sample_id'], sp['split']))
    dataset = dict(zip(sp['sample_id'], sp['dataset']))
    face = {}
    for sid, digest, status in zip(m2['sample_id'], m2['face_png_sha256'], m2['final_status']):
        require(status == 'COMPLETE', 'M2 face not COMPLETE: %s' % sid)
        face[sid] = digest
    require(set(split) == set(used) == set(face), 'referenced samples missing from split_v1/M2')
    require(set(split.values()) == {'TRAIN'}, 'a referenced sample is not TRAIN')
    out = {}
    for sid in used:
        require(dataset[sid] == sample_datasets[sid], 'dataset mismatch %s' % sid)
        out[sid] = {'split': split[sid], 'dataset': dataset[sid], 'face_png_sha256': face[sid]}
    return out


def recheck_checkpoint_unchanged(before):
    """After inference: same bytes, size and mtime (no mutation)."""
    for key, rec in before['files'].items():
        p = Path(rec['path'])
        require(p.stat().st_size == rec['size'] and p.stat().st_mtime_ns == rec['mtime_ns']
                and sha256_file(p) == rec['sha256'], 'checkpoint %s changed during inference' % p)
    return True


# ================================================================== populations
@dataclasses.dataclass(frozen=True)
class BankRow:
    index: int                 # E03/E05: canonical 0-based global pairs_train_v1 position; DSDG: 0-based stream index
    row_id: str                # pair_id (E03/E05) or synthetic_id (DSDG)
    pair_id: object = None
    target_live_id: object = None
    source_spoof_id: object = None
    dataset: object = None
    synthetic_id: object = None


class Population:
    def __init__(self, method_id, seed, kind, rows, provenance):
        require(method_id in METHODS, 'unknown method')
        require(kind in POPULATION_KINDS.values(), 'unknown population kind %r' % (kind,))
        self.method_id, self.seed, self.kind = method_id, seed, kind
        self.rows = tuple(rows)
        self.provenance = dict(provenance)
        require(self.rows, 'empty population')
        idx = [r.index for r in self.rows]
        require(all(a < b for a, b in zip(idx, idx[1:])) and idx[0] >= 0, 'row indices not strictly increasing')
        require(len({r.row_id for r in self.rows}) == len(self.rows), 'duplicate row ids')
        if method_id in PAIR_METHODS:
            for r in self.rows:
                require(r.synthetic_id is None and r.row_id == r.pair_id, 'pair rows are identified by pair_id')
                require(isinstance(r.pair_id, str) and PAIR_ID_RE.match(r.pair_id) is not None,
                        'non-TRAIN pair id %r' % (r.pair_id,))
                require(r.target_live_id != r.source_spoof_id and r.dataset in DATASETS, 'pair %s' % r.pair_id)
            if kind == 'PRODUCTION_FROZEN_TRAIN_POPULATION':
                require(idx == list(range(N_SYN)), 'production population must be positions 0..8837')
            if kind == 'S7_TOY_POPULATION':
                require(len(self.rows) == TOY_N, 'S7 toy population must have 186 pairs')
        else:
            pat = FIXTURE_SYNTHETIC_ID_RE if kind == 'SYNTHETIC_FIXTURE' else SYNTHETIC_ID_RE
            for r in self.rows:
                require(r.pair_id is None and r.target_live_id is None and r.source_spoof_id is None
                        and r.dataset is None, 'DSDG rows carry no pair lineage (DSDG-Q4)')
                require(r.row_id == r.synthetic_id == synthetic_id(method_id, seed, r.index)
                        and pat.match(r.row_id) is not None, 'DSDG synthetic id %r' % (r.row_id,))
            require(idx == list(range(len(self.rows))), 'DSDG stream indices must be 0..n-1')
            if kind == 'PRODUCTION_FROZEN_TRAIN_POPULATION':
                require(len(self.rows) == N_SYN, 'production population must be 8838 stream samples')
            if kind == 'S7_TOY_POPULATION':
                require(len(self.rows) == TOY_N, 'S7 toy population must be stream indices 0..185')

    def record(self):
        return dict(self.provenance, method_id=self.method_id, seed=self.seed, kind=self.kind, rows=len(self.rows),
                    row_ids_sha256=sha256_bytes('\n'.join(r.row_id for r in self.rows).encode()),
                    first_index=self.rows[0].index, last_index=self.rows[-1].index)


def dsdg_population(method_id, seed, n, kind, provenance=None):
    rows = [BankRow(index=i, row_id=synthetic_id(method_id, seed, i), synthetic_id=synthetic_id(method_id, seed, i))
            for i in range(n)]
    prov = dict(provenance or {}, stream=dsdg_stream_plan(n), bank_seed=dsdg_bank_seed(method_id, seed))
    return Population(method_id, seed, kind, rows, prov)


def load_toy_manifest(root=None):
    p = (Path(root) if root is not None else ROOT) / TOY_MANIFEST_PATH
    raw = p.read_bytes()
    require(sha256_bytes(raw) == TOY_MANIFEST_SHA256, 'toy manifest sha256 differs from the approved S6 record')
    man = json.loads(raw.decode('utf-8'))
    require(man.get('test_rows_read') == 0, 'toy manifest must record zero TEST rows')
    rows = man['toy_pairs']
    require(len(rows) == TOY_N, 'toy manifest must hold 186 pairs')
    return rows


def pair_population(method_id, seed, kind, relation, *, positions=None, provenance=None):
    """`relation`: the frozen pairs_train_v1 rows in canonical order as (pair_id, target_live_id, source_spoof_id,
    dataset). PRODUCTION takes every row; S7_TOY takes the toy positions (each toy row must equal the relation row at
    its global position)."""
    require(method_id in PAIR_METHODS, 'pair populations exist only for E03/E05')
    relation = [tuple(r) for r in relation]
    ids = [r[0] for r in relation]
    require(ids == sorted(ids) and len(set(ids)) == len(ids), 'relation not in ascending unique pair_id order')
    prov = dict(provenance or {})
    if kind == 'S7_TOY_POPULATION':
        toy = load_toy_manifest()
        positions = [t['position'] for t in toy]
        for t in toy:
            require(tuple(relation[t['position']]) == (t['pair_id'], t['target_live_id'], t['source_spoof_id'],
                                                       t['dataset']), 'toy row %s differs from the frozen relation'
                    % t['pair_id'])
        prov.update(toy_manifest=TOY_MANIFEST_PATH, toy_manifest_sha256=TOY_MANIFEST_SHA256)
    elif kind == 'PRODUCTION_FROZEN_TRAIN_POPULATION':
        require(positions is None and len(relation) == N_SYN, 'production takes all 8838 frozen rows')
        positions = list(range(N_SYN))
    else:
        require(positions is not None, 'fixture populations need explicit positions')
    rows = [BankRow(index=i, row_id=relation[i][0], pair_id=relation[i][0], target_live_id=relation[i][1],
                    source_spoof_id=relation[i][2], dataset=relation[i][3]) for i in positions]
    if method_id == 'E05':
        prov['pair_seeds_sha256'] = sha256_bytes(','.join(str(e05_pair_seed(seed, r.pair_id)) for r in rows).encode())
    return Population(method_id, seed, kind, rows, prov)


# ================================================================== TRAIN-only face reader
def read_bytes_noatime(path):
    fd = os.open(str(path), os.O_RDONLY | os.O_NOATIME | getattr(os, 'O_NOFOLLOW', 0))
    try:
        chunks = []
        while True:
            b = os.read(fd, 1 << 20)
            if not b:
                return b''.join(chunks)
            chunks.append(b)
    finally:
        os.close(fd)


class TrainFaceReader:
    """Canonical faces of an allowlisted TRAIN sample set only. Every read is O_NOATIME with an atime stat check, the
    bytes must equal the M2 face SHA256 before any decoding, and no sample outside the allowlist (hence no VAL or TEST
    sample) has a path. Decoding is the adapter's (each method keeps its training decoder)."""

    def __init__(self, faces_root, samples):
        root = Path(faces_root)
        require(root.is_absolute() and root.is_dir() and not root.is_symlink(), 'faces root %s' % root)
        require(os.path.realpath(str(root)) == str(root), 'faces root must be resolved')
        for sid, rec in samples.items():
            require(rec.get('split') == 'TRAIN', 'sample %s is not TRAIN' % sid)
            require(rec.get('dataset') in DATASETS, 'sample %s dataset' % sid)
            require(isinstance(rec.get('face_png_sha256'), str) and len(rec['face_png_sha256']) == 64,
                    'sample %s face sha256' % sid)
        self.root, self.samples = root, {k: dict(v) for k, v in samples.items()}
        self.reads, self.atime_checked_reads, self.opened = 0, 0, []

    def path(self, sample_id):
        rec = self.samples.get(sample_id)
        require(rec is not None, 'sample %r is outside the TRAIN allowlist' % (sample_id,))
        d = self.root / rec['dataset']
        p = d / ('%s.png' % sample_id)
        require(not d.is_symlink() and not p.is_symlink(), 'symlinked face refused: %s' % p)
        return p

    def read(self, sample_id):
        p = self.path(sample_id)
        try:
            a0 = os.stat(str(p)).st_atime_ns
            raw = read_bytes_noatime(p)
            a1 = os.stat(str(p)).st_atime_ns
        except OSError as exc:
            raise LearnedBankError('face unreadable %s: %s' % (sample_id, exc))
        require(a0 == a1, 'atime of %s changed across an O_NOATIME read' % sample_id)
        digest = sha256_bytes(raw)
        require(digest == self.samples[sample_id]['face_png_sha256'], 'face sha256 mismatch %s' % sample_id)
        self.reads += 1
        self.atime_checked_reads += 1
        self.opened.append(sample_id)
        return raw, digest

    def record(self):
        return {'faces_root': str(self.root), 'allowlisted_samples': len(self.samples), 'reads': self.reads,
                'atime_checked_reads': self.atime_checked_reads,
                'distinct_samples_read': len(set(self.opened)),
                'splits_allowlisted': sorted({v['split'] for v in self.samples.values()}),
                'o_noatime': True}


# ================================================================== encoding / create-only writes
def encode_png(arr):
    """Deterministic lossless PNG with the E01/E02 bank writer settings (no metadata chunks)."""
    from PIL import Image
    im = Image.fromarray(np.ascontiguousarray(arr))
    require(im.mode == 'RGB', 'PNG mode %s != RGB' % im.mode)
    buf = io.BytesIO()
    im.save(buf, format='PNG', compress_level=PNG_COMPRESS_LEVEL, optimize=False)
    return buf.getvalue()


def decode_png(raw):
    from PIL import Image
    with Image.open(io.BytesIO(raw)) as im:
        require(im.format == 'PNG' and im.mode == 'RGB' and im.size == (CANONICAL, CANONICAL),
                'bank PNG is not a 256x256 RGB PNG')
        return np.array(im)


def write_exclusive(path, data, mode=0o444):
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        view = memoryview(data)
        while view:
            view = view[os.write(fd, view):]
        os.fsync(fd)
    finally:
        os.close(fd)


# ================================================================== manifest schema
MANIFEST = 'bank_manifest.jsonl'
MANIFEST_SHA = 'bank_manifest.sha256'
PROVENANCE = 'bank_provenance.json'
BANK_DIR = 'bank'
MANIFEST_FIELDS = (
    'schema', 'row_index', 'row_id', 'method_id', 'bank_seed', 'generator_seed', 'mode', 'label', 'synthetic_id',
    'pair_id', 'target_live_id', 'source_spoof_id', 'dataset', 'target_face_sha256', 'source_face_sha256',
    'deterministic_noise_seed', 'dsdg_batch_index', 'dsdg_batch_offset', 'checkpoint_identity_sha256',
    'config_sha256', 'contract_sha256', 'conversion_rule', 'generation_status', 'failure_reason', 'output_path',
    'output_sha256', 'output_pixel_sha256', 'raw_output_finite', 'raw_output_min', 'raw_output_max')
CONVERSION_RULES = {
    'E03': 'tf.cast(tf.clip_by_value(synth1, 0.0, 1.0) * 255, tf.uint8) [stdn model/utils.py:24,37]',
    'E05': 'util.tensor2im(mix[0]): clip((x+1)/2*255, 0, 255).astype(uint8) [swapping_autoencoder util/util.py:125-131]',
    'E06b': "np.transpose((255 * spoof).astype('uint8'), (1, 2, 0)) [DSDG generated.py:63]",
    'E06c': "np.transpose((255 * spoof).astype('uint8'), (1, 2, 0)) [DSDG generated.py:63]",
}


@dataclasses.dataclass
class GenResult:
    row: BankRow
    pixels: object               # HWC uint8 RGB or None
    raw_finite: bool
    raw_min: object
    raw_max: object
    failure_reason: object = None
    noise_seed: object = None
    target_face_sha256: object = None
    source_face_sha256: object = None


def _manifest_row(adapter, population, auth_mode, res, out_path, out_sha, out_px):
    r = res.row
    ok = out_path is not None
    batch = offset = None
    if population.method_id in DSDG_METHODS:
        batch, offset = dsdg_location(r.index)
    return {
        'schema': SCHEMA, 'row_index': r.index, 'row_id': r.row_id, 'method_id': population.method_id,
        'bank_seed': population.seed, 'generator_seed': adapter.generator_seed, 'mode': auth_mode, 'label': 'spoof',
        'synthetic_id': r.synthetic_id, 'pair_id': r.pair_id, 'target_live_id': r.target_live_id,
        'source_spoof_id': r.source_spoof_id, 'dataset': r.dataset, 'target_face_sha256': res.target_face_sha256,
        'source_face_sha256': res.source_face_sha256, 'deterministic_noise_seed': res.noise_seed,
        'dsdg_batch_index': batch, 'dsdg_batch_offset': offset,
        'checkpoint_identity_sha256': adapter.checkpoint_identity_sha256, 'config_sha256': adapter.config_sha256,
        'contract_sha256': adapter.contract_sha256, 'conversion_rule': CONVERSION_RULES[population.method_id],
        'generation_status': 'success' if ok else 'failure', 'failure_reason': None if ok else res.failure_reason,
        'output_path': out_path, 'output_sha256': out_sha, 'output_pixel_sha256': out_px,
        'raw_output_finite': bool(res.raw_finite),
        'raw_output_min': None if res.raw_min is None else float(res.raw_min),
        'raw_output_max': None if res.raw_max is None else float(res.raw_max)}


def environment_record():
    import PIL
    return {'python_executable': sys.executable, 'python_version': platform.python_version(),
            'numpy': np.__version__, 'pillow': PIL.__version__, 'host': platform.node(),
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}


def _git_head():
    try:
        return subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL, universal_newlines=True).stdout.strip() or None
    except OSError:
        return None


# ================================================================== writer
def _check_mode(adapter, population, auth, run_dir):
    require(isinstance(auth, Authorization) and auth.mode in MODES, 'an Authorization is required')
    require(population.method_id == adapter.method_id and population.seed == adapter.bank_seed,
            'adapter and population disagree on method/seed')
    require(adapter.contract_sha256 == auth.contract_sha256, 'adapter and authorization bind different contracts')
    require(population.kind == POPULATION_KINDS[auth.mode], '%s needs a %s population (got %s)'
            % (auth.mode, POPULATION_KINDS[auth.mode], population.kind))
    if auth.mode == 'FIXTURE':
        require(adapter.is_fixture is True, 'fixture mode accepts only fixture-weight adapters')
        require('faces_256' not in Path(run_dir).parts, 'fixture banks never live beside benchmark faces')
        return
    require(adapter.is_fixture is False, '%s requires the real approved checkpoint' % auth.mode)
    require(population.method_id in auth.methods and population.seed in auth.seeds,
            'authorization does not cover %s seed %d' % (population.method_id, population.seed))
    require(adapter.checkpoint_identity_sha256 == checkpoint_identity_sha256(population.method_id, population.seed),
            'adapter checkpoint is not the approved checkpoint of %s seed %d' % (population.method_id,
                                                                              population.seed))
    require(adapter.generator_seed == population.seed, 'bank seed s must use generator seed s (D12)')
    if auth.mode == 'PRODUCTION':
        storage = load_storage_authority()
        require(Path(run_dir) == production_bank_dir(population.method_id, population.seed, storage),
                'production banks are written only at their registered location')


def write_bank(adapter, population, auth, run_dir, *, reader=None, progress=None, extra_provenance=None):
    """Generate one bank. Create-only and fail-closed: `run_dir` must not exist (its parent must); every file is
    O_EXCL 0444; per-row non-finite outputs are counted failures; any other exception aborts the run, keeps the partial
    evidence and writes an ABORTED provenance record. Returns the provenance dict."""
    run_dir = Path(run_dir)
    _check_mode(adapter, population, auth, run_dir)
    parent = run_dir.parent
    require(parent.is_dir() and not parent.is_symlink(), 'parent %s must be an existing real directory' % parent)
    require(not os.path.lexists(str(run_dir)), 'run directory exists: %s (banks are never overwritten)' % run_dir)
    os.mkdir(str(run_dir), 0o755)
    os.mkdir(str(run_dir / BANK_DIR), 0o755)
    t0, start = time.monotonic(), time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    rows = []
    prov = {'schema': SCHEMA, 'record': 'bank_provenance', 'method_id': population.method_id,
            'bank_seed': population.seed, 'mode': auth.mode, 'authorization': auth.record(),
            'population': population.record(), 'adapter': adapter.identity(), 'environment': environment_record(),
            'git_head': _git_head(), 'start_utc': start, 'run_dir': str(run_dir)}
    if extra_provenance:
        prov['extra'] = extra_provenance
    try:
        for res in adapter.generate(population, reader):
            require(res.row == population.rows[len(rows)], 'adapter yielded rows out of population order')
            out_path = out_sha = out_px = None
            if res.pixels is not None:
                require(res.raw_finite, '%s: pixels returned for a non-finite raw output' % res.row.row_id)
                px = res.pixels
                require(isinstance(px, np.ndarray) and px.dtype == np.uint8 and px.shape == (CANONICAL, CANONICAL, 3),
                        '%s: output is not canonical 256x256x3 uint8' % res.row.row_id)
                raw = encode_png(px)
                require(np.array_equal(decode_png(raw), px), '%s: PNG round trip changed pixels' % res.row.row_id)
                out_path = '%s/%s.png' % (BANK_DIR, res.row.row_id)
                write_exclusive(run_dir / out_path, raw)
                out_sha, out_px = sha256_bytes(raw), pixel_sha256(px)
            else:
                require(not res.raw_finite and res.failure_reason, '%s: a failure row needs a non-finite output and '
                        'a reason (only non-finite outputs are counted failures)' % res.row.row_id)
            rows.append(_manifest_row(adapter, population, auth.mode, res, out_path, out_sha, out_px))
            if progress is not None:
                progress(len(rows), len(population.rows))
        require(len(rows) == len(population.rows), 'adapter produced %d of %d rows' % (len(rows), len(population.rows)))
        body = ''.join(canonical_json(r) + '\n' for r in rows).encode('ascii')
        write_exclusive(run_dir / MANIFEST, body)
        manifest_sha = sha256_bytes(body)
        write_exclusive(run_dir / MANIFEST_SHA, ('%s  %s\n' % (manifest_sha, MANIFEST)).encode())
        failures = [r for r in rows if r['generation_status'] != 'success']
        prov.update(status='completed' if not failures else 'completed_with_failures', rows=len(rows),
                    outputs=len(rows) - len(failures), failure_count=len(failures),
                    failures=[{'row_id': r['row_id'], 'failure_reason': r['failure_reason']} for r in failures],
                    manifest_sha256=manifest_sha, adapter_finalize=adapter.finalize(),
                    reader=reader.record() if reader is not None else None,
                    wall_time_seconds=time.monotonic() - t0, end_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        check = read_bank(run_dir, expect_method=population.method_id, expect_seed=population.seed,
                          expect_mode=auth.mode, expect_rows=len(population.rows), _skip_provenance=True)
        require(check['ok'], 'post-write verification failed: %s' % check['problems'][:5])
        prov['post_write_verification'] = {k: check[k] for k in ('ok', 'rows', 'files_checked', 'problems')}
        write_exclusive(run_dir / PROVENANCE, (json.dumps(prov, indent=1, sort_keys=True) + '\n').encode())
        os.chmod(str(run_dir / BANK_DIR), 0o555)
        os.chmod(str(run_dir), 0o555)
        return prov
    except BaseException as exc:
        prov.update(status='aborted', failure_reason='%s: %s' % (type(exc).__name__, exc), rows_before_abort=len(rows),
                    wall_time_seconds=time.monotonic() - t0)
        try:
            write_exclusive(run_dir / ('ABORTED_' + PROVENANCE), (json.dumps(prov, indent=1, sort_keys=True,
                                                                              default=str) + '\n').encode())
        except OSError:
            pass
        raise


# ================================================================== reader
def read_bank(run_dir, *, expect_method=None, expect_seed=None, expect_mode=None, expect_rows=None,
              expect_checkpoint_identity=None, verify_pixels=True, _skip_provenance=False):
    """Independent verification of a learned bank. Returns {'ok', 'problems', 'rows', ...}; never raises on content."""
    run_dir = Path(run_dir)
    problems = []
    try:
        body = (run_dir / MANIFEST).read_bytes()
        recorded = (run_dir / MANIFEST_SHA).read_text().split()
    except OSError as exc:
        return {'ok': False, 'problems': ['unreadable bank: %s' % exc], 'rows': []}
    if recorded != [sha256_bytes(body), MANIFEST]:
        problems.append('manifest sha256 file mismatch')
    rows = []
    for n, line in enumerate(body.decode('ascii').splitlines()):
        try:
            r = json.loads(line)
        except ValueError:
            problems.append('line %d is not JSON' % n)
            continue
        if canonical_json(r) != line:
            problems.append('line %d is not canonical JSON' % n)
        if set(r) != set(MANIFEST_FIELDS):
            problems.append('line %d fields differ from the schema' % n)
            continue
        rows.append(r)
    if not rows:
        problems.append('empty manifest')
    idx = [r['row_index'] for r in rows]
    if not all(a < b for a, b in zip(idx, idx[1:])):
        problems.append('row_index not strictly increasing')
    if len({r['row_id'] for r in rows}) != len(rows):
        problems.append('duplicate row ids')
    if expect_rows is not None and len(rows) != expect_rows:
        problems.append('rows %d != expected %d' % (len(rows), expect_rows))
    for r in rows:
        m = r['method_id']
        if r['schema'] != SCHEMA or r['label'] != 'spoof' or m not in METHODS:
            problems.append('%s: schema/label/method' % r['row_id'])
            continue
        if (expect_method and m != expect_method) or (expect_seed is not None and r['bank_seed'] != expect_seed) \
                or (expect_mode and r['mode'] != expect_mode):
            problems.append('%s: method/seed/mode differ from the expected bank' % r['row_id'])
        if expect_checkpoint_identity and r['checkpoint_identity_sha256'] != expect_checkpoint_identity:
            problems.append('%s: checkpoint identity' % r['row_id'])
        if r['conversion_rule'] != CONVERSION_RULES[m]:
            problems.append('%s: conversion rule' % r['row_id'])
        if m in PAIR_METHODS:
            if not (isinstance(r['pair_id'], str) and PAIR_ID_RE.match(r['pair_id']) and r['row_id'] == r['pair_id']
                    and r['synthetic_id'] is None and r['target_live_id'] and r['source_spoof_id']
                    and r['dataset'] in DATASETS and r['dsdg_batch_index'] is None):
                problems.append('%s: pair lineage' % r['row_id'])
            if m == 'E05' and r['deterministic_noise_seed'] != e05_pair_seed(r['generator_seed'], r['pair_id']):
                problems.append('%s: E05 pair seed' % r['row_id'])
            if m == 'E03' and r['deterministic_noise_seed'] is not None:
                problems.append('%s: E03 is not stochastic at inference' % r['row_id'])
        else:
            pat = FIXTURE_SYNTHETIC_ID_RE if r['mode'] == 'FIXTURE' else SYNTHETIC_ID_RE
            if not (r['pair_id'] is None and r['target_live_id'] is None and r['source_spoof_id'] is None
                    and r['dataset'] is None and r['synthetic_id'] == r['row_id'] and pat.match(r['row_id'])
                    and r['row_id'] == synthetic_id(m, r['bank_seed'], r['row_index'])
                    and (r['dsdg_batch_index'], r['dsdg_batch_offset']) == dsdg_location(r['row_index'])
                    and r['deterministic_noise_seed'] == dsdg_bank_seed(m, r['bank_seed'])):
                problems.append('%s: DSDG synthetic identity' % r['row_id'])
        if r['generation_status'] == 'success':
            if r['output_path'] != '%s/%s.png' % (BANK_DIR, r['row_id']) or not r['raw_output_finite']:
                problems.append('%s: success row path/finite' % r['row_id'])
        elif r['generation_status'] == 'failure':
            if r['output_path'] is not None or r['raw_output_finite'] or not r['failure_reason']:
                problems.append('%s: failure row' % r['row_id'])
        else:
            problems.append('%s: generation_status' % r['row_id'])
    expected = {r['output_path'] for r in rows if r['generation_status'] == 'success'}
    try:
        actual = {'%s/%s' % (BANK_DIR, p.name) for p in (run_dir / BANK_DIR).iterdir()}
    except OSError:
        actual = set()
    if expected != actual:
        problems.append('bank file set differs: missing %s extra %s' % (sorted(expected - actual)[:3],
                                                                         sorted(actual - expected)[:3]))
    checked = 0
    if verify_pixels:
        for r in rows:
            if r['generation_status'] != 'success' or r['output_path'] not in actual:
                continue
            p = run_dir / r['output_path']
            if p.is_symlink() or stat.S_IMODE(os.lstat(str(p)).st_mode) & 0o222:
                problems.append('%s: output is a symlink or writable' % r['row_id'])
            raw = p.read_bytes()
            if sha256_bytes(raw) != r['output_sha256']:
                problems.append('%s: file sha256' % r['row_id'])
            try:
                arr = decode_png(raw)
            except Exception as exc:  # decode failures are reported, not raised
                problems.append('%s: decode %s' % (r['row_id'], exc))
                continue
            if arr.dtype != np.uint8 or arr.shape != (CANONICAL, CANONICAL, 3) \
                    or pixel_sha256(arr) != r['output_pixel_sha256']:
                problems.append('%s: pixels' % r['row_id'])
            checked += 1
    prov_ok = None
    if not _skip_provenance:
        prov_ok = (run_dir / PROVENANCE).is_file() and not (run_dir / ('ABORTED_' + PROVENANCE)).exists()
        if not prov_ok:
            problems.append('missing provenance or aborted run')
    return {'ok': not problems, 'problems': problems[:50], 'rows': rows, 'files_checked': checked,
            'manifest_sha256': sha256_bytes(body), 'provenance_ok': prov_ok,
            'failures': sum(1 for r in rows if r['generation_status'] != 'success')}


def load_bank_pixels(run_dir, row):
    return decode_png((Path(run_dir) / row['output_path']).read_bytes())


def compare_runs(run_a, run_b):
    """C-Q2 bitwise replay: identical manifests and identical bank file bytes."""
    a, b = Path(run_a), Path(run_b)
    out = {'manifest_equal': (a / MANIFEST).read_bytes() == (b / MANIFEST).read_bytes()}
    fa = sorted(p.name for p in (a / BANK_DIR).iterdir())
    fb = sorted(p.name for p in (b / BANK_DIR).iterdir())
    out['file_set_equal'] = fa == fb
    diff = [n for n in fa if n in set(fb) and (a / BANK_DIR / n).read_bytes() != (b / BANK_DIR / n).read_bytes()]
    out['files_compared'] = len(fa)
    out['files_differing'] = diff[:20]
    out['bitwise_identical'] = out['manifest_equal'] and out['file_set_equal'] and not diff
    return out


# ================================================================== no-training guards (shared by adapters)
class TrainingForbidden(RuntimeError):
    pass


def forbid_training_torch(torch):
    """Make optimizer construction/steps, backward passes and torch.save raise inside an inference process."""
    existing = getattr(torch, '_learned_bank_training_guards', None)
    if existing is not None:                     # idempotent: one shared counter set per process
        return existing
    counters = {'optimizer_constructions': 0, 'optimizer_steps': 0, 'backward_passes': 0, 'torch_saves': 0}

    def forbid(name):
        def forbidden(*args, **kwargs):
            counters[name] += 1
            raise TrainingForbidden('learned-bank inference forbids ' + name)
        return forbidden
    torch.optim.Optimizer.__init__ = forbid('optimizer_constructions')
    torch.optim.Optimizer.step = forbid('optimizer_steps')
    torch.Tensor.backward = forbid('backward_passes')
    torch.autograd.backward = forbid('backward_passes')
    torch.save = forbid('torch_saves')
    torch.set_grad_enabled(False)
    torch._learned_bank_training_guards = counters
    return counters

