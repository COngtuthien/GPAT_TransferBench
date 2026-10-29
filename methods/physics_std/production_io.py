"""E04 Physics-STD production I/O: frozen identities, owner-resolved data/RNG bindings, schedule (framework-free).

Imports neither TensorFlow nor pyarrow, so tools/run_e04.py --preflight-only and the laptop tests use exactly the
same constants and checks as the TF1 controller (gpat-m6-e03, Python 3.8). Authorities:
  * configs/methods/e04_physics_std.yaml (frozen, 41861794...) and its snapshot; A3 section 4; A4; A2-06;
  * M6D3d training-graph owner resolution (methods/physics_std/training_contract.py) and the M6D3e realization;
  * the owner production resolution D1-D9 (configs/amendments/e04_production_resolution.yaml, 2026-10-29).
Fidelity: CONTROLLED_ADAPTATION (reporting label "Physics-STD (controlled geometry/depth reconstruction)").
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
METHOD_ID, TRACK = 'E04', 'A_FAIR_COMMON_IDENTITY_FREE'
FIDELITY = 'CONTROLLED_ADAPTATION'
REPORTING_LABEL = 'Physics-STD (controlled geometry/depth reconstruction)'
SCIENTIFIC, QUALIFICATION = 'SCIENTIFIC', 'QUALIFICATION'
SCIENTIFIC_SEEDS = (42, 1337, 2026)
QUALIFICATION_SEED = 60904          # engineering only; never an experiment seed; unused by any earlier harness
QUALIFICATION_PARTS = ('qualification', 'e04_production', 'E04')
QUALIFICATION_CASES = ('det_a', 'det_b', 'terminal_probe', 'failure_probe')
RESUME = 'NOT_QUALIFIED_FRESH_ONLY'

# ---------------------------------------------------------------- frozen identities
CONFIG_PATH = 'configs/methods/e04_physics_std.yaml'
CONFIG_SHA = '418617942ebd87c45cb92bb7dc9ad96a6500a104489a657f9ca3345c3686576a'
SNAPSHOT_PATH = 'frozen_config_snapshot/' + CONFIG_PATH
SPEC_SHA = 'f7d2371682ad9504f59f0c6984e4b1a741f855f1b1f60838f9d60b160281489e'
A4 = ('configs/amendments/e04_a4_execution_resolution.yaml',
      '7579b7177e8b48513bb7a9057e6f700cf9b1f576b5c577887c07bbda9cf35355')
M6D3D = ('configs/amendments/e04_training_graph_resolution.yaml',
         'f62f28c8fe8e090f47436544287243c308fb73b32fb26b1b0e55e6bb6b98e3cb')
M6D3E_DOC = ('docs/spec/amendments/GPAT_TransferBench_v1_0_E04_Training_Graph_Implementation_M6D3e.md', None)
RULING_YAML = ('configs/amendments/e04_production_resolution.yaml',
               '94c9c37bdb23e6e34673fa65e8e4dfc623b961f9e9c10f5e4f0d241840b5505e')
RULING_DOC = ('docs/spec/amendments/GPAT_TransferBench_v1_0_E04_Production_Owner_Resolution_Addendum.md',
              'f0d4a799eb7932deaf9b75cc60cf26427ac2117e120875101145c28b35d1df4c')
LOCK = ('environments/e04_training.lock.json', 'a9629d46c2464f011fb68e0f071ecd4cc21cf0ede5af68faf9ab0d4595fefbdd')
RUNTIME_RECORD = ('environments/e04_training.runtime.json',
                  '3dc558858ab55fd401c63fd964d6653c5aba8482fc3187b08b4bfc78dff10ff9')
E03_LOCK_SHA = 'b2c6c2eadca1b93a0bdf6798990821cb889e199643a5bdb5368ed0b12fdc8c62'
LOGGING_CONTRACT = 'configs/run_logging_v1.yaml'
PAIRS_TRAIN = ('manifests/pairs_train_v1.parquet', 'a5e4fdaef236f15730c7e3885b537e08e995faffbe654167fc940f44bbc75243',
               8838)
SPLIT_MANIFEST_SHA = 'fb9aeb369a124fc96ba855ef2ce269236c4a743fe960e73ab739412c9cb5092d'
QUALIFIED_WORKER_GRAPH_SHA = '7e3e84ccc40f8390819b0121b417e5ebfea78de3efddb1e306d07e70bd4c27e2'   # M6D3e
PREPASS_DIRNAME, PACK_DIRNAME = 'e04_geometry_prepass_v1', 'e04_feed_pack_v1'
PREPASS_MANIFEST_SHA256 = '274a4b0a52077cae18a5477565429410b24479236e8835415cf0eb783f026279'   # frozen CPU pre-pass (D1-D3)
PACK_MANIFEST_SHA256 = 'db1908d56efb23758fee49f88d2158374b85572a213a269419881bd02037520b'      # E04 feed pack v1
PYTHON_EXECUTABLE = '/home/student20261/miniconda3/envs/gpat-m6-e03/bin/python'
ENVIRONMENT_NAME = 'gpat-m6-e03'
GPU_MODEL, GPU_UUID = 'NVIDIA GeForce RTX 3090', 'GPU-b722cd9d-c9fa-223c-915e-cc09d9b862b7'
LOCK_DRIVER, OBSERVED_DRIVER = '595.84', '580.178.04'
DRIVER_POLICY = 'OWNER_APPROVED_DRIVER_VERSION_ONLY_RUNTIME_COMPATIBILITY (E04 D9, conditional narrow scope)'
PRODUCTION_FILES = ('methods/physics_std/production_io.py', 'methods/physics_std/production_runner.py',
                    'methods/physics_std/feed_pack.py', 'methods/physics_std/prepass.py', 'tools/run_e04.py',
                    'methods/physics_std/training_graph.py', 'methods/physics_std/network.py',
                    'methods/physics_std/losses.py', 'methods/physics_std/trace_transfer.py',
                    'methods/physics_std/training_contract.py')

# ---------------------------------------------------------------- frozen schedule (config + A4 + M6D3d)
IMAGE_SIZE, MAP_SIZE, BATCH, PAIRS_PER_ITERATION, ROWS = 256, 32, 8, 4, 8838
ITERATIONS = 150000
LR0, LR_FACTOR, LR_EVERY, D_LR_MULTIPLIER = 5e-5, 10, 45000, 0.5
CHECKPOINT_RULE, TERMINAL_STATE = 'BASELINE_FINAL_STATE_V1', 'iteration_150000'
CHECKPOINT_EVERY = 15000            # implementation-defined periodic cadence (config: periodic_plus_terminal)
CHECKPOINT_SUFFIXES = ('.data-00000-of-00001', '.index', '.meta')
PERIODIC_REASON = 'Periodic/terminal retention'           # PhysicsSTDAdapter.validate_checkpoint_event
TERMINAL_PROBE_LABEL = 'QUALIFICATION_TERMINAL_SEMANTICS_PROBE_NOT_150000_ITERATIONS'
QUALIFICATION_LABELS = ('QUALIFICATION_ONLY', 'NOT_SCIENTIFIC', 'NOT_A_TRAINED_E04_CHECKPOINT')
VAL_NULL_REASON = 'NO_VAL_PROTOCOL_IN_SOURCE_OR_FROZEN_CONTRACT'
STEP_LOSSES = {1: ('loss_G_step1', 'L_depth', 'L_G', 'L_P', 'L_R', 'P_positive', 'P0_negative_prior'),
               2: ('loss_D', 'L_D'), 3: ('loss_G_step3', 'L_S', 'L_H')}

# ---------------------------------------------------------------- launch environment (E04 training lock)
PRECISION_ENV = {
    'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'CUDA_VISIBLE_DEVICES': '0', 'NVIDIA_TF32_OVERRIDE': '0',
    'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONNOUSERSITE': '1', 'TF_CUDNN_DETERMINISTIC': '1', 'TF_DETERMINISTIC_OPS': '1',
    'TF_ENABLE_AUTO_MIXED_PRECISION': '0', 'TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE': '0',
    'TF_ENABLE_CUBLAS_TENSOR_OP_MATH_FP32': '0', 'TF_ENABLE_CUDNN_RNN_TENSOR_OP_MATH_FP32': '0',
    'TF_ENABLE_CUDNN_TENSOR_OP_MATH_FP32': '0', 'TF_ENABLE_ONEDNN_OPTS': '0', 'TF_XLA_FLAGS': '--tf_xla_auto_jit=0'}

REFUSED_OPTIONS = tuple('--' + x for x in (
    'iterations', 'total-iterations', 'steps', 'max-steps', 'batch', 'batch-size', 'lr', 'learning-rate', 'decay',
    'lr-schedule', 'optimizer', 'beta1', 'beta2', 'epsilon', 'alpha', 'beta', 'lambda', 'loss', 'losses', 'augment',
    'augmentation', 'flip', 'brightness', 'pairs', 'pair-manifest', 'split', 'pack', 'feed-pack', 'prepass',
    'geometry', 'landmarks', 'faces', 'faces-root', 'checkpoint', 'checkpoint-rule', 'checkpoint-every', 'ckpt',
    'output', 'output-root', 'run-root', 'run-dir', 'runtime-root', 'root', 'fidelity', 'test', 'test-split', 'val',
    'eval', 'evaluate', 'best', 'select', 'resume', 'restore', 'from-checkpoint', 'init-checkpoint', 'pretrained',
    'limit', 'qualification', 'qualification-seed', 'source', 'config', 'pair-seed', 'hard-seed', 'seed-offset',
    'workers', 'device'))


class E04Stop(RuntimeError):
    """A production gate failed: STOP_AND_REPORT, never retried or worked around."""


def require(value, message):
    if not value:
        raise E04Stop('E04 STOP: ' + message)


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(str(path), 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------- seeds and streams (D4, D7)
def validate_mode_seed(mode, seed):
    require(mode in (SCIENTIFIC, QUALIFICATION), 'mode')
    require(type(seed) is int, 'integer seed')
    if mode == SCIENTIFIC:
        require(seed in SCIENTIFIC_SEEDS, 'scientific seed must be one of 42, 1337, 2026')
    else:
        require(seed == QUALIFICATION_SEED and seed not in SCIENTIFIC_SEEDS, 'qualification seed must be 60904')


def derive_seed(domain, seed):
    """1 + (u32_be(SHA256("GPAT_TRANSFERBENCH|E04|<domain>|<seed>")[0:4]) mod 2147483647); D4 PAIR_ORDER, D7
    HARD_SAMPLE. The two values are not guaranteed distinct; correctness relies only on separate RNG objects."""
    require(domain in ('PAIR_ORDER', 'HARD_SAMPLE'), 'RNG domain')
    d = hashlib.sha256(('GPAT_TRANSFERBENCH|E04|%s|%d' % (domain, seed)).encode('utf-8')).digest()
    return 1 + int.from_bytes(d[:4], 'big') % 2147483647


def rng_plan(seed):
    return {'tf_graph_seed': seed, 'pair_order_seed': derive_seed('PAIR_ORDER', seed),
            'hard_sample_seed': derive_seed('HARD_SAMPLE', seed),
            'generators': 'two separate numpy.random.RandomState (legacy MT19937) objects',
            'pair_rule': 'pass p = rng.permutation(8838) over pairs_train_v1 stored (ascending pair_id) order; '
                         'concatenated; iteration t = rows [4t, 4t+4); no drop / padding / deduplication',
            'hard_rule': 'before Step 1: u = uniform(0,1,(4,1,1,3)) -> float32 scales; k = uniform(0,1,(4,1,1,1)) '
                         '>= 0.5 -> float32 keep; the same arrays for Steps 1-3',
            'process_global_python_numpy': 'seeded with the run seed; never used by the feed'}


class PairStream:
    """D4: one RandomState(PAIR_ORDER seed); successive full permutations; 4 rows per iteration."""

    def __init__(self, seed):
        self.rng = np.random.RandomState(derive_seed('PAIR_ORDER', seed))
        self.buffer, self.passes, self.drawn = [], 0, 0

    def next_rows(self):
        while len(self.buffer) < PAIRS_PER_ITERATION:
            self.buffer.extend(int(x) for x in self.rng.permutation(ROWS))
            self.passes += 1
        rows, self.buffer = self.buffer[:PAIRS_PER_ITERATION], self.buffer[PAIRS_PER_ITERATION:]
        start = self.drawn
        self.drawn += PAIRS_PER_ITERATION
        return rows, [(start + j) // ROWS for j in range(PAIRS_PER_ITERATION)]


class HardStream:
    """D7: one RandomState(HARD_SAMPLE seed); 16 uniform doubles per iteration, the qualified realization."""

    def __init__(self, seed):
        self.rng = np.random.RandomState(derive_seed('HARD_SAMPLE', seed))
        self.iterations = 0

    def next(self):
        u = self.rng.uniform(0, 1, (4, 1, 1, 3))
        k = self.rng.uniform(0, 1, (4, 1, 1, 1))
        self.iterations += 1
        return u.astype(np.float32), (k >= .5).astype(np.float32)


def stream_hashes(seed, iterations=ITERATIONS):
    """Full-stream sha256 in the approved vector convention (tools/e04_rng_test_vectors.py)."""
    p, h = PairStream(seed), HardStream(seed)
    rows, hh = [], hashlib.sha256()
    for _ in range(iterations):
        rows.extend(p.next_rows()[0])
        s, k = h.next()
        hh.update(s.tobytes())
        hh.update(k.tobytes())
    return {'pair_sha256_int32_le': hashlib.sha256(np.asarray(rows, '<i4').tobytes()).hexdigest(),
            'hard_sha256': hh.hexdigest(), 'pair_passes_generated': p.passes}


# ---------------------------------------------------------------- schedule
def expected_learning_rates(iteration):
    g = LR0 * LR_FACTOR ** (-(iteration // LR_EVERY))
    return g, g * D_LR_MULTIPLIER


def checkpoint_prefix(iteration):
    return 'iteration_%06d' % iteration


def scientific_checkpoints():
    return list(range(CHECKPOINT_EVERY, ITERATIONS + 1, CHECKPOINT_EVERY))


def checkpoint_kind(iteration):
    selected = iteration == ITERATIONS
    return ('selected' if selected else 'periodic', selected, CHECKPOINT_RULE if selected else PERIODIC_REASON)


# ---------------------------------------------------------------- roots
def run_root(runtime_root, mode, seed, case=None, run_id=None):
    validate_mode_seed(mode, seed)
    rt = Path(runtime_root)
    if mode == SCIENTIFIC:
        require(case is None and run_id is None, 'scientific roots carry no case/run_id')
        return rt / 'runs' / 'm6' / METHOD_ID / ('seed_%d' % seed)
    require(case in QUALIFICATION_CASES and run_id, 'qualification case/run_id')
    root = rt.joinpath(*QUALIFICATION_PARTS) / ('%s-q%d-%s' % (case, seed, run_id))
    require('runs' not in root.relative_to(rt).parts, 'qualification roots live outside runs/')
    return root


def pack_root(runtime_root):
    return Path(runtime_root) / 'data' / 'processed' / PACK_DIRNAME


def prepass_root(runtime_root):
    return Path(runtime_root) / 'data' / 'processed' / PREPASS_DIRNAME


EXEC_CONFIG = 'configs/execution/m5_gpu_3090.yaml'


def storage_from_exec_config(rel=EXEC_CONFIG, root=ROOT):
    """The E03 storage resolution (unchanged module; same containment rules), with E04's pack root."""
    from methods.stdn.production_io import storage_from_exec_config as e03_storage
    s = e03_storage(rel, root)
    s['pack_root'] = str(pack_root(s['runtime_root']))
    s['prepass_root'] = str(prepass_root(s['runtime_root']))
    return s


# ---------------------------------------------------------------- contract / identities
def verify_contract(root=ROOT):
    """Byte identities of every frozen authority and the owner resolution; returns the parsed frozen config."""
    root = Path(root)
    raw = (root / CONFIG_PATH).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == CONFIG_SHA and raw == (root / SNAPSHOT_PATH).read_bytes(),
            'frozen E04 config sha256 / snapshot identity')
    cfg = yaml.safe_load(raw)
    for rel, digest in (A4, M6D3D, RULING_YAML, RULING_DOC, LOCK, RUNTIME_RECORD):
        require(sha256_file(root / rel) == digest, rel + ' sha256')
    import importlib.util
    spec = importlib.util.spec_from_file_location('e04_training_contract', root / 'methods/physics_std/training_contract.py')
    tc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tc)
    contract = tc.load_contract(root)                     # M6D3d digests + immutable inputs + snapshot
    require(contract['status'] == 'E04_TRAINING_GRAPH_CONTRACT_RESOLVED', 'M6D3d contract')
    require(cfg['spec_sha256'] == SPEC_SHA and cfg['method_id'] == METHOD_ID and cfg['fidelity_class'] == FIDELITY,
            'config identity')
    require(cfg['seeds']['experiment_seeds'] == list(SCIENTIFIC_SEEDS), 'frozen seeds')
    t, o = cfg['training'], cfg['optimizer']
    require((t['input_resolution'], t['total_iterations'], t['batch_size']) == (IMAGE_SIZE, ITERATIONS, BATCH),
            'frozen training values')
    require(o['learning_rate'] == LR0 and o['lr_schedule'] == {'type': 'step_divide', 'factor': LR_FACTOR,
                                                               'every_iterations': LR_EVERY}, 'frozen optimizer')
    c = cfg['checkpoint']
    require(c['rule'] == CHECKPOINT_RULE and c['terminal_state'] == TERMINAL_STATE and
            c['cadence'] == 'implementation_defined_periodic_plus_terminal' and c['selection_uses_val'] is False and
            c['selection_uses_test'] is False and c['selection_scope'] == 'WITHIN_SEED', 'checkpoint rule')
    require(cfg['data']['splits']['TEST']['allowed'] is False and cfg['data']['pair_manifest_sha256'] == PAIRS_TRAIN[1]
            and cfg['data']['split_manifest_sha256'] == SPLIT_MANIFEST_SHA, 'data firewall')
    require(contract['optimization']['generator_learning_rate_multiplier'] == 1.0 and
            contract['optimization']['discriminator_learning_rate_multiplier'] == D_LR_MULTIPLIER and
            contract['optimization']['step_order'] == ['step1_generator_eq23', 'step2_discriminator_L_D',
                                                       'step3_generator_eq24'], 'M6D3d schedule')
    ruling = json.loads((root / RULING_YAML[0]).read_text())
    require(ruling['status'] == 'OWNER_APPROVED' and ruling['approved'] is True and
            ruling['scientific_seed_execution_authorized'] is False, 'owner resolution status')
    d = ruling['decisions']
    require(all(d['D%d' % i]['owner_status'] == 'APPROVED' for i in range(1, 9)) and
            d['D9']['owner_status'] == 'APPROVED_CONDITIONAL_NARROW_SCOPE', 'D1-D9 approval')
    for s in SCIENTIFIC_SEEDS:
        require(derive_seed('PAIR_ORDER', s) != 0 and derive_seed('HARD_SAMPLE', s) != 0, 'seed derivation')
    require((derive_seed('PAIR_ORDER', 42), derive_seed('HARD_SAMPLE', 42), derive_seed('PAIR_ORDER', 1337),
             derive_seed('HARD_SAMPLE', 1337), derive_seed('PAIR_ORDER', 2026), derive_seed('HARD_SAMPLE', 2026)) ==
            (679869707, 320493565, 1349475396, 35554372, 692890966, 1053992267), 'approved seed table')
    require(d['D4']['draws'] == ITERATIONS * PAIRS_PER_ITERATION and d['D6']['val_fields'].endswith(VAL_NULL_REASON),
            'D4/D6 bindings')
    return cfg


def production_file_hashes(root=ROOT):
    return {rel: sha256_file(Path(root) / rel) for rel in PRODUCTION_FILES if (Path(root) / rel).is_file()}


def identities(root=ROOT):
    return {'config_sha256': CONFIG_SHA, 'spec_sha256': SPEC_SHA, 'a4_sha256': A4[1], 'm6d3d_sha256': M6D3D[1],
            'ruling_yaml_sha256': RULING_YAML[1], 'ruling_doc_sha256': RULING_DOC[1],
            'environment_lock_sha256': LOCK[1], 'runtime_record_sha256': RUNTIME_RECORD[1],
            'e03_environment_lock_sha256': E03_LOCK_SHA,
            'logging_contract_sha256': sha256_file(Path(root) / LOGGING_CONTRACT),
            'pairs_train_v1_sha256': PAIRS_TRAIN[1], 'split_manifest_sha256': SPLIT_MANIFEST_SHA,
            'prepass_manifest_sha256': PREPASS_MANIFEST_SHA256, 'pack_manifest_sha256': PACK_MANIFEST_SHA256,
            'qualified_worker_graph_sha256': QUALIFIED_WORKER_GRAPH_SHA, 'fidelity': FIDELITY,
            'production_files_sha256': production_file_hashes(root)}


def launch_environment_errors(environ, seed):
    expected = dict(PRECISION_ENV, PYTHONHASHSEED=str(seed))
    return sorted(k for k, v in expected.items() if environ.get(k) != v)


def refused_arguments(argv):
    out = []
    for a in argv:
        name = a.split('=', 1)[0]
        if name in REFUSED_OPTIONS or (name.startswith('--') and name not in ('--seed', '--execution-config',
                                                                              '--preflight-only')):
            out.append(name)
    return sorted(set(out))


def environment_differences(lock, runtime_record, live):
    """E04 training lock identity except the NVIDIA driver version (D9, conditional narrow scope)."""
    diff = []
    if live.get('python') != lock['python']:
        diff.append('python')
    if live.get('python_executable_sha256') != lock['python_executable_sha256']:
        diff.append('python_executable_sha256')
    if live.get('environment_path') != lock['environment_path']:
        diff.append('environment_path')
    if live.get('package_versions') != lock['package_versions']:
        diff.append('package_versions')
    if live.get('native_runtime') != lock['native_runtime']:
        diff.append('native_runtime')
    if live.get('precision_env') != lock['precision_env']:
        diff.append('precision_env')
    if live.get('session_config') != runtime_record['session_config']:
        diff.append('session_config')
    old, new = (x.split(', ') for x in (lock['gpu'], live.get('gpu', '')))
    if len(old) != 3 or len(new) != 3 or (old[0], old[2]) != (new[0], new[2]) or new[2] != GPU_UUID:
        diff.append('gpu(model/uuid)')
    driver = {'lock_driver_version': old[1] if len(old) == 3 else None,
              'observed_driver_version': new[1] if len(new) == 3 else None, 'driver_policy': DRIVER_POLICY,
              'gpu_model': new[0] if new else None, 'gpu_uuid': new[2] if len(new) == 3 else None}
    if driver['observed_driver_version'] != driver['lock_driver_version'] and \
            (driver['lock_driver_version'], driver['observed_driver_version']) != (LOCK_DRIVER, OBSERVED_DRIVER):
        diff.append('gpu_driver(version outside the owner-approved pair)')
    return sorted(diff), driver
