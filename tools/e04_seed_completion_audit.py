"""E04 seed closure: read-only completion audit of one E04 Physics-STD scientific run (CONTROLLED_ADAPTATION, track
A_FAIR_COMMON_IDENTITY_FREE, scientific launch authority 18ba00e, production qualification b0f0ffd under the
owner-approved D12 procedural reproducibility ruling c60481b).

Three phases (the two collectors never write anything; this file must stay parseable by Python 3.8):

  collect-tf  (GPU host, gpat-m6-e03 Python 3.8 / TF 1.15.5, exact launch environment of the audited seed,
              `python -I -B - collect-tf --seed <s> --repo <gpu repo> < this file`) -- prints one JSON document.
              Re-verifies the frozen contract, identities and the pinned pre-pass / feed-pack manifests with the
              production verifiers, re-reads the live environment against the E04 training lock (D9: driver version
              only), rebuilds the qualified M6D3e construction in an unexecuted graph (no session), opens all ten TF
              Saver checkpoints with tf.train.NewCheckpointReader (no session, no restore) and records every variable's
              finiteness, the counters (global_iteration, minibatch_phase, both Adam beta powers) and the runner's
              state-fingerprint convention, computes the production stream hashes, and recomputes the dense sampling
              offsets of a fixed iteration sample from the feed pack geometry (no image is opened).
  collect     (GPU host, a Python 3.11 env with pyarrow -- gpat-m6-e05, CUDA_VISIBLE_DEVICES="",
              `python -I -B - collect --seed <s> --repo <gpu repo> < this file`) -- prints one JSON document.
              Lists the run root (path/size/sha256) before and after, scans every metrics.jsonl line against the frozen
              schedule (run_start, 10 x (15000 iterations, checkpoint event)), re-derives every pair row and every hard
              draw from the D4/D7 rule text with numpy RandomState (independent of the production module), recomputes
              the hard and depth input sha256 of every iteration from those draws and the hash-verified feed pack, maps
              every consumed face through split_v1 and the M2 accounting, compares the stat-only faces_256 atime
              snapshots (~/workdir/e04_s<seed>_atime_{pre,post}.tsv) and reads the durable host markers, wtmp and the
              journal boot list. No face image is opened.
  build       (repository, stdlib only) -- gates both documents and writes outputs/audit/E04_SEED<s>_{COMPLETION.json,
              COMPLETION.md, CHECKPOINT_MANIFEST.txt, EVIDENCE_SHA256.txt}. COMPLETION.json embeds both collected
              documents so the build can be re-derived on any host.

Process-exit evidence (E04 launch contract, E04_SEED42_LAUNCH): the launch wrapper writes the runner's exit status to
~/workdir/e04_s<seed>.exit and the end time to e04_s<seed>.end_time immediately after the runner returns. Completion is
proven by that durable marker together with the runner's own completion record, the run summary and the run root; the
live tmux pane is recorded but is not required (a later host reboot removes the tmux server). Boot continuity over the
whole run window is a separate gate, so a reboot during training cannot pass.

No training, restore, selection, VAL, TEST, bank or downstream code is run. D12: GPU numerical variation between
independent executions is disclosed and permitted; it never excuses a missing/invalid update, non-finite state or a
corrupted checkpoint, all of which are gated here.
"""
import argparse
import hashlib
import json
import math
import os
import struct
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

CLASSIFICATION = 'SCIENTIFIC_TRAINING_COMPLETION_AUDIT'
RUNTIME = '/home/student20261/workdir/GPAT_TransferBench_runtime'
WORKDIR = '/home/student20261/workdir'
AUTHORITY = '18ba00e94813f005876c8d6235b0b848226b2c1d'                # E04 scientific launch authority (all seeds)
QUALIFICATION_AUTHORITY = 'b0f0ffd06b26e1c2d542df9e09101417ee3b687c'  # E04_PRODUCTION_QUALIFIED_..., 52/52 (D12)
D12_RULING_COMMIT = 'c60481b91922c1c72c73a603219585ec4a78c1e4'
D12_ACTIVATION_COMMIT = 'c676459'
D12_CRITERIA_SHA = '9f1a2471f0c9e6ad24b9cc4727779361b763ec84b7e43dc221fc6b411c199d0d'
CONFIG_SHA = '418617942ebd87c45cb92bb7dc9ad96a6500a104489a657f9ca3345c3686576a'
RUN_IDS = {42: '40befa6004aee85c', 1337: 'a42fdd2a2accaf0a', 2026: 'e7100eb9a5ab0537'}
DERIVED = {42: (679869707, 320493565), 1337: (1349475396, 35554372), 2026: (692890966, 1053992267)}
FIDELITY = 'CONTROLLED_ADAPTATION'
REPORTING_LABEL = 'Physics-STD (controlled geometry/depth reconstruction)'
PINNED_FILES = {
    'outputs/audit/E04_RNG_TEST_VECTORS.json': '35eded87f8da406752bd10337cf626bb4a003993ccd6c6492219f7a8a9bb2eac',
    'outputs/audit/E04_PRODUCTION_QUALIFICATION_D12.json':
        '53db7557cb6205f477f5bc9c5ae39271115b4241e897d6e36a7d6c9adf659c50',
    'configs/amendments/e04_d12_procedural_reproducibility.yaml':
        'a22093b1dbdfa3ea5cc3534e7deaa387b11f39d61405bb42ce5541bef9387754',
    'configs/methods/e04_physics_std.yaml': CONFIG_SHA,
    'frozen_config_snapshot/configs/methods/e04_physics_std.yaml': CONFIG_SHA,
    'environments/e04_training.lock.json': 'a9629d46c2464f011fb68e0f071ecd4cc21cf0ede5af68faf9ab0d4595fefbdd',
    'environments/e04_training.runtime.json': '3dc558858ab55fd401c63fd964d6653c5aba8482fc3187b08b4bfc78dff10ff9',
    'configs/amendments/e04_production_resolution.yaml':
        '94c9c37bdb23e6e34673fa65e8e4dfc623b961f9e9c10f5e4f0d241840b5505e',
    'manifests/pairs_train_v1.parquet': 'a5e4fdaef236f15730c7e3885b537e08e995faffbe654167fc940f44bbc75243',
    'manifests/split_v1.parquet': 'fb9aeb369a124fc96ba855ef2ce269236c4a743fe960e73ab739412c9cb5092d',
}
EXPECTED = {
    'method_id': 'E04', 'host': 'nd-System-Product-Name', 'fidelity': FIDELITY, 'resume': 'NOT_QUALIFIED_FRESH_ONLY',
    'checkpoint_rule': 'BASELINE_FINAL_STATE_V1', 'environment_name': 'gpat-m6-e03',
    'python_executable': '/home/student20261/miniconda3/envs/gpat-m6-e03/bin/python',
    'gpu_model': 'NVIDIA GeForce RTX 3090', 'gpu_uuid': 'GPU-b722cd9d-c9fa-223c-915e-cc09d9b862b7',
    'identities': {
        'a4_sha256': '7579b7177e8b48513bb7a9057e6f700cf9b1f576b5c577887c07bbda9cf35355',
        'config_sha256': CONFIG_SHA,
        'e03_environment_lock_sha256': 'b2c6c2eadca1b93a0bdf6798990821cb889e199643a5bdb5368ed0b12fdc8c62',
        'environment_lock_sha256': 'a9629d46c2464f011fb68e0f071ecd4cc21cf0ede5af68faf9ab0d4595fefbdd',
        'fidelity': FIDELITY,
        'logging_contract_sha256': 'd33622947d1951d71bbda4cf86b96c6500943d354fe607beb6cbc47f51c3de15',
        'm6d3d_sha256': 'f62f28c8fe8e090f47436544287243c308fb73b32fb26b1b0e55e6bb6b98e3cb',
        'pack_manifest_sha256': 'db1908d56efb23758fee49f88d2158374b85572a213a269419881bd02037520b',
        'pairs_train_v1_sha256': 'a5e4fdaef236f15730c7e3885b537e08e995faffbe654167fc940f44bbc75243',
        'prepass_manifest_sha256': '274a4b0a52077cae18a5477565429410b24479236e8835415cf0eb783f026279',
        'production_files_sha256': {
            'methods/physics_std/feed_pack.py': '1472abf97e90e1e2d26031aae1eb9d7f8ac7c9481e56eedb32fc2824d576e9de',
            'methods/physics_std/losses.py': '94476c2b4bddc9e9692d478e271c145c1d595b7d2410fd2773606dfcc0c3ff02',
            'methods/physics_std/network.py': '9d691809d487c5c9c27b5c9e08389225f606f8d72f1017d2991cd71e6214da21',
            'methods/physics_std/prepass.py': '3d424ace8d12b961094263a401b22383f61e71ccabee87889a5293c2790f807b',
            'methods/physics_std/production_io.py': 'a6a17f948283a74f71048b8e8412f2f8c4fab6fd98fe39fa7e3a1fed95c205c0',
            'methods/physics_std/production_runner.py':
                '68047bd5b08476755541f222b86645cda87e1e3aeb0405c18268763b693c85bd',
            'methods/physics_std/trace_transfer.py': '8152a7643858cb523344dcfa8fbe1a84168082d545d6ca68f7435cf2a06882cd',
            'methods/physics_std/training_contract.py':
                'f5c7962699320014a9711605a3fddc4d8d673f6c91fd2df9af5d216f628b36bd',
            'methods/physics_std/training_graph.py': 'af17ab2789f203834d9fd9121b357dd3e44e49689fccac6be25024926bf69f51',
            'tools/run_e04.py': 'c46a35748cf393dff8fb85fd39405d8890a6f1a38e909589b54f04e996737ffb'},
        'qualified_worker_graph_sha256': '7e3e84ccc40f8390819b0121b417e5ebfea78de3efddb1e306d07e70bd4c27e2',
        'ruling_doc_sha256': 'f0d4a799eb7932deaf9b75cc60cf26427ac2117e120875101145c28b35d1df4c',
        'ruling_yaml_sha256': '94c9c37bdb23e6e34673fa65e8e4dfc623b961f9e9c10f5e4f0d241840b5505e',
        'runtime_record_sha256': '3dc558858ab55fd401c63fd964d6653c5aba8482fc3187b08b4bfc78dff10ff9',
        'spec_sha256': 'f7d2371682ad9504f59f0c6984e4b1a741f855f1b1f60838f9d60b160281489e',
        'split_manifest_sha256': 'fb9aeb369a124fc96ba855ef2ce269236c4a743fe960e73ab739412c9cb5092d'},
    'structure': {'D': 72, 'G': 72, 'bn_update_ops': {'1': [30, 32], '2': [30, 32], '3': [30, 30]},
                  'construction_operations': 11315,
                  'construction_sha256_without_seed_attrs':
                      '61f1ed5868d6774abb809915c2e9dfd0a50c5313064672389033ca46dd951e1d',
                  'global_variables': 500, 'shared_G_slots': True, 'trainable_parameters': 1763902,
                  'trainable_variables': 144},
    'pack_counts': {'live': 3830, 'pairs': 8838, 'samples': 12668, 'spoof': 8838},
}
DRIVER_RULING = {'lock_driver_version': '595.84', 'observed_driver_version': '580.178.04',
                 'driver_policy': 'OWNER_APPROVED_DRIVER_VERSION_ONLY_RUNTIME_COMPATIBILITY (E04 D9, conditional '
                                  'narrow scope)',
                 'scope': ('only the NVIDIA driver version may differ; GPU model, GPU UUID and every other lock field '
                           'are exact gates; this is not evidence that arbitrary driver changes are acceptable')}
PRECISION_ENV = {
    'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'CUDA_VISIBLE_DEVICES': '0', 'NVIDIA_TF32_OVERRIDE': '0',
    'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONNOUSERSITE': '1', 'TF_CUDNN_DETERMINISTIC': '1', 'TF_DETERMINISTIC_OPS': '1',
    'TF_ENABLE_AUTO_MIXED_PRECISION': '0', 'TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE': '0',
    'TF_ENABLE_CUBLAS_TENSOR_OP_MATH_FP32': '0', 'TF_ENABLE_CUDNN_RNN_TENSOR_OP_MATH_FP32': '0',
    'TF_ENABLE_CUDNN_TENSOR_OP_MATH_FP32': '0', 'TF_ENABLE_ONEDNN_OPTS': '0', 'TF_XLA_FLAGS': '--tf_xla_auto_jit=0'}
ITERATIONS, EVERY, ROWS, PAIRS, BATCH = 150000, 15000, 8838, 4, 8
CHECKPOINTS = list(range(EVERY, ITERATIONS + 1, EVERY))
LR0, LR_FACTOR, LR_EVERY = 5e-5, 10, 45000
TOL_LR = 1e-6                                                   # relative; float32 read-back of the TF schedule
N_VARIABLES = 500
SUFFIXES = ('.data-00000-of-00001', '.index', '.meta')
PERIODIC_REASON = 'Periodic/terminal retention'
STEP_LOSSES = {'step1': ['L_G', 'L_P', 'L_R', 'L_depth', 'P0_negative_prior', 'P_positive', 'loss_G_step1'],
               'step2': ['L_D', 'loss_D'], 'step3': ['L_H', 'L_S', 'loss_G_step3']}
VAL_NULL_REASON = 'NO_VAL_PROTOCOL_IN_SOURCE_OR_FROZEN_CONTRACT'
REASONS = {'epoch': 'PhySTD schedule is iteration-based (150000 complete minibatch iterations); no epoch exists',
           'train_metrics': 'PhySTD exposes no TRAIN metric beyond its losses; none is fabricated',
           'val_losses': VAL_NULL_REASON, 'val_metrics': VAL_NULL_REASON}
ITERATION_KEYS = sorted(['epoch', 'global_step', 'gpu_memory_bytes', 'hard_keep', 'hard_scales', 'input_sha256',
                         'iteration', 'learning_rate', 'learning_rates', 'missing_field_reasons', 'pair_ids',
                         'pair_passes', 'pair_rows', 'record_type', 'step_seconds', 'train_losses', 'train_metrics',
                         'utc', 'val_losses', 'val_metrics', 'wall_clock_seconds'])
INPUT_KEYS = ['depth', 'hard_keep', 'hard_scales', 'offsets', 'rgb']
SMALL_FILES = ('run_manifest.json', 'run_summary.json', 'checkpoint_index.json', 'resolved_config.yaml',
               'metrics.jsonl', 'stdout.log', 'stderr.log', 'generation_log.jsonl')
RUN_ROOT_FILES = len(SMALL_FILES) + len(CHECKPOINTS) * len(SUFFIXES) + 1
PANE_FAIL_PATTERNS = ('Traceback', 'ResourceExhausted', 'out of memory', 'OOM', 'E04 STOP', 'REFUSED', 'Killed',
                      'Segmentation fault', 'core dumped', 'NaN', 'non-finite', 'Error')
LOG_PATTERNS = ('Traceback', 'Error', 'error', 'OutOfMemory', 'out of memory', 'Killed', 'nan', 'inf', 'Warning')
MARKER_SLACK = 60                                # s: wrapper markers written right after the runner's end_utc
BLOCK = 1500                                     # publication summary block (100 blocks)
OFFSET_SAMPLE = sorted(set(list(range(64)) + [c - 1 for c in CHECKPOINTS] + [c for c in CHECKPOINTS[:-1]]))
FLT_MIN = 1.1754943508222875e-38                 # smallest normal float32
BETA1, BETA2 = 0.9, 0.999
BETA_NAMES = {'G': ('beta1_power', 'beta2_power'), 'D': ('beta1_power_1', 'beta2_power_1')}


def paths(seed):
    w = '%s/e04_s%d' % (WORKDIR, seed)
    return {'run_root': '%s/runs/m6/E04/seed_%d' % (RUNTIME, seed), 'tmux': 'e04_physics_std_seed%d' % seed,
            'exit': w + '.exit', 'command': w + '.command', 'launch_time': w + '.launch_time',
            'end_time': w + '.end_time', 'pane_log': w + '.pane.log', 'preflight': w + '_preflight.json',
            'atime_pre': w + '_atime_pre.tsv', 'atime_post': w + '_atime_post.tsv'}


def evidence(seed):
    a = 'outputs/audit/E04_SEED%d_' % seed
    return {'completion': a + 'COMPLETION.json', 'manifest': a + 'CHECKPOINT_MANIFEST.txt',
            'report': a + 'COMPLETION.md', 'sha256': a + 'EVIDENCE_SHA256.txt'}


def command(seed):
    return '%s tools/run_e04.py --seed %d --execution-config configs/execution/m5_gpu_3090.yaml' % (
        EXPECTED['python_executable'], seed)


def wrapper_invocation(seed):
    return 'python -u -B tools/run_e04.py --seed %d --execution-config configs/execution/m5_gpu_3090.yaml' % seed


def run_id(seed, commit=AUTHORITY):
    return hashlib.sha256(('E04|%d|%s|%s' % (seed, CONFIG_SHA, commit)).encode('utf-8')).hexdigest()[:16]


def derive_seed(domain, seed):
    """D4/D7 rule text: 1 + (u32_be(SHA256("GPAT_TRANSFERBENCH|E04|<domain>|<seed>")[0:4]) mod 2147483647)."""
    d = hashlib.sha256(('GPAT_TRANSFERBENCH|E04|%s|%d' % (domain, seed)).encode('utf-8')).digest()
    return 1 + int.from_bytes(d[:4], 'big') % 2147483647


def expected_lr(t):
    g = LR0 * LR_FACTOR ** (-(t // LR_EVERY))
    return g, g * 0.5


def checkpoint_kind(it):
    sel = it == ITERATIONS
    return ('selected' if sel else 'periodic', sel, EXPECTED['checkpoint_rule'] if sel else PERIODIC_REASON)


def prefix(it):
    return 'checkpoints/iteration_%06d' % it


def f32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]


def beta_power_sequence(beta, n):
    """TF1 Adam beta power after k = 0..n applications: starts at float32(beta), multiplied by float32(beta) in float32
    (IEEE round-to-nearest; the float64 product of two float32 values is exact) after every apply_gradients."""
    b = f32(beta)
    out, p = [b], b
    for _ in range(n):
        p = f32(p * b)
        out.append(p)
    return out


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(str(path), 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def utc(ts):
    return datetime.fromtimestamp(ts, timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def now_utc():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def epoch_seconds(iso):
    return datetime.strptime(iso, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc).timestamp()


def tree_snapshot(root):
    out = []
    for dirpath, dirnames, filenames in os.walk(str(root)):
        dirnames.sort()
        for name in sorted(filenames):
            p = Path(dirpath) / name
            st = p.stat()
            out.append([str(p.relative_to(root)), st.st_size, st.st_mtime_ns])
    return sorted(out)


def sha_listing(root):
    """LC_ALL=C (byte-order) sorted `<path> <size> <sha256>` lines; digest = sha256 of the newline-terminated lines."""
    lines = ['%s %d %s' % (p, s, sha256_file(Path(root) / p)) for p, s, _ in tree_snapshot(root)]
    return sha256_bytes(''.join(x + '\n' for x in lines).encode()), len(lines), lines


def finite(v):
    return isinstance(v, float) and math.isfinite(v)


def git_state():
    def git(*a):
        return subprocess.run(['git'] + list(a), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                              universal_newlines=True)
    anc = {name: git('merge-base', '--is-ancestor', c, 'HEAD').returncode == 0
           for name, c in (('qualification', QUALIFICATION_AUTHORITY), ('d12_ruling', D12_RULING_COMMIT),
                           ('launch_authority', AUTHORITY))}
    return {'head': git('rev-parse', 'HEAD').stdout.strip(), 'branch': git('branch', '--show-current').stdout.strip(),
            'status_porcelain': git('status', '--porcelain').stdout, 'ancestors_of_head': anc}


def iteration_records(path, wanted=None):
    """Yields (line number, record) for every iteration record (optionally only iterations in `wanted`)."""
    with open(str(path), 'rb') as f:
        for n, line in enumerate(f, 1):
            if wanted is not None and b'"record_type": "epoch"' not in line:
                continue
            r = json.loads(line)
            if r.get('record_type') == 'epoch' and (wanted is None or r.get('iteration') in wanted):
                yield n, r


# ============================================================================ collect-tf (GPU host, gpat-m6-e03)
def collect_tf(seed, repo):
    sys.path.insert(0, str(repo))
    os.chdir(str(repo))
    root = Path(paths(seed)['run_root'])
    before = tree_snapshot(root)
    import platform
    import numpy as np
    from methods.physics_std import feed_pack
    from methods.physics_std import production_io as pio
    from methods.physics_std import production_runner as pr
    from methods.physics_std.trace_transfer import dense_offsets
    import tensorflow as tf
    pio.verify_contract()
    ids = pio.identities()
    storage = pio.storage_from_exec_config()
    manifests = {}
    for key, pinned, name in (('pack_root', pio.PACK_MANIFEST_SHA256, 'pack_manifest.json'),
                              ('prepass_root', pio.PREPASS_MANIFEST_SHA256, 'prepass_manifest.json')):
        mpath = Path(storage[key]) / name
        payloads = json.loads(mpath.read_text())['payloads']
        manifests[key] = {'manifest_sha256': sha256_file(mpath), 'pinned': pinned,
                          'payloads': len(payloads),
                          'payloads_match': sum(sha256_file(Path(storage[key]) / p) == row['sha256']
                                                for p, row in sorted(payloads.items()))}
    pack = feed_pack.load(pio.pack_root(RUNTIME), pio.PACK_MANIFEST_SHA256)
    lock = json.loads(Path(pio.LOCK[0]).read_text())
    record = json.loads(Path(pio.RUNTIME_RECORD[0]).read_text())
    live = pr.live_environment(tf)
    diff, driver = pio.environment_differences(lock, record, live)
    # the qualified M6D3e construction, rebuilt alone in an unexecuted graph (Controller order; no session)
    graph = tf.Graph()
    with graph.as_default():
        from methods.physics_std.training_graph import TrainingGraph
        with tf.device('/gpu:0'):
            tf.set_random_seed(seed)
            model = TrainingGraph(tf, Path(repo))
        construction = pr.construction_signature(graph)
        gvars = sorted((v.op.name, v.shape.as_list(), v.dtype.base_dtype.name) for v in tf.global_variables())
        structure = {'construction_sha256_without_seed_attrs': construction[0],
                     'construction_operations': construction[1], 'global_variables': len(gvars),
                     'trainable_variables': len(tf.trainable_variables()), 'G': len(model.G), 'D': len(model.D),
                     'trainable_parameters': int(sum(np.prod(v.shape.as_list()) for v in tf.trainable_variables())),
                     'shared_G_slots': model.slot_references[1] == model.slot_references[3],
                     'bn_update_ops': {str(k): [len(x) for x in v] for k, v in model.bn_paths.items()}}
        trainable = sorted(v.op.name for v in tf.trainable_variables())
    ckpts, prev = [], None
    for it in CHECKPOINTS:
        pfx = root / prefix(it)
        files = {}
        for suffix in SUFFIXES:
            p = Path(str(pfx) + suffix)
            files[suffix] = {'path': prefix(it) + suffix, 'exists': p.is_file(),
                             'size': p.stat().st_size if p.is_file() else None,
                             'sha256': sha256_file(p) if p.is_file() else None}
        reader = tf.train.NewCheckpointReader(str(pfx))
        shapes = reader.get_variable_to_shape_map()
        dtypes = {k: v.name for k, v in reader.get_variable_to_dtype_map().items()}
        values = {n: reader.get_tensor(n) for n in shapes}
        floating = [n for n in values if np.issubdtype(np.asarray(values[n]).dtype, np.floating)]
        per = {n + ':0': hashlib.sha256(np.ascontiguousarray(values[n]).tobytes()).hexdigest() for n in values}
        names = sorted((n, list(shapes[n]), dtypes[n]) for n in shapes)
        rec = {'iteration': it, 'files': files, 'variables': len(values), 'floating': len(floating),
               'nonfinite_variables': sum(int(not np.isfinite(values[n]).all()) for n in floating),
               'global_iteration': int(values['global_iteration']), 'minibatch_phase': int(values['minibatch_phase']),
               'beta_powers': {k: float(values[k]) for k in sorted(sum(BETA_NAMES.values(), ()))},
               'state_sha256': hashlib.sha256(json.dumps(per, sort_keys=True).encode()).hexdigest(),
               'variable_set_equals_graph': names == gvars,
               'variable_set_sha256': sha256_bytes(json.dumps(names).encode()),
               'adam_slots': {s: sum(n.endswith('/' + s) for n in values)
                              for s in ('G_Adam', 'G_Adam_1', 'D_Adam', 'D_Adam_1')},
               'trainable_differs_from_previous': (sum(int(not np.array_equal(values[n], prev[n])) for n in trainable)
                                                   if prev is not None else None),
               'trainable_abs_max': max(float(np.abs(values[n]).max()) for n in trainable)}
        ckpts.append(rec)
        prev = values
    prev = values = None
    state = (root / 'checkpoints' / 'checkpoint').read_text().splitlines()
    streams = pio.stream_hashes(seed)
    # dense sampling offsets of a fixed iteration sample, from the pack geometry only (no image is opened)
    offsets = {'iterations': OFFSET_SAMPLE, 'checked': 0, 'match': 0, 'mismatch': []}
    for _, r in iteration_records(root / 'metrics.jsonl', set(OFFSET_SAMPLE)):
        recs = [pack.rows[x] for x in r['pair_rows']]
        o = np.stack([dense_offsets(pack.q140_xy(b), pack.q140_xy(a)) for _, a, b, _ in recs]).astype(np.float32)
        ok = hashlib.sha256(o.tobytes()).hexdigest() == r['input_sha256']['offsets']
        offsets['checked'] += 1
        offsets['match'] += ok
        if not ok:
            offsets['mismatch'].append(r['iteration'])
    after = tree_snapshot(root)
    return {
        'phase': 'collect-tf', 'seed': seed,
        'collector': {'python': sys.version.split()[0], 'executable': sys.executable, 'tensorflow': tf.__version__,
                      'numpy': np.__version__, 'utc': now_utc(), 'platform': platform.platform(),
                      'launch_environment_exact': all(os.environ.get(k) == v for k, v in
                                                      dict(PRECISION_ENV, PYTHONHASHSEED=str(seed)).items())},
        'git': git_state(), 'identities_recomputed': ids, 'storage': storage, 'manifests': manifests,
        'pack': {'manifest_sha256': pack.manifest_sha256, 'counts': pack.manifest['counts'],
                 'test_rows': pack.manifest['test_rows'], 'val_rows': pack.manifest['val_rows'],
                 'rows': len(pack.rows), 'prepass_manifest_sha256': pack.manifest['prepass_manifest_sha256']},
        'environment_recheck': {'differences_other_than_driver': diff, 'driver': driver, 'gpu': live['gpu'],
                                'tensorflow': live['tensorflow'], 'python': live['python'],
                                'native_runtime': live['native_runtime'], 'packages': len(live['package_versions']),
                                'session_config': live['session_config']},
        'structure': structure, 'checkpoints': ckpts, 'checkpoint_state_file': state,
        'production_stream_hashes': streams, 'offsets_sample': offsets,
        'run_root_tree_unchanged': before == after, 'run_root_tree_sha256': sha256_bytes(json.dumps(after).encode()),
    }


# ============================================================================ collect (GPU host, Python 3.11 + pyarrow)
def load_pack(pack_dir):
    import numpy as np
    raw = (pack_dir / 'pack_manifest.json').read_bytes()
    m = json.loads(raw)
    ok = all(sha256_file(pack_dir / n) == r['sha256'] for n, r in m['payloads'].items())
    pairs = json.loads((pack_dir / 'pairs_train.json').read_text())
    s = json.loads((pack_dir / 'samples.json').read_text())
    samples = {r[0]: dict(zip(s['columns'], r)) for r in s['rows']}
    m0 = np.frombuffer((pack_dir / 'live_m0.f32le').read_bytes(), '<f4').reshape(m['shapes']['live_m0.f32le'])
    return {'manifest_sha256': sha256_bytes(raw), 'payloads_match': ok, 'test_rows': m['test_rows'],
            'val_rows': m['val_rows'], 'counts': m['counts'], 'rows': pairs['rows'], 'split': pairs['split'],
            'samples': samples, 'm0': m0, 'prepass_manifest_sha256': m['prepass_manifest_sha256']}


def scan_metrics(path, seed, pack):
    """Every line of metrics.jsonl against the frozen schedule and an independent rule-text replay of D4/D7."""
    import numpy as np
    pseed, hseed = derive_seed('PAIR_ORDER', seed), derive_seed('HARD_SAMPLE', seed)
    prng, hrng = np.random.RandomState(pseed), np.random.RandomState(hseed)
    buf, passes_made = [], 0
    zeros = np.zeros((4, 32, 32), np.float32)
    chk = {k: 0 for k in ('keys', 'index', 'global_step', 'lr', 'losses', 'pair_rows', 'pair_ids', 'pair_passes',
                          'hard_draws', 'hard_input_sha256', 'depth_input_sha256', 'input_keys', 'nulls', 'reasons',
                          'timing', 'gpu_memory', 'utc')}
    rle, lines, bad, i = [], 0, 0, 0
    events, utcs, rows_all, hard_h = {}, [], [], hashlib.sha256()
    max_rel_lr, wall_prev, mem = 0.0, -1.0, []
    blocks = {}
    step_seconds = []
    with open(str(path), 'rb') as f:
        raw = f.read()
    trailing_newline = raw.endswith(b'\n')
    for line in raw.decode('utf-8').splitlines():
        lines += 1
        try:
            r = json.loads(line)
        except ValueError:
            bad += 1
            continue
        if r.get('record_type') == 'event':
            kind = 'EVENT:' + str(r.get('event'))
            events.setdefault(r.get('event'), []).append(dict(r.get('payload') or {}, _line=lines, _utc=r.get('utc'),
                                                              _seed=r.get('experiment_seed'),
                                                              _method=r.get('method_id')))
            utcs.append(r.get('utc'))
            if rle and rle[-1][0] == kind:
                rle[-1][1] += 1
            else:
                rle.append([kind, 1])
            continue
        kind = 'ITERATION' if r.get('record_type') == 'epoch' else 'OTHER:' + str(r.get('record_type'))
        if rle and rle[-1][0] == kind:
            rle[-1][1] += 1
        else:
            rle.append([kind, 1])
        if kind != 'ITERATION':
            continue
        utcs.append(r.get('utc'))
        chk['keys'] += sorted(r) == ITERATION_KEYS
        chk['index'] += r.get('iteration') == i and r.get('epoch') is None
        chk['global_step'] += r.get('global_step') == i + 1
        g, d = expected_lr(i)
        lr = r.get('learning_rates') or {}
        rel = max(abs(lr.get('G', -1) - g) / g, abs(lr.get('D', -1) - d) / d)
        max_rel_lr = max(max_rel_lr, rel)
        chk['lr'] += rel <= TOL_LR and r.get('learning_rate') == lr.get('G') and sorted(lr) == ['D', 'G']
        ls = r.get('train_losses') or {}
        ok = sorted(ls) == sorted(STEP_LOSSES) and all(sorted(ls[s]) == STEP_LOSSES[s] and
                                                       all(finite(v) for v in ls[s].values()) for s in STEP_LOSSES)
        chk['losses'] += ok
        while len(buf) < PAIRS:
            buf.extend(int(x) for x in prng.permutation(ROWS))
            passes_made += 1
        want, buf = buf[:PAIRS], buf[PAIRS:]
        pr_ = r.get('pair_rows')
        chk['pair_rows'] += pr_ == want
        recs = [pack['rows'][x] for x in want]
        chk['pair_ids'] += r.get('pair_ids') == [x[0] for x in recs]
        chk['pair_passes'] += r.get('pair_passes') == [(PAIRS * i + j) // ROWS for j in range(PAIRS)]
        u = hrng.uniform(0, 1, (4, 1, 1, 3)).astype(np.float32)
        k = (hrng.uniform(0, 1, (4, 1, 1, 1)) >= .5).astype(np.float32)
        chk['hard_draws'] += ([float(x) for x in u.ravel()] == r.get('hard_scales') and
                              [float(x) for x in k.ravel()] == r.get('hard_keep'))
        sh = r.get('input_sha256') or {}
        chk['input_keys'] += sorted(sh) == INPUT_KEYS
        chk['hard_input_sha256'] += (sh.get('hard_scales') == sha256_bytes(u.tobytes()) and
                                     sh.get('hard_keep') == sha256_bytes(k.tobytes()))
        depth = np.concatenate([np.stack([pack['m0'][pack['samples'][x[1]]['m0_row']] for x in recs]), zeros])
        chk['depth_input_sha256'] += sh.get('depth') == sha256_bytes(depth[..., None].astype(np.float32).tobytes())
        rows_all.extend(want)
        hard_h.update(u.tobytes())
        hard_h.update(k.tobytes())
        chk['nulls'] += all(r.get(x) is None for x in ('train_metrics', 'val_losses', 'val_metrics', 'epoch'))
        chk['reasons'] += r.get('missing_field_reasons') == REASONS
        ss, wc = r.get('step_seconds'), r.get('wall_clock_seconds')
        chk['timing'] += finite(ss) and ss > 0 and finite(wc) and wc > wall_prev
        wall_prev = wc if finite(wc) else wall_prev
        gm = r.get('gpu_memory_bytes')
        chk['gpu_memory'] += type(gm) is int and gm > 0
        chk['utc'] += isinstance(r.get('utc'), str)
        mem.append(gm if type(gm) is int else 0)
        step_seconds.append(ss if finite(ss) else 0.0)
        b = blocks.setdefault(i // BLOCK, {'n': 0, 'sum': {}, 'seconds': []})
        b['n'] += 1
        b['seconds'].append(ss if finite(ss) else 0.0)
        for s in STEP_LOSSES:
            for key, v in (ls.get(s) or {}).items():
                b['sum'][s + '/' + key] = b['sum'].get(s + '/' + key, 0.0) + (v if finite(v) else float('nan'))
        i += 1
    full = len(rows_all) // ROWS
    pblocks = [rows_all[x * ROWS:(x + 1) * ROWS] for x in range(full)]
    tail = rows_all[full * ROWS:]
    summary = []
    for bi in sorted(blocks):
        b = blocks[bi]
        sec = sorted(b['seconds'])
        g, d = expected_lr(bi * BLOCK)
        summary.append({'block': bi, 'first_iteration': bi * BLOCK, 'iterations': b['n'], 'lr_G': g, 'lr_D': d,
                        'median_step_seconds': sec[len(sec) // 2],
                        'loss_means': {k: v / b['n'] for k, v in sorted(b['sum'].items())}})
    secs = sorted(step_seconds)
    c = {'lines': lines, 'bad_lines': bad, 'trailing_newline': trailing_newline, 'rle': rle,
         'iteration_records': i, 'checks': chk, 'max_relative_lr_error': max_rel_lr,
         'final_global_step': i, 'utc_nondecreasing': all(a <= b for a, b in zip(utcs, utcs[1:])),
         'first_utc': utcs[0] if utcs else None, 'last_utc': utcs[-1] if utcs else None,
         'derived_seeds': {'pair_order': pseed, 'hard_sample': hseed},
         'streams': {'pair_rows': len(rows_all),
                     'pair_sha256_int32_le': sha256_bytes(np.asarray(rows_all, '<i4').tobytes()),
                     'hard_sha256': hard_h.hexdigest(), 'permutations_generated': passes_made,
                     'full_passes': full, 'permutation_passes': sum(sorted(x) == list(range(ROWS)) for x in pblocks),
                     'tail_rows': len(tail), 'tail_distinct': len(set(tail)) == len(tail),
                     'identical_consecutive_passes': sum(a == b for a, b in zip(pblocks, pblocks[1:])),
                     'distinct_pair_rows': len(set(rows_all)), 'head': rows_all[:8], 'last': rows_all[-4:]},
         'step_seconds': ({'median': secs[len(secs) // 2], 'p05': secs[len(secs) // 20],
                           'p95': secs[(len(secs) * 19) // 20], 'max': secs[-1], 'min': secs[0], 'sum': sum(secs)}
                          if secs else None),
         'gpu_memory_max_bytes': max(mem) if mem else None, 'block_summary': summary}
    consumed = sorted({x for r in set(rows_all) for x in pack['rows'][r][1:3]})
    return c, events, consumed


def collect_atime(seed, split, consumed):
    p = paths(seed)

    def load(f):
        out = {}
        with open(f) as h:
            for line in h:
                a, m, path = line.rstrip('\n').split('\t')
                out[path] = (float(a), float(m))
        return out
    pre, post = load(p['atime_pre']), load(p['atime_post'])
    t_pre = epoch_seconds(Path(p['atime_pre'][:-4] + '.time').read_text().strip())
    t_post = epoch_seconds(Path(p['atime_post'][:-4] + '.time').read_text().strip())
    stats, manifests = {}, []
    for path, (a, m) in sorted(pre.items()):
        if '/manifests/' in path:
            a2, m2 = post.get(path, (None, None))
            manifests.append({'file': os.path.basename(path), 'atime_changed': a2 != a, 'mtime_unchanged': m2 == m})
            continue
        sid = os.path.basename(path)[:-4]
        meta = split.get(sid)
        ds, sp = (meta['dataset'], meta['split']) if meta else (os.path.basename(os.path.dirname(path)), 'UNMAPPED')
        s = stats.setdefault('%s|%s' % (ds, sp), {'files': 0, 'sensitive_whole_window': 0, 'atime_changed': 0,
                                                  'changed_consumed': 0, 'changed_not_consumed': 0, 'consumed': 0,
                                                  'consumed_sensitive': 0, 'mtime_changed': 0, 'missing_post': 0})
        sens = (a <= m) or (t_pre - a >= 86400)    # relatime: the first read in the window updates atime
        s['files'] += 1
        s['sensitive_whole_window'] += sens
        s['consumed'] += sid in consumed
        s['consumed_sensitive'] += sid in consumed and sens
        if path not in post:
            s['missing_post'] += 1
            continue
        a2, m2 = post[path]
        s['mtime_changed'] += m2 != m
        if a2 != a:
            s['atime_changed'] += 1
            s['changed_consumed' if sid in consumed else 'changed_not_consumed'] += 1
    return {'pre_snapshot_utc': utc(t_pre), 'post_snapshot_utc': utc(t_post), 'files_pre': len(pre),
            'files_post': len(post), 'same_paths': sorted(pre) == sorted(post), 'by_class': stats,
            'manifests': manifests, 'mount': 'ext4 rw,relatime (/)', 'pre_sha256': sha256_file(p['atime_pre']),
            'post_sha256': sha256_file(p['atime_post'])}


def marker_file(path):
    q = Path(path)
    if not q.is_file():
        return None
    st = q.stat()
    return {'size': st.st_size, 'mtime': st.st_mtime_ns / 1e9, 'mtime_utc': utc(st.st_mtime), 'sha256': sha256_file(q)}


def boot_provenance():
    def run(*a):
        r = subprocess.run(list(a), stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        return r.returncode, r.stdout, r.stderr.strip()
    out = {'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip().replace('-', ''),
           'btime': int(next(x.split()[1] for x in Path('/proc/stat').read_text().splitlines()
                             if x.startswith('btime '))),
           'kernel_release': os.uname().release}
    rc, so, _ = run('journalctl', '--list-boots', '-o', 'json', '--no-pager')
    out['journal_boots'] = json.loads(so) if rc == 0 and so.strip() else None
    rc, so, _ = run('last', '-x', '--time-format', 'iso', 'reboot', 'shutdown')
    out['wtmp'] = [x for x in so.splitlines() if x.startswith(('reboot ', 'shutdown '))][:40] if rc == 0 else None
    rc, so, se = run('tmux', 'ls', '-F', '#{session_name}')
    out['tmux_ls'] = {'rc': rc, 'sessions': sorted(so.splitlines()), 'stderr': se}
    return out


def host_state(seed):
    p = paths(seed)

    def sh(*a):
        return subprocess.run(list(a), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                              universal_newlines=True).stdout.strip()
    pane = sh('tmux', 'list-panes', '-t', p['tmux'], '-F', '#{pane_dead} #{pane_dead_status}').split()
    procs = [x for x in sh('ps', '-eo', 'pid,args').splitlines() if 'python' in x and 'tools/run_e04.py' in x]
    pane_log = Path(p['pane_log']).read_bytes().decode('utf-8', errors='replace').splitlines()
    others = sorted(x for x in sh('tmux', 'ls', '-F', '#{session_name}').splitlines()
                    if x.startswith('e04_') and x != p['tmux'])
    siblings = {}
    for x in others:                      # evidence that a leftover E04 session is a finished, earlier sibling seed
        sp = sh('tmux', 'list-panes', '-t', x, '-F', '#{pane_dead} #{pane_dead_status}').split()
        sib = [s for s in sorted(RUN_IDS) if s != seed and paths(s)['tmux'] == x]
        q = paths(sib[0]) if sib else None
        siblings[x] = {'seed': sib[0] if sib else None, 'panes': [sp] if sp else [],
                       'exit_file_raw': (Path(q['exit']).read_bytes().decode('utf-8', errors='replace')
                                         if q and Path(q['exit']).is_file() else None),
                       'end_time': (Path(q['end_time']).read_text().strip()
                                    if q and Path(q['end_time']).is_file() else None)}
    return {'tmux_pane_dead': pane[0] == '1' if pane else None,
            'tmux_pane_dead_status': int(pane[1]) if len(pane) > 1 and pane[0] == '1' else None,
            'runner_processes': procs, 'exit_file': Path(p['exit']).read_text().strip(),
            'exit_file_raw': Path(p['exit']).read_bytes().decode('utf-8', errors='replace'),
            'launch_time': Path(p['launch_time']).read_text().strip(),
            'end_time': Path(p['end_time']).read_text().strip(),
            'command_file': Path(p['command']).read_text(), 'command_file_sha256': sha256_file(p['command']),
            'preflight': json.loads(Path(p['preflight']).read_text()),
            'pane_log_lines': len(pane_log), 'pane_log_last': [x.strip() for x in pane_log if x.strip()][-1:],
            'pane_log_fail_hits': {k: sum(k in x for x in pane_log) for k in PANE_FAIL_PATTERNS},
            'other_e04_tmux': sorted(x for x in sh('tmux', 'ls', '-F', '#{session_name}').splitlines()
                                     if x.startswith('e04_')),
            'sibling_e04_sessions': siblings,
            'gpu_query': sh('nvidia-smi', '--query-gpu=name,driver_version,uuid', '--format=csv,noheader'),
            'marker_files': {k: marker_file(p[k]) for k in ('command', 'launch_time', 'pane_log', 'exit', 'end_time')},
            'boot': boot_provenance()}


def collect(seed, repo):
    os.chdir(str(repo))
    import pyarrow.parquet as pq
    import yaml
    root = Path(paths(seed)['run_root'])
    before_tree = tree_snapshot(root)
    before_digest, _, _ = sha_listing(root)
    g = git_state()
    man = json.loads((root / 'run_manifest.json').read_text())
    summ = json.loads((root / 'run_summary.json').read_text())
    cindex = json.loads((root / 'checkpoint_index.json').read_text())
    resolved = yaml.safe_load((root / 'resolved_config.yaml').read_text())
    pack = load_pack(Path(RUNTIME) / 'data' / 'processed' / 'e04_feed_pack_v1')
    split = {r['sample_id']: r for r in pq.read_table('manifests/split_v1.parquet', columns=[
        'sample_id', 'dataset', 'label_binary', 'split']).to_pylist()}
    acc = {r['sample_id']: r['face_png_sha256'] for r in pq.read_table(
        'manifests/m2_sample_accounting.parquet', columns=['sample_id', 'face_png_sha256']).to_pylist()}
    metrics, events, consumed = scan_metrics(root / 'metrics.jsonl', seed, pack)
    recs = [pack['samples'][s] for s in consumed]
    faces = {'samples': len(consumed), 'pack_samples': len(pack['samples']),
             'split_v1_train': sum(split.get(s, {}).get('split') == 'TRAIN' for s in consumed),
             'pack_split_train': sum(r['split'] == 'TRAIN' for r in recs),
             'label_matches_split_v1': sum(split.get(r['sample_id'], {}).get('label_binary') == r['label_binary']
                                           for r in recs),
             'dataset_matches_split_v1': sum(split.get(r['sample_id'], {}).get('dataset') == r['dataset']
                                             for r in recs),
             'face_png_sha256_match_m2_accounting': sum(acc.get(r['sample_id']) == r['face_png_sha256'] for r in recs),
             'by_role': {role: sum(r['role'] == role for r in recs) for role in ('live', 'spoof')}}
    atime = collect_atime(seed, split, set(consumed))
    ck_events = [{k: e.get(k) for k in ('iteration', 'label_iteration', 'prefix', 'files', 'variables',
                                        'round_trip_equal', 'state_sha256', 'state_unchanged_by_save', 'all_finite',
                                        'terminal_semantics_probe', 'labels', 'resume_sidecar', 'checkpoint_type',
                                        'selected_for_final', '_seed', '_method', '_line', '_utc')}
                 for e in events.get('e04_checkpoint_event', [])]
    start = (events.get('e04_run_start') or [{}])[0]
    logs = {}
    for name in ('stdout.log', 'stderr.log', 'generation_log.jsonl'):
        text = (root / name).read_bytes().decode('utf-8', errors='replace')
        logs[name] = {'bytes': len(text.encode()), 'lines': len(text.splitlines()),
                      'pattern_hits': {k: sum(k in x for x in text.splitlines()) for k in LOG_PATTERNS}}
    rt = {rel: sha256_file(rel) for rel in sorted(PINNED_FILES)}
    host = host_state(seed)
    digest, count, lines = sha_listing(root)
    after_tree = tree_snapshot(root)
    return {
        'phase': 'collect', 'seed': seed,
        'collector': {'python': sys.version.split()[0], 'executable': sys.executable, 'utc': now_utc()},
        'git': g, 'host': host, 'files_sha256': rt,
        'run_manifest': man, 'run_summary': summ,
        'checkpoint_index_meta': {k: v for k, v in cindex.items() if k != 'checkpoints'},
        'checkpoint_index': cindex['checkpoints'],
        'resolved': {k: resolved['_resolved'].get(k) for k in ('rng', 'schedule', 'checkpoint_rule', 'resume', 'feed',
                                                               'runner_mode', 'scientific_run', 'fidelity', 'labels',
                                                               'identities')},
        'metrics': metrics, 'metrics_sha256': sha256_file(root / 'metrics.jsonl'),
        'event_counts': {k: len(v) for k, v in sorted(events.items())},
        'run_start': {k: start.get(k) for k in ('seeding', 'identities', 'structure', 'gpu_driver', 'fidelity', 'pack',
                                                'environment_identity_equals_lock_except_driver', 'session_config',
                                                'python38_pathlib_is_relative_to_backport', '_line', '_seed',
                                                '_method', '_utc')},
        'checkpoint_events': ck_events,
        'pack': {'manifest_sha256': pack['manifest_sha256'], 'payloads_match': pack['payloads_match'],
                 'test_rows': pack['test_rows'], 'val_rows': pack['val_rows'], 'rows': len(pack['rows']),
                 'split': pack['split'], 'counts': pack['counts'], 'samples': len(pack['samples']),
                 'non_train_samples': sum(r['split'] != 'TRAIN' for r in pack['samples'].values()),
                 'prepass_manifest_sha256': pack['prepass_manifest_sha256']},
        'consumed_faces': faces,
        'checkpoints_dir_listing': sorted(x.name for x in (root / 'checkpoints').iterdir()),
        'partial_files': sorted(str(x.relative_to(root)) for x in root.rglob('*.partial')),
        'resume_files': sorted(str(x.relative_to(root)) for x in root.rglob('*resume*')),
        'tmp_files': sorted(str(x.relative_to(root)) for x in root.rglob('*tmp*')),
        'logs': logs, 'atime_firewall': atime,
        'run_root_sha_listing_digest': digest, 'run_root_file_count': count,
        'run_root_sha_listing_digest_before': before_digest, 'run_root_sha_listing': lines,
        'run_root_tree_sha256': sha256_bytes(json.dumps(after_tree).encode()),
        'run_root_unchanged_whole_audit': before_tree == after_tree and before_digest == digest,
        'parent_listing': sorted(os.listdir(str(root.parent))),
    }


# ============================================================================ build (repository)
class Gates:
    def __init__(self):
        self.results = []

    def __call__(self, name, ok, detail=None):
        self.results.append(dict({'gate': name, 'pass': bool(ok)}, **({'detail': detail} if detail is not None else {})))
        return ok

    @property
    def failed(self):
        return [r['gate'] for r in self.results if not r['pass']]


def expected_rle():
    out = [['EVENT:e04_run_start', 1]]
    for _ in CHECKPOINTS:
        out += [['ITERATION', EVERY], ['EVENT:e04_checkpoint_event', 1]]
    return out


def manifest_text(c, t):
    seed = c['seed']
    head = ('# E04 Physics-STD seed%d run %s (authority %s): iteration checkpoint_type selected_for_final size_bytes '
            'sha256 path\n' % (seed, run_id(seed), AUTHORITY[:7]))
    body = ''
    for r in t['checkpoints']:
        kind, sel, _ = checkpoint_kind(r['iteration'])
        for suffix in SUFFIXES:
            f = r['files'][suffix]
            body += '%06d %s %s %d %s %s\n' % (r['iteration'], kind, str(sel).lower(), f['size'], f['sha256'],
                                               f['path'])
    return head + body + ('# checkpoints/checkpoint (TF Saver state file) is part of the run-root listing; '
                          'resume sidecars: none (E04 is fresh-only, NOT_QUALIFIED_FRESH_ONLY)\n')


def wtmp_record(line):
    """`last -x --time-format iso` line -> (kind, start epoch seconds, end epoch seconds or None, still running)."""
    f = line.split()
    kind = {('reboot', 'system', 'boot'): 'reboot', ('shutdown', 'system', 'down'): 'shutdown'}.get(tuple(f[:3]))
    try:
        start = datetime.strptime(f[4], '%Y-%m-%dT%H:%M:%S%z').timestamp()
    except (IndexError, ValueError):
        return None, None, None, False
    end = None
    if len(f) > 6 and f[5] == '-':
        try:
            end = datetime.strptime(f[6], '%Y-%m-%dT%H:%M:%S%z').timestamp()
        except ValueError:
            end = None
    return kind, start, end, 'still running' in line


def boot_continuity(c):
    """No reboot or shutdown inside [launch_time, end_time]: the last wtmp boot before launch is followed by no boot or
    shutdown record until after the end marker and no journal boot starts inside the window. Then exactly one of:
      (a) rebooted after the run: the current boot started after the run ended (so the host stayed up through
          completion and only rebooted afterwards) -- the seed-42 case;
      (b) uninterrupted: no wtmp boot record exists after launch, the run boot is the wtmp 'still running' record, the
          kernel btime (/proc/stat) precedes launch and equals that record's start within 120 s, and the current
          journal boot (same boot_id) has entries both before launch and after the end marker -- the host has not
          rebooted at all since before the run started."""
    host, rp = c['host'], c['host']['boot']
    launch, end = epoch_seconds(host['launch_time']), epoch_seconds(host['end_time'])
    recs = [(x,) + wtmp_record(x) for x in (rp.get('wtmp') or [])]
    boots = sorted([r for r in recs if r[1] == 'reboot'], key=lambda r: r[2])
    run_boot = [r for r in boots if r[2] <= launch]
    run_boot = run_boot[-1] if run_boot else None
    inside = [r[0] for r in recs if r[1] in ('reboot', 'shutdown') and launch <= r[2] <= end]
    later = [r for r in boots if r[2] > launch]
    jb = rp.get('journal_boots') or []
    j_inside = [b['boot_id'] for b in jb if launch <= b['first_entry'] / 1e6 <= end]
    j_after = [b for b in jb if b['last_entry'] / 1e6 >= end and b['first_entry'] / 1e6 <= end]
    rebooted_after = bool(later) and min(r[2] for r in later) > end and rp['btime'] > end
    current = [b for b in jb if b.get('boot_id') == rp.get('boot_id')]
    uninterrupted = (run_boot is not None and not later and run_boot[4] and rp['btime'] <= launch and
                     abs(rp['btime'] - run_boot[2]) <= 120 and len(current) == 1 and
                     current[0]['first_entry'] / 1e6 <= launch and current[0]['last_entry'] / 1e6 >= end)
    ok = run_boot is not None and inside == [] and j_inside == [] and (rebooted_after or uninterrupted)
    return ok, {'run_boot_wtmp': run_boot[0] if run_boot else None,
                'next_boot_wtmp': later[0][0] if later else None,
                'records_inside_window': inside, 'journal_boots_starting_inside_window': j_inside,
                'journal_boot_alive_at_end': [[b['boot_id'], utc(b['first_entry'] / 1e6), utc(b['last_entry'] / 1e6)]
                                              for b in j_after],
                'current_boot_id': rp.get('boot_id'), 'current_btime_utc': utc(rp['btime']),
                'kernel_release_now': rp.get('kernel_release'),
                'shutdowns_after_end': [r[0] for r in recs if r[1] == 'shutdown' and r[2] > end]}


def finished_siblings(host, seed):
    """Every e04_* tmux session other than this seed's must be (a) the session of another registered E04 seed, (b) a
    single dead pane with exit status 0, (c) backed by that seed's durable exit marker '0\\n' and (d) an end marker
    strictly before this seed's launch -- i.e. a finished earlier run that cannot have overlapped this one. Any other
    leftover session fails. A document without sibling evidence keeps the original rule (only this seed's session)."""
    own, launch = paths(seed)['tmux'], epoch_seconds(host['launch_time'])
    others = [x for x in host['other_e04_tmux'] if x != own]
    if not others:
        return True, {}
    ev = host.get('sibling_e04_sessions') or {}
    out, ok = {}, True
    for x in others:
        e = ev.get(x)
        good = (e is not None and e['seed'] in RUN_IDS and e['seed'] != seed and paths(e['seed'])['tmux'] == x and
                e['panes'] == [['1', '0']] and e['exit_file_raw'] == '0\n' and e['end_time'] is not None and
                epoch_seconds(e['end_time']) < launch)
        ok = ok and good
        out[x] = {'finished_before_launch': good, 'seed': e and e['seed'], 'end_time': e and e['end_time']}
    return ok, out


def build(c, t, repo_root):
    seed = c['seed']
    G = Gates()
    man, summ, m, rs, host = c['run_manifest'], c['run_summary'], c['metrics'], c['run_start'], c['host']
    E, f = EXPECTED, c['files_sha256']
    rid = run_id(seed)
    pseed, hseed = derive_seed('PAIR_ORDER', seed), derive_seed('HARD_SAMPLE', seed)
    vectors = json.loads((Path(repo_root) / 'outputs/audit/E04_RNG_TEST_VECTORS.json').read_text())['seeds'][str(seed)]
    d12 = json.loads((Path(repo_root) / 'configs/amendments/e04_d12_procedural_reproducibility.yaml').read_text())
    qual = json.loads((Path(repo_root) / 'outputs/audit/E04_PRODUCTION_QUALIFICATION_D12.json').read_text())

    # A. identity / authority
    G('A.seed_pair', t['seed'] == c['seed'] == seed and seed in RUN_IDS and seed in DERIVED)
    G('A.run_id_recomputed', rid == RUN_IDS[seed] == man['run_id'] == summ['run_id'], rid)
    G('A.manifest_identity', man['method_id'] == E['method_id'] and man['experiment_seed'] == seed and
      man['git_commit'] == AUTHORITY and man['git_dirty'] is False and man['config_sha256'] == CONFIG_SHA and
      man['fidelity'] == FIDELITY and man['reporting_label'] == REPORTING_LABEL and man['resume'] == E['resume'] and
      man['runner_mode'] == 'SCIENTIFIC' and man['scientific_run'] is True and man['snapshot_verified'] is True and
      man['qualification_seed'] is None and man['qualification_case'] is None and man['source_commits'] == {})
    G('A.manifest_identities_exact', man['identities'] == E['identities'] == rs['identities'] ==
      t['identities_recomputed'] == c['resolved']['identities'])
    G('A.command_line_exact', man['command_line'] == command(seed) and '--resume' not in host['command_file'] and
      '--preflight-only' not in host['command_file'] and
      wrapper_invocation(seed) in host['command_file'] and 'PYTHONHASHSEED=%d ' % seed in host['command_file'] and
      all('%s=%s' % kv in host['command_file'] for kv in PRECISION_ENV.items()) and
      'PATH=/home/student20261/miniconda3/envs/gpat-m6-e03/bin:' in host['command_file'])
    pf = host['preflight']
    G('A.preflight_seed', pf['experiment_seed'] == seed and pf['git_commit'] == AUTHORITY and pf['git_dirty'] is False
      and pf['run_dir_exists'] is False and pf['identities'] == E['identities'] and pf['graph_executed'] is False and
      pf['tensorflow_imported'] is False and pf['images_decoded'] == 0 and pf['checkpoints'] == CHECKPOINTS and
      pf['iterations'] == ITERATIONS and pf['batch'] == BATCH and pf['resume'] == E['resume'] and
      pf['rng']['pair_order_seed'] == pseed and pf['rng']['hard_sample_seed'] == hseed)
    G('A.gpu_checkout_authority', c['git']['head'] == t['git']['head'] == AUTHORITY and
      c['git']['branch'] == t['git']['branch'] == 'm6-baselines' and
      c['git']['status_porcelain'] == t['git']['status_porcelain'] == '')
    G('A.authority_ancestry', all(c['git']['ancestors_of_head'].values()) and all(t['git']['ancestors_of_head'].values()),
      c['git']['ancestors_of_head'])
    G('A.host', man['host'] == E['host'] and man['environment']['host'] == E['host'])
    G('A.frozen_files', all(f.get(k) == v for k, v in PINNED_FILES.items()),
      sorted(k for k, v in PINNED_FILES.items() if f.get(k) != v))
    G('A.single_fresh_start', c['event_counts'].get('e04_run_start') == 1 and rs['_line'] == 1 and
      rs['_seed'] == seed and rs['_method'] == 'E04' and c['resolved']['resume'] == E['resume'] and
      {'seed_%d' % seed, '.seed_%d.lock' % seed} <= set(c['parent_listing']) and
      set(c['parent_listing']) <= {x % s for s in DERIVED for x in ('seed_%d', '.seed_%d.lock')}, c['parent_listing'])
    G('A.fidelity_disclosed', rs['fidelity'] == man['fidelity'] == c['resolved']['fidelity'] == FIDELITY)

    # B. completion (durable E04 launch-contract evidence; boot continuity over the run window)
    last = host['pane_log_last'][0] if len(host['pane_log_last']) == 1 else None
    G('B.exit_marker_zero', host['exit_file'] == '0' and host['exit_file_raw'] == '0\n' and
      ('%s\necho $? > %s\ndate -u +%%FT%%TZ > %s\n' % (wrapper_invocation(seed), paths(seed)['exit'],
                                                       paths(seed)['end_time'])) in host['command_file'],
      host['exit_file_raw'])
    st, en = epoch_seconds(man['start_utc']), epoch_seconds(man['end_utc'])
    la, em = epoch_seconds(host['launch_time']), epoch_seconds(host['end_time'])
    mk = host['marker_files']
    G('B.window_and_markers', la <= st <= en <= em <= en + MARKER_SLACK and summ['start_utc'] == man['start_utc'] and
      summ['end_utc'] == man['end_utc'] and all(isinstance(mk.get(k), dict) for k in mk) and
      mk['command']['sha256'] == host['command_file_sha256'] and mk['command']['mtime'] <= la + 1 and
      la <= mk['launch_time']['mtime'] <= la + 1 and mk['exit']['size'] == 2 and
      all(en <= mk[k]['mtime'] <= en + MARKER_SLACK for k in ('pane_log', 'exit', 'end_time')) and
      mk['exit']['mtime'] <= mk['end_time']['mtime'],
      {'launch_time': host['launch_time'], 'start_utc': man['start_utc'], 'end_utc': man['end_utc'],
       'end_time': host['end_time'], 'marker_mtimes_utc': {k: (v or {}).get('mtime_utc') for k, v in mk.items()}})
    ok, cont = boot_continuity(c)
    G('B.no_reboot_during_run', ok, cont)
    G('B.completion_status', man['completion_status'] == summ['completion_status'] == 'completed' and
      summ['failure_reason'] is None and summ['test_split_accessed'] is False)
    G('B.completion_record', last == '{"status": "completed", "run_dir": "%s"}' % paths(seed)['run_root'] and
      all(v == 0 for v in host['pane_log_fail_hits'].values()),
      dict(host['pane_log_fail_hits'], last_line=last))
    G('B.no_runner_process', host['runner_processes'] == [], host['runner_processes'])
    G('B.records_written', summ['records_written'] == m['lines'] == ITERATIONS + len(CHECKPOINTS) + 1,
      summ['records_written'])
    G('B.summary_counts', summ['global_iteration'] == ITERATIONS and summ['face_reads'] == ITERATIONS * BATCH and
      summ['checkpoint_selection_rule'] == E['checkpoint_rule'])
    G('B.logs_clean', all(v['bytes'] == 0 for v in c['logs'].values()), {k: v['bytes'] for k, v in c['logs'].items()})

    # C. metrics log: lifecycle, continuity, completeness
    G('C.metrics_parse', m['bad_lines'] == 0 and m['trailing_newline'] and m['utc_nondecreasing'])
    G('C.lifecycle_order', m['rle'] == expected_rle(), m['rle'][:5])
    G('C.iteration_records_150000', m['iteration_records'] == ITERATIONS)
    G('C.event_set', c['event_counts'] == {'e04_checkpoint_event': len(CHECKPOINTS), 'e04_run_start': 1},
      c['event_counts'])
    for k, v in sorted(m['checks'].items()):
        G('C.iteration_%s' % k, v == ITERATIONS, v)
    G('C.window_matches_log', m['first_utc'] == man['start_utc'] and man['start_utc'] <= m['last_utc'] <=
      man['end_utc'], [m['first_utc'], m['last_utc']])

    # D. optimizer / schedule accounting (frozen three-step controller)
    tk = {r['iteration']: r for r in t['checkpoints']}
    G('D.final_global_iteration_150000', m['final_global_step'] == ITERATIONS == summ['global_iteration'] and
      tk.get(ITERATIONS, {}).get('global_iteration') == ITERATIONS)
    G('D.checkpoint_counters', all(tk.get(it, {}).get('global_iteration') == it and
                                   tk.get(it, {}).get('minibatch_phase') == 0 for it in CHECKPOINTS),
      {it: [tk.get(it, {}).get('global_iteration'), tk.get(it, {}).get('minibatch_phase')] for it in CHECKPOINTS})
    seq2, seq1 = beta_power_sequence(BETA2, 2 * ITERATIONS), beta_power_sequence(BETA1, 2 * ITERATIONS)
    beta = {'exact_normal_range': [], 'flushed_or_underflowed': [], 'bad': []}
    for it in CHECKPOINTS:
        bp = tk.get(it, {}).get('beta_powers', {})
        for opt, applies in (('G', 2 * it), ('D', it)):
            for name, seq in zip(BETA_NAMES[opt], (seq1, seq2)):
                want, got = seq[applies], bp.get(name)
                if want >= FLT_MIN:
                    (beta['exact_normal_range'] if got == want else beta['bad']).append([it, name, applies, got])
                elif got in (0.0, want):
                    beta['flushed_or_underflowed'].append([it, name, applies, got])
                else:
                    beta['bad'].append([it, name, applies, got])
    G('D.adam_beta_powers_exact', beta['bad'] == [] and len(beta['exact_normal_range']) >= 4,
      {'exact_normal_range': beta['exact_normal_range'], 'bad': beta['bad'],
       'flushed_or_underflowed': len(beta['flushed_or_underflowed'])})
    G('D.lr_staircase', m['max_relative_lr_error'] <= TOL_LR and m['checks']['lr'] == ITERATIONS,
      m['max_relative_lr_error'])
    sched = c['resolved']['schedule']
    G('D.resolved_schedule', sched['iterations'] == ITERATIONS and sched['checkpoint_every'] == EVERY and
      sched['batch'] == '4 target_live + 4 source_spoof' and
      sched['lr'] == '5e-5 * 10^-floor(t / 45000); D x 0.5' and sched['per_iteration'] == [
          'step1: BN(original G, D), apply G Eq.23', 'step2: BN(original G, D), apply D L_D',
          'step3: BN(original G, hard G), apply G Eq.24; global_iteration += 1'])
    G('D.optimizer_applications', m['checks']['losses'] == ITERATIONS and m['checks']['global_step'] == ITERATIONS,
      {'G': 2 * m['iteration_records'], 'D': m['iteration_records']})

    # E. data feed and RNG (D4, D7, D8, D2/D3)
    rng = c['resolved']['rng']
    G('E.derived_seeds', (pseed, hseed) == DERIVED[seed] == (rng['pair_order_seed'], rng['hard_sample_seed']) ==
      (rs['seeding']['pair_order_seed'], rs['seeding']['hard_sample_seed']) ==
      (m['derived_seeds']['pair_order'], m['derived_seeds']['hard_sample']) ==
      (vectors['pair_seed'], vectors['hard_seed']) and rng['tf_graph_seed'] == rs['seeding']['tf_graph_seed'] == seed
      and rs['seeding']['python_numpy_global'] == seed, [pseed, hseed])
    s, ps = m['streams'], t['production_stream_hashes']
    G('E.pair_stream_equals_approved_vector', s['pair_rows'] == PAIRS * ITERATIONS and
      s['pair_sha256_int32_le'] == vectors['pair_stream']['sha256_int32_le'] == ps['pair_sha256_int32_le'],
      s['pair_sha256_int32_le'])
    G('E.hard_stream_equals_approved_vector',
      s['hard_sha256'] == vectors['hard_stream']['sha256_float32_scales_then_keep_per_iteration'] == ps['hard_sha256'],
      s['hard_sha256'])
    G('E.pair_passes', s['full_passes'] == vectors['pair_stream']['full_passes'] == s['permutation_passes'] and
      s['tail_rows'] == vectors['pair_stream']['tail_rows'] and s['tail_distinct'] and
      s['identical_consecutive_passes'] == 0 and s['distinct_pair_rows'] == ROWS and
      s['permutations_generated'] == ps['pair_passes_generated'] == vectors['pair_stream']['permutations_generated']
      and s['head'][:4] == vectors['pair_stream']['iteration0_rows'] and
      s['last'] == vectors['pair_stream']['iteration149999_rows'],
      {k: s[k] for k in ('full_passes', 'tail_rows', 'permutations_generated')})
    G('E.record_by_record_replay', all(m['checks'][k] == ITERATIONS for k in (
        'pair_rows', 'pair_ids', 'pair_passes', 'hard_draws', 'hard_input_sha256', 'depth_input_sha256')))
    os_ = t['offsets_sample']
    G('E.offsets_sample_from_pack_geometry', os_['checked'] == os_['match'] == len(OFFSET_SAMPLE) and
      os_['mismatch'] == [], {'checked': os_['checked'], 'match': os_['match']})
    cf = c['consumed_faces']
    G('E.consumed_faces_train_only', cf['samples'] == cf['pack_samples'] == cf['split_v1_train'] ==
      cf['pack_split_train'] == cf['label_matches_split_v1'] == cf['dataset_matches_split_v1'] ==
      cf['face_png_sha256_match_m2_accounting'] == E['pack_counts']['samples'] and
      cf['by_role'] == {'live': E['pack_counts']['live'], 'spoof': E['pack_counts']['spoof']}, cf)
    pk, tp = c['pack'], t['pack']
    G('E.feed_pack_and_prepass', pk['manifest_sha256'] == tp['manifest_sha256'] ==
      E['identities']['pack_manifest_sha256'] == rs['pack']['manifest_sha256'] and pk['payloads_match'] and
      pk['test_rows'] == pk['val_rows'] == tp['test_rows'] == tp['val_rows'] == 0 and pk['non_train_samples'] == 0
      and pk['split'] == 'TRAIN' and pk['counts'] == tp['counts'] == rs['pack']['counts'] == E['pack_counts'] and
      pk['rows'] == tp['rows'] == ROWS and pk['prepass_manifest_sha256'] == tp['prepass_manifest_sha256'] ==
      E['identities']['prepass_manifest_sha256'] and
      all(v['manifest_sha256'] == v['pinned'] and v['payloads'] == v['payloads_match'] > 0
          for v in t['manifests'].values()), t['manifests'])
    fd = c['resolved']['feed']
    G('E.no_augmentation_no_val', fd['augmentation'] == 'NONE (D5)' and fd['val'] == 'NONE (D6)' and
      fd['encoding'] == 'RGB uint8 / 255 (D8)' and fd['relation'] == 'manifests/pairs_train_v1.parquet' and
      man['val_split_accessed'] is False and man['val_role'] == 'NOT_USED_D6' and
      m['checks']['nulls'] == ITERATIONS)

    # F. graph structure / numerics
    G('F.graph_structure_run_start', rs['structure'] == E['structure'], rs['structure'])
    G('F.graph_structure_rebuilt', t['structure'] == E['structure'], t['structure'])
    G('F.losses_finite', m['checks']['losses'] == ITERATIONS)
    G('F.checkpoint_state_finite', all(tk.get(it, {}).get('nonfinite_variables') == 0 for it in CHECKPOINTS))

    # G. checkpoints (BASELINE_FINAL_STATE_V1)
    ev = {e['iteration']: e for e in c['checkpoint_events']}
    idx = {}
    for r in c['checkpoint_index']:
        idx.setdefault(r['global_step'], []).append(r)
    G('G.ten_checkpoints', sorted(tk) == sorted(ev) == sorted(idx) == CHECKPOINTS and
      len(c['checkpoint_index']) == len(CHECKPOINTS) * len(SUFFIXES) and
      c['checkpoints_dir_listing'] == sorted(['checkpoint'] + ['iteration_%06d%s' % (it, sfx) for it in CHECKPOINTS
                                                               for sfx in SUFFIXES]))
    bad = {'files': [], 'events': [], 'index': [], 'reader': [], 'state': [], 'trained': []}
    for it in CHECKPOINTS:
        r, x, ix = tk.get(it, {}), ev.get(it, {}), idx.get(it, [])
        kind, sel, reason = checkpoint_kind(it)
        files = r.get('files', {})
        if not all(files.get(sfx, {}).get('exists') for sfx in SUFFIXES):
            bad['files'].append(it)
        want = [{'path': files.get(sfx, {}).get('path'), 'epoch': None, 'global_step': it,
                 'file_size_bytes': files.get(sfx, {}).get('size'), 'sha256': files.get(sfx, {}).get('sha256'),
                 'checkpoint_type': kind, 'selected_for_final': sel, 'selection_reason': reason} for sfx in SUFFIXES]
        if x.get('files') != want:
            bad['events'].append([it, 'files'])
        if sorted(ix, key=lambda q: q['path']) != sorted(want, key=lambda q: q['path']):
            bad['index'].append(it)
        if not (x.get('label_iteration') == it and x.get('variables') == x.get('round_trip_equal') == N_VARIABLES and
                x.get('all_finite') is True and x.get('state_unchanged_by_save') is True and
                x.get('terminal_semantics_probe') is False and x.get('labels') == [] and
                x.get('resume_sidecar') is None and x.get('checkpoint_type') == kind and
                x.get('selected_for_final') is sel and x.get('prefix') == prefix(it) and x.get('_seed') == seed):
            bad['events'].append([it, 'payload'])
        if not (r.get('variables') == N_VARIABLES and r.get('nonfinite_variables') == 0 and
                r.get('variable_set_equals_graph') is True and
                r.get('adam_slots') == {'G_Adam': 72, 'G_Adam_1': 72, 'D_Adam': 72, 'D_Adam_1': 72}):
            bad['reader'].append(it)
        if r.get('state_sha256') != x.get('state_sha256'):
            bad['state'].append(it)
        if it > EVERY and not (r.get('trainable_differs_from_previous') or 0) > 0:
            bad['trained'].append(it)
    G('G.files_exist', bad['files'] == [], bad['files'])
    G('G.events_match_files', bad['events'] == [], bad['events'][:10])
    G('G.index_matches_files', bad['index'] == [], bad['index'])
    G('G.reader_complete_finite', bad['reader'] == [], bad['reader'])
    G('G.reader_state_equals_logged_live_state', bad['state'] == [], bad['state'])
    G('G.training_changed_weights_every_checkpoint', bad['trained'] == [], bad['trained'])
    G('G.variable_set_constant', len({r['variable_set_sha256'] for r in t['checkpoints']}) == 1)
    lines = {x['path']: x for x in ({'path': ln.split(' ')[0], 'size': int(ln.split(' ')[1]),
                                      'sha256': ln.split(' ')[2]} for ln in c['run_root_sha_listing'])}
    G('G.listing_matches_reader_phase', all(lines.get(fr['path'], {}).get('sha256') == fr['sha256'] and
                                            lines.get(fr['path'], {}).get('size') == fr['size']
                                            for r in t['checkpoints'] for fr in r['files'].values()))
    ck_dir = paths(seed)['run_root'] + '/checkpoints/'
    G('G.saver_state_latest_150000', t['checkpoint_state_file'] == [
        'model_checkpoint_path: "%siteration_150000"' % ck_dir,
        'all_model_checkpoint_paths: "%siteration_150000"' % ck_dir])
    fin = tk.get(ITERATIONS, {}).get('files', {})
    G('G.final_150000_selected', summ['checkpoint_selection_rule'] == E['checkpoint_rule'] and
      summ['final_or_selected_checkpoint_path'] == ck_dir + 'iteration_150000' and
      summ['final_or_selected_checkpoint_sha256'] == fin.get(SUFFIXES[0], {}).get('sha256') and
      [x['path'] for x in summ['final_checkpoint_files']] == [prefix(ITERATIONS) + sfx for sfx in SUFFIXES] and
      all(x['selected_for_final'] and x['global_step'] == ITERATIONS and x['sha256'] == fin.get(sfx, {}).get('sha256')
          for x, sfx in zip(summ['final_checkpoint_files'], SUFFIXES)) and
      c['checkpoint_index_meta']['selected_final'] == prefix(ITERATIONS) and
      c['checkpoint_index_meta']['checkpoint_rule'] == E['checkpoint_rule'] and
      c['checkpoint_index_meta']['scientific_checkpoints'] is True and
      sum(r['selected_for_final'] for r in c['checkpoint_index']) == len(SUFFIXES) and
      c['resolved']['checkpoint_rule'] == {'rule': E['checkpoint_rule'], 'terminal': 'iteration_150000',
                                           'selected': prefix(ITERATIONS)})
    G('G.no_partial_no_sidecar', c['partial_files'] == [] and c['resume_files'] == [] and c['tmp_files'] == [] and
      c['checkpoint_index_meta']['resume_sidecars'] is None)

    # H. firewall
    a = c['atime_firewall']
    by = a['by_class']
    G('H.snapshots_consistent', a['files_pre'] == a['files_post'] and a['same_paths'] and
      a['pre_snapshot_utc'] <= summ['start_utc'] and a['post_snapshot_utc'] >= summ['end_utc'] and
      all(v['missing_post'] == v['mtime_changed'] == 0 for v in by.values()) and
      'UNMAPPED' not in {k.split('|')[1] for k in by})
    tests = {k: v for k, v in by.items() if k.endswith('|TEST')}
    G('H.test_zero', len(tests) == 3 and all(v['atime_changed'] == 0 and v['consumed'] == 0 and
                                              v['sensitive_whole_window'] == v['files'] for v in tests.values()))
    vals = {k: v for k, v in by.items() if k.endswith('|VAL')}
    G('H.val_zero', len(vals) == 3 and all(v['atime_changed'] == 0 and v['consumed'] == 0 for v in vals.values()))
    G('H.changes_only_consumed_faces', all(v['changed_not_consumed'] == 0 for v in by.values()))
    G('H.manifests_unmodified', all(x['mtime_unchanged'] for x in a['manifests']) and len(a['manifests']) > 0)
    G('H.test_flags', man['test_split_accessed'] is False and summ['test_split_accessed'] is False and
      man['test_code_path_present'] is False)

    # I. environment / driver ruling / D12
    er = t['environment_recheck']
    G('I.driver_ruling_run_start', rs['gpu_driver'] == {
        'driver_policy': DRIVER_RULING['driver_policy'], 'gpu_model': E['gpu_model'], 'gpu_uuid': E['gpu_uuid'],
        'lock_driver_version': DRIVER_RULING['lock_driver_version'],
        'observed_driver_version': DRIVER_RULING['observed_driver_version']})
    G('I.environment_equals_lock_except_driver', rs['environment_identity_equals_lock_except_driver'] is True and
      er['differences_other_than_driver'] == [] and er['driver'] == rs['gpu_driver'])
    G('I.live_gpu', host['gpu_query'] == '%s, %s, %s' % (E['gpu_model'], DRIVER_RULING['observed_driver_version'],
                                                          E['gpu_uuid']) == er['gpu'])
    G('I.runtime', er['tensorflow'] == man['tensorflow_version'] == '1.15.5' and
      er['native_runtime'] == {'cudaRuntimeGetVersion': 11080, 'cuda_query_rc': 0, 'cudnnGetVersion': 8907} and
      man['python_executable'] == t['collector']['executable'] == E['python_executable'] and
      t['collector']['launch_environment_exact'] is True and er['session_config'] == rs['session_config'])
    G('I.python38_backport_disclosed', rs['python38_pathlib_is_relative_to_backport'] is True)
    G('I.d12_authority', d12['status'] == 'OWNER_APPROVED' and d12['approved'] is True and
      d12['acceptance_criteria_sha256'] == D12_CRITERIA_SHA and
      qual['status'] == 'E04_PRODUCTION_QUALIFIED_READY_FOR_SCIENTIFIC_EXECUTION' and
      qual['gates_passed'] == qual['gates_total'] == 52 and qual['d12']['acceptance_criteria_sha256'] == D12_CRITERIA_SHA
      and all(qual['d12']['criteria'].values()))

    # J. run-root integrity / exclusivity
    G('J.run_root_unchanged', c['run_root_unchanged_whole_audit'] and t['run_root_tree_unchanged'] and
      c['run_root_sha_listing_digest'] == c['run_root_sha_listing_digest_before'] and
      c['run_root_tree_sha256'] == t['run_root_tree_sha256'])
    G('J.run_root_files', c['run_root_file_count'] == len(c['run_root_sha_listing']) == RUN_ROOT_FILES,
      c['run_root_file_count'])
    G('J.metrics_sha_in_listing', lines.get('metrics.jsonl', {}).get('sha256') == c['metrics_sha256'])
    sib_ok, sib = finished_siblings(host, seed)
    G('J.no_other_e04_runner', host['runner_processes'] == [] and sib_ok,
      host['other_e04_tmux'] if not sib else {'sessions': host['other_e04_tmux'], 'finished_earlier_siblings': sib})

    failed = G.failed
    status = 'E04_SEED%d_COMPLETED_CLEANLY' % seed if not failed else 'E04_SEED%d_NEEDS_REVIEW' % seed
    text = manifest_text(c, t)
    by_split = {}
    for k, v in by.items():
        sp = by_split.setdefault(k.split('|')[1], {'files': 0, 'sensitive_whole_window': 0, 'atime_changed': 0,
                                                   'changed_consumed': 0, 'changed_not_consumed': 0, 'consumed': 0})
        for key in sp:
            sp[key] += v[key]
    bs = m['block_summary']
    doc = {
        'milestone': 'E04-SEED%d-COMPLETION' % seed, 'classification': CLASSIFICATION, 'status': status,
        'failed_gates': failed, 'gates': G.results, 'gates_passed': len(G.results) - len(failed),
        'gates_total': len(G.results),
        'method': {'id': 'E04', 'name': 'Physics-STD', 'fidelity': FIDELITY, 'reporting_label': REPORTING_LABEL,
                   'track': 'A_FAIR_COMMON_IDENTITY_FREE'},
        'seed': seed, 'run_id': rid, 'run_id_rule': 'sha256("E04|%d|%s|%s")[:16]' % (seed, CONFIG_SHA, AUTHORITY),
        'scientific_authority': AUTHORITY, 'qualification_authority': QUALIFICATION_AUTHORITY,
        'd12': {'ruling_commit': D12_RULING_COMMIT, 'activation_commit': D12_ACTIVATION_COMMIT,
                'acceptance_criteria_sha256': D12_CRITERIA_SHA,
                'amendment_sha256': PINNED_FILES['configs/amendments/e04_d12_procedural_reproducibility.yaml'],
                'disclosure': ('D12 procedural reproducibility: independent GPU executions of the frozen graph are not '
                               'bitwise identical (GPU ScatterNd gradients of the warp GatherNd ops and the forward '
                               'AddN T_A); this run is not compared bitwise to any other execution. It never excuses '
                               'a missing or wrong optimizer update, non-finite state or a corrupted checkpoint, which '
                               'are gated above (D, F, G).')},
        'config_sha256': CONFIG_SHA, 'resume': E['resume'],
        'run_root': paths(seed)['run_root'], 'start_utc': summ['start_utc'], 'end_utc': summ['end_utc'],
        'training_duration_seconds': summ['training_duration_seconds'],
        'counts': {'iterations': m['iteration_records'], 'final_global_iteration': m['final_global_step'],
                   'g_applications': 2 * m['iteration_records'], 'd_applications': m['iteration_records'],
                   'checkpoints': len(tk), 'checkpoint_files': len(c['checkpoint_index']),
                   'metrics_records': m['lines'], 'face_reads': summ['face_reads'],
                   'pair_rows_consumed': m['streams']['pair_rows']},
        'optimizer_counter_evidence': {
            'beta_power_rule': 'TF1 Adam: beta_power = float32(beta)^(applications + 1) by float32 multiplication',
            'G_optimizer': 'G_Adam (beta1_power, beta2_power): applied at Step 1 and Step 3 (shared slots)',
            'D_optimizer': 'D_Adam (beta1_power_1, beta2_power_1): applied at Step 2',
            'exact_normal_range': beta['exact_normal_range'],
            'flushed_or_underflowed': beta['flushed_or_underflowed'],
            'note': ('beta1 powers underflow to 0 after ~1000 applications; beta2 powers are exact float32 values '
                     'identifying the application count until they leave the normal float32 range, after which the '
                     'GPU flushes them to zero. global_iteration and minibatch_phase are exact at every checkpoint.')},
        'rng': {'experiment_seed': seed, 'tf_graph_seed': seed, 'pair_order_seed': pseed, 'hard_sample_seed': hseed,
                'pair_sha256_int32_le': s['pair_sha256_int32_le'], 'hard_sha256': s['hard_sha256'],
                'approved_vectors': 'outputs/audit/E04_RNG_TEST_VECTORS.json',
                'record_by_record_replay': {k: m['checks'][k] for k in ('pair_rows', 'pair_ids', 'pair_passes',
                                                                        'hard_draws', 'hard_input_sha256',
                                                                        'depth_input_sha256')},
                'offsets_sample': {'checked': os_['checked'], 'match': os_['match']},
                'rgb_input': ('not recomputed (would open face images); the runner SHA256-checks every face PNG '
                              'against the feed pack before decoding')},
        'loss_diagnostics': {
            'first_block': bs[0] if bs else None, 'last_block': bs[-1] if bs else None,
            'block_size': BLOCK, 'blocks': len(bs),
            'note': 'training losses only; no VAL/TEST metric exists or is inferred; nothing is selected by loss'},
        'publication_data': {
            'per_iteration_source': paths(seed)['run_root'] + '/metrics.jsonl',
            'metrics_jsonl_sha256': c['metrics_sha256'],
            'fields': ['train_losses (step1: L_G, L_P, L_R, L_depth, P0_negative_prior, P_positive, loss_G_step1; '
                       'step2: L_D, loss_D; step3: L_H, L_S, loss_G_step3)', 'learning_rates (G, D)',
                       'step_seconds', 'wall_clock_seconds', 'gpu_memory_bytes', 'utc', 'pair_rows', 'pair_ids',
                       'hard_scales', 'hard_keep', 'input_sha256'],
            'block_summary_in_collected': 'collected.metrics.block_summary (%d x %d iterations)' % (len(bs), BLOCK)},
        'runtime': {'step_seconds': m['step_seconds'], 'tf_max_bytes_in_use_max': m['gpu_memory_max_bytes'],
                    'slow_blocks': [[b['first_iteration'], round(b['median_step_seconds'], 3)] for b in bs
                                    if b['median_step_seconds'] >= 1.25 * m['step_seconds']['median']],
                    'gpu_sharing': ('per-iteration step_seconds are the contention record; concurrent GPU workloads '
                                    'were not sampled continuously during the run (launch record: no other compute '
                                    'process at launch; nonexclusive exception authorised); slower blocks are '
                                    'consistent with intermittent sharing and are not a validity criterion')},
        'trainable_change': {
            'changed_vs_previous_checkpoint': {r['iteration']: r['trainable_differs_from_previous']
                                               for r in t['checkpoints'][1:]},
            'note': ('a trainable that keeps its exact float32 value between checkpoints late in training is float32 '
                     'absorption: at lr 5e-8 (G) / 2.5e-8 (D) an Adam step is below half an ulp of a value near 1 '
                     '(BN gamma); this is expected arithmetic, not a missing update')},
        'checkpoints': {'files': [{'iteration': r['iteration'], 'sha256': {x: r['files'][x]['sha256'] for x in SUFFIXES},
                                   'size': {x: r['files'][x]['size'] for x in SUFFIXES},
                                   'state_sha256': r['state_sha256']} for r in t['checkpoints']],
                        'final_selected': {'prefix': prefix(ITERATIONS), 'rule': E['checkpoint_rule'],
                                           'path': ck_dir + 'iteration_150000',
                                           'sha256': {x: fin.get(x, {}).get('sha256') for x in SUFFIXES},
                                           'state_sha256': tk.get(ITERATIONS, {}).get('state_sha256')},
                        'manifest_file': evidence(seed)['manifest'], 'manifest_sha256': sha256_bytes(text.encode()),
                        'resume_sidecars': 0},
        'firewall': {'test_access': 0 if not failed else None, 'by_split': by_split,
                     'test_files': by_split.get('TEST', {}).get('files'),
                     'test_changed': by_split.get('TEST', {}).get('atime_changed'),
                     'scope': 'TRAIN (pairs_train_v1 via the E04 feed pack) only; no VAL, no TEST', 'atime': a},
        'driver_ruling': DRIVER_RULING,
        'environment': {'python_executable': E['python_executable'], 'tensorflow': er['tensorflow'],
                        'native_runtime': er['native_runtime'], 'packages': er['packages'],
                        'run_platform': man['platform'], 'audit_platform': t['collector']['platform'],
                        'python38_pathlib_backport': rs['python38_pathlib_is_relative_to_backport']},
        'host_continuity': cont,
        'run_root_integrity': {'file_count': c['run_root_file_count'],
                               'sha_listing_digest': c['run_root_sha_listing_digest'],
                               'sha_listing_convention': ('LC_ALL=C sorted "<path> <size> <sha256>" lines, sha256 of '
                                                          'the newline-terminated listing'),
                               'metrics_jsonl_sha256': c['metrics_sha256'],
                               'unchanged_whole_audit': c['run_root_unchanged_whole_audit']},
        'host_markers': {'exit': host['exit_file'], 'launch_time': host['launch_time'], 'end_time': host['end_time'],
                         'command_file_sha256': host['command_file_sha256'], 'tmux': paths(seed)['tmux'],
                         'tmux_live_state': {'pane_dead': host['tmux_pane_dead'],
                                             'tmux_ls': host['boot']['tmux_ls']}},
        'confirmations': {'resume_used': False, 'val_checkpoint_selection': False, 'test_selection': False,
                          'best_checkpoint_logic': False, 'best_seed_logic': False, 'bank_generation': False,
                          'downstream_evaluation': False, 'run_root_mutated': False, 'downstream_metric': None},
        'collected': c, 'collected_tf': t,
    }
    return doc, text


def report(doc):
    d, c, t = doc, doc['collected'], doc['collected_tf']
    seed, n = d['seed'], d['counts']
    fin = d['checkpoints']['final_selected']
    hc = d['host_continuity']
    lb, fb = d['loss_diagnostics']['last_block'], d['loss_diagnostics']['first_block']
    keys = ['step1/loss_G_step1', 'step1/L_depth', 'step2/loss_D', 'step3/loss_G_step3']
    rows = ''.join('| %s | %d | %d | %d | %d | %d | %d |\n' % (k, v['files'], v['sensitive_whole_window'],
                                                              v['atime_changed'], v['changed_consumed'],
                                                              v['changed_not_consumed'], v['consumed'])
                   for k, v in sorted(d['firewall']['by_split'].items()))
    ex = d['optimizer_counter_evidence']['exact_normal_range']
    rt = d['runtime']['step_seconds']
    return '\n'.join([
        '# E04 seed%d: scientific run %s' % (seed, 'completed cleanly' if not d['failed_gates'] else 'NEEDS REVIEW'),
        '',
        '**Status:** `%s`, %d/%d gates passing. The machine-readable record is `E04_SEED%d_COMPLETION.json`; it '
        'embeds both read-only host collections, so `tools/e04_seed_completion_audit.py build` can re-derive it.'
        % (d['status'], d['gates_passed'], d['gates_total'], seed),
        '',
        'This commit is reporting evidence only. The E04 scientific launch authority stays **`%s`** (production '
        'qualification `%s`, D12 ruling `%s`, criteria `%s…`).' % (AUTHORITY, QUALIFICATION_AUTHORITY[:7],
                                                                   D12_RULING_COMMIT[:7], D12_CRITERIA_SHA[:8]),
        '',
        '| | |', '|---|---|',
        '| method | E04 **Physics-STD**, track %s, fidelity `%s` (reporting label "%s") |'
        % (d['method']['track'], d['method']['fidelity'], d['method']['reporting_label']),
        '| run | seed = %d, run_id = **`%s`** = sha256("E04\\|%d\\|41861794…\\|18ba00e…")[:16] (recomputed) |'
        % (seed, d['run_id'], seed),
        '| resume | `%s`: one run_start, fresh run root |' % d['resume'],
        '| window | %s → %s, %.0f s of training; step time median %.3f s (p05 %.3f, p95 %.3f, max %.1f) |'
        % (d['start_utc'], d['end_utc'], d['training_duration_seconds'], rt['median'], rt['p05'], rt['p95'],
           rt['max']),
        '| exit | `~/workdir/e04_s%d.exit` = %s, written by the launch wrapper after the runner returned; end marker '
        '%s; the runner completion record is the last pane-log line; no runner process; stdout/stderr/generation_log '
        'empty |' % (seed, d['host_markers']['exit'], d['host_markers']['end_time']),
        ('| host | no reboot or shutdown inside the run window (run boot `%s`; next boot `%s`, after the run ended); '
         'the tmux server is absent now (later reboots), so the live pane is not used as evidence |'
         % (' '.join((hc['run_boot_wtmp'] or '').split()[3:5]), ' '.join((hc['next_boot_wtmp'] or '').split()[3:5]))
         if hc['next_boot_wtmp'] else
         '| host | no reboot or shutdown inside the run window: the host has not rebooted since before launch (run boot '
         '= current boot `%s`, wtmp "still running", boot_id `%s`, btime %s; journal boot spans %s → %s); the tmux '
         'session is still present (pane dead after the wrapper finished) but is not required as evidence |'
         % (' '.join((hc['run_boot_wtmp'] or '').split()[3:5]), hc['current_boot_id'], hc['current_btime_utc'],
            hc['journal_boot_alive_at_end'][0][1] if hc['journal_boot_alive_at_end'] else None,
            hc['journal_boot_alive_at_end'][0][2] if hc['journal_boot_alive_at_end'] else None)),
        '',
        '## Completion accounting',
        '',
        '- **Iterations = %d** (contiguous 0..149999, global_step = iteration + 1), **final global_iteration = %d**, '
        'minibatch_phase 0 at every checkpoint.' % (n['iterations'], n['final_global_iteration']),
        '- **G applications = %d** (Step 1 Eq.23 and Step 3 Eq.24, one shared G Adam), **D applications = %d** '
        '(Step 2 L_D). Each iteration record holds the losses fetched in the three apply runs; the graph asserts the '
        'G → D → G phase order.' % (n['g_applications'], n['d_applications']),
        '- **Adam beta powers:** %d checkpoint values in the normal float32 range equal float32(0.999)^(applications+1) '
        'exactly, e.g. %s. The rest are flushed to 0 as expected.'
        % (len(ex), '; '.join('%s at %d = %d applications' % (x[1], x[0], x[2]) for x in ex[:3])),
        '- **Learning rate:** G 5e-5·10^-⌊t/45000⌋, D half; the largest relative float32 read-back error is %.2g.'
        % c['metrics']['max_relative_lr_error'],
        '- **Metrics log:** %d records (1 run_start, %d iterations, %d checkpoint events), every line parses, no '
        'missing or duplicate iteration, every loss component finite.' % (n['metrics_records'], n['iterations'],
                                                                          n['checkpoints']),
        '',
        'Training-loss block means (iterations %d–%d and %d–%d), for orientation only:'
        % (fb['first_iteration'], fb['first_iteration'] + fb['iterations'] - 1, lb['first_iteration'],
           lb['first_iteration'] + lb['iterations'] - 1),
        '',
        '| loss | first block | last block |', '|---|---|---|'] + [
        '| %s | %.6g | %.6g |' % (k, fb['loss_means'][k], lb['loss_means'][k]) for k in keys] + [
        '',
        '## Data feed and RNG (D4, D7, D8, D2/D3)',
        '',
        '- **Seeds:** experiment %d; PAIR_ORDER %d; HARD_SAMPLE %d; TF graph seed %d.'
        % (seed, d['rng']['pair_order_seed'], d['rng']['hard_sample_seed'], seed),
        '- **Streams:** pair stream `%s` and hard stream `%s` equal the approved vectors and the production stream '
        'function.' % (d['rng']['pair_sha256_int32_le'], d['rng']['hard_sha256']),
        '- **Record by record:** an independent numpy RandomState replay of the rule text matches the pair rows, pair '
        'ids, pass indices and hard draws of all %d iterations. The hard and depth input sha256 recomputed from the '
        'replay and the feed pack also match. The dense offsets of %d sampled iterations match, recomputed from the '
        'pack geometry.' % (n['iterations'], d['rng']['offsets_sample']['match']),
        '- **Faces:** %d TRAIN samples consumed (the whole pack). Split, label, dataset and M2 face sha256 match for '
        'all of them. Feed pack `%s`, pre-pass `%s`, 0 VAL and 0 TEST rows.'
        % (c['consumed_faces']['samples'], c['pack']['manifest_sha256'], c['pack']['prepass_manifest_sha256']),
        '',
        '## Checkpoints (%s)' % fin['rule'],
        '',
        '- iteration_015000 … iteration_150000 (30 files) exist and equal the index and the checkpoint events.',
        '- Every checkpoint holds the 500 graph variables, all finite. Its state fingerprint equals the live-state '
        'fingerprint logged by the runner at save time.',
        '- Trainables that changed against the previous checkpoint (of 144): %s. In the final block, the unchanged '
        'ones are float32 absorption, not a missing update: at lr 5e-8 an Adam step is below half an ulp of a BN '
        'gamma near 1.' % ', '.join('%d: %d' % (int(k), v) for k, v in
                                    sorted(d['trainable_change']['changed_vs_previous_checkpoint'].items(),
                                           key=lambda kv: int(kv[0]))),
        '- The final checkpoint is **iteration_150000**, selected by the frozen rule, not by loss or VAL:',
        '  - path `%s`' % fin['path'],
        '  - `.data-00000-of-00001` **`%s`**' % fin['sha256']['.data-00000-of-00001'],
        '  - `.index` `%s`' % fin['sha256']['.index'],
        '  - `.meta` `%s`' % fin['sha256']['.meta'],
        '- No partial, temporary or resume file. Full list: `E04_SEED%d_CHECKPOINT_MANIFEST.txt`.' % seed,
        '',
        '## Firewall',
        '',
        'Stat-only atime snapshots: pre %s, post %s, %d files. No image was opened by the audit.'
        % (c['atime_firewall']['pre_snapshot_utc'], c['atime_firewall']['post_snapshot_utc'],
           c['atime_firewall']['files_pre']),
        '',
        '| split | files | atime-sensitive | changed | changed, consumed | changed, not consumed | consumed |',
        '|---|---|---|---|---|---|---|',
        rows.rstrip('\n'),
        '',
        '**TEST = 0** and **VAL = 0**: none of those faces changed, and no consumed sample is VAL or TEST. '
        '`test_split_accessed` and `val_split_accessed` are false.',
        '',
        '## Environment, D9 and D12',
        '',
        '- Driver: lock 595.84, observed **580.178.04**, `%s`. GPU model and UUID are exact, and there is no other '
        'lock difference, both at run_start and on the live re-check.' % d['driver_ruling']['driver_policy'],
        '- TensorFlow %s, CUDA runtime %d, cuDNN %d, %d locked packages; Python 3.8 pathlib backport disclosed.'
        % (d['environment']['tensorflow'], d['environment']['native_runtime']['cudaRuntimeGetVersion'],
           d['environment']['native_runtime']['cudnnGetVersion'], d['environment']['packages']),
        '- OS kernel during the run: `%s`. At audit time: `%s`. The kernel is not part of the E04 lock; it is '
        'disclosed only.' % (d['environment']['run_platform'], d['environment']['audit_platform']),
        '- D12: %s' % d['d12']['disclosure'],
        '- GPU sharing: not sampled continuously. %d of %d blocks of %d iterations have a median step time of at least '
        '1.25 × the overall median. That is consistent with intermittent sharing and is not a validity criterion.'
        % (len(d['runtime']['slow_blocks']), d['loss_diagnostics']['blocks'], d['loss_diagnostics']['block_size']),
        ('- Host after completion: the run boot ended without a wtmp shutdown record when the host restarted at `%s` '
         '(after the run ended). Later shutdowns: %s. Kernel updates installed after the run (unattended upgrades) '
         'changed only the OS kernel; the NVIDIA driver version is unchanged.'
         % (' '.join((hc['next_boot_wtmp'] or '').split()[4:5]),
            '; '.join('`%s`' % ' '.join(x.split()[3:5]) for x in hc['shutdowns_after_end']) or 'none')
         if hc['next_boot_wtmp'] else
         '- Host after completion: no reboot or shutdown since before launch (the audit ran in the same boot, '
         '`%s`); the kernel during the run and at audit time are the same boot.' % hc['current_boot_id']),
        '',
        '## Run-root integrity',
        '',
        '- %d files; LC_ALL=C `<path> <size> <sha256>` listing digest **`%s`**, identical before and after the audit.'
        % (d['run_root_integrity']['file_count'], d['run_root_integrity']['sha_listing_digest']),
        '- metrics.jsonl `%s`. It holds every per-iteration loss, learning rate and timing value needed for later '
        'plots.' % d['run_root_integrity']['metrics_jsonl_sha256'],
        '',
        '## Boundary',
        '',
        'No downstream metric, synthetic bank, best-checkpoint or best-seed logic. ACER, APCER, BPCER, EER and ROC '
        'need the later frozen downstream pipeline; none is derived from training losses.',
        ''])


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd')
    for name in ('collect', 'collect-tf'):
        a = sub.add_parser(name)
        a.add_argument('--seed', type=int, required=True, choices=sorted(RUN_IDS))
        a.add_argument('--repo', required=True)
    b = sub.add_parser('build')
    b.add_argument('--collected', required=True)
    b.add_argument('--collected-tf', required=True)
    b.add_argument('--repo-root', default=str(Path(globals().get('__file__', 'tools/x')).resolve().parents[1]))
    args = ap.parse_args(argv)
    if args.cmd == 'collect-tf':
        json.dump(collect_tf(args.seed, args.repo), sys.stdout, sort_keys=True)
        return 0
    if args.cmd == 'collect':
        json.dump(collect(args.seed, args.repo), sys.stdout, sort_keys=True)
        return 0
    c = json.loads(Path(args.collected).read_text())
    t = json.loads(Path(args.collected_tf).read_text())
    doc, text = build(c, t, args.repo_root)
    root, ev = Path(args.repo_root), evidence(c['seed'])
    (root / ev['manifest']).write_text(text)
    (root / ev['completion']).write_text(json.dumps(doc, indent=2, sort_keys=True) + '\n')
    (root / ev['report']).write_text(report(doc))
    (root / ev['sha256']).write_text(''.join('%s  %s\n' % (sha256_file(root / ev[k]), ev[k])
                                             for k in ('manifest', 'completion', 'report')))
    print(doc['status'], doc['gates_passed'], '/', doc['gates_total'], doc['failed_gates'])
    return 0 if not doc['failed_gates'] else 1


if __name__ == '__main__':
    sys.exit(main())
