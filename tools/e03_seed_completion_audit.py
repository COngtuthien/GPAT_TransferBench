"""E03 seed closure: read-only completion audit of one E03 STDN scientific run
(FAITHFUL_OFFICIAL_CORE_WITH_TRACK_A_CONTROLLED_DATA_ADAPTATION_AND_RUNTIME_COMPATIBILITY, track
A_FAIR_COMMON_IDENTITY_FREE, training authority d547fc9 for every E03 seed).

Three phases (the two collectors never write anything; this file must stay parseable by Python 3.8):

  collect-tf  (GPU host, gpat-m6-e03 Python 3.8 / TF 1.15.5, exact launch environment of the audited seed,
              `python -I -B - collect-tf --seed <s> --repo <gpu repo> < this file`) -- prints one JSON document.
              Re-verifies the frozen contract, source cache and feed pack with the production verifiers, re-reads the
              live environment against the E03 lock (owner driver ruling: version only), opens all 50 official TF
              Saver checkpoints with tf.train.NewCheckpointReader (no session, no restore) and checks every variable
              finite, global_step = 3000 e, the variable set/shapes against the M6D2b oracle, and re-derives the
              TRAIN/VAL pair-row streams in a separate CPU-only graph (graph seed = experiment seed, shuffle op seed =
              split seed, reshuffle_each_iteration=True, repeat) without reading the run's metrics.
  collect     (GPU host, a Python 3.11 env with pyarrow -- gpat-m6-e05, CUDA_VISIBLE_DEVICES="",
              `python -I -B - collect --seed <s> --repo <gpu repo> < this file`) -- prints one JSON document.
              Lists the run root (path/size/sha256) before and after, scans every metrics.jsonl line against the
              official schedule (2000 TRAIN -> ckpt-e -> 500 VAL per epoch, G/D/global_step accounting, staircase
              learning rate), re-derives every TRAIN/VAL flip from numpy RandomState(split seed) (independent of TF),
              maps every consumed face through the hash-verified feed pack, split_v1 and the M2 accounting, and
              compares the stat-only faces_256 atime snapshots (~/workdir/e03_s<seed>_atime_{pre,post}.tsv). No
              face image is opened.
  build       (repository, stdlib only) -- gates both documents and writes outputs/audit/E03_SEED<s>_{COMPLETION.json,
              COMPLETION.md, CHECKPOINT_MANIFEST.txt, EVIDENCE_SHA256.txt}. COMPLETION.json embeds both collected
              documents so the build can be re-derived on any host.

No training, restore, selection, TEST, bank or downstream code is run. VAL losses are reported as diagnostics only.
"""
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

CLASSIFICATION = 'SCIENTIFIC_TRAINING_COMPLETION_AUDIT'
RUNTIME = '/home/student20261/workdir/GPAT_TransferBench_runtime'
WORKDIR = '/home/student20261/workdir'
AUTHORITY = 'd547fc9f8932a273d719ae589aade85a9d989b85'
CONFIG_SHA = '08dde851bc9e8ac688c794a6a6999204a7e5b75fb244fcd97e46b22dcb302fe0'
RUN_IDS = {42: 'a565b76bc7349532'}
VAL_SEEDS = {42: 1355875380, 1337: 1050417158, 2026: 1242808125}     # owner ruling V (qualified mappings)
FIDELITY = 'FAITHFUL_OFFICIAL_CORE_WITH_TRACK_A_CONTROLLED_DATA_ADAPTATION_AND_RUNTIME_COMPATIBILITY'
EXPECTED = {
    'method_id': 'E03', 'method_name': 'STDN', 'host': 'nd-System-Product-Name', 'fidelity': FIDELITY,
    'track': 'A_FAIR_COMMON_IDENTITY_FREE', 'resume': 'NOT_QUALIFIED_FRESH_ONLY',
    'checkpoint_rule': 'OFFICIAL_LATEST_FINAL_CKPT_50', 'environment_name': 'gpat-m6-e03',
    'python_executable': '/home/student20261/miniconda3/envs/gpat-m6-e03/bin/python',
    'gpu_model': 'NVIDIA GeForce RTX 3090', 'gpu_uuid': 'GPU-b722cd9d-c9fa-223c-915e-cc09d9b862b7',
    'source_commit': 'c79f1f8c615d2b8471b3df29da881bb18dd54c90',
    'source_tree': '991a4f127990f98a108b8426113396f1c39f558d',
    'identities': {
        'compatibility_manifest_sha256': '4cf2d14ed302f15418b887dfef5e45312328bc1e9e4273c2ea74ef607e447f84',
        'config_sha256': CONFIG_SHA,
        'environment_lock_sha256': 'b2c6c2eadca1b93a0bdf6798990821cb889e199643a5bdb5368ed0b12fdc8c62',
        'fidelity': FIDELITY,
        'logging_contract_sha256': 'd33622947d1951d71bbda4cf86b96c6500943d354fe607beb6cbc47f51c3de15',
        'pack_manifest_sha256': 'c8c9850edf783a7fce392c934e721feb29a0b88a9223612a39bd33ee46813833',
        'pair_manifest_sha256': {'TRAIN': 'a5e4fdaef236f15730c7e3885b537e08e995faffbe654167fc940f44bbc75243',
                                 'VAL': '84d124919a52cd8d84a89766f464a4dcde1aeaa7791218f6356813a40904f872'},
        'preprocess_sha256': '265b75755ecede0d9c01f6d1e22f4d232767ec1e15faf8d7d55de11a68996e9e',
        'production_files_sha256': {
            'methods/stdn/feed_pack.py': '12125864491d5142d253406b22c7f080ea7ba31ef7ac8b06406f80b5ce9b8cf0',
            'methods/stdn/production_io.py': 'ba4e2938eec887d7baffc8858ec7b9c69de89eea6663c9b9eb2fa5bdc96ae4e1',
            'methods/stdn/production_runner.py': '17a72e929d2e86f9f2b4414ff99ffd0a0d29b48e33ebbfba19cde5061932d07f',
            'tools/run_e03.py': '59a888a63b4468b84f399ab698e0ade9c2f211b064dfa3160163838aa59e34ad'},
        'ruling_addendum_sha256': 'ad3bf3b13815ca013fbd72237cf4bebe40473729f1e00507fe0ac6d209446f8a',
        'ruling_yaml_sha256': '4bf6f87967a837be7c26efb61a1c3cc13285c16e52c8ca5c7de424b696b85d3b',
        'runtime_addendum_sha256': 'f5133f81ed1a0567a7bb1426fc23e05d43e9d689d6478b0f37ec2377832d9d0e',
        'source_commit': 'c79f1f8c615d2b8471b3df29da881bb18dd54c90',
        'source_files_sha256': 'ef4975ecf8d27d9f20681889c281816dca24ffb243cb26d3f3128e691e633f53',
        'source_tree': '991a4f127990f98a108b8426113396f1c39f558d',
        'spec_sha256': 'f7d2371682ad9504f59f0c6984e4b1a741f855f1b1f60838f9d60b160281489e',
        'split_manifest_sha256': 'fb9aeb369a124fc96ba855ef2ce269236c4a743fe960e73ab739412c9cb5092d'},
    'pack_counts': {'landmark_rows': 15321, 'm2_rows_not_taken': 5294, 'pairs': {'TRAIN': 8838, 'VAL': 1905},
                    'samples_by_role': {'TRAIN_live': 3830, 'TRAIN_spoof': 8838, 'VAL_live': 748, 'VAL_spoof': 1905},
                    'samples_by_split': {'TRAIN': 12668, 'VAL': 2653}, 'unique_samples': 15321},
}
DRIVER_RULING = {'lock_driver_version': '595.84', 'observed_driver_version': '580.178.04',
                 'driver_policy': 'OWNER_APPROVED_DRIVER_VERSION_ONLY_RUNTIME_COMPATIBILITY',
                 'scope': ('only the NVIDIA driver version may differ; GPU model, GPU UUID and every other lock field '
                           'are exact gates; this is not evidence that arbitrary driver changes are acceptable')}
PRECISION_ENV = {
    'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'CUDA_VISIBLE_DEVICES': '0', 'NVIDIA_TF32_OVERRIDE': '0',
    'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONNOUSERSITE': '1', 'TF_CUDNN_DETERMINISTIC': '1', 'TF_DETERMINISTIC_OPS': '1',
    'TF_ENABLE_AUTO_MIXED_PRECISION': '0', 'TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE': '0',
    'TF_ENABLE_CUBLAS_TENSOR_OP_MATH_FP32': '0', 'TF_ENABLE_CUDNN_RNN_TENSOR_OP_MATH_FP32': '0',
    'TF_ENABLE_CUDNN_TENSOR_OP_MATH_FP32': '0', 'TF_ENABLE_ONEDNN_OPTS': '0', 'TF_XLA_FLAGS': '--tf_xla_auto_jit=0'}
MAX_EPOCH, STEPS, VAL_STEPS, BATCH, G_D_RATIO = 50, 2000, 500, 2, 2
ROWS = {'TRAIN': 8838, 'VAL': 1905}
GS_PER_EPOCH = STEPS + STEPS // G_D_RATIO                      # 3000
FINAL_GS = MAX_EPOCH * GS_PER_EPOCH                             # 150000
TRAIN_STEPS, VAL_TOTAL = MAX_EPOCH * STEPS, MAX_EPOCH * VAL_STEPS
CONSUMED = {'TRAIN': TRAIN_STEPS * BATCH, 'VAL': VAL_TOTAL * BATCH}     # pair rows actually run
PER_EPOCH_ROWS = {'TRAIN': STEPS * BATCH, 'VAL': VAL_STEPS * BATCH}
LR0, DECAY, DECAY_STEPS = 6e-5, 0.9, 20000
TOL_LR = 1e-6                                                   # relative; float32 read-back of exponential_decay
N_VARIABLES = 747
SUFFIXES = ('.data-00000-of-00001', '.index', '.meta')
LOSS_KEYS = ('a_loss', 'd_loss', 'g_loss')
BRIGHTNESS_MAX = 0.25 + 1e-6                                    # dataset.py random_brightness(0.25); float32 mean
FEED_LOOKAHEAD_MAX = 4 * BATCH * 2                              # faces parsed beyond consumption (prefetch + batch)
SMALL_FILES = ('run_manifest.json', 'run_summary.json', 'checkpoint_index.json', 'resolved_config.yaml',
               'metrics.jsonl', 'stdout.log', 'stderr.log', 'generation_log.jsonl')
PANE_FAIL_PATTERNS = ('Traceback', 'ResourceExhausted', 'out of memory', 'OOM', 'E03 STOP', 'REFUSED', 'Killed',
                      'Segmentation fault', 'core dumped', 'NaN', 'non-finite')
LOG_PATTERNS = ('Traceback', 'Error', 'error', 'OutOfMemory', 'out of memory', 'Killed', 'nan', 'inf', 'Warning')
RUN_ROOT_FILES = len(SMALL_FILES) + MAX_EPOCH * len(SUFFIXES) + 1 + MAX_EPOCH * (STEPS // 200 + VAL_STEPS // 50)


def paths(seed):
    w = '%s/e03_s%d' % (WORKDIR, seed)
    return {'run_root': '%s/runs/m6/E03/seed_%d' % (RUNTIME, seed), 'tmux': 'e03_stdn_seed%d' % seed,
            'exit': w + '.exit', 'command': w + '.command', 'launch_time': w + '.launch_time',
            'end_time': w + '.end_time', 'pane_log': w + '.pane.log', 'preflight': w + '_preflight.json',
            'atime_pre': w + '_atime_pre.tsv', 'atime_post': w + '_atime_post.tsv'}


def evidence(seed):
    a = 'outputs/audit/E03_SEED%d_' % seed
    return {'completion': a + 'COMPLETION.json', 'manifest': a + 'CHECKPOINT_MANIFEST.txt',
            'report': a + 'COMPLETION.md', 'sha256': a + 'EVIDENCE_SHA256.txt'}


def command(seed):
    return '%s tools/run_e03.py --seed %d --execution-config configs/execution/m5_gpu_3090.yaml' % (
        EXPECTED['python_executable'], seed)


def wrapper_invocation(seed):
    return 'python -u -B tools/run_e03.py --seed %d --execution-config configs/execution/m5_gpu_3090.yaml' % seed


def run_id(seed, commit=AUTHORITY):
    return hashlib.sha256(('E03|%d|%s|%s' % (seed, CONFIG_SHA, commit)).encode()).hexdigest()[:16]


def val_seed(seed):
    """Owner ruling V: 1 + (u32 big-endian of SHA256("GPAT_TRANSFERBENCH|E03|VAL_DATA|<seed>")[0:4] mod 2^31-1)."""
    d = hashlib.sha256(('GPAT_TRANSFERBENCH|E03|VAL_DATA|' + str(seed)).encode('utf-8')).digest()
    return 1 + int.from_bytes(d[:4], 'big') % 2147483647


def expected_lr(global_step):
    return LR0 * DECAY ** (global_step // DECAY_STEPS)


def applications_after(i):
    """Official counters after zero-based TRAIN step i of the whole run (step % 2 == 0 runs g_op + d_op)."""
    e, s = divmod(i, STEPS)
    g = i + 1
    d = e * (STEPS // G_D_RATIO) + s // G_D_RATIO + 1
    return g, d


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


def seq_digest(seq):
    return sha256_bytes(json.dumps([int(x) for x in seq]).encode())


def stream_digests(seq, split):
    """Whole-stream digest plus one digest per epoch block of consumed pair rows."""
    k = PER_EPOCH_ROWS[split]
    return {'rows': len(seq), 'sha256': seq_digest(seq),
            'epoch_sha256': [seq_digest(seq[i * k:(i + 1) * k]) for i in range(len(seq) // k)],
            'head': [int(x) for x in seq[:8]]}


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
        return subprocess.run(['git'] + list(a), stdout=subprocess.PIPE, universal_newlines=True).stdout
    return {'head': git('rev-parse', 'HEAD').strip(), 'branch': git('branch', '--show-current').strip(),
            'status_porcelain': git('status', '--porcelain')}


# ============================================================================ collect-tf (GPU host, gpat-m6-e03)
def predict_streams(tf, np, seed):
    """The run's shuffle ops rebuilt alone in a CPU-only graph; returns the consumed-length pair-row streams."""
    out = {}
    for split, op_seed in (('TRAIN', seed), ('VAL', val_seed(seed))):
        graph = tf.Graph()
        with graph.as_default():
            tf.set_random_seed(seed)                                    # production_runner.seed_process
            ds = tf.data.Dataset.from_tensor_slices(np.arange(ROWS[split], dtype=np.int64))
            ds = ds.shuffle(ROWS[split], seed=op_seed, reshuffle_each_iteration=True).repeat(-1)
            nxt = ds.batch(CONSUMED[split]).make_one_shot_iterator().get_next()
            with tf.Session(graph=graph, config=tf.ConfigProto(device_count={'GPU': 0})) as sess:
                seq = sess.run(nxt).tolist()
        out[split] = dict(stream_digests(seq, split), graph_seed=seed, op_seed=op_seed)
    return out


def collect_tf(seed, repo):
    sys.path.insert(0, str(repo))
    os.chdir(str(repo))
    root = Path(paths(seed)['run_root'])
    before = tree_snapshot(root)
    import importlib.util
    import numpy as np
    from methods.stdn import feed_pack
    from methods.stdn import production_io as pio
    from methods.stdn import production_runner as pr
    import tensorflow as tf
    cfg = pio.verify_contract()
    source = pio.verify_source()
    checkout = pio.verify_source_checkout()
    ids = pio.identities(cfg, source)
    pack = feed_pack.load(pio.pack_root(RUNTIME), pio.PACK_MANIFEST_SHA256)
    lock = json.loads(Path(pio.LOCK[0]).read_text())
    live = pr.live_environment(tf)
    diff, driver = pio.environment_differences(lock, live)
    spec = importlib.util.spec_from_file_location('e03_m6d2b_oracle', 'methods/stdn/runtime_qualification.py')
    oracle_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle_mod)
    oracle = {k[:-2]: v for k, v in oracle_mod.expected_variables().items()}
    ckpts, prev, names0 = [], None, None
    for e in range(1, MAX_EPOCH + 1):
        prefix = root / 'checkpoints' / ('ckpt-%d' % e)
        files = {}
        for suffix in SUFFIXES:
            p = Path(str(prefix) + suffix)
            files[suffix] = {'path': 'checkpoints/' + p.name, 'exists': p.is_file(),
                             'size': p.stat().st_size if p.is_file() else None,
                             'sha256': sha256_file(p) if p.is_file() else None}
        reader = tf.train.NewCheckpointReader(str(prefix))
        shapes = reader.get_variable_to_shape_map()
        dtypes = {k: v.name for k, v in reader.get_variable_to_dtype_map().items()}
        values = {n: reader.get_tensor(n) for n in shapes}
        floating = [n for n in values if np.issubdtype(np.asarray(values[n]).dtype, np.floating)]
        names = sorted((n, list(shapes[n]), dtypes[n]) for n in shapes)
        names0 = names if names0 is None else names0
        rec = {'epoch': e, 'files': files, 'variables': len(values), 'floating': len(floating),
               'nonfinite_variables': sum(int(not np.isfinite(values[n]).all()) for n in floating),
               'global_step': int(values['global_step']) if 'global_step' in values else None,
               'variable_set_sha256': sha256_bytes(json.dumps(names).encode()),
               'variable_set_equals_ckpt1': names == names0,
               'trainable_oracle_present': sum(list(shapes.get(k, [])) == v for k, v in oracle.items()),
               'differs_from_previous': (sum(int(not np.array_equal(values[n], prev[n])) for n in values)
                                         if prev is not None else None),
               'trainable_differs_from_previous': (sum(int(not np.array_equal(values[n], prev[n])) for n in oracle)
                                                   if prev is not None else None),
               'loss_ema_values': {n: float(values[n]) for n in sorted(values) if n.endswith('/avg')}}
        ckpts.append(rec)
        prev = values
    prev = values = None
    state = (root / 'checkpoints' / 'checkpoint').read_text().splitlines()
    streams = predict_streams(tf, np, seed)
    after = tree_snapshot(root)
    return {
        'phase': 'collect-tf', 'seed': seed,
        'collector': {'python': sys.version.split()[0], 'executable': sys.executable, 'tensorflow': tf.__version__,
                      'numpy': np.__version__, 'utc': now_utc(),
                      'launch_environment_exact': all(os.environ.get(k) == v for k, v in
                                                      dict(PRECISION_ENV, PYTHONHASHSEED=str(seed)).items())},
        'git': git_state(), 'identities_recomputed': ids,
        'source_checkout': {k: checkout[k] for k in ('commit', 'tree', 'verification')},
        'pack': {'manifest_sha256': pack.manifest_sha256, 'counts': pack.manifest['counts'],
                 'test_rows': pack.manifest['test_rows'], 'rows': {s: len(v) for s, v in pack.rows.items()}},
        'environment_recheck': {'differences_other_than_driver': diff, 'driver': driver,
                                'live_gpu_driver': live['gpu_driver'], 'live_gpu_uuid': live['gpu_uuid'],
                                'live_tensorflow': live['tensorflow'], 'live_python': live['python'],
                                'live_cuda_cudnn': live['cuda_cudnn_native'], 'packages': len(live['packages'])},
        'trainable_oracle_variables': len(oracle),
        'checkpoints': ckpts, 'checkpoint_state_file': state,
        'predicted_streams': streams,
        'run_root_tree_unchanged': before == after, 'run_root_tree_sha256': sha256_bytes(json.dumps(after).encode()),
    }


# ============================================================================ collect (GPU host, Python 3.11 + pyarrow)
def load_pack(pack_dir):
    raw = (pack_dir / 'pack_manifest.json').read_bytes()
    m = json.loads(raw)
    ok = all(sha256_file(pack_dir / n) == r['sha256'] for n, r in m['payloads'].items())
    docs = {n: json.loads((pack_dir / n).read_text()) for n in ('pairs_train.json', 'pairs_val.json', 'samples.json')}
    samples = {r[0]: dict(zip(docs['samples.json']['columns'], r)) for r in docs['samples.json']['rows']}
    return {'manifest_sha256': sha256_bytes(raw), 'payloads_match': ok, 'test_rows': m['test_rows'],
            'rows': {'TRAIN': docs['pairs_train.json']['rows'], 'VAL': docs['pairs_val.json']['rows']},
            'samples': samples}


def pass_structure(seq, n):
    full = len(seq) // n
    blocks = [seq[k * n:(k + 1) * n] for k in range(full)]
    tail = seq[full * n:]
    return {'full_passes': full, 'permutation_passes': sum(sorted(b) == list(range(n)) for b in blocks),
            'tail_rows': len(tail), 'tail_distinct': len(set(tail)) == len(tail),
            'identical_consecutive_passes': sum(a == b for a, b in zip(blocks, blocks[1:]))}


def scan_metrics(path, seed, pack):
    import numpy as np
    rle, lines, bad = [], 0, 0
    tr = {k: 0 for k in ('index', 'kind', 'schedule', 'global_step', 'continuity', 'lr', 'losses', 'pairs', 'flips',
                         'brightness', 'figure', 'nulls')}
    va = {k: 0 for k in ('index', 'kind', 'global_step', 'lr', 'losses', 'pairs', 'flips', 'figure', 'nulls', 'role')}
    rows = {'TRAIN': [], 'VAL': []}
    flips = {'TRAIN': [], 'VAL': []}
    events, utcs, figures = {}, [], set()
    max_rel_lr, gs_prev, i_tr, i_va = 0.0, 0, 0, 0
    epoch_losses = {'TRAIN': {}, 'VAL': {}}
    step_seconds, gpu_mem, delta_abs_max = [], [], 0.0
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
            payload = r.get('payload') or {}
            events.setdefault(r.get('event'), []).append(dict(payload, _line=lines, _utc=r.get('utc'),
                                                              _seed=r.get('experiment_seed')))
            key = (kind, payload.get('epoch'))
        else:
            kind = r.get('record_kind')
            key = (kind, r.get('epoch'))
            utcs.append(r.get('utc'))
            step_seconds.append(r.get('step_seconds'))
            gpu_mem.append(r.get('gpu_memory_bytes'))
            if r.get('figure'):
                figures.add(r['figure'])
        if rle and rle[-1][0] == key[0] and rle[-1][1] == key[1]:
            rle[-1][2] += 1
        else:
            rle.append([key[0], key[1], 1])
        if kind == 'TRAIN_STEP':
            e, s = divmod(i_tr, STEPS)
            g, d = applications_after(i_tr)
            gd = s % G_D_RATIO == 0
            tr['index'] += (r['epoch'] == e + 1 and r['train_step_in_epoch'] == s and r['train_step_total'] == i_tr)
            tr['kind'] += r['record_type'] == 'epoch'
            tr['schedule'] += (r['optimizer_ops_applied'] == (['g_op', 'd_op'] if gd else ['g_op']) and
                               r['applications'] == {'G': g, 'D': d})
            tr['global_step'] += r['global_step'] == r['shared_global_step_after'] == g + d
            tr['continuity'] += (r['shared_global_step_before'] == gs_prev and
                                 r['shared_global_step_after'] - gs_prev == (2 if gd else 1))
            exp = expected_lr(gs_prev)
            rel = max(abs(r['learning_rates'][k] - exp) / exp for k in ('G', 'D'))
            max_rel_lr = max(max_rel_lr, rel)
            tr['lr'] += (rel <= TOL_LR and r['learning_rate'] == r['learning_rates']['G'] == r['learning_rates']['D'])
            ls = r['train_losses']
            tr['losses'] += sorted(ls) == list(LOSS_KEYS) and all(finite(ls[k]) for k in LOSS_KEYS)
            pr_ = [int(x) for x in r['pair_rows']]
            tr['pairs'] += (len(pr_) == BATCH and r['pair_ids'] == [pack['rows']['TRAIN'][x][0] for x in pr_])
            fl = r['flips_live_spoof']
            tr['flips'] += len(fl) == BATCH and all(x in ([0, 0], [0, 1], [1, 0], [1, 1]) for x in fl)
            dl = r['brightness_delta_live_spoof']
            dmax = max(abs(x) for p in dl for x in p)
            delta_abs_max = max(delta_abs_max, dmax)
            tr['brightness'] += (len(dl) == BATCH and all(finite(x) for p in dl for x in p) and dmax <= BRIGHTNESS_MAX)
            want = 'diagnostics/figures/Epoch-%d-%d.png' % (e + 1, s + 1) if s % 200 == 0 else None
            tr['figure'] += r['figure'] == want
            tr['nulls'] += (r['val_losses'] is None and r['val_metrics'] is None and r['train_metrics'] is None)
            rows['TRAIN'] += pr_
            flips['TRAIN'] += [list(x) for x in fl]
            epoch_losses['TRAIN'].setdefault(e + 1, []).append([ls[k] for k in LOSS_KEYS])
            gs_prev = r['global_step']
            i_tr += 1
        elif kind == 'VAL_STEP':
            e, s = divmod(i_va, VAL_STEPS)
            va['index'] += r['epoch'] == e + 1 and r['val_step_in_epoch'] == s
            va['kind'] += r['record_type'] == 'epoch'
            va['global_step'] += r['global_step'] == gs_prev == (e + 1) * GS_PER_EPOCH
            exp = expected_lr(gs_prev)
            va['lr'] += abs(r['learning_rate'] - exp) / exp <= TOL_LR
            ls = r['val_losses']
            va['losses'] += sorted(ls) == list(LOSS_KEYS) and all(finite(ls[k]) for k in LOSS_KEYS)
            pr_ = [int(x) for x in r['pair_rows']]
            va['pairs'] += (len(pr_) == BATCH and r['pair_ids'] == [pack['rows']['VAL'][x][0] for x in pr_])
            fl = r['flips_live_spoof']
            va['flips'] += len(fl) == BATCH and all(x in ([0, 0], [0, 1], [1, 0], [1, 1]) for x in fl)
            want = 'diagnostics/figures/Epoch-%d-Val-%d.png' % (e + 1, s + 1) if s % 50 == 0 else None
            va['figure'] += r['figure'] == want
            va['nulls'] += (r['train_losses'] is None and r['train_metrics'] is None and r['val_metrics'] is None and
                            'brightness_delta_live_spoof' not in r and 'applications' not in r)
            va['role'] += r['val_role'] == 'DIAGNOSTICS_ONLY_NEVER_SELECTS'
            rows['VAL'] += pr_
            flips['VAL'] += [list(x) for x in fl]
            epoch_losses['VAL'].setdefault(e + 1, []).append([ls[k] for k in LOSS_KEYS])
            i_va += 1
    # independent flip prediction: the parser draws live then spoof from numpy RandomState(split seed)
    flip_match = {}
    for split, s in (('TRAIN', seed), ('VAL', val_seed(seed))):
        n = len(flips[split])
        pred = (np.random.RandomState(s).rand(2 * n) > 0.5).astype(int).reshape(-1, 2).tolist()
        flip_match[split] = {'seed': s, 'pairs': n, 'match': sum(a == b for a, b in zip(flips[split], pred)),
                             'live_flip_rate': sum(x[0] for x in flips[split]) / max(n, 1),
                             'spoof_flip_rate': sum(x[1] for x in flips[split]) / max(n, 1)}
    loss_epochs = {sp: {str(e): [sum(v[k] for v in vals) / len(vals) for k in range(3)]
                        for e, vals in sorted(d.items())} for sp, d in epoch_losses.items()}
    secs = sorted(x for x in step_seconds if isinstance(x, float))
    c = {'lines': lines, 'bad_lines': bad, 'trailing_newline': trailing_newline, 'rle': rle,
         'train_records': i_tr, 'val_records': i_va, 'train_checks': tr, 'val_checks': va,
         'max_relative_lr_error': max_rel_lr, 'final_global_step': gs_prev,
         'utc_nondecreasing': all(a <= b for a, b in zip(utcs, utcs[1:])), 'first_utc': utcs[0] if utcs else None,
         'last_utc': utcs[-1] if utcs else None, 'figures_referenced': sorted(figures),
         'streams': {sp: dict(stream_digests(rows[sp], sp), passes=pass_structure(rows[sp], ROWS[sp]))
                     for sp in rows},
         'flips': flip_match, 'brightness_delta_abs_max': delta_abs_max,
         'loss_epoch_means': {'keys': list(LOSS_KEYS), 'TRAIN': loss_epochs['TRAIN'], 'VAL': loss_epochs['VAL']},
         'step_seconds': {'median': secs[len(secs) // 2], 'max': secs[-1], 'min': secs[0]} if secs else None,
         'gpu_memory_max_bytes': max(x for x in gpu_mem if x is not None) if any(gpu_mem) else None}
    consumed = {sp: sorted({x for r in set(rows[sp]) for x in pack['rows'][sp][r][1:3]}) for sp in rows}
    c['distinct_pair_rows'] = {sp: len(set(rows[sp])) for sp in rows}
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


def host_state(seed):
    p = paths(seed)

    def sh(*a):
        return subprocess.run(list(a), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                              universal_newlines=True).stdout.strip()
    pane = sh('tmux', 'list-panes', '-t', p['tmux'], '-F', '#{pane_dead} #{pane_dead_status}').split()
    procs = [x for x in sh('ps', '-eo', 'pid,args').splitlines() if 'python' in x and 'tools/run_e03.py' in x]
    pane_log = Path(p['pane_log']).read_bytes().decode('utf-8', errors='replace').splitlines()
    return {'tmux_pane_dead': pane[0] == '1' if pane else None,
            'tmux_pane_dead_status': int(pane[1]) if len(pane) > 1 and pane[0] == '1' else None,
            'runner_processes': procs, 'exit_file': Path(p['exit']).read_text().strip(),
            'exit_file_mtime_utc': utc(Path(p['exit']).stat().st_mtime),
            'launch_time': Path(p['launch_time']).read_text().strip(),
            'end_time': Path(p['end_time']).read_text().strip(),
            'command_file': Path(p['command']).read_text(), 'command_file_sha256': sha256_file(p['command']),
            'preflight': json.loads(Path(p['preflight']).read_text()),
            'pane_log_lines': len(pane_log), 'pane_log_last': [x.strip() for x in pane_log if x.strip()][-1:],
            'pane_log_fail_hits': {k: sum(k in x for x in pane_log) for k in PANE_FAIL_PATTERNS},
            'other_e03_tmux': sorted(x for x in sh('tmux', 'ls', '-F', '#{session_name}').splitlines()
                                     if x.startswith('e03_stdn_seed')),
            'gpu_query': sh('nvidia-smi', '--query-gpu=name,driver_version,uuid', '--format=csv,noheader')}


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
    pack = load_pack(Path(RUNTIME) / 'data' / 'processed' / 'e03_feed_pack_v1')
    split = {r['sample_id']: r for r in pq.read_table('manifests/split_v1.parquet', columns=[
        'sample_id', 'dataset', 'label_binary', 'split']).to_pylist()}
    acc = {r['sample_id']: r['face_png_sha256'] for r in pq.read_table(
        'manifests/m2_sample_accounting.parquet', columns=['sample_id', 'face_png_sha256']).to_pylist()}
    metrics, events, consumed = scan_metrics(root / 'metrics.jsonl', seed, pack)
    faces = {}
    for sp, ids in consumed.items():
        recs = [pack['samples'][s] for s in ids]
        faces[sp] = {'samples': len(ids),
                     'split_v1_matches': sum(split.get(s, {}).get('split') == sp for s in ids),
                     'pack_split_matches': sum(r['split'] == sp for r in recs),
                     'label_matches_split_v1': sum(split.get(r['sample_id'], {}).get('label_binary') ==
                                                   r['label_binary'] for r in recs),
                     'dataset_matches_split_v1': sum(split.get(r['sample_id'], {}).get('dataset') == r['dataset']
                                                     for r in recs),
                     'face_png_sha256_match_m2_accounting': sum(acc.get(r['sample_id']) == r['face_png_sha256']
                                                                for r in recs),
                     'pack_samples_of_split': sum(r['split'] == sp for r in pack['samples'].values())}
    all_consumed = set(consumed['TRAIN']) | set(consumed['VAL'])
    atime = collect_atime(seed, split, all_consumed)
    ck_events = [{k: e.get(k) for k in ('epoch', 'prefix', 'files', 'variables', 'round_trip_equal', 'all_finite',
                                        'global_step', 'state_unchanged_by_save', 'terminal_semantics_probe', 'labels',
                                        'resume_sidecar', 'checkpoint_type', 'selected_for_final', '_seed')}
                 for e in events.get('e03_checkpoint_event', [])]
    start = (events.get('e03_run_start') or [{}])[0]
    fig_dir = root / 'diagnostics' / 'figures'
    logs = {}
    for name in ('stdout.log', 'stderr.log', 'generation_log.jsonl'):
        text = (root / name).read_bytes().decode('utf-8', errors='replace')
        logs[name] = {'bytes': len(text.encode()), 'lines': len(text.splitlines()),
                      'pattern_hits': {k: sum(k in x for x in text.splitlines()) for k in LOG_PATTERNS}}
    rt = {}
    for rel in ('configs/methods/e03_stdn.yaml', 'frozen_config_snapshot/configs/methods/e03_stdn.yaml',
                'environments/e03_stdn.lock.json', 'manifests/split_v1.parquet', 'manifests/pairs_train_v1.parquet',
                'manifests/val_pairs_v1.parquet', 'manifests/m2_sample_accounting.parquet'):
        rt[rel] = sha256_file(rel)
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
        'resolved': {k: resolved['_resolved'].get(k) for k in ('rng', 'schedule', 'checkpoint_rule', 'resume',
                                                               'runner_mode', 'scientific_run', 'fidelity', 'labels')},
        'metrics': metrics, 'metrics_sha256': sha256_file(root / 'metrics.jsonl'),
        'event_counts': {k: len(v) for k, v in sorted(events.items())},
        'run_start': {k: start.get(k) for k in ('seeding', 'identities', 'structure', 'gpu_driver', 'fidelity', 'pack',
                                                'environment_identity_equals_lock_except_driver', 'session_config',
                                                'python38_pathlib_is_relative_to_backport', '_line', '_seed')},
        'checkpoint_events': ck_events,
        'pack': {'manifest_sha256': pack['manifest_sha256'], 'payloads_match': pack['payloads_match'],
                 'test_rows': pack['test_rows'], 'rows': {s: len(v) for s, v in pack['rows'].items()},
                 'samples': len(pack['samples']),
                 'test_samples': sum(r['split'] == 'TEST' for r in pack['samples'].values())},
        'consumed_faces': faces,
        'figures_on_disk': sorted('diagnostics/figures/' + x.name for x in fig_dir.iterdir()),
        'checkpoints_dir_listing': sorted(x.name for x in (root / 'checkpoints').iterdir()),
        'partial_files': sorted(str(x.relative_to(root)) for x in root.rglob('*.partial')),
        'resume_files': sorted(str(x.relative_to(root)) for x in root.rglob('*resume*')),
        'runner_state_files': sorted(str(x.relative_to(root)) for x in root.rglob('*runner_state*')),
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
    out = [['EVENT:e03_run_start', None, 1]]
    for e in range(1, MAX_EPOCH + 1):
        out += [['TRAIN_STEP', e, STEPS], ['EVENT:e03_checkpoint_event', e, 1], ['VAL_STEP', e, VAL_STEPS]]
    return out


def expected_figures():
    out = set()
    for e in range(1, MAX_EPOCH + 1):
        out |= {'diagnostics/figures/Epoch-%d-%d.png' % (e, s + 1) for s in range(0, STEPS, 200)}
        out |= {'diagnostics/figures/Epoch-%d-Val-%d.png' % (e, s + 1) for s in range(0, VAL_STEPS, 50)}
    return sorted(out)


def checkpoint_kind(e):
    sel = e == MAX_EPOCH
    return ('selected' if sel else 'official', sel,
            EXPECTED['checkpoint_rule'] if sel else 'Official cadence; not final selection')


def manifest_text(c, t):
    seed = c['seed']
    head = ('# E03 STDN seed%d run %s (authority %s): epoch global_step checkpoint_type selected_for_final '
            'size_bytes sha256 path\n' % (seed, RUN_IDS[seed], AUTHORITY[:7]))
    body = ''
    for r in t['checkpoints']:
        kind, sel, _ = checkpoint_kind(r['epoch'])
        for suffix in SUFFIXES:
            f = r['files'][suffix]
            body += '%02d %d %s %s %d %s %s\n' % (r['epoch'], r['global_step'], kind, str(sel).lower(), f['size'],
                                                  f['sha256'], f['path'])
    return head + body + ('# checkpoints/checkpoint (TF Saver state file) is part of the run-root listing; '
                          'runner-state / resume sidecars: none (E03 is fresh-only, NOT_QUALIFIED_FRESH_ONLY)\n')


def build(c, t, repo_root):
    seed = c['seed']
    G = Gates()
    man, summ, m, rs, host = c['run_manifest'], c['run_summary'], c['metrics'], c['run_start'], c['host']
    E, f = EXPECTED, c['files_sha256']
    rid = run_id(seed)
    vs = val_seed(seed)

    # A. identity / authority
    G('A.seed_pair', t['seed'] == c['seed'] == seed and seed in RUN_IDS)
    G('A.run_id_recomputed', rid == RUN_IDS[seed] == man['run_id'] == summ['run_id'], rid)
    G('A.manifest_identity', man['method_id'] == E['method_id'] and man['experiment_seed'] == seed and
      man['git_commit'] == AUTHORITY and man['git_dirty'] is False and man['config_sha256'] == CONFIG_SHA and
      man['fidelity'] == E['fidelity'] and man['resume'] == E['resume'] and man['runner_mode'] == 'SCIENTIFIC' and
      man['scientific_run'] is True and man['snapshot_verified'] is True and
      man['source_commits'] == {'https://github.com/yaojieliu/ECCV20-STDN': E['source_commit']})
    G('A.manifest_identities_exact', man['identities'] == E['identities'] == rs['identities'] ==
      t['identities_recomputed'])
    G('A.command_line_exact', man['command_line'] == command(seed) and '--resume' not in host['command_file'] and
      wrapper_invocation(seed) in host['command_file'] and 'PYTHONHASHSEED=%d ' % seed in host['command_file'] and
      all('%s=%s' % kv in host['command_file'] for kv in PRECISION_ENV.items()) and
      'PATH=/home/student20261/miniconda3/envs/gpat-m6-e03/bin:' in host['command_file'])
    pf = host['preflight']
    G('A.preflight_seed', pf['experiment_seed'] == seed and pf['git_commit'] == AUTHORITY and pf['git_dirty'] is False
      and pf['run_dir_exists'] is False and pf['identities'] == E['identities'] and pf['graph_executed'] is False and
      pf['rng']['TRAIN'] == {'parser_seed': seed, 'shuffle_seed': seed} and
      pf['rng']['VAL'] == {'parser_seed': vs, 'shuffle_seed': vs})
    G('A.gpu_checkout_authority', c['git']['head'] == t['git']['head'] == AUTHORITY and
      c['git']['branch'] == t['git']['branch'] == 'm6-baselines' and
      c['git']['status_porcelain'] == t['git']['status_porcelain'] == '')
    G('A.host', man['host'] == E['host'] and man['environment']['host'] == E['host'])
    G('A.frozen_files', f['configs/methods/e03_stdn.yaml'] == f['frozen_config_snapshot/configs/methods/e03_stdn.yaml']
      == CONFIG_SHA and f['environments/e03_stdn.lock.json'] == E['identities']['environment_lock_sha256'] and
      f['manifests/split_v1.parquet'] == E['identities']['split_manifest_sha256'] and
      f['manifests/pairs_train_v1.parquet'] == E['identities']['pair_manifest_sha256']['TRAIN'] and
      f['manifests/val_pairs_v1.parquet'] == E['identities']['pair_manifest_sha256']['VAL'])
    G('A.source_cache', t['source_checkout'] == {'commit': E['source_commit'], 'tree': E['source_tree'],
                                                 'verification': 'PASS'})
    G('A.single_fresh_start', c['event_counts'].get('e03_run_start') == 1 and rs['_line'] == 1 and
      'seed_%d' % seed in c['parent_listing'] and E['resume'] == c['resolved']['resume'])
    G('A.fidelity_disclosed', rs['fidelity'] == man['fidelity'] == c['resolved']['fidelity'] == FIDELITY and
      man['historical_fidelity_field'] == 'FAITHFUL_OFFICIAL')

    # B. completion
    G('B.process_exited_zero', host['exit_file'] == '0' and host['tmux_pane_dead'] is True and
      host['tmux_pane_dead_status'] == 0 and host['runner_processes'] == [])
    G('B.completion_status', man['completion_status'] == summ['completion_status'] == 'completed' and
      summ['failure_reason'] is None)
    G('B.summary_counts', summ['shared_global_step'] == FINAL_GS and
      summ['optimizer_applications'] == {'G': TRAIN_STEPS, 'D': TRAIN_STEPS // G_D_RATIO} and
      summ['train_steps'] == TRAIN_STEPS and summ['val_steps'] == VAL_TOTAL)
    G('B.records_written', summ['records_written'] == m['lines'] == TRAIN_STEPS + VAL_TOTAL + MAX_EPOCH + 1,
      summ['records_written'])
    G('B.pane_log_clean', all(v == 0 for v in host['pane_log_fail_hits'].values()) and
      host['pane_log_last'] == ['{"status": "completed", "run_dir": "%s"}' % paths(seed)['run_root']],
      host['pane_log_fail_hits'])
    G('B.window', host['launch_time'] <= man['start_utc'] <= man['end_utc'] <= host['end_time'] and
      summ['start_utc'] == man['start_utc'] and summ['end_utc'] == man['end_utc'])
    G('B.logs_clean', all(v['bytes'] == 0 for v in c['logs'].values()), {k: v['bytes'] for k, v in c['logs'].items()})

    # C. lifecycle order and counts
    G('C.metrics_parse', m['bad_lines'] == 0 and m['trailing_newline'] and m['utc_nondecreasing'])
    G('C.lifecycle_order', m['rle'] == expected_rle(), len(m['rle']))
    G('C.train_records_100000', m['train_records'] == TRAIN_STEPS)
    G('C.val_records_25000', m['val_records'] == VAL_TOTAL)
    G('C.event_set', c['event_counts'] == {'e03_checkpoint_event': MAX_EPOCH, 'e03_run_start': 1},
      c['event_counts'])
    for k, v in sorted(m['train_checks'].items()):
        G('C.train_%s' % k, v == TRAIN_STEPS, v)
    for k, v in sorted(m['val_checks'].items()):
        G('C.val_%s' % k, v == VAL_TOTAL, v)

    # D. optimizer / schedule accounting
    G('D.final_global_step_150000', m['final_global_step'] == FINAL_GS == summ['shared_global_step'])
    G('D.final_applications', summ['optimizer_applications'] == {'G': 100000, 'D': 50000})
    G('D.lr_staircase', m['max_relative_lr_error'] <= TOL_LR, m['max_relative_lr_error'])
    G('D.resolved_schedule', c['resolved']['schedule']['final_global_step'] == FINAL_GS and
      c['resolved']['schedule']['train_steps_per_epoch'] == STEPS and
      c['resolved']['schedule']['val_steps_per_epoch'] == VAL_STEPS and
      c['resolved']['schedule']['max_epoch'] == MAX_EPOCH and
      c['resolved']['schedule']['decay_steps_shared_global'] == DECAY_STEPS)

    # E. data feed (owner rulings V / R1 / R2)
    rng = c['resolved']['rng']
    G('E.val_seed_rule', vs == VAL_SEEDS[seed] and rng['TRAIN'] == {'parser_seed': seed, 'shuffle_seed': seed} and
      rng['VAL'] == {'parser_seed': vs, 'shuffle_seed': vs} and rng['tf_graph_seed'] == seed and
      rs['seeding']['TRAIN'] == rng['TRAIN'] and rs['seeding']['VAL'] == rng['VAL'] and
      rs['seeding']['tf_graph_seed'] == seed)
    for sp in ('TRAIN', 'VAL'):
        st, pr_ = m['streams'][sp], t['predicted_streams'][sp]
        G('E.%s_stream_equals_independent_tf_replay' % sp, st['rows'] == pr_['rows'] == CONSUMED[sp] and
          st['sha256'] == pr_['sha256'] and st['epoch_sha256'] == pr_['epoch_sha256'] and
          pr_['op_seed'] == (seed if sp == 'TRAIN' else vs) and pr_['graph_seed'] == seed,
          sum(a == b for a, b in zip(st['epoch_sha256'], pr_['epoch_sha256'])))
        ps = st['passes']
        G('E.%s_reshuffled_full_passes' % sp, ps['full_passes'] == CONSUMED[sp] // ROWS[sp] ==
          ps['permutation_passes'] and ps['tail_distinct'] and ps['identical_consecutive_passes'] == 0 and
          m['distinct_pair_rows'][sp] == ROWS[sp], ps)
        fl = m['flips'][sp]
        G('E.%s_flips_equal_numpy_prediction' % sp, fl['pairs'] == CONSUMED[sp] == fl['match'] and
          fl['seed'] == (seed if sp == 'TRAIN' else vs), fl['match'])
        cf = c['consumed_faces'][sp]
        G('E.%s_faces_in_split' % sp, cf['samples'] == cf['split_v1_matches'] == cf['pack_split_matches'] ==
          cf['label_matches_split_v1'] == cf['dataset_matches_split_v1'] ==
          cf['face_png_sha256_match_m2_accounting'] == cf['pack_samples_of_split'] ==
          E['pack_counts']['samples_by_split'][sp], cf)
    G('E.brightness_train_only', m['brightness_delta_abs_max'] <= BRIGHTNESS_MAX, m['brightness_delta_abs_max'])
    G('E.feed_pack', c['pack']['manifest_sha256'] == t['pack']['manifest_sha256'] ==
      E['identities']['pack_manifest_sha256'] and c['pack']['payloads_match'] and
      c['pack']['test_rows'] == t['pack']['test_rows'] == 0 and c['pack']['test_samples'] == 0 and
      t['pack']['counts'] == rs['pack']['counts'] == E['pack_counts'] and
      c['pack']['rows'] == t['pack']['rows'] == ROWS)
    reads = summ['face_reads']
    G('E.face_reads_lifecycle', all(0 <= reads[sp] - 2 * CONSUMED[sp] <= FEED_LOOKAHEAD_MAX for sp in ROWS) and
      sorted(reads) == ['TRAIN', 'VAL'], reads)

    # F. graph structure / numerics
    st = rs['structure']
    G('F.official_graph_structure', st['global_variables'] == N_VARIABLES and st['trainable_variables'] == 170 and
      st['adam_ops'] == 170 and st['bn_moving'] == 60 and st['weight_ema'] == 170 and st['loss_ema'] == 2 and
      st['trainable_equals_m6d2b_oracle'] is True and st['val_reachable_state_mutating_ops'] == [] and
      st['val_fused_bn_is_training_true'] == 0 and st['val_reads_train_iterator'] is False and
      st['val_step_returns_no_train_op'] is True and st['variables_created_by_feed'] == 0 and
      st['variables_created_by_val_graph'] == 0)
    G('F.losses_finite', m['train_checks']['losses'] == TRAIN_STEPS and m['val_checks']['losses'] == VAL_TOTAL)

    # G. checkpoints
    tk = {r['epoch']: r for r in t['checkpoints']}
    ev = {e['epoch']: e for e in c['checkpoint_events']}
    idx = {}
    for r in c['checkpoint_index']:
        idx.setdefault(r['epoch'], []).append(r)
    epochs = list(range(1, MAX_EPOCH + 1))
    G('G.fifty_checkpoints', sorted(tk) == sorted(ev) == sorted(idx) == epochs and
      len(c['checkpoint_index']) == MAX_EPOCH * len(SUFFIXES) and
      c['checkpoints_dir_listing'] == sorted(['checkpoint'] + ['ckpt-%d%s' % (e, s) for e in epochs
                                                               for s in SUFFIXES]))
    bad = {'files': [], 'events': [], 'index': [], 'reader': [], 'trained': []}
    for e in epochs:
        r, x, ix = tk.get(e, {}), ev.get(e, {}), idx.get(e, [])
        kind, sel, reason = checkpoint_kind(e)
        files = r.get('files', {})
        if not all(files.get(s, {}).get('exists') for s in SUFFIXES):
            bad['files'].append(e)
        ev_files = {fr['path']: fr for fr in x.get('files', [])}
        ix_files = {fr['path']: fr for fr in ix}
        for s in SUFFIXES:
            fr = files.get(s, {})
            p = fr.get('path')
            want = {'path': p, 'epoch': e, 'global_step': e * GS_PER_EPOCH, 'file_size_bytes': fr.get('size'),
                    'sha256': fr.get('sha256'), 'checkpoint_type': kind, 'selected_for_final': sel,
                    'selection_reason': reason}
            if ev_files.get(p) != want:
                bad['events'].append([e, s])
            if ix_files.get(p) != want:
                bad['index'].append([e, s])
        if not (x.get('global_step') == e * GS_PER_EPOCH and x.get('variables') == x.get('round_trip_equal') ==
                N_VARIABLES and x.get('all_finite') is True and x.get('state_unchanged_by_save') is True and
                x.get('terminal_semantics_probe') is False and x.get('labels') == [] and
                x.get('resume_sidecar') is None and x.get('checkpoint_type') == kind and
                x.get('selected_for_final') is sel and x.get('prefix') == 'checkpoints/ckpt-%d' % e and
                x.get('_seed') == seed):
            bad['events'].append([e, 'payload'])
        if not (r.get('variables') == N_VARIABLES and r.get('nonfinite_variables') == 0 and
                r.get('global_step') == e * GS_PER_EPOCH and r.get('variable_set_equals_ckpt1') is True and
                r.get('trainable_oracle_present') == t['trainable_oracle_variables'] == 170):
            bad['reader'].append(e)
        if e > 1 and not (r.get('trainable_differs_from_previous') or 0) > 0:
            bad['trained'].append(e)
    G('G.files_exist', bad['files'] == [], bad['files'])
    G('G.events_match_files', bad['events'] == [], bad['events'][:10])
    G('G.index_matches_files', bad['index'] == [], bad['index'][:10])
    G('G.reader_finite_global_step', bad['reader'] == [], bad['reader'])
    G('G.training_changed_weights_every_epoch', bad['trained'] == [], bad['trained'])
    lines = {x['path']: x for x in ({'path': ln.split(' ')[0], 'size': int(ln.split(' ')[1]),
                                      'sha256': ln.split(' ')[2]} for ln in c['run_root_sha_listing'])}
    G('G.listing_matches_reader_phase', all(lines.get(fr['path'], {}).get('sha256') == fr['sha256'] and
                                            lines.get(fr['path'], {}).get('size') == fr['size']
                                            for r in t['checkpoints'] for fr in r['files'].values()))
    state = t['checkpoint_state_file']
    ck_dir = paths(seed)['run_root'] + '/checkpoints/'
    G('G.saver_state_latest_ckpt50', state == ['model_checkpoint_path: "%sckpt-50"' % ck_dir] +
      ['all_model_checkpoint_paths: "%sckpt-%d"' % (ck_dir, e) for e in epochs])
    fin = tk.get(MAX_EPOCH, {}).get('files', {})
    G('G.final_ckpt50_selected', summ['checkpoint_selection_rule'] == E['checkpoint_rule'] and
      summ['final_or_selected_checkpoint_path'] == ck_dir + 'ckpt-50' and
      summ['final_or_selected_checkpoint_sha256'] == fin.get(SUFFIXES[0], {}).get('sha256') and
      [x['path'] for x in summ['final_checkpoint_files']] == ['checkpoints/ckpt-50' + s for s in SUFFIXES] and
      all(x['selected_for_final'] and x['global_step'] == FINAL_GS and
          x['sha256'] == fin.get(s, {}).get('sha256') for x, s in zip(summ['final_checkpoint_files'], SUFFIXES)) and
      c['checkpoint_index_meta']['selected_final'] == 'checkpoints/ckpt-50' and
      c['checkpoint_index_meta']['checkpoint_rule'] == E['checkpoint_rule'] and
      sum(r['selected_for_final'] for r in c['checkpoint_index']) == len(SUFFIXES))
    G('G.no_partial_no_sidecar', c['partial_files'] == [] and c['runner_state_files'] == [] and
      c['resume_files'] == [] and c['checkpoint_index_meta']['resume_sidecars'] is None)
    G('G.figures', c['figures_on_disk'] == expected_figures() == m['figures_referenced'])

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
    G('H.changes_only_consumed_faces', all(v['changed_not_consumed'] == 0 for v in by.values()))
    G('H.manifests_unmodified', all(x['mtime_unchanged'] for x in a['manifests']) and len(a['manifests']) > 0)
    G('H.test_flags', man['test_split_accessed'] is False and summ['test_split_accessed'] is False and
      man['test_code_path_present'] is False and man['val_role'] == 'DIAGNOSTICS_ONLY_NEVER_SELECTS')

    # I. environment / driver ruling
    er = t['environment_recheck']
    G('I.driver_ruling_run_start', rs['gpu_driver'] == {
        'driver_policy': DRIVER_RULING['driver_policy'], 'gpu_model': E['gpu_model'], 'gpu_uuid': E['gpu_uuid'],
        'lock_driver_version': DRIVER_RULING['lock_driver_version'],
        'observed_driver_version': DRIVER_RULING['observed_driver_version']})
    G('I.environment_equals_lock_except_driver', rs['environment_identity_equals_lock_except_driver'] is True and
      er['differences_other_than_driver'] == [] and er['driver'] == rs['gpu_driver'])
    G('I.live_gpu', host['gpu_query'] == '%s, %s, %s' % (E['gpu_model'], DRIVER_RULING['observed_driver_version'],
                                                          E['gpu_uuid']) and er['live_gpu_uuid'] == E['gpu_uuid'])
    G('I.runtime', er['live_tensorflow'] == man['tensorflow_version'] == '1.15.5' and
      er['live_cuda_cudnn']['cudaRuntimeGetVersion'] == 11080 and er['live_cuda_cudnn']['cudnnGetVersion'] == 8907 and
      man['python_executable'] == t['collector']['executable'] == E['python_executable'] and
      t['collector']['launch_environment_exact'] is True)
    G('I.python38_backport_disclosed', rs['python38_pathlib_is_relative_to_backport'] is True)

    # J. run-root integrity
    G('J.run_root_unchanged', c['run_root_unchanged_whole_audit'] and t['run_root_tree_unchanged'] and
      c['run_root_sha_listing_digest'] == c['run_root_sha_listing_digest_before'] and
      c['run_root_tree_sha256'] == t['run_root_tree_sha256'])
    G('J.run_root_files', c['run_root_file_count'] == len(c['run_root_sha_listing']) == RUN_ROOT_FILES,
      c['run_root_file_count'])
    G('J.single_e03_session', host['other_e03_tmux'] == sorted(set(host['other_e03_tmux'])) and
      paths(seed)['tmux'] in host['other_e03_tmux'])

    failed = G.failed
    status = 'E03_SEED%d_COMPLETED_CLEANLY' % seed if not failed else 'E03_SEED%d_NEEDS_REVIEW' % seed
    text = manifest_text(c, t)
    by_split = {}
    for k, v in by.items():
        s = by_split.setdefault(k.split('|')[1], {'files': 0, 'sensitive_whole_window': 0, 'atime_changed': 0,
                                                  'changed_consumed': 0, 'changed_not_consumed': 0, 'consumed': 0})
        for key in s:
            s[key] += v[key]
    lm = m['loss_epoch_means']
    doc = {
        'milestone': 'E03-SEED%d-COMPLETION' % seed, 'classification': CLASSIFICATION, 'status': status,
        'failed_gates': failed, 'gates': G.results, 'gates_passed': len(G.results) - len(failed),
        'gates_total': len(G.results),
        'method': {'id': E['method_id'], 'name': E['method_name'], 'fidelity': E['fidelity'], 'track': E['track']},
        'seed': seed, 'run_id': rid, 'run_id_rule': 'sha256("E03|%d|%s|%s")[:16]' % (seed, CONFIG_SHA, AUTHORITY),
        'scientific_authority': AUTHORITY, 'config_sha256': CONFIG_SHA, 'resume': E['resume'],
        'run_root': paths(seed)['run_root'], 'start_utc': summ['start_utc'], 'end_utc': summ['end_utc'],
        'training_duration_seconds': summ['training_duration_seconds'],
        'counts': {'epochs': len(tk), 'train_steps': m['train_records'], 'val_steps': m['val_records'],
                   'val_blocks': sum(1 for x in m['rle'] if x[0] == 'VAL_STEP'),
                   'g_applications': summ['optimizer_applications']['G'],
                   'd_applications': summ['optimizer_applications']['D'], 'final_global_step': m['final_global_step'],
                   'checkpoints': len(tk), 'checkpoint_files': len(c['checkpoint_index'])},
        'rng': {'TRAIN': {'parser_seed': seed, 'shuffle_seed': seed}, 'VAL': {'parser_seed': vs, 'shuffle_seed': vs},
                'tf_graph_seed': seed, 'val_seed_rule': rng['val_seed_rule'],
                'train_stream_sha256': m['streams']['TRAIN']['sha256'],
                'val_stream_sha256': m['streams']['VAL']['sha256'], 'flips': m['flips']},
        'loss_diagnostics': {'keys': lm['keys'], 'train_epoch_means': {e: lm['TRAIN'][e] for e in ('1', '50')},
                             'val_epoch_means': {e: lm['VAL'][e] for e in ('1', '50')},
                             'note': 'VAL losses are diagnostics only; no selection, threshold or ranking uses them'},
        'checkpoints': {'files': [{'epoch': r['epoch'], 'global_step': r['global_step'],
                                   'sha256': {s: r['files'][s]['sha256'] for s in SUFFIXES},
                                   'size': {s: r['files'][s]['size'] for s in SUFFIXES}} for r in t['checkpoints']],
                        'final_selected': {'prefix': 'checkpoints/ckpt-50', 'rule': E['checkpoint_rule'],
                                           'sha256': {s: fin.get(s, {}).get('sha256') for s in SUFFIXES}},
                        'manifest_file': evidence(seed)['manifest'], 'manifest_sha256': sha256_bytes(text.encode()),
                        'resume_sidecars': 0},
        'firewall': {'test_access': 0 if not failed else None, 'by_split': by_split,
                     'test_files': by_split.get('TEST', {}).get('files'),
                     'test_changed': by_split.get('TEST', {}).get('atime_changed'),
                     'scope': 'TRAIN (pairs_train_v1) for optimisation; VAL (val_pairs_v1) diagnostics only; no TEST',
                     'atime': a, 'relatime_note': ('faces read less than 24 h before the pre snapshot (E03 '
                                                   'qualification) are not atime-sensitive; TRAIN/VAL use is proven '
                                                   'by the stream/manifest/content evidence, TEST by atime')},
        'driver_ruling': DRIVER_RULING,
        'environment': {'python_executable': E['python_executable'], 'tensorflow': er['live_tensorflow'],
                        'cuda_cudnn_native': er['live_cuda_cudnn'], 'packages': er['packages'],
                        'python38_pathlib_backport': rs['python38_pathlib_is_relative_to_backport']},
        'run_root_integrity': {'file_count': c['run_root_file_count'],
                               'sha_listing_digest': c['run_root_sha_listing_digest'],
                               'sha_listing_convention': ('LC_ALL=C sorted "<path> <size> <sha256>" lines, sha256 of '
                                                          'the newline-terminated listing'),
                               'metrics_jsonl_sha256': c['metrics_sha256'],
                               'unchanged_whole_audit': c['run_root_unchanged_whole_audit']},
        'host_markers': {'exit': host['exit_file'], 'launch_time': host['launch_time'], 'end_time': host['end_time'],
                         'command_file_sha256': host['command_file_sha256'], 'tmux': paths(seed)['tmux']},
        'confirmations': {'resume_used': False, 'val_checkpoint_selection': False, 'test_selection': False,
                          'best_epoch_logic': False, 'best_seed_logic': False, 'bank_generation': False,
                          'downstream_evaluation': False, 'run_root_mutated': False, 'downstream_metric': None},
        'collected': c, 'collected_tf': t,
    }
    return doc, text


def report(doc):
    d, c, t = doc, doc['collected'], doc['collected_tf']
    seed, n = d['seed'], d['counts']
    lm = d['loss_diagnostics']
    fin = d['checkpoints']['final_selected']['sha256']
    rows = ''.join('| %s | %d | %d | %d | %d | %d | %d |\n' % (k, v['files'], v['sensitive_whole_window'],
                                                              v['atime_changed'], v['changed_consumed'],
                                                              v['changed_not_consumed'], v['consumed'])
                   for k, v in sorted(d['firewall']['by_split'].items()))
    fmt = lambda xs: ' / '.join('%.4g' % x for x in xs)  # noqa: E731
    return '\n'.join([
        '# E03 seed%d: scientific run %s' % (seed, 'completed cleanly' if not d['failed_gates'] else 'NEEDS REVIEW'),
        '',
        '**Status:** `%s`, %d/%d gates passing. The machine-readable record is `E03_SEED%d_COMPLETION.json`; it '
        'embeds both read-only host collections, so `tools/e03_seed_completion_audit.py build` can re-derive it.'
        % (d['status'], d['gates_passed'], d['gates_total'], seed),
        '',
        'This commit is reporting evidence only. The E03 scientific training authority stays **`%s`**.' % AUTHORITY,
        '',
        '| | |', '|---|---|',
        '| method | E03 **STDN**, track %s |' % d['method']['track'],
        '| execution fidelity | `%s` (not a fully native reproduction) |' % d['method']['fidelity'],
        '| run | seed = %d, run_id = **`%s`** = sha256("E03\\|%d\\|08dde851…\\|d547fc9…")[:16] (recomputed) |'
        % (seed, d['run_id'], seed),
        '| resume | `%s`: one run_start, command `tools/run_e03.py --seed %d --execution-config '
        'configs/execution/m5_gpu_3090.yaml` |' % (d['resume'], seed),
        '| window | %s → %s, %.0f s of training, median %.3f s per step |'
        % (d['start_utc'], d['end_utc'], d['training_duration_seconds'], c['metrics']['step_seconds']['median']),
        '| exit | `~/workdir/e03_s%d.exit` = %s; tmux `%s` pane dead with status 0; no runner process; '
        'stdout/stderr/generation_log empty |' % (seed, d['host_markers']['exit'], d['host_markers']['tmux']),
        '',
        '## Completion accounting',
        '',
        '- **Epochs = %d**, **TRAIN steps = %d**, **VAL steps = %d** in %d blocks of 500.' % (
            n['epochs'], n['train_steps'], n['val_steps'], n['val_blocks']),
        '- **G applications = %d**, **D applications = %d**, **final shared global_step = %d**. Every TRAIN record '
        'matches the official schedule: even steps run g_op + d_op (+2), odd steps g_op (+1).'
        % (n['g_applications'], n['d_applications'], n['final_global_step']),
        '- **Lifecycle order:** metrics.jsonl is exactly run_start, then for each epoch 2000 TRAIN, the ckpt-e event, '
        '500 VAL. VAL never changes global_step.',
        '- **Learning rate:** 6e-5 × 0.9^⌊global_step/20000⌋ on the shared global_step; the largest relative '
        'float32 read-back error is %.2g.' % c['metrics']['max_relative_lr_error'],
        '- **Losses:** all g/d/a losses are finite in every TRAIN and VAL record.',
        '',
        'Loss means (%s), diagnostics only:' % ' / '.join(lm['keys']),
        '',
        '| | epoch 1 | epoch 50 |', '|---|---|---|',
        '| TRAIN | %s | %s |' % (fmt(lm['train_epoch_means']['1']), fmt(lm['train_epoch_means']['50'])),
        '| VAL (diagnostics only) | %s | %s |' % (fmt(lm['val_epoch_means']['1']), fmt(lm['val_epoch_means']['50'])),
        '',
        '## Data feed (owner rulings V, R1, R2)',
        '',
        '- **Seeds:** TRAIN parser/shuffle = %d; VAL parser/shuffle = %d (`%s`); graph seed = %d.'
        % (seed, d['rng']['VAL']['parser_seed'], d['rng']['val_seed_rule'], seed),
        '- **Pair-row streams:** all %d TRAIN and %d VAL rows equal a separate CPU-only TF replay of the shuffle ops; '
        'every full pass is a fresh permutation (reshuffle_each_iteration=True).'
        % (c['metrics']['streams']['TRAIN']['rows'], c['metrics']['streams']['VAL']['rows']),
        '- **Flips:** %d/%d TRAIN and %d/%d VAL pairs equal an independent numpy RandomState prediction, so VAL never '
        'perturbed the TRAIN stream.' % (d['rng']['flips']['TRAIN']['match'], d['rng']['flips']['TRAIN']['pairs'],
                                         d['rng']['flips']['VAL']['match'], d['rng']['flips']['VAL']['pairs']),
        '- **Faces:** %d TRAIN and %d VAL samples consumed; split_v1, labels, datasets and M2 face sha256 match for all.'
        % (c['consumed_faces']['TRAIN']['samples'], c['consumed_faces']['VAL']['samples']),
        '- **Feed pack:** manifest `%s`, payloads verified, 0 TEST rows.' % c['pack']['manifest_sha256'],
        '',
        '## Checkpoints (%s)' % d['checkpoints']['final_selected']['rule'],
        '',
        '- ckpt-1 … ckpt-50 (150 files) exist and equal the index and the checkpoint events; each event lies between '
        'its epoch\'s TRAIN and VAL blocks.',
        '- Every checkpoint holds 747 finite variables with global_step = 3000·epoch; the 170 trainable variables '
        'match the M6D2b oracle and change between consecutive checkpoints.',
        '- The TF Saver state names ckpt-50 as latest. The final checkpoint is **ckpt-50**:',
        '  - `.data-00000-of-00001` **`%s`**' % fin['.data-00000-of-00001'],
        '  - `.index` `%s`' % fin['.index'],
        '  - `.meta` `%s`' % fin['.meta'],
        '- No partial, resume or runner-state file. Full list: `E03_SEED%d_CHECKPOINT_MANIFEST.txt`.' % seed,
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
        '**TEST = 0**: every TEST face was atime-sensitive and none changed; no consumed sample is TEST; '
        '`test_split_accessed` is false. VAL was read for diagnostics only and never selects.',
        '',
        '## Environment (owner driver ruling)',
        '',
        '- Driver: lock 595.84, observed **580.178.04**, `%s`; GPU model and UUID exact; no other lock difference '
        '(run_start and live re-check).' % d['driver_ruling']['driver_policy'],
        '- TensorFlow %s, CUDA runtime %d, cuDNN %d, %d packages equal to the lock; Python 3.8 pathlib backport '
        'disclosed.' % (d['environment']['tensorflow'], d['environment']['cuda_cudnn_native']['cudaRuntimeGetVersion'],
                        d['environment']['cuda_cudnn_native']['cudnnGetVersion'], d['environment']['packages']),
        '',
        '## Run-root integrity',
        '',
        '- %d files; LC_ALL=C `<path> <size> <sha256>` listing digest **`%s`**, identical before and after the audit.'
        % (d['run_root_integrity']['file_count'], d['run_root_integrity']['sha_listing_digest']),
        '- metrics.jsonl `%s`' % d['run_root_integrity']['metrics_jsonl_sha256'],
        '',
        '## Boundary',
        '',
        'No downstream metric, bank, best-epoch or best-seed logic. ckpt-50 is the final checkpoint by the frozen '
        'rule, not by VAL.',
        ''])


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd')
    for name in ('collect', 'collect-tf'):
        a = sub.add_parser(name)
        a.add_argument('--seed', type=int, required=True, choices=sorted(VAL_SEEDS))
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
