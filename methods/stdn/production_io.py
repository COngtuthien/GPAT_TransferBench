"""E03 STDN production I/O: frozen identities, owner-ruled data/RNG bindings, schedule accounting (framework-free).

Imports neither TensorFlow nor pyarrow, so tools/run_e03.py --preflight-only and the laptop tests (Python 3.11) use
exactly the same constants and checks as the TF1 controller (Python 3.8). Authorities:
  * configs/methods/e03_stdn.yaml (frozen, 08dde851...) and its snapshot;
  * the M6D2b runtime compatibility addendum + environments/e03_stdn.lock.json (runtime layer);
  * the owner resolution docs/spec/amendments/GPAT_TransferBench_v1_0_E03_Production_Data_RNG_Owner_Resolution_
    Addendum.md + configs/amendments/e03_production_data_rng_resolution.yaml (rulings V, R1, R2, R3; D1-D3);
  * the pinned source yaojieliu/ECCV20-STDN@c79f1f8c (train.py / model/*.py unchanged, never patched).
"""
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
METHOD_ID, DISPLAY_NAME, TRACK = 'E03', 'STDN', 'A_FAIR_COMMON_IDENTITY_FREE'
FIDELITY = 'FAITHFUL_OFFICIAL_CORE_WITH_TRACK_A_CONTROLLED_DATA_ADAPTATION_AND_RUNTIME_COMPATIBILITY'
FIDELITY_LAYERS = {
    'official_unchanged': 'STDN graph, losses, optimizer graph, G/D schedule, shared global_step, learning-rate rule, '
                          'BN, weight/loss EMA, hard-sample mechanism, checkpoint cadence, ckpt-50 final rule, '
                          '50 x 2000 TRAIN steps, 500 VAL steps per epoch',
    'controlled_benchmark_adaptation': 'Track-A frozen frame-pair relations (pairs_train_v1 / val_pairs_v1), M2 '
                                       'canonical faces + landmarks_px256 via the E03 feed pack, independent '
                                       'deterministic TRAIN/VAL data RNG, serialized deterministic feed',
    'runtime_compatibility': 'qualified NVIDIA TF 1.15.5+nv22.12 runtime on the RTX 3090 (M6D2b)'}
SCIENTIFIC, QUALIFICATION = 'SCIENTIFIC', 'QUALIFICATION'
SCIENTIFIC_SEEDS = (42, 1337, 2026)
QUALIFICATION_SEED = 60903
QUALIFICATION_PARTS = ('qualification', 'e03_production', 'E03')
RESUME = 'NOT_QUALIFIED_FRESH_ONLY'

# ---------------------------------------------------------------- frozen identities
CONFIG_PATH = 'configs/methods/e03_stdn.yaml'
CONFIG_SHA = '08dde851bc9e8ac688c794a6a6999204a7e5b75fb244fcd97e46b22dcb302fe0'
SNAPSHOT_PATH = 'frozen_config_snapshot/' + CONFIG_PATH
SPEC_SHA = 'f7d2371682ad9504f59f0c6984e4b1a741f855f1b1f60838f9d60b160281489e'
RUNTIME_ADDENDUM = ('docs/spec/amendments/GPAT_TransferBench_v1_0_E03_Runtime_Compatibility_Addendum.md',
                    'f5133f81ed1a0567a7bb1426fc23e05d43e9d689d6478b0f37ec2377832d9d0e')
RULING_ADDENDUM = ('docs/spec/amendments/GPAT_TransferBench_v1_0_E03_Production_Data_RNG_Owner_Resolution_Addendum.md',
                   'ad3bf3b13815ca013fbd72237cf4bebe40473729f1e00507fe0ac6d209446f8a')
RULING_YAML = ('configs/amendments/e03_production_data_rng_resolution.yaml',
               '4bf6f87967a837be7c26efb61a1c3cc13285c16e52c8ca5c7de424b696b85d3b')
LOCK = ('environments/e03_stdn.lock.json', 'b2c6c2eadca1b93a0bdf6798990821cb889e199643a5bdb5368ed0b12fdc8c62')
COMPAT_MANIFEST = ('environments/e03_stdn.compatibility_patch_manifest.json',
                   '4cf2d14ed302f15418b887dfef5e45312328bc1e9e4273c2ea74ef607e447f84')
LOGGING_CONTRACT = 'configs/run_logging_v1.yaml'
SOURCE_PIN = 'c79f1f8c615d2b8471b3df29da881bb18dd54c90'
SOURCE_ROOT = 'third_party/source_cache/stdn'
SOURCE_EVIDENCE = 'outputs/audit/M6D2A_E03_ENVIRONMENT_RESOLUTION.json'
PAIR_MANIFESTS = {'TRAIN': ('manifests/pairs_train_v1.parquet',
                            'a5e4fdaef236f15730c7e3885b537e08e995faffbe654167fc940f44bbc75243', 8838),
                  'VAL': ('manifests/val_pairs_v1.parquet',
                          '84d124919a52cd8d84a89766f464a4dcde1aeaa7791218f6356813a40904f872', 1905)}
SPLIT_MANIFEST = ('manifests/split_v1.parquet', 'fb9aeb369a124fc96ba855ef2ce269236c4a743fe960e73ab739412c9cb5092d')
PREPROCESS = ('configs/frozen/preprocess_v1.yaml', '265b75755ecede0d9c01f6d1e22f4d232767ec1e15faf8d7d55de11a68996e9e')
# E03 feed pack v1 (owner decision D1), built by tools/e03_feed_pack.py; non-circular pin of its manifest.
PACK_DIRNAME = 'e03_feed_pack_v1'
PACK_MANIFEST_SHA256 = 'c8c9850edf783a7fce392c934e721feb29a0b88a9223612a39bd33ee46813833'
PYTHON_EXECUTABLE = '/home/student20261/miniconda3/envs/gpat-m6-e03/bin/python'
ENVIRONMENT_NAME = 'gpat-m6-e03'
# The E03 lock records the GPU as "name, capability, driver, memory" without a UUID; the UUID of the same physical
# RTX 3090 is pinned from the committed E05/E06 host evidence.
GPU_MODEL, GPU_UUID = 'NVIDIA GeForce RTX 3090', 'GPU-b722cd9d-c9fa-223c-915e-cc09d9b862b7'
LOCK_DRIVER, OBSERVED_DRIVER = '595.84', '580.178.04'
DRIVER_POLICY = 'OWNER_APPROVED_DRIVER_VERSION_ONLY_RUNTIME_COMPATIBILITY'
PRODUCTION_FILES = ('methods/stdn/production_io.py', 'methods/stdn/production_runner.py', 'methods/stdn/feed_pack.py',
                    'tools/run_e03.py')

# ---------------------------------------------------------------- official schedule (pinned source, unchanged)
IMAGE_SIZE, MAP_SIZE, BATCH_SIZE, G_D_RATIO = 256, 32, 2, 2
MAX_EPOCH, STEPS_PER_EPOCH, VAL_STEPS = 50, 2000, 500
LOG_FR_TRAIN, LOG_FR_VAL = int(STEPS_PER_EPOCH / 10), int(VAL_STEPS / 10)      # model/config.py LOG_FR_TRAIN/TEST
LEARNING_RATE, DECAY_RATE, DECAY_STEPS = 6e-5, 0.9, 20000                      # 10.0 epochs x 2000 (shared step)
BETA1, BETA2, EPSILON, WEIGHT_EMA, LOSS_EMA = 0.9, 0.999, 1e-8, 0.9999, 0.9
GLOBAL_STEP_PER_EPOCH = STEPS_PER_EPOCH + STEPS_PER_EPOCH // G_D_RATIO           # 3000
FINAL_GLOBAL_STEP = MAX_EPOCH * GLOBAL_STEP_PER_EPOCH                           # 150000
CHECKPOINT_RULE = 'OFFICIAL_LATEST_FINAL_CKPT_50'
CHECKPOINT_SUFFIXES = ('.data-00000-of-00001', '.index', '.meta')
TERMINAL_PROBE_LABEL = 'QUALIFICATION_TERMINAL_SEMANTICS_PROBE_NOT_50_EPOCHS'
QUALIFICATION_LABELS = ('QUALIFICATION_ONLY', 'NOT_SCIENTIFIC', 'NOT_A_TRAINED_STDN_CHECKPOINT')

# ---------------------------------------------------------------- launch environment (M6D2b lock precision_env)
PRECISION_ENV = {
    'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'CUDA_VISIBLE_DEVICES': '0', 'NVIDIA_TF32_OVERRIDE': '0',
    'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONNOUSERSITE': '1', 'TF_CUDNN_DETERMINISTIC': '1', 'TF_DETERMINISTIC_OPS': '1',
    'TF_ENABLE_AUTO_MIXED_PRECISION': '0', 'TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE': '0',
    'TF_ENABLE_CUBLAS_TENSOR_OP_MATH_FP32': '0', 'TF_ENABLE_CUDNN_RNN_TENSOR_OP_MATH_FP32': '0',
    'TF_ENABLE_CUDNN_TENSOR_OP_MATH_FP32': '0', 'TF_ENABLE_ONEDNN_OPTS': '0', 'TF_XLA_FLAGS': '--tf_xla_auto_jit=0'}

# Every option a scientific CLI must refuse (only --seed / --execution-config / --preflight-only exist).
REFUSED_OPTIONS = tuple('--' + x for x in (
    'epochs', 'max-epoch', 'max_epoch', 'steps', 'steps-per-epoch', 'train-steps', 'val-steps', 'val_steps',
    'batch', 'batch-size', 'batch_size', 'g-d-ratio', 'gd-ratio', 'lr', 'learning-rate', 'decay', 'lr-decay',
    'decay-steps', 'optimizer', 'beta1', 'beta2', 'epsilon', 'ema', 'loss', 'losses', 'augment', 'augmentation',
    'flip', 'brightness', 'no-flip', 'no-brightness', 'pairs', 'pair-manifest', 'val-pairs', 'split',
    'split-manifest', 'pack', 'feed-pack', 'landmarks', 'faces', 'faces-root', 'checkpoint', 'checkpoint-rule',
    'ckpt', 'output', 'output-root', 'run-root', 'run-dir', 'runtime-root', 'log-dir', 'root', 'fidelity', 'test',
    'test-split', 'eval', 'evaluate', 'best', 'best-checkpoint', 'select', 'resume', 'restore', 'from-checkpoint',
    'init-checkpoint', 'pretrained', 'limit', 'limit-steps', 'qualification', 'qualification-seed', 'max-steps',
    'source', 'source-root', 'config', 'resolution', 'image-size', 'map-size', 'seed-offset', 'val-seed',
    'shuffle-seed', 'prefetch', 'workers', 'num-parallel-calls'))


class E03Stop(RuntimeError):
    """A production gate failed: STOP_AND_REPORT, never retried or worked around."""


def require(value, message):
    if not value:
        raise E03Stop('E03 STOP: ' + message)


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------- seeds (ruling V, decision D2)
def validate_mode_seed(mode, seed):
    require(mode in (SCIENTIFIC, QUALIFICATION), 'mode')
    require(type(seed) is int, 'integer seed')
    if mode == SCIENTIFIC:
        require(seed in SCIENTIFIC_SEEDS, 'scientific seed must be one of 42, 1337, 2026')
    else:
        require(seed == QUALIFICATION_SEED and seed not in SCIENTIFIC_SEEDS, 'qualification seed must be 60903')


def val_parser_seed(seed):
    """1 + (uint32 big-endian of SHA256(UTF8("GPAT_TRANSFERBENCH|E03|VAL_DATA|" + str(seed)))[0:4] mod 2147483647)."""
    digest = hashlib.sha256(('GPAT_TRANSFERBENCH|E03|VAL_DATA|' + str(seed)).encode('utf-8')).digest()
    return 1 + int.from_bytes(digest[:4], 'big') % 2147483647


def rng_plan(seed):
    """Split-specific RNG objects (never process-global): parser draws + explicit tf.data shuffle op seeds."""
    v = val_parser_seed(seed)
    return {'tf_graph_seed': seed, 'TRAIN': {'parser_seed': seed, 'shuffle_seed': seed},
            'VAL': {'parser_seed': v, 'shuffle_seed': v},
            'process_global_python_numpy': 'seeded with experiment_seed; never used by the feed',
            'val_seed_rule': '1 + (u32_be(SHA256(UTF8("GPAT_TRANSFERBENCH|E03|VAL_DATA|" + str(seed))))[0:4] '
                             'mod 2147483647)'}


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


EXEC_CONFIG = 'configs/execution/m5_gpu_3090.yaml'


def storage_from_exec_config(rel=EXEC_CONFIG, root=ROOT):
    """gpatbench m2b.resolve_roots containment semantics (Python 3.8: relative_to instead of is_relative_to)."""
    require(rel == EXEC_CONFIG, 'execution config must be ' + EXEC_CONFIG)
    path = Path(root) / rel
    cfg = yaml.safe_load(path.read_text())
    runtime_root = Path(cfg['storage']['runtime_root']).resolve()
    for key, value in cfg['storage']['roots'].items():
        try:
            Path(value).resolve().relative_to(runtime_root)
        except ValueError:
            require(False, 'execution root %s escapes %s' % (key, runtime_root))
    faces = Path(cfg['storage']['roots']['faces_256_root'])
    require(faces == runtime_root / 'data' / 'processed' / 'faces_256', 'faces root = <runtime_root>/data/processed/'
                                                                         'faces_256')
    require(not faces.is_symlink() and faces.resolve() == faces, 'faces root is not a symlink')
    return {'exec_config': rel, 'exec_config_sha256': sha256_file(path), 'runtime_root': str(runtime_root),
            'faces_256_root': str(faces), 'pack_root': str(pack_root(runtime_root))}


QUALIFICATION_CASES = ('val_a', 'val_b', 'no_val', 'terminal_probe')


# ---------------------------------------------------------------- schedule accounting (official)
def is_gd_step(step):
    """train.py:168: step % config.G_D_RATIO == 0 -> sess.run(losses + [g_op, d_op, fig])."""
    return step % G_D_RATIO == 0


def applications_after(epoch_index, step):
    """Counters after zero-based TRAIN step `step` of zero-based epoch `epoch_index` (full official epochs before)."""
    g = epoch_index * STEPS_PER_EPOCH + step + 1
    d = epoch_index * (STEPS_PER_EPOCH // G_D_RATIO) + step // G_D_RATIO + 1
    return {'G': g, 'D': d, 'global_step': g + d}


def expected_learning_rate(global_step):
    """tf.train.exponential_decay(6e-5, global_step, 20000, 0.9, staircase=True) on the SHARED global_step."""
    return LEARNING_RATE * DECAY_RATE ** (global_step // DECAY_STEPS)


def decay_boundaries():
    return [k * DECAY_STEPS for k in range(1, FINAL_GLOBAL_STEP // DECAY_STEPS + 1) if k * DECAY_STEPS < FINAL_GLOBAL_STEP]


def is_train_figure_step(step):
    return step % LOG_FR_TRAIN == 0                     # train.py:181


def is_val_figure_step(step):
    return step % LOG_FR_VAL == 0                       # train.py:201


def train_figure_name(epoch_index, step):
    return 'Epoch-%d-%d.png' % (epoch_index + 1, step + 1)            # train.py:182


def val_figure_name(epoch_index, step):
    return 'Epoch-%d-Val-%d.png' % (epoch_index + 1, step + 1)        # train.py:202


def checkpoint_prefix(epoch):
    return 'ckpt-%d' % epoch                            # saver.save(..., LOG_DIR + '/ckpt', global_step=epoch+1)


def checkpoint_kind(epoch):
    selected = epoch == MAX_EPOCH
    return ('selected' if selected else 'official', selected,
            CHECKPOINT_RULE if selected else 'Official cadence; not final selection')


def epoch_plan():
    """The full official order of one scientific run (used by tests and the controller)."""
    plan = []
    for e in range(MAX_EPOCH):
        plan += [('train', e, STEPS_PER_EPOCH), ('checkpoint', e, checkpoint_prefix(e + 1)), ('val', e, VAL_STEPS)]
    return plan + [('complete', MAX_EPOCH - 1, None)]


# ---------------------------------------------------------------- contract / identities
def verify_contract(root=ROOT):
    """Byte identities of every frozen authority; returns the parsed frozen config. No framework import."""
    root = Path(root)
    raw = (root / CONFIG_PATH).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == CONFIG_SHA and raw == (root / SNAPSHOT_PATH).read_bytes(),
            'frozen E03 config sha256 / snapshot identity')
    cfg = yaml.safe_load(raw)
    for rel, digest in (RUNTIME_ADDENDUM, RULING_ADDENDUM, RULING_YAML, LOCK, COMPAT_MANIFEST, PREPROCESS):
        require(sha256_file(root / rel) == digest, rel + ' sha256')
    require(cfg['spec_sha256'] == SPEC_SHA and cfg['method_id'] == METHOD_ID and cfg['display_name'] == DISPLAY_NAME,
            'config identity')
    require(cfg['source']['pinned_commit'] == SOURCE_PIN and cfg['seeds']['experiment_seeds'] == list(SCIENTIFIC_SEEDS),
            'source pin / seeds')
    t, o = cfg['training'], cfg['optimizer']
    require((t['input_resolution'], t['map_size'], t['batch_size'], t['g_d_ratio'], t['max_epoch'],
             t['steps_per_epoch'], t['val_steps']) == (IMAGE_SIZE, MAP_SIZE, BATCH_SIZE, G_D_RATIO, MAX_EPOCH,
                                                       STEPS_PER_EPOCH, VAL_STEPS), 'frozen training values')
    require((o['learning_rate'], o['lr_decay'], o['beta1'], o['beta2'], o['epsilon'], o['weight_ema_decay'],
             o['loss_ema_decay']) == (LEARNING_RATE, {'type': 'exponential', 'rate': DECAY_RATE,
                                                      'every_steps': DECAY_STEPS, 'staircase': True},
                                      BETA1, BETA2, EPSILON, WEIGHT_EMA, LOSS_EMA), 'frozen optimizer values')
    require(cfg['checkpoint']['rule'] == CHECKPOINT_RULE and cfg['checkpoint']['selection_uses_val'] is False and
            cfg['checkpoint']['selection_uses_test'] is False, 'checkpoint rule')
    require(cfg['data']['splits']['TEST']['allowed'] is False and cfg['data']['split_manifest_sha256'] ==
            SPLIT_MANIFEST[1] and cfg['data']['preprocess_contract_sha256'] == PREPROCESS[1], 'data firewall')
    ruling = yaml.safe_load((root / RULING_YAML[0]).read_text())
    require(ruling['status'] == 'OWNER_RESOLVED' and ruling['resolved_execution_fidelity'] == FIDELITY, 'owner ruling')
    v = ruling['rulings']['V']
    require(v['decision'] == 'SEPARATE_DETERMINISTIC_SPLIT_RNG_STREAMS' and all(
        val_parser_seed(int(s)) == x for s, x in list(v['val_parser_seeds'].items()) +
        list(v['qualification_val_parser_seed'].items())), 'ruling V seeds')
    for split, (path, digest, rows) in PAIR_MANIFESTS.items():
        r = ruling['rulings']['R1' if split == 'TRAIN' else 'R2']
        require((r['decision'], r['sha256'], r['rows']) == (path, digest, rows), 'ruling relation ' + split)
    require(ruling['rulings']['R3']['decision'] == 'RETAIN_OFFICIAL_DIAGNOSTIC_FIGURES', 'ruling R3')
    return cfg


def verify_source(root=ROOT):
    """All 25 recorded source-cache files byte-identical; the four removed upstream weight files stay absent."""
    root = Path(root)
    ev = json.loads((root / SOURCE_EVIDENCE).read_text())['laptop_source']
    require(ev['pin'] == SOURCE_PIN, 'source evidence pin')
    src = root / SOURCE_ROOT
    hashes = {}
    for rel, row in sorted(ev['files'].items()):
        hashes[rel] = sha256_file(src / rel)
        require(hashes[rel] == row['sha256'], 'source file ' + rel)
    require(all(not (src / rel).exists() for rel in ev['deleted']), 'upstream weights absent')
    return {'commit': SOURCE_PIN, 'tree': ev['tree'], 'files': len(hashes),
            'files_sha256': hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()}


STDN_REQUIRED_FILES = ('README.md', 'train.py', 'test.py', 'model/config.py', 'model/model.py', 'model/dataset.py',
                       'model/warp.py', 'model/loss.py', 'model/utils.py')     # methods/stdn/source.py REQUIRED_FILES


def verify_source_checkout(root=ROOT):
    """Python 3.8 equivalent of methods.common.learned.verify_source for E03 (whose str.removesuffix needs 3.9):
    the same pin entry, checkout top level, HEAD, HEAD^{tree}, origin URL, and per-file git blob + cited sha256."""
    import subprocess
    root = Path(root)
    cfg = yaml.safe_load((root / CONFIG_PATH).read_text())
    source = cfg['source']
    pins = json.loads((root / source['provenance']).read_text())['sources']
    matches = [p for p in pins.values() if METHOD_ID in p.get('method_ids', []) and p['repository'] ==
               source['repository']]
    require(len(matches) == 1 and matches[0]['pinned_commit'] == source['pinned_commit'] == SOURCE_PIN, 'source pin')
    pin = matches[0]
    src = (root / source['local_source_path']).resolve()

    def git(*a):
        env = dict(__import__('os').environ, GIT_NO_LAZY_FETCH='1', GIT_TERMINAL_PROMPT='0')
        r = subprocess.run(['git', '-C', str(src)] + list(a), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           universal_newlines=True)
        require(r.returncode == 0, 'source git verification failed: %s' % (a,))
        return r.stdout.strip()

    def strip_git(url):
        url = url[:-4] if url.endswith('.git') else url
        return url.rstrip('/')
    require((src / '.git').exists() and Path(git('rev-parse', '--show-toplevel')).resolve() == src, 'source checkout')
    require(git('rev-parse', 'HEAD') == SOURCE_PIN and git('rev-parse', 'HEAD^{tree}') == pin['commit_tree'],
            'source commit/tree')
    require(strip_git(git('remote', 'get-url', 'origin')) == strip_git(source['repository']), 'source origin')
    verified = {}
    for rel in sorted(set(STDN_REQUIRED_FILES) | set(pin['cited_files'])):
        path = src / rel
        require(path.is_file() and not path.is_symlink(), 'source file ' + rel)
        path.resolve().relative_to(src)
        entry = git('ls-tree', 'HEAD', '--', rel).split()
        require(len(entry) == 4 and entry[0] in ('100644', '100755') and entry[1] == 'blob', 'pinned regular ' + rel)
        raw = path.read_bytes()
        require(hashlib.sha1(('blob %d\0' % len(raw)).encode() + raw).hexdigest() == entry[2], 'pinned blob ' + rel)
        cited = pin['cited_files'].get(rel)
        require(not cited or (hashlib.sha256(raw).hexdigest() == cited['sha256'] and len(raw) == cited['size_bytes']),
                'cited digest ' + rel)
        verified[rel] = hashlib.sha256(raw).hexdigest()
    return {'repository': source['repository'], 'commit': SOURCE_PIN, 'tree': pin['commit_tree'], 'root': str(src),
            'files_sha256': verified, 'verification': 'PASS'}


def production_file_hashes(root=ROOT):
    return {rel: sha256_file(Path(root) / rel) for rel in PRODUCTION_FILES if (Path(root) / rel).is_file()}


def identities(cfg, source, root=ROOT):
    return {'config_sha256': CONFIG_SHA, 'spec_sha256': SPEC_SHA, 'runtime_addendum_sha256': RUNTIME_ADDENDUM[1],
            'ruling_addendum_sha256': RULING_ADDENDUM[1], 'ruling_yaml_sha256': RULING_YAML[1],
            'environment_lock_sha256': LOCK[1], 'compatibility_manifest_sha256': COMPAT_MANIFEST[1],
            'logging_contract_sha256': sha256_file(Path(root) / LOGGING_CONTRACT),
            'source_commit': source['commit'], 'source_tree': source['tree'],
            'source_files_sha256': source['files_sha256'], 'pack_manifest_sha256': PACK_MANIFEST_SHA256,
            'pair_manifest_sha256': {s: v[1] for s, v in PAIR_MANIFESTS.items()},
            'split_manifest_sha256': SPLIT_MANIFEST[1], 'preprocess_sha256': PREPROCESS[1],
            'fidelity': FIDELITY, 'production_files_sha256': production_file_hashes(root)}


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


def environment_differences(lock, live):
    """Exact lock identity except the NVIDIA driver version (owner-approved driver-version-only compatibility)."""
    diff = []
    for key in ('python', 'tensorflow', 'environment_path', 'packages', 'cuda_cudnn_native', 'precision_env',
                'session_config'):
        if lock[key] != live.get(key):
            diff.append(key)
    old, new = (x.split(', ') for x in (lock['gpu_driver'], live.get('gpu_driver', '')))
    if len(old) != 4 or len(new) != 4 or (old[0], old[1], old[3]) != (new[0], new[1], new[3]):
        diff.append('gpu_driver(model/capability/memory)')
    if live.get('gpu_uuid') != GPU_UUID:
        diff.append('gpu_uuid')
    driver = {'lock_driver_version': old[2] if len(old) == 4 else None,
              'observed_driver_version': new[2] if len(new) == 4 else None, 'driver_policy': DRIVER_POLICY,
              'gpu_model': new[0] if new else None, 'gpu_uuid': live.get('gpu_uuid')}
    if driver['observed_driver_version'] != driver['lock_driver_version']:
        if (driver['lock_driver_version'], driver['observed_driver_version']) != (LOCK_DRIVER, OBSERVED_DRIVER):
            diff.append('gpu_driver(version outside the owner-approved pair)')
    return sorted(diff), driver
