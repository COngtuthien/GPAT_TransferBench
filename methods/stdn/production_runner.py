"""E03 STDN production controller (Python 3.8, NVIDIA TF 1.15.5): the unchanged official graph on the Track-A feed.

Execution fidelity: FAITHFUL_OFFICIAL_CORE_WITH_TRACK_A_CONTROLLED_DATA_ADAPTATION_AND_RUNTIME_COMPATIBILITY.

Official and unchanged (imported from the hash-verified pinned source, never patched or re-implemented):
  train._step (Gen / Disc_s / losses / hard samples / get_train_op: Adam, exponential decay on the shared global_step,
  loss EMA, weight EMA), contrib batch_norm (inline moving-stat updates), model.warp.generate_offset_map, and the
  train.py epoch order: 2000 TRAIN steps (step % 2 == 0 -> ONE sess.run(losses + [g_op, d_op, fig]); otherwise
  sess.run(losses + [g_op, fig])), saver.save(..., global_step=epoch+1), then 500 VAL steps sess.run(losses_val +
  [fig_val]); figures at step % 200 / step % 50 == 0 from the fig of that same run.
Controlled benchmark adaptation (owner rulings V/R1/R2/R3, decisions D1-D3):
  one dataset element = one frozen pair row (pairs_train_v1 / val_pairs_v1 via the verified E03 feed pack);
  from_tensor_slices -> shuffle(full population, explicit split seed, reshuffle_each_iteration=True) -> repeat(-1)
  -> map(num_parallel_calls=1) -> batch(2) -> prefetch(1); the parser draws flips from a split-specific
  numpy.random.RandomState (TRAIN: experiment seed; VAL: the D2 derived seed), never from process-global RNG; TRAIN
  keeps the official tf.image.random_brightness(0.25), VAL has none; landmarks are M2 landmarks_px256 / 256 with the
  exact flip rule of methods/stdn/landmarks.py.
Runtime compatibility: the M6D2b session configuration and precision environment (environments/e03_stdn.lock.json).

The upstream main() (and its Saver.restore resume path) is never called. Runs are fresh only.
"""
import ctypes
import fcntl
import getpass
import hashlib
import importlib
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import random
import socket
import subprocess
import sys
import sysconfig
import time
from types import SimpleNamespace
import uuid

import numpy as np


def _is_relative_to(self, *other):
    """CPython 3.9 pathlib.PurePath.is_relative_to, verbatim semantics."""
    try:
        self.relative_to(*other)
        return True
    except ValueError:
        return False


# Benchmark-infrastructure compatibility (NOT an STDN/TensorFlow API shim): the shared, hash-pinned
# methods/common/config.py uses PurePath.is_relative_to (Python >= 3.9); the qualified E03 runtime is Python 3.8.
# The exact 3.9 semantics are installed only when missing, only in this process, and disclosed in run_start.
PY38_PATHLIB_BACKPORT = not hasattr(Path, 'is_relative_to')
if PY38_PATHLIB_BACKPORT:
    import pathlib
    pathlib.PurePath.is_relative_to = _is_relative_to

from methods.common.config import load_method_config  # noqa: E402
from methods.common.learned import PreparationError, serialized_tf_map  # noqa: E402
from methods.common.learned_runlog import LearnedRunContext  # noqa: E402
from methods.common.runlog import (RunContext, RunDirectoryError, atomic_write_bytes, atomic_write_json,  # noqa: E402
                                   compute_run_id, git_commit, load_logging_contract)
from methods.stdn import feed_pack  # noqa: E402
from methods.stdn import production_io as pio  # noqa: E402
from methods.stdn.landmarks import LandmarkAdapter  # noqa: E402

require = pio.require
ROOT = pio.ROOT
SCIENTIFIC_PLAN = tuple((pio.STEPS_PER_EPOCH, 'checkpoint', pio.VAL_STEPS) for _ in range(pio.MAX_EPOCH))


# ================================================================= run context (run_logging_v1)
class E03RunContext(LearnedRunContext):
    """SCIENTIFIC: the unchanged LearnedRunContext (runs/m6/E03/seed_<seed>, frozen-rule checkpoint metadata).
    QUALIFICATION: the same writers bound to <rt>/qualification/e03_production/E03/<case>-q60903-<run_id>."""

    def __init__(self, *, mode, seed, runtime_root, environment, missing_environment_reasons, identities, case=None,
                 command_line=None):
        pio.validate_mode_seed(mode, seed)
        config = load_method_config(pio.METHOD_ID)
        self.mode, self.case, self.identities = mode, case, dict(identities)
        require(config['_runtime']['config_sha256'] == pio.CONFIG_SHA == identities['config_sha256'], 'config identity')
        if mode == pio.SCIENTIFIC:
            require(case is None, 'scientific runs have no qualification case')
            super().__init__(method_id=pio.METHOD_ID, seed=seed, config=config, runtime_root=runtime_root,
                             command_line=command_line, resume=False, environment=environment,
                             missing_environment_reasons=missing_environment_reasons)
            self.qualification_seed = None
            return
        self.contract = load_logging_contract()
        self.method_id, self.seed, self.qualification_seed = pio.METHOD_ID, None, seed
        self.config, self.runtime_root, self.resume = config, Path(runtime_root), False
        self.source_commits = {config['source']['repository']: config['source']['pinned_commit']}
        self.command_line = command_line if command_line is not None else ' '.join(sys.argv)
        self.config_sha256 = config['_runtime']['config_sha256']
        self.config_path = config['_runtime']['config_path']
        self.git_commit = git_commit()
        self.run_id = compute_run_id(pio.METHOD_ID, seed, self.config_sha256, self.git_commit)
        self.run_dir = pio.run_root(runtime_root, mode, seed, case=case, run_id=self.run_id)
        self.start_utc = self._metrics_fh = self._generation_fh = self._closed_summary = self._lock_fh = None
        self._records, self.run_uuid, self._started = 0, str(uuid.uuid4()), time.monotonic()
        self._validate_metadata(environment)
        self._environment, self._environment_reasons = dict(environment), dict(missing_environment_reasons)

    @property
    def seed_value(self):
        return self.seed if self.mode == pio.SCIENTIFIC else self.qualification_seed

    def open(self):
        if self.mode == pio.SCIENTIFIC:
            return super().open()
        if self._metrics_fh is not None or self._closed_summary is not None:
            raise RunDirectoryError('RunContext objects may be opened only once')
        self.run_dir.parent.mkdir(parents=True, exist_ok=True)
        self._lock_fh = (self.run_dir.parent / ('.%s.lock' % self.run_dir.name)).open('a')
        try:
            fcntl.flock(self._lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return self._open_locked()
        except BaseException:
            for handle in (self._metrics_fh, self._generation_fh, self._lock_fh):
                if handle is not None:
                    handle.close()
            self._metrics_fh = self._generation_fh = self._lock_fh = None
            raise

    def _open_locked(self):
        if self.run_dir.exists():          # fresh only in both modes (E03 resume is not qualified)
            raise RunDirectoryError('STOP_AND_REPORT: %s exists; E03 runs are fresh only (resume is not qualified)'
                                    % self.run_dir)
        result = super()._open_locked()
        index = json.loads(self.path('checkpoint_index').read_text())
        index.update(note='Official TF Saver checkpoints actually written (one record per physical file: .data, '
                          '.index, .meta). ckpt-1..ckpt-49 official cadence; ckpt-50 = OFFICIAL_LATEST_FINAL_CKPT_50. '
                          'Each ckpt-e is saved after the epoch TRAIN block and before its VAL block. No resume sidecar.',
                     runner_mode=self.mode, checkpoint_rule=pio.CHECKPOINT_RULE,
                     selected_final=('checkpoints/' + pio.checkpoint_prefix(pio.MAX_EPOCH)
                                     if self.mode == pio.SCIENTIFIC else None),
                     scientific_checkpoints=self.mode == pio.SCIENTIFIC, resume_sidecars=None)
        if self.mode == pio.QUALIFICATION:
            index.update(labels=self.labels(), qualification_seed=self.qualification_seed, qualification_case=self.case,
                         selected_final_reason='QUALIFICATION_ONLY: no scientific selection exists')
        atomic_write_json(self.path('checkpoint_index'), index)
        return result

    def labels(self):
        if self.mode == pio.SCIENTIFIC:
            return []
        return list(pio.QUALIFICATION_LABELS) + ([pio.TERMINAL_PROBE_LABEL] if self.case == 'terminal_probe' else [])

    def _resolved_config(self):
        resolved = super()._resolved_config()
        resolved['_resolved'].update({
            'runner_mode': self.mode, 'qualification_seed': self.qualification_seed, 'qualification_case': self.case,
            'scientific_run': self.mode == pio.SCIENTIFIC, 'labels': self.labels(),
            'note': 'Verbatim copy of the frozen config above; execution bindings below are the owner rulings and the '
                    'M6D2b runtime layer. No scientific value was overridden.',
            'historical_fidelity_field': self.config['fidelity_class'], 'fidelity': pio.FIDELITY,
            'fidelity_layers': pio.FIDELITY_LAYERS, 'identities': self.identities,
            'owner_resolution': {'addendum': pio.RULING_ADDENDUM[0], 'addendum_sha256': pio.RULING_ADDENDUM[1],
                                 'yaml': pio.RULING_YAML[0], 'yaml_sha256': pio.RULING_YAML[1]},
            'relations': {s: {'path': v[0], 'sha256': v[1], 'rows': v[2]} for s, v in pio.PAIR_MANIFESTS.items()},
            'feed_pack': {'dir': pio.PACK_DIRNAME, 'manifest_sha256': pio.PACK_MANIFEST_SHA256},
            'rng': pio.rng_plan(self.seed_value),
            'schedule': {'train_steps_per_epoch': pio.STEPS_PER_EPOCH, 'val_steps_per_epoch': pio.VAL_STEPS,
                         'max_epoch': pio.MAX_EPOCH, 'even_step': 'sess.run(losses + [g_op, d_op, fig])',
                         'odd_step': 'sess.run(losses + [g_op, fig])', 'global_step_per_epoch': pio.GLOBAL_STEP_PER_EPOCH,
                         'final_global_step': pio.FINAL_GLOBAL_STEP, 'decay_steps_shared_global': pio.DECAY_STEPS,
                         'epoch_order': ['train', 'saver.save(global_step=epoch+1)', 'val']},
            'feed': {'pipeline': 'from_tensor_slices -> shuffle(N, seed, reshuffle_each_iteration=True) -> repeat(-1) '
                                 '-> map(num_parallel_calls=1) -> batch(2) -> prefetch(1)',
                     'TRAIN_augmentation': 'independent live/spoof flip (RandomState draw > 0.5), random_brightness 0.25',
                     'VAL_augmentation': 'independent live/spoof flip (VAL RandomState draw > 0.5), no brightness',
                     'landmarks': 'M2 landmarks_px256 / 256; flip: x -> 1 - x then iBUG-68 lm_reverse_list'},
            'figures': {'TRAIN': 'step % 200 == 0', 'VAL': 'step % 50 == 0', 'dir': 'diagnostics/figures',
                        'source': 'fig / fig_val of the same official sess.run'},
            'checkpoint_rule': {'rule': pio.CHECKPOINT_RULE, 'cadence': 'ckpt-1..ckpt-50 (one per epoch, before VAL)',
                                'selected': 'checkpoints/' + pio.checkpoint_prefix(pio.MAX_EPOCH)},
            'resume': pio.RESUME,
            'run_id_rule': 'run_logging_v1: sha256(method_id|seed|config_sha256|git_commit)[:16]'})
        return resolved

    def _manifest(self, **kwargs):
        m = super()._manifest(**kwargs)
        m.update(runner_mode=self.mode, scientific_run=self.mode == pio.SCIENTIFIC,
                 qualification_seed=self.qualification_seed, qualification_case=self.case, fidelity=pio.FIDELITY,
                 historical_fidelity_field=self.config['fidelity_class'], identities=self.identities,
                 resume=pio.RESUME, val_split_accessed=True, val_role='DIAGNOSTICS_ONLY_NEVER_SELECTS',
                 test_code_path_present=False)
        if self.mode == pio.QUALIFICATION:
            m['labels'] = self.labels()
            m['missing_field_reasons']['experiment_seed'] = ('QUALIFICATION_ONLY: seed %d is not an experiment seed'
                                                             % self.qualification_seed)
        return m

    def record_checkpoint(self, **kwargs):
        if self.mode == pio.SCIENTIFIC:
            return super().record_checkpoint(**kwargs)
        kind, selected, reason = pio.checkpoint_kind(kwargs['epoch'])
        require((kwargs['checkpoint_type'], kwargs['selected_for_final'], kwargs['selection_reason']) ==
                (kind, selected, reason), 'qualification checkpoint semantics equal the production cadence')
        require(selected is (self.case == 'terminal_probe'), 'only the terminal-semantics probe records ckpt-50')
        return RunContext.record_checkpoint(self, **kwargs)


# ================================================================= environment (M6D2b lock, driver-only tolerance)
def session_config(tf):
    """The M6D2b qualified session configuration (environments/e03_stdn.lock.json session_config)."""
    from tensorflow.core.protobuf import rewriter_config_pb2
    c = tf.ConfigProto(allow_soft_placement=False, intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    c.gpu_options.allow_growth = True
    c.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.OFF
    r = c.graph_options.rewrite_options
    r.disable_meta_optimizer = True
    r.auto_mixed_precision = rewriter_config_pb2.RewriterConfig.OFF
    return c


def live_environment(tf):
    import importlib.metadata as md
    env = Path(sys.executable).parents[1]
    site = env / 'lib/python3.8/site-packages/nvidia'
    cuda = ctypes.CDLL(str(site / 'cuda_runtime/lib/libcudart.so.11.0'))
    cudnn = ctypes.CDLL(str(site / 'cudnn/lib/libcudnn.so.8'))
    version = ctypes.c_int()
    rc = cuda.cudaRuntimeGetVersion(ctypes.byref(version))
    cudnn.cudnnGetVersion.restype = ctypes.c_size_t
    smi = lambda q: subprocess.check_output(['nvidia-smi', '--query-gpu=' + q, '--format=csv,noheader']).decode().strip()  # noqa: E731
    return {'python': sys.version, 'executable': sys.executable, 'tensorflow': tf.__version__,
            'environment_path': str(env),
            # TF/setuptools add vendored packages to sys.path at import; inventory the fixed environment paths only
            # (the M6D2b lock was taken in a process without TensorFlow imported).
            'packages': sorted([{'name': d.metadata['Name'], 'version': d.version} for d in md.distributions(
                path=sorted({sysconfig.get_path('purelib'), sysconfig.get_path('platlib')}))],
                key=lambda d: d['name'].lower()),
            'cuda_cudnn_native': {'cudaRuntimeGetVersion': version.value, 'cuda_query_rc': rc,
                                  'cudnnGetVersion': int(cudnn.cudnnGetVersion())},
            'precision_env': {k: os.environ.get(k) for k in pio.PRECISION_ENV},
            'session_config': str(session_config(tf)),
            'gpu_driver': smi('name,compute_cap,driver_version,memory.total'), 'gpu_uuid': smi('uuid')}


def verify_environment(tf):
    lock = json.loads((ROOT / pio.LOCK[0]).read_text())
    live = live_environment(tf)
    require(live['executable'] == pio.PYTHON_EXECUTABLE or Path(live['executable']).resolve() ==
            Path(pio.PYTHON_EXECUTABLE).resolve(), 'E03 Python executable')
    diff, driver = pio.environment_differences(lock, live)
    require(not diff, 'gpat-m6-e03 differs from the E03 lock: ' + ', '.join(diff))
    return live, driver


def environment_record(tf, live):
    gpu = live['gpu_driver'].split(', ')
    env = {'host': socket.gethostname(), 'user': getpass.getuser(), 'platform': platform.platform(),
           'python_version': platform.python_version(), 'numpy_version': np.__version__, 'gpu_model': gpu[0],
           'gpu_count': len(tf.config.experimental.list_physical_devices('GPU')),
           'cuda_version': str(live['cuda_cudnn_native']['cudaRuntimeGetVersion']),
           'cudnn_version': str(live['cuda_cudnn_native']['cudnnGetVersion']), 'framework': 'tensorflow',
           'framework_version': tf.__version__, 'pytorch_version': None, 'tensorflow_version': tf.__version__,
           'dependency_fingerprint': pio.LOCK[1], 'environment_lock_path': pio.LOCK[0],
           'environment_name': pio.ENVIRONMENT_NAME, 'python_executable': sys.executable,
           'gpu_note': 'single RTX 3090; TF1.15 FP32; M6D2b session/precision layer'}
    return env, {'pytorch_version': 'E03 is a TensorFlow 1.x method; PyTorch is not part of gpat-m6-e03'}


# ================================================================= Track-A feed (rulings R1/R2/V, D1)
def make_landmark_adapter():
    """The unchanged methods/stdn/landmarks.py LandmarkAdapter (pinned lm_reverse_list, /256, 1 - x + permutation).

    Its constructor calls methods.common.learned.verify_source, which uses str.removesuffix (Python >= 3.9). For
    the construction only, that call is answered by production_io.verify_source_checkout -- the same pin/checkout/
    HEAD/tree/origin/blob/cited-digest checks written for Python 3.8 -- and the original binding is restored.
    """
    from methods.stdn import landmarks as landmarks_module
    verified = pio.verify_source_checkout()
    original = landmarks_module.validate_source
    landmarks_module.validate_source = lambda config, source_root=None: verified
    try:
        adapter = LandmarkAdapter(load_method_config(pio.METHOD_ID))
    finally:
        landmarks_module.validate_source = original
    require(adapter.permutation.tolist() == LANDMARK_PERMUTATION and adapter.width == pio.IMAGE_SIZE,
            'pinned iBUG-68 lm_reverse_list and width 256')
    return adapter


# dataset.py:96-100 (one-based list, minus 1), as an independent cross-check of the adapter's AST extraction
LANDMARK_PERMUTATION = [x - 1 for x in (17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 27, 26, 25, 24, 23,
                                        22, 21, 20, 19, 18, 28, 29, 30, 31, 36, 35, 34, 33, 32, 46, 45, 44, 43, 48, 47,
                                        40, 39, 38, 37, 42, 41, 55, 54, 53, 52, 51, 50, 49, 60, 59, 58, 57, 56, 65, 64,
                                        63, 62, 61, 68, 67, 66)]


class FaceReader:
    """Canonical faces of feed-pack samples only (TRAIN/VAL); every PNG is SHA256-checked before decoding."""

    def __init__(self, faces_root, pack):
        self.root, self.pack, self.reads = Path(faces_root), pack, {'TRAIN': 0, 'VAL': 0}

    def __call__(self, sample_id):
        from PIL import Image
        rec = self.pack.sample(sample_id)                 # KeyError for any non-pack sample: no TEST path exists
        raw = (self.root / rec['dataset'] / (sample_id + '.png')).read_bytes()
        require(hashlib.sha256(raw).hexdigest() == rec['face_png_sha256'], 'canonical face sha256 ' + sample_id)
        image = Image.open(io.BytesIO(raw))
        require(image.mode == 'RGB' and image.size == (pio.IMAGE_SIZE, pio.IMAGE_SIZE), 'canonical RGB256 face')
        self.reads[rec['split']] += 1
        return image


class SplitParser:
    """py_func body: the official _parse_function semantics on one frozen pair row, with a split-only RNG object."""

    def __init__(self, split, pack, reader, landmarks, warp, seed, keep_log=False):
        import cv2
        self.split, self.rows, self.pack, self.reader, self.lm, self.warp = (
            split, pack.pair_rows(split), pack, reader, landmarks, warp)
        self.cv2, self.rng, self.seed = cv2, np.random.RandomState(seed), seed
        self.calls = self.draws = 0
        self.log = [] if keep_log else None

    def rng_state_sha256(self):
        name, keys, pos, has_gauss, cached = self.rng.get_state()
        return hashlib.sha256(keys.tobytes() + repr((name, pos, has_gauss, cached)).encode()).hexdigest()

    def _side(self, sample_id):
        size = pio.IMAGE_SIZE
        image = np.array(self.reader(sample_id).resize((size, size)), np.float32)   # official resize (no-op at 256)
        flip = bool(self.rng.rand() > 0.5)                                           # dataset.py:117/136/180/200
        self.draws += 1
        if flip:
            image = self.cv2.flip(image, 1)
        points = self.lm.from_cache({'landmarks_px256': self.pack.landmarks_px256(sample_id)}, flip=flip)
        return image, points, flip

    def __call__(self, index):
        index = int(index)
        row = self.rows[index]
        image_li, lm_li, f_li = self._side(row['target_live_id'])            # live first, as in the source
        image_sp, lm_sp, f_sp = self._side(row['source_spoof_id'])
        reg = self.warp.generate_offset_map(lm_sp, lm_li)                    # dataset.py:142/206
        self.calls += 1
        if self.log is not None:
            self.log.append((index, int(f_li), int(f_sp)))
        return (np.array(image_li, np.float32) / 255, np.array(image_sp, np.float32) / 255, reg.astype(np.float32),
                np.array([f_li, f_sp], np.int32), np.int64(index))


def make_iterator(tf, parser, n, shuffle_seed, brightness):
    size = pio.IMAGE_SIZE
    dataset = tf.data.Dataset.from_tensor_slices(np.arange(n, dtype=np.int64))
    dataset = dataset.shuffle(n, seed=shuffle_seed, reshuffle_each_iteration=True).repeat(-1)

    def parse_fn(index):
        li, sp, reg, flips, idx = tf.py_func(parser, [index], [tf.float32, tf.float32, tf.float32, tf.int32, tf.int64])
        li = tf.ensure_shape(li, [size, size, 3])
        sp = tf.ensure_shape(sp, [size, size, 3])
        reg = tf.ensure_shape(reg, [size, size, 3])
        flips, idx = tf.ensure_shape(flips, [2]), tf.ensure_shape(idx, [])
        if brightness:                                                  # dataset.py:151 (TRAIN only)
            bl, bs = tf.image.random_brightness(li, 0.25), tf.image.random_brightness(sp, 0.25)
            image = tf.stack([bl, bs], axis=0)
            delta = tf.stack([tf.reduce_mean(bl - li), tf.reduce_mean(bs - sp)])
        else:
            image = tf.stack([li, sp], axis=0)
            delta = tf.zeros([2], tf.float32)
        return image, reg, idx, flips, delta
    dataset = serialized_tf_map(dataset, parse_fn)
    dataset = dataset.batch(batch_size=pio.BATCH_SIZE).prefetch(1)
    return dataset.make_one_shot_iterator().get_next()


# ================================================================= controller
class Tee:
    def __init__(self, stream, path):
        self.stream, self.fh = stream, open(path, 'a', encoding='utf-8')

    def write(self, text):
        self.stream.write(text)
        self.fh.write(text)
        return len(text)

    def flush(self):
        self.stream.flush()
        self.fh.flush()


def seed_process(tf, seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)
    return dict(pio.rng_plan(seed), python_numpy_global=seed)


class Controller:
    """Builds the unchanged official graph on the Track-A feed and executes an epoch plan."""

    def __init__(self, tf, *, mode, seed, pack, faces_root, keep_parser_log=False):
        pio.validate_mode_seed(mode, seed)
        self.tf, self.mode, self.seed, self.pack = tf, mode, seed, pack
        src = str(ROOT / pio.SOURCE_ROOT)
        if src not in sys.path:
            sys.path.insert(0, src)
        self.train_mod = importlib.import_module('train')
        self.warp = importlib.import_module('model.warp')
        config_cls = importlib.import_module('model.config').Config
        require(Path(self.train_mod.__file__).resolve() == (ROOT / pio.SOURCE_ROOT / 'train.py').resolve(),
                'official train.py from the pinned source cache')
        self.config = config_cls.__new__(config_cls)        # class constants; the LOG_DIR-writing __init__ is not used
        require((self.config.BATCH_SIZE, self.config.IMAGE_SIZE, self.config.G_D_RATIO, self.config.STEPS_PER_EPOCH,
                 self.config.STEPS_PER_EPOCH_VAL, self.config.MAX_EPOCH, self.config.LOG_FR_TRAIN,
                 self.config.LOG_FR_TEST, self.config.NUM_EPOCHS_PER_DECAY * self.config.STEPS_PER_EPOCH) ==
                (pio.BATCH_SIZE, pio.IMAGE_SIZE, pio.G_D_RATIO, pio.STEPS_PER_EPOCH, pio.VAL_STEPS, pio.MAX_EPOCH,
                 pio.LOG_FR_TRAIN, pio.LOG_FR_VAL, pio.DECAY_STEPS), 'pinned model/config.py constants')
        self.seeding = seed_process(tf, seed)
        plan = pio.rng_plan(seed)
        landmarks = make_landmark_adapter()
        self.reader = FaceReader(faces_root, pack)
        self.parsers = {s: SplitParser(s, pack, self.reader, landmarks, self.warp, plan[s]['parser_seed'],
                                       keep_log=keep_parser_log) for s in ('TRAIN', 'VAL')}
        n = {s: len(pack.rows[s]) for s in ('TRAIN', 'VAL')}
        require(n == {s: v[2] for s, v in pio.PAIR_MANIFESTS.items()}, 'pack relation sizes')
        # official order: Dataset(train), Dataset(val), _step(train), _step(val), Saver
        self.train_next = make_iterator(tf, self.parsers['TRAIN'], n['TRAIN'], plan['TRAIN']['shuffle_seed'], True)
        self.val_next = make_iterator(tf, self.parsers['VAL'], n['VAL'], plan['VAL']['shuffle_seed'], False)
        n_vars_data = len(tf.global_variables())
        self.losses, self.g_op, self.d_op, self.fig = self.train_mod._step(
            self.config, SimpleNamespace(nextit=self.train_next[:2]), training_nn=True)
        n_vars_train = len(tf.global_variables())
        self.losses_val, g_val, d_val, self.fig_val = self.train_mod._step(
            self.config, SimpleNamespace(nextit=self.val_next[:2]), training_nn=False)
        self.saver = tf.train.Saver(max_to_keep=50)
        graph = tf.get_default_graph()
        self.global_step = tf.train.get_global_step()
        self.variables = tf.global_variables()
        adam = [o for o in graph.get_operations() if o.type in ('ApplyAdam', 'ResourceApplyAdam')]
        self.lr = {g: next(o.inputs[5] for o in adam if o.inputs[0].name.startswith(p))
                   for g, p in (('G', 'STDN/'), ('D', 'Disc/'))}
        try:
            from tensorflow.contrib.memory_stats import MaxBytesInUse
            self.max_bytes = MaxBytesInUse()
        except Exception:                                    # pragma: no cover - reported as a null with reason
            self.max_bytes = None
        self.structure = self._structure(graph, adam, n_vars_data, n_vars_train, g_val, d_val)

    def _structure(self, graph, adam, n_vars_data, n_vars_train, g_val, d_val):
        tf = self.tf
        tv = tf.trainable_variables()
        names = [v.name for v in self.variables]
        seen, stack = set(), [t.op for t in self.losses_val + [self.fig_val]]
        while stack:
            op = stack.pop()
            if op.name not in seen:
                seen.add(op.name)
                stack.extend(x.op for x in op.inputs)
                stack.extend(op.control_inputs)
        val_ops = [graph.get_operation_by_name(n) for n in seen]
        mutating = sorted(o.name for o in val_ops if o.type.startswith(('Assign', 'Apply', 'ResourceApply', 'Scatter'))
                          or o.type in ('AssignAdd', 'AssignSub', 'AssignVariableOp', 'CountUpTo'))
        s = {'global_variables': len(self.variables), 'trainable_variables': len(tv),
             'trainable_G': sum(v.name.startswith('STDN/') for v in tv),
             'trainable_D': sum(v.name.startswith('Disc/') for v in tv), 'adam_ops': len(adam),
             'bn_moving': sum('/moving_' in n for n in names),
             'weight_ema': sum('ExponentialMovingAverage' in n for n in names),
             'loss_ema': sum(n.endswith('/avg:0') for n in names),
             'variables_created_by_feed': n_vars_data, 'variables_created_by_val_graph': len(self.variables) - n_vars_train,
             'val_step_returns_no_train_op': g_val is None and d_val is None,
             'val_reachable_state_mutating_ops': mutating,
             'val_fused_bn_is_training_true': sum(1 for o in val_ops if o.type.startswith('FusedBatchNorm') and
                                                  o.get_attr('is_training')),
             'val_fused_bn_is_training_false': sum(1 for o in val_ops if o.type.startswith('FusedBatchNorm') and
                                                   not o.get_attr('is_training')),
             'val_reads_train_iterator': any(o.name == self.train_next[0].op.name for o in val_ops)}
        qual = importlib.util.spec_from_file_location('e03_m6d2b_oracle', ROOT / 'methods/stdn/runtime_qualification.py')
        oracle = importlib.util.module_from_spec(qual)
        qual.loader.exec_module(oracle)
        s['trainable_equals_m6d2b_oracle'] = {v.name: v.shape.as_list() for v in tv} == oracle.expected_variables()
        require(s['trainable_equals_m6d2b_oracle'] and (s['global_variables'], s['trainable_variables'], s['adam_ops'],
                s['bn_moving'], s['weight_ema'], s['loss_ema'], s['variables_created_by_feed'],
                s['variables_created_by_val_graph']) == (747, 170, 170, 60, 170, 2, 0, 0) and
                s['val_step_returns_no_train_op'] and not mutating and s['val_fused_bn_is_training_true'] == 0 and
                not s['val_reads_train_iterator'], 'unchanged official graph structure')
        return s

    # ------------------------------------------------------------- state readers (read-only session runs)
    def fingerprint(self, sess):
        values = sess.run(self.variables)
        per = {v.name: hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest() for v, x in zip(self.variables, values)}
        finite = all(np.isfinite(x).all() for x in values if np.issubdtype(np.asarray(x).dtype, np.floating))
        return {'sha256': hashlib.sha256(json.dumps(per, sort_keys=True).encode()).hexdigest(), 'per_variable': per,
                'all_finite': bool(finite)}

    def counters(self, sess):
        fetch = [self.global_step, self.lr['G'], self.lr['D']] + ([self.max_bytes] if self.max_bytes is not None else [])
        r = sess.run(fetch)
        return {'global_step': int(r[0]), 'lr_G': float(r[1]), 'lr_D': float(r[2]),
                'gpu_memory_bytes': int(r[3]) if self.max_bytes is not None else None}


def write_figure(run_dir, name, image):
    import cv2
    ok, buf = cv2.imencode('.png', image)              # identical encoder to the official cv2.imwrite
    require(ok, 'figure encode')
    path = run_dir / 'diagnostics' / 'figures' / name
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(path, buf.tobytes())
    return 'diagnostics/figures/' + name


def write_checkpoint(ctx, ctl, sess, epoch, *, terminal_semantics_probe=False):
    """Official saver.save(sess, <dir>/ckpt, global_step=epoch); one index record per physical file."""
    tf = ctl.tf
    before = ctl.counters(sess)
    if terminal_semantics_probe:
        require(ctx.mode == pio.QUALIFICATION and ctx.case == 'terminal_probe' and epoch == pio.MAX_EPOCH,
                'the ckpt-50 semantics probe is qualification-only')
    ckpt_dir = ctx.run_dir / 'checkpoints'
    prefix = ckpt_dir / pio.checkpoint_prefix(epoch)
    require(not any(ckpt_dir.glob(prefix.name + '.*')), 'checkpoint path is fresh')
    written = ctl.saver.save(sess, str(ckpt_dir / 'ckpt'), global_step=epoch)
    require(written == str(prefix), 'official checkpoint prefix')
    after = ctl.counters(sess)
    require((before['global_step'], before['lr_G'], before['lr_D']) == (after['global_step'], after['lr_G'],
                                                                       after['lr_D']), 'checkpointing changes no state')
    reader = tf.train.NewCheckpointReader(str(prefix))
    shapes = reader.get_variable_to_shape_map()
    names = {v.op.name for v in ctl.variables}
    require(set(shapes) == names, 'checkpoint holds every global variable')
    live = dict(zip((v.op.name for v in ctl.variables), sess.run(ctl.variables)))
    stored = {n: reader.get_tensor(n) for n in names}
    equal = sum(np.array_equal(stored[n], live[n]) for n in names)
    finite = all(np.isfinite(x).all() for x in stored.values() if np.issubdtype(np.asarray(x).dtype, np.floating))
    require(equal == len(names) and finite, 'checkpoint round trip equals live state and is finite')
    kind, selected, reason = pio.checkpoint_kind(epoch)
    files = []
    for suffix in pio.CHECKPOINT_SUFFIXES:
        p = Path(str(prefix) + suffix)
        meta = dict(path='checkpoints/' + p.name, epoch=epoch, global_step=after['global_step'],
                    file_size_bytes=p.stat().st_size, sha256=pio.sha256_file(p), checkpoint_type=kind,
                    selected_for_final=selected, selection_reason=reason)
        ctx.record_checkpoint(**meta)
        files.append(meta)
    event = {'epoch': epoch, 'prefix': 'checkpoints/' + prefix.name, 'files': files, 'variables': len(names),
             'round_trip_equal': equal, 'all_finite': bool(finite), 'global_step': after['global_step'],
             'state_unchanged_by_save': True, 'terminal_semantics_probe': bool(terminal_semantics_probe),
             'labels': ctx.labels(), 'resume_sidecar': None, 'checkpoint_type': kind, 'selected_for_final': selected}
    ctx.log_event('e03_checkpoint_event', event)
    return event


def _losses(values):
    return {'g_loss': float(values[0]), 'd_loss': float(values[1]), 'a_loss': float(values[2])}


def run_plan(ctx, ctl, sess, plan, *, on_point=None, on_step=None):
    """Execute epochs [(train_steps, 'checkpoint' | 'probe50' | None, val_steps)]. SCIENTIFIC == SCIENTIFIC_PLAN.

    on_point / on_step are qualification observers (read-only state fingerprints); SCIENTIFIC passes neither.
    """
    if ctx.mode == pio.SCIENTIFIC:
        require(tuple(plan) == SCIENTIFIC_PLAN and on_point is None and on_step is None,
                'scientific runs execute exactly 50 x (2000 TRAIN, ckpt, 500 VAL) with no observer')
    t0 = time.monotonic()
    nan_reason = {'train_metrics': 'STDN exposes no TRAIN metric beyond its g/d/a losses; none is fabricated.',
                  'val_metrics': 'STDN exposes no VAL metric beyond its g/d/a losses; none is fabricated.'}
    apps = {'G': 0, 'D': 0}
    state = ctl.counters(sess)
    require(state['global_step'] == 0, 'fresh global_step')
    train_total = 0
    checkpoints = []
    for e, (n_train, ckpt, n_val) in enumerate(plan):
        for step in range(n_train):
            gd = pio.is_gd_step(step)
            fetch = ctl.losses + ([ctl.g_op, ctl.d_op, ctl.fig] if gd else [ctl.g_op, ctl.fig]) + list(ctl.train_next[2:])
            ts = time.monotonic()
            r = sess.run(fetch)
            dt = time.monotonic() - ts
            fig, idx, flips, delta = r[-4], r[-3], r[-2], r[-1]
            losses = _losses(r[:3])
            require(all(np.isfinite(v) for v in losses.values()), 'non-finite STDN loss')
            apps['G'] += 1
            apps['D'] += int(gd)
            after = ctl.counters(sess)
            require(after['global_step'] - state['global_step'] == (2 if gd else 1) and
                    after['global_step'] == apps['G'] + apps['D'], 'shared global_step +2 (G+D) / +1 (G)')
            figure = write_figure(ctx.run_dir, pio.train_figure_name(e, step), fig) if pio.is_train_figure_step(step) \
                else None
            ctx.log_epoch({
                'epoch': e + 1, 'record_kind': 'TRAIN_STEP', 'train_step_in_epoch': step, 'train_step_total': train_total,
                'global_step': after['global_step'], 'shared_global_step_before': state['global_step'],
                'shared_global_step_after': after['global_step'],
                'optimizer_ops_applied': ['g_op', 'd_op'] if gd else ['g_op'],
                'applications': dict(apps), 'learning_rate': state['lr_G'],
                'learning_rates': {'G': state['lr_G'], 'D': state['lr_D']},
                'learning_rate_semantics': 'exponential_decay read from the shared global_step before this run',
                'train_losses': losses, 'train_metrics': None, 'val_losses': None, 'val_metrics': None,
                'pair_rows': [int(x) for x in idx],
                'pair_ids': [ctl.pack.rows['TRAIN'][int(x)][0] for x in idx],
                'flips_live_spoof': [[int(a), int(b)] for a, b in flips],
                'brightness_delta_live_spoof': [[float(a), float(b)] for a, b in delta],
                'figure': figure, 'step_seconds': dt, 'wall_clock_seconds': time.monotonic() - t0,
                'gpu_memory_bytes': after['gpu_memory_bytes'],
                'missing_field_reasons': dict(nan_reason, val_losses='TRAIN step; VAL losses are logged on VAL records',
                                              **({'gpu_memory_bytes': 'tf.contrib.memory_stats unavailable'}
                                                 if after['gpu_memory_bytes'] is None else {}))})
            state = after
            train_total += 1
            if on_step:
                on_step('train', e, step, sess, gd)
        if ckpt:
            if on_point:
                on_point('before_checkpoint', e, sess)
            checkpoints.append(write_checkpoint(ctx, ctl, sess, pio.MAX_EPOCH if ckpt == 'probe50' else e + 1,
                                                terminal_semantics_probe=ckpt == 'probe50'))
        if on_point:
            on_point('before_val', e, sess)
        for step in range(n_val):
            ts = time.monotonic()
            r = sess.run(ctl.losses_val + [ctl.fig_val] + list(ctl.val_next[2:]))
            dt = time.monotonic() - ts
            fig, idx, flips = r[3], r[4], r[5]
            losses = _losses(r[:3])
            require(all(np.isfinite(v) for v in losses.values()), 'non-finite STDN VAL loss')
            after = ctl.counters(sess)
            require(after['global_step'] == state['global_step'], 'VAL never changes the shared global_step')
            figure = write_figure(ctx.run_dir, pio.val_figure_name(e, step), fig) if pio.is_val_figure_step(step) \
                else None
            ctx.log_epoch({
                'epoch': e + 1, 'record_kind': 'VAL_STEP', 'val_step_in_epoch': step, 'global_step': after['global_step'],
                'learning_rate': state['lr_G'], 'train_losses': None, 'train_metrics': None, 'val_losses': losses,
                'val_metrics': None, 'val_role': 'DIAGNOSTICS_ONLY_NEVER_SELECTS',
                'pair_rows': [int(x) for x in idx], 'pair_ids': [ctl.pack.rows['VAL'][int(x)][0] for x in idx],
                'flips_live_spoof': [[int(a), int(b)] for a, b in flips], 'figure': figure, 'step_seconds': dt,
                'wall_clock_seconds': time.monotonic() - t0, 'gpu_memory_bytes': after['gpu_memory_bytes'],
                'missing_field_reasons': dict(nan_reason, train_losses='VAL step; no optimizer is applied',
                                              **({'gpu_memory_bytes': 'tf.contrib.memory_stats unavailable'}
                                                 if after['gpu_memory_bytes'] is None else {}))})
            if on_step:
                on_step('val', e, step, sess, False)
        if on_point:
            on_point('after_val', e, sess)
    return {'applications': apps, 'global_step': state['global_step'], 'train_steps': train_total,
            'checkpoints': checkpoints, 'seconds': time.monotonic() - t0}


def prepare(tf, mode, seed, runtime_root, faces_root, *, keep_parser_log=False):
    """Contract, source, environment and pack verification, then the graph. Creates no run root."""
    cfg = pio.verify_contract()
    source = pio.verify_source()
    ids = pio.identities(cfg, source)
    live, driver = verify_environment(tf)
    pack = feed_pack.load(pio.pack_root(runtime_root), pio.PACK_MANIFEST_SHA256)
    ctl = Controller(tf, mode=mode, seed=seed, pack=pack, faces_root=faces_root, keep_parser_log=keep_parser_log)
    env, reasons = environment_record(tf, live)
    return ctl, ids, live, driver, env, reasons


def run_scientific(*, seed, runtime_root, faces_root, command_line=None):
    """Full 50-epoch SCIENTIFIC run, fresh only. Never called by the qualification harness."""
    import tensorflow as tf
    from methods.common.runlog import git_dirty
    mode = pio.SCIENTIFIC
    pio.validate_mode_seed(mode, seed)
    require(not git_dirty(), 'SCIENTIFIC runs require a clean git worktree')
    errors = pio.launch_environment_errors(os.environ, seed)
    require(not errors, 'launch environment: ' + ', '.join(errors))
    target = pio.run_root(runtime_root, mode, seed)
    require(not target.exists(), '%s exists; E03 scientific runs are fresh only (resume is not qualified)' % target)
    ctl, ids, live, driver, env, reasons = prepare(tf, mode, seed, runtime_root, faces_root)
    ctx = E03RunContext(mode=mode, seed=seed, runtime_root=runtime_root, environment=env,
                        missing_environment_reasons=reasons, identities=ids, command_line=command_line)
    with ctx:
        streams = (sys.stdout, sys.stderr)
        sys.stdout, sys.stderr = Tee(sys.stdout, ctx.path('stdout')), Tee(sys.stderr, ctx.path('stderr'))
        try:
            ctx.log_event('e03_run_start', {
                'seeding': ctl.seeding, 'identities': ids, 'structure': ctl.structure, 'gpu_driver': driver,
                'environment_identity_equals_lock_except_driver': True, 'fidelity': pio.FIDELITY,
                'python38_pathlib_is_relative_to_backport': PY38_PATHLIB_BACKPORT,
                'pack': {'manifest_sha256': ctl.pack.manifest_sha256, 'counts': ctl.pack.manifest['counts']},
                'session_config': live['session_config']})
            with tf.Session(config=session_config(tf)) as sess:
                sess.run(tf.initializers.global_variables())     # train.py:153 (fresh; no Saver.restore)
                result = run_plan(ctx, ctl, sess, SCIENTIFIC_PLAN)
            require(result['global_step'] == pio.FINAL_GLOBAL_STEP and result['applications'] ==
                    {'G': 100000, 'D': 50000}, 'final shared global_step 150000, G 100000, D 50000')
            final = [f for c in result['checkpoints'] for f in c['files'] if f['selected_for_final']]
            data = next(f for f in final if f['path'].endswith('.data-00000-of-00001'))
            ctx.close(completion_status='completed', summary={
                'final_or_selected_checkpoint_path': str(ctx.run_dir / 'checkpoints' / pio.checkpoint_prefix(50)),
                'final_or_selected_checkpoint_sha256': data['sha256'],
                'final_checkpoint_files': final, 'checkpoint_selection_rule': pio.CHECKPOINT_RULE,
                'training_duration_seconds': result['seconds'], 'shared_global_step': result['global_step'],
                'optimizer_applications': result['applications'], 'train_steps': result['train_steps'],
                'val_steps': pio.MAX_EPOCH * pio.VAL_STEPS, 'face_reads': ctl.reader.reads,
                'missing_field_reasons': {'seed_level_evaluation_metrics': 'Downstream evaluation is a later milestone',
                                          'peak_vram_bytes': 'per-step tf.contrib MaxBytesInUse is in metrics.jsonl'}})
        finally:
            sys.stdout, sys.stderr = streams
    return ctx.run_dir
