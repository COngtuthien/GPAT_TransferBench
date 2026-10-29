"""E04 Physics-STD production controller (Python 3.8, NVIDIA TF 1.15.5, gpat-m6-e03; CONTROLLED_ADAPTATION).

Unchanged and imported (never re-implemented): methods/physics_std/training_graph.TrainingGraph (M6D3d contract,
M6D3e realization: generator, four discriminators, losses Eq.23 / L_D / Eq.24, two Adam instances with shared G slots,
phase guards G -> D -> G, global_iteration +1 after Step 3, lr 5e-5 * 10^-floor(t/45000), D at half) and
trace_transfer.dense_offsets. The per-minibatch execution is exactly the qualified M6D3e worker order: for each step
1, 2, 3, run that step's BN update collections in order (1: original G, D; 2: original G, D; 3: original G, hard G),
then that step's apply. The step's losses are fetched in the apply run itself (pre-apply values, no extra state).
Owner resolution D1-D9 (configs/amendments/e04_production_resolution.yaml):
  D4 pair rows from one RandomState(PAIR_ORDER seed) permutation stream over pairs_train_v1; D7 hard draws from a
  separate RandomState(HARD_SAMPLE seed) before Step 1, reused for Steps 1-3; D8 RGB uint8 / 255 once; D2/D3 geometry
  from the frozen CPU pre-pass via the E04 feed pack (live M0, Q140 xy; spoof depth exact zero); D5 no augmentation;
  D6 no VAL loop (VAL fields null with reason); D9 driver-version-only tolerance on the E04 training lock.
Checkpoints: TF Saver of every global variable every 15000 iterations; iteration_150000 = BASELINE_FINAL_STATE_V1.
Fresh only; no Saver.restore exists in this module.
"""
import ctypes
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import sysconfig
import time
import uuid

import numpy as np


def _is_relative_to(self, *other):
    """CPython 3.9 pathlib.PurePath.is_relative_to, verbatim semantics (the E03 Python 3.8 backport)."""
    try:
        self.relative_to(*other)
        return True
    except ValueError:
        return False


PY38_PATHLIB_BACKPORT = not hasattr(Path, 'is_relative_to')
if PY38_PATHLIB_BACKPORT:
    import pathlib
    pathlib.PurePath.is_relative_to = _is_relative_to

from methods.common.config import load_method_config  # noqa: E402
from methods.common.learned_runlog import LearnedRunContext  # noqa: E402
from methods.common.runlog import (RunContext, RunDirectoryError, atomic_write_json, compute_run_id,  # noqa: E402
                                   git_commit, load_logging_contract)
from methods.physics_std import feed_pack  # noqa: E402
from methods.physics_std import production_io as pio  # noqa: E402

require = pio.require
ROOT = pio.ROOT


# ================================================================= run context (run_logging_v1)
class E04RunContext(LearnedRunContext):
    """SCIENTIFIC: runs/m6/E04/seed_<seed>. QUALIFICATION: <rt>/qualification/e04_production/E04/<case>-q60904-<id>."""

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
        self.source_commits = {}
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
        if self.run_dir.exists():                           # fresh only in both modes (resume is not qualified)
            raise RunDirectoryError('STOP_AND_REPORT: %s exists; E04 runs are fresh only (resume is not qualified)'
                                    % self.run_dir)
        result = super()._open_locked()
        index = json.loads(self.path('checkpoint_index').read_text())
        index.update(note='TF Saver checkpoints actually written (one record per physical file: .data, .index, .meta). '
                          'Periodic every %d iterations; iteration_150000 = BASELINE_FINAL_STATE_V1 (A2-06). '
                          'Each is written after Step 3 of its iteration (phase 0). No resume sidecar.' % pio.CHECKPOINT_EVERY,
                     runner_mode=self.mode, checkpoint_rule=pio.CHECKPOINT_RULE,
                     selected_final=('checkpoints/' + pio.checkpoint_prefix(pio.ITERATIONS)
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
            'scientific_run': self.mode == pio.SCIENTIFIC, 'labels': self.labels(), 'fidelity': pio.FIDELITY,
            'reporting_label': pio.REPORTING_LABEL, 'identities': self.identities,
            'note': 'Verbatim frozen config above; the execution bindings below are the M6D3d/M6D3e realization and '
                    'the owner production resolution D1-D9. No scientific value was overridden.',
            'owner_resolution': {'yaml': pio.RULING_YAML[0], 'yaml_sha256': pio.RULING_YAML[1],
                                 'document': pio.RULING_DOC[0], 'document_sha256': pio.RULING_DOC[1]},
            'rng': pio.rng_plan(self.seed_value),
            'schedule': {'iterations': pio.ITERATIONS, 'batch': '4 target_live + 4 source_spoof',
                         'per_iteration': ['step1: BN(original G, D), apply G Eq.23',
                                           'step2: BN(original G, D), apply D L_D',
                                           'step3: BN(original G, hard G), apply G Eq.24; global_iteration += 1'],
                         'lr': '5e-5 * 10^-floor(t / 45000); D x 0.5', 'checkpoint_every': pio.CHECKPOINT_EVERY},
            'feed': {'relation': pio.PAIRS_TRAIN[0], 'pack': pio.PACK_DIRNAME,
                     'pack_manifest_sha256': pio.PACK_MANIFEST_SHA256,
                     'prepass_manifest_sha256': pio.PREPASS_MANIFEST_SHA256,
                     'encoding': 'RGB uint8 / 255 (D8)', 'augmentation': 'NONE (D5)', 'val': 'NONE (D6)'},
            'checkpoint_rule': {'rule': pio.CHECKPOINT_RULE, 'terminal': pio.TERMINAL_STATE,
                                'selected': 'checkpoints/' + pio.checkpoint_prefix(pio.ITERATIONS)},
            'resume': pio.RESUME})
        return resolved

    def _manifest(self, **kwargs):
        m = super()._manifest(**kwargs)
        m.update(runner_mode=self.mode, scientific_run=self.mode == pio.SCIENTIFIC,
                 qualification_seed=self.qualification_seed, qualification_case=self.case, fidelity=pio.FIDELITY,
                 reporting_label=pio.REPORTING_LABEL, identities=self.identities, resume=pio.RESUME,
                 val_split_accessed=False, val_role='NOT_USED_D6', test_code_path_present=False)
        if self.mode == pio.QUALIFICATION:
            m['labels'] = self.labels()
            m['missing_field_reasons']['experiment_seed'] = ('QUALIFICATION_ONLY: seed %d is not an experiment seed'
                                                             % self.qualification_seed)
        return m

    def record_checkpoint(self, **kwargs):
        """E04 rule (PhysicsSTDAdapter.validate_checkpoint_event semantics; the shared checkpoint_plan has no E04
        branch): iteration <= 150000, selected only at 150000, zero extra optimizer steps."""
        it = kwargs['global_step']
        kind, selected, reason = pio.checkpoint_kind(it)
        require(type(it) is int and 0 < it <= pio.ITERATIONS, 'checkpoint iteration within budget')
        require((kwargs['checkpoint_type'], kwargs['selected_for_final'], kwargs['selection_reason']) ==
                (kind, selected, reason), 'checkpoint semantics equal the E04 rule')
        if self.mode == pio.QUALIFICATION:
            require(selected is (self.case == 'terminal_probe'), 'only the terminal-semantics probe records 150000')
        return RunContext.record_checkpoint(self, **kwargs)


# ================================================================= environment (E04 training lock, D9)
def session_config(tf):
    """The qualified M6D3e session configuration (environments/e04_training.runtime.json)."""
    cfg = tf.ConfigProto(allow_soft_placement=True, intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    cfg.gpu_options.allow_growth = True
    cfg.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.OFF
    cfg.graph_options.rewrite_options.disable_meta_optimizer = True
    cfg.graph_options.rewrite_options.auto_mixed_precision = 2
    if 'auto_mixed_precision_mkl' in cfg.graph_options.rewrite_options.DESCRIPTOR.fields_by_name:
        cfg.graph_options.rewrite_options.auto_mixed_precision_mkl = 2
    return cfg


def live_environment(tf):
    import importlib.metadata as md
    lock = json.loads((ROOT / pio.LOCK[0]).read_text())
    env = Path(sys.executable).parents[1]
    site = env / 'lib/python3.8/site-packages/nvidia'
    cuda = ctypes.CDLL(str(site / 'cuda_runtime/lib/libcudart.so.11.0'))
    cudnn = ctypes.CDLL(str(site / 'cudnn/lib/libcudnn.so.8'))
    version = ctypes.c_int()
    rc = cuda.cudaRuntimeGetVersion(ctypes.byref(version))
    cudnn.cudnnGetVersion.restype = ctypes.c_size_t
    norm = lambda n: n.lower().replace('_', '-')  # noqa: E731
    dists = {norm(d.metadata['Name']): d.version for d in md.distributions(
        path=sorted({sysconfig.get_path('purelib'), sysconfig.get_path('platlib')}))}
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,uuid', '--format=csv,noheader'])
    return {'python': sys.version, 'executable': sys.executable,
            'python_executable_sha256': pio.sha256_file(Path(sys.executable).resolve()),
            'environment_path': str(env), 'tensorflow': tf.__version__,
            'package_versions': {n: dists.get(norm(n)) for n in lock['package_versions']},
            'native_runtime': {'cudaRuntimeGetVersion': version.value, 'cuda_query_rc': rc,
                               'cudnnGetVersion': int(cudnn.cudnnGetVersion())},
            'precision_env': {k: os.environ.get(k) for k in lock['precision_env']},
            'session_config': str(session_config(tf)), 'gpu': gpu.decode().strip()}


def verify_environment(tf):
    lock = json.loads((ROOT / pio.LOCK[0]).read_text())
    record = json.loads((ROOT / pio.RUNTIME_RECORD[0]).read_text())
    live = live_environment(tf)
    require(Path(live['executable']).resolve() == Path(pio.PYTHON_EXECUTABLE).resolve(), 'E04 training Python')
    diff, driver = pio.environment_differences(lock, record, live)
    require(not diff, 'gpat-m6-e03 differs from the E04 training lock: ' + ', '.join(diff))
    return live, driver


def environment_record(tf, live):
    import platform
    import socket
    import getpass
    gpu = live['gpu'].split(', ')
    env = {'host': socket.gethostname(), 'user': getpass.getuser(), 'platform': platform.platform(),
           'python_version': platform.python_version(), 'numpy_version': np.__version__, 'gpu_model': gpu[0],
           'gpu_count': len(tf.config.experimental.list_physical_devices('GPU')),
           'cuda_version': str(live['native_runtime']['cudaRuntimeGetVersion']),
           'cudnn_version': str(live['native_runtime']['cudnnGetVersion']), 'framework': 'tensorflow',
           'framework_version': tf.__version__, 'pytorch_version': None, 'tensorflow_version': tf.__version__,
           'dependency_fingerprint': pio.LOCK[1], 'environment_lock_path': pio.LOCK[0],
           'environment_name': pio.ENVIRONMENT_NAME, 'python_executable': sys.executable,
           'gpu_note': 'single RTX 3090; TF1.15 FP32; M6D3e session/precision layer'}
    return env, {'pytorch_version': 'E04 training is TensorFlow 1.x; PyTorch is used only by the CPU pre-pass env'}


# ================================================================= feed (D4, D7, D8, D2)
class FaceReader:
    """Canonical faces of feed-pack samples only (TRAIN); every PNG is SHA256-checked before decoding."""

    def __init__(self, faces_root, pack):
        self.root, self.pack, self.reads = Path(faces_root), pack, 0

    def __call__(self, sample_id):
        from PIL import Image
        rec = self.pack.sample(sample_id)                     # KeyError for any non-pack sample: no TEST path
        raw = (self.root / rec['dataset'] / (sample_id + '.png')).read_bytes()
        require(hashlib.sha256(raw).hexdigest() == rec['face_png_sha256'], 'canonical face sha256 ' + sample_id)
        image = Image.open(io.BytesIO(raw))
        require(image.mode == 'RGB' and image.size == (pio.IMAGE_SIZE, pio.IMAGE_SIZE), 'canonical RGB256 face')
        self.reads += 1
        return np.asarray(image, dtype=np.uint8)


class Feed:
    def __init__(self, seed, pack, reader):
        from methods.physics_std.trace_transfer import dense_offsets
        self.pack, self.reader, self.dense_offsets = pack, reader, dense_offsets
        self.pairs, self.hard = pio.PairStream(seed), pio.HardStream(seed)

    def next(self):
        rows, passes = self.pairs.next_rows()
        recs = [self.pack.rows[r] for r in rows]                       # (pair_id, live, spoof, dataset)
        live, spoof = [r[1] for r in recs], [r[2] for r in recs]
        rgb = np.stack([self.reader(s) for s in live + spoof]).astype(np.float32) / 255     # D8, once
        depth = np.stack([self.pack.live_m0(s) for s in live] +
                         [np.zeros((pio.MAP_SIZE, pio.MAP_SIZE), np.float32)] * 4)[..., None]
        offsets = np.stack([self.dense_offsets(self.pack.q140_xy(b), self.pack.q140_xy(a)) for a, b in zip(live, spoof)])
        scales, keep = self.hard.next()                                                     # D7, before Step 1
        arrays = {'rgb': rgb, 'depth': depth.astype(np.float32), 'offsets': offsets.astype(np.float32),
                  'hard_scales': scales, 'hard_keep': keep}
        require(rgb.shape == (8, 256, 256, 3) and 0 <= rgb.min() and rgb.max() <= 1, 'rgb [8,256,256,3] in [0,1]')
        require(arrays['depth'].shape == (8, 32, 32, 1) and not arrays['depth'][4:].any() and
                0 <= arrays['depth'].min() and arrays['depth'].max() <= 1, 'depth live [0,1], spoof exact zero')
        require(arrays['offsets'].shape == (4, 256, 256, 2) and np.isfinite(arrays['offsets']).all(), 'offsets')
        info = {'pair_rows': rows, 'pair_ids': [r[0] for r in recs], 'pair_passes': passes,
                'hard_scales': [float(x) for x in scales.ravel()], 'hard_keep': [float(x) for x in keep.ravel()],
                'input_sha256': {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in arrays.items()}}
        return arrays, info


# ================================================================= controller
def construction_signature(graph):
    """sha256 of the serialized GraphDef with per-op RNG seed attributes removed (the seed values follow the run
    seed; everything else must equal the qualified M6D3e construction)."""
    gd = graph.as_graph_def()
    for node in gd.node:
        for attr in ('seed', 'seed2'):
            if attr in node.attr:
                del node.attr[attr]
    return hashlib.sha256(gd.SerializeToString(deterministic=True)).hexdigest(), len(graph.get_operations())


def seed_process(tf, seed):
    random.seed(seed)
    np.random.seed(seed)
    return dict(pio.rng_plan(seed), python_numpy_global=seed)


class Controller:
    def __init__(self, tf, *, mode, seed, pack, faces_root):
        pio.validate_mode_seed(mode, seed)
        self.tf, self.mode, self.seed, self.pack = tf, mode, seed, pack
        self.seeding = seed_process(tf, seed)
        from methods.physics_std.training_graph import TrainingGraph
        graph = tf.get_default_graph()
        require(not graph.get_operations(), 'empty default graph')
        with tf.device('/gpu:0'):                      # the qualified M6D3e construction
            tf.set_random_seed(seed)
            self.model = TrainingGraph(tf, ROOT)
        self.construction = construction_signature(graph)
        self.saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=None)
        self.variables = tf.global_variables()
        self.trainables = tf.trainable_variables()
        try:
            from tensorflow.contrib.memory_stats import MaxBytesInUse
            self.max_bytes = MaxBytesInUse()
        except Exception:                               # pragma: no cover - reported as a null with reason
            self.max_bytes = None
        m = self.model
        self.structure = {'construction_sha256_without_seed_attrs': self.construction[0],
                          'construction_operations': self.construction[1],
                          'trainable_variables': len(self.trainables), 'global_variables': len(self.variables),
                          'G': len(m.G), 'D': len(m.D),
                          'trainable_parameters': int(sum(np.prod(v.shape.as_list()) for v in self.trainables)),
                          'shared_G_slots': m.slot_references[1] == m.slot_references[3],
                          'bn_update_ops': {k: [len(x) for x in v] for k, v in m.bn_paths.items()}}
        require(self.structure['trainable_parameters'] == 1763902 and self.structure['trainable_variables'] == 144 and
                self.structure['shared_G_slots'] and len(m.G) + len(m.D) == 144, 'M6D3e graph inventory')
        self.reader = FaceReader(faces_root, pack)
        self.feed = Feed(seed, pack, self.reader)

    def feed_dict(self, arrays):
        m = self.model
        return {m.rgb: arrays['rgb'], m.depth: arrays['depth'], m.offsets: arrays['offsets'],
                m.hard_scales: arrays['hard_scales'], m.hard_keep: arrays['hard_keep']}

    def counters(self, sess):
        m = self.model
        fetch = [m.iteration, m.phase, m.lr_G, m.lr_D] + ([self.max_bytes] if self.max_bytes is not None else [])
        r = sess.run(fetch)
        return {'iteration': int(r[0]), 'phase': int(r[1]), 'lr_G': float(r[2]), 'lr_D': float(r[3]),
                'gpu_memory_bytes': int(r[4]) if self.max_bytes is not None else None}

    def fingerprint(self, sess):
        values = sess.run(self.variables)
        per = {v.name: hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest() for v, x in zip(self.variables, values)}
        finite = all(np.isfinite(x).all() for x in values if np.issubdtype(np.asarray(x).dtype, np.floating))
        return {'sha256': hashlib.sha256(json.dumps(per, sort_keys=True).encode()).hexdigest(), 'per_variable': per,
                'all_finite': bool(finite)}

    def iteration(self, sess, fd, on_step=None):
        """One complete PhySTD minibatch iteration in the qualified order; returns the per-step losses."""
        m, out = self.model, {}
        for step in (1, 2, 3):
            if on_step:
                on_step('before', step, sess, fd)
            for updates in m.bn_paths[step]:
                sess.run(updates, fd)
            keys = pio.STEP_LOSSES[step]
            r = sess.run([m.apply[step]] + [m.losses[k] for k in keys], fd)
            out['step%d' % step] = {k: float(v) for k, v in zip(keys, r[1:])}
            require(all(np.isfinite(v) for v in out['step%d' % step].values()), 'non-finite step %d loss' % step)
            if on_step:
                on_step('after', step, sess, fd)
        return out


def write_checkpoint(ctx, ctl, sess, iteration, *, terminal_semantics_probe=False):
    tf = ctl.tf
    before = ctl.fingerprint(sess)
    if terminal_semantics_probe:
        require(ctx.mode == pio.QUALIFICATION and ctx.case == 'terminal_probe', 'the 150000 probe is qualification-only')
    label = pio.ITERATIONS if terminal_semantics_probe else iteration
    ckpt_dir = ctx.run_dir / 'checkpoints'
    prefix = ckpt_dir / pio.checkpoint_prefix(label)
    require(not any(ckpt_dir.glob(prefix.name + '.*')), 'checkpoint path is fresh')
    written = ctl.saver.save(sess, str(prefix), write_meta_graph=True)
    require(written == str(prefix), 'checkpoint prefix')
    after = ctl.fingerprint(sess)
    require(before['sha256'] == after['sha256'], 'checkpointing changes no state')
    reader = tf.train.NewCheckpointReader(str(prefix))
    names = {v.op.name for v in ctl.variables}
    require(set(reader.get_variable_to_shape_map()) == names, 'checkpoint holds every global variable')
    live = dict(zip((v.op.name for v in ctl.variables), sess.run(ctl.variables)))
    equal = sum(np.array_equal(reader.get_tensor(n), live[n]) for n in names)
    require(equal == len(names), 'checkpoint round trip equals live state')
    counters = ctl.counters(sess)
    require(counters['iteration'] == iteration and counters['phase'] == 0, 'checkpoint after Step 3 (phase 0)')
    kind, selected, reason = pio.checkpoint_kind(label)
    files = []
    for suffix in pio.CHECKPOINT_SUFFIXES:
        p = Path(str(prefix) + suffix)
        meta = dict(path='checkpoints/' + p.name, epoch=None, global_step=label, file_size_bytes=p.stat().st_size,
                    sha256=pio.sha256_file(p), checkpoint_type=kind, selected_for_final=selected,
                    selection_reason=reason)
        ctx.record_checkpoint(**meta)
        files.append(meta)
    event = {'iteration': iteration, 'label_iteration': label, 'prefix': 'checkpoints/' + prefix.name, 'files': files,
             'variables': len(names), 'round_trip_equal': equal, 'state_sha256': after['sha256'],
             'state_unchanged_by_save': True, 'all_finite': after['all_finite'],
             'terminal_semantics_probe': bool(terminal_semantics_probe), 'labels': ctx.labels(),
             'resume_sidecar': None, 'checkpoint_type': kind, 'selected_for_final': selected}
    ctx.log_event('e04_checkpoint_event', event)
    return event


REASONS = {'epoch': 'PhySTD schedule is iteration-based (150000 complete minibatch iterations); no epoch exists',
           'train_metrics': 'PhySTD exposes no TRAIN metric beyond its losses; none is fabricated',
           'val_losses': pio.VAL_NULL_REASON, 'val_metrics': pio.VAL_NULL_REASON}


def run_plan(ctx, ctl, sess, iterations, checkpoints, *, probe=False, on_step=None, on_iteration=None,
             fail_after=None):
    """SCIENTIFIC: exactly (150000, every 15000, no probe, no observer). QUALIFICATION: labelled shorter plans."""
    if ctx.mode == pio.SCIENTIFIC:
        require((iterations, list(checkpoints), probe, on_step, on_iteration, fail_after) ==
                (pio.ITERATIONS, pio.scientific_checkpoints(), False, None, None, None),
                'scientific runs execute exactly 150000 iterations with the frozen checkpoint cadence')
    t0 = time.monotonic()
    state = ctl.counters(sess)
    require((state['iteration'], state['phase']) == (0, 0), 'fresh global_iteration / phase')
    events = []
    for t in range(iterations):
        arrays, info = ctl.feed.next()
        fd = ctl.feed_dict(arrays)
        exp = pio.expected_learning_rates(t)
        require(abs(state['lr_G'] - exp[0]) <= 1e-6 * exp[0] and abs(state['lr_D'] - exp[1]) <= 1e-6 * exp[1],
                'learning rate schedule (G, D = G / 2) at iteration %d' % t)
        ts = time.monotonic()
        losses = ctl.iteration(sess, fd, on_step=(lambda *a: on_step(t, *a)) if on_step else None)
        dt = time.monotonic() - ts
        after = ctl.counters(sess)
        require((after['iteration'], after['phase']) == (t + 1, 0), 'global_iteration +1 after Step 3, phase 0')
        ctx.log_epoch(dict(info, **{
            'epoch': None, 'global_step': after['iteration'], 'iteration': t, 'learning_rate': state['lr_G'],
            'learning_rates': {'G': state['lr_G'], 'D': state['lr_D']}, 'train_losses': losses, 'train_metrics': None,
            'val_losses': None, 'val_metrics': None, 'step_seconds': dt, 'wall_clock_seconds': time.monotonic() - t0,
            'gpu_memory_bytes': after['gpu_memory_bytes'],
            'missing_field_reasons': dict(REASONS, **({'gpu_memory_bytes': 'tf.contrib.memory_stats unavailable'}
                                                       if after['gpu_memory_bytes'] is None else {}))}))
        if on_iteration:
            on_iteration(t, sess, info, losses)
        state = after
        if fail_after is not None and t + 1 == fail_after:
            raise pio.E04Stop('E04 STOP: QUALIFICATION_INJECTED_FAILURE after iteration %d' % (t + 1))
        if t + 1 in checkpoints:
            events.append(write_checkpoint(ctx, ctl, sess, t + 1, terminal_semantics_probe=probe and
                                           t + 1 == max(checkpoints)))
    return {'iterations': state['iteration'], 'checkpoints': events, 'seconds': time.monotonic() - t0,
            'face_reads': ctl.reader.reads}


def prepare(tf, mode, seed, runtime_root, faces_root):
    """Contract, environment and pack verification, then the graph. Creates no run root."""
    pio.verify_contract()
    ids = pio.identities()
    require(pio.PACK_MANIFEST_SHA256 and pio.PREPASS_MANIFEST_SHA256, 'feed pack and pre-pass manifests are pinned')
    live, driver = verify_environment(tf)
    pack = feed_pack.load(pio.pack_root(runtime_root), pio.PACK_MANIFEST_SHA256)
    require(pack.manifest['prepass_manifest_sha256'] == pio.PREPASS_MANIFEST_SHA256, 'pack -> pre-pass binding')
    ctl = Controller(tf, mode=mode, seed=seed, pack=pack, faces_root=faces_root)
    env, reasons = environment_record(tf, live)
    return ctl, ids, live, driver, env, reasons


def run_start_payload(ctl, ids, driver, live):
    return {'seeding': ctl.seeding, 'identities': ids, 'structure': ctl.structure, 'gpu_driver': driver,
            'environment_identity_equals_lock_except_driver': True, 'fidelity': pio.FIDELITY,
            'python38_pathlib_is_relative_to_backport': PY38_PATHLIB_BACKPORT,
            'pack': {'manifest_sha256': ctl.pack.manifest_sha256, 'counts': ctl.pack.manifest['counts']},
            'session_config': live['session_config']}


def run_scientific(*, seed, runtime_root, faces_root, command_line=None):
    """Full 150000-iteration SCIENTIFIC run, fresh only. Never called by the qualification harness."""
    import tensorflow as tf
    from methods.common.runlog import git_dirty
    mode = pio.SCIENTIFIC
    pio.validate_mode_seed(mode, seed)
    require(not git_dirty(), 'SCIENTIFIC runs require a clean git worktree')
    errors = pio.launch_environment_errors(os.environ, seed)
    require(not errors, 'launch environment: ' + ', '.join(errors))
    target = pio.run_root(runtime_root, mode, seed)
    require(not target.exists(), '%s exists; E04 scientific runs are fresh only' % target)
    ctl, ids, live, driver, env, reasons = prepare(tf, mode, seed, runtime_root, faces_root)
    ctx = E04RunContext(mode=mode, seed=seed, runtime_root=runtime_root, environment=env,
                        missing_environment_reasons=reasons, identities=ids, command_line=command_line)
    with ctx:
        ctx.log_event('e04_run_start', run_start_payload(ctl, ids, driver, live))
        with tf.Session(config=session_config(tf)) as sess:
            sess.run(tf.global_variables_initializer())
            result = run_plan(ctx, ctl, sess, pio.ITERATIONS, pio.scientific_checkpoints())
        require(result['iterations'] == pio.ITERATIONS, 'final global_iteration 150000')
        final = [f for c in result['checkpoints'] for f in c['files'] if f['selected_for_final']]
        data = next(f for f in final if f['path'].endswith('.data-00000-of-00001'))
        ctx.close(completion_status='completed', summary={
            'final_or_selected_checkpoint_path': str(ctx.run_dir / 'checkpoints' / pio.checkpoint_prefix(pio.ITERATIONS)),
            'final_or_selected_checkpoint_sha256': data['sha256'], 'final_checkpoint_files': final,
            'checkpoint_selection_rule': pio.CHECKPOINT_RULE, 'training_duration_seconds': result['seconds'],
            'global_iteration': result['iterations'], 'face_reads': result['face_reads'],
            'missing_field_reasons': {'seed_level_evaluation_metrics': 'Downstream evaluation is a later milestone',
                                      'peak_vram_bytes': 'per-iteration tf.contrib MaxBytesInUse is in metrics.jsonl'}})
    return ctx.run_dir
