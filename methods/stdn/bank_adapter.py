"""E03 STDN inference-only bank adapter (TF1, Python 3.8; learned_bank_generation_contract.yaml E03-Q1/Q2/Q3/Q5, Q4).

Graph (built once, batch 1, from the hash-verified pinned source c79f1f8; no line of STDN is patched):
  img       = source spoof RGB256 / 255                     placeholder [1,256,256,3]   (dataset.py:144)
  img_live  = target live RGB256 / 255                      placeholder [1,256,256,3]
  reg       = generate_offset_map(lm_sp, lm_li)             placeholder [1,256,256,3]   (dataset.py:142, numpy)
  M, s, b, C, T = Gen(img, training_nn=False, scope='STDN')                             (test.py:55, E03-Q1)
  recon1     = (1-s)*img - b - tf.image.resize_images(C, [imsize, imsize]) - T           (train.py:47)
  trace      = img - recon1                                                              (train.py:48)
  trace_warp = warping(trace, reg, imsize)          (train.py:49; `trace[bsize:]` is the spoof half of the training
                                                     batch [live; spoof], which here is the whole spoof-only batch)
  synth1     = img_live + trace_warp                (train.py:50; `img[:bsize]` is the live half)
  pixels     = tf.cast(tf.clip_by_value(synth1, 0.0, 1.0) * 255, tf.uint8)               (utils.py:24, 37; Q4)
Restore: tf.train.Saver(max_to_keep=50,) over the graph's global variables, which are exactly the raw STDN/*
variables and BN moving statistics (no EMA shadow exists in this graph; E03-Q2, test.py:58). With training_nn=False
the official layers create the conv weights/biases as non-trainable variables; they keep the checkpoint's raw names
(134 global variables: 44 conv weights/biases, 54 trainable BN parameters, 36 moving statistics), exactly as in the
official test graph; the 170 ExponentialMovingAverage shadows in ckpt-50 are never restored.
No brightness, no flip (E03-Q3); landmarks are the frozen e03_feed_pack_v1 landmarks_px256 through the unchanged
LandmarkAdapter.from_cache(flip=False) (/256); faces decode as in the E03 FaceReader (PIL, RGB 256x256, resize no-op).
Only the synthetic spoof image is returned (E03-Q5).
"""
from __future__ import annotations

import importlib
import io
import json
import sys
from pathlib import Path

import numpy as np

from methods.common import learned_bank as lb

IMSIZE = 256
SYNTHESIS_LINES = {  # pinned train.py text the graph reproduces (checked byte-exact by verify_synthesis_lines)
    44: "  M, s, b, C, T = Gen(img, training_nn=training_nn, scope='STDN')",
    47: '  recon1 = (1-s)*img - b - tf.image.resize_images(C, [imsize, imsize]) - T',
    48: '  trace = img - recon1',
    49: '  trace_warp = warping(trace[bsize:,...], reg, imsize)',
    50: '  synth1 = img[:bsize,...]+ trace_warp',
}
CONVERSION_LINES = {('model/utils.py', 24): '        fig = tf.clip_by_value(fig, 0.0, 1.0)',
                    ('model/utils.py', 37): '    img = tf.cast(column * 255, tf.uint8)',
                    ('test.py', 55): '  _M, _s, _b, _C, _T, _imname = _step(config, dataset_test, False)',
                    ('test.py', 58): '  saver = tf.train.Saver(max_to_keep=50,)',
                    ('model/dataset.py', 142): '            reg_map_sp = generate_offset_map(lm_sp, lm_li)'}
MUTATING_OP_PREFIXES = ('Assign', 'Apply', 'ResourceApply', 'Scatter', 'ResourceScatter')
RANDOM_OP_TYPES = ('RandomUniform', 'RandomStandardNormal', 'TruncatedNormal', 'RandomShuffle', 'Multinomial',
                   'RandomUniformInt', 'StatelessRandomUniform', 'PyFunc', 'EagerPyFunc')


def require(ok, message):
    if not ok:
        raise lb.LearnedBankError('E03 bank: ' + message)


def verify_synthesis_lines(repo_root=None):
    src = lb.verify_pinned_sources('STDN', repo_root)
    root = Path(src['root'])
    train = (root / 'train.py').read_text().splitlines()
    for line, text in SYNTHESIS_LINES.items():
        require(train[line - 1] == text, 'train.py:%d differs from the frozen synthesis line' % line)
    for (rel, line), text in CONVERSION_LINES.items():
        require((root / rel).read_text().splitlines()[line - 1] == text, '%s:%d differs' % (rel, line))
    return src


def _import_official(source_root):
    root = str(Path(source_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    model = importlib.import_module('model.model')
    warp = importlib.import_module('model.warp')
    for mod in (model, warp):
        require(Path(mod.__file__).resolve().parent == Path(root) / 'model', 'official module from pinned source')
    return model, warp


class STDNGraph:
    def __init__(self, tf, source_root):
        self.tf = tf
        model, self.warp = _import_official(source_root)
        imsize = IMSIZE
        self.img = tf.placeholder(tf.float32, [1, imsize, imsize, 3], name='bank_source_spoof')
        self.img_live = tf.placeholder(tf.float32, [1, imsize, imsize, 3], name='bank_target_live')
        self.reg = tf.placeholder(tf.float32, [1, imsize, imsize, 3], name='bank_offset_map')
        img, reg = self.img, self.reg
        M, s, b, C, T = model.Gen(img, training_nn=False, scope='STDN')
        recon1 = (1-s)*img - b - tf.image.resize_images(C, [imsize, imsize]) - T
        trace = img - recon1
        trace_warp = self.warp.warping(trace, reg, imsize)
        self.synth1 = self.img_live + trace_warp
        fig = tf.clip_by_value(self.synth1, 0.0, 1.0)
        self.pixels = tf.cast(fig * 255, tf.uint8)
        self.variables = tf.global_variables()
        self.saver = tf.train.Saver(max_to_keep=50,)
        self.structure = self._structure()

    def _structure(self):
        tf = self.tf
        names = [v.name for v in self.variables]
        require(names and all(n.startswith('STDN/') for n in names), 'graph variables must all be STDN/*')
        require(not any('ExponentialMovingAverage' in n or '/avg' in n for n in names), 'no EMA variables')
        require(not tf.get_collection(tf.GraphKeys.UPDATE_OPS), 'no BN update ops in test mode')
        seen, stack = set(), [self.synth1.op, self.pixels.op]
        while stack:
            op = stack.pop()
            if op.name in seen:
                continue
            seen.add(op.name)
            stack.extend(x.op for x in op.inputs)
            stack.extend(op.control_inputs)
        graph = tf.get_default_graph()
        types = sorted({graph.get_operation_by_name(n).type for n in seen})
        mutating = [t for t in types if t.startswith(MUTATING_OP_PREFIXES)]
        random_ops = [t for t in types if t in RANDOM_OP_TYPES]
        require(not mutating and not random_ops, 'inference graph has mutating %s / random %s ops'
                % (mutating, random_ops))
        trainable = tf.trainable_variables()
        return {'global_variables': len(names), 'trainable_variables': len(trainable),
                'bn_moving_statistics': sum('moving_' in n for n in names), 'ema_variables': 0,
                'update_ops': 0, 'reachable_ops': len(seen), 'mutating_reachable_ops': mutating,
                'random_reachable_ops': random_ops, 'training_nn': False, 'batch_size': 1,
                'variable_names_sha256': lb.sha256_bytes('\n'.join(sorted(names)).encode())}

    def checkpoint_coverage(self, prefix):
        """Every graph variable exists in the checkpoint with the same shape (index read only; no tensor data)."""
        listed = dict(self.tf.train.list_variables(str(prefix)))
        missing = [v.name for v in self.variables if v.op.name not in listed]
        shape = [v.name for v in self.variables if v.op.name in listed
                 and list(listed[v.op.name]) != v.shape.as_list()]
        require(not missing and not shape, 'checkpoint lacks %s / shape mismatch %s' % (missing[:3], shape[:3]))
        return {'checkpoint_variables': len(listed), 'graph_variables_covered': len(self.variables),
                'ema_in_checkpoint': sum('ExponentialMovingAverage' in k for k in listed),
                'ema_restored': 0}


def decode_face(raw):
    """methods.stdn.production_runner.FaceReader + SplitParser._side decoding: PIL, RGB 256x256, resize (no-op at 256),
    float32 / 255."""
    from PIL import Image
    image = Image.open(io.BytesIO(raw))
    require(image.mode == 'RGB' and image.size == (IMSIZE, IMSIZE), 'canonical RGB256 face')
    arr = np.array(image.resize((IMSIZE, IMSIZE)), np.float32)
    return np.array(arr, np.float32) / 255


class STDNBankAdapter:
    def __init__(self, tf, sess, graph, landmarks, landmark_source, seed, *, is_fixture, checkpoint_identity_sha256,
                 contract_sha256, config_sha256, identity_extra):
        self.tf, self.sess, self.graph, self.landmarks, self.landmark_source = tf, sess, graph, landmarks, \
            landmark_source
        self.method_id, self.bank_seed, self.generator_seed = 'E03', seed, seed
        self.is_fixture = is_fixture
        self.checkpoint_identity_sha256 = checkpoint_identity_sha256
        self.contract_sha256, self.config_sha256 = contract_sha256, config_sha256
        self._identity = dict(identity_extra, graph=graph.structure)
        self.forwards = 0

    def identity(self):
        return dict(self._identity, method_id='E03', bank_seed=self.bank_seed, generator_seed=self.generator_seed,
                    is_fixture=self.is_fixture, checkpoint_identity_sha256=self.checkpoint_identity_sha256,
                    config_sha256=self.config_sha256, contract_sha256=self.contract_sha256)

    def points(self, sample_id):
        return self.landmarks.from_cache({'landmarks_px256': self.landmark_source(sample_id)}, flip=False)

    def synthesize(self, live, spoof, lm_li, lm_sp):
        reg = self.graph.warp.generate_offset_map(lm_sp, lm_li).astype(np.float32)
        synth, pixels = self.sess.run([self.graph.synth1, self.graph.pixels], feed_dict={
            self.graph.img: spoof[None], self.graph.img_live: live[None], self.graph.reg: reg[None]})
        self.forwards += 1
        return synth[0], pixels[0]

    def generate(self, population, reader):
        require(population.method_id == 'E03' and population.seed == self.bank_seed, 'population identity')
        require(reader is not None, 'E03 needs the TRAIN face reader')
        for row in population.rows:
            raw_li, sha_li = reader.read(row.target_live_id)             # live first, as in the source parser
            raw_sp, sha_sp = reader.read(row.source_spoof_id)
            synth, pixels = self.synthesize(decode_face(raw_li), decode_face(raw_sp),
                                            self.points(row.target_live_id), self.points(row.source_spoof_id))
            finite = bool(np.isfinite(synth).all())
            yield lb.GenResult(row=row, pixels=pixels if finite else None, raw_finite=finite,
                               raw_min=float(synth.min()) if finite else None,
                               raw_max=float(synth.max()) if finite else None,
                               failure_reason=None if finite else 'NONFINITE_GENERATOR_OUTPUT', noise_seed=None,
                               target_face_sha256=sha_li, source_face_sha256=sha_sp)

    def finalize(self):
        return {'forwards': self.forwards, 'batch_size': 1, 'stochastic_at_inference': False,
                'trace_sidecar_written': False}


# ------------------------------------------------------------------ TRAIN-only feed-pack access
def load_train_pack(pack_dir, expected_manifest_sha256):
    """The frozen e03_feed_pack_v1, TRAIN side only: manifest SHA256; payload hashes of samples.json, landmarks and
    pairs_train.json (pairs_val.json is never opened); TRAIN pair order hash; per-row landmark hashes. Sample metadata
    rows of other splits inside samples.json are dropped before any face path exists."""
    from methods.stdn import feed_pack as fp
    root = Path(pack_dir)
    raw_manifest = (root / fp.MANIFEST).read_bytes()
    require(lb.sha256_bytes(raw_manifest) == expected_manifest_sha256, 'feed pack manifest sha256')
    m = json.loads(raw_manifest.decode('utf-8'))
    require(m.get('schema') == fp.SCHEMA and m.get('version') == fp.VERSION and m.get('test_rows') == 0, 'pack schema')
    raw = {}
    for name in ('samples.json', 'landmarks_px256.f32le', 'pairs_train.json'):
        raw[name] = (root / name).read_bytes()
        require(len(raw[name]) == m['payloads'][name]['bytes']
                and lb.sha256_bytes(raw[name]) == m['payloads'][name]['sha256'], 'payload integrity ' + name)
    s = json.loads(raw['samples.json'].decode('utf-8'))
    require(s['columns'] == list(fp.SAMPLE_COLUMNS), 'samples columns')
    n = len(s['rows'])
    lm = np.frombuffer(raw['landmarks_px256.f32le'], dtype=fp.LANDMARK_DTYPE)
    require(lm.size == n * 68 * 2, 'landmark payload size')
    lm = lm.reshape((n,) + fp.LANDMARK_SHAPE)
    train = {}
    for i, r in enumerate(s['rows']):
        rec = dict(zip(fp.SAMPLE_COLUMNS, r))
        require(rec['landmark_row'] == i and rec['split'] in ('TRAIN', 'VAL'), 'sample row')
        if rec['split'] != 'TRAIN':
            continue
        require(lb.sha256_bytes(fp.landmark_bytes(lm[i])) == rec['landmarks_le_f4_sha256'], 'landmark row')
        train[rec['sample_id']] = {'split': 'TRAIN', 'dataset': rec['dataset'],
                                   'face_png_sha256': rec['face_png_sha256'], 'label_binary': rec['label_binary'],
                                   'landmarks_px256': lm[i].copy()}
    d = json.loads(raw['pairs_train.json'].decode('utf-8'))
    require(d['columns'] == list(fp.PAIR_COLUMNS) and d['split'] == 'TRAIN', 'pairs_train header')
    require(lb.sha256_bytes('\n'.join(r[0] for r in d['rows']).encode()) == m['pair_id_order_sha256']['TRAIN'],
            'TRAIN pair order')
    relation = []
    for pid, live, spoof, ds in d['rows']:
        require(live in train and spoof in train and train[live]['label_binary'] == 0
                and train[spoof]['label_binary'] == 1 and train[live]['dataset'] == train[spoof]['dataset'] == ds,
                'TRAIN pair ' + pid)
        relation.append((pid, live, spoof, ds))
    require(len(relation) == lb.N_SYN, 'TRAIN relation size')
    return {'relation': relation, 'samples': train, 'manifest_sha256': expected_manifest_sha256,
            'payloads_read': sorted(raw), 'pairs_val_opened': False}


# ------------------------------------------------------------------ constructors
def precision_environment_errors(environ, *, gpu):
    from methods.stdn import production_io as pio
    want = dict(pio.PRECISION_ENV)
    if not gpu:
        want['CUDA_VISIBLE_DEVICES'] = ''
    return sorted(k for k, v in want.items() if environ.get(k) != v)


def make_session(tf):
    from methods.stdn.production_runner import session_config
    return tf.Session(config=session_config(tf))


def make_landmarks():
    from methods.stdn.production_runner import make_landmark_adapter
    return make_landmark_adapter()


def load_real(tf, seed, run_root, landmark_source, *, repo_root=None):
    """S7 / production (GPU): graph, raw-variable restore of the approved ckpt-50 of `seed`."""
    import os
    _, contract_sha = lb.load_generation_contract()
    cfg = lb.verify_config_identity('E03')
    src = verify_synthesis_lines(repo_root)
    errs = precision_environment_errors(os.environ, gpu=True)
    require(not errs, 'E03 precision environment differs: %s' % errs)
    ckpt = lb.verify_checkpoint('E03', seed, run_root)
    graph = STDNGraph(tf, src['root'])
    prefix = Path(run_root) / 'checkpoints' / 'ckpt-50'
    coverage = graph.checkpoint_coverage(prefix)
    sess = make_session(tf)
    graph.saver.restore(sess, str(prefix))
    tf.get_default_graph().finalize()
    adapter = STDNBankAdapter(tf, sess, graph, make_landmarks(), landmark_source, seed, is_fixture=False,
                              checkpoint_identity_sha256=ckpt['checkpoint_identity_sha256'],
                              contract_sha256=contract_sha, config_sha256=cfg['sha256'],
                              identity_extra={'checkpoint': ckpt, 'source': src, 'coverage': coverage,
                                              'framework': 'tensorflow ' + tf.__version__})
    return adapter, ckpt
