"""E04 Physics-STD inference-only bank adapter (TF1, Python 3.8) under configs/amendments/e04_bank_generation_contract.yaml.

Owner-approved decisions (2026-11-25):
  E04-Q1 A_MOVING_STATISTICS_PER_PAIR   the frozen generator topology (network.generator) with batch normalization in
                                        inference mode (training=False, the restored moving statistics), one source
                                        spoof per forward (batch 1); no UPDATE_OPS, no batch statistics
  E04-Q2 A_PAPER_SPOOF_SYNTHESIS        out = (1-P_w)*(I_live + B~_w + C~_w + T_w) + P_w*I_source_w with the operations
                                        and operand order of training_graph.py:32-39 (compose below)
  E04-Q3 A_PREDECESSOR_CLIP_TRUNCATE    (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8); non-finite rows are counted
                                        failures before any conversion
  E04-Q4 A_NONE                         the RGB image only
  E04-Q5 A_GPU_FP32_BITWISE_REPLAY      S7/production on the RTX 3090 in gpat-m6-e03 (E04 training lock, D9 policy)
  E04-Q6 A_ADDITIVE_E04_EXTENSION       methods.common.learned_bank writer/reader, PNG, create-only roots

Graph (built once per seed):
  source  [1,256,256,3]  source spoof, float32(uint8)/255 (D8)        placeholder bank_source_spoof
  live    [1,256,256,3]  target live, float32(uint8)/255 (D8)          placeholder bank_target_live
  offsets [1,256,256,2]  dense_offsets(source_xy=spoof Q140, target_xy=live Q140) (D2, scipy)  placeholder
  g = network.generator(tf, source, InferenceLayers(tf))   B, C, T, P = sigmoid(raw[...,9]), B_low = low32(B), C_low
  synth = compose(tf, g, live, source, offsets)            warp (CPU, bilinear, border clamp) of B_low, C_low, T, P and
                                                           the source; trace_transfer.synthesis(live, w_src, wB+wC+wT, wP)
Restore: tf.train.Saver over exactly the 102 generator variables (72 trainable + 30 BN moving statistics) by their
trained names; optimizer slots, counters and the discriminator are never in the graph. The depth branch and the
reconstruction tensors are built by network.generator but never fetched, so never executed.

Disclosure (contract): training-time synthesis used batch statistics of 8-image batches; bank inference uses the stored
moving statistics (accumulated over original and harder batches). This is a controlled adaptation, not official
Physics-STD test-time behaviour. No line of network.py, trace_transfer.py or training_graph.py is changed.
"""
from __future__ import annotations

import hashlib
import inspect
import io
from pathlib import Path

import numpy as np

from methods.common import learned_bank as lb
from methods.physics_std import network
from methods.physics_std.trace_transfer import dense_offsets, synthesis, warp

METHOD_ID = 'E04'
IMSIZE = 256
Q = 140
FROZEN_FILES = {   # E04 bank-generation contract frozen_inputs_sha256 (byte-unchanged training implementation)
    'methods/physics_std/network.py': '9d691809d487c5c9c27b5c9e08389225f606f8d72f1017d2991cd71e6214da21',
    'methods/physics_std/trace_transfer.py': '8152a7643858cb523344dcfa8fbe1a84168082d545d6ca68f7435cf2a06882cd',
    'methods/physics_std/training_graph.py': 'af17ab2789f203834d9fd9121b357dd3e44e49689fccac6be25024926bf69f51',
}
FROZEN_SYNTHESIS_LINES = {   # training_graph.py lines the composition reproduces (checked byte-exact)
    32: "        b,c,t = g['B_low'][4:],g['C_low'][4:],g['T'][4:]",
    36: "        self.warped = {k: warp(tf, value, self.offsets, 'warp_'+k) for k,value in",
    37: "                       [('B',b),('C',c),('T',t),('P',p),('source',source)]}",
    39: "        self.spoof = synthesis(tf,live,w['source'],w['B']+w['C']+w['T'],w['P'])",
}
PREDECESSOR_CONVERSION_LINES = {('model/utils.py', 24): '        fig = tf.clip_by_value(fig, 0.0, 1.0)',
                                ('model/utils.py', 37): '    img = tf.cast(column * 255, tf.uint8)'}
N_TRAINABLE, N_MOVING, N_FUSED_BN = 72, 30, 15
MUTATING_OP_PREFIXES = ('Assign', 'Apply', 'ResourceApply', 'Scatter', 'ResourceScatter')
RANDOM_OP_TYPES = ('RandomUniform', 'RandomStandardNormal', 'TruncatedNormal', 'RandomShuffle', 'Multinomial',
                   'RandomUniformInt', 'StatelessRandomUniform', 'PyFunc', 'EagerPyFunc')


def require(ok, message):
    if not ok:
        raise lb.LearnedBankError('E04 bank: ' + message)


# ================================================================== frozen-source identity
def verify_frozen_sources(repo_root=None):
    root = Path(repo_root) if repo_root is not None else lb.ROOT
    files = {rel: lb.sha256_file(root / rel) for rel in sorted(FROZEN_FILES)}
    for rel, digest in FROZEN_FILES.items():
        require(files[rel] == digest, '%s changed (%s)' % (rel, files[rel]))
    lines = (root / 'methods/physics_std/training_graph.py').read_text().splitlines()
    for n, text in FROZEN_SYNTHESIS_LINES.items():
        require(lines[n - 1] == text, 'training_graph.py:%d differs from the frozen synthesis line' % n)
    stdn = lb.verify_pinned_sources('STDN', repo_root)
    for (rel, n), text in PREDECESSOR_CONVERSION_LINES.items():
        require((Path(stdn['root']) / rel).read_text().splitlines()[n - 1] == text, 'predecessor %s:%d' % (rel, n))
    return {'files': files, 'synthesis_lines': sorted(FROZEN_SYNTHESIS_LINES),
            'predecessor_reference': {'commit': stdn['commit'], 'lines': ['%s:%d' % k for k in
                                                                          sorted(PREDECESSOR_CONVERSION_LINES)]}}


# ================================================================== inference-mode layers (E04-Q1)
class InferenceLayers(network.Layers):
    """network.Layers with exactly one change: tf.layers.batch_normalization(training=False). Same variable scopes,
    names, shapes, initializers, momentum and epsilon; the attention gate is inherited unchanged."""

    def conv(self, x, channels, name, stride=1, kernel=3, transpose=False, output=False):
        tf = self.tf
        with tf.variable_scope(name):
            cin = x.shape.as_list()[-1]
            shape = [kernel, kernel, channels, cin] if transpose else [kernel, kernel, cin, channels]
            w = tf.get_variable('kernel', shape, dtype=tf.float32,
                                initializer=tf.random_normal_initializer(mean=0., stddev=.02))
            b = tf.get_variable('bias', [channels], dtype=tf.float32, initializer=tf.zeros_initializer())
            self.initializers[w.name] = {'distribution': 'Normal', 'mean': 0., 'stddev': .02}
            self.initializers[b.name] = {'distribution': 'constant', 'value': 0., 'role': 'bias'}
            if transpose:
                static = x.shape.as_list()
                target = [static[0], static[1] * stride, static[2] * stride, channels]
                y = tf.nn.conv2d_transpose(x, w, target, [1, stride, stride, 1], padding='SAME')
                y.set_shape(target)
            else:
                y = tf.nn.conv2d(x, w, [1, stride, stride, 1], padding='SAME')
            y = tf.nn.bias_add(y, b)
            if not output:
                y = tf.layers.batch_normalization(y, momentum=.99, epsilon=1e-5,
                    center=True, scale=True, training=False, fused=True, name='BN')
                scope = tf.get_variable_scope().name
                self.initializers[scope + '/BN/gamma:0'] = {'distribution': 'constant', 'value': 1., 'role': 'BN scale'}
                self.initializers[scope + '/BN/beta:0'] = {'distribution': 'constant', 'value': 0., 'role': 'BN offset'}
                y = tf.nn.leaky_relu(y, alpha=.2, name='leaky_relu')
            return y


def inference_layers_differ_only_in_bn_mode():
    """Source-level proof: InferenceLayers.conv == network.Layers.conv with training=True -> training=False only."""
    frozen = inspect.getsource(network.Layers.conv)
    ours = inspect.getsource(InferenceLayers.conv)
    return frozen.count('training=True') == 1 and ours == frozen.replace('training=True', 'training=False')


# ================================================================== frozen synthesis composition (E04-Q2)
def compose(tf, g, live, source, offsets):
    """training_graph.py:32-39 for an already-sliced decomposition `g` of the source spoof(s): warp B_low, C_low, T, P
    and the source image with the same offsets, then synthesis(live, w_source, w_B + w_C + w_T, w_P)."""
    p = g['P']
    b, c, t = g['B_low'], g['C_low'], g['T']
    warped = {k: warp(tf, value, offsets, 'warp_' + k) for k, value in
              [('B', b), ('C', c), ('T', t), ('P', p), ('source', source)]}
    w = warped
    return synthesis(tf, live, w['source'], w['B'] + w['C'] + w['T'], w['P']), warped


class E04InferenceGraph:
    def __init__(self, tf, *, layers_cls=InferenceLayers):
        require(layers_cls is InferenceLayers, 'bank inference uses InferenceLayers only (E04-Q1)')
        self.tf = tf
        shape = [1, IMSIZE, IMSIZE, 3]
        self.source = tf.placeholder(tf.float32, shape, name='bank_source_spoof')
        self.live = tf.placeholder(tf.float32, shape, name='bank_target_live')
        self.offsets = tf.placeholder(tf.float32, [1, IMSIZE, IMSIZE, 2], name='bank_sampling_offsets')
        self.decomposition = network.generator(tf, self.source, layers_cls(tf))
        self.synth, self.warped = compose(tf, self.decomposition, self.live, self.source, self.offsets)
        self.variables = tf.global_variables()
        self.saver = tf.train.Saver(var_list={v.op.name: v for v in self.variables}, max_to_keep=None)
        self.structure = self._structure()

    def _structure(self):
        tf = self.tf
        graph = tf.get_default_graph()
        names = sorted(v.op.name for v in self.variables)
        trainable = [v for v in tf.trainable_variables()]
        moving = [n for n in names if n.endswith(('/moving_mean', '/moving_variance'))]
        require(all(n.startswith('G/') for n in names), 'graph variables must all be G/*')
        require(not any(n.startswith('D/') or 'Adam' in n or 'ExponentialMovingAverage' in n for n in names),
                'no discriminator, optimizer or EMA variable')
        require(len(trainable) == N_TRAINABLE and len(moving) == N_MOVING and len(names) == N_TRAINABLE + N_MOVING,
                'generator variable counts %d/%d/%d' % (len(trainable), len(moving), len(names)))
        require(not tf.get_collection(tf.GraphKeys.UPDATE_OPS), 'no BN update ops in inference mode')
        bn = [op for op in graph.get_operations() if op.type.startswith('FusedBatchNorm')]
        require(len(bn) == N_FUSED_BN and all(op.get_attr('is_training') is False for op in bn),
                'every FusedBatchNorm must be inference mode')
        reads = []
        for op in bn:
            mean, var = (_variable_name(op.inputs[i]) for i in (3, 4))
            require(mean == op.name.rsplit('/', 1)[0] + '/moving_mean'
                    and var == op.name.rsplit('/', 1)[0] + '/moving_variance', 'BN %s statistics source' % op.name)
            reads.append([op.name, mean, var])
        seen, stack = {}, [self.synth.op]
        while stack:
            op = stack.pop()
            if op.name in seen:
                continue
            seen[op.name] = op
            stack.extend(x.op for x in op.inputs)
            stack.extend(op.control_inputs)
        types = sorted({o.type for o in seen.values()})
        mutating = [t for t in types if t.startswith(MUTATING_OP_PREFIXES)]
        random_ops = [t for t in types if t in RANDOM_OP_TYPES]
        gradients = [o.name for o in graph.get_operations() if o.name.startswith('gradients')]
        require(not mutating and not random_ops and not gradients, 'mutating %s / random %s / gradient ops %s'
                % (mutating, random_ops, gradients[:3]))
        placeholders = sorted(o.name for o in seen.values() if o.type == 'Placeholder')
        require(placeholders == ['bank_sampling_offsets', 'bank_source_spoof', 'bank_target_live'], 'synth inputs')
        reached = sorted({_variable_name(o.outputs[0]) for o in seen.values()
                          if o.type in ('VariableV2', 'VarHandleOp')})
        shapes = {v.op.name: v.shape.as_list() for v in self.variables}
        return {'global_variables': len(names), 'trainable_variables': len(trainable), 'bn_moving_statistics': len(moving),
                'fused_batch_norm': len(bn), 'fused_batch_norm_is_training': False, 'bn_statistics_reads_sha256':
                lb.sha256_bytes(lb.canonical_json(sorted(reads)).encode()), 'update_ops': 0,
                'reachable_ops': len(seen), 'reachable_op_types_sha256': lb.sha256_bytes('\n'.join(types).encode()),
                'reachable_variables': len(reached), 'mutating_reachable_ops': mutating,
                'random_reachable_ops': random_ops, 'gradient_ops': 0, 'batch_size': 1, 'placeholders': placeholders,
                'variable_names_sha256': lb.sha256_bytes('\n'.join(names).encode()),
                'variable_shapes_sha256': lb.sha256_bytes(lb.canonical_json(shapes).encode()),
                'cpu_pinned_reachable_ops': sum(1 for o in seen.values() if 'cpu:0' in o.device.lower())}

    def checkpoint_coverage(self, prefix):
        """Every graph variable exists in the checkpoint with the same shape (index only); returns what is NOT
        restored (optimizer slots, counters, discriminator)."""
        listed = dict(self.tf.train.list_variables(str(prefix)))
        missing = [v.op.name for v in self.variables if v.op.name not in listed]
        shape = [v.op.name for v in self.variables if v.op.name in listed
                 and list(listed[v.op.name]) != v.shape.as_list()]
        require(not missing and not shape, 'checkpoint lacks %s / shape mismatch %s' % (missing[:3], shape[:3]))
        restored = {v.op.name for v in self.variables}
        skipped = sorted(k for k in listed if k not in restored)
        return {'checkpoint_variables': len(listed), 'restored_variables': len(restored),
                'not_restored': len(skipped), 'not_restored_classes': {
                    'discriminator': sum(k.startswith('D/') and 'Adam' not in k for k in skipped),
                    'optimizer_slots': sum('Adam' in k for k in skipped),
                    'other': sum(not k.startswith('D/') and 'Adam' not in k for k in skipped)},
                'restored_names_sha256': lb.sha256_bytes('\n'.join(sorted(restored)).encode())}


def _variable_name(tensor):
    op, n = tensor.op, 0
    while op.type in ('Identity', 'ReadVariableOp') and op.inputs and n < 8:
        op, n = op.inputs[0].op, n + 1
    return op.name if op.type in ('VariableV2', 'VarHandleOp', 'Variable') else None


def moving_statistics_finite(tf, prefix):
    reader = tf.train.NewCheckpointReader(str(prefix))
    names = sorted(k for k in reader.get_variable_to_shape_map()
                   if k.startswith('G/') and k.endswith(('/moving_mean', '/moving_variance')))
    vals = {k: reader.get_tensor(k) for k in names}
    ok = len(names) == N_MOVING and all(np.isfinite(v).all() for v in vals.values()) and \
        all(float(v.min()) > 0 for k, v in vals.items() if k.endswith('moving_variance'))
    return {'ok': bool(ok), 'count': len(names),
            'min_moving_variance': float(min(v.min() for k, v in vals.items() if k.endswith('moving_variance'))),
            'values_sha256': lb.sha256_bytes(b''.join(hashlib.sha256(np.ascontiguousarray(vals[k]).tobytes()).digest()
                                                      for k in names))}


# ================================================================== numpy side (E04-Q3, D8, D2)
def convert_uint8(out):
    """E04-Q3 A_PREDECESSOR_CLIP_TRUNCATE, verbatim; the caller has already rejected non-finite outputs."""
    x = np.asarray(out)
    require(x.dtype == np.float32, 'synthesis output must be float32')
    require(bool(np.isfinite(x).all()), 'non-finite output reached conversion')
    y = np.clip(x, 0.0, 1.0) * 255.0
    require(y.dtype == np.float32, 'conversion left float32 (%s)' % y.dtype)
    return y.astype(np.uint8)


def decode_face(raw):
    """production_runner.FaceReader decoding (PIL, RGB 256x256, uint8) then D8: float32(uint8) / 255 once."""
    from PIL import Image
    image = Image.open(io.BytesIO(raw))
    require(image.mode == 'RGB' and image.size == (IMSIZE, IMSIZE), 'canonical RGB256 face')
    return np.asarray(image, dtype=np.uint8).astype(np.float32) / 255


def offsets_for(q_spoof, q_live):
    """D2: dense_offsets(source_xy = spoof Q140, target_xy = live Q140), float32 [256,256,2] (pinned scipy)."""
    q_spoof, q_live = (np.asarray(q, dtype=np.float32) for q in (q_spoof, q_live))
    require(q_spoof.shape == (Q, 2) and q_live.shape == (Q, 2), 'Q140 [140,2] geometry')
    off = dense_offsets(q_spoof, q_live).astype(np.float32)
    require(off.shape == (IMSIZE, IMSIZE, 2) and bool(np.isfinite(off).all()), 'offsets')
    return off


# ================================================================== adapter
class E04BankAdapter:
    def __init__(self, tf, sess, graph, q140_source, seed, *, is_fixture, checkpoint_identity_sha256,
                 contract_sha256, config_sha256, identity_extra):
        self.tf, self.sess, self.graph, self.q140_source = tf, sess, graph, q140_source
        self.method_id, self.bank_seed, self.generator_seed = METHOD_ID, seed, seed
        self.is_fixture = is_fixture
        self.checkpoint_identity_sha256 = checkpoint_identity_sha256
        self.contract_sha256, self.config_sha256 = contract_sha256, config_sha256
        self._identity = dict(identity_extra, graph=getattr(graph, 'structure', None))
        self.forwards = 0

    def identity(self):
        return dict(self._identity, method_id=METHOD_ID, bank_seed=self.bank_seed, generator_seed=self.generator_seed,
                    is_fixture=self.is_fixture, checkpoint_identity_sha256=self.checkpoint_identity_sha256,
                    config_sha256=self.config_sha256, contract_sha256=self.contract_sha256,
                    decisions=dict(lb.E04_APPROVED_DECISIONS), conversion_rule=lb.CONVERSION_RULES[METHOD_ID])

    def synthesize(self, live, spoof, q_live, q_spoof):
        off = offsets_for(q_spoof, q_live)
        synth = self.sess.run(self.graph.synth, feed_dict={
            self.graph.source: spoof[None], self.graph.live: live[None], self.graph.offsets: off[None]})
        self.forwards += 1
        return synth[0]

    def generate(self, population, reader):
        require(population.method_id == METHOD_ID and population.seed == self.bank_seed, 'population identity')
        require(reader is not None, 'E04 needs the TRAIN face reader')
        for row in population.rows:
            raw_li, sha_li = reader.read(row.target_live_id)
            raw_sp, sha_sp = reader.read(row.source_spoof_id)
            synth = self.synthesize(decode_face(raw_li), decode_face(raw_sp), self.q140_source(row.target_live_id),
                                    self.q140_source(row.source_spoof_id))
            finite = bool(np.isfinite(synth).all())
            yield lb.GenResult(row=row, pixels=convert_uint8(synth) if finite else None, raw_finite=finite,
                               raw_min=float(synth.min()) if finite else None,
                               raw_max=float(synth.max()) if finite else None,
                               failure_reason=None if finite else 'NONFINITE_GENERATOR_OUTPUT', noise_seed=None,
                               target_face_sha256=sha_li, source_face_sha256=sha_sp)

    def finalize(self):
        return {'forwards': self.forwards, 'batch_size': 1, 'stochastic_at_inference': False,
                'auxiliary_outputs_written': False, 'bn_mode': 'inference_moving_statistics'}


# ================================================================== TRAIN-only feed-pack access (D2, D4 order)
def load_e04_inputs(runtime_root):
    """The frozen e04_feed_pack_v1 (TRAIN closure only, manifest pinned) cross-checked against split_v1 / M2 accounting:
    returns the 8838-row relation in ascending pair_id, the TRAIN allowlist and a Q140 lookup."""
    from methods.physics_std import feed_pack, production_io as pio
    pack = feed_pack.load(pio.pack_root(runtime_root), pio.PACK_MANIFEST_SHA256)
    relation = [tuple(r) for r in pack.rows]
    require(len(relation) == lb.N_SYN, 'relation size')
    used = {}
    for _, live, spoof, ds in relation:
        used[live], used[spoof] = ds, ds
    allow = lb.train_face_allowlist_parquet(used)
    for sid, rec in allow.items():
        require(rec['face_png_sha256'] == pack.sample(sid)['face_png_sha256'], 'pack face sha256 %s' % sid)
    return {'relation': relation, 'allowlist': allow, 'q140': pack.q140_xy,
            'pack_manifest_sha256': pack.manifest_sha256}


# ================================================================== constructors
def precision_environment_errors(environ, *, gpu):
    from methods.physics_std import production_io as pio
    want = dict(pio.PRECISION_ENV)
    if not gpu:
        want['CUDA_VISIBLE_DEVICES'] = ''
    return sorted(k for k, v in want.items() if environ.get(k) != v)


def make_session(tf):
    from methods.physics_std.production_runner import session_config
    return tf.Session(config=session_config(tf))


def load_real(tf, seed, run_root, q140_source, *, repo_root=None):
    """S7 / production (GPU, E04-Q5): environment lock, checkpoint hashes, graph, generator-only restore."""
    import os
    from methods.physics_std.production_runner import verify_environment
    _, contract_sha = lb.load_e04_generation_contract()
    cfg = lb.verify_config_identity(METHOD_ID)
    src = verify_frozen_sources(repo_root)
    errs = precision_environment_errors(os.environ, gpu=True)
    require(not errs, 'E04 precision environment differs: %s' % errs)
    live_env, driver = verify_environment(tf)
    ckpt = lb.verify_checkpoint(METHOD_ID, seed, run_root)
    prefix = Path(run_root) / 'checkpoints' / 'iteration_150000'
    graph = E04InferenceGraph(tf)
    coverage = graph.checkpoint_coverage(prefix)
    moving = moving_statistics_finite(tf, prefix)
    require(moving['ok'], 'generator moving statistics not finite / positive')
    sess = make_session(tf)
    graph.saver.restore(sess, str(prefix))
    tf.get_default_graph().finalize()
    adapter = E04BankAdapter(tf, sess, graph, q140_source, seed, is_fixture=False,
                             checkpoint_identity_sha256=ckpt['checkpoint_identity_sha256'],
                             contract_sha256=contract_sha, config_sha256=cfg['sha256'],
                             identity_extra={'checkpoint': ckpt, 'source': src, 'coverage': coverage,
                                             'moving_statistics': moving, 'driver': driver,
                                             'environment_session_config': live_env['session_config'],
                                             'framework': 'tensorflow ' + tf.__version__})
    return adapter, ckpt
