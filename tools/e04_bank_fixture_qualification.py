#!/usr/bin/env python3
"""CPU synthetic-fixture qualification of the E04 Physics-STD bank adapter (NOT S7, NOT production, NOT benchmark data).

Run on the GPU host from a fresh clean clone, CUDA hidden, in the locked gpat-m6-e03 interpreter:

    env -i HOME=$HOME PATH=/usr/bin:/bin <PRECISION_ENV with CUDA_VISIBLE_DEVICES=''> nice -n 19 \
      ~/miniconda3/envs/gpat-m6-e03/bin/python -B tools/e04_bank_fixture_qualification.py \
      --work <fresh dir> --out <evidence.json> --runtime-root <runtime root>

Inputs are artificial: random-init fixture weights saved from the frozen TrainingGraph, synthetic noise/gradient PNG
"faces" and synthetic Q140 geometry in the work dir. The three real E04 checkpoints are only hash-verified (before and
after) and structurally inspected (variable index, generator moving statistics); no real-weight forward is run. An
audit hook counts every open under <runtime_root>/data (benchmark faces, feed pack, pre-pass): it must stay zero.
CPU results prove neither GPU bitwise replay nor GPU numerical equality: that is E04 S7 (E04-Q5).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from methods.common import learned_bank as lb  # noqa: E402

FIXTURE_SEED = 60912          # qualification only; never an experiment seed
OPENS = {'runtime_data': 0, 'paths': []}


def install_open_counter(runtime_root):
    guard = str(Path(runtime_root) / 'data')

    def hook(event, args):
        if event == 'open' and args and isinstance(args[0], (str, bytes, os.PathLike)):
            p = os.fsdecode(args[0])
            if p.startswith(guard):
                OPENS['runtime_data'] += 1
                if len(OPENS['paths']) < 5:
                    OPENS['paths'].append(p)
    sys.addaudithook(hook)


class Gates:
    def __init__(self):
        self.items = {}

    def __call__(self, name, ok, detail=None):
        self.items[name] = {'pass': bool(ok), 'detail': detail}
        sys.stderr.write('%s %s\n' % ('PASS' if ok else 'FAIL', name))
        return ok

    def record(self):
        failed = sorted(k for k, v in self.items.items() if not v['pass'])
        return {'gates': self.items, 'gates_total': len(self.items), 'gates_passed': len(self.items) - len(failed),
                'failed_gates': failed}


# ------------------------------------------------------------------ synthetic inputs (never benchmark data)
def synthetic_faces(root, n_pairs, rng_seed):
    from PIL import Image
    rng = np.random.default_rng(rng_seed)
    samples, relation = {}, []
    yy, xx = np.mgrid[0:256, 0:256]
    for i in range(n_pairs):
        ids = []
        for role in ('live', 'spoof'):
            sid = hashlib.sha256(('e04fixture|%s|%d' % (role, i)).encode()).hexdigest()
            d = Path(root) / 'casia_fasd'
            d.mkdir(parents=True, exist_ok=True)
            base = np.stack([(xx + 11 * i) % 256, (yy * 2 + 5 * i) % 256, (xx + yy) % 256], -1)
            img = np.clip(base + rng.integers(-40, 40, (256, 256, 3)), 0, 255).astype(np.uint8)
            buf = io.BytesIO()
            Image.fromarray(img).save(buf, format='PNG')
            (d / (sid + '.png')).write_bytes(buf.getvalue())
            samples[sid] = {'split': 'TRAIN', 'dataset': 'casia_fasd',
                            'face_png_sha256': hashlib.sha256(buf.getvalue()).hexdigest()}
            ids.append(sid)
        relation.append(('PTR%06d' % (i + 1), ids[0], ids[1], 'casia_fasd'))
    return samples, relation


def synthetic_q140(sample_ids, rng_seed):
    rng = np.random.default_rng(rng_seed)
    template = rng.uniform(48, 208, (140, 2))
    template[:4] = [[40, 40], [216, 40], [40, 216], [216, 216]]       # hull corners
    return {sid: (template + rng.normal(0, 3, (140, 2))).astype(np.float32) for sid in sample_ids}


# ------------------------------------------------------------------ numpy references (independent transcription)
def np_resize_legacy(x, size):
    """tf.image.resize_bilinear(align_corners=False, half_pixel_centers=False), NHWC, float64."""
    n, h, w, c = x.shape

    def axis(out, inn):
        src = np.arange(out) * (inn / out)
        i0 = np.floor(src).astype(int)
        i1 = np.minimum(i0 + 1, inn - 1)
        return i0, i1, src - i0
    y0, y1, fy = axis(size, h)
    x0, x1, fx = axis(size, w)
    top = x[:, y0][:, :, x0] * (1 - fx)[None, None, :, None] + x[:, y0][:, :, x1] * fx[None, None, :, None]
    bot = x[:, y1][:, :, x0] * (1 - fx)[None, None, :, None] + x[:, y1][:, :, x1] * fx[None, None, :, None]
    return top * (1 - fy)[None, :, None, None] + bot * fy[None, :, None, None]


def np_lowpass(x, size):
    return np_resize_legacy(np_resize_legacy(x, size), 256)


def np_warp(x, off):
    n, h, w, c = x.shape
    yy, xx = np.mgrid[:h, :w].astype(np.float64)
    sx = np.clip(xx[None] + off[..., 0], 0, w - 1)
    sy = np.clip(yy[None] + off[..., 1], 0, h - 1)
    x0, y0 = np.floor(sx), np.floor(sy)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    b = np.arange(n)[:, None, None]

    def s(px, py):
        return x[b, py.astype(int), px.astype(int)]
    dx, dy = (sx - x0)[..., None], (sy - y0)[..., None]
    return (1 - dx) * (1 - dy) * s(x0, y0) + dx * (1 - dy) * s(x1, y0) + (1 - dx) * dy * s(x0, y1) + dx * dy * s(x1, y1)


def maxabs(a, b):
    return float(np.max(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))))


# ------------------------------------------------------------------ the qualification
def run(args, g):
    from methods.physics_std import bank_adapter as ba, network
    from methods.physics_std.production_io import PRECISION_ENV
    from methods.physics_std.trace_transfer import dense_offsets
    errs = ba.precision_environment_errors(os.environ, gpu=False)
    g('precision_env_cpu', not errs, errs)
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    g('cuda_hidden_tf115', not tf.test.is_gpu_available() and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
      and tf.__version__ == '1.15.5', tf.__version__)
    ev = {'tensorflow': tf.__version__, 'precision_env': {k: os.environ.get(k) for k in sorted(PRECISION_ENV)}}
    # ---------------- governance
    rec, csha = lb.load_e04_generation_contract()
    g('A.e04_contract_owner_approved', rec['decisions'] == lb.E04_APPROVED_DECISIONS, csha)
    g('A.frozen_config', lb.verify_config_identity('E04')['sha256'] == lb.METHOD_CONFIGS['E04'][1])
    src = ba.verify_frozen_sources()
    g('A.frozen_training_sources_and_lines', src['files'] == ba.FROZEN_FILES, src)
    g('A.inference_layers_differ_only_in_bn_mode', ba.inference_layers_differ_only_in_bn_mode())
    try:
        lb.load_s7_authority('E04')
        s7 = None
    except lb.LearnedBankBlocked as exc:
        s7 = exc.status
    g('A.e04_s7_refused', s7 == lb.STATUS_S7, s7)
    prod = []
    for s in lb.SEEDS:
        try:
            lb.load_production_authority('E04', s)
            prod.append(None)
        except lb.LearnedBankBlocked as exc:
            prod.append(exc.status)
    g('A.e04_production_refused', prod == [lb.STATUS_PRODUCTION] * 3, prod)
    pf = lb.storage_preflight()
    g('A.storage_live_ok_without_e04', pf['ok'] and lb.registered_bank_methods() == lb.METHODS
      and 'E04' not in os.listdir(pf['bank_root']), pf['checks'])
    regress = {}
    for m in lb.METHODS:
        regress[m] = lb.load_s7_authority(m).record()['contract_sha256'] is not None and all(
            lb.load_production_authority(m, s).mode == 'PRODUCTION' for s in lb.SEEDS)
    g('A.existing_four_methods_authorities_unchanged', all(regress.values()), regress)
    work = Path(args.work)
    # ---------------- B. fixture checkpoint from the frozen TrainingGraph (same variable set as a real checkpoint)
    from methods.physics_std.training_graph import TrainingGraph
    fx = work / 'fixture_ckpt'
    fx.mkdir()
    tf.reset_default_graph()
    tf.set_random_seed(FIXTURE_SEED)
    m = TrainingGraph(tf, ROOT)
    with ba.make_session(tf) as s0:
        s0.run(tf.global_variables_initializer())
        for v in tf.global_variables():               # non-trivial generator moving statistics
            if v.op.name.startswith('G/') and v.op.name.endswith('moving_mean'):
                s0.run(v.assign(tf.random.normal(v.shape, stddev=0.05, seed=FIXTURE_SEED)))
            if v.op.name.startswith('G/') and v.op.name.endswith('moving_variance'):
                s0.run(v.assign(tf.random.uniform(v.shape, 0.5, 1.5, seed=FIXTURE_SEED)))
        trained = {v.op.name: v.shape.as_list() for v in tf.global_variables()
                   if v.op.name.startswith('G/') and 'Adam' not in v.op.name}
        values0 = {v.op.name: s0.run(v) for v in tf.global_variables() if v.op.name in trained}
        prefix = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=None).save(
            s0, str(fx / 'iteration_fixture'), write_meta_graph=True)
    fixture_files = {p.name: lb.sha256_file(p) for p in sorted(fx.iterdir())}
    ev['fixture_checkpoint'] = fixture_files
    # ---------------- C. inference graph integrity and restore
    tf.reset_default_graph()
    graph = ba.E04InferenceGraph(tf)
    st = graph.structure
    names_shapes = {v.op.name: v.shape.as_list() for v in graph.variables}
    g('C.variable_names_shapes_equal_trained_generator', names_shapes == trained and len(trained) == 102,
      {'inference': len(names_shapes), 'trained_generator': len(trained)})
    g('C.bn_inference_mode_reads_moving_statistics', st['fused_batch_norm'] == 15
      and st['fused_batch_norm_is_training'] is False and st['bn_moving_statistics'] == 30, st)
    g('C.no_update_optimizer_gradient_random_ops', st['update_ops'] == 0 and st['gradient_ops'] == 0
      and not st['mutating_reachable_ops'] and not st['random_reachable_ops'] and not any(
          'Adam' in o.name or o.type.startswith('Apply') for o in tf.get_default_graph().get_operations()))
    g('C.no_discriminator', not any(o.name.startswith('D/') for o in tf.get_default_graph().get_operations()))
    cov = graph.checkpoint_coverage(prefix)
    g('C.generator_only_restore', cov['restored_variables'] == 102 and cov['not_restored_classes']['discriminator'] > 0
      and cov['not_restored_classes']['optimizer_slots'] > 0, cov)
    sess = ba.make_session(tf)
    graph.saver.restore(sess, prefix)
    tf.get_default_graph().finalize()
    g('C.restore_exact', all(np.array_equal(sess.run(v), values0[v.op.name]) for v in graph.variables))
    ev['graph'] = st
    # ---------------- D. fixture banks through the common writer/reader
    faces = work / 'faces'
    samples, relation = synthetic_faces(faces, 6, 7)
    q140 = synthetic_q140(samples, 11)
    fixture_identity = lb.sha256_bytes(lb.canonical_json(fixture_files).encode())
    adapter = ba.E04BankAdapter(tf, sess, graph, lambda sid: q140[sid], FIXTURE_SEED, is_fixture=True,
                                checkpoint_identity_sha256=fixture_identity, contract_sha256=csha,
                                config_sha256=lb.METHOD_CONFIGS['E04'][1],
                                identity_extra={'fixture_checkpoint': fixture_files, 'coverage': cov,
                                                'framework': 'tensorflow ' + tf.__version__})
    auth = lb.e04_fixture_authorization('e04_cpu')
    positions = [0, 2, 3, 5]

    def mk(pos=positions):
        return lb.pair_population('E04', FIXTURE_SEED, 'SYNTHETIC_FIXTURE', relation, positions=pos)
    prov_a = lb.write_bank(adapter, mk(), auth, work / 'E04_A', reader=lb.TrainFaceReader(faces, samples))
    prov_b = lb.write_bank(adapter, mk(), auth, work / 'E04_B', reader=lb.TrainFaceReader(faces, samples))
    ra = lb.read_bank(work / 'E04_A', expect_method='E04', expect_mode='FIXTURE', expect_rows=len(positions))
    rb = lb.read_bank(work / 'E04_B', expect_method='E04', expect_mode='FIXTURE', expect_rows=len(positions))
    g('D.reader_ok', ra['ok'] and rb['ok'] and ra['failures'] == 0 == rb['failures'], ra['problems'][:3] + rb['problems'][:3])
    g('D.png_256x256x3_uint8', ra['files_checked'] == len(positions) == rb['files_checked'])
    cmp = lb.compare_runs(work / 'E04_A', work / 'E04_B')
    g('D.cpu_replay_bitwise', cmp['bitwise_identical'], cmp)
    g('D.create_only_immutable', all(oct(os.stat(p).st_mode & 0o777) == '0o444' for p in (work / 'E04_A/bank').iterdir())
      and oct(os.stat(work / 'E04_A').st_mode & 0o777) == '0o555')
    before = sorted(p.name for p in (work / 'E04_A' / 'bank').iterdir())
    try:
        lb.write_bank(adapter, mk([1]), auth, work / 'E04_A', reader=lb.TrainFaceReader(faces, samples))
        refused = None
    except lb.LearnedBankError as exc:
        refused = str(exc)
    g('D.overwrite_refused', refused is not None and 'exists' in refused
      and sorted(p.name for p in (work / 'E04_A' / 'bank').iterdir()) == before, refused)
    g('D.o_noatime_reads', prov_a['reader']['atime_checked_reads'] == 2 * len(positions), prov_a['reader'])
    rows = ra['rows']
    g('D.manifest_e04_identity', all(r['method_id'] == 'E04' and r['deterministic_noise_seed'] is None
                                     and r['conversion_rule'] == lb.CONVERSION_RULES['E04']
                                     and r['contract_sha256'] == csha and r['row_id'] == r['pair_id']
                                     and r['checkpoint_identity_sha256'] == fixture_identity for r in rows))
    g('D.lineage_roles', [(r['pair_id'], r['target_live_id'], r['source_spoof_id']) for r in rows] ==
      [relation[i][:3] for i in positions])
    g('D.no_auxiliary_outputs', sorted(p.name for p in (work / 'E04_A').iterdir()) ==
      sorted([lb.BANK_DIR, lb.MANIFEST, lb.MANIFEST_SHA, lb.PROVENANCE])
      and all(p.suffix == '.png' for p in (work / 'E04_A' / 'bank').iterdir()))
    # one row, recomputed outside the writer
    pid, live_id, spoof_id, _ = relation[2]
    live = ba.decode_face((faces / 'casia_fasd' / (live_id + '.png')).read_bytes())
    spoof = ba.decode_face((faces / 'casia_fasd' / (spoof_id + '.png')).read_bytes())
    from PIL import Image
    g('D.input_scale_d8', live.dtype == np.float32 and np.array_equal(
        live, np.asarray(Image.open(faces / 'casia_fasd' / (live_id + '.png')), np.uint8).astype(np.float32) / 255))
    synth = adapter.synthesize(live, spoof, q140[live_id], q140[spoof_id])
    row = next(r for r in rows if r['pair_id'] == pid)
    px = ba.convert_uint8(synth)
    g('D.conversion_equals_independent_reference', np.array_equal(px, lb.reference_e04_uint8(synth))
      and lb.pixel_sha256(px) == row['output_pixel_sha256'] and row['raw_output_min'] == float(synth.min()),
      [row['raw_output_min'], row['raw_output_max']])
    ev['fixture_synthesis_range'] = [float(synth.min()), float(synth.max())]      # recorded, not a gate
    g('D.moving_statistics_not_updated', all(np.array_equal(sess.run(v), values0[v.op.name]) for v in graph.variables))
    # ---------------- E. numpy reference of decomposition, low-pass, warp and synthesis (same restored weights)
    off = ba.offsets_for(q140[spoof_id], q140[live_id])
    fetch = {k: graph.decomposition[k] for k in ('decoder_raw', 'B', 'C', 'T', 'P', 'B_low', 'C_low')}
    fetch.update({'w_' + k: v for k, v in graph.warped.items()})
    fetch['synth'] = graph.synth
    out = sess.run(fetch, {graph.source: spoof[None], graph.live: live[None], graph.offsets: off[None]})
    raw = out['decoder_raw'].astype(np.float64)
    e = {'heads_B_C_T_slices': max(maxabs(out['B'], raw[..., 0:3]), maxabs(out['C'], raw[..., 3:6]),
                                   maxabs(out['T'], raw[..., 6:9])),
         'P_sigmoid_channel_9': maxabs(out['P'], 1 / (1 + np.exp(-raw[..., 9:10]))),
         'B_low32': maxabs(out['B_low'], np_lowpass(out['B'].astype(np.float64), 32)),
         'C_low128': maxabs(out['C_low'], np_lowpass(out['C'].astype(np.float64), 128))}
    o64 = off[None].astype(np.float64)
    for k, src_t in (('B', out['B_low']), ('C', out['C_low']), ('T', out['T']), ('P', out['P']), ('source', spoof[None])):
        e['warp_' + k] = maxabs(out['w_' + k], np_warp(np.asarray(src_t, np.float64), o64))
    ref = (1 - out['w_P'].astype(np.float64)) * (live[None] + (out['w_B'].astype(np.float64) + out['w_C'] + out['w_T'])) \
        + out['w_P'] * out['w_source'].astype(np.float64)
    e['synthesis_equation'] = maxabs(out['synth'], ref)
    e['synth_equals_adapter'] = maxabs(out['synth'][0], synth)
    g('E.numpy_reference_decomposition_lowpass_warp_synthesis', all(v <= 2e-5 for v in e.values())
      and e['synth_equals_adapter'] == 0.0, e)
    # roles and offset direction: an exact integer translation spoof = live + (5, 3)
    ql = q140[live_id]
    qs = (ql + np.array([5, 3], np.float32)).astype(np.float32)
    off_t = ba.offsets_for(qs, ql)
    inside = off_t[100:156, 100:156]
    g('E.offsets_direction_source_minus_target', np.allclose(inside[..., 0], 5, atol=1e-4)
      and np.allclose(inside[..., 1], 3, atol=1e-4) and np.array_equal(off_t[0, 0], [0, 0]),
      [float(inside[..., 0].mean()), float(inside[..., 1].mean())])
    o2 = sess.run(graph.warped['source'], {graph.source: spoof[None], graph.live: live[None],
                                           graph.offsets: np.round(off_t)[None].astype(np.float32)})
    g('E.warp_reads_source_at_p_plus_offset', np.array_equal(o2[0, 120, 120], spoof[123, 125]))
    swapped = adapter.synthesize(spoof, live, q140[spoof_id], q140[live_id])
    g('E.roles_matter', maxabs(swapped, synth) > 1e-3, maxabs(swapped, synth))
    g('E.dense_offsets_is_frozen_function', np.array_equal(off, dense_offsets(q140[spoof_id], q140[live_id])))
    sess.close()
    # ---------------- F. frozen-graph comparison: composition == TrainingGraph.synthesized_spoof (same BN inputs)
    tf.reset_default_graph()
    m = TrainingGraph(tf, ROOT)
    g8 = network.generator(tf, m.rgb, network.Layers(tf))       # frozen layers: batch statistics, shared variables
    sliced = {k: g8[k][4:] for k in ('B_low', 'C_low', 'T', 'P')}
    spoof_c, warped_c = ba.compose(tf, sliced, m.rgb[:4], m.rgb[4:], m.offsets)
    s1 = ba.make_session(tf)
    tf.train.Saver(var_list=tf.global_variables()).restore(s1, prefix)
    ids = [relation[i] for i in range(4)]
    rgb8 = np.stack([ba.decode_face((faces / 'casia_fasd' / (r[1] + '.png')).read_bytes()) for r in ids] +
                    [ba.decode_face((faces / 'casia_fasd' / (r[2] + '.png')).read_bytes()) for r in ids])
    off4 = np.stack([ba.offsets_for(q140[r[2]], q140[r[1]]) for r in ids])
    a, b, wa, wb = s1.run([m.spoof, spoof_c, m.warped, warped_c], {m.rgb: rgb8, m.offsets: off4})
    g('F.composition_equals_frozen_synthesized_spoof_bitwise', np.array_equal(a, b)
      and all(np.array_equal(wa[k], wb[k]) for k in wa), maxabs(a, b))
    s1.close()
    # ---------------- G. BN-mode isolation: inference BN with moving stats := batch stats reproduces training mode
    tf.reset_default_graph()
    xs = tf.placeholder(tf.float32, [1, 256, 256, 3])
    xl = tf.placeholder(tf.float32, [1, 256, 256, 3])
    xo = tf.placeholder(tf.float32, [1, 256, 256, 2])
    gt = network.generator(tf, xs, network.Layers(tf))
    st_t, _ = ba.compose(tf, gt, xl, xs, xo)
    bn_t = {o.name: o.inputs[0] for o in tf.get_default_graph().get_operations() if o.type.startswith('FusedBatchNorm')}
    s2 = ba.make_session(tf)
    tf.train.Saver(var_list={v.op.name: v for v in tf.global_variables()}).restore(s2, prefix)
    feed_t = {xs: spoof[None], xl: live[None], xo: off[None]}
    t_out, t_bn = s2.run([[gt['B'], gt['C'], gt['T'], gt['P'], st_t], bn_t], feed_t)
    s2.close()
    tf.reset_default_graph()
    gi = ba.E04InferenceGraph(tf)
    assigns = []
    for name, x in t_bn.items():
        base = name.rsplit('/', 1)[0]
        mean = x.astype(np.float64).mean(axis=(0, 1, 2))
        var = x.astype(np.float64).var(axis=(0, 1, 2))
        v_m = [v for v in gi.variables if v.op.name == base + '/moving_mean'][0]
        v_v = [v for v in gi.variables if v.op.name == base + '/moving_variance'][0]
        assigns += [tf.assign(v_m, mean.astype(np.float32)), tf.assign(v_v, var.astype(np.float32))]
    s3 = ba.make_session(tf)
    gi.saver.restore(s3, prefix)
    fetch_i = [gi.decomposition['B'], gi.decomposition['C'], gi.decomposition['T'], gi.decomposition['P'], gi.synth]
    feed_i = {gi.source: spoof[None], gi.live: live[None], gi.offsets: off[None]}
    own = s3.run(fetch_i, feed_i)                    # fixture's stored moving statistics (the approved mode)
    s3.run(assigns)
    iso = s3.run(fetch_i, feed_i)                    # moving statistics := this image's batch statistics
    s3.close()
    names = ['B', 'C', 'T', 'P', 'synth']
    d_iso = {n: maxabs(x, y) for n, x, y in zip(names, iso, t_out)}
    d_own = {n: maxabs(x, y) for n, x, y in zip(names, own, t_out)}
    g('G.bn_mode_is_the_only_difference', len(t_bn) == 15 and d_iso['synth'] <= 1e-4
      and d_own['synth'] > 100 * max(d_iso['synth'], 1e-7),
      {'equal_statistics_max_abs': d_iso, 'approved_moving_statistics_max_abs': d_own})
    ev['bn_mode_isolation'] = {'equal_statistics_max_abs': d_iso, 'approved_moving_statistics_max_abs': d_own}
    # ---------------- H. conversion cases, non-finite accounting, firewall
    f32 = np.float32
    cases = np.array([-1.0, -0.0, 0.0, 0.5 / 255, 1 / 255, 0.5, 127.5 / 255, 254.5 / 255, np.nextafter(f32(1), f32(0)),
                      1.0, 1.5, 3.0], dtype=f32).reshape(1, 4, 3)
    got = ba.convert_uint8(cases).ravel().tolist()
    want_f32 = [int(np.float32(min(max(v, f32(0)), f32(1))) * f32(255)) for v in cases.ravel()]   # float32, truncate
    g('H.clip_truncate_cases', got == want_f32 and got[5] == 127 and got[6] == 127 and got[7] == 254 and got[9] == 255
      and got[0] == 0 and got[10] == 255, {'inputs': [float(v) for v in cases.ravel()], 'got': got})
    g('H.not_rounding', got[5] != int(np.rint(0.5 * 255)), [got[5], int(np.rint(0.5 * 255))])
    try:
        ba.convert_uint8(np.array([[[np.nan, 0, 0]]], f32))
        nan_refused = False
    except lb.LearnedBankError:
        nan_refused = True
    g('H.nonfinite_never_converted', nan_refused)

    class NaNAdapter(ba.E04BankAdapter):
        def synthesize(self, live_, spoof_, ql_, qs_):
            out_ = np.zeros((256, 256, 3), np.float32)
            out_[0, 0, 0] = np.nan
            return out_
    tf.reset_default_graph()
    nan_ad = NaNAdapter(tf, None, None, lambda sid: q140[sid], FIXTURE_SEED, is_fixture=True,
                        checkpoint_identity_sha256=fixture_identity, contract_sha256=csha,
                        config_sha256=lb.METHOD_CONFIGS['E04'][1], identity_extra={'probe': 'non-finite accounting'})
    prov_n = lb.write_bank(nan_ad, mk([1]), auth, work / 'E04_NAN', reader=lb.TrainFaceReader(faces, samples))
    rn = lb.read_bank(work / 'E04_NAN', expect_method='E04', expect_mode='FIXTURE', expect_rows=1)
    g('H.nonfinite_counted_failure_no_file', rn['ok'] and rn['failures'] == 1 and prov_n['failure_count'] == 1
      and rn['rows'][0]['failure_reason'] == 'NONFINITE_GENERATOR_OUTPUT' and rn['rows'][0]['output_path'] is None
      and not list((work / 'E04_NAN' / 'bank').iterdir()), rn['rows'][0]['generation_status'])
    rdr = lb.TrainFaceReader(faces, samples)
    try:
        rdr.read('VAL_OR_TEST_SAMPLE')
        outside = False
    except lb.LearnedBankError:
        outside = True
    try:
        lb.TrainFaceReader(faces, {'x': {'split': 'VAL', 'dataset': 'casia_fasd', 'face_png_sha256': '0' * 64}})
        val_ok = False
    except lb.LearnedBankError:
        val_ok = True
    g('H.val_test_refused', outside and val_ok)
    g('H.fixture_checkpoint_unchanged', {p.name: lb.sha256_file(p) for p in sorted(fx.iterdir())} == fixture_files)
    # ---------------- I. the three real E04 checkpoints: hash before/after, index coverage, moving statistics
    real = {}
    for seed in lb.SEEDS:
        run_root = Path(args.runtime_root) / 'runs' / 'm6' / 'E04' / ('seed_%d' % seed)
        before = lb.verify_checkpoint('E04', seed, run_root)
        tf.reset_default_graph()
        gr = ba.E04InferenceGraph(tf)
        pre = run_root / 'checkpoints' / 'iteration_150000'
        cov_r = gr.checkpoint_coverage(pre)
        mov = ba.moving_statistics_finite(tf, pre)
        lb.recheck_checkpoint_unchanged(before)
        after = lb.verify_checkpoint('E04', seed, run_root)
        real[str(seed)] = {'run_id': before['run_id'], 'checkpoint_identity_sha256': before['checkpoint_identity_sha256'],
                           'files': {k: v['sha256'] for k, v in before['files'].items()}, 'coverage': cov_r,
                           'moving_statistics': mov, 'unchanged_after': after['files'] == before['files']}
        g('I.E04_seed%d_checkpoint_verified_covered_unchanged' % seed, real[str(seed)]['unchanged_after']
          and cov_r['restored_variables'] == 102 and cov_r['checkpoint_variables'] == 500 and mov['ok']
          and before['checkpoint_identity_sha256'] == lb.checkpoint_identity_sha256('E04', seed), cov_r)
    ev['real_checkpoints'] = real
    ev['banks'] = {'A': prov_a['manifest_sha256'], 'B': prov_b['manifest_sha256'], 'nonfinite': prov_n['manifest_sha256']}
    ev['source'] = src
    return ev


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--work', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--runtime-root', required=True)
    args = ap.parse_args(argv)
    work, out = Path(args.work), Path(args.out)
    if out.exists() or work.exists():
        raise SystemExit('refusing to overwrite %s / %s' % (out, work))
    work.mkdir()
    install_open_counter(args.runtime_root)
    g = Gates()
    t0 = time.monotonic()
    started = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    try:
        evidence, error = run(args, g), None
    except Exception as exc:  # recorded, never hidden
        import traceback
        evidence, error = {}, traceback.format_exc()
        g('harness_completed', False, '%s: %s' % (type(exc).__name__, exc))
    g('zero_opens_under_runtime_data', OPENS['runtime_data'] == 0, OPENS)
    rec = dict(g.record(), schema='e04_bank_fixture_qualification_v1', method='E04',
               label='SYNTHETIC_FIXTURE_QUALIFICATION_NOT_S7_NOT_BENCHMARK', fixture_seed=FIXTURE_SEED,
               git_head=lb._git_head(), environment=lb.environment_record(), started_utc=started,
               wall_seconds=round(time.monotonic() - t0, 1), evidence=evidence, error=error,
               gpu_bitwise_reproducibility_proven=False, gpu_numerical_equality_proven=False, work=str(work))
    out.write_text(json.dumps(rec, indent=1, sort_keys=True, default=str) + '\n')
    print(json.dumps({'gates_passed': rec['gates_passed'], 'gates_total': rec['gates_total'],
                      'failed': rec['failed_gates']}))
    return 0 if not rec['failed_gates'] else 1


if __name__ == '__main__':
    sys.exit(main())
