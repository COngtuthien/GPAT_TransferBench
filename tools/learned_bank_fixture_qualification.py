#!/usr/bin/env python3
"""CPU synthetic-fixture qualification of the four learned-bank adapters (NOT S7, NOT production, NOT benchmark data).

Run on the GPU host from an isolated clone, one family per locked environment, CUDA hidden, idle priority:

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 19 ionice -c3 \
      <env python> -B tools/learned_bank_fixture_qualification.py <dsdg|e05|e03> \
      --work <fresh dir> --out <evidence.json> --runtime-root <runtime root (checkpoints read-only)>

Inputs are artificial: random-init fixture weights (DSDG Decoder_s / STDN Gen from the pinned source; PCGAN interface
fixtures, because the real StyleGAN2 kernels are CUDA-only), synthetic PNG faces in the work dir, synthetic landmarks.
The twelve real checkpoints are only hash-verified and structurally inspected (no forward, no model output), with hashes
before and after. An audit hook counts every open under <runtime_root>/data (benchmark faces, manifests of record): it
must stay zero. Nothing here proves GPU bitwise reproducibility; that is S7 (C-Q2/C-Q3) after E04 completes.
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

FIXTURE_SEED = 60911          # qualification only; never an experiment seed
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


def common_gates(g, family, methods):
    _, csha = lb.load_generation_contract()
    g('contract_owner_approved', True, csha)
    g('pinned_sources_verified', bool(lb.verify_pinned_sources(family)), family)
    g('frozen_configs', all(lb.verify_config_identity(m)['sha256'] == lb.METHOD_CONFIGS[m][1] for m in methods))
    for m in methods:
        try:
            lb.load_s7_authority(m)
            refused = None
        except lb.LearnedBankBlocked as exc:
            refused = exc.status
        g('s7_refused_%s' % m, refused == lb.STATUS_E04, refused)
        refusals = []
        for s in lb.SEEDS:
            try:
                lb.load_production_authority(m, s)
                refusals.append(None)
            except lb.LearnedBankBlocked as exc:
                refusals.append(exc.status)
        g('production_refused_%s' % m, all(r in (lb.STATUS_E04, lb.STATUS_PRODUCTION) for r in refusals), refusals)
    return csha


def real_checkpoints(g, method, runtime_root, inspect):
    """Hash-verify (before/after) and structurally inspect the approved final checkpoints; never a forward."""
    out = {}
    for seed in lb.SEEDS:
        run_root = Path(runtime_root) / 'runs' / 'm6' / method / ('seed_%d' % seed)
        before = lb.verify_checkpoint(method, seed, run_root)
        info = inspect(run_root, before)
        lb.recheck_checkpoint_unchanged(before)
        after = lb.verify_checkpoint(method, seed, run_root)
        out[seed] = {'checkpoint_identity_sha256': before['checkpoint_identity_sha256'],
                     'files': {k: v['sha256'] for k, v in before['files'].items()}, 'structure': info,
                     'unchanged_after': after['files'] == before['files']}
        g('%s_seed%d_checkpoint_verified_unchanged' % (method, seed), out[seed]['unchanged_after'], before['run_id'])
    return out


def synthetic_faces(root, n_pairs, rng_seed):
    """Artificial PNG 'faces' (uniform noise + gradients) and a fixture relation; never benchmark data."""
    from PIL import Image
    rng = np.random.default_rng(rng_seed)
    samples, relation = {}, []
    yy, xx = np.mgrid[0:256, 0:256]
    for i in range(n_pairs):
        ids = []
        for role in ('live', 'spoof'):
            sid = hashlib.sha256(('fixture|%s|%d' % (role, i)).encode()).hexdigest()
            d = Path(root) / 'casia_fasd'
            d.mkdir(parents=True, exist_ok=True)
            base = np.stack([(xx + 7 * i) % 256, (yy * 2) % 256, (xx + yy) % 256], -1)
            img = np.clip(base + rng.integers(-40, 40, (256, 256, 3)), 0, 255).astype(np.uint8)
            buf = io.BytesIO()
            Image.fromarray(img).save(buf, format='PNG')
            (d / (sid + '.png')).write_bytes(buf.getvalue())
            samples[sid] = {'split': 'TRAIN', 'dataset': 'casia_fasd',
                            'face_png_sha256': hashlib.sha256(buf.getvalue()).hexdigest()}
            ids.append(sid)
        relation.append(('PTR%06d' % (i + 1), ids[0], ids[1], 'casia_fasd'))
    return samples, relation


def bank_gates(g, prefix, run_a, run_b, method, rows, retry):
    ra = lb.read_bank(run_a, expect_method=method, expect_mode='FIXTURE', expect_rows=rows)
    rb = lb.read_bank(run_b, expect_method=method, expect_mode='FIXTURE', expect_rows=rows)
    g(prefix + '_reader_ok', ra['ok'] and rb['ok'], ra['problems'][:3] + rb['problems'][:3])
    g(prefix + '_no_failures', ra['failures'] == 0 and rb['failures'] == 0)
    g(prefix + '_256x256x3_uint8_png', ra['files_checked'] == rows)
    cmp = lb.compare_runs(run_a, run_b)
    g(prefix + '_cpu_replay_bitwise', cmp['bitwise_identical'], cmp)
    g(prefix + '_create_only_immutable', all(oct(os.stat(p).st_mode & 0o777) == '0o444'
                                              for p in (Path(run_a) / 'bank').iterdir())
      and oct(os.stat(run_a).st_mode & 0o777) == '0o555')
    before = sorted(p.name for p in (Path(run_a) / 'bank').iterdir())
    try:
        retry(run_a)
        refused = None
    except lb.LearnedBankError as exc:
        refused = str(exc)
    g(prefix + '_overwrite_refused', refused is not None and 'exists' in refused
      and sorted(p.name for p in (Path(run_a) / 'bank').iterdir()) == before, refused)
    return ra


# =========================================================================================== DSDG (E06b, E06c)
def run_dsdg(args, g):
    import torch
    from methods.common.upstream import upstream_modules
    from methods.dsdg import bank_adapter as da
    g('cuda_hidden', not torch.cuda.is_available() and os.environ.get('CUDA_VISIBLE_DEVICES') == '')
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '2')))
    common_gates(g, 'DSDG', lb.DSDG_METHODS)
    # randn(generator=g) is the official zeros().normal_(0, 1) draw (generated.py:47-48) on the same generator state
    g1, g2 = torch.Generator().manual_seed(123), torch.Generator().manual_seed(123)
    a = [torch.randn(100, 128, generator=g1) for _ in range(4)]
    b = [torch.zeros(100, 128).normal_(0, 1, generator=g2) for _ in range(4)]
    g('randn_equals_official_normal_draw', all(torch.equal(x, y) for x, y in zip(a, b)))
    work = Path(args.work)
    evidence = {'methods': {}}
    fixture_ckpts = {}
    for method in lb.DSDG_METHODS:                 # written before any training guard is installed
        fixture_ckpts[method] = da.write_fixture_checkpoint(torch, method, FIXTURE_SEED, _fresh(work / (
            '%s_fixture' % method)))
    auth = lb.fixture_authorization('dsdg_cpu')
    for method in lb.DSDG_METHODS:
        ckpt = fixture_ckpts[method]
        pop = lambda n: lb.dsdg_population(method, FIXTURE_SEED, n, 'SYNTHETIC_FIXTURE')  # noqa: E731
        adapter = da.build_fixture(torch, method, FIXTURE_SEED, ckpt)
        g('%s_official_load_model_full_coverage' % method, adapter.identity()['load']['all_netG_entries_from_checkpoint'])
        g('%s_netG_parameters' % method, da.parameter_report(adapter.netG) == da.NETG_PARAMETERS)
        g('%s_eval_mode' % method, not adapter.netG.training)
        t0 = time.monotonic()
        prov_a = lb.write_bank(adapter, pop(lb.TOY_N), auth, work / ('%s_A' % method))
        t_bank = time.monotonic() - t0
        g('%s_discarded_tail_14' % method, prov_a['adapter_finalize']['discarded_final_batch_samples'] == 14)
        adapter_b = da.build_fixture(torch, method, FIXTURE_SEED, ckpt)        # independent load, fresh root
        prov_b = lb.write_bank(adapter_b, pop(lb.TOY_N), auth, work / ('%s_B' % method))
        g('%s_latent_stream_identical' % method, prov_a['adapter_finalize']['latent_stream_sha256'] ==
          prov_b['adapter_finalize']['latent_stream_sha256'])
        ra = bank_gates(g, method, work / ('%s_A' % method), work / ('%s_B' % method), method, lb.TOY_N,
                        lambda d: lb.write_bank(da.build_fixture(torch, method, FIXTURE_SEED, ckpt), pop(3), auth, d))
        ids_ok = all(r['row_id'] == lb.synthetic_id(method, FIXTURE_SEED, r['row_index']) and r['pair_id'] is None
                     and r['target_live_id'] is None and r['source_spoof_id'] is None for r in ra['rows'])
        g('%s_synthetic_ids_null_lineage' % method, ids_ok)
        # independent recomputation with the official draw expression and the reference conversion
        gen = torch.Generator().manual_seed(lb.dsdg_bank_seed(method, FIXTURE_SEED))
        recomputed = {}
        for batch in range(2):
            noise = torch.zeros(100, 128).normal_(0, 1, generator=gen)
            noise_s = torch.zeros(100, 128).normal_(0, 1, generator=gen)
            with torch.no_grad():
                fake = adapter.netG(torch.cat((noise_s, noise, noise), dim=1))
            spoof = fake[:, 0:3].numpy()
            for i in (0, 85, 99):
                idx = batch * 100 + i
                if idx < lb.TOY_N:
                    recomputed[idx] = lb.pixel_sha256(lb.reference_dsdg_uint8(spoof[i]))
            if batch == 0:
                g('%s_sigmoid_range' % method, float(fake.min()) >= 0.0 and float(fake.max()) <= 1.0)
                g('%s_output_shape' % method, tuple(fake.shape) == (100, 6, 256, 256) and fake.dtype == torch.float32)
        by_index = {r['row_index']: r['output_pixel_sha256'] for r in ra['rows']}
        g('%s_independent_recompute' % method, all(by_index[k] == v for k, v in recomputed.items()), sorted(recomputed))
        # toy prefix == longer-stream rows (the S7 toy rows must later equal the production rows)
        lb.write_bank(da.build_fixture(torch, method, FIXTURE_SEED, ckpt), pop(250), auth, work / ('%s_C250' % method))
        same = all((work / ('%s_A' % method) / 'bank' / n).read_bytes() ==
                   (work / ('%s_C250' % method) / 'bank' / n).read_bytes()
                   for n in sorted(p.name for p in (work / ('%s_A' % method) / 'bank').iterdir()))
        g('%s_toy_prefix_equals_longer_stream' % method, same)
        g('%s_fixture_checkpoint_unchanged' % method, lb.sha256_file(ckpt['path']) == ckpt['sha256'])
        guards = adapter.identity()['training_guards']
        g('%s_no_optimizer_backward_save' % method, all(v == 0 for v in guards.values()), dict(guards))
        evidence['methods'][method] = {'fixture_checkpoint': ckpt, 'bank_A': prov_a['manifest_sha256'],
                                       'bank_B': prov_b['manifest_sha256'], 'bank_seconds_cpu': round(t_bank, 1),
                                       'stream_seed_fixture': lb.dsdg_bank_seed(method, FIXTURE_SEED),
                                       'load': adapter.identity()['load']}
    m6, m7 = (work / 'E06b_A' / 'bank'), (work / 'E06c_A' / 'bank')
    g('E06b_E06c_distinct_streams', lb.dsdg_bank_seed('E06b', FIXTURE_SEED) != lb.dsdg_bank_seed('E06c', FIXTURE_SEED)
      and (m6 / 'E06b_S60911_000000.png').read_bytes() != (m7 / 'E06c_S60911_000000.png').read_bytes())
    # guards are effective
    try:
        torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
        blocked = False
    except lb.TrainingForbidden:
        blocked = True
    g('optimizer_construction_forbidden', blocked)

    def inspect(run_root, before):
        with upstream_modules(da.source_root(), da.UPSTREAM_MODULES, da.UPSTREAM_ROOTS) as mods:
            obj = torch.load(before['files']['pth']['path'], map_location='cpu', weights_only=False)
            sd = obj['model'].state_dict()
            ref = torch.nn.DataParallel(mods['networks'].Decoder_s(hdim=128)).state_dict()
        return {'epoch': obj.get('epoch'), 'wrapper': type(obj['model']).__name__,
                'inner': type(getattr(obj['model'], 'module', obj['model'])).__name__,
                'keys_equal_decoder_s': sorted(sd) == sorted(ref),
                'shapes_equal': all(tuple(sd[k].shape) == tuple(ref[k].shape) for k in ref if k in sd),
                'all_finite': all(bool(torch.isfinite(v).all()) for v in sd.values() if v.is_floating_point())}
    for method in lb.DSDG_METHODS:
        real = real_checkpoints(g, method, args.runtime_root, inspect)
        for seed, rec in real.items():
            s = rec['structure']
            g('%s_seed%d_checkpoint_structure' % (method, seed), s['epoch'] == 200 and s['inner'] == 'Decoder_s'
              and s['keys_equal_decoder_s'] and s['shapes_equal'] and s['all_finite'], s)
        evidence['methods'][method]['real_checkpoints'] = real
    evidence['torch'] = torch.__version__
    return evidence


def _fresh(p):
    p.mkdir()
    return p


# =========================================================================================== E05 PCGAN
def run_e05(args, g):
    import torch
    from methods.pcgan import bank_adapter as pa
    g('cuda_hidden', not torch.cuda.is_available() and os.environ.get('CUDA_VISIBLE_DEVICES') == '')
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '2')))
    csha = common_gates(g, 'PCGAN', ('E05',))
    tensor2im, seg_sha = pa.pinned_tensor2im()
    rng = np.random.default_rng(3)
    vals = [rng.normal(0, 1.4, (3, 256, 256)).astype(np.float32) for _ in range(4)]
    vals.append(np.array([-1.5, -1.0, -0.0039, 0.0, 0.999, 1.0, 2.0], np.float32)[None, None].repeat(3, 0))
    g('tensor2im_equals_reference', all(np.array_equal(tensor2im(torch.from_numpy(v)), lb.reference_e05_uint8(v))
                                        for v in vals), seg_sha)
    face = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
    x = pa.canonical_input(face)
    ref = (np.ascontiguousarray(face.transpose(2, 0, 1)).astype(np.float32) / np.float32(255) - np.float32(0.5)) \
        / np.float32(0.5)
    g('input_mapping', x.dtype == np.float32 and x.shape == (3, 256, 256) and np.array_equal(x, ref)
      and float(x.min()) >= -1 and float(x.max()) <= 1)
    enc_text = (ROOT / 'third_party/source_cache/swapping_autoencoder/models/networks/encoder.py').read_text()
    g('encoder_draws_no_random_numbers', not any(t in enc_text for t in ('normal_(', 'randn', 'rand(', 'dropout',
                                                                       'Dropout')))
    E, G = pa.build_fixture(torch, FIXTURE_SEED)
    guards = lb.forbid_training_torch(torch)
    adapter = pa.PCGANBankAdapter(torch, FIXTURE_SEED, E, G, tensor2im, torch.device('cpu'), is_fixture=True,
                                  checkpoint_identity_sha256='f1' * 32, contract_sha256=csha,
                                  config_sha256=lb.METHOD_CONFIGS['E05'][1],
                                  identity_extra={'fixture': 'INTERFACE_FIXTURE_NOT_PCGAN_ARCHITECTURE',
                                                  'training_guards': guards, 'tensor2im_segment_sha256': seg_sha})
    xs = torch.from_numpy(pa.canonical_input(face))[None]
    xt = torch.from_numpy(pa.canonical_input(rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)))[None]
    m1, m2 = adapter.mix(xs, xt, 1234), adapter.mix(xs, xt, 1234)
    m3 = adapter.mix(xs, xt, 1235)
    with torch.no_grad():
        sp, gl = E(xs)
        _, glt = E(xt)
        u1, u2 = G(sp, glt), G(sp, glt)
    g('seed_fixes_noise', torch.equal(m1, m2) and not torch.equal(m1, m3))
    g('fresh_noise_each_forward_without_seed', not torch.equal(u1, u2))
    faces = Path(args.work) / 'faces'
    samples, relation = synthetic_faces(faces, 12, 11)
    auth = lb.fixture_authorization('e05_cpu')
    positions = [0, 3, 4, 7, 11]
    pop = lb.pair_population('E05', FIXTURE_SEED, 'SYNTHETIC_FIXTURE', relation, positions=positions)
    prov_a = lb.write_bank(adapter, pop, auth, Path(args.work) / 'E05_A', reader=lb.TrainFaceReader(faces, samples))
    adapter.encoder_forwards = adapter.generator_forwards = 0
    prov_b = lb.write_bank(adapter, lb.pair_population('E05', FIXTURE_SEED, 'SYNTHETIC_FIXTURE', relation,
                                                       positions=positions), auth, Path(args.work) / 'E05_B',
                           reader=lb.TrainFaceReader(faces, samples))
    ra = bank_gates(g, 'E05', Path(args.work) / 'E05_A', Path(args.work) / 'E05_B', 'E05', len(positions),
                    lambda d: lb.write_bank(adapter, lb.pair_population('E05', FIXTURE_SEED, 'SYNTHETIC_FIXTURE',
                                                                        relation, positions=[1]), auth, d,
                                            reader=lb.TrainFaceReader(faces, samples)))
    g('pair_lineage_and_positions', [r['row_index'] for r in ra['rows']] == positions and all(
        r['pair_id'] == relation[r['row_index']][0] and r['target_live_id'] == relation[r['row_index']][1]
        and r['source_spoof_id'] == relation[r['row_index']][2]
        and r['source_face_sha256'] == samples[r['source_spoof_id']]['face_png_sha256'] for r in ra['rows']))
    g('pair_seeds', all(r['deterministic_noise_seed'] == lb.e05_pair_seed(FIXTURE_SEED, r['pair_id'])
                        for r in ra['rows']))
    g('o_noatime_reads', prov_a['reader']['atime_checked_reads'] == 2 * len(positions)
      and prov_a['reader']['splits_allowlisted'] == ['TRAIN'])
    g('one_generator_forward_per_pair', prov_b['adapter_finalize']['generator_forwards'] == len(positions)
      and prov_b['adapter_finalize']['reconstruction_forwards'] == 0)
    # order independence: a single-pair bank reproduces the row; an independent recompute matches
    lb.write_bank(adapter, lb.pair_population('E05', FIXTURE_SEED, 'SYNTHETIC_FIXTURE', relation, positions=[7]),
                  auth, Path(args.work) / 'E05_single', reader=lb.TrainFaceReader(faces, samples))
    g('order_independent', (Path(args.work) / 'E05_A/bank/PTR000008.png').read_bytes() ==
      (Path(args.work) / 'E05_single/bank/PTR000008.png').read_bytes())
    from PIL import Image
    pid, live, spoof, _ = relation[4]
    src = np.array(Image.open(faces / 'casia_fasd' / (spoof + '.png')))
    tgt = np.array(Image.open(faces / 'casia_fasd' / (live + '.png')))
    with torch.no_grad():
        s_sp, _ = E(torch.from_numpy(pa.canonical_input(src))[None])
        _, t_gl = E(torch.from_numpy(pa.canonical_input(tgt))[None])
        torch.manual_seed(lb.e05_pair_seed(FIXTURE_SEED, pid))
        out = G(s_sp, t_gl)
    row = next(r for r in ra['rows'] if r['pair_id'] == pid)
    g('independent_recompute', lb.pixel_sha256(lb.reference_e05_uint8(out[0].numpy())) == row['output_pixel_sha256'])
    g('no_optimizer_backward_save', all(v == 0 for v in guards.values()), dict(guards))
    try:
        torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))])
        blocked = False
    except lb.TrainingForbidden:
        blocked = True
    g('optimizer_construction_forbidden', blocked)

    def inspect(run_root, before):
        return pa.checkpoint_structure(torch, before['files']['pth']['path'])
    real = real_checkpoints(g, 'E05', args.runtime_root, inspect)
    for seed, rec in real.items():
        s = rec['structure']
        g('E05_seed%d_checkpoint_structure' % seed, s['format'] == pa.FORMAT and s['benchmark_iteration'] == 4000
          and all(v['all_finite'] and v['dtypes'] == ['torch.float32'] for v in s['models'].values()), s)
    g('real_forward_not_run_on_cpu', True, 'StyleGAN2 custom CUDA kernels: real E05 forward is S7 GPU work')
    return {'bank_A': prov_a['manifest_sha256'], 'bank_B': prov_b['manifest_sha256'], 'real_checkpoints': real,
            'tensor2im_segment_sha256': seg_sha, 'fixture': 'INTERFACE_FIXTURE_NOT_PCGAN_ARCHITECTURE',
            'torch': torch.__version__}


# =========================================================================================== E03 STDN (TF1)
def run_e03(args, g):
    from methods.stdn import bank_adapter as sa
    errs = sa.precision_environment_errors(os.environ, gpu=False)
    g('precision_env_cpu', not errs, errs)
    import tensorflow as tf
    g('cuda_hidden', not tf.test.is_gpu_available() and os.environ.get('CUDA_VISIBLE_DEVICES') == '')
    csha = common_gates(g, 'STDN', ('E03',))
    src = sa.verify_synthesis_lines()
    g('synthesis_and_conversion_lines_exact', True, sorted(sa.SYNTHESIS_LINES))
    work = Path(args.work)
    # fixture weights: random init under a fixed graph seed, saved, then restored into a fresh inference graph
    fx = work / 'E03_fixture'
    fx.mkdir()
    tf.reset_default_graph()
    tf.set_random_seed(FIXTURE_SEED)
    gr = sa.STDNGraph(tf, src['root'])
    with sa.make_session(tf) as s0:
        s0.run(tf.global_variables_initializer())
        for v in tf.global_variables():                       # non-trivial BN moving statistics for test mode
            if 'moving_mean' in v.name:
                s0.run(v.assign(tf.random.normal(v.shape, stddev=0.05, seed=FIXTURE_SEED)))
            if 'moving_variance' in v.name:
                s0.run(v.assign(tf.random.uniform(v.shape, 0.5, 1.5, seed=FIXTURE_SEED)))
        values0 = {v.name: s0.run(v) for v in tf.global_variables()}
        prefix = gr.saver.save(s0, str(fx / 'fixture-ckpt'), write_meta_graph=False)
    fixture_files = {p.name: lb.sha256_file(p) for p in sorted(fx.iterdir())}
    tf.reset_default_graph()
    graph = sa.STDNGraph(tf, src['root'])
    g('graph_structure', graph.structure['ema_variables'] == 0 and graph.structure['update_ops'] == 0
      and not graph.structure['mutating_reachable_ops'] and not graph.structure['random_reachable_ops'],
      graph.structure)
    coverage = graph.checkpoint_coverage(prefix)
    sess = sa.make_session(tf)
    graph.saver.restore(sess, prefix)
    tf.get_default_graph().finalize()
    g('raw_variable_restore_exact', all(np.array_equal(sess.run(v), values0[v.name]) for v in graph.variables))
    landmarks = sa.make_landmarks()
    faces = work / 'faces'
    samples, relation = synthetic_faces(faces, 6, 5)
    lrng = np.random.default_rng(9)
    template = np.stack([np.linspace(60, 196, 68), 128 + 50 * np.sin(np.linspace(0, 6.0, 68))], 1)
    lms = {sid: (template + lrng.normal(0, 4, (68, 2))).astype('<f4') for sid in samples}
    adapter = sa.STDNBankAdapter(tf, sess, graph, landmarks, lambda sid: lms[sid], FIXTURE_SEED, is_fixture=True,
                                 checkpoint_identity_sha256=lb.sha256_bytes(lb.canonical_json(fixture_files).encode()),
                                 contract_sha256=csha, config_sha256=lb.METHOD_CONFIGS['E03'][1],
                                 identity_extra={'fixture_checkpoint': fixture_files, 'coverage': coverage,
                                                 'framework': 'tensorflow ' + tf.__version__})
    auth = lb.fixture_authorization('e03_cpu')
    positions = [0, 2, 5]
    mk = lambda: lb.pair_population('E03', FIXTURE_SEED, 'SYNTHETIC_FIXTURE', relation, positions=positions)  # noqa
    prov_a = lb.write_bank(adapter, mk(), auth, work / 'E03_A', reader=lb.TrainFaceReader(faces, samples))
    prov_b = lb.write_bank(adapter, mk(), auth, work / 'E03_B', reader=lb.TrainFaceReader(faces, samples))
    ra = bank_gates(g, 'E03', work / 'E03_A', work / 'E03_B', 'E03', len(positions),
                    lambda d: lb.write_bank(adapter, lb.pair_population('E03', FIXTURE_SEED, 'SYNTHETIC_FIXTURE',
                                                                        relation, positions=[1]), auth, d,
                                            reader=lb.TrainFaceReader(faces, samples)))
    g('o_noatime_reads', prov_a['reader']['atime_checked_reads'] == 2 * len(positions))
    g('not_stochastic', all(r['deterministic_noise_seed'] is None for r in ra['rows']))
    # independent checks on one row: landmarks unflipped /256, offset direction (spoof, live), in-graph conversion
    pid, live, spoof, _ = relation[2]
    pts_li, pts_sp = adapter.points(live), adapter.points(spoof)
    g('landmarks_unflipped_div256', np.array_equal(pts_li, lms[live].astype(np.float32) / 256)
      and np.array_equal(pts_sp, lms[spoof].astype(np.float32) / 256))
    li = sa.decode_face((faces / 'casia_fasd' / (live + '.png')).read_bytes())
    sp = sa.decode_face((faces / 'casia_fasd' / (spoof + '.png')).read_bytes())
    from PIL import Image
    g('input_scale', li.dtype == np.float32 and float(li.max()) <= 1.0 and np.array_equal(
        li, np.asarray(Image.open(faces / 'casia_fasd' / (live + '.png')), np.float32) / 255))
    reg = graph.warp.generate_offset_map(pts_sp, pts_li).astype(np.float32)
    synth, px = adapter.synthesize(li, sp, pts_li, pts_sp)
    row = next(r for r in ra['rows'] if r['pair_id'] == pid)
    g('conversion_in_graph_equals_reference', np.array_equal(px, lb.reference_e03_uint8(synth))
      and lb.pixel_sha256(px) == row['output_pixel_sha256'])
    g('offset_map_finite', bool(np.isfinite(reg).all()) and reg.shape == (256, 256, 3))
    g('synthesis_float_range_recorded', row['raw_output_min'] is not None and row['raw_output_max'] is not None,
      [row['raw_output_min'], row['raw_output_max']])
    g('fixture_checkpoint_unchanged', {p.name: lb.sha256_file(p) for p in sorted(fx.iterdir())} == fixture_files)
    try:
        tf.Variable(0.0)
        added = True
    except RuntimeError:
        added = False
    g('graph_finalized_no_new_ops', not added)
    sess.close()

    def inspect(run_root, before):
        tf.reset_default_graph()
        gr2 = sa.STDNGraph(tf, src['root'])
        return gr2.checkpoint_coverage(Path(run_root) / 'checkpoints' / 'ckpt-50')
    real = real_checkpoints(g, 'E03', args.runtime_root, inspect)
    for seed, rec in real.items():
        s = rec['structure']
        g('E03_seed%d_checkpoint_covers_graph' % seed, s['graph_variables_covered'] == len(graph.variables)
          and s['ema_restored'] == 0, s)
    return {'bank_A': prov_a['manifest_sha256'], 'bank_B': prov_b['manifest_sha256'], 'real_checkpoints': real,
            'graph': graph.structure, 'tensorflow': tf.__version__, 'fixture_checkpoint': fixture_files}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('family', choices=('dsdg', 'e05', 'e03'))
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
        evidence = {'dsdg': run_dsdg, 'e05': run_e05, 'e03': run_e03}[args.family](args, g)
        error = None
    except Exception as exc:  # recorded, never hidden
        import traceback
        evidence, error = {}, traceback.format_exc()
        g('harness_completed', False, '%s: %s' % (type(exc).__name__, exc))
    g('zero_opens_under_runtime_data', OPENS['runtime_data'] == 0, OPENS)
    rec = dict(g.record(), schema='learned_bank_fixture_qualification_v1', family=args.family,
               label='SYNTHETIC_FIXTURE_QUALIFICATION_NOT_S7_NOT_BENCHMARK', fixture_seed=FIXTURE_SEED,
               git_head=lb._git_head(), environment=lb.environment_record(), started_utc=started,
               wall_seconds=round(time.monotonic() - t0, 1), evidence=evidence, error=error,
               gpu_bitwise_reproducibility_proven=False, work=str(work))
    out.write_text(json.dumps(rec, indent=1, sort_keys=True, default=str) + '\n')
    print(json.dumps({'family': args.family, 'gates_passed': rec['gates_passed'], 'gates_total': rec['gates_total'],
                      'failed': rec['failed_gates']}))
    return 0 if not rec['failed_gates'] else 1


if __name__ == '__main__':
    sys.exit(main())
