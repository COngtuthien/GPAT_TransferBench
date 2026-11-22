#!/usr/bin/env python3
"""Learned-method bank CLI (E03 / E05 / E06b / E06c), fail-closed.

    python -B tools/learned_bank.py gates   --method M [--seed S]          # read-only authority report (any host)
    python -B tools/learned_bank.py storage-preflight                     # read-only HDD bank-root checks
    python -B tools/learned_bank.py s7      --method M --label A|B --out-root DIR --runtime-root RT
    python -B tools/learned_bank.py s7-compare --method M --out-root DIR
    python -B tools/learned_bank.py production --method M --seed S --runtime-root RT

`s7` and `production` evaluate their authority gates FIRST and exit 3 with a JSON status before any framework import,
checkpoint open or face read:
  LEARNED_BANK_BLOCKED_BY_SCIENTIFIC_AUTHORITY   contract absent / unapproved / not the approved decisions
  LEARNED_BANK_BLOCKED_BY_E04_COMPLETION         E04 seed-2026 completion audit PASS + family closure absent (C-Q5)
  LEARNED_BANK_BLOCKED_BY_S7_LAUNCH_AUTHORITY    no owner S7 launch authority record
  LEARNED_BANK_BLOCKED_BY_PRODUCTION_AUTHORITY   no per-family owner production authority binding a passed S7 record
  LEARNED_BANK_BLOCKED_BY_STORAGE_AUTHORITY      no activated, live-verified HDD storage authority
At d0b4e4e none of the S7 / production authority records exists, so both commands refuse. Fixture qualification is a
separate harness (tools/learned_bank_fixture_qualification.py) and can never stand in for S7 or production.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from methods.common import learned_bank as lb  # noqa: E402

EXIT_BLOCKED = 3


def emit(obj):
    sys.stdout.write(json.dumps(obj, sort_keys=True, default=str) + '\n')
    sys.stdout.flush()


def gate_report(method, seed=None):
    out = {'method': method}
    try:
        _, sha = lb.load_generation_contract()
        out['contract'] = {'ok': True, 'sha256': sha}
    except lb.LearnedBankBlocked as exc:
        out['contract'] = {'ok': False, 'status': exc.status, 'reason': str(exc)}
    out['e04'] = lb.e04_completion_state()
    for name, fn in (('s7', lambda: lb.load_s7_authority(method)),
                     ('production', lambda: lb.load_production_authority(method, seed or 42))):
        try:
            out[name] = {'ok': True, 'authorization': fn().record()}
        except lb.LearnedBankBlocked as exc:
            out[name] = {'ok': False, 'status': exc.status, 'reason': str(exc)}
    try:
        lb.load_storage_authority()
        out['storage_authority'] = {'ok': True}
    except lb.LearnedBankBlocked as exc:
        out['storage_authority'] = {'ok': False, 'status': exc.status, 'reason': str(exc)}
    return out


# ------------------------------------------------------------------ real populations (called only after the gates)
def e03_inputs(runtime_root, seed, kind):
    from methods.stdn import production_io as pio
    from methods.stdn.bank_adapter import load_train_pack
    pack = load_train_pack(pio.pack_root(runtime_root), pio.PACK_MANIFEST_SHA256)
    pop = lb.pair_population('E03', seed, kind, pack['relation'],
                             provenance={'relation_source': 'e03_feed_pack_v1 pairs_train.json',
                                         'pack_manifest_sha256': pack['manifest_sha256'], 'pairs_val_opened': False})
    used = sorted({r.target_live_id for r in pop.rows} | {r.source_spoof_id for r in pop.rows})
    samples = {sid: {k: pack['samples'][sid][k] for k in ('split', 'dataset', 'face_png_sha256')} for sid in used}
    reader = lb.TrainFaceReader(Path(runtime_root) / 'data' / 'processed' / 'faces_256', samples)
    return pop, reader, (lambda sid: pack['samples'][sid]['landmarks_px256'])


def e05_inputs(runtime_root, seed, kind):
    from methods.pcgan import production_io as pio
    rows, _, evidence = pio.read_pair_manifest()
    relation = [(r['pair_id'], r['target_live_id'], r['source_spoof_id'], r['dataset']) for r in rows]
    pop = lb.pair_population('E05', seed, kind, relation, provenance={'relation_source': lb.PAIRS_PATH,
                                                                      'relation_evidence': evidence})
    used = {}
    for r in pop.rows:
        used[r.target_live_id] = r.dataset
        used[r.source_spoof_id] = r.dataset
    reader = lb.TrainFaceReader(Path(runtime_root) / 'data' / 'processed' / 'faces_256',
                                lb.train_face_allowlist_parquet(used))
    return pop, reader


def real_adapter(method, seed, runtime_root, landmark_source=None):
    run_root = Path(runtime_root) / 'runs' / 'm6' / method / ('seed_%d' % seed)
    if method == 'E03':
        import tensorflow as tf
        from methods.stdn.bank_adapter import load_real
        tf.reset_default_graph()
        return load_real(tf, seed, run_root, landmark_source)
    import torch
    if method == 'E05':
        from methods.pcgan import production_io as pio
        from methods.pcgan.bank_adapter import load_real
        return load_real(torch, seed, run_root, pio.build_root(runtime_root))
    from methods.dsdg.bank_adapter import load_real
    return load_real(torch, method, seed, run_root)


def generate_real(auth, method, seed, runtime_root, run_dir, kind):
    landmark_source = reader = None
    if method == 'E03':
        pop, reader, landmark_source = e03_inputs(runtime_root, seed, kind)
    elif method == 'E05':
        pop, reader = e05_inputs(runtime_root, seed, kind)
    else:
        pop = lb.dsdg_population(method, seed, lb.TOY_N if kind == 'S7_TOY_POPULATION' else lb.N_SYN, kind)
    adapter, ckpt = real_adapter(method, seed, runtime_root, landmark_source)
    prov = lb.write_bank(adapter, pop, auth, run_dir, reader=reader)
    lb.recheck_checkpoint_unchanged(ckpt)
    prov['checkpoint_unchanged_after'] = True
    if method != 'E03':                    # profiling gate (S7 run A): peak allocator memory of this process
        import torch
        prov['peak_cuda_memory_allocated_bytes'] = torch.cuda.max_memory_allocated()
    else:
        prov['peak_cuda_memory_allocated_bytes'] = None   # TF1: sampled with nvidia-smi during run A (S7 plan)
    return prov


# ------------------------------------------------------------------ commands
def cmd_s7(args):
    try:
        auth = lb.load_s7_authority(args.method)
    except lb.LearnedBankBlocked as exc:
        emit({'command': 's7', 'method': args.method, 'status': exc.status, 'reason': str(exc)})
        return EXIT_BLOCKED
    out_root = Path(args.out_root)
    rt = Path(args.runtime_root)
    lb.require(out_root.is_absolute() and out_root.is_dir() and not out_root.is_symlink(), 'out root')
    lb.require(not lb._is_within(out_root.resolve(), rt.resolve()), 'S7 roots lie outside the runtime root')
    label_dir = out_root / ('run_%s' % args.label)
    os.mkdir(str(label_dir))                       # fresh root per run label; an existing one is refused
    method_dir = label_dir / args.method
    os.mkdir(str(method_dir))
    results = {}
    for seed in lb.SEEDS:
        prov = generate_real(auth, args.method, seed, rt, method_dir / ('seed_%d' % seed), 'S7_TOY_POPULATION')
        results[seed] = {k: prov.get(k) for k in ('status', 'rows', 'outputs', 'failure_count', 'manifest_sha256',
                                                  'checkpoint_unchanged_after', 'peak_cuda_memory_allocated_bytes')}
    emit({'command': 's7', 'method': args.method, 'label': args.label, 'results': results})
    return 0


def cmd_s7_compare(args):
    out = {}
    for seed in lb.SEEDS:
        a = Path(args.out_root) / 'run_A' / args.method / ('seed_%d' % seed)
        b = Path(args.out_root) / 'run_B' / args.method / ('seed_%d' % seed)
        ra = lb.read_bank(a, expect_method=args.method, expect_seed=seed, expect_mode='S7_TOY', expect_rows=lb.TOY_N,
                          expect_checkpoint_identity=lb.checkpoint_identity_sha256(args.method, seed))
        rb = lb.read_bank(b, expect_method=args.method, expect_seed=seed, expect_mode='S7_TOY', expect_rows=lb.TOY_N,
                          expect_checkpoint_identity=lb.checkpoint_identity_sha256(args.method, seed))
        out[seed] = {'run_A_reader_ok': ra['ok'], 'run_B_reader_ok': rb['ok'], 'failures': [ra.get('failures'),
                     rb.get('failures')], 'replay': lb.compare_runs(a, b)}
    emit({'command': 's7-compare', 'method': args.method, 'seeds': out})
    return 0 if all(v['run_A_reader_ok'] and v['run_B_reader_ok'] and v['replay']['bitwise_identical']
                    and v['failures'] == [0, 0] for v in out.values()) else 1


def cmd_production(args):
    try:
        auth = lb.load_production_authority(args.method, args.seed)
    except lb.LearnedBankBlocked as exc:
        emit({'command': 'production', 'method': args.method, 'seed': args.seed, 'status': exc.status,
              'reason': str(exc)})
        return EXIT_BLOCKED
    storage = lb.load_storage_authority()
    run_dir = lb.production_bank_dir(args.method, args.seed, storage)
    method_dir = run_dir.parent
    if not method_dir.exists():
        os.mkdir(str(method_dir), 0o700)
    prov = generate_real(auth, args.method, args.seed, Path(args.runtime_root), run_dir,
                         'PRODUCTION_FROZEN_TRAIN_POPULATION')
    emit({'command': 'production', 'method': args.method, 'seed': args.seed,
          'result': {k: prov.get(k) for k in ('status', 'rows', 'outputs', 'failure_count', 'manifest_sha256')}})
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    g = sub.add_parser('gates')
    g.add_argument('--method', required=True, choices=lb.METHODS)
    g.add_argument('--seed', type=int, choices=lb.SEEDS)
    sub.add_parser('storage-preflight')
    s = sub.add_parser('s7')
    s.add_argument('--method', required=True, choices=lb.METHODS)
    s.add_argument('--label', required=True, choices=('A', 'B'))
    s.add_argument('--out-root', required=True)
    s.add_argument('--runtime-root', required=True)
    c = sub.add_parser('s7-compare')
    c.add_argument('--method', required=True, choices=lb.METHODS)
    c.add_argument('--out-root', required=True)
    p = sub.add_parser('production')
    p.add_argument('--method', required=True, choices=lb.METHODS)
    p.add_argument('--seed', required=True, type=int, choices=lb.SEEDS)
    p.add_argument('--runtime-root', required=True)
    args = ap.parse_args(argv)
    if args.cmd == 'gates':
        emit(gate_report(args.method, args.seed))
        return 0
    if args.cmd == 'storage-preflight':
        emit(lb.storage_preflight())
        return 0
    return {'s7': cmd_s7, 's7-compare': cmd_s7_compare, 'production': cmd_production}[args.cmd](args)


if __name__ == '__main__':
    sys.exit(main())
