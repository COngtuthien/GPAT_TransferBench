#!/usr/bin/env python3
"""E04 fixed geometry pre-pass CLI (owner resolution D1-D3; methods/physics_std/prepass.py).

GPU host, gpat-m6-e04-geometry, CPU only, from the repository root:
  GPAT_E04_GEOMETRY_RUNTIME=$PWD/environments/e04_geometry.runtime.json CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
      python -B tools/e04_geometry_prepass.py run --out <rt>/qualification/e04_prepass/<name>
  python -B tools/e04_geometry_prepass.py compare <dir_a> <dir_b>
  python -B tools/e04_geometry_prepass.py freeze --src <dir_a> --compare <dir_b>     # -> <rt>/data/processed/<FROZEN>

`run` refuses an existing output directory and never writes elsewhere. `freeze` copies an accepted (full-population)
run only if a second independent run is byte-identical, verifies the copy and makes it read-only.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RUNTIME = Path('/home/student20261/workdir/GPAT_TransferBench_runtime')
FROZEN = 'e04_geometry_prepass_v1'

from methods.physics_std import prepass  # noqa: E402


def compare(a, b):
    a, b = Path(a), Path(b)
    names = sorted(p.name for p in a.iterdir())
    out = {'files_a': names, 'files_b': sorted(p.name for p in b.iterdir()), 'differing': []}
    for n in names:
        if n == prepass.MANIFEST:
            continue
        if prepass.sha256_file(a / n) != prepass.sha256_file(b / n):
            out['differing'].append(n)
    ma, mb = (json.loads((x / prepass.MANIFEST).read_text()) for x in (a, b))
    strip = lambda m: {k: v for k, v in m.items() if k not in ('seconds',)}  # noqa: E731
    out['manifest_equal_except_seconds'] = strip(ma) == strip(mb)
    out['identical'] = out['files_a'] == out['files_b'] and not out['differing'] and out['manifest_equal_except_seconds']
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--out', required=True)
    r.add_argument('--limit', type=int, help='qualification smoke only; output is marked not acceptable')
    c = sub.add_parser('compare')
    c.add_argument('a')
    c.add_argument('b')
    f = sub.add_parser('freeze')
    f.add_argument('--src', required=True)
    f.add_argument('--compare', required=True)
    args = ap.parse_args(argv)
    if args.cmd == 'run':
        out = Path(args.out).resolve()
        if not str(out).startswith(str(RUNTIME / 'qualification') + '/'):
            raise SystemExit('pre-pass runs write only under <runtime_root>/qualification/')
        res = prepass.run(out, root=ROOT, runtime_root=RUNTIME, limit=args.limit)
        print(json.dumps(res, sort_keys=True))
        return 0
    if args.cmd == 'compare':
        res = compare(args.a, args.b)
        print(json.dumps(res, sort_keys=True))
        return 0 if res['identical'] else 1
    src = Path(args.src)
    m = json.loads((src / prepass.MANIFEST).read_text())
    cmp_ = compare(src, args.compare)
    if not (m['acceptable'] and cmp_['identical']):
        raise SystemExit('freeze refused: the source must be a full run byte-identical to an independent second run')
    dst = RUNTIME / 'data' / 'processed' / FROZEN
    if dst.exists():
        raise SystemExit('freeze refused: %s exists (frozen assets are never overwritten)' % dst)
    dst.mkdir(parents=True)
    for p in sorted(src.iterdir()):
        shutil.copyfile(p, dst / p.name)
    manifest_sha = prepass.sha256_file(dst / prepass.MANIFEST)
    prepass.load(dst, manifest_sha)
    for p in dst.iterdir():
        os.chmod(p, 0o444)
    os.chmod(dst, 0o555)
    print(json.dumps({'frozen': str(dst), 'manifest_sha256': manifest_sha, 'compare': cmp_}, sort_keys=True))
    return 0


if __name__ == '__main__':
    sys.exit(main())
