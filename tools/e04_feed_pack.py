#!/usr/bin/env python3
"""Build / verify the E04 feed pack v1 (methods/physics_std/feed_pack.py) on the GPU host.

  python -B tools/e04_feed_pack.py build --prepass-manifest-sha256 <sha>   # -> <rt>/data/processed/e04_feed_pack_v1
  python -B tools/e04_feed_pack.py verify --root <pack dir> --manifest-sha256 <sha>

Inputs: the hash-verified E03 feed pack (TRAIN rows only are used) and the frozen geometry pre-pass. Stdlib + numpy
(Python 3.8 or 3.12). Refuses an existing output directory; makes the pack read-only after verification.
"""
import argparse
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RUNTIME = Path('/home/student20261/workdir/GPAT_TransferBench_runtime')

from methods.physics_std import feed_pack, prepass  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build')
    b.add_argument('--prepass-manifest-sha256', required=True)
    v = sub.add_parser('verify')
    v.add_argument('--root', required=True)
    v.add_argument('--manifest-sha256', required=True)
    args = ap.parse_args(argv)
    if args.cmd == 'verify':
        p = feed_pack.load(args.root, args.manifest_sha256)
        print(json.dumps({'verified': True, 'manifest_sha256': p.manifest_sha256, 'counts': p.manifest['counts'],
                          'payloads': p.manifest['payloads']}, sort_keys=True))
        return 0
    from methods.stdn import feed_pack as e03_pack
    from methods.common.runlog import git_commit, git_dirty
    e03 = e03_pack.load(RUNTIME / 'data/processed' / prepass.E03_PACK_DIRNAME, prepass.E03_PACK_MANIFEST_SHA256)
    out = RUNTIME / 'data/processed' / feed_pack.DIRNAME
    src = RUNTIME / 'data/processed/e04_geometry_prepass_v1'
    provenance = {'builder': 'tools/e04_feed_pack.py', 'builder_sha256': prepass.sha256_file(Path(__file__)),
                  'verifier_sha256': prepass.sha256_file(ROOT / 'methods/physics_std/feed_pack.py'),
                  'repository_commit': git_commit(), 'repository_dirty': git_dirty(),
                  'e03_feed_pack_manifest_sha256': prepass.E03_PACK_MANIFEST_SHA256,
                  'pairs_train_v1_sha256': prepass.PAIRS_TRAIN_SHA256,
                  'owner_resolution_sha256': prepass.RULING[1], 'prepass_dir': str(src)}
    sha = feed_pack.build(out, e03_pack=e03, prepass_dir=src, prepass_manifest_sha256=args.prepass_manifest_sha256,
                          provenance=provenance)
    feed_pack.load(out, sha)
    for p in out.iterdir():
        os.chmod(p, 0o444)
    os.chmod(out, 0o555)
    print(json.dumps({'built': str(out), 'manifest_sha256': sha}, sort_keys=True))
    return 0


if __name__ == '__main__':
    sys.exit(main())
