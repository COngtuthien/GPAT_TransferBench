"""E06c three-seed closure, host phase: read-only recheck of the three E06c run roots (final netG sha256/size,
metrics sha256, C-collation run-root listing digest and file count, resolved_config.yaml text). Files are opened
'rb' only; nothing is unpickled or written. Run on the GPU host: `python -I - < tools/e06c_family_host_recheck.py`.
"""
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

BASE = Path('/home/student20261/workdir/GPAT_TransferBench_runtime/runs/m6/E06c')
SEEDS = (42, 1337, 2026)
FINAL_NETG = 'checkpoints/netG_model_epoch_200_iter_0.pth'


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


def listing_digest(root):
    """`find . -type f -printf "%P %s %T@\\n" | LC_ALL=C sort | sha256sum` (same convention as the seed closures)."""
    lines = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            p = Path(dirpath) / name
            st = p.stat()
            m = st.st_mtime_ns
            lines.append(f'{p.relative_to(root)} {st.st_size} {m // 10**9}.{m % 10**9:09d}0')
    return hashlib.sha256(''.join(f'{x}\n' for x in sorted(lines)).encode()).hexdigest(), len(lines)


def main():
    out = {'utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'), 'parent_listing': sorted(os.listdir(BASE)),
           'seeds': {}}
    for s in SEEDS:
        root = BASE / f'seed_{s}'
        before = listing_digest(root)
        g = root / FINAL_NETG
        rec = {'final_netG_sha256': sha256_file(g), 'final_netG_bytes': g.stat().st_size,
               'metrics_jsonl_sha256': sha256_file(root / 'metrics.jsonl'),
               'resolved_config_yaml': (root / 'resolved_config.yaml').read_bytes().decode(),
               'resolved_config_sha256': sha256_file(root / 'resolved_config.yaml')}
        rec.update(listing_digest=before[0], file_count=before[1],
                   listing_stable_during_recheck=listing_digest(root) == before)
        out['seeds'][str(s)] = rec
    json.dump(out, sys.stdout, sort_keys=True)


if __name__ == '__main__':
    main()
