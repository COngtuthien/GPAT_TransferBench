"""E06b three-seed closure, host phase: read-only recheck of the three successful E06b run roots (final netG
sha256/size, metrics sha256, C-collation run-root listing digests -- path/size/mtime and path/size/sha256 -- and file
count, resolved_config.yaml text) plus the archived seed42 attempt-1 root (path/size/sha256 digest only). Files are
opened 'rb' only; nothing is unpickled or written.
Run on the GPU host: `python -I -B - < tools/e06b_family_host_recheck.py`.
"""
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

RUNTIME = Path('/home/student20261/workdir/GPAT_TransferBench_runtime')
BASE = RUNTIME / 'runs/m6/E06b'
ATTEMPT1 = RUNTIME / 'diagnostics/m6/E06B_SEED42_ATTEMPT1_d3c0ddd5dc82a4a1_CHECKPOINT_WRITER_FAILURE'
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


def tree(root):
    out = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            p = Path(dirpath) / name
            st = p.stat()
            out.append((str(p.relative_to(root)), st.st_size, st.st_mtime_ns))
    return sorted(out)


def listing_digest(root):
    """`find . -type f -printf "%P %s %T@\\n" | LC_ALL=C sort | sha256sum` (same convention as the seed closures)."""
    lines = sorted(f'{p} {s} {m // 10**9}.{m % 10**9:09d}0' for p, s, m in tree(root))
    return hashlib.sha256(''.join(f'{x}\n' for x in lines).encode()).hexdigest(), len(lines)


def sha_listing_digest(root):
    """LC_ALL=C sorted `<path> <size> <sha256>` lines; digest = sha256 of the newline-terminated lines."""
    lines = [f'{p} {s} {sha256_file(Path(root) / p)}' for p, s, _ in tree(root)]
    return hashlib.sha256(''.join(f'{x}\n' for x in lines).encode()).hexdigest(), len(lines)


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
        sha_digest, sha_count = sha_listing_digest(root)
        rec.update(listing_digest=before[0], file_count=before[1], sha_listing_digest=sha_digest,
                   sha_listing_count=sha_count, listing_stable_during_recheck=listing_digest(root) == before)
        out['seeds'][str(s)] = rec
    a1_digest, a1_count = sha_listing_digest(ATTEMPT1) if ATTEMPT1.is_dir() else (None, 0)
    out['attempt1_archive'] = {'path': str(ATTEMPT1), 'exists': ATTEMPT1.is_dir(), 'sha_listing_digest': a1_digest,
                               'file_count': a1_count, 'under_runs': str(ATTEMPT1).startswith(f'{RUNTIME}/runs/')}
    json.dump(out, sys.stdout, sort_keys=True)


if __name__ == '__main__':
    main()
