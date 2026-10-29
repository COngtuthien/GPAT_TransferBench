"""E03 three-seed closure, host phase: read-only recheck of the three E03 STDN scientific run roots (ckpt-50 file
sha256/size, metrics sha256, LC_ALL=C path/size/sha256 run-root listing digest and file count, stable across the
recheck, resolved_config.yaml text) plus the final host state (no E03 or other method process, GPU scientific
checkout HEAD/branch/cleanliness, GPU compute apps, E03 tmux panes and tmux sessions, kernel boot id and boot time).
Files are opened 'rb' only; nothing is deserialised, written or deleted.
Run on the GPU host: `python -I -B - < tools/e03_family_host_recheck.py`.
"""
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

RUNTIME = Path('/home/student20261/workdir/GPAT_TransferBench_runtime')
REPO = Path('/home/student20261/workdir/GPAT_TransferBench')
BASE = RUNTIME / 'runs/m6/E03'
SEEDS = (42, 1337, 2026)
FINAL_FILES = tuple('checkpoints/ckpt-50' + s for s in ('.data-00000-of-00001', '.index', '.meta'))


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


def sha_listing_digest(root):
    """LC_ALL=C sorted `<path> <size> <sha256>` lines; digest = sha256 of the newline-terminated lines."""
    lines = [f'{p} {s} {sha256_file(Path(root) / p)}' for p, s, _ in tree(root)]
    return hashlib.sha256(''.join(f'{x}\n' for x in lines).encode()).hexdigest(), len(lines)


def sh(*a, cwd=None):
    return subprocess.run(list(a), capture_output=True, text=True, cwd=cwd).stdout


def main():
    out = {'utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'), 'parent_listing': sorted(os.listdir(BASE)),
           'seeds': {}}
    for s in SEEDS:
        root = BASE / f'seed_{s}'
        before = tree(root)
        rec = {'final_checkpoint': {c: {'sha256': sha256_file(root / c), 'bytes': (root / c).stat().st_size}
                                    for c in FINAL_FILES},
               'metrics_jsonl_sha256': sha256_file(root / 'metrics.jsonl'),
               'resolved_config_yaml': (root / 'resolved_config.yaml').read_bytes().decode(),
               'resolved_config_sha256': sha256_file(root / 'resolved_config.yaml')}
        digest, count = sha_listing_digest(root)
        rec.update(sha_listing_digest=digest, file_count=count, listing_stable_during_recheck=tree(root) == before)
        out['seeds'][str(s)] = rec
    procs = [x.strip() for x in sh('ps', '-eo', 'pid,args').splitlines()[1:]
             if x.split()[0] != str(os.getpid())]                  # this recheck runs under a gpat-* python too
    out['host_state'] = {
        'hostname': sh('hostname').strip(),
        'gpu': sh('nvidia-smi', '--query-gpu=name,driver_version,uuid', '--format=csv,noheader').strip(),
        'gpu_compute_apps': sh('nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                               '--format=csv,noheader').strip().splitlines(),
        'method_processes': [x for x in procs if 'python' in x.split(' ', 2)[1] and
                             any(k in x for k in ('tools/run_e0', 'run_e0', 'bank', 'downstream', 'gpat'))],
        'git_head': sh('git', 'rev-parse', 'HEAD', cwd=REPO).strip(),
        'git_branch': sh('git', 'branch', '--show-current', cwd=REPO).strip(),
        'git_status_porcelain': sh('git', 'status', '--porcelain', cwd=REPO),
        'e03_tmux_panes': {t: sh('tmux', 'list-panes', '-t', t, '-F', '#{pane_dead} #{pane_dead_status}').strip()
                           for t in (f'e03_stdn_seed{s}' for s in SEEDS)},
        'tmux_sessions': sorted(sh('tmux', 'ls', '-F', '#{session_name}').splitlines()),
        'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip().replace('-', ''),
        'btime': int(next(x.split()[1] for x in Path('/proc/stat').read_text().splitlines() if x.startswith('btime '))),
        'runs_m6_listing': sorted(os.listdir(RUNTIME / 'runs/m6'))}
    json.dump(out, sys.stdout, sort_keys=True)


if __name__ == '__main__':
    main()
