"""E06b seed42 closure: read-only completion audit of the E06b DSDG-NATIVE / seed 42 scientific run, ATTEMPT 2
(run_id aa96129289d080ba, training authority d9ecbef = checkpoint-context fix, FAITHFUL_OFFICIAL_WITH_RUNTIME_COMPATIBILITY).
Attempt 1 (d3c0ddd5dc82a4a1, 24f7049) is FAILED_TECHNICAL_CHECKPOINT_WRITER_BUG history and is only checked to be
still archived, untouched and outside runs/.

Two phases:

  collect  (GPU host, gpat-m6-e06c env, `python -I - collect --repo <gpu repo> < this file`) -- read-only, prints one
           JSON document. Files are opened 'rb'. There are no E06b runner-state sidecars; the 63 official .pth files
           (whole-module DataParallel pickles) are never unpickled with their classes: data.pkl is walked by a stub
           unpickler that imports nothing, and every float tensor storage (archive/data/<key>) is read as raw bytes
           and checked finite. The run root is listed (path/size/mtime_ns and path/size/sha256) and its small files
           hashed before and after the audit. VAL/TEST/SiW data are never opened: firewall evidence is the run
           manifest/summary/metrics plus a stat-only before/after atime comparison of the faces_256 snapshot taken
           at launch (~/workdir/e06b_s42_r2_atime_{pre,post}.tsv). The native relation and the 8-worker loader draws
           are re-derived independently from split_v1 (TRAIN filter, allowlisted columns) without repo code.
  build    (repository) -- gates the collected document and writes outputs/audit/E06B_SEED42_{COMPLETION.json,
           CHECKPOINT_MANIFEST.txt}. COMPLETION.json embeds the collected document so the build can be re-derived.

No selection, VAL, TEST, bank, downstream or training code is run.
"""
import argparse
import collections
import hashlib
import io
import json
import math
import os
import pickle
import statistics
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

CLASSIFICATION = 'SCIENTIFIC_TRAINING_COMPLETION_AUDIT'
STATUS_OK = 'E06B_SEED42_COMPLETED_CLEANLY'
STATUS_BLOCKED = 'E06B_SEED42_NEEDS_REVIEW'
RUNTIME = '/home/student20261/workdir/GPAT_TransferBench_runtime'
WORKDIR = '/home/student20261/workdir'
EXPECTED = {
    'method_id': 'E06b', 'method_name': 'DSDG-NATIVE', 'seed': 42, 'attempt': 2, 'run_id': 'aa96129289d080ba',
    'git_commit': 'd9ecbef7295fc6b41ced6a4966b49810d36a8c87',
    'config_sha256': '9d665dc2c909d421b8e54964407e27bb40d133f11bceec2e2cb29810f268417f',
    'run_root': f'{RUNTIME}/runs/m6/E06b/seed_42', 'host': 'nd-System-Product-Name',
    'fidelity': 'FAITHFUL_OFFICIAL_WITH_RUNTIME_COMPATIBILITY', 'track': 'B_NATIVE_FULL_SECONDARY',
    'execution_mode': 'GLOBAL_STATISTIC_PRESERVING_MICROBATCH_V2', 'checkpoint_rule': 'OFFICIAL_GENERATOR_EPOCH_200',
    'learning_rate': 2e-4, 'environment_name': 'gpat-m6-e06c',
    'python_executable': '/home/student20261/miniconda3/envs/gpat-m6-e06c/bin/python',
}
ATTEMPT1 = {'run_id': 'd3c0ddd5dc82a4a1', 'authority': '24f7049b894330574b84d11baeb40ed4a7fb3c6b',
            'classification': 'FAILED_TECHNICAL_CHECKPOINT_WRITER_BUG',
            'archive': f'{RUNTIME}/diagnostics/m6/E06B_SEED42_ATTEMPT1_d3c0ddd5dc82a4a1_CHECKPOINT_WRITER_FAILURE',
            'listing_digest_path_size_sha256': '8ce8d5a8be3ec6179ef4299a51ab7903383bb963307e809d50c7e2dd9d91f6ca',
            'file_count': 14}
EPOCHS, STEPS_PER_EPOCH, SPOOF_ROWS, B, WORKERS = 200, 16, 3720, 240, 8
TOTAL_STEPS = EPOCHS * STEPS_PER_EPOCH                       # 3200
CHECKPOINT_EPOCHS = [1] + list(range(10, EPOCHS + 1, 10))    # 21 events
OFFICIAL = (('netE_nir', 'netE_spoof_'), ('netE_vis', 'netE_live_'), ('netG', 'netG_'))
FINAL_NETG = 'checkpoints/netG_model_epoch_200_iter_0.pth'
LOSS_KEYS = ('loss_rec', 'loss_kl', 'loss_mmd', 'loss_ip', 'loss_pair', 'loss_cls', 'loss_ort', 'total_loss')
CLASS_INDEX = {'print': 0, 'replay': 1}
DATASETS = ('casia_fasd', 'msu_mfsd')
SMALL_FILES = ('run_manifest.json', 'run_summary.json', 'checkpoint_index.json', 'resolved_config.yaml',
               'metrics.jsonl', 'stdout.log', 'stderr.log', 'generation_log.jsonl')
LOG_PATTERNS = ('Traceback', 'Error', 'error', 'OutOfMemory', 'out of memory', 'CUDA', 'Killed', 'signal',
                'SIGTERM', 'SIGKILL', 'No space', 'Errno', 'LearnedRunContext', 'has no attribute', 'DataLoader',
                'worker', 'STOP', 'Warning', 'nan', 'inf')
RUN_WINDOW = ('2026-10-21T17:18:10Z', '2026-10-21T20:58:11Z')
ATIME_PRE, ATIME_POST = f'{WORKDIR}/e06b_s42_r2_atime_pre.tsv', f'{WORKDIR}/e06b_s42_r2_atime_post.tsv'
EXIT_FILE = f'{WORKDIR}/e06b_s42_r2.exit'
COMMAND = ('/home/student20261/miniconda3/envs/gpat-m6-e06c/bin/python tools/run_e06b.py --seed 42 '
           '--execution-config configs/execution/m5_gpu_3090.yaml')
EVIDENCE = {'completion': 'outputs/audit/E06B_SEED42_COMPLETION.json',
            'manifest': 'outputs/audit/E06B_SEED42_CHECKPOINT_MANIFEST.txt',
            'report': 'outputs/audit/E06B_SEED42_COMPLETION.md',
            'sha256': 'outputs/audit/E06B_SEED42_EVIDENCE_SHA256.txt'}


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def utc(ts):
    return datetime.fromtimestamp(ts, timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def epoch_seconds(iso):
    return datetime.strptime(iso, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc).timestamp()


def tree_snapshot(root):
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            p = Path(dirpath) / name
            st = p.stat()
            out.append([str(p.relative_to(root)), st.st_size, st.st_mtime_ns])
    return out


def listing_digest(root):
    """`find . -type f -printf "%P %s %T@\\n" | LC_ALL=C sort | sha256sum` (same convention as earlier closures)."""
    lines = [f'{p} {s} {m // 10**9}.{m % 10**9:09d}0' for p, s, m in tree_snapshot(root)]
    return sha256_bytes(''.join(f'{x}\n' for x in sorted(lines)).encode()), len(lines)


def sha_listing(root):
    """LC_ALL=C (byte-order) sorted `<path> <size> <sha256>` lines; digest = sha256 of the newline-terminated lines."""
    rows = sorted((p, s) for p, s, _ in tree_snapshot(root))
    lines = [f'{p} {s} {sha256_file(Path(root) / p)}' for p, s in rows]
    return sha256_bytes(''.join(f'{x}\n' for x in lines).encode()), len(lines), lines


def small_hashes(root):
    return {n: sha256_file(root / n) for n in SMALL_FILES if (root / n).exists()}


def expected_batch(iteration):
    return (B, [20] * 12) if iteration < STEPS_PER_EPOCH - 1 else (120, [20] * 6)


# ============================================================================ official .pth (no class import)
class _Stub:
    """Placeholder for any pickled global: records nothing executable, accepts any construction/state."""
    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, *args, **kwargs):
        self.args = args

    def __setstate__(self, state):
        self.state = state

    def __setitem__(self, key, value):
        pass

    def append(self, value):
        pass

    def extend(self, values):
        pass


class _StorageRef(tuple):
    pass


class _NoImportUnpickler(pickle.Unpickler):
    """Resolves every global to a _Stub subclass (nothing is imported or executed) and records storage references."""
    def __init__(self, f):
        super().__init__(f)
        self.refs = {}

    def find_class(self, module, name):
        return type(f'{module}.{name}', (_Stub,), {'qualname': f'{module}.{name}'})

    def persistent_load(self, pid):
        kind, storage_type, key, location, numel = pid
        if kind != 'storage':
            raise pickle.UnpicklingError(f'unexpected persistent id {kind!r}')
        self.refs[str(key)] = (storage_type.qualname, int(numel))
        return _StorageRef((storage_type.qualname, str(key), int(numel)))


FLOAT_DTYPES = {'torch.FloatStorage': 'f4', 'torch.DoubleStorage': 'f8', 'torch.HalfStorage': 'f2'}


def official_storage_report(path):
    """Every float storage of a torch zip checkpoint read as raw bytes (numpy view) and checked finite."""
    import numpy as np
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        pkl = next(n for n in names if n.endswith('/data.pkl'))
        prefix = pkl[:-len('data.pkl')]
        up = _NoImportUnpickler(io.BytesIO(z.read(pkl)))
        up.load()
        byteorder = z.read(prefix + 'byteorder').decode().strip() if prefix + 'byteorder' in names else 'little'
        types = collections.Counter()
        floats = elements = nonfinite = 0
        for key, (stype, numel) in sorted(up.refs.items()):
            types[stype] += 1
            if stype not in FLOAT_DTYPES:
                continue
            raw = z.read(prefix + 'data/' + key)
            dt = np.dtype(('<' if byteorder == 'little' else '>') + FLOAT_DTYPES[stype])
            if len(raw) != numel * dt.itemsize:
                raise ValueError(f'{path}: storage {key} size {len(raw)} != {numel} x {dt.itemsize}')
            vals = np.frombuffer(raw, dtype=dt)
            floats += 1
            elements += int(vals.size)
            nonfinite += int(vals.size - np.isfinite(vals).sum())
    return {'storages': len(up.refs), 'storage_types': dict(sorted(types.items())), 'float_storages': floats,
            'float_elements': elements, 'nonfinite_elements': nonfinite, 'byteorder': byteorder}


# ============================================================================ collect (GPU host, read-only)
def scan_metrics(path):
    bad_json, nonfinite, schema, steps, events = [], [], [], [], {}
    per_epoch = {}
    step_seconds, mem, cls_nonzero = [], [], []
    first_utc = last_utc = None
    n = 0

    def reject(c):
        raise ValueError(c)
    with open(path, 'rb') as f:
        for n, raw in enumerate(f, 1):
            try:
                d = json.loads(raw, parse_constant=reject)
            except ValueError as e:
                bad_json.append([n, str(e)[:80]])
                continue
            if d.get('record_type') == 'event':
                events.setdefault(d['event'], []).append(d['payload'])
                continue
            if d.get('record_type') != 'epoch':
                schema.append([n, 'record_type', d.get('record_type')])
                continue
            s, e, it = d['global_step'], d['epoch'], d['iteration_in_epoch']
            steps.append(s)
            first_utc = first_utc or d['utc']
            last_utc = d['utc']
            size, chunks = expected_batch(it)
            tl = d['train_losses']
            checks = {'epoch': e == (s - 1) // STEPS_PER_EPOCH + 1, 'iteration': it == (s - 1) % STEPS_PER_EPOCH,
                      'batch': d['actual_global_batch_size'] == size, 'chunks': d['microbatch_sizes'] == chunks,
                      'lr': d['learning_rate'] == EXPECTED['learning_rate'],
                      'mode': d['runner_mode'] == 'SCIENTIFIC' and d['experiment_seed'] == EXPECTED['seed']
                      and d['method_id'] == EXPECTED['method_id'],
                      'execution_mode': d['execution_mode'] == EXPECTED['execution_mode'],
                      'granularity': d['record_granularity'] == 'GLOBAL_OPTIMIZER_STEP',
                      'optimizer_applications': d['optimizer_applications_total'] == s,
                      'owned_gradients_finite_55': d['owned_gradients_finite'] == 55,
                      'netCls_grad_nonzero': isinstance(d['netCls_grad_nonzero'], int) and d['netCls_grad_nonzero'] > 0,
                      'val_null': d['val_losses'] is None and d['val_metrics'] is None,
                      'no_test_key': not any('test' in k.lower() for k in d),
                      'loss_keys': sorted(tl) == sorted(LOSS_KEYS),
                      'cls_pair_active': tl['loss_cls'] > 0.0 and tl['loss_pair'] > 0.0}
            for k, ok in checks.items():
                if not ok:
                    schema.append([n, k, s])
            nums = list(tl.values()) + [d['step_seconds'], d['wall_clock_seconds']]
            if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in nums):
                nonfinite.append(s)
            row = per_epoch.setdefault(e, {'sizes': [], 'hashes': [], 'losses': {k: [] for k in LOSS_KEYS}})
            row['sizes'].append(d['actual_global_batch_size'])
            row['hashes'].append(d['batch_pair_id_sha256'])
            for k in LOSS_KEYS:
                row['losses'][k].append(tl[k])
            step_seconds.append(d['step_seconds'])
            mem.append(d['gpu_memory_bytes'])
            cls_nonzero.append(d['netCls_grad_nonzero'])
    c = collections.Counter(steps)
    dup = sorted(s for s, k in c.items() if k > 1)
    missing = sorted(set(range(1, TOTAL_STEPS + 1)) - set(steps))
    loss_ranges = {k: [min(v for r in per_epoch.values() for v in r['losses'][k]),
                       max(v for r in per_epoch.values() for v in r['losses'][k])] for k in LOSS_KEYS}
    trend = {str(e): {k: statistics.fmean(per_epoch[e]['losses'][k]) for k in LOSS_KEYS}
             for e in CHECKPOINT_EPOCHS if e in per_epoch}
    return {
        'raw_lines': n, 'step_records': len(steps), 'event_counts': {k: len(v) for k, v in sorted(events.items())},
        'corrupted_lines': bad_json, 'schema_violations': schema[:50], 'schema_violation_count': len(schema),
        'nonfinite_steps': nonfinite, 'duplicates': dup, 'missing': missing[:50], 'missing_count': len(missing),
        'step_min': min(steps), 'step_max': max(steps), 'ordered': steps == sorted(steps),
        'contiguous': steps == list(range(1, TOTAL_STEPS + 1)),
        'epochs_present': sorted(per_epoch), 'epoch_sizes_ok': sum(
            r['sizes'] == [B] * 15 + [120] for r in per_epoch.values()),
        'epoch_rows': sorted({sum(r['sizes']) for r in per_epoch.values()}),
        'first_step_utc': first_utc, 'last_step_utc': last_utc,
        'loss_ranges': loss_ranges, 'loss_mean_at_checkpoint_epochs': trend,
        'netCls_grad_nonzero_min': min(cls_nonzero),
        'step_seconds': {'median': statistics.median(step_seconds), 'p95': sorted(step_seconds)[int(0.95 * len(
            step_seconds))], 'max': max(step_seconds), 'sum': sum(step_seconds)},
        'peak_step_gpu_memory_bytes': max(mem),
    }, events, {e: r['hashes'] for e, r in per_epoch.items()}


def predict_native(split_path, seed, epochs):
    """Independent re-derivation (no repo code) of the native relation and the seeded loader draws: torch map-style
    RandomSampler with generator G (per epoch: int64 base seed, randperm(n), a second randperm(n) at exhaustion);
    8 workers, worker w python `random` seeded (base + w) mod 2**32, batch b served by worker b % 8, each spoof row
    drawing its live partner by random.choice over its subject's TRAIN live pool (sorted by UTF-8 bytes)."""
    import random
    import pyarrow.parquet as pq
    import torch
    cols = ['sample_id', 'dataset', 'subject_id_global', 'label_binary', 'attack_macro', 'split']
    rows = [r for r in pq.read_table(split_path, columns=cols, filters=[('split', '==', 'TRAIN')]).to_pylist()
            if r['dataset'] in DATASETS]
    by_id = {r['sample_id']: r for r in rows}
    bkey = lambda s: s.encode('utf-8')   # noqa: E731
    pools, spoof = collections.defaultdict(list), []
    for r in rows:
        (pools[r['subject_id_global']] if r['label_binary'] == 0 else spoof).append(r['sample_id'])
    pools = {k: tuple(sorted(v, key=bkey)) for k, v in pools.items()}
    spoof = sorted(spoof, key=bkey)
    n = len(spoof)
    G = torch.Generator()
    G.manual_seed(seed)
    out, pair_audit = {}, collections.Counter()
    for epoch in range(1, epochs + 1):
        base = int(torch.empty((), dtype=torch.int64).random_(generator=G).item())
        perm = torch.randperm(n, generator=G).tolist()
        torch.randperm(n, generator=G)
        batches = [perm[i:i + B] for i in range(0, n, B)]
        streams = [random.Random((base + w) % 2 ** 32) for w in range(WORKERS)]
        draws = [None] * len(batches)
        for w in range(WORKERS):
            for b in range(w, len(batches), WORKERS):
                draws[b] = [streams[w].choice(pools[by_id[spoof[i]]['subject_id_global']]) for i in batches[b]]
        hashes, order = [], []
        for idx, lives in zip(batches, draws):
            sids = [spoof[i] for i in idx]
            order += sids
            hashes.append(sha256_bytes('\n'.join(f'{s}>{l}' for s, l in zip(sids, lives)).encode()))
            for s, l in zip(sids, lives):
                rs, rl = by_id[s], by_id[l]
                pair_audit['pairs'] += 1
                pair_audit['same_subject'] += rs['subject_id_global'] == rl['subject_id_global']
                pair_audit['partner_live'] += rl['label_binary'] == 0 and rl['attack_macro'] == 'live'
                pair_audit['spoof_is_spoof'] += rs['label_binary'] == 1 and rs['attack_macro'] in CLASS_INDEX
                pair_audit['train_split'] += rs['split'] == rl['split'] == 'TRAIN'
                pair_audit['casia_msu'] += rs['dataset'] in DATASETS and rl['dataset'] == rs['dataset']
                pair_audit[f"class_{CLASS_INDEX[rs['attack_macro']]}"] += 1
        out[epoch] = {'batches': hashes, 'epoch_order': sha256_bytes('\n'.join(order).encode()),
                      'is_permutation': sorted(perm) == list(range(n)), 'sizes': [len(b) for b in batches]}
    population = {ds: {'spoof': sum(by_id[s]['dataset'] == ds for s in spoof),
                       'live': sum(len(v) for k, v in pools.items() if k.startswith(ds + '::'))} for ds in DATASETS}
    return out, dict(pair_audit), {'spoof_rows': n, 'live_pools': len(pools), 'by_dataset': population,
                                   'train_rows': len(rows)}, torch.__version__


def collect_atime(split_path):
    import pyarrow.parquet as pq
    t = pq.read_table(split_path, columns=['sample_id', 'dataset', 'split']).to_pylist()
    meta = {r['sample_id']: (r['dataset'], r['split']) for r in t}

    def load(p):
        out = {}
        for line in open(p):
            a, m, path = line.rstrip('\n').split('\t')
            out[path] = (float(a), float(m))
        return out
    pre, post = load(ATIME_PRE), load(ATIME_POST)
    t_pre = epoch_seconds(Path(ATIME_PRE.replace('.tsv', '.time')).read_text().strip())
    t_post = epoch_seconds(Path(ATIME_POST.replace('.tsv', '.time')).read_text().strip())
    lo, hi = epoch_seconds(RUN_WINDOW[0]), epoch_seconds(RUN_WINDOW[1])
    stats, manifests = {}, []
    for path, (a, m) in sorted(pre.items()):
        if '/manifests/' in path:
            a2 = post[path][0]
            manifests.append({'file': os.path.basename(path), 'atime_pre_utc': utc(a), 'atime_post_utc': utc(a2),
                              'changed': a2 != a, 'mtime_unchanged': post[path][1] == m})
            continue
        sid = os.path.basename(path)[:-4]
        ds, sp = meta.get(sid, (os.path.basename(os.path.dirname(path)), 'UNMAPPED'))
        s = stats.setdefault(f'{ds}|{sp}', {'files': 0, 'sensitive_whole_window': 0, 'atime_changed': 0,
                                           'changed_inside_run_window': 0, 'changed_outside_run_window': 0,
                                           'mtime_changed': 0, 'missing_post': 0})
        s['files'] += 1
        s['sensitive_whole_window'] += (a <= m) or (lo - a >= 86400)   # relatime: any read in the window updates
        if path not in post:
            s['missing_post'] += 1
            continue
        a2, m2 = post[path]
        s['mtime_changed'] += m2 != m
        if a2 != a:
            s['atime_changed'] += 1
            s['changed_inside_run_window' if lo - 60 <= a2 <= hi + 60 else 'changed_outside_run_window'] += 1
    return {'pre_snapshot_utc': utc(t_pre), 'post_snapshot_utc': utc(t_post), 'files_pre': len(pre),
            'files_post': len(post), 'run_window_utc': list(RUN_WINDOW), 'by_class': stats, 'manifests': manifests,
            'mount': 'ext4 rw,relatime (/)', 'pre_sha256': sha256_file(ATIME_PRE), 'post_sha256': sha256_file(ATIME_POST)}


def collect(run_root, repo):
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    import subprocess
    from methods.dsdg import native
    from methods.dsdg import native_runner as nr
    root = Path(run_root)
    before_tree, before_small = tree_snapshot(root), small_hashes(root)
    before_sha_digest, _, _ = sha_listing(root)
    git = {'head': subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip(),
           'branch': subprocess.run(['git', 'branch', '--show-current'], capture_output=True, text=True).stdout.strip(),
           'status_porcelain': subprocess.run(['git', 'status', '--porcelain'], capture_output=True,
                                              text=True).stdout}
    adapter = native.DSDGNativeAdapter()
    config = nr.verify_contract(adapter.config)         # frozen config + snapshot + contract + M6F-C gates
    ids = nr.identities(config)
    source = nr.verify_source(adapter, ids)
    man = json.loads((root / 'run_manifest.json').read_text())
    summ = json.loads((root / 'run_summary.json').read_text())
    cindex = json.loads((root / 'checkpoint_index.json').read_text())
    metrics, events, hashes = scan_metrics(root / 'metrics.jsonl')
    split_path = Path(repo) / 'manifests' / 'split_v1.parquet'
    pred, pair_audit, population, torch_version = predict_native(split_path, EXPECTED['seed'], EPOCHS)
    batch_match = sum(i < len(hashes.get(e, [])) and hashes[e][i] == h
                      for e, p in pred.items() for i, h in enumerate(p['batches']))
    complete = {p['epoch']: p for p in events.get('e06b_epoch_complete', [])}
    epoch_events_ok = sum(c['rows'] == SPOOF_ROWS and c['optimizer_steps'] == STEPS_PER_EPOCH
                          and c['global_step_end'] == e * STEPS_PER_EPOCH for e, c in complete.items())
    epoch_order_match = sum(complete.get(e, {}).get('epoch_spoof_order_sha256') == p['epoch_order']
                            for e, p in pred.items())
    official = []
    for c in cindex['checkpoints']:
        p = root / c['path']
        rep = official_storage_report(p) if p.is_file() else None
        official.append(dict(c, exists=p.is_file(), size_actual=p.stat().st_size if p.is_file() else None,
                             sha256_actual=sha256_file(p) if p.is_file() else None, storages=rep))
    on_disk = sorted(str(p.relative_to(root)) for p in (root / 'checkpoints').iterdir())
    logs = {}
    for name in ('stdout.log', 'stderr.log'):
        text = (root / name).read_text(errors='replace')
        lines = text.splitlines()
        logs[name] = {'bytes': len(text.encode()), 'lines': len(lines),
                      'checkpoint_saved_lines': sum(l.startswith('Checkpoint saved to ') for l in lines),
                      'other_lines': [l for l in lines if not l.startswith('Checkpoint saved to ')][:20],
                      'pattern_hits': {k: sum(k in l for l in lines if not l.startswith('Checkpoint saved to '))
                                       for k in LOG_PATTERNS}}
    lightcnn = Path(RUNTIME) / 'third_party_weights/lightcnn/LightCNN_29Layers_V2_checkpoint.pth.tar'
    atime = collect_atime(split_path)
    a1 = Path(ATTEMPT1['archive'])
    a1_digest, a1_count, _ = sha_listing(a1) if a1.is_dir() else (None, 0, None)
    a1_manifest = json.loads((a1 / 'run_manifest.json').read_text()) if a1.is_dir() else {}
    digest, count = listing_digest(root)
    sha_digest, sha_count, sha_lines = sha_listing(root)
    after_tree, after_small = tree_snapshot(root), small_hashes(root)
    return {
        'collector': {'torch_version': torch_version, 'python': sys.version.split()[0],
                      'utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')},
        'git': git, 'identities_recomputed': {k: v for k, v in ids.items() if k != 'source_files_sha256'},
        'source_recomputed': {'commit': source['commit'], 'files_sha256_equals_lock': True},
        'run_manifest': man, 'run_summary': summ,
        'checkpoint_index_meta': {k: v for k, v in cindex.items() if k != 'checkpoints'},
        'metrics': metrics, 'metrics_sha256': sha256_file(root / 'metrics.jsonl'),
        'run_start_event': events.get('e06b_run_start', [None])[0],
        'run_start_events': len(events.get('e06b_run_start', [])),
        'checkpoint_events': [{'epoch': c['epoch'], 'global_step': c['global_step'], 'official': c['official'],
                               'resume_sidecar': c['resume_sidecar']} for c in events.get('e06b_checkpoint_event', [])],
        'visualization_epochs': [v['epoch'] for v in events.get('e06b_visualization_diagnostic', [])],
        'visualization_roles': sorted({f"{v['role']}|{v['data']}|{v['not_validation']}|{v['selects_nothing']}"
                                       for v in events.get('e06b_visualization_diagnostic', [])}),
        'order_prediction': {'torch_version': torch_version,
                             'predicted_steps': sum(len(p['batches']) for p in pred.values()),
                             'pair_hash_match': batch_match, 'epoch_order_match': epoch_order_match,
                             'permutations': sum(p['is_permutation'] for p in pred.values()),
                             'predicted_epoch_plans_ok': sum(p['sizes'] == [B] * 15 + [120] for p in pred.values()),
                             'epoch_complete_events_ok': epoch_events_ok},
        'pairing_audit': pair_audit, 'population_rederived': population,
        'official_checkpoints': official, 'checkpoints_dir_listing': on_disk,
        'runner_state_files': sorted(str(p.relative_to(root)) for p in root.rglob('*runner_state*')),
        'resume_index_present': (root / 'resume_state_index.json').exists(),
        'logs': logs, 'exit_file': Path(EXIT_FILE).read_text().strip(),
        'exit_file_mtime_utc': utc(Path(EXIT_FILE).stat().st_mtime),
        'lightcnn': {'sha256': sha256_file(lightcnn), 'bytes': lightcnn.stat().st_size},
        'config_file_sha256': sha256_file(Path(repo) / man['config_path']),
        'split_manifest_sha256': sha256_file(split_path),
        'atime_firewall': atime,
        'attempt1_archive': {'path': str(a1), 'exists': a1.is_dir(), 'listing_digest_path_size_sha256': a1_digest,
                             'file_count': a1_count, 'run_id': a1_manifest.get('run_id'),
                             'git_commit': a1_manifest.get('git_commit'),
                             'completion_status': a1_manifest.get('completion_status'),
                             'under_runs': str(a1).startswith(f'{RUNTIME}/runs/')},
        'small_file_sha256': after_small, 'run_root_listing_digest': digest, 'run_root_file_count': count,
        'run_root_sha_listing_digest': sha_digest, 'run_root_sha_listing_count': sha_count,
        'run_root_sha_listing_digest_before': before_sha_digest, 'run_root_sha_listing': sha_lines,
        'run_root_tree_sha256': sha256_bytes(json.dumps(after_tree).encode()),
        'run_root_unchanged_whole_audit': (before_tree == after_tree and before_small == after_small
                                           and before_sha_digest == sha_digest),
        'parent_listing': sorted(os.listdir(root.parent)),
    }


# ============================================================================ build (repository)
class Gates:
    def __init__(self):
        self.results = []

    def __call__(self, name, ok, detail=None):
        self.results.append({'gate': name, 'pass': bool(ok), **({'detail': detail} if detail is not None else {})})
        return ok

    @property
    def failed(self):
        return [r['gate'] for r in self.results if not r['pass']]


def manifest_text(c):
    """Deterministic checkpoint manifest: official checkpoints sorted by epoch/path (E06b has no sidecars)."""
    head = (f"# E06b DSDG-NATIVE seed42 attempt 2 run {EXPECTED['run_id']} (authority {EXPECTED['git_commit'][:7]}): "
            'epoch global_step checkpoint_type selected_for_final size_bytes sha256 path\n')
    rows = sorted(c['official_checkpoints'], key=lambda r: (r['epoch'], r['path']))
    body = ''.join(f"{r['epoch']:03d} {r['global_step']} {r['checkpoint_type']} {str(r['selected_for_final']).lower()} "
                   f"{r['file_size_bytes']} {r['sha256']} {r['path']}\n" for r in rows)
    return head + body + '# runner-state / resume sidecars: none (E06b is fresh-only)\n'


def build(c, repo_root):
    repo = Path(repo_root)
    G = Gates()
    man, summ, m, rs = c['run_manifest'], c['run_summary'], c['metrics'], c['run_start_event'] or {}
    E = EXPECTED

    # A. host / authority
    G('A.host', man['host'] == E['host'] and man['environment']['host'] == E['host'])
    G('A.gpu_checkout_authority', c['git']['head'] == E['git_commit'] and c['git']['branch'] == 'm6-baselines'
      and c['git']['status_porcelain'] == '', c['git'])
    G('A.exit_zero', c['exit_file'] == '0', c['exit_file'])

    # B. run identity
    for k, v in (('method_id', E['method_id']), ('experiment_seed', E['seed']), ('run_id', E['run_id']),
                 ('git_commit', E['git_commit']), ('git_dirty', False), ('config_sha256', E['config_sha256']),
                 ('completion_status', 'completed'), ('test_split_accessed', False), ('snapshot_verified', True),
                 ('environment_name', E['environment_name']), ('python_executable', E['python_executable']),
                 ('command_line', COMMAND)):
        G(f'B.manifest.{k}', man.get(k) == v, man.get(k))
    run_id = sha256_bytes(f"{E['method_id']}|{E['seed']}|{E['config_sha256']}|{E['git_commit']}".encode())[:16]
    G('B.run_id_rule', run_id == E['run_id'], run_id)
    G('B.no_resume_flag', '--resume' not in man['command_line'] and c['run_start_events'] == 1)
    for k, v in (('run_id', E['run_id']), ('experiment_seed', E['seed']), ('completion_status', 'completed'),
                 ('checkpoint_selection_rule', E['checkpoint_rule']), ('optimizer_applications', TOTAL_STEPS),
                 ('test_split_accessed', False), ('failure_reason', None), ('records_written', m['raw_lines'])):
        G(f'B.summary.{k}', summ.get(k) == v, summ.get(k))
    G('B.not_attempt1', E['run_id'] != ATTEMPT1['run_id'] and man['git_commit'] != ATTEMPT1['authority'])

    # C. config / data / environment identities
    G('C.identities_equal_recomputed', rs.get('identities') == c['identities_recomputed'])
    G('C.fidelity', rs.get('runtime_fidelity_assessment') == E['fidelity']
      and c['identities_recomputed']['runtime_fidelity_assessment'] == E['fidelity'])
    G('C.execution_mode', rs.get('execution_mode') == E['execution_mode'])
    G('C.config_file_sha256', c['config_file_sha256'] == E['config_sha256'] == rs['identities']['config_sha256'])
    G('C.split_manifest_sha256', c['split_manifest_sha256'] == rs['identities']['split_manifest_sha256'])
    G('C.source_commit', c['source_recomputed']['commit'] == rs['identities']['source_commit']
      and man['source_commits'] == {'https://github.com/JDAI-CV/FaceX-Zoo': rs['identities']['source_commit']})
    G('C.lightcnn', c['lightcnn']['sha256'] == rs['identities']['lightcnn_sha256'] == rs['lightcnn']['sha256']
      and c['lightcnn']['bytes'] == rs['lightcnn']['bytes'] == 123844849)
    G('C.environment_lock', man['dependency_fingerprint'] == rs['identities']['environment_lock_sha256'])
    G('C.loader_contract', {k: rs['loader'][k] for k in ('batch_size', 'num_workers', 'persistent_workers',
                                                          'drop_last', 'shuffle', 'len_batches', 'rows')}
      == {'batch_size': B, 'num_workers': WORKERS, 'persistent_workers': False, 'drop_last': False, 'shuffle': True,
          'len_batches': STEPS_PER_EPOCH, 'rows': SPOOF_ROWS}, rs.get('loader'))
    G('C.precision', rs['precision'] == {'activation_checkpointing': False, 'amp': False, 'autocast_cuda': False,
                                         'cudnn_benchmark': False, 'cudnn_deterministic': True, 'cudnn_tf32': False,
                                         'default_dtype': 'torch.float32', 'matmul_precision': 'highest',
                                         'matmul_tf32': False}, rs.get('precision'))
    G('C.seeding', rs['seeding']['seed'] == E['seed'] and rs['seeding']['role'] == 'EXPERIMENT_SEED')
    pop = rs['population']
    G('C.population_logged_and_rederived', pop['total']['spoof'] == SPOOF_ROWS and set(pop) == set(DATASETS) | {'total'}
      and all(c['population_rederived']['by_dataset'][d] == {'spoof': pop[d]['spoof'], 'live': pop[d]['live']}
              for d in DATASETS) and c['population_rederived']['spoof_rows'] == SPOOF_ROWS, pop)

    # D. training completion / step accounting / order and native pairing
    G('D.step_records_3200', m['step_records'] == TOTAL_STEPS and m['raw_lines'] == TOTAL_STEPS + 243)
    G('D.contiguous_1_3200', m['contiguous'] and m['step_min'] == 1 and m['step_max'] == TOTAL_STEPS
      and not m['duplicates'] and m['missing_count'] == 0)
    G('D.epochs_1_200', m['epochs_present'] == list(range(1, EPOCHS + 1)))
    G('D.epoch_plan_15x240_120', m['epoch_sizes_ok'] == EPOCHS and m['epoch_rows'] == [SPOOF_ROWS])
    G('D.record_schema', m['schema_violation_count'] == 0, m['schema_violations'])
    G('D.event_counts', m['event_counts'] == {'e06b_checkpoint_event': 21, 'e06b_epoch_complete': EPOCHS,
                                              'e06b_run_start': 1, 'e06b_visualization_diagnostic': 21},
      m['event_counts'])
    o = c['order_prediction']
    G('D.pair_hash_prediction_3200', o['predicted_steps'] == TOTAL_STEPS and o['pair_hash_match'] == TOTAL_STEPS, o)
    G('D.epoch_spoof_order_prediction_200', o['epoch_order_match'] == EPOCHS and o['permutations'] == EPOCHS
      and o['predicted_epoch_plans_ok'] == EPOCHS, o)
    G('D.every_spoof_row_once_per_epoch', o['epoch_complete_events_ok'] == EPOCHS)
    pa, n_pairs = c['pairing_audit'], EPOCHS * SPOOF_ROWS
    G('D.native_pairing_all_epochs', pa['pairs'] == n_pairs and all(
        pa[k] == n_pairs for k in ('same_subject', 'partner_live', 'spoof_is_spoof', 'train_split', 'casia_msu')), pa)
    G('D.class_mapping_print0_replay1', pa['class_0'] == EPOCHS * 2080 and pa['class_1'] == EPOCHS * 1640, pa)

    # E. numerical health
    G('E.no_corrupted_lines', m['corrupted_lines'] == [])
    G('E.all_losses_finite', m['nonfinite_steps'] == [])
    G('E.loss_cls_active', m['loss_ranges']['loss_cls'][0] > 0.0, m['loss_ranges']['loss_cls'])
    G('E.loss_pair_active', m['loss_ranges']['loss_pair'][0] > 0.0, m['loss_ranges']['loss_pair'])
    G('E.netCls_grad_nonzero_every_step', m['netCls_grad_nonzero_min'] > 0, m['netCls_grad_nonzero_min'])
    off = c['official_checkpoints']
    G('E.official_storages_finite', all(r['storages'] and r['storages']['nonfinite_elements'] == 0
                                        and r['storages']['float_elements'] > 0 for r in off))
    lg = c['logs']
    G('E.stderr_empty', lg['stderr.log']['bytes'] == 0)
    G('E.stdout_only_checkpoint_lines', lg['stdout.log']['other_lines'] == []
      and lg['stdout.log']['checkpoint_saved_lines'] == 63 and not any(lg['stdout.log']['pattern_hits'].values()))

    # F. checkpoints
    G('F.official_count_63', len(off) == 63 and len(c['checkpoint_events']) == 21)
    G('F.official_epochs', sorted({r['epoch'] for r in off}) == CHECKPOINT_EPOCHS
      and [e['epoch'] for e in c['checkpoint_events']] == CHECKPOINT_EPOCHS)
    G('F.three_files_per_event', all(sorted(r['path'] for r in off if r['epoch'] == e) == sorted(
        f'checkpoints/{p}model_epoch_{e}_iter_0.pth' for _, p in OFFICIAL) for e in CHECKPOINT_EPOCHS))
    G('F.official_files_match_index', all(r['exists'] and r['size_actual'] == r['file_size_bytes']
                                          and r['sha256_actual'] == r['sha256'] for r in off))
    G('F.official_global_steps', all(r['global_step'] == r['epoch'] * STEPS_PER_EPOCH for r in off))
    sel = [r for r in off if r['selected_for_final']]
    G('F.single_selected_final_netG', len(sel) == 1 and sel[0]['path'] == FINAL_NETG and sel[0]['epoch'] == EPOCHS
      and sel[0]['checkpoint_type'] == 'selected' and sel[0]['selection_reason'] == E['checkpoint_rule'])
    G('F.non_final_not_selected', all(r['selected_for_final'] is False and r['checkpoint_type'] ==
                                      ('terminal' if r['epoch'] == EPOCHS else 'periodic')
                                      for r in off if r['path'] != FINAL_NETG))
    G('F.summary_final_netG', bool(sel) and summ['final_or_selected_checkpoint_path'] == f"{E['run_root']}/{FINAL_NETG}"
      and summ['final_or_selected_checkpoint_sha256'] == sel[0]['sha256'])
    ev_paths = {(e['epoch'], o2['path'], o2['sha256']) for e in c['checkpoint_events'] for o2 in e['official']}
    G('F.events_match_index', ev_paths == {(r['epoch'], r['path'], r['sha256']) for r in off})
    G('F.no_resume_sidecar', all(e['resume_sidecar'] is None for e in c['checkpoint_events'])
      and c['runner_state_files'] == [] and not c['resume_index_present'])
    G('F.checkpoints_dir_exact', c['checkpoints_dir_listing'] == sorted(r['path'] for r in off))  # no .partial/stray

    # G. firewall
    a = c['atime_firewall']
    by = a['by_class']
    test = {k: v for k, v in by.items() if k.endswith('|TEST') and not k.startswith('siw')}
    val = {k: v for k, v in by.items() if k.endswith('|VAL') and not k.startswith('siw')}
    siw = {k: v for k, v in by.items() if k.startswith('siw')}
    G('G.snapshot_complete', a['files_pre'] == a['files_post'] == 20627 and 'UNMAPPED' not in ''.join(by))
    G('G.test_faces_atime_unchanged', test and all(v['atime_changed'] == 0 and v['missing_post'] == 0
                                                   for v in test.values()), test)
    G('G.test_faces_whole_window_sensitive', test and all(v['sensitive_whole_window'] == v['files']
                                                          for v in test.values()))
    G('G.val_faces_atime_unchanged', val and all(v['atime_changed'] == 0 for v in val.values()), val)
    G('G.val_faces_whole_window_sensitive', val and all(v['sensitive_whole_window'] == v['files']
                                                        for v in val.values()))
    G('G.siw_atime_unchanged', siw and all(v['atime_changed'] == 0 for v in siw.values()), siw)
    G('G.only_casia_msu_train_read', all(v['atime_changed'] == 0 for k, v in by.items()
                                         if k not in ('casia_fasd|TRAIN', 'msu_mfsd|TRAIN')))
    G('G.no_face_mtime_change', all(v['mtime_changed'] == 0 and v['missing_post'] == 0 for v in by.values()))
    mf = {x['file']: x for x in a['manifests']}
    G('G.val_pairs_manifest_untouched', not mf['val_pairs_v1.parquet']['changed'], mf['val_pairs_v1.parquet'])
    G('G.only_split_manifest_read_in_run', all(not x['changed'] for n, x in mf.items() if n != 'split_v1.parquet')
      and all(x['mtime_unchanged'] for x in mf.values()), a['manifests'])
    G('G.no_val_records', m['schema_violation_count'] == 0)   # every step record checked val null / no test key
    G('G.visualization_is_train_only', c['visualization_epochs'] == CHECKPOINT_EPOCHS and c['visualization_roles'] == [
        'UPSTREAM_TRAINING_SAMPLE_VISUALIZATION|TRAIN (last global batch of the epoch)|True|True'])

    # H. run-root integrity / attempt 1 history
    G('H.run_root_unchanged_whole_audit', c['run_root_unchanged_whole_audit'])
    G('H.sha_listing_before_equals_after', c['run_root_sha_listing_digest_before'] == c['run_root_sha_listing_digest'])
    G('H.parent_only_seed42', [p for p in c['parent_listing'] if not p.startswith('.')] == ['seed_42'],
      c['parent_listing'])
    a1 = c['attempt1_archive']
    G('H.attempt1_archived_untouched_nonselectable', a1['exists'] and not a1['under_runs']
      and a1['listing_digest_path_size_sha256'] == ATTEMPT1['listing_digest_path_size_sha256']
      and a1['file_count'] == ATTEMPT1['file_count'] and a1['run_id'] == ATTEMPT1['run_id']
      and a1['git_commit'] == ATTEMPT1['authority'] and a1['completion_status'] == 'failed', a1)

    text = manifest_text(c)
    sel0 = sel[0] if sel else {}
    doc = {
        'classification': CLASSIFICATION, 'status': STATUS_BLOCKED if G.failed else STATUS_OK,
        'failed_gates': G.failed, 'gates': G.results,
        'method': {'id': 'E06b', 'name': E['method_name'], 'fidelity': E['fidelity'], 'track': E['track'],
                   'execution_mode': E['execution_mode']},
        'seed': E['seed'], 'attempt': E['attempt'], 'run_id': E['run_id'], 'run_root': E['run_root'],
        'host': E['host'], 'ssh_address_note': 'GPU host reached as student20261@10.31.131.201 for launch and audit',
        'training_authority_commit': E['git_commit'],
        'authority_note': ('Scientific training authority for all successful E06b seeds is d9ecbef (checkpoint-context '
                           'fix); this closure commit is evidence/reporting only.'),
        'successful_scientific_seed': 'ATTEMPT 2 (run aa96129289d080ba)',
        'attempt1': dict(ATTEMPT1, selectable=False, eligible_for_bank=False, eligible_for_downstream=False,
                         role='failed technical history only (epoch 1 / step 16, 0 checkpoints)'),
        'frozen': {'config_sha256': E['config_sha256'], 'identities': rs.get('identities'),
                   'resolved_config_sha256': c['small_file_sha256']['resolved_config.yaml']},
        'training': {'completed_epoch': EPOCHS, 'global_step': m['step_max'], 'step_records': m['step_records'],
                     'optimizer_applications': summ['optimizer_applications'], 'steps_per_epoch': STEPS_PER_EPOCH,
                     'epoch_plan': '15 x 240 + 1 x 120 = 3720 CASIA+MSU TRAIN spoof rows (micro-batches 12 x 20 / '
                                   '6 x 20)',
                     'resume': 'none (fresh-only; E06b resume not qualified)',
                     'run_window_utc': [man['start_utc'], man['end_utc']],
                     'first_last_step_utc': [m['first_step_utc'], m['last_step_utc']],
                     'training_duration_seconds': summ['training_duration_seconds'],
                     'step_seconds': m['step_seconds'], 'peak_step_gpu_memory_bytes': m['peak_step_gpu_memory_bytes'],
                     'order_prediction': c['order_prediction'], 'pairing_audit': c['pairing_audit']},
        'numerical_health': {'nonfinite_steps': len(m['nonfinite_steps']), 'corrupted_lines': len(m['corrupted_lines']),
                             'loss_ranges': m['loss_ranges'],
                             'loss_mean_at_checkpoint_epochs': m['loss_mean_at_checkpoint_epochs'],
                             'netCls_grad_nonzero_min': m['netCls_grad_nonzero_min'],
                             'note': ('loss_cls = lambda_type * CE (K = 2) and loss_pair (lambda_pair = 5) are active; '
                                      'netCls is intentionally outside the optimizer (gradients logged, not applied)'),
                             'official_storage_check': ('data.pkl walked by a no-import stub unpickler; every float '
                                                        'storage read raw and checked finite'),
                             'final_netG_storages': sel0.get('storages'), 'logs': c['logs']},
        'checkpoints': {'events': len(c['checkpoint_events']), 'epochs': CHECKPOINT_EPOCHS, 'official_files': len(off),
                        'resume_sidecars': len(c['runner_state_files']),
                        'final_netG': {'path': FINAL_NETG, 'size_bytes': sel0.get('file_size_bytes'),
                                       'sha256': sel0.get('sha256'), 'epoch': sel0.get('epoch'),
                                       'global_step': sel0.get('global_step'),
                                       'checkpoint_type': sel0.get('checkpoint_type'),
                                       'selected_for_final': sel0.get('selected_for_final'),
                                       'selection_rule': E['checkpoint_rule']},
                        'manifest_file': EVIDENCE['manifest'], 'manifest_sha256': sha256_bytes(text.encode())},
        'firewall': {'test_access': 0 if not G.failed else None, 'val_access': 0 if not G.failed else None,
                     'siw_access': 0 if not G.failed else None, 'test_split_accessed': man['test_split_accessed'],
                     'atime': a, 'structural': ('CanonicalFaceReader resolves only the CASIA+MSU TRAIN native '
                                                'relation; the runner has no VAL pass; selection rule is '
                                                'OFFICIAL_GENERATOR_EPOCH_200'),
                     'upstream_test_block': ('the upstream "test" block writes TRAIN-sample visualization PNGs '
                                             '(diagnostics/visualization); it is not benchmark TEST'),
                     'train_note': ('CASIA/MSU TRAIN was read by attempt 1 about 30 min before the pre snapshot, so '
                                    'relatime leaves most TRAIN atimes unchanged; TEST/VAL are fully atime-sensitive'),
                     'auditor_reads': ('split_v1.parquet is read by the runner at start and by the early-qualification '
                                       'predictor (2026-10-21T17:19-17:21Z); this audit re-reads it after the post '
                                       'snapshot')},
        'run_root_integrity': {'file_count': c['run_root_file_count'], 'listing_digest': c['run_root_listing_digest'],
                               'listing_digest_convention':
                                   'find . -type f -printf "%P %s %T@\\n" | LC_ALL=C sort | sha256sum',
                               'sha_listing_digest': c['run_root_sha_listing_digest'],
                               'sha_listing_convention': ('LC_ALL=C sorted "<path> <size> <sha256>" lines, sha256 of '
                                                          'the newline-terminated listing'),
                               'tree_sha256': c['run_root_tree_sha256'], 'metrics_jsonl_sha256': c['metrics_sha256'],
                               'small_file_sha256': c['small_file_sha256'], 'final_netG_sha256': sel0.get('sha256'),
                               'unchanged_whole_audit': c['run_root_unchanged_whole_audit']},
        'confirmations': {'resume_used': False, 'val_checkpoint_selection': False, 'test_selection': False,
                          'best_epoch_logic': False, 'best_seed_logic': False, 'bank_generation': False,
                          'downstream_evaluation': False, 'run_root_mutated': False, 'attempt1_selected': False,
                          'additional_seed_started': False, 'other_method_started': False, 'gpat_started': False},
        'collected': c,
    }
    return doc, text


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('collect')
    a.add_argument('--run-root', default=EXPECTED['run_root'])
    a.add_argument('--repo', required=True)
    b = sub.add_parser('build')
    b.add_argument('--collected', required=True)
    b.add_argument('--repo-root', default=str(Path(globals().get('__file__', 'tools/x')).resolve().parents[1]))
    args = ap.parse_args(argv)
    if args.cmd == 'collect':
        json.dump(collect(args.run_root, args.repo), sys.stdout, sort_keys=True)
        return 0
    c = json.loads(Path(args.collected).read_text())
    doc, text = build(c, args.repo_root)
    root = Path(args.repo_root)
    (root / EVIDENCE['manifest']).write_text(text)
    (root / EVIDENCE['completion']).write_text(json.dumps(doc, indent=2, sort_keys=True) + '\n')
    print(doc['status'], doc['failed_gates'])
    return 0 if doc['status'] == STATUS_OK else 1


if __name__ == '__main__':
    sys.exit(main())
