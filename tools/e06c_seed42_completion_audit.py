"""E06c seed42 closure: read-only completion audit of the E06c DSDG-BIN-IDFREE / seed 42 scientific run
(run_id 27dcaf23aac18d7a, launch authority 84809868, CONTROLLED_ADAPTATION, DEV-020).

Two phases:

  collect  (GPU host, gpat-m6-e06c env, `python -I - collect --repo <gpu repo> < this file`) -- read-only, prints one
           JSON document. Files are opened 'rb'; engineering resume sidecars are opened only with
           torch.load(weights_only=True, map_location='cpu'). The whole-module DataParallel official .pth files are
           never unpickled: their tensor storages are read as raw zip members and compared with the sidecar tensors.
           The run root is re-listed (path/size/mtime_ns) and its small files re-hashed before and after the audit.
           VAL/TEST data are never opened: firewall evidence comes from the run manifest/summary/metrics, the
           structural TRAIN-only reader, and a stat-only before/after atime comparison of the faces_256 snapshot
           taken at launch (~/workdir/e06c_s42_atime_{pre,post}.tsv).
  build    (repository) -- gates the collected document and writes outputs/audit/E06C_SEED42_{COMPLETION.json,
           CHECKPOINT_MANIFEST.txt}. COMPLETION.json embeds the collected document so the build can be re-derived.

No selection, VAL, TEST, bank, downstream or training code is run.
"""
import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

CLASSIFICATION = 'SCIENTIFIC_TRAINING_COMPLETION_AUDIT'
STATUS_OK = 'E06C_SEED42_COMPLETED_CLEANLY'
STATUS_BLOCKED = 'E06C_SEED42_NEEDS_REVIEW'
RUNTIME = '/home/student20261/workdir/GPAT_TransferBench_runtime'
WORKDIR = '/home/student20261/workdir'
EXPECTED = {
    'method_id': 'E06c', 'method_name': 'DSDG-BIN-IDFREE', 'seed': 42, 'run_id': '27dcaf23aac18d7a',
    'git_commit': '84809868acf472af5ba8cf6ab47165ae038e1012',
    'config_sha256': '7176dd4cd49007320ab0c44519e7efde60568098503077303218bd913b6b5002',
    'train_relation_sha256': 'a5e4fdaef236f15730c7e3885b537e08e995faffbe654167fc940f44bbc75243',
    'run_root': f'{RUNTIME}/runs/m6/E06c/seed_42',
    'host': 'nd-System-Product-Name', 'fidelity_class': 'CONTROLLED_ADAPTATION', 'deviation': 'DEV-020',
    'execution_mode': 'GLOBAL_STATISTIC_PRESERVING_MICROBATCH_V2', 'nominal_global_batch': 240,
    'max_microbatch': 20, 'tail_batch': 198, 'drop_last': False, 'runner_mode': 'SCIENTIFIC',
    'checkpoint_rule': 'OFFICIAL_GENERATOR_EPOCH_200', 'learning_rate': 2e-4,
}
SPEC_PATH = 'docs/spec/GPAT_TransferBench_v1_0_Frozen_Specification_2026.docx'
SPEC_SHA256 = 'f7d2371682ad9504f59f0c6984e4b1a741f855f1b1f60838f9d60b160281489e'
EPOCHS, STEPS_PER_EPOCH, TRAIN_ROWS = 200, 37, 8838
TOTAL_STEPS = EPOCHS * STEPS_PER_EPOCH                       # 7400
CHECKPOINT_EPOCHS = [1] + list(range(10, EPOCHS + 1, 10))    # 21 events
OFFICIAL = (('netE_nir', 'netE_spoof_'), ('netE_vis', 'netE_live_'), ('netG', 'netG_'))
FINAL_NETG = 'checkpoints/netG_model_epoch_200_iter_0.pth'
LOSS_KEYS = ('loss_rec', 'loss_kl', 'loss_mmd', 'loss_ip', 'loss_pair', 'loss_cls', 'loss_ort', 'total_loss')
SIDECAR_MODELS = ('netE_nir', 'netE_vis', 'netG', 'netCls')
SMALL_FILES = ('run_manifest.json', 'run_summary.json', 'checkpoint_index.json', 'resume_state_index.json',
               'resolved_config.yaml', 'stdout.log', 'stderr.log', 'generation_log.jsonl')
LOG_PATTERNS = ('Traceback', 'Error', 'error', 'OutOfMemory', 'out of memory', 'CUDA', 'Killed', 'signal',
                'SIGTERM', 'SIGKILL', 'No space', 'Errno', 'STOP', 'Warning', 'nan', 'inf')
RUN_WINDOW = ('2026-10-18T06:44:08Z', '2026-10-19T00:39:46Z')
ATIME_PRE, ATIME_POST = f'{WORKDIR}/e06c_s42_atime_pre.tsv', f'{WORKDIR}/e06c_s42_atime_post.tsv'
EXIT_FILE = f'{WORKDIR}/e06c_s42.exit'
EVIDENCE = {'completion': 'outputs/audit/E06C_SEED42_COMPLETION.json',
            'manifest': 'outputs/audit/E06C_SEED42_CHECKPOINT_MANIFEST.txt',
            'report': 'outputs/audit/E06C_SEED42_COMPLETION.md',
            'sha256': 'outputs/audit/E06C_SEED42_EVIDENCE_SHA256.txt'}


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
    """`find . -type f -printf "%P %s %T@\\n" | sort | sha256sum` (same convention as earlier closures)."""
    lines = [f'{p} {s} {m // 10**9}.{m % 10**9:09d}0' for p, s, m in tree_snapshot(root)]
    return sha256_bytes(''.join(f'{x}\n' for x in sorted(lines)).encode()), len(lines)


def small_hashes(root):
    return {n: sha256_file(root / n) for n in SMALL_FILES if (root / n).exists()}


def expected_batch(iteration):
    return (240, [20] * 12) if iteration < STEPS_PER_EPOCH - 1 else (198, [20] * 9 + [18])


# ============================================================================ collect (GPU host, read-only)
def scan_metrics(path):
    bad_json, nonfinite, schema, steps, events = [], [], [], [], {}
    per_epoch = {}
    step_seconds, mem = [], []
    first_utc = last_utc = None

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
            checks = {'epoch': e == (s - 1) // STEPS_PER_EPOCH + 1, 'iteration': it == (s - 1) % STEPS_PER_EPOCH,
                      'batch': d['actual_global_batch_size'] == size, 'chunks': d['microbatch_sizes'] == chunks,
                      'lr': d['learning_rate'] == EXPECTED['learning_rate'],
                      'mode': d['runner_mode'] == 'SCIENTIFIC' and d['experiment_seed'] == EXPECTED['seed'],
                      'execution_mode': d['execution_mode'] == EXPECTED['execution_mode'],
                      'granularity': d['record_granularity'] == 'GLOBAL_OPTIMIZER_STEP',
                      'optimizer_applications': d['optimizer_applications_total'] == s,
                      'owned_gradients_finite_55': d['owned_gradients_finite'] == 55,
                      'val_null': d['val_losses'] is None and d['val_metrics'] is None,
                      'loss_keys': sorted(d['train_losses']) == sorted(LOSS_KEYS),
                      'cls_pair_zero': d['train_losses']['loss_cls'] == 0.0 and d['train_losses']['loss_pair'] == 0.0}
            for k, ok in checks.items():
                if not ok:
                    schema.append([n, k, s])
            tl = d['train_losses']
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
    dup = sorted({s for s in steps if steps.count(s) > 1}) if len(steps) != len(set(steps)) else []
    missing = sorted(set(range(1, TOTAL_STEPS + 1)) - set(steps))
    loss_ranges = {k: [min(r['losses'][k][i] for r in per_epoch.values() for i in range(len(r['losses'][k]))),
                       max(r['losses'][k][i] for r in per_epoch.values() for i in range(len(r['losses'][k])))]
                   for k in LOSS_KEYS}
    trend = {str(e): {k: statistics.fmean(per_epoch[e]['losses'][k]) for k in LOSS_KEYS}
             for e in CHECKPOINT_EPOCHS if e in per_epoch}
    return {
        'raw_lines': n, 'step_records': len(steps), 'event_counts': {k: len(v) for k, v in sorted(events.items())},
        'corrupted_lines': bad_json, 'schema_violations': schema[:50], 'schema_violation_count': len(schema),
        'nonfinite_steps': nonfinite, 'duplicates': dup, 'missing': missing[:50], 'missing_count': len(missing),
        'step_min': min(steps), 'step_max': max(steps), 'ordered': steps == sorted(steps),
        'contiguous': steps == list(range(1, TOTAL_STEPS + 1)),
        'epochs_present': sorted(per_epoch), 'epoch_sizes_ok': sum(
            r['sizes'] == [240] * 36 + [198] for r in per_epoch.values()),
        'epoch_rows': sorted({sum(r['sizes']) for r in per_epoch.values()}),
        'first_step_utc': first_utc, 'last_step_utc': last_utc,
        'loss_ranges': loss_ranges, 'loss_mean_at_checkpoint_epochs': trend,
        'step_seconds': {'median': statistics.median(step_seconds), 'p95': sorted(step_seconds)[int(0.95 * len(
            step_seconds))], 'max': max(step_seconds), 'sum': sum(step_seconds)},
        'peak_step_gpu_memory_bytes': max(mem),
    }, events, {e: r['hashes'] for e, r in per_epoch.items()}


def predict_order(relation_path, seed, epochs):
    """Independent CPU prediction of the seeded DataLoader draws (torch map-style RandomSampler, shuffle=True):
    per epoch G -> int64 base seed, randperm(n), and a second randperm(n) drawn at sampler exhaustion."""
    import pyarrow.parquet as pq
    import torch
    ids = pq.read_table(relation_path, columns=['pair_id']).column('pair_id').to_pylist()
    n = len(ids)
    G = torch.Generator()
    G.manual_seed(seed)
    out = {}
    for epoch in range(1, epochs + 1):
        torch.empty((), dtype=torch.int64).random_(generator=G)
        perm = torch.randperm(n, generator=G).tolist()
        torch.randperm(n, generator=G)
        batches = [sha256_bytes('\n'.join(ids[i] for i in perm[b:b + 240]).encode()) for b in range(0, n, 240)]
        out[epoch] = {'batches': batches, 'epoch_order': sha256_bytes('\n'.join(ids[i] for i in perm).encode()),
                      'is_permutation': sorted(perm) == list(range(n))}
    return out, torch.__version__


def tensor_report(tensors, torch):
    count = elements = nonfinite = 0
    for t in tensors:
        if torch.is_tensor(t) and t.is_floating_point():
            count += 1
            elements += t.numel()
            nonfinite += int((~torch.isfinite(t)).sum())
    return {'float_tensors': count, 'elements': elements, 'nonfinite_elements': nonfinite}


def storage_sha256s(t):
    return sha256_bytes(t.detach().cpu().contiguous().numpy().tobytes())


def official_storages(path):
    """sha256 of every raw tensor storage in a torch zip checkpoint (archive/data/<k>); no unpickling."""
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if '/data/' in n and not n.endswith('/')]
        return sorted(sha256_bytes(z.read(n)) for n in names)


def audit_sidecar(path, entry, torch, identities, run_id, final_official=None):
    st = torch.load(path, map_location='cpu', weights_only=True)
    models = {m: tensor_report(st['models'][m].values(), torch) for m in SIDECAR_MODELS}
    opt = st['optimizer']
    adam_tensors, steps = [], set()
    for s in opt['state'].values():
        adam_tensors += [s['exp_avg'], s['exp_avg_sq']]
        steps.add(float(s['step']))
    out = {'path': entry['path'], 'epoch': entry['completed_epoch'],
           'identity_ok': (st['kind'] == 'E06C_ENGINEERING_RESUME_STATE' and st['method_id'] == 'E06c'
                           and st['runner_mode'] == 'SCIENTIFIC' and st['seed'] == EXPECTED['seed']
                           and st['run_id'] == run_id and st['scientific_checkpoint'] is False
                           and st['selection_candidate'] is False and st['identities'] == identities
                           and st['execution_mode'] == EXPECTED['execution_mode']
                           and st['resolved_config_sha256'] == entry['resolved_config_sha256']),
           'completed_epoch': st['completed_epoch'], 'global_step': st['global_step'],
           'optimizer_applications_total': st['optimizer_applications_total'],
           'models': models, 'adam': dict(tensor_report(adam_tensors, torch), param_states=len(opt['state']),
                                          step_values=sorted(steps)),
           'adam_lr': sorted({g['lr'] for g in opt['param_groups']})}
    if final_official is not None:   # tensor-storage identity: official whole-module pickles vs sidecar state_dicts
        out['official_storage_match'] = {}
        for model, rel in final_official.items():
            side = sorted(storage_sha256s(t) for t in st['models'][model].values())
            off = official_storages(rel)
            out['official_storage_match'][model] = {'sidecar_tensors': len(side), 'official_storages': len(off),
                                                    'equal_multiset': side == off}
    del st
    return out


def collect_atime(relation_path, split_path):
    import pyarrow.parquet as pq
    t = pq.read_table(split_path, columns=['sample_id', 'split'])
    split = dict(zip(t.column('sample_id').to_pylist(), t.column('split').to_pylist()))
    r = pq.read_table(relation_path, columns=['source_spoof_id', 'target_live_id'])
    rel = set(r.column('source_spoof_id').to_pylist()) | set(r.column('target_live_id').to_pylist())

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
        c = split.get(sid, 'UNMAPPED') + ('_IN_TRAIN_RELATION' if sid in rel else '')
        s = stats.setdefault(c, {'files': 0, 'sensitive_whole_window': 0, 'atime_changed': 0,
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
    import torch
    from methods.dsdg import runner_io as rio
    root = Path(run_root)
    before_tree, before_small = tree_snapshot(root), small_hashes(root)
    git = {'head': subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip(),
           'branch': subprocess.run(['git', 'branch', '--show-current'], capture_output=True, text=True).stdout.strip(),
           'status_porcelain': subprocess.run(['git', 'status', '--porcelain'], capture_output=True,
                                              text=True).stdout}
    contract = rio.load_contract()            # verifies every bound input SHA256 in the GPU checkout
    identities = rio.identities(contract)
    man = json.loads((root / 'run_manifest.json').read_text())
    summ = json.loads((root / 'run_summary.json').read_text())
    cindex = json.loads((root / 'checkpoint_index.json').read_text())
    rindex = json.loads((root / 'resume_state_index.json').read_text())
    metrics, events, hashes = scan_metrics(root / 'metrics.jsonl')
    relation = Path(repo) / rio.TRAIN_RELATION
    pred, torch_version = predict_order(relation, EXPECTED['seed'], EPOCHS)
    batch_match = sum(hashes.get(e, [])[i] == h for e, p in pred.items() for i, h in enumerate(p['batches'])
                      if i < len(hashes.get(e, [])))
    complete = {p['epoch']: p for p in events.get('e06c_epoch_complete', [])}
    epoch_events_ok = sum(c['rows'] == TRAIN_ROWS and c['unique_pair_ids'] == TRAIN_ROWS
                          and c['optimizer_steps'] == STEPS_PER_EPOCH and c['global_step_end'] == e * STEPS_PER_EPOCH
                          for e, c in complete.items())
    epoch_order_match = sum(complete.get(e, {}).get('epoch_pair_order_sha256') == p['epoch_order']
                            for e, p in pred.items())
    # official checkpoints: re-hash every indexed file
    official = []
    for c in cindex['checkpoints']:
        p = root / c['path']
        official.append(dict(c, exists=p.is_file(), size_actual=p.stat().st_size if p.is_file() else None,
                             sha256_actual=sha256_file(p) if p.is_file() else None))
    on_disk = sorted(str(p.relative_to(root)) for p in (root / 'checkpoints').iterdir())
    final_official = {m: root / 'checkpoints' / f'{prefix}model_epoch_200_iter_0.pth' for m, prefix in OFFICIAL}
    sidecars = []
    for entry in rindex['entries']:
        p = root / entry['path']
        sha = sha256_file(p)
        rep = audit_sidecar(p, entry, torch, identities, EXPECTED['run_id'],
                            final_official if entry['completed_epoch'] == EPOCHS else None)
        sidecars.append(dict(rep, size=p.stat().st_size, size_indexed=entry['file_size_bytes'], sha256=sha,
                             sha256_indexed=entry['sha256'], global_step_indexed=entry['global_step']))
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
    atime = collect_atime(relation, Path(repo) / 'manifests' / 'split_v1.parquet')
    digest, count = listing_digest(root)
    after_tree, after_small = tree_snapshot(root), small_hashes(root)
    parent = sorted(os.listdir(root.parent))
    return {
        'collector': {'torch_version': torch_version, 'python': sys.version.split()[0],
                      'utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')},
        'git': git, 'identities_recomputed': identities, 'run_manifest': man, 'run_summary': summ,
        'checkpoint_index_meta': {k: v for k, v in cindex.items() if k != 'checkpoints'},
        'resume_index_meta': {k: v for k, v in rindex.items() if k != 'entries'},
        'metrics': metrics, 'metrics_sha256': sha256_file(root / 'metrics.jsonl'),
        'run_start_event': events.get('e06c_run_start', [None])[0],
        'resume_reconciliation_events': len(events.get('e06c_resume_reconciliation', [])),
        'checkpoint_events': [{'epoch': c['epoch'], 'global_step': c['global_step'],
                               'official': c['official'], 'resume_state': c['resume_state']}
                              for c in events.get('e06c_checkpoint_event', [])],
        'visualization_epochs': [v['epoch'] for v in events.get('e06c_visualization_diagnostic', [])],
        'visualization_roles': sorted({f"{v['role']}|{v['data']}|{v['not_validation']}|{v['selects_nothing']}"
                                       for v in events.get('e06c_visualization_diagnostic', [])}),
        'order_prediction': {'torch_version': torch_version, 'predicted_steps': sum(len(p['batches'])
                                                                                   for p in pred.values()),
                             'batch_hash_match': batch_match, 'epoch_order_match': epoch_order_match,
                             'permutations': sum(p['is_permutation'] for p in pred.values()),
                             'epoch_complete_events_ok': epoch_events_ok},
        'official_checkpoints': official, 'checkpoints_dir_listing': on_disk, 'sidecars': sidecars,
        'logs': logs, 'exit_file': Path(EXIT_FILE).read_text().strip(),
        'exit_file_mtime_utc': utc(Path(EXIT_FILE).stat().st_mtime),
        'lightcnn': {'sha256': sha256_file(lightcnn), 'bytes': lightcnn.stat().st_size},
        'train_relation_sha256': sha256_file(relation),
        'config_file_sha256': sha256_file(Path(repo) / man['config_path']),
        'm6d5e_files_now': {k: sha256_file(Path(repo) / k) for k in man['m6d5e_file_sha256']},
        'atime_firewall': atime,
        'small_file_sha256': after_small, 'run_root_listing_digest': digest, 'run_root_file_count': count,
        'run_root_tree_sha256': sha256_bytes(json.dumps(after_tree).encode()),
        'run_root_unchanged_whole_audit': before_tree == after_tree and before_small == after_small,
        'parent_listing': parent,
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
    """Deterministic checkpoint manifest: official checkpoints then engineering sidecars, sorted by epoch/path."""
    head = (f"# E06c DSDG-BIN-IDFREE seed42 run {EXPECTED['run_id']} (authority {EXPECTED['git_commit'][:8]}): "
            'epoch global_step checkpoint_type selected_for_final size_bytes sha256 path\n')
    rows = sorted(c['official_checkpoints'], key=lambda r: (r['epoch'], r['path']))
    body = ''.join(f"{r['epoch']:03d} {r['global_step']} {r['checkpoint_type']} {str(r['selected_for_final']).lower()} "
                   f"{r['file_size_bytes']} {r['sha256']} {r['path']}\n" for r in rows)
    side = sorted(c['sidecars'], key=lambda r: r['epoch'])
    body += '# engineering resume sidecars (ENGINEERING_RESUME_ONLY; not scientific checkpoints, never selected)\n'
    body += ''.join(f"{r['epoch']:03d} {r['global_step']} engineering_resume false {r['size']} {r['sha256']} "
                    f"{r['path']}\n" for r in side)
    return head + body


def build(c, repo_root):
    repo = Path(repo_root)
    G = Gates()
    man, summ, m = c['run_manifest'], c['run_summary'], c['metrics']
    E = EXPECTED

    # A. host / authority
    G('A.host', man['host'] == E['host'] and man['environment']['host'] == E['host'])
    G('A.gpu_checkout_authority', c['git']['head'] == E['git_commit'] and c['git']['branch'] == 'm6-baselines'
      and c['git']['status_porcelain'] == '', c['git'])
    G('A.exit_zero', c['exit_file'] == 'E06C_EXIT=0', c['exit_file'])

    # B. run identity
    for k, v in (('method_id', E['method_id']), ('experiment_seed', E['seed']), ('run_id', E['run_id']),
                 ('git_commit', E['git_commit']), ('git_dirty', False), ('runner_mode', 'SCIENTIFIC'),
                 ('scientific_run', True), ('qualification_only', False), ('fidelity_class', E['fidelity_class']),
                 ('deviation', E['deviation']), ('execution_mode', E['execution_mode']),
                 ('nominal_global_batch', 240), ('max_microbatch', 20), ('tail_batch', 198), ('drop_last', False),
                 ('config_sha256', E['config_sha256']), ('completion_status', 'completed'),
                 ('test_split_accessed', False), ('val_split_accessed', False), ('snapshot_verified', True)):
        G(f'B.manifest.{k}', man.get(k) == v, man.get(k))
    run_id = sha256_bytes(f"{E['method_id']}|{E['seed']}|{E['config_sha256']}|{E['git_commit']}".encode())[:16]
    G('B.run_id_rule', run_id == E['run_id'], run_id)
    G('B.command_line', man['command_line'].endswith('tools/run_e06c.py --seed 42 --execution-config '
                                                     'configs/execution/m5_gpu_3090.yaml')
      and '--resume-state' not in man['command_line'])
    for k, v in (('run_id', E['run_id']), ('experiment_seed', E['seed']), ('completion_status', 'completed'),
                 ('checkpoint_selection_rule', E['checkpoint_rule']), ('optimizer_applications', TOTAL_STEPS),
                 ('test_split_accessed', False), ('failure_reason', None)):
        G(f'B.summary.{k}', summ.get(k) == v, summ.get(k))

    # C. config / data / environment identities
    ids = man['identities']
    G('C.identities_equal_recomputed_contract', ids == c['identities_recomputed'])
    G('C.config_file_sha256', c['config_file_sha256'] == E['config_sha256'] == ids['config_sha256'])
    G('C.train_relation_sha256', c['train_relation_sha256'] == E['train_relation_sha256']
      == ids['train_relation_sha256'])
    G('C.lightcnn', c['lightcnn']['sha256'] == ids['lightcnn_sha256'] and c['lightcnn']['bytes'] == 123844849)
    G('C.environment_lock', man['dependency_fingerprint'] == ids['environment_lock_sha256']
      and man['environment_name'] == 'gpat-m6-e06c')
    G('C.m6d5e_files_unchanged', c['m6d5e_files_now'] == man['m6d5e_file_sha256'])
    G('C.resolved_config_sha256', c['small_file_sha256']['resolved_config.yaml'] == man['resolved_config_sha256'])
    G('C.spec_sha256', sha256_file(repo / SPEC_PATH) == SPEC_SHA256)
    G('C.repo_config_sha256', sha256_file(repo / man['config_path']) == E['config_sha256'])

    # D. training completion / step accounting
    G('D.step_records_7400', m['step_records'] == TOTAL_STEPS and m['raw_lines'] == TOTAL_STEPS + 243)
    G('D.contiguous_1_7400', m['contiguous'] and m['step_min'] == 1 and m['step_max'] == TOTAL_STEPS
      and not m['duplicates'] and m['missing_count'] == 0)
    G('D.epochs_1_200', m['epochs_present'] == list(range(1, EPOCHS + 1)))
    G('D.epoch_plan_36x240_198', m['epoch_sizes_ok'] == EPOCHS and m['epoch_rows'] == [TRAIN_ROWS])
    G('D.record_schema', m['schema_violation_count'] == 0, m['schema_violations'])
    G('D.event_counts', m['event_counts'] == {'e06c_checkpoint_event': 21, 'e06c_epoch_complete': EPOCHS,
                                              'e06c_run_start': 1, 'e06c_visualization_diagnostic': 21},
      m['event_counts'])
    G('D.fresh_no_resume', c['resume_reconciliation_events'] == 0 and c['run_start_event'] is not None)
    o = c['order_prediction']
    G('D.batch_order_prediction_7400', o['predicted_steps'] == TOTAL_STEPS and o['batch_hash_match'] == TOTAL_STEPS, o)
    G('D.epoch_order_prediction_200', o['epoch_order_match'] == EPOCHS and o['permutations'] == EPOCHS, o)
    G('D.every_pair_once_per_epoch', o['epoch_complete_events_ok'] == EPOCHS)

    # E. numerical health
    G('E.no_corrupted_lines', m['corrupted_lines'] == [])
    G('E.all_losses_finite', m['nonfinite_steps'] == [])
    G('E.cls_pair_zero', m['loss_ranges']['loss_cls'] == [0.0, 0.0] and m['loss_ranges']['loss_pair'] == [0.0, 0.0])
    side = sorted(c['sidecars'], key=lambda r: r['epoch'])
    G('E.sidecar_parameters_finite', all(r['models'][n]['nonfinite_elements'] == 0 and r['models'][n]['elements'] > 0
                                         for r in side for n in SIDECAR_MODELS))
    G('E.sidecar_adam_finite', all(r['adam']['nonfinite_elements'] == 0 and r['adam']['elements'] > 0 for r in side))
    G('E.sidecar_adam_steps', all(r['adam']['step_values'] == [float(r['global_step'])] for r in side))
    G('E.sidecar_adam_lr', all(r['adam_lr'] == [E['learning_rate']] for r in side))
    lg = c['logs']
    G('E.stderr_empty', lg['stderr.log']['bytes'] == 0)
    G('E.stdout_only_checkpoint_lines', lg['stdout.log']['other_lines'] == []
      and lg['stdout.log']['checkpoint_saved_lines'] == 63 and not any(lg['stdout.log']['pattern_hits'].values()))

    # F. checkpoints
    off = c['official_checkpoints']
    G('F.official_count_63', len(off) == 63 and len(c['checkpoint_events']) == 21)
    G('F.official_epochs', sorted({r['epoch'] for r in off}) == CHECKPOINT_EPOCHS
      and [e['epoch'] for e in c['checkpoint_events']] == CHECKPOINT_EPOCHS)
    G('F.official_files_match_index', all(r['exists'] and r['size_actual'] == r['file_size_bytes']
                                          and r['sha256_actual'] == r['sha256'] for r in off))
    G('F.official_global_steps', all(r['global_step'] == r['epoch'] * STEPS_PER_EPOCH for r in off))
    sel = [r for r in off if r['selected_for_final']]
    G('F.single_selected_final_netG', len(sel) == 1 and sel[0]['path'] == FINAL_NETG and sel[0]['epoch'] == EPOCHS
      and sel[0]['checkpoint_type'] == 'selected' and sel[0]['selection_reason'] == E['checkpoint_rule'])
    G('F.non_final_not_selected', all(r['selected_for_final'] is False and r['checkpoint_type'] ==
                                      ('terminal' if r['epoch'] == EPOCHS else 'periodic')
                                      for r in off if r['path'] != FINAL_NETG))
    G('F.summary_final_netG', summ['final_or_selected_checkpoint_path'] == f"{E['run_root']}/{FINAL_NETG}"
      and summ['final_or_selected_checkpoint_sha256'] == sel[0]['sha256'] if sel else False)
    G('F.index_meta', c['checkpoint_index_meta'].get('selected_final') == 'netG_model_epoch_200_iter_0.pth'
      and c['checkpoint_index_meta'].get('checkpoint_rule') == E['checkpoint_rule']
      and c['checkpoint_index_meta'].get('scientific_checkpoints') is True)
    ev_paths = {(e['epoch'], o2['path'], o2['sha256']) for e in c['checkpoint_events'] for o2 in e['official']}
    G('F.events_match_index', ev_paths == {(r['epoch'], r['path'], r['sha256']) for r in off})
    G('F.sidecars_21', [r['epoch'] for r in side] == CHECKPOINT_EPOCHS)
    G('F.sidecar_files_match_index', all(r['size'] == r['size_indexed'] and r['sha256'] == r['sha256_indexed']
                                         for r in side))
    G('F.sidecar_identity', all(r['identity_ok'] and r['completed_epoch'] == r['epoch']
                                and r['global_step'] == r['global_step_indexed'] == r['epoch'] * STEPS_PER_EPOCH
                                and r['optimizer_applications_total'] == r['global_step'] for r in side))
    G('F.sidecar_events_match', {(e['epoch'], e['resume_state']['sha256']) for e in c['checkpoint_events']}
      == {(r['epoch'], r['sha256']) for r in side})
    G('F.resume_index_meta', c['resume_index_meta'].get('scientific_checkpoint') is False
      and c['resume_index_meta'].get('selection_candidate') is False
      and c['resume_index_meta'].get('run_id') == E['run_id'])
    final = side[-1] if side else {}
    G('F.final_official_equals_final_sidecar', bool(final.get('official_storage_match')) and all(
        v['equal_multiset'] and v['sidecar_tensors'] == v['official_storages'] > 0
        for v in final['official_storage_match'].values()), final.get('official_storage_match'))
    expected_listing = sorted([r['path'] for r in off] + [r['path'] for r in side])
    G('F.checkpoints_dir_exact', sorted(c['checkpoints_dir_listing']) == expected_listing)   # no .partial/stray file

    # G. firewall
    a = c['atime_firewall']
    test = {k: v for k, v in a['by_class'].items() if k.split('_')[0] == 'TEST'}    # split_v1 labels TRAIN/VAL/TEST
    val = {k: v for k, v in a['by_class'].items() if k.split('_')[0] == 'VAL'}
    G('G.test_faces_atime_unchanged', test and all(v['atime_changed'] == 0 and v['missing_post'] == 0
                                                   for v in test.values()), test)
    G('G.test_faces_whole_window_sensitive', test and all(v['sensitive_whole_window'] == v['files']
                                                          for v in test.values()))
    G('G.val_faces_atime_unchanged', val and all(v['atime_changed'] == 0 for v in val.values()), val)
    G('G.no_non_relation_face_read', all(v['atime_changed'] == 0 for k, v in a['by_class'].items()
                                         if not k.endswith('_IN_TRAIN_RELATION')))
    G('G.no_face_mtime_change', all(v['mtime_changed'] == 0 and v['missing_post'] == 0
                                    for v in a['by_class'].values()))
    mf = {x['file']: x for x in a['manifests']}
    G('G.val_test_manifests_untouched_in_run', not mf['val_pairs_v1.parquet']['changed'] and all(
        not x['changed'] or x['atime_post_utc'] < RUN_WINDOW[0] for n, x in mf.items()
        if n not in ('pairs_train_v1.parquet',)), a['manifests'])
    G('G.no_val_records', m['schema_violation_count'] == 0)   # every step record checked val_losses/val_metrics null
    G('G.visualization_is_train_only', c['visualization_epochs'] == CHECKPOINT_EPOCHS and c['visualization_roles'] == [
        'UPSTREAM_TRAINING_SAMPLE_VISUALIZATION|TRAIN (last global batch of the epoch)|True|True'])

    # H. run-root integrity
    G('H.run_root_unchanged_whole_audit', c['run_root_unchanged_whole_audit'])
    G('H.parent_only_seed42', [p for p in c['parent_listing'] if not p.startswith('.')] == ['seed_42'],
      c['parent_listing'])

    text = manifest_text(c)
    sel0 = sel[0] if sel else {}
    doc = {
        'classification': CLASSIFICATION, 'status': STATUS_BLOCKED if G.failed else STATUS_OK,
        'failed_gates': G.failed, 'gates': G.results,
        'method': {'id': 'E06c', 'name': E['method_name'], 'fidelity_class': E['fidelity_class'],
                   'deviation': E['deviation'], 'execution_mode': E['execution_mode']},
        'seed': E['seed'], 'run_id': E['run_id'], 'run_root': E['run_root'], 'host': E['host'],
        'ssh_address_note': 'GPU host reached as student20261@10.31.131.201 (network-address change only; same '
                            'hostname, paths and run identity as the launch on 100.121.84.44)',
        'launch_authority_commit': E['git_commit'],
        'authority_note': 'Scientific training authority for all E06c seeds stays 84809868; this closure commit is '
                          'evidence/reporting only.',
        'frozen': {'config_sha256': E['config_sha256'], 'spec_sha256': SPEC_SHA256,
                   'train_relation_sha256': E['train_relation_sha256'], 'resolved_config_sha256':
                   man['resolved_config_sha256'], 'identities': ids},
        'training': {'completed_epoch': EPOCHS, 'global_step': m['step_max'], 'step_records': m['step_records'],
                     'optimizer_applications': summ['optimizer_applications'], 'steps_per_epoch': STEPS_PER_EPOCH,
                     'epoch_plan': '36 x 240 + 1 x 198 = 8838 TRAIN pairs', 'run_window_utc': [man['start_utc'],
                                                                                              man['end_utc']],
                     'first_last_step_utc': [m['first_step_utc'], m['last_step_utc']],
                     'training_duration_seconds': summ['training_duration_seconds'],
                     'step_seconds': m['step_seconds'], 'peak_step_gpu_memory_bytes': m['peak_step_gpu_memory_bytes'],
                     'order_prediction': c['order_prediction']},
        'numerical_health': {'nonfinite_steps': len(m['nonfinite_steps']), 'corrupted_lines': len(m['corrupted_lines']),
                             'loss_ranges': m['loss_ranges'],
                             'loss_mean_at_checkpoint_epochs': m['loss_mean_at_checkpoint_epochs'],
                             'note': 'loss_cls = loss_pair = 0 by construction (one-logit CE, lambda_pair = 0)',
                             'final_sidecar': {k: final.get(k) for k in ('models', 'adam', 'official_storage_match')},
                             'sidecars_checked': len(side),
                             'method': 'runner_state_epoch_<E>.pth via torch.load(weights_only=True, '
                                       'map_location=cpu); official whole-module .pth never unpickled',
                             'logs': c['logs']},
        'checkpoints': {'events': len(c['checkpoint_events']), 'epochs': CHECKPOINT_EPOCHS, 'official_files': len(off),
                        'final_netG': {'path': FINAL_NETG, 'size_bytes': sel0.get('file_size_bytes'),
                                       'sha256': sel0.get('sha256'), 'epoch': sel0.get('epoch'),
                                       'global_step': sel0.get('global_step'),
                                       'checkpoint_type': sel0.get('checkpoint_type'),
                                       'selected_for_final': sel0.get('selected_for_final'),
                                       'selection_rule': E['checkpoint_rule']},
                        'engineering_sidecars': {'count': len(side), 'role': 'ENGINEERING_RESUME_ONLY; not '
                                                 'scientific checkpoints; never selected'},
                        'manifest_file': EVIDENCE['manifest'], 'manifest_sha256': sha256_bytes(text.encode())},
        'firewall': {'test_benchmark_access': 0 if not G.failed else None, 'val_checkpoint_selection_access': 0,
                     'val_split_accessed': man['val_split_accessed'], 'test_split_accessed': man['test_split_accessed'],
                     'atime': a, 'structural': 'CanonicalFaceReader resolves only TRAIN-relation sample_ids; the '
                     'runner has no VAL pass; selection rule is OFFICIAL_GENERATOR_EPOCH_200',
                     'upstream_test_block': 'the upstream "test" block writes TRAIN-sample visualization PNGs '
                                            '(diagnostics/visualization); it is not benchmark TEST',
                     'auditor_reads': 'split_v1.parquet + inventory.parquet atime 2026-10-18T06:43:31Z = pre-launch '
                                      'auditor read; split_v1.parquet + pairs_train_v1.parquet re-read by this audit '
                                      'AFTER the post snapshot'},
        'run_root_integrity': {'file_count': c['run_root_file_count'], 'listing_digest': c['run_root_listing_digest'],
                               'listing_digest_convention': 'find . -type f -printf "%P %s %T@\\n" | sort | sha256sum',
                               'tree_sha256': c['run_root_tree_sha256'], 'metrics_jsonl_sha256': c['metrics_sha256'],
                               'small_file_sha256': c['small_file_sha256'],
                               'final_netG_sha256': sel0.get('sha256'),
                               'final_sidecar_sha256': final.get('sha256'),
                               'unchanged_whole_audit': c['run_root_unchanged_whole_audit']},
        'confirmations': {'val_checkpoint_selection': False, 'test_selection': False, 'best_seed_logic': False,
                          'bank_generation': False, 'downstream_evaluation': False, 'run_root_mutated': False,
                          'seed2026_started': False, 'other_method_started': False, 'gpat_started': False},
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
