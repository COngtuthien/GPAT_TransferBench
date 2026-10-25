"""E05 seed closure: read-only completion audit of one E05 PCGAN (controlled architecture resolution) scientific run
(CONTROLLED_ADAPTATION, track A_FAIR_COMMON_IDENTITY_FREE, training authority efdf753 for every E05 seed).

Two phases:

  collect  (GPU host, gpat-m6-e05 env, launch environment of the audited seed, CUDA_VISIBLE_DEVICES=0,
           `python -I -B - collect --seed <s> --repo <gpu repo> < this file`) -- read-only, prints one JSON document.
           Run-root files are opened 'rb'. The two checkpoints are read with torch.load(weights_only=True) only (no
           arbitrary pickle code) and every floating tensor is checked finite. The run root is listed
           (path/size/sha256) before and after the audit. VAL/TEST faces are never opened: firewall evidence is the
           run manifest/summary/metrics, the content-level match of every consumed face PNG digest against the M2
           sample accounting of TRAIN samples, and a stat-only before/after atime comparison of the faces_256
           snapshot taken at launch (~/workdir/e05_s<seed>_atime_{pre,post}.tsv). The sampler order is re-derived
           independently (one int64 draw, then randperm(8838) from torch.Generator().manual_seed(seed)). The live
           environment is re-read and compared with the lock under the owner driver ruling (version only).
  build    (repository, stdlib only) -- gates the collected document and writes
           outputs/audit/E05_SEED<s>_{COMPLETION.json, CHECKPOINT_MANIFEST.txt}. COMPLETION.json embeds the collected
           document so the build can be re-derived on any host.

No training, selection, VAL, TEST, bank or downstream code is run.
"""
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

CLASSIFICATION = 'SCIENTIFIC_TRAINING_COMPLETION_AUDIT'
RUNTIME = '/home/student20261/workdir/GPAT_TransferBench_runtime'
WORKDIR = '/home/student20261/workdir'
AUTHORITY = 'efdf753156045dd2bed52229d9eac869b18d72f8'
CONFIG_SHA = '478756e150c427832315800acba954cedf778bac520eee3bde8cabf7359e71de'
RUN_IDS = {42: '18463bf5cd471e55', 1337: '036c806bb2f339b7'}
EXPECTED = {
    'method_id': 'E05', 'method_name': 'PCGAN (controlled architecture resolution)', 'host': 'nd-System-Product-Name',
    'fidelity': 'CONTROLLED_ADAPTATION', 'track': 'A_FAIR_COMMON_IDENTITY_FREE', 'resume': 'NOT_QUALIFIED_FRESH_ONLY',
    'checkpoint_rule': 'BASELINE_FINAL_STATE_V1', 'learning_rate': 1e-6, 'environment_name': 'gpat-m6-e05',
    'python_executable': '/home/student20261/miniconda3/envs/gpat-m6-e05/bin/python',
    'python_sha256': 'd3255ce6eac7a06a653a6f238d703ebf863423912b39ec957334ce2974364244',
    'extensions': {'fused_bias_act': '5afdb2369817ffea68530b1e88d75596a77ed08a397f83122b618a18225718ab',
                   'upfirdn2d': '40ba9db73e1790cd3d904a1fed09c5441180c6995a9a396030cfaa52efd1d081'},
    'pair_manifest_sha256': 'a5e4fdaef236f15730c7e3885b537e08e995faffbe654167fc940f44bbc75243',
    'lock_sha256': '849a100430e36048acde80d58801ae8908c15ae43719e97f4dcb90d77564ac50',
    'source_commit': '6baa180f1184ee79a6b967f9d80ee0e02a979ac7',
    'gpu_model': 'NVIDIA GeForce RTX 3090', 'gpu_uuid': 'GPU-b722cd9d-c9fa-223c-915e-cc09d9b862b7',
}
DRIVER_RULING = {'lock_driver_version': '595.84', 'observed_driver_version': '580.178.04',
                 'driver_policy': 'OWNER_APPROVED_DRIVER_VERSION_ONLY_RUNTIME_COMPATIBILITY',
                 'scope': ('only the NVIDIA driver version may differ; GPU model, GPU UUID and every other lock field '
                           'are exact gates; this is not evidence that arbitrary driver changes are acceptable')}
ITERATIONS, PAIR_ROWS = 4000, 8838
G_TERMS = ('L_rec', 'L_recblur', 'L_advrec', 'L_advmix', 'L_pat')
D_TERMS = ('L_D_real', 'L_D_rec', 'L_D_mix', 'L_D_patch_real', 'L_D_patch_fake')
LOSS_KEYS = G_TERMS + ('L_G_total',) + D_TERMS + ('L_D_image', 'L_D_patch', 'L_D_total')
GROUP_TENSORS = {'D': 80, 'G': 59}
MODELS = ('Encoder', 'Generator', 'ImageD', 'PatchD')
CHECKPOINTS = {1: ('checkpoints/iteration_0001.pth', 'periodic', False),
               4000: ('checkpoints/iteration_4000.pth', 'selected', True)}
FORBIDDEN_RECORD_KEYS = ('test', 'subject', 'image')
SMALL_FILES = ('run_manifest.json', 'run_summary.json', 'checkpoint_index.json', 'resolved_config.yaml',
               'metrics.jsonl', 'stdout.log', 'stderr.log', 'generation_log.jsonl')
LOG_PATTERNS = ('Traceback', 'Error', 'error', 'OutOfMemory', 'out of memory', 'Killed', 'signal', 'SIGTERM',
                'SIGKILL', 'No space', 'Errno', 'DataLoader', 'checkpoint', 'non-finite', 'nonfinite', 'nan', 'inf',
                'contract', 'Warning')
TOL_G_TOTAL = 1e-5                                   # relative; FP32 rounding of a 5-term sum


def paths(seed):
    return {'run_root': f'{RUNTIME}/runs/m6/E05/seed_{seed}', 'tmux': f'm6_e05_s{seed}',
            'exit': f'{WORKDIR}/e05_s{seed}.exit', 'command': f'{WORKDIR}/e05_s{seed}.command',
            'atime_pre': f'{WORKDIR}/e05_s{seed}_atime_pre.tsv', 'atime_post': f'{WORKDIR}/e05_s{seed}_atime_post.tsv'}


def evidence(seed):
    a = f'outputs/audit/E05_SEED{seed}_'
    return {'completion': a + 'COMPLETION.json', 'manifest': a + 'CHECKPOINT_MANIFEST.txt',
            'report': a + 'COMPLETION.md', 'sha256': a + 'EVIDENCE_SHA256.txt'}


def command(seed, flags=''):
    return (f"{EXPECTED['python_executable']} {flags}tools/run_e05.py --seed {seed} "
            '--execution-config configs/execution/m5_gpu_3090.yaml')


def run_id(seed, commit=AUTHORITY):
    return hashlib.sha256(f'E05|{seed}|{CONFIG_SHA}|{commit}'.encode()).hexdigest()[:16]


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
    return sorted(out)


def sha_listing(root):
    """LC_ALL=C (byte-order) sorted `<path> <size> <sha256>` lines; digest = sha256 of the newline-terminated lines."""
    lines = [f'{p} {s} {sha256_file(Path(root) / p)}' for p, s, _ in tree_snapshot(root)]
    return sha256_bytes(''.join(f'{x}\n' for x in lines).encode()), len(lines), lines


def small_hashes(root):
    return {n: sha256_file(root / n) for n in SMALL_FILES if (root / n).exists()}


def finite(v):
    return isinstance(v, float) and math.isfinite(v)


# ============================================================================ collect (GPU host, read-only)
def predict_order(seed, n=PAIR_ROWS):
    """torch RandomSampler(generator=G) with G.manual_seed(seed): one int64 base draw, then randperm(n)."""
    import torch
    g = torch.Generator()
    g.manual_seed(seed)
    torch.empty((), dtype=torch.int64).random_(generator=g)
    return torch.randperm(n, generator=g).tolist(), torch.__version__


def scan_metrics(path, seed, pairs, split, faces):
    records, events, bad_lines = [], {}, 0
    with open(path, 'rb') as f:
        raw = f.read()
    for line in raw.decode('utf-8').splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            bad_lines += 1
            continue
        if r.get('record_type') == 'epoch':
            records.append(r)
        else:
            events.setdefault(r.get('event'), []).append(r.get('payload'))
    pred, torch_version = predict_order(seed)
    n = len(records)
    steps = [r['global_step'] for r in records]
    c = {'lines': len(raw.decode('utf-8').splitlines()), 'bad_lines': bad_lines, 'records': n,
         'trailing_newline': raw.endswith(b'\n'),
         'global_step_is_1_to_n': steps == list(range(1, n + 1)),
         'benchmark_iteration_equals_global_step': sum(r['benchmark_iteration'] == r['global_step'] for r in records),
         'distinct_global_steps': len(set(steps)), 'first_step': steps[0] if steps else None,
         'last_step': steps[-1] if steps else None, 'epoch_values': sorted({r['epoch'] for r in records}),
         'seed_values': sorted({r['experiment_seed'] for r in records}),
         'runner_modes': sorted({r['runner_mode'] for r in records}),
         'granularity': sorted({r['record_granularity'] for r in records}),
         'utc_nondecreasing': all(a['utc'] <= b['utc'] for a, b in zip(records, records[1:])),
         'first_utc': records[0]['utc'] if records else None, 'last_utc': records[-1]['utc'] if records else None}
    c['forbidden_key_records'] = sum(any(any(w in k.lower() for w in FORBIDDEN_RECORD_KEYS) for k in r)
                                     for r in records)
    # data order / scope
    sampler = [r['sampler_index'] for r in records]
    c['order'] = {'torch_version': torch_version, 'predicted_positions': min(n, len(pred)),
                  'match': sum(s == pred[i] for i, s in enumerate(sampler)),
                  'distinct_sampler_indices': len(set(sampler)),
                  'distinct_pair_ids': len({r['pair_id'] for r in records}),
                  'sampler_order_sha256': sha256_bytes(json.dumps(sampler).encode()),
                  'predicted_order_sha256': sha256_bytes(json.dumps(pred[:n]).encode()),
                  'pair_id_order_sha256': sha256_bytes('\n'.join(r['pair_id'] for r in records).encode())}
    roles = manifest_row = pair_hash = png = train = 0
    datasets, consumed = {}, set()
    for r in records:
        row = pairs[r['sampler_index']]
        manifest_row += (row['pair_id'] == r['pair_id'] and row['source_spoof_id'] == r['source_spoof_id'] and
                         row['target_live_id'] == r['target_live_id'] and row['dataset'] == r['dataset'] and
                         row['split'] == 'TRAIN')
        key = f"{r['pair_id']}|{r['source_spoof_id']}>{r['target_live_id']}"
        pair_hash += sha256_bytes(key.encode()) == r['pair_sha256']
        s, t = split.get(r['source_spoof_id']), split.get(r['target_live_id'])
        roles += bool(s and t and s['label_binary'] == 1 and t['label_binary'] == 0)
        train += bool(s and t and s['split'] == t['split'] == 'TRAIN' and s['dataset'] == t['dataset'] == r['dataset'])
        png += (faces.get(r['source_spoof_id']) == r['source_png_sha256'] and
                faces.get(r['target_live_id']) == r['target_png_sha256'])
        datasets[r['dataset']] = datasets.get(r['dataset'], 0) + 1
        consumed.update((r['source_spoof_id'], r['target_live_id']))
    c['pairs'] = {'manifest_row_match': manifest_row, 'pair_sha256_match': pair_hash,
                  'source_spoof_target_live': roles, 'both_faces_train_same_dataset': train,
                  'face_png_sha256_match_m2_accounting': png, 'records_by_dataset': dict(sorted(datasets.items())),
                  'consumed_sample_ids': len(consumed)}
    c['input_contract'] = sum(r['input_shape'] == [1, 3, 256, 256] and r['input_dtype'] == 'torch.float32' and
                              -1.0 <= r['input_range'][0] <= r['input_range'][1] <= 1.0 and r['batch_size'] == 1
                              for r in records)
    c['applications_exact'] = sum(r['optimizer_applications'] == {
        'D': r['global_step'], 'G': r['global_step'], 'hook_D': r['global_step'], 'hook_G': r['global_step'],
        'total': 2 * r['global_step']} for r in records)
    c['final_applications'] = records[-1]['optimizer_applications'] if records else None
    c['learning_rates_exact'] = sum(r['learning_rates'] == {'D': EXPECTED['learning_rate'], 'G': EXPECTED['learning_rate']}
                                    and r['learning_rate'] == EXPECTED['learning_rate'] for r in records)
    # numerics
    c['loss_keys_exact'] = sum(sorted(r['train_losses']) == sorted(LOSS_KEYS) for r in records)
    c['losses_finite'] = sum(all(finite(r['train_losses'][k]) for k in LOSS_KEYS) for r in records)
    c['nonfinite_values'] = sum(not finite(v) for r in records for v in r['train_losses'].values())
    c['g_terms_active'] = sum(all(r['train_losses'][k] > 0 for k in G_TERMS) for r in records)
    rel = [abs(sum(r['train_losses'][k] for k in G_TERMS) - r['train_losses']['L_G_total']) /
           abs(r['train_losses']['L_G_total']) for r in records]
    c['g_total_unit_sum_max_relative_error'] = max(rel) if rel else None
    drel = [abs(r['train_losses']['L_D_image'] + r['train_losses']['L_D_patch'] - r['train_losses']['L_D_total']) /
            abs(r['train_losses']['L_D_total']) for r in records]
    c['d_total_sum_max_relative_error'] = max(drel) if drel else None
    c['d_losses_finite_positive'] = sum(all(r['train_losses'][k] > 0 for k in D_TERMS + ('L_D_total',))
                                        for r in records)
    c['loss_stats'] = {k: {'first': records[0]['train_losses'][k], 'last': records[-1]['train_losses'][k],
                           'min': min(r['train_losses'][k] for r in records),
                           'max': max(r['train_losses'][k] for r in records),
                           'mean': sum(r['train_losses'][k] for r in records) / n,
                           'mean_last_100': sum(r['train_losses'][k] for r in records[-100:]) / min(100, n)}
                       for k in LOSS_KEYS} if records else None
    c['gradients_exact'] = sum(all(r['gradients'][g]['tensors'] == r['gradients'][g]['non_none'] ==
                                   r['gradients'][g]['finite'] == GROUP_TENSORS[g] and
                                   r['gradients'][g]['inactive_group_non_none'] == 0 for g in GROUP_TENSORS)
                               for r in records)
    c['gradients_all_nonzero'] = sum(all(r['gradients'][g]['nonzero'] == GROUP_TENSORS[g] for g in GROUP_TENSORS)
                                     for r in records)
    sec = sorted(r['step_seconds'] for r in records)
    c['step_seconds'] = {'median': sec[n // 2], 'max': sec[-1], 'min': sec[0]} if records else None
    c['gpu_memory_max_bytes'] = max(r['gpu_memory_bytes'] for r in records) if records else None
    c['missing_field_reasons'] = sorted({json.dumps(r['missing_field_reasons'], sort_keys=True) for r in records})
    c['null_val_fields'] = sum(r['val_losses'] is None and r['val_metrics'] is None for r in records)
    return c, events, consumed


def checkpoint_report(torch, path, ref=None):
    obj = torch.load(path, map_location='cpu', weights_only=True)
    out = {'format': obj.get('format'), 'benchmark_iteration': obj.get('benchmark_iteration'),
           'top_keys': sorted(obj), 'models': {}, 'provenance': obj.get('provenance')}
    tensors = floating = nonfinite = differs = 0
    for name in sorted(obj['models']):
        state = obj['models'][name]
        m = {'tensors': len(state), 'elements': 0, 'floating': 0}
        for key, v in state.items():
            tensors += 1
            m['elements'] += v.numel()
            if v.is_floating_point():
                floating += 1
                m['floating'] += 1
                nonfinite += int(not bool(torch.isfinite(v).all()))
            if ref is not None:
                differs += int(not torch.equal(v, ref['models'][name][key]))
        out['models'][name] = m
    out.update(tensors=tensors, floating_tensors=floating, nonfinite_floating_tensors=nonfinite,
               tensors_differing_from_iteration_1=differs if ref is not None else None)
    return out, obj


def collect_atime(seed, split, consumed):
    p = paths(seed)

    def load(f):
        out = {}
        with open(f) as h:
            for line in h:
                a, m, path = line.rstrip('\n').split('\t')
                out[path] = (float(a), float(m))
        return out
    pre, post = load(p['atime_pre']), load(p['atime_post'])
    t_pre = epoch_seconds(Path(p['atime_pre'][:-4] + '.time').read_text().strip())
    t_post = epoch_seconds(Path(p['atime_post'][:-4] + '.time').read_text().strip())
    stats, manifests = {}, []
    for path, (a, m) in sorted(pre.items()):
        if '/manifests/' in path:
            a2, m2 = post.get(path, (None, None))
            manifests.append({'file': os.path.basename(path), 'atime_pre_utc': utc(a),
                              'atime_post_utc': utc(a2) if a2 is not None else None, 'atime_changed': a2 != a,
                              'mtime_unchanged': m2 == m})
            continue
        sid = os.path.basename(path)[:-4]
        meta = split.get(sid)
        ds, sp = (meta['dataset'], meta['split']) if meta else (os.path.basename(os.path.dirname(path)), 'UNMAPPED')
        s = stats.setdefault(f'{ds}|{sp}', {'files': 0, 'sensitive_whole_window': 0, 'atime_changed': 0,
                                           'changed_consumed': 0, 'changed_not_consumed': 0, 'consumed': 0,
                                           'consumed_sensitive': 0, 'consumed_changed': 0, 'mtime_changed': 0,
                                           'missing_post': 0})
        sens = (a <= m) or (t_pre - a >= 86400)    # relatime: the first read in the window updates atime
        s['files'] += 1
        s['sensitive_whole_window'] += sens
        s['consumed'] += sid in consumed
        s['consumed_sensitive'] += sid in consumed and sens
        if path not in post:
            s['missing_post'] += 1
            continue
        a2, m2 = post[path]
        s['mtime_changed'] += m2 != m
        if a2 != a:
            s['atime_changed'] += 1
            s['changed_consumed' if sid in consumed else 'changed_not_consumed'] += 1
            s['consumed_changed'] += sid in consumed
    return {'pre_snapshot_utc': utc(t_pre), 'post_snapshot_utc': utc(t_post), 'files_pre': len(pre),
            'files_post': len(post), 'same_paths': sorted(pre) == sorted(post), 'by_class': stats,
            'manifests': manifests, 'mount': 'ext4 rw,relatime (/)', 'pre_sha256': sha256_file(p['atime_pre']),
            'post_sha256': sha256_file(p['atime_post'])}


def host_state(seed):
    p = paths(seed)

    def sh(*a):
        r = subprocess.run(list(a), capture_output=True, text=True)
        return r.stdout.strip()
    pane = sh('tmux', 'list-panes', '-t', p['tmux'], '-F', '#{pane_dead} #{pane_dead_status}').split()
    capture = sh('tmux', 'capture-pane', '-p', '-J', '-t', p['tmux'], '-S', '-5000').splitlines()
    procs = [x for x in sh('pgrep', '-af', 'tools/run_e05.py').splitlines()
             if f'--seed {seed} ' in x + ' ' and 'pgrep' not in x]
    gpu = sh('nvidia-smi', '--query-gpu=name,driver_version,uuid', '--format=csv,noheader')
    return {'tmux_pane_dead': pane[0] == '1' if pane else None,
            'tmux_pane_dead_status': int(pane[1]) if len(pane) > 1 and pane[0] == '1' else None,
            'tmux_capture_nonempty_lines': [x for x in capture if x.strip()][-10:],
            'tmux_capture_pattern_hits': {k: sum(k in x for x in capture) for k in LOG_PATTERNS},
            'runner_processes': procs, 'exit_file': Path(p['exit']).read_text().strip(),
            'exit_file_mtime_utc': utc(Path(p['exit']).stat().st_mtime),
            'command_file': Path(p['command']).read_text().strip(), 'gpu_query': gpu}


def collect(seed, repo):
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    import pyarrow.parquet as pq
    import torch
    from methods.pcgan import production_io as pio
    from methods.pcgan import production_runner as pr
    from methods.pcgan.runtime_qualification import environment
    root = Path(paths(seed)['run_root'])
    before_tree, before_small = tree_snapshot(root), small_hashes(root)
    before_digest, _, _ = sha_listing(root)

    def git(*a):
        return subprocess.run(['git', *a], capture_output=True, text=True).stdout
    g = {'head': git('rev-parse', 'HEAD').strip(), 'branch': git('branch', '--show-current').strip(),
         'status_porcelain': git('status', '--porcelain')}
    man = json.loads((root / 'run_manifest.json').read_text())
    summ = json.loads((root / 'run_summary.json').read_text())
    cindex = json.loads((root / 'checkpoint_index.json').read_text())
    pairs = pq.read_table(pio.PAIR_MANIFEST, columns=list(pio.ALLOWLIST)).to_pylist()
    split = {r['sample_id']: r for r in pq.read_table('manifests/split_v1.parquet', columns=[
        'sample_id', 'dataset', 'label_binary', 'split']).to_pylist()}
    acc = pq.read_table('manifests/m2_sample_accounting.parquet', columns=['sample_id', 'face_png_sha256']).to_pylist()
    faces = {r['sample_id']: r['face_png_sha256'] for r in acc}
    metrics, events, consumed = scan_metrics(root / 'metrics.jsonl', seed, pairs, split, faces)
    metrics['m2_accounting_unique_sample_ids'] = len(faces) == len(acc)
    consumed_rows = [split[s] for s in consumed if s in split]
    metrics['consumed_splits'] = sorted({r['split'] for r in consumed_rows})
    ckpt = []
    ref = None
    for c in sorted(cindex['checkpoints'], key=lambda x: x['global_step']):
        p = root / c['path']
        rep, obj = checkpoint_report(torch, p, ref) if p.is_file() else (None, None)
        if c['global_step'] == 1:
            ref = obj
        ckpt.append(dict(c, exists=p.is_file(), size_actual=p.stat().st_size if p.is_file() else None,
                         sha256_actual=sha256_file(p) if p.is_file() else None, load=rep))
        del obj
    ref = None
    logs = {}
    for name in ('stdout.log', 'stderr.log', 'generation_log.jsonl'):
        text = (root / name).read_bytes().decode('utf-8', errors='replace')
        logs[name] = {'bytes': len(text.encode()), 'lines': len(text.splitlines()),
                      'pattern_hits': {k: sum(k in x for x in text.splitlines()) for k in LOG_PATTERNS}}
    pr.configure_precision(torch)                      # the runner's FP32 flags, so precision compares like-for-like
    lock = json.loads(Path(pio.LOCK_PATH).read_text())
    live = environment()
    diff, driver = pio.environment_differences(lock['identity'], live, seed)
    start = events.get('e05_run_start', [{}])[0] or {}
    bc = (start.get('runtime_closure') or {}).get('build_closure') or {}
    ext = {k: {'recorded': v['extension']['sha256'], 'path': v['extension']['path'],
               'on_disk': sha256_file(v['extension']['path'])} for k, v in (bc.get('extensions') or {}).items()}
    atime = collect_atime(seed, split, consumed)
    digest, count, lines = sha_listing(root)
    after_tree, after_small = tree_snapshot(root), small_hashes(root)
    return {
        'seed': seed,
        'collector': {'torch_version': torch.__version__, 'python': sys.version.split()[0],
                      'executable': sys.executable, 'utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')},
        'git': g, 'host': host_state(seed),
        'files_sha256': {'config': sha256_file(pio.CONFIG_PATH), 'config_snapshot': sha256_file(pio.SNAPSHOT_PATH),
                         'pair_manifest': sha256_file(pio.PAIR_MANIFEST), 'lock': sha256_file(pio.LOCK_PATH),
                         'split_manifest': sha256_file('manifests/split_v1.parquet'),
                         'm2_sample_accounting': sha256_file('manifests/m2_sample_accounting.parquet'),
                         'python_executable': sha256_file(EXPECTED['python_executable']),
                         'production_files': pio.production_file_hashes()},
        'run_manifest': man, 'run_summary': summ,
        'checkpoint_index_meta': {k: v for k, v in cindex.items() if k != 'checkpoints'},
        'metrics': metrics, 'metrics_sha256': sha256_file(root / 'metrics.jsonl'),
        'event_counts': {k: len(v) for k, v in sorted(events.items())},
        'run_start': {k: start.get(k) for k in ('seeding', 'precision', 'identities', 'source_commit', 'gpu_driver',
                                                'environment_identity_equals_lock', 'environment_python_sha256',
                                                'loader', 'pair_manifest', 'fidelity_class')},
        'run_start_jit_build_invoked': bc.get('jit_build_invoked'),
        'extensions': ext,
        'checkpoint_events': [{k: e.get(k) for k in ('path', 'global_step', 'benchmark_iteration',
                                                     'actual_benchmark_iteration', 'checkpoint_type',
                                                     'selected_for_final', 'selection_reason', 'sha256',
                                                     'file_size_bytes', 'tensors_round_trip_equal',
                                                     'optimizer_applications', 'terminal_semantics_probe',
                                                     'resume_sidecar', 'labels')}
                              for e in events.get('e05_checkpoint_event', [])],
        'checkpoints': ckpt, 'checkpoints_dir_listing': sorted(x.name for x in (root / 'checkpoints').iterdir()),
        'partial_files': sorted(str(x.relative_to(root)) for x in root.rglob('*.partial')),
        'runner_state_files': sorted(str(x.relative_to(root)) for x in root.rglob('*runner_state*')),
        'resume_files': sorted(str(x.relative_to(root)) for x in root.rglob('*resume*')),
        'logs': logs,
        'environment_recheck': {'differences_other_than_driver': diff, 'driver': driver,
                                'live_gpu_driver': live['gpu_driver'], 'lock_gpu_driver': lock['identity']['gpu_driver'],
                                'live_torch': live['torch'], 'live_torchvision': live['torchvision'],
                                'live_cuda': live['cuda'], 'live_cudnn': live['cudnn'],
                                'live_python_sha256': live['python_sha256'], 'packages': len(live['packages'])},
        'atime_firewall': atime,
        'small_file_sha256': after_small, 'run_root_sha_listing_digest': digest, 'run_root_file_count': count,
        'run_root_sha_listing_digest_before': before_digest, 'run_root_sha_listing': lines,
        'run_root_tree_sha256': sha256_bytes(json.dumps(after_tree).encode()),
        'run_root_unchanged_whole_audit': (before_tree == after_tree and before_small == after_small
                                           and before_digest == digest),
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
    seed = c['seed']
    head = (f'# E05 PCGAN (controlled architecture resolution) seed{seed} run {RUN_IDS[seed]} '
            f'(authority {AUTHORITY[:7]}): iteration checkpoint_type selected_for_final selection_reason size_bytes '
            'sha256 path\n')
    body = ''.join(f"{r['global_step']:04d} {r['checkpoint_type']} {str(r['selected_for_final']).lower()} "
                   f"{r['selection_reason'].replace(' ', '_')} {r['file_size_bytes']} {r['sha256']} {r['path']}\n"
                   for r in sorted(c['checkpoints'], key=lambda r: r['global_step']))
    return head + body + '# runner-state / resume sidecars: none (E05 is fresh-only, NOT_QUALIFIED_FRESH_ONLY)\n'


def build(c, repo_root):
    seed = c['seed']
    G = Gates()
    man, summ, m, rs, host = c['run_manifest'], c['run_summary'], c['metrics'], c['run_start'], c['host']
    E, f = EXPECTED, c['files_sha256']
    rid = run_id(seed)

    # A. identity / authority
    G('A.run_id_recomputed', rid == RUN_IDS[seed] == man['run_id'] == summ['run_id'], rid)
    G('A.manifest_identity', man['method_id'] == E['method_id'] and man['experiment_seed'] == seed and
      man['git_commit'] == AUTHORITY and man['git_dirty'] is False and man['config_sha256'] == CONFIG_SHA and
      man['fidelity_class'] == E['fidelity'] and man['resume'] == E['resume'])
    G('A.command_line_exact', man['command_line'] == command(seed) and '--resume' not in man['command_line'] and
      host['command_file'].endswith(command(seed, '-u -B ')) and f'PYTHONHASHSEED={seed} ' in host['command_file'])
    G('A.gpu_checkout_authority', c['git']['head'] == AUTHORITY and c['git']['branch'] == 'm6-baselines' and
      c['git']['status_porcelain'] == '')
    G('A.host', man['environment']['host'] == E['host'])
    G('A.frozen_files', f['config'] == f['config_snapshot'] == CONFIG_SHA and
      f['pair_manifest'] == E['pair_manifest_sha256'] and f['lock'] == E['lock_sha256'] and
      f['python_executable'] == E['python_sha256'])
    G('A.run_start_identities', rs['identities']['config_sha256'] == CONFIG_SHA and
      rs['identities']['pair_manifest_sha256'] == E['pair_manifest_sha256'] and
      rs['identities']['environment_lock_sha256'] == E['lock_sha256'] and
      rs['identities']['production_files_sha256'] == f['production_files'] and
      rs['source_commit'] == E['source_commit'] and rs['fidelity_class'] == E['fidelity'])
    G('A.single_fresh_start', c['event_counts'].get('e05_run_start') == 1 and
      f'seed_{seed}' in c['parent_listing'])

    # B. completion
    G('B.process_exited_zero', host['exit_file'] == '0' and host['tmux_pane_dead'] is True and
      host['tmux_pane_dead_status'] == 0 and host['runner_processes'] == [])
    G('B.completion_status', man['completion_status'] == summ['completion_status'] == 'completed' and
      summ['failure_reason'] is None)
    G('B.summary_iterations', summ['benchmark_iterations'] == ITERATIONS and
      summ['optimizer_applications'] == {'D': ITERATIONS, 'G': ITERATIONS, 'hook_D': ITERATIONS,
                                         'hook_G': ITERATIONS, 'total': 2 * ITERATIONS})
    G('B.records_written', summ['records_written'] == m['lines'] == ITERATIONS + sum(c['event_counts'].values()),
      summ['records_written'])
    G('B.tmux_capture_clean', host['tmux_capture_pattern_hits']['Traceback'] == 0 and
      any('"status": "completed"' in x for x in host['tmux_capture_nonempty_lines']))

    # C. metrics continuity
    G('C.records_4000', m['records'] == ITERATIONS and m['bad_lines'] == 0 and m['trailing_newline'])
    G('C.global_step_1_to_4000', m['global_step_is_1_to_n'] and m['distinct_global_steps'] == ITERATIONS and
      m['first_step'] == 1 and m['last_step'] == ITERATIONS and
      m['benchmark_iteration_equals_global_step'] == ITERATIONS)
    G('C.record_constants', m['epoch_values'] == [1] and m['seed_values'] == [seed] and
      m['runner_modes'] == ['SCIENTIFIC'] and m['granularity'] == ['BENCHMARK_ITERATION_D_THEN_G'] and
      m['utc_nondecreasing'])
    G('C.no_forbidden_record_keys', m['forbidden_key_records'] == 0 and m['null_val_fields'] == ITERATIONS)
    G('C.event_set', c['event_counts'] == {'e05_checkpoint_event': 2, 'e05_run_start': 1}, c['event_counts'])

    # D. optimizer accounting
    G('D.applications_every_iteration', m['applications_exact'] == ITERATIONS)
    G('D.final_applications', m['final_applications'] == {'D': ITERATIONS, 'G': ITERATIONS, 'hook_D': ITERATIONS,
                                                          'hook_G': ITERATIONS, 'total': 2 * ITERATIONS})
    G('D.learning_rates_frozen', m['learning_rates_exact'] == ITERATIONS)

    # E. data order and scope
    o, p = m['order'], m['pairs']
    G('E.predicted_order_4000', o['predicted_positions'] == ITERATIONS and o['match'] == ITERATIONS and
      o['sampler_order_sha256'] == o['predicted_order_sha256'])
    G('E.distinct_pairs_4000', o['distinct_sampler_indices'] == o['distinct_pair_ids'] == ITERATIONS)
    G('E.pair_manifest_rows', p['manifest_row_match'] == p['pair_sha256_match'] == ITERATIONS and
      rs['pair_manifest']['sha256'] == E['pair_manifest_sha256'] and rs['pair_manifest']['splits'] == ['TRAIN'] and
      rs['pair_manifest']['rows'] == PAIR_ROWS and rs['pair_manifest']['subject_columns_read'] == [])
    G('E.spoof_source_live_target', p['source_spoof_target_live'] == ITERATIONS)
    G('E.train_only', p['both_faces_train_same_dataset'] == ITERATIONS and m['consumed_splits'] == ['TRAIN'])
    G('E.face_content_is_train_face', p['face_png_sha256_match_m2_accounting'] == ITERATIONS and
      m['m2_accounting_unique_sample_ids'])
    G('E.input_contract', m['input_contract'] == ITERATIONS)
    G('E.loader', rs['loader']['batch_size'] == 1 and rs['loader']['num_workers'] == 0 and
      rs['loader']['shuffle'] is True and rs['loader']['drop_last'] is False and
      rs['loader']['generator_seed'] == seed and rs['loader']['rows'] == PAIR_ROWS)

    # F. numerics
    G('F.loss_keys', m['loss_keys_exact'] == ITERATIONS)
    G('F.losses_finite', m['losses_finite'] == ITERATIONS and m['nonfinite_values'] == 0)
    G('F.g_terms_active', m['g_terms_active'] == ITERATIONS)
    G('F.g_total_unit_sum', m['g_total_unit_sum_max_relative_error'] <= TOL_G_TOTAL,
      m['g_total_unit_sum_max_relative_error'])
    G('F.d_losses', m['d_losses_finite_positive'] == ITERATIONS and m['d_total_sum_max_relative_error'] <= TOL_G_TOTAL,
      m['d_total_sum_max_relative_error'])
    G('F.gradients_80_59', m['gradients_exact'] == ITERATIONS)
    logs_ok = all(v['pattern_hits'][k] == 0 for v in c['logs'].values() for k in LOG_PATTERNS)
    G('F.logs_clean', logs_ok and c['logs']['stderr.log']['bytes'] == 0, {k: v['bytes'] for k, v in c['logs'].items()})

    # G. checkpoints
    ck = {x['global_step']: x for x in c['checkpoints']}
    ev = {x['global_step']: x for x in c['checkpoint_events']}
    G('G.exactly_two', sorted(ck) == sorted(ev) == sorted(CHECKPOINTS) and
      c['checkpoints_dir_listing'] == sorted(Path(v[0]).name for v in CHECKPOINTS.values()))
    for it, (path, kind, sel) in CHECKPOINTS.items():
        x, e = ck.get(it, {}), ev.get(it, {})
        ld = x.get('load') or {}
        prov = ld.get('provenance') or {}
        apps = {'D': it, 'G': it, 'hook_D': it, 'hook_G': it, 'total': 2 * it}
        reason = E['checkpoint_rule'] if sel else 'Official cadence; not final selection'
        G(f'G.iteration_{it}_index', x.get('path') == path and x.get('checkpoint_type') == kind and
          x.get('selected_for_final') is sel and x.get('selection_reason') == reason and x.get('epoch') == 1)
        G(f'G.iteration_{it}_file', x.get('exists') and x.get('size_actual') == x.get('file_size_bytes') and
          x.get('sha256_actual') == x.get('sha256') == e.get('sha256'))
        G(f'G.iteration_{it}_event', e.get('path') == path and e.get('checkpoint_type') == kind and
          e.get('selected_for_final') is sel and e.get('actual_benchmark_iteration') == it and
          e.get('optimizer_applications') == apps and e.get('terminal_semantics_probe') is False and
          e.get('resume_sidecar') is None and e.get('tensors_round_trip_equal') == ld.get('tensors') == 167)
        G(f'G.iteration_{it}_weights_only_load', ld.get('format') == 'E05_BASELINE_STATE_V1' and
          ld.get('benchmark_iteration') == it and ld.get('top_keys') == ['benchmark_iteration', 'format', 'models',
                                                                       'provenance'] and
          sorted(ld.get('models', {})) == sorted(MODELS) and ld.get('nonfinite_floating_tensors') == 0)
        G(f'G.iteration_{it}_provenance', prov.get('run_id') == RUN_IDS[seed] and prov.get('git_commit') == AUTHORITY
          and prov.get('experiment_seed') == seed and prov.get('config_sha256') == CONFIG_SHA and
          prov.get('benchmark_iteration') == prov.get('actual_benchmark_iteration') == it and
          prov.get('optimizer_applications') == apps and prov.get('checkpoint_type') == kind and
          prov.get('selected_for_final') is sel and prov.get('runner_mode') == 'SCIENTIFIC' and
          prov.get('terminal_semantics_probe') is False and prov.get('source_commit') == E['source_commit'])
    G('G.no_extra_application_at_write', m['applications_exact'] == ITERATIONS)   # iteration 2 reads total 4
    G('G.state_changed_by_training', (ck.get(4000, {}).get('load') or {}).get('tensors_differing_from_iteration_1', 0)
      > 0)
    sel = [x for x in c['checkpoints'] if x['selected_for_final']]
    G('G.one_selected', len(sel) == 1 and sel[0]['global_step'] == ITERATIONS and
      summ['final_or_selected_checkpoint_sha256'] == sel[0]['sha256'] and
      summ['final_or_selected_checkpoint_path'] == f"{paths(seed)['run_root']}/{CHECKPOINTS[4000][0]}" and
      summ['checkpoint_selection_rule'] == E['checkpoint_rule'] and
      c['checkpoint_index_meta']['selected_final'] == CHECKPOINTS[4000][0])
    G('G.no_partial_no_sidecar', c['partial_files'] == [] and c['runner_state_files'] == [] and
      c['resume_files'] == [] and c['checkpoint_index_meta']['resume_sidecars'] is None)

    # H. firewall
    a = c['atime_firewall']
    by = a['by_class']
    vt = {k: v for k, v in by.items() if k.split('|')[1] in ('VAL', 'TEST')}
    G('H.snapshots_consistent', a['files_pre'] == a['files_post'] and a['same_paths'] and
      a['pre_snapshot_utc'] <= summ['start_utc'] and a['post_snapshot_utc'] >= summ['end_utc'] and
      all(v['missing_post'] == v['mtime_changed'] == 0 for v in by.values()) and 'UNMAPPED' not in
      {k.split('|')[1] for k in by})
    G('H.val_zero', all(v['atime_changed'] == 0 and v['sensitive_whole_window'] == v['files'] and v['consumed'] == 0
                        for k, v in vt.items() if k.endswith('|VAL')) and any(k.endswith('|VAL') for k in vt))
    G('H.test_zero', all(v['atime_changed'] == 0 and v['sensitive_whole_window'] == v['files'] and v['consumed'] == 0
                         for k, v in vt.items() if k.endswith('|TEST')) and any(k.endswith('|TEST') for k in vt))
    G('H.train_changes_only_consumed', all(v['changed_not_consumed'] == 0 for k, v in by.items()
                                           if k.endswith('|TRAIN')))
    G('H.manifests_unmodified', all(x['mtime_unchanged'] for x in a['manifests']))
    G('H.test_split_flag', man['test_split_accessed'] is False and summ['test_split_accessed'] is False)

    # I. environment / driver ruling
    d, er = rs['gpu_driver'], c['environment_recheck']
    G('I.driver_ruling_run_start', d == {'driver_version_changed': True, 'gpu_model': E['gpu_model'],
                                         'gpu_uuid': E['gpu_uuid'],
                                         'live_driver_version': DRIVER_RULING['observed_driver_version'],
                                         'lock_driver_version': DRIVER_RULING['lock_driver_version']})
    G('I.environment_equals_lock_except_driver', rs['environment_identity_equals_lock'] is True and
      er['differences_other_than_driver'] == [] and er['driver'] == d)
    G('I.live_gpu', host['gpu_query'] == er['live_gpu_driver'] ==
      f"{E['gpu_model']}, {DRIVER_RULING['observed_driver_version']}, {E['gpu_uuid']}")
    G('I.python_binary', rs['environment_python_sha256'] == er['live_python_sha256'] == E['python_sha256'] and
      man['environment']['python_executable'] == c['collector']['executable'] == E['python_executable'])
    G('I.extensions_no_rebuild', c['run_start_jit_build_invoked'] is False and
      {k: v['recorded'] for k, v in c['extensions'].items()} == E['extensions'] and
      all(v['on_disk'] == v['recorded'] for v in c['extensions'].values()))
    G('I.precision_seeding', rs['precision'] == {'amp': False, 'autocast_cpu': False, 'autocast_cuda': False,
                                                 'cudnn_benchmark': False, 'cudnn_deterministic': True,
                                                 'cudnn_tf32': False, 'default_dtype': 'torch.float32',
                                                 'matmul_precision': 'highest', 'matmul_tf32': False} and
      rs['seeding']['seed'] == seed and rs['seeding']['role'] == 'EXPERIMENT_SEED')

    # J. run-root integrity
    G('J.run_root_unchanged', c['run_root_unchanged_whole_audit'] and
      c['run_root_sha_listing_digest'] == c['run_root_sha_listing_digest_before'])
    G('J.run_root_files', c['run_root_file_count'] == len(c['run_root_sha_listing']) == 10 and
      all(n in c['small_file_sha256'] for n in SMALL_FILES))

    failed = G.failed
    status = f'E05_SEED{seed}_COMPLETED_CLEANLY' if not failed else f'E05_SEED{seed}_NEEDS_REVIEW'
    text = manifest_text(c)
    sel0 = sel[0] if sel else {}
    vt_sum = lambda s, key: sum(v[key] for k, v in by.items() if k.endswith('|' + s))  # noqa: E731
    doc = {
        'milestone': f'E05-SEED{seed}-COMPLETION', 'classification': CLASSIFICATION, 'status': status,
        'failed_gates': failed, 'gates': G.results,
        'method': {'id': E['method_id'], 'name': E['method_name'], 'fidelity': E['fidelity'], 'track': E['track']},
        'seed': seed, 'run_id': rid, 'run_id_rule': f'sha256("E05|{seed}|{CONFIG_SHA}|{AUTHORITY}")[:16]',
        'scientific_authority': AUTHORITY, 'config_sha256': CONFIG_SHA, 'resume': E['resume'],
        'run_root': paths(seed)['run_root'], 'start_utc': summ['start_utc'], 'end_utc': summ['end_utc'],
        'training_duration_seconds': summ['training_duration_seconds'],
        'iterations': m['records'], 'optimizer_applications': m['final_applications'],
        'distinct_train_pairs': m['order']['distinct_pair_ids'],
        'pair_order': {'predictor': 'torch.Generator().manual_seed(seed); one int64 draw; randperm(8838)',
                       'positions_matched': m['order']['match'], 'positions': m['order']['predicted_positions'],
                       'pair_id_order_sha256': m['order']['pair_id_order_sha256'],
                       'records_by_dataset': m['pairs']['records_by_dataset']},
        'losses': {'g_terms': list(G_TERMS), 'g_total_unit_sum_max_relative_error':
                   m['g_total_unit_sum_max_relative_error'], 'stats': m['loss_stats']},
        'checkpoints': {'files': [{k: x[k] for k in ('path', 'global_step', 'checkpoint_type', 'selected_for_final',
                                                     'selection_reason', 'file_size_bytes', 'sha256')}
                                  for x in sorted(c['checkpoints'], key=lambda r: r['global_step'])],
                        'final_selected': {'path': sel0.get('path'), 'sha256': sel0.get('sha256'),
                                           'rule': E['checkpoint_rule']},
                        'manifest_file': evidence(seed)['manifest'], 'manifest_sha256': sha256_bytes(text.encode()),
                        'resume_sidecars': 0},
        'firewall': {'val_access': 0 if not failed else None, 'test_access': 0 if not failed else None,
                     'val_files': vt_sum('VAL', 'files'), 'test_files': vt_sum('TEST', 'files'),
                     'val_changed': vt_sum('VAL', 'atime_changed'), 'test_changed': vt_sum('TEST', 'atime_changed'),
                     'scope': 'TRAIN only (pairs_train_v1 order + M2 face PNG digests of every consumed face)',
                     'atime': a, 'relatime_note': ('TRAIN faces read by another run less than 24 h before the pre '
                                                   'snapshot are not atime-sensitive; TRAIN use is proven by the '
                                                   'order/manifest/content evidence, not by atime')},
        'driver_ruling': DRIVER_RULING,
        'environment': {'python_executable': E['python_executable'], 'python_sha256': E['python_sha256'],
                        'extensions_sha256': E['extensions'], 'torch': c['environment_recheck']['live_torch'],
                        'torchvision': c['environment_recheck']['live_torchvision'],
                        'cuda': c['environment_recheck']['live_cuda'], 'cudnn': c['environment_recheck']['live_cudnn'],
                        'jit_build_invoked': c['run_start_jit_build_invoked']},
        'run_root_integrity': {'file_count': c['run_root_file_count'],
                               'sha_listing_digest': c['run_root_sha_listing_digest'],
                               'sha_listing_convention': ('LC_ALL=C sorted "<path> <size> <sha256>" lines, sha256 of '
                                                          'the newline-terminated listing'),
                               'metrics_jsonl_sha256': c['metrics_sha256'], 'small_file_sha256': c['small_file_sha256'],
                               'checkpoint_sha256': {x['path']: x['sha256'] for x in c['checkpoints']},
                               'unchanged_whole_audit': c['run_root_unchanged_whole_audit']},
        'resolution_wording': ('controlled implementation resolutions (owner approved); not native, faithful or '
                               'official PCGAN choices'),
        'confirmations': {'resume_used': False, 'val_checkpoint_selection': False, 'test_selection': False,
                          'best_iteration_logic': False, 'best_seed_logic': False, 'bank_generation': False,
                          'downstream_evaluation': False, 'run_root_mutated': False, 'downstream_metric': None},
        'collected': c,
    }
    return doc, text


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('collect')
    a.add_argument('--seed', type=int, required=True, choices=sorted(RUN_IDS))
    a.add_argument('--repo', required=True)
    b = sub.add_parser('build')
    b.add_argument('--collected', required=True)
    b.add_argument('--repo-root', default=str(Path(globals().get('__file__', 'tools/x')).resolve().parents[1]))
    args = ap.parse_args(argv)
    if args.cmd == 'collect':
        json.dump(collect(args.seed, args.repo), sys.stdout, sort_keys=True)
        return 0
    c = json.loads(Path(args.collected).read_text())
    doc, text = build(c, args.repo_root)
    root, ev = Path(args.repo_root), evidence(c['seed'])
    (root / ev['manifest']).write_text(text)
    (root / ev['completion']).write_text(json.dumps(doc, indent=2, sort_keys=True) + '\n')
    print(doc['status'], doc['failed_gates'])
    return 0 if not doc['failed_gates'] else 1


if __name__ == '__main__':
    sys.exit(main())
