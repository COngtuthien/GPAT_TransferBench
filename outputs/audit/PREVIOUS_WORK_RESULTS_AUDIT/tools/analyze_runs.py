"""Read-only analysis of the 12 completed previous-work training runs (E06c, E06b, E05, E03 x seeds 42/1337/2026).

Inputs: a local copy of each run root's small files (metrics.jsonl, run_manifest.json, run_summary.json,
checkpoint_index.json, resume_state_index.json, resolved_config.yaml, stdout/stderr/generation logs) and the
host listing file (path/size/sha256 and path/size/mtime of every run-root file, computed on the GPU host).
Writes only into --out. Never opens checkpoints, images, VAL/TEST data or predictions.

usage: python3 -I analyze_runs.py --runs <dir> --listings <file> --out <dir>
"""
import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

SEEDS = (42, 1337, 2026)
METHODS = ('E06c', 'E06b', 'E05', 'E03')
BUDGET = {'E06c': dict(steps=7400, epochs=200, steps_per_epoch=37),
          'E06b': dict(steps=3200, epochs=200, steps_per_epoch=16),
          'E05': dict(steps=4000),
          'E03': dict(train_steps=100000, val_steps=25000, epochs=50, g_apps=100000, d_apps=50000, global_step=150000)}
CKPT_EVENTS = {'E06c': 21, 'E06b': 21, 'E05': 2, 'E03': 50}
CKPT_FILES = {'E06c': 63, 'E06b': 63, 'E05': 2, 'E03': 150}
# closure-recorded values (outputs/audit/<M>_THREE_SEED_FINAL_CHECKPOINTS.csv); verified, not trusted blindly
CLOSURE_CSV = {'E06c': 'E06C', 'E06b': 'E06B', 'E05': 'E05', 'E03': 'E03'}
DSDG_AUX = ('loss_kl', 'loss_mmd', 'loss_ip', 'loss_pair', 'loss_cls', 'loss_ort')


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def finite_walk(obj, path=''):
    """Yield (path, value) for every non-finite float in a JSON value."""
    if isinstance(obj, float):
        if not math.isfinite(obj):
            yield path, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from finite_walk(v, path + '.' + k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from finite_walk(v, '%s[%d]' % (path, i))


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def load_listing(path):
    sha, mtime = defaultdict(list), defaultdict(list)
    for line in open(path):
        line = line.rstrip('\n')
        if line.startswith('SHA '):
            _, run, rest = line.split(' ', 2)
            sha[run].append(rest)
        elif line.startswith('MTIME '):
            _, run, rest = line.split(' ', 2)
            mtime[run].append(rest)
    return sha, mtime


def listing_digest(lines):
    return hashlib.sha256(''.join(l + '\n' for l in lines).encode()).hexdigest()


def closure_rows(repo):
    rows = {}
    for m, tag in CLOSURE_CSV.items():
        with open(repo / 'outputs/audit' / ('%s_THREE_SEED_FINAL_CHECKPOINTS.csv' % tag)) as f:
            for r in csv.DictReader(f):
                rows[(m, int(r['seed']))] = r
    return rows


def analyze(method, seed, root, sha_lines, mtime_lines, closure):
    d = root / method / ('seed_%d' % seed)
    res = {'method_id': method, 'seed': seed, 'issues': [], 'checks': {}}
    chk = res['checks']
    man = json.loads((d / 'run_manifest.json').read_text())
    summ = json.loads((d / 'run_summary.json').read_text())
    cidx = json.loads((d / 'checkpoint_index.json').read_text())
    res['files_sha256'] = {p.name: sha256_file(p) for p in sorted(d.iterdir()) if p.is_file()}

    # ---- host listing: payload presence and size/sha of every indexed checkpoint file
    host = {}
    for l in sha_lines:
        path, size, digest = l.rsplit(' ', 2)
        host[path] = (int(size), digest)
    res['host_file_count'] = len(sha_lines)
    res['host_sha_listing_digest'] = listing_digest(sha_lines)
    res['host_mtime_listing_digest'] = listing_digest(mtime_lines)
    entries = cidx['checkpoints']
    missing = [e['path'] for e in entries if e['path'] not in host]
    mismatch = [e['path'] for e in entries if e['path'] in host and host[e['path']] != (e['file_size_bytes'], e['sha256'])]
    ckpt_dir_files = sorted(p for p in host if p.startswith('checkpoints/'))
    chk['checkpoint_index_entries'] = len(entries)
    chk['checkpoint_payloads_missing'] = missing
    chk['checkpoint_payloads_size_or_sha_mismatch'] = mismatch
    chk['checkpoints_dir_files_on_host'] = len(ckpt_dir_files)
    for name in ('metrics.jsonl', 'run_summary.json', 'checkpoint_index.json', 'run_manifest.json'):
        chk['host_sha_equals_copy:' + name] = host.get(name, (None, None))[1] == res['files_sha256'].get(name)
    sel = [e for e in entries if e['selected_for_final']]
    res['final_checkpoint'] = {'path': summ['final_or_selected_checkpoint_path'].split('/' + d.name + '/')[-1],
                               'sha256': summ['final_or_selected_checkpoint_sha256'],
                               'rule': summ['checkpoint_selection_rule'],
                               'selected_index_entries': [(e['path'], e['sha256'], e['file_size_bytes']) for e in sel]}
    fin_path = res['final_checkpoint']['path']
    if method == 'E03':
        fin_path = 'checkpoints/ckpt-50.data-00000-of-00001'
    chk['final_checkpoint_on_host_sha_equal'] = host.get(fin_path, (None, None))[1] == summ['final_or_selected_checkpoint_sha256']
    if method == 'E06c':
        rsi = json.loads((d / 'resume_state_index.json').read_text())
        side = [v for v in rsi.values() if isinstance(v, list)][0]
        chk['resume_sidecars_indexed'] = len(side)
        chk['resume_sidecars_on_host_equal'] = all(host.get(e['path'], (None, None)) == (e['file_size_bytes'], e['sha256'])
                                                    for e in side)

    # ---- closure cross-check
    c = closure
    res['closure_commit'] = c['seed_closure_commit']
    chk['closure_metrics_sha_equal'] = c['metrics_sha256'] == res['files_sha256']['metrics.jsonl']
    if method == 'E06c':
        chk['closure_listing_digest_equal'] = c['run_root_listing_digest'] == res['host_mtime_listing_digest']
    elif method == 'E06b':
        chk['closure_listing_digest_equal'] = (c['run_root_sha_listing_digest'] == res['host_sha_listing_digest'] and
                                               c['run_root_mtime_listing_digest'] == res['host_mtime_listing_digest'])
    else:
        chk['closure_listing_digest_equal'] = c['run_root_listing_digest'] == res['host_sha_listing_digest']
    final_closure = c.get('final_checkpoint_sha256') or c.get('final_netG_sha256') or c.get('final_data_sha256')
    chk['closure_final_sha_equal'] = final_closure == summ['final_or_selected_checkpoint_sha256']

    # ---- provenance
    res['provenance'] = {k: man.get(k) for k in ('run_id', 'run_uuid', 'git_commit', 'git_dirty', 'config_path',
                                                  'config_sha256', 'host', 'user', 'command_line', 'start_utc',
                                                  'end_utc', 'completion_status', 'logging_contract',
                                                  'python_executable', 'python_version', 'framework',
                                                  'framework_version', 'cuda_version', 'cudnn_version', 'gpu_model',
                                                  'dependency_fingerprint', 'environment_lock_path',
                                                  'environment_name', 'source_commits', 'test_split_accessed')}
    res['provenance']['run_id_recomputed'] = hashlib.sha256(
        ('%s|%d|%s|%s' % (method, seed, man['config_sha256'], man['git_commit'])).encode()).hexdigest()[:16]
    chk['run_id_formula'] = res['provenance']['run_id_recomputed'] == man['run_id'] == summ['run_id']
    res['summary'] = {k: summ.get(k) for k in ('completion_status', 'failure_reason', 'training_duration_seconds',
                                                'wall_clock_seconds', 'peak_vram_bytes', 'peak_vram_reason',
                                                'records_written', 'start_utc', 'end_utc', 'optimizer_applications',
                                                'test_split_accessed', 'seed_level_evaluation_metrics')}

    # ---- metrics.jsonl
    recs = []
    bad_lines = 0
    for line in open(d / 'metrics.jsonl'):
        try:
            recs.append(json.loads(line))
        except ValueError:
            bad_lines += 1
    chk['unparseable_lines'] = bad_lines
    chk['records_total'] = len(recs)
    chk['records_equal_summary'] = len(recs) == summ['records_written']
    events = Counter(r.get('event') for r in recs if r.get('record_type') == 'event')
    res['event_counts'] = dict(events)
    nonfinite = [(i, p) for i, r in enumerate(recs) for p, _ in finite_walk(r)]
    chk['nonfinite_values'] = len(nonfinite)
    utcs = [r['utc'] for r in recs]
    chk['utc_non_decreasing'] = all(a <= b for a, b in zip(utcs, utcs[1:]))
    seeds_seen = {r.get('experiment_seed') for r in recs if 'experiment_seed' in r}
    chk['experiment_seed_fields_equal_seed'] = seeds_seen <= {seed}
    modes = {r.get('runner_mode') for r in recs if 'runner_mode' in r}
    chk['runner_mode_values'] = sorted(m for m in modes if m)

    if method == 'E03':
        train = [r for r in recs if r.get('record_kind') == 'TRAIN_STEP']
        val = [r for r in recs if r.get('record_kind') == 'VAL_STEP']
    else:
        train = [r for r in recs if r.get('record_type') == 'epoch']
        val = []
    ck_ev = [r for r in recs if r.get('event', '').endswith('_checkpoint_event')]
    chk['checkpoint_events'] = len(ck_ev)
    chk['checkpoint_events_equal_budget'] = len(ck_ev) == CKPT_EVENTS[method]
    chk['checkpoint_files_equal_budget'] = len(entries) == CKPT_FILES[method]
    res['checkpoint_chronology'] = [{'utc': r['utc'], 'epoch': r['payload'].get('epoch'),
                                     'global_step': r['payload'].get('global_step',
                                                                     r['payload'].get('benchmark_iteration'))}
                                    for r in ck_ev]
    chk['wall_clock_non_decreasing'] = all(a['wall_clock_seconds'] <= b['wall_clock_seconds']
                                           for a, b in zip(train, train[1:]))

    # step indexing
    if method == 'E03':
        idx = [r['train_step_total'] for r in train]
        chk['train_steps'] = len(train)
        chk['val_steps'] = len(val)
        # train_step_total is 0-based (0..99999); train_step_in_epoch is 0-based (0..1999)
        chk['train_index_contiguous'] = idx == list(range(BUDGET['E03']['train_steps']))
        chk['train_step_in_epoch_mapping'] = all(r['train_step_in_epoch'] == r['train_step_total'] % 2000 and
                                                 r['epoch'] == r['train_step_total'] // 2000 + 1 for r in train)
        chk['val_steps_equal_budget'] = len(val) == BUDGET['E03']['val_steps']
        gs_ok = all(r['shared_global_step_before'] + (2 if 'd_op' in r['optimizer_ops_applied'] else 1) ==
                    r['shared_global_step_after'] == r['global_step'] for r in train)
        chain = all(a['shared_global_step_after'] == b['shared_global_step_before'] for a, b in zip(train, train[1:]))
        chk['shared_global_step_consistent'] = gs_ok and chain and train[-1]['global_step'] == 150000
        chk['final_applications'] = train[-1]['applications']
        chk['applications_equal_budget'] = train[-1]['applications'] == {'G': 100000, 'D': 50000}
        lr_err = max(abs(r['learning_rate'] - 6e-5 * 0.9 ** (r['shared_global_step_before'] // 20000)) /
                     (6e-5 * 0.9 ** (r['shared_global_step_before'] // 20000)) for r in train)
        chk['lr_schedule_max_rel_err'] = lr_err
        chk['lr_G_equals_D'] = all(r['learning_rates']['G'] == r['learning_rates']['D'] for r in train)
        chk['val_gs_unchanged'] = all(v['global_step'] % 3000 == 0 for v in val)
    else:
        idx = [r['global_step'] for r in train]
        n = BUDGET[method]['steps']
        chk['train_steps'] = len(train)
        chk['train_index_contiguous'] = idx == list(range(1, n + 1))
        chk['duplicate_steps'] = len(idx) - len(set(idx))
        if method in ('E06c', 'E06b'):
            spe = BUDGET[method]['steps_per_epoch']
            chk['epoch_mapping'] = all(r['epoch'] == (r['global_step'] - 1) // spe + 1 and
                                       r['iteration_in_epoch'] == (r['global_step'] - 1) % spe for r in train)
            chk['optimizer_counter_equals_step'] = all(r['optimizer_applications_total'] == r['global_step'] for r in train)
            chk['lr_values'] = sorted({r['learning_rate'] for r in train})
            chk['owned_gradients_finite_min'] = min(r['owned_gradients_finite'] for r in train)
            # DSDG pinned assembly (train_generator.py:174-178): epoch<2 -> 0.01 * aux terms
            err = 0.0
            for r in train:
                L = r['train_losses']
                w = 0.01 if r['epoch'] < 2 else 1.0
                tot = L['loss_rec'] + w * sum(L[k] for k in DSDG_AUX)
                err = max(err, abs(tot - L['total_loss']) / abs(L['total_loss']))
            chk['total_loss_assembly_max_rel_err'] = err
        else:  # E05
            chk['optimizer_counter_consistent'] = all(
                r['optimizer_applications'] == {'D': r['global_step'], 'G': r['global_step'], 'hook_D': r['global_step'],
                                                'hook_G': r['global_step'], 'total': 2 * r['global_step']} for r in train)
            chk['lr_values'] = sorted({v for r in train for v in r['learning_rates'].values()})
            g = max(abs(sum(r['train_losses'][k] for k in ('L_rec', 'L_recblur', 'L_advrec', 'L_advmix', 'L_pat')) -
                        r['train_losses']['L_G_total']) / abs(r['train_losses']['L_G_total']) for r in train)
            dd = max(abs(r['train_losses']['L_D_image'] + r['train_losses']['L_D_patch'] - r['train_losses']['L_D_total'])
                     / abs(r['train_losses']['L_D_total']) for r in train)
            chk['L_G_total_unit_sum_max_rel_err'] = g
            chk['L_D_total_sum_max_rel_err'] = dd
            chk['gradient_records_all_finite_nonzero'] = all(
                r['gradients'][k]['finite'] == r['gradients'][k]['tensors'] == r['gradients'][k]['nonzero']
                for r in train for k in ('D', 'G'))

    # timing / memory
    step_s = [r['step_seconds'] for r in train]
    mem = [r['gpu_memory_bytes'] for r in train + val if r.get('gpu_memory_bytes') is not None]
    res['timing'] = {'train_step_seconds_sum': sum(step_s), 'train_step_seconds_median': statistics.median(step_s),
                     'last_wall_clock_seconds': train[-1]['wall_clock_seconds'] if method != 'E03' else
                     max(r['wall_clock_seconds'] for r in train + val),
                     'summary_training_duration_seconds': summ['training_duration_seconds'],
                     'first_record_utc': utcs[0], 'last_record_utc': utcs[-1],
                     'val_step_seconds_sum': sum(r['step_seconds'] for r in val) if val else None}
    res['peak_gpu_memory_bytes_from_metrics'] = max(mem) if mem else None
    chk['peak_vram_summary_equals_metrics_max'] = (summ['peak_vram_bytes'] is None or
                                                   summ['peak_vram_bytes'] == res['peak_gpu_memory_bytes_from_metrics'])

    # losses: final record, final-window mean, first-window mean
    keys = sorted(train[0]['train_losses'])
    res['loss_keys'] = keys
    if method in ('E06c', 'E06b'):
        last_ep = [r for r in train if r['epoch'] == 200]
        first_ep = [r for r in train if r['epoch'] == 1]
        window = 'epoch 200 mean (%d steps)' % len(last_ep)
    elif method == 'E05':
        last_ep, first_ep = train[-100:], train[:100]
        window = 'iterations 3901-4000 mean'
    else:
        last_ep = [r for r in train if r['epoch'] == 50]
        first_ep = [r for r in train if r['epoch'] == 1]
        window = 'epoch 50 TRAIN mean (%d steps)' % len(last_ep)
    res['loss_final_window'] = window
    res['loss_final_record'] = train[-1]['train_losses']
    res['loss_final_window_mean'] = {k: mean([r['train_losses'][k] for r in last_ep]) for k in keys}
    res['loss_first_window_mean'] = {k: mean([r['train_losses'][k] for r in first_ep]) for k in keys}
    if val:
        res['val_loss_epoch50_mean'] = {k: mean([v['val_losses'][k] for v in val if v['epoch'] == 50])
                                        for k in sorted(val[0]['val_losses'])}
    res['final_learning_rate'] = train[-1].get('learning_rates', train[-1]['learning_rate'])

    # binned curves (postprocessing only: exact means of logged per-step values)
    curve = []
    if method == 'E05':
        bins = [(b, train[b * 100:(b + 1) * 100]) for b in range(40)]
        for b, rs in bins:
            curve.append(dict(bin_start_iteration=rs[0]['global_step'], bin_end_iteration=rs[-1]['global_step'],
                              n=len(rs), lr_G=rs[-1]['learning_rates']['G'], lr_D=rs[-1]['learning_rates']['D'],
                              step_seconds_mean=mean([r['step_seconds'] for r in rs]),
                              gpu_memory_bytes_max=max(r['gpu_memory_bytes'] for r in rs),
                              **{'train_' + k: mean([r['train_losses'][k] for r in rs]) for k in keys}))
    else:
        by = defaultdict(list)
        for r in train:
            by[r['epoch']].append(r)
        vby = defaultdict(list)
        for v in val:
            vby[v['epoch']].append(v)
        for e in sorted(by):
            rs = by[e]
            row = dict(epoch=e, first_global_step=rs[0]['global_step'], last_global_step=rs[-1]['global_step'], n=len(rs),
                       learning_rate_last=rs[-1]['learning_rate'],
                       step_seconds_mean=mean([r['step_seconds'] for r in rs]),
                       gpu_memory_bytes_max=max(r['gpu_memory_bytes'] for r in rs),
                       **{'train_' + k: mean([r['train_losses'][k] for r in rs]) for k in keys})
            if vby.get(e):
                row.update({'val_n': len(vby[e])}, **{'val_' + k: mean([v['val_losses'][k] for v in vby[e]])
                                                      for k in sorted(vby[e][0]['val_losses'])})
            curve.append(row)
    res['curve_rows'] = len(curve)
    res['issues'] = [k for k, v in chk.items() if v is False or (k == 'nonfinite_values' and v) or
                     (k == 'unparseable_lines' and v) or (isinstance(v, list) and k.startswith('checkpoint_payloads') and v)]
    return res, curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True)
    ap.add_argument('--listings', required=True)
    ap.add_argument('--repo', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    root, out, repo = Path(a.runs), Path(a.out), Path(a.repo)
    (out / 'curves').mkdir(parents=True, exist_ok=True)
    sha, mtime = load_listing(a.listings)
    closures = closure_rows(repo)
    allres = []
    for m in METHODS:
        for s in SEEDS:
            key = '%s/seed_%d' % (m, s)
            res, curve = analyze(m, s, root, sha[key], mtime[key], closures[(m, s)])
            allres.append(res)
            with open(out / 'curves' / ('%s_seed%d_curve.csv' % (m, s)), 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(curve[0].keys()) +
                                   [k for k in curve[-1] if k not in curve[0]], lineterminator='\n')
                w.writeheader()
                w.writerows(curve)
            print(key, 'issues:', res['issues'])
    (out / 'run_analysis.json').write_text(json.dumps(allres, indent=1, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
