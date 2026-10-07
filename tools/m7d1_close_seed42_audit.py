"""M7D1-CLOSE-SEED42: read-only completion audit of the GPAT-B0 / E08 / seed 42 scientific run (run_id 7b799fbd6d0426da).

Two phases:

  collect  (GPU host, inside the gpat-m7-gpu env)  -- reads the scientific run root and prints one JSON document to
           stdout. It opens every file read-only ('rb' / torch.load(weights_only=True, map_location='cpu')), writes
           nothing anywhere, and re-lists the run root (name/size/mtime_ns) plus the small JSON/log hashes before and
           after to prove the run root was not mutated. It never opens VAL/TEST data: firewall evidence comes only from
           the runner-produced run_summary/run_manifest/metrics.
  build    (repository) -- validates the collected document against the M7D1-CLOSE gates and writes
           outputs/audit/M7D1_E08_SEED42_COMPLETION.json and outputs/audit/M7D1_E08_SEED42_EMA_SHA256.txt.

No selection, bank, downstream or training code is imported or run.
"""
import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

CLASSIFICATION = 'SCIENTIFIC_TRAINING_COMPLETION_AUDIT'
STATUS_OK = 'M7D1_E08_SEED42_TRAINING_AUDITED_COMPLETE'
STATUS_BLOCKED = 'M7D1_E08_SEED42_AUDIT_BLOCKED'

EXPECTED = {
    'method': 'GPAT-B0', 'experiment': 'E08', 'seed': 42, 'run_id': '7b799fbd6d0426da',
    'code_commit': '058e976e5a538c4d18e1177f0eb27727cfb735cd',
    'config_sha256': '0a268d2c9d1914a9e687d08cdfb96d5bf793eaa22559476b8e68d3cf58035d6a',
    'pair_manifest_sha256': 'a5e4fdaef236f15730c7e3885b537e08e995faffbe654167fc940f44bbc75243',
    'env_lock_sha256': '24c983ebbb308acacd63f32837532e5f9166114962f524e993f243f2ff746114',
    'amp_amendment_path': 'configs/amendments/gpat_m7d1_amp_retry_resolution.yaml',
    'amp_amendment_sha256': '4dc8a838cbdce72c454313b4951684dfad8d90f82b8a6a8a090d355e8499e0a7',
    'amp_policy': 'ATOMIC_AMP_BACKOFF_RETRY',
    'unique_ids_sha256': '27a8dff74625704c99b7dcbbf38a5b70ee9eaede0fe7b0ea8d6206719d5ebb1d',
    'run_root': '/home/student20261/workdir/GPAT_TransferBench_runtime/runs/m7/E08/seed_42',
}
EPOCHS = 60
GROUPS_PER_EPOCH = 1105
TOTAL_UPDATES = EPOCHS * GROUPS_PER_EPOCH          # 66300
MICROBATCHES_PER_EPOCH = 2210
TRAIN_ROWS = 8838
CANDIDATE_EPOCHS = tuple(range(10, EPOCHS + 1))    # 51
TERMINAL_MARKERS = ('NON_FINITE_LOSS', 'FAILED_NUMERICAL_POST_STEP', 'FAIL_CLOSED_AMP_OVERFLOW_FINAL',
                    'FAIL_CLOSED_AMP_OVERFLOW', 'out of memory', 'OutOfMemoryError')
SMALL_FILES = ('run_manifest.json', 'run_summary.json', 'checkpoint_index.json', 'resolved_config.yaml',
               'stdout.log', 'stderr.log')


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


def candidate_name(epoch):
    return f'ema_epoch_{epoch:02d}.pt'


def ema_manifest_text(rows):
    """Deterministic sha256sum-style manifest, sorted by epoch; paths relative to the run root."""
    rows = sorted(rows, key=lambda r: r['epoch'])
    return ''.join(f"{r['sha256']}  checkpoints/ema_candidates/{r['filename']}\n" for r in rows)


def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


# ============================================================================ collect (GPU host, read-only)
def tree_snapshot(root):
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            p = Path(dirpath) / name
            st = p.stat()
            out.append([str(p.relative_to(root)), st.st_size, st.st_mtime_ns])
    return out


def small_hashes(root):
    return {n: sha256_file(root / n) for n in SMALL_FILES if (root / n).exists()}


def scan_metrics(path):
    groups_by_update, dup_updates, bad_status, nonfinite = {}, [], [], []
    retries, epochs, other_types, markers = [], [], {}, {}
    epoch_group_pairs = {}
    raw_lines = 0
    with open(path, 'rb') as f:
        for raw in f:
            raw_lines += 1
            text = raw.decode('utf-8')
            for m in TERMINAL_MARKERS:
                if m in text:
                    markers[m] = markers.get(m, 0) + 1
            d = json.loads(text)
            t = d.get('record_type')
            if t == 'optimizer_group':
                u = d['global_update']
                if d.get('group_status') != 'COMPLETE':
                    bad_status.append([u, d.get('group_status')])
                    continue
                if u in groups_by_update:
                    dup_updates.append(u)
                key = (d['epoch'], d['group'])
                epoch_group_pairs[key] = epoch_group_pairs.get(key, 0) + 1
                vals = dict(d.get('train_losses') or {})
                vals['D_grad_norm'] = d.get('D_grad_norm')
                vals['G_grad_norm'] = d.get('G_grad_norm')
                bad = sorted(k for k, v in vals.items() if not finite(v))
                if bad:
                    nonfinite.append([u, bad])
                groups_by_update[u] = {k: d.get(k) for k in (
                    'epoch', 'group', 'group_samples', 'microbatch_sizes', 'amp_attempts', 'amp_policy',
                    'D_scale', 'D_scale_before', 'G_scale', 'G_scale_before', 'ema_updated')}
                groups_by_update[u]['n_loss_terms'] = len(d.get('train_losses') or {})
            elif t == 'amp_retry':
                retries.append(d)
            elif t == 'epoch':
                epochs.append(d)
            else:
                other_types[str(t)] = other_types.get(str(t), 0) + 1

    updates = sorted(groups_by_update)
    g = groups_by_update
    d_scales = [v['D_scale'] for v in g.values()] + [v['D_scale_before'] for v in g.values()]
    g_scales = [v['G_scale'] for v in g.values()] + [v['G_scale_before'] for v in g.values()]
    for r in retries:
        for k in ('old_scale', 'new_scale', 'scales_after_restore'):
            for opt, s in (r.get(k) or {}).items():
                (d_scales if opt.startswith('D') else g_scales).append(s)
    per_epoch = {}
    for v in g.values():
        e = per_epoch.setdefault(v['epoch'], {'groups': 0, 'samples': 0, 'microbatches': 0, 'max_group': 0})
        e['groups'] += 1
        e['samples'] += v['group_samples']
        e['microbatches'] += len(v['microbatch_sizes'])
        e['max_group'] = max(e['max_group'], v['group'])
    tail_groups = {str(u): {k: g[u][k] for k in ('epoch', 'group', 'group_samples', 'microbatch_sizes')}
                   for u in updates if g[u]['group_samples'] != 8 or g[u]['microbatch_sizes'] != [4, 4]}
    last = g[updates[-1]] if updates else None
    retry_targets = {}
    for r in retries:
        retry_targets.setdefault(r['global_update_attempted'], []).append(r)
    retry_audit = []
    for u, rs in sorted(retry_targets.items()):
        target = g.get(u, {})
        retry_audit.append({
            'global_update_attempted': u,
            'epoch': rs[0]['epoch'], 'group': rs[0]['group'],
            'retry_numbers': [r['retry_number'] for r in rs],
            'offending_optimizers': sorted({o for r in rs for o in r['offending_optimizers']}),
            'offending_parameters': sorted({p for r in rs for ps in (r.get('offending_parameters') or {}).values()
                                            for p in ps}),
            'stages': sorted({r.get('stage') for r in rs}),
            'scale_transitions': [{'retry_number': r['retry_number'], 'old_scale': r['old_scale'],
                                   'new_scale': r['new_scale'], 'scales_after_restore': r['scales_after_restore']}
                                  for r in rs],
            'all_optimizer_steps_taken_zero': all(r['optimizer_steps_taken'] == 0 for r in rs),
            'all_optimizer_update_false': all(r['optimizer_update'] is False for r in rs),
            'all_pre_group_state_restored': all(r['pre_group_state_restored'] is True for r in rs),
            'all_amp_policy': sorted({r.get('amp_policy') for r in rs}),
            'complete_record_count': sum(1 for x in [u] if x in g) + dup_updates.count(u),
            'complete_epoch_group': [target.get('epoch'), target.get('group')],
            'complete_amp_attempts': target.get('amp_attempts'),
            'complete_D_scale_before': target.get('D_scale_before'), 'complete_D_scale': target.get('D_scale'),
            'complete_G_scale_before': target.get('G_scale_before'), 'complete_G_scale': target.get('G_scale'),
            'previous_update_completed': (u - 1) in g or u == 1,
            'update_not_advanced_on_retry': all(r['global_update_attempted'] == u for r in rs),
        })
    # the retry event must sit between the COMPLETE record of u-1 and that of u in the log: check ordering
    order_ok = True
    with open(path, 'rb') as f:
        last_complete = 0
        for raw in f:
            d = json.loads(raw)
            if d.get('record_type') == 'optimizer_group' and d.get('group_status') == 'COMPLETE':
                last_complete = d['global_update']
            elif d.get('record_type') == 'amp_retry' and d['global_update_attempted'] != last_complete + 1:
                order_ok = False
    return {
        'metrics_lines': raw_lines,
        'record_type_counts': {'optimizer_group_complete': len(g), 'amp_retry': len(retries), 'epoch': len(epochs),
                               'other': other_types},
        'update_min': updates[0] if updates else None, 'update_max': updates[-1] if updates else None,
        'updates_contiguous_1_to_max': updates == list(range(1, len(updates) + 1)),
        'missing_updates': sorted(set(range(1, TOTAL_UPDATES + 1)) - set(updates))[:50],
        'duplicate_complete_updates': dup_updates[:50],
        'duplicate_epoch_group_pairs': [list(k) for k, c in epoch_group_pairs.items() if c > 1][:50],
        'non_complete_group_records': bad_status[:50],
        'non_finite_records': nonfinite[:50], 'non_finite_record_count': len(nonfinite),
        'loss_terms_per_record': sorted({v['n_loss_terms'] for v in g.values()}),
        'amp_policies': sorted({str(v['amp_policy']) for v in g.values()}),
        'terminal_marker_counts': markers,
        'per_epoch': {str(k): per_epoch[k] for k in sorted(per_epoch)},
        'non_full_groups': tail_groups,
        'last_group': last, 'last_update': updates[-1] if updates else None,
        'scales': {'D_min': min(d_scales), 'D_max': max(d_scales), 'G_min': min(g_scales), 'G_max': max(g_scales),
                   'D_final': last['D_scale'] if last else None, 'G_final': last['G_scale'] if last else None},
        'amp_attempts_gt1_updates': sorted(u for u, v in g.items() if v['amp_attempts'] != 1),
        'retries': retry_audit, 'retry_log_order_ok': order_ok,
        'epoch_records': [{k: e.get(k) for k in ('epoch', 'global_step', 'optimizer_groups', 'microbatches', 'rows',
                                                 'ema_active', 'val_losses', 'val_metrics')} for e in epochs],
    }


def ckpt_meta_keys_with(meta, words):
    out = []

    def walk(obj, prefix):
        if isinstance(obj, dict):
            for k, v in obj.items():
                name = f'{prefix}.{k}' if prefix else str(k)
                if any(w in str(k).lower() for w in words):
                    out.append(name)
                walk(v, name)
    walk(meta, '')
    return out


def collect(run_root):
    import torch  # read-only use: torch.load(weights_only=True, map_location='cpu')
    root = Path(run_root)
    before_tree, before_small = tree_snapshot(root), small_hashes(root)
    manifest = json.loads((root / 'run_manifest.json').read_bytes())
    summary = json.loads((root / 'run_summary.json').read_bytes())
    index = json.loads((root / 'checkpoint_index.json').read_bytes())
    metrics = scan_metrics(root / 'metrics.jsonl')
    metrics_sha = sha256_file(root / 'metrics.jsonl')
    logs = {}
    for n in ('stdout.log', 'stderr.log'):
        b = (root / n).read_bytes()
        logs[n] = {'size': len(b), 'markers': {m: b.decode('utf-8', 'replace').count(m) for m in TERMINAL_MARKERS}}

    cdir = root / 'checkpoints' / 'ema_candidates'
    cand_files = sorted(p.name for p in cdir.iterdir())
    candidates = []
    for name in cand_files:
        p = cdir / name
        payload = torch.load(p, map_location='cpu', weights_only=True)
        meta = payload.get('metadata', {})
        candidates.append({
            'filename': name, 'file_size_bytes': p.stat().st_size, 'sha256': sha256_file(p),
            'payload_keys': sorted(payload), 'kind': payload.get('kind'),
            'n_e_art_ema_tensors': len(payload.get('e_art_ema', {})),
            'n_g_res_ema_tensors': len(payload.get('g_res_ema', {})),
            'metadata': meta,
            'val_fields': ckpt_meta_keys_with({'metadata': meta}, ('val',)),
            'test_fields': ckpt_meta_keys_with({'metadata': meta}, ('test',)),
        })
        del payload

    rp = root / 'checkpoints' / 'recovery' / 'latest.pt'
    rpay = torch.load(rp, map_location='cpu', weights_only=True)
    scalers = {k: {kk: vv for kk, vv in v.items()} for k, v in rpay.get('scalers', {}).items()}
    recovery = {
        'path': 'checkpoints/recovery/latest.pt', 'file_size_bytes': rp.stat().st_size, 'sha256': sha256_file(rp),
        'kind': rpay.get('kind'), 'labels': rpay.get('labels'), 'payload_keys': sorted(rpay),
        'position': rpay.get('position'), 'provenance': rpay.get('provenance'),
        'identity_map_sha256': rpay.get('identity_map_sha256'),
        'scalers': json.loads(json.dumps(scalers, default=float)),
        'modules': sorted(rpay.get('modules', {})), 'ema_modules': sorted(rpay.get('ema') or {}),
        'optimizers': sorted(rpay.get('optimizers', {})),
        'val_fields': ckpt_meta_keys_with({'position': rpay.get('position'), 'provenance': rpay.get('provenance')},
                                          ('val',)),
        'test_fields': ckpt_meta_keys_with({'position': rpay.get('position'), 'provenance': rpay.get('provenance')},
                                           ('test',)),
    }
    del rpay
    after_tree, after_small = tree_snapshot(root), small_hashes(root)
    return {
        'run_root': str(root), 'collected_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'torch_version': str(torch.__version__),
        'run_manifest': manifest, 'run_summary': summary, 'checkpoint_index': index,
        'small_file_sha256': before_small, 'metrics_sha256': metrics_sha, 'metrics': metrics, 'logs': logs,
        'candidate_dir_listing': cand_files, 'candidates': candidates, 'recovery': recovery,
        'run_root_tree_before': before_tree, 'run_root_tree_after': after_tree,
        'run_root_unchanged': before_tree == after_tree and before_small == after_small,
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


def build(c, repo_root, authority):
    G = Gates()
    E = EXPECTED
    man, summ, idx, m, rec = c['run_manifest'], c['run_summary'], c['checkpoint_index'], c['metrics'], c['recovery']
    prov = man.get('provenance', {})

    # A. authority
    G('A.repo_authority', authority['branch'] == 'm6-baselines' and authority['head'] == E['code_commit']
      and authority['origin'] == E['code_commit'] and authority['ls_remote'] == E['code_commit']
      and authority['clean'], authority)
    G('A.manifest_code_authority', man.get('git_commit') == E['code_commit'] and man.get('git_dirty') is False
      and prov.get('code_commit') == E['code_commit'])
    G('A.run_root_unchanged_during_audit', c['run_root_unchanged'])
    G('A.run_root_path', c['run_root'] == E['run_root'])

    # B. completion
    G('B.completion_status', man.get('completion_status') == 'completed' and summ.get('completion_status') == 'completed')
    G('B.identity', man.get('method_id') == E['method'] and summ.get('method_id') == E['method']
      and man.get('experiment_id') == E['experiment'] and man.get('experiment_seed') == E['seed']
      and summ.get('experiment_seed') == E['seed'] and man.get('run_id') == E['run_id']
      and summ.get('run_id') == E['run_id'] and idx.get('run_id') == E['run_id']
      and man.get('runner_mode') == 'SCIENTIFIC' and man.get('qualification_seed') is None
      and man.get('labels') == [])
    G('B.global_update', summ.get('global_update') == TOTAL_UPDATES)
    rc = m['record_type_counts']
    G('B.record_counts', rc['optimizer_group_complete'] == TOTAL_UPDATES and rc['epoch'] == EPOCHS
      and rc['other'] == {} and m['metrics_lines'] == TOTAL_UPDATES + EPOCHS + rc['amp_retry'], rc)
    G('B.update_sequence_exact', m['updates_contiguous_1_to_max'] and m['update_min'] == 1
      and m['update_max'] == TOTAL_UPDATES and not m['missing_updates'] and not m['duplicate_complete_updates']
      and not m['duplicate_epoch_group_pairs'])
    G('B.all_group_status_complete', not m['non_complete_group_records'])
    G('B.all_losses_and_grad_norms_finite', m['non_finite_record_count'] == 0)
    G('B.no_terminal_markers', not m['terminal_marker_counts']
      and all(not any(v['markers'].values()) for v in c['logs'].values()), m['terminal_marker_counts'])
    pe = m['per_epoch']
    G('B.per_epoch_groups', sorted(int(k) for k in pe) == list(range(1, EPOCHS + 1)) and all(
        v['groups'] == GROUPS_PER_EPOCH and v['max_group'] == GROUPS_PER_EPOCH and v['samples'] == TRAIN_ROWS
        and v['microbatches'] == MICROBATCHES_PER_EPOCH for v in pe.values()))
    tails = m['non_full_groups']
    G('B.only_tail_groups_partial', len(tails) == EPOCHS and all(
        t['group'] == GROUPS_PER_EPOCH and t['group_samples'] == 6 and t['microbatch_sizes'] == [4, 2]
        for t in tails.values()))
    er = m['epoch_records']
    G('B.epoch_records', [e['epoch'] for e in er] == list(range(1, EPOCHS + 1)) and all(
        e['global_step'] == e['epoch'] * GROUPS_PER_EPOCH and e['optimizer_groups'] == GROUPS_PER_EPOCH
        and e['microbatches'] == MICROBATCHES_PER_EPOCH and e['rows'] == TRAIN_ROWS
        and e['val_losses'] is None and e['val_metrics'] is None for e in er))
    fe = er[-1]
    G('B.final_epoch_record', fe['epoch'] == EPOCHS and fe['global_step'] == TOTAL_UPDATES
      and fe['optimizer_groups'] == GROUPS_PER_EPOCH and fe['microbatches'] == MICROBATCHES_PER_EPOCH
      and fe['rows'] == TRAIN_ROWS and fe['ema_active'] is True, fe)
    lg = m['last_group']
    G('B.final_group', m['last_update'] == TOTAL_UPDATES and lg['epoch'] == EPOCHS
      and lg['group'] == GROUPS_PER_EPOCH, lg)
    G('B.amp_policy', m['amp_policies'] == [E['amp_policy']] and man.get('amp_policy') == E['amp_policy'])

    # C. AMP retries
    retries = m['retries']
    retry_ok = all(
        r['all_optimizer_steps_taken_zero'] and r['all_optimizer_update_false'] and r['all_pre_group_state_restored']
        and r['update_not_advanced_on_retry'] and r['previous_update_completed'] and r['complete_record_count'] == 1
        and r['complete_epoch_group'] == [r['epoch'], r['group']]
        and r['retry_numbers'] == list(range(1, len(r['retry_numbers']) + 1))
        and r['complete_amp_attempts'] == len(r['retry_numbers']) + 1
        and r['all_amp_policy'] == [E['amp_policy']] for r in retries)
    G('C.retry_events_recovered', retry_ok)
    G('C.retry_log_order', m['retry_log_order_ok'])
    G('C.attempts_consistent', sorted(m['amp_attempts_gt1_updates']) == [r['global_update_attempted'] for r in retries])
    G('C.no_fail_closed_final', 'FAIL_CLOSED_AMP_OVERFLOW_FINAL' not in m['terminal_marker_counts'])
    G('C.no_duplicate_train_group', not m['duplicate_epoch_group_pairs'])

    # D. firewall (runner evidence only)
    acc = summ.get('access', {})
    G('D.firewall', man.get('val_split_accessed') is False and man.get('test_split_accessed') is False
      and summ.get('val_split_accessed') is False and summ.get('test_split_accessed') is False
      and all(acc.get(k) == 0 for k in ('non_train_images', 'val_images', 'test_images', 'val_metadata',
                                        'test_metadata')), acc)
    G('D.unique_ids_sha256', acc.get('unique_ids_sha256') == E['unique_ids_sha256'])
    G('D.train_open_accounting', acc.get('train_images') == acc.get('opened_events') == 2 * TRAIN_ROWS * EPOCHS,
      'two roles opened per TRAIN row per epoch; AMP retries re-opened nothing')

    # E. recovery
    rpos, rprov = rec.get('position') or {}, rec.get('provenance') or {}
    idx_rec = [r for r in idx['checkpoints'] if r.get('role') == 'recovery']
    G('E.recovery_present_and_indexed', len(idx_rec) == 1 and idx_rec[0]['sha256'] == rec['sha256']
      and idx_rec[0]['file_size_bytes'] == rec['file_size_bytes'])
    G('E.recovery_kind_labels', rec['kind'] == 'GPAT_RECOVERY_CHECKPOINT' and 'RECOVERY_ONLY' in (rec['labels'] or [])
      and 'NOT_ELIGIBLE_FOR_VAL_SELECTION' in rec['labels'])
    G('E.recovery_final_boundary', rpos.get('global_update') == TOTAL_UPDATES and rpos.get('stage') == 'generator'
      and rpos.get('epoch') == EPOCHS + 1 and rpos.get('next_group') == 1, rpos)
    G('E.recovery_provenance', rprov.get('code_commit') == E['code_commit'] and rprov.get('method') == E['method']
      and rprov.get('config_sha256') == E['config_sha256'] and rprov.get('gpu_env_lock_sha256') == E['env_lock_sha256']
      and rprov.get('source_manifest_sha256') == E['pair_manifest_sha256'])
    G('E.recovery_no_val_test_fields', not rec['val_fields'] and not rec['test_fields'])

    # F. EMA candidates
    cands = c['candidates']
    expected_names = [candidate_name(e) for e in CANDIDATE_EPOCHS]
    G('F.candidate_set_exact', c['candidate_dir_listing'] == expected_names and len(cands) == 51,
      len(c['candidate_dir_listing']))
    idx_c = {Path(r['path']).name: r for r in idx['checkpoints'] if r.get('role') == 'ema_candidate'}
    rows, cand_ok, sel_ok, valtest_ok = [], True, True, True
    for cd in cands:
        md = cd['metadata']
        ep = md.get('epoch')
        ir = idx_c.get(cd['filename'], {})
        ok = (cd['filename'] == candidate_name(ep) and md.get('global_update') == ep * GROUPS_PER_EPOCH
              and cd['kind'] == 'GPAT_EMA_CANDIDATE' and md.get('kind') == 'GPAT_EMA_CANDIDATE'
              and md.get('method') == E['method'] and md.get('seed') == E['seed']
              and md.get('code_commit') == E['code_commit'] and md.get('config_sha256') == E['config_sha256']
              and md.get('gpu_env_lock_sha256') == E['env_lock_sha256']
              and md.get('source_manifest_sha256') == E['pair_manifest_sha256']
              and md.get('ema_scope') == ['E_art', 'G_res'] and md.get('ema_decay') == 0.999
              and cd['payload_keys'] == ['e_art_ema', 'g_res_ema', 'kind', 'metadata']
              and ir.get('sha256') == cd['sha256'] and ir.get('file_size_bytes') == cd['file_size_bytes']
              and ir.get('epoch') == ep and ir.get('global_step') == ep * GROUPS_PER_EPOCH)
        sel = md.get('selected') is False and ir.get('selected_for_final') is False
        vt = not cd['val_fields'] and not cd['test_fields']
        cand_ok &= ok
        sel_ok &= sel
        valtest_ok &= vt
        rows.append({'filename': cd['filename'], 'epoch': ep, 'expected_global_update': ep * GROUPS_PER_EPOCH,
                     'embedded_global_update': md.get('global_update'), 'file_size_bytes': cd['file_size_bytes'],
                     'sha256': cd['sha256'], 'code_commit': md.get('code_commit'), 'method': md.get('method'),
                     'seed': md.get('seed'), 'kind': md.get('kind'), 'ema_scope': md.get('ema_scope'),
                     'ema_decay': md.get('ema_decay'), 'selected': md.get('selected'),
                     'index_selected_for_final': ir.get('selected_for_final'),
                     'val_fields': cd['val_fields'], 'test_fields': cd['test_fields']})
    rows.sort(key=lambda r: r['epoch'])
    G('F.candidate_identity_provenance', cand_ok)
    G('F.no_selection_state', sel_ok and summ.get('final_or_selected_checkpoint_path') is None
      and summ.get('final_or_selected_checkpoint_sha256') is None and summ.get('seed_level_evaluation_metrics') is None)
    G('F.no_val_test_fields', valtest_ok)
    G('F.epochs_10_to_60', [r['epoch'] for r in rows] == list(CANDIDATE_EPOCHS))
    G('F.index_complete', len(idx['checkpoints']) == 52 and len(idx_c) == 51)

    # G. provenance (repo files hashed against the expected authorities)
    repo = Path(repo_root)
    repo_hash = {
        'config': sha256_file(repo / man.get('config_path', 'configs/methods/gpat_b0.yaml')),
        'env_lock': sha256_file(repo / man.get('environment_lock_path', 'environments/gpat_m7_gpu.lock.json')),
        'amp_amendment': sha256_file(repo / E['amp_amendment_path']),
    }
    G('G.config_sha256', man.get('config_sha256') == prov.get('config_sha256') == repo_hash['config']
      == E['config_sha256'])
    G('G.pair_manifest_sha256', prov.get('source_manifest_sha256') == E['pair_manifest_sha256'])
    G('G.env_lock_sha256', man.get('dependency_fingerprint') == prov.get('gpu_env_lock_sha256')
      == repo_hash['env_lock'] == E['env_lock_sha256'])
    G('G.amp_amendment_sha256', repo_hash['amp_amendment'] == E['amp_amendment_sha256']
      and authority.get('amp_amendment_in_head') is True and man.get('amp_policy') == E['amp_policy'])

    manifest_text = ema_manifest_text(rows)
    retry_updates = [r['global_update_attempted'] for r in retries]
    sc = m['scales']
    status = STATUS_OK if not G.failed else STATUS_BLOCKED
    doc = {
        'milestone': 'M7D1-CLOSE-SEED42', 'classification': CLASSIFICATION, 'status': status,
        'failed_gates': G.failed,
        'method': E['method'], 'experiment': E['experiment'], 'seed': E['seed'], 'run_id': E['run_id'],
        'run_root': c['run_root'], 'authority_commit': E['code_commit'], 'authority': authority,
        'completion_status': man.get('completion_status'),
        'final_epoch': fe['epoch'], 'global_update': summ.get('global_update'),
        'optimizer_group_count': rc['optimizer_group_complete'], 'epoch_count': rc['epoch'],
        'update_sequence': {'first': m['update_min'], 'last': m['update_max'],
                            'contiguous': m['updates_contiguous_1_to_max'], 'missing': m['missing_updates'],
                            'duplicates': m['duplicate_complete_updates']},
        'final_epoch_record': fe, 'final_group': lg,
        'tail_group_policy': 'group 1105 of every epoch has group_samples=6, microbatch_sizes=[4,2] (8838 = 1104*8+6); expected, not an anomaly',
        'terminal_numerical_failure_count': sum(m['terminal_marker_counts'].values()),
        'terminal_marker_counts': m['terminal_marker_counts'],
        'non_finite_record_count': m['non_finite_record_count'],
        'amp_retry_summary': {
            'policy': E['amp_policy'], 'amendment': E['amp_amendment_path'],
            'total_retry_events': rc['amp_retry'], 'affected_update_count': len(retries),
            'affected_global_updates': retry_updates,
            'offending_optimizers': sorted({o for r in retries for o in r['offending_optimizers']}),
            'max_retries_for_one_update': max((len(r['retry_numbers']) for r in retries), default=0),
            'min_D_scale': sc['D_min'], 'min_G_scale': sc['G_min'], 'max_D_scale': sc['D_max'],
            'max_G_scale': sc['G_max'], 'events': retries,
        },
        'final_scalers': {'D_scale': sc['D_final'], 'G_scale': sc['G_final'],
                          'recovery_checkpoint_scalers': rec['scalers']},
        'firewall': {k: acc.get(k) for k in ('train_images', 'opened_events', 'unique_ids', 'unique_ids_sha256',
                                             'non_train_images', 'val_images', 'test_images', 'val_metadata',
                                             'test_metadata')}
        | {'val_split_accessed': summ.get('val_split_accessed'), 'test_split_accessed': summ.get('test_split_accessed'),
           'evidence_source': 'runner-produced run_summary.json/run_manifest.json only; no VAL/TEST file opened'},
        'recovery_checkpoint': {k: rec[k] for k in ('path', 'file_size_bytes', 'sha256', 'kind', 'labels', 'position',
                                                    'provenance', 'modules', 'ema_modules', 'optimizers')}
        | {'boundary_note': 'position.epoch=61/next_group=1 is the next epoch to run: last completed epoch is 60, '
                            'global_update 66300'},
        'ema_candidates': {'count': len(rows), 'epoch_range': [rows[0]['epoch'], rows[-1]['epoch']],
                           'sha256_manifest_file': 'outputs/audit/M7D1_E08_SEED42_EMA_SHA256.txt',
                           'sha256_manifest_digest': sha256_bytes(manifest_text.encode('utf-8')),
                           'candidates': rows},
        'provenance': {'code_commit': man.get('git_commit'), 'config_path': man.get('config_path'),
                       'config_sha256': man.get('config_sha256'),
                       'pair_manifest_sha256': prov.get('source_manifest_sha256'),
                       'environment_lock_path': man.get('environment_lock_path'),
                       'environment_lock_sha256': man.get('dependency_fingerprint'),
                       'm7c3_record_sha256': prov.get('m7c3_record_sha256'),
                       'teacher_sha256': prov.get('teacher_sha256'),
                       'amp_amendment_path': E['amp_amendment_path'], 'amp_amendment_sha256': repo_hash['amp_amendment'],
                       'amp_amendment_binding': 'via code commit 058e976 (amendment added in that commit) and '
                                                'run_manifest.amp_policy=ATOMIC_AMP_BACKOFF_RETRY; the manifest does '
                                                'not embed the amendment hash',
                       'repo_file_sha256': repo_hash},
        'run_root_files': {'small_file_sha256': c['small_file_sha256'], 'metrics_jsonl_sha256': c['metrics_sha256'],
                           'metrics_jsonl_lines': m['metrics_lines'],
                           'unchanged_during_audit': c['run_root_unchanged'],
                           'file_count': len(c['run_root_tree_after'])},
        'observations': [],
        'confirmations': {'val_selection_performed': False, 'test_access': False, 'bank_generated': False,
                          'downstream_run': False, 'training_resumed_or_restarted': False,
                          'run_root_modified': False, 'seed_1337_or_2026_started': False, 'b1_b2_b3_started': False},
        'gates': G.results,
        'collected_utc': c['collected_utc'],
        'audit_timestamp_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
    }
    return doc, manifest_text


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('collect')
    a.add_argument('--run-root', default=EXPECTED['run_root'])
    b = sub.add_parser('build')
    b.add_argument('--collected', required=True)
    b.add_argument('--authority', required=True, help='JSON file with branch/head/origin/ls_remote/clean')
    b.add_argument('--observations', default=None, help='optional JSON list of non-gating observations')
    b.add_argument('--repo-root', default=str(Path(__file__).resolve().parents[1]))
    args = ap.parse_args(argv)
    if args.cmd == 'collect':
        json.dump(collect(args.run_root), sys.stdout, sort_keys=True)
        return 0
    c = json.loads(Path(args.collected).read_text())
    authority = json.loads(Path(args.authority).read_text())
    doc, manifest_text = build(c, args.repo_root, authority)
    if args.observations:
        doc['observations'] = json.loads(Path(args.observations).read_text())
    out = Path(args.repo_root) / 'outputs' / 'audit'
    (out / 'M7D1_E08_SEED42_EMA_SHA256.txt').write_text(manifest_text)
    (out / 'M7D1_E08_SEED42_COMPLETION.json').write_text(json.dumps(doc, indent=2, sort_keys=True) + '\n')
    print(doc['status'], doc['failed_gates'])
    return 0 if doc['status'] == STATUS_OK else 1


if __name__ == '__main__':
    sys.exit(main())
