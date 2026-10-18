"""M7D1-N7A closure: OWNER-STOPPED confirmed generator collapse of the M7D1-A1 GPAT-B0 / E08 / seed 1337 scientific
run (run_id 62d7cfd98022a95a, launch authority 4466e12, curriculum GPAT-TransferBench-v1.0+M7D1-A1).

This is NOT a completed 60-epoch run, NOT a numerical hard failure and NOT an AMP hard failure. The owner authorized a
graceful stop after a direct TRAIN-only output diagnostic (tools/m7d1_n7a_seed1337_direct_output_diag.py) confirmed
that the generator output equals the live target (identity collapse). The scientific run was stopped with SIGTERM to
the main process only; the runner closed itself as `interrupted` at u15107 (mid-epoch 14).

Phases:
  collect  (GPU host, gpat-m7-gpu env, `python -I - collect < this file`) -- read-only, prints JSON. Opens no image,
           no split file and no VAL/TEST path; reads the run root (metrics, manifests, checkpoint bytes for hashing),
           the preserved epoch-13 recovery copy, and the listings of the two seed42 roots.
  build    (repository) -- gates + writes outputs/audit/M7D1_A1_E08_SEED1337_{STOPPED_COLLAPSE.json,
           COLLAPSE_DIRECT.json,STOPPED_COLLAPSE.md}.

No training, resume, replay, fix, selection, VAL/TEST, bank or downstream code is run.
"""
import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys

MILESTONE = 'M7D1-N7A'
STATUS = 'OWNER_STOPPED_CONFIRMED_GENERATOR_COLLAPSE'
STATUS_BLOCKED = 'M7D1_N7A_SEED1337_CLOSURE_NEEDS_REVIEW'
METHOD, EXPERIMENT, SEED = 'GPAT-B0', 'E08', 1337
RUN_ID = '62d7cfd98022a95a'
LAUNCH_COMMIT = '4466e1258c24e4a5d448dd5b30e8d8768dfe7174'
CURRICULUM_ID = 'GPAT-TransferBench-v1.0+M7D1-A1'
CONFIG_SHA256 = '0a268d2c9d1914a9e687d08cdfb96d5bf793eaa22559476b8e68d3cf58035d6a'
BRANCH = 'm6-baselines'
RUNTIME = '/home/student20261/workdir/GPAT_TransferBench_runtime'
RUN_ROOT = f'{RUNTIME}/runs/m7_a1/E08/seed_1337'
LAUNCH_CLONE = '/home/student20261/workdir/GPAT_TransferBench_M7A1_4466e12'
DIAG_DIR = f'{RUNTIME}/diagnostics/m7/M7D1_N7A_seed1337'
EPOCH13_COPY = f'{DIAG_DIR}/recovery_latest_epoch13_end_u14365.pt'
SEED42_ROOTS = {'a1_seed42': (f'{RUNTIME}/runs/m7_a1/E08/seed_42',
                              '9b1a85eaeaf635450807d43a7e19e97d447d6a18f2585c5fc7720b8efadfadf3', 59),
                'v1_0_seed42': (f'{RUNTIME}/runs/m7/E08/seed_42',
                                'ac64a06351fa8eb45e3efbef8fc64d297d6d3f0d78ee62f8b3cfcd26b6de893a', 59)}
LAUNCH_UTC = '2026-10-17T15:02:30Z'

# Observed during the session (GPU host shell output, recorded at the time; not re-derivable afterwards).
STOP = {
    'authorization': 'owner decision checkpoint: stop gracefully if direct output collapse is confirmed',
    'diagnostic_utc': '2026-10-17T19:09:22Z', 'update_before_diagnostic': 14948,
    'pre_stop_snapshot_utc': '2026-10-17T19:11:20Z', 'update_at_pre_stop_snapshot': 15089,
    'signal': 'SIGTERM', 'target': 'main training process only (PID 3120153; its 4 DataLoader workers were not '
                                    'signalled)',
    'sent_utc': '2026-10-17T19:11:35Z', 'exited_after_seconds': 24, 'process_gone_utc': '2026-10-17T19:11:59Z',
    'cli_stdout_last_line': '{"status": "finished", "run_dir": "' + RUN_ROOT + '"}',
    'shell_exit_status': 0,
    'exit_status_interpretation': 'EXIT_STATUS=0 and the CLI word "finished" only mean the graceful-stop path returned '
                                  'normally; the runner recorded completion_status=interrupted. It is NOT scientific '
                                  'completion.',
}
PRE_STOP = {'listing_digest': '84f5b8cd3fe37406a7e49dbaccb2d5744445753d8f7172efcef32de25394d2f8', 'file_count': 11,
            'sha256': {'metrics.jsonl': '6e292e3e0d35894930d795d293db49c5c56f6b6d59e1975702b46f21b34e42ae',
                       'run_manifest.json': '2965569d6cbf382329ad4f9d6ed62e0717543112222b93dae51be0c7608163d5',
                       'checkpoint_index.json': 'f4ab956350ce5dd011a0675053d46151634b099a4d999840b31b58fd452f2526',
                       'resolved_config.yaml': 'dcca6fdf3c1b56fc60220344174a0d30ef7ba6cd75165695a632ffe2c8d5c6d7',
                       'checkpoints/recovery/latest.pt':
                           '0f845ec5530b0d7ab6eca521eba374e9efd5698da28e1d58271c74ed48ade2b8'}}
POST_STOP = {'listing_digest': '07fd4e0bfeef18f13ed8977d5cafd8fc7d40c22d002ab4069552c75779df04b7', 'file_count': 12}
EXPECTED_CLOSEOUT_CHANGES = {
    'added': ['run_summary.json'],
    'rewritten': ['checkpoint_index.json', 'checkpoints/recovery/latest.pt', 'metrics.jsonl', 'run_manifest.json'],
    'unchanged': ['checkpoints/ema_candidates/ema_epoch_10.pt', 'checkpoints/ema_candidates/ema_epoch_11.pt',
                  'checkpoints/ema_candidates/ema_epoch_12.pt', 'checkpoints/ema_candidates/ema_epoch_13.pt',
                  'resolved_config.yaml', 'stderr.log', 'stdout.log'],
}
LAST_COMPLETE_UPDATE, LAST_COMPLETE_EPOCH = 15107, 13
UPDATES_PER_EPOCH = 1105
ONSET = {'grad_spike_update': 2167, 'signature_from': 2174}
ONSET_WINDOW = (2150, 2200)
SIGNATURE = 'budget >= 0.0099 and bg < 1e-6 and lm < 1e-6 (M7D1-N2 historical-collapse training signature)'
SCHEDULE_FIELDS = ('lambda_adv', 'scale_hf', 'lambda_con', 'lambda_spec', 'learning_rate', 'epoch', 'group')
EVIDENCE = {'closure': 'outputs/audit/M7D1_A1_E08_SEED1337_STOPPED_COLLAPSE.json',
            'direct': 'outputs/audit/M7D1_A1_E08_SEED1337_COLLAPSE_DIRECT.json',
            'report': 'outputs/audit/M7D1_A1_E08_SEED1337_STOPPED_COLLAPSE.md',
            'collected': 'outputs/audit/M7D1_A1_E08_SEED1337_STOPPED_COLLECTED.json',
            'diag_tool': 'tools/m7d1_n7a_seed1337_direct_output_diag.py',
            'sha256': 'outputs/audit/M7D1_A1_E08_SEED1337_STOPPED_COLLAPSE_EVIDENCE_SHA256.txt'}


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()


def listing_digest(root):
    """Same convention as earlier sessions: `find . -type f -printf "%P %s %T@\\n" | sort | sha256sum`."""
    lines = []
    for dirpath, _, filenames in os.walk(root):
        for n in filenames:
            p = Path(dirpath) / n
            st = p.stat()
            lines.append(f'{p.relative_to(root)} {st.st_size} {st.st_mtime_ns // 10**9}.{st.st_mtime_ns % 10**9:09d}0')
    return sha256_bytes(''.join(f'{x}\n' for x in sorted(lines)).encode()), len(lines)


def finite_tree(x):
    if isinstance(x, bool) or x is None or isinstance(x, str):
        return True
    if isinstance(x, (int, float)):
        return math.isfinite(x)
    if isinstance(x, dict):
        return all(finite_tree(v) for v in x.values())
    if isinstance(x, list):
        return all(finite_tree(v) for v in x)
    return True


def median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else None


# ---------------------------------------------------------------------------- A1 schedule (production reference)
def expected_schedule(u, rc):
    cur = rc.curriculum_a1(u)
    return {'lambda_adv': cur['lambda_adv'], 'scale_hf': cur['s_hf'], 'lambda_con': cur['lambda_con'],
            'lambda_spec': cur['lambda_spec'], 'learning_rate': rc.main_lr(u),
            'epoch': (u - 1) // rc.UPDATES_PER_EPOCH + 1, 'group': (u - 1) % rc.UPDATES_PER_EPOCH + 1}


# ============================================================================ collect (GPU host, read-only)
def metrics_audit(path, rc):
    counts, retries, non_complete, nonfinite, mism, epochs = {}, [], [], [], {}, []
    prev, dups, gaps, last = 0, [], [], None
    sig_first, sig_run, sig_longest, sig_count, sig_run_start = None, 0, (0, None), 0, None
    window, per_epoch, scales = [], {}, {'D_min': math.inf, 'G_min': math.inf}
    gmax_pre = (0.0, None)
    with open(path, 'rb') as f:
        for raw in f:
            d = json.loads(raw)
            rt = d.get('record_type')
            key = rt + ('_' + d['group_status'] if rt == 'optimizer_group' else '')
            counts[key] = counts.get(key, 0) + 1
            # amp_retry records describe the overflowed (discarded) attempt, so only accepted records are checked
            if rt != 'amp_retry' and not finite_tree(d):
                nonfinite.append([rt, d.get('global_update', d.get('epoch'))])
            if rt == 'amp_retry':
                retries.append({k: d.get(k) for k in ('global_update_attempted', 'offending_optimizers', 'old_scale',
                                                      'new_scale', 'attempt')})
                continue
            if rt == 'epoch':
                epochs.append(d.get('epoch'))
                continue
            if rt != 'optimizer_group':
                continue
            if d.get('group_status') != 'COMPLETE':
                non_complete.append([d.get('global_update'), d.get('group_status')])
                continue
            u = d['global_update']
            if u == prev:
                dups.append(u)
            elif u != prev + 1:
                gaps.append([prev, u])
            prev, last = u, d
            for k, v in expected_schedule(u, rc).items():
                if not (type(d.get(k)) is type(v) and d.get(k) == v):
                    mism.setdefault(k, []).append([u, d.get(k), v])
            scales['D_min'] = min(scales['D_min'], d['D_scale'])
            scales['G_min'] = min(scales['G_min'], d['G_scale'])
            tl = d['train_losses']
            s = tl['budget'] >= 0.0099 and tl['bg'] < 1e-6 and tl['lm'] < 1e-6
            if s:
                sig_count += 1
                sig_first = sig_first or u
                if sig_run == 0:
                    sig_run_start = u
                sig_run += 1
                if sig_run > sig_longest[0]:
                    sig_longest = (sig_run, sig_run_start)
            else:
                sig_run = 0
            if u < ONSET['grad_spike_update'] and d['G_grad_norm'] > gmax_pre[0]:
                gmax_pre = (d['G_grad_norm'], u)
            e = per_epoch.setdefault(d['epoch'], {k: [] for k in ('budget', 'bg', 'lm', 'id', 'G_grad_norm',
                                                                     'D_total')})
            for k in ('budget', 'bg', 'lm', 'id', 'D_total'):
                e[k].append(tl[k])
            e['G_grad_norm'].append(d['G_grad_norm'])
            e.setdefault('signature', 0)
            e['signature'] += s
            if ONSET_WINDOW[0] <= u <= ONSET_WINDOW[1]:
                window.append({'u': u, 'epoch': d['epoch'], 'group': d['group'], 'stage': rc.curriculum_a1(u)['stage'],
                               'lambda_adv': d['lambda_adv'], 'scale_hf': d['scale_hf'],
                               'amp_attempts': d['amp_attempts'], 'G_grad_norm': d['G_grad_norm'],
                               'D_grad_norm': d['D_grad_norm'], 'G_scale': d['G_scale'], 'D_scale': d['D_scale'],
                               'budget': tl['budget'], 'bg': tl['bg'], 'lm': tl['lm'], 'id': tl['id'],
                               'low': tl['low'], 'gadv': tl['gadv'], 'G_total': tl['G_total'],
                               'D_total': tl['D_total'], 'signature': s})
    per_epoch_summary = {str(k): {'updates': len(v['budget']), 'signature_updates': v['signature'],
                                  **{f'{m}_median': median(v[m]) for m in ('budget', 'bg', 'lm', 'id', 'G_grad_norm',
                                                                           'D_total')},
                                  'G_grad_norm_min': min(v['G_grad_norm'])}
                         for k, v in sorted(per_epoch.items())}
    return {
        'record_counts': counts, 'update_first': 1 if counts.get('optimizer_group_COMPLETE') else None,
        'update_last': prev, 'duplicates': dups, 'gaps': gaps, 'non_complete_group_records': non_complete,
        'nonfinite_records': nonfinite, 'epoch_records': epochs,
        'last_complete': {k: last[k] for k in ('global_update', 'epoch', 'group', 'utc', 'G_scale', 'D_scale',
                                               'G_grad_norm', 'D_grad_norm')},
        'schedule_audit': {'fields': list(SCHEDULE_FIELDS), 'updates_checked': prev,
                           'mismatch_total': sum(len(v) for v in mism.values()),
                           'mismatches_first': {k: v[:5] for k, v in mism.items()}},
        'amp': {'retry_events': retries, 'retry_event_count': len(retries),
                'retry_updates': sorted({r['global_update_attempted'] for r in retries}),
                'min_scales': scales, 'final_scales': {'D': last['D_scale'], 'G': last['G_scale']}},
        'signature': {'definition': SIGNATURE, 'updates': sig_count, 'first': sig_first,
                      'longest_run': sig_longest[0], 'longest_run_start': sig_longest[1],
                      'run_at_end': sig_run, 'run_at_end_start': sig_run_start if sig_run else None},
        'G_grad_norm_max_before_spike': {'value': gmax_pre[0], 'update': gmax_pre[1]},
        'onset_window': window, 'per_epoch': per_epoch_summary,
    }


def recovery_payload(path):
    import torch
    b = Path(path).read_bytes()
    p = torch.load(io.BytesIO(b), map_location='cpu', weights_only=True)
    prov = p.get('provenance', {})
    return {'path': str(path), 'sha256': sha256_bytes(b), 'size': len(b), 'kind': p.get('kind'),
            'position': p.get('position'), 'labels': p.get('labels'),
            'provenance': {k: prov.get(k) for k in ('code_commit', 'curriculum_id', 'config_sha256', 'method')},
            'stat_mtime_ns': Path(path).stat().st_mtime_ns}


def git(root, *args):
    return subprocess.run(['git', '-C', str(root), *args], capture_output=True, text=True, check=True).stdout.strip()


def processes():
    out = []
    for pid in os.listdir('/proc'):
        if not pid.isdigit() or int(pid) == os.getpid():
            continue
        try:
            cmd = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
        except OSError:
            continue
        if 'gpatbench.cli' in cmd and 'train-generator' in cmd:
            out.append([int(pid), cmd.strip()])
    return out


def collect():
    sys.dont_write_bytecode = True
    sys.path.insert(0, LAUNCH_CLONE)
    from methods.gpat import runtime_contract as rc
    assert rc.CURRICULUM_ID == CURRICULUM_ID
    s42_before = {k: listing_digest(v[0]) for k, v in SEED42_ROOTS.items()}
    run_before = listing_digest(RUN_ROOT)
    files = {}
    for dirpath, _, filenames in os.walk(RUN_ROOT):
        for n in filenames:
            p = Path(dirpath) / n
            files[str(p.relative_to(RUN_ROOT))] = {'size': p.stat().st_size, 'sha256': sha256_file(p)}
    root = Path(RUN_ROOT)
    c = {
        'run_root': RUN_ROOT,
        'run_root_listing': {'digest': run_before[0], 'file_count': run_before[1]},
        'run_root_files': files,
        'run_manifest': json.loads((root / 'run_manifest.json').read_text()),
        'run_summary': json.loads((root / 'run_summary.json').read_text()),
        'checkpoint_index': json.loads((root / 'checkpoint_index.json').read_text()),
        'metrics': metrics_audit(root / 'metrics.jsonl', rc),
        'recovery_final': recovery_payload(root / 'checkpoints' / 'recovery' / 'latest.pt'),
        'recovery_epoch13_copy': recovery_payload(EPOCH13_COPY),
        'diag_dir_listing': sorted(os.listdir(DIAG_DIR)),
        'a1_runs_listing': sorted(str(p.relative_to(f'{RUNTIME}/runs/m7_a1')) for p in
                                  Path(f'{RUNTIME}/runs/m7_a1').glob('*/*')),
        'launch_clone': {'path': LAUNCH_CLONE, 'head': git(LAUNCH_CLONE, 'rev-parse', 'HEAD'),
                         'branch': git(LAUNCH_CLONE, 'branch', '--show-current'),
                         'porcelain': git(LAUNCH_CLONE, 'status', '--porcelain'),
                         'ignored': git(LAUNCH_CLONE, 'status', '--porcelain', '--ignored')},
        'training_processes': processes(),
    }
    run_after = listing_digest(RUN_ROOT)
    c['run_root_listing']['digest_after_collect'] = run_after[0]
    c['seed42_roots'] = {k: {'path': v[0], 'digest_before': s42_before[k][0], 'file_count': s42_before[k][1],
                             'digest_after': listing_digest(v[0])[0]} for k, v in SEED42_ROOTS.items()}
    return c


# ============================================================================ build (repository)
class Gates:
    def __init__(self):
        self.results, self.failed = [], []

    def __call__(self, name, ok, detail=None):
        self.results.append({'gate': name, 'pass': bool(ok), **({'detail': detail} if detail is not None else {})})
        if not ok:
            self.failed.append(name)


def compact_direct(raw, diag_tool_sha256, raw_sha256):
    """Compact durable form of the scratch n7a_diag.json (no new measurement: values copied, rounded never)."""
    keep = ('x_hat_minus_target_mean_abs_per_image', 'x_hat_minus_target_max_abs',
            'x_hat_minus_target_mean_abs_u8_levels_per_image', 'u8_x_hat_equals_u8_target_count',
            'u8_pixels_differing_fraction_per_image', 'u8_max_level_diff', 'x_hat_minus_source_mean_abs_per_image',
            'A_mean_per_image', 'A_pixel_quantiles', 'M_mean_per_image', 'M_pixel_quantiles', 'M_fraction_gt_0p99',
            'M_fraction_lt_0p01', 'delta_abs_mean', 'delta_HF_abs_quantiles')
    role = {'s1337_a1': 'subject: M7D1-A1 seed1337 (this run)',
            's42_v10': 'reference: historical v1.0 seed42 (known collapsed, M7D1-N2 F01)',
            's42_a1': 'reference: healthy M7D1-A1 seed42 (completed N6B, selected N6C)'}
    states = {}
    for k, s in raw['states'].items():
        states[k] = {'run': k.split('/')[0], 'role': role[k.split('/')[0]], 'checkpoint_sha256': s['sha256'],
                     **({'global_update': s['global_update']} if 'global_update' in s else {}),
                     **({'position': s['position']} if 'position' in s else {}),
                     **{m: s[m] for m in keep}}
    fw = raw['firewall']
    wo = fw['write_opens']
    clone_pyc = [w[0] for w in wo if w[0].startswith(LAUNCH_CLONE + '/') and '/__pycache__/' in w[0]]
    other_writes = [w[0] for w in wo if w[0] not in clone_pyc]
    return {
        'milestone': MILESTONE, 'kind': 'DIAGNOSTIC_ONLY direct generator-output confirmation (TRAIN only)',
        'conclusion': 'DIRECT_COLLAPSE_CONFIRMED',
        'collected_utc': raw['collected_utc'], 'torch': raw['torch'], 'code': raw['clone_head'],
        'forward': 'production GPATCore (methods.gpat.model) at the pinned 4466e12 launch clone, eval mode, CPU fp32, '
                   'no grad, deterministic algorithms, 4 threads, no teacher, no discriminator',
        'scale_hf': raw['scale_hf'],
        'scale_hf_reason': 'curriculum stage-2 value in force at epochs 10..13 (u <= 14365 < 16576) for all three runs',
        'diag_tool': {'path': EVIDENCE['diag_tool'], 'sha256': diag_tool_sha256},
        'raw_result_sha256': raw_sha256,
        'subset': {'size_pairs': len(raw['subset']), 'images': 2 * len(raw['subset']), 'rule': raw['subset_rule'],
                   'datasets': sorted({r['dataset'] for r in raw['subset']}), 'rows': raw['subset']},
        'states': states,
        'firewall': {'access_log': fw['access_log'], 'faces_files_opened': fw['faces_files_opened'],
                     'faces_opened_all_in_subset': fw['faces_opened_all_in_subset'],
                     'suspicious_val_test_paths': fw['suspicious_val_test_paths'],
                     'run_root_files_opened': fw['run_root_files_opened'],
                     'run_root_write_opens': [p for p in other_writes if p.startswith(RUNTIME)],
                     'other_write_opens': other_writes,
                     'launch_clone_pycache_write_opens': len(clone_pyc),
                     'launch_clone_pycache_note': 'git-ignored __pycache__ entries in the launch clone written by the '
                                                  'diagnostic interpreter; removed afterwards, clone porcelain clean'},
        'other_roots_listing': raw['other_roots_listing'],
    }


def direct_findings(direct):
    st = direct['states']
    subj = {k: v for k, v in st.items() if v['run'] == 's1337_a1'}
    v10 = {k: v for k, v in st.items() if v['run'] == 's42_v10'}
    ok = {k: v for k, v in st.items() if v['run'] == 's42_a1'}
    rng = lambda d, m, f='median': [min(v[m][f] for v in d.values()), max(v[m][f] for v in d.values())]
    return {
        's1337_a1': {'states': sorted(subj), 'exact_u8_target_copies': {k: v['u8_x_hat_equals_u8_target_count']
                                                                          for k, v in subj.items()},
                     'x_hat_minus_target_mean_abs_median_range': rng(subj, 'x_hat_minus_target_mean_abs_per_image'),
                     'x_hat_minus_target_max_abs': max(v['x_hat_minus_target_max_abs'] for v in subj.values()),
                     'u8_max_level_diff': max(v['u8_max_level_diff'] for v in subj.values()),
                     'A_mean_median_range': rng(subj, 'A_mean_per_image'),
                     'A_pixel_max': max(v['A_pixel_quantiles']['max'] for v in subj.values()),
                     'M_mean_median_range': rng(subj, 'M_mean_per_image'),
                     'M_fraction_lt_0p01': sorted({v['M_fraction_lt_0p01'] for v in subj.values()}),
                     'delta_HF_abs_p50_range': [min(v['delta_HF_abs_quantiles']['p50'] for v in subj.values()),
                                                max(v['delta_HF_abs_quantiles']['p50'] for v in subj.values())],
                     'hf_residual': 'saturated at |delta_HF| = s_hf but completely masked out (M ~ 0 everywhere)'},
        's42_v10': {'states': sorted(v10), 'exact_u8_target_copies': {k: v['u8_x_hat_equals_u8_target_count']
                                                                        for k, v in v10.items()},
                    'A_mean_median_range': rng(v10, 'A_mean_per_image'),
                    'M_mean_median_range': rng(v10, 'M_mean_per_image'),
                    'failure_mode': 'same target-copy failure mode'},
        's42_a1': {'states': sorted(ok), 'exact_u8_target_copies': {k: v['u8_x_hat_equals_u8_target_count']
                                                                      for k, v in ok.items()},
                   'x_hat_minus_target_mean_abs_median_range': rng(ok, 'x_hat_minus_target_mean_abs_per_image'),
                   'A_mean_median_range': rng(ok, 'A_mean_per_image'),
                   'M_mean_median_range': rng(ok, 'M_mean_per_image'),
                   'u8_pixels_differing_fraction_median_range': rng(ok, 'u8_pixels_differing_fraction_per_image'),
                   'failure_mode': 'healthy: nonzero x_hat - target edit, nonzero artifact map, open mask'},
    }


def build(c, direct, repo_root):
    G = Gates()
    man, summ, m = c['run_manifest'], c['run_summary'], c['metrics']
    rf, r13 = c['recovery_final'], c['recovery_epoch13_copy']
    prov = man['provenance']

    G('A.identity', man['method_id'] == METHOD and man['experiment_id'] == EXPERIMENT and man['run_id'] == RUN_ID
      and man['experiment_seed'] == SEED and man['runner_mode'] == 'SCIENTIFIC' and man['git_commit'] == LAUNCH_COMMIT
      and man['git_dirty'] is False and man['config_sha256'] == CONFIG_SHA256
      and prov.get('curriculum_id') == CURRICULUM_ID and c['run_root'] == RUN_ROOT)
    lc = c['launch_clone']
    G('A.launch_clone', lc['head'] == LAUNCH_COMMIT and lc['branch'] == BRANCH and lc['porcelain'] == ''
      and lc['ignored'] == '!! third_party/source_cache/'
      and man['command_line'] == f'{LAUNCH_CLONE}/gpatbench/cli.py train-generator --method {METHOD} --seed {SEED}', lc)

    # B. interrupted, not completed
    G('B.interrupted_not_completed', man['completion_status'] == 'interrupted'
      and summ['completion_status'] == 'interrupted' and summ['global_update'] == LAST_COMPLETE_UPDATE
      and summ['final_or_selected_checkpoint_path'] is None and summ['final_or_selected_checkpoint_sha256'] is None)
    G('B.no_process_left', c['training_processes'] == [], c['training_processes'])

    # C. metrics integrity
    G('C.complete_contiguous', m['record_counts'].get('optimizer_group_COMPLETE') == LAST_COMPLETE_UPDATE
      and m['update_first'] == 1 and m['update_last'] == LAST_COMPLETE_UPDATE and m['duplicates'] == []
      and m['gaps'] == [] and m['non_complete_group_records'] == [])
    G('C.epoch_records', m['epoch_records'] == list(range(1, LAST_COMPLETE_EPOCH + 1))
      and m['record_counts'].get('epoch') == LAST_COMPLETE_EPOCH)
    G('C.last_complete_mid_epoch14', m['last_complete']['global_update'] == LAST_COMPLETE_UPDATE
      and m['last_complete']['epoch'] == 14 and m['last_complete']['group'] == LAST_COMPLETE_UPDATE - 13 * 1105)
    G('C.a1_schedule', m['schedule_audit']['updates_checked'] == LAST_COMPLETE_UPDATE
      and m['schedule_audit']['mismatch_total'] == 0)
    G('C.finite', m['nonfinite_records'] == [])
    G('C.record_types', set(m['record_counts']) == {'optimizer_group_COMPLETE', 'epoch', 'amp_retry'},
      m['record_counts'])
    amp = m['amp']
    G('C.amp_recovered', amp['retry_event_count'] == 9 and len(amp['retry_updates']) == 8
      and amp['retry_updates'] == [1798, 2132, 7771, 9802, 12055, 12146, 14057, 14173]
      and amp['min_scales'] == {'D_min': 8192.0, 'G_min': 65536.0}
      and amp['final_scales'] == {'D': 16384.0, 'G': 2097152.0})

    # D. firewall
    acc = summ['access']
    G('D.no_val_test', man['val_split_accessed'] is False and man['test_split_accessed'] is False
      and summ['val_split_accessed'] is False and summ['test_split_accessed'] is False
      and acc['val_images'] == acc['test_images'] == acc['non_train_images'] == 0
      and acc['val_metadata'] == acc['test_metadata'] == 0 and acc['train_images'] == acc['opened_events'] > 0, acc)
    fw = direct['firewall']
    G('D.diagnostic_train_only', fw['access_log']['train_images'] == 64 and fw['access_log']['val_images'] == 0
      and fw['access_log']['test_images'] == 0 and fw['access_log']['non_train_images'] == 0
      and fw['faces_files_opened'] == 64 and fw['faces_opened_all_in_subset'] is True
      and fw['suspicious_val_test_paths'] == [] and fw['run_root_write_opens'] == [], fw)

    # E. checkpoints: no selection, recovery states
    ix = c['checkpoint_index']['checkpoints']
    ema = [r for r in ix if r['role'] == 'ema_candidate']
    G('E.no_selection', c['checkpoint_index']['labels'] == [] and all(r['selected_for_final'] is False for r in ix)
      and [r['epoch'] for r in ema] == [10, 11, 12, 13])
    G('E.recovery_final', rf['kind'] == 'GPAT_RECOVERY_CHECKPOINT' and rf['position']['epoch'] == 14
      and rf['position']['next_group'] == 743 and rf['position']['global_update'] == LAST_COMPLETE_UPDATE
      and rf['provenance']['code_commit'] == LAUNCH_COMMIT and 'RECOVERY_ONLY' in rf['labels']
      and 'NOT_ELIGIBLE_FOR_VAL_SELECTION' in rf['labels']
      and c['run_root_files']['checkpoints/recovery/latest.pt']['sha256'] == rf['sha256'], rf['position'])
    G('E.recovery_epoch13_copy', r13['sha256'] == PRE_STOP['sha256']['checkpoints/recovery/latest.pt']
      and r13['position']['epoch'] == 14 and r13['position']['next_group'] == 1
      and r13['position']['global_update'] == 14365 and r13['provenance']['code_commit'] == LAUNCH_COMMIT
      and 'RECOVERY_ONLY' in r13['labels'] and c['diag_dir_listing'] == [Path(EPOCH13_COPY).name], r13['position'])

    # F. root integrity
    rl = c['run_root_listing']
    G('F.run_root_unchanged_since_stop', rl['digest'] == rl['digest_after_collect'] == POST_STOP['listing_digest']
      and rl['file_count'] == POST_STOP['file_count'], rl)
    files = c['run_root_files']
    changed = sorted(k for k, v in PRE_STOP['sha256'].items() if files.get(k, {}).get('sha256') != v)
    G('F.closeout_changes_exact', sorted(set(files) - set(PRE_STOP['sha256']) - set(EXPECTED_CLOSEOUT_CHANGES[
        'unchanged'])) == EXPECTED_CLOSEOUT_CHANGES['added']
      and changed == EXPECTED_CLOSEOUT_CHANGES['rewritten']
      and files['resolved_config.yaml']['sha256'] == PRE_STOP['sha256']['resolved_config.yaml']
      and files['stderr.log']['size'] == 0 and files['stdout.log']['size'] == 0
      and all(files[e['path']]['sha256'] == e['sha256'] for e in ema), changed)
    G('F.a1_runs_only_e08_seed42_seed1337', c['a1_runs_listing'] == ['E08/seed_1337', 'E08/seed_42'],
      c['a1_runs_listing'])
    for k, (path, digest, n) in SEED42_ROOTS.items():
        s = c['seed42_roots'][k]
        G(f'F.{k}_root_unchanged', s['digest_before'] == s['digest_after'] == digest and s['file_count'] == n, s)
        G(f'F.{k}_root_unchanged_during_diagnostic',
          direct['other_roots_listing']['before'][{'a1_seed42': 's42_a1', 'v1_0_seed42': 's42_v10'}[k]][0]
          == direct['other_roots_listing']['after'][{'a1_seed42': 's42_a1', 'v1_0_seed42': 's42_v10'}[k]][0] == digest)

    # G. direct collapse confirmation
    st = direct['states']
    subj = [v for v in st.values() if v['run'] == 's1337_a1']
    G('G.direct_states', sorted(st) == sorted(
        [f'{r}/ema_epoch_{e:02d}' for r in ('s1337_a1', 's42_a1', 's42_v10') for e in (10, 11, 12, 13)]
        + ['s1337_a1/recovery_live', 's1337_a1/recovery_ema']) and direct['code'] == LAUNCH_COMMIT
      and direct['subset']['size_pairs'] == 32)
    G('G.subject_ema_hashes_match_index', all(st[f's1337_a1/ema_epoch_{e["epoch"]:02d}']['checkpoint_sha256']
                                              == e['sha256'] for e in ema)
      and st['s1337_a1/recovery_live']['checkpoint_sha256'] == r13['sha256'])
    G('G.subject_exact_target_copy', all(v['u8_x_hat_equals_u8_target_count'] == 32 and v['u8_max_level_diff'] == 0
                                         and v['M_fraction_lt_0p01'] == 1.0
                                         and v['x_hat_minus_target_max_abs'] < 1e-6 for v in subj))
    G('G.historical_v10_same_failure', all(v['u8_x_hat_equals_u8_target_count'] == 32 for v in st.values()
                                           if v['run'] == 's42_v10'))
    G('G.healthy_a1_seed42_open', all(v['u8_x_hat_equals_u8_target_count'] == 0 and v['M_fraction_gt_0p99'] > 0.99
                                      and v['A_mean_per_image']['median'] > 1e-3 for v in st.values()
                                      if v['run'] == 's42_a1'))

    # H. onset (scalar trajectory, stage 1)
    win = {r['u']: r for r in m['onset_window']}
    sp = win[ONSET['grad_spike_update']]
    sg = m['signature']
    G('H.onset_stage1_no_adv', sp['stage'] == 1 and sp['epoch'] == 2 and sp['lambda_adv'] == 0.0
      and all(r['lambda_adv'] == 0.0 and r['stage'] == 1 for r in m['onset_window'])
      and sp['amp_attempts'] == 1 and sp['G_grad_norm'] > 2.0
      and win[ONSET['grad_spike_update'] - 1]['G_grad_norm'] < 0.1, sp)
    gap = win[ONSET['signature_from'] + 1]
    G('H.signature_sustained_to_stop', sg['first'] == ONSET['signature_from']
      and sg['run_at_end_start'] == ONSET['signature_from'] + 2
      and sg['run_at_end'] == LAST_COMPLETE_UPDATE - ONSET['signature_from'] - 1
      and sg['updates'] == LAST_COMPLETE_UPDATE - ONSET['signature_from']
      and gap['signature'] is False and 1e-6 <= gap['lm'] < 1.1e-6 and gap['bg'] < 1e-6 and gap['budget'] >= 0.0099,
      dict(sg, single_miss=gap))

    findings = direct_findings(direct)
    doc = {
        'milestone': MILESTONE, 'status': STATUS if not G.failed else STATUS_BLOCKED, 'final_run_status': STATUS,
        'failed_gates': G.failed, 'gates': G.results,
        'this_is_not': ['a completed 60-epoch scientific run', 'a numerical hard failure', 'an AMP hard failure',
                        'a selectable or bank-eligible result'],
        'classification': 'owner-authorized graceful stop after direct TRAIN-only confirmation of catastrophic '
                          'generator identity collapse',
        'method': METHOD, 'experiment': EXPERIMENT, 'seed': SEED, 'run_id': RUN_ID,
        'launch_authority_commit': LAUNCH_COMMIT, 'curriculum_id': CURRICULUM_ID, 'run_root': RUN_ROOT,
        'launch_utc': LAUNCH_UTC, 'launch_clone': lc,
        'provenance': prov,
        'environment': {k: man[k] for k in ('python_version', 'pytorch_version', 'cuda_version', 'cudnn_version',
                                            'gpu_model', 'environment_lock_path', 'amp_policy')},
        'completion': {'manifest_completion_status': man['completion_status'],
                       'summary_completion_status': summ['completion_status'],
                       'last_complete_update': LAST_COMPLETE_UPDATE, 'last_complete_epoch': LAST_COMPLETE_EPOCH,
                       'last_complete_record': m['last_complete'],
                       'planned_updates': 66300, 'planned_epochs': 60,
                       'completed_fraction_of_plan': LAST_COMPLETE_UPDATE / 66300},
        'owner_stop': STOP,
        'integrity': {'record_counts': m['record_counts'],
                      'update_sequence': {'first': m['update_first'], 'last': m['update_last'],
                                          'contiguous': m['gaps'] == [] and m['duplicates'] == [],
                                          'duplicates': m['duplicates'], 'gaps': m['gaps']},
                      'non_complete_group_records': m['non_complete_group_records'],
                      'nonfinite_records': m['nonfinite_records'], 'epoch_records': m['epoch_records'],
                      'a1_schedule_audit': m['schedule_audit'], 'amp': amp,
                      'terminal_amp_failure': False, 'numerical_hard_failure': False},
        'firewall': {'scientific_run_access': acc, 'manifest_val_split_accessed': man['val_split_accessed'],
                     'manifest_test_split_accessed': man['test_split_accessed'],
                     'diagnostic_access': fw['access_log'], 'val_access': 0, 'test_access': 0},
        'checkpoints': {'ema_candidates': [{k: r[k] for k in ('epoch', 'global_step', 'sha256', 'file_size_bytes',
                                                              'path', 'selected_for_final')} for r in ema],
                        'selection_performed': False,
                        'note': 'EMA candidates 10..13 are training artifacts of a collapsed run; no VAL selection '
                                'was or will be run on them.'},
        'recovery_states': {
            'final_mid_epoch_after_stop': {'path': 'checkpoints/recovery/latest.pt (inside the scientific root)',
                                           'sha256': rf['sha256'], 'size': rf['size'], 'position': rf['position'],
                                           'labels': rf['labels'], 'provenance': rf['provenance'],
                                           'selectable_ema_checkpoint': False},
            'epoch13_end_copy': {'path': EPOCH13_COPY.replace(RUNTIME, '<runtime>'), 'sha256': r13['sha256'],
                                 'size': r13['size'], 'position': r13['position'], 'labels': r13['labels'],
                                 'provenance': r13['provenance'], 'selectable_ema_checkpoint': False,
                                 'note': 'byte-identical copy of the epoch-13-end recovery/latest.pt taken before the '
                                         'stop (the stop overwrote latest.pt); outside the scientific root'},
            'pre_onset_state_available': False,
            'pre_onset_note': 'recovery/latest.pt is overwritten at every epoch end and the first EMA candidate is '
                              'epoch 10, so no state before u2167 survives; the nearest exact pre-onset state is a '
                              'fresh start (u0) at 4466e12 / seed1337 / SCIENTIFIC mode.'},
        'root_integrity': {'pre_stop': PRE_STOP, 'post_stop': POST_STOP, 'now': rl,
                           'closeout_expected_changes': EXPECTED_CLOSEOUT_CHANGES,
                           'closeout_observed_rewritten': changed,
                           'run_root_files': files,
                           'seed42_roots': c['seed42_roots'], 'a1_runs_listing': c['a1_runs_listing']},
        'onset': {
            'grad_spike_update': ONSET['grad_spike_update'], 'signature_sustained_from': ONSET['signature_from'],
            'epoch': sp['epoch'], 'stage': sp['stage'], 'lambda_adv': sp['lambda_adv'], 'scale_hf': sp['scale_hf'],
            'G_grad_norm_before': win[ONSET['grad_spike_update'] - 1]['G_grad_norm'],
            'G_grad_norm_at_spike_pre_clip': sp['G_grad_norm'], 'amp_attempts_at_spike': sp['amp_attempts'],
            'nearest_amp_retry_before': max(u for u in amp['retry_updates'] if u < ONSET['grad_spike_update']),
            'G_grad_norm_max_before_spike': m['G_grad_norm_max_before_spike'],
            'signature': sg,
            'signature_note': f"first at u{sg['first']}; single non-signature update u{ONSET['signature_from'] + 1} "
                              f"(lm {gap['lm']:.4g}, just above the 1e-6 threshold; budget and bg in signature); "
                              f"then uninterrupted u{sg['run_at_end_start']}..u{LAST_COMPLETE_UPDATE} "
                              f"({sg['run_at_end']} updates) through the owner stop",
            'window': m['onset_window'], 'per_epoch': m['per_epoch'],
            'attribution': 'THIS FAILURE CANNOT BE ATTRIBUTED TO THE M7D1-A1 ADVERSARIAL TRANSITION, because it '
                           'begins before adversarial loss is active (lambda_adv = 0 throughout stage 1, u < 5526).',
            'root_cause': 'NOT INFERRED (next milestone: deterministic fresh-from-u1 replay with pre-onset '
                          'snapshots; the seed42 N3 adversarial mechanism is not assumed).'},
        'direct_collapse': {'evidence': EVIDENCE['direct'], 'conclusion': direct['conclusion'],
                            'subset_size_pairs': direct['subset']['size_pairs'], 'subset_rule': direct['subset']['rule'],
                            'findings': findings},
        'confirmations': {'resumed': False, 'metrics_modified': False, 'run_deleted_or_quarantined': False,
                          'val_run': False, 'test_run': False, 'selection_run': False, 'replay_started': False,
                          'fix_implemented': False, 'production_code_modified': False, 'seed_2026_started': False,
                          'b1_b2_b3_started': False, 'bank_or_downstream_started': False, 'n7b_started': False},
        'evidence': EVIDENCE,
    }
    return doc


def report(doc):
    o, f, i = doc['onset'], doc['direct_collapse']['findings'], doc['integrity']
    rs = doc['recovery_states']
    s = f['s1337_a1']
    lines = [
        f"# M7D1-N7A — GPAT-B0 / E08 / seed1337 (run {RUN_ID}): {doc['final_run_status']}",
        '',
        f"Status: **{doc['status']}** (failed gates: {doc['failed_gates'] or 'none'}).",
        '',
        'This run is **not** a completed 60-epoch run, **not** a numerical hard failure and **not** an AMP hard '
        'failure. It is an owner-authorized graceful stop after direct TRAIN-only confirmation of catastrophic '
        'generator identity collapse. No checkpoint of this run is selectable or bank-eligible.',
        '',
        '## Identity',
        f"- launch authority `{LAUNCH_COMMIT}` (pinned launch clone `{LAUNCH_CLONE}`), curriculum `{CURRICULUM_ID}`",
        f"- run root `{RUN_ROOT}`, launched {LAUNCH_UTC}, SCIENTIFIC mode, config `{CONFIG_SHA256}`",
        f"- environment: {doc['environment']['gpu_model']}, torch {doc['environment']['pytorch_version']}, "
        f"CUDA {doc['environment']['cuda_version']}",
        '',
        '## Stop',
        f"- last COMPLETE update **u{LAST_COMPLETE_UPDATE}** (epoch 14, group "
        f"{doc['completion']['last_complete_record']['group']}); last complete epoch **{LAST_COMPLETE_EPOCH}**; "
        f"{LAST_COMPLETE_UPDATE}/66300 planned updates",
        f"- {STOP['signal']} to the main process only at {STOP['sent_utc']}; graceful exit after "
        f"{STOP['exited_after_seconds']} s; shell EXIT_STATUS={STOP['shell_exit_status']}; manifest/summary "
        f"completion_status = `{doc['completion']['manifest_completion_status']}`",
        f"- {STOP['exit_status_interpretation']}",
        '',
        '## Direct collapse confirmation (TRAIN only, 32 pairs)',
        f"- rule: {doc['direct_collapse']['subset_rule']}; production GPATCore forward at 4466e12, CPU fp32, "
        f"s_hf 0.10",
        f"- seed1337 ({len(s['states'])} states: EMA 10–13, recovery live/EMA at u14365): u8 x_hat == u8 target in "
        f"32/32 for every state; median mean |x_hat − x_t| {s['x_hat_minus_target_mean_abs_median_range'][0]:.3g}; "
        f"max {s['x_hat_minus_target_max_abs']:.3g}; A median {s['A_mean_median_range'][0]:.2g}–"
        f"{s['A_mean_median_range'][1]:.2g}; M median {s['M_mean_median_range'][0]:.2g}–"
        f"{s['M_mean_median_range'][1]:.2g}; M < 0.01 on 100% of pixels; HF residual saturated but masked out",
        f"- historical v1.0 seed42 (EMA 10–13): 32/32 exact target copies (same failure mode); A median "
        f"{f['s42_v10']['A_mean_median_range'][0]:.2g}–{f['s42_v10']['A_mean_median_range'][1]:.2g}",
        f"- healthy A1 seed42 (EMA 10–13): 0/32 exact copies; mean |x_hat − x_t| "
        f"{f['s42_a1']['x_hat_minus_target_mean_abs_median_range'][0]:.3g}–"
        f"{f['s42_a1']['x_hat_minus_target_mean_abs_median_range'][1]:.3g}; A median "
        f"{f['s42_a1']['A_mean_median_range'][0]:.3g}–{f['s42_a1']['A_mean_median_range'][1]:.3g}; M median ≈ 1",
        '',
        '## Training-time onset',
        f"- u{o['grad_spike_update']} (epoch {o['epoch']}, stage {o['stage']}, lambda_adv {o['lambda_adv']}): "
        f"G grad norm {o['G_grad_norm_before']:.3g} → {o['G_grad_norm_at_spike_pre_clip']:.3g} (pre-clip), "
        f"amp_attempts {o['amp_attempts_at_spike']} (no AMP event; nearest earlier retry u"
        f"{o['nearest_amp_retry_before']})",
        f"- historical-collapse signature ({o['signature']['definition']}): {o['signature_note']}",
        '- budget pinned near the artifact-min hinge, id/lm/bg collapse rapidly; generator gradient later decays to '
        f"~{doc['onset']['per_epoch']['14']['G_grad_norm_median']:.1g} (epoch-14 median)",
        f"- **{o['attribution']}**",
        f"- root cause: {o['root_cause']}",
        '',
        '## Integrity',
        f"- COMPLETE u1..u{LAST_COMPLETE_UPDATE} contiguous, 0 duplicates, 0 gaps; epoch records 1..13",
        f"- A1 schedule mismatches {i['a1_schedule_audit']['mismatch_total']} over "
        f"{i['a1_schedule_audit']['updates_checked']} updates (fields: {', '.join(i['a1_schedule_audit']['fields'])})",
        f"- non-finite records {len(i['nonfinite_records'])}; AMP: {i['amp']['retry_event_count']} recovered retries "
        f"on {len(i['amp']['retry_updates'])} updates {i['amp']['retry_updates']}, no terminal AMP failure",
        f"- VAL access 0, TEST access 0 (scientific run: {doc['firewall']['scientific_run_access']['train_images']} "
        f"TRAIN image opens; diagnostic: 64 TRAIN images)",
        '',
        '## Recovery states (neither is a selectable EMA checkpoint)',
        f"- A. final mid-epoch after stop: u{rs['final_mid_epoch_after_stop']['position']['global_update']}, epoch "
        f"{rs['final_mid_epoch_after_stop']['position']['epoch']} / next group "
        f"{rs['final_mid_epoch_after_stop']['position']['next_group']}, sha256 "
        f"`{rs['final_mid_epoch_after_stop']['sha256']}`",
        f"- B. epoch-13-end copy `{rs['epoch13_end_copy']['path']}`: u"
        f"{rs['epoch13_end_copy']['position']['global_update']}, sha256 `{rs['epoch13_end_copy']['sha256']}`",
        f"- pre-onset state available: no — {rs['pre_onset_note']}",
        '',
        '## Root integrity',
        f"- seed1337 root listing: pre-stop `{PRE_STOP['listing_digest']}` ({PRE_STOP['file_count']} files) → "
        f"post-stop `{POST_STOP['listing_digest']}` ({POST_STOP['file_count']} files), unchanged since",
        f"- close-out changes: added {EXPECTED_CLOSEOUT_CHANGES['added']}, rewritten "
        f"{EXPECTED_CLOSEOUT_CHANGES['rewritten']}; resolved_config, EMA candidates and logs unchanged",
        *[f"- {k} root `{v['digest_after']}` ({v['file_count']} files) unchanged"
          for k, v in doc['root_integrity']['seed42_roots'].items()],
        '',
        'No resume, replay, fix, VAL, TEST, selection, seed2026, B1/B2/B3, bank or downstream work was started.',
        '',
    ]
    return '\n'.join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('collect')
    b = sub.add_parser('build')
    b.add_argument('--collected', default=None)
    b.add_argument('--raw-direct', default=None, help='scratch n7a_diag.json (first build only)')
    b.add_argument('--repo-root', default=str(Path(globals().get('__file__', 'tools/x')).resolve().parents[1]))
    args = ap.parse_args(argv)
    if args.cmd == 'collect':
        json.dump(collect(), sys.stdout, sort_keys=True)
        return 0
    root = Path(args.repo_root)
    if args.collected:
        (root / EVIDENCE['collected']).write_text(
            json.dumps(json.loads(Path(args.collected).read_text()), indent=1, sort_keys=True) + '\n')
    c = json.loads((root / EVIDENCE['collected']).read_text())
    if args.raw_direct:
        raw_b = Path(args.raw_direct).read_bytes()
        direct = compact_direct(json.loads(raw_b), sha256_file(root / EVIDENCE['diag_tool']), sha256_bytes(raw_b))
        (root / EVIDENCE['direct']).write_text(json.dumps(direct, indent=1, sort_keys=True) + '\n')
    direct = json.loads((root / EVIDENCE['direct']).read_text())
    doc = build(c, direct, root)
    (root / EVIDENCE['closure']).write_text(json.dumps(doc, indent=1, sort_keys=True) + '\n')
    (root / EVIDENCE['report']).write_text(report(doc))
    print(doc['status'], doc['failed_gates'])
    return 0 if doc['status'] == STATUS else 1


if __name__ == '__main__':
    sys.exit(main())
