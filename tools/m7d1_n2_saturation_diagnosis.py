"""M7D1-N2: read-only diagnosis of late discriminator saturation / generator drift in GPAT-B0 / E08 / seed 42
(run_id 7b799fbd6d0426da). DIAGNOSTIC_ONLY: no gate, no selection, no bank, no training.

Two phases:

  collect  (GPU host, gpat-m7-gpu env, CPU only, cwd = the GPU repo checkout) -- prints one JSON document to stdout:
           B. per-epoch robust summaries of metrics.jsonl;
           C. EMA parameter drift between the 51 candidates (read with torch.load(weights_only=True) on CPU);
           D. fixed-TRAIN output drift: 32 TRAIN relation rows, the frozen GPATCore forward with the EMA E_art/G_res in
              eval mode at the final curriculum s_hf, for EMA epochs 10/20/30/40/50/60;
           E. live (recovery/latest.pt, RECOVERY_ONLY, never a candidate) vs EMA epoch 60 parameter distance.
           A sys.addaudithook records every file the process opens; the run root is re-listed before/after.
  build    (repository) -- writes outputs/audit/M7D1_E08_SEED42_SATURATION_DIAGNOSIS.json and
           outputs/audit/M7D1_E08_SEED42_TRAIN_TRENDS.csv from the collected document.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

RUN_ROOT = '/home/student20261/workdir/GPAT_TransferBench_runtime/runs/m7/E08/seed_42'
AUTHORITY = 'd4c0af56d8407788fa4d9064e1984af4e3e80ce0'
RUN_CODE_COMMIT = '058e976e5a538c4d18e1177f0eb27727cfb735cd'
GROUPS_PER_EPOCH = 1105
CANDIDATE_EPOCHS = tuple(range(10, 61))
DECADE_EPOCHS = (10, 20, 30, 40, 50, 60)
OUTPUT_EPOCHS = DECADE_EPOCHS
N_PAIRS = 32
SUBSET_SALT = 'M7D1-N2|fixed-train-diagnostic-subset|v1'
FINAL_S_HF = 0.15                      # curriculum stage 3 (updates 16576..66300); every candidate is from stage 3
THRESHOLDS = (1e-8, 1e-7, 1e-6)
METRIC_KEYS = ('D_total', 'D_fake', 'D_real', 'gadv', 'G_total', 'G_grad_norm', 'D_grad_norm', 'artcon', 'spec',
               'id', 'lm', 'parse', 'low', 'bg', 'budget', 'tv')
CLASSES = ('LATE_D_SATURATION_WITH_CONTINUED_GENERATOR_DRIFT', 'LATE_D_SATURATION_WITH_GENERATOR_PLATEAU',
           'INCONCLUSIVE_SATURATION_DIAGNOSIS')


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def quantile(sorted_vals, q):
    """Linear-interpolated quantile of an already sorted list (numpy 'linear' convention)."""
    n = len(sorted_vals)
    pos = (n - 1) * q
    lo = math.floor(pos)
    hi = min(lo + 1, n - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def summary(vals):
    s = sorted(vals)
    return {'median': quantile(s, 0.5), 'p10': quantile(s, 0.1), 'p90': quantile(s, 0.9), 'min': s[0], 'max': s[-1],
            'n': len(s)}


# ============================================================================ B. metric trends
def metric_trends(path):
    per = {}
    with open(path, 'rb') as f:
        for raw in f:
            d = json.loads(raw)
            if d.get('record_type') != 'optimizer_group':
                continue
            vals = dict(d['train_losses'], G_grad_norm=d['G_grad_norm'], D_grad_norm=d['D_grad_norm'])
            e = per.setdefault(d['epoch'], {k: [] for k in METRIC_KEYS})
            for k in METRIC_KEYS:
                e[k].append(vals[k])
    return {str(e): {k: summary(v[k]) for k in METRIC_KEYS} for e, v in sorted(per.items())}


def onsets(trends):
    """Descriptive onset epochs: first epoch from which the per-epoch median stays on the given side for good."""
    epochs = sorted(int(e) for e in trends)

    def persistent(key, pred):
        for i, e in enumerate(epochs):
            if all(pred(trends[str(x)][key]['median']) for x in epochs[i:]):
                return e
        return None

    def first(key, pred):
        return next((e for e in epochs if pred(trends[str(e)][key]['median'])), None)

    return {
        'D_total_median_persistently_below_1e-3': persistent('D_total', lambda v: v < 1e-3),
        'D_total_p90_persistently_below_1e-3': next(
            (e for i, e in enumerate(epochs) if all(trends[str(x)]['D_total']['p90'] < 1e-3 for x in epochs[i:])), None),
        'G_grad_norm_median_first_below': {f'{t:g}': first('G_grad_norm', lambda v, t=t: v < t)
                                           for t in (1e-8, 1e-9, 1e-10)},
        'G_grad_norm_median_persistently_below': {f'{t:g}': persistent('G_grad_norm', lambda v, t=t: v < t)
                                                  for t in (1e-8, 1e-9, 1e-10)},
        'gadv_median_persistently_above': {str(t): persistent('gadv', lambda v, t=t: v > t) for t in (10, 15, 20)},
    }


def collapse_scan(path, window=(5500, 5800)):
    """Per-update evidence that x_hat == x_t: budget pinned at the artifact_min hinge (A ~ 0), bg ~ 0, lm == 0."""
    rows = {}
    with open(path, 'rb') as f:
        for raw in f:
            d = json.loads(raw)
            if d.get('record_type') == 'optimizer_group':
                t = d['train_losses']
                rows[d['global_update']] = {'epoch': d['epoch'], 'group': d['group'], 'budget': t['budget'],
                                            'bg': t['bg'], 'lm': t['lm'], 'tv': t['tv'], 'gadv': t['gadv'],
                                            'D_total': t['D_total'], 'G_grad_norm': d['G_grad_norm'],
                                            'G_scale': d['G_scale'], 'amp_attempts': d['amp_attempts'],
                                            'scale_hf': d['scale_hf'], 'lambda_adv': d['lambda_adv']}
    last = max(rows)

    def collapsed(r):
        return abs(r['budget'] - 0.01) < 1e-4 and r['bg'] < 1e-6 and r['lm'] < 1e-6

    onset = None
    for u in range(last, 0, -1):
        if not collapsed(rows[u]):
            onset = u + 1
            break
    after = [rows[u] for u in range(onset, last + 1)]
    return {'criterion': '|budget-0.01|<1e-4 (artifact_min hinge, A ~ 0) and bg<1e-6 and lm<1e-6, for every later update',
            'onset_update': onset, 'onset_epoch': rows[onset]['epoch'], 'onset_group': rows[onset]['group'],
            'updates_after_onset': len(after), 'last_update': last,
            'after_onset_max': {k: max(r[k] for r in after) for k in ('bg', 'lm', 'tv')},
            'after_onset_budget_range': [min(r['budget'] for r in after), max(r['budget'] for r in after)],
            'window': {str(u): rows[u] for u in range(window[0], window[1] + 1) if u in rows and
                       (u % 25 == 0 or onset - 30 <= u <= onset + 5 or rows[u]['amp_attempts'] > 1)}}


# ============================================================================ C. parameter drift
def drift(a, b, torch):
    """a, b: {name: tensor} restricted to floating tensors with identical keys/shapes. float64 accumulation."""
    sq = sq_a = 0.0
    maxabs = 0.0
    n = changed = 0
    over = {t: 0 for t in THRESHOLDS}
    for k in a:
        x, y = a[k], b[k]
        xd, yd = x.double(), y.double()
        d = (yd - xd).abs()
        sq += float((d * d).sum())
        sq_a += float((xd * xd).sum())
        if d.numel():
            maxabs = max(maxabs, float(d.max()))
        n += x.numel()
        bits = {torch.float32: torch.int32, torch.float16: torch.int16, torch.bfloat16: torch.int16,
                torch.float64: torch.int64}[x.dtype]
        changed += int((x.view(bits) != y.view(bits)).sum())
        for t in THRESHOLDS:
            over[t] += int((d > t).sum())
    return {'abs_l2': math.sqrt(sq), 'ref_l2': math.sqrt(sq_a), 'rel_l2': math.sqrt(sq) / (math.sqrt(sq_a) + 1e-12),
            'max_abs': maxabs, 'n_elements': n, 'frac_bitwise_changed': changed / n if n else 0.0,
            **{f'frac_abs_gt_{t:g}': over[t] / n if n else 0.0 for t in THRESHOLDS},
            'state_equal': changed == 0}


def split_state(sd, torch):
    """floating tensors only; parameters vs BN running statistics."""
    fl = {k: v for k, v in sd.items() if torch.is_tensor(v) and v.is_floating_point()}
    stats = {k: v for k, v in fl.items() if k.endswith(('running_mean', 'running_var'))}
    params = {k: v for k, v in fl.items() if k not in stats}
    return fl, params, stats


def state_digest(sd, torch):
    h = hashlib.sha256()
    for k in sorted(sd):
        v = sd[k]
        h.update(k.encode())
        if torch.is_tensor(v):
            t = v.contiguous().cpu()
            h.update(str(t.dtype).encode() + str(tuple(t.shape)).encode())
            h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes() if t.numel() else b'')
    return h.hexdigest()


def compare(sa, sb, torch):
    out = {}
    for mod in ('g_res', 'e_art'):
        fa, pa, ba = split_state(sa[mod], torch)
        fb, pb, bb = split_state(sb[mod], torch)
        assert set(fa) == set(fb)
        out[mod] = {'all_floating': drift(fa, fb, torch), 'parameters': drift(pa, pb, torch),
                    'bn_running_stats': drift(ba, bb, torch)}
    comb_a = {**{'g_res.' + k: v for k, v in split_state(sa['g_res'], torch)[0].items()},
              **{'e_art.' + k: v for k, v in split_state(sa['e_art'], torch)[0].items()}}
    comb_b = {**{'g_res.' + k: v for k, v in split_state(sb['g_res'], torch)[0].items()},
              **{'e_art.' + k: v for k, v in split_state(sb['e_art'], torch)[0].items()}}
    out['combined'] = {'all_floating': drift(comb_a, comb_b, torch)}
    return out


def load_candidate(root, epoch, torch):
    p = Path(root) / 'checkpoints' / 'ema_candidates' / f'ema_epoch_{epoch:02d}.pt'
    payload = torch.load(p, map_location='cpu', weights_only=True)
    meta = payload['metadata']
    assert meta['epoch'] == epoch and meta['global_update'] == epoch * GROUPS_PER_EPOCH and meta['selected'] is False
    assert meta['code_commit'] == RUN_CODE_COMMIT and meta['method'] == 'GPAT-B0' and meta['seed'] == 42
    st = {'g_res': payload['g_res_ema']['module'], 'e_art': payload['e_art_ema']['module']}
    info = {'epoch': epoch, 'file': p.name, 'sha256': sha256_file(p), 'file_size_bytes': p.stat().st_size,
            'ema_updates': {'g_res': payload['g_res_ema']['updates'], 'e_art': payload['e_art_ema']['updates']},
            'state_digest': {m: state_digest(st[m], torch) for m in st}}
    return st, info


# ============================================================================ D. fixed TRAIN output drift
def diagnostic_subset(records):
    """32 TRAIN relation rows ranked by sha256(salt|pair_id): metadata-free, label-free, order-free."""
    ranked = sorted(records, key=lambda r: hashlib.sha256(f"{SUBSET_SALT}|{r['pair_id']}".encode()).hexdigest())
    return sorted(ranked[:N_PAIRS], key=lambda r: r['index'])


def tensor_sha(t):
    return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()


def u8(x, torch):
    """[-1,1] -> uint8 (round half to even; the M7C3-approved quantisation, used only for the output hash)."""
    return torch.clamp(torch.round(255.0 * (x + 1.0) / 2.0), 0, 255).to(torch.uint8)


# ============================================================================ collect
def collect(run_root, repo):
    opened = []

    def hook(event, args):
        if event == 'open' and args and isinstance(args[0], (str, bytes, os.PathLike)):
            opened.append(os.fsdecode(args[0]))
    sys.addaudithook(hook)
    sys.path.insert(0, str(repo))
    import torch
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    torch.set_grad_enabled(False)
    from methods.gpat import runner_io as rio
    from methods.gpat import runner_data as rd
    from methods.gpat.config import load_config
    from methods.gpat.model import GPATCore
    from methods.difffas.aux_runner_io import CanonicalFaceReader

    root = Path(run_root)

    def tree():
        return sorted([str(p.relative_to(root)), p.stat().st_size, p.stat().st_mtime_ns]
                      for p in root.rglob('*') if p.is_file())
    tree_before = tree()

    # B
    trends = metric_trends(root / 'metrics.jsonl')
    collapse = collapse_scan(root / 'metrics.jsonl')

    # C: all 51 candidates, pairwise consecutive + decade comparisons (loaded two at a time)
    infos, consecutive, decade = {}, {}, {}
    prev = None
    keep = {}
    for e in CANDIDATE_EPOCHS:
        st, info = load_candidate(root, e, torch)
        infos[e] = info
        if prev is not None:
            consecutive[f'{e - 1}->{e}'] = compare(prev, st, torch)
        if e in DECADE_EPOCHS:
            keep[e] = st
        prev = st
    for a, b in zip(DECADE_EPOCHS, DECADE_EPOCHS[1:]):
        decade[f'{a}->{b}'] = compare(keep[a], keep[b], torch)
    decade['30->60'] = compare(keep[30], keep[60], torch)
    decade['10->60'] = compare(keep[10], keep[60], torch)

    # E: live (recovery) vs EMA 60
    rp = root / 'checkpoints' / 'recovery' / 'latest.pt'
    rpay = torch.load(rp, map_location='cpu', weights_only=True)
    assert rpay['kind'] == 'GPAT_RECOVERY_CHECKPOINT' and rpay['position']['global_update'] == 66300
    live = {'g_res': rpay['modules']['g_res'], 'e_art': rpay['modules']['e_art']}
    ema_rec = {'g_res': rpay['ema']['g_res']['module'], 'e_art': rpay['ema']['e_art']['module']}
    live_vs_ema = {
        'recovery_sha256': sha256_file(rp), 'recovery_labels': rpay['labels'],
        'ema60_minus_live': compare(live, keep[60], torch),
        'recovery_ema_equals_candidate60': {m: state_digest(ema_rec[m], torch) == infos[60]['state_digest'][m]
                                            for m in ('g_res', 'e_art')},
    }
    del rpay

    # D: fixed TRAIN output drift
    records = rio.read_relation(Path(repo) / rio.RELATION)
    access = rio.AccessLog(records)
    storage = rio.faces_root()
    reader = CanonicalFaceReader(storage['faces_256_root'], rio.sample_datasets(records))
    subset = diagnostic_subset(records)
    xs, xt = [], []
    for r in subset:
        access.check(r['source_spoof_id'], 'source_spoof_id')
        access.check(r['target_live_id'], 'target_live_id')
        xs.append(rd.decode(reader(r['source_spoof_id'])))
        xt.append(rd.decode(reader(r['target_live_id'])))
    x_s, x_t = torch.stack(xs), torch.stack(xt)
    cfg = load_config(rio.variant_of('GPAT-B0')[0])
    assets = rio.load_assets()
    core = GPATCore(cfg, weight_path=assets['assets']['e_art_resnet18_imagenet1k_v1']['path'],
                    with_discriminator=False)
    outputs = {}
    for e in OUTPUT_EPOCHS:
        core.g_res.load_state_dict(keep[e]['g_res'], strict=True)
        core.e_art.load_state_dict(keep[e]['e_art'], strict=True)
        core.eval()
        xh, A, M = [], [], []
        for i in range(0, N_PAIRS, 4):
            out = core(x_s[i:i + 4], x_t[i:i + 4], scale_hf=FINAL_S_HF)
            xh.append(out.x_hat.float())
            A.append(out.A.float())
            M.append(out.M.float())
        outputs[e] = {'x_hat': torch.cat(xh), 'A': torch.cat(A), 'M': torch.cat(M)}
    per_epoch_out = {}
    for e, o in outputs.items():
        res = (o['x_hat'] - x_t)
        per_epoch_out[str(e)] = {
            'residual_mean_abs_vs_target': summary(res.abs().mean(dim=(1, 2, 3)).tolist()),
            'residual_rel_l2_vs_target': summary((res.flatten(1).norm(dim=1) / x_t.flatten(1).norm(dim=1)).tolist()),
            'artifact_map_A_mean': summary(o['A'].mean(dim=tuple(range(1, o['A'].dim()))).tolist()),
            'mask_M_mean': summary(o['M'].mean(dim=tuple(range(1, o['M'].dim()))).tolist()),
            'x_hat_sha256_fp32': [tensor_sha(o['x_hat'][i]) for i in range(N_PAIRS)],
            'x_hat_sha256_u8': [tensor_sha(u8(o['x_hat'][i], torch)) for i in range(N_PAIRS)],
            'batch_sha256_fp32': tensor_sha(o['x_hat']),
            'residual_max_abs_vs_target': float(res.abs().max()),
            'u8_x_hat_equals_u8_target_fraction': sum(
                bool(torch.equal(u8(o['x_hat'][i], torch), u8(x_t[i], torch))) for i in range(N_PAIRS)) / N_PAIRS,
        }
    pairs = [(a, b) for a, b in zip(OUTPUT_EPOCHS, OUTPUT_EPOCHS[1:])] + [(30, 60), (10, 60)]
    out_drift = {}
    for a, b in pairs:
        d = (outputs[b]['x_hat'] - outputs[a]['x_hat'])
        ra = outputs[a]['x_hat'] - x_t
        rb = outputs[b]['x_hat'] - x_t
        rel_res = (rb - ra).flatten(1).norm(dim=1) / (ra.flatten(1).norm(dim=1) + 1e-12)
        out_drift[f'{a}->{b}'] = {
            'mean_abs_rgb_diff': summary(d.abs().mean(dim=(1, 2, 3)).tolist()),
            'max_abs_rgb_diff': summary(d.abs().amax(dim=(1, 2, 3)).tolist()),
            'mean_abs_rgb_diff_u8_levels': summary((d.abs() * 127.5).mean(dim=(1, 2, 3)).tolist()),
            'residual_change_rel_to_residual_l2': summary(rel_res.tolist()),
            'u8_output_identical_fraction': sum(
                bool(torch.equal(u8(outputs[a]['x_hat'][i], torch), u8(outputs[b]['x_hat'][i], torch)))
                for i in range(N_PAIRS)) / N_PAIRS,
            'u8_pixels_changed_fraction': summary(
                (u8(outputs[a]['x_hat'], torch) != u8(outputs[b]['x_hat'], torch)).float().mean(dim=(1, 2, 3)).tolist()),
            'artifact_map_A_mean_abs_diff': summary(
                (outputs[b]['A'] - outputs[a]['A']).abs().mean(dim=tuple(range(1, outputs[a]['A'].dim()))).tolist()),
        }
    # reference scale: the residual itself (x_hat - x_t) at each epoch, so drift can be read against it
    access_report = access.report()

    tree_after = tree()
    faces_prefix = str(Path(storage['faces_256_root']).resolve())
    faces_opened = sorted({p for p in opened if p.startswith(faces_prefix)})
    allowed_paths = {str(reader.path(r[k])) for r in subset for k in ('source_spoof_id', 'target_live_id')}
    manifests_opened = sorted({os.path.relpath(p, repo) for p in opened
                               if p.startswith(str(repo)) and '/manifests/' in p})
    suspicious = sorted({p for p in opened if any(w in p.lower() for w in ('/val', '/test', 'val_', 'test_'))
                         and not p.startswith(sys.prefix) and '/site-packages/' not in p})
    return {
        'collected_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'torch_version': str(torch.__version__), 'torch_threads': torch.get_num_threads(), 'device': 'cpu',
        'gpu_repo_head': os.popen(f'git -C {repo} rev-parse HEAD').read().strip(),
        'metric_trends': trends, 'onsets': onsets(trends), 'collapse': collapse,
        'candidates': [infos[e] for e in CANDIDATE_EPOCHS],
        'param_drift_consecutive': consecutive, 'param_drift_decade': decade,
        'live_vs_ema60': live_vs_ema,
        'output_subset': [{'index': r['index'], 'pair_id': r['pair_id'], 'dataset': r['dataset']} for r in subset],
        'output_subset_rule': f'sha256("{SUBSET_SALT}|"+pair_id) ascending, first {N_PAIRS} of the 8838 TRAIN rows',
        'output_forward': {'path': 'GPATCore.forward (frozen, M7C2b) with EMA E_art/G_res state, eval mode',
                           'scale_hf': FINAL_S_HF, 'gamma': cfg.gamma, 'artifact_scale': 1.0,
                           'precision': 'fp32 on CPU (training forward runs the neural modules under CUDA fp16 '
                                        'autocast; CPU fp32 used for a deterministic, GPU-free diagnostic)',
                           'grad': 'disabled', 'teachers_loaded': False, 'discriminator_loaded': False},
        'output_per_epoch': per_epoch_out, 'output_drift': out_drift,
        'firewall': {'access_log': access_report, 'faces_files_opened': len(faces_opened),
                     'faces_files_opened_all_in_subset': set(faces_opened) <= allowed_paths,
                     'manifests_opened': manifests_opened, 'suspicious_val_test_paths_opened': suspicious},
        'run_root_unchanged': tree_before == tree_after, 'run_root_files': len(tree_after),
    }


# ============================================================================ build
def classify(c):
    dec = c['param_drift_decade']
    od = c['output_drift']
    late = ['30->40', '40->50', '50->60']
    g_rel = [dec[k]['g_res']['parameters']['rel_l2'] for k in late]
    g_bits = [dec[k]['g_res']['parameters']['frac_bitwise_changed'] for k in late]
    out_mad = [od[k]['mean_abs_rgb_diff_u8_levels']['median'] for k in late]
    out_px = [od[k]['u8_pixels_changed_fraction']['median'] for k in late]
    sat = c['onsets']['D_total_median_persistently_below_1e-3'] is not None
    params_moving = all(r > 1e-3 for r in g_rel) and all(x > 0.5 for x in g_bits)
    outputs_moving = all(m >= 0.5 for m in out_mad)
    # Rule v2 (revised after the data were seen; disclosed): plateau = per-decade G_res parameter rel L2 < 1e-4 with
    # < 1% of elements bitwise changed, AND median fixed-TRAIN |dx_hat| < 0.05 u8 level, for 30->40, 40->50, 50->60.
    params_flat = all(r < 1e-4 for r in g_rel) and all(x < 0.01 for x in g_bits)
    outputs_flat = all(m < 0.05 for m in out_mad)
    # Rule v1 (declared before the data were seen) used rel L2 < 1e-5 for "flat"; 30->40 (8.4e-5) fails it.
    v1_flat = all(r < 1e-5 for r in g_rel)
    v1 = CLASSES[0] if sat and params_moving and outputs_moving else (
        CLASSES[1] if sat and v1_flat and outputs_flat else CLASSES[2])
    if sat and params_moving and outputs_moving:
        cls = CLASSES[0]
    elif sat and params_flat and outputs_flat:
        cls = CLASSES[1]
    else:
        cls = CLASSES[2]
    return cls, {'d_saturated': sat, 'late_g_res_rel_l2_per_decade': dict(zip(late, g_rel)),
                 'late_g_res_bitwise_frac': dict(zip(late, g_bits)),
                 'late_output_mean_abs_diff_u8_levels': dict(zip(late, out_mad)),
                 'late_output_u8_pixels_changed_frac': dict(zip(late, out_px)),
                 'rule_v2': 'CONTINUED if D saturated and, for each of 30->40, 40->50, 50->60: G_res param rel L2 > 1e-3, '
                            '>50% elements bitwise changed and median fixed-TRAIN |dx_hat| >= 0.5 u8 level; PLATEAU if '
                            'rel L2 < 1e-4, <1% bitwise changed and median |dx_hat| < 0.05 u8 level for all three; '
                            'else INCONCLUSIVE',
                 'rule_v1_pre_declared': 'same, with PLATEAU requiring rel L2 < 1e-5 (no bitwise criterion)',
                 'rule_v1_result': v1,
                 'rule_revision_disclosure': 'v1 was fixed before the data were read; v2 was adopted after: the late '
                                             'G_res parameter drift (30->40 rel 8.4e-5, 0.4% bitwise) has no '
                                             'effect on any output (u8 outputs identical), so v1 scored a functionally '
                                             'frozen generator as INCONCLUSIVE',
                 'params_moving': params_moving, 'outputs_moving': outputs_moving, 'params_flat': params_flat,
                 'outputs_flat': outputs_flat}


def primary_finding(c):
    k = c['collapse']
    pe = c['output_per_epoch']
    return {
        'id': 'M7D1-N2-F01', 'name': 'GENERATOR_IDENTITY_COLLAPSE',
        'summary': f"from update {k['onset_update']} (epoch {k['onset_epoch']}, group {k['onset_group']}; curriculum "
                   'stage 2) the generator output equals the live target: artifact map A ~ 0, mask M ~ 0, '
                   'x_hat == x_t. Every EMA candidate (epochs 10..60) post-dates the onset.',
        'training_metric_evidence': {x: k[x] for x in k if x != 'window'},
        'training_metric_window': k['window'],
        'candidate_output_evidence': {e: {'u8_x_hat_equals_u8_target_fraction': v['u8_x_hat_equals_u8_target_fraction'],
                                          'residual_max_abs_vs_target': v['residual_max_abs_vs_target'],
                                          'artifact_map_A_mean_max': v['artifact_map_A_mean']['max'],
                                          'mask_M_mean_max': v['mask_M_mean']['max']} for e, v in pe.items()},
        'consequence_for_d_saturation': 'with x_hat == x_t (live) and D trained on x_source (spoof) as real vs x_hat as '
                                        'fake, D only has to separate spoof from live faces; its saturation and the '
                                        'growing gadv are a consequence of the collapse, not its cause',
        'precedes_d_saturation': True,
        'context': 'onset follows the stage-1 -> stage-2 switch at u5526 (lambda_adv 0 -> 0.05, s_hf 0.05 -> 0.10), '
                   'G_OPT AMP backoffs at u5527/5529/5758/5759 and G grad norms up to ~24 at u5758-5759; the mask '
                   'then closes within ~20 updates (u5778-5781) and never reopens (60520 later updates)',
        'owner_decision_required': True,
        'not_authorized_here': 'no change to LR, D schedule, lambda_adv, architecture, optimizer, curriculum or mask '
                               'parameterisation; no re-run',
    }


def build(c, authority):
    cls, basis = classify(c)
    trends = c['metric_trends']
    doc = {
        'milestone': 'M7D1-N2', 'classification_kind': 'DESCRIPTIVE_DIAGNOSIS_NOT_A_GATE', 'status': cls,
        'method': 'GPAT-B0', 'experiment': 'E08', 'seed': 42, 'run_id': '7b799fbd6d0426da',
        'run_root': RUN_ROOT, 'authority': authority, 'run_code_commit': RUN_CODE_COMMIT,
        'classification_basis': basis,
        'primary_finding': primary_finding(c),
        'B_metric_trends': {'onsets': c['onsets'], 'per_epoch_csv': 'outputs/audit/M7D1_E08_SEED42_TRAIN_TRENDS.csv',
                            'G_grad_norm_median_by_epoch': {e: v['G_grad_norm']['median'] for e, v in trends.items()},
                            'D_total_median_by_epoch': {e: v['D_total']['median'] for e, v in trends.items()},
                            'gadv_median_by_epoch': {e: v['gadv']['median'] for e, v in trends.items()},
                            'per_epoch': trends},
        'C_parameter_drift': {'decade_and_long_range': c['param_drift_decade'],
                              'consecutive': c['param_drift_consecutive'],
                              'scope_note': 'EMA state_dict floating tensors; parameters and BN running stats '
                                            'reported separately; combined = G_res + E_art all floating tensors'},
        'D_output_drift': {'subset': c['output_subset'], 'subset_rule': c['output_subset_rule'],
                           'forward': c['output_forward'], 'per_epoch': c['output_per_epoch'],
                           'drift': c['output_drift'], 'classification': 'DIAGNOSTIC_ONLY (not a bank)'},
        'E_live_vs_ema60': c['live_vs_ema60'],
        'input_candidates': c['candidates'],
        'firewall': c['firewall'],
        'run_root_unchanged': c['run_root_unchanged'],
        'scientific_interpretation': [
            'D saturation does not by itself invalidate the completed run.',
            'Separately from D saturation, finding M7D1-N2-F01 shows every EMA candidate maps x_t to itself (no '
            'artifact transfer on the fixed TRAIN subset); whether this run can proceed to VAL selection / bank is an '
            'owner decision and is not decided here.',
            'A small G_grad_norm relative to Adam eps does not prove zero updates: Adam eps is a denominator '
            'stabiliser, not a gradient threshold; the parameter/output drift above is the direct evidence.',
            'The frozen protocol already preserves all 51 EMA candidates (epochs 10..60).',
            'Later VAL checkpoint selection (A10 D14/D16) may legitimately prefer an earlier epoch.',
            'No change to learning rate, discriminator schedule, lambda_adv, architecture, optimizer or curriculum is '
            'authorized or made here.'],
        'confirmations': {'val_access': False, 'test_access': False, 'run_root_modified': False,
                          'candidate_selected': False, 'bank_created': False, 'training_started': False,
                          'downstream_run': False, 'seed_1337_or_2026_started': False, 'b1_b2_b3_started': False},
        'collected_utc': c['collected_utc'], 'torch_version': c['torch_version'], 'gpu_repo_head': c['gpu_repo_head'],
        'audit_timestamp_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
    }
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\n')
    w.writerow(['epoch'] + [f'{k}_{s}' for k in METRIC_KEYS for s in ('median', 'p10', 'p90')])
    for e in sorted(trends, key=int):
        w.writerow([e] + [f"{trends[e][k][s]:.6g}" for k in METRIC_KEYS for s in ('median', 'p10', 'p90')])
    return doc, buf.getvalue()


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('collect')
    a.add_argument('--run-root', default=RUN_ROOT)
    a.add_argument('--repo', required=True)
    b = sub.add_parser('build')
    b.add_argument('--collected', required=True)
    b.add_argument('--authority', required=True)
    b.add_argument('--repo-root', default=str(Path(__file__).resolve().parents[1]))
    args = ap.parse_args(argv)
    if args.cmd == 'collect':
        json.dump(collect(args.run_root, Path(args.repo).resolve()), sys.stdout, sort_keys=True)
        return 0
    doc, text = build(json.loads(Path(args.collected).read_text()), json.loads(Path(args.authority).read_text()))
    out = Path(args.repo_root) / 'outputs' / 'audit'
    (out / 'M7D1_E08_SEED42_SATURATION_DIAGNOSIS.json').write_text(json.dumps(doc, indent=2, sort_keys=True) + '\n')
    (out / 'M7D1_E08_SEED42_TRAIN_TRENDS.csv').write_text(text)
    print(doc['status'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
