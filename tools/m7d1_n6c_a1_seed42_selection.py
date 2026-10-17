"""M7D1-N6C: frozen VAL-only checkpoint selection for M7D1-A1 GPAT-B0 / E08 / seed 42 (run 92e91ddb78f1e242).

Input: the raw per-candidate VAL metrics produced on the GPU host by tools/m7d1_n6c_a1_seed42_val_eval.py (collected
read-only into outputs/audit/M7D1_A1_E08_SEED42_N6C_COLLECTED.json). No model, no image and no split file is opened
here; TEST is never involved.

Selection rule (frozen spec 10.6; A10 D14 IMPLEMENTATION_CLARIFICATION, configs/amendments/gpat_a10_m7_contract_resolution
.yaml D14; weights also pinned by tools/m7a_gpat_contract_resolution_preflight.py):
  candidates  EMA epochs 10..60 inclusive, N = 51 (recovery/latest.pt is never a candidate)
  orientation higher is better for ID, Dice, ArtSim; NME and LFErr are ranked as -NME, -LFErr
  rank        ascending 1..N on the oriented value; ties get the average rank
  percentile  p = (rank - 1) / (N - 1)
  score       0.25 p(ID) + 0.15 p(Dice) + 0.25 p(ArtSim) + 0.20 p(-NME) + 0.15 p(-LFErr)
  winner      maximum score; tie -> higher raw ArtSim; then the earlier epoch
Ranks, percentiles and scores are computed with exact rationals (fractions.Fraction; the weights are the decimal
literals above), so score ties are exact rather than float-rounding artifacts; floats are reported alongside.

Phases:  build  (repository) -- gates + outputs/audit/M7D1_A1_E08_SEED42_{VAL_METRICS.csv,SELECTION.json,SELECTION.md,
                                 N6C_EVIDENCE_SHA256.txt}
"""
import argparse
import csv
import hashlib
import io
import json
import math
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MILESTONE = 'M7D1-N6C'
STATUS_OK = 'M7D1_N6C_A1_SEED42_SELECTED'
STATUS_BLOCKED = 'M7D1_N6C_SELECTION_BLOCKED'
RUN_ID = '92e91ddb78f1e242'
LAUNCH_COMMIT = '4466e1258c24e4a5d448dd5b30e8d8768dfe7174'
CLOSURE_COMMIT = 'ffffa9bd3a2f7d94f0d7fa228764dc425d2ff563'
CURRICULUM_ID = 'GPAT-TransferBench-v1.0+M7D1-A1'
RUN_ROOT = '/home/student20261/workdir/GPAT_TransferBench_runtime/runs/m7_a1/E08/seed_42'
RUN_ROOT_LISTING_DIGEST = '9b1a85eaeaf635450807d43a7e19e97d447d6a18f2585c5fc7720b8efadfadf3'
EVAL_ROOT = '/home/student20261/workdir/GPAT_TransferBench_runtime/eval/m7_a1/E08/seed_42/N6C'
EMA_MANIFEST = 'outputs/audit/M7D1_A1_E08_SEED42_EMA_MANIFEST.txt'
EMA_MANIFEST_DIGEST = '51f9648ebefa0d41f85277ce646280d33c02911b9de04171ddd61b8e9ba79406'
METRICS_SHA256 = 'fa05cfb6151ad932d52dc6c810c8d86711ee748e228ec376632fdf9b1f71717e'
VAL_PAIRS = 'manifests/val_pairs_v1.parquet'
VAL_PAIRS_SHA256 = '84d124919a52cd8d84a89766f464a4dcde1aeaa7791218f6356813a40904f872'
SPLIT_MANIFEST_SHA256 = 'fb9aeb369a124fc96ba855ef2ce269236c4a743fe960e73ab739412c9cb5092d'
VAL_ROWS = 1905
EVAL_TOOL = 'tools/m7d1_n6c_a1_seed42_val_eval.py'
EPOCHS = tuple(range(10, 61))
N = len(EPOCHS)
METRICS = ('ID', 'Dice', 'ArtSim', 'NME', 'LFErr')
ORIENTATION = {'ID': 1, 'Dice': 1, 'ArtSim': 1, 'NME': -1, 'LFErr': -1}
RANK_NAMES = {'ID': 'ID', 'Dice': 'Dice', 'ArtSim': 'ArtSim', 'NME': 'neg_NME', 'LFErr': 'neg_LFErr'}
WEIGHTS = {'ID': Fraction('0.25'), 'Dice': Fraction('0.15'), 'ArtSim': Fraction('0.25'), 'NME': Fraction('0.20'),
           'LFErr': Fraction('0.15')}
TOP_K = 5
EVIDENCE = {'collected': 'outputs/audit/M7D1_A1_E08_SEED42_N6C_COLLECTED.json',
            'csv': 'outputs/audit/M7D1_A1_E08_SEED42_VAL_METRICS.csv',
            'selection': 'outputs/audit/M7D1_A1_E08_SEED42_SELECTION.json',
            'report': 'outputs/audit/M7D1_A1_E08_SEED42_SELECTION.md',
            'sha256': 'outputs/audit/M7D1_A1_E08_SEED42_N6C_EVIDENCE_SHA256.txt'}
CSV_COLUMNS = ('epoch', 'checkpoint', 'checkpoint_sha256', 'checkpoint_size_bytes', 'global_update', 'ID', 'Dice',
               'ArtSim', 'NME', 'LFErr', 'evaluated_val_pairs', 'status', 'warnings', 'wall_seconds',
               'rank_ID', 'rank_Dice', 'rank_ArtSim', 'rank_neg_NME', 'rank_neg_LFErr',
               'p_ID', 'p_Dice', 'p_ArtSim', 'p_neg_NME', 'p_neg_LFErr', 'selection_score',
               'selection_score_exact', 'position')
OWNER_DECISIONS = {
    'eval_s_hf': {'value': 0.15, 'class': 'OWNER_EVALUATION_CLARIFICATION', 'date': '2026-10-17',
                  'basis': 'spec 9.3 / gpat_b0.yaml delta_scale_hf; the frozen architecture value for every candidate '
                           '(the A1 training curriculum used 0.10 for updates <= 16575)'},
    'generator_precision': {'value': 'R06_FP16_AUTOCAST_E_ART_G_RES', 'class': 'OWNER_EVALUATION_CLARIFICATION',
                            'date': '2026-10-17',
                            'basis': 'M7B R-06: fp16 autocast for E_art/G_res as in training; fp32 DWT/IDWT/'
                                     'composition/x_hat/A and fp32 frozen evaluators'}}


class SelectionError(RuntimeError):
    pass


def require(ok, msg):
    if not ok:
        raise SelectionError(f'{MILESTONE}: {msg}')


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path):
    return sha256_bytes(Path(path).read_bytes())


# ============================================================================ frozen rule (pure)
def average_ranks(values):
    """1-based ascending ranks; tied values share the average of the ranks they occupy (A10 D14)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [None] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        r = Fraction(i + 1 + j + 1, 2)
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    return ranks


def percentile(rank, n):
    return (Fraction(rank) - 1) / (n - 1)


def validate_rows(rows):
    require(len(rows) == N, f'exactly {N} candidate rows (got {len(rows)})')
    epochs = [r['epoch'] for r in rows]
    require(sorted(epochs) == list(EPOCHS) and len(set(epochs)) == N, 'one row per epoch 10..60, no duplicate')
    for r in rows:
        require('recovery' not in str(r.get('checkpoint', '')) and not str(r.get('checkpoint', '')).endswith(
            'latest.pt'), 'recovery/latest.pt is never a selection candidate')
        for m in METRICS:
            v = r[m]
            require(isinstance(v, float) and math.isfinite(v), f'epoch {r["epoch"]} {m} must be a finite float')
    counts = {r.get('evaluated_val_pairs') for r in rows}
    require(len(counts) == 1, f'identical VAL sample count for all candidates (got {sorted(counts)})')


def rank_candidates(rows):
    """rows: [{'epoch', 'ID', 'Dice', 'ArtSim', 'NME', 'LFErr', ...}] -> rows ordered best-first with ranks,
    percentiles and exact scores; plus the winner / tie-break record."""
    validate_rows(rows)
    rows = sorted(rows, key=lambda r: r['epoch'])
    out = [dict(r) for r in rows]
    for m in METRICS:
        oriented = [ORIENTATION[m] * r[m] for r in rows]
        for r, rank in zip(out, average_ranks(oriented)):
            r[f'rank_{RANK_NAMES[m]}'] = rank
            r[f'p_{RANK_NAMES[m]}'] = percentile(rank, N)
    for r in out:
        r['score_exact'] = sum((WEIGHTS[m] * r[f'p_{RANK_NAMES[m]}'] for m in METRICS), Fraction(0))
    ordered = sorted(out, key=lambda r: (-r['score_exact'], -r['ArtSim'], r['epoch']))
    for pos, r in enumerate(ordered, 1):
        r['position'] = pos
    best = ordered[0]['score_exact']
    tied = [r for r in ordered if r['score_exact'] == best]
    tie = {'max_score_shared_by_epochs': sorted(r['epoch'] for r in tied), 'tie_break_required': len(tied) > 1,
           'tie_break_stage': None}
    if len(tied) > 1:
        best_art = max(r['ArtSim'] for r in tied)
        art_tied = [r for r in tied if r['ArtSim'] == best_art]
        tie['tie_break_stage'] = 'higher_raw_ArtSim' if len(art_tied) == 1 else 'earlier_epoch'
        tie['after_ArtSim_epochs'] = sorted(r['epoch'] for r in art_tied)
    return ordered, tie


def frac_str(f):
    return f'{f.numerator}/{f.denominator}'


# ============================================================================ evidence (repository)
def load_ema_manifest():
    raw = (ROOT / EMA_MANIFEST).read_bytes()
    require(sha256_bytes(raw) == EMA_MANIFEST_DIGEST, 'N6B EMA manifest digest')
    rows = {}
    for line in raw.decode('utf-8').splitlines():
        if not line.startswith('#'):
            e, size, sha, rel = line.split(' ')
            rows[int(e)] = {'size': int(size), 'sha256': sha, 'rel': rel}
    require(sorted(rows) == list(EPOCHS), 'manifest epochs 10..60')
    return rows


def log_fields(text):
    out = {}
    for line in text.splitlines():
        for key in ('repo_head', 'tool_sha256', 'run_root_listing_pre', 'run_root_listing_post', 'EXIT_STATUS',
                    'dirty_lines', 'phase', 'end_utc'):
            if line.startswith(key + '='):
                out[key] = line.split('=', 1)[1].split(' ')[0]
            elif f' {key}=' in line:
                out[key] = line.split(f' {key}=', 1)[1].split(' ')[0]
    return out


def candidate_rows(collected, manifest):
    full = collected['eval']['full']
    require(sorted(int(e) for e in full) == list(EPOCHS), 'collected full results for epochs 10..60')
    rows = []
    for e in EPOCHS:
        res = full[str(e)]
        c = res['candidate']
        m = manifest[e]
        require(c['epoch'] == e and c['sha256'] == m['sha256'] and c['size_bytes'] == m['size'] and
                c['rel'] == m['rel'], f'epoch {e}: evaluated candidate equals the N6B EMA manifest entry')
        require(res['status'] == 'COMPLETE' and res['label'] == 'VAL_ONLY_N6C_EVALUATION', f'epoch {e} status')
        rows.append({'epoch': e, 'checkpoint': c['rel'], 'checkpoint_sha256': c['sha256'],
                     'checkpoint_size_bytes': c['size_bytes'], 'global_update': c['global_update'],
                     **{k: float(res['metrics'][k]) for k in METRICS}, 'evaluated_val_pairs': res['evaluated_pairs'],
                     'pair_ids_sha256': res['pair_ids_sha256'], 'status': res['status'],
                     'warnings': ';'.join(res['warnings']), 'wall_seconds': res['wall_seconds'],
                     'config_digest': res['config_digest'], 'tool_sha256': res['tool_sha256']})
    require(len({r['config_digest'] for r in rows}) == 1, 'one evaluation configuration for all candidates')
    require(len({r['pair_ids_sha256'] for r in rows}) == 1, 'identical VAL pair list / order for all candidates')
    require(len({r['tool_sha256'] for r in rows}) == 1, 'one evaluator file for all candidates')
    require(all(r['evaluated_val_pairs'] == VAL_ROWS for r in rows), f'all {VAL_ROWS} VAL pairs evaluated')
    return rows


def csv_text(ordered):
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\n')
    w.writerow(CSV_COLUMNS)
    for r in sorted(ordered, key=lambda r: r['epoch']):
        w.writerow([r['epoch'], r['checkpoint'], r['checkpoint_sha256'], r['checkpoint_size_bytes'],
                    r['global_update'], *[repr(r[m]) for m in METRICS], r['evaluated_val_pairs'], r['status'],
                    r['warnings'], r['wall_seconds'],
                    *[str(float(r[f'rank_{RANK_NAMES[m]}'])) for m in METRICS],
                    *[repr(float(r[f'p_{RANK_NAMES[m]}'])) for m in METRICS],
                    repr(float(r['score_exact'])), frac_str(r['score_exact']), r['position']])
    return buf.getvalue()


def read_csv_rows(text):
    """Raw metric rows of the committed CSV (what an independent re-ranking starts from)."""
    rows = []
    for rec in csv.DictReader(io.StringIO(text)):
        rows.append({'epoch': int(rec['epoch']), 'checkpoint': rec['checkpoint'],
                     'checkpoint_sha256': rec['checkpoint_sha256'], 'evaluated_val_pairs': int(rec['evaluated_val_pairs']),
                     **{m: float(rec[m]) for m in METRICS}})
    return rows


def public(r):
    return {'epoch': r['epoch'], 'checkpoint': r['checkpoint'], 'checkpoint_path': f"{RUN_ROOT}/{r['checkpoint']}",
            'checkpoint_sha256': r['checkpoint_sha256'], 'raw_metrics': {m: r[m] for m in METRICS},
            'ranks': {RANK_NAMES[m]: float(r[f'rank_{RANK_NAMES[m]}']) for m in METRICS},
            'percentiles': {RANK_NAMES[m]: float(r[f'p_{RANK_NAMES[m]}']) for m in METRICS},
            'selection_score': float(r['score_exact']), 'selection_score_exact': frac_str(r['score_exact']),
            'position': r['position']}


def metric_ranges(rows):
    out = {}
    for m in METRICS:
        vals = [r[m] for r in rows]
        lo, hi = min(vals), max(vals)
        out[m] = {'min': lo, 'max': hi, 'argmin_epoch': min(rows, key=lambda r: (r[m], r['epoch']))['epoch'],
                  'argmax_epoch': min(rows, key=lambda r: (-r[m], r['epoch']))['epoch'],
                  'distinct_values': len(set(vals))}
    return out


def build(collected_path):
    raw = Path(collected_path).read_bytes()
    collected = json.loads(raw)
    ev = collected['eval']
    manifest = load_ema_manifest()
    rows = candidate_rows(collected, manifest)
    ordered, tie = rank_candidates(rows)
    winner, runner = ordered[0], ordered[1]
    csv_body = csv_text(ordered)
    # independent re-derivation from the CSV text itself
    re_ordered, re_tie = rank_candidates(read_csv_rows(csv_body))
    require(re_ordered[0]['epoch'] == winner['epoch'] and re_tie == tie, 'CSV re-ranking reproduces the winner')

    sessions = ev['sessions']
    require(len(sessions) >= 1, 'full-phase session record')
    fws = [s['firewall'] for s in sessions] + ([ev['smoke']['firewall']] if ev['smoke'] else []) + \
          [s['firewall'] for s in ev['sanity']]
    for fw in fws:
        require(not fw['denied'] and fw['non_val_image_opens'] == 0 and fw['other_manifest_opens'] == 0 and
                fw['run_root_write_opens'] == 0 and fw['test_images_accessed'] == 0 and
                fw['test_metadata_accessed'] == 0, 'firewall: VAL only, no TEST, run root read-only')
    logs = {name: log_fields(text) for name, text in sorted(collected['logs'].items())}
    for name, f in logs.items():
        if f.get('EXIT_STATUS') == '0':
            require(f['run_root_listing_pre'] == RUN_ROOT_LISTING_DIGEST == f['run_root_listing_post'],
                    f'{name}: scientific run root listing unchanged')
            require(f['repo_head'] == CLOSURE_COMMIT and f['dirty_lines'] == '0', f'{name}: GPU repo at authority')
    tool_sha = sha256_file(ROOT / EVAL_TOOL)
    require({r['tool_sha256'] for r in rows} == {tool_sha}, 'committed evaluator == the evaluator that ran')

    sanity = [s for s in ev['sanity'] if s['result']['candidate']['epoch'] == winner['epoch']]
    require(len(sanity) == 1, 'one post-selection sanity record for the winner')
    sanity = sanity[0]
    repro = sanity['reproducibility']
    require(all(v['bitwise_equal_mean'] and v['per_pair_bitwise_equal'] for v in repro.values()),
            'winner re-evaluation reproduces the full-phase metrics bitwise')
    cfg = sessions[-1]['config']
    require(cfg['eval_s_hf'] == OWNER_DECISIONS['eval_s_hf']['value'] and
            cfg['generator_precision'] == OWNER_DECISIONS['generator_precision']['value'], 'owner decisions applied')

    top = [public(r) for r in ordered[:TOP_K]]
    selection = {
        'milestone': MILESTONE, 'status': STATUS_OK, 'label': 'BEST_EMA_SEED42',
        'authority_commit': CLOSURE_COMMIT, 'launch_authority_commit': LAUNCH_COMMIT,
        'run': {'method': 'GPAT-B0', 'experiment': 'E08', 'seed': 42, 'run_id': RUN_ID, 'curriculum_id': CURRICULUM_ID,
                'run_root': RUN_ROOT, 'run_root_listing_digest': RUN_ROOT_LISTING_DIGEST,
                'metrics_jsonl_sha256': METRICS_SHA256, 'complete_updates': 66300, 'epochs': 60},
        'ema_candidates': {'count': N, 'epochs': [EPOCHS[0], EPOCHS[-1]], 'manifest': EMA_MANIFEST,
                           'manifest_digest': EMA_MANIFEST_DIGEST, 'recovery_latest_selectable': False},
        'val_split': {'manifest': VAL_PAIRS, 'sha256': VAL_PAIRS_SHA256, 'rows': VAL_ROWS,
                      'split_manifest_sha256': SPLIT_MANIFEST_SHA256, 'split_column': 'VAL for every row',
                      'evaluated_pairs_per_candidate': VAL_ROWS, 'pair_ids_sha256': rows[0]['pair_ids_sha256'],
                      'authority': 'frozen spec 10.6; A10 manifest binding'},
        'evaluators': {'config': cfg, 'tool': EVAL_TOOL, 'tool_sha256': tool_sha},
        'metric_definitions': {
            'ID': 'cos(AdaFace(x_t), AdaFace(x_hat)); runner.Teachers.adaface (adaface_input INTER_AREA matrix, BGR); '
                  'spec 13',
            'Dice': 'A10 D09: FaceXFormer parsing argmax masks of x_t and x_hat, classes 1..10, a class omitted for a '
                    'pair only if absent from both masks, mean over valid classes',
            'ArtSim': 'A10 D13: cos of ArtifactProbe v1 (b5ace6c2...) L2-normalized penultimate embeddings of x_s and '
                      'x_hat; teacher_preprocess.artifact_probe_input_float (INTER_AREA 224, HP 9x9 sigma 1.5)',
            'NME': 'spec 13 + A10 D14: mean over 68 landmarks of the L2 distance between FaceXFormer landmarks of x_hat '
                   'and x_t, divided by sqrt(w*h) of the bounding box of x_t\'s 68 landmarks (native coordinates)',
            'LFErr': 'M7C2a N-08: ||LL_syn_internal - LL_t||_1 / (||LL_t||_1 + 1e-8) per pair (exactly 0 at gamma=0)',
            'aggregation': 'per pair, then the mean over all 1905 VAL pairs (math.fsum)',
            'representation': 'x_hat_float pre-export, no clamp, each evaluator through its own adapter (M7B R-03)',
            'evaluator_precision': 'fp32 frozen evaluators (M7B R-06)'},
        'owner_decisions_n6c': OWNER_DECISIONS,
        'selection_rule': {'authority': 'frozen spec 10.6; A10 D14 (gpat_a10_m7_contract_resolution.yaml)',
                           'orientation': {m: ('higher' if ORIENTATION[m] > 0 else 'lower') + '_is_better'
                                           for m in METRICS},
                           'rank': 'ascending 1..N on the oriented value; average rank for ties',
                           'percentile': 'p = (rank - 1) / (N - 1)', 'N': N,
                           'weights': {RANK_NAMES[m]: float(WEIGHTS[m]) for m in METRICS},
                           'weights_sum': float(sum(WEIGHTS.values())),
                           'tie_break': ['higher_raw_ArtSim', 'earlier_epoch'],
                           'arithmetic': 'exact rationals (fractions.Fraction) for ranks, percentiles and scores',
                           'test_used': False},
        'metric_ranges': metric_ranges(rows),
        'winner': public(winner), 'runner_up': public(runner),
        'margin_to_runner_up': float(winner['score_exact'] - runner['score_exact']),
        'margin_to_runner_up_exact': frac_str(winner['score_exact'] - runner['score_exact']),
        'tie_break': tie, 'top_5': top,
        'reproducibility': {'csv_reranking_same_winner': True, 'winner_full_val_reevaluation': repro,
                            'nondeterminism_observed': False},
        'post_selection_sanity': {'label': 'POST_SELECTION_SANITY_ONLY', 'epoch': winner['epoch'],
                                  'diagnostics_summary': sanity['diagnostics_summary'],
                                  'affects_selection': False},
        'smoke': {'label': 'SMOKE_NOT_USED_FOR_SELECTION', 'epoch': ev['smoke']['result']['candidate']['epoch'],
                  'pairs': ev['smoke']['result']['evaluated_pairs'], 'metrics': ev['smoke']['result']['metrics'],
                  'smoke_pass': ev['smoke']['smoke_pass']} if ev['smoke'] else None,
        'firewall': {'sessions': fws, 'test_images_accessed': 0, 'test_metadata_accessed': 0,
                     'train_images_accessed': 0},
        'launcher_logs': logs,
        'eval_root_files': ev['files'],
        'confirmations': {'test_accessed': False, 'training_started': False, 'seeds_1337_2026_started': False,
                          'b1_b2_b3_started': False, 'bank_generation': False, 'downstream_fas': False,
                          'scientific_run_root_modified': False, 'checkpoint_copied_or_renamed': False,
                          'ensemble': False},
    }
    return selection, csv_body, raw


def report_md(sel):
    w, ru = sel['winner'], sel['runner_up']
    lines = [
        '# M7D1-N6C — VAL-only best-EMA selection, M7D1-A1 GPAT-B0 / E08 / seed 42', '',
        f"Status: **{sel['status']}**. Run `{RUN_ID}` (launch authority `{LAUNCH_COMMIT[:7]}`, closure authority "
        f"`{CLOSURE_COMMIT[:7]}`), curriculum `{CURRICULUM_ID}`.", '',
        '## What was done', '',
        f"All 51 EMA candidates (epochs 10..60, N6B manifest digest `{EMA_MANIFEST_DIGEST[:16]}…`) were evaluated on "
        f"the full VAL relation `{VAL_PAIRS}` ({VAL_ROWS} pairs, SHA-256 `{VAL_PAIRS_SHA256[:16]}…`). Each candidate "
        'was hash-checked against the manifest before loading. `recovery/latest.pt` was never opened as a candidate. '
        'TEST was never readable: an audit-hook firewall denies any non-VAL canonical face and any manifest other '
        'than `val_pairs_v1`.', '',
        'Owner evaluation decisions for N6C (2026-10-17): the residual scale `s_hf = 0.15` (spec 9.3 / '
        '`delta_scale_hf`) for every candidate, and the M7B R-06 fp16 autocast for the E_art/G_res forward. Every '
        'other convention comes from existing authority: A10 D09/D13/D14, M7B R-03/R-06, M7C2a N-06/N-08, and the '
        'M7C3 evaluator adapters.', '',
        '## Rule (frozen spec 10.6, A10 D14)', '',
        '`score = 0.25·p(ID) + 0.15·p(Dice) + 0.25·p(ArtSim) + 0.20·p(−NME) + 0.15·p(−LFErr)`, where '
        '`p = (rank − 1)/(N − 1)` and ties get the average rank. Ties on score go to the higher raw ArtSim, then to '
        'the earlier epoch. Arithmetic is exact (rationals).', '',
        '## Raw VAL metric ranges over the 51 candidates', '',
        '| metric | min (epoch) | max (epoch) | distinct values |', '|---|---|---|---|']
    for m, r in sel['metric_ranges'].items():
        lines.append(f"| {m} | {r['min']:.6g} ({r['argmin_epoch']}) | {r['max']:.6g} ({r['argmax_epoch']}) | "
                     f"{r['distinct_values']} |")
    lines += ['', 'LFErr is exactly 0 for every candidate (γ = 0, M7C2a N-08). All 51 candidates therefore tie on '
              'it, and p(−LFErr) = 0.5 for each of them.', '',
              '## Top 5', '', '| pos | epoch | score | ID | Dice | ArtSim | NME | LFErr | checkpoint SHA-256 |',
              '|---|---|---|---|---|---|---|---|---|']
    for t in sel['top_5']:
        m = t['raw_metrics']
        lines.append(f"| {t['position']} | {t['epoch']} | {t['selection_score']:.6f} | {m['ID']:.6f} | "
                     f"{m['Dice']:.6f} | {m['ArtSim']:.6f} | {m['NME']:.6f} | {m['LFErr']:.3g} | "
                     f"`{t['checkpoint_sha256'][:16]}…` |")
    tb = sel['tie_break']
    lines += ['', '## BEST_EMA_SEED42', '',
              f"- Epoch **{w['epoch']}**: `{w['checkpoint_path']}`",
              f"- SHA-256 `{w['checkpoint_sha256']}`",
              f"- Raw metrics: " + ', '.join(f'{k} {v:.6g}' for k, v in w['raw_metrics'].items()),
              f"- Percentiles: " + ', '.join(f'{k} {v:.4f}' for k, v in w['percentiles'].items()),
              f"- Score {w['selection_score']:.6f} (`{w['selection_score_exact']}`)",
              f"- Runner-up: epoch {ru['epoch']}, score {ru['selection_score']:.6f}; margin "
              f"{sel['margin_to_runner_up']:.6f}",
              f"- Tie-break required: {tb['tie_break_required']}", '',
              '## Reproducibility', '',
              '- An independent re-ranking from the committed CSV gives the same winner.',
              '- A full re-evaluation of the winner on VAL reproduces all five metrics bitwise: both the means and '
              'every per-pair value.', '',
              '## Post-selection sanity (POST_SELECTION_SANITY_ONLY; does not affect the selection)', '']
    for k, v in sel['post_selection_sanity']['diagnostics_summary'].items():
        lines.append(f"- {k}: mean {v['mean']:.6g}, min {v['min']:.6g}, median {v['median']:.6g}, max {v['max']:.6g}")
    lines += ['', sel.get('sanity_interpretation', ''), '',
              '## Firewall', '', 'Across the smoke, full and sanity sessions: TEST images 0, TEST metadata 0, TRAIN '
              'images 0, 0 denied opens, and 0 writes into the scientific run root. The run-root listing digest is '
              f"`{RUN_ROOT_LISTING_DIGEST[:16]}…` both before and after every phase.", '',
              'Not started: seed 1337/2026, B1/B2/B3, bank generation, downstream FAS, TEST evaluation and any '
              'training.', '']
    return '\n'.join(lines)


def write_evidence(selection, csv_body, collected_raw, interpretation):
    selection['sanity_interpretation'] = interpretation
    files = {EVIDENCE['collected']: collected_raw,
             EVIDENCE['csv']: csv_body.encode('utf-8'),
             EVIDENCE['selection']: (json.dumps(selection, indent=1, sort_keys=True) + '\n').encode('utf-8'),
             EVIDENCE['report']: report_md(selection).encode('utf-8')}
    for rel, data in files.items():
        (ROOT / rel).write_bytes(data)
    manifest = ''.join(f'{sha256_bytes(files[rel])}  {rel}\n' for rel in sorted(files))
    (ROOT / EVIDENCE['sha256']).write_text(manifest)
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('phase', choices=('build',))
    ap.add_argument('--collected', required=True)
    ap.add_argument('--interpretation', required=True, help='post-selection sanity interpretation (text file)')
    a = ap.parse_args(argv)
    selection, csv_body, raw = build(a.collected)
    print(write_evidence(selection, csv_body, raw, Path(a.interpretation).read_text().strip()))
    print(json.dumps({'winner': selection['winner'], 'runner_up': selection['runner_up']['epoch'],
                      'margin': selection['margin_to_runner_up'], 'tie_break': selection['tie_break']}, indent=1))


if __name__ == '__main__':
    main()
