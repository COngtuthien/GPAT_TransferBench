"""M7D1-N4: GPAT-B0 / E08 / seed 42 identity-collapse REPAIR QUALIFICATION.
DIAGNOSTIC_ONLY -- not a scientific run, not a protocol amendment, not an approved repair. Every branch is a diagnostic
variant; nothing here changes a production config, the frozen specification or the scientific run root.

Root cause (M7D1-N3, d88563a): lambda_adv 0 -> 0.05 at the stage-2 boundary u5526 against a stage-1-overconfident D;
gadv dominates the G update and, through the shared G_res trunk, saturates the mask sigmoid closed (identity collapse,
onset u5781). N4 evaluates curriculum-only lambda_adv repairs from the SAME exact pre-u5526 state that N3 fork A
reproduced 325/325 bitwise (N3 `snapshots/replay/pre_update_5526.pt`, sha256 BASE_SHA256).

Branches (only lambda_adv differs; s_hf, lambda_con, lambda_spec, LR, D updates, AMP policy, optimizer, clipping, EMA,
data order are the frozen scientific ones):
  R0  CONTROL           frozen stage 2: lambda_adv = 0.05 from u5526 (the unpatched runtime_contract.curriculum)
  R1  ONE-EPOCH RAMP    lambda_adv = 0.05 * ((u - 5526) / (6630 - 5526)) for 5526 <= u <= 6630, then 0.05
  R2  FIVE-EPOCH RAMP   lambda_adv = 0.05 * ((u - 5526) / (11050 - 5526)) for 5526 <= u <= 11050, then 0.05
  R3  LOW FIXED 0.01    lambda_adv = 0.01 from u5526 (fallback diagnostic: changes the final objective)
  R4  LOW FIXED 0.02    lambda_adv = 0.02 from u5526 (fallback diagnostic: changes the final objective)
  R5  D reset / soft start: R5_UNSUPPORTED_EXACT_STATE (no recorded authority for the original D initialization)
  R6  mask floor: lowest priority, only if R1-R4 give no viable repair (not implemented unless needed)
The ramps are written 0.05 * (ratio) rather than (0.05 * n) / d so that both endpoints are exactly 0.0 and 0.05 in
binary floating point ((0.05 * 5524) / 5524 == 0.049999999999999996); the two forms are equal as real numbers.

Scenarios (fresh process each, gpat-m7-gpu, PYTHONHASHSEED=42 + rio.LAUNCH_ENVIRONMENT, short TMPDIR):
  branch   --branch RX [--end U]   u5526..U from the base snapshot; per-update mask/artifact/loss rows; snapshots
  analyze  --branch RX --updates   N3 per-loss attribution + Adam-step split (+ shared-trunk shares) on branch snapshots
  collect                          CPU/stdlib: compact collected document from the N4 diagnostic root
Writes go only to <runtime_root>/diagnostics/m7/M7D1_N4_seed42/ (and TMPDIR); an audit hook refuses writes under
<runtime_root>/runs, the N3 diagnostic root and the repository configs.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_n3():
    spec = importlib.util.spec_from_file_location('m7d1_n3_collapse_root_cause',
                                                  ROOT / 'tools' / 'm7d1_n3_collapse_root_cause.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


n3 = _load_n3()
N3_PARTS = n3.DIAG_PARTS
N4_PARTS = ('diagnostics', 'm7', 'M7D1_N4_seed42')
N3_SNAP_KIND = n3.SNAP_KIND
N4_SNAP_KIND = 'GPAT_M7D1_N4_DIAGNOSTIC_PRE_UPDATE_SNAPSHOT'
n3.DIAG_PARTS = N4_PARTS                 # every N3 helper (DiagContext, save_snapshot, ...) now writes under N4 only
n3.SNAP_KIND = N4_SNAP_KIND

METHOD, SEED, SCI_RUN_ID = n3.METHOD, n3.SEED, n3.SCI_RUN_ID
LABELS = n3.LABELS + ['DIAGNOSTIC_REPAIR_VARIANT', 'NOT_AN_APPROVED_PROTOCOL_CHANGE']
AUTHORITY_COMMIT = 'd88563a46757d171a5e2af359eebc4fc18c305eb'
BASE_SNAPSHOT = 'snapshots/replay/pre_update_5526.pt'          # relative to the N3 diagnostic root
BASE_SHA256 = '3b77d738e810c5cad05f7974e7c688fb71719fcd1547bac270f8c2346b8eaf59'
BASE_POSITION = {'stage': 'generator', 'epoch': 6, 'next_group': 1, 'global_update': 5525}
START, PHASE_A_END, STAGE2_LAST = 5526, 5850, 16575
LAMBDA_ADV_FINAL = 0.05
MASK = n3.MASK
BRANCHES = {
    'R0': {'kind': 'CONTROL_FROZEN_STAGE2', 'name': 'CONTROL'},
    'R1': {'kind': 'LINEAR_RAMP', 'name': 'ADV_RAMP_1E', 'ramp_end': 6630},
    'R2': {'kind': 'LINEAR_RAMP', 'name': 'ADV_RAMP_5E', 'ramp_end': 11050},
    'R3': {'kind': 'CONSTANT', 'name': 'LOW_ADV_001', 'value': 0.01},
    'R4': {'kind': 'CONSTANT', 'name': 'LOW_ADV_002', 'value': 0.02},
}
LONG_BRANCHES = ('R1', 'R2')
DEFAULT_END = {'R0': PHASE_A_END, 'R1': STAGE2_LAST, 'R2': STAGE2_LAST, 'R3': PHASE_A_END, 'R4': PHASE_A_END}
ATTRIBUTION_UPDATES = {
    'R0': (5526, 5600, 5700, 5750, 5781, 5850),
    'R1': (5526, 5600, 5700, 5750, 5781, 5850, 6630, 6631, 8840, 11050, 13800, 16575),
    'R2': (5526, 5600, 5700, 5750, 5781, 5850, 6630, 8840, 11050, 11051, 13800, 16575),
    'R3': (5850,), 'R4': (5850,),
}
R5_STATUS = {'status': 'R5_UNSUPPORTED_EXACT_STATE',
             'reason': 'the scientific run records no initial-D state or digest (run_manifest has no init record, '
                       'only the final recovery/latest.pt exists, the N3 snapshots start at u5525); a reset would need '
                       'an invented definition (seed, D_OPT/D_SCALER state), which N4 forbids'}

# ----------------------------------------------------------------------------- predeclared rules (before any result)
COLLAPSE_RULE = {
    'name': 'N4_IDENTITY_COLLAPSE_V1',
    'metric': 'M_mean: mean of M = sigmoid(float(mask_logit)) over every mask pixel of every sample of the accepted '
              '(COMPLETE) forward of update u (sample-weighted over the microbatches of the group)',
    'threshold': 0.01, 'consecutive_complete_updates': 10,
    'onset_update': 'first update of the first run of 10 consecutive updates with M_mean < 0.01',
    'event_update': 'the 10th update of that run (first update at which the 10-update condition holds)',
    'declared': 'M7D1-N4 instruction section 7, implemented before any repair branch was run',
}
ARTIFACT_RULE = {
    'name': 'N4_ARTIFACT_RETENTION_CHARACTERIZATION_V1',
    'metric': 'per-update group mean of A, epoch median', 'reference_floor': 0.01,
    'reference_floor_source': 'frozen budget hinge ARTIFACT_MIN (A10 D10.6, methods/gpat/losses.py)',
    'retained': 'every completed post-ramp stage-2 epoch has median group-mean A >= 0.01 and no declared collapse',
    'declared': 'before any repair branch was run; characterization used for preference criterion 3 only',
}
LONG_HARD_GATES = ('declared identity collapse', 'terminal numerical failure', 'OOM', 'FAIL_CLOSED_AMP_OVERFLOW_FINAL',
                   'non-finite loss/parameter', 'update discontinuity', 'TRAIN-order mismatch', 'provenance mismatch',
                   'VAL/TEST access', 'scientific-run mutation')
PREFERENCE = ('passes hard gates', 'prevents the known collapse through u16575', 'preserves meaningful artifact output',
              'preserves eventual lambda_adv = 0.05', 'smallest protocol deviation', 'no architecture change',
              'no optimizer / D reset')
FIDELITY_TERMS = ('id', 'lm', 'parse', 'low', 'bg')
COLLAPSE_MARGIN = 50                      # long branches stop 50 updates after a declared collapse event


def lambda_adv(branch, u):
    """Diagnostic lambda_adv of generator update u (stage 2 only)."""
    if not START <= u <= STAGE2_LAST:
        raise ValueError(f'N4 branches are defined on u{START}..u{STAGE2_LAST} only, got u{u}')
    b = BRANCHES[branch]
    if b['kind'] == 'CONTROL_FROZEN_STAGE2':
        return LAMBDA_ADV_FINAL
    if b['kind'] == 'LINEAR_RAMP':
        end = b['ramp_end']
        return LAMBDA_ADV_FINAL * ((u - START) / (end - START)) if u <= end else LAMBDA_ADV_FINAL
    return b['value']


def branch_curriculum(branch, real):
    """real(u) with only lambda_adv replaced on u5526..u16575; R0 returns real(u) unchanged."""
    def curriculum(u):
        c = real(u)
        if u < START or branch == 'R0':
            return c
        if u > STAGE2_LAST or c['stage'] != 2:
            raise AssertionError(f'N4 branch {branch} left the frozen stage-2 window at u{u}')
        return dict(c, lambda_adv=lambda_adv(branch, u))
    curriculum.n4_branch = branch
    return curriculum


def collapse_event(rows, rule=COLLAPSE_RULE):
    """rows: [(u, M_mean)] in update order. Returns the declared event or None."""
    need, thr = rule['consecutive_complete_updates'], rule['threshold']
    run, prev = [], None
    for u, m in rows:
        if prev is not None and u != prev + 1:
            run = []
        prev = u
        run = run + [u] if m < thr else []
        if len(run) == need:
            return {'onset_update': run[0], 'event_update': run[-1]}
    return None


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


file_sha = n3.file_sha
TOOL_SHA256_AT_START = file_sha(__file__)


def runtime_root():
    return n3.runtime_root()


def n4_root():
    return runtime_root().joinpath(*N4_PARTS)


def n3_root():
    return runtime_root().joinpath(*N3_PARTS)


# ----------------------------------------------------------------------------- write firewall
def install_write_firewall():
    rt = str(runtime_root().resolve())
    allowed = [str(n4_root()), os.environ.get('TMPDIR', '/nonexistent'), '/dev/shm', '/dev/null', '/proc/']
    forbidden = [rt + '/runs', str(n3_root()), str(ROOT / 'configs'), str(ROOT / 'docs')]
    record = {'denied': [], 'outside_allowed': []}

    def hook(event, args):
        if event != 'open' or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        mode = args[1] if len(args) > 1 and isinstance(args[1], str) else ''
        flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
        writing = any(c in mode for c in 'wax+') or bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
        if not writing:
            return
        p = os.path.abspath(os.fsdecode(args[0]))
        if any(p.startswith(f) for f in forbidden):
            record['denied'].append(p)
            raise PermissionError(f'M7D1-N4 firewall: write refused (scientific / N3 / config root): {p}')
        if not any(p.startswith(a) for a in allowed):
            record['outside_allowed'].append(p)
    sys.addaudithook(hook)
    return record


# ----------------------------------------------------------------------------- base state
def load_base():
    """The exact N3 pre-u5526 state (fork A base): hash pinned, N3 kind/labels, position after u5525."""
    from methods.gpat import runner_checkpoint as ck
    path = n3_root() / BASE_SNAPSHOT
    meta = json.loads(path.with_suffix('.json').read_text())
    got = file_sha(path)
    assert got == BASE_SHA256 == meta['sha256'], f'base snapshot hash {got}'
    snap = ck.load(path)
    assert snap['kind'] == N3_SNAP_KIND and snap['labels'] == n3.LABELS
    assert snap['global_update_attempted'] == START
    assert {k: snap['position'][k] for k in BASE_POSITION} == BASE_POSITION, snap['position']
    assert snap['ema'] is not None, 'EMA must be active after u5525'
    return snap, {'path': str(path), 'sha256': got, 'size': path.stat().st_size, 'state_digest': meta['state_digest'],
                  'code_head_at_capture': meta['code_head']}


# ----------------------------------------------------------------------------- per-forward artifact statistics
class CoreHook:
    """Forward hook on GPATCore (called as self.core(x_source, x_target, scale_hf=...)); read-only, returns None."""

    def __init__(self, tr):
        self.buf = []
        tr.core.register_forward_hook(self)

    def __call__(self, module, inputs, out):
        import torch
        with torch.no_grad():
            x_t = inputs[1]
            hf = (out.delta_LH.abs() + out.delta_HL.abs() + out.delta_HH.abs()).float()
            self.buf.append({'logit': out.raw[:, MASK].detach().float().clone(),
                             'M': torch.sigmoid(out.raw[:, MASK].detach().float()),
                             'A': out.A.detach().float().clone(),
                             'res_t': (out.x_hat.detach().float() - x_t.float()).abs().mean(dim=1),
                             'MdHF': (out.M.detach().float() * hf),
                             'dHF_mean': float(hf.mean()) / 3.0,
                             'tanh_hf_absmean': float(torch.tanh(out.raw[:, 3:12].detach().float()).abs().mean()),
                             'n': int(x_t.shape[0])})
        return None

    def take(self, n_mb):
        import torch
        b = self.buf[-n_mb:]
        self.buf = []
        cat = {k: torch.cat([x[k] for x in b]) for k in ('logit', 'M', 'A', 'res_t', 'MdHF')}
        q = lambda t, ps: torch.quantile(t.flatten(), torch.tensor(ps, device=t.device)).tolist()   # noqa: E731
        M, A, L = cat['M'], cat['A'], cat['logit']
        mq = q(M, [0.01, 0.1, 0.5, 0.9, 0.99])
        aq = q(A, [0.1, 0.5, 0.9])
        rq = q(cat['res_t'], [0.5, 0.9])
        img_A = A.flatten(1).mean(1)
        w = [x['n'] for x in b]
        return {
            'logit_mean': float(L.mean()), 'logit_std': float(L.std()), 'logit_min': float(L.min()),
            'logit_max': float(L.max()),
            'M_mean': float(M.mean()), 'M_std': float(M.std()), 'M_min': float(M.min()), 'M_max': float(M.max()),
            'M_p01': mq[0], 'M_p10': mq[1], 'M_p50': mq[2], 'M_p90': mq[3], 'M_p99': mq[4],
            'frac_M_lt_1e-1': float((M < 1e-1).float().mean()), 'frac_M_lt_1e-2': float((M < 1e-2).float().mean()),
            'frac_M_lt_1e-3': float((M < 1e-3).float().mean()), 'frac_M_gt_0.5': float((M > 0.5).float().mean()),
            'sigmoid_deriv_mean': float((M * (1 - M)).mean()),
            'tanh_hf_absmean': sum(x['tanh_hf_absmean'] * n for x, n in zip(b, w)) / sum(w),
            'abs_delta_HF_mean': sum(x['dHF_mean'] * n for x, n in zip(b, w)) / sum(w),
            'M_times_abs_delta_HF_mean': float(cat['MdHF'].mean()),
            'abs_xhat_minus_target_mean': float(cat['res_t'].mean()), 'abs_xhat_minus_target_p50': rq[0],
            'abs_xhat_minus_target_p90': rq[1], 'abs_xhat_minus_target_max': float(cat['res_t'].max()),
            'A_mean': float(A.mean()), 'A_p10': aq[0], 'A_p50': aq[1], 'A_p90': aq[2], 'A_max': float(A.max()),
            'A_image_mean_min': float(img_A.min()), 'frac_images_A_lt_artifact_min': float((img_A < 0.01).float().mean()),
            'samples': sum(w)}


class N4Wrapper(n3.StepWrapper):
    """N3 StepWrapper (pre-update snapshots + N3 mask hook; never changes the wrapped step) plus per-update artifact
    statistics, a streamed per-update row, the declared collapse tracker and an optional post-collapse stop."""

    def __init__(self, tr, snap_dir, snap_updates, branch, rows_path, stop, long_run):
        super().__init__(tr, snap_dir, snap_updates, None)
        self.branch, self.stop, self.long_run = branch, stop, long_run
        self.core_hook = CoreHook(tr)
        self.rows_fh = open(rows_path, 'a', encoding='utf-8')
        self.m_rows = []
        self.collapse = None

    def __call__(self, group, u):
        import hashlib as _h
        epoch, g = self.tr.position['epoch'], self.tr.position['next_group']
        idx = [mb['index'].tolist() for mb in group]
        self.core_hook.buf = []
        rec = super().__call__(group, u)
        ext = self.core_hook.take(len(group))
        n3row = self.mask_rows[-1]
        tl = rec['train_losses']
        row = {'global_update': u, 'epoch': epoch, 'group': g, 'branch': self.branch,
               'group_indices_sha256': _h.sha256(json.dumps(idx).encode()).hexdigest(),
               'lambda_adv': rec['lambda_adv'], 'learning_rate': rec['learning_rate'], 'scale_hf': rec['scale_hf'],
               'lambda_con': rec['lambda_con'], 'lambda_spec': rec['lambda_spec'],
               'G_grad_norm': rec['G_grad_norm'], 'D_grad_norm': rec['D_grad_norm'],
               'G_clip_coefficient': min(1.0, 1.0 / (rec['G_grad_norm'] + 1e-6)),
               'D_clip_coefficient': min(1.0, 1.0 / (rec['D_grad_norm'] + 1e-6)),
               'G_scale': rec['G_scale'], 'D_scale': rec['D_scale'], 'G_scale_before': rec['G_scale_before'],
               'D_scale_before': rec['D_scale_before'], 'amp_attempts': rec['amp_attempts'],
               'group_status': rec['group_status'], 'ema_updated': rec['ema_updated'],
               'losses': {k: tl[k] for k in sorted(tl)}, 'n3_mask': {k: n3row[k] for k in sorted(n3row)
                                                                     if k != 'global_update'},
               **ext}
        self.rows_fh.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
        self.rows_fh.flush()
        if rec['group_status'] == 'COMPLETE':
            self.m_rows.append((u, ext['M_mean']))
            if self.collapse is None:
                self.collapse = collapse_event(self.m_rows[-COLLAPSE_RULE['consecutive_complete_updates']:])
                if self.collapse is not None:
                    print(f'N4 {self.branch}: declared collapse onset u{self.collapse["onset_update"]} '
                          f'event u{self.collapse["event_update"]}', flush=True)
        if self.long_run and self.collapse is not None and u >= self.collapse['event_update'] + COLLAPSE_MARGIN:
            self.stop.requested = True
        return rec


def _patch_failure_recording(tr):
    def record_failure(exc, *, stage, epoch, attempted, role):
        tr.ctx.log('numerical_failure', {**exc.record, 'stage': stage, 'epoch': epoch,
                                         'attempted_group_or_batch': attempted, 'position_unchanged': dict(tr.position),
                                         'diagnostic': 'N4 branch stops; no recovery exists for diagnostic branches'})
    tr.record_failure = record_failure


def _check_launch():
    tmp = os.environ.get('TMPDIR', '')
    assert tmp and len(tmp) <= 24, 'launch with a short TMPDIR (AF_UNIX path length), e.g. /tmp/gpat_n4'
    for k in ('TMP', 'TEMP'):
        assert os.environ.get(k) == tmp, f'{k} must equal TMPDIR'


def scenario_branch(args):
    import torch
    from methods.gpat import runner as R
    from methods.gpat import runtime_contract as rc
    branch = args.branch
    end = args.end or DEFAULT_END[branch]
    assert START < end <= STAGE2_LAST
    assert end >= PHASE_A_END or (args.name or '').startswith('smoke_'), 'a short branch must be named smoke_*'
    real = rc.curriculum
    if branch != 'R0':
        rc.curriculum = branch_curriculum(branch, real)          # this diagnostic process only; configs untouched
    name = args.name or f'branch_{branch}'
    tr = n3.setup(name, {'runner_mode': 'DIAGNOSTIC_REPAIR_QUALIFICATION_FROM_N3_PRE_U5526', 'labels': LABELS,
                         'purpose': f'M7D1-N4 repair branch {branch} u{START}..u{end}', 'branch': branch,
                         'branch_definition': BRANCHES[branch], 'base_snapshot_sha256': BASE_SHA256,
                         'collapse_rule': COLLAPSE_RULE})
    tr.suppressed_candidates = []
    tr.write_candidate = lambda epoch: tr.suppressed_candidates.append(epoch)
    _patch_failure_recording(tr)
    snap, base = load_base()
    n3.apply_snapshot(tr, snap)
    del snap
    assert tr.step.ema is tr.ema and tr.ema is not None, 'EMA must be live on the real step'
    assert tr.position['global_update'] == START - 1
    stop = n3.Stop()
    snaps = sorted(int(x) for x in args.updates.split(',')) if args.updates else ATTRIBUTION_UPDATES[branch]
    w = N4Wrapper(tr, n4_root() / 'snapshots' / name, [x for x in snaps if x <= end], branch,
                  tr.ctx.run_dir / 'per_update.jsonl', stop, long_run=end > PHASE_A_END)
    tr.step = w
    t0 = time.monotonic()
    status, failure = 'COMPLETED_TO_END', None
    try:
        tr.run_generator_groups(stop, limit=end - START + 1)
    except R.NumericalStop as exc:
        status, failure = 'HARD_FAILURE_NUMERICAL', {'type': type(exc).__name__, 'record': exc.record}
    except torch.cuda.OutOfMemoryError as exc:
        status, failure = 'HARD_FAILURE_OOM', {'type': 'OutOfMemoryError', 'message': str(exc)[:500]}
    if status == 'COMPLETED_TO_END' and tr.position['global_update'] != end:
        status = 'STOPPED_AFTER_DECLARED_COLLAPSE' if w.collapse else 'STOPPED_UNEXPECTEDLY'
    w.rows_fh.close()
    recs, retries = {}, []
    for line in (tr.ctx.run_dir / 'metrics.jsonl').read_text().splitlines():
        d = json.loads(line)
        if d['record_type'] == 'optimizer_group':
            recs[d['global_update']] = d
        elif d['record_type'] == 'amp_retry':
            retries.append(d)
    n3.write_jsonl(tr.ctx.run_dir / 'mask_per_update.jsonl', w.mask_rows)
    last = tr.position['global_update']
    res = {'branch': branch, 'definition': BRANCHES[branch], 'end_requested': end, 'last_completed_update': last,
           'status': status, 'failure': failure, 'collapse': w.collapse, 'base_snapshot': base,
           'records': len(recs), 'records_contiguous': sorted(recs) == list(range(START, last + 1)),
           'seconds_stepping': time.monotonic() - t0,
           'amp_retries': [{k: r[k] for k in ('global_update_attempted', 'offending_optimizers', 'old_scale',
                                              'new_scale')} for r in retries],
           'snapshots': w.snaps, 'suppressed_recovery_saves': tr.suppressed_recovery_saves,
           'suppressed_candidate_epochs': tr.suppressed_candidates, 'access': tr.access.report(),
           'peak_reserved_GiB': round(torch.cuda.max_memory_reserved() / 2 ** 30, 3)}
    if branch == 'R0':
        res['parity'] = r0_parity(recs, retries, w.mask_rows, min(last, PHASE_A_END))
    tr.ctx.close('diagnostic_completed' if status.startswith(('COMPLETED', 'STOPPED_AFTER')) else status,
                 {'access': res['access'], 'status': status})
    return res


def r0_parity(recs, retries, mask_rows, end):
    """R0 vs the scientific metrics.jsonl (read-only) and vs N3 fork A (records + N3 mask rows), u5526..end."""
    sci, sci_retry = n3.scientific_records()
    ups = range(START, end + 1)
    mism_sci = {u: n3.diff_fields(recs[u], sci[u]) for u in ups if u in recs}
    fa = {}
    for line in (n3_root() / 'fork_A' / 'metrics.jsonl').read_text().splitlines():
        d = json.loads(line)
        if d['record_type'] == 'optimizer_group':
            fa[d['global_update']] = d
    mism_fa = {u: n3.diff_fields(recs[u], fa[u]) for u in ups if u in recs}
    fa_mask = {json.loads(x)['global_update']: json.loads(x)
               for x in (n3_root() / 'fork_A' / 'mask_per_update.jsonl').read_text().splitlines()}
    mine_mask = {r['global_update']: r for r in mask_rows}
    mask_mism = [u for u in ups if json.dumps(mine_mask.get(u), sort_keys=True) != json.dumps(fa_mask.get(u),
                                                                                                sort_keys=True)]
    mine_r = {}
    for r in retries:
        mine_r.setdefault(r['global_update_attempted'], []).append(r)
    rmism = sorted(x for x in set(mine_r) | {y for y in sci_retry if START <= y <= end}
                   if [n3.strip(r) for r in mine_r.get(x, [])] != [n3.strip(r) for r in sci_retry.get(x, [])])
    return {'updates_compared': len(ups),
            'bitwise_equal_vs_scientific': sum(1 for u in ups if u in recs and not mism_sci[u]),
            'mismatched_vs_scientific': sorted(u for u, v in mism_sci.items() if v)[:100],
            'bitwise_equal_vs_n3_fork_A': sum(1 for u in ups if u in recs and not mism_fa[u]),
            'mismatched_vs_n3_fork_A': sorted(u for u, v in mism_fa.items() if v)[:100],
            'n3_mask_rows_bitwise_equal_fork_A': len(ups) - len(mask_mism), 'n3_mask_row_mismatch': mask_mism[:100],
            'amp_retry_event_mismatch_vs_scientific': rmism,
            'amp_retry_updates': sorted(mine_r)}


# ----------------------------------------------------------------------------- attribution
def _trunk_wrap(orig):
    """Adds shared-trunk (g_res except the mask-head row) and mask-row shares of each Adam-step part to N3 adam_parts."""
    def adam_parts(tr, u, unit, W, clip, used, offs, gsl):
        import torch
        parts, pred, info = orig(tr, u, unit, W, clip, used, offs, gsl)
        ew0, ew1 = offs['g_res.ending.weight']
        per_out = (ew1 - ew0) // 13
        n = parts['TOTAL'].numel()
        mrow = torch.zeros(n)
        mrow[ew0 + MASK * per_out: ew0 + (MASK + 1) * per_out] = 1
        mrow[offs['g_res.ending.bias'][0] + MASK] = 1
        pref = torch.cat([torch.full((k,), 1.0 if nm.startswith('g_res.') else 0.0) for nm, k in gsl])
        trunk = pref * (1 - mrow)
        trunk, mrow = trunk.double(), mrow.double()                     # float64: 43M-element dot products
        tot = parts['TOTAL'].double()
        tt, tm = tot * trunk, tot * mrow
        info['shared_trunk'] = {}
        for k, v in parts.items():
            if k.startswith('TOTAL_'):
                continue
            v = v.double()
            info['shared_trunk'][k] = {
                'trunk_l2': float((v * trunk).norm()),
                'trunk_projection_share': float(torch.dot(v * trunk, tt) / (torch.dot(tt, tt) + 1e-30)),
                'mask_row_projection_share': float(torch.dot(v * mrow, tm) / (torch.dot(tm, tm) + 1e-30)),
                'all_projection_share': float(torch.dot(v, tot) / (torch.dot(tot, tot) + 1e-30))}
        return parts, pred, info
    return adam_parts


def scenario_analyze(args):
    import torch
    from methods.gpat import runtime_contract as rc
    branch = args.branch
    src = args.source or f'branch_{branch}'
    ups = tuple(int(x) for x in args.updates.split(',')) if args.updates else ATTRIBUTION_UPDATES[branch]
    tr = n3.setup(args.name or f'analyze_{branch}', {'runner_mode': 'DIAGNOSTIC_REPAIR_ATTRIBUTION', 'labels': LABELS,
                                                     'purpose': f'N3 attribution on N4 {src} snapshots'})
    n3.adam_parts = _trunk_wrap(n3.adam_parts)
    out = n3.analyze_snapshots(tr, n4_root() / 'snapshots' / src, ups, branch_curriculum(branch, rc.curriculum),
                               recorded=n3.recorded_groups(src))
    res = {'branch': branch, 'snapshots': out, 'access': tr.access.report(),
           'peak_reserved_GiB': round(torch.cuda.max_memory_reserved() / 2 ** 30, 3)}
    tr.ctx.close('diagnostic_completed', {'access': res['access']})
    return res


# ----------------------------------------------------------------------------- collect (CPU, stdlib + runtime_contract)
SCREEN_KEYS = ('lambda_adv', 'learning_rate', 'scale_hf', 'lambda_con', 'lambda_spec', 'G_grad_norm', 'D_grad_norm',
               'G_clip_coefficient', 'D_clip_coefficient', 'G_scale', 'D_scale', 'amp_attempts', 'logit_mean',
               'logit_std', 'logit_min', 'logit_max', 'M_mean', 'M_std', 'M_min', 'M_max', 'M_p01', 'M_p10',
               'M_p50', 'M_p90', 'M_p99', 'frac_M_lt_1e-1', 'frac_M_lt_1e-2', 'frac_M_lt_1e-3', 'frac_M_gt_0.5',
               'sigmoid_deriv_mean', 'tanh_hf_absmean', 'abs_delta_HF_mean', 'M_times_abs_delta_HF_mean',
               'abs_xhat_minus_target_mean', 'A_mean', 'A_p10', 'A_p50', 'A_p90', 'A_max')
LOSS_KEYS = ('D_total', 'D_real', 'D_fake', 'gadv', 'G_total', 'budget', 'artcon', 'spec', 'id', 'lm', 'parse', 'bg',
             'tv', 'low')
LONG_KEYS = ('lambda_adv', 'G_grad_norm', 'D_grad_norm', 'G_scale', 'D_scale', 'amp_attempts', 'M_mean', 'M_p10',
             'M_p50', 'frac_M_lt_1e-2', 'sigmoid_deriv_mean', 'logit_mean', 'tanh_hf_absmean',
             'M_times_abs_delta_HF_mean', 'abs_xhat_minus_target_mean', 'A_mean', 'A_p90')
LONG_LOSS_KEYS = ('D_total', 'D_real', 'D_fake', 'gadv', 'G_total', 'budget', 'artcon', 'spec', 'bg', 'id', 'lm')
ATTR_TERMS = ('gadv', 'artcon', 'spec', 'budget', 'id', 'lm', 'parse', 'low', 'bg', 'tv', 'momentum')


def _jsonl(path):
    with open(path) as f:
        return [json.loads(x) for x in f if x.strip()]


def _q(vals, p):
    s = sorted(vals)
    if not s:
        return None
    k = (len(s) - 1) * p
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _dist(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return {'n': len(vals), 'mean': sum(vals) / len(vals), 'min': min(vals), 'p10': _q(vals, 0.1),
            'p50': _q(vals, 0.5), 'p90': _q(vals, 0.9), 'max': max(vals)}


def schedule_verification(branch, rows, sci):
    """Every row: lambda_adv == lambda_adv(branch, u) exactly; LR, s_hf, lambda_con, lambda_spec, epoch/group equal to
    the scientific record of the same update and to runtime_contract (frozen)."""
    from methods.gpat import runtime_contract as rc
    bad_lambda, bad_other, bad_rc = [], [], []
    for r in rows:
        u = r['global_update']
        if r['lambda_adv'] != lambda_adv(branch, u):
            bad_lambda.append(u)
        s = sci[u]
        if any(r[k] != s[k] for k in ('learning_rate', 'scale_hf', 'lambda_con', 'lambda_spec', 'epoch', 'group')):
            bad_other.append(u)
        c = rc.curriculum(u)
        if (r['learning_rate'], r['scale_hf'], r['lambda_con'], r['lambda_spec']) != (
                rc.main_lr(u), c['s_hf'], c['lambda_con'], c['lambda_spec']):
            bad_rc.append(u)
    ups = [r['global_update'] for r in rows]
    return {'rows': len(rows), 'lambda_adv_exact_mismatch': bad_lambda[:50],
            'other_terms_vs_scientific_mismatch': bad_other[:50], 'other_terms_vs_frozen_contract_mismatch': bad_rc[:50],
            'contiguous_from_5526': ups == list(range(START, START + len(ups))),
            'lambda_adv_at': {str(u): r['lambda_adv'] for r in rows for u in (r['global_update'],)
                              if u in (5526, 5527, 6078, 6629, 6630, 6631, 8288, 11049, 11050, 11051, 16575)}}


def _compact(r, keys, loss_keys):
    out = {k: r[k] for k in keys}
    out.update({'L_' + k: r['losses'].get(k) for k in loss_keys})
    return out


def collect_branch(branch, sci, sci_groups_sha=None):
    d = n4_root()
    res_path = d / 'results' / f'branch_{branch}.json'
    res = json.loads(res_path.read_text()) if res_path.exists() else None
    rows = _jsonl(d / f'branch_{branch}' / 'per_update.jsonl')
    ev = collapse_event([(r['global_update'], r['M_mean']) for r in rows if r['group_status'] == 'COMPLETE'])
    screen = [dict(global_update=r['global_update'], epoch=r['epoch'], group=r['group'],
                   **_compact(r, SCREEN_KEYS, LOSS_KEYS)) for r in rows if r['global_update'] <= PHASE_A_END]
    out = {'result': res, 'result_sha256': file_sha(res_path) if res else None,
           'per_update_sha256': file_sha(d / f'branch_{branch}' / 'per_update.jsonl'),
           'metrics_sha256': file_sha(d / f'branch_{branch}' / 'metrics.jsonl'),
           'rows': len(rows), 'last_update': rows[-1]['global_update'] if rows else None,
           'collapse_rederived': ev, 'schedule': schedule_verification(branch, rows, sci),
           '_group_sha': {r['global_update']: r['group_indices_sha256'] for r in rows},
           'screen': screen,
           'nonfinite_rows': [r['global_update'] for r in rows
                              if any(isinstance(v, float) and not math.isfinite(v)
                                     for v in list(r['losses'].values()) + [r['M_mean'], r['A_mean'], r['G_grad_norm']])]}
    if branch in LONG_BRANCHES:
        keep = long_keep(branch, rows, ev)
        out['long_sampling'] = LONG_SAMPLING
        out['long'] = [dict(global_update=r['global_update'], epoch=r['epoch'], **_compact(r, LONG_KEYS, LONG_LOSS_KEYS))
                       for r in rows if r['global_update'] in keep]
        by_ep = {}
        for r in rows:
            by_ep.setdefault(r['epoch'], []).append(r)
        out['epochs'] = {str(e): epoch_summary(rs) for e, rs in sorted(by_ep.items())}
    a_path = d / 'results' / f'analyze_{branch}.json'
    if a_path.exists():
        out['attribution'] = compact_attribution(json.loads(a_path.read_text()))
        out['attribution_sha256'] = file_sha(a_path)
    return out


LONG_SAMPLING = ('every 5th update from u5526, every AMP-retry update, u(ramp_end)-20..u(ramp_end)+20, the last 20 '
                 'updates, and collapse onset-30..event+30; epoch summaries use every update')


def long_keep(branch, rows, ev):
    ups = [r['global_update'] for r in rows]
    keep = {u for u in ups if (u - START) % 5 == 0}
    keep |= {r['global_update'] for r in rows if r['amp_attempts'] > 1}
    e = BRANCHES[branch].get('ramp_end')
    if e:
        keep |= set(range(e - 20, e + 21))
    keep |= set(ups[-20:])
    if ev:
        keep |= set(range(ev['onset_update'] - 30, ev['event_update'] + 31))
    return keep & set(ups)


def epoch_summary(rs):
    f = lambda k: [r[k] for r in rs]                                   # noqa: E731
    fl = lambda k: [r['losses'].get(k) for r in rs]                    # noqa: E731
    return {'updates': [rs[0]['global_update'], rs[-1]['global_update'], len(rs)],
            'lambda_adv': [rs[0]['lambda_adv'], rs[-1]['lambda_adv']],
            'M_mean': _dist(f('M_mean')), 'M_p10': _dist(f('M_p10')), 'frac_M_lt_1e-2': _dist(f('frac_M_lt_1e-2')),
            'sigmoid_deriv_mean': _dist(f('sigmoid_deriv_mean')), 'A_mean': _dist(f('A_mean')),
            'A_p90': _dist(f('A_p90')), 'abs_xhat_minus_target_mean': _dist(f('abs_xhat_minus_target_mean')),
            'tanh_hf_absmean': _dist(f('tanh_hf_absmean')),
            'M_times_abs_delta_HF_mean': _dist(f('M_times_abs_delta_HF_mean')),
            'G_grad_norm': _dist(f('G_grad_norm')), 'frac_G_grad_norm_gt_1': sum(1 for x in f('G_grad_norm') if x > 1)
            / len(rs), 'D_grad_norm': _dist(f('D_grad_norm')), 'amp_retry_updates': sum(1 for x in f('amp_attempts')
                                                                                         if x > 1),
            'D_total': _dist(fl('D_total')), 'D_real': _dist(fl('D_real')), 'D_fake': _dist(fl('D_fake')),
            'gadv': _dist(fl('gadv')), 'budget': _dist(fl('budget')), 'artcon': _dist(fl('artcon')),
            'id': _dist(fl('id')), 'lm': _dist(fl('lm')), 'bg': _dist(fl('bg')), 'G_total': _dist(fl('G_total')),
            'frac_budget_active': sum(1 for x in fl('budget') if x and x > 0) / len(rs),
            'frac_M_mean_lt_0.01': sum(1 for x in f('M_mean') if x < 0.01) / len(rs)}


def compact_attribution(a):
    out = {}
    for u, r in sorted(a['snapshots'].items(), key=lambda kv: int(kv[0])):
        le = r['logit_effect']
        st = r['adam_step'].get('shared_trunk', {})
        out[u] = {
            'curriculum': r['curriculum'], 'G_scale_used': r['G_scale_used'],
            'mask_logit_mean': r['mask_logit']['mean'], 'M_mean': r['M']['mean'], 'A_mean': r['A']['mean'],
            'abs_xhat_minus_target_mean': r['abs_xhat_minus_target']['mean'],
            'D_real_logit_mean': r['D_real_logits']['mean'], 'D_fake_logit_mean': r['D_fake_logits']['mean'],
            'BCE_fake_as_real_gadv': r['BCE_fake_as_real_gadv'],
            'G_grad_l2_preclip': r['total']['G_grad_l2_preclip'], 'clip_coefficient': r['total']['clip_coefficient'],
            'recorded_G_grad_norm': r.get('recorded_G_grad_norm'),
            'attribution_vs_recorded_rel': r.get('attribution_vs_recorded_G_grad_norm_rel'),
            'next_snapshot_check': r.get('next_snapshot_check'),
            'grad_projection_share': {k: r['terms'][k]['projection_share_of_total'] for k in r['terms']},
            'grad_l2_weighted': {k: r['terms'][k]['all_G_grad_l2_weighted'] for k in r['terms']},
            'lin_dlogit_mean': {k: le[k]['lin_dlogit_mean'] for k in le if k not in ('base_fp32', 'additivity')},
            'additivity': le['additivity'],
            'step_trunk_projection_share': {k: v['trunk_projection_share'] for k, v in st.items()},
            'step_all_projection_share': {k: v['all_projection_share'] for k, v in st.items()},
            'step_mask_row_projection_share': {k: v['mask_row_projection_share'] for k, v in st.items()},
            'momentum_l2_vs_current_l2': r['adam_step']['momentum_l2_vs_current_l2'],
            'snapshot': r['snapshot']}
    return out


def scenario_collect(args):
    sci_rows = {}
    with open(n3.sci_dir() / 'metrics.jsonl', 'rb') as f:                    # read-only
        for raw in f:
            d = json.loads(raw)
            if d.get('record_type') == 'optimizer_group' and START <= d['global_update'] <= STAGE2_LAST:
                sci_rows[d['global_update']] = d
    sci_metrics_sha = file_sha(n3.sci_dir() / 'metrics.jsonl')
    branches = {}
    for b in BRANCHES:
        if (n4_root() / f'branch_{b}' / 'per_update.jsonl').exists():
            branches[b] = collect_branch(b, sci_rows)
    order = train_order_check(branches)
    for d in branches.values():
        del d['_group_sha']
    sci_mtimes = {str(p.relative_to(n3.sci_dir())): int(p.stat().st_mtime) for p in sorted(n3.sci_dir().rglob('*'))
                  if p.is_file()}
    return {'milestone': 'M7D1-N4', 'labels': LABELS, 'method': METHOD, 'experiment': 'E08', 'seed': SEED,
            'scientific_run_id': SCI_RUN_ID, 'authority_commit': AUTHORITY_COMMIT,
            'n3_root_cause': {'commit': AUTHORITY_COMMIT, 'classification':
                              'ADVERSARIAL_IMBALANCE / CURRICULUM_TRANSITION_INSTABILITY',
                              'mechanism': 'MASK_SIGMOID_SATURATION through shared G_res trunk',
                              'fork_A_collapse_u': 5781, 'fork_C_collapse_u': 5716},
            'base_snapshot': {'n3_relative_path': BASE_SNAPSHOT, 'sha256': BASE_SHA256,
                              'position_after': BASE_POSITION},
            'branch_definitions': BRANCHES, 'r5': R5_STATUS, 'collapse_rule': COLLAPSE_RULE,
            'artifact_rule': ARTIFACT_RULE, 'long_hard_gates': LONG_HARD_GATES, 'preference': PREFERENCE,
            'scientific_metrics_sha256': sci_metrics_sha, 'scientific_root_file_mtimes': sci_mtimes,
            'scientific_stage2_lr_sample': {str(u): sci_rows[u]['learning_rate'] for u in (5526, 6630, 11050, 16575)},
            'train_order': order, 'branches': branches}


# ----------------------------------------------------------------------------- finalize (pure, from the collected doc)
POST_RAMP_EPOCH_RULE = 'complete 1105-update epochs whose every update has lambda_adv == 0.05'


def train_order_check(branches):
    """(collect) Every branch sees the same TRAIN groups at the same update (group-index digests): R0 is the reference
    for u <= 5850, R1 / R2 cross-check each other on the long window; u5526 must equal the base snapshot group."""
    ref = dict(branches['R0']['_group_sha']) if 'R0' in branches else {}
    for b in LONG_BRANCHES:
        if b in branches:
            for u, h in branches[b]['_group_sha'].items():
                ref.setdefault(u, h)
    out = {}
    for b, d in branches.items():
        g = d['_group_sha']
        shared = [u for u in g if u in ref]
        out[b] = {'updates_compared': len(shared), 'reference_branches': sorted({'R0', *LONG_BRANCHES} & set(branches)),
                  'mismatch': sorted(u for u in shared if g[u] != ref[u])[:50],
                  'sequence_sha256': sha(json.dumps([[u, g[u]] for u in sorted(g)]).encode())}
    return out


def post_ramp_epochs(d):
    return [e for e, s in sorted(d.get('epochs', {}).items(), key=lambda kv: int(kv[0]))
            if s['updates'][2] == 1105 and s['lambda_adv'] == [LAMBDA_ADV_FINAL, LAMBDA_ADV_FINAL]
            and all(r['lambda_adv'] == LAMBDA_ADV_FINAL for r in d['long'] if str(r['epoch']) == e)]


def hard_gate_failures(b, d, order):
    r = d['result']
    f = []
    if r is None:
        return ['result missing']
    if r['status'] != 'COMPLETED_TO_END':
        f.append(f'status {r["status"]}')
    if r['failure']:
        f.append(f'failure {r["failure"]["type"]}')
    if d['collapse_rederived'] is not None or r['collapse'] is not None:
        f.append('declared identity collapse')
    if d['collapse_rederived'] != r['collapse']:
        f.append('collapse re-derivation differs')
    if d['nonfinite_rows']:
        f.append('non-finite loss/M/A/grad-norm rows')
    if not (r['records_contiguous'] and d['schedule']['contiguous_from_5526']):
        f.append('update discontinuity')
    if d['rows'] != r['records'] or r['last_completed_update'] != d['last_update']:
        f.append('per-update rows differ from optimizer records')
    sch = d['schedule']
    if sch['lambda_adv_exact_mismatch'] or sch['other_terms_vs_scientific_mismatch'] or \
            sch['other_terms_vs_frozen_contract_mismatch']:
        f.append('curriculum/LR mismatch')
    if order[b]['mismatch']:
        f.append('TRAIN-order mismatch')
    a = r['access']
    if a['val_images'] or a['val_metadata'] or a['test_images'] or a['test_metadata'] or a['non_train_images']:
        f.append('VAL/TEST/non-TRAIN access')
    if r['write_firewall']['denied'] or r['write_firewall']['outside_allowed']:
        f.append('write firewall event')
    if r['code_head'] != AUTHORITY_COMMIT or r['base_snapshot']['sha256'] != BASE_SHA256:
        f.append('provenance mismatch')
    return f


def r0_parity_status(d):
    p = d['result']['parity']
    n = PHASE_A_END - START + 1
    ok = (p['updates_compared'] == n and p['bitwise_equal_vs_scientific'] == n and p['bitwise_equal_vs_n3_fork_A'] == n
          and p['n3_mask_rows_bitwise_equal_fork_A'] == n and not p['amp_retry_event_mismatch_vs_scientific'])
    ev = d['collapse_rederived']
    return {'status': 'R0_PARITY_PASS' if ok and ev is not None else 'R0_PARITY_FAIL', 'records_bitwise': ok,
            'collapse': ev, 'collapse_onset_equals_n3_fork_A_5781': bool(ev and ev['onset_update'] == 5781),
            'amp_retry_updates': p['amp_retry_updates'], **{k: p[k] for k in (
                'updates_compared', 'bitwise_equal_vs_scientific', 'bitwise_equal_vs_n3_fork_A',
                'n3_mask_rows_bitwise_equal_fork_A')}}


def phase_a(d):
    ev = d['collapse_rederived']
    if ev is not None and ev['onset_update'] <= PHASE_A_END:
        return {'result': 'COLLAPSED', **ev}
    if d['last_update'] is None or d['last_update'] < PHASE_A_END:
        return {'result': 'INCOMPLETE'}
    return {'result': 'EARLY_COLLAPSE_NOT_OBSERVED'}


def build_qualification(c):
    B = c['branches']
    order = c['train_order']
    out = {'milestone': 'M7D1-N4', 'labels': c['labels'], 'authority_commit': c['authority_commit'],
           'n3_root_cause': c['n3_root_cause'], 'base_snapshot': c['base_snapshot'],
           'branch_definitions': c['branch_definitions'], 'r5': c['r5'], 'collapse_rule': c['collapse_rule'],
           'artifact_rule': c['artifact_rule'], 'post_ramp_epoch_rule': POST_RAMP_EPOCH_RULE,
           'long_hard_gates': c['long_hard_gates'], 'preference': c['preference'], 'train_order': order,
           'scientific_metrics_sha256': c['scientific_metrics_sha256'], 'branches': {}}
    r0 = r0_parity_status(B['R0']) if 'R0' in B else {'status': 'R0_MISSING'}
    out['r0_parity'] = r0
    for b, d in B.items():
        r = d['result']
        e = {'phase_a': phase_a(d), 'hard_gate_failures': hard_gate_failures(b, d, order),
             'last_update': d['last_update'], 'status': r['status'] if r else None,
             'amp_retries': len(r['amp_retries']) if r else None,
             'amp_retry_updates': sorted({x['global_update_attempted'] for x in r['amp_retries']}) if r else None,
             'schedule': d['schedule'], 'eventual_lambda_adv': lambda_adv(b, STAGE2_LAST)}
        if b == 'R0':
            e['hard_gate_failures'] = [x for x in e['hard_gate_failures'] if x != 'declared identity collapse']
            e['role'] = 'CONTROL (must collapse)'
        if 'epochs' in d:
            pre = post_ramp_epochs(d)
            meds = {ep: d['epochs'][ep]['A_mean']['p50'] for ep in pre}
            e['post_ramp_epochs'] = pre
            e['post_ramp_epoch_median_A_mean'] = meds
            e['artifact_retained'] = bool(pre) and all(v >= c['artifact_rule']['reference_floor'] for v in meds.values()) \
                and d['collapse_rederived'] is None
            e['long_horizon_reached'] = d['last_update'] == STAGE2_LAST
        if b != 'R0':
            long_ok = e.get('long_horizon_reached', False) and not e['hard_gate_failures']
            e['qualification'] = ('QUALIFIED' if long_ok and e.get('artifact_retained') else
                                  'HARD_GATES_PASSED_ARTIFACT_NOT_RETAINED' if long_ok else
                                  'REJECTED' if e['hard_gate_failures'] and 'long' in d else
                                  'COLLAPSED_IN_PHASE_A' if e['phase_a']['result'] == 'COLLAPSED' else
                                  'NOT_LONG_QUALIFIED')
        out['branches'][b] = e
    labels = {b: BRANCHES[b]['name'] + '_QUALIFIED' for b in ('R1', 'R2', 'R3', 'R4')}
    qualified = [b for b in ('R1', 'R2', 'R3', 'R4') if out['branches'].get(b, {}).get('qualification') == 'QUALIFIED']
    out['qualified'] = {b: labels[b] for b in qualified}
    out['primary_repair_candidate'] = qualified[0] if qualified and r0['status'] == 'R0_PARITY_PASS' else None
    if r0['status'] != 'R0_PARITY_PASS':
        out['verdict'] = 'M7D1_N4_QUALIFICATION_BLOCKED'
    elif out['primary_repair_candidate']:
        out['verdict'] = 'M7D1_N4_REPAIR_CANDIDATE_QUALIFIED'
    else:
        out['verdict'] = 'M7D1_N4_NO_REPAIR_QUALIFIED'
    out['gradient_summary'] = gradient_summary(c)
    return out


def _fid(dct):
    return sum(dct.get(k, 0.0) or 0.0 for k in FIDELITY_TERMS)


def gradient_rows(c):
    rows = []
    for b, d in sorted(c['branches'].items()):
        for u, a in sorted(d.get('attribution', {}).items(), key=lambda kv: int(kv[0])):
            gs, st, ld = a['grad_projection_share'], a['step_trunk_projection_share'], a['lin_dlogit_mean']
            row = {'branch': b, 'global_update': int(u), 'lambda_adv': a['curriculum']['lambda_adv'],
                   'M_mean': a['M_mean'], 'mask_logit_mean': a['mask_logit_mean'], 'A_mean': a['A_mean'],
                   'D_real_logit_mean': a['D_real_logit_mean'], 'D_fake_logit_mean': a['D_fake_logit_mean'],
                   'G_grad_l2_preclip': a['G_grad_l2_preclip'], 'clip_coefficient': a['clip_coefficient'],
                   'attribution_vs_recorded_rel': a['attribution_vs_recorded_rel'],
                   'momentum_l2_vs_current_l2': a['momentum_l2_vs_current_l2']}
            for k in ('gadv', 'artcon', 'spec', 'budget', 'tv'):
                row[f'grad_share_{k}'] = gs.get(k)
                row[f'trunk_step_share_{k}'] = st.get(k)
                row[f'lin_dlogit_{k}'] = ld.get(k)
            row['grad_share_fidelity'] = _fid(gs)
            row['trunk_step_share_fidelity'] = _fid(st)
            row['lin_dlogit_fidelity'] = _fid(ld)
            row['trunk_step_share_momentum'] = st.get('momentum')
            row['lin_dlogit_momentum'] = ld.get('momentum')
            row['lin_dlogit_TOTAL'] = ld.get('TOTAL')
            rows.append(row)
    return rows


GRAD_FIELDS = ('branch', 'global_update', 'lambda_adv', 'M_mean', 'mask_logit_mean', 'A_mean', 'D_real_logit_mean',
               'D_fake_logit_mean', 'G_grad_l2_preclip', 'clip_coefficient', 'attribution_vs_recorded_rel',
               'momentum_l2_vs_current_l2',
               *[f'{p}_{k}' for k in ('gadv', 'artcon', 'spec', 'budget', 'tv', 'fidelity')
                 for p in ('grad_share', 'trunk_step_share', 'lin_dlogit')],
               'trunk_step_share_momentum', 'lin_dlogit_momentum', 'lin_dlogit_TOTAL')


def gradient_summary(c):
    rows = gradient_rows(c)
    by = {}
    for r in rows:
        by.setdefault(r['branch'], {})[r['global_update']] = {
            'lambda_adv': r['lambda_adv'], 'grad_share_gadv': r['grad_share_gadv'],
            'trunk_step_share_gadv': r['trunk_step_share_gadv'], 'lin_dlogit_gadv': r['lin_dlogit_gadv'],
            'lin_dlogit_TOTAL': r['lin_dlogit_TOTAL'], 'M_mean': r['M_mean']}
    return by


def screen_rows(c):
    rows = []
    for b, d in sorted(c['branches'].items()):
        for r in d['screen']:
            rows.append({'branch': b, **r})
    return rows


SCREEN_FIELDS = ('branch', 'global_update', 'epoch', 'group', *SCREEN_KEYS, *('L_' + k for k in LOSS_KEYS))
LONG_EPOCH_STATS = ('M_mean', 'M_p10', 'frac_M_lt_1e-2', 'sigmoid_deriv_mean', 'A_mean', 'A_p90',
                    'abs_xhat_minus_target_mean', 'tanh_hf_absmean', 'M_times_abs_delta_HF_mean', 'G_grad_norm',
                    'D_grad_norm', 'D_total', 'D_real', 'D_fake', 'gadv', 'budget', 'artcon', 'id', 'lm', 'bg',
                    'G_total')


def long_rows(c):
    rows = []
    for b, d in sorted(c['branches'].items()):
        for e, s in sorted(d.get('epochs', {}).items(), key=lambda kv: int(kv[0])):
            row = {'branch': b, 'epoch': int(e), 'first_update': s['updates'][0], 'last_update': s['updates'][1],
                   'updates': s['updates'][2], 'lambda_adv_first': s['lambda_adv'][0],
                   'lambda_adv_last': s['lambda_adv'][1], 'frac_G_grad_norm_gt_1': s['frac_G_grad_norm_gt_1'],
                   'amp_retry_updates': s['amp_retry_updates'], 'frac_budget_active': s['frac_budget_active'],
                   'frac_M_mean_lt_0.01': s['frac_M_mean_lt_0.01']}
            for k in LONG_EPOCH_STATS:
                for q in ('min', 'p10', 'p50', 'p90', 'max'):
                    row[f'{k}_{q}'] = (s[k] or {}).get(q)
            rows.append(row)
    return rows


LONG_FIELDS = ('branch', 'epoch', 'first_update', 'last_update', 'updates', 'lambda_adv_first', 'lambda_adv_last',
               'frac_G_grad_norm_gt_1', 'amp_retry_updates', 'frac_budget_active', 'frac_M_mean_lt_0.01',
               *[f'{k}_{q}' for k in LONG_EPOCH_STATS for q in ('min', 'p10', 'p50', 'p90', 'max')])


def finalize(collected_path):
    raw = Path(collected_path).read_bytes()
    c = json.loads(raw)
    doc = build_qualification(c)
    doc['source_collected_sha256'] = sha(raw)
    return {'QUALIFICATION.json': json.dumps(doc, indent=1, sort_keys=True) + '\n',
            'SCREEN.csv': n3.render_csv(screen_rows(c), SCREEN_FIELDS),
            'LONG.csv': n3.render_csv(long_rows(c), LONG_FIELDS),
            'GRADIENTS.csv': n3.render_csv(gradient_rows(c), GRAD_FIELDS)}


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--scenario', choices=('branch', 'analyze', 'collect'), required=True)
    ap.add_argument('--branch', choices=sorted(BRANCHES))
    ap.add_argument('--out', required=True)
    ap.add_argument('--end', type=int, default=None, help='branch: last update (default per branch)')
    ap.add_argument('--updates', default=None, help='branch: snapshot updates; analyze: updates to attribute')
    ap.add_argument('--name', default=None)
    ap.add_argument('--source', default=None, help='analyze: snapshot directory name (default branch_<RX>)')
    args = ap.parse_args()
    out_path = Path(args.out).resolve()
    assert str(out_path).startswith(str(n4_root()) + '/') or args.scenario == 'collect', out_path
    if args.scenario != 'collect':
        _check_launch()
    fw = install_write_firewall()
    t0 = time.monotonic()
    if args.scenario == 'collect':
        doc = scenario_collect(args)
        doc['write_firewall_collect'] = fw
        doc['collect_tool_sha256'] = file_sha(__file__)
        out_path.write_text(json.dumps(doc, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n')
        print('collect: branches', sorted(doc['branches']), flush=True)
        return
    out = {'branch': scenario_branch, 'analyze': scenario_analyze}[args.scenario](args)
    out['write_firewall'] = fw
    out['seconds_total'] = round(time.monotonic() - t0, 1)
    out['code_head'] = n3.git_head()
    out['tool_sha256'] = file_sha(__file__)
    out['tool_sha256_at_start'] = TOOL_SHA256_AT_START
    out['n3_tool_sha256'] = file_sha(ROOT / 'tools' / 'm7d1_n3_collapse_root_cause.py')
    out_path.write_text(json.dumps(out, indent=1, sort_keys=True, default=str) + '\n')
    print(f'{args.scenario} {args.branch}: {out.get("status", "done")} in {out["seconds_total"]} s', flush=True)


if __name__ == '__main__':
    main()
