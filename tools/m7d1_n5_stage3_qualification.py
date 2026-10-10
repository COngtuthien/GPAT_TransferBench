"""M7D1-N5: GPAT-B0 / E08 / seed 42 STAGE-3 ADVERSARIAL-TRANSITION QUALIFICATION.
DIAGNOSTIC_ONLY -- not a scientific run, not a protocol amendment, not an approved repair. Every branch is a diagnostic
variant; nothing here changes a production config, the frozen specification, the scientific run root, or the N3 / N4
diagnostic roots.

Context: N3 (d88563a) localized the seed-42 identity collapse to lambda_adv 0 -> 0.05 at u5526 against an
overconfident D (MASK_SIGMOID_SATURATION via the shared G_res trunk). N4 (6fc875e) qualified the diagnostic stage-2
repair R1 (lambda_adv = 0.05 * ((u - 5526) / (6630 - 5526)) on u5526..u6630, then 0.05 to u16575). The frozen stage-3
boundary u16576 changes lambda_adv 0.05 -> 0.10 and s_hf 0.10 -> 0.15; N5 qualifies that second transition.

Exact start state. N4 saved PRE-update snapshots only; the last R1 one is N4 `snapshots/branch_R1/pre_update_16575.pt`
(sha256 N4_R1_SNAPSHOT_SHA256, state digest taken from the live R1 state at capture). The `anchor` scenario restores it,
re-executes the single update u16575 under the R1 curriculum (lambda_adv 0.05 == frozen stage 2), FAILS CLOSED unless
the u16575 optimizer record, AMP-retry events, N3 mask row and N4 per-update row are bitwise equal to N4 R1, runs the
epoch-15 end exactly as the runner does (recovery save / EMA candidate intercepted, never written), draws the u16576
group and captures `snapshots/anchor/pre_update_16576.pt` WITHOUT stepping it. Every branch starts from that snapshot,
i.e. from the exact R1 state after u16575 (position epoch 16, next_group 1, global_update 16575).

Branches (only the listed stage-3 terms differ; LR, lambda_con, lambda_spec, D updates, AMP policy, optimizers,
clipping, EMA and TRAIN order are the frozen scientific ones):
  S0  ORIGINAL_STAGE3       frozen stage 3 from u16576: s_hf 0.15, lambda_adv 0.10 (unpatched runtime_contract)
  S1  ADV_RAMP_1E           s_hf 0.15; lambda_adv = 0.05 + 0.05 * ((u - 16576) / (17680 - 16576)) to u17680, then 0.10
  S2  ADV_RAMP_5E           s_hf 0.15; lambda_adv = 0.05 + 0.05 * ((u - 16576) / (22100 - 16576)) to u22100, then 0.10
  S3  ONLY_S_HF (causal)    s_hf 0.15, lambda_adv held 0.05 -- short screen only, NOT a protocol candidate
  S4  ONLY_ADV_JUMP (causal) lambda_adv 0.10, s_hf held 0.10 -- short screen only, NOT a protocol candidate
Phase A: u16576..u16900 for S0..S4. Phase B: S0, S1, S2 continue in the SAME process to u27625 (no restart).

Scenarios (fresh process each, gpat-m7-gpu, PYTHONHASHSEED=42 + rio.LAUNCH_ENVIRONMENT, short TMPDIR):
  anchor                           exact R1 post-u16575 state (see above)
  branch   --branch SX [--end U]   u16576..U from the anchor snapshot; per-update rows; pre-update snapshots
  analyze  --branch SX --updates   N3 per-loss attribution + N4 Adam-step / shared-trunk split on branch snapshots
  collect                          CPU/stdlib: compact collected document from the N5 diagnostic root
Writes go only to <runtime_root>/diagnostics/m7/M7D1_N5_seed42/ (and TMPDIR); an audit hook refuses writes under
<runtime_root>/runs, the N3 and N4 diagnostic roots and the repository configs / docs / methods.
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


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'tools' / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


n4 = _load('m7d1_n4_repair_qualification', 'm7d1_n4_repair_qualification.py')
n3 = n4.n3
N3_PARTS, N4_PARTS = n4.N3_PARTS, n4.N4_PARTS
N5_PARTS = ('diagnostics', 'm7', 'M7D1_N5_seed42')
N4_SNAP_KIND = n4.N4_SNAP_KIND
N5_SNAP_KIND = 'GPAT_M7D1_N5_DIAGNOSTIC_PRE_UPDATE_SNAPSHOT'
SNAP_LABELS = list(n3.LABELS)            # snapshot payloads carry the N3 label list (as in N4)
n3.DIAG_PARTS = N5_PARTS                 # every N3 helper (DiagContext, save_snapshot, ...) now writes under N5 only
n3.SNAP_KIND = N5_SNAP_KIND

METHOD, SEED, SCI_RUN_ID = n3.METHOD, n3.SEED, n3.SCI_RUN_ID
LABELS = n3.LABELS + ['DIAGNOSTIC_REPAIR_VARIANT', 'NOT_AN_APPROVED_PROTOCOL_CHANGE', 'STAGE3_TRANSITION_QUALIFICATION']
AUTHORITY_COMMIT = '6fc875ef1c63f0d512e877643e8005879eb8c0cc'
N4_AUTHORITY_COMMIT = n4.AUTHORITY_COMMIT                       # code head of the N4 branches (d88563a)
N4_R1_SNAPSHOT = 'snapshots/branch_R1/pre_update_16575.pt'      # relative to the N4 diagnostic root
N4_R1_SNAPSHOT_SHA256 = 'ea26d463fa67b1adbfe70d56f997da1e32dea394a536abe2d9156e22f741a612'
N4_R1_POSITION = {'stage': 'generator', 'epoch': 15, 'next_group': 1105, 'global_update': 16574}
ANCHOR_SNAPSHOT = 'snapshots/anchor/pre_update_16576.pt'        # relative to the N5 diagnostic root
ANCHOR_POSITION = {'stage': 'generator', 'epoch': 16, 'next_group': 1, 'global_update': 16575}
REPLAY_U, START, PHASE_A_END, PHASE_B_END = 16575, 16576, 16900, 27625
LAMBDA2, LAMBDA3, S_HF2, S_HF3 = 0.05, 0.10, 0.10, 0.15
MASK = n3.MASK
BRANCHES = {
    'S0': {'kind': 'CONTROL_FROZEN_STAGE3', 'name': 'ORIGINAL_STAGE3', 'protocol_candidate': True},
    'S1': {'kind': 'LINEAR_RAMP', 'name': 'ADV_RAMP_1E', 'ramp_end': 17680, 'protocol_candidate': True},
    'S2': {'kind': 'LINEAR_RAMP', 'name': 'ADV_RAMP_5E', 'ramp_end': 22100, 'protocol_candidate': True},
    'S3': {'kind': 'ONLY_S_HF', 'name': 'ONLY_S_HF_CHANGE', 'protocol_candidate': False},
    'S4': {'kind': 'ONLY_ADV_JUMP', 'name': 'ONLY_ADV_JUMP', 'protocol_candidate': False},
}
LONG_BRANCHES = ('S0', 'S1', 'S2')
SHORT_BRANCHES = ('S3', 'S4')
DEFAULT_END = {'S0': PHASE_B_END, 'S1': PHASE_B_END, 'S2': PHASE_B_END, 'S3': PHASE_A_END, 'S4': PHASE_A_END}
SCREEN_ATTR = (16576, 16577, 16580, 16600, 16700, 16900)
ATTRIBUTION_UPDATES = {
    'S0': SCREEN_ATTR + (17680, 17681, 22100, 22101, 27625),
    'S1': SCREEN_ATTR + (17680, 17681, 22100, 22101, 27625),
    'S2': SCREEN_ATTR + (17680, 17681, 22100, 22101, 27625),
    'S3': SCREEN_ATTR, 'S4': SCREEN_ATTR,
}

# ----------------------------------------------------------------------------- predeclared rules (before any result)
COLLAPSE_RULE = {
    'name': 'N5_IDENTITY_COLLAPSE_V1',
    'metric': 'M_mean: mean of M = sigmoid(float(mask_logit)) over every mask pixel of every sample of the accepted '
              '(COMPLETE) forward of update u (sample-weighted over the microbatches of the group)',
    'threshold': 0.01, 'consecutive_complete_updates': 10,
    'onset_update': 'first update of the first run of 10 consecutive updates with M_mean < 0.01',
    'event_update': 'the 10th update of that run (first update at which the 10-update condition holds)',
    'declared': 'M7D1-N5 instruction section 6, implemented before any N5 branch was run; identical to '
                'N4_IDENTITY_COLLAPSE_V1',
}
ARTIFACT_RULE = {
    'name': 'N5_ARTIFACT_RETENTION_CHARACTERIZATION_V1',
    'metric': 'per-update group mean of A, epoch median', 'reference_floor': 0.01,
    'reference_floor_source': 'frozen budget hinge ARTIFACT_MIN (A10 D10.6, methods/gpat/losses.py); same rule as N4',
    'retained': 'every complete stage-3 epoch of the branch window (epochs 16..25) has median group-mean A >= 0.01 and '
                'no declared collapse',
    'declared': 'before any N5 branch was run; characterization used for preference criterion 3 only',
}
OPEN_MASK_NOTE = ('open-mask saturation (M -> 1, sigmoid derivative -> 0) is characterized and reported; no pass/fail '
                  'threshold is attached to it (instruction section 13)')
HARD_GATES = ('N5 formal identity collapse', 'non-finite loss', 'non-finite parameter', 'terminal AMP overflow',
              'corrupted optimizer/scaler state', 'TRAIN order mismatch', 'curriculum/LR mismatch',
              'provenance mismatch', 'write-firewall violation', 'VAL/TEST access', 'scientific-root mutation')
PREFERENCE = ('all hard gates pass', 'no closed-mask collapse', 'meaningful artifact output remains',
              'original eventual lambda_adv = 0.10 restored', 'smallest deviation from the frozen curriculum',
              'no architecture change')
DECISION_ORDER = (('S0', 'STAGE3_ORIGINAL_TRANSITION_QUALIFIED'), ('S1', 'STAGE3_ADV_RAMP_1E_QUALIFIED'),
                  ('S2', 'STAGE3_ADV_RAMP_5E_QUALIFIED'))
COLLAPSE_MARGIN = 50                      # a branch stops 50 updates after a declared collapse event
FIDELITY_TERMS = n4.FIDELITY_TERMS


def lambda_adv(branch, u):
    """Diagnostic lambda_adv of generator update u (stage 3 window only)."""
    if not START <= u <= DEFAULT_END[branch]:
        raise ValueError(f'N5 branch {branch} is defined on u{START}..u{DEFAULT_END[branch]} only, got u{u}')
    b = BRANCHES[branch]
    if b['kind'] in ('CONTROL_FROZEN_STAGE3', 'ONLY_ADV_JUMP'):
        return LAMBDA3
    if b['kind'] == 'ONLY_S_HF':
        return LAMBDA2
    end = b['ramp_end']
    return LAMBDA2 + LAMBDA2 * ((u - START) / (end - START)) if u <= end else LAMBDA3


def s_hf(branch, u):
    if not START <= u <= DEFAULT_END[branch]:
        raise ValueError(f'N5 branch {branch} is defined on u{START}..u{DEFAULT_END[branch]} only, got u{u}')
    return S_HF2 if BRANCHES[branch]['kind'] == 'ONLY_ADV_JUMP' else S_HF3


def branch_curriculum(branch, real):
    """real(u) with only s_hf / lambda_adv replaced on the branch window; S0 returns real(u) unchanged."""
    def curriculum(u):
        c = real(u)
        if u < START or branch == 'S0':
            return c
        if c['stage'] != 3:
            raise AssertionError(f'N5 branch {branch} outside the frozen stage-3 window at u{u}')
        return dict(c, s_hf=s_hf(branch, u), lambda_adv=lambda_adv(branch, u))
    curriculum.n5_branch = branch
    return curriculum


def proposed_schedule(stage3_branch):
    """Complete diagnostic lambda_adv(u) for u1..u66300: frozen stage 1, N4-R1 stage-2 ramp, the chosen stage-3 rule."""
    def f(u):
        if not 1 <= u <= 66300:
            raise ValueError(u)
        if u <= 5525:
            return 0.0
        if u <= 16575:
            return n4.lambda_adv('R1', u)
        b = BRANCHES[stage3_branch]
        if b['kind'] == 'LINEAR_RAMP' and u <= b['ramp_end']:
            return LAMBDA2 + LAMBDA2 * ((u - START) / (b['ramp_end'] - START))
        return LAMBDA3
    return f


def collapse_event(rows, rule=COLLAPSE_RULE):
    return n4.collapse_event(rows, rule)


sha = n4.sha
file_sha = n3.file_sha
TOOL_SHA256_AT_START = file_sha(__file__)


def runtime_root():
    return n3.runtime_root()


def n5_root():
    return runtime_root().joinpath(*N5_PARTS)


def n4_root():
    return runtime_root().joinpath(*N4_PARTS)


def n3_root():
    return runtime_root().joinpath(*N3_PARTS)


# ----------------------------------------------------------------------------- write firewall
def install_write_firewall():
    rt = str(runtime_root().resolve())
    allowed = [str(n5_root()) + '/', os.environ.get('TMPDIR', '/nonexistent'), '/dev/shm', '/dev/null', '/proc/']
    forbidden = [rt + '/runs', str(n3_root()), str(n4_root()), str(ROOT / 'configs'), str(ROOT / 'docs'),
                 str(ROOT / 'methods')]
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
        if any(p == f or p.startswith(f + '/') for f in forbidden):
            record['denied'].append(p)
            raise PermissionError(f'M7D1-N5 firewall: write refused (scientific / N3 / N4 / repo root): {p}')
        if not any(p.startswith(a) for a in allowed):
            record['outside_allowed'].append(p)
    sys.addaudithook(hook)
    return record


# ----------------------------------------------------------------------------- per-forward statistics
class CoreHook(n4.CoreHook):
    """N4 artifact statistics (unchanged keys and code) plus open-mask characterization."""

    def take(self, n_mb):
        import torch
        b = self.buf[-n_mb:]
        M = torch.cat([x['M'] for x in b])
        L = torch.cat([x['logit'] for x in b])
        per_img = M.flatten(1)
        lq = torch.quantile(L.flatten(), torch.tensor([0.01, 0.5, 0.99], device=L.device)).tolist()
        extra = {'frac_M_gt_0.9': float((M > 0.9).float().mean()), 'frac_M_gt_0.99': float((M > 0.99).float().mean()),
                 'frac_M_gt_0.999': float((M > 0.999).float().mean()),
                 'M_spatial_var_mean': float(per_img.var(dim=1).mean()),
                 'M_image_mean_min': float(per_img.mean(dim=1).min()),
                 'M_image_mean_max': float(per_img.mean(dim=1).max()),
                 'logit_p01': lq[0], 'logit_p50': lq[1], 'logit_p99': lq[2]}
        out = super().take(n_mb)
        out.update(extra)
        return out


class DHook:
    """Forward hook on the discriminator. Per microbatch the step calls D three times in this order (runner.py):
    D(x_hat) for gadv, D(x_source) real, D(x_hat.detach()) fake. Read-only, returns None."""
    KEYS = ('gadv_input', 'real', 'fake')

    def __init__(self, tr):
        self.buf = []
        tr.core.discriminator.register_forward_hook(self)

    def __call__(self, module, inputs, out):
        import torch
        with torch.no_grad():
            z = out.detach().float()
            self.buf.append({'mean': float(z.mean()), 'absmean': float(z.abs().mean()),
                             'sig_mean': float(torch.sigmoid(z).mean()), 'n': int(z.shape[0])})
        return None

    def take(self, n_mb):
        b = self.buf[-3 * n_mb:]
        self.buf = []
        assert len(b) == 3 * n_mb, f'expected {3 * n_mb} D forwards, got {len(b)}'
        out = {}
        for i, k in enumerate(self.KEYS):
            calls = b[i::3]
            w = [c['n'] for c in calls]
            for f in ('mean', 'absmean', 'sig_mean'):
                out[f'Dlogit_{k}_{f}'] = sum(c[f] * n for c, n in zip(calls, w)) / sum(w)
        return out


class N5Wrapper(n3.StepWrapper):
    """N3 StepWrapper (pre-update snapshots + N3 mask hook; never changes the wrapped step) plus the N4 per-update row,
    open-mask and D-logit statistics, the declared collapse tracker and the post-collapse stop."""

    def __init__(self, tr, snap_dir, snap_updates, branch, rows_path, stop):
        super().__init__(tr, snap_dir, snap_updates, None)
        self.branch, self.stop = branch, stop
        self.core_hook = CoreHook(tr)
        self.d_hook = DHook(tr)
        self.rows_fh = open(rows_path, 'a', encoding='utf-8')
        self.m_rows = []
        self.collapse = None

    def __call__(self, group, u):
        epoch, g = self.tr.position['epoch'], self.tr.position['next_group']
        idx = [mb['index'].tolist() for mb in group]
        self.core_hook.buf = []
        self.d_hook.buf = []
        rec = super().__call__(group, u)
        ext = self.core_hook.take(len(group))
        dext = self.d_hook.take(len(group))
        n3row = self.mask_rows[-1]
        tl = rec['train_losses']
        row = {'global_update': u, 'epoch': epoch, 'group': g, 'branch': self.branch,
               'group_indices_sha256': hashlib.sha256(json.dumps(idx).encode()).hexdigest(),
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
               **ext, **dext}
        self.rows_fh.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
        self.rows_fh.flush()
        if rec['group_status'] == 'COMPLETE':
            self.m_rows.append((u, ext['M_mean']))
            if self.collapse is None:
                self.collapse = collapse_event(self.m_rows[-COLLAPSE_RULE['consecutive_complete_updates']:])
                if self.collapse is not None:
                    print(f'N5 {self.branch}: declared collapse onset u{self.collapse["onset_update"]} '
                          f'event u{self.collapse["event_update"]}', flush=True)
        if self.collapse is not None and u >= self.collapse['event_update'] + COLLAPSE_MARGIN:
            self.stop.requested = True
        return rec


class AnchorCaptured(Exception):
    pass


class AnchorWrapper(N5Wrapper):
    """Steps u16575 exactly like N4 R1; at u16576 saves the pre-update snapshot and stops WITHOUT stepping."""

    def __call__(self, group, u):
        if u == START:
            self.snaps.append(n3.save_snapshot(n3.snapshot_payload(self.tr, group, u), self.snap_dir, u))
            raise AnchorCaptured()
        assert u == REPLAY_U, u
        return super().__call__(group, u)


def _patch_failure_recording(tr):
    def record_failure(exc, *, stage, epoch, attempted, role):
        tr.ctx.log('numerical_failure', {**exc.record, 'stage': stage, 'epoch': epoch,
                                         'attempted_group_or_batch': attempted, 'position_unchanged': dict(tr.position),
                                         'diagnostic': 'N5 branch stops; no recovery exists for diagnostic branches'})
    tr.record_failure = record_failure


def _check_launch():
    tmp = os.environ.get('TMPDIR', '')
    assert tmp and len(tmp) <= 24, 'launch with a short TMPDIR (AF_UNIX path length), e.g. /tmp/gpat_n5'
    for k in ('TMP', 'TEMP'):
        assert os.environ.get(k) == tmp, f'{k} must equal TMPDIR'


def _metrics(path):
    recs, retries, other = {}, [], []
    for line in Path(path).read_text().splitlines():
        d = json.loads(line)
        if d['record_type'] == 'optimizer_group':
            recs[d['global_update']] = d
        elif d['record_type'] == 'amp_retry':
            retries.append(d)
        else:
            other.append(d)
    return recs, retries, other


def _setup(name, extra):
    tr = n3.setup(name, {'labels': LABELS, 'collapse_rule': COLLAPSE_RULE, **extra})
    tr.suppressed_candidates = []
    tr.write_candidate = lambda epoch: tr.suppressed_candidates.append(epoch)
    _patch_failure_recording(tr)
    return tr


# ----------------------------------------------------------------------------- anchor (exact R1 state after u16575)
def load_n4_r1():
    from methods.gpat import runner_checkpoint as ck
    path = n4_root() / N4_R1_SNAPSHOT
    meta = json.loads(path.with_suffix('.json').read_text())
    got = file_sha(path)
    assert got == N4_R1_SNAPSHOT_SHA256 == meta['sha256'], f'N4 R1 snapshot hash {got}'
    r1 = json.loads((n4_root() / 'results' / 'branch_R1.json').read_text())
    listed = [s for s in r1['snapshots'] if s['global_update_attempted'] == REPLAY_U]
    assert len(listed) == 1 and listed[0]['sha256'] == got and listed[0]['state_digest'] == meta['state_digest']
    assert r1['status'] == 'COMPLETED_TO_END' and r1['last_completed_update'] == REPLAY_U
    assert r1['code_head'] == N4_AUTHORITY_COMMIT and meta['code_head'] == N4_AUTHORITY_COMMIT
    snap = ck.load(path)
    assert snap['kind'] == N4_SNAP_KIND and snap['labels'] == SNAP_LABELS
    assert snap['global_update_attempted'] == REPLAY_U
    assert {k: snap['position'][k] for k in N4_R1_POSITION} == N4_R1_POSITION, snap['position']
    assert snap['ema'] is not None
    return snap, {'path': str(path), 'n4_relative_path': N4_R1_SNAPSHOT, 'sha256': got, 'size': path.stat().st_size,
                  'state_digest': meta['state_digest'], 'code_head_at_capture': meta['code_head'],
                  'group_indices': meta['group_indices'], 'kind': snap['kind'],
                  'position': {k: snap['position'][k] for k in snap['position']},
                  'n4_branch_R1_result_sha256': file_sha(n4_root() / 'results' / 'branch_R1.json'),
                  'n4_branch_R1_tool_sha256': r1['tool_sha256']}


def scenario_anchor(args):
    import torch
    tr = _setup('anchor', {'runner_mode': 'DIAGNOSTIC_EXACT_R1_STATE_AFTER_U16575',
                           'purpose': 'replay N4 R1 u16575 from its pre-update snapshot (bitwise-verified) and capture '
                                      'the pre-u16576 state', 'n4_r1_snapshot_sha256': N4_R1_SNAPSHOT_SHA256})
    snap, base = load_n4_r1()
    n3.apply_snapshot(tr, snap)                                 # asserts the N4 capture-time state digest
    del snap
    assert tr.step.ema is tr.ema and tr.ema is not None
    assert tr.position['global_update'] == REPLAY_U - 1
    stop = n3.Stop()
    w = AnchorWrapper(tr, n5_root() / 'snapshots' / 'anchor', [START], 'ANCHOR_R1', tr.ctx.run_dir / 'per_update.jsonl',
                      stop)
    tr.step = w
    captured = False
    try:
        tr.run_generator_groups(stop, limit=2)
    except AnchorCaptured:
        captured = True
    w.rows_fh.close()
    n3.write_jsonl(tr.ctx.run_dir / 'mask_per_update.jsonl', w.mask_rows)
    recs, retries, other = _metrics(tr.ctx.run_dir / 'metrics.jsonl')
    parity = anchor_parity(recs, retries, w.mask_rows)
    epoch_recs = [d for d in other if d['record_type'] == 'epoch']
    checks = {'captured': captured, 'position_after_u16575': dict(tr.position),
              'position_ok': {k: tr.position[k] for k in ANCHOR_POSITION} == ANCHOR_POSITION,
              'epoch15_end_logged': [d['epoch'] for d in epoch_recs] == [15],
              'suppressed_recovery_saves': tr.suppressed_recovery_saves,
              'suppressed_candidate_epochs': tr.suppressed_candidates,
              'epoch_end_side_effects_intercepted': tr.suppressed_recovery_saves == [REPLAY_U]
              and tr.suppressed_candidates == [15]}
    ok = captured and checks['position_ok'] and checks['epoch15_end_logged'] and \
        checks['epoch_end_side_effects_intercepted'] and parity['all_bitwise']
    snaps = w.snaps
    assert len(snaps) == (1 if captured else 0)
    res = {'scenario': 'anchor', 'base_n4_r1_snapshot': base, 'replayed_update': REPLAY_U, 'parity': parity,
           'checks': checks, 'status': 'ANCHOR_EXACT' if ok else 'ANCHOR_PARITY_FAIL',
           'anchor_snapshot': snaps[0] if snaps else None, 'access': tr.access.report(),
           'peak_reserved_GiB': round(torch.cuda.max_memory_reserved() / 2 ** 30, 3)}
    tr.ctx.close('diagnostic_completed' if ok else 'ANCHOR_PARITY_FAIL', {'access': res['access'],
                                                                          'status': res['status']})
    return res


def anchor_parity(recs, retries, mask_rows):
    """u16575 replay vs N4 branch_R1 (read-only): optimizer record, AMP retries, N3 mask row, N4 per-update row."""
    d4 = n4_root() / 'branch_R1'
    r4, ret4, _ = _metrics(d4 / 'metrics.jsonl')
    mine = recs.get(REPLAY_U)
    rec_diff = n3.diff_fields(mine, r4[REPLAY_U]) if mine else ['<missing>']
    rm = [n3.strip(r) for r in retries if r['global_update_attempted'] == REPLAY_U]
    r4m = [n3.strip(r) for r in ret4 if r['global_update_attempted'] == REPLAY_U]
    m4 = [d for d in _jsonl(d4 / 'mask_per_update.jsonl') if d['global_update'] == REPLAY_U]
    mm = [r for r in mask_rows if r['global_update'] == REPLAY_U]
    mask_equal = len(mm) == 1 and len(m4) == 1 and json.dumps(mm[0], sort_keys=True) == json.dumps(m4[0],
                                                                                                    sort_keys=True)
    p4 = [d for d in _jsonl(d4 / 'per_update.jsonl') if d['global_update'] == REPLAY_U]
    p4 = p4[0] if len(p4) == 1 else None
    mine_rows = [json.loads(x) for x in (n5_root() / 'anchor' / 'per_update.jsonl').read_text().splitlines()]
    pm = [r for r in mine_rows if r['global_update'] == REPLAY_U]
    row_diff = ['<missing>'] if not (pm and p4) else [k for k in sorted(p4) if k != 'branch' and
                                                      json.dumps(pm[0].get(k)) != json.dumps(p4[k])]
    return {'record_field_mismatch': rec_diff, 'amp_retry_events_equal': rm == r4m, 'amp_retry_events': len(rm),
            'n3_mask_row_equal': mask_equal, 'n4_per_update_row_field_mismatch': row_diff,
            'n4_record_lambda_adv': r4[REPLAY_U]['lambda_adv'],
            'all_bitwise': not rec_diff and rm == r4m and mask_equal and not row_diff}


def load_anchor():
    res = json.loads((n5_root() / 'results' / 'anchor.json').read_text())
    assert res['status'] == 'ANCHOR_EXACT', 'anchor parity did not pass: fail closed'
    path = n5_root() / ANCHOR_SNAPSHOT
    snap, meta = n3.load_snapshot(path)                       # sha256 == meta, kind N5, N3 labels
    assert meta['sha256'] == res['anchor_snapshot']['sha256'] and meta['path'] == ANCHOR_SNAPSHOT
    assert snap['global_update_attempted'] == START
    assert {k: snap['position'][k] for k in ANCHOR_POSITION} == ANCHOR_POSITION, snap['position']
    assert snap['ema'] is not None
    return snap, {'path': str(path), 'n5_relative_path': ANCHOR_SNAPSHOT, 'sha256': meta['sha256'],
                  'size': path.stat().st_size, 'state_digest': meta['state_digest'], 'code_head': meta['code_head'],
                  'group_indices': meta['group_indices'],
                  'anchor_result_sha256': file_sha(n5_root() / 'results' / 'anchor.json')}


# ----------------------------------------------------------------------------- branches
def scenario_branch(args):
    import torch
    from methods.gpat import runner as R
    from methods.gpat import runtime_contract as rc
    branch = args.branch
    end = args.end or DEFAULT_END[branch]
    assert START < end <= DEFAULT_END[branch]
    assert end == DEFAULT_END[branch] or (args.name or '').startswith('smoke_'), 'a short branch must be named smoke_*'
    real = rc.curriculum
    if branch != 'S0':
        rc.curriculum = branch_curriculum(branch, real)          # this diagnostic process only; configs untouched
    name = args.name or f'branch_{branch}'
    tr = _setup(name, {'runner_mode': 'DIAGNOSTIC_STAGE3_QUALIFICATION_FROM_N4_R1_U16575',
                       'purpose': f'M7D1-N5 stage-3 branch {branch} u{START}..u{end}', 'branch': branch,
                       'branch_definition': BRANCHES[branch]})
    snap, base = load_anchor()
    n3.apply_snapshot(tr, snap)
    del snap
    assert tr.step.ema is tr.ema and tr.ema is not None, 'EMA must be live on the real step'
    assert tr.position['global_update'] == START - 1
    stop = n3.Stop()
    snaps = sorted(int(x) for x in args.updates.split(',')) if args.updates else ATTRIBUTION_UPDATES[branch]
    w = N5Wrapper(tr, n5_root() / 'snapshots' / name, [x for x in snaps if x <= end], branch,
                  tr.ctx.run_dir / 'per_update.jsonl', stop)
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
    recs, retries, _ = _metrics(tr.ctx.run_dir / 'metrics.jsonl')
    n3.write_jsonl(tr.ctx.run_dir / 'mask_per_update.jsonl', w.mask_rows)
    last = tr.position['global_update']
    nonfinite_params = [n for n, p in tr.core.named_parameters() if not bool(torch.isfinite(p).all())]
    res = {'branch': branch, 'definition': BRANCHES[branch], 'end_requested': end, 'last_completed_update': last,
           'status': status, 'failure': failure, 'collapse': w.collapse, 'start_snapshot': base,
           'records': len(recs), 'records_contiguous': sorted(recs) == list(range(START, last + 1)),
           'seconds_stepping': time.monotonic() - t0, 'nonfinite_parameters_at_end': nonfinite_params,
           'final_scalers': {'G_SCALER': tr.g_scaler.state_dict(), 'D_SCALER': tr.d_scaler.state_dict()},
           'amp_retries': [{k: r[k] for k in ('global_update_attempted', 'offending_optimizers', 'old_scale',
                                              'new_scale')} for r in retries],
           'snapshots': w.snaps, 'suppressed_recovery_saves': tr.suppressed_recovery_saves,
           'suppressed_candidate_epochs': tr.suppressed_candidates, 'access': tr.access.report(),
           'peak_reserved_GiB': round(torch.cuda.max_memory_reserved() / 2 ** 30, 3)}
    tr.ctx.close('diagnostic_completed' if status.startswith(('COMPLETED', 'STOPPED_AFTER')) else status,
                 {'access': res['access'], 'status': status})
    return res


# ----------------------------------------------------------------------------- attribution
def scenario_analyze(args):
    import torch
    from methods.gpat import runtime_contract as rc
    branch = args.branch
    src = args.source or f'branch_{branch}'
    ups = tuple(int(x) for x in args.updates.split(',')) if args.updates else ATTRIBUTION_UPDATES[branch]
    tr = _setup(args.name or f'analyze_{branch}', {'runner_mode': 'DIAGNOSTIC_STAGE3_ATTRIBUTION',
                                                   'purpose': f'N3 attribution on N5 {src} snapshots'})
    n3.adam_parts = n4._trunk_wrap(n3.adam_parts)
    have = {int(p.stem.split('_')[-1]) for p in (n5_root() / 'snapshots' / src).glob('pre_update_*.pt')}
    out = n3.analyze_snapshots(tr, n5_root() / 'snapshots' / src, tuple(u for u in ups if u in have),
                               branch_curriculum(branch, rc.curriculum), recorded=n3.recorded_groups(src))
    res = {'branch': branch, 'snapshots': out, 'requested': list(ups), 'missing': [u for u in ups if u not in have],
           'access': tr.access.report(), 'peak_reserved_GiB': round(torch.cuda.max_memory_reserved() / 2 ** 30, 3)}
    tr.ctx.close('diagnostic_completed', {'access': res['access']})
    return res


# ----------------------------------------------------------------------------- collect (CPU, stdlib + runtime_contract)
EXTRA_MASK_KEYS = ('frac_M_gt_0.9', 'frac_M_gt_0.99', 'frac_M_gt_0.999', 'M_spatial_var_mean', 'M_image_mean_min',
                   'M_image_mean_max', 'logit_p01', 'logit_p50', 'logit_p99')
D_KEYS = tuple(f'Dlogit_{k}_{f}' for k in DHook.KEYS for f in ('mean', 'absmean', 'sig_mean'))
SCREEN_KEYS = n4.SCREEN_KEYS + ('G_scale_before', 'D_scale_before') + EXTRA_MASK_KEYS + D_KEYS
LOSS_KEYS = n4.LOSS_KEYS
LONG_KEYS = n4.LONG_KEYS + ('scale_hf', 'frac_M_gt_0.99', 'M_spatial_var_mean', 'Dlogit_real_mean',
                            'Dlogit_fake_mean')
LONG_LOSS_KEYS = n4.LONG_LOSS_KEYS
SCI_ORDER_FIELDS = ('epoch', 'group', 'group_samples', 'microbatch_sizes', 'sample_weights', 'learning_rate',
                    'lambda_con', 'lambda_spec')
_jsonl, _q, _dist, _compact = n4._jsonl, n4._q, n4._dist, n4._compact


def schedule_verification(branch, rows, sci):
    """Every row: lambda_adv / s_hf == the branch definition exactly; LR, lambda_con, lambda_spec == runtime_contract
    (frozen stage 3); epoch / group / group sizes / weights / LR / lambda_con / lambda_spec == the scientific record of
    the same update (read-only)."""
    from methods.gpat import runtime_contract as rc
    bad_lambda, bad_shf, bad_rc, bad_sci = [], [], [], []
    for r in rows:
        u = r['global_update']
        if r['lambda_adv'] != lambda_adv(branch, u):
            bad_lambda.append(u)
        if r['scale_hf'] != s_hf(branch, u):
            bad_shf.append(u)
        c = rc.curriculum(u)
        if c['stage'] != 3 or (r['learning_rate'], r['lambda_con'], r['lambda_spec']) != (
                rc.main_lr(u), c['lambda_con'], c['lambda_spec']):
            bad_rc.append(u)
        s = sci.get(u)
        if s is None or r['_sci'] is None or any(json.dumps(r['_sci'][k]) != json.dumps(s[k])
                                                 for k in SCI_ORDER_FIELDS):
            bad_sci.append(u)
    ups = [r['global_update'] for r in rows]
    probe = (16576, 16577, 17128, 17679, 17680, 17681, 19338, 22099, 22100, 22101, 27625)
    return {'rows': len(rows), 'lambda_adv_exact_mismatch': bad_lambda[:50], 's_hf_exact_mismatch': bad_shf[:50],
            'lr_con_spec_vs_frozen_contract_mismatch': bad_rc[:50], 'order_and_lr_vs_scientific_mismatch': bad_sci[:50],
            'contiguous_from_16576': ups == list(range(START, START + len(ups))),
            'lambda_adv_at': {str(r['global_update']): r['lambda_adv'] for r in rows if r['global_update'] in probe},
            's_hf_at': {str(r['global_update']): r['scale_hf'] for r in rows if r['global_update'] in probe}}


def collect_branch(branch, sci):
    d = n5_root()
    res_path = d / 'results' / f'branch_{branch}.json'
    res = json.loads(res_path.read_text()) if res_path.exists() else None
    rows = _jsonl(d / f'branch_{branch}' / 'per_update.jsonl')
    recs, retries, _ = _metrics(d / f'branch_{branch}' / 'metrics.jsonl')
    for r in rows:                     # the branch's own optimizer record (group sizes / weights) for the order check
        r['_sci'] = {k: recs[r['global_update']][k] for k in SCI_ORDER_FIELDS} if r['global_update'] in recs else None
    ev = collapse_event([(r['global_update'], r['M_mean']) for r in rows if r['group_status'] == 'COMPLETE'])
    screen = [dict(global_update=r['global_update'], epoch=r['epoch'], group=r['group'],
                   **_compact(r, SCREEN_KEYS, LOSS_KEYS)) for r in rows if r['global_update'] <= PHASE_A_END]
    out = {'result': res, 'result_sha256': file_sha(res_path) if res else None,
           'per_update_sha256': file_sha(d / f'branch_{branch}' / 'per_update.jsonl'),
           'metrics_sha256': file_sha(d / f'branch_{branch}' / 'metrics.jsonl'),
           'rows': len(rows), 'last_update': rows[-1]['global_update'] if rows else None,
           'collapse_rederived': ev, 'schedule': schedule_verification(branch, rows, sci),
           'rows_equal_records': sorted(r['global_update'] for r in rows) == sorted(recs),
           'amp_retry_integrity': amp_retry_integrity(rows, retries),
           '_group_sha': {r['global_update']: r['group_indices_sha256'] for r in rows},
           'screen': screen, 'phase_a_summary': window_summary([r for r in rows if r['global_update'] <= PHASE_A_END]),
           'transition_first_rows': [dict(global_update=r['global_update'], **_compact(r, SCREEN_KEYS, LOSS_KEYS))
                                     for r in rows[:5]],
           'nonfinite_rows': [r['global_update'] for r in rows
                              if any(isinstance(v, float) and not math.isfinite(v)
                                     for v in list(r['losses'].values()) + [r['M_mean'], r['A_mean'], r['G_grad_norm'],
                                                                            r['D_grad_norm']])]}
    if branch in LONG_BRANCHES:
        keep = long_keep(branch, rows, ev)
        out['long_sampling'] = LONG_SAMPLING
        out['long'] = [dict(global_update=r['global_update'], epoch=r['epoch'],
                            **_compact(r, LONG_KEYS, LONG_LOSS_KEYS)) for r in rows if r['global_update'] in keep]
        by_ep = {}
        for r in rows:
            by_ep.setdefault(r['epoch'], []).append(r)
        out['epochs'] = {str(e): epoch_summary(rs) for e, rs in sorted(by_ep.items())}
    a_path = d / 'results' / f'analyze_{branch}.json'
    if a_path.exists():
        a = json.loads(a_path.read_text())
        out['attribution'] = n4.compact_attribution(a)
        out['attribution_missing'] = a.get('missing', [])
        out['attribution_sha256'] = file_sha(a_path)
        out['attribution_access'] = a['access']
        out['attribution_write_firewall'] = a.get('write_firewall')
    return out


def _halved(e):
    """One backoff event: scales are per-optimizer dicts over exactly the offending optimizers, each halved."""
    keys = sorted(e['offending_optimizers'])
    return sorted(e['new_scale']) == keys and sorted(e['old_scale']) == keys and \
        all(e['new_scale'][k] == e['old_scale'][k] / 2 for k in keys)


def amp_retry_integrity(rows, retries):
    """attempts == retries + 1 per update; each backoff halves; scales only fall at retry updates."""
    by = {}
    for r in retries:
        by.setdefault(r['global_update_attempted'], []).append(r)
    bad = []
    prev = None
    for r in rows:
        u = r['global_update']
        ev = by.get(u, [])
        if r['amp_attempts'] != len(ev) + 1 or not all(_halved(e) for e in ev):
            bad.append(u)
        if prev is not None and not ev and (r['G_scale_before'] < prev['G_scale'] or
                                            r['D_scale_before'] < prev['D_scale']):
            bad.append(u)
        prev = r
    return {'retry_events': len(retries), 'retry_updates': sorted(by), 'bad_updates': sorted(set(bad))[:50]}


LONG_SAMPLING = ('every 5th update from u16576, the first 30 updates, every AMP-retry update, u(ramp_end)-20..'
                 'u(ramp_end)+20 for S1/S2 ends 17680 and 22100, the last 20 updates, and collapse onset-30..event+30; '
                 'epoch summaries use every update')


def long_keep(branch, rows, ev):
    ups = [r['global_update'] for r in rows]
    keep = {u for u in ups if (u - START) % 5 == 0}
    keep |= set(range(START, START + 30))
    keep |= {r['global_update'] for r in rows if r['amp_attempts'] > 1}
    for e in (17680, 22100):
        keep |= set(range(e - 20, e + 21))
    keep |= set(ups[-20:])
    if ev:
        keep |= set(range(ev['onset_update'] - 30, ev['event_update'] + 31))
    return keep & set(ups)


def window_summary(rs):
    if not rs:
        return None
    f = lambda k: [r[k] for r in rs]                                   # noqa: E731
    fl = lambda k: [r['losses'].get(k) for r in rs]                    # noqa: E731
    return {'updates': [rs[0]['global_update'], rs[-1]['global_update'], len(rs)],
            'lambda_adv': [rs[0]['lambda_adv'], rs[-1]['lambda_adv']], 's_hf': [rs[0]['scale_hf'], rs[-1]['scale_hf']],
            **{k: _dist(f(k)) for k in ('M_mean', 'M_p01', 'M_p10', 'M_p50', 'frac_M_lt_1e-2', 'frac_M_gt_0.99',
                                        'M_spatial_var_mean', 'sigmoid_deriv_mean', 'logit_mean', 'A_mean', 'A_p90',
                                        'abs_xhat_minus_target_mean', 'tanh_hf_absmean', 'M_times_abs_delta_HF_mean',
                                        'G_grad_norm', 'D_grad_norm', 'Dlogit_real_mean', 'Dlogit_fake_mean',
                                        'Dlogit_real_sig_mean', 'Dlogit_fake_sig_mean')},
            **{k: _dist(fl(k)) for k in ('D_total', 'D_real', 'D_fake', 'gadv', 'budget', 'artcon', 'spec', 'id',
                                         'lm', 'parse', 'low', 'bg', 'tv', 'G_total')},
            'frac_G_clipped': sum(1 for x in f('G_grad_norm') if x > 1) / len(rs),
            'frac_D_clipped': sum(1 for x in f('D_grad_norm') if x > 1) / len(rs),
            'amp_retry_updates': sum(1 for x in f('amp_attempts') if x > 1),
            'amp_retry_attempts_extra': sum(x - 1 for x in f('amp_attempts')),
            'min_G_scale': min(min(f('G_scale')), min(f('G_scale_before'))),
            'min_D_scale': min(min(f('D_scale')), min(f('D_scale_before'))),
            'frac_budget_active': sum(1 for x in fl('budget') if x and x > 0) / len(rs),
            'frac_M_mean_lt_0.01': sum(1 for x in f('M_mean') if x < 0.01) / len(rs)}


def epoch_summary(rs):
    return window_summary(rs)


def scenario_collect(args):
    from methods.gpat import runtime_contract as rc
    sci_rows = {}
    with open(n3.sci_dir() / 'metrics.jsonl', 'rb') as f:                    # read-only
        for raw in f:
            d = json.loads(raw)
            if d.get('record_type') == 'optimizer_group' and START <= d['global_update'] <= PHASE_B_END:
                sci_rows[d['global_update']] = d
    sci_metrics_sha = file_sha(n3.sci_dir() / 'metrics.jsonl')
    anchor = json.loads((n5_root() / 'results' / 'anchor.json').read_text())
    branches = {}
    for b in BRANCHES:
        if (n5_root() / f'branch_{b}' / 'per_update.jsonl').exists():
            branches[b] = collect_branch(b, sci_rows)
    order = train_order_check(branches, anchor)
    for d in branches.values():
        del d['_group_sha']
    listing = {k: (n5_root() / 'results' / f'scientific_root_{k}.txt') for k in ('pre', 'post')}
    listing = {k: p.read_text().split()[0] if p.exists() else None for k, p in listing.items()}
    return {'milestone': 'M7D1-N5', 'labels': LABELS, 'method': METHOD, 'experiment': 'E08', 'seed': SEED,
            'scientific_run_id': SCI_RUN_ID, 'authority_commit': AUTHORITY_COMMIT,
            'history': {'n3': {'commit': 'd88563a46757d171a5e2af359eebc4fc18c305eb',
                               'classification': 'ADVERSARIAL_IMBALANCE / CURRICULUM_TRANSITION_INSTABILITY',
                               'mechanism': 'MASK_SIGMOID_SATURATION through shared G_res trunk',
                               'scientific_collapse_u': 5781},
                        'n4': {'commit': AUTHORITY_COMMIT, 'primary_candidate': 'R1', 'qualified': ['R1', 'R2'],
                               'qualified_through': 16575}},
            'stage_boundary': {'u16575': rc.curriculum(16575), 'u16576': rc.curriculum(16576)},
            'anchor': anchor, 'anchor_sha256': file_sha(n5_root() / 'results' / 'anchor.json'),
            'branch_definitions': BRANCHES, 'collapse_rule': COLLAPSE_RULE, 'artifact_rule': ARTIFACT_RULE,
            'open_mask_note': OPEN_MASK_NOTE, 'hard_gates': HARD_GATES, 'preference': PREFERENCE,
            'scientific_metrics_sha256': sci_metrics_sha, 'scientific_root_listing_sha256': listing['post'],
            'scientific_root_listing_pre_sha256': listing['pre'],
            'scientific_root_listing_command': SCI_LISTING_COMMAND,
            'train_order': order, 'branches': branches}


SCI_LISTING_COMMAND = ('cd <scientific run root> && find . -type f -printf "%P %s %T@\\n" | sort | sha256sum '
                       '(same command as N4; launcher writes results/scientific_root_{pre,post}.txt)')


# ----------------------------------------------------------------------------- finalize (pure, from the collected doc)
def train_order_check(branches, anchor):
    """(collect) Every branch sees the same TRAIN groups at the same update (group-index digests); u16576 equals the
    anchor snapshot group."""
    ref = {}
    for b in ('S0', 'S1', 'S2', 'S3', 'S4'):
        if b in branches:
            for u, h in branches[b]['_group_sha'].items():
                ref.setdefault(u, h)
    a16576 = sha(json.dumps(anchor['anchor_snapshot']['group_indices']).encode()) if anchor.get('anchor_snapshot') \
        else None
    out = {}
    for b, d in branches.items():
        g = d['_group_sha']
        shared = [u for u in g if u in ref]
        out[b] = {'updates_compared': len(shared), 'mismatch': sorted(u for u in shared if g[u] != ref[u])[:50],
                  'u16576_equals_anchor_group': g.get(START) == a16576,
                  'sequence_sha256': sha(json.dumps([[u, g[u]] for u in sorted(g)]).encode())}
    return out


def complete_epochs(d):
    return [e for e, s in sorted(d.get('epochs', {}).items(), key=lambda kv: int(kv[0])) if s['updates'][2] == 1105]


def hard_gate_failures(b, d, c):
    r = d['result']
    f = []
    if r is None:
        return ['result missing']
    if r['failure']:
        t = r['failure']['type']
        f.append('terminal AMP overflow' if 'Amp' in t else 'non-finite parameter' if 'PostStep' in t
                 else 'non-finite loss' if 'Loss' in t else f'hard failure {t}')
    if r['status'] not in ('COMPLETED_TO_END', 'STOPPED_AFTER_DECLARED_COLLAPSE') or \
            (r['status'] == 'STOPPED_AFTER_DECLARED_COLLAPSE' and not d['collapse_rederived']):
        f.append(f'status {r["status"]}')
    if d['collapse_rederived'] is not None or r['collapse'] is not None:
        f.append('N5 formal identity collapse')
    if d['collapse_rederived'] != r['collapse']:
        f.append('collapse re-derivation differs')
    if d['nonfinite_rows']:
        f.append('non-finite loss')
    if r['nonfinite_parameters_at_end']:
        f.append('non-finite parameter')
    if not (r['records_contiguous'] and d['schedule']['contiguous_from_16576'] and d['rows_equal_records']):
        f.append('update discontinuity')
    if d['amp_retry_integrity']['bad_updates']:
        f.append('corrupted optimizer/scaler state')
    sch = d['schedule']
    if sch['lambda_adv_exact_mismatch'] or sch['s_hf_exact_mismatch'] or sch['lr_con_spec_vs_frozen_contract_mismatch']:
        f.append('curriculum/LR mismatch')
    if sch['order_and_lr_vs_scientific_mismatch'] or c['train_order'][b]['mismatch'] or \
            not c['train_order'][b]['u16576_equals_anchor_group']:
        f.append('TRAIN order mismatch')
    a = r['access']
    if a['val_images'] or a['val_metadata'] or a['test_images'] or a['test_metadata'] or a['non_train_images']:
        f.append('VAL/TEST access')
    if r['write_firewall']['denied'] or r['write_firewall']['outside_allowed']:
        f.append('write-firewall violation')
    if r['code_head'] != AUTHORITY_COMMIT or r['start_snapshot']['sha256'] != c['anchor']['anchor_snapshot']['sha256'] \
            or c['anchor']['status'] != 'ANCHOR_EXACT':
        f.append('provenance mismatch')
    if not c['scientific_root_listing_pre_sha256'] or \
            c['scientific_root_listing_sha256'] != c['scientific_root_listing_pre_sha256']:
        f.append('scientific-root mutation')
    return f


def phase_a(d):
    ev = d['collapse_rederived']
    if ev is not None and ev['onset_update'] <= PHASE_A_END:
        return {'result': 'COLLAPSED', **ev}
    if d['last_update'] is None or d['last_update'] < PHASE_A_END:
        return {'result': 'INCOMPLETE'}
    return {'result': 'NO_FORMAL_COLLAPSE'}


def phase_a_outcomes(pa):
    """Predeclared mapping of instruction section 9 (letters A..E) from the formal Phase-A collapse results."""
    col = {b: pa.get(b, {}).get('result') == 'COLLAPSED' for b in BRANCHES}
    done = all(pa.get(b, {}).get('result') in ('COLLAPSED', 'NO_FORMAL_COLLAPSE') for b in BRANCHES)
    out = []
    if not done:
        return ['INCOMPLETE']
    if not col['S0']:
        out.append('A')
    if col['S0'] and col['S4'] and not col['S3']:
        out.append('B')
    if col['S0'] and not col['S4']:
        out.append('C')
    if col['S3']:
        out.append('D')
    if col['S0'] and not col['S1'] and not col['S2']:
        out.append('E')
    return out


def anchor_status(a):
    p = a['parity']
    return {'status': a['status'], 'all_bitwise': p['all_bitwise'], 'record_field_mismatch': p['record_field_mismatch'],
            'amp_retry_events_equal': p['amp_retry_events_equal'], 'n3_mask_row_equal': p['n3_mask_row_equal'],
            'n4_per_update_row_field_mismatch': p['n4_per_update_row_field_mismatch'], 'checks': a['checks'],
            'base_n4_r1_snapshot': {k: a['base_n4_r1_snapshot'][k] for k in (
                'n4_relative_path', 'sha256', 'size', 'position', 'code_head_at_capture', 'kind',
                'n4_branch_R1_result_sha256', 'n4_branch_R1_tool_sha256')},
            'base_state_digest': a['base_n4_r1_snapshot']['state_digest'],
            'anchor_snapshot': None if a['anchor_snapshot'] is None else {
                k: a['anchor_snapshot'][k] for k in ('path', 'sha256', 'size', 'global_update_attempted', 'kind',
                                                     'code_head', 'group_indices', 'state_digest')},
            'access': a['access'], 'write_firewall': a.get('write_firewall'), 'code_head': a.get('code_head')}


def build_qualification(c):
    B = c['branches']
    out = {'milestone': 'M7D1-N5', 'labels': c['labels'], 'authority_commit': c['authority_commit'],
           'history': c['history'], 'stage_boundary': c['stage_boundary'],
           'branch_definitions': c['branch_definitions'], 'collapse_rule': c['collapse_rule'],
           'artifact_rule': c['artifact_rule'], 'open_mask_note': c['open_mask_note'],
           'hard_gates': c['hard_gates'], 'preference': c['preference'], 'train_order': c['train_order'],
           'scientific_metrics_sha256': c['scientific_metrics_sha256'],
           'scientific_root_listing': {'pre': c['scientific_root_listing_pre_sha256'],
                                       'post': c['scientific_root_listing_sha256']},
           'branches': {}}
    an = anchor_status(c['anchor'])
    an_ok = an['status'] == 'ANCHOR_EXACT' and an['all_bitwise'] and not an['write_firewall']['denied'] and \
        not an['write_firewall']['outside_allowed'] and an['code_head'] == AUTHORITY_COMMIT
    out['anchor'] = dict(an, gate='PASS' if an_ok else 'FAIL')
    pa = {}
    for b, d in B.items():
        r = d['result']
        e = {'phase_a': phase_a(d), 'hard_gate_failures': hard_gate_failures(b, d, c), 'last_update': d['last_update'],
             'status': r['status'] if r else None, 'collapse': d['collapse_rederived'],
             'amp_retries': len(r['amp_retries']) if r else None,
             'amp_retry_updates': sorted({x['global_update_attempted'] for x in r['amp_retries']}) if r else None,
             'schedule': d['schedule'], 'eventual_lambda_adv': lambda_adv(b, DEFAULT_END[b]),
             'protocol_candidate': BRANCHES[b]['protocol_candidate'], 'phase_a_summary': d['phase_a_summary']}
        pa[b] = e['phase_a']
        if b in LONG_BRANCHES:
            eps = complete_epochs(d)
            meds = {ep: d['epochs'][ep]['A_mean']['p50'] for ep in eps}
            e['complete_epochs'] = eps
            e['epoch_median_A_mean'] = meds
            e['artifact_retained'] = len(eps) == 10 and all(v >= c['artifact_rule']['reference_floor']
                                                            for v in meds.values()) and d['collapse_rederived'] is None
            e['long_horizon_reached'] = d['last_update'] == PHASE_B_END
            ok = e['long_horizon_reached'] and not e['hard_gate_failures']
            e['qualification'] = ('QUALIFIED' if ok and e['artifact_retained'] else
                                  'HARD_GATES_PASSED_ARTIFACT_NOT_RETAINED' if ok else 'REJECTED')
        else:
            e['role'] = 'CAUSAL_DIAGNOSTIC (not a protocol candidate)'
            e['qualification'] = 'NOT_A_CANDIDATE'
        out['branches'][b] = e
    out['phase_a_outcomes'] = phase_a_outcomes(pa)
    qualified = [b for b in LONG_BRANCHES if out['branches'].get(b, {}).get('qualification') == 'QUALIFIED']
    out['qualified'] = qualified
    chosen = next(((b, lab) for b, lab in DECISION_ORDER if b in qualified), None) if an_ok else None
    out['preferred_stage3_branch'] = chosen[0] if chosen else None
    out['stage3_status'] = chosen[1] if chosen else ('QUALIFICATION_BLOCKED' if not an_ok else
                                                     'NO_STAGE3_REPAIR_QUALIFIED')
    out['verdict'] = ('M7D1_N5_QUALIFICATION_BLOCKED' if not an_ok else 'M7D1_N5_STAGE3_TRANSITION_QUALIFIED' if chosen
                      else 'M7D1_N5_NO_STAGE3_REPAIR_QUALIFIED')
    if chosen:
        f = proposed_schedule(chosen[0])
        out['proposed_lambda_adv_schedule'] = {
            'status': 'DIAGNOSTIC_RECOMMENDATION_ONLY (owner approval required; no config changed)',
            'definition': schedule_text(chosen[0]),
            'probe': {str(u): f(u) for u in (1, 5525, 5526, 6078, 6630, 6631, 16575, 16576, 17128, 17680, 17681,
                                             22100, 22101, 27625, 66300)}}
    out['gradient_summary'] = n4.gradient_summary(c)
    return out


def schedule_text(b):
    s = ['u1..u5525: 0.0 (frozen stage 1)',
         'u5526..u6630: 0.05 * ((u - 5526) / (6630 - 5526)) (N4 R1 stage-2 ramp)',
         'u6631..u16575: 0.05 (frozen stage 2 value)']
    k = BRANCHES[b]
    if k['kind'] == 'LINEAR_RAMP':
        e = k['ramp_end']
        s += [f'u16576..u{e}: 0.05 + 0.05 * ((u - 16576) / ({e} - 16576)) (N5 {b} stage-3 ramp)',
              f'u{e + 1}..u66300: 0.10 (frozen stage 3 value)']
    else:
        s += ['u16576..u66300: 0.10 (frozen stage 3, unchanged)']
    s += ['s_hf, lambda_con, lambda_spec, LR: frozen schedule unchanged',
          'qualified only on seed 42 through u27625; not evaluated beyond u27625']
    return s


def gradient_rows(c):
    return n4.gradient_rows(c)


def screen_rows(c):
    rows = []
    for b, d in sorted(c['branches'].items()):
        for r in d['screen']:
            rows.append({'branch': b, **r})
    return rows


SCREEN_FIELDS = ('branch', 'global_update', 'epoch', 'group', *SCREEN_KEYS, *('L_' + k for k in LOSS_KEYS))
LONG_EPOCH_STATS = ('M_mean', 'M_p01', 'M_p10', 'M_p50', 'frac_M_lt_1e-2', 'frac_M_gt_0.99', 'M_spatial_var_mean',
                    'sigmoid_deriv_mean', 'logit_mean', 'A_mean', 'A_p90', 'abs_xhat_minus_target_mean',
                    'tanh_hf_absmean', 'M_times_abs_delta_HF_mean', 'G_grad_norm', 'D_grad_norm', 'Dlogit_real_mean',
                    'Dlogit_fake_mean', 'Dlogit_real_sig_mean', 'Dlogit_fake_sig_mean', 'D_total', 'D_real', 'D_fake',
                    'gadv', 'budget', 'artcon', 'spec', 'id', 'lm', 'bg', 'G_total')
LONG_SCALARS = ('frac_G_clipped', 'frac_D_clipped', 'amp_retry_updates', 'amp_retry_attempts_extra', 'min_G_scale',
                'min_D_scale', 'frac_budget_active', 'frac_M_mean_lt_0.01')


def long_rows(c):
    rows = []
    for b, d in sorted(c['branches'].items()):
        for e, s in sorted(d.get('epochs', {}).items(), key=lambda kv: int(kv[0])):
            row = {'branch': b, 'epoch': int(e), 'first_update': s['updates'][0], 'last_update': s['updates'][1],
                   'updates': s['updates'][2], 'lambda_adv_first': s['lambda_adv'][0],
                   'lambda_adv_last': s['lambda_adv'][1], 's_hf': s['s_hf'][1], **{k: s[k] for k in LONG_SCALARS}}
            for k in LONG_EPOCH_STATS:
                for q in ('min', 'p10', 'p50', 'p90', 'max'):
                    row[f'{k}_{q}'] = (s[k] or {}).get(q)
            rows.append(row)
    return rows


LONG_FIELDS = ('branch', 'epoch', 'first_update', 'last_update', 'updates', 'lambda_adv_first', 'lambda_adv_last',
               's_hf', *LONG_SCALARS, *[f'{k}_{q}' for k in LONG_EPOCH_STATS for q in ('min', 'p10', 'p50', 'p90',
                                                                                         'max')])


def finalize(collected_path):
    raw = Path(collected_path).read_bytes()
    c = json.loads(raw)
    doc = build_qualification(c)
    doc['source_collected_sha256'] = sha(raw)
    return {'QUALIFICATION.json': json.dumps(doc, indent=1, sort_keys=True) + '\n',
            'SCREEN.csv': n3.render_csv(screen_rows(c), SCREEN_FIELDS),
            'LONG.csv': n3.render_csv(long_rows(c), LONG_FIELDS),
            'GRADIENTS.csv': n3.render_csv(gradient_rows(c), n4.GRAD_FIELDS)}


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--scenario', choices=('anchor', 'branch', 'analyze', 'collect'), required=True)
    ap.add_argument('--branch', choices=sorted(BRANCHES))
    ap.add_argument('--out', required=True)
    ap.add_argument('--end', type=int, default=None, help='branch: last update (default per branch)')
    ap.add_argument('--updates', default=None, help='branch: snapshot updates; analyze: updates to attribute')
    ap.add_argument('--name', default=None)
    ap.add_argument('--source', default=None, help='analyze: snapshot directory name (default branch_<SX>)')
    args = ap.parse_args()
    out_path = Path(args.out).resolve()
    assert str(out_path).startswith(str(n5_root()) + '/'), out_path
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
    out = {'anchor': scenario_anchor, 'branch': scenario_branch, 'analyze': scenario_analyze}[args.scenario](args)
    out['write_firewall'] = fw
    out['seconds_total'] = round(time.monotonic() - t0, 1)
    out['code_head'] = n3.git_head()
    out['tool_sha256'] = file_sha(__file__)
    out['tool_sha256_at_start'] = TOOL_SHA256_AT_START
    out['n4_tool_sha256'] = file_sha(ROOT / 'tools' / 'm7d1_n4_repair_qualification.py')
    out['n3_tool_sha256'] = file_sha(ROOT / 'tools' / 'm7d1_n3_collapse_root_cause.py')
    out_path.write_text(json.dumps(out, indent=1, sort_keys=True, default=str) + '\n')
    print(f'{args.scenario} {args.branch}: {out.get("status", "done")} in {out["seconds_total"]} s', flush=True)


if __name__ == '__main__':
    main()
