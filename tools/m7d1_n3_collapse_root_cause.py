"""M7D1-N3: localize the causal mechanism of the GPAT-B0 / E08 / seed 42 generator identity collapse (M7D1-N2-F01).
DIAGNOSTIC_ONLY -- not a scientific run, not a scientific replacement, not a protocol change.

No exact state at or before update 5525 exists for seed 42 (the scientific run keeps only the final latest.pt; the
quarantined attempts and the M7D1-N1 qualification hold u1105 / pre-u1971 states of code 6cf271e), so this harness
replays GPAT-B0 seed 42 from update 1 with the SCIENTIFIC seed-42 epoch order and determinism policy (exactly as
run_scientific) inside a diagnostic context. Every output lives under
<runtime_root>/diagnostics/m7/M7D1_N3_seed42/ ; the scientific run root is only read (its metrics.jsonl, for parity),
and an audit hook refuses any write under <runtime_root>/runs/.

Scenarios (each a fresh process, gpat-m7-gpu, PYTHONHASHSEED=42 + rio.LAUNCH_ENVIRONMENT):
  replay           u1..u5850 with the production Trainer/GeneratorStep (ATOMIC_AMP_BACKOFF_RETRY); per-update parity
                   against the scientific records and amp_retry events; DIAGNOSTIC pre-update snapshots at SNAP_UPDATES;
                   per-update mask statistics (forward hook on G_res); mask-channel optimizer trace u5740..5790.
  analyze          from each replay snapshot: mask/residual/A statistics, per-loss gradient attribution, sigmoid
                   saturation decomposition, discriminator path; scaler sensitivity at u5758/u5759.
  fork_<X>         X in A..F (+ pairwise): from the pre-u5526 snapshot, u5526..u5850 under a diagnostic curriculum
                   override (monkeypatched runtime_contract.curriculum in that process only); snapshots at
                   FORK_SNAP_UPDATES; then the same attribution on those snapshots.
"""
import argparse
import copy
import hashlib
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

from methods.gpat import runner_io as rio                   # noqa: E402

METHOD, SEED = 'GPAT-B0', 42
SCI_RUN_ID = '7b799fbd6d0426da'
DIAG_PARTS = ('diagnostics', 'm7', 'M7D1_N3_seed42')
LABELS = ['DIAGNOSTIC_ONLY', 'NOT_SCIENTIFIC', 'NOT_A_REPLACEMENT_RUN', 'NOT_ELIGIBLE_FOR_VAL_SELECTION',
          'NOT_ELIGIBLE_FOR_BANK', 'NOT_ELIGIBLE_FOR_PAPER_RESULT']
SNAP_KIND = 'GPAT_M7D1_N3_DIAGNOSTIC_PRE_UPDATE_SNAPSHOT'
REPLAY_END = 5850
SNAP_UPDATES = (5525, 5526, 5700, 5757, 5758, 5759, 5777, 5778, 5779, 5780, 5781, 5782, 5800)
PARITY_REPORT = (1, 1105, 1971, 5525, 5526, 5700, 5750, 5758, 5759, 5777, 5778, 5779, 5780, 5781, 5800)
TRACE = (5740, 5790)
FORK_FROM = 5526                  # pre-update snapshot of u5526 == state after u5525 (end of epoch 5, EMA active)
FORK_END = 5850
FORK_SNAP_UPDATES = (5526, 5600, 5700, 5758, 5781, 5800, 5850)
FORK_A_DENSE = tuple(range(5757, 5783))      # fork A == scientific trajectory: dense pre-update snapshots around u5781
MASK = 12
LIN_EPS = 1.0 / 64
TERMS = ('id', 'lm', 'parse', 'low', 'artcon', 'spec', 'budget', 'tv', 'bg', 'gadv')
VOLATILE = ('utc', 'wall_seconds', 'peak_allocated_bytes')
STAGE1 = {'s_hf': 0.05, 'lambda_adv': 0.0, 'lambda_con': 0.5, 'lambda_spec': 0.25}
STAGE2 = {'s_hf': 0.10, 'lambda_adv': 0.05, 'lambda_con': 1.0, 'lambda_spec': 0.5}
FORKS = {   # values used for every u >= 5526 (all forks end at u5850 < 16576, i.e. inside the frozen stage 2)
    'A': dict(STAGE2),
    'B': dict(STAGE1, s_hf=0.10),
    'C': dict(STAGE1, lambda_adv=0.05),
    'D': dict(STAGE1, lambda_con=1.0),
    'E': dict(STAGE1, lambda_spec=0.5),
    'F': dict(STAGE1),
}


def add_fork(name, **over):
    FORKS[name] = dict(STAGE1, **over)


# pairwise combinations (only run on demand)
add_fork('BC', s_hf=0.10, lambda_adv=0.05)
add_fork('BD', s_hf=0.10, lambda_con=1.0)
add_fork('BE', s_hf=0.10, lambda_spec=0.5)
add_fork('CD', lambda_adv=0.05, lambda_con=1.0)
add_fork('CE', lambda_adv=0.05, lambda_spec=0.5)
add_fork('DE', lambda_con=1.0, lambda_spec=0.5)
add_fork('BCD', s_hf=0.10, lambda_adv=0.05, lambda_con=1.0)
add_fork('BDE', s_hf=0.10, lambda_con=1.0, lambda_spec=0.5)
add_fork('CDE', lambda_adv=0.05, lambda_con=1.0, lambda_spec=0.5)
add_fork('BCE', s_hf=0.10, lambda_adv=0.05, lambda_spec=0.5)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def runtime_root():
    return Path(rio.faces_root()['runtime_root'])


def diag_root():
    return runtime_root().joinpath(*DIAG_PARTS)


def sci_dir():
    return rio.run_root(runtime_root(), rio.SCIENTIFIC, METHOD, SEED)


# ----------------------------------------------------------------------------- write firewall
def install_write_firewall():
    rt = str(runtime_root().resolve())
    allowed = [str(diag_root()), os.environ.get('TMPDIR', '/nonexistent'), '/dev/shm', '/dev/null', '/proc/']
    forbidden = rt + '/runs'
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
        if p.startswith(forbidden):
            record['denied'].append(p)
            raise PermissionError(f'M7D1-N3 firewall: write under the scientific runs root refused: {p}')
        if not any(p.startswith(a) for a in allowed):
            record['outside_allowed'].append(p)
    sys.addaudithook(hook)
    return record


# ----------------------------------------------------------------------------- torch helpers
def state_digest(obj):
    import torch
    h = hashlib.sha256()

    def walk(node, key=''):
        if torch.is_tensor(node):
            h.update(key.encode() + str(node.dtype).encode() + str(tuple(node.shape)).encode())
            t = node.detach().contiguous().cpu()
            h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes() if t.numel() else b'')
        elif isinstance(node, dict):
            for k in sorted(node, key=str):
                walk(node[k], f'{key}/{k}')
        elif isinstance(node, (list, tuple)):
            for i, v in enumerate(node):
                walk(v, f'{key}[{i}]')
        else:
            h.update(f'{key}={json.dumps(node, default=str)}'.encode())
    walk(obj)
    return h.hexdigest()


MODULES = ('e_art', 'g_res', 'discriminator')


def full_state(tr):
    from methods.gpat import runner_checkpoint as ck
    out = {}
    for k in MODULES:
        m = getattr(tr.core, k)
        out[f'params_{k}'] = state_digest({n: p for n, p in m.named_parameters()})
        out[f'buffers_{k}'] = state_digest({n: b for n, b in m.named_buffers()})
    out['G_OPT'] = state_digest(tr.g_opt.state_dict())
    out['D_OPT'] = state_digest(tr.d_opt.state_dict())
    out['G_SCALER'] = tr.g_scaler.state_dict()
    out['D_SCALER'] = tr.d_scaler.state_dict()
    out['rng'] = state_digest(ck.rng_state())
    out['ema'] = None if tr.ema is None else {k: state_digest(v.state_dict()) for k, v in tr.ema.items()}
    out['position'] = dict(tr.position)
    return out


class DiagContext:
    """Diagnostic stand-in for GPATRunContext: labelled metrics.jsonl / manifest / summary in its own directory; it can
    never record a checkpoint and has no checkpoint directory."""

    def __init__(self, name, extra=None):
        self.run_dir = diag_root() / name
        rel = self.run_dir.relative_to(runtime_root()).parts
        assert rel[0] == 'diagnostics' and 'runs' not in rel, 'never a scientific root'
        assert not self.run_dir.exists(), f'diagnostic root exists (move aside, never overwrite): {self.run_dir}'
        self.run_dir.mkdir(parents=True)
        self.manifest = {'labels': LABELS, 'runner_mode': 'DIAGNOSTIC_REPLAY_OF_SCIENTIFIC_E08_SEED42',
                         'method': METHOD, 'replayed_seed': SEED, 'scientific': False, 'writes_checkpoints': False,
                         'scientific_run_id_compared': SCI_RUN_ID, 'code_head': git_head(), **(extra or {})}
        (self.run_dir / 'run_manifest.json').write_text(json.dumps(self.manifest, indent=1) + '\n')
        self._fh = (self.run_dir / 'metrics.jsonl').open('a', encoding='utf-8')

    @property
    def ckpt_dir(self):
        raise AssertionError('diagnostic context never writes run checkpoints')

    def log(self, record_type, record):
        self._fh.write(json.dumps({'record_type': record_type, 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                                   **record}, sort_keys=True, allow_nan=False) + '\n')
        self._fh.flush()

    def record_checkpoint(self, entry):
        raise AssertionError('diagnostic context never records checkpoints')

    def close(self, status, summary):
        self._fh.close()
        (self.run_dir / 'run_summary.json').write_text(json.dumps({**self.manifest, 'status': status, **summary},
                                                                  indent=1, default=str) + '\n')


def git_head():
    import subprocess
    return subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], capture_output=True).stdout.decode().strip()


class Stop:
    requested = False


def setup(name, extra=None):
    """Trainer in SCIENTIFIC seed-42 order and determinism (exactly as run_scientific) with a diagnostic context."""
    import torch
    from methods.gpat import runtime_contract as rc
    from methods.gpat import runner as R
    assert os.environ.get('PYTHONHASHSEED') == str(SEED), 'launch with PYTHONHASHSEED=42 (as the scientific run)'
    for k, v in rio.LAUNCH_ENVIRONMENT.items():
        assert os.environ.get(k) == v, f'launch with {k}={v}'
    rc.apply_qualification_determinism(SEED, gpu=True)
    storage = rio.faces_root()
    tr = R.Trainer(mode=rio.SCIENTIFIC, method=METHOD, seed=SEED, runtime_root=storage['runtime_root'],
                   faces_root=storage['faces_256_root'], assets=rio.load_assets())
    tr.ctx = DiagContext(name, extra)                    # the scientific GPATRunContext is never opened
    tr.suppressed_recovery_saves = []
    tr.save_recovery = lambda: tr.suppressed_recovery_saves.append(tr.position['global_update'])

    def no_candidate(epoch):
        raise AssertionError('no EMA candidate is ever written by a diagnostic run')
    tr.write_candidate = no_candidate
    torch.cuda.reset_peak_memory_stats()
    return tr


# ----------------------------------------------------------------------------- per-forward mask statistics hook
class MaskHook:
    def __init__(self, tr):
        self.buf = []
        tr.core.g_res.register_forward_hook(self)

    def __call__(self, module, inputs, out):
        import torch
        with torch.no_grad():
            r = out[:, MASK].float()
            m = torch.sigmoid(r)
            hf = torch.tanh(out[:, 3:12].float()).abs().mean()
            v = torch.stack([r.mean(), r.std(), r.min(), r.max(), m.mean(), m.min(), m.max(),
                             (m < 1e-1).float().mean(), (m < 1e-2).float().mean(), (m < 1e-3).float().mean(),
                             (m > 0.5).float().mean(), (m * (1 - m)).mean(), hf]).tolist()
        self.buf.append(dict(zip(('logit_mean', 'logit_std', 'logit_min', 'logit_max', 'M_mean', 'M_min', 'M_max',
                                  'frac_M_lt_1e-1', 'frac_M_lt_1e-2', 'frac_M_lt_1e-3', 'frac_M_gt_0.5',
                                  'sigmoid_deriv_mean', 'tanh_hf_absmean'), v)))
        return None

    def take(self, n_mb):
        out = self.buf[-n_mb:]
        n_calls = len(self.buf)
        self.buf = []
        return out, n_calls


def merge_mb(stats):
    """Average the per-microbatch summaries (min/max combined as min/max)."""
    out = {}
    for k in stats[0]:
        vals = [s[k] for s in stats]
        out[k] = min(vals) if k.endswith('_min') else max(vals) if k.endswith('_max') else sum(vals) / len(vals)
    return out


# ----------------------------------------------------------------------------- snapshots
def snapshot_payload(tr, group, u):
    from methods.gpat import runner_checkpoint as ck
    return {'kind': SNAP_KIND, 'labels': LABELS, 'global_update_attempted': u,
            'modules': {k: getattr(tr.core, k).state_dict() for k in MODULES},
            'optimizers': {'G_OPT': tr.g_opt.state_dict(), 'D_OPT': tr.d_opt.state_dict()},
            'scalers': {'G_SCALER': tr.g_scaler.state_dict(), 'D_SCALER': tr.d_scaler.state_dict()},
            'ema': None if tr.ema is None else {k: v.state_dict() for k, v in tr.ema.items()},
            'position': ck.primitives(tr.position), 'rng': ck.rng_state(),
            'group': [{k: v.detach().cpu().clone() for k, v in mb.items()} for mb in group],
            'group_indices': [mb['index'].tolist() for mb in group],
            'group_pair_ids': [[tr.records[i]['pair_id'] for i in mb['index'].tolist()] for mb in group],
            'state_digest': ck.primitives(full_state(tr)), 'code_head': git_head()}


def save_snapshot(payload, directory, u):
    import torch
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'pre_update_{u}.pt'
    assert not path.exists(), path
    tmp = path.with_suffix('.tmp')
    torch.save(payload, tmp)
    os.replace(tmp, path)
    meta = {'path': str(path.relative_to(diag_root())), 'sha256': file_sha(path), 'size': path.stat().st_size,
            'global_update_attempted': u, 'labels': LABELS, 'kind': SNAP_KIND,
            'state_digest': payload['state_digest'], 'group_indices': payload['group_indices'],
            'group_pair_ids': payload['group_pair_ids'], 'code_head': payload['code_head']}
    path.with_suffix('.json').write_text(json.dumps(meta, indent=1) + '\n')
    return meta


def load_snapshot(path):
    from methods.gpat import runner_checkpoint as ck
    meta = json.loads(Path(path).with_suffix('.json').read_text())
    assert file_sha(path) == meta['sha256'], f'snapshot hash {path}'
    snap = ck.load(path)
    assert snap['kind'] == SNAP_KIND and snap['labels'] == LABELS
    return snap, meta


def reset_scaler(scaler, state):
    if scaler._scale is not None:
        scaler.update(new_scale=float(state['scale']))
    scaler.load_state_dict(state)


def apply_snapshot(tr, snap):
    """Restore the exact pre-update state (modules incl. buffers, optimizers, scalers, EMA, position, RNG) and verify it
    against the digest taken from the live state at capture time. Returns the group on the device."""
    from methods.gpat import runner as R
    from methods.gpat import runner_checkpoint as ck
    for k in MODULES:
        getattr(tr.core, k).load_state_dict(snap['modules'][k], strict=True)
    tr.g_opt.load_state_dict(copy.deepcopy(snap['optimizers']['G_OPT']))
    tr.d_opt.load_state_dict(copy.deepcopy(snap['optimizers']['D_OPT']))
    reset_scaler(tr.g_scaler, snap['scalers']['G_SCALER'])
    reset_scaler(tr.d_scaler, snap['scalers']['D_SCALER'])
    if snap['ema'] is not None:
        if tr.ema is None:
            tr.activate_ema()
        for k in ('e_art', 'g_res'):
            tr.ema[k].load_state_dict(copy.deepcopy(snap['ema'][k]))
    else:
        assert tr.ema is None
    for p in tr.core.parameters():
        p.grad = None
    tr.position = dict(snap['position'])
    ck.restore_rng(snap['rng'])
    got = ck.primitives(full_state(tr))
    assert got == snap['state_digest'], {k: (got[k], snap['state_digest'][k]) for k in got
                                         if got[k] != snap['state_digest'][k]}
    return [R.to_device(dict(mb)) for mb in snap['group']]


# ----------------------------------------------------------------------------- parity
def strip(rec):
    return {k: v for k, v in rec.items() if k not in VOLATILE}


def scientific_records():
    groups, retries = {}, {}
    with open(sci_dir() / 'metrics.jsonl', 'rb') as f:                 # read-only
        for raw in f:
            d = json.loads(raw)
            if d.get('record_type') == 'optimizer_group' and d['global_update'] <= REPLAY_END:
                groups[d['global_update']] = d
            elif d.get('record_type') == 'amp_retry' and d['global_update_attempted'] <= REPLAY_END:
                retries.setdefault(d['global_update_attempted'], []).append(d)
    return groups, retries


def diff_fields(a, b, prefix=''):
    out = []
    for k in sorted(set(a) | set(b)):
        if k in VOLATILE:
            continue
        va, vb = a.get(k, '<missing>'), b.get(k, '<missing>')
        if isinstance(va, dict) and isinstance(vb, dict):
            out += diff_fields(va, vb, f'{prefix}{k}.')
        elif json.dumps(va) != json.dumps(vb):
            out.append(f'{prefix}{k}')
    return out


# ----------------------------------------------------------------------------- stepping wrapper
class StepWrapper:
    """Wraps the production GeneratorStep: pre-update snapshots, mask-hook flush, mask-channel optimizer trace and
    pre-clip gradient capture. Never changes what the wrapped step computes."""

    def __init__(self, tr, snap_dir, snap_updates, trace=None):
        self.tr, self.real = tr, tr.step
        self.snap_dir, self.snap_updates, self.trace = snap_dir, set(snap_updates), trace
        self.hook = MaskHook(tr)
        self.snaps, self.mask_rows, self.trace_rows = [], [], []
        self.u = None
        self.pre_clip = None
        self.real.inspect = self.inspect
        orig = tr.g_opt.step
        tr.g_opt.step = self.opt_step(orig)

    def __getattr__(self, k):
        return getattr(self.real, k)

    def __setattr__(self, k, v):
        # 'ema': Trainer.activate_ema() assigns step.ema (end of epoch 5). The u1..u5850 replay ran with only
        # 'on_retry' forwarded, so its EMA stayed at the epoch-5 initialization (ema_updated False for u>5525; the EMA
        # is write-only, G/D/optimizer/scaler states are unaffected -- see results/parity.json).
        if k in ('on_retry', 'ema'):
            setattr(self.real, k, v)
        else:
            object.__setattr__(self, k, v)

    def tracing(self):
        return self.trace is not None and self.trace[0] <= self.u <= self.trace[1]

    def inspect(self, stage, step):
        import torch
        if stage != 'G' or not self.tracing():
            return
        e = self.tr.core.g_res.ending
        gw, gb = e.weight.grad, e.bias.grad
        norms = [p.grad.float().norm() for _, p in step.g_named if p.grad is not None]
        self.pre_clip = {'G_total_norm_unscaled_preclip': float(torch.stack(norms).norm()),
                         'mask_w_grad_norm': float(gw[MASK].float().norm()) if gw is not None else None,
                         'mask_b_grad': float(gb[MASK]) if gb is not None else None,
                         'mask_w_grad_finite': bool(torch.isfinite(gw[MASK]).all()) if gw is not None else None,
                         'G_scale_at_unscale': float(self.tr.g_scaler.get_scale())}

    def opt_step(self, orig):
        def step(*a, **kw):
            import torch
            if not self.tracing():
                return orig(*a, **kw)
            e = self.tr.core.g_res.ending
            w0, b0 = e.weight[MASK].detach().clone(), float(e.bias[MASK])
            gw = e.weight.grad[MASK].detach().clone()
            gb = float(e.bias.grad[MASK])
            res = orig(*a, **kw)
            st_w, st_b = self.tr.g_opt.state[e.weight], self.tr.g_opt.state[e.bias]
            self.trace_rows.append({
                'global_update': self.u, **(self.pre_clip or {}),
                'mask_w_grad_postclip_norm': float(gw.float().norm()), 'mask_b_grad_postclip': gb,
                'mask_w_exp_avg_norm': float(st_w['exp_avg'][MASK].norm()),
                'mask_w_exp_avg_sq_mean': float(st_w['exp_avg_sq'][MASK].mean()),
                'mask_b_exp_avg': float(st_b['exp_avg'][MASK]), 'mask_b_exp_avg_sq': float(st_b['exp_avg_sq'][MASK]),
                'mask_w_delta_norm': float((e.weight[MASK].detach() - w0).float().norm()),
                'mask_w_norm': float(e.weight[MASK].detach().float().norm()),
                'mask_b_value': float(e.bias[MASK]), 'mask_b_delta': float(e.bias[MASK]) - b0,
                'adam_step': float(st_b['step']), 'lr': float(self.tr.g_opt.param_groups[0]['lr'])})
            self.pre_clip = None
            return res
        return step

    def __call__(self, group, u):
        self.u = u
        if u in self.snap_updates:
            self.snaps.append(save_snapshot(snapshot_payload(self.tr, group, u), self.snap_dir, u))
        self.hook.buf = []
        rec = self.real(group, u)
        stats, calls = self.hook.take(len(group))
        self.mask_rows.append({'global_update': u, 'forward_calls': calls, **merge_mb(stats),
                               'mask_bias': float(self.tr.core.g_res.ending.bias[MASK])})
        return rec


def install(tr, snap_dir, snap_updates, trace=None):
    w = StepWrapper(tr, snap_dir, snap_updates, trace)
    tr.step = w
    return w


def write_jsonl(path, rows):
    with open(path, 'w') as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + '\n')


# ----------------------------------------------------------------------------- scenario: replay
def scenario_replay(args):
    end = args.limit or REPLAY_END
    name = 'replay' if end == REPLAY_END else f'smoke_replay_{end}'
    tr = setup(name, {'purpose': f'u1..u{end} deterministic diagnostic replay'})
    smoke = end != REPLAY_END
    w = install(tr, diag_root() / 'snapshots' / name, (end,) if smoke else SNAP_UPDATES, (1, end) if smoke else TRACE)
    t0 = time.monotonic()
    tr.run_generator_groups(Stop(), limit=end)
    secs = time.monotonic() - t0
    assert tr.position['global_update'] == end
    # parity
    mine = {}
    retries = {}
    for line in (tr.ctx.run_dir / 'metrics.jsonl').read_text().splitlines():
        d = json.loads(line)
        if d['record_type'] == 'optimizer_group':
            mine[d['global_update']] = d
        elif d['record_type'] == 'amp_retry':
            retries.setdefault(d['global_update_attempted'], []).append(d)
    sci, sci_retry = scientific_records()
    mism = {u: diff_fields(mine[u], sci[u]) for u in range(1, end + 1)}
    mism = {u: v for u, v in mism.items() if v}
    rmism = sorted(u for u in set(retries) | {x for x in sci_retry if x <= end}
                   if [strip(r) for r in retries.get(u, [])] != [strip(r) for r in sci_retry.get(u, [])])
    report = {u: {'bitwise_equal': u not in mism, 'differing_fields': mism.get(u, []),
                  'replay': {k: mine[u][k] for k in ('learning_rate', 'scale_hf', 'lambda_adv', 'lambda_con',
                                                     'lambda_spec', 'G_grad_norm', 'D_grad_norm', 'G_scale',
                                                     'D_scale', 'amp_attempts', 'epoch', 'group')},
                  'replay_losses': mine[u]['train_losses'], 'scientific_losses': sci[u]['train_losses'],
                  'amp_retry_events_replay': len(retries.get(u, [])),
                  'amp_retry_events_scientific': len(sci_retry.get(u, []))} for u in PARITY_REPORT if u <= end}
    write_jsonl(tr.ctx.run_dir / 'mask_per_update.jsonl', w.mask_rows)
    write_jsonl(tr.ctx.run_dir / 'mask_channel_trace.jsonl', w.trace_rows)
    out = {'updates': end, 'seconds': secs, 'bitwise_equal_updates': end - len(mism),
           'mismatched_updates': sorted(mism)[:200], 'first_mismatch': min(mism) if mism else None,
           'mismatch_fields_example': mism[min(mism)] if mism else None,
           'amp_retry_updates_replay': sorted(retries), 'amp_retry_updates_scientific': sorted(sci_retry),
           'amp_retry_event_mismatch': rmism, 'parity_report': report, 'snapshots': w.snaps,
           'suppressed_recovery_saves': tr.suppressed_recovery_saves, 'access': tr.access.report()}
    tr.ctx.close('diagnostic_completed', {'access': out['access'], 'bitwise_equal_updates': out['bitwise_equal_updates']})
    return out


# ----------------------------------------------------------------------------- attribution
def qstats(t):
    import torch
    t = t.detach().float().flatten()
    qs = torch.tensor([0.01, 0.1, 0.5, 0.9, 0.99], device=t.device)
    if t.numel() > 16_000_000:
        t = t[torch.linspace(0, t.numel() - 1, 16_000_000, device=t.device).long()]
    q = torch.quantile(t, qs).tolist()
    return {'mean': float(t.mean()), 'std': float(t.std()), 'min': float(t.min()), 'max': float(t.max()),
            'p01': q[0], 'p10': q[1], 'p50': q[2], 'p90': q[3], 'p99': q[4]}


def signed(t):
    import torch
    t = t.detach().float().flatten()
    n = t.numel()
    return {'l2': float(t.norm()), 'mean': float(t.mean()), 'median': float(t.median()),
            'frac_pos': float((t > 0).sum()) / n, 'frac_neg': float((t < 0).sum()) / n, 'sum': float(t.sum())}


def tendency(s, ref):
    """Gradient descent applies -dL/dlogit: a positive mean gradient lowers the logits (CLOSE_MASK)."""
    if s['l2'] <= 1e-3 * ref or s['l2'] == 0.0:
        return 'MIXED_OR_NEGLIGIBLE'
    if s['mean'] > 0 and s['frac_pos'] >= 0.6:
        return 'CLOSE_MASK'
    if s['mean'] < 0 and s['frac_neg'] >= 0.6:
        return 'OPEN_MASK'
    return 'MIXED_OR_NEGLIGIBLE'


def attribute(tr, group, u, cur, S=None, backoffs=0):
    """Per-term gradient attribution on group u from the current (pre-update) state, without stepping. Each term's
    gradient is taken separately (unit coefficient), scaled by the live G scale like the training backward and then
    unscaled; weighted gradients = curriculum/fixed weight * unit gradient. BN buffers and RNG are restored after."""
    import torch
    from methods.gpat import losses
    from methods.gpat import runner as R
    from methods.gpat import runner_checkpoint as ck
    core, step = tr.core, tr.step.real if hasattr(tr.step, 'real') else tr.step
    S = float(tr.g_scaler.get_scale()) if S is None else S
    bufs = {n: b.detach().clone() for n, b in core.named_buffers()}
    rng = ck.rng_state()
    weights_fixed = dict(losses.FIXED_WEIGHTS)
    W = {**weights_fixed, 'artcon': cur['lambda_con'], 'spec': cur['lambda_spec'], 'gadv': cur['lambda_adv']}
    named = step.g_named
    params = [p for _, p in named]
    sizes = [int(mb['x_source'].shape[0]) for mb in group]
    wmb = [s / sum(sizes) for s in sizes]
    nump = sum(p.numel() for p in params)
    gsl = [(n, p.numel()) for n, p in named]
    offs, o = {}, 0
    for n, k in gsl:
        offs[n] = (o, o + k)
        o += k
    unit = {k: torch.zeros(nump, dtype=torch.float32) for k in TERMS}      # CPU (GPU memory is shared)
    bw = {k: (W[k] if W[k] else 1.0) for k in TERMS}    # backward with the TRAINED weight (fp16 range as in training)
    per = {k: {'raw_mask': [], 'raw_hf': [], 'dM': [], 'dxhat': []} for k in TERMS}
    fwd = {'logit': [], 'M': [], 'sig': [], 'dLH': [], 'dHL': [], 'dHH': [], 'MdHF': [], 'res_t': [], 'res_s': [],
           'A': [], 'd_real': [], 'd_fake': []}
    comps_rec = {k: 0.0 for k in TERMS}
    used = [False] * len(params)
    nonfinite_terms = set()
    core.discriminator.requires_grad_(False)
    for j, mb in enumerate(group):
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            out = core(mb['x_source'], mb['x_target'], scale_hf=cur['s_hf'])
        comps = step.components(mb, out, cur)
        inputs = [out.raw, out.M, out.x_hat] + params
        for k in TERMS:
            comps_rec[k] += wmb[j] * float(comps[k])
            g = torch.autograd.grad(wmb[j] * bw[k] * comps[k].float() * S, inputs, retain_graph=True,
                                    allow_unused=True)
            g = [None if x is None else x.float() / (S * bw[k]) for x in g]
            if any(x is not None and not bool(torch.isfinite(x).all()) for x in g):
                nonfinite_terms.add(k)
            graw, gM, gx = g[0], g[1], g[2]
            if graw is None:
                graw = torch.zeros_like(out.raw, dtype=torch.float32)
            per[k]['raw_mask'].append(graw[:, MASK].detach())
            per[k]['raw_hf'].append(graw[:, 3:12].detach())
            per[k]['dM'].append(torch.zeros_like(out.M) if gM is None else gM.detach())
            per[k]['dxhat'].append(torch.zeros_like(out.x_hat) if gx is None else gx.detach())
            for i, x in enumerate(g[3:]):
                used[i] = used[i] or x is not None
            flat = torch.cat([torch.zeros(p.numel(), device=R.DEV) if x is None else x.reshape(-1)
                              for x, p in zip(g[3:], params)])
            unit[k] += flat.cpu()
            del flat, g
        with torch.no_grad():
            r = out.raw[:, MASK].float()
            fwd['logit'].append(r)
            fwd['M'].append(out.M.float())
            fwd['sig'].append((out.M * (1 - out.M)).float())
            for b in ('LH', 'HL', 'HH'):
                fwd['d' + b].append(getattr(out, 'delta_' + b).abs().float())
            fwd['MdHF'].append((out.M * (out.delta_LH.abs() + out.delta_HL.abs() + out.delta_HH.abs())).float())
            fwd['res_t'].append((out.x_hat.float() - mb['x_target'].float()).abs())
            fwd['res_s'].append((out.x_hat.float() - mb['x_source'].float()).abs())
            fwd['A'].append(out.A.float())
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                fwd['d_real'].append(core.discriminator(mb['x_source']).float())
                fwd['d_fake'].append(core.discriminator(out.x_hat).float())
        del out, comps
    core.discriminator.requires_grad_(True)
    with torch.no_grad():
        for n, b in core.named_buffers():
            b.copy_(bufs[n])
    ck.restore_rng(rng)
    if nonfinite_terms and S > 1.0:                      # same backoff as ATOMIC_AMP_BACKOFF_RETRY (factor 0.5)
        del unit, per, fwd
        return attribute(tr, group, u, cur, S / 2.0, backoffs + 1)
    total = sum(W[k] * unit[k] for k in TERMS)
    tnorm = float(total.norm())
    clip = min(1.0, 1.0 / (tnorm + 1e-6))

    def sl(vec, prefix):
        return torch.cat([vec[offs[n][0]:offs[n][1]] for n, _ in gsl if n.startswith(prefix)])
    ew0, ew1 = offs['g_res.ending.weight']
    wsz = ew1 - ew0
    per_out = wsz // 13
    mw = slice(ew0 + MASK * per_out, ew0 + (MASK + 1) * per_out)
    eb0, _ = offs['g_res.ending.bias']
    cat = lambda xs: torch.cat([x.flatten() for x in xs])       # noqa: E731
    mask_unit_ref = max(float(cat(per[k]['raw_mask']).norm()) * max(W[k], 0) for k in TERMS) or 1.0
    terms = {}
    for k in TERMS:
        gk = W[k] * unit[k]
        rm_u = cat(per[k]['raw_mask'])
        s_w = signed(W[k] * rm_u)
        s_u = signed(rm_u)
        terms[k] = {
            'weight': W[k], 'value': comps_rec[k], 'weighted_value': W[k] * comps_rec[k],
            'mask_logit_grad_weighted': s_w, 'mask_logit_grad_unit': s_u,
            'tendency_weighted': tendency(s_w, mask_unit_ref) if W[k] else 'ZERO_WEIGHT',
            'tendency_unit': tendency(s_u, max(float(cat(per[x]['raw_mask']).norm()) for x in TERMS) or 1.0),
            'hf_raw_grad_l2_weighted': W[k] * float(cat(per[k]['raw_hf']).norm()),
            'dL_dM_l2_unit': float(cat(per[k]['dM']).norm()), 'dL_dxhat_l2_unit': float(cat(per[k]['dxhat']).norm()),
            'mask_w_grad_l2_weighted': float(gk[mw].norm()), 'mask_b_grad_weighted': float(gk[eb0 + MASK]),
            'g_res_grad_l2_weighted': float(sl(gk, 'g_res.').norm()),
            'e_art_grad_l2_weighted': float(sl(gk, 'e_art.').norm()),
            'all_G_grad_l2_weighted': float(gk.norm()),
            'all_G_grad_l2_unit': float(unit[k].norm()),
            'cosine_with_total': float(torch.dot(gk, total) / (gk.norm() * total.norm() + 1e-30)),
            'projection_share_of_total': float(torch.dot(gk, total) / (total.norm() ** 2 + 1e-30)),
            'nonfinite': k in nonfinite_terms,
        }
    total_mask = sum(W[k] * cat(per[k]['raw_mask']) for k in TERMS)
    fc = {k: torch.cat(v) for k, v in fwd.items()}
    hf_abs = torch.cat([fc['dLH'], fc['dHL'], fc['dHH']], dim=1)
    logit = fc['logit']
    Mv = fc['M']
    res = {
        'global_update': u, 'curriculum': cur, 'G_scale_used': S, 'attribution_scale_backoffs': backoffs,
        'live_G_scale': float(tr.g_scaler.get_scale()), 'microbatch_sizes': sizes,
        'mask_logit': qstats(logit), 'M': {**qstats(Mv), 'frac_lt_1e-1': float((Mv < 1e-1).float().mean()),
                                           'frac_lt_1e-2': float((Mv < 1e-2).float().mean()),
                                           'frac_lt_1e-3': float((Mv < 1e-3).float().mean()),
                                           'frac_gt_0.5': float((Mv > 0.5).float().mean())},
        'sigmoid_derivative': qstats(fc['sig']),
        'abs_delta_LH': qstats(fc['dLH']), 'abs_delta_HL': qstats(fc['dHL']), 'abs_delta_HH': qstats(fc['dHH']),
        'abs_delta_HF_mean': float(hf_abs.mean()), 'M_times_abs_delta_HF': qstats(fc['MdHF']),
        'abs_xhat_minus_target': qstats(fc['res_t']), 'abs_xhat_minus_source': qstats(fc['res_s']),
        'A': qstats(fc['A']), 'A_image_means': fc['A'].flatten(1).mean(1).tolist(),
        'D_real_logits': qstats(fc['d_real']), 'D_fake_logits': qstats(fc['d_fake']),
        'BCE_real': float(torch.nn.functional.binary_cross_entropy_with_logits(
            fc['d_real'], torch.ones_like(fc['d_real']))),
        'BCE_fake_as_fake': float(torch.nn.functional.binary_cross_entropy_with_logits(
            fc['d_fake'], torch.zeros_like(fc['d_fake']))),
        'BCE_fake_as_real_gadv': float(torch.nn.functional.binary_cross_entropy_with_logits(
            fc['d_fake'], torch.ones_like(fc['d_fake']))),
        'terms': terms,
        'total': {'G_grad_l2_preclip': tnorm, 'clip_coefficient': clip, 'G_grad_l2_postclip': tnorm * clip,
                  'g_res_l2': float(sl(total, 'g_res.').norm()), 'e_art_l2': float(sl(total, 'e_art.').norm()),
                  'mask_logit_grad': signed(total_mask),
                  'mask_logit_tendency': tendency(signed(total_mask), float(total_mask.norm()) * 1e-3 + 1e-30),
                  'mask_w_grad_l2': float(total[mw].norm()), 'mask_b_grad': float(total[eb0 + MASK])},
        'nonfinite_terms': sorted(nonfinite_terms),
    }
    parts, pred_delta, adam_info = adam_parts(tr, u, unit, W, clip, used, offs, gsl)
    res['adam_step'] = adam_info
    res['logit_effect'] = logit_effect(tr, group, cur, parts, offs, gsl)
    res['_pred_delta'] = pred_delta                      # popped by analyze_snapshots (next-snapshot check)
    return res


def adam_parts(tr, u, unit, W, clip, used, offs, gsl):
    """Exact linear split of this update's G Adam step (torch.optim.Adam, betas (0.5, 0.999), no weight decay):
    delta = -lr/(1-b1^t) * (b1*m + (1-b1)*c*sum_k W_k g_k) / (sqrt(v_new)/sqrt(1-b2^t) + eps), with v_new and the clip
    coefficient c of the full gradient, so delta = momentum part + one part per weighted loss term."""
    import torch
    from methods.gpat import runtime_contract as rc
    step = tr.step.real if hasattr(tr.step, 'real') else tr.step
    grp = tr.g_opt.param_groups[0]
    b1, b2 = grp['betas']
    eps, lr = grp['eps'], rc.main_lr(u)
    ms, vs, ts, live = [], [], set(), []
    for (n, p), uu in zip(step.g_named, used):
        st = tr.g_opt.state.get(p)
        ok = bool(uu) and st is not None and len(st) > 0
        live.append(torch.full((p.numel(),), float(ok)))
        if ok:
            ms.append(st['exp_avg'].detach().float().reshape(-1).cpu())
            vs.append(st['exp_avg_sq'].detach().float().reshape(-1).cpu())
            ts.add(float(st['step']))
        else:
            ms.append(torch.zeros(p.numel()))
            vs.append(torch.zeros(p.numel()))
    assert len(ts) == 1, f'Adam step counts differ across G params: {ts}'
    t1 = ts.pop() + 1
    live = torch.cat(live)
    m, v = torch.cat(ms), torch.cat(vs)
    g = clip * sum(W[k] * unit[k] for k in TERMS)
    v_new = b2 * v + (1 - b2) * g * g
    denom = v_new.sqrt() / math.sqrt(1 - b2 ** t1) + eps
    ss = lr / (1 - b1 ** t1)
    parts = {'momentum': -ss * b1 * m / denom * live}
    for k in TERMS:
        if W[k]:
            parts[k] = -ss * (1 - b1) * clip * W[k] * unit[k] / denom * live
    total = sum(parts.values())
    ew0, ew1 = offs['g_res.ending.weight']
    per_out = (ew1 - ew0) // 13
    mrow = torch.zeros_like(total)
    mrow[ew0 + MASK * per_out: ew0 + (MASK + 1) * per_out] = 1
    mrow[offs['g_res.ending.bias'][0] + MASK] = 1
    pref = torch.cat([torch.full((k,), 1.0 if n.startswith('g_res.') else 0.0) for n, k in gsl])
    parts['TOTAL'] = total
    parts['TOTAL_mask_head_row_only'] = total * mrow
    parts['TOTAL_g_res_except_mask_row'] = total * pref * (1 - mrow)
    parts['TOTAL_e_art_only'] = total * (1 - pref)
    info = {'adam_t': t1, 'lr': lr, 'betas': [b1, b2], 'eps': eps, 'clip_coefficient': clip,
            'delta_l2': {k: float(x.norm()) for k, x in parts.items()},
            'momentum_l2_vs_current_l2': float(parts['momentum'].norm()) / (float((total - parts['momentum']).norm()) + 1e-30),
            'denominator': qstats(denom), 'v_old_mask_row_mean': float(v[mrow.bool()].mean()),
            'v_old_all_mean': float(v.mean())}
    return parts, total.detach().clone(), info


def logit_effect(tr, group, cur, parts, offs, gsl):
    """Effect of each Adam-step part on the mask logit / M / x_hat residual / A of this group: fp32 forward (autocast
    off) at theta and theta + part, BN buffers restored after every forward, theta restored bit-exactly at the end."""
    import torch
    from methods.gpat import runner_checkpoint as ck
    core = tr.core
    step = tr.step.real if hasattr(tr.step, 'real') else tr.step
    params = [p for _, p in step.g_named]
    theta0 = [p.detach().clone() for p in params]
    bufs = {n: b.detach().clone() for n, b in core.named_buffers()}
    rng = ck.rng_state()

    def fwd():
        L, Ms, R, A = [], [], [], []
        with torch.no_grad():
            for mb in group:
                out = core(mb['x_source'].float(), mb['x_target'].float(), scale_hf=cur['s_hf'])
                L.append(out.raw[:, MASK].float())
                Ms.append(out.M.float())
                R.append((out.x_hat.float() - mb['x_target'].float()).abs().flatten(1).mean(1))
                A.append(out.A.float().flatten(1).mean(1))
                for n, b in core.named_buffers():
                    b.copy_(bufs[n])
        return torch.cat(L), torch.cat(Ms), torch.cat(R), torch.cat(A)

    def setp(vec, scale=1.0):
        with torch.no_grad():
            o = 0
            for p, t0 in zip(params, theta0):
                k = p.numel()
                p.copy_(t0 + (0 if vec is None else scale * vec[o:o + k].view_as(p).to(p.device, p.dtype)))
                o += k
    base = fwd()
    out = {'base_fp32': {'logit_mean': float(base[0].mean()), 'M_mean': float(base[1].mean()),
                         'abs_xhat_minus_target_mean': float(base[2].mean()), 'A_mean': float(base[3].mean())}}
    sig = (base[1] * (1 - base[1]))
    for name, vec in parts.items():
        setp(vec)
        r = fwd()
        dl = (r[0] - base[0]).flatten()
        q = torch.quantile(dl[torch.linspace(0, dl.numel() - 1, min(dl.numel(), 4_000_000), device=dl.device).long()],
                           torch.tensor([0.1, 0.5, 0.9], device=dl.device)).tolist()
        out[name] = {'dlogit_mean': float(dl.mean()), 'dlogit_p10': q[0], 'dlogit_p50': q[1], 'dlogit_p90': q[2],
                     'dlogit_frac_neg': float((dl < 0).float().mean()),
                     'dlogit_sigweighted_mean': float(((r[0] - base[0]) * sig).sum() / (sig.sum() + 1e-30)),
                     'dM_mean': float((r[1] - base[1]).mean()), 'dres_mean': float((r[2] - base[2]).mean()),
                     'dA_mean': float((r[3] - base[3]).mean()),
                     'direction': 'CLOSE_MASK' if float(dl.mean()) < 0 else 'OPEN_MASK'}
        setp(vec, LIN_EPS)                               # first-order (directional derivative), additive over parts
        r = fwd()
        dl = (r[0] - base[0]) / LIN_EPS
        out[name].update({'lin_dlogit_mean': float(dl.mean()), 'lin_dlogit_frac_neg': float((dl < 0).float().mean()),
                          'lin_dlogit_sigweighted_mean': float((dl * sig).sum() / (sig.sum() + 1e-30)),
                          'lin_dM_mean': float(((r[1] - base[1]) / LIN_EPS).mean()),
                          'lin_dres_mean': float(((r[2] - base[2]) / LIN_EPS).mean()),
                          'lin_direction': 'CLOSE_MASK' if float(dl.mean()) < 0 else 'OPEN_MASK'})
    setp(None)
    ck.restore_rng(rng)
    with torch.no_grad():
        for n, b in core.named_buffers():
            b.copy_(bufs[n])
    terms_sum = sum(out[k]['dlogit_mean'] for k in parts if k not in ('TOTAL',) and not k.startswith('TOTAL_'))
    out['additivity'] = {'sum_of_parts_dlogit_mean': terms_sum, 'TOTAL_dlogit_mean': out['TOTAL']['dlogit_mean'],
                         'lin_sum_of_parts': sum(out[k]['lin_dlogit_mean'] for k in parts
                                                 if k != 'TOTAL' and not k.startswith('TOTAL_')),
                         'lin_TOTAL': out['TOTAL']['lin_dlogit_mean'], 'lin_eps': LIN_EPS}
    return out


def grads_at_scale(tr, group, u, cur, S):
    """Training-equivalent G gradient (sum of weighted terms, sample-weighted microbatches) at loss scale S, unscaled.
    Returns the flat fp32 gradient and the names of parameters with non-finite unscaled gradients."""
    import torch
    from methods.gpat import losses
    from methods.gpat import runner_checkpoint as ck
    core, step = tr.core, tr.step.real if hasattr(tr.step, 'real') else tr.step
    bufs = {n: b.detach().clone() for n, b in core.named_buffers()}
    rng = ck.rng_state()
    sizes = [int(mb['x_source'].shape[0]) for mb in group]
    wmb = [s / sum(sizes) for s in sizes]
    params = [p for _, p in step.g_named]
    acc = [torch.zeros_like(p, dtype=torch.float32) for p in params]
    core.discriminator.requires_grad_(False)
    for j, mb in enumerate(group):
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            out = core(mb['x_source'], mb['x_target'], scale_hf=cur['s_hf'])
        comps = step.components(mb, out, cur)
        total, _ = losses.assemble_generator_loss(comps, cur, lambda_type=0.0, lambda_idadv=0.0)
        g = torch.autograd.grad(wmb[j] * total * S, params, allow_unused=True)
        for a, x in zip(acc, g):
            if x is not None:
                a += x.float()
        del out, comps
    core.discriminator.requires_grad_(True)
    with torch.no_grad():
        for n, b in core.named_buffers():
            b.copy_(bufs[n])
    ck.restore_rng(rng)
    flat = torch.cat([a.reshape(-1) / S for a in acc])
    bad = [n for (n, _), a in zip(step.g_named, acc) if not bool(torch.isfinite(a).all())]
    return flat, bad


def scaler_sensitivity(tr, group, u, cur, scales):
    import torch
    res, ref = {}, None
    for S in scales:
        flat, bad = grads_at_scale(tr, group, u, cur, S)
        fin = torch.isfinite(flat)
        rec = {'nonfinite_params': bad[:30], 'nonfinite_param_count': len(bad),
               'finite_fraction': float(fin.float().mean()),
               'norm_finite_part': float(flat[fin].norm())}
        if not bad:
            if ref is None:
                ref = (S, flat)
            else:
                rec['rel_diff_vs_scale_%g' % ref[0]] = float((flat - ref[1]).norm() / (ref[1].norm() + 1e-30))
        res[str(S)] = rec
    return res


def recorded_groups(run_name):
    rec = {}
    p = diag_root() / run_name / 'metrics.jsonl'
    if p.exists():
        for line in p.read_text().splitlines():
            d = json.loads(line)
            if d['record_type'] == 'optimizer_group':
                rec[d['global_update']] = d
    return rec


def analyze_snapshots(tr, snap_dir, updates, curriculum_fn, scaler_updates=(), recorded=None):
    import torch
    out = {}
    prev = None                                          # (u, predicted theta_{u+1} - theta_u, theta_u) on CPU
    step = tr.step.real if hasattr(tr.step, 'real') else tr.step
    for u in updates:
        path = snap_dir / f'pre_update_{u}.pt'
        snap, meta = load_snapshot(path)
        group = apply_snapshot(tr, snap)
        cur = curriculum_fn(u)
        theta = torch.cat([p.detach().float().reshape(-1) for _, p in step.g_named]).cpu()
        if prev is not None and prev[0] == u - 1:
            actual = theta - prev[2]
            out[str(u - 1)]['next_snapshot_check'] = {
                'pred_vs_actual_rel_err': float((prev[1] - actual).norm() / (actual.norm() + 1e-30)),
                'actual_delta_l2': float(actual.norm()), 'pred_delta_l2': float(prev[1].norm())}
        r = attribute(tr, group, u, cur)
        assert ck_state_equal(tr, snap), f'attribution perturbed the state at u{u}'
        prev = (u, r.pop('_pred_delta'), theta)
        if recorded and u in recorded:
            rg = recorded[u]['G_grad_norm']
            r['recorded_G_grad_norm'] = rg
            r['attribution_vs_recorded_G_grad_norm_rel'] = abs(r['total']['G_grad_l2_preclip'] - rg) / (rg + 1e-30)
        r['snapshot'] = {'path': meta['path'], 'sha256': meta['sha256']}
        if u in scaler_updates:
            s0 = float(snap['scalers']['G_SCALER']['scale'])
            r['scaler_sensitivity'] = scaler_sensitivity(tr, group, u, cur, (s0, s0 / 2, s0 / 4))
            assert ck_state_equal(tr, snap), f'scaler probe perturbed the state at u{u}'
        out[str(u)] = r
        del snap, group
    return out


def ck_state_equal(tr, snap):
    from methods.gpat import runner_checkpoint as ck
    got = ck.primitives(full_state(tr))
    return got == snap['state_digest']


def frozen_curriculum(u):
    from methods.gpat import runtime_contract as rc
    return rc.curriculum(u)


def scenario_analyze(args):
    import torch
    from methods.gpat import runner as R
    src = args.source or 'replay'
    ups = tuple(int(x) for x in args.updates.split(',')) if args.updates else SNAP_UPDATES
    tr = setup(args.name or ('analyze' if src == 'replay' else f'analyze_{src}'),
               {'purpose': f'per-loss gradient attribution on {src} snapshots', 'fork_curriculum': args.fork})
    sd = diag_root() / 'snapshots' / src
    cur_fn = make_fork_curriculum(args.fork) if args.fork else frozen_curriculum
    out = analyze_snapshots(tr, sd, ups, cur_fn,
                            scaler_updates=(5758, 5759) if src in ('replay', 'fork_A') else (),
                            recorded=recorded_groups(src))
    res = {'snapshots': out, 'access': tr.access.report(),
           'peak_reserved_GiB': round(torch.cuda.max_memory_reserved() / 2 ** 30, 3)}
    tr.ctx.close('diagnostic_completed', {'access': res['access']})
    return res


# ----------------------------------------------------------------------------- forks
def make_fork_curriculum(name):
    from methods.gpat import runtime_contract as rc
    real_curriculum = rc.curriculum
    values = FORKS[name]

    def fork_curriculum(u):
        c = real_curriculum(u)
        if u < FORK_FROM:
            return c
        assert u < 16576, 'forks stay inside the frozen stage-2 window'
        return {'stage': f'FORK_{name}', **values}
    return fork_curriculum


def scenario_fork(args):
    """Steps only (no attribution: `--scenario analyze --source fork_<X> --fork <X>` does that with the same
    curriculum). Fork A uses the frozen stage-2 values, i.e. it re-runs the scientific trajectory: its records are
    compared field by field (EMA included) with the scientific metrics.jsonl."""
    import torch
    from methods.gpat import runtime_contract as rc
    name = args.fork
    end = args.end or FORK_END
    values = FORKS[name]
    rc.curriculum = make_fork_curriculum(name)          # this diagnostic process only; configs untouched
    run_name = args.name or f'fork_{name}'
    tr = setup(run_name, {'purpose': f'curriculum fork {name} from pre-u{FORK_FROM}, u{FORK_FROM}..u{end}',
                                'fork_values_for_u_ge_5526': values})
    snap, meta = load_snapshot(diag_root() / 'snapshots' / 'replay' / f'pre_update_{FORK_FROM}.pt')
    apply_snapshot(tr, snap)
    del snap
    assert tr.step.ema is tr.ema and tr.ema is not None, 'EMA must be live on the real step'
    sd = diag_root() / 'snapshots' / run_name
    snaps = (sorted(int(x) for x in args.updates.split(',')) if args.updates else
             sorted(set(FORK_SNAP_UPDATES + (FORK_A_DENSE if name == 'A' else ())) | {end}))
    w = install(tr, sd, [x for x in snaps if x <= end], (5740, min(5790, end)))
    t0 = time.monotonic()
    tr.run_generator_groups(Stop(), limit=end - FORK_FROM + 1)
    assert tr.position['global_update'] == end
    recs, retries = {}, []
    for line in (tr.ctx.run_dir / 'metrics.jsonl').read_text().splitlines():
        d = json.loads(line)
        if d['record_type'] == 'optimizer_group':
            recs[d['global_update']] = d
        elif d['record_type'] == 'amp_retry':
            retries.append(d)
    write_jsonl(tr.ctx.run_dir / 'mask_per_update.jsonl', w.mask_rows)
    write_jsonl(tr.ctx.run_dir / 'mask_channel_trace.jsonl', w.trace_rows)
    res = {'fork': name, 'values': values, 'end': end, 'seconds_stepping': time.monotonic() - t0,
           'from_snapshot': meta['sha256'], 'records': len(recs),
           'amp_retries': [{k: r[k] for k in ('global_update_attempted', 'offending_optimizers', 'old_scale',
                                              'new_scale')} for r in retries],
           'snapshots': w.snaps, 'access': tr.access.report(),
           'peak_reserved_GiB': round(torch.cuda.max_memory_reserved() / 2 ** 30, 3)}
    if name == 'A':
        sci, sci_retry = scientific_records()
        mism = {u: diff_fields(recs[u], sci[u]) for u in range(FORK_FROM, end + 1)}
        mism = {u: v for u, v in mism.items() if v}
        mine_r = {}
        for r in retries:
            mine_r.setdefault(r['global_update_attempted'], []).append(r)
        res['parity_vs_scientific'] = {
            'updates_compared': end - FORK_FROM + 1, 'bitwise_equal_updates': end - FORK_FROM + 1 - len(mism),
            'mismatched_updates': sorted(mism)[:100], 'first_mismatch_fields': mism[min(mism)] if mism else None,
            'amp_retry_event_mismatch': sorted(x for x in set(mine_r) | {y for y in sci_retry if FORK_FROM <= y <= end}
                                               if [strip(r) for r in mine_r.get(x, [])] !=
                                               [strip(r) for r in sci_retry.get(x, [])])}
    tr.ctx.close('diagnostic_completed', {'access': res['access']})
    return res


# ----------------------------------------------------------------------------- collect / diagnose (CPU, stdlib)
COLLAPSE_M = 0.01                     # a run is COLLAPSED once the per-update mean M (mask hook) drops below this
SINGLE_FORKS = ('A', 'B', 'C', 'D', 'E', 'F')
NECESSITY_FORK = 'BDE'                # stage 2 without the lambda_adv change
PARTS = ('gadv', 'artcon', 'bg', 'lm', 'id', 'parse', 'tv', 'spec', 'budget', 'low', 'momentum')
WINDOWS = ((5757, 5769), (5770, 5776), (5777, 5780))
TRAINING_FIELDS_EXCLUDED = ('ema_updated',)


def _groups(path, end):
    out = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if d.get('record_type') == 'optimizer_group' and d['global_update'] <= end:
                out[d['global_update']] = d
    return out


def _rows(path):
    with open(path) as f:
        return {d['global_update']: d for d in map(json.loads, f)}


def _compact_snapshot(r):
    le = r['logit_effect']
    keep = ('mean', 'p01', 'p50', 'p99')
    return {
        'logit_fp32_mean': le['base_fp32']['logit_mean'], 'M_fp32_mean': le['base_fp32']['M_mean'],
        'abs_xhat_minus_target_fp32_mean': le['base_fp32']['abs_xhat_minus_target_mean'],
        'A_fp32_mean': le['base_fp32']['A_mean'],
        'mask_logit': {k: r['mask_logit'][k] for k in keep}, 'M': {k: r['M'][k] for k in r['M'] if k in keep or
                                                                    k.startswith('frac')},
        'sigmoid_derivative': {k: r['sigmoid_derivative'][k] for k in keep},
        'abs_xhat_minus_target': {k: r['abs_xhat_minus_target'][k] for k in keep},
        'A': {k: r['A'][k] for k in keep}, 'abs_delta_HF_mean': r['abs_delta_HF_mean'],
        'D_real_logit_mean': r['D_real_logits']['mean'], 'D_fake_logit_mean': r['D_fake_logits']['mean'],
        'BCE_fake_as_real_gadv': r['BCE_fake_as_real_gadv'],
        'G_scale_used': r['G_scale_used'], 'attribution_scale_backoffs': r['attribution_scale_backoffs'],
        'G_grad_l2_preclip': r['total']['G_grad_l2_preclip'], 'clip_coefficient': r['total']['clip_coefficient'],
        'recorded_G_grad_norm': r.get('recorded_G_grad_norm'),
        'attribution_vs_recorded_G_grad_norm_rel': r.get('attribution_vs_recorded_G_grad_norm_rel'),
        'next_snapshot_pred_vs_actual_rel_err': (r.get('next_snapshot_check') or {}).get('pred_vs_actual_rel_err'),
        'adam': {k: r['adam_step'][k] for k in ('adam_t', 'lr', 'momentum_l2_vs_current_l2', 'v_old_mask_row_mean',
                                                'v_old_all_mean')} | {'delta_l2_TOTAL': r['adam_step']['delta_l2']['TOTAL']},
        'terms': {k: {'weight': v['weight'], 'value': v['value'], 'projection_share_of_total':
                      v['projection_share_of_total'], 'cosine_with_total': v['cosine_with_total'],
                      'all_G_grad_l2_weighted': v['all_G_grad_l2_weighted'],
                      'direct_dL_dlogit_weighted_mean': v['mask_logit_grad_weighted']['mean'],
                      'direct_dL_dlogit_weighted_l2': v['mask_logit_grad_weighted']['l2'],
                      'direct_tendency_weighted': v['tendency_weighted']} for k, v in r['terms'].items()},
        'step_effect_lin': {k: {'dlogit_mean': le[k]['lin_dlogit_mean'], 'dlogit_sigweighted_mean':
                                le[k]['lin_dlogit_sigweighted_mean'], 'dM_mean': le[k]['lin_dM_mean'],
                                'dres_mean': le[k]['lin_dres_mean'], 'direction': le[k]['lin_direction']}
                            for k in le if k not in ('base_fp32', 'additivity')},
        'step_effect_full': {k: {'dlogit_mean': le[k]['dlogit_mean'], 'dM_mean': le[k]['dM_mean'],
                                 'dres_mean': le[k]['dres_mean']} for k in le if k not in ('base_fp32', 'additivity')},
        'additivity': le['additivity'],
        'scaler_sensitivity': r.get('scaler_sensitivity'),
    }


def _traj(rows, groups, ups):
    return {str(u): {'logit_mean': rows[u]['logit_mean'], 'logit_min': rows[u]['logit_min'],
                     'logit_max': rows[u]['logit_max'], 'M_mean': rows[u]['M_mean'],
                     'frac_M_lt_1e-2': rows[u]['frac_M_lt_1e-2'], 'sigmoid_deriv_mean': rows[u]['sigmoid_deriv_mean'],
                     'tanh_hf_absmean': rows[u]['tanh_hf_absmean'], 'mask_bias': rows[u]['mask_bias'],
                     'G_grad_norm': groups[u]['G_grad_norm'], 'D_grad_norm': groups[u]['D_grad_norm'],
                     'G_scale': groups[u]['G_scale'], 'D_scale': groups[u]['D_scale'],
                     'amp_attempts': groups[u]['amp_attempts'],
                     'losses': {k: groups[u]['train_losses'][k] for k in ('budget', 'bg', 'lm', 'id', 'artcon', 'spec',
                                                                          'tv', 'gadv', 'D_total')}}
            for u in ups if u in rows}


def scenario_collect(args):
    """CPU only: compact evidence document from the diagnostic results (never touches the scientific root except
    reading its metrics.jsonl for parity)."""
    D = diag_root()
    res = D / 'results'
    sci = _groups(sci_dir() / 'metrics.jsonl', REPLAY_END)
    rep_g = _groups(D / 'replay' / 'metrics.jsonl', REPLAY_END)
    diff = {}
    for u in range(1, REPLAY_END + 1):
        for f in diff_fields(rep_g[u], sci[u]):
            diff.setdefault(f, []).append(u)
    replay = json.loads((res / 'replay.json').read_text())
    rows = _rows(D / 'replay' / 'mask_per_update.jsonl')
    ups = sorted(set(range(1, REPLAY_END + 1, 50)) | set(range(5500, REPLAY_END + 1, 5)) |
                 set(range(5740, 5801)) | set(PARITY_REPORT))
    doc = {'labels': LABELS, 'kind': 'GPAT_M7D1_N3_COLLECTED', 'method': METHOD, 'experiment': 'E08', 'seed': SEED,
           'scientific_run_id': SCI_RUN_ID, 'collapse_M_threshold': COLLAPSE_M,
           'replay': {'updates': REPLAY_END, 'seconds': replay['seconds'], 'tool_sha256': replay['tool_sha256'],
                      'code_head': replay['code_head'], 'write_firewall': replay['write_firewall'],
                      'access': replay['access'],
                      'records_compared': len(rep_g), 'differing_fields': {k: [min(v), max(v), len(v)]
                                                                          for k, v in diff.items()},
                      'amp_retry_updates_replay': replay['amp_retry_updates_replay'],
                      'amp_retry_updates_scientific': replay['amp_retry_updates_scientific'],
                      'amp_retry_event_mismatch': replay['amp_retry_event_mismatch'],
                      'representative': {str(u): {'bitwise_equal_training_fields':
                                                  not [f for f in diff_fields(rep_g[u], sci[u])
                                                       if f not in TRAINING_FIELDS_EXCLUDED],
                                                  'differing_fields': diff_fields(rep_g[u], sci[u]),
                                                  **{k: rep_g[u][k] for k in ('scale_hf', 'lambda_adv', 'lambda_con',
                                                                              'lambda_spec', 'G_grad_norm',
                                                                              'D_grad_norm', 'G_scale', 'D_scale',
                                                                              'amp_attempts')}}
                                         for u in PARITY_REPORT},
                      'snapshots': [{k: s[k] for k in ('path', 'sha256', 'global_update_attempted')}
                                    for s in replay['snapshots']]},
           'replay_trajectory': _traj(rows, rep_g, ups),
           'mask_channel_trace': [r for r in map(json.loads, (D / 'replay' / 'mask_channel_trace.jsonl')
                                                 .read_text().splitlines()) if 5740 <= r['global_update'] <= 5790],
           'forks': {}, 'attribution': {}}
    for name, f in (('replay', 'analyze_replay.json'), ('fork_A', 'analyze_fork_A.json')):
        a = json.loads((res / f).read_text())
        doc['attribution'][name] = {'tool_sha256': a['tool_sha256'], 'write_firewall': a['write_firewall'],
                                    'access': a['access'],
                                    'snapshots': {u: _compact_snapshot(r) for u, r in a['snapshots'].items()}}
    for name in SINGLE_FORKS + (NECESSITY_FORK,):
        p = res / f'fork_{name}.json'
        if not p.exists():
            continue
        fr = json.loads(p.read_text())
        frows = _rows(D / f'fork_{name}' / 'mask_per_update.jsonl')
        fg = _groups(D / f'fork_{name}' / 'metrics.jsonl', 10 ** 9)
        first = next((u for u in sorted(frows) if frows[u]['M_mean'] < COLLAPSE_M), None)
        fups = sorted(set(range(FORK_FROM, fr['end'] + 1, 10)) | {fr['end']} |
                      (set(range(first - 8, first + 3)) if first else set()))
        doc['forks'][name] = {'values': fr['values'], 'end': fr['end'], 'records': fr['records'],
                              'amp_retries': fr['amp_retries'], 'tool_sha256': fr['tool_sha256'],
                              'write_firewall': fr['write_firewall'], 'access': fr['access'],
                              'parity_vs_scientific': fr.get('parity_vs_scientific'),
                              'first_collapse_update': first,
                              'min_M_mean': min(r['M_mean'] for r in frows.values()),
                              'final_M_mean': frows[fr['end']]['M_mean'],
                              'final_logit_mean': frows[fr['end']]['logit_mean'],
                              'max_G_grad_norm': max(g['G_grad_norm'] for g in fg.values()),
                              'median_G_grad_norm': sorted(g['G_grad_norm'] for g in fg.values())[len(fg) // 2],
                              'trajectory': _traj(frows, fg, fups)}
    return doc


def diagnose(c):
    """Pure function of the collected document (re-derived by the tests)."""
    rp = c['replay']
    train_diff = {k: v for k, v in rp['differing_fields'].items() if k not in TRAINING_FIELDS_EXCLUDED}
    fa = c['forks'].get('A', {}).get('parity_vs_scientific') or {}
    parity = {
        'replay_records': rp['records_compared'],
        'replay_training_fields_bitwise_equal_all': not train_diff and rp['records_compared'] == REPLAY_END,
        'replay_only_differing_field': sorted(rp['differing_fields']),
        'amp_retry_events_equal': not rp['amp_retry_event_mismatch'],
        'fork_A_full_parity_u5526_u5850': fa.get('bitwise_equal_updates') == fa.get('updates_compared') == 325
        and not fa.get('amp_retry_event_mismatch'),
    }
    parity['status'] = ('PARITY_PASS_TRAINING_STATE' if parity['replay_training_fields_bitwise_equal_all'] and
                        parity['amp_retry_events_equal'] and parity['fork_A_full_parity_u5526_u5850']
                        else 'PARITY_FAIL')
    collapsed = {n: f['first_collapse_update'] for n, f in c['forks'].items()}
    singles = {n: collapsed.get(n) for n in SINGLE_FORKS if n in collapsed}
    changed = {'B': 's_hf', 'C': 'lambda_adv', 'D': 'lambda_con', 'E': 'lambda_spec'}
    sufficient = sorted(changed[n] for n in changed if singles.get(n) is not None)
    not_sufficient = sorted(changed[n] for n in changed if n in singles and singles[n] is None)
    nec = collapsed.get(NECESSITY_FORK, 'NOT_RUN')
    att = c['attribution']['fork_A']['snapshots']
    windows = {}
    for a, b in WINDOWS:
        s = {k: sum(att[str(u)]['step_effect_lin'].get(k, {'dlogit_mean': 0.0})['dlogit_mean'] for u in range(a, b + 1))
             for k in PARTS}
        tot = sum(att[str(u)]['step_effect_lin']['TOTAL']['dlogit_mean'] for u in range(a, b + 1))
        closing = sum(min(0.0, v) for v in s.values())
        windows[f'{a}-{b}'] = {'lin_dlogit_by_part': s, 'lin_dlogit_total': tot,
                               'gadv_share_of_closing': (min(0.0, s['gadv']) / closing) if closing else 0.0,
                               'largest_closing_part': min(s, key=s.get)}
    flip = windows['5777-5780']
    primary = None
    if (singles.get('C') is not None and singles.get('F') is None and nec is None and
            flip['largest_closing_part'] == 'gadv'):
        primary = 'LAMBDA_ADV_STAGE2_ACTIVATION__ADVERSARIAL_G_GRADIENT_CLOSES_MASK_VIA_SHARED_TRUNK'
    return {'parity': parity, 'fork_first_collapse_update': collapsed,
            'sufficient_single_changes': sufficient, 'not_sufficient_single_changes': not_sufficient,
            'necessity_fork': {'fork': NECESSITY_FORK, 'values': c['forks'].get(NECESSITY_FORK, {}).get('values'),
                               'first_collapse_update': nec},
            'attribution_windows_fork_A': windows,
            'primary_root_cause': primary or 'INCONCLUSIVE',
            'status': 'ROOT_CAUSE_LOCALIZED' if primary and parity['status'].startswith('PARITY_PASS')
            else 'INCONCLUSIVE'}


# ----------------------------------------------------------------------------- finalize (CPU, stdlib, derivation only)
AUTHORITY_COMMIT = '058e976e5a538c4d18e1177f0eb27727cfb735cd'     # code commit of the scientific run (close audit)
N2_COMMIT = 'b229bdc2e75148b883ba86f9191b126a1d45e96f'            # M7D1-N2 finding F01; head of every N3 process
COLLAPSE_ONSET = 5781
STAGE2_FIRST = 5526
FORK_CHANGED = {'A': ['s_hf', 'lambda_adv', 'lambda_con', 'lambda_spec'], 'B': ['s_hf'], 'C': ['lambda_adv'],
                'D': ['lambda_con'], 'E': ['lambda_spec'], 'F': [], 'BDE': ['s_hf', 'lambda_con', 'lambda_spec']}
FORK_LABEL = {'A': 'exact stage 2', 'B': 'only s_hf change', 'C': 'only lambda_adv change',
              'D': 'only lambda_con change', 'E': 'only lambda_spec change', 'F': 'hold stage-1 values',
              'BDE': 'stage 2 without the lambda_adv change (necessity test)'}
FIDELITY = ('bg', 'lm', 'id', 'parse')
GRAD_FIELDS = ('source', 'update', 'loss_term', 'loss_weight', 'loss_value', 'weighted_loss',
               'gradient_norm_weighted', 'gradient_projection_share', 'gradient_cosine_with_total',
               'direct_dL_dlogit_weighted_mean', 'direct_dL_dlogit_weighted_l2', 'direct_tendency',
               'adam_part_lin_dlogit_mean', 'adam_part_lin_dlogit_sigweighted_mean', 'adam_part_lin_dM_mean',
               'adam_part_lin_dres_mean', 'adam_part_full_dlogit_mean', 'adam_part_full_dM_mean',
               'adam_part_full_dres_mean', 'mask_direction', 'adam_t', 'lr', 'clip_coefficient',
               'G_grad_l2_preclip', 'adam_delta_l2_total', 'mask_logit_fp32_mean', 'M_fp32_mean')
FORK_FIELDS = ('fork_id', 'description', 'changed_factors', 's_hf', 'lambda_adv', 'lambda_con', 'lambda_spec',
               'start_update', 'end_update', 'records', 'collapse', 'collapse_update', 'min_M_mean',
               'final_M_mean', 'final_mask_logit_mean', 'final_frac_M_lt_1e-2', 'final_tanh_hf_absmean',
               'final_budget_loss', 'final_bg_loss', 'final_gadv_loss', 'median_G_grad_norm', 'max_G_grad_norm',
               'amp_retry_updates', 'parity_vs_scientific')
ADAM_PARTS = ('momentum', 'TOTAL', 'TOTAL_mask_head_row_only', 'TOTAL_g_res_except_mask_row', 'TOTAL_e_art_only')


def _csv_cell(v):
    if v is None:
        return ''
    if isinstance(v, bool):
        return 'true' if v else 'false'
    if isinstance(v, float):
        return repr(v)
    return str(v)


def render_csv(rows, fields):
    import csv
    import io
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\n')
    w.writerow(fields)
    for r in rows:
        w.writerow([_csv_cell(r.get(f)) for f in fields])
    return buf.getvalue()


def gradient_rows(c):
    rows = []
    for src in ('replay', 'fork_A'):
        snaps = c['attribution'][src]['snapshots']
        for u in sorted(snaps, key=int):
            s = snaps[u]
            common = {'source': src, 'update': int(u), 'adam_t': s['adam']['adam_t'], 'lr': s['adam']['lr'],
                      'clip_coefficient': s['clip_coefficient'], 'G_grad_l2_preclip': s['G_grad_l2_preclip'],
                      'adam_delta_l2_total': s['adam']['delta_l2_TOTAL'],
                      'mask_logit_fp32_mean': s['logit_fp32_mean'], 'M_fp32_mean': s['M_fp32_mean']}
            for k in list(TERMS) + list(ADAM_PARTS):
                r = dict(common, loss_term=k)
                t = s['terms'].get(k)
                if t is not None:
                    r.update(loss_weight=t['weight'], loss_value=t['value'], weighted_loss=t['weight'] * t['value'],
                             gradient_norm_weighted=t['all_G_grad_l2_weighted'],
                             gradient_projection_share=t['projection_share_of_total'],
                             gradient_cosine_with_total=t['cosine_with_total'],
                             direct_dL_dlogit_weighted_mean=t['direct_dL_dlogit_weighted_mean'],
                             direct_dL_dlogit_weighted_l2=t['direct_dL_dlogit_weighted_l2'],
                             direct_tendency=t['direct_tendency_weighted'])
                lin, full = s['step_effect_lin'].get(k), s['step_effect_full'].get(k)
                if lin is not None:
                    r.update(adam_part_lin_dlogit_mean=lin['dlogit_mean'],
                             adam_part_lin_dlogit_sigweighted_mean=lin['dlogit_sigweighted_mean'],
                             adam_part_lin_dM_mean=lin['dM_mean'], adam_part_lin_dres_mean=lin['dres_mean'],
                             adam_part_full_dlogit_mean=full['dlogit_mean'], adam_part_full_dM_mean=full['dM_mean'],
                             adam_part_full_dres_mean=full['dres_mean'], mask_direction=lin['direction'])
                elif t is not None and t['weight'] == 0:
                    r['mask_direction'] = 'ZERO_WEIGHT'
                rows.append(r)
    return rows


def fork_rows(c):
    rows = []
    for name in SINGLE_FORKS + (NECESSITY_FORK,):
        f = c['forks'][name]
        last = f['trajectory'][str(f['end'])]
        p = f['parity_vs_scientific']
        rows.append({'fork_id': name, 'description': FORK_LABEL[name],
                     'changed_factors': '+'.join(FORK_CHANGED[name]) or 'none',
                     **{k: f['values'][k] for k in ('s_hf', 'lambda_adv', 'lambda_con', 'lambda_spec')},
                     'start_update': FORK_FROM, 'end_update': f['end'], 'records': f['records'],
                     'collapse': f['first_collapse_update'] is not None, 'collapse_update': f['first_collapse_update'],
                     'min_M_mean': f['min_M_mean'], 'final_M_mean': f['final_M_mean'],
                     'final_mask_logit_mean': f['final_logit_mean'], 'final_frac_M_lt_1e-2': last['frac_M_lt_1e-2'],
                     'final_tanh_hf_absmean': last['tanh_hf_absmean'], 'final_budget_loss': last['losses']['budget'],
                     'final_bg_loss': last['losses']['bg'], 'final_gadv_loss': last['losses']['gadv'],
                     'median_G_grad_norm': f['median_G_grad_norm'], 'max_G_grad_norm': f['max_G_grad_norm'],
                     'amp_retry_updates': ';'.join(str(r['global_update_attempted']) for r in f['amp_retries']),
                     'parity_vs_scientific': (None if p is None else
                                              f"{p['bitwise_equal_updates']}/{p['updates_compared']}")})
    return rows


def build_root_cause(c, run_root_verification):
    """Final root-cause document; every number is read from the collected document `c` (or the recorded read-only
    run-root verification)."""
    d = c['diagnosis']
    att = c['attribution']['fork_A']['snapshots']
    rt = c['replay_trajectory']
    w = d['attribution_windows_fork_A']
    flip = w['5777-5780']['lin_dlogit_by_part']
    fid = {k: flip[k] for k in FIDELITY}
    traj_u = [u for u in (1105, 5525, 5526, 5700, 5750, 5758, 5759, 5770, 5775, 5776, 5777, 5778, 5779, 5780,
                          5781, 5782, 5790, 5800, 5850) if str(u) in rt]
    sc = {u: c['attribution']['replay']['snapshots'][u]['scaler_sensitivity'] for u in ('5758', '5759')}
    clip = {u: att[str(u)]['clip_coefficient'] for u in range(5757, 5781)}
    access_blocks = ([('replay', c['replay']['access'])] +
                     [(f'analyze_{k}', v['access']) for k, v in c['attribution'].items()] +
                     [(f'fork_{k}', v['access']) for k, v in c['forks'].items()])
    fw_blocks = ([('replay', c['replay']['write_firewall'])] +
                 [(f'analyze_{k}', v['write_firewall']) for k, v in c['attribution'].items()] +
                 [(f'fork_{k}', v['write_firewall']) for k, v in c['forks'].items()])
    ok_access = all(a[k] == 0 for _, a in access_blocks for k in ('val_images', 'test_images', 'val_metadata',
                                                                   'test_metadata', 'non_train_images'))
    ok_fw = all(not f['denied'] and not f['outside_allowed'] for _, f in fw_blocks)
    forks = {r['fork_id']: {k: r[k] for k in ('description', 'changed_factors', 's_hf', 'lambda_adv', 'lambda_con',
                                              'lambda_spec', 'start_update', 'end_update', 'collapse',
                                              'collapse_update', 'min_M_mean', 'final_M_mean')} for r in fork_rows(c)}
    return {
        'kind': 'GPAT_M7D1_N3_COLLAPSE_ROOT_CAUSE', 'milestone': 'M7D1-N3', 'labels': c['labels'],
        'status': d['status'],
        'scientific_run': {'method': c['method'], 'experiment': c['experiment'], 'seed': c['seed'],
                           'run_id': c['scientific_run_id'], 'authority_commit': AUTHORITY_COMMIT,
                           'finding_commit_M7D1_N2': N2_COMMIT, 'diagnostic_processes_code_head': c['replay']['code_head']},
        'classification': {
            'primary': 'ADVERSARIAL_IMBALANCE / CURRICULUM_TRANSITION_INSTABILITY',
            'mechanism': 'MASK_SIGMOID_SATURATION through shared G_res trunk',
            'diagnose_rule_result': d['primary_root_cause'],
            'statement': ('Activation of the generator adversarial loss (lambda_adv 0 -> 0.05) at the stage-2 '
                          'boundary against a discriminator trained for 5525 updates with lambda_adv = 0: the gadv '
                          'gradient dominates the generator update; through the shared G_res trunk it drives the '
                          'mask logit into negative sigmoid saturation (absorbing x_hat == x_t).'),
            'lambda_adv_sufficient_and_necessary_scope': (
                'ONLY within the tested diagnostic window u5526..u5850 from the exact pre-u5526 state of seed 42 '
                '(one seed, 325 updates); not generalized beyond that window or beyond seed 42'),
            'sufficient_single_changes': d['sufficient_single_changes'],
            'not_sufficient_single_changes': d['not_sufficient_single_changes'],
            'necessity_fork': d['necessity_fork']},
        'timeline': {'stage2_transition_update': STAGE2_FIRST, 'collapse_onset_update': COLLAPSE_ONSET,
                     'collapse_definition': f'per-update mean M < {COLLAPSE_M} (mask forward hook)',
                     'fork_A_first_collapse_update': d['fork_first_collapse_update']['A']},
        'parity': {**d['parity'], 'replay_differing_fields': c['replay']['differing_fields'],
                   'representative_updates': {u: v['bitwise_equal_training_fields']
                                              for u, v in c['replay']['representative'].items()},
                   'amp_retry_updates': c['replay']['amp_retry_updates_scientific'],
                   'ema_harness_caveat': (
                       'The u1..u5850 replay ran with a diagnostic step wrapper that forwarded only on_retry, so '
                       'Trainer.activate_ema() set step.ema on the wrapper and the replay EMA stayed at its epoch-5 '
                       'initialization: ema_updated False on u5526..u5850 (325 records). The EMA is write-only in '
                       'GeneratorStep; G/D/optimizer/scaler states are unaffected. Fixed (wrapper forwards ema).'),
                   'fork_A_corrected_parity': c['forks']['A']['parity_vs_scientific']},
        'mask_logit_trajectory': {str(u): {k: rt[str(u)][k] for k in ('logit_mean', 'logit_min', 'logit_max',
                                                                      'M_mean', 'frac_M_lt_1e-2',
                                                                      'sigmoid_deriv_mean', 'tanh_hf_absmean',
                                                                      'mask_bias')} for u in traj_u},
        'hf_residual_active': {'tanh_hf_absmean': {str(u): rt[str(u)]['tanh_hf_absmean'] for u in traj_u
                                                   if u >= 5776},
                               'statement': 'HF residual remained active through and after the collapse; the '
                                            'collapse is a mask closure, not a dead HF branch'},
        'attribution': {
            'method': ('per-term gradients at the trained weight; exact linear split of the real G Adam step into '
                       'momentum + one part per weighted term; first-order (eps=1/64) and full-step effect on the '
                       'mask logit by fp32 forwards on the update group'),
            'self_checks': {
                'max_rel_err_vs_recorded_G_grad_norm': max(
                    (s['attribution_vs_recorded_G_grad_norm_rel'] or 0.0) for a in c['attribution'].values()
                    for s in a['snapshots'].values()),
                'max_rel_err_predicted_vs_next_snapshot_step': max(
                    (s['next_snapshot_pred_vs_actual_rel_err'] or 0.0) for a in c['attribution'].values()
                    for s in a['snapshots'].values())},
            'windows_fork_A': w,
            'flip_5777_5780': {'total_lin_dlogit': w['5777-5780']['lin_dlogit_total'],
                               'gadv_lin_dlogit': flip['gadv'], 'gadv_share_of_closing':
                               w['5777-5780']['gadv_share_of_closing'],
                               'momentum_lin_dlogit': flip['momentum'],
                               'fidelity_losses_lin_dlogit': fid, 'fidelity_losses_lin_dlogit_sum': sum(fid.values()),
                               'artcon_lin_dlogit': flip['artcon'], 'budget_lin_dlogit': flip['budget']},
            'momentum_after_collapse_u5781_lin_dlogit': att['5781']['step_effect_lin']['momentum']['dlogit_mean'],
            'trunk_vs_head_lin_dlogit': {str(u): {
                'g_res_except_mask_row': att[str(u)]['step_effect_lin']['TOTAL_g_res_except_mask_row']['dlogit_mean'],
                'mask_head_row': att[str(u)]['step_effect_lin']['TOTAL_mask_head_row_only']['dlogit_mean'],
                'e_art': att[str(u)]['step_effect_lin']['TOTAL_e_art_only']['dlogit_mean']}
                for u in range(5777, 5781)},
            'gadv_projection_share': {str(u): att[str(u)]['terms']['gadv']['projection_share_of_total']
                                      for u in (5526, 5700, 5757, 5777, 5778, 5779, 5780)}},
        'sigmoid_saturation': {
            'sigmoid_derivative_p50': {str(u): att[str(u)]['sigmoid_derivative']['p50']
                                       for u in (5776, 5777, 5778, 5779, 5780, 5781, 5782, 5800, 5850)},
            'budget_loss': {str(u): att[str(u)]['terms']['budget']['value'] for u in (5778, 5779, 5780, 5781, 5850)},
            'statement': ('after u5781 every mask logit is strongly negative (sigmoid derivative ~1e-21 and below); '
                          'the budget hinge on A = clip(M * ...) is pinned at 0.01 but has no gradient path to reopen '
                          'the mask; momentum and TV continue lowering the logit')},
        'amp_retry_role': {
            'role': 'RULED_OUT_AS_CAUSE (symptom of large gadv-dominated G gradients)',
            'updates': [5758, 5759],
            'scaler_sensitivity': sc,
            'facts': ['retries are atomic (no optimizer step on the overflowed attempt)',
                      'overflowing parameter g_res.ending.weight',
                      'accepted gradients scale-invariant to <= 6e-5 relative',
                      'fork C collapses with retries at different updates (5527, 5663)']},
        'grad_clipping_role': {
            'role': 'NOT_CAUSAL_NO_PROTECTIVE_EFFECT (no no-clip counterfactual was run)',
            'clip_coefficient_fork_A_5757_5780': {str(k): v for k, v in clip.items()},
            'statement': 'clipping shrinks the gradient norm but Adam normalizes per coordinate'},
        'forks': forks,
        'ruled_out': ['s_hf curriculum change (fork B)', 'lambda_con curriculum change (fork D)',
                      'lambda_spec curriculum change (fork E)', 's_hf+lambda_con+lambda_spec together (fork BDE)',
                      'AMP retry / loss scaling', 'gradient clipping as trigger', 'EMA (write-only)',
                      'replay/numerical artefact (bitwise parity)', 'dead HF branch',
                      'direct mask-head drift', 'data order (identical groups in every fork)',
                      'stage-3 change (collapse inside stage 2)'],
        'secondary_contributors': [
            'stage-1 discriminator over-confidence (D trains from u1 while lambda_adv = 0)',
            'mask parametrization without recovery path (sigmoid saturation; budget acts only when A < 0.01)',
            'Adam momentum carrying the closing direction; per-coordinate normalization defeats clipping',
            'weak closing preference of fidelity losses (bg, lm, id, parse); artcon closing in u5757-5769'],
        'limitations': [
            'forks end at u5850 (69 updates after fork-A collapse, 134 after fork-C): no-collapse for B, D, E, F, '
            'BDE holds only within 325 updates; one seed (42)',
            'attribution is first-order per update (full-step effects reported; signs agree on the flip totals)',
            'the mechanism by which gadv closes the mask (D keying on the generator HF residual) is an '
            'interpretation, not separately tested',
            'diagnose rules and the collapse threshold (mean M < 0.01) were written after the replay trajectory '
            'was seen',
            'the tool changed during the milestone (replay ran with sha 1610d939..., copy kept on the GPU host); '
            'result JSONs record the tool sha read at process end; the G/D update computation was never changed',
            'stage-1 partial mask closure (u~1751-2900, 15-22% of pixels, recovered) is not analyzed',
            'no no-clip counterfactual was run',
            'moved-aside smoke/aborted runs are not evidence'],
        'access_and_safety': {
            'val_test_access_zero_all_processes': ok_access,
            'write_firewall_clean_all_processes': ok_fw,
            'scientific_run_root_unchanged': run_root_verification,
            'bank': 'NOT_TOUCHED', 'seed1337_2026': 'NOT_LAUNCHED', 'B1_B2_B3': 'NOT_LAUNCHED',
            'val_selection': 'NOT_RUN', 'fix': 'NOT_IMPLEMENTED',
            'diagnostic_runtime_in_git': False},
        'owner_decision_required': ('any fix is a protocol change; VAL selection, bank and seeds 1337/2026 remain '
                                    'blocked on the owner decision'),
        'source_collected_sha256': None,
    }


def finalize(collected_path, run_root_verification):
    raw = Path(collected_path).read_bytes()
    c = json.loads(raw)
    doc = build_root_cause(c, run_root_verification)
    doc['source_collected_sha256'] = sha(raw)
    return (json.dumps(doc, indent=1, sort_keys=True) + '\n', render_csv(gradient_rows(c), GRAD_FIELDS),
            render_csv(fork_rows(c), FORK_FIELDS))


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--scenario', choices=('replay', 'analyze', 'fork', 'collect'), required=True)
    ap.add_argument('--fork', choices=sorted(FORKS))
    ap.add_argument('--out', required=True)
    ap.add_argument('--source', default=None, help='analyze: snapshot directory name (default replay)')
    ap.add_argument('--updates', default=None, help='analyze: updates to attribute; fork: snapshot updates')
    ap.add_argument('--limit', type=int, default=None, help='replay smoke: stop after this many updates')
    ap.add_argument('--end', type=int, default=None, help='fork: last update (default 5850)')
    ap.add_argument('--name', default=None, help='analyze/fork: diagnostic context name (default derived)')
    args = ap.parse_args()
    fw = install_write_firewall()
    t0 = time.monotonic()
    if args.scenario == 'collect':
        doc = scenario_collect(args)
        doc['diagnosis'] = diagnose(doc)
        doc['write_firewall_collect'] = fw
        doc['collect_tool_sha256'] = file_sha(__file__)
        Path(args.out).write_text(json.dumps(doc, indent=1, sort_keys=True) + '\n')
        print(f'collect: {doc["diagnosis"]["status"]} {doc["diagnosis"]["primary_root_cause"]}', flush=True)
        return
    out = {'replay': scenario_replay, 'analyze': scenario_analyze, 'fork': scenario_fork}[args.scenario](args)
    out['write_firewall'] = fw
    out['seconds_total'] = round(time.monotonic() - t0, 1)
    out['code_head'] = git_head()
    out['tool_sha256'] = file_sha(__file__)
    Path(args.out).write_text(json.dumps(out, indent=1, sort_keys=True, default=str) + '\n')
    print(f'{args.scenario} {args.fork or ""}: done in {out["seconds_total"]} s', flush=True)


if __name__ == '__main__':
    main()
