"""M7D1-N1 GPAT AMP overflow diagnosis + ATOMIC_AMP_BACKOFF_RETRY qualification (gpat-m7-gpu). QUALIFICATION ONLY.

Replays the failed scientific attempt 2 of GPAT-B0 / E08 / seed 42 (FAIL_CLOSED_AMP_OVERFLOW at update 1971) from a
byte-identical COPY of its end-epoch-1 recovery checkpoint (a34e73d1..., update 1105). It never writes into, resumes or
overwrites <runtime_root>/runs/m7/E08/seed_42: that run root is only hashed (before/after) and its metrics.jsonl is
only read to compare records. Every output lives under <runtime_root>/qualification/m7/M7D1_N1/ and is labelled
QUALIFICATION_ONLY / NOT_SCIENTIFIC / NOT_ELIGIBLE_FOR_BANK / SELECTION / PAPER_RESULT. The replay uses the scientific
seed-42 epoch order (the only way to reproduce the scientific group 866) inside a qualification context that writes no
checkpoint of any kind.

Scenarios (each a fresh process; `--all` runs them in this order):
  preserve_before  hash every file of the failed scientific run root; check its run_summary
  replay           M7C4 policy (runner at the authority commit, unmodified): load the recovery copy, run updates
                   1106..1970, capture the exact pre-update-1971 snapshot, attempt 1971 -> expect the overflow
  probe            from the snapshot: update 1971 at D scales 65536..4096 (G 65536), M7C4 fail-closed attempts
  equivalence      from the snapshot: A = ATOMIC_AMP_BACKOFF_RETRY; B = one attempt from the pre-group state at A's
                   accepted scale(s); C = negative control (retry WITHOUT restoring buffers/RNG)
  continue_        from the snapshot, real loader: update 1971 under the retry policy + 20 further updates (1972..1991)
  finite_path      new code from the recovery copy: updates 1106..1125 must equal the scientific records
  preserve_after   re-hash the failed scientific run root
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from methods.gpat import runner_io as rio                   # noqa: E402

AUTHORITY = '6cf271ebd0d658dcc5384da00265326e7bac6476'
METHOD, SEED, EXPERIMENT = 'GPAT-B0', 42, 'E08'
RECOVERY_SHA256 = 'a34e73d1d8e8f6571ee214aba221a7d0858893cc25208df6be24531239e66706'
RECOVERY_UPDATE = 1105
FAILED_UPDATE, FAILED_EPOCH, FAILED_GROUP = 1971, 2, 866
FAILED_D_PARAMS = ['discriminator.model.0.weight', 'discriminator.model.2.weight']
PROBE_D_SCALES = (65536.0, 32768.0, 16384.0, 8192.0, 4096.0)
POST_FAILURE_UPDATES = 20
FINITE_PATH_UPDATES = 20
QUAL_PARTS = ('qualification', 'm7', 'M7D1_N1')
LABELS = list(rio.QUALIFICATION_LABELS)
SNAPSHOT_KIND = 'GPAT_M7D1_PRE_UPDATE_SNAPSHOT'
SCENARIOS = ('preserve_before', 'replay', 'probe', 'equivalence', 'continue_', 'finite_path', 'preserve_after')
VOLATILE = ('utc', 'wall_seconds', 'peak_allocated_bytes')
ADDITIVE = ('amp_policy', 'amp_attempts')
ENVELOPE = ('record_type', 'epoch', 'group')     # metrics.jsonl envelope of a step record


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1 << 22), b''):
            h.update(block)
    return h.hexdigest()


def runtime_root():
    return Path(rio.faces_root()['runtime_root'])


def sci_dir():
    return rio.run_root(runtime_root(), rio.SCIENTIFIC, METHOD, SEED)


def qroot():
    return runtime_root().joinpath(*QUAL_PARTS)


def strip(rec, extra=()):
    return {k: v for k, v in rec.items() if k not in VOLATILE + tuple(extra)}


# ----------------------------------------------------------------------------- Part A (no torch)
def scenario_preserve(_):
    d = sci_dir()
    files = {str(p.relative_to(d)): {'sha256': file_sha(p), 'size': p.stat().st_size}
             for p in sorted(d.rglob('*')) if p.is_file()}
    summary = json.loads((d / 'run_summary.json').read_text())
    manifest = json.loads((d / 'run_manifest.json').read_text())
    return {'run_dir': str(d), 'files': files, 'tree_sha256': sha(json.dumps(files, sort_keys=True).encode()),
            'completion_status': summary['completion_status'], 'global_update': summary['global_update'],
            'access': summary['access'], 'val_split_accessed': summary['val_split_accessed'],
            'test_split_accessed': summary['test_split_accessed'], 'manifest_status': manifest['completion_status'],
            'run_id': manifest['run_id'], 'git_commit': manifest['git_commit'],
            'recovery_sha256': files['checkpoints/recovery/latest.pt']['sha256']}


scenario_preserve_before = scenario_preserve_after = scenario_preserve


# ----------------------------------------------------------------------------- torch helpers
def state_digest(obj):
    import torch
    h = hashlib.sha256()

    def walk(node, key=''):
        if torch.is_tensor(node):
            h.update(key.encode() + str(node.dtype).encode() + str(tuple(node.shape)).encode())
            h.update(node.detach().contiguous().cpu().numpy().tobytes())
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


MODULES = ('e_art', 'g_res', 'discriminator', 'attack_head', 'identity_head')


def full_state(tr):
    """Every mutable training state component, digested separately (parameters, buffers, optimizers, scalers, RNG)."""
    from methods.gpat import runner_checkpoint as ck
    out = {}
    for k in MODULES:
        m = getattr(tr.core, k)
        if m is not None:
            out[f'params_{k}'] = state_digest({n: p for n, p in m.named_parameters()})
            out[f'buffers_{k}'] = state_digest({n: b for n, b in m.named_buffers()})
    out['e_art_bn_buffers'] = state_digest({n: b for n, b in tr.core.e_art.named_buffers()})
    out['all_core_buffers'] = state_digest({n: b for n, b in tr.core.named_buffers()})
    out['G_OPT'] = state_digest(tr.g_opt.state_dict())
    out['D_OPT'] = state_digest(tr.d_opt.state_dict())
    out['G_SCALER'] = tr.g_scaler.state_dict()
    out['D_SCALER'] = tr.d_scaler.state_dict()
    out['rng'] = state_digest(ck.rng_state())
    out['ema'] = None if tr.ema is None else {k: state_digest(v.state_dict()) for k, v in tr.ema.items()}
    out['position'] = dict(tr.position)
    return out


class ReplayContext:
    """Qualification-only stand-in for GPATRunContext: labelled metrics.jsonl/manifest/summary under M7D1_N1; no
    checkpoint can be recorded; the last safe recovery it reports is the scientific one (by hash, read-only)."""

    FILES = {'metrics': 'metrics.jsonl', 'checkpoint_index': 'checkpoint_index.json', 'run_manifest': 'run_manifest.json',
             'run_summary': 'run_summary.json'}

    def __init__(self, name):
        self.run_dir = qroot() / name
        assert 'runs' not in self.run_dir.relative_to(runtime_root()).parts, 'never a scientific root'
        assert not self.run_dir.exists(), f'qualification root exists (move it aside, never overwrite): {self.run_dir}'
        self.run_dir.mkdir()
        self.mode = 'QUALIFICATION_REPLAY_OF_SCIENTIFIC_E08_SEED42'
        self.manifest = {'labels': LABELS, 'runner_mode': self.mode, 'method': METHOD, 'replayed_seed': SEED,
                         'authority_commit': AUTHORITY, 'scientific': False, 'writes_checkpoints': False}
        (self.run_dir / 'run_manifest.json').write_text(json.dumps(self.manifest, indent=1) + '\n')
        (self.run_dir / 'checkpoint_index.json').write_text(json.dumps({'labels': LABELS, 'checkpoints': [{
            'path': 'runs/m7/E08/seed_42/checkpoints/recovery/latest.pt (scientific, read-only; never written here)',
            'sha256': RECOVERY_SHA256, 'global_step': RECOVERY_UPDATE, 'role': 'recovery'}]}, indent=1) + '\n')
        self._fh = (self.run_dir / 'metrics.jsonl').open('a', encoding='utf-8')

    def path(self, key):
        return self.run_dir / self.FILES[key]

    @property
    def ckpt_dir(self):
        raise AssertionError('qualification replay never writes checkpoints')

    def log(self, record_type, record):
        self._fh.write(json.dumps({'record_type': record_type, 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                                   **record}, sort_keys=True, allow_nan=False) + '\n')
        self._fh.flush()

    def record_checkpoint(self, entry):
        raise AssertionError('qualification replay never records checkpoints')

    def close(self, status, summary):
        self._fh.close()
        (self.run_dir / 'run_summary.json').write_text(json.dumps({**self.manifest, 'status': status, **summary},
                                                                  indent=1, default=str) + '\n')


def metrics(path, record_type=None):
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    return [r for r in rows if record_type is None or r['record_type'] == record_type]


def setup(name, *, policy=None):
    """Trainer (SCIENTIFIC seed-42 order and determinism, exactly as run_scientific) with a qualification context."""
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
    tr.ctx = ReplayContext(name)                        # the scientific GPATRunContext is never opened
    if policy is not None:
        tr.step.amp_policy = policy
    counts = {'D': 0, 'G': 0}
    for key, opt in (('D', tr.d_opt), ('G', tr.g_opt)):
        orig = opt.step
        opt.step = (lambda *a, _k=key, _f=orig, **kw: (counts.__setitem__(_k, counts[_k] + 1), _f(*a, **kw))[1])
    tr.opt_steps = counts
    torch.cuda.reset_peak_memory_stats()
    return tr


def recovery_copy():
    """Byte-identical copy of the scientific end-epoch-1 recovery into the qualification root (read-only source)."""
    dst = qroot() / 'inputs' / 'recovery_end_epoch1_u1105.pt'
    if not dst.exists():
        dst.parent.mkdir(exist_ok=True)
        src = sci_dir() / 'checkpoints' / 'recovery' / 'latest.pt'
        assert file_sha(src) == RECOVERY_SHA256, 'scientific recovery hash'
        tmp = dst.with_suffix('.tmp')
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
    assert file_sha(dst) == RECOVERY_SHA256, 'recovery copy hash'
    return dst


def load_from_recovery(tr):
    p = tr.load_recovery(recovery_copy())
    assert tr.position == {'stage': 'generator', 'epoch': 2, 'next_group': 1, 'global_update': RECOVERY_UPDATE,
                           'warmup_epoch': 1, 'warmup_next_batch': 1, 'warmup_step': 0}, tr.position
    return p


def scientific_records():
    return {r['global_update']: r for r in metrics(sci_dir() / 'metrics.jsonl', 'optimizer_group')}


def finish(tr, out):
    import torch
    out['access'] = tr.access.report()
    out['optimizer_steps'] = dict(tr.opt_steps)
    out['peak_reserved_GiB'] = round(torch.cuda.max_memory_reserved() / 2 ** 30, 3)
    out['run_dir'] = str(tr.ctx.run_dir)
    out['labels'] = LABELS
    out['runner_sha256'] = file_sha(ROOT / 'methods/gpat/runner.py')
    tr.ctx.close('qualification_completed', {'access': out['access']})
    return out


# ----------------------------------------------------------------------------- snapshot (Part C)
def snapshot_payload(tr, group, u):
    from methods.gpat import runner_checkpoint as ck
    return {'kind': SNAPSHOT_KIND, 'labels': LABELS, 'global_update_attempted': u,
            'modules': {k: getattr(tr.core, k).state_dict() for k in MODULES if getattr(tr.core, k) is not None},
            'optimizers': {'G_OPT': tr.g_opt.state_dict(), 'D_OPT': tr.d_opt.state_dict()},
            'scalers': {'G_SCALER': tr.g_scaler.state_dict(), 'D_SCALER': tr.d_scaler.state_dict()},
            'position': ck.primitives(tr.position), 'rng': ck.rng_state(),
            'group': [{k: v.detach().cpu().clone() for k, v in mb.items()} for mb in group],
            'group_indices': [mb['index'].tolist() for mb in group],
            'group_pair_ids': [[tr.records[i]['pair_id'] for i in mb['index'].tolist()] for mb in group],
            'state_digest': ck.primitives(full_state(tr))}


def group_digest(group):
    return [{k: state_digest(v) for k, v in sorted(mb.items())} for mb in group]


def reset_scaler(scaler, state):
    if scaler._scale is not None:
        scaler.update(new_scale=float(state['scale']))     # clears any per-optimizer record of a previous probe
    scaler.load_state_dict(state)


def apply_snapshot(tr, snap, *, d_scaler_state=None, g_scaler_state=None):
    """Restore the exact pre-update-1971 state into tr (all modules incl. buffers, optimizers, scalers, position, RNG);
    returns the group on the device. Verified against the digest taken from the live state at capture time."""
    from methods.gpat import runner as R
    from methods.gpat import runner_checkpoint as ck
    for k in MODULES:
        m = getattr(tr.core, k)
        assert (m is None) == (k not in snap['modules']), k
        if m is not None:
            m.load_state_dict(snap['modules'][k], strict=True)
    # deep copies: Optimizer.load_state_dict keeps CPU tensors (Adam `step`) by reference, and a later in-place
    # optimizer step would otherwise alter the in-memory base snapshot
    tr.g_opt.load_state_dict(copy.deepcopy(snap['optimizers']['G_OPT']))
    tr.d_opt.load_state_dict(copy.deepcopy(snap['optimizers']['D_OPT']))
    reset_scaler(tr.g_scaler, snap['scalers']['G_SCALER'])
    reset_scaler(tr.d_scaler, snap['scalers']['D_SCALER'])
    for p in tr.core.parameters():
        p.grad = None
    tr.position = dict(snap['position'])
    ck.restore_rng(snap['rng'])
    got = ck.primitives(full_state(tr))
    assert got == snap['state_digest'], {k: (got[k], snap['state_digest'][k]) for k in got
                                         if got[k] != snap['state_digest'][k]}
    if d_scaler_state is not None:
        reset_scaler(tr.d_scaler, d_scaler_state)
    if g_scaler_state is not None:
        reset_scaler(tr.g_scaler, g_scaler_state)
    return [R.to_device(dict(mb)) for mb in snap['group']]


def load_snapshot():
    from methods.gpat import runner_checkpoint as ck
    path = qroot() / 'snapshot' / 'pre_update_1971.pt'
    meta = json.loads((qroot() / 'snapshot' / 'pre_update_1971.json').read_text())
    assert file_sha(path) == meta['sha256'], 'snapshot hash'
    snap = ck.load(path)
    assert snap['kind'] == SNAPSHOT_KIND and snap['labels'] == LABELS
    return snap, meta


# ----------------------------------------------------------------------------- Part B + C: replay under M7C4
def scenario_replay(_):
    import torch
    from methods.gpat import runner as R
    from methods.gpat import runner_checkpoint as ck
    head = subprocess.run(['git', '-C', str(ROOT), 'show', f'{AUTHORITY}:methods/gpat/runner.py'],
                          capture_output=True).stdout
    runner_is_authority = sha(head) == file_sha(ROOT / 'methods/gpat/runner.py')
    assert runner_is_authority, 'Part B must run the unmodified M7C4 runner'
    tr = setup('replay_epoch2_m7c4_policy')
    load_from_recovery(tr)
    real = tr.step
    snap_info = {}

    def step(group, u):
        if u == FAILED_UPDATE:
            torch.cuda.synchronize()
            d = qroot() / 'snapshot'
            d.mkdir(exist_ok=True)
            payload = snapshot_payload(tr, group, u)
            info = ck.atomic_save(payload, d / 'pre_update_1971.pt')
            snap_info.update(info, group_digest=group_digest(payload['group']), state_digest=payload['state_digest'],
                             group_indices=payload['group_indices'], group_pair_ids=payload['group_pair_ids'],
                             labels=LABELS, kind=SNAPSHOT_KIND)
            (d / 'pre_update_1971.json').write_text(json.dumps(snap_info, indent=1, sort_keys=True) + '\n')
        return real(group, u)
    tr.step = step
    t0 = time.monotonic()
    failure = None
    try:
        tr.run_generator_groups(R.StopFlag(), limit=FAILED_UPDATE - RECOVERY_UPDATE)
    except R.AmpOverflowStop as exc:
        failure = {'type': type(exc).__name__, 'record': exc.record}
    seconds = time.monotonic() - t0
    ours = metrics(tr.ctx.path('metrics'), 'optimizer_group')
    sci = scientific_records()
    compared = [strip(r) == strip(sci[r['global_update']]) for r in ours]
    sci_fail = metrics(sci_dir() / 'metrics.jsonl', 'numerical_failure')[-1]
    our_fail = metrics(tr.ctx.path('metrics'), 'numerical_failure')
    keys = ('failure_type', 'global_update_attempted', 'offending_optimizers', 'offending_parameter_count',
            'offending_parameters', 'D_scale', 'G_scale', 'optimizer_steps_taken', 'epoch', 'attempted_group_or_batch',
            'position_unchanged', 'stage')
    out = {'runner_is_authority_m7c4': runner_is_authority, 'updates': [r['global_update'] for r in ours],
           'records_equal_scientific': all(compared), 'records_compared': len(compared),
           'first_mismatch': next((r['global_update'] for r, c in zip(ours, compared) if not c), None),
           'all_losses_finite': all(_finite_tree(r['train_losses']) for r in ours), 'failure': failure,
           'failure_record_logged': our_fail[-1] if our_fail else None,
           'failure_equal_scientific': bool(our_fail) and {k: our_fail[-1].get(k) for k in keys} ==
           {k: sci_fail.get(k) for k in keys}, 'scientific_failure': {k: sci_fail.get(k) for k in keys},
           'snapshot': snap_info, 'post_failure_state': ck.primitives(full_state(tr)), 'wall_seconds': seconds,
           'scales_at_1970': {'D': ours[-1]['D_scale'], 'G': ours[-1]['G_scale']} if ours else None}
    return finish(tr, out)


def _finite_tree(node):
    if isinstance(node, dict):
        return all(_finite_tree(v) for v in node.values())
    if isinstance(node, (list, tuple)):
        return all(_finite_tree(v) for v in node)
    if isinstance(node, float):
        return node == node and abs(node) != float('inf')
    return True


# ----------------------------------------------------------------------------- Part D: loss-scale probe
def boundary_inspector(tr, sink):
    import torch
    from methods.gpat import runner as R

    def norm(named):
        gs = [p.grad.detach().float().norm() for _, p in named if p.grad is not None]
        return float(torch.norm(torch.stack(gs))) if gs else 0.0

    def inspect(stage, step):
        if stage == 'G':                                   # both scalers unscaled; nothing stepped yet
            bad_d, bad_g = R.nonfinite(step.d_named), R.nonfinite(step.g_named)
            sink.update(D_finite=not bad_d, D_offending=bad_d, D_grad_norm_unscaled=None if bad_d else norm(step.d_named),
                        G_finite=not bad_g, G_offending=bad_g, G_grad_norm_unscaled=None if bad_g else norm(step.g_named),
                        D_scale=float(step.d_scaler.get_scale()), G_scale=float(step.g_scaler.get_scale()),
                        D_inf_per_param={n: int((~torch.isfinite(p.grad)).sum()) for n, p in step.d_named
                                         if p.grad is not None and not bool(torch.isfinite(p.grad).all())})
    return inspect


def scenario_probe(_):
    from methods.gpat import runner as R
    snap, meta = load_snapshot()
    tr = setup('probe_update1971', policy=R.AMP_FAIL_CLOSED)
    base = snap['state_digest']
    probes = []
    for s in PROBE_D_SCALES:
        group = apply_snapshot(tr, snap, d_scaler_state={**snap['scalers']['D_SCALER'], 'scale': s})
        before = dict(tr.opt_steps)
        sink = {}
        tr.step.inspect = boundary_inspector(tr, sink)
        tr.step.capture = []
        rec, err = None, None
        try:
            rec = tr.step(group, FAILED_UPDATE)
        except R.NumericalStop as exc:
            err = {'type': type(exc).__name__, 'record': exc.record}
        steps = {k: tr.opt_steps[k] - before[k] for k in before}
        probes.append({'D_scale': s, 'G_scale': sink.get('G_scale'), **sink, 'stop': err,
                       'losses_per_microbatch': [{'l_g': c['l_g'], 'l_d': c['l_d'], 'components': c['components']}
                                                 for c in tr.step.capture],
                       'group_record_losses': rec['train_losses'] if rec else None,
                       'D_grad_norm_record': rec['D_grad_norm'] if rec else None,
                       'optimizer_steps': steps, 'either_optimizer_stepped': any(steps.values()),
                       'post_step_params_finite': (rec is not None and not R.nonfinite(tr.step.d_named, None)
                                                   and not R.nonfinite(tr.step.g_named, None)) if rec else None})
    # the base snapshot is never altered: reload once more and re-verify, and re-hash the file
    apply_snapshot(tr, snap)
    finite = [p['D_scale'] for p in probes if p['D_finite'] and p['G_finite']]
    fails = [p['D_scale'] for p in probes if not p['D_finite']]
    if 65536.0 in fails and finite:
        cls = 'LOSS_SCALE_OVERFLOW_CONFIRMED'
    elif fails and not finite:
        cls = 'TRUE_NUMERICAL_INSTABILITY_NOT_SCALER_ONLY'
    else:
        cls = 'UNRESOLVED'
    out = {'probes': probes, 'finite_D_scales': finite, 'overflow_D_scales': fails,
           'lowest_finite_D_scale': min(finite) if finite else None,
           'highest_finite_D_scale': max(finite) if finite else None, 'classification': cls,
           'snapshot_sha256_after': file_sha(qroot() / 'snapshot' / 'pre_update_1971.pt'),
           'snapshot_sha256': meta['sha256'], 'base_state_digest_reverified': True, 'base_state_digest': base}
    return finish(tr, out)


# ----------------------------------------------------------------------------- Part F: restore equivalence
def scenario_equivalence(_):
    from methods.gpat import runner as R
    from methods.gpat import runner_checkpoint as ck
    snap, meta = load_snapshot()
    tr = setup('equivalence_update1971')
    out = {}

    def run(label, policy, **scales):
        group = apply_snapshot(tr, snap, **scales)
        tr.step.amp_policy = policy
        tr.step.capture, tr.step.inspect = [], None
        events = []
        tr.step.on_retry = events.append
        before = dict(tr.opt_steps)
        rec = tr.step(group, FAILED_UPDATE)
        return {'record': rec, 'events': events, 'capture': tr.step.capture,
                'optimizer_steps': {k: tr.opt_steps[k] - before[k] for k in before},
                'state': ck.primitives(full_state(tr))}

    # observe the failed attempt itself: which mutable state a never-stepped attempt changes
    group = apply_snapshot(tr, snap)
    tr.step.amp_policy, tr.step.capture = R.AMP_FAIL_CLOSED, []
    try:
        tr.step(group, FAILED_UPDATE)
        out['failed_attempt'] = {'overflowed': False}
    except R.AmpOverflowStop as exc:
        after = ck.primitives(full_state(tr))
        out['failed_attempt'] = {'overflowed': True, 'record': exc.record,
                                 'changed_components': sorted(k for k in after if after[k] != snap['state_digest'][k]),
                                 'e_art_bn_mutated': after['e_art_bn_buffers'] != snap['state_digest']['e_art_bn_buffers'],
                                 'rng_consumed': after['rng'] != snap['state_digest']['rng'],
                                 'params_unchanged': all(after[k] == snap['state_digest'][k] for k in after
                                                         if k.startswith('params_')),
                                 'optimizers_unchanged': after['G_OPT'] == snap['state_digest']['G_OPT'] and
                                 after['D_OPT'] == snap['state_digest']['D_OPT'],
                                 # the attempt sets lr = main_lr(1971) (a pure function of u, re-set by every attempt);
                                 # moments / step counts and every other hyper-parameter must be untouched
                                 'optimizer_state_tensors_unchanged': all(
                                     state_digest(o.state_dict()['state']) == state_digest(snap['optimizers'][k]['state'])
                                     for k, o in (('G_OPT', tr.g_opt), ('D_OPT', tr.d_opt))),
                                 'optimizer_hyperparameters_except_lr_unchanged': all(
                                     [{h: v for h, v in g.items() if h != 'lr'} for g in o.state_dict()['param_groups']] ==
                                     [{h: v for h, v in g.items() if h != 'lr'} for g in snap['optimizers'][k]['param_groups']]
                                     for k, o in (('G_OPT', tr.g_opt), ('D_OPT', tr.d_opt))),
                                 'optimizer_lr_set_to_main_lr_u': sorted({g['lr'] for o in (tr.g_opt, tr.d_opt)
                                                                          for g in o.param_groups}) ==
                                 [__import__('methods.gpat.runtime_contract', fromlist=['x']).main_lr(FAILED_UPDATE)],
                                 'snapshot_lr': sorted({g['lr'] for k in ('G_OPT', 'D_OPT')
                                                        for g in snap['optimizers'][k]['param_groups']})}
    a = run('A_atomic_retry', R.AMP_ATOMIC_RETRY)
    accepted = {'D_SCALER': a['record']['D_scale_before'], 'G_SCALER': a['record']['G_scale_before']}
    backed = {k: accepted[k] != snap['scalers'][k]['scale'] for k in accepted}
    scales = {('d_scaler_state' if k == 'D_SCALER' else 'g_scaler_state'):
              {**snap['scalers'][k], 'scale': accepted[k], '_growth_tracker': 0} for k in accepted if backed[k]}
    b = run('B_single_attempt_at_accepted_scale', R.AMP_FAIL_CLOSED, **scales)
    comp = {k: a['state'][k] == b['state'][k] for k in a['state']}
    rec_equal = strip(a['record'], ADDITIVE) == strip(b['record'], ADDITIVE)
    # C: negative control -- the overflowed attempt, then a retry WITHOUT restoring buffers / RNG
    group = apply_snapshot(tr, snap)
    tr.step.amp_policy, tr.step.capture = R.AMP_FAIL_CLOSED, []
    try:
        tr.step(group, FAILED_UPDATE)
    except R.AmpOverflowStop:
        pass
    for p in tr.core.parameters():
        p.grad = None
    for key, sc in (('D_SCALER', tr.d_scaler), ('G_SCALER', tr.g_scaler)):
        reset_scaler(sc, {**snap['scalers'][key], 'scale': accepted[key], '_growth_tracker': 0} if backed[key]
                     else snap['scalers'][key])
    before = dict(tr.opt_steps)
    try:
        rec_c, c_stop = tr.step(group, FAILED_UPDATE), None
    except R.NumericalStop as exc:
        rec_c, c_stop = None, {'type': type(exc).__name__, 'record': exc.record}
    c_state = ck.primitives(full_state(tr))
    out.update({
        'A': {'record': a['record'], 'events': a['events'], 'optimizer_steps': a['optimizer_steps'], 'state': a['state']},
        'B': {'record': b['record'], 'optimizer_steps': b['optimizer_steps'], 'state': b['state']},
        'accepted_scales': accepted, 'backed_off': backed,
        'component_equal': comp, 'all_state_equal': all(comp.values()), 'group_record_equal': rec_equal,
        'capture_equal': a['capture'] == b['capture'],
        'bn_equal': comp['e_art_bn_buffers'] and comp['all_core_buffers'], 'rng_equal': comp['rng'],
        'trainable_params_equal': all(v for k, v in comp.items() if k.startswith('params_')),
        'optimizer_states_equal': comp['G_OPT'] and comp['D_OPT'],
        'scaler_states_equal': comp['G_SCALER'] and comp['D_SCALER'],
        'post_group_state_digest': state_digest(a['state']),
        'negative_control_no_restore': {
            'record': rec_c, 'stop': c_stop, 'optimizer_steps': {k: tr.opt_steps[k] - before[k] for k in before},
            'differs_from_B': {k: c_state[k] != b['state'][k] for k in c_state},
            'bn_differs_from_B': c_state['e_art_bn_buffers'] != b['state']['e_art_bn_buffers'],
            'record_differs_from_B': rec_c is None or strip(rec_c, ADDITIVE) != strip(b['record'], ADDITIVE)}})
    return finish(tr, out)


# ----------------------------------------------------------------------------- Part G: exact group + 20 updates
def scenario_continue_(_):
    from methods.gpat import runner as R
    from methods.gpat import runner_checkpoint as ck
    snap, meta = load_snapshot()
    tr = setup('continue_update1971_plus20')
    assert tr.step.amp_policy == R.AMP_ATOMIC_RETRY == R.PRODUCTION_AMP_POLICY, 'production Trainer policy'
    apply_snapshot(tr, snap)
    assert tr.position == {'stage': 'generator', 'epoch': FAILED_EPOCH, 'next_group': FAILED_GROUP,
                           'global_update': FAILED_UPDATE - 1, 'warmup_epoch': 1, 'warmup_next_batch': 1,
                           'warmup_step': 0}, tr.position
    real = tr.step
    seen = {}
    snap_group = meta['group_digest']

    class Wrapped:                                     # keeps on_retry/active visible to the Trainer
        def __getattr__(self, k):
            return getattr(real, k)

        def __setattr__(self, k, v):
            setattr(real, k, v)

        def __call__(self, group, u):
            if u == FAILED_UPDATE:
                cpu = [{k: v.detach().cpu() for k, v in mb.items()} for mb in group]
                seen['group_digest_equal_snapshot'] = group_digest(cpu) == snap_group
                if not seen['group_digest_equal_snapshot']:
                    raise R.TrainingStop('GPAT runner STOP: the loader did not reproduce the snapshot group 866')
                seen['state_at_entry_equal_snapshot'] = ck.primitives(full_state(tr)) == snap['state_digest']
            seen.setdefault('steps_before', []).append((u, dict(tr.opt_steps)))
            return real(group, u)
    tr.step = Wrapped()
    n = POST_FAILURE_UPDATES + 1
    tr.run_generator_groups(R.StopFlag(), limit=n)
    recs = metrics(tr.ctx.path('metrics'), 'optimizer_group')
    retries = metrics(tr.ctx.path('metrics'), 'amp_retry')
    plan = tr.generator_plan(FAILED_EPOCH)
    consumed = [i for r in tr.access.opened for i in [r[1]]]
    planned_ids = [tr.records[i][role] for g in plan[FAILED_GROUP - 1:FAILED_GROUP - 1 + n] for mb in g for i in mb
                   for role in ('source_spoof_id', 'target_live_id')]
    out = {'updates': [r['global_update'] for r in recs], 'records': recs, 'amp_retry_events': retries,
           'contiguous': [r['global_update'] for r in recs] == list(range(FAILED_UPDATE, FAILED_UPDATE + n)),
           'all_losses_finite': all(_finite_tree(r['train_losses']) for r in recs),
           'all_grad_norms_finite': all(_finite_tree([r['D_grad_norm'], r['G_grad_norm']]) for r in recs),
           'optimizer_steps_total': dict(tr.opt_steps), 'd_g_one_to_one': tr.opt_steps == {'D': n, 'G': n},
           'access_events_equal_plan': consumed == planned_ids, 'access_events': len(consumed),
           'expected_access_events': len(planned_ids), **seen,
           'update1971_record': next(r for r in recs if r['global_update'] == FAILED_UPDATE),
           'final_position': dict(tr.position), 'final_state': ck.primitives(full_state(tr))}
    out.pop('steps_before', None)
    return finish(tr, out)


# ----------------------------------------------------------------------------- finite path unchanged under new code
def scenario_finite_path(_):
    from methods.gpat import runner as R
    tr = setup('finite_path_new_code_1106_1125')
    assert tr.step.amp_policy == R.AMP_ATOMIC_RETRY
    load_from_recovery(tr)
    tr.run_generator_groups(R.StopFlag(), limit=FINITE_PATH_UPDATES)
    ours = metrics(tr.ctx.path('metrics'), 'optimizer_group')
    sci = scientific_records()
    eq = [strip(r, ADDITIVE) == strip(sci[r['global_update']]) for r in ours]
    out = {'updates': [r['global_update'] for r in ours], 'records_equal_scientific': all(eq) and len(eq) ==
           FINITE_PATH_UPDATES, 'amp_attempts': [r['amp_attempts'] for r in ours],
           'retry_events': metrics(tr.ctx.path('metrics'), 'amp_retry')}
    return finish(tr, out)


# ----------------------------------------------------------------------------- evidence assembly (laptop, no torch)
def assemble(workdir):
    w = Path(workdir)
    sc = {n: json.loads((w / f'{n}.json').read_text()) for n in SCENARIOS}
    orch = json.loads((w / 'orchestration.json').read_text())
    g = {}
    pb, pa = sc['preserve_before'], sc['preserve_after']
    g['A_failed_run_preserved'] = (pb['tree_sha256'] == pa['tree_sha256'] and
                                   pb['completion_status'] == 'FAIL_CLOSED_AMP_OVERFLOW' and pb['global_update'] == 1970
                                   and pb['access']['val_images'] == 0 and pb['access']['test_images'] == 0 and
                                   pb['access']['val_metadata'] == 0 and pb['access']['test_metadata'] == 0 and
                                   not pb['val_split_accessed'] and not pb['test_split_accessed'] and
                                   pb['recovery_sha256'] == RECOVERY_SHA256)
    r = sc['replay']
    f = (r['failure'] or {}).get('record', {})
    g['B_overflow_reproduced_exactly'] = (
        r['runner_is_authority_m7c4'] and r['updates'] == list(range(RECOVERY_UPDATE + 1, FAILED_UPDATE)) and
        r['records_equal_scientific'] and r['all_losses_finite'] and r['failure']['type'] == 'AmpOverflowStop' and
        f.get('global_update_attempted') == FAILED_UPDATE and f.get('offending_optimizers') == ['D_OPT'] and
        f.get('offending_parameters', {}).get('D_OPT') == FAILED_D_PARAMS and f.get('offending_parameter_count') ==
        {'D_OPT': 2, 'G_OPT': 0} and r['failure_equal_scientific'] and r['optimizer_steps'] == {'D': 865, 'G': 865})
    g['C_snapshot_captured'] = bool(r['snapshot'].get('sha256')) and r['snapshot']['group_indices'] is not None
    p = sc['probe']
    first = p['probes'][0]
    g['D_probe_classification'] = p['classification'] == 'LOSS_SCALE_OVERFLOW_CONFIRMED'
    g['D_probe_snapshot_reproduces_failure'] = (not first['D_finite'] and first['D_offending'] == FAILED_D_PARAMS and
                                                first['G_finite'] and not first['either_optimizer_stepped'])
    g['D_probe_base_unaltered'] = p['snapshot_sha256_after'] == p['snapshot_sha256'] == r['snapshot']['sha256']
    e = sc['equivalence']
    g['F_restore_equivalence'] = (e['all_state_equal'] and e['group_record_equal'] and e['capture_equal'] and
                                  e['bn_equal'] and e['rng_equal'] and e['trainable_params_equal'] and
                                  e['optimizer_states_equal'] and e['scaler_states_equal'] and
                                  e['A']['optimizer_steps'] == {'D': 1, 'G': 1} and
                                  e['B']['optimizer_steps'] == {'D': 1, 'G': 1} and len(e['A']['events']) >= 1)
    g['F_failed_attempt_leaves_params_and_optimizer_state'] = (e['failed_attempt']['overflowed'] and
                                                   e['failed_attempt']['params_unchanged'] and
                                                   e['failed_attempt']['optimizer_state_tensors_unchanged'] and
                                                   e['failed_attempt']['optimizer_hyperparameters_except_lr_unchanged']
                                                   and e['failed_attempt']['optimizer_lr_set_to_main_lr_u'])
    c = sc['continue_']
    acc = c['update1971_record']
    g['G_exact_group_retry'] = (c['group_digest_equal_snapshot'] and c['state_at_entry_equal_snapshot'] and
                                acc['D_scale_before'] == e['accepted_scales']['D_SCALER'] and
                                strip(acc, ADDITIVE + ENVELOPE) == strip(e['A']['record'], ADDITIVE) and
                                acc['amp_attempts'] == len(e['A']['events']) + 1 and
                                [ev['global_update_attempted'] for ev in c['amp_retry_events']][:len(e['A']['events'])]
                                == [FAILED_UPDATE] * len(e['A']['events']))
    g['G_post_failure_updates'] = (c['contiguous'] and len(c['updates']) == POST_FAILURE_UPDATES + 1 and
                                   c['all_losses_finite'] and c['all_grad_norms_finite'] and c['d_g_one_to_one'] and
                                   c['access_events_equal_plan'] and
                                   c['access']['opened_events'] == 16 * (POST_FAILURE_UPDATES + 1))
    fp = sc['finite_path']
    g['finite_path_unchanged_under_new_code'] = (fp['records_equal_scientific'] and
                                                 fp['updates'] == list(range(1106, 1106 + FINITE_PATH_UPDATES)) and
                                                 set(fp['amp_attempts']) == {1} and not fp['retry_events'])
    firewall = {k: sum(s['access'][k] for n, s in sc.items() if not n.startswith('preserve') and 'access' in s)
                for k in ('train_images', 'non_train_images', 'val_images', 'test_images', 'val_metadata',
                          'test_metadata')}
    g['firewall'] = all(firewall[k] == 0 for k in firewall if k != 'train_images')
    g['all_scenarios_exit_0'] = all(v['exit_code'] == 0 for v in orch.values()) and set(orch) == set(SCENARIOS)
    return {'milestone': 'M7D1-N1', 'kind': 'gpat_amp_overflow_diagnosis_and_atomic_retry_qualification',
            'authority_commit': AUTHORITY, 'labels': LABELS, 'replayed': {'method': METHOD, 'experiment': EXPERIMENT,
                                                                          'seed': SEED},
            'failed_attempt_evidence': {'recovery_sha256': RECOVERY_SHA256, 'recovery_global_update': RECOVERY_UPDATE,
                                        'failed_update': FAILED_UPDATE, 'epoch': FAILED_EPOCH, 'group': FAILED_GROUP},
            'gates': g, 'classification': p['classification'], 'firewall_qualification': firewall,
            'orchestration': orch, 'scenarios': sc, 'status': 'PASS' if all(g.values()) else 'FAIL'}


# ----------------------------------------------------------------------------- orchestration
def run_scenario(name, args):
    out = globals()['scenario_' + name](args)
    Path(args.out).write_text(json.dumps(out, indent=1, sort_keys=True, default=str) + '\n')
    print(f'{name}: done')


def orchestrate(args):
    wd = Path(args.workdir)
    wd.mkdir(parents=True, exist_ok=True)
    names = args.only.split(',') if args.only else SCENARIOS
    path = wd / 'orchestration.json'
    results = json.loads(path.read_text()) if path.exists() else {}
    for name in names:
        out = wd / f'{name}.json'
        t0 = time.monotonic()
        r = subprocess.run([sys.executable, '-B', __file__, '--scenario', name, '--out', str(out)],
                           env=dict(os.environ), capture_output=True, text=True)
        (wd / f'{name}.log').write_text(r.stdout + r.stderr)
        results[name] = {'exit_code': r.returncode, 'seconds': round(time.monotonic() - t0, 1),
                         'runner_sha256': file_sha(ROOT / 'methods/gpat/runner.py')}
        path.write_text(json.dumps(results, indent=1) + '\n')
        print(f'{name}: exit {r.returncode} ({results[name]["seconds"]} s)', flush=True)
        if r.returncode != 0:
            break


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--scenario', choices=SCENARIOS)
    ap.add_argument('--out')
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--only', help='comma-separated scenario subset for --all')
    ap.add_argument('--workdir')
    ap.add_argument('--assemble', action='store_true')
    args = ap.parse_args()
    if args.assemble:
        ev = assemble(args.workdir)
        Path(args.out).write_text(json.dumps(ev, indent=1, sort_keys=True, default=str) + '\n')
        print(json.dumps({'status': ev['status'], 'gates': ev['gates']}, indent=1))
        return
    if args.all:
        orchestrate(args)
    else:
        run_scenario(args.scenario, args)


if __name__ == '__main__':
    main()
