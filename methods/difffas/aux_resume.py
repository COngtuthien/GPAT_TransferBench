"""M6D6f E07c auxiliary-encoder exact epoch-boundary resume (Amendment A8; CONTROLLED_ADAPTATION, DEV-021).

Pinned models/pretrain_classifier.py has no resume mechanism. A8 (owner-frozen,
DETERMINISTIC_IMPLEMENTATION_CLARIFICATION) defines ONE_LOGICAL_RUN recovery from infrastructure
interruption only:

  * the only resume point is the latest COMMITTED epoch boundary E (epoch 0 included); a
    partial epoch E+1 is logically rolled back and replayed from its beginning;
  * the engineering sidecar checkpoints/resume/runner_state_epoch_<EEE>.pth is self-contained
    (model state_dict, SGD state_dict with momentum, torch CPU/CUDA, Python and NumPy RNG,
    identities, metrics boundary, step counters); it is NOT the scientific checkpoint, never a
    selection candidate and never consumable by main DiffFAS;
  * the scientific checkpoint stays checkpoints/encoder_final.pkl (whole nn.Module, M6D6c/M6D6e),
    overwritten after every epoch; nothing here changes its format or cadence;
  * commit order: <name>.partial (fsync) -> SHA256 -> os.replace -> resume_state_index.json LAST ->
    explicit pruning of the previous sidecar bytes (history retained);
  * restore: exact indexed path, SHA256 well-formed before open, bytes SHA256 before torch.load,
    torch.load(weights_only=True), identities, model, optimizer, Python/NumPy RNG, torch CPU RNG,
    torch CUDA RNG, A7 state, and only then the next iterator;
  * metrics.jsonl is never truncated: post-boundary records are classified
    SUPERSEDED_BY_RESUME_ROLLBACK by an appended reconciliation record.

Torch is never imported at module level (the CLI uses the static gates before importing Torch).
"""
import hashlib
import io
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time
import uuid

from methods.common.config import ROOT, sha256_file
from methods.common.learned import PreparationError
from methods.common.runlog import atomic_write_json
from methods.difffas import aux_runner_io as aio

A8_OVERLAY = 'configs/amendments/e07c_a8_aux_resume_policy.yaml'
A8_OVERLAY_SHA256 = 'a098ce509156151ee6ce7ead1e9afa5711264b676a597e2514b1058ed46f471a'
RESUME_KIND = 'E07C_AUX_ENGINEERING_RESUME_STATE'
SCHEMA_VERSION = 1
RESUME_DIR = ('checkpoints', 'resume')
INDEX_NAME = 'resume_state_index.json'
SIDECAR = re.compile(r'runner_state_epoch_(\d{3})\.pth')
SUPERSEDED = 'SUPERSEDED_BY_RESUME_ROLLBACK'
HEX64 = re.compile(r'[0-9a-f]{64}')
CODE_FILES = ('methods/difffas/aux_runner_io.py', 'methods/difffas/aux_runner.py', 'methods/difffas/aux_resume.py',
              'tools/run_e07c_aux.py', aio.CONTRACT_PATH, A8_OVERLAY)
ENGINEERING_LABELS = {'scientific_checkpoint': False, 'official_difffas_checkpoint': False, 'selection_candidate': False,
                      'consumable_by_main_difffas': False, 'reportable_as_trained_model': False}
DEVICE = 0                      # CUDA_VISIBLE_DEVICES=0: the one execution device


class ResumeStop(PreparationError):
    """An A8 resume gate failed (identity drift, corrupted state, torn log, non-bitwise restore): STOP_AND_REPORT."""


def require(value, message):
    if not value:
        raise ResumeStop('E07c aux resume STOP: ' + message)


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def fsync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


# ================================================================= A8 authority (static)
def load_policy(path=None):
    """A8 overlay (JSON subset of YAML): pinned SHA256, bound document and authorities re-verified."""
    raw = Path(path if path is not None else ROOT / A8_OVERLAY).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == A8_OVERLAY_SHA256, 'A8 overlay SHA256 differs from the pinned value')
    policy = json.loads(raw)
    doc = policy['amendment_document']
    require(sha256_file(ROOT / doc['path']) == doc['sha256'], 'A8 document SHA256')
    for rel, digest in policy['bound_authority_sha256'].items():
        require(sha256_file(ROOT / rel) == digest, 'A8-bound authority SHA256 ' + rel)
    require((policy['classification'], policy['method_id'], policy['fidelity']['fidelity_class'],
             policy['fidelity']['deviation'], policy['fidelity']['new_deviation'],
             policy['fidelity']['new_fidelity_class']) ==
            ('DETERMINISTIC_IMPLEMENTATION_CLARIFICATION', 'E07c', 'CONTROLLED_ADAPTATION', 'DEV-021', False, False),
            'A8 classification / fidelity')
    require(policy['logical_run']['identity'] == 'ONE_LOGICAL_RUN' and
            policy['boundary']['rule'] == 'EXACT_EPOCH_BOUNDARY_ONLY' and
            policy['sidecar']['kind'] == RESUME_KIND and policy['sidecar']['unsafe_load'] == 'FORBIDDEN' and
            policy['cli']['resume_latest'] == 'NOT_SUPPORTED', 'A8 frozen semantics')
    return {'sha256': A8_OVERLAY_SHA256, 'document_sha256': doc['sha256'], 'policy': policy}


def a8_identities():
    p = load_policy()
    return {'a8_overlay_sha256': p['sha256'], 'a8_document_sha256': p['document_sha256']}


def code_hashes():
    return {rel: sha256_file(ROOT / rel) for rel in CODE_FILES}


def git_identity():
    def git(*args):
        return subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True, check=True).stdout
    porcelain = git('status', '--porcelain')
    return {'git_commit': git('rev-parse', 'HEAD').decode().strip(),
            'git_branch': git('branch', '--show-current').decode().strip(),
            'git_dirty': bool(porcelain.strip()), 'git_status_porcelain_sha256': aio.sha(porcelain)}


def process_identity(ctx, storage, precision_state):
    """Everything a resume must find unchanged (A8 section 12). Primitives only."""
    ident = dict(ctx.identities)
    ident.update(git_identity(), mode=ctx.mode, seed=ctx.aux_seed, run_id=ctx.run_id, run_label=aio.RUN_LABEL,
                 python_executable=sys.executable, environment=dict(ctx._environment),
                 exec_config_sha256=storage['exec_config_sha256'], faces_256_root=storage['faces_256_root'],
                 runtime_root=storage['runtime_root'], code_sha256=code_hashes(),
                 precision_state=dict(precision_state),
                 launch_environment={k: os.environ.get(k) for k in sorted(aio.LAUNCH_ENVIRONMENT)})
    require(ident['git_commit'] == ctx.git_commit, 'git commit changed during the process')
    if ctx.mode == aio.SCIENTIFIC:
        require(not ident['git_dirty'], 'SCIENTIFIC resume state requires a clean worktree')
    # JSON primitives only (e.g. torch.__version__ is a TorchVersion str subclass that weights_only=True refuses)
    return json.loads(json.dumps(ident, sort_keys=True))


# ================================================================= index / paths (static; the CLI uses these)
def resume_dir(run_dir):
    return Path(run_dir).joinpath(*RESUME_DIR)


def index_path(run_dir):
    return Path(run_dir) / INDEX_NAME


def sidecar_name(epoch):
    require(type(epoch) is int and 0 <= epoch <= aio.EPOCHS, 'sidecar epoch')
    return f'runner_state_epoch_{epoch:03d}.pth'


def sidecar_rel(epoch):
    return '/'.join(RESUME_DIR + (sidecar_name(epoch),))


def read_index(run_dir):
    p = index_path(run_dir)
    require(p.is_file() and not p.is_symlink(), 'resume_state_index.json missing')
    index = json.loads(p.read_text())
    require(index.get('kind') == RESUME_KIND and index.get('schema_version') == SCHEMA_VERSION, 'index kind/schema')
    current = [e for e in index['entries'] if e['status'] == 'COMMITTED_CURRENT']
    require(len(current) == 1 and current[0] == index['committed'], 'the index authorizes exactly ONE committed state')
    return index


def resolve_committed(run_dir, supplied):
    """The explicit --resume-state path must be exactly the index's committed sidecar inside this run root.

    Static (no Torch, no deserialization). The SHA256 recorded in the index must be well formed
    before the file is opened; the bytes are hashed later, immediately before torch.load.
    """
    run_dir = Path(run_dir)
    require(run_dir.is_dir() and not run_dir.is_symlink(), f'run root missing or not a directory: {run_dir}')
    run_dir = run_dir.resolve()
    raw = Path(supplied)
    require(raw.is_absolute(), '--resume-state must be an absolute path (no discovery, no relative lookup)')
    require(not raw.is_symlink() and not raw.parent.is_symlink(), 'resume sidecar path must not be a symlink')
    path = raw.resolve()
    require(path.parent == resume_dir(run_dir) and path == raw, 'resume sidecar must be inside <run_root>/checkpoints/resume')
    require(SIDECAR.fullmatch(path.name) is not None, 'resume sidecar name')
    index = read_index(run_dir)
    entry = index['committed']
    require(str(path.relative_to(run_dir)) == entry['path'],
            f"--resume-state is not the committed sidecar authorized by {INDEX_NAME} ({entry['path']})")
    require(isinstance(entry.get('sha256'), str) and HEX64.fullmatch(entry['sha256']) is not None,
            'indexed SHA256 is malformed (refused before any file open)')
    require(type(entry.get('size_bytes')) is int and entry['size_bytes'] > 0, 'indexed size')
    require(entry['path'] == sidecar_rel(entry['completed_epoch']) and
            entry['global_step'] == entry['completed_epoch'] * aio.STEPS_PER_EPOCH, 'indexed boundary')
    require(path.is_file(), 'committed sidecar bytes are missing')
    return path, entry, index


def read_verified(path, entry):
    """Bytes whose SHA256 and size equal the index; the same in-memory bytes are then deserialized."""
    require(HEX64.fullmatch(entry['sha256']) is not None, 'malformed SHA256')
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    with os.fdopen(fd, 'rb') as fh:
        raw = fh.read()
    digest = hashlib.sha256(raw).hexdigest()
    require(len(raw) == entry['size_bytes'] and digest == entry['sha256'],
            'resume sidecar SHA256/size mismatch; refusing to deserialize')
    return raw


def load_sidecar(torch, raw):
    """Safe deserialization only: tensors and primitives (weights_only=True); the unsafe pickle mode is never used."""
    return torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)


# ================================================================= state capture / restore
def rng_state(torch):
    import numpy as np
    version, internal, gauss = random.getstate()
    name, keys, pos, has_gauss, cached = np.random.get_state()
    return {'torch_cpu': torch.get_rng_state(),
            'torch_cuda_device': torch.cuda.get_rng_state(DEVICE), 'torch_cuda_device_index': DEVICE,
            'torch_cuda_all': list(torch.cuda.get_rng_state_all()),
            'python': {'version': int(version), 'internal': torch.tensor(internal, dtype=torch.int64),
                       'gauss_next': gauss},
            'numpy': {'bit_generator': str(name), 'keys': torch.from_numpy(keys.astype(np.int64)), 'pos': int(pos),
                      'has_gauss': int(has_gauss), 'cached_gaussian': float(cached)}}


def rng_digest(torch):
    """SHA256 of every RNG stream in its exact serialized representation."""
    import numpy as np
    name, keys, pos, has_gauss, cached = np.random.get_state()
    return {'torch_cpu_sha256': aio.sha(torch.get_rng_state().numpy().tobytes()),
            'torch_cuda_device_sha256': aio.sha(torch.cuda.get_rng_state(DEVICE).numpy().tobytes()),
            'torch_cuda_all_sha256': aio.sha(b''.join(s.numpy().tobytes() for s in torch.cuda.get_rng_state_all())),
            'python_sha256': aio.sha(repr(random.getstate()).encode()),
            'numpy_sha256': aio.sha(repr((name, keys.tolist(), int(pos), int(has_gauss), float(cached))).encode())}


def cpu_tree(torch, value):
    if isinstance(value, torch.Tensor):
        return value.detach().to('cpu', copy=True)
    if isinstance(value, dict):
        return {k: cpu_tree(torch, v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(torch, v) for v in value)
    return value


def tensor_digest(state):
    """Order-preserving SHA256 of a flat {name: tensor} mapping (bitwise)."""
    h = hashlib.sha256()
    for k, t in state.items():
        h.update(k.encode() + b'\0' + str(t.dtype).encode() + repr(tuple(t.shape)).encode() + b'\0')
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def optimizer_digest(opt_state):
    moments = {str(k): v['momentum_buffer'] for k, v in sorted(opt_state['state'].items())
               if v.get('momentum_buffer') is not None}
    groups = json.dumps(opt_state['param_groups'], sort_keys=True)
    return {'momentum_buffers': len(moments), 'momentum_sha256': tensor_digest(moments),
            'param_groups_sha256': aio.sha(groups.encode())}


def metrics_boundary(ctx):
    raw = ctx.path('metrics').read_bytes()
    require(not raw or raw.endswith(b'\n'), 'metrics.jsonl must end with a complete record at a commit boundary')
    records = [json.loads(x) for x in raw.splitlines()]
    steps = [r['global_step'] for r in records if r.get('record_type') == 'epoch']
    return {'bytes': len(raw), 'prefix_sha256': aio.sha(raw), 'lines': raw.count(b'\n'),
            'step_records_physical': len(steps), 'last_physical_global_step': steps[-1] if steps else 0}


def build_state(torch, trainer, ctx, *, identity, counters, checkpoint, session):
    model_state = {k: v.detach().to('cpu', copy=True) for k, v in trainer.model.state_dict().items()}
    opt_state = cpu_tree(torch, trainer.optimizer.state_dict())
    return {'kind': RESUME_KIND, 'schema_version': SCHEMA_VERSION, 'amendment': 'A8', 'method_id': aio.METHOD_ID,
            'role': 'AUXILIARY_CONDITIONING_ENCODER', **ENGINEERING_LABELS,
            'runner_mode': ctx.mode, 'seed': ctx.aux_seed, 'seed_field': ctx.seed_field,
            'logical_run': {'identity': 'ONE_LOGICAL_RUN', 'run_id': ctx.run_id, 'run_uuid': ctx.run_uuid,
                            'start_utc': ctx.start_utc, 'run_label': aio.RUN_LABEL, 'run_dir': str(ctx.run_dir)},
            'completed_epoch': trainer.completed_epoch, 'global_step': trainer.global_step,
            'identities': identity, 'model': model_state, 'model_sha256': tensor_digest(model_state),
            'optimizer': opt_state, 'optimizer_digest': optimizer_digest(opt_state),
            'rng': rng_state(torch), 'rng_digest': rng_digest(torch),
            'scientific_checkpoint_file': checkpoint, 'metrics_boundary': metrics_boundary(ctx),
            'counters': dict(counters), 'process_session': dict(session), 'written_utc': utc()}


def validate_state(state, *, identity, ctx, entry):
    """Every A8 identity gate; returns the list of checked fields."""
    require(isinstance(state, dict) and state.get('kind') == RESUME_KIND and state.get('schema_version') == SCHEMA_VERSION,
            'sidecar kind/schema')
    require(all(state.get(k) is v for k, v in ENGINEERING_LABELS.items()), 'sidecar engineering labels')
    require((state['method_id'], state['runner_mode'], state['seed'], state['seed_field']) ==
            (aio.METHOD_ID, ctx.mode, ctx.aux_seed, ctx.seed_field), 'method/mode/seed')
    lr = state['logical_run']
    require((lr['identity'], lr['run_id'], lr['run_uuid'], lr['start_utc'], lr['run_dir']) ==
            ('ONE_LOGICAL_RUN', ctx.run_id, ctx.run_uuid, ctx.start_utc, str(ctx.run_dir)),
            'logical run identity (run_id / run_uuid / start_utc / root)')
    require((state['completed_epoch'], state['global_step']) == (entry['completed_epoch'], entry['global_step']) and
            state['global_step'] == state['completed_epoch'] * aio.STEPS_PER_EPOCH, 'completed epoch / global step')
    drift = sorted(k for k in set(identity) | set(state['identities']) if identity.get(k) != state['identities'].get(k))
    require(not drift, 'identity drift: ' + ', '.join(drift))
    c = state['counters']
    require(c['logical_authoritative_optimizer_steps'] == state['global_step'] and
            c['physical_optimizer_steps_executed'] == c['logical_authoritative_optimizer_steps'] +
            c['superseded_optimizer_steps'], 'sidecar step counters')
    return sorted(identity)


def restore(torch, trainer, state):
    """A8 section 7 steps 6-12 on the already-constructed production components. No RNG is consumed after 9."""
    import numpy as np
    from methods.difffas import execution_policy as ep
    model, optimizer = trainer.model, trainer.optimizer
    out = {}
    # 6. model state (strict), bitwise-verified
    missing = model.load_state_dict(state['model'], strict=True)
    require(not missing.missing_keys and not missing.unexpected_keys, 'strict model state')
    live = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    require(list(live) == list(state['model']) and all(torch.equal(live[k], state['model'][k]) for k in live) and
            tensor_digest(live) == state['model_sha256'], 'restored model state is bitwise the sidecar state')
    out['model_sha256'] = state['model_sha256']
    # 7. optimizer state
    optimizer.load_state_dict(state['optimizer'])
    # 8. optimizer tensors / devices / hyperparameters
    g = optimizer.param_groups
    require(len(g) == 1 and all(a is b for a, b in zip(g[0]['params'], model.parameters())) and
            (g[0]['lr'], g[0]['momentum'], g[0]['weight_decay'], g[0]['dampening'], g[0]['nesterov']) ==
            (aio.LR, aio.MOMENTUM, aio.WEIGHT_DECAY, 0, False), 'restored SGD group')
    params = list(model.parameters())
    saved = state['optimizer']['state']
    require(len(optimizer.state) == len(saved), 'momentum-buffer inventory')
    for i, p in enumerate(params):
        if i in saved:
            buf = optimizer.state[p]['momentum_buffer']
            require(buf.device == p.device and buf.dtype == p.dtype == torch.float32 and buf.shape == p.shape and
                    torch.equal(buf.cpu(), saved[i]['momentum_buffer']), f'momentum buffer {i} restored bitwise on device')
        else:
            require(p not in optimizer.state, f'parameter {i} has no momentum (never stepped)')
    live_opt = optimizer_digest(cpu_tree(torch, optimizer.state_dict()))
    require(live_opt == state['optimizer_digest'], 'restored optimizer digest')
    out['optimizer_digest'] = live_opt
    # 9. Python / NumPy RNG
    r = state['rng']
    random.setstate((r['python']['version'], tuple(int(x) for x in r['python']['internal'].tolist()),
                     r['python']['gauss_next']))
    n = r['numpy']
    np.random.set_state((n['bit_generator'], n['keys'].numpy().astype(np.uint32), n['pos'], n['has_gauss'],
                         n['cached_gaussian']))
    # 10. torch CPU RNG
    torch.set_rng_state(r['torch_cpu'])
    # 11. torch CUDA RNG (all devices, then the execution device explicitly)
    torch.cuda.set_rng_state_all(r['torch_cuda_all'])
    torch.cuda.set_rng_state(r['torch_cuda_device'], r['torch_cuda_device_index'])
    require(rng_digest(torch) == state['rng_digest'], 'restored RNG streams are bitwise the committed streams')
    out['rng_digest'] = state['rng_digest']
    # 12. A7 execution state
    require(ep.precision_state() == ep.EXPECTED_STATE and state['identities']['precision_state'] == ep.precision_state(),
            'A7 precision state')
    require(not torch.is_autocast_enabled('cuda'), 'autocast disabled')
    trainer.completed_epoch, trainer.global_step = state['completed_epoch'], state['global_step']
    trainer.optimizer_applications = state['global_step']      # logical cumulative (the step hook continues from here)
    out['order'] = ['model', 'optimizer', 'optimizer tensors/devices', 'python+numpy RNG', 'torch CPU RNG',
                    'torch CUDA RNG', 'A7 state']
    return out


# ================================================================= metrics reconciliation (append-only)
def read_records(path):
    raw = Path(path).read_bytes()
    require(raw.endswith(b'\n') or not raw, 'torn final metrics record: STOP_AND_REPORT (never repaired automatically)')
    return raw, [json.loads(x) for x in raw.splitlines()]


def logical_trajectory(records):
    """Authoritative step records: file order, each e07c_aux_resume_reconciliation truncates to its boundary."""
    logical = []
    for r in records:
        if r.get('record_type') == 'epoch':
            logical.append(r)
        elif r.get('record_type') == 'event' and r.get('event') == 'e07c_aux_resume_reconciliation':
            g = r['payload']['committed_boundary']['global_step']
            logical = [x for x in logical if x['global_step'] <= g]
    return logical


def reconcile_metrics(ctx, state):
    """Read-only analysis of metrics.jsonl against the committed boundary (nothing is truncated)."""
    b = state['metrics_boundary']
    raw, records = read_records(ctx.path('metrics'))
    require(len(raw) >= b['bytes'] and aio.sha(raw[:b['bytes']]) == b['prefix_sha256'],
            'metrics.jsonl prefix differs from the committed boundary (rewritten log): STOP_AND_REPORT')
    before = [json.loads(x) for x in raw[:b['bytes']].splitlines()]
    after = [json.loads(x) for x in raw[b['bytes']:].splitlines()]
    g, e = state['global_step'], state['completed_epoch']
    logical_before = logical_trajectory(before)
    require([r['global_step'] for r in logical_before] == list(range(1, g + 1)),
            'authoritative trajectory up to the boundary is exactly steps 1..G')
    steps = [r for r in after if r.get('record_type') == 'epoch']
    require(all(r['global_step'] > g and r['epoch'] == e + 1 for r in steps),
            'post-boundary step records belong to epoch E+1 only')
    epochs_done = [r['payload']['epoch'] for r in after if r.get('event') == 'e07c_aux_epoch_complete']
    ckpts = [r['payload']['epoch'] for r in after if r.get('event') == 'e07c_aux_checkpoint_event']
    require(set(epochs_done) <= {e + 1} and set(ckpts) <= {e + 1}, 'post-boundary epoch events belong to epoch E+1 only')
    prior = [r for r in after if r.get('event') == 'e07c_aux_resume_reconciliation']
    return {'policy': 'APPEND_OR_EXPLICITLY_RECONCILE', 'overwrite': False, 'truncated': False,
            'committed_boundary': {'completed_epoch': e, 'global_step': g, 'metrics_bytes': b['bytes'],
                                   'metrics_prefix_sha256': b['prefix_sha256']},
            'metrics_bytes_at_resume': len(raw), 'metrics_records_at_resume': len(records),
            'post_boundary_records': len(after),
            'superseded': {'classification': SUPERSEDED, 'step_records': len(steps),
                           'global_step_range': [steps[0]['global_step'], steps[-1]['global_step']] if steps else None,
                           'epochs': sorted({r['epoch'] for r in steps}),
                           'epoch_complete_events': sorted(set(epochs_done)),
                           'checkpoint_events': sorted(set(ckpts)),
                           'loss_hex': [r.get('loss_hex') for r in steps],
                           'previous_reconciliations_after_boundary': len(prior)},
            'replay': {'from_epoch': e + 1, 'from_global_step': g + 1}}


def reconcile_files(ctx, state, entry, index):
    """Stale post-boundary files: recorded (size, SHA256) and removed; the committed sidecar is never touched."""
    run_dir = ctx.run_dir
    rdir = resume_dir(run_dir)
    removed, pruned = [], []
    history = {x['path']: x for x in index['entries']}
    for name in sorted(os.listdir(rdir)):
        p = rdir / name
        rel = str(p.relative_to(run_dir))
        if rel == entry['path']:
            continue
        require(p.is_file() and not p.is_symlink(), 'unexpected object in checkpoints/resume: ' + name)
        record = {'path': rel, 'size_bytes': p.stat().st_size, 'sha256': sha256_file(p)}
        h = history.get(rel)
        if h is not None and h['status'] == 'SUPERSEDED_BY_NEWER_BOUNDARY' and not h['bytes_pruned']:
            require(record['sha256'] == h['sha256'], 'retained superseded sidecar bytes changed')
            pruned.append(dict(record, reason='committed-then-superseded sidecar whose pruning was interrupted'))
        else:
            m = SIDECAR.fullmatch(name) or SIDECAR.fullmatch(name.removesuffix('.partial'))
            require(m is not None and int(m.group(1)) > entry['completed_epoch'],
                    'unindexed sidecar at or before the committed boundary: ' + name)
            removed.append(dict(record, reason='UNCOMMITTED post-boundary sidecar (never authorized, never loaded)'))
        os.remove(p)
    ck = aio.checkpoint_path(run_dir)
    partial = ck.with_name(ck.name + '.partial')
    if partial.exists():
        removed.append({'path': str(partial.relative_to(run_dir)), 'size_bytes': partial.stat().st_size,
                        'sha256': sha256_file(partial), 'reason': 'stale whole-module .partial (never replaced)'})
        os.remove(partial)
    fsync_dir(rdir)
    fsync_dir(ck.parent)
    return removed, pruned


def classify_scientific_checkpoint(ctx, state):
    """encoder_final.pkl on disk: committed-boundary bytes, superseded post-boundary bytes, or absent at E=0."""
    ck = aio.checkpoint_path(ctx.run_dir)
    committed = state['scientific_checkpoint_file']
    if not ck.exists():
        require(state['completed_epoch'] == 0 and committed is None, 'scientific checkpoint missing after epoch >= 1')
        return {'state': 'ABSENT_AT_EPOCH_0'}
    digest = sha256_file(ck)
    if committed is not None and digest == committed['sha256']:
        return {'state': 'MATCHES_COMMITTED_BOUNDARY', 'sha256': digest}
    require(state['completed_epoch'] < aio.EPOCHS, 'final scientific checkpoint bytes differ from the committed epoch 200')
    indexed = [c for c in json.loads(ctx.path('checkpoint_index').read_text())['checkpoints']
               if c['epoch'] == state['completed_epoch'] + 1 and c['sha256'] == digest]
    return {'state': SUPERSEDED + ('_INDEXED' if indexed else '_UNINDEXED'), 'sha256': digest,
            'size_bytes': ck.stat().st_size, 'epoch': state['completed_epoch'] + 1,
            'note': 'post-boundary whole-module bytes; never authoritative; overwritten when epoch E+1 is replayed'}


# ================================================================= the logical run
class LogicalRun:
    """A8 driver shared by the scientific CLI and the M6D6f qualification harness.

    fresh:  begin_fresh()  -> epoch-0 sidecar committed before the first iterator
    resume: begin_resume() -> verified load, identity gates, file/metrics reconciliation, restore
    each completed epoch: commit_boundary(epoch) -> encoder_final.pkl save, then sidecar commit, then prune.
    """

    def __init__(self, torch, trainer, ctx, config, *, storage, precision):
        self.torch, self.trainer, self.ctx, self.config = torch, trainer, ctx, config
        self.storage, self.precision = dict(storage), dict(precision)
        self.session = {'process_session_id': str(uuid.uuid4()), 'pid': os.getpid(), 'started_utc': utc()}
        self.counters = {'logical_authoritative_optimizer_steps': 0, 'physical_optimizer_steps_executed': 0,
                         'superseded_optimizer_steps': 0, 'process_sessions': 1}
        self._apps_at_start = 0
        self.identity = None
        self.verified = None

    # ------------------------------------------------------------ helpers
    def _identity(self):
        from methods.difffas import execution_policy as ep
        return process_identity(self.ctx, self.storage, ep.precision_state())

    def _counters_now(self):
        c = dict(self.counters)
        executed_here = self.trainer.optimizer_applications - self._apps_at_start
        c['physical_optimizer_steps_executed'] += executed_here
        c['logical_authoritative_optimizer_steps'] = self.trainer.global_step
        require(c['physical_optimizer_steps_executed'] == c['logical_authoritative_optimizer_steps'] +
                c['superseded_optimizer_steps'], 'physical = logical + superseded')
        return c

    def _commit(self, checkpoint):
        """Sidecar transaction: .partial (fsync) -> SHA256 -> os.replace -> index LAST -> explicit prune."""
        torch, ctx, t = self.torch, self.ctx, self.trainer
        epoch = t.completed_epoch
        require(t.global_step == epoch * aio.STEPS_PER_EPOCH, 'sidecar only at a completed epoch boundary')
        rdir = resume_dir(ctx.run_dir)
        rdir.mkdir(exist_ok=True)
        final = rdir / sidecar_name(epoch)
        partial = final.with_name(final.name + '.partial')
        require(not final.exists() and not partial.exists(), 'fresh sidecar name')
        counters = self._counters_now()
        state = build_state(torch, t, ctx, identity=self.identity, counters=counters, checkpoint=checkpoint,
                            session=self.session)
        with open(partial, 'xb') as fh:
            torch.save(state, fh)
            fh.flush()
            os.fsync(fh.fileno())
        size, digest = partial.stat().st_size, sha256_file(partial)
        os.replace(partial, final)
        fsync_dir(rdir)
        require(sha256_file(final) == digest, 'replaced sidecar bytes')
        entry = {'path': sidecar_rel(epoch), 'completed_epoch': epoch, 'global_step': t.global_step,
                 'size_bytes': size, 'sha256': digest, 'status': 'COMMITTED_CURRENT', 'bytes_pruned': False,
                 'committed_utc': utc(), 'process_session_id': self.session['process_session_id'],
                 'model_sha256': state['model_sha256'], 'optimizer_digest': state['optimizer_digest'],
                 'rng_digest': state['rng_digest'], 'metrics_boundary': state['metrics_boundary'],
                 'scientific_checkpoint_file': checkpoint, 'counters': counters}
        index = json.loads(index_path(ctx.run_dir).read_text())
        previous = index['committed']
        for e in index['entries']:
            if e['status'] == 'COMMITTED_CURRENT':
                e['status'] = 'SUPERSEDED_BY_NEWER_BOUNDARY'
        index['entries'].append(entry)
        index['committed'] = entry
        atomic_write_json(index_path(ctx.run_dir), index)            # LAST: the commit point
        fsync_dir(ctx.run_dir)
        pruned = None
        if previous is not None:                                     # only after the new index is committed
            old = ctx.run_dir / previous['path']
            require(sha256_file(old) == previous['sha256'], 'previous sidecar bytes before pruning')
            os.remove(old)
            fsync_dir(rdir)
            for e in index['entries']:
                if e['path'] == previous['path']:
                    e.update(bytes_pruned=True, pruned_utc=utc(), pruned_after_commit_of=entry['path'])
            index['pruning_log'].append({'path': previous['path'], 'sha256': previous['sha256'],
                                         'size_bytes': previous['size_bytes'], 'utc': utc(),
                                         'reason': f"explicit prune after commit of {entry['path']}"})
            atomic_write_json(index_path(ctx.run_dir), index)
            pruned = previous['path']
        ctx.log_event('e07c_aux_resume_state_committed', {
            'path': entry['path'], 'sha256': digest, 'size_bytes': size, 'completed_epoch': epoch,
            'global_step': t.global_step, 'pruned_previous': pruned, 'counters': counters,
            'process_session_id': self.session['process_session_id'], **ENGINEERING_LABELS})
        return entry

    # ------------------------------------------------------------ fresh
    def begin_fresh(self):
        """Epoch-0 boundary: after seed/model/cuda/optimizer/criterion/loader/A7, before iter(loader)."""
        ctx, t = self.ctx, self.trainer
        require(t.completed_epoch == t.global_step == t.optimizer_applications == 0, 'untrained fresh components')
        self.identity = self._identity()
        require(not index_path(ctx.run_dir).exists() and not resume_dir(ctx.run_dir).exists(), 'fresh root')
        atomic_write_json(index_path(ctx.run_dir), {
            'kind': RESUME_KIND, 'schema_version': SCHEMA_VERSION, 'amendment': 'A8',
            'a8_overlay_sha256': A8_OVERLAY_SHA256, 'method_id': aio.METHOD_ID, 'run_id': ctx.run_id,
            'run_uuid': ctx.run_uuid, 'runner_mode': ctx.mode, ctx.seed_field: ctx.aux_seed, **ENGINEERING_LABELS,
            'note': ('A8 engineering resume sidecars only. Exactly ONE entry is COMMITTED_CURRENT; it is the only '
                     'file --resume-state may name. Never a scientific checkpoint, selection candidate or main-DiffFAS '
                     'input; checkpoints/encoder_final.pkl (whole nn.Module) is the scientific checkpoint.'),
            'retention': 'COMMITTED_CURRENT only; previous bytes pruned explicitly after the next index commit',
            'committed': None, 'entries': [], 'pruning_log': [], 'stale_file_log': [],
            'sessions': [dict(self.session, kind='FRESH')]})
        entry = self._commit(checkpoint=None)
        return entry

    # ------------------------------------------------------------ resume
    def verify_before_open(self, supplied):
        """Read-only: exact indexed path, SHA256 before load, weights_only load, identity gates."""
        ctx = self.ctx
        path, entry, index = resolve_committed(ctx.run_dir, supplied)
        manifest = json.loads(ctx.path('run_manifest').read_text())
        require((manifest['run_id'], manifest['method_id'], manifest['runner_mode'], manifest[ctx.seed_field]) ==
                (ctx.run_id, aio.METHOD_ID, ctx.mode, ctx.aux_seed), 'run_manifest identity')
        require(manifest['identities'] == ctx.identities, 'run_manifest identities (config/A3/A6/A7/A8/lock/data)')
        require((index['run_id'], index['runner_mode'], index[ctx.seed_field], index['run_uuid']) ==
                (ctx.run_id, ctx.mode, ctx.aux_seed, manifest['run_uuid']), 'resume index identity')
        raw = read_verified(path, entry)
        state = load_sidecar(self.torch, raw)
        del raw
        self.verified = {'path': path, 'entry': entry, 'index': index, 'state': state,
                         'sha256_verified_before_load': True, 'weights_only': True}
        return self.verified

    def begin_resume(self):
        """After ctx.open(resume=True): identity gates, file + metrics reconciliation, restore, reconciliation record."""
        ctx, v = self.ctx, self.verified
        require(v is not None, 'verify_before_open first')
        state, entry, index = v['state'], v['entry'], v['index']
        self.identity = self._identity()
        checked = validate_state(state, identity=self.identity, ctx=ctx, entry=entry)
        metrics = reconcile_metrics(ctx, state)
        science = classify_scientific_checkpoint(ctx, state)
        removed, pruned = reconcile_files(ctx, state, entry, index)
        restored = restore(self.torch, self.trainer, state)                   # nothing consumes RNG after this
        c = state['counters']
        new_superseded = metrics['superseded']['step_records']
        self.counters = {'logical_authoritative_optimizer_steps': state['global_step'],
                         'physical_optimizer_steps_executed': c['physical_optimizer_steps_executed'] + new_superseded,
                         'superseded_optimizer_steps': c['superseded_optimizer_steps'] + new_superseded,
                         'process_sessions': c['process_sessions'] + 1}
        self._apps_at_start = self.trainer.optimizer_applications
        idx = json.loads(index_path(ctx.run_dir).read_text())
        for e in idx['entries']:
            if any(p['path'] == e['path'] for p in pruned):
                e.update(bytes_pruned=True, pruned_utc=utc(), pruned_after_commit_of=entry['path'])
        idx['pruning_log'].extend(dict(p, utc=utc()) for p in pruned)
        idx['stale_file_log'].extend(dict(r, utc=utc(), boundary=entry['path']) for r in removed)
        idx['sessions'].append(dict(self.session, kind='RESUME', restored_from=entry['path'],
                                    restored_sha256=entry['sha256']))
        atomic_write_json(index_path(ctx.run_dir), idx)
        payload = {'amendment': 'A8', 'logical_run': 'ONE_LOGICAL_RUN', 'run_id': ctx.run_id, 'run_uuid': ctx.run_uuid,
                   'process_session_id': self.session['process_session_id'],
                   'restored_sidecar': {'path': entry['path'], 'sha256': entry['sha256'], 'size_bytes': entry['size_bytes'],
                                        'sha256_verified_before_load': True, 'weights_only': True},
                   'identity_fields_checked': checked, 'scientific_checkpoint_on_disk': science,
                   'stale_files_removed': removed, 'interrupted_prunes_completed': pruned,
                   'restore': restored, 'counters_after_reconciliation': dict(self.counters), **metrics}
        ctx.log_event('e07c_aux_resume_reconciliation', payload)
        return payload

    # ------------------------------------------------------------ every completed epoch
    def commit_boundary(self, epoch):
        """pretrain_classifier.py:45 whole-module save (unchanged), THEN the engineering sidecar."""
        t = self.trainer
        require(epoch == t.completed_epoch and t.global_step == epoch * aio.STEPS_PER_EPOCH, 'completed epoch boundary')
        ck = t.checkpoint_event(epoch, self.ctx, self.config)
        entry = self._commit(checkpoint={'path': ck['path'], 'epoch': epoch, 'sha256': ck['sha256'],
                                         'size_bytes': ck['file_size_bytes'], 'format': ck['format']})
        return ck, entry

    def summary(self):
        c = self._counters_now()
        committed = read_index(self.ctx.run_dir)['committed']
        return {'logical_authoritative_optimizer_steps': c['logical_authoritative_optimizer_steps'],
                'physical_optimizer_steps_executed': c['physical_optimizer_steps_executed'],
                'superseded_optimizer_steps': c['superseded_optimizer_steps'],
                'committed_boundary': {'completed_epoch': committed['completed_epoch'],
                                       'global_step': committed['global_step'], 'sidecar': committed['path']},
                'uncommitted_in_progress_steps': c['logical_authoritative_optimizer_steps'] - committed['global_step'],
                'process_sessions': c['process_sessions'],
                'physical_counting': 'optimizer steps executed by every process of this logical run (A8 section 10)'}
