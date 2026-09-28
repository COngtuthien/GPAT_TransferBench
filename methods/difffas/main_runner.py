"""M6D6j E07c MAIN DiffFAS production runner (CONTROLLED_ADAPTATION, DEV-021). Torch is imported by the caller.

Reuses methods/difffas/main_graph.py (M6D6i) for the qualified training graph and adds statement-exact, AST-verified
replicas of the remaining pinned FAS_train.py loop body:

    report slot         FAS_train.py:86-98    report_slot()          (display only; no RNG, no state)
    periodic save slot  FAS_train.py:102-114  checkpoint_payload() + checkpoint_path_expr() -> main_checkpoint store
    visualization slot  FAS_train.py:117-163  visualize()            (OWNER D1: EXECUTE_EXACT_SOURCE)

Iteration order (MainRunner.iteration):
    train (main_graph.train_iteration: randint, training_losses, zero_grad, backward, scheduler.step, optimizer.step,
    EMA) -> report slot -> per-step metrics record -> periodic save slot [write, close, hash, index, verify, then prune the
    verified successor's direct predecessor (M6D6iR)] -> visualization slot -> at global_step 884000 ONLY: terminal
    checkpoint [write, verify, prune model_880000.pt] (OWNER D2: AFTER_FINAL_VISUALIZATION_BEFORE_RUN_COMPLETION).
Visualization is part of the trajectory: it creates a second TRAIN iterator (CPU RNG), torch.randperm (CPU RNG), runs
the pinned 250-step DDPM p_sample_loop with cond_scale 2 in TRAIN mode (CUDA RNG, dropout, BatchNorm running-buffer
updates) and writes a diagnostic PNG. It is never skipped, isolated, reduced or RNG-restored; there is no switch.
Bookkeeping (metrics, hashing, index, pruning) consumes no RNG and never touches model state.
"""
import ast
from contextlib import contextmanager
import fcntl
import functools
import getpass
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import socket
import sys
import time
import uuid

from methods.common.config import ROOT, load_method_config, sha256_file
from methods.common.learned import PreparationError
from methods.common.learned_runlog import LearnedRunContext
from methods.common.runlog import RunContext, RunDirectoryError, atomic_write_json, git_commit, load_logging_contract
from methods.difffas import main_checkpoint as mc
from methods.difffas import main_graph as mg
from methods.difffas import main_runner_io as mio
from methods.difffas.source import tree

LOCK_PATH = 'environments/e07c.lock.json'
TRAIN_SCRIPT = 'FAS_train.py'
require = mio.require
PRINT_LINES = (86, 96, 97, 98)
SAVE_IF_LINE, VIS_IF_LINE = 102, 117
EXPECTED_DEFAULTS = {'print_loss_every_iters': mio.PRINT_EVERY, 'save_checkpoints_every_iters': mio.SAVE_EVERY,
                     'save_images_every_iters': mio.VISUALIZE_EVERY, 'sample_algorithm': mio.SAMPLE_ALGORITHM,
                     'sample_initial_noise': mio.SAMPLE_INITIAL_NOISE, 'cond_scale': mio.COND_SCALE,
                     'DDIM_skip': mio.DDIM_SKIP, 'batch_size': mio.BATCH_SIZE, 'max_epochs': mio.EPOCHS}


class TrainingStop(RuntimeError):
    """A real-data gate failed (finiteness, batch plan, order): STOP_AND_REPORT."""


class MainMemoryBlocked(TrainingStop):
    """CUDA OOM at exact B=4: BLOCKED_BY_MAIN_B4_MEMORY. No fallback exists."""


def gate(value, message):
    if not value:
        raise TrainingStop('E07c main STOP_AND_REPORT: ' + message)


# ================================================================= source replicas (AST-verified)
def report_slot(args, progress_bar, loss_list, loss_mean_list, loss_vb_list, epoch, iters):
    if iters % args.print_loss_every_iters == 0:
        avg_loss = sum(loss_list) / len(loss_list)
        avg_loss_vb = sum(loss_vb_list) / len(loss_vb_list)
        avg_loss_mean = sum(loss_mean_list) / len(loss_mean_list)


        progress_bar.set_description(
            f"Loss: {avg_loss:.4f}, Loss_vb: {avg_loss_vb:.4f}, Loss_mean: {avg_loss_mean:.4f}, Epoch: {epoch+1}, Steps: {iters}"
        )

    loss_list = []
    loss_mean_list = []
    loss_vb_list = []


def checkpoint_payload(model, ema, scheduler, optimizer, conf):
    model_module = model
    payload = {
        "model": model_module.state_dict(),
        "ema": ema.state_dict(),
        "scheduler": scheduler.state_dict(),
        "optimizer": optimizer.state_dict(),
        "conf": conf,
    }
    return payload


def checkpoint_path_expr(conf, iters):
    return conf.training.ckpt_path + f"/model_{str(iters).zfill(6)}.pt"


def visualize(torch, torchvision, ddim_steps, diffusion, model, encoder, betas, val_loader, args, iters):
    print ('Generating samples at iters number ' + str(iters))
    val_loader_iter = iter(val_loader)
    val_batch = next(val_loader_iter)
    val_img = val_batch['content'].cuda()
    val_pose = val_batch['style_spoof'].cuda()
    num_samples = val_pose.size(0)
    shuffle_idx = torch.randperm(num_samples)
    val_pose_shuffled = val_pose[shuffle_idx]
    val_GT = val_batch['GT'].cuda()
    with torch.no_grad():
        if args.sample_algorithm == 'ddpm':
            print('Sampling algorithm used: DDPM')
            samples = diffusion.p_sample_loop(
                model,
                encoder,
                x_cond=[val_pose_shuffled, val_img],
                progress=True,
                cond_scale=args.cond_scale,
                sample_initial_noise=args.sample_initial_noise,
                means_size=args.means_size,
                var_size=args.var_size,
                use_pair=args.use_pair
            )
        elif args.sample_algorithm == 'ddim':
            print('Sampling algorithm used: DDIM')
            ddim_skip = args.DDIM_skip

            seq = range(0, args.sample_initial_noise, ddim_skip)
            xs, x0_preds = ddim_steps(
                encoder,
                seq,
                model,
                betas.cuda(),
                [val_pose_shuffled, val_img],
                diffusion = diffusion,
                cond_scale=args.cond_scale,
                sample_initial_noise=args.sample_initial_noise,
                means_size=args.means_size,
                var_size=args.var_size,
                use_pair=args.use_pair)
            samples = xs.cuda()
    grid = torch.cat([val_img, val_pose_shuffled[:, :3], samples], -1)
    grid = grid * 0.5 + 0.5
    img_name = f"{args.save_img_path}{iters}_output.png"
    torchvision.utils.save_image(grid, img_name, nrow=grid.shape[0] // 2)


def _own():
    raw = Path(__file__).read_bytes()
    mod = ast.parse(raw)
    fns = {n.name: n for n in mod.body if isinstance(n, ast.FunctionDef)}
    classes = {n.name: n for n in mod.body if isinstance(n, ast.ClassDef)}
    return raw, fns, classes


def _body(fn):
    return [s for s in fn.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]


ITERATION_ORDER = ['train', 'report', 'metrics', 'save', 'visualize', 'terminal']
SLOT_TESTS = {'iters % self.args.save_checkpoints_every_iters == 0': 'save',
              'iters % self.args.save_images_every_iters == 0': 'visualize',
              'iters == self.schedule.total': 'terminal'}
SLOT_CALLS = {'mg.train_iteration': 'train', 'report_slot': 'report', 'self.log_step': 'metrics'}


def iteration_order(runner_class):
    """Slot sequence of MainRunner.iteration from its AST (top-level statements only)."""
    fn = next(n for n in runner_class.body if isinstance(n, ast.FunctionDef) and n.name == 'iteration')
    order = []
    for s in _body(fn):
        if isinstance(s, ast.If) and ast.unparse(s.test) in SLOT_TESTS:
            order.append(SLOT_TESTS[ast.unparse(s.test)])
            continue
        value = s.value if isinstance(s, (ast.Assign, ast.Expr)) else None
        if isinstance(value, ast.Call) and ast.unparse(value.func) in SLOT_CALLS:
            order.append(SLOT_CALLS[ast.unparse(value.func)])
    return order


def verify_production_source(source):
    """AST equality of every replica with the pinned statements; frozen defaults; iteration-slot order."""
    module = tree(source, TRAIN_SCRIPT)
    raw, own, classes = _own()
    train = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == 'train')
    inner = next(n for n in ast.walk(train) if isinstance(n, ast.For) and n.lineno == 41)
    by_line = {s.lineno: s for s in inner.body}
    pinned_report = [ast.unparse(by_line[line]) for line in PRINT_LINES]
    require([ast.unparse(s) for s in _body(own['report_slot'])] == pinned_report, 'report slot == FAS_train.py:86-98')
    save_if, vis_if = by_line[SAVE_IF_LINE], by_line[VIS_IF_LINE]
    require(ast.unparse(save_if.test) == 'iters % args.save_checkpoints_every_iters == 0' and
            ast.unparse(vis_if.test) == 'iters % args.save_images_every_iters == 0', 'pinned save / visualization tests')
    call = save_if.body[1].value
    require(ast.unparse(save_if.body[0]) == 'model_module = model' and ast.unparse(call.func) == 'torch.save',
            'pinned save statements')
    pay = _body(own['checkpoint_payload'])
    require(ast.unparse(pay[0]) == 'model_module = model' and ast.unparse(pay[1].value) == ast.unparse(call.args[0]) and
            [k.value for k in call.args[0].keys] == list(mc.PAYLOAD_KEYS), 'payload == FAS_train.py:105-112')
    require(ast.unparse(_body(own['checkpoint_path_expr'])[0].value) == ast.unparse(call.args[1]),
            'checkpoint path == FAS_train.py:113')
    require([ast.unparse(s) for s in _body(own['visualize'])] == [ast.unparse(s) for s in vis_if.body],
            'visualize body == FAS_train.py:118-163')
    defaults = mg.pinned_argument_defaults(source)
    require({k: defaults[k] for k in EXPECTED_DEFAULTS} == EXPECTED_DEFAULTS, 'pinned argparse defaults')
    order = iteration_order(classes['MainRunner'])
    require(order == ITERATION_ORDER, 'iteration slot order ' + json.dumps(order))
    diffusion = tree(source, 'diffusion.py')
    gd = next(n for n in diffusion.body if isinstance(n, ast.ClassDef) and n.name == 'GaussianDiffusion')
    loop = next(n for n in gd.body if isinstance(n, ast.FunctionDef) and n.name == 'p_sample_loop')
    step = next(n for n in gd.body if isinstance(n, ast.FunctionDef) and n.name == 'p_sample')
    progressive = next(n for n in gd.body if isinstance(n, ast.FunctionDef) and n.name == 'p_sample_loop_progressive')
    require('noise = self.q_sample(x_cond[1], torch.tensor([sample_initial_noise]).cuda(), noise=noise)' in
            ast.unparse(loop) and 'noise = th.randn_like(x)' in ast.unparse(step) and
            'indices = list(range(sample_initial_noise))[::-1]' in ast.unparse(progressive),
            'pinned DDPM sampling path (q_sample start, per-step randn_like, 250 reverse steps)')
    return {'train_script_sha256': source['files_sha256'][TRAIN_SCRIPT],
            'diffusion_sha256': source['files_sha256']['diffusion.py'],
            'main_runner_sha256': hashlib.sha256(raw).hexdigest(),
            'replicas': {'report_slot': list(PRINT_LINES), 'checkpoint_payload': [103, 105, 112],
                         'checkpoint_path_expr': [113], 'visualize': [118, 163]},
            'triggers': {'save': 'iters % args.save_checkpoints_every_iters == 0 (:102)',
                         'visualize': 'iters % args.save_images_every_iters == 0 (:117)'},
            'pinned_defaults': {k: defaults[k] for k in EXPECTED_DEFAULTS},
            'iteration_order': order, 'sampling': {'algorithm': 'ddpm', 'function': 'GaussianDiffusion.p_sample_loop',
                                                   'reverse_steps': mio.SAMPLE_INITIAL_NOISE,
                                                   'model_forwards_per_step': 2, 'train_mode': True},
            'training_script_imported': False}


def production_args(source, config, run_dir):
    """main_graph.source_args + the pinned save / visualization / print defaults; nothing is overridable."""
    args, info = mg.source_args(source, config)
    defaults = mg.pinned_argument_defaults(source)
    for key in ('print_loss_every_iters', 'save_checkpoints_every_iters', 'save_images_every_iters', 'sample_algorithm',
                'sample_initial_noise', 'cond_scale', 'DDIM_skip'):
        setattr(args, key, defaults[key])
    s = config['sampler']
    require((args.cond_scale, args.sample_initial_noise, args.DDIM_skip) ==
            (s['cond_scale'], s['sample_initial_noise'], s['DDIM_skip']), 'visualization defaults == frozen sampler values')
    vis_dir = Path(run_dir) / 'diagnostics' / 'visualization'
    args.save_img_path = str(vis_dir) + '/'          # FAS_train.py concatenates f"{save_img_path}{iters}_output.png"
    return args, dict(info, visualization={k: defaults[k] for k in ('save_images_every_iters', 'sample_algorithm',
                                                                    'sample_initial_noise', 'cond_scale', 'DDIM_skip')},
                      save_img_path=args.save_img_path)


class Schedule:
    """Loop constants. Production code only ever uses Schedule.frozen(); unit tests may pass small fakes."""
    def __init__(self, iterations_per_epoch, epochs):
        self.iterations_per_epoch, self.epochs = iterations_per_epoch, epochs
        self.total = iterations_per_epoch * epochs

    @classmethod
    def frozen(cls):
        return cls(mio.ITERATIONS_PER_EPOCH, mio.EPOCHS)

    def is_frozen(self):
        return (self.iterations_per_epoch, self.epochs, self.total) == (mio.ITERATIONS_PER_EPOCH, mio.EPOCHS,
                                                                        mio.TOTAL_ITERATIONS)


# ================================================================= run context (run_logging_v1)
class E07cMainRunContext(LearnedRunContext):
    """run_logging_v1 writers for the MAIN E07c run root. SCIENTIFIC: <rt>/runs/m6/E07c/seed_<seed> (experiment seed).
    QUALIFICATION: <rt>/qualification/m6d6j/E07c/q60608-<id> (experiment_seed = null). A fresh context refuses an
    existing root; there is no resume in M6D6j (MAIN_CHECKPOINT_RESUME is unqualified)."""
    def __init__(self, *, mode, seed, runtime_root, environment, missing_environment_reasons, identities,
                 command_line=None):
        mio.validate_mode_seed(mode, seed)
        config = load_method_config(mio.METHOD_ID)
        require(identities['config_sha256'] == config['_runtime']['config_sha256'], 'config identity')
        self.mode, self.identities, self.contract = mode, dict(identities), load_logging_contract()
        self.method_id, self.config, self.runtime_root, self.resume = mio.METHOD_ID, config, Path(runtime_root), False
        self.run_seed = seed
        self.seed = seed if mode == mio.SCIENTIFIC else None
        self.source_commits = {config['source']['repository']: config['source']['pinned_commit']}
        self.command_line = command_line if command_line is not None else ' '.join(sys.argv)
        self.config_sha256, self.config_path = config['_runtime']['config_sha256'], config['_runtime']['config_path']
        self.git_commit = git_commit()
        if mode == mio.SCIENTIFIC:
            from methods.common.runlog import compute_run_id
            self.run_id = compute_run_id(mio.METHOD_ID, seed, self.config_sha256, self.git_commit)
            self.run_dir = mio.run_root(runtime_root, mode, seed)
        else:
            self.run_id = mio.qualification_id(self.config_sha256, self.git_commit)
            self.run_dir = mio.run_root(runtime_root, mode, seed, self.run_id)
        self.start_utc = self._metrics_fh = self._generation_fh = self._closed_summary = self._lock_fh = None
        self._records, self.run_uuid, self._started = 0, str(uuid.uuid4()), time.monotonic()
        required = {'host', 'user', 'platform', 'python_version', 'numpy_version', 'gpu_model', 'gpu_count',
                    'cuda_version', 'cudnn_version', 'framework', 'framework_version', 'pytorch_version',
                    'dependency_fingerprint', 'environment_lock_path', 'tensorflow_version'}
        if not required <= set(environment):
            raise PreparationError('explicit learned environment fields are missing')
        if any(v is None and not missing_environment_reasons.get(k) for k, v in environment.items()):
            raise PreparationError('null environment fields require reasons')
        self._validate_metadata(environment)
        self._environment, self._environment_reasons = dict(environment), dict(missing_environment_reasons)

    @property
    def seed_field(self):
        return 'experiment_seed' if self.mode == mio.SCIENTIFIC else 'qualification_seed'

    def open(self):
        if self._metrics_fh is not None or self._closed_summary is not None:
            raise RunDirectoryError('RunContext objects may be opened only once')
        self.run_dir.parent.mkdir(parents=True, exist_ok=True)
        self._lock_fh = (self.run_dir.parent / f'.{self.run_dir.name}.lock').open('a')
        try:
            fcntl.flock(self._lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if self.run_dir.exists():
                raise RunDirectoryError(f'STOP_AND_REPORT: E07c main run root already exists: {self.run_dir} (never '
                                        'overwritten; MAIN_CHECKPOINT_RESUME is unqualified)')
            return self._open_locked()
        except BaseException:
            for handle in (self._metrics_fh, self._generation_fh, self._lock_fh):
                if handle is not None:
                    handle.close()
            self._metrics_fh = self._generation_fh = self._lock_fh = None
            raise

    def _open_locked(self):
        result = RunContext._open_locked(self)
        (self.run_dir / 'diagnostics' / 'visualization').mkdir(parents=True)
        index = json.loads(self.path('checkpoint_index').read_text())
        index.update(note=('Every checkpoint actually written. Periodic bytes may be pruned only under M6D6iR '
                           '(successor-gated, explicit, logged); records are never removed.'),
                     experiment_seed=self.seed, runner_mode=self.mode, **{self.seed_field: self.run_seed},
                     checkpoint_rule='BASELINE_FINAL_STATE_V1', retention_policy=
                     'configs/amendments/e07c_m6d6ir_checkpoint_retention.yaml',
                     retention_policy_sha256=self.identities['m6d6ir_retention_record_sha256'])
        if self.mode == mio.QUALIFICATION:
            index.update(labels=list(mio.QUALIFICATION_LABELS))
        atomic_write_json(self.path('checkpoint_index'), index)
        return result

    def _resolved_config(self):
        resolved = RunContext._resolved_config(self)
        ids = self.identities
        resolved['_resolved'].update({
            'experiment_seed': self.seed, self.seed_field: self.run_seed, 'runner_mode': self.mode,
            'scientific_run': self.mode == mio.SCIENTIFIC, 'qualification_only': self.mode == mio.QUALIFICATION,
            'labels': list(mio.QUALIFICATION_LABELS) if self.mode == mio.QUALIFICATION else [],
            'identities': ids,
            'bound_overlays': [
                {'path': 'configs/amendments/e07c_a7_execution_policy.yaml', 'sha256': ids['a7_overlay_sha256']},
                {'path': 'configs/amendments/e07c_m6d6h_aux_encoder_freeze.yaml',
                 'sha256': ids['m6d6h_freeze_record_sha256']},
                {'path': 'configs/amendments/e07c_m6d6ir_checkpoint_retention.yaml',
                 'sha256': ids['m6d6ir_retention_record_sha256']},
                {'path': mio.CONTRACT_PATH, 'sha256': ids['m6d6j_contract_sha256']}],
            'relation': {'path': mio.RELATION, 'sha256': ids['relation_sha256'], 'rows': mio.TRAIN_ROWS},
            'loop': {'batch_size': mio.BATCH_SIZE, 'shuffle': mio.SHUFFLE, 'num_workers': mio.WORKERS,
                     'drop_last': mio.DROP_LAST, 'iterations_per_epoch': mio.ITERATIONS_PER_EPOCH,
                     'tail_batch': mio.TAIL_BATCH, 'epochs': mio.EPOCHS, 'total_iterations': mio.TOTAL_ITERATIONS,
                     'save_every': mio.SAVE_EVERY, 'visualize_every': mio.VISUALIZE_EVERY,
                     'visualization': 'OWNER D1: EXECUTE_EXACT_SOURCE (ddpm, 250 steps, cond_scale 2, train mode)',
                     'terminal': 'OWNER D2: AFTER_FINAL_VISUALIZATION_BEFORE_RUN_COMPLETION'},
            'logging_contract': {'version': 'run_logging_v1', 'sha256': ids['logging_contract_sha256'],
                                 'record_per': 'OPTIMIZER_STEP'},
            'resume': 'NOT_AVAILABLE (MAIN_CHECKPOINT_RESUME unqualified)'})
        return resolved

    def _manifest(self, **kwargs):
        m = LearnedRunContext._manifest(self, **kwargs)
        m.update(experiment_seed=self.seed, runner_mode=self.mode, **{self.seed_field: self.run_seed},
                 scientific_run=self.mode == mio.SCIENTIFIC, qualification_only=self.mode == mio.QUALIFICATION,
                 fidelity_class='CONTROLLED_ADAPTATION', deviation='DEV-021', identities=self.identities,
                 environment_lock_sha256=self.identities['environment_lock_sha256'],
                 m6d6j_file_sha256=mio.m6d6j_file_hashes(), val_split_accessed=False, resume_policy='NONE')
        if self.mode == mio.QUALIFICATION:
            m['missing_field_reasons']['experiment_seed'] = 'QUALIFICATION_ONLY_NOT_A_SCIENTIFIC_RUN'
            m['labels'] = list(mio.QUALIFICATION_LABELS)
        return m

    def log_event(self, event, payload=None):
        self._append({'record_type': 'event', 'event': event, 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                      'method_id': self.method_id, 'experiment_seed': self.seed, self.seed_field: self.run_seed,
                      'payload': payload or {}})

    def log_step(self, record):
        required = self.contract['trajectory']['required_fields']
        reasons = record.get('missing_field_reasons', {})
        if any(k not in record or (record[k] is None and not reasons.get(k)) for k in required):
            raise PreparationError('trajectory fields require values or explicit nulls with reasons')
        if any('test' in str(k).lower() for k in record):
            raise PreparationError('TEST must not appear in the trajectory')
        self._append({'record_type': 'step', 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **record})

    def close(self, *, completion_status='completed', summary=None, failure_reason=None):
        supplied = dict(summary or {})
        require(not supplied.get('test_split_accessed'), 'TEST is prohibited')
        reasons = dict(supplied.get('missing_field_reasons', {}))
        if self.mode == mio.QUALIFICATION:
            reasons.setdefault('experiment_seed', 'QUALIFICATION_ONLY_NOT_A_SCIENTIFIC_RUN')
        for key in ('final_or_selected_checkpoint_path', 'final_or_selected_checkpoint_sha256',
                    'seed_level_evaluation_metrics', 'training_duration_seconds', 'peak_vram_bytes', 'failure_reason',
                    'output_manifest_path'):
            if supplied.get(key) is None:
                reasons.setdefault(key, 'Not supplied by caller; no execution/evaluation result inferred')
        supplied.update(missing_field_reasons=reasons, runner_mode=self.mode, **{self.seed_field: self.run_seed})
        supplied.setdefault('peak_vram_reason', reasons.get('peak_vram_bytes'))
        self._validate_metadata(supplied)
        return RunContext.close(self, completion_status=completion_status, summary=supplied,
                                failure_reason=failure_reason)


# ================================================================= setup (A7 order)
def environment_record(torch):
    import numpy as np
    env = {'host': socket.gethostname(), 'user': getpass.getuser(), 'platform': platform.platform(),
           'python_version': platform.python_version(), 'numpy_version': np.__version__,
           'gpu_model': torch.cuda.get_device_name(0), 'gpu_count': torch.cuda.device_count(),
           'cuda_version': torch.version.cuda, 'cudnn_version': torch.backends.cudnn.version(),
           'framework': 'torch', 'framework_version': torch.__version__, 'pytorch_version': torch.__version__,
           'tensorflow_version': None, 'dependency_fingerprint': sha256_file(ROOT / LOCK_PATH),
           'environment_lock_path': LOCK_PATH, 'environment_name': 'gpat-m6-e07c', 'python_executable': sys.executable,
           'gpu_note': 'single RTX 3090; exact B=4 FP32 (A7); no microbatch/accumulation/AMP'}
    return env, {'tensorflow_version': 'E07c is a PyTorch method; TensorFlow is not part of gpat-m6-e07c'}


def seed_process(torch, mode, seed, config):
    """SCIENTIFIC: seed_adapter.apply_seed(seed, cuda=True). QUALIFICATION: the same framework calls with 60608 (the
    scientific API is never called with a qualification seed). FAS_train.py:180 seed_torch semantics."""
    import numpy as np
    from methods.difffas import seed_adapter
    mio.validate_mode_seed(mode, seed)
    require(os.environ.get('PYTHONHASHSEED') == str(seed), f'launch with PYTHONHASHSEED={seed}')
    if mode == mio.SCIENTIFIC:
        plan = seed_adapter.apply_seed(seed, cuda=True, config=config)
        out = {'seed': seed, 'role': 'EXPERIMENT_SEED', 'plan_scope': plan['scope'],
               'hook': 'methods/difffas/seed_adapter.py::apply_seed'}
    else:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        out = {'seed': seed, 'role': 'QUALIFICATION_SEED_NOT_AN_EXPERIMENT_SEED',
               'hook': 'main_runner.seed_process (same calls as seed_adapter.apply_seed; scientific API not called)'}
    out.update(pythonhashseed=os.environ.get('PYTHONHASHSEED'),
               torch_cpu_rng_sha256=mio.sha(torch.get_rng_state().numpy().tobytes()),
               torch_cuda_rng_sha256=mio.sha(torch.cuda.get_rng_state().numpy().tobytes()))
    return out


def configure_precision(torch, config):
    from methods.difffas import execution_policy as ep
    for key, value in mio.LAUNCH_ENVIRONMENT.items():
        require(os.environ.get(key) == value, f'launch with {key}={value} (environments/e07c.lock.json)')
    state = ep.apply_e07c_precision_policy(config)
    require(not torch.is_autocast_enabled('cuda'), 'autocast disabled')
    return dict(state, grad_scaler=False, amp=False, activation_checkpointing=False)


def payload_bytes(torch, payload):
    """Tensor bytes of the payload (successor free-space gate estimate)."""
    total = 0

    def walk(node):
        nonlocal total
        if torch.is_tensor(node):
            total += node.numel() * node.element_size()
        elif isinstance(node, dict):
            for v in node.values():
                walk(v)
        elif isinstance(node, (list, tuple)):
            for v in node:
                walk(v)
    walk({k: payload[k] for k in ('model', 'ema', 'scheduler', 'optimizer')})
    return total + (64 << 20)


class MainRunner:
    """Holds the constructed source objects and executes the frozen iteration body."""
    def __init__(self, *, torch, torchvision, ddim_steps, model, ema, encoder, optimizer, scheduler, betas, diffusion,
                 conf, args, loader, dataset, ctx, store, schedule, mode, seed, tqdm):
        self.torch, self.torchvision, self.ddim_steps = torch, torchvision, ddim_steps
        self.model, self.ema, self.encoder, self.optimizer, self.scheduler = model, ema, encoder, optimizer, scheduler
        self.betas, self.diffusion, self.conf, self.args = betas, diffusion, conf, args
        self.loader, self.dataset, self.ctx, self.store, self.schedule = loader, dataset, ctx, store, schedule
        self.mode, self.seed, self.tqdm = mode, seed, tqdm
        self.pending, self.trace, self.visualizations, self.terminal = [], [], [], None
        self.t0 = time.monotonic()
        self.peak_step_gpu_memory_bytes = 0
        dataset.on_item = self.pending.append

    def iteration(self, batch, epoch, iters, progress_bar):
        """ONE pinned loop body (FAS_train.py:42-163) + benchmark bookkeeping; returns the new global step."""
        torch = self.torch
        items, self.pending[:] = list(self.pending), []
        gate(len(items) == batch['GT'].shape[0] and len(items) in (mio.BATCH_SIZE, mio.TAIL_BATCH), 'batch items')
        torch.cuda.reset_peak_memory_stats()
        self.trace.append(('train', iters + 1))
        out = mg.train_iteration(torch, self.conf, self.args, batch, iters, self.model, self.ema, self.encoder,
                                 self.diffusion, self.betas, self.optimizer, self.scheduler, 'cuda')
        iters = out['iters']
        gate(mio.finite(out['loss_list'] + out['loss_mean_list'] + out['loss_vb_list']), 'non-finite loss')
        self.trace.append(('report', iters))
        report_slot(self.args, progress_bar, out['loss_list'], out['loss_mean_list'], out['loss_vb_list'], epoch, iters)
        self.log_step(out, epoch, iters, items)
        if iters % self.args.save_checkpoints_every_iters == 0:
            self.trace.append(('save', iters))
            self.checkpoint_transition(iters, 'periodic')
        if iters % self.args.save_images_every_iters == 0:
            self.trace.append(('visualize', iters))
            self.visualize_slot(iters)
        if iters == self.schedule.total:
            self.trace.append(('terminal', iters))
            self.checkpoint_transition(iters, 'terminal')
        return iters

    def log_step(self, out, epoch, iters, items):
        torch = self.torch
        peak = torch.cuda.max_memory_allocated()
        self.peak_step_gpu_memory_bytes = max(self.peak_step_gpu_memory_bytes, peak)
        tracks = '\n'.join(r['track_pair_id'] for r in items).encode('utf-8')
        record = mio.step_record(mode=self.mode, seed=self.seed, epoch=epoch + 1, global_step=iters,
                                 iteration=(iters - 1) % self.schedule.iterations_per_epoch,
                                 learning_rate=self.optimizer.param_groups[0]['lr'], loss=out['loss_list'][0],
                                 mse=out['loss_mean_list'][0], vb=out['loss_vb_list'][0], batch_size=len(items),
                                 batch_track_sha256=mio.sha(tracks), wall_clock_seconds=time.monotonic() - self.t0,
                                 gpu_memory_bytes=peak)
        self.ctx.log_step(record)
        return record

    def checkpoint_transition(self, iters, kind):
        """Save slot / terminal: exact payload -> safe write -> index -> verify -> successor-gated prune."""
        payload = checkpoint_payload(self.model, self.ema, self.scheduler, self.optimizer, self.conf)
        expected = checkpoint_path_expr(self.conf, iters)
        entry = self.store.write(global_step=iters, epoch=iters // self.schedule.iterations_per_epoch, kind=kind,
                                 payload=payload, estimated_bytes=payload_bytes(self.torch, payload))
        gate(str(self.store.run_dir / entry['path']) == expected, 'store path == FAS_train.py:113 path')
        self.ctx.log_event('e07c_main_checkpoint_verified', {k: entry[k] for k in mc.INDEX_FIELDS})
        pruned = self.store.transition(entry['path'])
        if pruned is not None:
            self.ctx.log_event('e07c_main_checkpoint_pruned', pruned)
        if kind == 'terminal':
            self.terminal = entry
        return entry, pruned

    def visualize_slot(self, iters):
        self.pending[:] = []
        t0 = time.monotonic()
        visualize(self.torch, self.torchvision, self.ddim_steps, self.diffusion, self.model, self.encoder, self.betas,
                  self.loader, self.args, iters)
        items, self.pending[:] = list(self.pending), []
        png = Path(f'{self.args.save_img_path}{iters}_output.png')
        gate(png.is_file() and len(items) == mio.BATCH_SIZE, 'visualization output and first TRAIN batch')
        info = {'global_step': iters, 'png': str(png.relative_to(self.ctx.run_dir)), 'png_sha256': sha256_file(png),
                'png_bytes': png.stat().st_size, 'items': [r['track_pair_id'] for r in items],
                'seconds': round(time.monotonic() - t0, 3)}
        self.visualizations.append(info)
        self.ctx.log_event('e07c_main_visualization', info)
        return info

    def run(self):
        """The frozen scientific loop (FAS_train.py:37-41); called only by run_scientific."""
        iters = 0
        for epoch in range(self.args.max_epochs):
            progress_bar = mg.epoch_progress(self.tqdm, self.loader)
            for batch in progress_bar:
                iters = self.iteration(batch, epoch, iters, progress_bar)
        gate(iters == self.schedule.total and self.terminal is not None and
             self.terminal['global_step'] == self.schedule.total, 'terminal state after the full budget')
        return iters


@contextmanager
def build_production(torch, *, mode, seed, runtime_root, faces_root, ctx, source, config, schedule, tqdm, on_stage=None):
    """A7 order: seed -> precision -> transform -> dataset -> DataLoader -> model/EMA/AdamW/scheduler -> fresh refusal
    -> betas/diffusion -> main_runner_encoder -> encoder.eval(); yields the MainRunner inside the encoder context."""
    import torchvision
    from torchvision import transforms
    from torch.utils.data import DataLoader
    from tensorfn import load_config
    from methods.common.upstream import upstream_modules
    from methods.difffas import aux_checkpoint as seam
    from methods.difffas import execution_policy as ep
    from methods.difffas import runtime_qualification as rq
    stage = on_stage or (lambda name, **kw: None)
    records, relation = mio.read_relation()
    datasets, membership = mio.read_train_membership(records)
    reader = mio.CanonicalFaceReader(faces_root, datasets)
    with upstream_modules(Path(source['root']), rq.UPSTREAM_MAIN, rq.UPSTREAM_MAIN_ROOTS) as modules:
        dc, dmod = modules['config.diffconfig'], modules['diffusion']
        DiffConf = load_config(dc.DiffusionConfig, str(Path(source['root']) / 'config/diffusion.conf'), (), False)
        diff_config = mg.verify_diff_config(DiffConf, config)
        DiffConf.training.ckpt_path = str(ctx.run_dir / 'checkpoints')       # FAS_train.py:278 analogue
        args, args_info = production_args(source, config, ctx.run_dir)
        seeding = seed_process(torch, mode, seed, config)
        stage('seed')
        precision = configure_precision(torch, config)
        stage('precision')
        transform = mg.build_transform(transforms)
        stage('transform')
        dataset = mio.MainTrainDataset(records, reader, transform)
        stage('dataset', dataset=dataset, reader=reader)
        loader = mg.build_dataloader(DataLoader, dataset, args)
        require((loader.batch_size, loader.num_workers, loader.drop_last, type(loader.sampler).__name__,
                 loader.generator, len(loader)) ==
                (mio.BATCH_SIZE, mio.WORKERS, mio.DROP_LAST, 'RandomSampler', None, mio.ITERATIONS_PER_EPOCH),
                'source DataLoader semantics (2210 batches, drop_last=False, no generator, 0 workers)')
        stage('dataloader')
        model, ema, optimizer, scheduler, betas, diffusion = mg.build_training_objects(
            dc.get_model_conf, DiffConf, dmod.create_gaussian_diffusion, args)
        stage('objects')
        validate = (mc.scientific_validator(config, seed, mio.ITERATIONS_PER_EPOCH) if mode == mio.SCIENTIFIC else
                    mc.qualification_validator(mio.ITERATIONS_PER_EPOCH, mio.EPOCHS, mio.TERMINAL_STEP))
        store = mc.CheckpointStore(ctx.run_dir, validate=validate, serializer=lambda p, fh: torch.save(p, fh),
                                   labels=mio.QUALIFICATION_LABELS if mode == mio.QUALIFICATION else ())
        with ep.main_runner_encoder(runtime_root, seam.OWNER_FROZEN_SHA256, config) as encoder:
            encoder.eval()
            stage('encoder_eval')
            runner = MainRunner(torch=torch, torchvision=torchvision, ddim_steps=dmod.ddim_steps, model=model, ema=ema,
                                encoder=encoder, optimizer=optimizer, scheduler=scheduler, betas=betas,
                                diffusion=diffusion, conf=DiffConf, args=args, loader=loader, dataset=dataset,
                                ctx=ctx, store=store, schedule=schedule, mode=mode, seed=seed, tqdm=tqdm)
            runner.evidence = {'relation': relation, 'membership': membership, 'seeding': seeding,
                               'precision': precision, 'diff_config': diff_config, 'args': args_info,
                               'upstream_modules': {'diffusion': dmod, 'unet_autoenc': modules['models.unet_autoenc']}}
            yield runner


def quiet_tqdm():
    """Non-scientific I/O handling: the FAS_train.py:40 tqdm wrapper with its display disabled (iteration identical;
    no RNG). The trajectory is retained by the per-step metrics.jsonl records instead of carriage-return bars."""
    from tqdm import tqdm
    return functools.partial(tqdm, disable=True)


# ================================================================= scientific entry (tools/run_e07c_main.py)
def run_scientific(*, seed, runtime_root, faces_root, command_line=None):
    """ONE 400-epoch E07c main scientific seed. NOT launched in M6D6j; fresh only (no resume)."""
    import torch
    from methods.common.runlog import git_dirty
    from methods.difffas import DiffFASAdapter
    mode = mio.SCIENTIFIC
    mio.validate_mode_seed(mode, seed)
    require(not git_dirty(), 'SCIENTIFIC runs require a clean git worktree')
    contract = mio.load_contract()
    ids = mio.identities(contract)
    adapter = DiffFASAdapter()
    config = adapter.config
    source = adapter.validate_source()
    require((source['commit'], source['tree']) == (ids['source_commit'], ids['source_tree']), 'source identity')
    source_evidence = verify_production_source(source)
    env, reasons = environment_record(torch)
    schedule = Schedule.frozen()
    ctx = E07cMainRunContext(mode=mode, seed=seed, runtime_root=runtime_root, environment=env,
                             missing_environment_reasons=reasons, identities=ids, command_line=command_line)
    with ctx:
        streams = mio.tee_run_logs(ctx.run_dir)
        try:
            with build_production(torch, mode=mode, seed=seed, runtime_root=runtime_root, faces_root=faces_root,
                                  ctx=ctx, source=source, config=config, schedule=schedule,
                                  tqdm=quiet_tqdm()) as runner:
                ctx.log_event('e07c_main_run_start', {k: v for k, v in runner.evidence.items()
                                                      if k != 'upstream_modules'} | {'source': source_evidence})
                t0 = time.monotonic()
                iters = runner.run()
                index = runner.store.read_index()['checkpoints']
                created = [e for e in index if e['checkpoint_type'] in ('periodic', 'terminal')]
                require(iters == mio.TOTAL_ITERATIONS and len(created) == len(mio.PERIODIC_STEPS) + 1 and
                        [e['global_step'] for e in created] == list(mio.PERIODIC_STEPS) + [mio.TERMINAL_STEP] and
                        sum(e['bytes_pruned'] for e in index) == len(mio.PERIODIC_STEPS) and
                        [e['path'] for e in index if e['bytes_present']] == [runner.terminal['path']],
                        '89 created, 88 pruned, 1 retained terminal')
                ctx.close(summary={
                    'final_or_selected_checkpoint_path': str(ctx.run_dir / runner.terminal['path']),
                    'final_or_selected_checkpoint_sha256': runner.terminal['sha256'],
                    'final_checkpoint_roles': runner.terminal['logical_roles'],
                    'completed_epoch': mio.EPOCHS, 'global_step': iters, 'optimizer_steps': iters,
                    'checkpoints_created': len(created), 'checkpoints_pruned': len(mio.PERIODIC_STEPS),
                    'checkpoints_retained': 1, 'visualizations_executed': len(runner.visualizations),
                    'training_duration_seconds': time.monotonic() - t0,
                    'peak_vram_bytes': runner.peak_step_gpu_memory_bytes, 'test_split_accessed': False,
                    'missing_field_reasons': {'seed_level_evaluation_metrics':
                                              'Generator training: evaluation happens in later milestones (M8+)'}})
        finally:
            mio.untee(streams)
    return ctx.run_dir
