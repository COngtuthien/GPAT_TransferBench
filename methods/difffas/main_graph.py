"""E07c MAIN DiffFAS training core, reconstructed statement-for-statement from pinned FAS_train.py (M6D6i).

Reusable by the future main production runner (M6D6j), which must call these functions instead of
writing a second copy. The pinned training script is never imported or executed (source.py): each
replica below is AST-compared with the pinned statements by verify_source_equivalence().

Scope (QUALIFIED IN M6D6i: MAIN_DIFFFAS_TRAINING_GRAPH):
    build_transform          FAS_train.py:184-188
    build_dataloader         FAS_train.py:205
    build_training_objects   FAS_train.py:216-219, 222-223, 226 (fresh path only), 234-235
    epoch_progress           FAS_train.py:40
    train_iteration          FAS_train.py:31-33 (per-iteration lists) and 42-83
    accumulate               FAS_train.py:21-26

Not here (later milestones): the dataset, the epoch loop, the print cadence (86-98), the checkpoint
save branch (102-114), the visualization/sampling branch (117-163), resume (226-232) and run logging.
The encoder is NOT built here: FAS_train.py:34-35 is replaced by
methods.difffas.execution_policy.main_runner_encoder (A7) followed by encoder.eval() in the caller.

No I/O of data, checkpoints or run roots; no torch import at module level (torch, DataLoader,
transforms and tqdm are passed in by the caller); no seeding and no scientific-seed decision.
Source quirks are preserved, including scheduler.step() BEFORE optimizer.step() (FAS_train.py:76-77)
and EMA accumulation of named_parameters only (buffers are never copied).
"""
import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace

from methods.common.config import load_method_config
from methods.common.learned import authoritative, PreparationError
from .source import tree

TRAIN_SCRIPT = 'FAS_train.py'
# (replica function, number of leading replica statements not compared, pinned FAS_train.py lines)
REPLICAS = (
    ('build_transform', 0, (184,)),
    ('build_dataloader', 0, (205,)),
    ('build_training_objects', 1, (216, 217, 218, 219, 222, 223, 226, 234, 235)),
    ('epoch_progress', 0, (40,)),
    ('train_iteration', 0, (31, 32, 33, 42, 43, 44, 45, 46, 47, 48, 50, 57, 70, 71, 72, 74, 75, 76, 77,
                            79, 80, 81, 83)),
)
TEST_ONLY_LINES = {226: 'resume branch: only the test `args.pretrain_path is not None` is replicated; the '
                        'body (torch.load + load_state_dict) is replaced by a refusal (MAIN_CHECKPOINT_RESUME '
                        'unqualified)'}
ACCUMULATE_LINES = (21, 26)
# argparse defaults (FAS_train.py:247-273) that the frozen E07c config binds; use_pair is the A1 official flag.
ARG_BINDINGS = {'batch_size': 'training.batch_size', 'guidance_prob': 'training.guidance_prob',
                'means_size': 'training.means_size', 'var_size': 'training.var_size',
                'max_epochs': 'training.max_epochs', 'use_pair': 'training.use_pair'}
A1_OVERRIDES = {'use_pair': {'pinned_default': True, 'frozen': False,
                             'authority': 'A1 official_code_path.use_pair=false (official flag, not a source edit)'}}


def accumulate(model1, model2, decay=0.9999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def build_transform(transforms):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
    ])
    return transform


def build_dataloader(DataLoader, dataset, args):
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader


def build_training_objects(get_model_conf, DiffConf, create_gaussian_diffusion, args):
    """A7 steps 5-13 (use_pair=False branch). Fresh run only; returns the objects train() receives."""
    if args.use_pair is not False or args.pretrain_path is not None:
        raise PreparationError('E07c main graph: A1 use_pair=False and a FRESH run (pretrain_path None) only')
    model = get_model_conf().make_model()
    model = model.to(args.device)
    ema = get_model_conf().make_model()
    ema = ema.to(args.device)
    optimizer = DiffConf.training.optimizer.make(model.parameters())
    scheduler = DiffConf.training.scheduler.make(optimizer)
    if args.pretrain_path is not None:
        raise PreparationError('MAIN_CHECKPOINT_RESUME is not qualified; FAS_train.py:227-232 is not implemented')
    betas = DiffConf.diffusion.beta_schedule.make()
    diffusion = create_gaussian_diffusion(betas, predict_xstart = False)
    return model, ema, optimizer, scheduler, betas, diffusion


def epoch_progress(tqdm, loader):
    progress_bar = tqdm(loader, desc="Training")
    return progress_bar


def train_iteration(torch, conf, args, batch, iters, model, ema, encoder, diffusion, betas, optimizer, scheduler,
                    device):
    """One pass of the FAS_train.py:41 loop body up to :83 (EMA). The caller owns the loop and `iters`."""
    loss_list = []
    loss_mean_list = []
    loss_vb_list = []
    iters = iters + 1
    img = batch['content']
    target_img = batch['GT']
    target_pose = batch['style_spoof']
    img = img.to(device)
    target_img = target_img.to(device)
    target_pose = target_pose.to(device)

    time_t = torch.randint(
        0,
        conf.diffusion.beta_schedule["n_timestep"],
        (img.shape[0],),
        device=device,
    )

    loss_dict = diffusion.training_losses(
        model,
        encoder,
        x_start=target_img,
        t=time_t,
        betas=betas.cuda(),
        cond_input=[img, target_pose],
        prob=1 - args.guidance_prob,
        means_size = args.means_size,
        var_size = args.var_size,
        use_pair = args.use_pair
    )

    loss = loss_dict['loss'].mean()
    loss_mse = loss_dict['mse'].mean()
    loss_vb = loss_dict['vb'].mean()

    optimizer.zero_grad()
    loss.backward()
    scheduler.step()
    optimizer.step()

    loss_list.append(loss.detach().item())
    loss_mean_list.append(loss_mse.detach().item())
    loss_vb_list.append(loss_vb.detach().item())

    accumulate(ema, model, 0 if iters < conf.training.scheduler.warmup else 0.9999)
    return {'iters': iters, 'time_t': time_t, 'loss_dict': loss_dict, 'loss': loss, 'loss_mse': loss_mse,
            'loss_vb': loss_vb, 'loss_list': loss_list, 'loss_mean_list': loss_mean_list, 'loss_vb_list': loss_vb_list,
            'content_after_loss': img, 'GT': target_img, 'style_spoof': target_pose}


# ----------------------------------------------------------------------------- source traceability
def _pinned_statement(module, line):
    nodes = [n for n in ast.walk(module) if isinstance(n, ast.stmt) and n.lineno == line]
    if len(nodes) != 1:
        raise PreparationError(f'pinned {TRAIN_SCRIPT}:{line} is not exactly one statement')
    node = nodes[0]
    return ast.unparse(node.test) if line in TEST_ONLY_LINES else ast.unparse(node)


def _own_functions():
    raw = Path(__file__).read_bytes()
    return raw, {n.name: n for n in ast.parse(raw).body if isinstance(n, ast.FunctionDef)}


def _replica_statements(fn, lead):
    body = [s for s in fn.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
    return [ast.unparse(s.test) if isinstance(s, ast.If) and i >= lead and
            ast.unparse(s.test) == 'args.pretrain_path is not None' else ast.unparse(s)
            for i, s in enumerate(body)][lead:]


def verify_source_equivalence(source):
    """Every replica statement equals the pinned statement (ast.unparse), in the pinned order."""
    module = tree(source, TRAIN_SCRIPT)
    own_raw, own = _own_functions()
    evidence = []
    for name, lead, lines in REPLICAS:
        pinned = [_pinned_statement(module, line) for line in lines]
        replica = _replica_statements(own[name], lead)
        tail = replica[len(pinned):]
        if replica[:len(pinned)] != pinned or [s.split(' ')[0] for s in tail] not in ([], ['return']):
            raise PreparationError(f'main_graph.{name} is not statement-equal to pinned {TRAIN_SCRIPT}')
        if lines != tuple(sorted(lines)):
            raise PreparationError('replica line order')
        evidence.append({'replica': 'methods/difffas/main_graph.py::' + name, 'pinned_lines': list(lines),
                         'statements_compared': len(pinned), 'leading_guard_statements': lead,
                         'trailing': 'return' if tail else None})
    pinned_acc = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == 'accumulate')
    if ((pinned_acc.lineno, pinned_acc.end_lineno) != ACCUMULATE_LINES or
            ast.unparse(pinned_acc) != ast.unparse(own['accumulate'])):
        raise PreparationError('main_graph.accumulate is not the pinned FAS_train.py:21-26 function')
    evidence.append({'replica': 'methods/difffas/main_graph.py::accumulate', 'pinned_lines': list(ACCUMULATE_LINES),
                     'comparison': 'whole FunctionDef (signature, default decay=0.9999, body)'})
    train = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == 'train')
    inner = [n for n in ast.walk(train) if isinstance(n, ast.For) and n.lineno == 41]
    if len(inner) != 1:
        raise PreparationError('pinned batch loop FAS_train.py:41')
    order = [(s.lineno, ast.unparse(s)) for s in inner[0].body if s.lineno in (74, 75, 76, 77, 83)]
    if [o[1] for o in order] != ['optimizer.zero_grad()', 'loss.backward()', 'scheduler.step()', 'optimizer.step()',
                                 'accumulate(ema, model, 0 if iters < conf.training.scheduler.warmup else 0.9999)']:
        raise PreparationError('pinned step order zero_grad -> backward -> scheduler.step -> optimizer.step -> EMA')
    excluded = {s.lineno: ast.unparse(s.test) for s in inner[0].body if isinstance(s, ast.If)}
    if excluded != {86: 'iters % args.print_loss_every_iters == 0', 102: 'iters % args.save_checkpoints_every_iters == 0',
                    117: 'iters % args.save_images_every_iters == 0'}:
        raise PreparationError('pinned loop branches after :83 changed')
    return {'train_script': TRAIN_SCRIPT, 'train_script_sha256': source['files_sha256'][TRAIN_SCRIPT],
            'main_graph_sha256': hashlib.sha256(own_raw).hexdigest(), 'replicas': evidence,
            'test_only_lines': {str(k): v for k, v in TEST_ONLY_LINES.items()},
            'pinned_step_order': [{'line': line, 'statement': s} for line, s in order],
            'scheduler_step_before_optimizer_step': True,
            'branches_not_replicated': {str(k): v for k, v in excluded.items()},
            'lines_not_replicated': {'86-98': 'print cadence (display only)', '102-114': 'checkpoint save branch',
                                     '117-163': 'visualization / sampling branch', '227-232': 'resume body',
                                     '34-35': 'encoder load -> A7 main_runner_encoder + caller encoder.eval()'},
            'training_script_imported': False}


def pinned_argument_defaults(source):
    """argparse defaults in the pinned `if __name__ == '__main__':` block (AST only)."""
    module = tree(source, TRAIN_SCRIPT)
    out = {}
    for node in ast.walk(module):
        if (isinstance(node, ast.Call) and ast.unparse(node.func) == 'parser.add_argument' and node.args and
                isinstance(node.args[0], ast.Constant) and str(node.args[0].value).startswith('--')):
            kw = {k.arg: k.value for k in node.keywords}
            if 'default' in kw:
                out[node.args[0].value[2:]] = ast.literal_eval(kw['default'])
    return out


def source_args(source, config=None, *, device='cuda'):
    """The `args` namespace train() reads, bound to the frozen E07c config and checked against pinned defaults."""
    cfg = authoritative(config) if config is not None else load_method_config('E07c')
    defaults = pinned_argument_defaults(source)
    values = {}
    for arg, path in ARG_BINDINGS.items():
        section, key = path.split('.')
        values[arg] = cfg[section][key]
        expected = A1_OVERRIDES[arg]['frozen'] if arg in A1_OVERRIDES else defaults[arg]
        if values[arg] != expected or (arg in A1_OVERRIDES and defaults[arg] != A1_OVERRIDES[arg]['pinned_default']):
            raise PreparationError(f'frozen {path} differs from the pinned argparse default of --{arg}')
    if (defaults['pretrain_path'], defaults['device']) != (None, device):
        raise PreparationError('pinned --pretrain_path / --device defaults')
    args = SimpleNamespace(**values, device=device, pretrain_path=None)
    return args, {'values': dict(vars(args)), 'pinned_defaults': {k: defaults[k] for k in (*ARG_BINDINGS, 'pretrain_path',
                                                                                           'device')},
                  'a1_overrides': A1_OVERRIDES, 'fresh_run': True}


def verify_diff_config(DiffConf, config=None):
    """The tensorfn DiffConf (config/diffusion.conf) equals the frozen E07c optimizer/scheduler/diffusion values."""
    cfg = authoritative(config) if config is not None else load_method_config('E07c')
    o, d = cfg['optimizer'], cfg['diffusion']
    opt, sch, beta = DiffConf.training.optimizer, DiffConf.training.scheduler, DiffConf.diffusion.beta_schedule
    observed = {'optimizer': {'type': opt.type, 'lr': opt.lr, 'betas': list(opt.betas), 'eps': opt.eps,
                              'weight_decay': opt.weight_decay, 'amsgrad': opt.amsgrad},
                'scheduler': {'type': sch.type, 'lr': sch.lr, 'n_iter': sch.n_iter, 'warmup': sch.warmup,
                              'decay': list(sch.decay), 'initial_multiplier': sch.initial_multiplier,
                              'final_multiplier': sch.final_multiplier, 'plateau': sch.plateau},
                'beta_schedule': {k: beta[k] for k in ('schedule', 'n_timestep', 'linear_start', 'linear_end')}}
    s = o['scheduler']
    if (observed['optimizer']['type'] != o['name'].lower() or observed['optimizer']['lr'] != o['learning_rate'] or
            (observed['scheduler']['type'], observed['scheduler']['lr'], observed['scheduler']['n_iter'],
             observed['scheduler']['warmup'], observed['scheduler']['decay']) !=
            (s['type'], s['lr'], s['n_iter'], s['warmup'], s['decay']) or
            observed['beta_schedule'] != {k: d[k] for k in ('schedule', 'n_timestep', 'linear_start', 'linear_end')} or
            o['ema'] != {'decay': 0.9999, 'before_warmup': 0.0}):
        raise PreparationError('pinned config/diffusion.conf differs from the frozen E07c optimizer/scheduler/diffusion')
    return observed
