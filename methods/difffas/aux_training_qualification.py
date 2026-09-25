"""M6D6b E07c auxiliary conditioning-encoder TRAINING-GRAPH qualification, never a training runner.

Run in an isolated GPU process under gpat-m6-e07c. Constructs the source-native
custom_rn "resnet18" encoder through the committed A3 seam (fc -> Linear(512, 7)),
then executes EXACTLY ONE synthetic step in the order of the pinned
models/pretrain_classifier.py:36-41 loop body:

    optimizer.zero_grad(); _,_,_,outputs = model(x); loss = CE(outputs, y);
    loss.backward(); optimizer.step()

with optim.SGD(model.parameters(), lr=0.002, momentum=0.9, weight_decay=5e-3)
(pretrain_classifier.py:27) and nn.CrossEntropyLoss() (:28). Inputs/labels are
analytic in-memory tensors, never benchmark samples. No dataset, manifest,
image reader, scheduler, epoch loop, checkpoint writer or checkpoint loader
exists here. QUALIFICATION_ONLY; the in-memory encoder is discarded at exit.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import warnings

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from methods.difffas import runtime_qualification as rq  # noqa: E402  (M6D6a helpers, unchanged)

SEED = 60602  # qualification_seed only; never the auxiliary (42) or an experiment seed (42/1337/2026)
LABEL = 'AUX_ENCODER_SYNTHETIC_TRAINING_GRAPH_ONLY'
BUILD_PARTS = ('builds', 'e07c_difffas', 'm6d6b')
BATCH = 4                 # qualification_batch_size; NOT the scientific auxiliary batch (256)
SCIENTIFIC_AUX_BATCH = 256
K = 7
INPUT_PHASE = 37          # analytic synthetic image family shared with M6D6a (rq.synthetic), new phase
SYNTHETIC_LABELS = tuple((3 * i + 1) % K for i in range(BATCH))  # (1, 4, 0, 3): class indices only
# Source-native modules defined in ResNet.__init__ but never called by _forward_impl (custom_rn.py:273-274).
SOURCE_UNUSED_MODULES = ('norm', 'in1')
SOURCE_DISCONNECTED_PARAMETERS = ('norm.bias', 'norm.weight')
HYPER = {'lr': 0.002, 'momentum': 0.9, 'weight_decay': 5e-3}
ELIGIBILITY = {'QUALIFICATION_ONLY': True, 'NOT_ELIGIBLE_FOR_BANK': True,
               'NOT_ELIGIBLE_FOR_DOWNSTREAM': True, 'NOT_ELIGIBLE_FOR_REPORTING': True}


def require(value, message):
    if not value:
        raise RuntimeError('M6D6b gate: ' + message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def upstream_training_semantics(source):
    """AST evidence from the pinned auxiliary pretraining script; the script is NEVER imported or run."""
    from methods.difffas.source import tree
    rel = 'models/pretrain_classifier.py'
    mod = tree(source, rel)  # re-verifies the SHA256 recorded by validate_source
    top = {ast.unparse(n): n.lineno for n in mod.body}
    expected_top = ('resnet18 = resnet18()', 'resnet18.fc = nn.Linear(512, 17)', 'resnet18 = resnet18.cuda()',
                    'criteria = nn.CrossEntropyLoss()', 'num_epochs = 200')
    for stmt in expected_top:
        require(stmt in top, 'pinned pretrain_classifier statement: ' + stmt)
    opt = [n for n in mod.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'optimizer']
    require(len(opt) == 1, 'exactly one optimizer assignment')
    call = opt[0].value
    require(ast.unparse(call.func) == 'optim.SGD' and [ast.unparse(a) for a in call.args] ==
            ['resnet18.parameters()'], 'SGD over resnet18.parameters()')
    kw = {k.arg: ast.literal_eval(k.value) for k in call.keywords}
    require(kw == HYPER, 'SGD hyperparameters ' + json.dumps(kw))
    loader = next(n for n in mod.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'train_loader')
    loader_kw = {k.arg: ast.literal_eval(k.value) for k in loader.value.keywords}
    order_lines = [top['resnet18 = resnet18()'], top['resnet18.fc = nn.Linear(512, 17)'],
                   top['resnet18 = resnet18.cuda()'], opt[0].lineno, top['criteria = nn.CrossEntropyLoss()']]
    require(order_lines == sorted(order_lines), 'construct -> fc -> cuda -> SGD -> CE order')
    outer = next(n for n in mod.body if isinstance(n, ast.For))
    inner = next(n for n in outer.body if isinstance(n, ast.For))
    body = [(ast.unparse(n), n.lineno) for n in inner.body]
    step_seq = ['optimizer.zero_grad()', '_, _, _, outputs = resnet18(inputs)',
                'loss_asym = criteria(outputs, labels)', 'loss = loss_asym', 'loss.backward()', 'optimizer.step()']
    found = [next((ln for s, ln in body if s == want), None) for want in step_seq]
    require(None not in found and found == sorted(found), 'zero_grad -> forward -> CE(4th) -> backward -> step')
    calls = [ast.unparse(n.func) for n in ast.walk(mod) if isinstance(n, ast.Call)]
    mode_calls = sorted(c for c in calls if c.endswith(('.train', '.eval')))
    require(not mode_calls, 'pinned pretraining script never calls .train()/.eval()')
    names = {ast.unparse(n) for n in ast.walk(mod) if isinstance(n, (ast.Name, ast.Attribute))}
    precision_tokens = sorted(t for t in ('autocast', 'GradScaler', 'amp', 'allow_tf32', 'half', 'float16',
                                          'bfloat16', 'set_default_dtype', 'cudnn') if any(t in n for n in names))
    require(not precision_tokens, 'no explicit precision control in pinned pretraining script')
    require(not [c for c in calls if 'lr_scheduler' in c or 'clip_grad' in c], 'no scheduler / clipping')
    require(sum(c == 'torch.save' for c in calls) == 1, 'single end-of-epoch whole-module save (not executed here)')
    return {'file': rel, 'sha256': source['files_sha256'][rel], 'imported_or_executed': False,
            'construction': {s: top[s] for s in expected_top[:3]},
            'optimizer': {'line': opt[0].lineno, 'statement': ast.unparse(opt[0]), 'class': 'optim.SGD',
                          'params': 'resnet18.parameters()', 'hyperparameters': kw,
                          'constructed_after_fc_replacement_and_cuda': True},
            'criterion': {'line': top['criteria = nn.CrossEntropyLoss()'], 'statement': 'nn.CrossEntropyLoss()',
                          'arguments': 'NONE (mean reduction, no weight, no label smoothing)'},
            'step_order': dict(zip(step_seq, found)),
            'mode_calls': mode_calls,
            'mode_semantics': ('no .train()/.eval() anywhere; torch.nn.Module defaults to training=True, so '
                               'BatchNorm2d uses batch statistics and updates running stats'),
            'precision_controls': precision_tokens,
            'precision_semantics': ('NONE_EXPLICIT: no autocast, GradScaler, TF32 flag, dtype cast or cudnn flag; '
                                    'FP32 parameters under library defaults'),
            'scheduler': 'NONE', 'gradient_clipping': 'NONE',
            'loader': {'line': loader.lineno, **loader_kw},
            'epochs': 200, 'torch_save_calls_in_script': 1}


class Counters:
    """Counts pass-through training calls; forbids everything outside the one-step contract."""

    def __init__(self, torch):
        self.torch = torch
        self.n = {'optimizer_constructions': 0, 'sgd_constructions': 0, 'optimizer_step_calls': 0,
                  'backward_calls': 0, 'zero_grad_calls': 0, 'autograd_grad_calls': 0,
                  'scheduler_constructions': 0, 'grad_clipping_calls': 0,
                  'checkpoint_saves': 0, 'checkpoint_loads': 0}

    def install(self):
        torch, n = self.torch, self.n
        opt_init, sgd_step, sgd_zero = torch.optim.Optimizer.__init__, torch.optim.SGD.step, torch.optim.SGD.zero_grad
        autograd_backward = torch.autograd.backward

        def forbid(name):
            def forbidden(*args, **kwargs):
                n[name] += 1
                raise RuntimeError('M6D6b forbids ' + name)
            return forbidden

        def init(self_, *args, **kwargs):
            n['optimizer_constructions'] += 1
            require(type(self_) is torch.optim.SGD and n['optimizer_constructions'] == 1, 'exactly one SGD')
            n['sgd_constructions'] += 1
            return opt_init(self_, *args, **kwargs)

        def step(self_, *args, **kwargs):
            n['optimizer_step_calls'] += 1
            require(n['optimizer_step_calls'] == 1 and n['backward_calls'] == 1, 'exactly one step after one backward')
            return sgd_step(self_, *args, **kwargs)

        def zero_grad(self_, *args, **kwargs):
            n['zero_grad_calls'] += 1
            require(n['zero_grad_calls'] == 1 and n['backward_calls'] == 0, 'zero_grad once, before backward')
            return sgd_zero(self_, *args, **kwargs)

        def backward(*args, **kwargs):  # Tensor.backward delegates here
            n['backward_calls'] += 1
            require(n['backward_calls'] == 1, 'exactly one backward')
            return autograd_backward(*args, **kwargs)
        torch.optim.Optimizer.__init__ = init
        torch.optim.SGD.step, torch.optim.SGD.zero_grad = step, zero_grad
        for cls in {c for c in vars(torch.optim).values() if isinstance(c, type) and
                    issubclass(c, torch.optim.Optimizer) and c is not torch.optim.SGD}:
            cls.step = forbid('optimizer_step_calls')
        torch.optim.lr_scheduler.LRScheduler.__init__ = forbid('scheduler_constructions')
        torch.nn.utils.clip_grad_norm_ = forbid('grad_clipping_calls')
        torch.nn.utils.clip_grad_value_ = forbid('grad_clipping_calls')
        torch.autograd.backward = backward
        torch.autograd.grad = forbid('autograd_grad_calls')
        torch.save = forbid('checkpoint_saves')
        torch.load = forbid('checkpoint_loads')


def tensor_stats(t):
    a = t.detach().cpu().contiguous().numpy()
    return {'sha256': sha(a.tobytes()), 'finite': bool(t.isfinite().all().item()),
            'nonzero_elements': int((t != 0).sum().item()), 'numel': t.numel(),
            'max_abs': float(t.detach().abs().max().item()), 'l2': float(t.detach().double().norm().item())}


def qualify(build):
    firewall = rq.Firewall(build)
    sys.addaudithook(firewall)
    source, semantics, adapter, contracts = rq.source_identity()
    upstream = upstream_training_semantics(source)
    static = rq.contract(adapter, contracts)  # frozen-contract checks shared with M6D6a (static)
    tc = contracts['effective_encoder']['training_contract']
    require((tc['optimizer'], tc['learning_rate'], tc['momentum'], tc['weight_decay'], tc['scheduler'], tc['loss']) ==
            ('SGD', HYPER['lr'], HYPER['momentum'], HYPER['weight_decay'], 'NONE',
             'CrossEntropyLoss_on_fourth_forward_output'), 'frozen A3 auxiliary optimizer/loss contract')
    require(upstream['optimizer']['hyperparameters'] == HYPER, 'frozen contract = pinned source hyperparameters')
    require(tc['batch_size'] == SCIENTIFIC_AUX_BATCH == upstream['loader']['batch_size'] and BATCH != tc['batch_size'],
            'qualification batch distinct from the frozen scientific batch')
    aux_seed = contracts['effective_encoder']['auxiliary_encoder_training_seed']
    require(aux_seed == 42 and SEED not in (42, 1337, 2026), 'qualification seed distinct from scientific seeds')
    result = {'milestone': 'M6D6b', 'label': LABEL, 'qualification_seed': SEED,
              'auxiliary_encoder_training_seed': None, 'experiment_seed': None,
              'scientific_seed_consumed': False,
              'seed_note': 'frozen auxiliary_encoder_training_seed 42 is NOT used; 60602 seeds this graph check only',
              'qualification_batch_size': BATCH, 'scientific_aux_batch_size': SCIENTIFIC_AUX_BATCH,
              'batch_256_training_memory_not_qualified': True, **ELIGIBILITY,
              'contract': static, 'upstream_training_semantics': upstream, 'source_before': source,
              'executable_semantics': semantics}
    import numpy as np
    import torch
    import inspect
    rq.TORCH_LOAD_WEIGHTS_ONLY_DEFAULT = repr(inspect.signature(torch.load).parameters['weights_only'].default)
    library_defaults = {'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
                        'cudnn_tf32': torch.backends.cudnn.allow_tf32,
                        'cudnn_benchmark': torch.backends.cudnn.benchmark,
                        'cudnn_deterministic': torch.backends.cudnn.deterministic,
                        'default_dtype': str(torch.get_default_dtype()),
                        'float32_matmul_precision': torch.get_float32_matmul_precision()}
    # Engineering qualification controls identical to M6D6a; NOT a scientific precision policy.
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cudnn = adapter.config['training']['cudnn']
    torch.backends.cudnn.benchmark = cudnn['benchmark']
    torch.backends.cudnn.deterministic = cudnn['deterministic']
    torch.set_float32_matmul_precision('highest')
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    result['precision'] = {
        'label': 'ENGINEERING_QUALIFICATION_CONTROLS', 'freezes_scientific_precision_policy': False,
        'controls': {'dtype': 'float32', 'matmul_tf32': False, 'cudnn_tf32': False,
                     'cudnn_benchmark': cudnn['benchmark'], 'cudnn_deterministic': cudnn['deterministic'],
                     'float32_matmul_precision': 'highest', 'autocast': False, 'grad_scaler': False,
                     'deterministic_algorithms_forced': False},
        'library_defaults_observed_before_controls': library_defaults,
        'pinned_source_policy': upstream['precision_semantics'],
        'open_issue': ('pinned pretrain_classifier.py sets no precision/cudnn flag; under library defaults cuDNN '
                       'convolutions may use TF32 on Ampere. The production auxiliary precision policy still needs '
                       'an explicit decision; this qualification decides nothing.')}
    counters = Counters(torch)
    counters.install()
    before = rq.environment()
    result['environment_before'] = before
    from methods.difffas.encoder import encoder_model
    src_root = Path(source['root'])
    with warnings.catch_warnings(record=True) as caught, encoder_model(adapter.config) as model:
        warnings.simplefilter('always')
        # ------------------------------------------------------------ A. identity (A3/A6, no substitution)
        cls = type(model)
        enc_file = Path(sys.modules[cls.__module__].__file__).resolve()
        require(enc_file == (src_root / 'models/custom_rn.py').resolve(), 'encoder class from pinned custom_rn')
        require(sha(enc_file.read_bytes()) == contracts['overlay']['source']['sha256'], 'custom_rn SHA256 (A6)')
        import torchvision
        require(not isinstance(model, torchvision.models.ResNet) and not cls.__module__.startswith('torchvision'),
                'no torchvision ResNet')
        custom_rn = sys.modules[cls.__module__]
        layers = [len(getattr(model, f'layer{i}')) for i in (1, 2, 3, 4)]
        widths = [getattr(model, f'layer{i}')[-1].conv2.out_channels for i in (1, 2, 3, 4)]
        require(layers == [3, 4, 6, 3] and custom_rn.BasicBlock.expansion == 1 and
                {type(b).__name__ for i in (1, 2, 3, 4) for b in getattr(model, f'layer{i}')} == {'BasicBlock'},
                'pinned [3,4,6,3] BasicBlock topology')
        require(widths == [64, 256, 512, 512], 'pinned stage widths')
        require(type(model.fc) is torch.nn.Linear and (model.fc.in_features, model.fc.out_features) == (512, K) and
                model.fc.bias is not None, 'A3 head Linear(512, 7) with bias')
        with torch.random.fork_rng(devices=[]):
            reference = custom_rn.resnet18(pretrained=False)  # structure only; RNG isolated
        shape = lambda m: {n: (type(x).__name__, [list(p.shape) for p in x.parameters(recurse=False)])
                           for n, x in m.named_modules()}
        ours, ref = shape(model), shape(reference)
        differing = sorted(n for n in set(ours) | set(ref) if ours.get(n) != ref.get(n))
        require(differing == ['fc'], 'only the A3 head differs from pinned resnet18()')
        del reference
        # ------------------------------------------------------------ B. source order: .cuda() -> SGD -> CE
        default_training = {n: m.training for n, m in model.named_modules()}
        require(all(default_training.values()), 'constructed module defaults to training=True')
        model = model.cuda()  # pretrain_classifier.py:17
        model.train()         # idempotent: equals the source default (the script never calls train/eval)
        require(all(m.training for m in model.modules()), 'train mode on every submodule')
        params = list(model.named_parameters())
        optimizer = torch.optim.SGD(model.parameters(), lr=HYPER['lr'], momentum=HYPER['momentum'],
                                    weight_decay=HYPER['weight_decay'])   # pretrain_classifier.py:27
        criterion = torch.nn.CrossEntropyLoss()                             # pretrain_classifier.py:28
        group, = optimizer.param_groups
        require([id(p) for p in group['params']] == [id(p) for _, p in params], 'optimizer set == model.parameters()')
        require(all(p.requires_grad for _, p in params), 'every parameter trainable')
        group_h = {k: group[k] for k in ('lr', 'momentum', 'dampening', 'weight_decay', 'nesterov', 'maximize',
                                         'foreach', 'differentiable', 'fused')}
        require((group_h['lr'], group_h['momentum'], group_h['weight_decay'], group_h['dampening'],
                 group_h['nesterov'], group_h['maximize']) == (0.002, 0.9, 0.005, 0, False, False),
                'exact SGD hyperparameters')
        require((criterion.reduction, criterion.label_smoothing, criterion.weight, criterion.ignore_index) ==
                ('mean', 0.0, None, -100), 'plain CrossEntropyLoss')
        name_of = {id(p): n for n, p in params}
        state0 = rq.state_report(model)
        torch.cuda.synchronize()
        memory = {'after_construction_allocated_bytes': torch.cuda.memory_allocated(),
                  'after_construction_reserved_bytes': torch.cuda.memory_reserved()}
        torch.cuda.reset_peak_memory_stats()
        # ------------------------------------------------------------ C. synthetic batch (no benchmark data)
        x = rq.synthetic(torch, BATCH, INPUT_PHASE).cuda()
        y = torch.tensor(SYNTHETIC_LABELS, dtype=torch.long, device='cuda')
        require(list(x.shape) == [BATCH, 3, 256, 256] and x.dtype == torch.float32 and
                float(x.min()) >= -1 and float(x.max()) <= 1, 'synthetic input [B,3,256,256] in [-1,1]')
        require(y.dtype == torch.long and list(y.shape) == [BATCH] and int(y.min()) >= 0 and int(y.max()) < K,
                'synthetic target LongTensor[B] in [0,6]')
        called, bn_elements = set(), {}

        def seen(name):
            def hook(module, args):
                called.add(name)
                if isinstance(module, torch.nn.BatchNorm2d):
                    bn_elements[name] = args[0].numel() // args[0].shape[1]
            return hook
        hooks = [m.register_forward_pre_hook(seen(n)) for n, m in model.named_modules() if n]
        # ------------------------------------------------------------ D. one step, pinned order (:36-41)
        optimizer.zero_grad()
        require(all(p.grad is None for _, p in params), 'grads None after zero_grad (set_to_none)')
        outputs = model(x)
        for h in hooks:
            h.remove()
        require(len(outputs) == 4, 'four encoder outputs')
        logits = outputs[3]
        require(list(logits.shape) == [BATCH, K] and logits.requires_grad, 'fourth output logits [B,7]')
        feature_shapes = [list(o.shape) for o in outputs[:3]]
        require(feature_shapes == [[BATCH, 256, 32, 32], [BATCH, 512, 16, 16], [BATCH, 512, 8, 8]], 'A6 features')
        loss = criterion(logits, y)
        with torch.no_grad():
            reference_ce = -torch.log_softmax(logits.double(), 1)[torch.arange(BATCH, device='cuda'), y].mean()
        loss_value = float(loss.item())
        require(torch.isfinite(loss).item() and loss.dim() == 0, 'finite scalar CE')
        ce_abs_dev = abs(loss_value - float(reference_ce))
        require(ce_abs_dev < 1e-5, 'CE equals float64 -mean log_softmax[y]')
        uncalled = sorted({n for n, _ in model.named_modules() if n} - called)
        require(uncalled == sorted(SOURCE_UNUSED_MODULES), 'only source-native unused modules skipped: ' + str(uncalled))
        require(bn_elements and min(bn_elements.values()) > 1, 'BatchNorm2d has >1 value per channel at B=4')
        buffers_pre_backward = rq.state_report(model)['buffer_sha256']
        loss.backward()
        # ------------------------------------------------------------ E. gradients, BEFORE step
        grads = {}
        for n, p in params:
            grads[n] = None if p.grad is None else tensor_stats(p.grad)
        none_grad = sorted(n for n, g in grads.items() if g is None)
        require(none_grad == sorted(SOURCE_DISCONNECTED_PARAMETERS),
                'disconnected trainable parameters must be exactly the unused source norm: ' + str(none_grad))
        live = {n: g for n, g in grads.items() if g is not None}
        require(all(g['finite'] for g in live.values()), 'all gradients finite')
        nonzero = sorted(n for n, g in live.items() if g['nonzero_elements'])
        require(grads['fc.weight']['nonzero_elements'] > 0 and grads['fc.bias']['finite'] and
                grads['fc.bias']['nonzero_elements'] > 0, 'fc gradients exist and are non-zero')
        for stage in ('conv1.weight', 'layer1.0.conv1.weight', 'layer2.0.conv1.weight', 'layer3.0.conv1.weight',
                      'layer4.0.conv1.weight'):
            require(grads[stage]['nonzero_elements'] > 0, 'backbone gradient reaches ' + stage)
        total_norm = sum(g['l2'] ** 2 for g in live.values()) ** 0.5
        pre = {n: p.detach().clone() for n, p in params}
        pre_grad = {n: p.grad.detach().clone() for n, p in params if p.grad is not None}
        # ------------------------------------------------------------ F. exactly one optimizer step
        optimizer.step()
        torch.cuda.synchronize()
        peak = {'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                'peak_reserved_bytes': torch.cuda.max_memory_reserved()}
        post = rq.state_report(model)
        changed, unchanged, max_dev = [], [], 0.0
        for n, p in params:
            require(torch.isfinite(p).all().item(), 'finite updated parameter ' + n)
            (changed if not torch.equal(p, pre[n]) else unchanged).append(n)
            if n in pre_grad:  # first SGD step: buf = d_p = g + wd * p; p <- p - lr * buf (dampening 0, no nesterov)
                expected = pre[n] - HYPER['lr'] * (pre_grad[n] + HYPER['weight_decay'] * pre[n])
                max_dev = max(max_dev, float((p - expected).abs().max().item()))
        require(max_dev <= 1e-6, 'update matches first-step SGD formula')
        require('fc.weight' in changed and 'fc.bias' in changed, 'fc parameters updated')
        require(any(not n.startswith('fc.') for n in changed), 'backbone parameters updated')
        require(all(n in unchanged for n in SOURCE_DISCONNECTED_PARAMETERS), 'None-grad parameters untouched')
        require(set(changed) <= set(pre_grad), 'only parameters with gradients changed')
        buffers_changed_by_step = sorted(k for k in buffers_pre_backward
                                         if buffers_pre_backward[k] != post['buffer_sha256'][k])
        require(not buffers_changed_by_step, 'backward/step change no buffer')
        buffers_changed_by_forward = sorted(k for k in state0['buffer_sha256']
                                            if state0['buffer_sha256'][k] != buffers_pre_backward[k])
        require(all(k.endswith(('running_mean', 'running_var', 'num_batches_tracked'))
                    for k in buffers_changed_by_forward) and
                not any(k.startswith('norm.') for k in buffers_changed_by_forward), 'native BN train-mode stats only')
        momentum = sorted(name_of[id(p)] for p, s in optimizer.state.items() if 'momentum_buffer' in s)
        require(momentum == sorted(pre_grad), 'momentum buffers exactly for parameters with gradients')
        del pre, pre_grad
        result['encoder'] = {'class': cls.__name__, 'module': cls.__module__,
            'factory': 'models/custom_rn.py::resnet18(pretrained=False)',
            'construction_seam': 'methods/difffas/encoder.py::encoder_model (committed A3 seam)',
            'file_sha256': sha(enc_file.read_bytes()), 'topology': layers, 'block': 'BasicBlock',
            'stage_widths': widths, 'head': {'type': 'Linear', 'in_features': 512, 'out_features': K, 'bias': True},
            'structural_difference_vs_pinned_resnet18': differing, 'torchvision_substitute': False,
            'projection_or_adapter_modules_added': False, 'weights_loaded': False,
            'mode': {'default_training_all_modules': True, 'model_train_called': True, 'eval_called': False,
                     'source_evidence': 'pretrain_classifier.py has no .train()/.eval(); nn.Module default is train'},
            'batchnorm': {'mode': 'train (batch statistics; running stats updated)', 'batchnorm2d_layers':
                          len(bn_elements), 'min_values_per_channel_at_B4': min(bn_elements.values()),
                          'frozen_bn_or_running_stat_workaround': False},
            'source_unused_modules': list(SOURCE_UNUSED_MODULES),
            'source_unused_note': ('custom_rn.py:273 self.norm = nn.BatchNorm1d(18) and :274 self.in1 = '
                                   'nn.InstanceNorm2d(64) are built but never called by _forward_impl; norm.weight/'
                                   'norm.bias are inside resnet18.parameters() (the pinned SGD set) and receive no '
                                   'gradient, exactly as in the pinned source'),
            'state_initial': state0}
        result['synthetic_batch'] = {
            'label': 'SYNTHETIC_IN_MEMORY_NOT_BENCHMARK_SAMPLES', 'input_generator': 'rq.synthetic analytic '
            'sin/cos family (no RNG, no file)', 'input_phase': INPUT_PHASE,
            'input': rq.summarize(x), 'target_values': list(SYNTHETIC_LABELS),
            'target_rule': '(3*i + 1) % 7 for i in range(4): class INDICES only, no class names attached',
            'target_dtype': str(y.dtype), 'target_sha256': sha(y.cpu().numpy().tobytes())}
        result['forward'] = {'feature_shapes': feature_shapes, 'logits': rq.summarize(logits),
                             'logits_shape': list(logits.shape), 'loss_attached_to': 'outputs[3] only',
                             'called_modules': len(called)}
        result['loss'] = {'criterion': 'torch.nn.CrossEntropyLoss()', 'reduction': 'mean', 'label_smoothing': 0.0,
                          'weight': None, 'value': loss_value, 'finite': True,
                          'value_float32_hex': float(loss.detach().cpu()).hex(),
                          'float64_reference': float(reference_ce), 'abs_deviation_from_reference': ce_abs_dev}
        result['optimizer'] = {'class': 'torch.optim.SGD', 'param_groups': 1, 'hyperparameters': group_h,
            'scheduler': None, 'nesterov': False, 'grad_clipping': None,
            'optimized_tensors': len(params), 'optimized_parameters': sum(p.numel() for _, p in params),
            'param_set': 'model.parameters() (identity-verified, whole network incl. fc and source norm)',
            'momentum_buffers_after_step': len(momentum)}
        result['gradients'] = {'checked': 'after backward, before optimizer.step',
            'tensors_with_grad': len(live), 'tensors_with_nonzero_grad': len(nonzero),
            'tensors_with_all_zero_grad': sorted(set(live) - set(nonzero)),
            'none_grad_tensors': none_grad, 'none_grad_classification': 'SOURCE_NATIVE_UNUSED_MODULE',
            'unexpected_disconnected_tensors': [], 'all_finite': True,
            'max_abs': max(g['max_abs'] for g in live.values()), 'global_l2': total_norm,
            'fc_weight': grads['fc.weight'], 'fc_bias': grads['fc.bias'], 'stem_conv1': grads['conv1.weight'],
            'per_tensor': grads}
        result['parameter_update'] = {'changed_tensors': len(changed), 'unchanged_tensors': sorted(unchanged),
            'changed_parameters_outside_optimizer_scope': 0, 'fc_changed': True,
            'backbone_changed_tensors': sum(not n.startswith('fc.') for n in changed), 'all_finite_after': True,
            'sgd_first_step_formula_max_abs_deviation': max_dev, 'sgd_formula_tolerance': 1e-6,
            'aggregate_parameter_sha256_before': state0['aggregate_parameter_sha256'],
            'aggregate_parameter_sha256_after': post['aggregate_parameter_sha256'],
            'parameter_sha256_after': post['parameter_sha256'],
            'buffers_changed_by_train_mode_forward': buffers_changed_by_forward,
            'buffers_changed_by_backward_or_step': buffers_changed_by_step}
        result['cuda_memory'] = {'label': 'AUX_ENCODER_SYNTHETIC_TRAINING_STEP_MEMORY_OBSERVATION',
            'batch_size': BATCH, **memory, 'one_step': peak, 'qualifies_training_memory': False,
            'qualifies_scientific_batch_size': False,
            'statement': 'B256 scientific training memory remains unqualified'}
        del optimizer, outputs, logits, loss, x, y
    result['warnings'] = sorted({f'{w.category.__name__}: {w.message}' for w in caught})
    result['source_after'] = rq.source_identity()[0]
    require(result['source_after'] == source and source['worktree_status'] == '', 'pinned source unchanged, clean')
    result['environment_after'] = rq.environment()
    require(before == result['environment_after'], 'environment stable')
    n = counters.n
    require(n == {'optimizer_constructions': 1, 'sgd_constructions': 1, 'optimizer_step_calls': 1,
                  'backward_calls': 1, 'zero_grad_calls': 1, 'autograd_grad_calls': 0,
                  'scheduler_constructions': 0, 'grad_clipping_calls': 0,
                  'checkpoint_saves': 0, 'checkpoint_loads': 0}, 'exact counters ' + json.dumps(n))
    require(not firewall.denied and not any(firewall.attempts.values()), 'firewall: zero denied accesses')
    require(all(Path(a[0]).name in rq.ALLOWED_SUBPROCESSES or a in rq.ALLOWED_EXACT_ARGV
                for a in firewall.subprocesses), 'subprocess allowlist')
    result.update(status='PASS', counters=n, backward_calls=1, optimizer_step_calls=1, epochs=0,
        checkpoint_created=False, checkpoint_loaded=False, auxiliary_encoder_trained=False,
        scientific_training=False, main_difffas_model_constructed=False, main_difffas_training=False,
        benchmark_data_access=False, TRAIN_access=False, VAL_access=False, TEST_access=False,
        synthetic_bank=False, fidelity='CONTROLLED_ADAPTATION', compatibility_patch='NONE',
        firewall={'denied': firewall.denied, 'attempts': firewall.attempts, 'event_counts': firewall.events,
                  'subprocesses': firewall.subprocesses})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-root', type=Path, required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    build = args.build_root.resolve()
    require(not build.is_relative_to(ROOT), 'build root outside repository')
    require(build.parts[-3:] == BUILD_PARTS and build.is_relative_to(rq.RUNTIME), 'dedicated M6D6b build path')
    require(Path(args.output).name == args.output and args.output.endswith('.json'), 'evidence filename')
    require(not (build / args.output).exists(), 'fresh output name')
    build.mkdir(parents=True, exist_ok=True)
    tmp = Path(os.environ.get('TMPDIR', '/'))
    require(tmp.is_dir() and tmp.resolve().is_relative_to(build), 'TMPDIR inside the dedicated build root')
    result = qualify(build)
    (build / args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'status': result['status'], 'output': str(build / args.output)}))


if __name__ == '__main__':
    main()
