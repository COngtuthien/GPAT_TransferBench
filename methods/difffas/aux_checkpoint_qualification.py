"""M6D6c E07c auxiliary-encoder WHOLE-MODULE checkpoint serialization/loader qualification, never training.

Run in isolated GPU processes under gpat-m6-e07c, one role per process:

  --role writer  constructs the source-native custom_rn "resnet18" (A3 seam, fc ->
                 Linear(512, 7)) under qualification_seed 60603, moves it to CUDA
                 (pretrain_classifier.py:17) and writes ONE qualification-only
                 checkpoint with torch.save(model, path) (pretrain_classifier.py:45)
                 through methods/difffas/aux_checkpoint.py, then records a
                 synthetic eval forward reference.
  --role loader  verifies the recorded SHA256, proves wrong-SHA rejection BEFORE
                 torch.load, observes torch.load(path) / weights_only=True /
                 weights_only=False, then loads through the SHA-gated seam and
                 checks identity, parameters, buffers and bitwise forward equality.

No dataset, manifest, image reader, optimizer, backward pass or training exists
here. The checkpoint lives only under <runtime_root>/builds/e07c_difffas/m6d6c/
and is QUALIFICATION_ONLY / NOT_A_SCIENTIFIC_CHECKPOINT; it is deleted after
the loader processes.
"""
import argparse
import ast
import contextlib
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

SEED = 60603  # qualification_seed only; never the auxiliary (42) or an experiment seed (42/1337/2026)
LABEL = 'AUX_CHECKPOINT_SERIALIZATION_QUALIFICATION_ONLY'
BUILD_PARTS = ('builds', 'e07c_difffas', 'm6d6c')
CHECKPOINT_RELATIVE = ('checkpoint', 'e07c_aux_encoder_QUALIFICATION_ONLY.pkl')
BATCH = 4
K = 7
INPUT_PHASE = 53  # analytic synthetic image family shared with M6D6a/b (rq.synthetic), new phase
OUTPUTS = ('x32x32', 'x16x16', 'x8x8', 'logits')
OUTPUT_SHAPES = {'x32x32': [BATCH, 256, 32, 32], 'x16x16': [BATCH, 512, 16, 16], 'x8x8': [BATCH, 512, 8, 8],
                 'logits': [BATCH, K]}
ELIGIBILITY = {'QUALIFICATION_ONLY': True, 'NOT_ELIGIBLE_FOR_BANK': True, 'NOT_ELIGIBLE_FOR_DOWNSTREAM': True,
               'NOT_ELIGIBLE_FOR_REPORTING': True, 'NOT_A_SCIENTIFIC_CHECKPOINT': True}
# (name, torch.load keyword arguments, pinned custom_rn importable during the call)
PROBES = (('default_no_argument', {}, True), ('weights_only_true', {'weights_only': True}, True),
          ('weights_only_false', {'weights_only': False}, True),
          ('weights_only_false_without_pinned_module', {'weights_only': False}, False))
PRECISION_POLICY = 'PRECISION_POLICY_DECISION_REQUIRED'


def require(value, message):
    if not value:
        raise RuntimeError('M6D6c gate: ' + message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class CheckpointFirewall(rq.Firewall):
    """M6D6a firewall plus exactly one exception: the qualification checkpoint path.

    The writer may create it; loaders may only read it. Every other .pkl/.pth,
    runtime data, runs root, manifest or image stays denied.
    """

    def __init__(self, build, checkpoint, writable):
        super().__init__(build)
        self.checkpoint, self.writable = str(checkpoint), writable
        self.checkpoint_opens = {'read': 0, 'write': 0}

    def classify(self, event, args):
        path, hits = super().classify(event, args)
        if path == self.checkpoint and hits == ['blocked_weight_or_array']:
            flags = args[2] if event == 'open' and len(args) > 2 and isinstance(args[2], int) else 0
            mode = args[1] if event == 'open' and len(args) > 1 and isinstance(args[1], str) else ''
            writing = flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND) or set(mode) & set('wax+')
            if writing and not self.writable:
                return path, hits
            self.checkpoint_opens['write' if writing else 'read'] += 1
            return path, []
        return path, hits


class Trace:
    """Ordered audit trace: checkpoint opens, SHA verdict events, torch.save/load calls, pickle.find_class."""

    def __init__(self, checkpoint):
        from methods.difffas.aux_checkpoint import EVENT_REJECTED, EVENT_VERIFIED
        self.checkpoint, self.events = str(checkpoint), []
        self.names = {EVENT_VERIFIED: 'sha256_verified', EVENT_REJECTED: 'sha256_rejected'}

    def __call__(self, event, args):
        if event in self.names:
            self.events.append([self.names[event], args[1]])
        elif event == 'pickle.find_class':
            self.events.append(['find_class', f'{args[0]}.{args[1]}'])
        elif event == 'open' and args and isinstance(args[0], (str, bytes)) and \
                os.path.abspath(os.fsdecode(args[0])) == self.checkpoint:
            self.events.append(['open_checkpoint', args[1] if isinstance(args[1], str) else 'flags'])

    def take(self):
        out, self.events = self.events, []
        return out


class Counters:
    """Forbids optimizer/backward; counts QUALIFICATION_ONLY torch.save (writer) / torch.load (loader)."""

    def __init__(self, torch, trace, role):
        self.torch, self.trace, self.role = torch, trace, role
        self.n = {'optimizer_constructions': 0, 'optimizer_step_calls': 0, 'backward_calls': 0,
                  'autograd_grad_calls': 0, 'torch_save_calls': 0, 'torch_load_calls': 0}

    def install(self):
        torch, n, trace = self.torch, self.n, self.trace
        save, load = torch.save, torch.load

        def forbid(name):
            def forbidden(*args, **kwargs):
                n[name] += 1
                raise RuntimeError('M6D6c forbids ' + name)
            return forbidden

        def counted_save(obj, f, *args, **kwargs):
            n['torch_save_calls'] += 1
            trace.events.append(['torch.save', {'object': f'{type(obj).__module__}.{type(obj).__qualname__}',
                                                'extra_arguments': bool(args or kwargs)}])
            require(n['torch_save_calls'] == 1 and not args and not kwargs, 'exactly one plain torch.save(model, path)')
            return save(obj, f)

        def counted_load(f, *args, **kwargs):
            n['torch_load_calls'] += 1
            trace.events.append(['torch.load', {'source': 'path' if isinstance(f, (str, os.PathLike)) else 'bytes',
                                                'kwargs': {k: repr(v) for k, v in sorted(kwargs.items())}}])
            require(not args, 'keyword-only torch.load options')
            return load(f, **kwargs)
        torch.optim.Optimizer.__init__ = forbid('optimizer_constructions')
        for cls in {c for c in vars(torch.optim).values() if isinstance(c, type) and
                    issubclass(c, torch.optim.Optimizer)}:
            cls.step = forbid('optimizer_step_calls')
        torch.Tensor.backward = forbid('backward_calls')
        torch.autograd.backward = forbid('backward_calls')
        torch.autograd.grad = forbid('autograd_grad_calls')
        torch.save = counted_save if self.role == 'writer' else forbid('torch_save_calls')
        torch.load = counted_load if self.role == 'loader' else forbid('torch_load_calls')


def upstream_serialization_semantics(source):
    """AST evidence of the pinned writer/consumer pair; nothing is imported or executed."""
    from methods.difffas.source import tree
    writer_rel, consumer_rel, train_rel = 'models/pretrain_classifier.py', 'models/unet_autoenc.py', 'FAS_train.py'
    writer = tree(source, writer_rel)
    top = {ast.unparse(n): n.lineno for n in writer.body}
    require(top.get('resnet18 = resnet18.cuda()') == 17, 'writer moves the encoder to CUDA (:17)')
    outer = next(n for n in writer.body if isinstance(n, ast.For))
    saves = [n for n in ast.walk(writer) if isinstance(n, ast.Call) and ast.unparse(n.func) == 'torch.save']
    require(len(saves) == 1, 'exactly one torch.save in the pinned writer')
    save = saves[0]
    save_stmt = next(s for s in outer.body if save in list(ast.walk(s)))
    require(ast.unparse(save) == "torch.save(resnet18, './PADISI.pkl')" and not save.keywords and
            isinstance(save.args[0], ast.Name), 'writer saves the WHOLE module object (no state_dict)')
    require(save.lineno == 45 and top['resnet18 = resnet18.cuda()'] < save.lineno, 'save after .cuda(), line 45')
    consumer = tree(source, consumer_rel)
    cls = next(n for n in consumer.body if isinstance(n, ast.ClassDef) and n.name == 'BeatGANsAutoencModel')
    enc = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'encoder')
    body = [(ast.unparse(s), s.lineno) for s in enc.body]
    require([s for s, _ in body] == ['model_autoencoder = resnet18()', 'model_autoencoder = torch.load(path)',
                                     'model_autoencoder = model_autoencoder.cuda()', 'return model_autoencoder'],
            'consumer encoder(path) body')
    load = next(n for n in ast.walk(enc) if isinstance(n, ast.Call) and ast.unparse(n.func) == 'torch.load')
    require(not load.keywords and len(load.args) == 1, 'consumer torch.load(path) has no weights_only/map_location')
    imports = [ast.unparse(n) for n in consumer.body if isinstance(n, ast.ImportFrom) and n.module == 'custom_rn']
    train = tree(source, train_rel)
    fn = next(n for n in train.body if isinstance(n, ast.FunctionDef) and n.name == 'train')
    stmts = {ast.unparse(s): s.lineno for s in fn.body}
    require(stmts.get('encoder = model.encoder(args.pretrain_classifier)') == 34 and stmts.get('encoder.eval()') == 35,
            'FAS_train.py:34-35 loads then eval()s the encoder')
    return {'writer': {'file': writer_rel, 'sha256': source['files_sha256'][writer_rel], 'statement': ast.unparse(save),
                       'line': save.lineno, 'inside_epoch_loop': save_stmt is not None, 'saved_object': 'resnet18 (whole nn.Module)',
                       'cuda_before_save_line': top['resnet18 = resnet18.cuda()'], 'writer_import': 'from custom_rn import resnet18',
                       'pickled_class_module_name': 'custom_rn'},
            'consumer': {'file': consumer_rel, 'sha256': source['files_sha256'][consumer_rel],
                         'method': 'BeatGANsAutoencModel.encoder', 'lines': {s: ln for s, ln in body},
                         'torch_load_arguments': 'path only (no weights_only, no map_location)',
                         'consumer_module_import': imports},
            'caller': {'file': train_rel, 'sha256': source['files_sha256'][train_rel],
                       'statements': {'encoder = model.encoder(args.pretrain_classifier)': 34, 'encoder.eval()': 35}},
            'state_dict_anywhere_in_writer_or_consumer_encoder': False, 'imported_or_executed': False}


def archive_report(path):
    """Opcode disassembly of the zip archive's data.pkl (pickletools.genops): lists globals, never unpickles."""
    import pickletools
    import zipfile
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        data = [n for n in names if n.endswith('/data.pkl')]
        require(len(data) == 1, 'one data.pkl record')
        raw = z.read(data[0])
    found, strings, protocol = set(), [], None
    for op, arg, _ in pickletools.genops(raw):
        if op.name == 'PROTO':
            protocol = arg
        elif op.name in ('SHORT_BINUNICODE', 'BINUNICODE', 'UNICODE'):
            strings.append(arg)
        elif op.name == 'GLOBAL':
            found.add(arg.replace(' ', '.'))
        elif op.name == 'STACK_GLOBAL':
            found.add(f'{strings[-2]}.{strings[-1]}')
    prefix = data[0].rsplit('/', 1)[0] + '/'
    return {'format': 'torch zip archive', 'records': len(names), 'archive_prefix': prefix,
            'record_kinds': sorted({n[len(prefix):].split('/')[0] for n in names}),
            'storage_records': sum(n.startswith(prefix + 'data/') for n in names),
            'data_pkl_sha256': sha(raw), 'pickle_protocol': protocol, 'globals': sorted(found),
            'method': 'pickletools.genops opcode listing; no object constructed'}


def precision_setup(torch, adapter):
    library_defaults = {'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
                        'cudnn_tf32': torch.backends.cudnn.allow_tf32,
                        'cudnn_benchmark': torch.backends.cudnn.benchmark,
                        'cudnn_deterministic': torch.backends.cudnn.deterministic,
                        'default_dtype': str(torch.get_default_dtype()),
                        'float32_matmul_precision': torch.get_float32_matmul_precision()}
    # Engineering qualification controls identical to M6D6a/M6D6b; NOT a scientific precision policy.
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cudnn = adapter.config['training']['cudnn']
    torch.backends.cudnn.benchmark = cudnn['benchmark']
    torch.backends.cudnn.deterministic = cudnn['deterministic']
    torch.set_float32_matmul_precision('highest')
    return {'label': 'ENGINEERING_QUALIFICATION_CONTROLS', 'freezes_scientific_precision_policy': False,
            'controls': {'dtype': 'float32', 'matmul_tf32': False, 'cudnn_tf32': False,
                         'cudnn_benchmark': cudnn['benchmark'], 'cudnn_deterministic': cudnn['deterministic'],
                         'float32_matmul_precision': 'highest', 'autocast': False, 'grad_scaler': False,
                         'deterministic_algorithms_forced': False},
            'library_defaults_observed_before_controls': library_defaults,
            'production_policy': PRECISION_POLICY,
            'production_policy_note': ('no E07c authority (spec 8.7, A1/A2/A3/A6, frozen configs, pinned source) fixes '
                                       'the auxiliary/main precision; serialization qualification does not need it')}


def common(role, build, checkpoint):
    firewall = CheckpointFirewall(build, checkpoint, writable=role == 'writer')
    sys.addaudithook(firewall)
    trace = Trace(checkpoint)
    sys.addaudithook(trace)
    source, semantics, adapter, contracts = rq.source_identity()
    upstream = upstream_serialization_semantics(source)
    static = rq.contract(adapter, contracts)
    from methods.difffas.aux_checkpoint import FROZEN_FORMAT
    require(adapter.config['conditioning_encoder']['checkpoint_format'] == FROZEN_FORMAT, 'frozen whole-module format')
    require(contracts['effective_encoder']['auxiliary_encoder_training_seed'] == 42 and
            SEED not in (42, 1337, 2026), 'qualification seed distinct from scientific seeds')
    import inspect
    import numpy as np
    import torch
    signature = inspect.signature(torch.load)  # genuine signature, captured before Counters wraps torch.load
    parameter = signature.parameters['weights_only']
    rq.TORCH_LOAD_WEIGHTS_ONLY_DEFAULT = repr(parameter.default)
    precision = precision_setup(torch, adapter)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    counters = Counters(torch, trace, role)
    counters.install()
    result = {'milestone': 'M6D6c', 'role': role, 'label': LABEL, 'qualification_seed': SEED,
              'auxiliary_encoder_training_seed': None, 'experiment_seed': None, 'scientific_seed_consumed': False,
              'seed_note': 'frozen auxiliary_encoder_training_seed 42 is NOT used; 60603 seeds this check only',
              **ELIGIBILITY, 'checkpoint_path': '<runtime_root>/builds/e07c_difffas/m6d6c/' + '/'.join(CHECKPOINT_RELATIVE),
              'checkpoint_format_authority': {
                  'config': 'configs/methods/e07c_difffas_bin_idfree.yaml conditioning_encoder.checkpoint_format',
                  'value': adapter.config['conditioning_encoder']['checkpoint_format'],
                  'amendment': 'A3 5.2: torch.save(model) of the WHOLE nn.Module; A6 leaves it unchanged'},
              'contract': static, 'upstream_serialization_semantics': upstream, 'source_before': source,
              'executable_semantics': semantics, 'precision': precision,
              'torch_load_signature': str(signature),
              'torch_load_weights_only_default': repr(parameter.default),
              'torch_load_env_overrides': {k: os.environ.get(k) for k in ('TORCH_FORCE_WEIGHTS_ONLY_LOAD',
                                                                           'TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD')}}
    return firewall, trace, counters, adapter, contracts, source, result, torch


def finish(result, firewall, counters, source, expected_counters):
    result['source_after'] = rq.source_identity()[0]
    require(result['source_after'] == source and source['worktree_status'] == '', 'pinned source unchanged, clean')
    result['environment_after'] = rq.environment()
    require(result['environment_before'] == result['environment_after'], 'environment stable')
    require(counters.n == expected_counters, 'exact counters ' + json.dumps(counters.n))
    require(not firewall.denied and not any(firewall.attempts.values()), 'firewall: zero denied accesses')
    require(all(Path(a[0]).name in rq.ALLOWED_SUBPROCESSES or a in rq.ALLOWED_EXACT_ARGV
                for a in firewall.subprocesses), 'subprocess allowlist')
    result.update(status='PASS', counters=counters.n, backward_calls=0, optimizer_step_calls=0, epochs=0,
        auxiliary_encoder_trained=False, scientific_training=False, scientific_checkpoint_created=False,
        scientific_checkpoint_loaded=False, main_difffas_model_constructed=False, main_difffas_training=False,
        benchmark_data_access=False, TRAIN_access=False, VAL_access=False, TEST_access=False, synthetic_bank=False,
        fidelity='CONTROLLED_ADAPTATION', source_patch='NONE',
        firewall={'denied': firewall.denied, 'attempts': firewall.attempts, 'event_counts': firewall.events,
                  'checkpoint_opens': firewall.checkpoint_opens, 'subprocesses': firewall.subprocesses})
    return result


def module_types(model):
    return {n: f'{type(m).__module__}.{type(m).__qualname__}' for n, m in model.named_modules()}


def reference_forward(torch, model):
    x = rq.synthetic(torch, BATCH, INPUT_PHASE).cuda()
    with torch.no_grad():
        outs = model(x)
    require(len(outs) == 4, 'four encoder outputs')
    out = {}
    for name, t in zip(OUTPUTS, outs):
        require(list(t.shape) == OUTPUT_SHAPES[name], f'{name} shape {list(t.shape)}')
        out[name] = rq.summarize(t)
    return rq.summarize(x), out, outs


def writer(build, checkpoint):
    firewall, trace, counters, adapter, contracts, source, result, torch = common('writer', build, checkpoint)
    from methods.difffas.aux_checkpoint import encoder_identity, save_whole_module
    from methods.difffas.encoder import encoder_model
    result['environment_before'] = rq.environment()
    require(not checkpoint.exists(), 'fresh qualification checkpoint path')
    with warnings.catch_warnings(record=True) as caught, encoder_model(adapter.config) as model:
        warnings.simplefilter('always')
        custom_rn = sys.modules[type(model).__module__]
        identity = encoder_identity(model, custom_rn, contracts)
        import torchvision
        require(not isinstance(model, torchvision.models.ResNet), 'no torchvision ResNet')
        model = model.cuda()  # pretrain_classifier.py:17 (the pinned writer serializes the CUDA model)
        training = sorted({m.training for m in model.modules()})
        require(training == [True], 'source default train mode at save time (writer never calls eval)')
        state = rq.state_report(model)
        require(state['devices'] == ['cuda:0'] and state['dtypes'] == ['torch.float32'], 'CUDA FP32 parameters')
        types = module_types(model)
        checkpoint.parent.mkdir(parents=True, exist_ok=False)
        trace.take()
        saved = save_whole_module(model, checkpoint, adapter.config)
        save_trace = trace.take()
        require([e[0] for e in save_trace if e[0] != 'open_checkpoint'] == ['torch.save'], 'one torch.save call')
        require(rq.state_report(model) == state, 'save does not mutate the model')
        archive = archive_report(checkpoint)
        require({'custom_rn.ResNet', 'custom_rn.BasicBlock'} <= set(archive['globals']), 'pickle names custom_rn classes')
        require(not [g for g in archive['globals'] if g.startswith('torchvision')], 'no torchvision class in pickle')
        model.eval()  # reference forward in the consumer's mode (FAS_train.py:35), after the save
        x_summary, forward, _ = reference_forward(torch, model)
        require(rq.state_report(model) == state, 'eval forward leaves parameters and buffers unchanged')
        result['writer'] = {'construction_seam': 'methods/difffas/encoder.py::encoder_model (committed A3 seam)',
            'serialization_seam': 'methods/difffas/aux_checkpoint.py::save_whole_module',
            'identity': identity, 'custom_rn_file': str(Path(custom_rn.__file__).resolve().relative_to(ROOT)),
            'torchvision_substitute': False, 'projection_or_adapter_modules_added': False, 'weights_loaded': False,
            'training_flags_at_save': training, 'module_types': types, 'state_before_save': state,
            'fc': {'in_features': model.fc.in_features, 'out_features': model.fc.out_features,
                   'weight_shape': list(model.fc.weight.shape), 'bias_shape': list(model.fc.bias.shape)},
            'save_trace': save_trace}
        result['checkpoint'] = {**{k: saved[k] for k in ('size_bytes', 'sha256', 'serialization_api', 'format')},
                                'path': result['checkpoint_path'], 'archive': archive, **ELIGIBILITY}
        result['reference_forward'] = {'mode': 'eval (FAS_train.py:35), torch.no_grad', 'input': x_summary,
                                       'input_phase': INPUT_PHASE, 'outputs': forward,
                                       'input_label': 'SYNTHETIC_IN_MEMORY_NOT_BENCHMARK_SAMPLES'}
        del model
    result['warnings'] = sorted({f'{w.category.__name__}: {w.message}' for w in caught})
    return finish(result, firewall, counters, source, {'optimizer_constructions': 0, 'optimizer_step_calls': 0,
                  'backward_calls': 0, 'autograd_grad_calls': 0, 'torch_save_calls': 1, 'torch_load_calls': 0})


def outcome_of(error):
    text = str(error)
    if 'Weights only load failed' in text and 'custom_rn.ResNet' in text:
        category = 'WEIGHTS_ONLY_UNPICKLER_REJECTS_CUSTOM_MODULE_GLOBAL'
    elif isinstance(error, ModuleNotFoundError) and "'custom_rn'" in text:
        category = 'PINNED_CUSTOM_RN_MODULE_NOT_IMPORTABLE'
    else:
        category = 'OTHER'
    return {'outcome': 'REJECTED', 'exception_class': f'{type(error).__module__}.{type(error).__qualname__}',
            'category': category, 'message_head': text.strip().splitlines()[0][:240], 'message_sha256': sha(text.encode())}


def loader(build, checkpoint, expected, writer_name):
    firewall, trace, counters, adapter, contracts, source, result, torch = common('loader', build, checkpoint)
    from methods.common.config import sha256_file
    from methods.common.learned import PreparationError
    from methods.common.upstream import upstream_modules
    from methods.difffas.aux_checkpoint import LOAD_COMPATIBILITY, encoder_identity, load_verified_whole_module
    written = json.loads((build / writer_name).read_text())
    require(written['status'] == 'PASS' and written['checkpoint']['sha256'] == expected, 'expected SHA256 = writer record')
    result['environment_before'] = rq.environment()
    require(checkpoint.stat().st_size == written['checkpoint']['size_bytes'] and sha256_file(checkpoint) == expected,
            'checkpoint bytes unchanged since the writer')
    trace.take()
    # ------------------------------------------------------------ A. rejection BEFORE torch.load
    rejections = {}
    wrong = expected[:-1] + ('1' if expected[-1] == '0' else '0')
    for name, value in (('wrong_sha256_one_hex_digit', wrong), ('malformed_sha256', 'not-a-recorded-sha256')):
        loads = counters.n['torch_load_calls']
        try:
            with load_verified_whole_module(checkpoint, value, adapter.config):
                require(False, 'wrong SHA256 must not yield an encoder')
        except PreparationError as error:
            message = str(error)
        events = trace.take()
        require(counters.n['torch_load_calls'] == loads, name + ': torch.load never reached')
        require(not [e for e in events if e[0] in ('find_class', 'torch.load', 'sha256_verified')], name + ': no unpickle')
        rejections[name] = {'exception': 'PreparationError', 'message': message, 'torch_load_calls': 0,
                            'find_class_events': 0, 'events': [e[0] for e in events]}
    require(rejections['wrong_sha256_one_hex_digit']['events'] == ['open_checkpoint', 'sha256_rejected'],
            'wrong SHA: bytes hashed, rejected, nothing else')
    require(rejections['malformed_sha256']['events'] == [], 'malformed SHA: rejected before the file is opened')
    # ------------------------------------------------------------ B. torch.load behaviour probes (verified bytes)
    models_root = Path(source['root']) / 'models'
    probes = {}
    for name, kwargs, with_module in PROBES:
        require(sha256_file(checkpoint) == expected, name + ': SHA256 verified before the probe')
        trace.take()
        scope = upstream_modules(models_root, ('custom_rn',), {'custom_rn'}) if with_module else contextlib.nullcontext()
        with warnings.catch_warnings(record=True) as caught, scope as modules:
            warnings.simplefilter('always')
            try:
                obj = torch.load(str(checkpoint), **kwargs)
                state = rq.state_report(obj)
                outcome = {'outcome': 'LOADED', 'type': f'{type(obj).__module__}.{type(obj).__qualname__}',
                           'is_pinned_custom_rn_ResNet': bool(modules) and type(obj) is modules['custom_rn'].ResNet,
                           'devices': state['devices'],
                           'aggregate_parameter_sha256': state['aggregate_parameter_sha256'],
                           'aggregate_buffer_sha256': state['aggregate_buffer_sha256']}
                del obj
            except Exception as error:  # the observation IS the result; recorded, never retried
                outcome = outcome_of(error)
        events = trace.take()
        outcome.update(call='torch.load(path' + ''.join(f', {k}={v!r}' for k, v in kwargs.items()) + ')',
                       pinned_custom_rn_importable=with_module, sha256_verified_before_call=True,
                       find_class_globals=sorted({e[1] for e in events if e[0] == 'find_class'}),
                       warnings=sorted({f'{w.category.__name__}: {str(w.message).splitlines()[0][:160]}' for w in caught}))
        probes[name] = outcome
    d, t, f, m = (probes[n] for n, _, _ in PROBES)
    weights_only = 'WEIGHTS_ONLY_UNPICKLER_REJECTS_CUSTOM_MODULE_GLOBAL'
    require(d['outcome'] != 'LOADED', 'STOP: default torch.load already restores the module; no override justified')
    require(d['outcome'] == t['outcome'] == 'REJECTED' and d['category'] == t['category'] == weights_only and
            d['exception_class'] == t['exception_class'], 'BLOCKED_BY_RUNTIME_COMPATIBILITY: unexpected rejection kind')
    require(f['outcome'] == 'LOADED' and f['is_pinned_custom_rn_ResNet'] and f['devices'] == ['cuda:0'] and
            f['aggregate_parameter_sha256'] == written['writer']['state_before_save']['aggregate_parameter_sha256'] and
            f['aggregate_buffer_sha256'] == written['writer']['state_before_save']['aggregate_buffer_sha256'],
            'weights_only=False restores the source-native module')
    require(m['outcome'] == 'REJECTED' and m['category'] == 'PINNED_CUSTOM_RN_MODULE_NOT_IMPORTABLE',
            'module identity is required during unpickle')
    require(LOAD_COMPATIBILITY['weights_only'] is False, 'seam uses exactly the evidenced option')
    decision = {'observed': 'DEFAULT_AND_WEIGHTS_ONLY_TRUE_REJECT_WEIGHTS_ONLY_FALSE_RESTORES',
                'selected': 'EXPLICIT_WEIGHTS_ONLY_FALSE_AFTER_SHA256_VERIFICATION',
                'classification': LOAD_COMPATIBILITY['classification'],
                'default_resolves_to_weights_only_true': True, 'global_torch_load_patch': False,
                'scientific_adaptation': False, 'checkpoint_format_changed': False, 'source_modified': False}
    # ------------------------------------------------------------ C. SHA-gated seam load + round trip
    trace.take()
    loads = counters.n['torch_load_calls']
    ref = written['writer']
    with warnings.catch_warnings(record=True) as caught, load_verified_whole_module(checkpoint, expected,
                                                                                    adapter.config) as model:
        warnings.simplefilter('always')
        events = trace.take()
        kinds = [e[0] for e in events]
        require(counters.n['torch_load_calls'] == loads + 1, 'one seam torch.load')
        first_find = kinds.index('find_class')
        require(kinds[:3] == ['open_checkpoint', 'sha256_verified', 'torch.load'] and first_find > 2 and
                'open_checkpoint' not in kinds[1:], 'order: read -> SHA verified -> torch.load(bytes) -> unpickle')
        load_call = events[2][1]
        require(load_call == {'source': 'bytes', 'kwargs': {'weights_only': 'False'}}, 'explicit weights_only=False')
        resolved = sorted({e[1] for e in events if e[0] == 'find_class'})
        require('custom_rn.ResNet' in resolved and set(resolved) <= set(written['checkpoint']['archive']['globals']),
                'unpickle resolves custom_rn.ResNet; only archive globals')
        custom_rn = sys.modules['custom_rn']
        require(Path(custom_rn.__file__).resolve() == (Path(source['root']) / 'models/custom_rn.py').resolve(),
                'live custom_rn is the pinned file')
        identity = encoder_identity(model, custom_rn, contracts)
        require(identity == ref['identity'], 'identity equals writer')
        require(module_types(model) == ref['module_types'], 'module tree equals writer (no projection/substitution)')
        require(sorted({mm.training for mm in model.modules()}) == ref['training_flags_at_save'], 'saved mode restored')
        state = rq.state_report(model)
        require(state == ref['state_before_save'], 'every parameter and buffer bitwise equal to the pre-save state')
        buffers = dict(model.named_buffers())
        buffer_devices = sorted({str(b.device) for b in buffers.values()})
        buffer_dtypes = sorted({str(b.dtype) for b in buffers.values()})
        require(buffer_devices == ['cuda:0'] and state['devices'] == ['cuda:0'], 'on cuda:0 (unet_autoenc.py:76)')
        model.eval()  # FAS_train.py:35
        x_summary, forward, outs = reference_forward(torch, model)
        require(x_summary == written['reference_forward']['input'], 'identical synthetic input')
        equal = {n: forward[n]['sha256'] == written['reference_forward']['outputs'][n]['sha256'] for n in OUTPUTS}
        diff = {n: {k: forward[n][k] - written['reference_forward']['outputs'][n][k] for k in ('min', 'max', 'mean', 'l2')}
                for n in OUTPUTS if not equal[n]}
        require(all(equal.values()), 'bitwise forward equality: ' + json.dumps(diff))
        require(rq.state_report(model) == state, 'eval forward leaves the loaded state unchanged')
        result['seam_load'] = {'api': 'methods/difffas/aux_checkpoint.py::load_verified_whole_module',
            'event_order': kinds[:3] + ['find_class x%d' % kinds.count('find_class')],
            'torch_load_call': load_call, 'find_class_globals': resolved, 'identity': identity,
            'custom_rn_live_module_file': str(Path(custom_rn.__file__).resolve().relative_to(ROOT)),
            'class_is_pinned_module_class': type(model) is custom_rn.ResNet,
            'training_flags_after_load': ref['training_flags_at_save'],
            'fc': {'in_features': model.fc.in_features, 'out_features': model.fc.out_features},
            'parameters': state['parameters'], 'parameter_tensors': state['parameter_tensors'],
            'buffers': state['buffers'], 'parameter_devices': state['devices'], 'parameter_dtypes': state['dtypes'],
            'buffer_devices': buffer_devices, 'buffer_dtypes': buffer_dtypes,
            'parameter_tensors_bitwise_equal': sum(state['parameter_sha256'][k] == v
                                                   for k, v in ref['state_before_save']['parameter_sha256'].items()),
            'buffer_tensors_bitwise_equal': sum(state['buffer_sha256'][k] == v
                                                for k, v in ref['state_before_save']['buffer_sha256'].items()),
            'aggregate_parameter_sha256': state['aggregate_parameter_sha256'],
            'aggregate_buffer_sha256': state['aggregate_buffer_sha256'],
            'module_tree_equal': True, 'projection_or_adapter_modules_added': False}
        result['round_trip_forward'] = {'mode': 'eval (FAS_train.py:35), torch.no_grad', 'input': x_summary,
            'outputs': forward, 'bitwise_equal': equal, 'tolerance_used': 'NONE (bitwise)',
            'a6_feature_shapes': [forward[n]['shape'] for n in OUTPUTS[:3]], 'fourth_output_shape': forward['logits']['shape']}
        del model, outs
    require('custom_rn' not in sys.modules, 'custom_rn import scoped to the seam context')
    result['warnings'] = sorted({f'{w.category.__name__}: {w.message}' for w in caught})
    result['checkpoint_consumed'] = {'sha256': expected, 'size_bytes': written['checkpoint']['size_bytes'],
                                     'writer_evidence': writer_name, **ELIGIBILITY}
    result['rejection_before_deserialization'] = rejections
    result['probes'] = probes
    result['compatibility_decision'] = decision
    return finish(result, firewall, counters, source, {'optimizer_constructions': 0, 'optimizer_step_calls': 0,
                  'backward_calls': 0, 'autograd_grad_calls': 0, 'torch_save_calls': 0,
                  'torch_load_calls': len(PROBES) + 1})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--role', choices=('writer', 'loader'), required=True)
    parser.add_argument('--build-root', type=Path, required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--expected-sha256')
    parser.add_argument('--writer-evidence')
    args = parser.parse_args()
    build = args.build_root.resolve()
    require(not build.is_relative_to(ROOT), 'build root outside repository')
    require(build.parts[-3:] == BUILD_PARTS and build.is_relative_to(rq.RUNTIME), 'dedicated M6D6c build path')
    checkpoint = build.joinpath(*CHECKPOINT_RELATIVE)
    require(not checkpoint.is_relative_to(rq.RUNTIME / 'runs'), 'qualification checkpoint outside runs/')
    require(Path(args.output).name == args.output and args.output.endswith('.json'), 'evidence filename')
    require(not (build / args.output).exists(), 'fresh output name')
    tmp = Path(os.environ.get('TMPDIR', '/'))
    require(tmp.is_dir() and tmp.resolve().is_relative_to(build), 'TMPDIR inside the dedicated build root')
    if args.role == 'writer':
        require(args.expected_sha256 is None and args.writer_evidence is None, 'writer takes no SHA input')
        result = writer(build, checkpoint)
    else:
        require(args.writer_evidence and Path(args.writer_evidence).name == args.writer_evidence, 'writer evidence name')
        result = loader(build, checkpoint, args.expected_sha256, args.writer_evidence)
    (build / args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'status': result['status'], 'output': str(build / args.output)}))


if __name__ == '__main__':
    main()
