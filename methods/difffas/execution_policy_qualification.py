"""M6D6d E07c A7 execution-policy qualification: precision flags + throwaway-constructor RNG equivalence, never training.

Run in an isolated GPU process under gpat-m6-e07c (two fresh processes, qualification_seed 60604):

  A. apply methods/difffas/execution_policy.py::apply_e07c_precision_policy and record the
     library defaults before and the effective state after;
  B. from one identical CPU RNG state, compare the CPU RNG transition of
       helper      consume_upstream_encoder_loader_rng (cold custom_rn import, inside the window)
       direct      pinned custom_rn.resnet18(pretrained=False)
       call_site   pinned BeatGANsAutoencModel.encoder(path) run up to its torch.load, which is
                   INTERCEPTED before any I/O (nothing is opened, read or deserialized)
       helper_ctx  the helper again with the upstream main modules imported
       bypass      no constructor (control)
     bitwise at RNG-state byte level, plus the next torch.rand draw;
  C. synthetic eval forwards of the K7 encoder and the main BeatGANsAutoencModel under the
     applied policy (torch.no_grad).

No dataset, manifest, image, optimizer, backward, autocast, GradScaler, torch.save or real
torch.load exists here. The throwaway object is never moved to CUDA.
"""
import argparse
import gc
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

SEED = 60604  # qualification_seed only; never the auxiliary (42) or an experiment seed (42/1337/2026)
LABEL = 'EXECUTION_POLICY_QUALIFICATION_ONLY'
BUILD_PARTS = ('builds', 'e07c_difffas', 'm6d6d')
BATCH = 4
K = 7
NEXT_DRAW = 16  # torch.rand(16) after each path
INTERCEPT_PATH = '<M6D6d-intercepted-before-io:never-opened>'
PATHS = ('helper', 'direct', 'call_site', 'helper_ctx', 'bypass')
ENCODER_SHAPES = {'x32x32': [BATCH, 256, 32, 32], 'x16x16': [BATCH, 512, 16, 16], 'x8x8': [BATCH, 512, 8, 8],
                  'embg': [BATCH, K]}
MAIN_OUTPUT_SHAPE = [BATCH, 6, 256, 256]
INPUT_PHASES = {'encoder_input': 61, 'x_t': 67, 'x_cond': 71}  # analytic rq.synthetic family, new phases
COUNTERS = ('optimizer_constructions', 'optimizer_step_calls', 'backward_calls', 'autograd_grad_calls',
            'torch_save_calls', 'torch_load_calls', 'torch_load_intercepted_before_io', 'autocast_entries',
            'grad_scaler_constructions')


def require(value, message):
    if not value:
        raise RuntimeError('M6D6d gate: ' + message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class Intercepted(Exception):
    """Raised by the torch.load stand-in at unet_autoenc.py:75 before any I/O."""


class Counters:
    """Forbids optimizer/backward/autocast/GradScaler/torch.save/torch.load; one torch.load interception window."""

    def __init__(self, torch):
        self.torch, self.n, self.intercept = torch, {k: 0 for k in COUNTERS}, None

    def install(self):
        torch, n = self.torch, self.n

        def forbid(name):
            def forbidden(*args, **kwargs):
                n[name] += 1
                raise RuntimeError('M6D6d forbids ' + name)
            return forbidden

        def load(f, *args, **kwargs):
            if self.intercept is None:
                return forbid('torch_load_calls')()
            n['torch_load_intercepted_before_io'] += 1
            self.intercept(f, args, kwargs, sys._getframe(1))
            raise Intercepted()
        torch.optim.Optimizer.__init__ = forbid('optimizer_constructions')
        for cls in {c for c in vars(torch.optim).values() if isinstance(c, type) and
                    issubclass(c, torch.optim.Optimizer)}:
            cls.step = forbid('optimizer_step_calls')
        torch.Tensor.backward = forbid('backward_calls')
        torch.autograd.backward = forbid('backward_calls')
        torch.autograd.grad = forbid('autograd_grad_calls')
        torch.save = forbid('torch_save_calls')
        torch.load = load
        torch.amp.autocast_mode.autocast.__enter__ = forbid('autocast_entries')
        torch.amp.grad_scaler.GradScaler.__init__ = forbid('grad_scaler_constructions')


def rng_snapshot(torch, np):
    state = np.random.get_state()
    return {'cpu_sha256': sha(torch.get_rng_state().numpy().tobytes()),
            'cuda_sha256': sha(torch.cuda.get_rng_state().numpy().tobytes()),
            'python_random_sha256': sha(repr(random.getstate()).encode()),
            'numpy_sha256': sha(state[1].tobytes() + repr((state[0], *state[2:])).encode())}


def throwaway_instances():
    gc.collect()
    return sum(1 for o in gc.get_objects() if type(o).__name__ == 'ResNet' and
               type(o).__module__ in ('custom_rn', 'models.custom_rn'))


def plain(value):
    """True if a helper return value holds only plain data (no module, tensor or object reference)."""
    if isinstance(value, dict):
        return all(isinstance(k, str) and plain(v) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return all(plain(v) for v in value)
    return value is None or isinstance(value, (str, int, float, bool))


def rng_equivalence(torch, np, adapter, source, counters):
    """Section B: identical initial CPU RNG state -> five paths -> byte-level transition comparison.

    helper and direct run before the upstream main modules are imported; call_site,
    helper_ctx and bypass run with them imported (the future main-runner context).
    """
    from methods.common.upstream import upstream_modules
    from methods.difffas import execution_policy as ep
    torch.manual_seed(SEED)
    initial_cpu, initial_cuda = torch.get_rng_state(), torch.cuda.get_rng_state()
    initial = sha(initial_cpu.numpy().tobytes())
    paths, captured = {}, {}

    def run(name, fn):
        torch.set_rng_state(initial_cpu)
        torch.cuda.set_rng_state(initial_cuda)
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated()
        instances = throwaway_instances()
        before = rng_snapshot(torch, np)
        out = fn()
        after = rng_snapshot(torch, np)
        draw = torch.rand(NEXT_DRAW)
        record = {'before': before, 'after': after, 'cpu_state_changed': before['cpu_sha256'] != after['cpu_sha256'],
                  'next_rand': {'call': f'torch.rand({NEXT_DRAW})', 'sha256': sha(draw.numpy().tobytes()),
                                'first4': [float(v) for v in draw[:4]]}}
        if name in ('direct',):
            record['object'] = {'type': f'{type(out).__module__}.{type(out).__qualname__}',
                                'fc': [out.fc.in_features, out.fc.out_features],
                                'devices': sorted({str(p.device) for p in out.parameters()}),
                                'aggregate_parameter_sha256': rq.state_report(out)['aggregate_parameter_sha256']}
            del out
            out = None
        record['return_value'] = out if plain(out) else f'NON_PLAIN:{type(out).__qualname__}'
        record['return_value_plain_data'] = plain(out)
        del out
        torch.cuda.synchronize()
        record['cuda_memory_allocated_delta_bytes'] = torch.cuda.memory_allocated() - allocated
        record['throwaway_instances_before'], record['throwaway_instances_after'] = instances, throwaway_instances()
        paths[name] = record

    # helper first: its window includes the cold import of the pinned custom_rn module
    run('helper', lambda: ep.consume_upstream_encoder_loader_rng(adapter.config))
    with upstream_modules(Path(source['root']) / 'models', ('custom_rn',), {'custom_rn'}) as modules:
        ctor = modules['custom_rn'].resnet18
        run('direct', lambda: ctor(pretrained=False))
    require('custom_rn' not in sys.modules, 'custom_rn import scoped')

    def intercept(f, args, kwargs, frame):
        captured.update(argument=f, extra_args=len(args), kwargs=sorted(kwargs),
                        caller={'function': frame.f_code.co_name, 'file': Path(frame.f_code.co_filename).name,
                                'line': frame.f_lineno},
                        rng_at_torch_load=rng_snapshot(torch, np))
        obj = frame.f_locals['model_autoencoder']
        captured['object'] = {'type': f'{type(obj).__module__}.{type(obj).__qualname__}',
                              'fc': [obj.fc.in_features, obj.fc.out_features],
                              'devices': sorted({str(p.device) for p in obj.parameters()}),
                              'aggregate_parameter_sha256': rq.state_report(obj)['aggregate_parameter_sha256']}
        del obj

    with upstream_modules(Path(source['root']), rq.UPSTREAM_MAIN, rq.UPSTREAM_MAIN_ROOTS) as modules:
        ua = modules['models.unet_autoenc']

        def call_site():
            counters.intercept = intercept
            try:
                ua.BeatGANsAutoencModel.encoder(None, INTERCEPT_PATH)
            except Intercepted:
                pass
            else:
                require(False, 'upstream encoder(path) must reach torch.load')
            finally:
                counters.intercept = None
        run('call_site', call_site)
        run('helper_ctx', lambda: ep.consume_upstream_encoder_loader_rng(adapter.config))
        run('bypass', lambda: None)
        del ua
    # ------------------------------------------------------------------ gates
    replay = ('helper', 'direct', 'call_site', 'helper_ctx')
    for name in PATHS:
        p = paths[name]
        require(p['before']['cpu_sha256'] == initial, name + ': identical initial CPU RNG state')
        for k in ('cuda_sha256', 'python_random_sha256', 'numpy_sha256'):
            require(p['before'][k] == p['after'][k], f'{name}: {k} untouched (CPU-only consumption)')
        require(p['cuda_memory_allocated_delta_bytes'] == 0, name + ': no CUDA allocation')
        require(p['throwaway_instances_after'] == 0, name + ': throwaway object not alive afterwards')
    ref = paths['direct']
    for name in replay:
        require(paths[name]['cpu_state_changed'], name + ': constructor consumes CPU RNG')
        require(paths[name]['after']['cpu_sha256'] == ref['after']['cpu_sha256'], name + ': CPU RNG after == direct')
        require(paths[name]['next_rand']['sha256'] == ref['next_rand']['sha256'], name + ': next draw == direct')
    for name in ('helper', 'helper_ctx'):
        p = paths[name]
        require(p['return_value_plain_data'] and p['return_value']['role'] == 'RNG_COMPATIBILITY_ONLY' and
                p['return_value']['object_returned'] is False, name + ': returns plain record only')
    require(captured['caller'] == {'function': 'encoder', 'file': 'unet_autoenc.py', 'line': 75} and
            captured['argument'] == INTERCEPT_PATH and captured['extra_args'] == 0 and captured['kwargs'] == [],
            'interception at unet_autoenc.py:75 torch.load(path)')
    require(captured['rng_at_torch_load'] == paths['call_site']['after'], 'nothing between :75 and return consumes RNG')
    same = ('fc', 'devices', 'aggregate_parameter_sha256')
    require({k: captured['object'][k] for k in same} == {k: ref['object'][k] for k in same} and
            captured['object']['type'] == 'models.custom_rn.ResNet' and ref['object']['type'] == 'custom_rn.ResNet' and
            ref['object']['fc'] == [512, 17] and ref['object']['devices'] == ['cpu'],
            'call-site throwaway (consumer import models.custom_rn) == direct constructor output (CPU, 17-way head)')
    bypass = paths['bypass']
    require(not bypass['cpu_state_changed'] and bypass['next_rand']['sha256'] != ref['next_rand']['sha256'],
            'bypass control: no constructor -> different next draw')
    return {'initial_cpu_rng_sha256': initial, 'initial_seeding': f'torch.manual_seed({SEED})',
            'rng_state_api': 'torch.get_rng_state() bytes (CPU default generator, mt19937)',
            'order_of_paths': list(PATHS), 'paths': paths,
            'call_site_interception': {k: v for k, v in captured.items() if k != 'argument'} |
                                      {'argument': 'INTERCEPT_PATH (never opened)', 'io_performed': False,
                                       'deserialization': False},
            'equal': {'before_all_paths': True, 'after_helper_eq_direct': True, 'after_call_site_eq_direct': True,
                      'after_helper_ctx_eq_direct': True, 'next_rand_equal_replay_paths': True,
                      'bypass_next_rand_differs': True, 'cuda_rng_untouched': True,
                      'python_and_numpy_rng_untouched': True},
            'reference_after_cpu_rng_sha256': ref['after']['cpu_sha256'],
            'reference_next_rand_sha256': ref['next_rand']['sha256'],
            'bypass_next_rand_sha256': bypass['next_rand']['sha256']}


def runtime_forward(torch, adapter, dc, ua, applied):
    """Section C: synthetic eval forwards under the applied policy; no backward, optimizer or checkpoint."""
    from methods.difffas.encoder import encoder_model
    from methods.difffas import execution_policy as ep
    observed = []

    def probe(tag):
        def hook(module, args, kwargs=None):
            observed.append({'at': tag, 'state': ep.precision_state()})
        return hook
    out = {}
    with encoder_model(adapter.config) as enc:
        custom_rn = sys.modules[type(enc).__module__]
        require(type(enc) is custom_rn.ResNet and (enc.fc.in_features, enc.fc.out_features) == (512, K), 'K7 encoder')
        enc = enc.cuda().eval()
        enc.register_forward_pre_hook(probe('encoder'))
        model = dc.get_model_conf().make_model()
        require(type(model) is ua.BeatGANsAutoencModel and model.input_blocks[0][0].in_channels == 3,
                'pinned main model, 3-channel input')
        model = model.cuda().eval()
        model.register_forward_pre_hook(probe('main_model'), with_kwargs=True)
        stages = {}
        for name in ('middle_block', f'output_blocks.{len(model.output_blocks) - 1}'):
            model.get_submodule(name).register_forward_hook(
                lambda m, a, o, name=name: stages.__setitem__(name, o.detach().clone()))
        x = rq.synthetic(torch, BATCH, INPUT_PHASES['encoder_input']).cuda()
        x_t = rq.synthetic(torch, BATCH, INPUT_PHASES['x_t']).cuda()
        x_cond = rq.synthetic(torch, BATCH, INPUT_PHASES['x_cond']).cuda()
        t = torch.tensor([0, 250, 500, 999], device='cuda')  # rescale_timesteps=False: _scale_timesteps is identity
        cond_mask = torch.ones(BATCH, dtype=torch.bool, device='cuda')
        tcfg = adapter.config['training']
        with torch.no_grad():
            features = enc(x)
            for name, tensor in zip(ENCODER_SHAPES, features):
                require(list(tensor.shape) == ENCODER_SHAPES[name] and tensor.dtype == torch.float32, name + ' A6 shape')
                out['encoder_' + name] = rq.summarize(tensor)
            y = model(x=torch.cat([x_t], 1), encoder=enc, t=t, cond_mask=cond_mask, x_cond=x_cond, prob=1,
                      means_size=tcfg['means_size'], var_size=tcfg['var_size'])
            require(list(y.shape) == MAIN_OUTPUT_SHAPE and y.dtype == torch.float32, 'main output shape/dtype')
            out['main_model_output'] = rq.summarize(y)
            for name, tensor in sorted(stages.items()):
                out['main_' + name] = rq.summarize(tensor)
        require(len(stages) == 2, 'internal stages captured')
        del model, enc, features, y
    require(observed and all(o['state'] == applied for o in observed), 'policy in force during every forward')
    require(ep.precision_state() == applied, 'policy unchanged by upstream imports and forwards')
    return {'label': 'SYNTHETIC_FORWARD_UNDER_A7_POLICY', 'autograd': 'torch.no_grad', 'mode': 'eval',
            'input_phases': INPUT_PHASES, 'input_label': 'SYNTHETIC_IN_MEMORY_NOT_BENCHMARK_SAMPLES',
            'encoder': {'construction_seam': 'methods/difffas/encoder.py::encoder_model (A3 seam, fc -> Linear(512, 7))',
                        'untrained_in_memory_only': True, 'weights_loaded': False},
            'main_model': {'construction': 'config/diffconfig.py::get_model_conf().make_model()', 'use_pair': False,
                           'prob': 1, 'means_size': tcfg['means_size'], 'var_size': tcfg['var_size'],
                           'final_output_zero_note': 'zero-initialised final projection (M6D6a); internal stages recorded'},
            'outputs': out, 'finite_all': all(v['finite'] for v in out.values()),
            'a6_feature_shapes': [ENCODER_SHAPES[k] for k in ('x32x32', 'x16x16', 'x8x8')],
            'precision_state_inside_forwards': sorted({o['at'] for o in observed}),
            'architecture_changed': False, 'backward': False, 'optimizer_step': False, 'checkpoint': False}


def qualify(build):
    firewall = rq.Firewall(build)
    sys.addaudithook(firewall)
    source, semantics, adapter, contracts = rq.source_identity()
    static = rq.contract(adapter, contracts)
    from methods.difffas import execution_policy as ep
    loaded = ep.load_policy(adapter.config)
    policy = loaded['policy']
    loader_semantics = ep.upstream_loader_semantics(source)
    order = ep.upstream_main_order(source, policy)
    precision_requests = ep.upstream_precision_requests(source)
    require(precision_requests == [], 'pinned source requests no AMP/autocast/GradScaler/TF32/half precision')
    require(contracts['effective_encoder']['auxiliary_encoder_training_seed'] == 42 and
            SEED not in (42, 1337, 2026) and adapter.config['seeds']['experiment_seeds'] == [42, 1337, 2026],
            'qualification seed distinct from scientific seeds')
    import inspect
    import numpy as np
    import torch
    rq.TORCH_LOAD_WEIGHTS_ONLY_DEFAULT = repr(inspect.signature(torch.load).parameters['weights_only'].default)
    defaults = ep.precision_state()
    counters = Counters(torch)
    counters.install()
    applied = ep.apply_e07c_precision_policy(adapter.config)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    result = {'milestone': 'M6D6d', 'label': LABEL, 'qualification_seed': SEED,
              'auxiliary_encoder_training_seed': None, 'experiment_seed': None, 'scientific_seed_consumed': False,
              'seed_note': '60604 seeds this check only; auxiliary 42 and main 42/1337/2026 are not used',
              'contract': static, 'source_before': source, 'executable_semantics': semantics,
              'a7': {'overlay': ep.A7_OVERLAY, 'overlay_sha256': loaded['sha256'],
                     'document': policy['amendment_document'], 'classification': policy['classification']},
              'precision': {'library_defaults_before_policy': defaults, 'applied_state': applied,
                            'expected_state': ep.EXPECTED_STATE, 'api': 'execution_policy.apply_e07c_precision_policy',
                            'grad_scaler_constructed': False, 'autocast_entered': False,
                            'use_deterministic_algorithms': 'NOT_SET'},
              'upstream_loader_semantics': loader_semantics, 'upstream_main_order': order,
              'upstream_precision_requests': precision_requests}
    result['environment_before'] = rq.environment()
    from methods.common.upstream import upstream_modules
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result['rng_equivalence'] = rng_equivalence(torch, np, adapter, source, counters)
        with upstream_modules(Path(source['root']), rq.UPSTREAM_MAIN, rq.UPSTREAM_MAIN_ROOTS) as modules:
            dc, ua = modules['config.diffconfig'], modules['models.unet_autoenc']
            result['runtime_forward'] = runtime_forward(torch, adapter, dc, ua, applied)
    result['warnings'] = sorted({f'{w.category.__name__}: {w.message}' for w in caught})
    result['precision']['state_after_all_sections'] = ep.precision_state()
    require(result['precision']['state_after_all_sections'] == applied, 'policy stable to the end')
    result['source_after'] = rq.source_identity()[0]
    require(result['source_after'] == source and source['worktree_status'] == '', 'pinned source unchanged, clean')
    result['environment_after'] = rq.environment()
    require(result['environment_before'] == result['environment_after'], 'environment stable')
    expected = {k: 0 for k in COUNTERS} | {'torch_load_intercepted_before_io': 1}
    require(counters.n == expected, 'exact counters ' + json.dumps(counters.n))
    require(not firewall.denied and not any(firewall.attempts.values()), 'firewall: zero denied accesses')
    require(all(Path(a[0]).name in rq.ALLOWED_SUBPROCESSES or a in rq.ALLOWED_EXACT_ARGV
                for a in firewall.subprocesses), 'subprocess allowlist')
    result.update(status='PASS', counters=counters.n, backward_calls=0, optimizer_step_calls=0, epochs=0,
        torch_save_calls=0, torch_load_calls=0, auxiliary_encoder_trained=False, scientific_training=False,
        scientific_checkpoint_created=False, scientific_checkpoint_loaded=False, main_difffas_training=False,
        benchmark_data_access=False, TRAIN_access=False, VAL_access=False, TEST_access=False, synthetic_bank=False,
        fidelity='CONTROLLED_ADAPTATION', source_patch='NONE',
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
    require(build.parts[-3:] == BUILD_PARTS and build.is_relative_to(rq.RUNTIME), 'dedicated M6D6d build path')
    require(Path(args.output).name == args.output and args.output.endswith('.json'), 'evidence filename')
    require(not (build / args.output).exists(), 'fresh output name')
    tmp = Path(os.environ.get('TMPDIR', '/'))
    require(tmp.is_dir() and tmp.resolve().is_relative_to(build), 'TMPDIR inside the dedicated build root')
    result = qualify(build)
    (build / args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'status': result['status'], 'output': str(build / args.output)}))


if __name__ == '__main__':
    main()
