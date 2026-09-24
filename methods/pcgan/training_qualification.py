"""Two invocations of this synthetic-only harness qualify the M6D4c runner.

No benchmark data API or checkpoint writer. Existing generated source and CUDA
binaries are read-only; JIT load is routed to their verified binary entry points.
"""
import argparse
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import traceback
from types import SimpleNamespace

from .runtime_qualification import ROOT, Firewall, source_identity, environment, sha
from .training_runner import TrainingRunner, COUNTS, OVERLAY_SHA, require, explicit_pair
from .training_losses import generator_losses, discriminator_losses, G_TERMS, D_TERMS
from . import training_diagnostics as diag

SEED = 60401  # diagnostic only, never an experiment result seed
LOCK_SHA = '849a100430e36048acde80d58801ae8908c15ae43719e97f4dcb90d77564ac50'


class ReadOnlyRuntimeFirewall(Firewall):
    def __call__(self, event, args):
        super().__call__(event, args)
        if event == 'open' and args and isinstance(args[0], (str, bytes)):
            flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
            path = os.path.abspath(os.fsdecode(args[0]))
            if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
                if not (path.startswith('/tmp/') or path == '/dev/null'):
                    self.denied.append(dict(event=event, path=path))
                    raise RuntimeError('Read-only environment/source/build: ' + path)


def existing_runtime(build, result):
    """Verify full qualified closure, then import existing real extensions only."""
    import torch
    import torch.utils.cpp_extension
    source, adapter = source_identity()
    manifest_raw = (build / 'compatibility_patch_manifest.json').read_bytes()
    require(manifest_raw == (ROOT / 'environments/e05.compatibility_patch_manifest.json').read_bytes(),
            'exact M6D4a compatibility manifest')
    manifest = json.loads(manifest_raw)
    require(manifest['source'] == source, 'source identity')
    generated = Path(manifest['generated_root'])
    closure = {rel: sha((generated / rel).read_bytes()) for rel in manifest['generated_files_sha256']}
    require(closure == manifest['generated_files_sha256'], 'generated source closure')
    previous = json.loads((ROOT / 'outputs/audit/M6D4A_E05_SYNTHETIC_PROCESS_1.json').read_text())
    binaries = {('fused' if name == 'fused_bias_act' else name): record
                for name, record in previous['custom_ops'].items()}
    loads = []
    def load_existing(name, sources, **kwargs):
        require(name in binaries, 'only qualified extension')
        expected = binaries[name]
        for path in sources:
            rel = str(Path(path).relative_to(generated))
            require(sha(Path(path).read_bytes()) == closure[rel], 'extension source input')
        path = Path(expected['extension']['path'])
        require(path.is_relative_to(build), 'qualified binary location')
        require(sha(path.read_bytes()) == expected['extension']['sha256'], 'qualified binary hash')
        require((path.parent / 'build.ninja').read_text() == expected['build_ninja'], 'Ninja identity')
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        loads.append(name)
        return module
    torch.utils.cpp_extension.load = load_existing
    sys.path.insert(0, str(generated))
    import util
    require(util.is_custom_kernel_supported(), 'custom CUDA branch')
    from models.networks.encoder import StyleGAN2ResnetEncoder
    from models.networks.generator import StyleGAN2ResnetGenerator
    from models.networks.discriminator import StyleGAN2Discriminator
    from models.networks.patch_discriminator import StyleGAN2PatchDiscriminator
    act = importlib.import_module('models.networks.stylegan2_op.fused_act')
    up = importlib.import_module('models.networks.stylegan2_op.upfirdn2d')
    require(act.use_custom_kernel and up.use_custom_kernel and sorted(loads) == ['fused', 'upfirdn2d'],
            'both real CUDA extensions loaded, no fallback/rebuild')
    result['source_before'] = source
    result['build_closure'] = dict(manifest_sha256=sha(manifest_raw), generated_files_sha256=closure,
        extensions={k: dict(extension=v['extension'], build_ninja_sha256=sha(v['build_ninja'].encode()))
                    for k, v in previous['custom_ops'].items()}, jit_build_invoked=False)
    from .architecture import architecture_mapping
    arch = architecture_mapping(adapter.config, source)
    options = {k: v['value'] for k, v in arch['inherited_source_defaults'].items()}
    options.update({v['target'].removeprefix('opt.'): v['value'] for v in arch['settings']})
    opt = SimpleNamespace(**options)
    models = {name: cls(opt).cuda() for name, cls in zip(COUNTS,
        (StyleGAN2ResnetEncoder, StyleGAN2ResnetGenerator, StyleGAN2Discriminator, StyleGAN2PatchDiscriminator))}
    result['models'] = {name: dict(class_name=type(m).__name__, options=options,
        parameter_count=sum(p.numel() for p in m.parameters())) for name, m in models.items()}
    def crop(x):
        return util.apply_random_crop(x, opt.patch_size, (opt.patch_min_scale, opt.patch_max_scale),
                                      num_crops=opt.patch_num_crops)
    return models, crop, previous


def qualify(build, result):
    firewall = ReadOnlyRuntimeFirewall(build)
    sys.addaudithook(firewall)
    import numpy as np
    import torch
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision('highest')
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    def forbidden(*args, **kwargs):
        raise RuntimeError('Pretrained load/checkpoint writing forbidden')
    torch.load = torch.save = torch.nn.Module.load_state_dict = forbidden
    require(sha((ROOT / 'environments/e05.lock.json').read_bytes()) == LOCK_SHA, 'environment lock')
    result.update(diagnostic_seed=SEED, overlay_sha256=OVERLAY_SHA, environment_lock_sha256=LOCK_SHA,
                  benchmark_data_access=False, benchmark_training=False, checkpoints=0,
                  pretrained_loads=0, synthetic_bank=False, fidelity='CONTROLLED_ADAPTATION')
    result['environment_before'] = environment()
    models, crop, previous = existing_runtime(build, result)
    require(result['environment_before'] == previous['environment_before'], 'unchanged M6D4a runtime')
    runner = TrainingRunner(models, crop, root=ROOT)
    result['optimizer_membership'] = {}
    for group, entries in runner.groups.items():
        optimizer = runner.optimizers[group]
        require([id(p) for p in optimizer.param_groups[0]['params']] == [id(p) for _, p in entries],
                'live optimizer membership')
        result['optimizer_membership'][group] = dict(parameter_count=sum(p.numel() for _, p in entries),
            tensor_count=len(entries), lr=optimizer.param_groups[0]['lr'],
            betas=list(optimizer.param_groups[0]['betas']), weight_decay=optimizer.param_groups[0]['weight_decay'],
            entries=[dict(name=n, identity=id(p), shape=list(p.shape), numel=p.numel()) for n, p in entries])
    require(result['optimizer_membership']['G']['parameter_count'] == 4629193 and
            result['optimizer_membership']['D']['parameter_count'] == 53381666, 'group counts')
    result['optimizer_overlap'] = False
    result['parameters_initial'] = diag.parameters(models)
    result['optimizer_applications'] = dict(D=0, G=0, total=0)
    for key, optimizer in runner.optimizers.items():
        def counted(opt, args, kwargs, key=key):
            result['optimizer_applications'][key] += 1
            result['optimizer_applications']['total'] += 1
            require(result['optimizer_applications'][key] == 1, 'one application budget')
        optimizer.register_step_post_hook(counted)
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, 256), torch.linspace(-1, 1, 256), indexing='ij')
    src = torch.stack((xx, yy, torch.sin(xx * 7) * torch.cos(yy * 5)))[None].cuda().requires_grad_()
    tgt = torch.stack((torch.cos(xx * 3), torch.sin(yy * 4), xx * yy))[None].cuda().requires_grad_()
    result['inputs'] = dict(x_src=diag.tensor_record(src), x_tgt=diag.tensor_record(tgt))
    phase = ['G_probe']
    result['noise_rng'] = []
    result['forward_calls'] = []
    def rng_hash():
        return sha(torch.cuda.get_rng_state().cpu().numpy().tobytes())
    for name, module in models['Generator'].named_modules():
        if type(module).__name__ == 'NoiseInjection':
            def noise_pre(m, args, name=name):
                require(m.fixed_noise is None, 'fresh StyleGAN noise')
                result['noise_rng'].append(dict(phase=phase[0], module=name, before=rng_hash()))
            def noise_post(m, args, output):
                result['noise_rng'][-1]['after'] = rng_hash()
                require(result['noise_rng'][-1]['before'] != result['noise_rng'][-1]['after'], 'noise RNG advance')
            module.register_forward_pre_hook(noise_pre)
            module.register_forward_hook(noise_post)
    for component in ('Encoder', 'Generator'):
        def forward_call(m, args, component=component):
            result['forward_calls'].append(dict(component=component, phase=phase[0],
                benchmark_iteration=runner.benchmark_iteration, D_applications=runner.applications['D'],
                input_identity=id(args[0])))
        models[component].register_forward_pre_hook(forward_call)
    E, G, D, P = (models[k] for k in COUNTS)
    print('G connectivity: direct complete five-loss graph', flush=True)
    paths = explicit_pair(E, G, src, tgt)
    losses, tensors = generator_losses(src, tgt, paths['r_src'], paths['m'], D, P, crop)
    result['initial_G'] = diag.external_totals(losses)
    result['initial_G_tensors'] = {k: diag.tensor_record(v) for k, v in {**paths, **tensors}.items()}
    inputs = dict(x_src=src, x_tgt=tgt, **{k: v for k, v in paths.items() if k.startswith('z_')})
    result['G_connectivity'] = {}
    for i, term in enumerate(G_TERMS):
        print('Probe ' + term, flush=True)
        result['G_connectivity'][term] = diag.connectivity(losses[term], models, inputs,
                                                           retain_graph=i < len(G_TERMS)-1)
    diag.validate_connectivity(result['G_connectivity'], generator=True)
    del paths, losses, tensors, inputs
    phase[0] = 'D_probe'
    print('D connectivity: direct complete detached objective', flush=True)
    paths = explicit_pair(E, G, src, tgt)
    losses, tensors = discriminator_losses(src, tgt, paths['r_src'], paths['m'], D, P, crop)
    result['probe_D'] = diag.external_totals(losses)
    result['probe_D_tensors'] = {k: diag.tensor_record(v) for k, v in tensors.items()}
    inputs = dict(x_src=src, x_tgt=tgt, r_src=paths['r_src'], m=paths['m'])
    result['D_connectivity'] = {}
    for i, term in enumerate(D_TERMS):
        print('Probe ' + term, flush=True)
        result['D_connectivity'][term] = diag.connectivity(losses[term], models, inputs,
                                                          retain_graph=i < len(D_TERMS)-1)
    diag.validate_connectivity(result['D_connectivity'], generator=False)
    require(diag.parameters(models) == result['parameters_initial'], 'no probe parameter updates')
    require(result['optimizer_applications']['total'] == 0 and
            all(not opt.state for opt in runner.optimizers.values()), 'no optimizer probe state')
    del paths, losses, tensors, inputs
    src = src.detach()
    tgt = tgt.detach()
    result['events'] = []
    def observer(stage, live, state):
        result['events'].append(dict(stage=stage, benchmark_iteration=live.benchmark_iteration,
                                    applications=dict(live.applications)))
        print(stage, flush=True)
        if stage in ('before_D', 'before_G'):
            phase[0] = stage[-1] + '_step'
        if stage in ('D_forward', 'G_forward'):
            group = stage[0]
            result[group + '_step_losses'] = diag.external_totals(state['losses'])
            result[group + '_step_tensors'] = {k: diag.tensor_record(v) for k, v in
                                             {**state['paths'], **state['tensors']}.items()}
            if group == 'D':
                require(not state['tensors']['rec_detached'].requires_grad and
                        not state['tensors']['mix_detached'].requires_grad, 'D fake detach')
                result['D_fakes_detached'] = True
            else:
                require(live.applications == dict(D=1, G=0), 'fresh G after D')
                require(state['paths']['r_src'].requires_grad and state['paths']['m'].requires_grad,
                        'G generated paths retain graph')
        if stage in ('D_backward', 'G_backward'):
            group = stage[0]
            grads = diag.step_gradients(live.groups)
            require(grads['G' if group == 'D' else 'D']['non_none'] == 0, 'inactive gradients absent')
            result[group + '_step_gradients'] = grads
        if stage in ('after_D', 'after_G'):
            group = stage[-1]
            state_after = diag.parameters(models)
            result['parameters_post_' + group] = state_after
            baseline = result['parameters_initial' if group == 'D' else 'parameters_post_D']
            result[group + '_changes'] = diag.changes(baseline, state_after,
                                                      result[group + '_step_gradients'], group)
            result[group + '_adam'] = diag.adam_state(live.optimizers[group], live.groups[group])
    runner.cycle(src, tgt, observer)
    result['benchmark_iteration'] = runner.benchmark_iteration
    require(runner.benchmark_iteration == 1 and result['optimizer_applications'] == dict(D=1, G=1, total=2),
            'complete iteration and application budget')
    calls = result['forward_calls']
    require([r['component'] for r in calls if r['phase'] == 'G_step'] ==
            ['Encoder', 'Encoder', 'Generator', 'Generator'], 'fresh G paths')
    require(all(r['D_applications'] == 1 and r['benchmark_iteration'] == 0
                for r in calls if r['phase'] == 'G_step'), 'post-D recomputation at t=0')
    result['G_recomputation_verified'] = True
    result['source_after'] = source_identity()[0]
    require(result['source_before'] == result['source_after'], 'source unchanged')
    result['environment_after'] = environment()
    require(result['environment_before'] == result['environment_after'], 'environment unchanged')
    # Recheck exact generated source, binary and build recipes after all numerical work.
    manifest = json.loads((ROOT / 'environments/e05.compatibility_patch_manifest.json').read_text())
    require(all(sha((Path(manifest['generated_root']) / p).read_bytes()) == h
                for p, h in manifest['generated_files_sha256'].items()), 'source closure after')
    for record in previous['custom_ops'].values():
        path = Path(record['extension']['path'])
        require(sha(path.read_bytes()) == record['extension']['sha256'] and
                (path.parent / 'build.ninja').read_text() == record['build_ninja'], 'build closure after')
    result['build_closure_unchanged'] = True
    result['firewall'] = dict(denied=firewall.denied, events=firewall.events)
    result['peak_cuda_memory_bytes'] = torch.cuda.max_memory_allocated()
    result['checkpoint_policy'] = runner.contract['checkpoint']
    result['status'] = 'PASS'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--process-number', type=int, choices=(1, 2), required=True)
    args = parser.parse_args()
    require(args.output.is_relative_to('/tmp') and not args.output.exists(), 'new /tmp JSON evidence')
    result = dict(process_number=args.process_number, launched=True, status='STARTED')
    try:
        qualify(args.build_root.resolve(), result)
    except BaseException as exc:
        result.update(status='STOP_AND_REPORT', error_type=type(exc).__name__, error=str(exc),
                      traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(dict(status=result['status'], output=str(args.output))), flush=True)
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
