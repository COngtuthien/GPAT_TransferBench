"""A7 E07c production execution policy: FP32 / TF32-off precision and upstream encoder-loader RNG compatibility.

Owner-frozen benchmark execution policy (Amendment A7, DETERMINISTIC_IMPLEMENTATION_CLARIFICATION).
It is not an upstream-author choice and not a scientific deviation: fidelity stays
CONTROLLED_ADAPTATION / DEV-021 and the pinned source is never patched. The flags are
process-level and are applied only inside a dedicated E07c production process; nothing here
trains, reads benchmark data or reads/writes a checkpoint.
"""
from contextlib import contextmanager
import ast
import hashlib
from pathlib import Path

import yaml

from methods.common.config import ROOT, FrozenConfigError, load_method_config
from methods.common.learned import authoritative, PreparationError
from methods.common.upstream import upstream_modules
from .aux_checkpoint import load_frozen_aux_encoder
from .contract import validate_contract
from .source import tree, validate_source, verify_executable_semantics

A7_OVERLAY = 'configs/amendments/e07c_a7_execution_policy.yaml'
A7_OVERLAY_SHA256 = '3e4c758c1421aac6e993724a8a80b99e8b0754e75b983cec5ffca41479f492a3'
PRECISION = {'dtype': 'float32', 'autocast': False, 'grad_scaler': False, 'matmul_tf32': False, 'cudnn_tf32': False}
CUDNN = {'benchmark': False, 'deterministic': True}
# Observable process state once the policy is applied (matmul_tf32=False implies 'highest').
EXPECTED_STATE = {'default_dtype': 'torch.float32', 'matmul_tf32': False, 'cudnn_tf32': False,
                  'cudnn_benchmark': False, 'cudnn_deterministic': True, 'float32_matmul_precision': 'highest',
                  'autocast_cuda': False, 'autocast_cpu': False, 'deterministic_algorithms_forced': False}
THROWAWAY_CONSTRUCTOR = 'custom_rn.resnet18(pretrained=False)'
UPSTREAM_ENCODER_BODY = (('model_autoencoder = resnet18()', 74), ('model_autoencoder = torch.load(path)', 75),
                         ('model_autoencoder = model_autoencoder.cuda()', 76), ('return model_autoencoder', 77))
PRECISION_TOKENS = ('autocast', 'GradScaler', 'allow_tf32', 'float32_matmul_precision', '.half()',
                    'bfloat16', 'float16', 'torch.amp', 'cuda.amp')


def load_policy(config=None, *, overlay_path=None):
    """Verify the A7 overlay bytes and every identity it binds; return the parsed policy."""
    cfg = authoritative(config) if config is not None else load_method_config('E07c')
    raw = Path(overlay_path if overlay_path is not None else ROOT / A7_OVERLAY).read_bytes()
    if hashlib.sha256(raw).hexdigest() != A7_OVERLAY_SHA256:
        raise FrozenConfigError('A7 execution-policy overlay SHA256 mismatch')
    policy = yaml.safe_load(raw)
    contracts = validate_contract(cfg)
    doc = policy['amendment_document']
    if hashlib.sha256((ROOT / doc['path']).read_bytes()).hexdigest() != doc['sha256']:
        raise FrozenConfigError('A7 amendment document SHA256 mismatch')
    if (policy['base_config_path'] != cfg['_runtime']['config_path'] or
            policy['base_config_sha256'] != cfg['_runtime']['config_sha256'] or
            policy['amendment_a3'] != contracts['documents']['A3'] or
            policy['a6_overlay'] != {'path': contracts['overlay_path'], 'sha256': contracts['overlay_sha256']} or
            policy['source']['repository'] != cfg['source']['repository'] or
            policy['source']['pinned_commit'] != cfg['source']['pinned_commit']):
        raise FrozenConfigError('A7 base/A3/A6/source identity mismatch')
    fidelity = policy['fidelity']
    if ((fidelity['fidelity_class'], fidelity['deviation']) != (cfg['fidelity_class'], cfg['deviation']) or
            fidelity['new_fidelity_class'] or fidelity['new_deviation'] or
            policy['source']['source_pin_changed'] or policy['source']['official_source_modified']):
        raise PreparationError('A7 must not change fidelity, deviation or source')
    precision = {k: policy['precision'][k] for k in PRECISION}
    cudnn = {k: policy['cudnn'][k] for k in CUDNN}
    if precision != PRECISION or cudnn != CUDNN or cudnn != cfg['training']['cudnn']:
        raise PreparationError('A7 precision/cuDNN policy differs from the frozen values')
    rng = policy['rng_compatibility']
    if (rng['preserve_upstream_throwaway_encoder_constructor_rng'] is not True or rng['count'] != 1 or
            rng['constructor'] != THROWAWAY_CONSTRUCTOR or rng['device'] != 'cpu' or rng['conditioning_role'] or
            not policy['secure_loader']['mandatory']):
        raise PreparationError('A7 RNG-compatibility / secure-loader decision mismatch')
    return {'policy': policy, 'sha256': A7_OVERLAY_SHA256, 'contracts': contracts}


def precision_state():
    """Observed process precision/cuDNN state; reads flags only."""
    import torch
    return {'default_dtype': str(torch.get_default_dtype()),
            'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
            'cudnn_tf32': torch.backends.cudnn.allow_tf32,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'float32_matmul_precision': torch.get_float32_matmul_precision(),
            'autocast_cuda': torch.is_autocast_enabled('cuda'),
            'autocast_cpu': torch.is_autocast_enabled('cpu'),
            'deterministic_algorithms_forced': torch.are_deterministic_algorithms_enabled()}


def apply_e07c_precision_policy(config=None):
    """Set the A7 backend flags in THIS dedicated E07c process and return the verified state.

    FP32 default dtype; TF32 off for cuBLAS matmul and cuDNN; cuDNN autotuner off and
    deterministic cuDNN on. No autocast context is entered and no GradScaler exists in
    E07c. torch.use_deterministic_algorithms is deliberately NOT set (A7 section 2).
    """
    policy = load_policy(config)['policy']
    import torch
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = policy['precision']['matmul_tf32']
    torch.backends.cudnn.allow_tf32 = policy['precision']['cudnn_tf32']
    torch.backends.cudnn.benchmark = policy['cudnn']['benchmark']
    torch.backends.cudnn.deterministic = policy['cudnn']['deterministic']
    state = precision_state()
    if state != EXPECTED_STATE:
        raise PreparationError('A7 precision policy did not take effect: ' + repr(state))
    return state


def upstream_loader_semantics(source):
    """AST evidence of the pinned throwaway constructor; nothing is imported or executed."""
    consumer, rn = 'models/unet_autoenc.py', 'models/custom_rn.py'
    mod = tree(source, consumer)
    cls = next(n for n in mod.body if isinstance(n, ast.ClassDef) and n.name == 'BeatGANsAutoencModel')
    enc = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'encoder')
    body = tuple((ast.unparse(s), s.lineno) for s in enc.body)
    imports = [(ast.unparse(n), n.lineno) for n in mod.body if isinstance(n, ast.ImportFrom) and n.module == 'custom_rn']
    if body != UPSTREAM_ENCODER_BODY or imports != [('from .custom_rn import resnet18', 10)]:
        raise PreparationError('pinned BeatGANsAutoencModel.encoder(path) is not the expected throwaway + load')
    rn_tree = tree(source, rn)
    ctor = next(n for n in rn_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'resnet18')
    defaults = dict(zip([a.arg for a in ctor.args.args][-len(ctor.args.defaults):],
                        [ast.literal_eval(d) for d in ctor.args.defaults]))
    ret = next(ast.unparse(n.value) for n in ctor.body if isinstance(n, ast.Return))
    resnet = next(n for n in rn_tree.body if isinstance(n, ast.ClassDef) and n.name == 'ResNet')
    init = ast.unparse(next(n for n in resnet.body if isinstance(n, ast.FunctionDef) and n.name == '__init__'))
    if (defaults != {'pretrained': False, 'progress': True} or
            ret != "_resnet('resnet18', BasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)" or
            'self.fc = nn.Linear(512 * block.expansion, 17)' not in init or
            "nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')" not in init or
            'cuda' in init):
        raise PreparationError('pinned custom_rn.resnet18 constructor mismatch')
    return {'consumer': {'file': consumer, 'sha256': source['files_sha256'][consumer],
                         'method': 'BeatGANsAutoencModel.encoder', 'body': [list(b) for b in body],
                         'resnet18_import': 'from .custom_rn import resnet18 (line 10)'},
            'constructor': {'file': rn, 'sha256': source['files_sha256'][rn], 'function': 'resnet18',
                            'line': ctor.lineno, 'defaults': defaults, 'returns': ret,
                            'head': 'nn.Linear(512 * block.expansion, 17)', 'init': 'kaiming_normal_ on Conv2d; default nn.Linear init',
                            'cuda_in_constructor': False},
            'throwaway_overwritten_at_line': 75, 'imported_or_executed': False}


def upstream_precision_requests(source):
    """Occurrences of AMP/autocast/GradScaler/TF32/half-precision tokens in the verified pinned files."""
    hits = []
    for relative in sorted(source['files_sha256']):
        if not relative.endswith('.py'):
            continue
        text = (Path(source['root']) / relative).read_text()
        hits += [f'{relative}:{token}' for token in PRECISION_TOKENS if token in text]
    return hits


def upstream_main_order(source, policy):
    """Check every frozen A7 main-runner step against the pinned AST: exact statement at exact line, in order."""
    steps = []
    for step in policy['main_runner_order']:
        nodes = [n for n in ast.walk(tree(source, step['file'])) if isinstance(n, ast.stmt) and
                 n.lineno == step['line'] and ast.unparse(n).splitlines()[0] == step['statement']]
        if len(nodes) != 1:
            raise PreparationError(f"A7 order step {step['step']} not found in pinned {step['file']}:{step['line']}")
        steps.append({**step, 'sha256': source['files_sha256'][step['file']]})
    main = [s['line'] for s in steps if s['step'] <= 14]
    train = [s['line'] for s in steps if s['file'] == 'FAS_train.py' and s['step'] >= 15]
    loader = [s['line'] for s in steps if s['file'] == 'models/unet_autoenc.py']
    if ([s['step'] for s in steps] != list(range(1, len(steps) + 1)) or main != sorted(main) or
            train != sorted(train) or loader != [74, 75, 76] or
            not all(s['file'] == 'FAS_train.py' for s in steps if s['step'] <= 14)):
        raise PreparationError('A7 main-runner order is not the pinned source order')
    return steps


def consume_upstream_encoder_loader_rng(config=None):
    """RNG_COMPATIBILITY_ONLY: replay unet_autoenc.py:74 `resnet18()` immediately before the secure load (A7).

    Builds exactly one pinned custom_rn.resnet18(pretrained=False) on the CPU, with the default
    17-way head and NOT the A3 K=7 encoder, and drops it at once. It is never used, moved, saved,
    retained or returned. The only retained effect is the CPU RNG advance of the upstream
    constructor. No file, checkpoint, CUDA or manual RNG call.
    """
    cfg = authoritative(config) if config is not None else load_method_config('E07c')
    contracts = load_policy(cfg)['contracts']
    source = validate_source(cfg)
    verify_executable_semantics(source, contracts)
    upstream_loader_semantics(source)
    with upstream_modules(Path(source['root']) / 'models', ('custom_rn',), {'custom_rn'}) as modules:
        modules['custom_rn'].resnet18(pretrained=False)
    return {'role': 'RNG_COMPATIBILITY_ONLY', 'replays': 'models/unet_autoenc.py:74 model_autoencoder = resnet18()',
            'constructor': THROWAWAY_CONSTRUCTOR, 'device': 'cpu', 'object_returned': False}


@contextmanager
def main_runner_encoder(runtime_root, recorded_sha256, config=None):
    """A7 replacement for FAS_train.py:34 `encoder = model.encoder(path)`; the caller then calls .eval() (:35).

    Upstream order preserved: throwaway resnet18() (RNG only, unet_autoenc.py:74), then the
    mandatory M6D6c secure load (SHA256 before deserialize -> pinned custom_rn ->
    weights_only=False -> identity -> .cuda(), :75-76).
    """
    consume_upstream_encoder_loader_rng(config)
    with load_frozen_aux_encoder(runtime_root, recorded_sha256, config) as encoder:
        yield encoder
