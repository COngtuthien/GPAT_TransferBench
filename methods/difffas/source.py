"""Fail-closed DiffFAS provenance and AST evidence; never import training scripts."""
import ast
import hashlib
from pathlib import Path

from methods.common.learned import verify_source, PreparationError

REQUIRED_FILES = (
    'FAS_dataset.py', 'FAS_train.py', 'FAS_sample.py', 'diffusion.py',
    'config/diffconfig.py', 'config/diffusion.conf',
    'models/__init__.py', 'models/custom_rn.py', 'models/unet_autoenc.py',
    'models/pretrain_classifier.py', 'models/unet.py', 'models/blocks.py',
    'models/nn.py', 'models/latentnet.py', 'models/choices.py',
    'models/config_base.py', 'models/losses.py',
)


def validate_source(config, *, source_root=None):
    return verify_source(config, REQUIRED_FILES, source_root=source_root)


def tree(source, relative):
    raw = (Path(source['root']) / relative).read_bytes()
    if hashlib.sha256(raw).hexdigest() != source['files_sha256'][relative]:
        raise PreparationError('DiffFAS source changed after verification')
    return ast.parse(raw)


def symbol(source, relative, name, class_name=None):
    body = tree(source, relative).body
    if class_name:
        body = next(n for n in body if isinstance(n, ast.ClassDef) and n.name == class_name).body
    return next(n for n in body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)


def evidence(source):
    locations = [('FAS_train.py', 'seed_torch', None), ('FAS_train.py', 'train', None),
                 ('FAS_sample.py', 'main', None), ('diffusion.py', 'training_losses', 'GaussianDiffusion'),
                 ('diffusion.py', 'create_gaussian_diffusion', None), ('diffusion.py', 'ddim_steps', None),
                 ('models/custom_rn.py', 'resnet18', None),
                 ('models/custom_rn.py', '_forward_impl', 'ResNet'),
                 ('models/unet_autoenc.py', 'forward', 'BeatGANsAutoencModel'),
                 ('models/unet_autoenc.py', 'encoder', 'BeatGANsAutoencModel')]
    return [{'file': file, 'symbol': (cls + '.' if cls else '') + name,
             'line': symbol(source, file, name, cls).lineno,
             'sha256': source['files_sha256'][file]} for file, name, cls in locations]


def verify_executable_semantics(source, contracts):
    """Assert critical executable wiring, not just claims copied into a plan."""
    overlay = contracts['overlay']
    for relative, key in [('models/custom_rn.py', 'sha256'), ('models/unet_autoenc.py', 'consumer_sha256')]:
        if source['files_sha256'][relative] != overlay['source'][key]:
            raise PreparationError('A6 source digest mismatch')
    if any(source[k] != overlay['source'][v] for k, v in [('repository', 'repository'), ('commit', 'pinned_commit')]):
        raise PreparationError('A6 source identity mismatch')
    constructor = symbol(source, 'models/custom_rn.py', 'resnet18')
    call = next(n.value for n in constructor.body if isinstance(n, ast.Return))
    topology = ast.literal_eval(call.args[2])
    if ast.unparse(call.func) != '_resnet' or ast.unparse(call.args[1]) != 'BasicBlock':
        raise PreparationError('custom encoder constructor mismatch')
    forward = ast.unparse(symbol(source, 'models/unet_autoenc.py', 'forward', 'BeatGANsAutoencModel'))
    if 'x32x32, x16x16, x8x8, _ = self.encode(x_cond, encoder)' not in forward:
        raise PreparationError('encoder feature consumption mismatch')
    training = ast.unparse(symbol(source, 'diffusion.py', 'training_losses', 'GaussianDiffusion'))
    if ('torch.cat([x_t, img], 1) if use_pair_flag else torch.cat([x_t], 1)' not in training or
            'x_cond=target_pose' not in training or 'ModelMeanType.EPSILON: noise' not in training):
        raise PreparationError('A1 unpaired epsilon path mismatch')
    sample = symbol(source, 'FAS_sample.py', 'main')
    if 'torch.load(args.model_path)[\'model\']' not in ast.unparse(sample):
        raise PreparationError('sampler must load the model tensor set')
    return {'topology': topology, 'block': 'BasicBlock',
            'fourth_output_discarded': True, 'training_content_inert': True,
            'sampler_tensor_set': 'model', 'symbols': evidence(source)}
