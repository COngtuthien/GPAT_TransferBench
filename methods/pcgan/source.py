"""Verify architecture source without importing models or building CUDA ops."""
import ast
import hashlib
from pathlib import Path

from methods.common.learned import verify_source, PreparationError

REQUIRED_FILES = (
    'models/__init__.py', 'models/base_model.py', 'models/swapping_autoencoder_model.py',
    'models/networks/__init__.py', 'models/networks/base_network.py',
    'models/networks/encoder.py', 'models/networks/generator.py',
    'models/networks/discriminator.py', 'models/networks/patch_discriminator.py',
    'models/networks/stylegan2_layers.py', 'models/networks/loss.py',
    'models/networks/stylegan2_op/__init__.py',
    'models/networks/stylegan2_op/fused_act.py',
    'models/networks/stylegan2_op/fused_bias_act.cpp',
    'models/networks/stylegan2_op/fused_bias_act_kernel.cu',
    'models/networks/stylegan2_op/upfirdn2d.py',
    'models/networks/stylegan2_op/upfirdn2d.cpp',
    'models/networks/stylegan2_op/upfirdn2d_kernel.cu',
    'options/__init__.py', 'optimizers/swapping_autoencoder_optimizer.py',
    'util/__init__.py', 'util/util.py',
)


def validate_source(config, *, source_root=None):
    return verify_source(config, REQUIRED_FILES, source_root=source_root)


def source_tree(source, relative_path):
    path = Path(source['root']) / relative_path
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != source['files_sha256'][relative_path]:
        raise PreparationError('source changed after verification')
    return ast.parse(raw, filename=str(path))


def symbol_evidence(source, relative_path, class_name, function_name=None):
    tree = source_tree(source, relative_path)
    node = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name), None)
    if node is not None and function_name is not None:
        node = next((n for n in node.body if isinstance(n, ast.FunctionDef)
                     and n.name == function_name), None)
    if node is None:
        raise PreparationError(f'missing architecture symbol: {class_name}.{function_name}')
    return {'file': relative_path, 'symbol': class_name + ('.' + function_name if function_name else ''),
            'line': node.lineno, 'sha256': source['files_sha256'][relative_path]}


def literal_option(source, relative_path, name):
    """Read pinned parser defaults without executing source or duplicating values."""
    def value(node):
        # Pinned defaults include simple arithmetic, e.g. 256+128 and 1/8.
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Div)):
            left, right = value(node.left), value(node.right)
            return left + right if isinstance(node.op, ast.Add) else left / right
        return ast.literal_eval(node)
    matches = []
    for node in ast.walk(source_tree(source, relative_path)):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'add_argument' and node.args
                and isinstance(node.args[0], ast.Constant) and node.args[0].value == '--' + name):
            for keyword in node.keywords:
                if keyword.arg == 'default':
                    matches.append(value(keyword.value))
    if len(matches) != 1:
        raise PreparationError(f'expected one literal pinned option default: {name}')
    return matches[0]
