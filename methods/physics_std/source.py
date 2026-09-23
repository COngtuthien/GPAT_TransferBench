"""Verify the auxiliary source and bind unchanged NumPy-only upstream bodies."""
import ast
from pathlib import Path
import numpy as np

from methods.common.learned import verify_source, PreparationError

REQUIRED_FILES = (
    'TDDFA.py', 'utils/tddfa_util.py', 'utils/depth.py', 'utils/io.py',
    'bfm/bfm.py', 'Sim3DR/Sim3DR.py', 'Sim3DR/__init__.py',
    'Sim3DR/_init_paths.py', 'Sim3DR/lighting.py', 'Sim3DR/lib/rasterize.pyx',
    'Sim3DR/lib/rasterize.h', 'Sim3DR/lib/rasterize_kernel.cpp',
    'Sim3DR/setup.py', 'models/mobilenet_v1.py',
)


def validate_source(config, *, source_root=None):
    return verify_source(config, REQUIRED_FILES, source_root=source_root,
                         auxiliary_geometry=True)


def bind_function(source, relative_path, function_name, namespace, *, class_name=None):
    """Compile the exact verified function AST, excluding heavyweight imports.

    No algorithm is rewritten. A caller-supplied minimal namespace binds NumPy,
    official dependent functions, or the official rasterization extension.
    """
    path = Path(source['root']) / relative_path
    import hashlib
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != source['files_sha256'][relative_path]:
        raise PreparationError('source changed after verification')
    tree = ast.parse(raw, filename=str(path))
    body = tree.body
    if class_name is not None:
        body = next(n for n in body if isinstance(n, ast.ClassDef) and n.name == class_name).body
    node = next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == function_name)
    scope = {'np': np, **namespace}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
    return scope[function_name]
