"""Import pinned source without vendoring it or running its script entrypoint."""
from contextlib import contextmanager
import importlib
from pathlib import Path
import sys
import threading

from .learned import PreparationError

_IMPORT_LOCK = threading.RLock()


@contextmanager
def upstream_modules(root, module_names, package_roots):
    """Scoped imports for legacy absolute imports; refuses namespace collisions.

    The caller must verify the source immediately before entering. Intended for a
    dedicated runtime process, not concurrent model workers sharing sys.modules.
    Importing exposes upstream functions; it does not call main(), create a model,
    or load a checkpoint. No silent alternative implementation is used.
    """
    root = Path(root).resolve()
    with _IMPORT_LOCK:
        if any(n.split('.')[0] in package_roots for n in sys.modules):
            raise PreparationError('legacy upstream module namespace already occupied')
        before = set(sys.modules)
        sys.path.insert(0, str(root))
        try:
            result = {name: importlib.import_module(name) for name in module_names}
            for module in result.values():
                if not Path(module.__file__).resolve().is_relative_to(root):
                    raise PreparationError('upstream import escaped the verified source root')
            yield result
        finally:
            sys.path.remove(str(root))
            for name in set(sys.modules) - before:
                if name.split('.')[0] in package_roots:
                    sys.modules.pop(name, None)
