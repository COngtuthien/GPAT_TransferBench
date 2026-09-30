"""R-05 option D: bind the pinned NAFNet block definitions without importing BasicSR.

The pinned checkout (third_party/source_cache/nafnet, git-ignored) must sit at the commit and tree recorded in
third_party/source_pins.json, and every cited file must match its recorded SHA-256. Only SimpleGate, NAFBlock,
LayerNormFunction and LayerNorm2d are AST-selected and compiled unchanged into a namespace that holds only
torch, nn and F. No module-level upstream code runs, so the basicsr package, lmdb, get_root_logger,
local_arch.Local_Base, the NAFNet class and the arch registry are never imported. `ctx.saved_variables` stays
verbatim (a DeprecationWarning is expected). local_arch.py is verified for provenance and never executed.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

from methods.common.learned import git_read

ROOT = Path(__file__).resolve().parents[2]
SOURCE_PINS = ROOT / 'third_party/source_pins.json'
DEFAULT_ROOT = ROOT / 'third_party/source_cache/nafnet'
PIN_KEY = 'nafnet'
REPOSITORY = 'https://github.com/megvii-research/NAFNet'
PINNED_COMMIT = '2b4af71ebe098a92a75910c233a3965a3e93ede4'
ARCH = 'basicsr/models/archs/NAFNet_arch.py'
UTIL = 'basicsr/models/archs/arch_util.py'
LOCAL = 'basicsr/models/archs/local_arch.py'
LICENSE = 'LICENSE'
FILES_SHA256 = {
    ARCH: '01b22270cc93f1bb90c0e3e4490e98b023fcf73f8552860b4a9ee880ce5c6967',
    UTIL: '5a11af2e7c2d7a7b57c1fbd7e19cf0a50b4b4e8c7ae7dd203a915d7a707e7005',
    LOCAL: 'c4df2ba4d896442a0f6ec984accd6e68f31edce3afdf066add202c25a0d1af26',
    LICENSE: 'a29ecef3456149898f08e4c71b11b33e7d333664e087bc212e84e18ddd6599ad',
}
# Dependency order: LayerNorm2d uses LayerNormFunction; NAFBlock uses SimpleGate and LayerNorm2d.
SYMBOLS = ((UTIL, 'LayerNormFunction'), (UTIL, 'LayerNorm2d'), (ARCH, 'SimpleGate'), (ARCH, 'NAFBlock'))


class NAFSourceError(RuntimeError):
    """The pinned NAFNet source is unavailable or differs from its recorded provenance."""


def _pin():
    try:
        pin = json.loads(SOURCE_PINS.read_text())['sources'][PIN_KEY]
    except (OSError, ValueError, KeyError) as exc:
        raise NAFSourceError(f'NAFNet source provenance unavailable: {exc}') from exc
    if pin['repository'] != REPOSITORY or pin['pinned_commit'] != PINNED_COMMIT:
        raise NAFSourceError('NAFNet pin differs from the A10 section 6 pin')
    for rel, digest in FILES_SHA256.items():
        if pin['cited_files'][rel]['sha256'] != digest:
            raise NAFSourceError(f'recorded NAFNet provenance differs from A10 for {rel}')
    return pin


def verify_source(root=None) -> dict:
    """Verify commit, tree and per-file SHA-256; return the verified bytes of every cited file."""
    pin = _pin()
    root = Path(root if root is not None else DEFAULT_ROOT).resolve()
    if not (root / '.git').exists():
        raise NAFSourceError(f'pinned NAFNet source unavailable: {root}')
    try:
        if Path(git_read(root, 'rev-parse', '--show-toplevel')).resolve() != root:
            raise NAFSourceError('NAFNet source path resolves to a different checkout')
        if git_read(root, 'rev-parse', 'HEAD') != PINNED_COMMIT:
            raise NAFSourceError('NAFNet source commit mismatch')
        if git_read(root, 'rev-parse', 'HEAD^{tree}') != pin['commit_tree']:
            raise NAFSourceError('NAFNet source tree mismatch')
    except RuntimeError as exc:
        if isinstance(exc, NAFSourceError):
            raise
        raise NAFSourceError(str(exc)) from exc
    data = {}
    for rel, digest in FILES_SHA256.items():
        path = root / rel
        if not path.is_file() or path.is_symlink():
            raise NAFSourceError(f'pinned NAFNet file unavailable: {rel}')
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise NAFSourceError(f'pinned NAFNet file SHA-256 mismatch: {rel}')
        data[rel] = raw
    return {'root': str(root), 'commit': PINNED_COMMIT, 'tree': pin['commit_tree'], 'files': data}


def select_nodes(files: dict) -> list:
    """AST-select exactly the required top-level class definitions, in dependency order."""
    trees = {rel: ast.parse(files[rel], filename=rel) for rel in (UTIL, ARCH)}
    nodes = []
    for rel, name in SYMBOLS:
        found = [n for n in trees[rel].body if isinstance(n, ast.ClassDef) and n.name == name]
        if len(found) != 1:
            raise NAFSourceError(f'{name} must be defined exactly once at top level of {rel}')
        nodes.append((rel, found[0]))
    return nodes


def source_segments(files: dict) -> dict:
    """Verbatim source text of each selected definition, with its 1-based line range."""
    out = {}
    for rel, node in select_nodes(files):
        text = files[rel].decode('utf-8')
        out[node.name] = {'file': rel, 'first_line': node.lineno, 'last_line': node.end_lineno,
                          'source': ast.get_source_segment(text, node)}
    return out


def load(root=None) -> dict:
    """Return {'SimpleGate', 'NAFBlock', 'LayerNormFunction', 'LayerNorm2d'} bound from the pinned source."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    verified = verify_source(root)
    namespace = {'__name__': 'gpat_pinned_nafnet', 'torch': torch, 'nn': nn, 'F': F}
    for rel, node in select_nodes(verified['files']):
        code = compile(ast.Module(body=[node], type_ignores=[]), f"{verified['root']}/{rel}", 'exec')
        exec(code, namespace)
    return {name: namespace[name] for _, name in SYMBOLS}
