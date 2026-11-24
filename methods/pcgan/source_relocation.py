"""E05 owner ruling O2 (configs/amendments/e05_source_relocation_ruling.yaml): path-independent identity of the pinned
PCGAN upstream source (swapping-autoencoder-pytorch 6baa180f) and verification of the qualified M6D4a runtime closure.

The qualified compatibility manifest environments/e05.compatibility_patch_manifest.json records the upstream source
`root` as the absolute path of the original GPU checkout. That path stays documented as historical provenance, but it
is no longer the only permitted execution location: a byte-identical pinned source inside any clean execution checkout
is accepted, and ONLY the `root` field may differ. Every other identity field (repository, commit, tree, verification,
worktree status, the full files_sha256 inventory and every code file's sha256 + git blob) must be exactly equal, and is
additionally re-hashed here from disk. Stdlib only (Python >= 3.9); nothing is written.
"""
import hashlib
import json
from pathlib import Path

RULING_PATH = 'configs/amendments/e05_source_relocation_ruling.yaml'
RULING_SCHEMA = 'e05_source_relocation_ruling_v1'
SOURCE_REL = 'third_party/source_cache/swapping_autoencoder'
HISTORICAL_ROOT = '/home/student20261/workdir/GPAT_TransferBench/' + SOURCE_REL
RELOCATABLE_KEYS = ('root',)
EXECUTABLE_SUFFIXES = ('.py', '.cpp', '.cu', '.h', '.cuh', '.c', '.cc', '.so', '.pyx', '.pyd')
IMPORTED_MODULES = ('util', 'models.networks.encoder', 'models.networks.generator', 'models.networks.discriminator',
                    'models.networks.patch_discriminator', 'models.networks.stylegan2_op.fused_act',
                    'models.networks.stylegan2_op.upfirdn2d')


class SourceIdentityError(RuntimeError):
    pass


def _fail(message):
    raise SourceIdentityError('E05 source identity: ' + message)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _git_blob(raw):
    return hashlib.sha1(b'blob %d\0' % len(raw) + raw).hexdigest()


def _no_symlink_between(base, target):
    """Every path component from `base` (inclusive) down to `target` exists and is not a symlink."""
    p = Path(base)
    if p.is_symlink():
        return False
    for part in Path(target).relative_to(base).parts:
        p = p / part
        if p.is_symlink():
            return False
    return True


def load_ruling(checkout_root):
    p = Path(checkout_root) / RULING_PATH
    if not p.is_file() or p.is_symlink():
        _fail('owner ruling %s is missing' % RULING_PATH)
    raw = p.read_bytes()
    rec = json.loads(raw.decode('utf-8'))
    if rec.get('schema_version') != RULING_SCHEMA or rec.get('status') != 'OWNER_APPROVED' \
            or rec.get('approved') is not True or rec.get('decision') != 'O2_PATH_INDEPENDENT_SOURCE_IDENTITY' \
            or rec.get('relocatable_fields') != list(RELOCATABLE_KEYS) \
            or rec.get('historical_root') != HISTORICAL_ROOT:
        _fail('owner ruling %s is not the approved O2 record' % RULING_PATH)
    return rec, _sha(raw)


def verify_source_identity(manifest_source, live_source, checkout_root):
    """Accept the live pinned source iff it is content-identical to the qualified manifest source. Returns a record;
    raises SourceIdentityError otherwise."""
    checkout = Path(checkout_root)
    if not checkout.is_absolute() or str(checkout.resolve()) != str(checkout) or not checkout.is_dir():
        _fail('execution checkout root must be an absolute, resolved directory: %s' % checkout)
    if set(manifest_source) != set(live_source):
        _fail('identity field set differs: %s' % sorted(set(manifest_source) ^ set(live_source)))
    differing = sorted(k for k in manifest_source if manifest_source[k] != live_source[k])
    if any(k not in RELOCATABLE_KEYS for k in differing):
        _fail('content identity differs in %s' % [k for k in differing if k not in RELOCATABLE_KEYS])
    if manifest_source.get('root') != HISTORICAL_ROOT:
        _fail('qualified manifest root is not the documented historical root')
    if manifest_source.get('verification') != 'PASS' or manifest_source.get('worktree_status') != '':
        _fail('qualified source verification/worktree status')
    expected_root = checkout / SOURCE_REL
    root = Path(live_source['root'])
    if str(root) != str(expected_root) or str(root.resolve()) != str(expected_root) or not root.is_dir() \
            or not _no_symlink_between(checkout, root):
        _fail('pinned source must be the real directory <checkout>/%s (no symlink or traversal): %s'
              % (SOURCE_REL, root))
    if not (root / '.git').exists():
        _fail('pinned source checkout has no .git')
    # independent re-hash of every inventoried file (defence in depth over source_identity())
    for rel, digest in sorted(manifest_source['files_sha256'].items()):
        p = root / rel
        if '..' in Path(rel).parts or not p.is_file() or p.is_symlink() or not p.resolve().is_relative_to(root):
            _fail('missing/unsafe inventoried file %s' % rel)
        if _sha(p.read_bytes()) != digest:
            _fail('file bytes differ: %s' % rel)
    for rel, ident in sorted(manifest_source['code_files'].items()):
        p = root / rel
        if '..' in Path(rel).parts or not p.is_file() or p.is_symlink() or not p.resolve().is_relative_to(root):
            _fail('missing/unsafe code file %s' % rel)
        raw = p.read_bytes()
        if _sha(raw) != ident['sha256'] or _git_blob(raw) != ident['git_blob']:
            _fail('code file bytes differ: %s' % rel)
    # no executable source beyond the pinned code inventory (any symlink anywhere in the tree also fails)
    extra = []
    for p in sorted(root.rglob('*')):
        rel = p.relative_to(root)
        if rel.parts[0] == '.git':
            continue
        if p.is_symlink():
            _fail('symlink inside the pinned source: %s' % rel)
        if p.is_file() and p.suffix in EXECUTABLE_SUFFIXES and str(rel) not in manifest_source['code_files']:
            extra.append(str(rel))
    if extra:
        _fail('unexpected executable source: %s' % extra[:5])
    relocated = differing == ['root']
    out = {'relocated': relocated, 'root': str(root), 'historical_root': HISTORICAL_ROOT,
           'differing_fields': differing, 'commit': manifest_source['commit'], 'tree': manifest_source['tree'],
           'files_verified': len(manifest_source['files_sha256']), 'code_files_verified':
           len(manifest_source['code_files']), 'unexpected_executable_files': 0}
    if relocated:
        rec, digest = load_ruling(checkout)
        out['ruling'] = {'path': RULING_PATH, 'sha256': digest}
    return out


def verify_extension(path, expected, build, generated, closure, sources):
    """A qualified CUDA extension: every JIT source input is a closure file with its qualified hash, the binary lies
    inside the qualified build root and has its qualified sha256, and build.ninja is byte-identical."""
    build, generated, path = Path(build), Path(generated), Path(path)
    for s in sources:
        rel = str(Path(s).relative_to(generated))
        if rel not in closure or _sha(Path(s).read_bytes()) != closure[rel]:
            _fail('extension source input %s' % rel)
    if path.is_symlink() or not path.resolve().is_relative_to(build.resolve()):
        _fail('qualified binary location %s' % path)
    if _sha(path.read_bytes()) != expected['extension']['sha256']:
        _fail('qualified binary hash %s' % path)
    if (path.parent / 'build.ninja').read_text() != expected['build_ninja']:
        _fail('Ninja identity %s' % path)
    return {'path': str(path), 'sha256': expected['extension']['sha256']}


def verify_import_resolution(modules, generated, closure):
    """Every imported upstream module resolves to its file in the qualified generated closure (no shadowing)."""
    generated = Path(generated).resolve()
    out = {}
    for name in IMPORTED_MODULES:
        mod = modules.get(name)
        if mod is None or not getattr(mod, '__file__', None):
            _fail('module %s not imported' % name)
        p = Path(mod.__file__).resolve()
        if not p.is_relative_to(generated):
            _fail('module %s resolved outside the generated closure: %s' % (name, p))
        rel = str(p.relative_to(generated))
        if closure.get(rel) != _sha(p.read_bytes()):
            _fail('module %s file %s is not the qualified closure file' % (name, rel))
        out[name] = rel
    return out
