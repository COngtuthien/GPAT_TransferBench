"""Additive learned-method preparation; imports no deep-learning framework.

Plans are metadata, never training launches or evidence of checkpoints written.
Scientific inputs always come from the verified M6B config or pinned source.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess

import numpy as np

from .config import ROOT, FrozenConfigError, load_method_config, sha256_file


class PreparationError(RuntimeError):
    """A prerequisite failed; there is no replacement asset/source/default."""


def authoritative(config: dict) -> dict:
    canonical = load_method_config(config['method_id'])
    if config != canonical:
        raise FrozenConfigError('learned adapter refuses scientific/config metadata overrides')
    if not canonical['learned_method']:
        raise PreparationError('learned adapter requires a learned method')
    return canonical


def git_read(root: Path, *args: str) -> str:
    # No fetch, checkout, model import or data inspection. Disable lazy network fetch.
    env = dict(os.environ, GIT_NO_LAZY_FETCH='1', GIT_TERMINAL_PROMPT='0')
    try:
        return subprocess.run(['git', '-C', str(root), *args], env=env,
                              check=True, capture_output=True, text=True).stdout.strip()
    except subprocess.CalledProcessError as exc:
        raise PreparationError(f'source git verification failed: {args}') from exc


def verify_source(config: dict, required_files: tuple[str, ...], *, source_root=None) -> dict:
    cfg = authoritative(config)
    source = cfg['source']
    pins = json.loads((ROOT / source['provenance']).read_text())['sources']
    matches = [p for p in pins.values() if cfg['method_id'] in p['method_ids']
               and p['repository'] == source['repository']]
    if len(matches) != 1:
        raise PreparationError('source pin must identify exactly one repository')
    pin = matches[0]
    if any(pin[k] != source[k] for k in ('repository', 'pinned_commit')):
        raise PreparationError('config/source provenance mismatch')
    root = Path(source_root) if source_root is not None else ROOT / source['local_source_path']
    root = root.resolve()
    if not (root / '.git').exists():
        raise PreparationError(f'missing pinned source checkout: {root}')
    if Path(git_read(root, 'rev-parse', '--show-toplevel')).resolve() != root:
        raise PreparationError('source path resolves to a different checkout')
    if git_read(root, 'rev-parse', 'HEAD') != source['pinned_commit']:
        raise PreparationError('source commit mismatch')
    if git_read(root, 'rev-parse', 'HEAD^{tree}') != pin['commit_tree']:
        raise PreparationError('source tree mismatch')
    origin = git_read(root, 'remote', 'get-url', 'origin').removesuffix('.git').rstrip('/')
    if origin != source['repository'].removesuffix('.git').rstrip('/'):
        raise PreparationError('source repository identity mismatch')
    verified = {}
    # All cited files plus the executable import closure. Removed upstream weights
    # are intentionally NOT read or restored.
    for rel in sorted(set(required_files) | set(pin['cited_files'])):
        path = root / rel
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root):
            raise PreparationError(f'missing/unsafe required source file: {rel}')
        entry = git_read(root, 'ls-tree', 'HEAD', '--', rel).split()
        if len(entry) != 4 or entry[0] not in ('100644', '100755') or entry[1] != 'blob':
            raise PreparationError(f'required source not a pinned regular file: {rel}')
        raw = path.read_bytes()
        blob = hashlib.sha1(f'blob {len(raw)}\0'.encode() + raw).hexdigest()
        digest = hashlib.sha256(raw).hexdigest()
        if blob != entry[2]:
            raise PreparationError(f'working source differs from pinned blob: {rel}')
        cited = pin['cited_files'].get(rel)
        if cited and (digest != cited['sha256'] or len(raw) != cited['size_bytes']):
            raise PreparationError(f'source provenance digest mismatch: {rel}')
        verified[rel] = digest
    return {'repository': source['repository'], 'commit': source['pinned_commit'],
            'tree': pin['commit_tree'], 'root': str(root), 'files_sha256': verified,
            'verification': 'PASS'}


def verify_asset(asset: dict) -> dict:
    path = Path(asset['external_runtime_path'])
    if not path.is_absolute() or path.resolve().is_relative_to(ROOT):
        raise PreparationError('external asset must remain outside the repository/source cache')
    if not path.is_file():
        raise PreparationError(f'required external asset unavailable: {path}')
    size = path.stat().st_size
    if 'bytes' in asset and size != asset['bytes']:
        raise PreparationError('external asset size mismatch')
    digest = sha256_file(path)
    if 'sha256' in asset and digest != asset['sha256']:
        raise PreparationError('external asset SHA256 mismatch')
    return {'path': str(path), 'size_bytes': size, 'sha256': digest,
            'verification': 'PASS', 'deserialized': False}


def environment_report(framework: str, modules: tuple[str, ...]) -> dict:
    # Metadata only: do not import torch/TF or initialise CUDA while preparing.
    packages = importlib.metadata.packages_distributions()
    observed = {}
    for module in modules:
        distributions = packages.get(module, [])
        observed[module] = {name: importlib.metadata.version(name) for name in distributions} or None
    missing = [k for k, v in observed.items() if v is None]
    return {'python': platform.python_version(), 'framework': framework,
            'observed_packages': observed, 'missing_modules': missing,
            'execution_ready': False,
            'reason': 'Missing modules: ' + ', '.join(missing) if missing else
                      'Installed metadata only; framework/API/device compatibility not executed',
            'final_execution_environment_frozen': False, 'gpu_probe_performed': False}


def seed_plan(config: dict, seed: int, framework: str) -> dict:
    cfg = authoritative(config)
    if type(seed) is not int or seed not in cfg['seeds']['experiment_seeds']:
        raise PreparationError('seed must be one of the frozen experiment seeds')
    if framework not in ('tensorflow', 'torch'):
        raise PreparationError('unsupported framework')
    return {'seed': seed, 'python_random': seed, 'numpy_random': seed,
            'framework': framework, 'framework_seed': seed,
            'launch_environment': {'PYTHONHASHSEED': str(seed)},
            'cudnn_benchmark': False if framework == 'torch' else None,
            'cudnn_deterministic': True if framework == 'torch' else None,
            'force_deterministic_algorithms': False,
            'worker_seeding': ('torch.initial_seed() modulo 2**32 -> Python/NumPy; seeded loader generator'
                               if framework == 'torch' else 'serialized_tf_map: num_parallel_calls=1'),
            'determinism_limit': 'RNG seeding is not proof of cross-device/kernel bitwise determinism; '
                                 'TF1 concurrent py_func RNG calls must be serialized by the future runner.'}


def apply_framework_seed(config: dict, seed: int, framework: str, *, cuda=False) -> dict:
    """Explicit future-runtime hook. No framework import occurs during preparation.

    PYTHONHASHSEED must be set before interpreter launch, not changed ineffectually here.
    No torch.use_deterministic_algorithms() override is imposed.
    """
    plan = seed_plan(config, seed, framework)
    if os.environ.get('PYTHONHASHSEED') != str(seed):
        raise PreparationError('launch interpreter with the seed plan PYTHONHASHSEED')
    module = importlib.import_module(framework)
    if framework == 'tensorflow' and not hasattr(module, 'set_random_seed'):
        raise PreparationError('STDN requires the pinned-source TF1 API, including tf.contrib')
    random.seed(seed)
    np.random.seed(seed)
    if framework == 'tensorflow':
        module.set_random_seed(seed)
    else:
        module.manual_seed(seed)
        module.backends.cudnn.benchmark = plan['cudnn_benchmark']
        module.backends.cudnn.deterministic = plan['cudnn_deterministic']
        if cuda:
            module.cuda.manual_seed_all(seed)
    return plan


def seed_torch_worker(worker_id: int) -> None:
    """Pass as DataLoader.worker_init_fn; torch assigns the seeded worker initial seed."""
    torch = importlib.import_module('torch')
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def effective_batch(config: dict, *, physical_batch_size=None,
                    gradient_accumulation_steps=1, replica_factor=1, oom_reason=None) -> dict:
    cfg = authoritative(config)
    expected = cfg['training']['effective_batch_size']
    physical = expected if physical_batch_size is None else physical_batch_size
    values = (physical, gradient_accumulation_steps, replica_factor)
    if any(type(v) is not int or v <= 0 for v in values):
        raise PreparationError('batch/accumulation/replica values must be positive integers')
    # Official DataParallel receives a GLOBAL batch; four GPUs is not a factor of four.
    if replica_factor != 1:
        raise PreparationError('only official global-batch DataParallel is supported; replica factor = 1')
    if physical * gradient_accumulation_steps * replica_factor != expected:
        raise PreparationError(f'effective batch must equal frozen {expected}')
    if physical != expected and (not cfg['training']['physical_batch_reduction_allowed'] or
                                 not isinstance(oom_reason, str) or not oom_reason.strip()):
        raise PreparationError('batch reduction requires observed OOM and a logged deviation reason')
    return {'physical_batch_size': physical, 'gradient_accumulation_steps': gradient_accumulation_steps,
            'replica_factor': replica_factor, 'effective_batch_size': expected,
            'oom_deviation_reason': oom_reason,
            'loss_accounting': 'Accumulate sample-weighted losses over the effective batch. '
                               'MMD uses the full effective-batch latent means before abs; '
                               'averaging microbatch MMD is NOT equivalent.',
            'tail_policy': 'Retain official non-drop-last tail; do not fabricate/repeat samples.',
            'runtime_accumulation_executed': False}


def checkpoint_plan(config: dict, seed: int, *, selection_split=None) -> dict:
    cfg = authoritative(config)
    if type(seed) is not int or seed not in cfg['seeds']['experiment_seeds']:
        raise PreparationError('checkpoint is scoped to a frozen experiment seed')
    if selection_split is not None:
        raise PreparationError('official final checkpoint is not selected using any data split')
    policy = cfg['checkpoint']
    if (policy['selection_scope'] != 'WITHIN_SEED' or policy['selection_uses_val'] or
            policy['selection_uses_test'] or policy['baseline_final_state_v1_overrides_this']):
        raise PreparationError('invalid official checkpoint policy')
    if cfg['method_id'] == 'E03':
        epoch = cfg['training']['max_epoch']
        final = f'ckpt-{epoch}'
        if policy['rule'] != f'OFFICIAL_LATEST_FINAL_CKPT_{epoch}':
            raise PreparationError('STDN final checkpoint rule mismatch')
    elif cfg['method_id'] == 'E06c':
        epoch = cfg['training']['all_epochs']
        final = policy['official_default_file']
        if policy['rule'] != f'OFFICIAL_GENERATOR_EPOCH_{epoch}':
            raise PreparationError('DSDG final checkpoint rule mismatch')
    else:
        raise PreparationError('method not implemented in M6C2a')
    return {'method_id': cfg['method_id'], 'experiment_seed': seed,
            'rule': policy['rule'], 'cadence': policy['cadence'], 'final_epoch': epoch,
            'authoritative_path_basename': final, 'selection_scope': 'WITHIN_SEED',
            'selection_uses_val': False, 'selection_uses_test': False,
            'checkpoint_types_supported': ['official', 'periodic', 'terminal', 'selected'],
            'checkpoint_written': False, 'checkpoint_sha256': None,
            'missing_field_reasons': {'checkpoint_sha256': 'Preparation only; no checkpoint exists'}}


def checkpoint_metadata(config: dict, seed: int, *, epoch: int, global_step: int,
                        path: str, file_size_bytes: int, sha256: str,
                        checkpoint_type: str, selected_for_final: bool) -> dict:
    """Metadata for a future writer to pass to RunContext.record_checkpoint.

    Caller supplies measured size/hash AFTER writing; this helper writes no bytes.
    STDN multi-file checkpoints require one record per physical shard.
    """
    policy = checkpoint_plan(config, seed)
    if checkpoint_type not in policy['checkpoint_types_supported']:
        raise PreparationError('unknown checkpoint metadata type')
    if type(epoch) is not int or not 1 <= epoch <= policy['final_epoch']:
        raise PreparationError('checkpoint epoch outside frozen budget')
    if (selected_for_final or checkpoint_type in ('selected', 'terminal')) and epoch != policy['final_epoch']:
        raise PreparationError('only official final epoch may be selected')
    expected = policy['authoritative_path_basename']
    basename = Path(path).name
    if selected_for_final and basename != expected and not (
            config['method_id'] == 'E03' and basename.startswith(expected + '.')):
        raise PreparationError('selected checkpoint filename does not match official final')
    if (type(global_step) is not int or global_step < 0 or type(file_size_bytes) is not int or
            file_size_bytes <= 0 or len(sha256) != 64 or any(c not in '0123456789abcdef' for c in sha256)):
        raise PreparationError('actual checkpoint size/hash/step metadata required')
    return dict(path=path, epoch=epoch, global_step=global_step, file_size_bytes=file_size_bytes,
                sha256=sha256, checkpoint_type=checkpoint_type, selected_for_final=selected_for_final,
                selection_reason=policy['rule'] if selected_for_final else 'Official cadence; not final selection')


def mapping(config: dict, targets: dict[str, str], source_file: str) -> list[dict]:
    """Mapping registry contains FIELD NAMES, never a second set of scientific values."""
    result = []
    for target, config_path in targets.items():
        value = config
        for field in config_path.split('.'):
            value = value[field]
        result.append({'target': target, 'config_field': config_path, 'value': value,
                       'source_file': source_file})
    return result


def torch_loader_options(config: dict, seed: int) -> dict:
    """Future DataLoader RNG wiring; no loader, device or model is created here."""
    seed_plan(config, seed, 'torch')
    torch = importlib.import_module('torch')
    generator = torch.Generator()
    generator.manual_seed(seed)
    return {'generator': generator, 'worker_init_fn': seed_torch_worker}


def serialized_tf_map(dataset, callback):
    """TF1 py_func callbacks must consume Python/NumPy RNG in dataset order.

    Retains the callback's official frame choice/flip behavior; only the worker
    scheduling surface is serialized under the frozen implementation-only policy.
    """
    return dataset.map(callback, num_parallel_calls=1)
