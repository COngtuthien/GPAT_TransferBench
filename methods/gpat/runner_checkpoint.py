"""M7C4 GPAT checkpoints: recovery (resumable, never eligible for selection/bank/paper) and immutable EMA candidates.

Every payload holds tensors and JSON primitives only, so it loads with torch.load(weights_only=True). Writes are atomic:
<dir>/.tmp-* -> flush -> fsync -> os.replace -> fsync(dir); the SHA-256 and size are measured after the rename.
Recovery checkpoints exist only at optimizer boundaries (never a mid-accumulation state) and roll at
checkpoints/recovery/latest.pt (warmup: checkpoints/recovery/warmup_latest.pt). EMA candidates are written once per
epoch 10..60 to checkpoints/ema_candidates/ema_epoch_<NN>.pt, hold E_art EMA + G_res EMA only, carry selected=False
and no VAL metric; selection is a separate post-training process.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch

from methods.gpat import runner_io as rio

RECOVERY_KIND = 'GPAT_RECOVERY_CHECKPOINT'
WARMUP_RECOVERY_KIND = 'GPAT_WARMUP_RECOVERY_CHECKPOINT'
CANDIDATE_KIND = 'GPAT_EMA_CANDIDATE'
RECOVERY_LABELS = ('RECOVERY_ONLY', 'NOT_ELIGIBLE_FOR_VAL_SELECTION', 'NOT_ELIGIBLE_FOR_BANK',
                   'NOT_ELIGIBLE_FOR_PAPER_RESULT')
CANDIDATE_SELECTION_REASON = ('GPAT EMA candidate (epochs 10..60); not selected: selection is the separate '
                              'post-training VAL process (A10 D14/D16)')


def primitives(obj):
    """JSON round trip: str subclasses (e.g. TorchVersion) and tuples become plain JSON primitives."""
    return json.loads(json.dumps(obj, default=str))


# ----------------------------------------------------------------------------- RNG
def rng_state() -> dict:
    py = random.getstate()
    name, keys, pos, has_gauss, cached = np.random.get_state()
    return {'python': {'version': py[0], 'state': list(py[1]), 'gauss': py[2]},
            'numpy': {'name': name, 'keys': torch.from_numpy(np.asarray(keys, dtype=np.int64)), 'pos': int(pos),
                      'has_gauss': int(has_gauss), 'cached_gaussian': float(cached)},
            'torch_cpu': torch.get_rng_state(),
            'torch_cuda': [s for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []}


def restore_rng(state) -> None:
    py = state['python']
    random.setstate((py['version'], tuple(py['state']), py['gauss']))
    n = state['numpy']
    np.random.set_state((n['name'], n['keys'].numpy().astype(np.uint32), n['pos'], n['has_gauss'], n['cached_gaussian']))
    torch.set_rng_state(state['torch_cpu'])
    if state['torch_cuda']:
        torch.cuda.set_rng_state_all(state['torch_cuda'])


# ----------------------------------------------------------------------------- atomic files
def fsync_dir(path):
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1 << 22), b''):
            h.update(block)
    return h.hexdigest()


def atomic_save(payload, path) -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix='.tmp-', suffix='.pt')
    try:
        with os.fdopen(fd, 'wb') as fh:
            torch.save(payload, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        fsync_dir(path.parent)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return {'path': str(path), 'file_size_bytes': path.stat().st_size, 'sha256': file_sha256(path)}


def load(path):
    return torch.load(path, map_location='cpu', weights_only=True)


# ----------------------------------------------------------------------------- recovery (generator stage)
def recovery_payload(*, modules, optimizers, scalers, ema, position, provenance, identity_map_sha256, order):
    """modules: {'e_art','g_res','discriminator', 'attack_head'?, 'identity_head'?} -> state dicts.
    position: epoch, next_group (1-based group to run next), global_update (last completed), lr, curriculum."""
    rio.require(set(optimizers) == {'G_OPT', 'D_OPT'} and set(scalers) == {'G_SCALER', 'D_SCALER'}, 'G/D state')
    return {'kind': RECOVERY_KIND, 'labels': list(RECOVERY_LABELS),
            'modules': {k: v.state_dict() for k, v in modules.items() if v is not None},
            'optimizers': {k: v.state_dict() for k, v in optimizers.items()},
            'scalers': {k: v.state_dict() for k, v in scalers.items()},
            'ema': None if ema is None else {k: v.state_dict() for k, v in ema.items()},
            'position': primitives(position), 'order': primitives(order),
            'identity_map_sha256': identity_map_sha256, 'rng': rng_state(), 'provenance': primitives(provenance)}


def warmup_recovery_payload(*, e_art, attack_head, optimizer, scaler, position, provenance, order):
    return {'kind': WARMUP_RECOVERY_KIND, 'labels': list(RECOVERY_LABELS),
            'modules': {'e_art': e_art.state_dict(), 'attack_head': attack_head.state_dict()},
            'optimizers': {'WARMUP_OPT': optimizer.state_dict()}, 'scalers': {'WARMUP_SCALER': scaler.state_dict()},
            'position': primitives(position), 'order': primitives(order), 'rng': rng_state(),
            'provenance': primitives(provenance)}


def check_provenance(payload, expected):
    got = payload['provenance']
    for key, value in primitives(expected).items():
        rio.require(got.get(key) == value, f'resume refused: provenance {key} differs')


# ----------------------------------------------------------------------------- EMA candidates
def candidate_name(epoch):
    rio.require(epoch in rio.CANDIDATE_EPOCHS, 'EMA candidates exist for epochs 10..60 only')
    return f'ema_epoch_{epoch:02d}.pt'


def candidate_payload(*, e_art_ema, g_res_ema, metadata):
    meta = primitives(metadata)
    for key in ('method', 'seed', 'epoch', 'global_update', 'config_sha256', 'code_commit', 'gpu_env_lock_sha256',
                'source_manifest_sha256', 'teacher_sha256', 'ema_decay', 'gamma'):
        rio.require(key in meta, f'candidate metadata {key}')
    rio.require(not any('val' in k.lower() or 'test' in k.lower() for k in meta), 'no VAL/TEST field in a candidate')
    meta.update(selected=False, kind=CANDIDATE_KIND, ema_scope=['E_art', 'G_res'],
                excluded=['discriminator', 'attack_head', 'identity_head', 'live_weights', 'optimizer'])
    return {'kind': CANDIDATE_KIND, 'e_art_ema': e_art_ema.state_dict(), 'g_res_ema': g_res_ema.state_dict(),
            'metadata': meta}


def write_candidate(directory, epoch, payload) -> dict:
    path = Path(directory) / candidate_name(epoch)
    rio.require(not path.exists(), 'EMA candidates are immutable: ' + str(path))
    rio.require(set(payload) == {'kind', 'e_art_ema', 'g_res_ema', 'metadata'}, 'candidate payload keys')
    info = atomic_save(payload, path)
    return {**info, 'epoch': epoch, 'global_step': payload['metadata']['global_update'], 'checkpoint_type': 'periodic',
            'selected_for_final': False, 'selection_reason': CANDIDATE_SELECTION_REASON, 'role': 'ema_candidate'}
