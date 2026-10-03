"""M7C4 GPAT production datasets and loader (torch).

Order authority is runner_io (metadata only): the per-epoch plan of index lists is fixed before any worker runs, so the
pair order is independent of num_workers. Workers only decode. Each item carries its relation index so the main process
logs every image access through runner_io.AccessLog; the worker-side CanonicalFaceReader can resolve only relation ids.

Decode contract: canonical 256x256 RGB PNG -> uint8 -> x = u8 / 127.5 - 1 (fp32, CHW). No augmentation of any kind.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from methods.common.learned import seed_torch_worker
from methods.gpat import runner_io as rio

FACE = 256


def decode(img) -> torch.Tensor:
    """PIL RGB 256x256 -> fp32 [3, 256, 256] in [-1, 1] (x = u8 / 127.5 - 1)."""
    arr = np.asarray(img, dtype=np.uint8)
    rio.require(arr.shape == (FACE, FACE, 3), 'canonical 256x256x3 face')
    return torch.from_numpy(arr.copy()).permute(2, 0, 1).to(torch.float32) / 127.5 - 1.0


class GeneratorPairs(Dataset):
    """Item = one relation row: x_source (spoof), x_target (live), attack index, identity label/valid, index."""

    def __init__(self, records, reader, identity=None):
        rio.require(len(records) == rio.TRAIN_ROWS, 'full TRAIN relation')
        self.records, self.reader = tuple(records), reader
        self.labels, self.valid = identity if identity is not None else ([-1] * len(records), [False] * len(records))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        return {'index': i, 'x_source': decode(self.reader(r['source_spoof_id'])),
                'x_target': decode(self.reader(r['target_live_id'])), 'attack': r['attack_index'],
                'identity': self.labels[i], 'identity_valid': bool(self.valid[i])}


class WarmupSources(Dataset):
    """B1/B3 attack warmup: the TRAIN source spoof frame of each relation row (8838) and its attack index only."""

    def __init__(self, records, reader):
        rio.require(len(records) == rio.TRAIN_ROWS, 'full TRAIN relation')
        self.records, self.reader = tuple(records), reader

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        return {'index': i, 'x_source': decode(self.reader(r['source_spoof_id'])), 'attack': r['attack_index']}


class PlanSampler(Sampler):
    """Yields the current epoch's fixed list of index lists (set by the runner before each epoch / resume)."""

    def __init__(self):
        self.plan = []

    def set_plan(self, batches):
        self.plan = [list(b) for b in batches]

    def __iter__(self):
        return iter([list(b) for b in self.plan])

    def __len__(self):
        return len(self.plan)


def make_loader(dataset, sampler, seed, *, num_workers=None, pin_memory=None):
    workers = rio.LOADER['num_workers'] if num_workers is None else num_workers
    kwargs = {'batch_sampler': sampler, 'num_workers': workers,
              'pin_memory': rio.LOADER['pin_memory'] if pin_memory is None else pin_memory,
              'worker_init_fn': seed_torch_worker, 'generator': torch.Generator().manual_seed(seed)}
    if workers > 0:
        kwargs.update(persistent_workers=rio.LOADER['persistent_workers'], prefetch_factor=rio.LOADER['prefetch_factor'])
    return DataLoader(dataset, **kwargs)
