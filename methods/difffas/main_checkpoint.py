"""M6D6j E07c MAIN checkpoint store: safe write -> durable index -> verify -> M6D6iR successor-gated prune.

ONE implementation shared by scientific runs and qualification. It writes the exact source payload it is given
(FAS_train.py:105-114, built by main_runner.checkpoint_payload) with the given serializer (torch.save in production)
and never alters it: no weights-only, optimizer stripping, dtype change or compression.

Save transition (fail closed; any failure raises CheckpointStop = STOP_AND_REPORT):
    free-space gate (predecessor still retained) -> serialize to <exact path>.partial (O_EXCL) -> flush + fsync ->
    os.replace to the exact final path -> fsync directory -> stat -> SHA256 -> append the checkpoint_index.json
    entry durably (atomic write + fsync) -> re-read the index and re-hash the file -> mark index_verified durably.
Retention (configs/amendments/e07c_m6d6ir_checkpoint_retention.yaml):
    only after the successor entry is index_verified may its DIRECT predecessor be pruned, and only if that predecessor
    is periodic, non-selected, non-final, non-protected, present, at its exact recorded path, with its recorded size
    and SHA256. Removal is os.unlink of that one exact path; the index record is kept and gains bytes_present=false,
    bytes_pruned=true, pruned_utc, prune_reason, successor global_step and SHA256. No glob, no enumeration, no
    keep-latest-N, no background cleanup. The terminal checkpoint is protected and can never be pruned.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

from methods.common.learned import PreparationError
from methods.common.runlog import atomic_write_json

PAYLOAD_KEYS = ('model', 'ema', 'scheduler', 'optimizer', 'conf')
INDEX_FIELDS = ('path', 'epoch', 'global_step', 'file_size_bytes', 'sha256', 'checkpoint_type', 'selected_for_final',
                'selection_reason')
PERIODIC_REASON = 'Official cadence; not final selection'           # methods/common/learned.checkpoint_metadata
TERMINAL_REASON = 'BASELINE_FINAL_STATE_V1'
PROTECTED_ROLES = ('terminal', 'selected', 'authoritative_final', 'officially_required')
PRUNE_REASON = 'EXPLICIT_E07C_PERIODIC_RETENTION_POLICY'
RESERVE_BYTES = 1 << 30                  # headroom kept free beyond the successor's estimated size
HASH_CHUNK = 1 << 22


class CheckpointStop(RuntimeError):
    """Save / hash / index / verify / prune failure: STOP_AND_REPORT; scientific state is not changed further."""


class StorageStop(CheckpointStop):
    """Not enough space to create the successor while the predecessor is retained (never deleted early)."""


class ProtectedCheckpoint(CheckpointStop):
    """Refusal to prune a protected (terminal / selected / final / officially required) checkpoint."""


def stop(ok, message, kind=CheckpointStop):
    if not ok:
        raise kind('E07c checkpoint STOP_AND_REPORT: ' + message)


def checkpoint_name(iters):
    """FAS_train.py:113 f"/model_{str(iters).zfill(6)}.pt" (basename)."""
    return f"model_{str(iters).zfill(6)}.pt"


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(HASH_CHUNK), b''):
            h.update(block)
    return h.hexdigest()


def fsync_dir(path):
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


class CheckpointStore:
    """checkpoints/ + checkpoint_index.json of ONE run root (scientific or qualification).

    validate(meta) checks run_logging_v1 metadata before an entry is written (scientific mode passes
    methods.common.learned.checkpoint_metadata for the experiment seed). Fault hooks exist only for tests.
    """
    def __init__(self, run_dir, *, validate, serializer, labels=(), reserve_bytes=RESERVE_BYTES, faults=None):
        self.run_dir = Path(run_dir)
        self.ckpt_dir = self.run_dir / 'checkpoints'
        self.index_path = self.run_dir / 'checkpoint_index.json'
        stop(self.ckpt_dir.is_dir() and not self.ckpt_dir.is_symlink() and self.index_path.is_file(),
             'run root must be opened (checkpoints/ + checkpoint_index.json)')
        self.validate, self.serializer, self.labels = validate, serializer, list(labels)
        self.reserve_bytes, self.faults = reserve_bytes, dict(faults or {})
        self.events = []

    # ------------------------------------------------------------- index
    def read_index(self):
        return json.loads(self.index_path.read_text())

    def _write_index(self, index):
        if self.faults.get('index'):
            raise OSError('injected index failure')
        atomic_write_json(self.index_path, index)

    def entry(self, rel):
        matches = [e for e in self.read_index()['checkpoints'] if e['path'] == rel]
        stop(len(matches) == 1, 'exactly one index record for ' + rel)
        return matches[0]

    def exact_path(self, rel):
        stop(rel == 'checkpoints/' + Path(rel).name and Path(rel).name.startswith('model_') and rel.endswith('.pt'),
             'recorded path is checkpoints/model_<step>.pt: ' + rel)
        path = self.run_dir / rel
        stop(path.parent == self.ckpt_dir and not path.is_symlink(), 'exact recorded path inside checkpoints/')
        return path

    # ------------------------------------------------------------- save transition
    def write(self, *, global_step, epoch, kind, payload, estimated_bytes, extra=None):
        """Create, close, stat, hash, index and verify ONE checkpoint; returns the verified index entry."""
        stop(kind in ('periodic', 'terminal') and type(global_step) is int and global_step > 0, 'checkpoint kind/step')
        stop(tuple(payload) == PAYLOAD_KEYS, 'exact source payload keys model, ema, scheduler, optimizer, conf')
        rel = 'checkpoints/' + checkpoint_name(global_step)
        path = self.exact_path(rel)
        partial = path.with_name(path.name + '.partial')
        stop(not path.exists() and not partial.exists(), 'checkpoint path unused: ' + rel)
        stop(rel not in [e['path'] for e in self.read_index()['checkpoints']], 'no prior index record for ' + rel)
        free = shutil.disk_usage(self.ckpt_dir).free
        stop(free >= estimated_bytes + self.reserve_bytes,
             f'insufficient space for the successor ({free} free < {estimated_bytes} + {self.reserve_bytes} reserve); '
             'the predecessor is never deleted early', StorageStop)
        t0 = time.monotonic()
        try:
            with open(partial, 'xb') as fh:
                if self.faults.get('save'):
                    raise OSError('injected save failure')
                self.serializer(payload, fh)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(partial, path)
            fsync_dir(self.ckpt_dir)
        except BaseException as exc:
            if partial.exists():
                partial.unlink()        # the incomplete successor only; never another checkpoint
            raise CheckpointStop(f'E07c checkpoint STOP_AND_REPORT: write failed for {rel}: {exc!r}') from exc
        write_seconds = time.monotonic() - t0
        size = path.stat().st_size
        t1 = time.monotonic()
        try:
            if self.faults.get('hash'):
                raise OSError('injected hash failure')
            digest = file_sha256(path)
        except OSError as exc:
            raise CheckpointStop(f'E07c checkpoint STOP_AND_REPORT: hash failed for {rel}: {exc!r}') from exc
        hash_seconds = time.monotonic() - t1
        selected = kind == 'terminal'
        meta = self.validate(dict(path=rel, epoch=epoch, global_step=global_step, file_size_bytes=size, sha256=digest,
                                  checkpoint_type=kind, selected_for_final=selected))
        stop(tuple(meta) == INDEX_FIELDS and meta['selection_reason'] ==
             (TERMINAL_REASON if selected else PERIODIC_REASON), 'run_logging_v1 checkpoint metadata')
        entry = dict(meta, bytes_present=True, bytes_pruned=False, index_verified=False,
                     logical_roles=list(PROTECTED_ROLES) if selected else [],
                     payload_keys=list(PAYLOAD_KEYS), serialization='torch.save(payload, file) of the exact source '
                     'payload (FAS_train.py:105-114) to <path>.partial, fsync, os.replace', written_utc=_utc(),
                     write_seconds=round(write_seconds, 3), hash_seconds=round(hash_seconds, 3),
                     **({'labels': self.labels} if self.labels else {}), **(extra or {}))
        index = self.read_index()
        index['checkpoints'].append(entry)
        try:
            self._write_index(index)
        except OSError as exc:
            raise CheckpointStop(f'E07c checkpoint STOP_AND_REPORT: index write failed for {rel}: {exc!r}') from exc
        self.verify(rel)
        self.events.append({'event': 'checkpoint_verified', 'path': rel, 'global_step': global_step, 'kind': kind})
        return self.entry(rel)

    def verify(self, rel):
        """Re-read the durable index and re-hash the bytes; then record index_verified durably."""
        e = self.entry(rel)
        path = self.exact_path(rel)
        verified = (not self.faults.get('verify') and path.is_file() and path.stat().st_size == e['file_size_bytes']
                    and file_sha256(path) == e['sha256'] and e['bytes_present'] is True)
        stop(verified, 'SHA256/index verification failed for ' + rel)
        index = self.read_index()
        for item in index['checkpoints']:
            if item['path'] == rel:
                item.update(index_verified=True, verified_utc=_utc())
        self._write_index(index)
        return self.entry(rel)

    # ------------------------------------------------------------- retention (M6D6iR)
    @staticmethod
    def protected(e):
        return (e['checkpoint_type'] != 'periodic' or e['selected_for_final'] or
                bool(set(e.get('logical_roles', [])) & set(PROTECTED_ROLES)))

    def predecessor(self, successor_rel):
        entries = self.read_index()['checkpoints']
        position = [i for i, e in enumerate(entries) if e['path'] == successor_rel]
        stop(len(position) == 1, 'successor is indexed')
        return entries[position[0] - 1] if position[0] > 0 else None

    def prune(self, predecessor_rel, successor_rel):
        """Remove the bytes of ONE exact predecessor after its direct successor is verified; keep its record."""
        succ = self.entry(successor_rel)
        pred = self.entry(predecessor_rel)
        if self.protected(pred):
            raise ProtectedCheckpoint('E07c checkpoint refusal: protected checkpoint is never pruned: ' + predecessor_rel)
        stop(succ['index_verified'] is True and succ['bytes_present'] is True, 'successor verified before any prune')
        direct = self.predecessor(successor_rel)
        stop(direct is not None and direct['path'] == predecessor_rel and pred['global_step'] < succ['global_step'],
             'only the DIRECT predecessor of the verified successor may be pruned')
        stop(pred['bytes_present'] is True and pred['bytes_pruned'] is False, 'predecessor bytes are present')
        path = self.exact_path(predecessor_rel)
        stop(path.is_file() and path.stat().st_size == pred['file_size_bytes'] and file_sha256(path) == pred['sha256'],
             'predecessor bytes match their record before removal')
        try:
            if self.faults.get('prune'):
                raise OSError('injected prune failure')
            os.unlink(path)
            fsync_dir(self.ckpt_dir)
        except OSError as exc:
            raise CheckpointStop(f'E07c checkpoint STOP_AND_REPORT: prune failed for {predecessor_rel}: {exc!r}; the '
                                 'index still records its bytes as present') from exc
        index = self.read_index()
        for item in index['checkpoints']:
            if item['path'] == predecessor_rel:
                item.update(bytes_present=False, bytes_pruned=True, pruned_utc=_utc(), prune_reason=PRUNE_REASON,
                            successor_checkpoint_global_step=succ['global_step'],
                            successor_checkpoint_sha256=succ['sha256'])
        self._write_index(index)
        event = {'event': 'checkpoint_pruned', 'path': predecessor_rel, 'global_step': pred['global_step'],
                 'sha256': pred['sha256'], 'file_size_bytes': pred['file_size_bytes'],
                 'successor_path': successor_rel, 'successor_global_step': succ['global_step'],
                 'successor_sha256': succ['sha256'], 'prune_reason': PRUNE_REASON}
        self.events.append(event)
        return event

    def transition(self, successor_rel):
        """Successor-gated retention step: prune the direct predecessor if it is an eligible periodic checkpoint."""
        pred = self.predecessor(successor_rel)
        if pred is None or self.protected(pred) or pred['bytes_present'] is not True:
            return None
        return self.prune(pred['path'], successor_rel)

    # ------------------------------------------------------------- qualification-only cleanup
    def remove_qualification_bytes(self, rel):
        """Qualification cleanup policy (NOT the M6D6iR scientific retention policy): labelled roots only."""
        stop('QUALIFICATION_ONLY' in self.labels, 'qualification cleanup is refused outside a qualification root')
        e = self.entry(rel)
        path = self.exact_path(rel)
        stop(e['bytes_present'] and path.is_file() and file_sha256(path) == e['sha256'], 'cleanup of recorded bytes')
        os.unlink(path)
        fsync_dir(self.ckpt_dir)
        index = self.read_index()
        for item in index['checkpoints']:
            if item['path'] == rel:
                item.update(bytes_present=False, qualification_bytes_removed=True, qualification_removed_utc=_utc(),
                            qualification_cleanup_reason='QUALIFICATION_CLEANUP_POLICY_NOT_M6D6IR_RETENTION')
        self._write_index(index)
        return {'event': 'qualification_bytes_removed', 'path': rel, 'sha256': e['sha256'],
                'file_size_bytes': e['file_size_bytes']}


def qualification_validator(iterations_per_epoch, final_epoch, terminal_step):
    """The same rules as methods.common.learned.checkpoint_metadata for E07c, without an experiment seed."""
    def validate(meta):
        kind, selected, epoch, step = (meta['checkpoint_type'], meta['selected_for_final'], meta['epoch'],
                                       meta['global_step'])
        if (type(epoch) is not int or not 1 <= epoch <= final_epoch or epoch != step // iterations_per_epoch or
                (selected or kind == 'terminal') and (epoch != final_epoch or step != terminal_step) or
                type(meta['file_size_bytes']) is not int or meta['file_size_bytes'] <= 0 or len(meta['sha256']) != 64):
            raise PreparationError('checkpoint metadata outside the frozen E07c rule')
        return dict(meta, selection_reason=TERMINAL_REASON if selected else PERIODIC_REASON)
    return validate


def scientific_validator(config, seed, iterations_per_epoch):
    from methods.common.learned import checkpoint_metadata

    def validate(meta):
        if meta['epoch'] != meta['global_step'] // iterations_per_epoch:
            raise PreparationError('epoch = completed epochs at the checkpoint global step')
        return checkpoint_metadata(config, seed, **meta)
    return validate
