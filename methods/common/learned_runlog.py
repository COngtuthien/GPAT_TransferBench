"""Learned-method logging specialization; M6C1 runtime is retained unchanged."""
import json

from .learned import PreparationError, checkpoint_metadata
from .runlog import RunContext, atomic_write_json


class LearnedRunContext(RunContext):
    """Reuse atomic, append-only/resume semantics with explicit learned environment.

    The future runner must supply observed hardware/framework fields and null
    reasons. Static tests supply a labelled preparation environment in a temp dir.
    """
    def __init__(self, *, environment, missing_environment_reasons, **kwargs):
        super().__init__(**kwargs)
        if not self.config['learned_method']:
            raise PreparationError('LearnedRunContext requires a learned method')
        required = set(self._environment) - {'gpu_note'}
        required.add('tensorflow_version')
        if not required <= set(environment):
            raise PreparationError('explicit learned environment fields are missing')
        if any(v is None and not missing_environment_reasons.get(k) for k, v in environment.items()):
            raise PreparationError('null environment fields require reasons')
        self._validate_metadata(environment)
        self._environment = dict(environment)
        self._environment_reasons = dict(missing_environment_reasons)

    def _manifest(self, **kwargs):
        result = super()._manifest(**kwargs)
        result['missing_field_reasons'] = self._environment_reasons.copy()
        return result

    def _open_locked(self):
        result = super()._open_locked()
        path = self.path('checkpoint_index')
        index = json.loads(path.read_text())
        index['note'] = 'Only checkpoints actually written are indexed; preparation creates none.'
        atomic_write_json(path, index)
        return result

    def record_checkpoint(self, **kwargs):
        expected = checkpoint_metadata(self.config, self.seed, **{
            k: v for k, v in kwargs.items() if k != 'selection_reason'})
        if kwargs.get('selection_reason') != expected['selection_reason']:
            raise PreparationError('checkpoint reason disagrees with frozen official rule')
        return super().record_checkpoint(**expected)

    def log_epoch(self, record):
        def check(node):
            if isinstance(node, dict):
                if any('test' in str(k).lower() for k in node):
                    raise PreparationError('TEST fields are forbidden in learned trajectory')
                for value in node.values():
                    check(value)
            elif isinstance(node, (list, tuple)):
                for value in node:
                    check(value)
        check(record)
        return super().log_epoch(record)

    def close(self, *, summary=None, **kwargs):
        supplied = dict(summary or {})
        if ('checkpoint_selection_rule' in supplied and
                supplied['checkpoint_selection_rule'] != self.config['checkpoint']['rule']):
            raise PreparationError('summary cannot replace the frozen checkpoint rule')
        if supplied.get('test_split_accessed'):
            raise PreparationError('TEST is prohibited in learned runtime')
        reasons = dict(supplied.get('missing_field_reasons', {}))
        for key in ('final_or_selected_checkpoint_path', 'final_or_selected_checkpoint_sha256',
                    'seed_level_evaluation_metrics', 'training_duration_seconds', 'peak_vram_bytes',
                    'failure_reason', 'output_manifest_path'):
            if supplied.get(key) is None:
                reasons.setdefault(key, 'Not supplied by caller; no execution/evaluation result inferred')
        supplied['missing_field_reasons'] = reasons
        supplied.setdefault('peak_vram_reason', reasons.get('peak_vram_bytes'))
        self._validate_metadata(supplied)
        return super().close(summary=supplied, **kwargs)
