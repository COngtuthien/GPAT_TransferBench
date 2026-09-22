#!/usr/bin/env python
"""Read-only E07c plans, no model/manifest loading, weights, or runtime writes."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.m6c2a_preflight import StaticAccessAudit


def run_preflight():
    from methods.difffas import DiffFASAdapter
    adapter = DiffFASAdapter()
    auxiliary = adapter.build_auxiliary_plan()
    plans = [adapter.build_training_plan(seed) for seed in adapter.config['seeds']['experiment_seeds']]
    assert all(p['auxiliary_checkpoint'] == auxiliary['checkpoint'] for p in plans)
    return {'result': 'PASS', 'mode': 'STATIC_PREPARATION_ONLY', 'auxiliary_plan': auxiliary,
            'main_plans': plans, 'auxiliary_runs': auxiliary['runs'],
            'same_auxiliary_checkpoint_for_all_main_seeds': True,
            'model_constructed': False, 'training_launched': False, 'diffusion_sampling': False,
            'checkpoint_created': False, 'production_runtime_write': False, 'gpu_job': False}


def main():
    audit = StaticAccessAudit()
    sys.addaudithook(audit)
    result = run_preflight()
    result['file_open_audit'] = {**audit.report(), 'manifest_image_opens': audit.benchmark_image_opens}
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == '__main__':
    main()
