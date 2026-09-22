#!/usr/bin/env python
"""Read-only E04 preparation; no reconstruction, inference, training or outputs on disk."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run_preflight():
    from methods.physics_std import PhysicsSTDAdapter
    adapter = PhysicsSTDAdapter()
    plans = [adapter.build_training_plan(seed) for seed in adapter.config['seeds']['experiment_seeds']]
    return {'result': 'PASS', 'mode': 'STATIC_PREPARATION_ONLY', 'plans': plans,
            'training_launched': False, 'model_inference': False, 'gpu_job': False,
            'checkpoint_created': False, 'production_runtime_write': False,
            'environment_freeze_pending': True}


def main():
    import json
    from tools.m6c2a_preflight import StaticAccessAudit
    audit = StaticAccessAudit()
    sys.addaudithook(audit)
    result = run_preflight()
    result['file_open_audit'] = audit.report()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
