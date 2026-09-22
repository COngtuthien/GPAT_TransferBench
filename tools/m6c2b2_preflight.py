#!/usr/bin/env python
"""Read-only E05 static preparation with benchmark-image/manifest open firewall."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.m6c2a_preflight import StaticAccessAudit


def run_preflight():
    from methods.pcgan import PCGANAdapter
    adapter = PCGANAdapter()
    plans = [adapter.build_training_plan(seed) for seed in adapter.config['seeds']['experiment_seeds']]
    return {'result': 'PASS', 'mode': 'STATIC_PREPARATION_ONLY', 'plans': plans,
            'benchmark_image_execution': False, 'model_inference': False,
            'training_launched': False, 'checkpoint_created': False,
            'production_runtime_write': False, 'bank_generated': False, 'gpu_job': False}


def main():
    audit = StaticAccessAudit()
    sys.addaudithook(audit)
    result = run_preflight()
    result['file_open_audit'] = audit.report()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
