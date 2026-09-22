#!/usr/bin/env python
"""Read-only M6C2a source/asset/config preparation. Never launches training.

All outputs go to stdout. Redirect to /tmp if retaining a local report. The open
hook refuses benchmark data and images, including manifests, before imports.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class StaticAccessAudit:
    def __init__(self):
        self.opens = 0
        self.benchmark_image_opens = 0
        self.manifest_opens = 0
        self.rejected_data_opens = 0

    def __call__(self, event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes)):
            return
        import os
        path = Path(os.fsdecode(args[0])).absolute()
        self.opens += 1
        suffix = path.suffix.lower()
        image = suffix in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp', '.mp4', '.avi', '.mov'}
        manifest = path.is_relative_to(ROOT / 'manifests')
        data = (suffix in {'.parquet', '.npy', '.npz', '.h5', '.hdf5'} or
                path.is_relative_to(ROOT / 'data') or path.is_relative_to(ROOT / 'cache') or
                ('GPAT_TransferBench_runtime' in path.parts and
                 any(p in path.parts for p in ('data', 'cache', 'runs', 'banks'))))
        if image or manifest or data:
            self.benchmark_image_opens += int(image)
            self.manifest_opens += int(manifest)
            self.rejected_data_opens += 1
            raise RuntimeError(f'static preflight refuses data/image/manifest open: {path}')

    def report(self):
        return vars(self).copy()


def run_preflight():
    from methods.common.config import load_logging_contract
    from methods.stdn import STDNAdapter
    from methods.dsdg import DSDGAdapter
    logging = load_logging_contract()
    plans = []
    for cls in (STDNAdapter, DSDGAdapter):
        adapter = cls()
        for seed in adapter.config['seeds']['experiment_seeds']:
            plans.append(adapter.build_training_plan(seed))
    return {'result': 'PASS', 'mode': 'STATIC_PREPARATION_ONLY',
            'logging_contract': logging['version'], 'plans': plans,
            'training_launched': False, 'gpu_job': False, 'checkpoint_created': False,
            'production_runtime_write': False, 'environment_freeze_pending': True}


def main():
    import json
    audit = StaticAccessAudit()
    sys.addaudithook(audit)
    result = run_preflight()
    result['file_open_audit'] = audit.report()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
