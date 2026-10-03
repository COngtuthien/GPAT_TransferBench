"""Frozen CLI `python -m gpatbench.cli train-generator --method GPAT-B<k> --seed <42|1337|2026>` (spec section 24).

SCIENTIFIC mode only. Every static gate runs before Torch is imported: frozen method, scientific seed (the
qualification seed 70404 is refused), branch m6-baselines, clean committed worktree, PYTHONHASHSEED = seed, the
deterministic launch environment, the locked gpat-m7-gpu interpreter, the containment-checked faces root, SHA-256
verified teacher/backbone assets, and a fresh run root (or an explicit --resume of an existing one). There is no
option that changes any scientific value. Qualification uses tools/m7c4_gpat_runner_qualification.py instead.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from methods.common.config import ROOT
from methods.common.learned import PreparationError
from methods.gpat import runner_io as rio

BRANCH = 'm6-baselines'
GPU_ENV_PREFIX = '/home/student20261/miniconda3/envs/gpat-m7-gpu'


class Refused(SystemExit):
    """Fail-closed refusal (exit status 2); nothing was trained or written."""

    def __init__(self, message):
        print('GPAT SCIENTIFIC RUN REFUSED: ' + message, file=sys.stderr)
        super().__init__(2)


def preconditions(method, seed, *, execution_config=None, asset_config=None, resume=False, environ=None,
                  dirty=None, branch=None, check_interpreter=True, verify_assets=True):
    environ = os.environ if environ is None else environ
    try:
        rio.variant_of(method)
        rio.validate_mode_seed(rio.SCIENTIFIC, seed)
    except PreparationError as exc:
        raise Refused(str(exc)) from exc
    from methods.common.runlog import git_commit, git_dirty
    if branch is None:
        branch = subprocess.run(['git', '-C', str(ROOT), 'branch', '--show-current'], capture_output=True,
                                text=True).stdout.strip()
    if branch != BRANCH:
        raise Refused(f'wrong branch {branch!r} (expected {BRANCH})')
    if git_dirty() if dirty is None else dirty:
        raise Refused('git worktree is dirty; scientific runs require a clean committed worktree')
    if environ.get('PYTHONHASHSEED') != str(seed):
        raise Refused(f'launch with PYTHONHASHSEED={seed}')
    for key, value in rio.LAUNCH_ENVIRONMENT.items():
        if environ.get(key) != value:
            raise Refused(f'launch with {key}={value}')
    if check_interpreter and not os.path.realpath(sys.executable).startswith(GPU_ENV_PREFIX + '/'):
        raise Refused('wrong environment: the interpreter is not the locked gpat-m7-gpu python')
    try:
        storage = rio.faces_root(execution_config)
        assets = rio.load_assets(asset_config, verify_bytes=verify_assets)
    except (PreparationError, OSError) as exc:
        raise Refused(str(exc)) from exc
    run_dir = rio.run_root(storage['runtime_root'], rio.SCIENTIFIC, method, seed)
    if run_dir.exists() and not resume:
        raise Refused(f'{run_dir} already exists; it is never overwritten (pass --resume to continue it)')
    if resume and not (run_dir / 'checkpoints' / 'recovery').is_dir():
        raise Refused(f'--resume needs an existing run root with recovery checkpoints: {run_dir}')
    return {'method': method, 'experiment_id': rio.variant_of(method)[1], 'experiment_seed': seed,
            'run_dir': str(run_dir), 'git_commit': git_commit(), 'storage': storage, 'assets': assets,
            'resume': resume, 'torch_imported': 'torch' in sys.modules}


def main(args) -> int:
    plan = preconditions(args.method, args.seed, execution_config=args.execution_config,
                         asset_config=args.asset_config, resume=args.resume)
    if args.preflight_only:
        print(json.dumps({k: v for k, v in plan.items() if k != 'assets'}, indent=2, sort_keys=True))
        return 0
    from methods.gpat import runner
    run_dir = runner.run_scientific(method=args.method, seed=args.seed, runtime_root=plan['storage']['runtime_root'],
                                    faces_root=plan['storage']['faces_256_root'], assets=plan['assets'],
                                    resume=args.resume)
    print(json.dumps({'status': 'finished', 'run_dir': str(run_dir)}))
    return 0


def add_parser(sub):
    p = sub.add_parser('train-generator', help='M7: GPAT-B0..B3 scientific generator training (seeds 42/1337/2026)')
    p.add_argument('--method', required=True, choices=sorted(rio.VARIANTS))
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--execution-config', default=rio.EXEC_CONFIG)
    p.add_argument('--asset-config', default=rio.ASSET_CONFIG)
    p.add_argument('--resume', action='store_true', help='continue an existing run root from its recovery checkpoint')
    p.add_argument('--preflight-only', action='store_true', help='verify every gate; import no Torch; train nothing')
    p.set_defaults(func=main)
    return p
