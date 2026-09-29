#!/usr/bin/env python3
"""E06b DSDG-NATIVE SCIENTIFIC training CLI.

Target FAITHFUL_OFFICIAL; runtime fidelity FAITHFUL_OFFICIAL_WITH_RUNTIME_COMPATIBILITY (M6F-C owner decision).

NOT launched in M6F-D. Hard-binds: method E06b, experiment seed in {42, 1337, 2026}, all_epochs 200, the native
CASIA+MSU TRAIN relation from split_v1 (TRAIN filter, allowlisted columns), K = 2 (print 0 / replay 1),
lambda_pair = 5, GLOBAL_STATISTIC_PRESERVING_MICROBATCH_V2 (global 240, microbatch 20, drop_last=False, 15 x 240 + 120)
and <runtime_root>/runs/m6/E06b/seed_<seed>/. FRESH ONLY: resume is not qualified for E06b, so an existing seed
root is refused. There is no option to change epochs, lr, batch, microbatch, drop_last, precision, K, datasets,
lambdas, checkpoint selection or to enable qualification limits; such flags are refused before anything is imported.
Every identity (clean git, frozen config + snapshot, M6F-A contract, M6F-C runtime qualification, source pin,
DSDG environment lock, LightCNN, faces root) is verified before Torch is imported.

Launch (from a clean worktree, gpat-m6-e06c):
  CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=<seed> NVIDIA_TF32_OVERRIDE=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 python tools/run_e06b.py --seed <seed> \
      --execution-config configs/execution/m5_gpu_3090.yaml [--preflight-only]
"""
import argparse
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from methods.common.learned import PreparationError  # noqa: E402
from methods.dsdg import native_runner as nr  # noqa: E402

LAUNCH_ENVIRONMENT = {'NVIDIA_TF32_OVERRIDE': '0', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}


class Refused(SystemExit):
    """Fail-closed refusal (exit status 2); nothing was trained or written."""
    def __init__(self, message):
        print('E06b SCIENTIFIC RUN REFUSED: ' + message, file=sys.stderr)
        super().__init__(2)


def parse(argv):
    refused = nr.refused_arguments(argv)
    if refused:
        raise Refused('override/tuning/resume/qualification options are not accepted: ' + ', '.join(refused))
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                     allow_abbrev=False)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--execution-config', required=True,
                        help='infrastructure storage config (faces_256_root via m2b containment)')
    parser.add_argument('--preflight-only', action='store_true',
                        help='verify every gate, import no Torch, train nothing')
    try:
        return parser.parse_args(argv)
    except SystemExit as exc:
        raise Refused('unrecognized or malformed arguments') from exc


def preconditions(args, *, environ=None, dirty=None):
    """Every static gate; returns the plan. Raises Refused on any failure."""
    environ = os.environ if environ is None else environ
    try:
        nr.validate_mode_seed(nr.SCIENTIFIC, args.seed)
    except PreparationError as exc:
        raise Refused(f'{exc} (qualification seed {nr.QUALIFICATION_SEED} is never a scientific seed)') from exc
    from methods.common.runlog import git_commit, git_dirty
    if (git_dirty() if dirty is None else dirty):
        raise Refused('git worktree is dirty; scientific runs require a clean committed worktree')
    if environ.get('PYTHONHASHSEED') != str(args.seed):
        raise Refused(f'launch with PYTHONHASHSEED={args.seed}')
    for key, value in LAUNCH_ENVIRONMENT.items():
        if environ.get(key) != value:
            raise Refused(f'launch with {key}={value}')
    try:
        from methods.dsdg import native
        from methods.dsdg import runner_io as rio
        from methods.dsdg import runtime as m6d5a
        adapter = native.DSDGNativeAdapter()
        config = nr.verify_contract(adapter.config)
        ids = nr.identities(config)
        source = nr.verify_source(adapter, ids)
        lightcnn = nr.lightcnn_identity(m6d5a.LIGHTCNN)
        exec_rel = os.path.relpath(Path(args.execution_config).resolve(), ROOT)
        if exec_rel != nr.EXEC_CONFIG:
            raise Refused('execution config must be ' + nr.EXEC_CONFIG)
        storage = rio.faces_root_from_exec_config(exec_rel)
    except PreparationError as exc:
        raise Refused(str(exc)) from exc
    run_dir = nr.run_root(storage['runtime_root'], nr.SCIENTIFIC, args.seed)
    if run_dir.exists():
        raise Refused(f'{run_dir} exists; E06b scientific runs are fresh only (resume is not qualified)')
    return {'method_id': nr.METHOD_ID, 'experiment_seed': args.seed, 'all_epochs': nr.ALL_EPOCHS,
            'run_dir': str(run_dir), 'git_commit': git_commit(),
            'identities': {k: v for k, v in ids.items() if k != 'source_files_sha256'},
            'source_commit': source['commit'], 'lightcnn': lightcnn, 'storage': storage,
            'execution_mode': nr.EXECUTION_MODE, 'runtime_fidelity_assessment': nr.RUNTIME_FIDELITY,
            'resume': 'NOT_QUALIFIED_FRESH_ONLY', 'torch_imported': 'torch' in sys.modules}


def main(argv=None):
    args = parse(sys.argv[1:] if argv is None else argv)
    plan = preconditions(args)
    if args.preflight_only:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    run_dir = nr.run_scientific(seed=args.seed, runtime_root=plan['storage']['runtime_root'],
                                faces_root=plan['storage']['faces_256_root'],
                                command_line=' '.join([sys.executable] + sys.argv))
    print(json.dumps({'status': 'completed', 'run_dir': str(run_dir)}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
