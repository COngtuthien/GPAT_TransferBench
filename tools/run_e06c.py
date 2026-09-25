#!/usr/bin/env python3
"""E06c DSDG-BIN-IDFREE SCIENTIFIC training CLI (CONTROLLED_ADAPTATION, DEV-020). NOT launched in M6D5e.

Hard-binds: method E06c, experiment seed in {42, 1337, 2026}, all_epochs 200, the frozen
TRAIN relation manifests/pairs_train_v1.parquet, GLOBAL_STATISTIC_PRESERVING_MICROBATCH_V2
(global 240, microbatch 20, drop_last=False) and <runtime_root>/runs/m6/E06c/seed_<seed>/.
There is no option to change epochs, learning rate, batch, microbatch, drop_last, precision
or checkpoint selection; such flags are refused before anything is imported or executed.
Every identity (git cleanliness, source pin, environment lock, pair manifest, LightCNN,
M6D5c/M6D5d overlays, M6D5e contract) is verified before Torch is imported. Resume is
explicit: --resume-state must name an indexed runner_state_epoch_<E>.pth in that seed root.

Launch (from a clean worktree, gpat-m6-e06c):
  CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=<seed> NVIDIA_TF32_OVERRIDE=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 python tools/run_e06c.py --seed <seed> \
      --execution-config configs/execution/m5_gpu_3090.yaml [--resume-state checkpoints/runner_state_epoch_<E>.pth]
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
from methods.dsdg import runner_io as rio  # noqa: E402

LAUNCH_ENVIRONMENT = {'NVIDIA_TF32_OVERRIDE': '0', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}


class Refused(SystemExit):
    """Fail-closed refusal (exit status 2); nothing was trained or written."""
    def __init__(self, message):
        print('E06c SCIENTIFIC RUN REFUSED: ' + message, file=sys.stderr)
        super().__init__(2)


def parse(argv):
    refused = rio.refused_arguments(argv)
    if refused:
        raise Refused('override/tuning options are not accepted: ' + ', '.join(refused) +
                      ' (epochs, lr, batch, microbatch, drop_last, precision and checkpoint selection are frozen)')
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                     allow_abbrev=False)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--execution-config', required=True,
                        help='infrastructure storage config (faces_256_root via m2b containment)')
    parser.add_argument('--resume-state', default=None, help='explicit checkpoints/runner_state_epoch_<E>.pth')
    parser.add_argument('--preflight-only', action='store_true', help='verify every gate, import no Torch, train nothing')
    try:
        return parser.parse_args(argv)
    except SystemExit as exc:
        raise Refused('unrecognized or malformed arguments') from exc


def preconditions(args, *, environ=None, dirty=None):
    """Every static gate; returns the plan. Raises Refused on any failure."""
    environ = os.environ if environ is None else environ
    try:
        rio.validate_mode_seed(rio.SCIENTIFIC, args.seed)
    except PreparationError as exc:
        raise Refused(f'{exc} (qualification seed {rio.QUALIFICATION_SEED} is never a scientific seed)') from exc
    from methods.common.runlog import git_commit, git_dirty
    if (git_dirty() if dirty is None else dirty):
        raise Refused('git worktree is dirty; scientific runs require a clean committed worktree')
    if environ.get('PYTHONHASHSEED') != str(args.seed):
        raise Refused(f'launch with PYTHONHASHSEED={args.seed}')
    for key, value in LAUNCH_ENVIRONMENT.items():
        if environ.get(key) != value:
            raise Refused(f'launch with {key}={value}')
    try:
        from methods.dsdg import DSDGAdapter
        from methods.dsdg import runner
        from methods.dsdg import runtime as m6d5a
        contract = rio.load_contract()               # config, A1, overlays, logging, lock, pair manifest SHA256
        ids = rio.identities(contract)
        overlays = runner.verify_overlays()          # M6D5c V1 + M6D5d V2 owner overlays
        adapter = DSDGAdapter()
        source = adapter.validate_source()
        lock = json.loads((ROOT / runner.LOCK_PATH).read_text())
        if (source['commit'], source['tree']) != (ids['source_commit'], ids['source_tree']) or \
                source['files_sha256'] != lock['source']['files_sha256']:
            raise Refused('source identity differs from the environment lock')
        lightcnn = runner.lightcnn_identity(m6d5a.LIGHTCNN)
        storage = rio.faces_root_from_exec_config(os.path.relpath(Path(args.execution_config).resolve(), ROOT))
        if storage['exec_config_sha256'] != contract['bound_inputs_sha256'][rio.EXEC_CONFIG]:
            raise Refused('execution config is not the contract-bound ' + rio.EXEC_CONFIG)
    except PreparationError as exc:
        raise Refused(str(exc)) from exc
    run_dir = rio.run_root(storage['runtime_root'], rio.SCIENTIFIC, args.seed)
    occupied = run_dir.exists() and any(run_dir.iterdir())
    if occupied and args.resume_state is None:
        raise Refused(f'{run_dir} already contains a run; resume only with an explicit --resume-state')
    if args.resume_state is not None:
        if not occupied:
            raise Refused('--resume-state given but the seed directory holds no run')
        try:
            path, entry = rio.resume_entry(run_dir, args.resume_state)
        except PreparationError as exc:
            raise Refused(str(exc)) from exc
    return {'method_id': rio.METHOD_ID, 'experiment_seed': args.seed, 'all_epochs': rio.ALL_EPOCHS,
            'run_dir': str(run_dir), 'git_commit': git_commit(), 'identities': ids, 'overlays': overlays,
            'lightcnn': lightcnn, 'storage': storage, 'execution_mode': rio.EXECUTION_MODE,
            'resume_state': args.resume_state, 'torch_imported': 'torch' in sys.modules}


def main(argv=None):
    args = parse(sys.argv[1:] if argv is None else argv)
    plan = preconditions(args)
    if args.preflight_only:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    from methods.dsdg import runner
    run_dir = runner.run_scientific(seed=args.seed, runtime_root=plan['storage']['runtime_root'],
                                    faces_root=plan['storage']['faces_256_root'], resume_state=args.resume_state,
                                    command_line=' '.join([sys.executable] + sys.argv))
    print(json.dumps({'status': 'completed', 'run_dir': str(run_dir)}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
