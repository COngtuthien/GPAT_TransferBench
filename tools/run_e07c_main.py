#!/usr/bin/env python3
"""E07c MAIN DiffFAS SCIENTIFIC training CLI (CONTROLLED_ADAPTATION, DEV-021). NOT launched in M6D6j.

Hard-binds: experiment seed 42, 1337 or 2026; the A1 TRAIN relation (8838 rows); B=4, shuffle, drop_last=False,
0 workers; 400 epochs (884000 optimizer steps); AdamW / cycle scheduler / EMA from the pinned config; A7 FP32/TF32-off;
the owner-frozen auxiliary encoder via execution_policy.main_runner_encoder; checkpoints every 10000 steps plus the
A2-06 terminal at 884000 with M6D6iR retention; the pinned visualization every 1000 steps (owner D1, exact source);
terminal after the final visualization (owner D2); run_logging_v1 root <runtime_root>/runs/m6/E07c/seed_<seed>/.
There is no option to change any of them; override flags are refused before anything is imported. FRESH ONLY:
an existing run root is refused and there is no resume (MAIN_CHECKPOINT_RESUME is unqualified).

Launch (from a clean worktree on m6-baselines, gpat-m6-e07c):
  CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=<seed> NVIDIA_TF32_OVERRIDE=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 python tools/run_e07c_main.py --seed <seed> \
      --execution-config configs/execution/m5_gpu_3090.yaml
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from methods.common.learned import PreparationError  # noqa: E402
from methods.difffas import main_runner_io as mio  # noqa: E402

BRANCH = 'm6-baselines'


class Refused(SystemExit):
    """Fail-closed refusal (exit status 2); nothing was trained or written."""
    def __init__(self, message):
        print('E07c MAIN SCIENTIFIC RUN REFUSED: ' + message, file=sys.stderr)
        super().__init__(2)


def parse(argv):
    refused = mio.refused_arguments(argv)
    if refused:
        raise Refused('override/tuning options are not accepted: ' + ', '.join(refused) + ' (seed set, data, loop, '
                      'optimizer, precision, encoder, cadence, visualization, retention and resume are frozen)')
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                     allow_abbrev=False)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--execution-config', required=True,
                        help='infrastructure storage config (faces_256_root via m2b containment)')
    parser.add_argument('--preflight-only', action='store_true', help='verify every gate, import no Torch, train nothing')
    try:
        return parser.parse_args(argv)
    except SystemExit as exc:
        raise Refused('unrecognized or malformed arguments') from exc


def preconditions(args, *, environ=None, dirty=None, branch=None, check_interpreter=True, runtime_root=None):
    """Every static gate; returns the plan. Raises Refused on any failure. Imports no Torch."""
    environ = os.environ if environ is None else environ
    try:
        mio.validate_mode_seed(mio.SCIENTIFIC, args.seed)
    except PreparationError as exc:
        raise Refused(f'{exc} (qualification seed {mio.QUALIFICATION_SEED} is never scientific)') from exc
    from methods.common.runlog import git_commit, git_dirty
    if branch is None:
        branch = subprocess.run(['git', '-C', str(ROOT), 'branch', '--show-current'], capture_output=True,
                                text=True).stdout.strip()
    if branch != BRANCH:
        raise Refused(f'wrong branch/authority: {branch!r} (expected {BRANCH})')
    if (git_dirty() if dirty is None else dirty):
        raise Refused('git worktree is dirty; scientific runs require a clean committed worktree')
    if environ.get('PYTHONHASHSEED') != str(args.seed):
        raise Refused(f'launch with PYTHONHASHSEED={args.seed}')
    for key, value in mio.LAUNCH_ENVIRONMENT.items():
        if environ.get(key) != value:
            raise Refused(f'launch with {key}={value}')
    try:
        from methods.difffas import DiffFASAdapter
        from methods.difffas import execution_policy as ep
        from methods.difffas import main_runner
        from methods.difffas.aux_checkpoint import load_freeze_record
        contract = mio.load_contract()
        ids = mio.identities(contract)
        adapter = DiffFASAdapter()
        config = adapter.config
        if ep.load_policy(config)['sha256'] != ids['a7_overlay_sha256']:
            raise Refused('A7 overlay identity differs from the contract')
        load_freeze_record(config)
        source = adapter.validate_source()
        lock = json.loads((ROOT / 'environments/e07c.lock.json').read_text())
        if (source['commit'], source['tree']) != (ids['source_commit'], ids['source_tree']) or \
                source['commit'] != lock['source']['commit']:
            raise Refused('wrong source pin (differs from the contract / environment lock)')
        main_runner.verify_production_source(source)
        if check_interpreter and os.path.realpath(sys.executable) != os.path.realpath(lock['identity']['executable']):
            raise Refused('wrong environment: interpreter is not the locked gpat-m6-e07c python')
        storage = mio.faces_root_from_exec_config(os.path.relpath(Path(args.execution_config).resolve(), ROOT))
        if storage['exec_config_sha256'] != contract['bound_inputs_sha256'][mio.EXEC_CONFIG]:
            raise Refused('execution config is not the contract-bound ' + mio.EXEC_CONFIG)
    except PreparationError as exc:
        raise Refused(str(exc)) from exc
    run_dir = mio.run_root(runtime_root or storage['runtime_root'], mio.SCIENTIFIC, args.seed)
    if run_dir.exists():
        raise Refused(f'{run_dir} already exists; it is never overwritten and MAIN_CHECKPOINT_RESUME is unqualified')
    return {'method_id': mio.METHOD_ID, 'experiment_seed': args.seed, 'epochs': mio.EPOCHS,
            'total_iterations': mio.TOTAL_ITERATIONS, 'run_dir': str(run_dir), 'git_commit': git_commit(),
            'identities': ids, 'storage': storage, 'resume': None, 'visualization': 'OWNER D1: EXECUTE_EXACT_SOURCE',
            'terminal': 'OWNER D2: AFTER_FINAL_VISUALIZATION_BEFORE_RUN_COMPLETION',
            'retention': 'M6D6iR', 'torch_imported': 'torch' in sys.modules}


def main(argv=None):
    args = parse(sys.argv[1:] if argv is None else argv)
    plan = preconditions(args)
    if args.preflight_only:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    from methods.difffas import main_runner
    run_dir = main_runner.run_scientific(seed=args.seed, runtime_root=plan['storage']['runtime_root'],
                                         faces_root=plan['storage']['faces_256_root'],
                                         command_line=' '.join([sys.executable] + sys.argv))
    print(json.dumps({'status': 'completed', 'run_dir': str(run_dir)}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
