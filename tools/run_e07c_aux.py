#!/usr/bin/env python3
"""E07c auxiliary conditioning-encoder SCIENTIFIC training CLI (A3 5.4b; CONTROLLED_ADAPTATION, DEV-021).

NOT launched in M6D6e. Hard-binds: method E07c, auxiliary_encoder_training_seed 42 (the only
accepted seed), 200 epochs, batch 256, shuffle, 6 workers, drop_last=True, SGD lr=0.002
momentum=0.9 weight_decay=5e-3, no scheduler, CrossEntropyLoss on the fourth output, the
frozen K7 TRAIN population of manifests/split_v1.parquet (14467 rows), A7 FP32 / TF32-off,
the whole-module epoch-boundary save and <runtime_root>/runs/m6/E07c/aux_encoder/seed_42/.
There is no option to change any of them; such flags are refused before anything is imported
or executed. Every identity (git branch/cleanliness, source pin, A3/A6/A7, environment lock
and interpreter, contract-bound inputs) is verified before Torch is imported. There is no
resume (AUX_RESUME_NOT_QUALIFIED): an existing seed_42 root is refused.

Launch (from a clean worktree on m6-baselines, gpat-m6-e07c):
  CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=42 NVIDIA_TF32_OVERRIDE=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 python tools/run_e07c_aux.py --seed 42 \
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
from methods.difffas import aux_runner_io as aio  # noqa: E402

BRANCH = 'm6-baselines'


class Refused(SystemExit):
    """Fail-closed refusal (exit status 2); nothing was trained or written."""
    def __init__(self, message):
        print('E07c AUXILIARY SCIENTIFIC RUN REFUSED: ' + message, file=sys.stderr)
        super().__init__(2)


def parse(argv):
    refused = aio.refused_arguments(argv)
    if refused:
        raise Refused('override/tuning options are not accepted: ' + ', '.join(refused) +
                      ' (seed, epochs, batch, workers, drop_last, optimizer, precision, labels, data source, '
                      'checkpoint selection and resume are frozen)')
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


def interpreter_identity(lock):
    return {'executable': sys.executable, 'expected': lock['identity']['executable'],
            'match': os.path.realpath(sys.executable) == os.path.realpath(lock['identity']['executable'])}


def preconditions(args, *, environ=None, dirty=None, branch=None, check_interpreter=True):
    """Every static gate; returns the plan. Raises Refused on any failure. Imports no Torch."""
    environ = os.environ if environ is None else environ
    try:
        aio.validate_mode_seed(aio.SCIENTIFIC, args.seed)
    except PreparationError as exc:
        raise Refused(f'{exc} (qualification seed {aio.QUALIFICATION_SEED} is never scientific)') from exc
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
    for key, value in aio.LAUNCH_ENVIRONMENT.items():
        if environ.get(key) != value:
            raise Refused(f'launch with {key}={value}')
    try:
        from methods.difffas import DiffFASAdapter
        from methods.difffas import execution_policy as ep
        from methods.difffas.aux_training_qualification import upstream_training_semantics
        contract = aio.load_contract()          # config, A1, A3, A6, A7, logging, lock, split, class map, exec config
        ids = aio.identities(contract)
        adapter = DiffFASAdapter()
        config = adapter.config
        policy = ep.load_policy(config)         # A7 overlay + A3/A6/source binding
        if policy['sha256'] != ids['a7_overlay_sha256']:
            raise Refused('A7 overlay identity differs from the contract')
        source = adapter.validate_source()
        lock = json.loads((ROOT / 'environments/e07c.lock.json').read_text())
        if (source['commit'], source['tree']) != (ids['source_commit'], ids['source_tree']) or \
                source['commit'] != lock['source']['commit'] or \
                any(source['files_sha256'].get(k) != v for k, v in lock['source']['files_sha256'].items()):
            raise Refused('wrong source pin (differs from the contract / environment lock)')
        semantics = upstream_training_semantics(source)
        if (semantics['loader']['batch_size'], semantics['loader']['shuffle'], semantics['loader']['num_workers'],
                semantics['loader']['drop_last'], semantics['epochs']) != \
                (aio.BATCH_SIZE, aio.SHUFFLE, aio.WORKERS, aio.DROP_LAST, aio.EPOCHS):
            raise Refused('pinned pretrain_classifier.py loader/epochs differ from the runner constants')
        aux = config['conditioning_encoder']
        if (aux['auxiliary_encoder_training_seed'], aux['auxiliary_encoder_training_runs'],
                aux['training_contract']['epochs'], aux['substitute_objective']['classes']) != \
                (aio.SCIENTIFIC_SEED, 1, aio.EPOCHS, list(aio.CLASSES)):
            raise Refused('A3 auxiliary contract differs from the runner constants')
        if check_interpreter and not interpreter_identity(lock)['match']:
            raise Refused('wrong environment: interpreter is not the locked gpat-m6-e07c python')
        storage = aio.faces_root_from_exec_config(os.path.relpath(Path(args.execution_config).resolve(), ROOT))
        if storage['exec_config_sha256'] != contract['bound_inputs_sha256'][aio.EXEC_CONFIG]:
            raise Refused('execution config is not the contract-bound ' + aio.EXEC_CONFIG)
    except PreparationError as exc:
        raise Refused(str(exc)) from exc
    run_dir = aio.run_root(storage['runtime_root'], aio.SCIENTIFIC, args.seed)
    if run_dir.exists():
        raise Refused(f'{run_dir} already exists; auxiliary resume is NOT qualified and the root is never overwritten')
    return {'method_id': aio.METHOD_ID, 'role': 'AUXILIARY_CONDITIONING_ENCODER',
            'auxiliary_encoder_training_seed': args.seed, 'epochs': aio.EPOCHS, 'run_dir': str(run_dir),
            'checkpoint': str(aio.checkpoint_path(run_dir)), 'git_commit': git_commit(), 'identities': ids,
            'storage': storage, 'resume': 'AUX_RESUME_NOT_QUALIFIED', 'torch_imported': 'torch' in sys.modules}


def main(argv=None):
    args = parse(sys.argv[1:] if argv is None else argv)
    plan = preconditions(args)
    if args.preflight_only:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    from methods.difffas import aux_runner
    run_dir = aux_runner.run_scientific(runtime_root=plan['storage']['runtime_root'],
                                        faces_root=plan['storage']['faces_256_root'],
                                        command_line=' '.join([sys.executable] + sys.argv))
    print(json.dumps({'status': 'completed', 'run_dir': str(run_dir)}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
