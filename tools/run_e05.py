#!/usr/bin/env python3
"""E05 PCGAN (controlled architecture resolution) SCIENTIFIC training CLI.

CONTROLLED_ADAPTATION (A2-07 architecture resolution, A5 blur operator, M6D4c training-runner resolution); the
executable architecture basis is taesungp/swapping-autoencoder-pytorch@6baa180f. Not native/faithful PCGAN.

Hard-binds: method E05, experiment seed in {42, 1337, 2026}, 4000 benchmark iterations of one D then one G Adam
application (lr 1e-6, betas (0.9, 0.999)), batch 1 explicit pairs from manifests/pairs_train_v1.parquet (TRAIN only;
x_src = source_spoof_id, x_tgt = target_live_id), the five unit-weighted G losses, FP32 (no TF32/AMP/autocast,
cuDNN deterministic, no benchmark), checkpoint rule BASELINE_FINAL_STATE_V1 (terminal iteration 4000, selected within
the seed; no VAL/TEST selection) and <runtime_root>/runs/m6/E05/seed_<seed>/. FRESH ONLY: resume is not qualified for
E05, so an existing seed root is refused. There is no option to change iterations, batch, lr, betas, resolution, loss
weights, alpha/beta, precision, checkpoint selection, datasets/splits, pair manifest, output/run root, workers or to
enable qualification limits; such flags are refused before anything is imported. Every identity (clean git, frozen
config + snapshot + spec, A2/A5/M6D4c/M6D4e, source pin, E05 environment lock, pair manifest, faces root) is verified
before Torch is imported.

Launch (from a clean worktree, gpat-m6-e05, exact M6D4e launch environment):
  env PATH=/home/student20261/miniconda3/envs/gpat-m6-e05/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
      CUDA_VISIBLE_DEVICES=0 CUDA_HOME=/home/student20261/miniconda3/envs/gpat-m6-e05 \
      TORCH_EXTENSIONS_DIR=/home/student20261/workdir/GPAT_TransferBench_runtime/builds/e05_pcgan/torch_extensions \
      TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=2 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 NVIDIA_TF32_OVERRIDE=0 \
      CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=<seed> \
      python -u -B tools/run_e05.py --seed <seed> --execution-config configs/execution/m5_gpu_3090.yaml [--preflight-only]
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
BRANCH = 'm6-baselines'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from methods.common.learned import PreparationError  # noqa: E402
from methods.pcgan import production_io as pio  # noqa: E402


class Refused(SystemExit):
    """Fail-closed refusal (exit status 2); nothing was trained or written."""
    def __init__(self, message):
        print('E05 SCIENTIFIC RUN REFUSED: ' + message, file=sys.stderr)
        super().__init__(2)


def parse(argv):
    refused = pio.refused_arguments(argv)
    if refused:
        raise Refused('override/tuning/resume/qualification options are not accepted: ' + ', '.join(refused))
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                     allow_abbrev=False)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--execution-config', required=True,
                        help='infrastructure storage config (faces_256_root via m2b containment)')
    parser.add_argument('--preflight-only', action='store_true',
                        help='verify every gate, import no Torch, read no image, write nothing, train nothing')
    try:
        return parser.parse_args(argv)
    except SystemExit as exc:
        raise Refused('unrecognized or malformed arguments') from exc


def preconditions(args, *, environ=None, dirty=None, storage=None):
    """Every static gate; returns the plan. Raises Refused on any failure."""
    environ = os.environ if environ is None else environ
    try:
        pio.validate_mode_seed(pio.SCIENTIFIC, args.seed)
    except PreparationError as exc:
        raise Refused(f'{exc} (qualification seed {pio.QUALIFICATION_SEED} is never a scientific seed)') from exc
    from methods.common.runlog import git_commit, git_dirty
    if (git_dirty() if dirty is None else dirty):
        raise Refused('git worktree is dirty; scientific runs require a clean committed worktree')
    branch = subprocess.run(['git', '-C', str(ROOT), 'branch', '--show-current'], capture_output=True,
                            text=True).stdout.strip()
    if branch != BRANCH:
        raise Refused(f'scientific runs launch from branch {BRANCH}; got {branch!r}')
    errors = pio.launch_environment_errors(environ, args.seed)
    if errors:
        raise Refused('launch environment must be exactly: ' + ', '.join(errors))
    try:
        from methods.pcgan import PCGANAdapter
        config, contracts = pio.verify_contract()
        ids = pio.identities(config, contracts)
        source = pio.verify_source(PCGANAdapter(config))
        exec_rel = os.path.relpath(Path(args.execution_config).resolve(), ROOT)
        if exec_rel != pio.EXEC_CONFIG:
            raise Refused('execution config must be ' + pio.EXEC_CONFIG)
        storage = storage or pio.faces_root_from_exec_config(exec_rel)
        from methods.common.config import sha256_file
        if sha256_file(ROOT / pio.PAIR_MANIFEST) != pio.PAIR_MANIFEST_SHA:
            raise Refused('pair manifest SHA256 differs from the frozen value')
    except PreparationError as exc:
        raise Refused(str(exc)) from exc
    faces = Path(storage['faces_256_root'])
    if not faces.is_dir() or faces.is_symlink():
        raise Refused(f'canonical faces root {faces} must be an existing non-symlink directory')
    build = pio.build_root(storage['runtime_root'])
    if not (build / 'compatibility_patch_manifest.json').is_file():
        raise Refused(f'qualified E05 build root {build} is missing')
    run_dir = pio.run_root(storage['runtime_root'], pio.SCIENTIFIC, args.seed)
    if run_dir.exists():
        raise Refused(f'{run_dir} exists; E05 scientific runs are fresh only (resume is not qualified)')
    return {'method_id': pio.METHOD_ID, 'experiment_seed': args.seed, 'total_iterations': pio.TOTAL_ITERATIONS,
            'batch_size': pio.BATCH_SIZE, 'run_dir': str(run_dir), 'run_dir_exists': False,
            'git_commit': git_commit(), 'git_branch': branch, 'git_dirty': False, 'identities': ids,
            'source_commit': source['commit'], 'source_tree': source['tree'], 'storage': storage,
            'build_root': str(build), 'pair_manifest_sha256': pio.PAIR_MANIFEST_SHA,
            'checkpoint_iterations': pio.checkpoint_iterations(), 'checkpoint_rule': pio.CHECKPOINT_RULE,
            'launch_environment': dict(pio.LAUNCH_ENVIRONMENT, PYTHONHASHSEED=str(args.seed)),
            'fidelity_class': 'CONTROLLED_ADAPTATION', 'resume': 'NOT_QUALIFIED_FRESH_ONLY',
            'torch_imported': 'torch' in sys.modules, 'model_executed': False, 'images_read': 0}


def main(argv=None):
    args = parse(sys.argv[1:] if argv is None else argv)
    plan = preconditions(args)
    if args.preflight_only:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    from methods.pcgan import production_runner as pr
    run_dir = pr.run_scientific(seed=args.seed, runtime_root=plan['storage']['runtime_root'],
                                faces_root=plan['storage']['faces_256_root'],
                                command_line=' '.join([sys.executable] + sys.argv))
    print(json.dumps({'status': 'completed', 'run_dir': str(run_dir)}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
