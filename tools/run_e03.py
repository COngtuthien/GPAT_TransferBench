#!/usr/bin/env python3
"""E03 STDN SCIENTIFIC training CLI (Python 3.8, gpat-m6-e03, NVIDIA TF 1.15.5).

Execution fidelity: FAITHFUL_OFFICIAL_CORE_WITH_TRACK_A_CONTROLLED_DATA_ADAPTATION_AND_RUNTIME_COMPATIBILITY
(official STDN graph/losses/optimizer/schedule unchanged; Track-A pair relations, M2 faces/landmarks feed pack and
independent TRAIN/VAL data RNG per the owner rulings; M6D2b runtime compatibility layer).

Hard-binds: method E03, experiment seed in {42, 1337, 2026}, 50 epochs x (2000 TRAIN steps -- step % 2 == 0 runs
losses + [g_op, d_op, fig] in ONE sess.run, otherwise losses + [g_op, fig] --, saver.save ckpt-<epoch>, 500 VAL
diagnostic steps), batch 2 pairs from pairs_train_v1 (TRAIN) / val_pairs_v1 (VAL) through the verified E03 feed pack,
checkpoint rule OFFICIAL_LATEST_FINAL_CKPT_50 (no VAL/TEST selection) and <runtime_root>/runs/m6/E03/seed_<seed>/.
FRESH ONLY: an existing seed root is refused; there is no resume and no Saver.restore. No option changes epochs, steps,
VAL steps, batch, G:D ratio, lr/decay/optimizer, augmentation, relations, split, pack, checkpoint rule, output root,
fidelity or qualification limits; such flags are refused before anything is imported. Every identity (clean git,
frozen config + snapshot, owner rulings, runtime addendum, lock, source cache, feed pack manifest, faces root) is
verified before TensorFlow is imported.

Launch (clean worktree, exact M6D2b precision environment):
  env PATH=/home/student20261/miniconda3/envs/gpat-m6-e03/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
      CUDA_VISIBLE_DEVICES=0 NVIDIA_TF32_OVERRIDE=0 TF_ENABLE_AUTO_MIXED_PRECISION=0 \
      TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE=0 TF_ENABLE_CUBLAS_TENSOR_OP_MATH_FP32=0 \
      TF_ENABLE_CUDNN_TENSOR_OP_MATH_FP32=0 TF_ENABLE_CUDNN_RNN_TENSOR_OP_MATH_FP32=0 TF_XLA_FLAGS=--tf_xla_auto_jit=0 \
      TF_DETERMINISTIC_OPS=1 TF_CUDNN_DETERMINISTIC=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 TF_ENABLE_ONEDNN_OPTS=0 \
      PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONHASHSEED=<seed> \
      python -u -B tools/run_e03.py --seed <seed> --execution-config configs/execution/m5_gpu_3090.yaml [--preflight-only]
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

from methods.stdn import production_io as pio  # noqa: E402


class Refused(SystemExit):
    """Fail-closed refusal (exit status 2); nothing was trained or written."""
    def __init__(self, message):
        print('E03 SCIENTIFIC RUN REFUSED: ' + message, file=sys.stderr)
        super().__init__(2)


def parse(argv):
    refused = pio.refused_arguments(argv)
    if refused:
        raise Refused('override/tuning/resume/qualification options are not accepted: ' + ', '.join(refused))
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                     allow_abbrev=False)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--execution-config', required=True, help='infrastructure storage config (faces root)')
    parser.add_argument('--preflight-only', action='store_true',
                        help='verify every gate; import no TensorFlow, decode no image, load no landmark array, '
                             'create no run root, execute no graph')
    try:
        return parser.parse_args(argv)
    except SystemExit as exc:
        raise Refused('unrecognized or malformed arguments') from exc


def preconditions(args, *, environ=None, dirty=None, storage=None):
    """Every static gate; returns the plan. Raises Refused on any failure."""
    environ = os.environ if environ is None else environ
    try:
        pio.validate_mode_seed(pio.SCIENTIFIC, args.seed)
    except pio.E03Stop as exc:
        raise Refused('%s (qualification seed %d is never a scientific seed)' % (exc, pio.QUALIFICATION_SEED)) from exc
    from methods.common.runlog import git_commit, git_dirty
    if (git_dirty() if dirty is None else dirty):
        raise Refused('git worktree is dirty; scientific runs require a clean committed worktree')
    branch = subprocess.run(['git', '-C', str(ROOT), 'branch', '--show-current'], capture_output=True,
                            universal_newlines=True).stdout.strip()
    if branch != BRANCH:
        raise Refused('scientific runs launch from branch %s; got %r' % (BRANCH, branch))
    errors = pio.launch_environment_errors(environ, args.seed)
    if errors:
        raise Refused('launch environment must be exactly the M6D2b precision environment: ' + ', '.join(errors))
    try:
        cfg = pio.verify_contract()
        source = pio.verify_source()
        ids = pio.identities(cfg, source)
        rel = os.path.relpath(str(Path(args.execution_config).resolve()), str(ROOT))
        storage = storage or pio.storage_from_exec_config(rel)
        pack = Path(storage['pack_root'])
        manifest = pack / 'pack_manifest.json'
        if not manifest.is_file() or pio.sha256_file(manifest) != pio.PACK_MANIFEST_SHA256:
            raise Refused('E03 feed pack manifest at %s is missing or differs from %s' % (manifest,
                                                                                       pio.PACK_MANIFEST_SHA256))
        payloads = json.loads(manifest.read_text())['payloads']
        for name, row in sorted(payloads.items()):           # byte hashes only: no array or image is decoded
            if pio.sha256_file(pack / name) != row['sha256']:
                raise Refused('E03 feed pack payload %s differs from its manifest' % name)
    except pio.E03Stop as exc:
        raise Refused(str(exc)) from exc
    faces = Path(storage['faces_256_root'])
    if not faces.is_dir() or faces.is_symlink():
        raise Refused('canonical faces root %s must be an existing non-symlink directory' % faces)
    run_dir = pio.run_root(storage['runtime_root'], pio.SCIENTIFIC, args.seed)
    if run_dir.exists():
        raise Refused('%s exists; E03 scientific runs are fresh only (resume is not qualified)' % run_dir)
    return {'method_id': pio.METHOD_ID, 'display_name': pio.DISPLAY_NAME, 'experiment_seed': args.seed,
            'fidelity': pio.FIDELITY, 'rng': pio.rng_plan(args.seed), 'max_epoch': pio.MAX_EPOCH,
            'steps_per_epoch': pio.STEPS_PER_EPOCH, 'val_steps': pio.VAL_STEPS, 'batch_size': pio.BATCH_SIZE,
            'g_d_ratio': pio.G_D_RATIO, 'final_global_step': pio.FINAL_GLOBAL_STEP,
            'checkpoint_rule': pio.CHECKPOINT_RULE, 'run_dir': str(run_dir), 'run_dir_exists': False,
            'git_commit': git_commit(), 'git_branch': branch, 'git_dirty': False, 'identities': ids,
            'storage': storage, 'pack_manifest_sha256': pio.PACK_MANIFEST_SHA256,
            'launch_environment': dict(pio.PRECISION_ENV, PYTHONHASHSEED=str(args.seed)), 'resume': pio.RESUME,
            'tensorflow_imported': 'tensorflow' in sys.modules, 'graph_executed': False, 'images_decoded': 0,
            'landmark_arrays_loaded': 0}


def main(argv=None):
    args = parse(sys.argv[1:] if argv is None else argv)
    plan = preconditions(args)
    if args.preflight_only:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    from methods.stdn import production_runner as pr
    run_dir = pr.run_scientific(seed=args.seed, runtime_root=plan['storage']['runtime_root'],
                                faces_root=plan['storage']['faces_256_root'],
                                command_line=' '.join([sys.executable] + sys.argv))
    print(json.dumps({'status': 'completed', 'run_dir': str(run_dir)}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
