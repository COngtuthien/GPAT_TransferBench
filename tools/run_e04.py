#!/usr/bin/env python3
"""E04 Physics-STD SCIENTIFIC training CLI (Python 3.8, gpat-m6-e03, NVIDIA TF 1.15.5; CONTROLLED_ADAPTATION).

Hard-binds method E04, experiment seed in {42, 1337, 2026}, 150000 complete PhySTD iterations (per iteration: BN
updates + G Eq.23, BN updates + D L_D, BN updates + G Eq.24; global_iteration +1 after Step 3), batch 8 = 4 target_live
+ 4 source_spoof rows of pairs_train_v1 via the verified E04 feed pack, the D4/D7 RNG streams, D8 RGB / 255, no
augmentation (D5), no VAL loop (D6), checkpoints every 15000 iterations with iteration_150000 =
BASELINE_FINAL_STATE_V1, and <runtime_root>/runs/m6/E04/seed_<seed>/. FRESH ONLY: an existing seed root is refused;
there is no resume and no Saver.restore. No option changes iterations, batch, lr, optimizer, losses, augmentation,
relation, pack, checkpoint rule, output root, fidelity or qualification limits; such flags are refused before anything
is imported. Every identity (clean git, frozen config + snapshot, A4, M6D3d, owner resolution, E04 training lock, feed
pack and pre-pass manifests, faces root) is verified before TensorFlow is imported.

Launch (clean worktree, exact E04 training precision environment):
  env PATH=/home/student20261/miniconda3/envs/gpat-m6-e03/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
      CUDA_VISIBLE_DEVICES=0 NVIDIA_TF32_OVERRIDE=0 TF_ENABLE_AUTO_MIXED_PRECISION=0 \
      TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE=0 TF_ENABLE_CUBLAS_TENSOR_OP_MATH_FP32=0 \
      TF_ENABLE_CUDNN_TENSOR_OP_MATH_FP32=0 TF_ENABLE_CUDNN_RNN_TENSOR_OP_MATH_FP32=0 TF_XLA_FLAGS=--tf_xla_auto_jit=0 \
      TF_DETERMINISTIC_OPS=1 TF_CUDNN_DETERMINISTIC=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 TF_ENABLE_ONEDNN_OPTS=0 \
      PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONHASHSEED=<seed> \
      python -u -B tools/run_e04.py --seed <seed> --execution-config configs/execution/m5_gpu_3090.yaml [--preflight-only]
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

from methods.physics_std import production_io as pio  # noqa: E402


class Refused(SystemExit):
    """Fail-closed refusal (exit status 2); nothing was trained or written."""
    def __init__(self, message):
        print('E04 SCIENTIFIC RUN REFUSED: ' + message, file=sys.stderr)
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
                        help='verify every gate; import no TensorFlow, decode no image, create no run root')
    try:
        return parser.parse_args(argv)
    except SystemExit as exc:
        raise Refused('unrecognized or malformed arguments') from exc


def preconditions(args, *, environ=None, dirty=None, storage=None):
    environ = os.environ if environ is None else environ
    try:
        pio.validate_mode_seed(pio.SCIENTIFIC, args.seed)
    except pio.E04Stop as exc:
        raise Refused('%s (qualification seed %d is never a scientific seed)' % (exc, pio.QUALIFICATION_SEED)) from exc
    from methods.common.runlog import git_commit, git_dirty
    if (git_dirty() if dirty is None else dirty):
        raise Refused('git worktree is dirty; scientific runs require a clean committed worktree')
    branch = subprocess.run(['git', '-C', str(ROOT), 'branch', '--show-current'], stdout=subprocess.PIPE,
                            universal_newlines=True).stdout.strip()
    if branch != BRANCH:
        raise Refused('scientific runs launch from branch %s; got %r' % (BRANCH, branch))
    errors = pio.launch_environment_errors(environ, args.seed)
    if errors:
        raise Refused('launch environment must be exactly the E04 training precision environment: ' + ', '.join(errors))
    try:
        pio.verify_contract()
        ids = pio.identities()
        if not (pio.PACK_MANIFEST_SHA256 and pio.PREPASS_MANIFEST_SHA256):
            raise Refused('the E04 feed pack / pre-pass manifests are not pinned in production_io')
        rel = os.path.relpath(str(Path(args.execution_config).resolve()), str(ROOT))
        storage = storage or pio.storage_from_exec_config(rel)
        for key, pinned, name in (('pack_root', pio.PACK_MANIFEST_SHA256, 'pack_manifest.json'),
                                  ('prepass_root', pio.PREPASS_MANIFEST_SHA256, 'prepass_manifest.json')):
            manifest = Path(storage[key]) / name
            if not manifest.is_file() or pio.sha256_file(manifest) != pinned:
                raise Refused('%s is missing or differs from %s' % (manifest, pinned))
            for payload, row in sorted(json.loads(manifest.read_text())['payloads'].items()):
                if pio.sha256_file(Path(storage[key]) / payload) != row['sha256']:
                    raise Refused('%s payload %s differs from its manifest' % (key, payload))
    except pio.E04Stop as exc:
        raise Refused(str(exc)) from exc
    faces = Path(storage['faces_256_root'])
    if not faces.is_dir() or faces.is_symlink():
        raise Refused('canonical faces root %s must be an existing non-symlink directory' % faces)
    run_dir = pio.run_root(storage['runtime_root'], pio.SCIENTIFIC, args.seed)
    if run_dir.exists():
        raise Refused('%s exists; E04 scientific runs are fresh only (resume is not qualified)' % run_dir)
    return {'method_id': pio.METHOD_ID, 'experiment_seed': args.seed, 'fidelity': pio.FIDELITY,
            'reporting_label': pio.REPORTING_LABEL, 'rng': pio.rng_plan(args.seed), 'iterations': pio.ITERATIONS,
            'batch': pio.BATCH, 'checkpoint_rule': pio.CHECKPOINT_RULE, 'checkpoints': pio.scientific_checkpoints(),
            'run_dir': str(run_dir), 'run_dir_exists': False, 'git_commit': git_commit(), 'git_branch': branch,
            'git_dirty': False, 'identities': ids, 'storage': storage,
            'launch_environment': dict(pio.PRECISION_ENV, PYTHONHASHSEED=str(args.seed)), 'resume': pio.RESUME,
            'tensorflow_imported': 'tensorflow' in sys.modules, 'graph_executed': False, 'images_decoded': 0}


def main(argv=None):
    args = parse(sys.argv[1:] if argv is None else argv)
    plan = preconditions(args)
    if args.preflight_only:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    from methods.physics_std import production_runner as pr
    run_dir = pr.run_scientific(seed=args.seed, runtime_root=plan['storage']['runtime_root'],
                                faces_root=plan['storage']['faces_256_root'],
                                command_line=' '.join([sys.executable] + sys.argv))
    print(json.dumps({'status': 'completed', 'run_dir': str(run_dir)}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
