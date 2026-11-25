"""M10 ResNet-18 detector launcher CLI (one group / detector seed per invocation; see gpatbench/downstream/m10_launcher).

SCIENTIFIC (needs the ACTIVE configs/amendments/m10_resnet18_science_launch_authority.yaml):

    env -i HOME=$HOME PATH=/usr/bin:/bin LANG=C.UTF-8 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 \
        PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
        ~/workdir/envs/gpat-m8-downstream/bin/python -B tools/run_m10_detector.py \
        --group <G> --seed <S> --runtime-root ~/workdir/GPAT_TransferBench_runtime [--preflight-only]

NON_SCIENTIFIC_QUALIFICATION: add --fixture <dir> --qualification-out <dir outside the official roots>.
Exit codes: 0 success / preflight passed, 1 run failed (evidence kept), 2 preflight refused (nothing created).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gpatbench.downstream.m10_launcher import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
