#!/bin/bash
date -u +%FT%TZ > ~/workdir/m10_E00_s42.launch_time
cd ~/workdir/m10_launch/science_exec && env -i HOME=$HOME PATH=/usr/bin:/bin LANG=C.UTF-8 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 ~/workdir/envs/gpat-m8-downstream/bin/python -B tools/run_m10_detector.py --group E00 --seed 42 --runtime-root ~/workdir/GPAT_TransferBench_runtime; echo $? > ~/workdir/m10_E00_s42.exit
date -u +%FT%TZ > ~/workdir/m10_E00_s42.end_time
