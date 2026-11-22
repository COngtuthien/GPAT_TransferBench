#!/usr/bin/env python3
"""M8 global smoke orchestrator: select -> banks -> evaluator -> metrics -> plots, each in its own process.

    python -B tools/m8_smoke_run.py --mode {real,synthetic} --root <new smoke root> --bank-python <gpat-m5 python>
        --weights <resnet18-f37072fd.pth> [--faces-root <production faces_256>]

The bank stage runs under the locked gpat-m5 interpreter (Q4); the others under the current interpreter
(gpat-m8-downstream). Every stage runs at nice 19 / ionice idle with CUDA hidden and 8 CPU threads. The run stops at the
first failing stage and keeps its evidence. <root> must not exist. Writes <root>/smoke_run.json (commands, exit codes,
wall times). Outputs are SMOKE_NON_SCIENTIFIC.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGES = ("select", "banks", "evaluator", "metrics", "plots")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("real", "synthetic"), required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--bank-python", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--faces-root")
    a = ap.parse_args(argv)
    root = Path(a.root)
    root.mkdir(parents=True, exist_ok=False)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "8",
           "MKL_NUM_THREADS": "8", "PYTHONHASHSEED": "0"}
    prefix = ["nice", "-n", "19"] + (["ionice", "-c3"] if shutil.which("ionice") else [])
    run = {"mode": a.mode, "root": str(root), "label": "SMOKE_NON_SCIENTIFIC", "stages": [],
           "git_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"]).decode().strip(),
           "worktree_dirty": bool(subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"]).decode().strip())}
    ok = True
    for st in STAGES:
        py = a.bank_python if st == "banks" else sys.executable
        cmd = [py, "-B", str(ROOT / "tools" / "m8_smoke_stage.py"), st, "--mode", a.mode, "--root", str(root),
               "--weights", a.weights] + (["--faces-root", a.faces_root] if a.faces_root else [])
        t0 = time.time()
        p = subprocess.run(prefix + cmd, env=env, capture_output=True, text=True)
        rec = {"stage": st, "argv": prefix + cmd, "exit_code": p.returncode, "start_unix": t0,
               "wall_seconds": round(time.time() - t0, 2), "stdout_tail": p.stdout[-2000:], "stderr_tail": p.stderr[-4000:]}
        run["stages"].append(rec)
        print(json.dumps({"stage": st, "exit_code": p.returncode, "wall_seconds": rec["wall_seconds"]}), flush=True)
        if p.returncode != 0:
            ok = False
            break
    run["ok"] = ok
    (root / "smoke_run.json").write_text(json.dumps(run, indent=1) + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
