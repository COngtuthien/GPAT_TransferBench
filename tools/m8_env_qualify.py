#!/usr/bin/env python3
"""gpat-m8-downstream environment qualification V1-V6 (environments/m8_downstream.planned.json).

    <gpat-m8-downstream python> -B tools/m8_env_qualify.py all --evidence <new dir> --weights <resnet18-f37072fd.pth> \
        --m5-python <gpat-m5 python> --pre-dir <dir holding gpat_m5_freeze_pre.txt and gpat_m5_tree_pre.sha256>

V4 (HARD GATE for S9) runs the real smoke runner (resnet_runner.train_detector, E01-type mixed batches, the smoke
schedule) in two separate processes on a deterministic synthetic world and requires byte-identical step logs, epoch
records, VAL and holdout prediction files and detector records. V5 renders the same synthetic tables in two processes
and compares PNG bytes. No benchmark image is read; CUDA is hidden.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PLAN = "environments/m8_downstream.planned.json"
V4_FILES = ("steps.jsonl", "epochs.json", "predictions_val_select_epoch0.parquet", "predictions_val_select_epoch1.parquet",
            "predictions_holdout.parquet", "detector.json")


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def v4_child(out: Path, weights: str) -> int:
    from gpatbench.downstream import resnet_runner as R
    R.configure_determinism()
    import io
    from PIL import Image
    from gpatbench.downstream import toy_manifest
    from gpatbench.downstream.faceio import FaceReader
    out.mkdir(parents=True, exist_ok=False)
    man = toy_manifest.build_synthetic_world(out / "world")
    roles = man["roles"]
    reader = FaceReader(out / "world" / "faces_256")
    syn = [{"pair_id": f"SYN{i:04d}", "path": str(out / "world" / "faces_256" / f["face_rel_path"]),
            "sha256": f["face_sha256"]} for i, f in enumerate(roles["train_spoof"])]

    def load_syn(s):
        raw = Path(s["path"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == s["sha256"]
        with Image.open(io.BytesIO(raw)) as im:
            return im.copy()
    R.train_detector(method_id="E01", seed=42, live=roles["train_live"], real_spoof=roles["train_spoof"], synthetic=syn,
                     val_select=roles["val_select_live"] + roles["val_select_spoof"],
                     holdout=roles["val_holdout_live"] + roles["val_holdout_spoof"],
                     load_real=lambda f: reader.read(f["face_rel_path"], f["face_sha256"]), load_synthetic=load_syn,
                     weights_path=weights, epochs=2, warmup_epochs=1, out_dir=out / "detector",
                     split_manifest_sha256=hashlib.sha256(b"v4").hexdigest())
    return 0


def v5_child(out: Path) -> int:
    from gpatbench.downstream import render
    m = out / "metrics"
    m.mkdir(parents=True, exist_ok=False)
    meth = ("E00", "E01", "E02")
    lines = {"T06_smoke_downstream_resnet18_video.csv": ["method_id,ACER,AUC,EER"] +
             [f"{x},{0.1 * (i + 1)!r},{0.9 - 0.1 * i!r},{0.15 * (i + 1)!r}" for i, x in enumerate(meth)],
             "T13_smoke_attackwise_apcer.csv": ["method_id,attack_macro,APCER"] +
             [f"{x},{a},{0.05 * (i + j + 1)!r}" for i, x in enumerate(meth) for j, a in enumerate(("print", "replay"))],
             "T14_smoke_datasetwise_metrics.csv": ["method_id,dataset,ACER,AUC"] +
             [f"{x},{d},{0.1 + 0.05 * j!r},{0.8 - 0.05 * i!r}" for i, x in enumerate(meth)
              for j, d in enumerate(("casia_fasd", "msu_mfsd", "siwmv2"))],
             "curves_roc_det_points.csv": ["method_id,threshold,APCER,BPCER,FPR,TPR"] +
             [f"{x},{t!r},{t!r},{1 - t!r},{1 - t!r},{1 - t!r}" for x in meth for t in (0.0, 0.25, 0.5, 0.75, 1.0)],
             "training_curves.csv": ["method_id,kind,step,epoch,loss,val_acer,val_auc"] +
             [f"{x},train_step,{s},{s // 3},{0.7 - 0.05 * s!r},," for x in meth for s in range(6)] +
             [f"{x},val_epoch,,{e},,{0.3 - 0.1 * e!r},{0.7 + 0.1 * e!r}" for x in meth for e in range(2)]}
    for name, rows in lines.items():
        (m / name).write_text("\n".join(rows) + "\n")
    render.render_all(m, out / "figures")
    return 0


def run_children(kind: str, ev: Path, weights: str | None) -> dict:
    outs = []
    for i in (1, 2):
        o = ev / f"{kind}_process_{i}"
        cmd = [sys.executable, "-B", str(Path(__file__).resolve()), f"{kind}-child", "--out", str(o)] + \
              (["--weights", weights] if weights else [])
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8",
               "PYTHONDONTWRITEBYTECODE": "1"}
        p = subprocess.run(["nice", "-n", "19"] + cmd, env=env, capture_output=True, text=True)
        outs.append({"process": i, "exit_code": p.returncode, "stderr_tail": p.stderr[-3000:], "dir": o})
    res = {"processes": [{k: v for k, v in o.items() if k != "dir"} for o in outs]}
    if any(o["exit_code"] for o in outs):
        res["identical"] = False
        return res
    if kind == "v4":
        files = {f: [sha(o["dir"] / "detector" / f) for o in outs] for f in V4_FILES}
        steps = [json.loads(l) for l in (outs[0]["dir"] / "detector" / "steps.jsonl").read_text().splitlines()]
        res.update(files=files, steps=len(steps), first_step_lr_zero=float.fromhex(steps[0]["lr_backbone_hex"]) == 0.0,
                   loss_hex=[s["loss_hex"] for s in steps],
                   state_sha256=[e["state_sha256"] for e in json.loads((outs[0]["dir"] / "detector" / "epochs.json").read_text())])
    else:
        names = sorted(p.name for p in (outs[0]["dir"] / "figures").glob("*.png"))
        files = {n: [sha(o["dir"] / "figures" / n) for o in outs] for n in names}
        res.update(files=files)
    res["identical"] = all(len(set(v)) == 1 for v in files.values()) and bool(files)
    return res


def m5_snapshot(m5_python: str) -> tuple[str, str]:
    freeze = subprocess.check_output([m5_python, "-B", "-m", "pip", "list", "--format=freeze"])
    envdir = Path(m5_python).resolve().parents[1]
    listing = subprocess.check_output(f'cd "{envdir}" && find . -xdev -printf "%P\\t%s\\t%T@\\n" | LC_ALL=C sort',
                                      shell=True)
    return hashlib.sha256(freeze).hexdigest(), hashlib.sha256(listing).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("all", "v4-child", "v5-child"))
    ap.add_argument("--out")
    ap.add_argument("--evidence")
    ap.add_argument("--weights")
    ap.add_argument("--m5-python")
    ap.add_argument("--pre-dir")
    a = ap.parse_args(argv)
    if a.cmd == "v4-child":
        return v4_child(Path(a.out), a.weights)
    if a.cmd == "v5-child":
        return v5_child(Path(a.out))
    ev = Path(a.evidence)
    ev.mkdir(parents=True, exist_ok=False)
    plan = json.loads((ROOT / PLAN).read_text())
    gates, rec = {}, {}
    # V1
    pre_dir = Path(a.pre_dir)
    pre_freeze = hashlib.sha256((pre_dir / "gpat_m5_freeze_pre.txt").read_bytes()).hexdigest()
    pre_tree = (pre_dir / "gpat_m5_tree_pre.sha256").read_text().split()[0]
    now_freeze, now_tree = m5_snapshot(a.m5_python)
    gates["V1_gpat_m5_unchanged"] = (pre_freeze, pre_tree) == (now_freeze, now_tree)
    rec["V1"] = {"freeze_sha256_pre": pre_freeze, "freeze_sha256_now": now_freeze, "tree_sha256_pre": pre_tree,
                 "tree_sha256_now": now_tree}
    # V2
    import importlib.metadata as md
    import numpy, PIL, pyarrow, torch, torchvision, matplotlib
    want = {w["name"]: w["version"] for w in plan["added_wheels"]}
    want.update({"torch": "2.12.1+cu130", "torchvision": "0.27.1+cu130", "numpy": "2.4.6", "pillow": "12.3.0",
                 "pyarrow": "25.0.1", "packaging": "26.3", "python-dateutil": "2.9.0.post0"})
    got = {k: md.version(k) for k in want}
    import platform
    gates["V2_versions"] = got == want and platform.python_version() == "3.11.16"
    venv_site = Path(sys.prefix) / "lib" / "python3.11" / "site-packages"
    added_location = {w["name"]: str(Path(md.distribution(w["name"]).locate_file("")).resolve())
                      for w in plan["added_wheels"]}
    gates["V2_added_packages_in_overlay_only"] = all(Path(v) == venv_site.resolve() for v in added_location.values())
    rec["V2"] = {"versions": got, "python": platform.python_version(), "sys_prefix": sys.prefix,
                 "base_prefix": sys.base_prefix, "added_locations": added_location, "torch_cuda_used": False}
    # V3
    from gpatbench.downstream import resnet_runner as R
    model = R.build_model(a.weights, 42)
    gates["V3_resnet18_weights"] = sha(a.weights) == R.WEIGHTS_SHA256 and model.fc.in_features == 512
    rec["V3"] = {"weights_sha256": sha(a.weights), "fc_in_features": model.fc.in_features}
    del model
    # V4, V5
    rec["V4"] = run_children("v4", ev, a.weights)
    gates["V4_cpu_training_bitwise_two_processes"] = rec["V4"]["identical"] and rec["V4"].get("first_step_lr_zero", False)
    rec["V5"] = run_children("v5", ev, None)
    gates["V5_png_bytes_identical"] = rec["V5"]["identical"]
    # V6
    freeze_overlay = subprocess.check_output([sys.executable, "-B", "-m", "pip", "list", "--format=freeze", "--path",
                                              str(venv_site)]).decode()
    lock = {"schema_version": "m8_downstream_environment_lock_v1", "environment_name": "gpat-m8-downstream",
            "construction": "A_VENV_OVERLAY", "sys_prefix": sys.prefix, "base_prefix": sys.base_prefix,
            "python_vv": sys.version, "overlay_freeze": freeze_overlay.splitlines(),
            "base_gpat_m5_freeze_sha256": now_freeze, "base_gpat_m5_tree_sha256": now_tree,
            "wheels": plan["added_wheels"], "torch_config": torch.__config__.show(),
            "torch_threads_smoke": R.THREADS, "deterministic_algorithms": True, "cuda_visible_devices": "",
            "qualification_gates": None}
    rec["lock"] = lock
    gates["V6_lock_captured"] = bool(freeze_overlay.strip()) and len(lock["overlay_freeze"]) == 6
    lock["qualification_gates"] = gates
    (ev / "m8_downstream.lock.json").write_text(json.dumps(lock, indent=1, sort_keys=True) + "\n")
    out = {"record": "M8_DOWNSTREAM_ENVIRONMENT_QUALIFICATION", "gates": gates, "all_pass": all(gates.values()),
           "lock_sha256": sha(ev / "m8_downstream.lock.json"), **rec}
    out.pop("lock")
    (ev / "environment_qualification.json").write_text(json.dumps(out, indent=1, sort_keys=True, default=str) + "\n")
    print(json.dumps({"gates": gates, "all_pass": out["all_pass"]}, indent=1))
    return 0 if out["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
