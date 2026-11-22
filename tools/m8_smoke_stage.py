#!/usr/bin/env python3
"""One stage of the M8 global smoke pipeline (run by tools/m8_smoke_run.py; never alone in production).

    python -B tools/m8_smoke_stage.py {select,banks,evaluator,metrics,plots} --mode {real,synthetic} --root <smoke root>
        [--faces-root <production faces_256>] [--weights <resnet18-f37072fd.pth>]

Every stage refuses unless the owner-approved smoke and evaluator contracts are present, installs the faces-root
audit-hook firewall (allowlist; directory listings forbidden), writes <root>/<stage>/stage.json and firewall.json, and
exits 0 only if every stage gate holds. Outputs are SMOKE_NON_SCIENTIFIC.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SMOKE_CONTRACT = "configs/amendments/m8_smoke_test_contract.yaml"
EVAL_CONTRACT = "configs/amendments/m8_downstream_evaluator_contract.yaml"
FEASIBILITY = "outputs/audit/M8_TOY_SUBSET_FEASIBILITY.json"
SPLIT_SHA = "fb9aeb369a124fc96ba855ef2ce269236c4a743fe960e73ab739412c9cb5092d"
BANK_SEEDS = (42, 1337, 2026)
DETECTORS = ("E00", "E01", "E02")
DETECTOR_SEED = 42


def sha_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def contracts() -> dict:
    out = {}
    for p in (SMOKE_CONTRACT, EVAL_CONTRACT):
        d = json.loads((ROOT / p).read_text())
        if d.get("status") != "OWNER_APPROVED" or d.get("approved") is not True:
            raise SystemExit(f"{p} is not owner-approved")
        out[p] = sha_file(ROOT / p)
    return out


def schedule() -> dict:
    s = json.loads((ROOT / SMOKE_CONTRACT).read_text())["smoke_schedule_SMOKE_NON_SCIENTIFIC"]
    return {"epochs": int(s["epochs"]), "warmup_epochs": 1, "steps_per_epoch": int(s["steps_per_epoch"])}


def load_manifest(root: Path) -> dict:
    return json.loads((root / "select" / "toy_manifest.json").read_text())


def faces_root_for(args, root: Path) -> Path:
    return Path(args.faces_root) if args.mode == "real" else root / "select" / "world" / "faces_256"


def stage_select(args, root, out):
    from gpatbench.downstream import audit_hook, toy_manifest, toy_subset
    gates = {}
    if args.mode == "real":
        audit_hook.install(args.faces_root, [], out / "firewall.json")     # select opens no face at all
        man = toy_manifest.build_real(ROOT, args.faces_root)
        rec = json.loads((ROOT / FEASIBILITY).read_text())
        plan = toy_subset.plan(ROOT)
        for k in ("roles", "pooled", "toy_pairs", "feasibility", "shortfalls", "skipped_units"):
            gates[f"plan_equals_feasibility_record.{k}"] = json.loads(json.dumps(plan[k])) == rec[k]
        f = plan["feasibility"]
        gates["expected_counts"] = (f["train_live_frames"], f["train_real_spoof_frames"], f["toy_pairs"],
                                    f["live_pass_full_batches_per_epoch"]) == (96, 256, 186, 3)
        gates["expected_val_videos"] = (plan["pooled"]["VAL/select/0"]["videos"], plan["pooled"]["VAL/select/1"]["videos"],
                                        plan["pooled"]["VAL/holdout/0"]["videos"],
                                        plan["pooled"]["VAL/holdout/1"]["videos"]) == (12, 32, 12, 33)
        gates["eight_e01_operators"] = all(v > 0 for v in f["e01_operator_coverage"].values())
        gates["select_holdout_disjoint"] = all(f["select_holdout_disjoint"].values())
    else:
        man = toy_manifest.build_synthetic_world(out / "world")
        gates["synthetic_world_built"] = True
    gates["test_rows_read_zero"] = man["test_rows_read"] == 0
    gates["roles_nonempty"] = all(man["roles"][k] for k in toy_manifest.ROLE_KEYS)
    sha = toy_manifest.write_manifest(man, out / "toy_manifest.json")
    return gates, {"toy_manifest_sha256": sha, "toy_pairs": len(man["toy_pairs"]),
                   "frames": {k: len(v) for k, v in man["roles"].items()}}


def stage_banks(args, root, out):
    from methods.common import bank_writer as bw
    from methods.common.config import load_method_config
    from gpatbench.downstream import audit_hook
    man = load_manifest(root)
    faces = faces_root_for(args, root)
    audit_hook.install(faces, [s["face_rel_path"] for s in man["pair_samples"]], out / "firewall.json")
    auth = bw.smoke_authorization()
    positions = [p["position"] for p in man["toy_pairs"]]
    gates, info = {}, {"banks": {}}
    lock = None
    if args.mode == "real":
        live = bw.verify_locked_environment()
        lock = {"verified": True, "path": bw.ENV_LOCK_PATH, "live": live}
    for m in ("E01", "E02"):
        cfg = load_method_config(m)
        if args.mode == "real":
            inputs = bw.smoke_toy_inputs(faces.parents[2], cfg, positions)
            gates[f"{m}.toy_inputs_equal_manifest"] = [(r.pair_id, r.target_live_id, r.source_spoof_id) for r in inputs.pairs] \
                == [(p["pair_id"], p["target_live_id"], p["source_spoof_id"]) for p in man["toy_pairs"]]
        else:
            ps = {s["sample_id"]: s for s in man["pair_samples"]}
            inputs = bw.BankInputs(
                pairs=[bw.PairRow(p["position"], p["pair_id"], p["target_live_id"], p["source_spoof_id"], p["dataset"])
                       for p in man["toy_pairs"]],
                faces_root=faces, face_sha256={k: v["face_sha256"] for k, v in ps.items()},
                sample_split={k: "TRAIN" for k in ps}, sample_dataset={k: v["dataset"] for k, v in ps.items()},
                provenance={"kind": "SMOKE_SYNTHETIC_FIXTURE", "pairs": len(man["toy_pairs"])})
        ops = cfg["generation"].get("operator_set")
        for s in BANK_SEEDS:
            summ = bw.write_bank(m, s, inputs, auth, root / "banks", config=cfg, environment_lock=lock)
            rd = root / "banks" / "runs" / "m6" / m / f"seed_{s}"
            files = bw.verify_bank_files(rd, m, global_positions=True)
            replay = bw.replay_bank(rd, m, s, inputs, auth, config=cfg)
            rows = bw.read_manifest(rd)
            key = f"{m}.seed_{s}"
            gates[f"{key}.completed_no_failures"] = summ["completion_status"] == "completed" and summ["failure_count"] == 0
            gates[f"{key}.rows_equal_toy_pairs"] = [r["position"] for r in rows] == positions
            gates[f"{key}.file_verification"] = files["ok"]
            gates[f"{key}.full_replay"] = replay["ok"]
            if m == "E01":
                gates[f"{key}.operator_is_position_mod_8"] = all(r["selected_operator_name"] == ops[r["position"] % 8] for r in rows)
            else:
                gates[f"{key}.k40_n159_residual"] = summ["k_values"] == [40] and summ["n_eligible_values"] == [159] \
                    and summ["max_abs_imag"] <= 1e-8
            info["banks"][key] = {"rows": len(rows), "outputs": summ["output_count"], "failures": summ["failure_count"],
                                  "bank_manifest_sha256": summ["bank_manifest_sha256"], "run_id": summ["run_id"]}
    return gates, info


def stage_evaluator(args, root, out):
    from gpatbench.downstream import resnet_runner as R
    R.configure_determinism()
    from gpatbench.downstream import audit_hook
    from gpatbench.downstream.bank_reader import read_verified_bank
    from gpatbench.downstream.faceio import FaceReader
    from PIL import Image
    import io
    man = load_manifest(root)
    faces = faces_root_for(args, root)
    roles = man["roles"]
    allow = sorted({f["face_rel_path"] for k in roles for f in roles[k]})
    audit_hook.install(faces, allow, out / "firewall.json")
    reader = FaceReader(faces)
    load_real = lambda f: reader.read(f["face_rel_path"], f["face_sha256"])

    def load_synthetic(s):
        raw = Path(s["path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != s["sha256"]:
            raise R.RunnerError(f"synthetic output sha256 mismatch {s['pair_id']}")
        with Image.open(io.BytesIO(raw)) as im:
            if im.mode != "RGB" or im.size != (256, 256):
                raise R.RunnerError("synthetic output is not 256x256 RGB")
            return im.copy()

    sch = schedule()
    split_sha = SPLIT_SHA if args.mode == "real" else hashlib.sha256(b"synthetic-qualification").hexdigest()
    gates, info = {}, {}
    for m in DETECTORS:
        syn = []
        if m != "E00":
            b = read_verified_bank(root / "banks" / "runs" / "m6" / m / f"seed_{DETECTOR_SEED}", m, DETECTOR_SEED,
                                   global_positions=True)
            syn = [{"pair_id": s.pair_id, "path": str(s.path), "sha256": s.output_sha256} for s in b["samples"]]
        rec = R.train_detector(method_id=m, seed=DETECTOR_SEED, live=roles["train_live"], real_spoof=roles["train_spoof"],
                               synthetic=syn, val_select=roles["val_select_live"] + roles["val_select_spoof"],
                               holdout=roles["val_holdout_live"] + roles["val_holdout_spoof"], load_real=load_real,
                               load_synthetic=load_synthetic, weights_path=args.weights, epochs=sch["epochs"],
                               warmup_epochs=sch["warmup_epochs"], out_dir=out / m, split_manifest_sha256=split_sha)
        steps = [json.loads(l) for l in (out / m / "steps.jsonl").read_text().splitlines()]
        comp = (32, 32, 0) if m == "E00" else (32, 16, 16)
        gates[f"{m}.steps"] = len(steps) == sch["epochs"] * sch["steps_per_epoch"] if args.mode == "real" else len(steps) > 0
        gates[f"{m}.batch_composition"] = all((s["n_live"], s["n_real_spoof"], s["n_synthetic"]) == comp for s in steps)
        gates[f"{m}.first_step_lr_zero"] = float.fromhex(steps[0]["lr_backbone_hex"]) == 0.0
        gates[f"{m}.cpu_no_amp_deterministic"] = rec["device"] == "cpu" and rec["amp"] is False and rec["deterministic_algorithms"]
        gates[f"{m}.holdout_predicted_once"] = (out / m / "predictions_holdout.parquet").is_file()
        info[m] = rec
    led = reader.summary()
    (out / "faces_ledger.json").write_text(json.dumps({"summary": led, "files": reader.ledger}, indent=1, sort_keys=True) + "\n")
    gates["faces_atime_unchanged"] = led["atime_changed_files"] == 0
    return gates, info


def stage_metrics(args, root, out):
    from gpatbench.downstream import audit_hook, schemas, tables
    if args.mode == "real":
        audit_hook.install(args.faces_root, [], out / "firewall.json")
    gates = {}
    for m in DETECTORS:
        rows = tables.read_predictions(root / "evaluator" / m / "predictions_holdout.parquet")
        schemas.validate_prediction_rows(rows)
        gates[f"{m}.holdout_predictions_schema"] = True
    outputs = {}
    for name, data in tables.build_tables(root / "evaluator", ROOT).items():
        p = out / name
        if p.exists():
            raise SystemExit(f"refusing to overwrite {p}")
        p.write_bytes(data)
        outputs[name] = hashlib.sha256(data).hexdigest()
    gates["tables_written"] = len(outputs) == 7
    return gates, {"tables": outputs}


def stage_plots(args, root, out):
    from gpatbench.downstream import audit_hook, render
    if args.mode == "real":
        audit_hook.install(args.faces_root, [], out / "firewall.json")
    res = render.render_all(root / "metrics", out / "figures")
    figs = {f: sha_file(out / "figures" / f) for f in res["figures"]}
    return {"eight_figures": len(figs) == 8, "inputs_have_no_placeholders":
            all(c["placeholder_cells"] == 0 for c in res["input_checks"].values())}, {"figures": figs, "checks": res["input_checks"]}


STAGES = {"select": stage_select, "banks": stage_banks, "evaluator": stage_evaluator, "metrics": stage_metrics,
          "plots": stage_plots}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=list(STAGES))
    ap.add_argument("--mode", choices=("real", "synthetic"), required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--faces-root")
    ap.add_argument("--weights")
    a = ap.parse_args(argv)
    if a.mode == "real" and not a.faces_root:
        ap.error("--faces-root is required in real mode")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    root = Path(a.root)
    out = root / a.stage
    out.mkdir(parents=True, exist_ok=False)
    rec = {"stage": a.stage, "mode": a.mode, "label": "SMOKE_NON_SCIENTIFIC", "contracts": contracts(),
           "python": sys.executable, "argv": sys.argv}
    try:
        gates, info = STAGES[a.stage](a, root, out)
        rec.update(gates=gates, info=info, ok=bool(gates) and all(gates.values()))
    except BaseException as exc:
        rec.update(ok=False, error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    try:
        from gpatbench.downstream import audit_hook
        if audit_hook._STATE.get("installed"):
            fw = audit_hook.write_report()
            rec["firewall"] = {"violation_count": fw["violation_count"], "listing_attempts": fw["listing_attempts"],
                               "opens_under_faces_root": fw["opens_under_faces_root"],
                               "distinct_files_opened": fw["distinct_files_opened"]}
            if fw["violation_count"]:
                rec["ok"] = False
    except Exception as exc:                                  # pragma: no cover
        rec.update(ok=False, firewall_error=str(exc))
    (out / "stage.json").write_text(json.dumps(rec, indent=1, sort_keys=True, default=str) + "\n")
    print(json.dumps({"stage": a.stage, "ok": rec["ok"], "error": rec.get("error")}))
    return 0 if rec["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
