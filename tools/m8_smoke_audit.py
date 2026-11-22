#!/usr/bin/env python3
"""M8 global smoke AUDIT: replay identity, independent re-derivation, coverage, firewall and the smoke record.

    python -B tools/m8_smoke_audit.py audit --run-a <root A> --run-b <root B> --env-evidence <environment_qualification.json>
        --out <new evidence dir>
    python -B tools/m8_smoke_audit.py rederive --evidence <evidence dir>        # re-run the re-derivation on copied files

The re-derivation deliberately does NOT import gpatbench.downstream.metrics: video aggregation, the section-15
threshold sweep, D5 epoch selection, APCER/BPCER/ACER, pairwise AUC, D7 EER, D8 TPR@FPR, spoof F1 and the D4 decisions
are recomputed here from the prediction files with separate code. A record states smoke_test_passed true only when every
gate holds and gpatbench.downstream.smoke_evidence.validate_smoke_record accepts it.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import shutil
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SMOKE_CONTRACT = "configs/amendments/m8_smoke_test_contract.yaml"
EVAL_CONTRACT = "configs/amendments/m8_downstream_evaluator_contract.yaml"
DETECTORS = ("E00", "E01", "E02")
BANKS = [(m, s) for m in ("E01", "E02") for s in (42, 1337, 2026)]
COPY_SMALL = ("select/toy_manifest.json", "metrics", "plots/figures", "evaluator/faces_ledger.json")


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def deterministic_files(root: Path) -> dict:
    """relative path -> sha256 for every artifact the S9 replay must reproduce byte for byte."""
    pats = ["select/toy_manifest.json", "evaluator/faces_ledger.json", "metrics/*.csv", "plots/figures/*.png"]
    for m, s in BANKS:
        b = f"banks/runs/m6/{m}/seed_{s}"
        pats += [f"{b}/bank/*.png", f"{b}/edit_maps/*.png", f"{b}/bank_manifest.parquet", f"{b}/bank_manifest.sha256"]
    for m in DETECTORS:
        pats += [f"evaluator/{m}/{n}" for n in ("steps.jsonl", "epochs.json", "detector.json")] + \
                [f"evaluator/{m}/predictions_*.parquet"]
    pats += ["select/world/faces_256/*/*.png"]
    out = {}
    for pat in pats:
        for p in sorted(root.glob(pat)):
            out[str(p.relative_to(root))] = sha(p)
    return out


# ---------------------------------------------------------------- independent re-derivation
def _read_parquet(p):
    import pyarrow.parquet as pq
    return pq.read_table(p).to_pylist()


def _videos(rows):
    g = {}
    for r in rows:
        g.setdefault((r["dataset"], r["video_id"]), []).append(r)
    out = []
    for (ds, vid), fr in sorted(g.items()):
        ys = {f["y_true"] for f in fr}
        assert len(ys) == 1
        out.append({"dataset": ds, "video_id": vid, "y": ys.pop(), "score": math.fsum(f["p_frame"] for f in fr) / len(fr),
                    "macro": fr[0]["attack_macro"], "raw": fr[0]["attack_raw"], "reported": {f["video_score"] for f in fr}})
    return out


def _rates(y, s, t):
    ns = sum(1 for v in y if v == 1)
    nl = len(y) - ns
    spoof_live = sum(1 for v, x in zip(y, s) if v == 1 and not x >= t)
    live_spoof = sum(1 for v, x in zip(y, s) if v == 0 and x >= t)
    a, b = Fraction(spoof_live, ns), Fraction(live_spoof, nl)
    tp = ns - spoof_live
    f1 = Fraction(2 * tp, 2 * tp + live_spoof + spoof_live) if (2 * tp + live_spoof + spoof_live) else None
    return {"APCER": a, "BPCER": b, "ACER": (a + b) / 2, "accuracy": Fraction(tp + nl - live_spoof, ns + nl),
            "balanced_accuracy": 1 - (a + b) / 2, "F1_spoof": f1}


def _cands(s):
    u = sorted(set(s))
    return sorted({(a + b) / 2.0 for a, b in zip(u, u[1:])} | {0.0, 1.0})


def _auc(y, s):
    pos = [x for v, x in zip(y, s) if v == 1]
    neg = [x for v, x in zip(y, s) if v == 0]
    return Fraction(sum(2 if p > n else 1 if p == n else 0 for p in pos for n in neg), 2 * len(pos) * len(neg))


def _threshold(y, s):
    ev = [(_rates(y, s, t)["ACER"], t) for t in _cands(s)]
    best = min(a for a, _ in ev)
    c = [t for a, t in ev if a == best]
    d = min(abs(t - 0.5) for t in c)
    c = [t for t in c if abs(t - 0.5) == d]
    assert len(c) == 1, "unbroken tie"
    return c[0], best


def _eer(y, s):
    pts = [(_rates(y, s, t)["APCER"], _rates(y, s, t)["BPCER"]) for t in _cands(s)]
    for a, b in pts:
        if a == b:
            return a
    for (a0, b0), (a1, b1) in zip(pts, pts[1:]):
        if a0 - b0 < 0 < a1 - b1:
            al = (b0 - a0) / ((a1 - a0) - (b1 - b0))
            return a0 + al * (a1 - a0)
    raise AssertionError("no crossing")


def _tpr(y, s, x):
    ok = [1 - r["APCER"] for r in (_rates(y, s, t) for t in _cands(s)) if r["BPCER"] <= Fraction(x)]
    return max(ok) if ok else None


def _exact(v):
    return "N/A" if v is None else f"{v.numerator}/{v.denominator}" if isinstance(v, Fraction) else repr(v)


def rederive(evaluator_dir: Path, metrics_dir: Path) -> dict:
    t06 = {r["method_id"]: r for r in csv.DictReader(io.StringIO((metrics_dir / "T06_smoke_downstream_resnet18_video.csv").read_text()))}
    out, ok = {}, True
    for m in DETECTORS:
        d = evaluator_dir / m
        det = json.loads((d / "detector.json").read_text())
        epochs = json.loads((d / "epochs.json").read_text())
        per = []
        for e in range(det["epochs"]):
            rows = _read_parquet(d / f"predictions_val_select_epoch{e}.parquet")
            v = _videos(rows)
            y, s = [x["y"] for x in v], [x["score"] for x in v]
            t, acer = _threshold(y, s)
            per.append({"epoch": e, "threshold": t, "acer": acer, "auc": _auc(y, s),
                        "video_score_consistent": all(x["reported"] == {x["score"]} for x in v),
                        "threshold_matches_file": {r["threshold"] for r in rows} == {t},
                        "matches_epochs_json": (float.fromhex(epochs[e]["threshold_hex"]), Fraction(epochs[e]["val_acer"]),
                                                Fraction(epochs[e]["val_auc"])) == (t, acer, _auc(y, s))})
        best = min(per, key=lambda p: (p["acer"], -p["auc"], p["epoch"]))
        rows = _read_parquet(d / "predictions_holdout.parquet")
        v = _videos(rows)
        y, s = [x["y"] for x in v], [x["score"] for x in v]
        thr = best["threshold"]
        r = _rates(y, s, thr)
        r.update(AUC=_auc(y, s), EER=_eer(y, s), **{"TPR@FPR=1%": _tpr(y, s, "0.01"), "TPR@FPR=5%": _tpr(y, s, "0.05")})
        table = t06[m]
        cmp = {k: _exact(r[k]) == table[k + "_exact"] for k in r}
        gates = {"epoch_rederivations_consistent": all(p["video_score_consistent"] and p["threshold_matches_file"]
                                                       and p["matches_epochs_json"] for p in per),
                 "selected_epoch_matches": best["epoch"] == det["selected_epoch"],
                 "holdout_uses_frozen_threshold": {x["threshold"] for x in rows} == {thr}
                 and float.fromhex(det["frozen_threshold_hex"]) == thr,
                 "holdout_video_scores_consistent": all(x["reported"] == {x["score"]} for x in v),
                 "decisions_follow_D4": all(x["pred_frame"] == int(x["p_frame"] >= thr) and
                                            x["pred_video"] == int(x["video_score"] >= thr) for x in rows),
                 "t06_values_match": all(cmp.values())}
        ok &= all(gates.values())
        out[m] = {"gates": gates, "t06_comparison": cmp, "selected_epoch": best["epoch"], "threshold": thr,
                  "ACER": _exact(r["ACER"]), "AUC": _exact(r["AUC"]), "EER": _exact(r["EER"])}
    return {"ok": ok, "detectors": out, "implementation": "independent (tools/m8_smoke_audit.py; no gpatbench metrics import)"}


# ---------------------------------------------------------------- audit
def run_checks(root: Path) -> dict:
    run = json.loads((root / "smoke_run.json").read_text())
    st = {s: json.loads((root / s / "stage.json").read_text()) for s in ("select", "banks", "evaluator", "metrics", "plots")}
    fw = {s: r.get("firewall") for s, r in st.items() if r.get("firewall")}
    led = json.loads((root / "evaluator" / "faces_ledger.json").read_text())["summary"]
    return {"run": run, "stages": st, "firewall": fw, "ledger": led,
            "ok": run["ok"] and all(r["ok"] for r in st.values()) and all(x["exit_code"] == 0 for x in run["stages"])
            and all(f["violation_count"] == 0 and f["listing_attempts"] == 0 for f in fw.values())
            and led["atime_changed_files"] == 0}


def audit(a) -> int:
    ra, rb, out = Path(a.run_a), Path(a.run_b), Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    ca, cb = run_checks(ra), run_checks(rb)
    mode = ca["run"]["mode"]
    fa, fb = deterministic_files(ra), deterministic_files(rb)
    diff = sorted(k for k in set(fa) | set(fb) if fa.get(k) != fb.get(k))
    (out / "artifact_hashes_run_a.json").write_text(json.dumps(fa, indent=1, sort_keys=True) + "\n")
    (out / "artifact_hashes_run_b.json").write_text(json.dumps(fb, indent=1, sort_keys=True) + "\n")
    red_a = rederive(ra / "evaluator", ra / "metrics")
    env = json.loads(Path(a.env_evidence).read_text())
    man = json.loads((ra / "select" / "toy_manifest.json").read_text())
    positions = [p["position"] for p in man["toy_pairs"]]
    bs = ca["stages"]["banks"]["info"]["banks"]
    ev = ca["stages"]["evaluator"]
    steps = {m: [json.loads(l) for l in (ra / "evaluator" / m / "steps.jsonl").read_text().splitlines()] for m in DETECTORS}
    real = mode == "real"
    coverage = {
        "banks_all_methods_seeds": sorted(bs) == sorted(f"{m}.seed_{s}" for m, s in BANKS),
        "bank_rows_equal_toy_pairs": all(v["rows"] == len(positions) and v["failures"] == 0 for v in bs.values()),
        "toy_pairs_186": (len(positions) == 186) if real else len(positions) > 0,
        "detectors_E00_E01_E02_seed42": sorted(ev["info"]) == list(DETECTORS) and all(v["seed"] == 42 for v in ev["info"].values()),
        "six_steps_each": all(len(steps[m]) == 6 for m in DETECTORS),
        "frozen_batch_composition": all((s["n_live"], s["n_real_spoof"], s["n_synthetic"]) == ((32, 32, 0) if m == "E00" else (32, 16, 16))
                                        for m in DETECTORS for s in steps[m]),
        "cpu_no_amp": all(v["device"] == "cpu" and v["amp"] is False for v in ev["info"].values()),
    }
    firewall = {"violations": sum(f["violation_count"] for c in (ca, cb) for f in c["firewall"].values()),
                "listing_attempts": sum(f["listing_attempts"] for c in (ca, cb) for f in c["firewall"].values()),
                "test_rows_used": man["test_rows_read"], "atime_changed_files": ca["ledger"]["atime_changed_files"]
                + cb["ledger"]["atime_changed_files"], "faces_files_read_run_a": ca["ledger"]["files"],
                "allowlist_size": len(man["allowlist"]), "method": "parquet TRAIN/VAL row filter + allowlist audit hook "
                "(opens outside the allowlist and directory listings under faces_256 raise) + O_NOATIME reads"}
    gates = {"run_a_stages_ok": ca["ok"], "run_b_stages_ok": cb["ok"], "replay_identical": not diff and bool(fa),
             "rederivation_ok": red_a["ok"], "environment_V1_V6": env["all_pass"], **coverage,
             "firewall_clean": firewall["violations"] == 0 and firewall["listing_attempts"] == 0
             and firewall["test_rows_used"] == 0 and firewall["atime_changed_files"] == 0,
             "clean_source_tree": not ca["run"]["worktree_dirty"] and not cb["run"]["worktree_dirty"]
             and ca["run"]["git_commit"] == cb["run"]["git_commit"]}

    def cmds(stage):
        return [{"run": n, "argv": s["argv"], "exit_code": s["exit_code"], "wall_seconds": s["wall_seconds"]}
                for n, c in (("A", ca), ("B", cb)) for s in c["run"]["stages"] if s["stage"] == stage]

    def stage_gates(names):
        g = {}
        for n in names:
            g.update({f"A.{n}.{k}": v for k, v in ca["stages"][n].get("gates", {}).items()})
            g.update({f"B.{n}.{k}": v for k, v in cb["stages"][n].get("gates", {}).items()})
        return g

    bank_gates = stage_gates(["banks"])
    gen = {k: v for k, v in bank_gates.items() if "completed" in k or "operator" in k or "k40" in k or "toy_inputs" in k}
    bank = {k: v for k, v in bank_gates.items() if k not in gen}
    stages = {
        "generator": {"executed": True, "simulated": False, "gates": {**gen, **stage_gates(["select"])},
                      "commands": cmds("select") + cmds("banks")},
        "bank": {"executed": True, "simulated": False, "gates": {**bank, "replay_identical": gates["replay_identical"]},
                 "commands": cmds("banks")},
        "evaluator": {"executed": True, "simulated": False, "gates": {**stage_gates(["evaluator"]), **coverage},
                      "commands": cmds("evaluator")},
        "metrics": {"executed": True, "simulated": False, "gates": {**stage_gates(["metrics"]), "independent_rederivation": red_a["ok"]},
                    "commands": cmds("metrics")},
        "plots": {"executed": True, "simulated": False, "gates": stage_gates(["plots"]), "commands": cmds("plots")}}
    for s in stages.values():
        s["status"] = "PASS" if s["gates"] and all(s["gates"].values()) and all(c["exit_code"] == 0 for c in s["commands"]) else "FAIL"
    all_ok = all(gates.values()) and all(s["status"] == "PASS" for s in stages.values())
    passed = all_ok and real            # a synthetic qualification run never yields a passing smoke record
    record = {"schema_version": "m8_smoke_test_record_v1", "record": "M8_SMOKE_TEST_RECORD", "mode": mode,
              "label": "SMOKE_NON_SCIENTIFIC", "contract_sha256": sha(ROOT / SMOKE_CONTRACT),
              "evaluator_contract_sha256": sha(ROOT / EVAL_CONTRACT),
              "source": {"git_commit": ca["run"]["git_commit"], "worktree_dirty": ca["run"]["worktree_dirty"] or cb["run"]["worktree_dirty"]},
              "environment": {"lock_sha256": env["lock_sha256"], "gates": env["gates"]},
              "toy_selection": {"toy_manifest_sha256": fa.get("select/toy_manifest.json"), "toy_pairs": len(positions),
                                "frames": {k: len(v) for k, v in man["roles"].items()},
                                "siwmv2_unit_disclosure": man["siwmv2_unit_disclosure"]},
              "coverage": {"banks": bs, "detectors": {m: {k: ev["info"][m][k] for k in ("selected_epoch", "total_steps",
                                                                                         "holdout_predictions_sha256")}
                                                      for m in DETECTORS}},
              "stages": stages, "gates": gates,
              "independent_replay": {"identical": gates["replay_identical"], "artifacts_compared": len(fa), "differences": diff[:50]},
              "rederivation": red_a, "test_firewall": firewall, "scientific_results_produced": False,
              "smoke_test_passed": passed,
              "not_authorized_by_this_record": ["production banks", "binding of smoke_test_record", "O3 completion",
                                                "scientific downstream training", "TEST access"]}
    from gpatbench.downstream.smoke_evidence import validate_smoke_record
    v = validate_smoke_record(record, ROOT) if passed else {"admissible": True, "passed": False, "problems": ["gates failed"]}
    record["validator"] = {"admissible": v["admissible"], "passed": v["passed"], "problems": v["problems"]}
    (out / "smoke_record_candidate.json").write_text(json.dumps(record, indent=1, sort_keys=True, default=str) + "\n")
    for rel in COPY_SMALL:
        src = ra / rel
        dst = out / "run_a" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        (shutil.copytree if src.is_dir() else shutil.copy2)(src, dst)
    for m in DETECTORS:
        for f in (ra / "evaluator" / m).iterdir():
            if f.name != "timing.json":
                (out / "run_a" / "evaluator" / m).mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, out / "run_a" / "evaluator" / m / f.name)
    for m, s in BANKS:
        for n in ("bank_manifest.parquet", "bank_manifest.sha256"):
            d = out / "run_a" / "banks" / f"{m}_seed_{s}"
            d.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ra / "banks" / "runs" / "m6" / m / f"seed_{s}" / n, d / n)
    for n, c in (("A", ca), ("B", cb)):
        (out / f"run_{n.lower()}_smoke_run.json").write_text(json.dumps(c["run"], indent=1) + "\n")
        for s in c["stages"]:
            (out / f"run_{n.lower()}_stage_{s}.json").write_text(json.dumps(c["stages"][s], indent=1, sort_keys=True,
                                                                             default=str) + "\n")
    record["qualification_all_gates_pass"] = all_ok
    (out / "smoke_record_candidate.json").write_text(json.dumps(record, indent=1, sort_keys=True, default=str) + "\n")
    print(json.dumps({"mode": mode, "all_gates_pass": all_ok, "smoke_test_passed": passed, "validator": record["validator"],
                      "gates": gates}, indent=1))
    return 0 if (passed and v["passed"]) or (not real and all_ok) else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("audit", "rederive"))
    ap.add_argument("--run-a")
    ap.add_argument("--run-b")
    ap.add_argument("--env-evidence")
    ap.add_argument("--out")
    ap.add_argument("--evidence")
    a = ap.parse_args(argv)
    if a.cmd == "audit":
        return audit(a)
    ev = Path(a.evidence)
    r = rederive(ev / "run_a" / "evaluator", ev / "run_a" / "metrics")
    print(json.dumps({"ok": r["ok"], "detectors": {m: d["gates"] for m, d in r["detectors"].items()}}, indent=1))
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
