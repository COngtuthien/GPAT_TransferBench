"""M10 ResNet-18 TRAIN/VAL matrix closure (laptop side, stdlib only). VAL findings only; TEST is never read.

    python3 tools/m10_matrix_closure.py --host-listings DIR [--out-prefix outputs/audit/M10_TRAIN_VAL_MATRIX_CLOSURE]

Inputs (all committed): the launch authority, outputs/audit/M10_RUNS/M10_<G>_SEED<S>_AUDIT.json and the per-run
evidence folder outputs/audit/M10_RUNS/<G>_seed<S>/ (run_root_sha256.txt, evidence/, diagnostics.json).
--host-listings holds a fresh `sha256sum` listing of every run root re-collected on the GPU host
(<G>_seed<S>.sha256, same format as run_root_sha256.txt) so that every recorded output path is shown to still exist
with its recorded hash. Fails closed: the status is M10_ALL_24_TRAIN_VAL_RUNS_COMPLETED_AND_AUDITED only if every
gate passes for all 24 runs; otherwise it reports the exact partial state.
"""
import argparse
import hashlib
import json
import statistics
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUTH = "configs/amendments/m10_resnet18_science_launch_authority.yaml"
M10_ROOT = "/media/ext_data/student20261_gpat_m10_detectors"
DISCLOSURES = {
    "E05": "E05 PCGAN: controlled architecture resolution; low-contrast outputs; seed 1337 bank near-collapse "
           "disclosed and retained under RAW-MATCHED (O6), no quality filter "
           "(outputs/audit/LEARNED_BANK_PRODUCTION_E05_FAMILY_CLOSURE.json)",
    "E04": "E04 Physics-STD: CONTROLLED_ADAPTATION, not an official Physics-STD inference reproduction; clipped outputs "
           "included under RAW-MATCHED (outputs/audit/LEARNED_BANK_PRODUCTION_E04_FAMILY_CLOSURE.json)",
    "E06b": "E06b DSDG-NATIVE: track B_NATIVE_FULL_SECONDARY (O7); reported separately, never pooled into the common "
            "identity-free primary comparison (A_FAIR_COMMON_IDENTITY_FREE)",
    "E00": "E00: real-only binary baseline (no synthetic image)",
}


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def parse_listing(text):
    out = {}
    for line in text.splitlines():
        if line.strip():
            h, f = line.split("  ", 1)
            out[f] = h
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host-listings", required=True)
    ap.add_argument("--out-prefix", default="outputs/audit/M10_TRAIN_VAL_MATRIX_CLOSURE")
    a = ap.parse_args(argv)
    auth = json.loads((ROOT / AUTH).read_text())
    order = [(g, int(s)) for g, s in auth["matrix"]["order"]]
    tracks = auth["matrix"]["tracks"]
    runs, gates = [], {}
    for g, s in order:
        key = f"{g}|{s}"
        rec_p = ROOT / f"outputs/audit/M10_RUNS/M10_{g}_SEED{s}_AUDIT.json"
        ev = ROOT / f"outputs/audit/M10_RUNS/{g}_seed{s}"
        r = {"group": g, "seed": s, "track_id": tracks[g], "run_root": f"{M10_ROOT}/{g}/seed_{s}"}
        chk = {}
        if not rec_p.is_file() or not (ev / "evidence/run_manifest.json").is_file():
            chk["present"] = False
            r.update(status="NOT_COMPLETED_OR_NOT_AUDITED", checks=chk)
            runs.append(r)
            gates[key] = False
            continue
        A = json.loads(rec_p.read_text())
        M = json.loads((ev / "evidence/run_manifest.json").read_text())
        SEL = json.loads((ev / "evidence/detector/selection.json").read_text())
        EP = json.loads((ev / "evidence/detector/epochs.json").read_text())
        lst_txt = (ev / "run_root_sha256.txt").read_text()
        lst = parse_listing(lst_txt)
        host_p = Path(a.host_listings) / f"{g}_seed{s}.sha256"
        host = parse_listing(host_p.read_text()) if host_p.is_file() else None
        ck = M["checkpoint"]
        sel_ep = SEL["selected_epoch"]
        pred_rel = f"detector/predictions_val_epoch{sel_ep:02d}.parquet"
        chk["present"] = True
        chk["audit_pass"] = A["status"] == "PASS" and A["label"] == "SCIENTIFIC" and not A["failed_gates"] \
            and A["gates_passed"] == A["gates_total"] and A["group"] == g and A["seed"] == s \
            and A["run_root"] == r["run_root"]
        chk["manifest_identity"] = M["method_id"] == g and M["detector_seed"] == s and M["track_id"] == tracks[g] \
            and M["label"] == "SCIENTIFIC" and M["source_commit"] == auth["bindings"]["source"]["commit"] \
            and M["code_sha256"] == auth["bindings"]["source"]["code_sha256"] and M["test_accessed"] is False \
            and M["bank_seed"] == (None if g == "E00" else s) \
            and ((M["synthetic_bank"] is None) if g == "E00"
                 else M["synthetic_bank"]["manifest_sha256"] == auth["bank_families"][g]["banks"][str(s)])
        chk["manifest_hash_in_listing"] = lst.get("run_manifest.json") == sha(ev / "evidence/run_manifest.json")
        chk["evidence_copies_match_listing"] = all(
            lst.get(p.relative_to(ev / "evidence").as_posix()) == sha(p)
            for p in (ev / "evidence").rglob("*") if p.is_file() and "host" not in p.relative_to(ev / "evidence").parts
            and not p.name.endswith(".gz"))
        chk["checkpoint_hash_consistent"] = lst.get(ck["path"]) == ck["file_sha256"] == SEL["checkpoint_file_sha256"] \
            and ck["epoch"] == sel_ep and [f for f in lst if f.endswith(".pt")] == [ck["path"]]
        chk["val_predictions_all_epochs"] = len(M["val_predictions"]) == 40 and all(
            lst.get(v["path"]) == v["sha256"] and v["frames"] == 3121 and v["videos"] == 391 for v in M["val_predictions"])
        chk["selected_predictions_committed"] = (ev / "evidence" / pred_rel).is_file() \
            and sha(ev / "evidence" / pred_rel) == lst.get(pred_rel)
        chk["host_listing_identical"] = host is not None and host == lst
        chk["epochs_40_steps_7000"] = len(EP) == 40 and M["total_steps"] == 7000 and M["epochs"] == 40 \
            and M["optimizer_steps_applied"] + M["optimizer_steps_skipped"] == 7000
        win = min(EP, key=lambda e: (Fraction(e["val_acer"]), -Fraction(e["val_auc"]), e["epoch"]))
        chk["selection_rederived"] = win["epoch"] == sel_ep and win["threshold_hex"] == SEL["frozen_val_threshold_hex"]
        r.update(status="COMPLETED_AND_AUDITED" if all(chk.values()) else "CHECK_FAILED", checks=chk,
                 run_id=M["run_id"], execution_commit=M["execution_commit"], audit_record=str(rec_p.relative_to(ROOT)),
                 audit_record_sha256=sha(rec_p), audit_gates=f"{A['gates_passed']}/{A['gates_total']}",
                 checkpoint_path=f"{r['run_root']}/{ck['path']}", checkpoint_file_sha256=ck["file_sha256"],
                 checkpoint_state_sha256=ck["state_sha256"], selected_epoch=sel_ep,
                 frozen_val_threshold_hex=SEL["frozen_val_threshold_hex"],
                 frozen_val_threshold=float.fromhex(SEL["frozen_val_threshold_hex"]),
                 val_acer=SEL["val_acer"], val_auc=SEL["val_auc"],
                 val_acer_float=float(Fraction(SEL["val_acer"])), val_auc_float=float(Fraction(SEL["val_auc"])),
                 selected_predictions={"path": f"{r['run_root']}/{pred_rel}", "sha256": lst.get(pred_rel),
                                       "committed_copy": str((ev / "evidence" / pred_rel).relative_to(ROOT))},
                 run_root_listing_sha256=hashlib.sha256(lst_txt.encode()).hexdigest(), run_root_files=len(lst),
                 optimizer_steps_skipped=M["optimizer_steps_skipped"],
                 synthetic_bank_manifest_sha256=None if g == "E00" else M["synthetic_bank"]["manifest_sha256"],
                 start_utc=M["start_utc"], end_utc=M["end_utc"])
        gates[key] = all(chk.values())
        runs.append(r)
    summary = {}
    for g in auth["matrix"]["groups"]:
        rs = [r for r in runs if r["group"] == g and r["status"] == "COMPLETED_AND_AUDITED"]
        acer = [r["val_acer_float"] for r in rs]
        auc = [r["val_auc_float"] for r in rs]
        summary[g] = {"track_id": tracks[g], "seeds_completed": [r["seed"] for r in rs],
                      "val_video_acer_mean": statistics.fmean(acer) if acer else None,
                      "val_video_acer_sd_sample": statistics.stdev(acer) if len(acer) >= 2 else None,
                      "val_video_auc_mean": statistics.fmean(auc) if auc else None,
                      "val_video_auc_sd_sample": statistics.stdev(auc) if len(auc) >= 2 else None,
                      "selected_epochs": [r["selected_epoch"] for r in rs],
                      "pooled_in_primary_identity_free_comparison": tracks[g] == "A_FAIR_COMMON_IDENTITY_FREE"}
    done = sum(gates.values())
    status = "M10_ALL_24_TRAIN_VAL_RUNS_COMPLETED_AND_AUDITED" if done == 24 else "M10_PARTIAL_RUNS_COMPLETED"
    rec = {"record": "M10_TRAIN_VAL_MATRIX_CLOSURE", "schema": "m10_train_val_matrix_closure_v1", "status": status,
           "runs_completed_and_audited": done, "runs_total": 24,
           "launch_authority": {"path": AUTH, "sha256": sha(ROOT / AUTH)},
           "source_commit": auth["bindings"]["source"]["commit"],
           "training_configuration": {k: auth["bindings"][k] for k in ("frozen_config", "evaluator_contract",
                                                                       "owner_decisions", "precision_pin",
                                                                       "environment", "imagenet_weights", "gpu",
                                                                       "storage_authority")},
           "data_identity": auth["data"], "bank_families": auth["bank_families"],
           "runs": runs, "val_summary_by_group": summary,
           "summary_scope": "VAL findings only (video-level VAL ACER/AUC of each run's selected checkpoint; mean and "
                            "sample standard deviation over detector seeds). Not TEST results; no seed or checkpoint "
                            "was chosen from this cross-seed comparison.",
           "disclosures": DISCLOSURES,
           "test_accessed": False,
           "backup": "single copy of each selected checkpoint under the HDD detector root; committed SHA256 values are "
                     "checksums, not backups",
           "gates": gates}
    out = ROOT / a.out_prefix
    js = json.dumps(rec, indent=1, sort_keys=True, default=str) + "\n"
    Path(f"{out}.json").write_text(js)
    md = [f"# M10 ResNet-18 TRAIN/VAL matrix closure\n", f"Status: **{status}** ({done}/24 runs completed and audited)\n",
          "VAL findings only. TEST was not accessed. These are not benchmark (TEST) results.\n",
          "| Seed | " + " | ".join(auth["matrix"]["groups"]) + " |", "|---|" + "---|" * len(auth["matrix"]["groups"])]
    for s in auth["matrix"]["seeds"]:
        cells = []
        for g in auth["matrix"]["groups"]:
            r = next(x for x in runs if x["group"] == g and x["seed"] == s)
            cells.append(f"PASS ep{r['selected_epoch']} ACER {r['val_acer_float']:.4f}" if gates[f"{g}|{s}"]
                         else r["status"])
        md.append(f"| {s} | " + " | ".join(cells) + " |")
    md += ["", "## VAL summary by group (selected checkpoints; mean ± sample SD over seeds)", "",
           "| Group | Track | Seeds | VAL video ACER | VAL video AUC |", "|---|---|---|---|---|"]
    for g, v in summary.items():
        f = lambda m, sd: "n/a" if m is None else (f"{m:.4f}" + ("" if sd is None else f" ± {sd:.4f}"))  # noqa: E731
        md.append(f"| {g} | {v['track_id']} | {len(v['seeds_completed'])} | "
                  f"{f(v['val_video_acer_mean'], v['val_video_acer_sd_sample'])} | "
                  f"{f(v['val_video_auc_mean'], v['val_video_auc_sd_sample'])} |")
    md += ["", "## Runs", "", "| Group | Seed | Run ID | Audit | Epoch | Threshold | Checkpoint SHA256 |",
           "|---|---|---|---|---|---|---|"]
    for r in runs:
        if r.get("run_id"):
            md.append(f"| {r['group']} | {r['seed']} | {r['run_id']} | {r['audit_gates']} | {r['selected_epoch']} | "
                      f"{r['frozen_val_threshold']:.6g} | `{r['checkpoint_file_sha256']}` |")
    md += ["", "## Disclosures", ""] + [f"- {v}" for v in DISCLOSURES.values()]
    md += ["", "Checkpoint bytes: single copy on the HDD detector root (no independent backup).", ""]
    Path(f"{out}.md").write_text("\n".join(md))
    Path(f"{out}_SHA256.txt").write_text("".join(f"{sha(p)}  {Path(p).relative_to(ROOT)}\n"
                                                 for p in (f"{out}.json", f"{out}.md")))
    print(json.dumps({"status": status, "completed": done}))
    return 0 if done == 24 else 1


if __name__ == "__main__":
    raise SystemExit(main())
