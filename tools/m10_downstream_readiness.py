"""M10 downstream ResNet-18 scientific readiness record, run-manifest template and launch-authority DRAFT.

    python3 tools/m10_downstream_readiness.py
      reads the committed evidence under outputs/audit/M10_DOWNSTREAM_READINESS/ (host bank/input verification, host
      synthetic-fixture qualification, resource snapshot, laptop test summaries) and the protocol audit, and writes
        outputs/audit/M10_DOWNSTREAM_SCIENTIFIC_READINESS.{json,md} (+ _SHA256.txt)
        outputs/audit/M10_RUN_MANIFEST_TEMPLATE.json
        configs/amendments/m10_resnet18_science_launch_authority.DRAFT.yaml   (DRAFT; approved false; never activated)
Nothing here launches, authorizes or activates anything.
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EV = "outputs/audit/M10_DOWNSTREAM_READINESS"
OUT = "outputs/audit/M10_DOWNSTREAM_SCIENTIFIC_READINESS"
TEMPLATE = "outputs/audit/M10_RUN_MANIFEST_TEMPLATE.json"
DRAFT = "configs/amendments/m10_resnet18_science_launch_authority.DRAFT.yaml"
AUDIT = "outputs/audit/M10_DOWNSTREAM_PROTOCOL_AUDIT.json"
CODE = ("gpatbench/downstream/science_runner.py", "gpatbench/downstream/resnet_runner.py",
        "gpatbench/downstream/metrics.py", "gpatbench/downstream/science_banks.py",
        "gpatbench/downstream/science_inputs.py", "gpatbench/downstream/science_manifest.py",
        "gpatbench/downstream/firewall.py", "gpatbench/downstream/faceio.py", "gpatbench/downstream/bank_reader.py",
        "methods/common/learned_bank.py", "methods/common/bank_writer.py")
ENV_LOCK = "environments/m8_downstream.lock.json"
GROUPS = ("E00", "E01", "E02", "E03", "E04", "E05", "E06b", "E06c")
SEEDS = (42, 1337, 2026)


def sha(rel):
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def load(rel):
    return json.loads((ROOT / rel).read_text())


OWNER_DECISIONS = [
    {"id": "O1_science_precision", "blocking": True,
     "question": "D13 leaves the science AMP dtype and GradScaler policy (and the CUDA determinism settings) to be "
                 "pinned before M10. Which precision policy do all 24 runs use?",
     "options": {"A_CUDA_AMP_FP16_GRADSCALER_DETERMINISTIC": "torch.autocast(cuda, float16) + torch.amp.GradScaler('cuda') "
                 "defaults (init 65536, growth 2, backoff 0.5, interval 2000); scaler.unscale_ before clip_grad_norm_; "
                 "deterministic algorithms, cudnn deterministic, benchmark off, TF32 off, CUBLAS_WORKSPACE_CONFIG=:4096:8",
                 "B_CUDA_AMP_BF16_DETERMINISTIC": "torch.autocast(cuda, bfloat16), no GradScaler; same determinism settings",
                 "C_CUDA_FP32": "not eligible: spec 14 / downstream_resnet18.yaml fix amp: true"},
     "recommended": "A_CUDA_AMP_FP16_GRADSCALER_DETERMINISTIC",
     "why": "the standard CUDA AMP of the spec era; qualified bitwise replay on the RTX 3090 (synthetic fixtures); "
            "fastest measured step; skipped steps (inf gradients) are logged per step",
     "activation": "configs/amendments/m10_resnet18_science_precision.yaml {status OWNER_APPROVED, approved true, policy}"},
    {"id": "O2_threshold_tie_rule", "blocking": True,
     "question": "Spec 15 breaks ACER ties only by 'closest to 0.5'. When two minimum-ACER thresholds are equally close "
                 "to 0.5 (always the case for constant or fully inverted VAL scores: 0 and 1 at ACER 1/2) the frozen rule "
                 "selects nothing and the implementation aborts the whole run. Which completion applies?",
     "options": {"A_SMALLER_THRESHOLD": "take the smaller of the tied thresholds; the epoch stays eligible",
                 "B_LARGER_THRESHOLD": "take the larger of the tied thresholds",
                 "C_EPOCH_INELIGIBLE": "record the epoch with no threshold and exclude it from checkpoint selection; the "
                                       "run fails only if every epoch is ineligible",
                 "D_FAIL_CLOSED": "keep the current behaviour (any such epoch aborts the run for owner review)"},
     "recommended": "C_EPOCH_INELIGIBLE",
     "why": "a tie this exact arises from degenerate or symmetric score sets; excluding such an epoch never lets an "
            "arbitrary direction choice decide the frozen threshold, and D5 still selects among well-defined epochs. "
            "Reproduced deterministically (qualification Q6, tests/test_m10_downstream_readiness)",
     "activation": "owner pin record + implementation change in science_runner (re-qualification required)"},
    {"id": "O3_checkpoint_retention_and_run_root", "blocking": True,
     "question": "Which checkpoints are kept, and where do detector run roots live? The system SSD holding the runtime "
                 "root is ~97% full (~27 GB free); the approved HDD has ~3.0 TB free but only the bank root is registered.",
     "options": {"A_SELECTED_ONLY_HDD_NEW_ROOT": "keep only the selected checkpoint (~45 MB/run, ~1.1 GB for 24); run roots "
                                                 "under a new owner-approved HDD directory outside the bank root, e.g. "
                                                 "/media/ext_data/student20261_gpat_m10_detectors/<group>/seed_<s>/",
                 "B_SELECTED_ONLY_RUNTIME_ROOT": "selected checkpoint only, run roots under <runtime_root>/runs/m10/ on "
                                                 "the system SSD (~1.3 GB total; reduces SSD headroom)",
                 "C_ALL_EPOCHS_HDD_NEW_ROOT": "keep all 40 epoch checkpoints (~1.8 GB/run, ~43 GB) on a new HDD root"},
     "recommended": "A_SELECTED_ONLY_HDD_NEW_ROOT",
     "why": "D5 needs every epoch evaluated but only the selected checkpoint frozen; the SSD has little headroom; the HDD "
            "root must stay separate from the immutable bank root (its contents check refuses foreign entries)"},
    {"id": "O4_science_environment", "blocking": True,
     "question": "The evaluator contract authorized gpat-m8-downstream (environments/m8_downstream.lock.json) for the "
                 "smoke only. Approve it (unchanged) for the M10 scientific runs?",
     "options": {"A_APPROVE_UNCHANGED": "torch 2.12.1+cu130, torchvision 0.27.1, CUDA 13.0, driver 580.178.04 as locked",
                 "B_NEW_ENVIRONMENT": "a separately locked environment (needs qualification)"},
     "recommended": "A_APPROVE_UNCHANGED", "why": "the same stack ran the passed smoke and this qualification"},
    {"id": "O5_execution_order_and_gpu_sharing", "blocking": False,
     "question": "Execution order and GPU policy on the shared RTX 3090 (an unrelated user's jobs held ~10-12 GB and "
                 "~93% utilization during this assessment).",
     "options": {"A_SEED_MAJOR": "seeds 42 -> 1337 -> 2026; within a seed E00, E01, E02, E03, E04, E05, E06b, E06c; one "
                                 "detector process at a time; independent audit committed before the next run",
                 "B_GROUP_MAJOR": "E00 (42, 1337, 2026), then E01 ... E06c"},
     "recommended": "A_SEED_MAJOR", "why": "a complete comparison at seed 42 first; identical rules either way"},
    {"id": "O6_primary_table_view", "blocking": False,
     "question": "Spec 7: state the primary table (RAW-MATCHED or QUALITY-MATCHED) before inspecting TEST. All seven "
                 "previous-work banks have 8838 valid outputs; N_common across the final comparison depends on later GPAT "
                 "banks. Not needed before training; needed before any TEST evaluation.",
     "options": {"A_RAW_MATCHED_PRIMARY": "RAW-MATCHED primary, QUALITY-MATCHED secondary",
                 "B_QUALITY_MATCHED_PRIMARY": "QUALITY-MATCHED primary"},
     "recommended": "A_RAW_MATCHED_PRIMARY", "why": "isolates the raw generators; the M10 matrix here is RAW-MATCHED"},
    {"id": "O7_track_id_labels", "blocking": False,
     "question": "Prediction rows carry track_id. Confirm the A1 track per method (method_status.csv) and 'binary' for E00 "
                 "(spec 17), replacing the smoke runner's fixed 'binary'.",
     "options": {"A_CONFIRM": "E01-E05, E06c A_FAIR_COMMON_IDENTITY_FREE; E06b B_NATIVE_FULL_SECONDARY; E00 binary"},
     "recommended": "A_CONFIRM", "why": "keeps E06b's native-track identity visible in every prediction file"},
]


def build():
    from gpatbench.downstream import science_banks as SB, science_manifest as SM
    pa = load(AUDIT)
    bv = load(EV + "/bank_loader_verification.json")
    q = load(EV + "/qualification.json")
    res = load(EV + "/resources.json")
    tests = load(EV + "/tests.json")
    gates = {}
    gates["G01_protocol_authorities_verified"] = pa["authorities"]["configs/amendments/m8_downstream_evaluator_contract.yaml"] \
        == "67e4b62ddc6b81b8b9a8657e71b3979f6e6a186827183b052ae3d29b587fcbe0"
    gates["G02_seven_families_21_banks_reverified"] = bv["all_ok"] is True and len(bv["banks"]) == 21 and all(
        b["ok"] and b["outputs"] == 8838 and b["reader"]["files_checked"] == 8838 for b in bv["banks"].values())
    gates["G03_bank_manifests_equal_audits"] = all(
        bv["banks"]["%s|%d" % (m, s)]["binding"]["manifest_sha256"] == SB.bank_binding(ROOT, m, s)["manifest_sha256"]
        for m in SB.METHODS for s in SEEDS)
    gates["G04_e05_low_contrast_and_e04_clipped_included"] = all(
        bv["banks"]["E05|%d" % s]["e05_low_contrast_loaded_unchanged"] for s in SEEDS) and all(
        bv["banks"]["E04|%d" % s]["e04_clipped_rows_present"] and bv["banks"]["E04|%d" % s]["e04_widest_16_loaded_unchanged"]
        for s in SEEDS)
    gates["G05_populations_and_split_firewall"] = bv["populations"]["counts"] == pa["populations"]["counts"] \
        and bv["populations"]["pair_sources_equal_train_spoof"] and bv["populations"]["pair_targets_train_live"] \
        and bv["real_faces"]["ledger"]["atime_changed_files"] == 0
    gates["G06_augmentation_equivalence"] = q["Q1_augmentation_equivalence"]["pass"] is True
    gates["G07_runner_bitwise_equal_to_qualified_smoke_runner"] = q["Q2_cpu_equivalence_with_smoke_runner"]["pass"] is True
    gates["G08_cuda_policies_replay_and_checkpoint_reload"] = q["Q3_cuda_policies_replay"]["pass"] is True
    gates["G09_plots"] = q["Q5_plots"]["pass"] is True
    gates["G10_laptop_tests"] = tests["m10_readiness"]["failures"] == 0 and tests["m10_readiness"]["errors"] == 0 \
        and tests["full_suite"]["new_failures"] == [] and tests["full_suite"]["identical_to_baseline"] is True
    gates["G11_threshold_rule_complete"] = False            # O2 (qualification Q6 reproduces the gap)
    gates["G12_science_precision_pinned"] = (ROOT / "configs/amendments/m10_resnet18_science_precision.yaml").exists()
    gates["G13_checkpoint_retention_and_run_root_approved"] = False   # O3
    gates["G14_science_environment_approved"] = False                 # O4
    gates["G15_launcher_and_independent_run_auditor"] = False         # to implement after O1-O4
    gates["G16_resources"] = res["gpu"]["name"] == "NVIDIA GeForce RTX 3090" and res["gpu"]["total_mib"] >= 24000 \
        and res["hdd"]["free_bytes"] > 100 << 30
    engineering = [k for k in ("G01_protocol_authorities_verified", "G02_seven_families_21_banks_reverified",
                               "G03_bank_manifests_equal_audits", "G04_e05_low_contrast_and_e04_clipped_included",
                               "G05_populations_and_split_firewall", "G06_augmentation_equivalence",
                               "G07_runner_bitwise_equal_to_qualified_smoke_runner",
                               "G08_cuda_policies_replay_and_checkpoint_reload", "G09_plots", "G10_laptop_tests",
                               "G16_resources") if not gates[k]]
    status = ("M8_DOWNSTREAM_BLOCKED_BY_DATA_INTEGRITY" if not (gates["G02_seven_families_21_banks_reverified"]
                                                               and gates["G03_bank_manifests_equal_audits"])
              else "M8_DOWNSTREAM_BLOCKED_BY_IMPLEMENTATION" if engineering
              else "M8_DOWNSTREAM_SCIENTIFIC_READY_AUTHORITY_PENDING" if all(gates.values())
              else "M8_DOWNSTREAM_BLOCKED_BY_SCIENTIFIC_CONTRACT")
    q4 = q["Q4_throughput"]
    step = q4["gpu_train_step_seconds"]["CUDA_AMP_FP16_GRADSCALER_DETERMINISTIC"]["median_after_10"] \
        + q4["cpu_batch64_decode_augment_seconds"]["8"]
    val_epoch = 49 * (q4["gpu_eval_64_seconds"]["CUDA_AMP_FP16_GRADSCALER_DETERMINISTIC"] + q4["cpu_batch64_decode_augment_seconds"]["8"])
    per_run = 7000 * step + 40 * val_epoch
    plan = {"runs": [{"order": i + 1, "group": g, "detector_seed": s, "bank_seed": None if g == "E00" else s,
                      "run_root": "<approved detector root>/%s/seed_%d/" % (g, s)}
                     for i, (s, g) in enumerate((s, g) for s in SEEDS for g in GROUPS)],
            "estimate_provisional": {
                "basis": "synthetic-fixture throughput on the shared RTX 3090 (qualification Q4); decode+augment of "
                         "64 PNGs with 8 threads and an fp16-AMP train step, not overlapped; VAL 49 batches per epoch",
                "seconds_per_step": round(step, 4), "seconds_per_run": round(per_run), "hours_24_runs": round(24 * per_run / 3600, 1),
                "contention_note": "unrelated GPU jobs were running; real throughput may differ (measure in the first "
                                   "scientific run's audit, never extrapolate from generator jobs)"},
            "storage_per_run_mib": {"selected_checkpoint": 45, "val_predictions_40_epochs": 8, "steps_log": 3,
                                    "records": 1, "all_epoch_checkpoints_if_retained": 1800},
            "gpu_memory_mib": {k: v["peak_allocated_mib"] for k, v in q4["gpu_train_step_seconds"].items()},
            "gpu_free_floor_mib": 4096, "one_detector_process_at_a_time": True,
            "unrelated_gpu_jobs": "recorded, never touched"}
    doc = {"schema": "m10_downstream_scientific_readiness_v1", "date": "2026-11-26", "status": status, "gates": gates,
           "engineering_gates_failed": engineering,
           "blocking_owner_decisions": [d["id"] for d in OWNER_DECISIONS if d["blocking"]],
           "owner_decisions": OWNER_DECISIONS,
           "matrix": {"groups": list(GROUPS), "seeds": list(SEEDS), "runs": 24, "bank_mapping": "detector seed s = bank seed s (D12)",
                      "e00": "real-only (32 live + 32 real spoof)"},
           "evidence": {p.name: sha(str(p.relative_to(ROOT))) for p in sorted((ROOT / EV).glob("*")) if p.is_file()},
           "protocol_audit": {"path": AUDIT, "sha256": sha(AUDIT)},
           "code": {p: sha(p) for p in CODE},
           "environment_lock": {"path": ENV_LOCK, "sha256": sha(ENV_LOCK)},
           "qualification_summary": {"Q1": q["Q1_augmentation_equivalence"], "Q2": {k: v for k, v in
                                     q["Q2_cpu_equivalence_with_smoke_runner"].items()}, "Q3": q["Q3_cuda_policies_replay"],
                                     "Q4": q4, "Q6": q["Q6_threshold_tie_finding"], "label": q["label"]},
           "bank_verification_summary": {k: {"manifest_sha256": v["binding"]["manifest_sha256"], "outputs": v["outputs"],
                                             "audit": v["binding"]["audit_gates"], "random_loaded": v["random_loaded"],
                                             "pixel_range": v["random_pixel_range"]} for k, v in bv["banks"].items()},
           "resources": res, "plan": plan,
           "reproducibility": {"manifest_schema": "gpatbench/downstream/science_manifest.py (%s)" % SM.SCHEMA,
                               "template": TEMPLATE, "required_fields": sorted(SM.REQUIRED)},
           "backup": "no detector output exists yet; banks and future detectors live only on the GPU host; git holds "
                     "manifests, hashes and audits, which are not a backup",
           "not_done": ["no scientific detector training", "no TEST access or evaluation", "no DINOv3", "no GPAT",
                        "no generator or bank change", "no launch authority activated"]}
    return doc


def template():
    from gpatbench.downstream import science_manifest as SM
    desc = {k: t.__name__ if isinstance(t, type) else " | ".join(x.__name__ for x in t) for k, t in SM.REQUIRED.items()}
    return {"schema": SM.SCHEMA, "field_types": desc, "environment_keys": list(SM.ENV_KEYS), "gpu_keys": list(SM.GPU_KEYS),
            "selection_keys": list(SM.SELECTION_KEYS),
            "rules": ["E00: bank_seed null, synthetic_bank null", "other groups: bank_seed = detector_seed, synthetic_bank "
                      "= {method_id, seed, manifest_sha256, path, audit_record}", "SCIENTIFIC: epochs 40, warmup 3",
                      "one VAL prediction parquet (sha256) per epoch", "test_accessed false; selection_source VAL only",
                      "backup states whether an independent copy exists"],
            "validator": "gpatbench.downstream.science_manifest.validate"}


def draft(doc):
    from gpatbench.downstream import science_banks as SB
    fam = {}
    for m in SB.METHODS:
        b = SB.bank_binding(ROOT, m, 42)
        fam[m] = {"closure": b["closure"], "closure_sha256": b["closure_sha256"],
                  "banks": {str(s): SB.bank_binding(ROOT, m, s)["manifest_sha256"] for s in SEEDS}}
    return {"schema_version": "m10_resnet18_science_launch_authority_v1", "record": "M10_RESNET18_SCIENCE_LAUNCH_AUTHORITY",
            "status": "DRAFT_PENDING_OWNER_DECISIONS", "approved": False, "date_prepared": "2026-11-26",
            "activation": "not activatable until O1-O4 are decided, the chosen rules are implemented and re-qualified, the "
                          "M10 launcher and independent per-run auditor are qualified, and the owner commits "
                          "configs/amendments/m10_resnet18_science_launch_authority.yaml",
            "evaluator_contract": {"path": "configs/amendments/m8_downstream_evaluator_contract.yaml",
                                   "sha256": sha("configs/amendments/m8_downstream_evaluator_contract.yaml")},
            "frozen_config": {"path": "configs/frozen/downstream_resnet18.yaml", "sha256": sha("configs/frozen/downstream_resnet18.yaml")},
            "spec": {"path": "docs/spec/GPAT_TransferBench_v1_0_Frozen_Specification_2026.docx",
                     "sha256": sha("docs/spec/GPAT_TransferBench_v1_0_Frozen_Specification_2026.docx")},
            "matrix": {"groups": list(GROUPS), "seeds": list(SEEDS), "runs": 24, "e00": "real-only, no synthetic image",
                       "bank_mapping": "D12 SEED_MATCHED_RAW: detector seed s uses the bank of generator seed s; RAW-MATCHED "
                                       "(all 8838 valid outputs)", "view": "RAW-MATCHED (QUALITY-MATCHED: O6)"},
            "bank_families": fam,
            "detector_implementation": {"code_sha256": doc["code"], "source_commit": "<commit of the qualified launcher>"},
            "environment": {"name": "gpat-m8-downstream", "lock": doc["environment_lock"], "decision": "O4"},
            "data": {"split_v1": "fb9aeb369a124fc96ba855ef2ce269236c4a743fe960e73ab739412c9cb5092d",
                     "m2_accounting": "65bb83e69ea7cebcb37373971e5e8f5248609b7963fd9d32aa1c0ba4377c0662",
                     "pairs_train_v1": "a5e4fdaef236f15730c7e3885b537e08e995faffbe654167fc940f44bbc75243",
                     "populations": load(AUDIT)["populations"]["counts"]},
            "gpu_and_resources": {"gpu": "NVIDIA GeForce RTX 3090 GPU-b722cd9d-c9fa-223c-915e-cc09d9b862b7",
                                  "precision": "O1", "free_floor_mib": 4096, "one_detector_process_at_a_time": True,
                                  "unrelated_jobs": "recorded, never touched", "order": "O5", "run_root": "O3"},
            "selection": {"checkpoint": "D5 min VAL video ACER, max VAL video AUC, earliest epoch",
                          "threshold": "spec 15 + D4 (tie completion: O2)", "frozen_before_test": True},
            "metrics": "gpatbench/downstream/metrics.py (sha256 %s)" % doc["code"]["gpatbench/downstream/metrics.py"],
            "reproduction_and_audit": ["run manifest validated by gpatbench.downstream.science_manifest.validate",
                                       "independent per-run audit committed and pushed before the next run",
                                       "selected checkpoint sha256 and VAL threshold frozen and committed before any TEST"],
            "fail_closed": ["any gate failure stops the matrix; partial outputs and logs are preserved",
                            "no rerun into an existing run root", "non-finite loss or VAL logit stops the run",
                            "no TEST access in any training run", "bank or face hash mismatch stops the run"],
            "owner_decisions_required": [{k: d[k] for k in ("id", "blocking", "options", "recommended")} for d in OWNER_DECISIONS],
            "authorizes": {"scientific_downstream_training": False, "test_evaluation": False, "dinov3": False, "gpat": False}}


def report(d):
    g = "\n".join("- %s: %s" % (k, "pass" if v else "**NOT MET**") for k, v in d["gates"].items())
    od = "\n".join("- **%s**%s: %s Recommended `%s`." % (x["id"], " (blocking)" if x["blocking"] else "", x["question"],
                                                        x["recommended"]) for x in d["owner_decisions"])
    e = d["plan"]["estimate_provisional"]
    return "\n".join(["# M10 downstream ResNet-18 scientific readiness", "", "**Status:** `%s`." % d["status"], "",
                      "## Gates", "", g, "", "## Owner decisions", "", od, "", "## Plan (provisional)", "",
                      "24 runs, one at a time, seed-major (O5). Estimated %.3f s/step, ~%d s/run, ~%.1f h for 24 runs "
                      "(%s)." % (e["seconds_per_step"], e["seconds_per_run"], e["hours_24_runs"], e["contention_note"]),
                      "Storage ~57 MB/run with selected-only retention (~1.4 GB total).", "",
                      "## Not done", "", "\n".join("- " + x for x in d["not_done"]), ""])


def main():
    d = build()
    (ROOT / (OUT + ".json")).write_text(json.dumps(d, indent=1, sort_keys=True) + "\n")
    (ROOT / (OUT + ".md")).write_text(report(d))
    (ROOT / TEMPLATE).write_text(json.dumps(template(), indent=1, sort_keys=True) + "\n")
    (ROOT / DRAFT).write_text(json.dumps(draft(d), indent=1, sort_keys=True) + "\n")
    (ROOT / (OUT + "_SHA256.txt")).write_text("".join("%s  %s\n" % (sha(p), p) for p in (OUT + ".json", OUT + ".md",
                                                                                       TEMPLATE, DRAFT)))
    print(d["status"], [k for k, v in d["gates"].items() if not v])


if __name__ == "__main__":
    main()
