"""M10 ResNet-18 scientific TRAIN/VAL launch authority and its qualification record.

    python3 tools/m10_launch_authority.py --source-commit <qualified clean-clone commit> [--activate]

Reads the committed evidence under outputs/audit/M10_SCIENTIFIC_LAUNCH_QUALIFICATION/ (host GPU qualification of the
launcher + independent auditor on synthetic fixtures, host and laptop tests, storage verification, the 24-run
scientific dry preflight) and writes
    outputs/audit/M10_SCIENTIFIC_LAUNCH_QUALIFICATION.{json,md} (+ _SHA256.txt)
and, only with --activate AND every mandatory gate passing,
    configs/amendments/m10_resnet18_science_launch_authority.yaml  (OWNER_APPROVED_ACTIVE; TRAIN/VAL only)
If any gate fails the authority is not written (it stays inactive). Nothing here launches a run.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gpatbench.downstream import m10_launcher as L  # noqa: E402
from gpatbench.downstream import m10_storage as ST  # noqa: E402

DRAFT = "configs/amendments/m10_resnet18_science_launch_authority.DRAFT.yaml"
EV = "outputs/audit/M10_SCIENTIFIC_LAUNCH_QUALIFICATION"
OUT = "outputs/audit/M10_SCIENTIFIC_LAUNCH_QUALIFICATION"
GPU = {"name": "NVIDIA GeForce RTX 3090", "uuid": "GPU-b722cd9d-c9fa-223c-915e-cc09d9b862b7", "driver": "580.178.04",
       "free_floor_mib": 4096}
POPULATIONS = {"train_live": 5629, "train_real_spoof": 8838, "val": 3121, "val_videos": 391}


def sha(rel, root=ROOT):
    return hashlib.sha256((Path(root) / rel).read_bytes()).hexdigest()


def load(rel, root=ROOT):
    return json.loads((Path(root) / rel).read_text())


def launch_command(group, seed):
    return ("cd <clone at or after the source commit> && env -i HOME=$HOME PATH=/usr/bin:/bin LANG=C.UTF-8 "
            "CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 "
            "CUDA_VISIBLE_DEVICES=0 ~/workdir/envs/gpat-m8-downstream/bin/python -B tools/run_m10_detector.py "
            f"--group {group} --seed {seed} --runtime-root ~/workdir/GPAT_TransferBench_runtime; "
            f"echo $? > ~/workdir/m10_{group}_s{seed}.exit")


def audit_command(group, seed):
    return ("env -i HOME=$HOME PATH=/usr/bin:/bin LANG=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 "
            "CUDA_VISIBLE_DEVICES= ~/workdir/envs/gpat-m8-downstream/bin/python -B tools/m10_run_auditor.py "
            f"--repo . --run-root {ST.M10_ROOT}/{group}/seed_{seed} --group {group} --seed {seed} "
            f"--runtime-root ~/workdir/GPAT_TransferBench_runtime --exit-marker ~/workdir/m10_{group}_s{seed}.exit "
            f"--out <scratch>/M10_{group}_SEED{seed}_AUDIT.json   (then commit it as "
            f"{L.AUDIT_RECORD.format(group=group, seed=seed)})")


def build_authority(repo=ROOT, *, source_commit, qualification=None, in_memory_candidate=False) -> dict:
    repo = Path(repo)
    d = load(DRAFT, repo)
    fam = d["bank_families"]
    for m, f in fam.items():
        if sha(f["closure"], repo) != f["closure_sha256"]:
            raise SystemExit(f"{m} family closure changed")
    storage = ST.load_storage_authority(repo)
    a = {
        "schema_version": "m10_resnet18_science_launch_authority_v1",
        "record": "M10_RESNET18_SCIENCE_LAUNCH_AUTHORITY",
        "status": "OWNER_APPROVED_ACTIVE",
        "approved": True,
        "in_memory_qualification_candidate": bool(in_memory_candidate),
        "date_activated": "2026-11-26",
        "approval": "owner decisions O1-O7 of 2026-11-26 (configs/amendments/m10_resnet18_owner_decisions.yaml) "
                    "authorize activating this TRAIN/VAL launch authority once every mandatory qualification gate "
                    "passes; activation does not start any run",
        "authorizes": {"scientific_train_val_execution": True, "one_run_per_invocation": True,
                       "automatic_matrix_execution": False, "test_evaluation": False, "dinov3": False, "gpat": False,
                       "generator_or_bank_changes": False},
        "based_on_draft": {"path": DRAFT, "sha256": sha(DRAFT, repo)},
        "bindings": {
            "owner_decisions": {"path": L.OWNER_PATH, "sha256": sha(L.OWNER_PATH, repo)},
            "precision_pin": {"path": L.PIN_PATH, "sha256": sha(L.PIN_PATH, repo)},
            "evaluator_contract": {"path": L.CONTRACT[0], "sha256": L.CONTRACT[1]},
            "frozen_config": {"path": L.CONFIG[0], "sha256": L.CONFIG[1]},
            "spec": {"path": L.SPEC[0], "sha256": L.SPEC[1]},
            "environment": {"name": L.ENV_EXPECTED["name"], "lock_path": L.LOCK[0], "lock_sha256": L.LOCK[1],
                            "expected": L.ENV_EXPECTED},
            "imagenet_weights": {"file": "resnet18-f37072fd.pth",
                                 "sha256": "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"},
            "source": {"commit": source_commit, "code_sha256": L.code_sha256(repo),
                       "rule": "the launcher verifies a clean worktree whose HEAD descends from this commit and whose "
                               "code files hash exactly to these values"},
            "storage_authority": {"path": ST.STORAGE_AUTHORITY, "sha256": storage["_sha256"], "root": storage["root"],
                                  "filesystem_uuid": storage["filesystem_uuid"]},
            "gpu": GPU,
            "runtime_root": "/home/student20261/workdir/GPAT_TransferBench_runtime",
            "auditor": {"path": "tools/m10_run_auditor.py", "sha256": sha("tools/m10_run_auditor.py", repo)},
        },
        "data": {"split_v1": d["data"]["split_v1"], "m2_accounting": d["data"]["m2_accounting"],
                 "pairs_train_v1": d["data"]["pairs_train_v1"], "populations_counts": POPULATIONS},
        "bank_families": fam,
        "matrix": {"groups": list(L.GROUPS), "seeds": list(L.SEEDS), "runs": 24,
                   "order": [[g, s] for g, s in L.ORDER], "order_rule": "O5 A_SEED_MAJOR",
                   "bank_mapping": "D12 SEED_MATCHED_RAW: detector seed s uses the bank of generator seed s; all 8838 "
                                   "valid outputs (RAW-MATCHED, O6 primary); E04 clipped and E05 near-collapse "
                                   "outputs included, no quality filter",
                   "e00": "real-only (32 live + 32 real spoof), no synthetic image",
                   "tracks": {"E00": "binary", **{m: t for m, t in load(L.OWNER_PATH, repo)["O7_track_labels"]
                                                  ["mapping"].items() if m != "E00"}}},
        "training_protocol": {"epochs": 40, "warmup_epochs": 3, "steps_per_epoch": 175, "total_steps": 7000,
                              "warmup_steps": 525, "batch": {"mixed": {"live": 32, "real_spoof": 16, "synthetic": 16},
                                                             "E00": {"live": 32, "real_spoof": 32}},
                              "optimizer": "AdamW betas (0.9, 0.999) eps 1e-8, weight decay 1e-4 on all parameters; "
                                           "backbone lr 1e-4, head (fc) lr 5e-4",
                              "schedule": "D6: lr = base*k/W for k < W, then cosine to 1e-6 per step",
                              "grad_clip_norm": 1.0, "augmentation": "D14", "eval_preprocessing": "D2",
                              "workers": L.WORKERS, "threads": L.THREADS,
                              "workers_note": "decode/augment threads only; parameters drawn in sample order "
                                              "(bitwise-equal outputs qualified)"},
        "precision": {"decision": "O1 A_CUDA_AMP_FP16_GRADSCALER_DETERMINISTIC", "policy": "CUDA_AMP_FP16_GRADSCALER_DETERMINISTIC",
                      "grad_scaler": {"init_scale": 65536.0, "growth_factor": 2.0, "backoff_factor": 0.5,
                                      "growth_interval": 2000},
                      "step_order": "scale(loss).backward -> unscale_ -> clip_grad_norm_(1.0) -> step -> update",
                      "determinism": "deterministic algorithms, cudnn deterministic, benchmark off, TF32 off, "
                                     "CUBLAS_WORKSPACE_CONFIG=:4096:8",
                      "skipped_steps": "recorded per iteration; fail closed on a non-finite gradient at scale <= 1 or "
                                       "an epoch without an applied update (disclosed implementation of the owner rule)"},
        "selection": {"threshold": "spec 15 + O2 A_SMALLER_THRESHOLD (gpatbench/downstream/m10_selection.py)",
                      "checkpoint": "D5: min VAL video ACER, max VAL video AUC, earliest epoch over all 40 epochs",
                      "decision": "spoof iff score >= threshold", "test_used": False},
        "retention": {"decision": "O3 A_SELECTED_ONLY_HDD_NEW_ROOT", "root": ST.M10_ROOT,
                      "layout": "<m10_root>/<group>/seed_<seed>/",
                      "rule": "best-so-far persisted atomically and verified before the previous one is removed; "
                              "exactly the selected checkpoint remains"},
        "per_run_audit": {"tool": "tools/m10_run_auditor.py", "record": L.AUDIT_RECORD,
                          "rule": "the independent audit of a run must PASS and be committed and pushed before the "
                                  "next run of the O5 order may launch (enforced by the launcher)"},
        "test_firewall": "no TEST input exists in the launcher; an audit hook refuses any open of a face file outside "
                         "the TRAIN/VAL allowlist and any listing under the faces root; split_v1 is hashed in full and "
                         "its TEST rows are removed by the parquet row filter",
        "fail_closed": ["any preflight gate failure: nothing is created", "existing run root: refused (create-only)",
                        "concurrent detector process or held lock: refused",
                        "predecessor run not audited PASS and committed: refused",
                        "non-finite loss / VAL logit / genuine non-finite gradient: run stops, evidence kept",
                        "bank, face, weight or code hash mismatch: refused or stopped",
                        "insufficient GPU memory: resource blocker reported; batch size and precision never adjusted"],
        "launch_commands": {f"{g}|{s}": launch_command(g, s) for g, s in L.ORDER},
        "audit_commands": {f"{g}|{s}": audit_command(g, s) for g, s in L.ORDER},
    }
    if qualification is not None:
        a["activation_verification"] = qualification
    return a


# ------------------------------------------------------------------------------------------------ qualification
def gates(source_commit) -> dict:
    q = load(f"{EV}/host_qualification.json")
    ht = load(f"{EV}/host_tests.json")
    lt = load(f"{EV}/laptop_tests.json")
    sv = load(f"{EV}/storage_verification.json")
    owner = load(L.OWNER_PATH)
    g = {}
    g["Q01_owner_decisions_recorded"] = owner["status"] == "OWNER_APPROVED" and owner["approved"] is True and \
        subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", "25690cf5799d2dbf5809184ceca15e7dc20faeda",
                        source_commit]).returncode == 0
    g["Q02_frozen_files_unchanged"] = all(sha(p) == h for p, h in (L.CONTRACT, L.CONFIG, L.SPEC, L.LOCK))
    g["Q03_precision_pin_O1"] = load(L.PIN_PATH)["policy"] == "CUDA_AMP_FP16_GRADSCALER_DETERMINISTIC"
    g["Q04_storage_root_verified"] = sv["ok"] is True and sv["preflight"]["ok"] is True and \
        ST.load_storage_authority(ROOT)["root"] == ST.M10_ROOT and q["storage_preflight"]["ok"] is True
    g["Q05_environment_and_gpu_verified"] = q["identity"]["environment_gate"] is True and \
        q["identity"]["gpu"]["uuid"] == GPU["uuid"] and q["identity"]["gpu"]["driver"] == GPU["driver"] and \
        q["identity"]["weights_ok"] is True
    g["Q06_qualified_commit_clean_clone"] = q["identity"]["head"] == source_commit and q["identity"]["clean"] is True \
        and q["identity"]["code_sha256"] == L.code_sha256(ROOT)
    g["Q07_fixture_runs_O1"] = q["runs_pass"] is True
    g["Q08_bitwise_replay"] = q["replay"]["bitwise"] is True
    g["Q09_gradscaler_skip_and_fail_closed"] = q["skip_recovery"]["pass"] is True and q["fail_closed"]["pass"] is True
    g["Q10_auditor_pass_on_good_runs"] = q["audits_pass"] is True
    g["Q11_auditor_detects_injected_failures"] = q["negatives"]["pass"] is True and q["negatives"]["cases"] >= 17
    g["Q12_test_firewall"] = q["firewall"]["pass"] is True
    g["Q13_scientific_dry_preflight_24"] = q["dry_preflight"]["pass"] is True and q["dry_preflight"]["runs"] == 24
    g["Q14_host_tests_cuda"] = ht["exit_code"] == 0 and ht["failures"] == 0 and ht["errors"] == 0 and \
        ht["cuda_tests_ran"] is True
    g["Q15_laptop_tests"] = all(v["exit_code"] == 0 for v in lt["targeted"].values())
    g["Q16_full_regression_no_new_failures"] = lt["full_suite"]["new_failures"] == []
    g["Q17_resources"] = q["resources"]["gpu_free_mib"] >= GPU["free_floor_mib"] and \
        q["resources"]["hdd_free_bytes"] > 100 * 2 ** 30
    return g


def build(source_commit):
    g = gates(source_commit)
    q = load(f"{EV}/host_qualification.json")
    status = "M10_DOWNSTREAM_SCIENTIFIC_READY_AUTHORITY_ACTIVE" if all(g.values()) else \
        "M10_DOWNSTREAM_BLOCKED_BY_QUALIFICATION"
    ev = sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / EV).rglob("*") if p.is_file())
    doc = {"schema": "m10_scientific_launch_qualification_v1", "status": status, "source_commit": source_commit,
           "gates": g, "failed": [k for k, v in g.items() if not v],
           "evidence": {p: sha(p) for p in ev},
           "plan": {"order": [[gg, s] for gg, s in L.ORDER], "runs": 24, "one_at_a_time": True,
                    "throughput": q["resources"]["throughput_estimate"], "storage": q["resources"]["storage_estimate"],
                    "note": "throughput is provisional (synthetic fixtures, shared GPU); measure in the first "
                            "scientific run's audit"},
           "not_done": ["no scientific detector training", "no TEST access or evaluation", "no DINOv3", "no GPAT",
                        "no generator or bank change"]}
    return doc


def report(doc) -> str:
    lines = ["# M10 scientific launch qualification", "", f"**Status:** `{doc['status']}`", "",
             f"Qualified source commit (clean GPU-host clone): `{doc['source_commit']}`.", "", "## Gates", ""]
    lines += [f"- {k}: {'pass' if v else '**FAIL**'}" for k, v in doc["gates"].items()]
    lines += ["", "## Plan", "", "24 runs, one at a time, seed-major (O5): " +
              ", ".join(f"{g}/{s}" for g, s in doc["plan"]["order"]) + ".", "",
              f"Provisional throughput: {json.dumps(doc['plan']['throughput'], sort_keys=True)}.", "",
              f"Storage: {json.dumps(doc['plan']['storage'], sort_keys=True)}.", "", "## Not done", ""]
    lines += [f"- {x}" for x in doc["not_done"]]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-commit", required=True)
    ap.add_argument("--activate", action="store_true")
    a = ap.parse_args()
    doc = build(a.source_commit)
    (ROOT / (OUT + ".json")).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    (ROOT / (OUT + ".md")).write_text(report(doc))
    (ROOT / (OUT + "_SHA256.txt")).write_text("".join(f"{sha(OUT + x)}  {OUT + x}\n" for x in (".json", ".md")))
    print(json.dumps({"status": doc["status"], "failed": doc["failed"]}))
    if a.activate:
        if doc["failed"]:
            print("NOT ACTIVATED: failed gates", doc["failed"])
            return 1
        auth = build_authority(ROOT, source_commit=a.source_commit, qualification={
            "ok": True, "record": OUT + ".json", "record_sha256": sha(OUT + ".json"), "gates": doc["gates"]})
        (ROOT / L.AUTHORITY_PATH).write_text(json.dumps(auth, indent=1, sort_keys=True) + "\n")
        print("ACTIVATED", L.AUTHORITY_PATH, sha(L.AUTHORITY_PATH))
    return 0


if __name__ == "__main__":
    sys.exit(main())
