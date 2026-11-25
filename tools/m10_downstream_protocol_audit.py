"""M10 downstream ResNet-18 protocol audit: every frozen rule with its source, the 24-run matrix, the real populations,
the implementation gap analysis and the unresolved owner items. Laptop, read-only inputs (TRAIN/VAL rows only).

    python3 tools/m10_downstream_protocol_audit.py   -> outputs/audit/M10_DOWNSTREAM_PROTOCOL_AUDIT.{json,md}
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = "outputs/audit/M10_DOWNSTREAM_PROTOCOL_AUDIT"
CONTRACT = "configs/amendments/m8_downstream_evaluator_contract.yaml"
CONTRACT_SHA256 = "67e4b62ddc6b81b8b9a8657e71b3979f6e6a186827183b052ae3d29b587fcbe0"
FROZEN = "configs/frozen/downstream_resnet18.yaml"
FROZEN_SHA256 = "f117c57380404bccfdc42370290c4bd2141b69346d3ee3f044d020f8b90a447b"
SPEC_TXT = "docs/spec/spec_extracted_text.txt"
SPEC_DOCX = "docs/spec/GPAT_TransferBench_v1_0_Frozen_Specification_2026.docx"
SMOKE_CONTRACT = "configs/amendments/m8_smoke_test_contract.yaml"
GROUPS = ("E00", "E01", "E02", "E03", "E04", "E05", "E06b", "E06c")
SEEDS = (42, 1337, 2026)


def sha(rel):
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def spec(fragment):
    lines = (ROOT / SPEC_TXT).read_text().splitlines()
    hits = [i + 1 for i, x in enumerate(lines) if fragment in x]
    if len(hits) < 1:
        raise SystemExit("spec fragment not found: %r" % fragment)
    return "%s:%d" % (SPEC_TXT, hits[0])


def build():
    import yaml
    from gpatbench.downstream import science_inputs as SI, science_banks as SB, science_runner as SR
    c = json.loads((ROOT / CONTRACT).read_text())
    f = yaml.safe_load((ROOT / FROZEN).read_text())
    if sha(CONTRACT) != CONTRACT_SHA256 or sha(FROZEN) != FROZEN_SHA256:
        raise SystemExit("frozen evaluator authority changed")
    pe = c["pinned_execution_details"]
    tr = f["training"]
    protocol = {
        "architecture": {"value": "torchvision ResNet-18 IMAGENET1K_V1, fc -> Linear(512, 1), one logit",
                         "source": [FROZEN + " model", spec("| Head | replace FC with Linear(512,1) |")]},
        "initialization": {"value": "ImageNet weights resnet18-f37072fd.pth (sha256 %s, hash-verified); "
                                    "torch.manual_seed(seed) immediately before building the model (new fc head)"
                                    % c["based_on"]["resnet18_weights_sha256"],
                           "source": [CONTRACT + " rng_scheme.model", spec("| Backbone | torchvision ResNet-18 IMAGENET1K_V1 |")]},
        "input_and_normalization": {"value": "train: D14 RandomResizedCrop(224, (0.92,1.0), (0.95,1.05), BICUBIC) -> "
                                             "HFlip 0.5 -> ColorJitter(0.05,0.05,0.05,0.01) p 0.5 -> to_tensor -> ImageNet "
                                             "Normalize; eval (VAL/TEST): D2 full bicubic resize 256 -> 224 -> Normalize",
                                    "source": [CONTRACT + " D2, D14", FROZEN + " augmentation", spec("14.1 FAS-safe augmentation")]},
        "loss_and_target": {"value": pe["D3"], "source": [CONTRACT + " D3", spec("| Loss | BCEWithLogitsLoss |")]},
        "optimizer": {"value": "AdamW, betas (0.9, 0.999), eps 1e-8, weight decay %s on all parameters; backbone lr %s, "
                               "head (fc) lr %s; clip_grad_norm_ 1.0 before each step"
                               % (tr["weight_decay"], tr["lr_backbone"], tr["lr_head"]),
                      "source": [FROZEN + " training", CONTRACT + " D6"]},
        "schedule": {"value": "%d epochs; warmup %d epochs linear lr = base*k/W from 0; then cosine to %s per step (D6)"
                              % (tr["epochs"], tr["warmup_epochs"], tr["min_lr"]),
                     "source": [FROZEN + " training.epochs/warmup_epochs/scheduler/min_lr", CONTRACT + " D6",
                                spec("| Epochs | 40 |")]},
        "batch_and_composition": {"value": "batch 64: mixed = 32 live + 16 real spoof + 16 synthetic spoof; E00 = 32 live + "
                                           "32 real spoof; tensor order live, real spoof, synthetic",
                                  "source": [FROZEN + " training.mixed_batch/real_only_batch", CONTRACT + " D1",
                                             spec("| Mixed batch | 64 = 32 live + 16 real spoof + 16 synthetic spoof |")]},
        "epoch_and_sampler": {"value": pe["D1"], "rng": c["pinned_execution_details"]["rng_scheme"],
                              "source": [CONTRACT + " D1, rng_scheme"]},
        "synthetic_budget_and_view": {"value": "N_syn = N_TRAIN spoof frames = 8838 per bank; RAW-MATCHED: every valid "
                                               "output (all 21 banks: 8838 valid, 0 failures); detector seed s uses the bank "
                                               "of generator seed s (D12 SEED_MATCHED_RAW); no averaging or mixing of banks",
                                      "source": [CONTRACT + " D12", spec("Main intended budget: N_syn = N_train_spoof_frames."),
                                                 spec("RAW-MATCHED view: retain every numerically valid generated output")]},
        "quality_matched": {"value": "QUALITY-MATCHED: N_common = minimum valid/accepted outputs among compared methods, "
                                     "deterministic hash subsample; D12: for E01/E02 equal to RAW-MATCHED unless N_common "
                                     "forces subsampling. All seven previous-work banks have 8838 valid outputs and no "
                                     "native quality filter is part of their frozen generation contracts; N_common across the "
                                     "full comparison depends on the later GPAT banks. Both views are reported; the primary "
                                     "view must be declared before TEST (owner).",
                            "source": [spec("QUALITY-MATCHED view: each method may apply"),
                                       spec("Report both RAW-MATCHED and QUALITY-MATCHED downstream results"),
                                       CONTRACT + " D12"]},
        "device_precision_determinism": {"value": pe["D13"], "frozen": "amp: %s" % tr["amp"],
                                         "unresolved": "science AMP dtype and GradScaler policy (and CUDA determinism "
                                                       "settings) are not pinned",
                                         "source": [CONTRACT + " D13 (confirmed_items: deferred to before M10)",
                                                    spec("| AMP | enabled on CUDA |")]},
        "dataloader_workers": {"value": "smoke: num_workers 0 (D13); science: not pinned. The science runner draws every "
                                        "D14 augmentation parameter in sample order from the dedicated generator before "
                                        "decoding, so a decode/augment thread pool changes no random draw (qualified "
                                        "bitwise against the smoke runner)",
                               "source": [CONTRACT + " D13, D14"]},
        "validation_and_checkpoint_selection": {"value": pe["D5"], "source": [CONTRACT + " D5", FROZEN + " selection.checkpoint",
                                                                              spec("| Checkpoint rule | minimum VAL video-level ACER")]},
        "threshold_selection": {"value": "VAL video scores; minimize ACER over all unique score midpoints plus 0 and 1; "
                                         "tie-break threshold closest to 0.5; D4 spoof iff score >= threshold; frozen per "
                                         "seed/method and applied once to TEST",
                                "unresolved": "no rule when two minimum-ACER thresholds are equally close to 0.5 (e.g. "
                                              "constant or fully inverted VAL scores: thresholds 0 and 1 at ACER 1/2); the "
                                              "implementation fails closed (UnresolvedThresholdTie)",
                                "source": [spec("compute threshold on VAL video scores by minimizing ACER"), CONTRACT + " D4",
                                           FROZEN + " selection.threshold"]},
        "checkpoint_saving": {"value": "not pinned: D5 needs every epoch evaluated; only the selected checkpoint is "
                                       "required to be kept (frozen before TEST)",
                              "source": [spec("TEST untouched before checkpoint/threshold freeze")]},
        "video_aggregation": {"value": "video score = arithmetic mean of that canonical video's frame probabilities; "
                                       "video-level metrics primary, frame-level diagnostic at the video threshold (D9)",
                              "source": [spec("Video probability = arithmetic mean"), CONTRACT + " D9"]},
        "metrics": {"value": "APCER, BPCER, ACER (exact counts); AUC (Mann-Whitney, spoof positive); EER (D7 linear "
                             "interpolation); accuracy, balanced accuracy, spoof F1, TPR@FPR 1%/5% (D8, no interpolation); "
                             "dataset-wise ACER/AUC, attack_macro APCER, SiW-Mv2 raw 14-type APCER (D10)",
                    "source": [spec("APCER = # spoof videos predicted live / # spoof videos"), CONTRACT + " D7, D8, D10",
                               "gpatbench/downstream/metrics.py"]},
        "seed_aggregation": {"value": "mean +/- sample standard deviation over seeds 42/1337/2026",
                             "source": [spec("All self-run results are reported over seeds")]},
        "test_policy": {"value": "TEST sealed: never used for training, selection, thresholds, early stopping, quality "
                                 "filtering or inspection; evaluated once with the frozen VAL threshold in a separately "
                                 "authorized stage",
                        "source": [spec("Never train on TEST metadata"), "gpatbench/downstream/firewall.py"]},
    }
    status = {r["experiment_id"]: r for r in csv.DictReader(open(ROOT / "outputs/audit/method_status.csv"))}
    matrix = []
    for g in GROUPS:
        for s in SEEDS:
            row = {"group": g, "detector_seed": s, "track_id": SR.TRACKS[g],
                   "synthetic": None if g == "E00" else "bank %s seed %d (D12 seed-matched, RAW-MATCHED 8838)" % (g, s),
                   "batch": "32 live + 32 real spoof" if g == "E00" else "32 live + 16 real spoof + 16 synthetic"}
            if g != "E00":
                b = SB.bank_binding(ROOT, g, s)
                row.update(bank_manifest_sha256=b["manifest_sha256"], bank_path=b["path"], bank_audit=b["audit_record"],
                           fidelity=status[g]["fidelity_class"])
            else:
                row["fidelity"] = "REAL_ONLY_LOWER_BOUND"
            matrix.append(row)
    pop = SI.build_populations(ROOT)
    n = pop["counts"]
    spe = n["train_live"] // 32
    arith = {"steps_per_epoch": spe, "total_steps": 40 * spe, "warmup_steps": 3 * spe,
             "per_epoch_images": {"live": 32 * spe, "real_spoof_mixed": 16 * spe, "real_spoof_E00": 32 * spe,
                                  "synthetic": 16 * spe},
             "passes_over_pool_in_40_epochs": {"real_spoof_mixed": round(40 * 16 * spe / n["train_real_spoof"], 3),
                                               "real_spoof_E00": round(40 * 32 * spe / n["train_real_spoof"], 3),
                                               "synthetic": round(40 * 16 * spe / 8838, 3)},
             "live_frames_dropped_per_epoch": n["train_live"] - 32 * spe}
    gaps = [
        {"item": "smoke runner is CPU-only, no AMP, deterministic CPU settings", "resolution":
         "gpatbench/downstream/science_runner.py: named PrecisionPolicy (CUDA AMP fp16+GradScaler / bf16 candidates); "
         "SCIENTIFIC mode refuses without the owner precision pin", "status": "IMPLEMENTED_PENDING_OWNER_PIN"},
        {"item": "smoke runner scores a VAL holdout stand-in for TEST at the end of training", "resolution":
         "science runner has no TEST / holdout input at all; it ends at the frozen VAL selection", "status": "IMPLEMENTED"},
        {"item": "smoke runner keeps epoch states in memory and writes no checkpoint", "resolution":
         "selected checkpoint written, reloaded (weights_only) and re-hashed; ALL_EPOCHS retention optional",
         "status": "IMPLEMENTED_RETENTION_PENDING_OWNER"},
        {"item": "bank_reader reads E01/E02 banks only", "resolution":
         "gpatbench/downstream/science_banks.py resolves all seven methods through their audited bindings and family "
         "readers (bank_writer / learned_bank registry) with full verification", "status": "IMPLEMENTED"},
        {"item": "synthetic sample id is pair_id only (DSDG banks have none)", "resolution":
         "sample_id = pair_id or DSDG synthetic_id", "status": "IMPLEMENTED"},
        {"item": "prediction rows carry track_id 'binary' for every method", "resolution":
         "science runner sets the A1 track from method_status.csv (E06b B_NATIVE_FULL_SECONDARY)",
         "status": "IMPLEMENTED_CONFIRMATION_REQUESTED"},
        {"item": "single-threaded decode/augmentation (D13 smoke workers 0)", "resolution":
         "parameters pre-drawn in sample order; optional thread pool; bitwise equal to the smoke runner",
         "status": "IMPLEMENTED"},
        {"item": "no full-population input builder", "resolution": "gpatbench/downstream/science_inputs.py (TRAIN/VAL "
         "only, M2 face sha256, M3 grouping disjointness)", "status": "IMPLEMENTED"},
        {"item": "no scientific run manifest", "resolution": "gpatbench/downstream/science_manifest.py",
         "status": "IMPLEMENTED"},
        {"item": "no M10 launcher / orchestration and no scientific launch authority", "resolution":
         "launch-authority DRAFT only (not activated); launcher to be qualified after the owner decisions",
         "status": "NOT_IMPLEMENTED_BY_DESIGN"},
    ]
    doc = {
        "schema": "m10_downstream_protocol_audit_v1", "date": "2026-11-26",
        "authorities": {CONTRACT: sha(CONTRACT), FROZEN: sha(FROZEN), SMOKE_CONTRACT: sha(SMOKE_CONTRACT),
                        SPEC_DOCX: sha(SPEC_DOCX), SPEC_TXT: sha(SPEC_TXT),
                        "resnet18_weights_sha256": c["based_on"]["resnet18_weights_sha256"],
                        "contract_status": c["status"], "contract_authorizes": c["authorizes"]},
        "frozen_protocol": protocol,
        "matrix": {"groups": list(GROUPS), "seeds": list(SEEDS), "runs": len(matrix), "rows": matrix,
                   "excluded": {"E06a": "superseded by A1 (E06c DSDG-BIN-IDFREE is the Track-A DSDG row)",
                                "E07a/E07b/E07c": "DiffFAS family: superseded (E07a) / owner-excluded by A9"},
                   "dinov3": "M11, not in this matrix (D11; frozen config missing)"},
        "populations": {"counts": n, "pool_sha256": pop["pool_sha256"], "split_manifest_sha256": pop["split_manifest_sha256"],
                        "grouping": SI.GROUP_KEY, "siwmv2_subject_note": "SiW-Mv2 rows have no subject identity "
                        "(owner Q-14); SiW-Mv2 is content-group / video-disjoint, never claimed subject-disjoint",
                        "arithmetic": arith},
        "implementation_gaps": gaps,
    }
    return doc


def report(d):
    p = d["frozen_protocol"]
    a = d["populations"]["arithmetic"]
    return "\n".join(["# M10 downstream ResNet-18 protocol audit", "",
                      "Evaluator contract `%s` (`%s`), frozen config `%s` (`%s`). Contract scope: smoke only; "
                      "scientific downstream training is not authorized by it." % (CONTRACT, d["authorities"][CONTRACT],
                                                                               FROZEN, d["authorities"][FROZEN]), "",
                      "## Frozen rules", ""] + ["- **%s**: %s (%s)" % (k, v["value"] if isinstance(v["value"], str)
                                                                       else json.dumps(v["value"]), "; ".join(v["source"]))
                                                 + (" — UNRESOLVED: %s" % v["unresolved"] if "unresolved" in v else "")
                                                 for k, v in p.items()] + [
        "", "## Matrix", "", "%d runs: %s x seeds %s; E00 real-only; detector seed s uses bank seed s." % (
            d["matrix"]["runs"], ", ".join(d["matrix"]["groups"]), d["matrix"]["seeds"]), "",
        "## Populations", "", "TRAIN live %(train_live)d, TRAIN real spoof %(train_real_spoof)d, VAL %(val)d frames / "
        "%(val_videos)d videos." % d["populations"]["counts"],
        "Steps per epoch %d, total %d, warmup %d; live frames dropped per epoch %d; passes over the pools in 40 epochs: %s."
        % (a["steps_per_epoch"], a["total_steps"], a["warmup_steps"], a["live_frames_dropped_per_epoch"],
           json.dumps(a["passes_over_pool_in_40_epochs"])), "",
        "## Implementation gaps", ""] + ["- %s → %s (`%s`)" % (g["item"], g["resolution"], g["status"])
                                         for g in d["implementation_gaps"]] + [""])


def main():
    d = build()
    (ROOT / (OUT + ".json")).write_text(json.dumps(d, indent=1, sort_keys=True) + "\n")
    (ROOT / (OUT + ".md")).write_text(report(d))
    print(OUT, hashlib.sha256((ROOT / (OUT + ".json")).read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
