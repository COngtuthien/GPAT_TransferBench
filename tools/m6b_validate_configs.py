"""Static validator for the M6B baseline execution configs.

Checks the invariants frozen in M6B section 17. Parses YAML only: it loads no model, reads no
benchmark data and touches no TEST split.

Usage: python3 tools/m6b_validate_configs.py
Exit code 0 = all checks pass.
"""
import hashlib
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
METHODS = ROOT / "configs" / "methods"

EXPECTED = {
    "E01": ("e01_fas_aug.yaml", "FAITHFUL_OFFICIAL_WITH_DETERMINISM_CLARIFICATION", False),
    "E02": ("e02_freqsub.yaml", "SPEC_DEFINED", False),
    "E03": ("e03_stdn.yaml", "FAITHFUL_OFFICIAL", True),
    "E04": ("e04_physics_std.yaml", "CONTROLLED_ADAPTATION", True),
    "E05": ("e05_pcgan.yaml", "CONTROLLED_ADAPTATION", True),
    "E06c": ("e06c_dsdg_bin_idfree.yaml", "CONTROLLED_ADAPTATION", True),
    "E07c": ("e07c_difffas_bin_idfree.yaml", "CONTROLLED_ADAPTATION", True),
}
SEEDS = [42, 1337, 2026]
CKPT = {
    "E03": ("OFFICIAL_LATEST_FINAL_CKPT_50", None),
    "E04": ("BASELINE_FINAL_STATE_V1", "iteration_150000"),
    "E05": ("BASELINE_FINAL_STATE_V1", "iteration_4000"),
    "E06c": ("OFFICIAL_GENERATOR_EPOCH_200", None),
    "E07c": ("BASELINE_FINAL_STATE_V1", "end_of_400_epoch_budget"),
}

fails = []


def check(cond, msg):
    if cond:
        print(f"  OK   {msg}")
    else:
        print(f"  FAIL {msg}")
        fails.append(msg)


def main():
    cfgs, seen = {}, set()
    print("== parse + identity ==")
    for mid, (fn, fid, learned) in EXPECTED.items():
        p = METHODS / fn
        check(p.is_file(), f"{mid}: config exists ({fn})")
        if not p.is_file():
            continue
        d = yaml.safe_load(p.read_text(encoding="utf-8"))
        check(isinstance(d, dict), f"{mid}: parses to a mapping")
        cfgs[mid] = d
        check(d.get("method_id") == mid, f"{mid}: method_id matches filename")
        check(d["method_id"] not in seen, f"{mid}: method_id unique")
        seen.add(d["method_id"])
        check(d.get("fidelity_class") == fid, f"{mid}: fidelity_class == {fid}")
        check(d.get("status") == "CONFIG_FROZEN", f"{mid}: status == CONFIG_FROZEN")
        check(d.get("milestone_status") == "NOT_TRAINED", f"{mid}: milestone_status == NOT_TRAINED")
        check(d.get("logging_contract") == "run_logging_v1", f"{mid}: logging_contract pinned")

    print("== controlled-adaptation labels ==")
    for mid in ("E04", "E07c"):
        d = cfgs[mid]
        check(d["fidelity_class"] == "CONTROLLED_ADAPTATION", f"{mid}: CONTROLLED_ADAPTATION label present")
        check(bool(d.get("fidelity_provenance")), f"{mid}: fidelity_provenance non-empty")
        check("A3" in " ".join(d["fidelity_provenance"]), f"{mid}: provenance cites Amendment A3")

    print("== seeds ==")
    for mid, d in cfgs.items():
        s = d["seeds"]
        check(s["experiment_seeds"] == SEEDS, f"{mid}: experiment_seeds == {SEEDS}")
        check(s["runs_per_method"] == 3, f"{mid}: runs_per_method == 3")
        for k in ("best_seed", "preferred_seed", "presentation_seed",
                  "max_over_seeds_as_primary", "dropping_a_poor_seed"):
            check(s.get(k) == "FORBIDDEN", f"{mid}: {k} FORBIDDEN")
        check(s["primary_result_aggregation"] == "MEAN_AND_STD_OVER_ALL_THREE_SEEDS",
              f"{mid}: primary aggregation is mean+std over all three seeds")

    print("== E07c auxiliary encoder ==")
    ce = cfgs["E07c"]["conditioning_encoder"]
    check(ce["auxiliary_encoder_training_seed"] == 42, "E07c: auxiliary_encoder_training_seed == 42")
    check(ce["auxiliary_encoder_training_runs"] == 1, "E07c: auxiliary_encoder_training_runs == 1")
    check(ce["trained_exactly_once"] is True, "E07c: trained_exactly_once")
    check(ce["same_frozen_encoder_reused_for_main_seeds"] == SEEDS,
          "E07c: same frozen encoder reused for 42/1337/2026")
    check(ce["three_different_encoders_trained"] == "FORBIDDEN", "E07c: three encoders FORBIDDEN")
    check(ce["architecture_substituted"] is False, "E07c: encoder architecture NOT substituted")
    check(ce["removed"] is False and ce["disabled"] is False, "E07c: encoder not removed/disabled")
    check(ce["substitute_objective"]["K"] == 7, "E07c: substitute K == 7")
    check(ce["substitute_objective"]["split"] == "TRAIN_ONLY", "E07c: encoder data is TRAIN only")
    check(sum(ce["substitute_objective"]["train_counts"].values()) ==
          ce["substitute_objective"]["train_total"] == 14467, "E07c: TRAIN counts sum to 14467")

    print("== E04 fixed auxiliary geometry ==")
    ea = cfgs["E04"]["external_assets"]
    check(ea["fixed_auxiliary_reconstruction_assets"] is True, "E04: assets are fixed auxiliary")
    check(ea["retrained_or_rederived_per_experiment_seed"] is False, "E04: not re-derived per seed")
    check(ea["reused_for_experiment_seeds"] == SEEDS, "E04: assets reused for all three seeds")
    check(ea["geometry_engine"]["pinned_commit"] == "1b6c67601abffc1e9f248b291708aef0e43b55ae",
          "E04: 3DDFA_V2 commit pinned")
    q = cfgs["E04"]["controlled_reconstruction"]["q140_vertex_set"]
    check(q["Q"] == 140 and q["image_independent"] is True, "E04: Q=140 frozen and image-independent")
    check(q["rederived_per_experiment_seed"] is False, "E04: Q=140 not re-derived per seed")
    check(q["vertex_indices_sha256"] ==
          "1b884401377f5aadd3d56857f05fddf70e2c54a52a9eb2160cebc06a74031a1f", "E04: Q=140 hash pinned")

    print("== checkpoint rules ==")
    for mid, (rule, terminal) in CKPT.items():
        c = cfgs[mid]["checkpoint"]
        check(c["rule"] == rule, f"{mid}: checkpoint rule == {rule}")
        if terminal:
            check(c.get("terminal_state") == terminal, f"{mid}: terminal_state == {terminal}")
        check(c["selection_scope"] == "WITHIN_SEED", f"{mid}: checkpoint selection is WITHIN_SEED")
        check(c["selection_uses_test"] is False, f"{mid}: checkpoint selection never uses TEST")
        check(c["selection_uses_val"] is False, f"{mid}: no invented VAL-based selection")
    for mid in ("E03", "E06c"):
        check(cfgs[mid]["checkpoint"]["baseline_final_state_v1_overrides_this"] is False,
              f"{mid}: BASELINE_FINAL_STATE_V1 does NOT override the official rule")
    for mid in ("E01", "E02"):
        c = cfgs[mid]["checkpoint"]
        check(c["rule"] == "NOT_APPLICABLE_NO_TRAINED_GENERATOR", f"{mid}: no checkpoint rule (non-learned)")
        check(cfgs[mid]["training"] == "NOT_APPLICABLE_NON_LEARNED_METHOD", f"{mid}: no invented training")

    print("== TEST firewall ==")
    for mid, d in cfgs.items():
        t = d["data"]["splits"]["TEST"]
        check(t["allowed"] is False, f"{mid}: TEST not allowed")
        check(t["used_for"] == [], f"{mid}: TEST used_for empty")
        for k in ("never_training", "never_checkpoint_selection", "never_hyperparameter_selection",
                  "never_seed_selection"):
            check(t[k] is True, f"{mid}: TEST {k}")
        check(t["code_path_present"] is False, f"{mid}: no TEST code path")
        check(d["data"]["splits"]["VAL"]["may_select_checkpoint"] is False, f"{mid}: VAL may not select")

    print("== source pins / asset firewall ==")
    for mid in ("E01", "E03", "E06c", "E07c"):
        check(bool(cfgs[mid]["source"].get("pinned_commit")), f"{mid}: source commit pinned")
        check(cfgs[mid]["source"].get("weight_bytes_from_source_cache") == "NONE",
              f"{mid}: no weight bytes from source_cache")
    check(cfgs["E05"]["source"]["executable_architecture_basis"]["pinned_commit"] ==
          "6baa180f1184ee79a6b967f9d80ee0e02a979ac7", "E05: [34] architecture basis pinned")
    lc = cfgs["E06c"]["external_assets"][0]
    check(lc["weight_bytes_in_git"] is False and lc["weight_bytes_from_source_cache"] is False,
          "E06c: LightCNN bytes external, not in git, not in source_cache")
    check(cfgs["E04"]["external_assets"]["geometry_engine"]["weight_bytes_from_source_cache"] == "NONE",
          "E04: 3DDFA bytes not in source_cache")

    print("== snapshots ==")
    for mid, (fn, _, _) in EXPECTED.items():
        a = (METHODS / fn).read_bytes()
        snap = ROOT / "frozen_config_snapshot" / "configs" / "methods" / fn
        check(snap.is_file() and snap.read_bytes() == a, f"{mid}: byte-identical frozen snapshot")

    print(f"\n{'PASS' if not fails else 'FAIL'}: {len(fails)} failure(s)")
    for f in fails:
        print("   -", f)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
