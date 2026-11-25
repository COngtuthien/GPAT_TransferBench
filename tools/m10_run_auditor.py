"""Independent per-run auditor of one M10 ResNet-18 detector run (fails closed; never trusts the runner's own status).

    SCIENTIFIC:    python -B tools/m10_run_auditor.py --repo . --run-root <m10_root>/<G>/seed_<S> --group G --seed S \
                       --runtime-root <rt> [--exit-marker ~/workdir/m10_<G>_s<S>.exit] --out audit.json
    QUALIFICATION: python -B tools/m10_run_auditor.py --run-root <dir> --expect expect.json --out audit.json

Independence: this file imports nothing from the detector code path (science_runner, resnet_runner, metrics,
m10_selection, m10_launcher, m10_run_manifest). It re-implements from the frozen texts (spec 15/16, evaluator
contract D1/D5/D6/rng_scheme, owner decisions O1-O7) and recomputes:
  - the TRAIN/VAL populations from split_v1 + M2 (TEST rows removed by the parquet row filter, never parsed), the pool
    digests and the TEST-firewall allowlist;
  - every step's batch identity (sampler permutations, without-replacement pools with remainder discard);
  - every step's learning rates (D6), the GradScaler scale trajectory (O1: 65536 / x2 / x0.5 / 2000), skipped-update
    consistency, the clipping evidence (pre/post-clip norms) and loss finiteness;
  - every epoch's VAL video aggregation, the O2 threshold by brute force, ACER, AUC (pairwise) and decisions;
  - the D5 checkpoint selection over all epochs and the best-so-far trajectory;
  - the retained checkpoint's file SHA256 and, after a weights_only reload, its state SHA256 and finiteness.
The seed-matched synthetic sample list is obtained through the audited bank family readers
(gpatbench.downstream.science_banks.resolve_bank, every bank file re-hashed); that reader is part of the bank audits,
not of the detector runner. What cannot be reconstructed from the logs (e.g. the exact kernel order inside a CUDA
step) is not claimed; the corresponding evidence (recorded flags, observed dtypes) is checked for presence and value.
"""
import argparse
import hashlib
import json
import math
import os
import re
import stat
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "m10_resnet18_run_audit_v1"
CONTRACT_SHA = "67e4b62ddc6b81b8b9a8657e71b3979f6e6a186827183b052ae3d29b587fcbe0"
CONFIG_SHA = "f117c57380404bccfdc42370290c4bd2141b69346d3ee3f044d020f8b90a447b"
LOCK_SHA = "70202661f1de3188aa7a6460a640982cea98074e1c41146ff516ac6fef3d7b9c"
WEIGHTS_SHA = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
SPLIT_SHA = "fb9aeb369a124fc96ba855ef2ce269236c4a743fe960e73ab739412c9cb5092d"
M2_SHA = "65bb83e69ea7cebcb37373971e5e8f5248609b7963fd9d32aa1c0ba4377c0662"
AUTHORITY = "configs/amendments/m10_resnet18_science_launch_authority.yaml"
OWNER = "configs/amendments/m10_resnet18_owner_decisions.yaml"
PIN = "configs/amendments/m10_resnet18_science_precision.yaml"
STORAGE = "configs/amendments/m10_detector_storage_location.yaml"
M10_ROOT = "/media/ext_data/student20261_gpat_m10_detectors"
SEEDS = (42, 1337, 2026)
TRACKS = {"E00": "binary", "E01": "A_FAIR_COMMON_IDENTITY_FREE", "E02": "A_FAIR_COMMON_IDENTITY_FREE",
          "E03": "A_FAIR_COMMON_IDENTITY_FREE", "E04": "A_FAIR_COMMON_IDENTITY_FREE", "E05": "A_FAIR_COMMON_IDENTITY_FREE",
          "E06b": "B_NATIVE_FULL_SECONDARY", "E06c": "A_FAIR_COMMON_IDENTITY_FREE"}
# frozen detector values (configs/frozen/downstream_resnet18.yaml, evaluator contract D1/D6, owner decision O1)
LR_BACKBONE, LR_HEAD, MIN_LR, WD, CLIP = 1.0e-4, 5.0e-4, 1.0e-6, 1.0e-4, 1.0
MIXED, REAL_ONLY = {"live": 32, "real_spoof": 16, "synthetic": 16}, {"live": 32, "real_spoof": 32, "synthetic": 0}
O1_POLICY = {"name": "CUDA_AMP_FP16_GRADSCALER_DETERMINISTIC", "device": "cuda", "autocast_dtype": "float16",
             "grad_scaler": True, "deterministic": True}
O1_SCALER = {"init_scale": 65536.0, "growth_factor": 2.0, "backoff_factor": 0.5, "growth_interval": 2000, "enabled": True}
O1_DETERMINISM = {"use_deterministic_algorithms": True, "cudnn_deterministic": True, "cudnn_benchmark": False,
                  "matmul_allow_tf32": False, "cudnn_allow_tf32": False, "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
ENV = {"name": "gpat-m8-downstream", "python": "3.11.16", "torch": "2.12.1+cu130", "torchvision": "0.27.1+cu130",
       "cuda": "13.0", "cudnn": 92000, "sys_prefix": "/home/student20261/workdir/envs/gpat-m8-downstream",
       "lock_sha256": LOCK_SHA}
MANIFEST_KEYS = ("schema", "label", "run_id", "method_id", "track_id", "detector_seed", "bank_seed", "source_commit",
                 "execution_commit", "code_sha256", "evaluator_contract_sha256", "frozen_config_sha256",
                 "owner_decisions_sha256", "precision_pin_sha256", "launch_authority", "storage_authority",
                 "split_manifest_sha256", "m2_accounting_sha256", "pool_sha256", "pool_counts", "synthetic_bank",
                 "weights_sha256", "environment", "gpu", "precision_policy", "grad_scaler", "determinism",
                 "amp_observed", "optimizer", "scheduler", "rng", "command", "argv", "working_directory", "run_root",
                 "start_utc", "end_utc", "epochs", "warmup_epochs", "steps_per_epoch", "total_steps",
                 "optimizer_steps_applied", "optimizer_steps_skipped", "epoch_metrics", "checkpoint",
                 "val_predictions", "selection", "logs", "firewall", "exit_status", "audit", "reproduction",
                 "test_accessed", "backup")
RUN_ENTRIES = {"launch_preflight.json", "pools.json", "run_status.json", "firewall_report.json", "run_manifest.json",
               "exit_status.json", "detector"}


class Missing(Exception):
    pass


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def jload(p):
    p = Path(p)
    if not p.is_file():
        raise Missing(f"missing {p.name}")
    return json.loads(p.read_text())


# ------------------------------------------------------------------------------------------ independent re-derivations
def permutation(method, seed, pool, i, n):
    import numpy as np
    s = int.from_bytes(hashlib.sha256(f"gpatbench.downstream.sampler.v1|resnet18|{method}|{seed}|{pool}|{i}"
                                      .encode()).digest(), "big")
    return [int(x) for x in np.random.Generator(np.random.PCG64(s)).permutation(n)]


class PoolSim:
    def __init__(self, ids, method, seed, name):
        self.ids, self.m, self.s, self.name = ids, method, seed, name
        self.i, self.ptr, self.order = 0, 0, permutation(method, seed, name, 0, len(ids))

    def take(self, k):
        if len(self.order) - self.ptr < k:
            self.i += 1
            self.order, self.ptr = permutation(self.m, self.s, self.name, self.i, len(self.ids)), 0
        out = [self.ids[j] for j in self.order[self.ptr:self.ptr + k]]
        self.ptr += k
        return out


def lr_d6(k, base, W, T):
    if k < W:
        return base * k / W
    return MIN_LR + (base - MIN_LR) * 0.5 * (1.0 + math.cos(math.pi * (k - W) / (T - W)))


def sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def brute_threshold(y, s):
    """spec 15 + O2: min ACER over unique midpoints + {0, 1}; then min float abs(t - 0.5) (the approved
    implementation's comparison); then the smallest t (O2 A_SMALLER_THRESHOLD)."""
    n_sp, n_lv = sum(1 for v in y if v == 1), sum(1 for v in y if v == 0)
    u = sorted(set(s))
    cands = sorted(set([(a + b) / 2.0 for a, b in zip(u, u[1:])] + [0.0, 1.0]))

    def acer(t):
        ap = Fraction(sum(1 for yi, si in zip(y, s) if yi == 1 and not si >= t), n_sp)
        bp = Fraction(sum(1 for yi, si in zip(y, s) if yi == 0 and si >= t), n_lv)
        return (ap + bp) / 2
    rows = [(acer(t), t) for t in cands]
    best = min(a for a, _ in rows)
    m = [t for a, t in rows if a == best]
    d = {t: abs(t - 0.5) for t in m}
    dm = min(d.values())
    return min(t for t in m if d[t] == dm), best


def pair_auc(y, s):
    pos = [b for a, b in zip(y, s) if a == 1]
    neg = sorted(b for a, b in zip(y, s) if a == 0)
    import bisect
    tw = 0
    for p in pos:
        lo, hi = bisect.bisect_left(neg, p), bisect.bisect_right(neg, p)
        tw += 2 * lo + (hi - lo)
    return Fraction(tw, 2 * len(pos) * len(neg))


def state_digest(state):
    h = hashlib.sha256()
    for k in sorted(state):
        t = state[k].detach().cpu().contiguous()
        h.update(f"{k}|{t.dtype}|{tuple(t.shape)}|".encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def rebuild_populations(repo):
    """TRAIN/VAL pools from split_v1 + M2 (TEST rows filtered by the parquet reader; none reaches Python)."""
    import pyarrow.parquet as pq
    sp, m2 = Path(repo) / "manifests/split_v1.parquet", Path(repo) / "manifests/m2_sample_accounting.parquet"
    if sha_file(sp) != SPLIT_SHA or sha_file(m2) != M2_SHA:
        raise Missing("split_v1 / M2 hash differs")
    rows = pq.read_table(sp, columns=["sample_id", "dataset", "video_id", "label_binary", "split", "content_group_id",
                                      "subject_id_global", "allocation_group_id"],
                         filters=[("split", "in", ["TRAIN", "VAL"])]).to_pylist()
    if {r["split"] for r in rows} - {"TRAIN", "VAL"}:
        raise Missing("non TRAIN/VAL row")
    ids = sorted(r["sample_id"] for r in rows)
    face = {r["sample_id"]: r["face_png_sha256"] for r in pq.read_table(
        m2, columns=["sample_id", "face_png_sha256"], filters=[("sample_id", "in", ids)]).to_pylist()}
    pools = {"train_live": [], "train_real_spoof": [], "val": []}
    for r in sorted(rows, key=lambda x: x["sample_id"]):
        k = "val" if r["split"] == "VAL" else ("train_live" if int(r["label_binary"]) == 0 else "train_real_spoof")
        pools[k].append({"sample_id": r["sample_id"], "face_sha256": face[r["sample_id"]],
                         "rel": f"{r['dataset']}/{r['sample_id']}.png", "video": (r["dataset"], r["video_id"]),
                         "group": (r["dataset"], r["content_group_id"] if r["dataset"] == "siwmv2" else r["subject_id_global"]),
                         "alloc": r["allocation_group_id"]})
    tr = pools["train_live"] + pools["train_real_spoof"]
    disjoint = not ({f["video"] for f in tr} & {f["video"] for f in pools["val"]}) and \
        not ({f["group"] for f in tr} & {f["group"] for f in pools["val"]}) and \
        not ({f["alloc"] for f in tr} & {f["alloc"] for f in pools["val"]})
    return {"ids": {k: [f["sample_id"] for f in v] for k, v in pools.items()},
            "pool_sha256": {k: sha_bytes("\n".join(f"{f['sample_id']}|{f['face_sha256']}" for f in v).encode())
                            for k, v in pools.items()},
            "allowlist_sha256": sha_bytes("\n".join(sorted({f["rel"] for v in pools.values() for f in v})).encode()),
            "allowlist_size": len({f["rel"] for v in pools.values() for f in v}),
            "val_videos": len({f["video"] for f in pools["val"]}), "train_val_disjoint": disjoint}


def scientific_expectations(repo, group, seed, runtime_root, bank_verify="files"):
    import subprocess
    repo = Path(repo)
    a = jload(repo / AUTHORITY)
    if a.get("status") != "OWNER_APPROVED_ACTIVE" or a.get("approved") is not True:
        raise Missing("launch authority not active")
    owner = jload(repo / OWNER)
    if owner["O7_track_labels"]["mapping"] != TRACKS:
        raise Missing("owner O7 mapping differs from the auditor's confirmed mapping")
    pops = rebuild_populations(repo)
    syn, bank = [], None
    if group != "E00":
        sys.path.insert(0, str(repo))
        from gpatbench.downstream import science_banks as SB       # audited bank family readers (not the runner)
        rb = SB.resolve_bank(repo, group, seed, verify_pixels=False)
        syn = [s.sample_id for s in rb["samples"]]
        b = rb["binding"]
        bank = {"manifest_sha256": a["bank_families"][group]["banks"][str(seed)], "path": b["path"],
                "manifest_file": str(Path(b["path"]) / ("bank_manifest.parquet" if b["family"] == "bank_writer"
                                                         else "bank_manifest.jsonl")),
                "closure": b["closure"], "closure_sha256": a["bank_families"][group]["closure_sha256"],
                "samples_sha256": rb["samples_sha256"], "reader_ok": rb["reader"]["ok"], "family": b["family"]}
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    return {"profile": "scientific", "label": "SCIENTIFIC", "group": group, "seed": seed, "track": TRACKS[group],
            "epochs": 40, "warmup_epochs": 3, "run_root": os.path.join(M10_ROOT, group, "seed_%d" % seed),
            "check_group_dir": True, "authority_sha256": sha_file(repo / AUTHORITY), "authority": a,
            "owner_decisions_sha256": sha_file(repo / OWNER), "precision_pin_sha256": sha_file(repo / PIN),
            "storage_authority_sha256": sha_file(repo / STORAGE) if (repo / STORAGE).exists() else None,
            "source_commit": a["bindings"]["source"]["commit"], "code_sha256": a["bindings"]["source"]["code_sha256"],
            "env": ENV, "gpu": {"uuid": a["bindings"]["gpu"]["uuid"], "driver": a["bindings"]["gpu"]["driver"],
                                "name": "NVIDIA GeForce RTX 3090"},
            "pools": dict(pops["ids"], synthetic=syn), "pool_sha256": pops["pool_sha256"],
            "train_val_disjoint": pops["train_val_disjoint"], "val_videos": pops["val_videos"],
            "allowlist_sha256": pops["allowlist_sha256"], "allowlist_size": pops["allowlist_size"],
            "firewall_required": True, "bank": bank, "qualification_overrides": None, "grad_scaler": O1_SCALER,
            "exit_marker_required": True, "audit_head": head}


# ------------------------------------------------------------------------------------------------- the audit
def audit(run_root, exp, *, exit_marker=None) -> dict:
    gates, notes = {}, []
    rr = Path(run_root)
    det = rr / "detector"

    def gate(name, fn):
        try:
            ok, detail = fn()
        except Missing as exc:
            ok, detail = False, {"missing": str(exc)}
        except Exception as exc:                                    # noqa: BLE001
            ok, detail = False, {"error": f"{type(exc).__name__}: {exc}"}
        gates[name] = {"ok": bool(ok), **(detail or {})}
        return bool(ok)

    st = {}

    def load_all():
        st["manifest"] = jload(rr / "run_manifest.json")
        st["epochs"] = jload(det / "epochs.json")
        st["selection"] = jload(det / "selection.json")
        st["detector"] = jload(det / "detector.json")
        st["pools"] = jload(rr / "pools.json")
        st["best"] = jload(det / "best_so_far.json")
        p = det / "steps.jsonl"
        if not p.is_file():
            raise Missing("missing steps.jsonl")
        st["steps"] = [json.loads(x) for x in p.read_text().splitlines()]
        return True, {}
    loaded = gate("A00_required_evidence_present", load_all)
    M, E, SEL, DET, P, STEPS = (st.get(k) for k in ("manifest", "epochs", "selection", "detector", "pools", "steps"))
    group, seed = exp["group"], exp["seed"]
    real_only = group == "E00"
    comp = REAL_ONLY if real_only else MIXED

    def layout():
        if str(rr) != exp["run_root"] or os.path.realpath(rr) != str(rr) or rr.is_symlink():
            return False, {"run_root": str(rr), "expected": exp["run_root"]}
        entries = {p.name for p in rr.iterdir()}
        dentries = sorted(p.name for p in det.iterdir()) if det.is_dir() else []
        n = exp["epochs"]
        want_det = {"steps.jsonl", "epochs.json", "best_so_far.json", "selection.json", "detector.json",
                    *("predictions_val_epoch%02d.parquet" % e for e in range(n))}
        ck = [x for x in dentries if re.fullmatch(r"checkpoint_epoch\d\d\.pt", x)]
        extra = sorted(set(dentries) - want_det - set(ck))
        mode_ok = stat.S_IMODE(os.lstat(rr).st_mode) == 0o700
        ok = entries == RUN_ENTRIES and set(want_det) <= set(dentries) and len(ck) == 1 and not extra and mode_ok
        return ok, {"run_entries": sorted(entries), "unexpected_detector_entries": extra, "checkpoints": ck,
                    "run_root_mode_0700": mode_ok}
    gate("A01_run_root_layout_selected_only", layout)

    def group_dir():
        if not exp.get("check_group_dir"):
            return True, {"skipped": "qualification profile"}
        names = sorted(p.name for p in rr.parent.iterdir())
        ok = all(n in {"seed_%d" % s for s in SEEDS} for n in names) and rr.parent.parent == Path(M10_ROOT)
        return ok, {"group_dir_entries": names}
    gate("A02_no_overlapping_or_rerun_roots", group_dir)

    def exit_status():
        ex = jload(rr / "exit_status.json")
        rs = jload(rr / "run_status.json")
        ev = [e["status"] for e in rs.get("events", [])]
        ok = ex.get("exit_code") == 0 and ex.get("error") is None and rs.get("status") == "COMPLETED" \
            and ev == ["STARTED", "COMPLETED"] and ex.get("run_id") == (M or {}).get("run_id")
        ext = None
        if exit_marker:
            ext = Path(exit_marker).read_text().strip() if Path(exit_marker).is_file() else None
            ok = ok and ext == "0"
        elif exp.get("exit_marker_required"):
            ok, ext = False, "external exit marker required but not given"
        return ok, {"exit_code": ex.get("exit_code"), "events": ev, "external_marker": ext,
                    "failure_json_present": (det / "failure.json").exists()}
    gate("A03_exit_status_and_completion", exit_status)
    if not loaded:
        return finish(gates, notes, exp, rr)

    def manifest():
        missing = [k for k in MANIFEST_KEYS if k not in M]
        ok = not missing and M["schema"] == "m10_resnet18_run_manifest_v2" and M["label"] == exp["label"] \
            and M["method_id"] == group and M["detector_seed"] == seed and M["test_accessed"] is False \
            and M["audit"].get("status") == "PENDING_INDEPENDENT_AUDIT" and M["run_root"] == exp["run_root"] \
            and SEL["label"] == exp["label"] and DET["label"] == exp["label"] and M["exit_status"].get("exit_code") == 0 \
            and M["start_utc"] <= M["end_utc"] and bool(M["reproduction"]) and bool(M["command"]) \
            and "independent" in M["backup"] and not M["backup"].lower().startswith("independent copy exists")
        return ok, {"missing_keys": missing, "label": M.get("label")}
    gate("A04_run_manifest_complete", manifest)

    def authority():
        if exp["profile"] != "scientific":
            ok = M["launch_authority"] is None and M["owner_decisions_sha256"] == exp["owner_decisions_sha256"] \
                and M["precision_pin_sha256"] == exp["precision_pin_sha256"]
            return ok, {"profile": exp["profile"]}
        a = exp["authority"]
        ok = M["launch_authority"] == {"path": AUTHORITY, "sha256": exp["authority_sha256"]} \
            and M["owner_decisions_sha256"] == exp["owner_decisions_sha256"] == a["bindings"]["owner_decisions"]["sha256"] \
            and M["precision_pin_sha256"] == exp["precision_pin_sha256"] == a["bindings"]["precision_pin"]["sha256"] \
            and (M["storage_authority"] or {}).get("sha256") == exp["storage_authority_sha256"] \
            == a["bindings"]["storage_authority"]["sha256"] and [group, seed] in a["matrix"]["order"] \
            and a["authorizes"]["test_evaluation"] is False
        return ok, {"authority_sha256": exp["authority_sha256"]}
    gate("A05_launch_and_owner_authority", authority)

    gate("A06_source_commit_and_code", lambda: (
        M["source_commit"] == exp["source_commit"] and M["code_sha256"] == exp["code_sha256"]
        and re.fullmatch(r"[0-9a-f]{40}", M["execution_commit"]) is not None,
        {"source_commit": M["source_commit"], "execution_commit": M["execution_commit"]}))
    gate("A07_frozen_contract_and_config", lambda: (
        M["evaluator_contract_sha256"] == CONTRACT_SHA and M["frozen_config_sha256"] == CONFIG_SHA, {}))
    gate("A08_environment", lambda: (
        all(M["environment"].get(k) == v for k, v in exp["env"].items()),
        {"observed": {k: M["environment"].get(k) for k in exp["env"]}}))
    gate("A09_gpu_identity", lambda: (
        all(M["gpu"].get(k) == v for k, v in exp["gpu"].items()), {"observed": M["gpu"]}))
    gate("A10_imagenet_weights", lambda: (M["weights_sha256"] == WEIGHTS_SHA and DET["weights_sha256"] == WEIGHTS_SHA, {}))

    def track():
        import pyarrow.parquet as pq
        t = exp["track"]
        bad = 0
        for e in range(exp["epochs"]):
            p = det / ("predictions_val_epoch%02d.parquet" % e)
            if p.is_file():
                bad += sum(1 for v in pq.read_table(p, columns=["track_id"]).column(0).to_pylist() if v != t)
        ok = t == TRACKS[group] and M["track_id"] == t and SEL["track_id"] == t and DET["track_id"] == t and bad == 0
        return ok, {"track_id": M["track_id"], "expected": t, "prediction_rows_with_other_track": bad}
    gate("A11_track_label_O7", track)

    def pools():
        want = exp["pools"]
        ok = all(P[k] == want[k] for k in ("train_live", "train_real_spoof", "val", "synthetic")) \
            and P["pool_sha256"] == M["pool_sha256"] == exp["pool_sha256"] \
            and exp.get("train_val_disjoint", True) is True \
            and not (set(P["train_live"]) | set(P["train_real_spoof"])) & set(P["val"]) \
            and M["split_manifest_sha256"] == exp.get("split_sha256", SPLIT_SHA) \
            and M["m2_accounting_sha256"] == exp.get("m2_sha256", M2_SHA)
        return ok, {"counts": {k: len(P[k]) for k in ("train_live", "train_real_spoof", "val", "synthetic")}}
    gate("A12_train_val_split_identity", pools)

    def bank():
        b = exp["bank"]
        if real_only:
            return (M["bank_seed"] is None and M["synthetic_bank"] is None and P["synthetic"] == []
                    and DET["bank_seed"] is None and all(s["n_synthetic"] == 0 for s in STEPS)), {"E00": "real-only"}
        mb = M["synthetic_bank"] or {}
        disk = sha_file(b["manifest_file"]) if Path(b["manifest_file"]).is_file() else None
        ok = M["bank_seed"] == seed and DET["bank_seed"] == seed and mb.get("seed") == seed \
            and mb.get("method_id") == group and mb.get("manifest_sha256") == b["manifest_sha256"] == disk \
            and mb.get("samples_sha256") == b["samples_sha256"] and b.get("reader_ok", True) is True
        if b.get("closure"):
            ok = ok and sha_file(Path(exp.get("repo", ROOT)) / b["closure"]) == b["closure_sha256"]
        return ok, {"bank_seed": M["bank_seed"], "manifest_on_disk": disk, "expected": b["manifest_sha256"]}
    gate("A13_seed_matched_bank_mapping", bank)

    n_live = len(P["train_live"])
    S = n_live // 32
    T = exp["epochs"] * S
    W = exp["warmup_epochs"] * S

    gate("A14_batch_composition", lambda: (
        bool(STEPS) and all(s["n_live"] == comp["live"] and s["n_real_spoof"] == comp["real_spoof"]
                            and s["n_synthetic"] == comp["synthetic"] and s["batch"] == 64 for s in STEPS),
        {"composition": comp}))

    def steps():
        ok = [s["step"] for s in STEPS] == list(range(T)) and all(s["epoch"] == s["step"] // S for s in STEPS) \
            and M["epochs"] == DET["epochs"] == exp["epochs"] and M["warmup_epochs"] == exp["warmup_epochs"] \
            and M["steps_per_epoch"] == DET["steps_per_epoch"] == S and M["total_steps"] == DET["total_steps"] == T \
            and [e["epoch"] for e in E] == list(range(exp["epochs"]))
        return ok, {"steps": len(STEPS), "expected": T, "steps_per_epoch": S, "epochs": len(E)}
    gate("A15_epochs_and_step_count", steps)

    def sampler():
        live, rs = P["train_live"], P["train_real_spoof"]
        sp = PoolSim(rs, group, seed, "real_spoof")
        sy = None if real_only else PoolSim(P["synthetic"], group, seed, "synthetic")
        bad, k = [], 0
        for epoch in range(exp["epochs"]):
            order = permutation(group, seed, "live", epoch, n_live)
            for b in range(S):
                ids = [live[i] for i in order[b * 32:(b + 1) * 32]] + sp.take(comp["real_spoof"]) + \
                      ([] if real_only else sy.take(comp["synthetic"]))
                if k >= len(STEPS) or STEPS[k]["batch_ids_sha256"] != sha_bytes("\n".join(ids).encode()):
                    bad.append(k)
                k += 1
        passes = {"real_spoof": sp.i, "synthetic": None if real_only else sy.i}
        return not bad and DET["pool_passes"] == passes, {"mismatched_steps": bad[:20], "pool_passes": passes}
    gate("A16_sampler_batches_rederived", sampler)

    def lr():
        bad = [s["step"] for s in STEPS if s["lr_backbone_hex"] != lr_d6(s["step"], LR_BACKBONE, W, T).hex()
               or s["lr_head_hex"] != lr_d6(s["step"], LR_HEAD, W, T).hex()]
        sc = M["scheduler"]
        ok = not bad and sc.get("warmup_steps") == W and sc.get("total_steps") == T and sc.get("min_lr") == MIN_LR
        return ok, {"mismatched_steps": bad[:20], "W": W, "T": T}
    gate("A17_lr_schedule_D6", lr)

    def optimizer():
        o = M["optimizer"]
        g = {x["name"]: x for x in o.get("groups", [])}
        ok = o.get("class") == "AdamW" and set(g) == {"backbone", "head"} and o.get("grad_clip_norm") == CLIP \
            and o.get("head_names") == ["fc.weight", "fc.bias"] and g["head"]["n_tensors"] == 2 \
            and g["head"]["n_parameters"] == 513 and g["backbone"]["base_lr"] == LR_BACKBONE \
            and g["head"]["base_lr"] == LR_HEAD and all(
                x["betas"] == [0.9, 0.999] and x["eps"] == 1e-8 and x["weight_decay"] == WD and x["amsgrad"] is False
                and x["maximize"] is False for x in g.values()) and DET["optimizer"] == o
        return ok, {"groups": {k: {kk: v.get(kk) for kk in ("n_tensors", "n_parameters", "base_lr")} for k, v in g.items()}}
    gate("A18_optimizer_AdamW_groups", optimizer)

    def amp():
        if exp["profile"] == "scientific" or exp.get("expect_o1_policy", True):
            ok = M["precision_policy"] == O1_POLICY and M["determinism"] == O1_DETERMINISM \
                and DET["determinism_after"] == O1_DETERMINISM and M["grad_scaler"] == exp["grad_scaler"] \
                and M["amp_observed"] == {"train_logits_dtype": "torch.float16", "val_logits_dtype": "torch.float16"}
        else:
            ok = M["precision_policy"]["name"] == exp["policy_name"]
        return ok, {"policy": M["precision_policy"], "grad_scaler": M["grad_scaler"], "amp_observed": M["amp_observed"]}
    gate("A19_amp_gradscaler_policy_O1", amp)

    def scaler():
        cfg = M["grad_scaler"]
        if cfg is None:
            return exp["profile"] != "scientific" and all(s["grad_scale"] is None for s in STEPS), {"scaler": None}
        scale, tracker, bad = cfg["init_scale"], 0, []
        for s in STEPS:
            applied = s["optimizer_step_applied"]
            finite = math.isfinite(float.fromhex(s["grad_norm_hex"]))
            if s["grad_scale_before"] != scale or applied != finite or (not applied and scale <= 1.0):
                bad.append(s["step"])
            if applied:
                tracker += 1
                if tracker == cfg["growth_interval"]:
                    scale, tracker = scale * cfg["growth_factor"], 0
            else:
                scale, tracker = scale * cfg["backoff_factor"], 0
            if s["grad_scale"] != scale:
                bad.append(s["step"])
        applied = sum(1 for s in STEPS if s["optimizer_step_applied"])
        per_epoch = [sum(1 for s in STEPS if s["epoch"] == e and s["optimizer_step_applied"]) for e in range(exp["epochs"])]
        ok = not bad and M["optimizer_steps_applied"] == applied and M["optimizer_steps_skipped"] == T - applied \
            and all(n > 0 for n in per_epoch) and [e["optimizer_steps_applied"] for e in E] == per_epoch
        return ok, {"inconsistent_steps": sorted(set(bad))[:20], "applied": applied, "skipped": T - applied,
                    "final_scale": scale}
    gate("A20_gradscaler_trajectory_and_effective_steps", scaler)

    def clipping():
        bad = []
        for s in STEPS:
            if not s["optimizer_step_applied"]:
                continue
            pre = float.fromhex(s["grad_norm_hex"])
            post = None if s["grad_norm_post_clip_hex"] is None else float.fromhex(s["grad_norm_post_clip_hex"])
            if post is None or post > CLIP * (1 + 1e-5) or (pre <= CLIP and abs(post - pre) > 1e-4 * max(pre, 1e-12) + 1e-7) \
                    or (pre > CLIP and post < CLIP * (1 - 1e-4)):
                bad.append(s["step"])
        clipped = sum(1 for s in STEPS if s["optimizer_step_applied"] and float.fromhex(s["grad_norm_hex"]) > CLIP)
        return not bad, {"violations": bad[:20], "steps_where_clip_active": clipped,
                         "note": "post-clip norm ~= 1 whenever the unscaled pre-clip norm exceeds 1 evidences clipping "
                                 "on unscaled gradients (clipping scaled gradients would leave ~1/scale)"}
    gate("A21_gradient_clipping_after_unscale", clipping)

    def losses():
        bad = [s["step"] for s in STEPS if not math.isfinite(float.fromhex(s["loss_hex"]))]
        mean_bad = [e["epoch"] for e in E if e["train_loss_mean"] != math.fsum(
            float.fromhex(s["loss_hex"]) for s in STEPS if s["epoch"] == e["epoch"]) / S]
        return not bad and not mean_bad, {"nonfinite_steps": bad[:20], "epoch_mean_mismatch": mean_bad}
    gate("A22_training_loss_finite_and_epoch_metrics", losses)

    vals = {}

    def predictions():
        import pyarrow.parquet as pq
        bad = []
        val_ids = P["val"]
        for e in E:
            p = det / ("predictions_val_epoch%02d.parquet" % e["epoch"])
            if not p.is_file() or sha_file(p) != e["predictions_sha256"] or \
                    M["val_predictions"][e["epoch"]]["sha256"] != e["predictions_sha256"]:
                bad.append((e["epoch"], "file/sha"))
                continue
            rows = pq.read_table(p).to_pylist()
            if [r["sample_id"] for r in rows] != sorted(val_ids) or len(rows) != len(val_ids):
                bad.append((e["epoch"], "rows"))
                continue
            if any(r["method_id"] != group or r["seed"] != seed or r["evaluator_id"] != "resnet18"
                   or r["checkpoint_sha256"] != e["state_sha256"] or r["split_manifest_sha256"] != M["split_manifest_sha256"]
                   for r in rows):
                bad.append((e["epoch"], "row identity"))
            pf = [sigmoid(float.fromhex(r["logit_hex"])) for r in rows]
            if any(a != r["p_frame"] for a, r in zip(pf, rows)):
                bad.append((e["epoch"], "p_frame != sigmoid(logit)"))
            if sha_bytes("\n".join(r["logit_hex"] for r in rows).encode()) != e["logits_sha256"]:
                bad.append((e["epoch"], "logits sha"))
            vals[e["epoch"]] = rows
        return not bad, {"problems": bad[:20], "epochs_checked": len(vals), "val_frames": len(val_ids)}
    gate("A23_val_predictions_complete", predictions)

    recomputed = {}

    def aggregation():
        bad = []
        for ep, rows in vals.items():
            g = {}
            for r in rows:
                g.setdefault((r["dataset"], r["video_id"]), []).append(r)
            for key, rs in g.items():
                if len({r["y_true"] for r in rs}) != 1:
                    bad.append((ep, key, "labels"))
                v = math.fsum(r["p_frame"] for r in rs) / len(rs)
                if any(r["video_score"] != v for r in rs):
                    bad.append((ep, key, "score"))
            keys = sorted(g, key=lambda k: (str(k[0]), str(k[1])))
            recomputed[ep] = ([g[k][0]["y_true"] for k in keys], [math.fsum(r["p_frame"] for r in g[k]) / len(g[k])
                                                                    for k in keys], len(keys))
        nv = {ep: x[2] for ep, x in recomputed.items()}
        ok = not bad and len(recomputed) == exp["epochs"] and set(nv.values()) == {exp["val_videos"]}
        return ok, {"problems": bad[:20], "videos": sorted(set(nv.values()))}
    gate("A24_video_aggregation_mean_of_frames", aggregation)

    def metrics():
        bad = []
        for e in E:
            ep = e["epoch"]
            if ep not in recomputed:
                bad.append((ep, "missing"))
                continue
            y, s, _ = recomputed[ep]
            t, acer = brute_threshold(y, s)
            auc = pair_auc(y, s)
            if float.fromhex(e["threshold_hex"]) != t or e["threshold"] != t or Fraction(e["val_acer"]) != acer \
                    or Fraction(e["val_auc"]) != auc:
                bad.append((ep, "threshold/acer/auc"))
            if any(r["threshold"] != t or r["pred_frame"] != int(r["p_frame"] >= t)
                   or r["pred_video"] != int(r["video_score"] >= t) for r in vals[ep]):
                bad.append((ep, "decisions"))
        return not bad, {"problems": bad[:20]}
    gate("A25_val_threshold_acer_auc_recomputed_O2", metrics)

    def selection():
        key = lambda e: (Fraction(e["val_acer"]), -Fraction(e["val_auc"]), e["epoch"])   # noqa: E731
        win = min(E, key=key)
        run, flags = None, []
        for e in E:
            nb = run is None or key(e) < key(run)
            flags.append(nb)
            if nb:
                run = e
        rsn = SEL["selection_reasoning"]
        ok = SEL["selected_epoch"] == win["epoch"] == M["selection"]["selected_epoch"] \
            and SEL["frozen_val_threshold_hex"] == win["threshold_hex"] == M["selection"]["frozen_val_threshold_hex"] \
            and SEL["checkpoint_state_sha256"] == win["state_sha256"] and SEL["val_acer"] == win["val_acer"] \
            and SEL["val_auc"] == win["val_auc"] and [e["became_best"] for e in E] == flags \
            and rsn["best_so_far_trajectory"] == [e["epoch"] for e, f in zip(E, flags) if f] \
            and rsn["epochs_evaluated"] == exp["epochs"] and SEL["test_used"] is False \
            and "VAL" in SEL["selection_source"] and win["epoch"] in rsn["epochs_at_min_acer_and_max_auc"]
        return ok, {"selected_epoch": win["epoch"], "final_epoch": E[-1]["epoch"] if E else None,
                    "threshold_hex": win["threshold_hex"], "val_acer": win["val_acer"], "val_auc": win["val_auc"]}
    gate("A26_checkpoint_selection_D5_and_frozen_threshold", selection)

    def checkpoint():
        import torch
        name = "checkpoint_epoch%02d.pt" % SEL["selected_epoch"]
        p = det / name
        if not p.is_file():
            raise Missing(f"missing {name}")
        fsha = sha_file(p)
        state = torch.load(p, map_location="cpu", weights_only=True)
        sd = state_digest(state)
        finite = all(bool(torch.isfinite(t).all()) for t in state.values() if t.is_floating_point())
        import torchvision
        ref = torchvision.models.resnet18(weights=None)
        ref.fc = torch.nn.Linear(512, 1)
        shapes = {k: tuple(v.shape) for k, v in ref.state_dict().items()}
        shape_ok = {k: tuple(v.shape) for k, v in state.items()} == shapes
        ok = fsha == SEL["checkpoint_file_sha256"] == M["checkpoint"]["file_sha256"] == st["best"]["checkpoint_file_sha256"] \
            and sd == SEL["checkpoint_state_sha256"] == st["best"]["state_sha256"] and finite and shape_ok \
            and SEL["checkpoint_file"] == name and M["checkpoint"]["path"] == "detector/" + name
        return ok, {"file_sha256": fsha, "state_sha256": sd, "finite": finite, "resnet18_fc1_shapes": shape_ok}
    gate("A27_selected_checkpoint_file_and_reload", checkpoint)

    def consistency():
        L = M["logs"]
        ok = sha_file(det / "steps.jsonl") == L["steps"]["sha256"] and sha_file(det / "epochs.json") == L["epochs"]["sha256"] \
            and sha_file(det / "selection.json") == L["selection"]["sha256"] \
            and sha_file(det / "detector.json") == L["detector"]["sha256"] \
            and [x["epoch"] for x in M["epoch_metrics"]] == list(range(exp["epochs"])) and all(
                m["val_acer"] == e["val_acer"] and m["val_auc"] == e["val_auc"] and m["state_sha256"] == e["state_sha256"]
                and m["threshold_hex"] == e["threshold_hex"] and m["became_best"] == e["became_best"]
                for m, e in zip(M["epoch_metrics"], E)) and DET["qualification_overrides"] == exp["qualification_overrides"]
        return ok, {}
    gate("A28_manifest_logs_consistent", consistency)

    def firewall():
        fw = jload(rr / "firewall_report.json")
        ok = fw.get("violation_count") == 0 and fw.get("listing_attempts") == 0 and M["firewall"].get("violation_count") == 0 \
            and P["firewall_allowlist_sha256"] == exp["allowlist_sha256"] \
            and P["firewall_allowlist_size"] == exp["allowlist_size"] == fw.get("allowlist_size") \
            and fw.get("distinct_files_opened", 10 ** 9) <= exp["allowlist_size"] and M["test_accessed"] is False
        if exp.get("firewall_required"):
            ok = ok and M["firewall"].get("installed") is True
        return ok, {k: fw.get(k) for k in ("allowlist_size", "opens_under_faces_root", "distinct_files_opened",
                                           "violation_count", "listing_attempts")}
    gate("A29_test_firewall_zero_access", firewall)

    notes.append("generator checkpoints are not read by the detector path; their integrity is covered by the bank "
                 "family closures bound in the launch authority (closure sha256 re-checked in A13), not re-proved here")
    notes.append("CUDA kernel-level execution order is not reconstructible from logs; the O1 policy is evidenced by the "
                 "recorded policy/scaler/determinism records, observed float16 logits, the scale trajectory and the "
                 "clipping norms, plus the authority-bound code hashes (A06)")
    return finish(gates, notes, exp, rr)


def finish(gates, notes, exp, rr):
    failed = [k for k, v in gates.items() if not v["ok"]]
    return {"schema": SCHEMA, "label": exp["label"], "profile": exp["profile"], "group": exp["group"],
            "seed": exp["seed"], "run_root": str(rr), "status": "PASS" if not failed else "FAIL",
            "gates_passed": len(gates) - len(failed), "gates_total": len(gates), "failed_gates": failed,
            "gates": gates, "not_proven": notes, "auditor_sha256": sha_file(Path(__file__)),
            "independence": "imports nothing from science_runner / resnet_runner / metrics / m10_selection / "
                            "m10_launcher / m10_run_manifest"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--repo", default=str(ROOT))
    ap.add_argument("--group")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--runtime-root")
    ap.add_argument("--expect", help="qualification expectations (non-scientific)")
    ap.add_argument("--exit-marker")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.expect:
        exp = json.loads(Path(a.expect).read_text())
        if exp.get("profile") != "qualification" or exp.get("label") != "NON_SCIENTIFIC_QUALIFICATION":
            raise SystemExit("--expect is for non-scientific qualification only")
    else:
        exp = scientific_expectations(a.repo, a.group, a.seed, a.runtime_root)
    exp.setdefault("repo", a.repo)
    res = audit(a.run_root, exp, exit_marker=a.exit_marker)
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    out.write_text(json.dumps(res, indent=1, sort_keys=True, default=str) + "\n")
    print(json.dumps({"status": res["status"], "failed_gates": res["failed_gates"],
                      "gates": f"{res['gates_passed']}/{res['gates_total']}"}))
    return 0 if res["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
