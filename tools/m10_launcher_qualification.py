"""NON-SCIENTIFIC GPU qualification of the M10 launcher and the independent per-run auditor (synthetic fixtures only).

GPU host, fresh clean clone at the candidate commit, approved environment:
    env -i HOME=$HOME PATH=/usr/bin:/bin LANG=C.UTF-8 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 \
        PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
        ~/workdir/envs/gpat-m8-downstream/bin/python -B tools/m10_launcher_qualification.py \
        --out ~/workdir/m10_qualification/<commit> --runtime-root ~/workdir/GPAT_TransferBench_runtime [--dry-preflight]

  F1 fixture runs through the real CLI (tools/run_m10_detector.py --fixture): O1 policy from the owner pin, full
     preflight (source, frozen, environment, weights, GPU, lock, qualification root outside official roots), TEST
     firewall audit hook (fixture TEST decoys present on disk), best-so-far retention; groups E03, E00, E06b
  F2 bitwise replay of E03 (steps, epochs, checkpoint file SHA256)
  F3 GradScaler: an injected inf gradient at step 1 is skipped (scale 65536 -> 32768) and training recovers;
     scale 1 + inf gradient stops the run (fail closed); a non-finite loss stops the run with partial evidence
  F4 the create-only refusal: relaunching into an existing run root is refused with nothing changed
  F5 independent auditor (CLI) PASS on every good run; 17+ injected failures each FAIL with the expected gate
  F6 scientific dry preflight of all 24 runs against an IN-MEMORY candidate authority (never written): every gate
     except the O5 predecessor gate of the 23 later runs must pass, including full bank verification (pixels) and
     the TRAIN/VAL populations; no run root is created
  F7 resources: GPU snapshot, HDD capacity, the detector-root preflight, provisional throughput and storage footprint
No real face, bank image or TEST file is decoded for training; fixture metrics are not scientific findings.
"""
import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gpatbench.downstream import m10_fixture as FX  # noqa: E402
from gpatbench.downstream import m10_launcher as L  # noqa: E402
from gpatbench.downstream import m10_storage as ST  # noqa: E402

EPOCHS, WARMUP = 3, 1
O1_SCALER = {"init_scale": 65536.0, "growth_factor": 2.0, "backoff_factor": 0.5, "growth_interval": 2000, "enabled": True}


def run_cli(args, timeout=3600):
    t0 = time.monotonic()
    r = subprocess.run([sys.executable, "-B", str(ROOT / "tools/run_m10_detector.py"), *args], cwd=str(ROOT),
                       capture_output=True, text=True, timeout=timeout)
    return {"rc": r.returncode, "stdout_tail": r.stdout[-3000:], "stderr_tail": r.stderr[-3000:],
            "seconds": time.monotonic() - t0}


def audit_cli(run_root, expect_path, out_path):
    r = subprocess.run([sys.executable, "-B", str(ROOT / "tools/m10_run_auditor.py"), "--run-root", str(run_root),
                        "--expect", str(expect_path), "--out", str(out_path)], cwd=str(ROOT), capture_output=True,
                       text=True, timeout=3600)
    d = json.loads(Path(out_path).read_text()) if Path(out_path).is_file() else {"status": "NO_OUTPUT"}
    return {"rc": r.returncode, "status": d.get("status"), "failed_gates": d.get("failed_gates"),
            "gates": f"{d.get('gates_passed')}/{d.get('gates_total')}", "stderr_tail": r.stderr[-1500:]}


def du(p):
    return sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file())


# ------------------------------------------------------------------------------------------- injected failures
def _jpatch(path, fn):
    d = json.loads(Path(path).read_text())
    fn(d)
    Path(path).write_text(json.dumps(d))


def _flip_ckpt(d):
    ck = next((d / "detector").glob("checkpoint_*.pt"))
    b = bytearray(ck.read_bytes())
    b[len(b) // 2] ^= 0xFF
    ck.write_bytes(bytes(b))


def _alter_predictions(d):
    import pyarrow as pa
    import pyarrow.parquet as pq
    p = d / "detector/predictions_val_epoch00.parquet"
    rows = pq.read_table(p).to_pylist()
    rows[0]["p_frame"] = min(1.0, rows[0]["p_frame"] + 0.01)
    os.unlink(p)
    pq.write_table(pa.Table.from_pylist(rows), p, compression="zstd")
    new = L.sha_file(p)
    _jpatch(d / "detector/epochs.json", lambda e: e[0].update(predictions_sha256=new))
    _jpatch(d / "run_manifest.json", lambda m: m["val_predictions"][0].update(sha256=new))


def _drop_last_line(p):
    lines = Path(p).read_text().splitlines()
    Path(p).write_text("\n".join(lines[:-1]) + "\n")


NEGATIVES = {
    "tampered_checkpoint": (_flip_ckpt, "A27_selected_checkpoint_file_and_reload"),
    "missing_run_manifest": (lambda d: os.unlink(d / "run_manifest.json"), "A00_required_evidence_present"),
    "altered_threshold": (lambda d: _jpatch(d / "detector/selection.json",
                                            lambda s: s.update(frozen_val_threshold_hex=(0.123).hex())),
                          "A26_checkpoint_selection_D5_and_frozen_threshold"),
    "altered_val_predictions_rehashed": (_alter_predictions, "A23_val_predictions_complete"),
    "incorrect_epoch_count": (lambda d: _jpatch(d / "detector/epochs.json", lambda e: e.pop()),
                              "A15_epochs_and_step_count"),
    "incorrect_optimizer_steps": (lambda d: _drop_last_line(d / "detector/steps.jsonl"), "A15_epochs_and_step_count"),
    "wrong_amp_policy": (lambda d: _jpatch(d / "run_manifest.json", lambda m: m["precision_policy"].update(
        name="CUDA_AMP_BF16_DETERMINISTIC", autocast_dtype="bfloat16")), "A19_amp_gradscaler_policy_O1"),
    "wrong_scaler_config": (lambda d: _jpatch(d / "run_manifest.json", lambda m: m.update(
        grad_scaler=dict(m["grad_scaler"] or {}, init_scale=1024.0))), "A19_amp_gradscaler_policy_O1"),
    "wrong_environment": (lambda d: _jpatch(d / "run_manifest.json", lambda m: m["environment"].update(
        torch="2.13.0+cu130")), "A08_environment"),
    "wrong_bank_seed": (lambda d: _jpatch(d / "run_manifest.json", lambda m: m.update(bank_seed=1337)),
                        "A13_seed_matched_bank_mapping"),
    "wrong_track_label": (lambda d: _jpatch(d / "run_manifest.json", lambda m: m.update(track_id="binary")),
                          "A11_track_label_O7"),
    "wrong_gpu": (lambda d: _jpatch(d / "run_manifest.json", lambda m: m["gpu"].update(uuid="GPU-00000000")),
                  "A09_gpu_identity"),
    "missing_exit_marker": (lambda d: os.unlink(d / "exit_status.json"), "A03_exit_status_and_completion"),
    "partial_run": (lambda d: _jpatch(d / "run_status.json", lambda s: (s.update(status="STARTED"),
                                                                        s.update(events=s["events"][:1]))),
                    "A03_exit_status_and_completion"),
    "corrupted_output": (lambda d: (d / "detector/predictions_val_epoch01.parquet").write_bytes(
        (d / "detector/predictions_val_epoch01.parquet").read_bytes()[:100]), "A23_val_predictions_complete"),
    "altered_bank_manifest": (None, "A13_seed_matched_bank_mapping"),          # applied to the expectation's copy
    "unauthorized_test_read": (lambda d: _jpatch(d / "firewall_report.json", lambda f: f.update(
        violation_count=1, violations=[{"event": "open", "path": "fixture/T0000.png"}])),
        "A29_test_firewall_zero_access"),
    "duplicate_run_content": (lambda d: shutil.copy(next((d / "detector").glob("checkpoint_*.pt")),
                                                    d / "detector" / "checkpoint_epoch99.pt"),
                              "A01_run_root_layout_selected_only"),
    "rerun_leftover_dir": (lambda d: (d / "detector.rerun").mkdir(), "A01_run_root_layout_selected_only"),
    "altered_lr": (lambda d: Path(d / "detector/steps.jsonl").write_text("".join(
        json.dumps(dict(s, lr_head_hex=(1e-3).hex()) if s["step"] == 1 else s) + "\n"
        for s in (json.loads(x) for x in Path(d / "detector/steps.jsonl").read_text().splitlines()))),
        "A17_lr_schedule_D6"),
    "altered_batch_identity": (lambda d: Path(d / "detector/steps.jsonl").write_text("".join(
        json.dumps(dict(s, batch_ids_sha256="0" * 64) if s["step"] == 2 else s) + "\n"
        for s in (json.loads(x) for x in Path(d / "detector/steps.jsonl").read_text().splitlines()))),
        "A16_sampler_batches_rederived"),
    "test_id_in_val_pool": (lambda d: _jpatch(d / "pools.json", lambda p: p["val"].__setitem__(0, "T0000")),
                            "A12_train_val_split_identity"),
}


def negatives(run_root, exp, work):
    out, ok = {}, True
    for name, (mutate, gate) in NEGATIVES.items():
        base = work / name
        d = base / exp["group"] / "seed_42"
        shutil.copytree(run_root, d)
        os.chmod(d, 0o700)
        e = copy.deepcopy(exp)
        e["run_root"] = str(d)
        if (d / "run_manifest.json").exists():
            _jpatch(d / "run_manifest.json", lambda m: m.update(run_root=str(d)))
        if e["bank"]:
            bm = base / "bank_manifest.json"
            shutil.copy(e["bank"]["manifest_file"], bm)
            e["bank"]["manifest_file"] = str(bm)
        if name == "altered_bank_manifest":
            Path(e["bank"]["manifest_file"]).write_text(Path(e["bank"]["manifest_file"]).read_text().replace("Y0000", "Y9999"))
        else:
            mutate(d)
        ep = base / "expect.json"
        ep.write_text(json.dumps(e))
        r = audit_cli(d, ep, base / "audit.json")
        hit = r["status"] == "FAIL" and gate in (r["failed_gates"] or [])
        ok = ok and hit
        out[name] = {"expected_gate": gate, "detected": hit, "status": r["status"], "failed_gates": r["failed_gates"]}
    return {"pass": ok, "cases": len(NEGATIVES), "results": out}


# ------------------------------------------------------------------------------------------- dry preflight (F6)
def dry_preflight(rt, weights, gpu):
    import importlib.util
    spec = importlib.util.spec_from_file_location("la", str(ROOT / "tools/m10_launch_authority.py"))
    la = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(la)
    head = L._git(ROOT, "rev-parse", "HEAD")
    cand = la.build_authority(ROOT, source_commit=head, in_memory_candidate=True)
    res, ok = {}, True
    banks_seconds = {}
    for g, s in L.ORDER:
        t0 = time.monotonic()
        pre = L.preflight(repo=ROOT, group=g, seed=s, mode="SCIENTIFIC", authority=cand, authority_sha="in-memory",
                          m10_root=ST.M10_ROOT, runtime_root=rt, weights_path=weights, gpu=gpu)
        try:
            inputs = L.scientific_inputs(ROOT, rt, g, s)
            L.input_gates(pre, inputs, cand, g, s, "SCIENTIFIC")
        except Exception as exc:                                    # noqa: BLE001
            pre["gates"]["inputs"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        failed = sorted(k for k, v in pre["gates"].items() if not v["ok"])
        expect_failed = [] if (g, s) == L.ORDER[0] else ["order_predecessors_audited"]
        this_ok = failed == expect_failed and not os.path.lexists(ST.run_root_path(ST.M10_ROOT, g, s))
        ok = ok and this_ok
        bank = pre["gates"].get("bank_identity", {})
        res[f"{g}|{s}"] = {"ok": this_ok, "failed_gates": failed, "expected_failed": expect_failed,
                           "bank_manifest_sha256": bank.get("manifest_sha256"),
                           "populations": pre["gates"].get("populations", {}).get("counts")}
        banks_seconds[f"{g}|{s}"] = round(time.monotonic() - t0, 1)
    return {"pass": ok, "runs": len(res), "results": res, "seconds_per_preflight": banks_seconds,
            "candidate_authority": "in memory only; never written", "source_commit": head}


# ------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--runtime-root", required=True)
    ap.add_argument("--weights", default="~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth")
    ap.add_argument("--dry-preflight", action="store_true")
    a = ap.parse_args()
    out = Path(os.path.expanduser(a.out))
    out.mkdir(parents=False, exist_ok=False)
    rt = os.path.expanduser(a.runtime_root)
    weights = os.path.expanduser(a.weights)
    doc = {"label": "NON_SCIENTIFIC_QUALIFICATION (synthetic fixtures; no real or bank image trained on; no TEST)",
           "utc_start": L.utc()}
    env = L.env_identity()
    gpu = L.gpu_identity()
    head = L._git(ROOT, "rev-parse", "HEAD")
    clean = L._git(ROOT, "status", "--porcelain", "--untracked-files=normal") == ""
    code = L.code_sha256(ROOT)
    env_gate = all(env.get(k) == v for k, v in L.ENV_EXPECTED.items()) and env["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    doc["identity"] = {"head": head, "clean": clean, "code_sha256": code, "environment": env,
                       "environment_gate": env_gate, "gpu": gpu["gpus"][0], "unrelated_compute_apps": gpu["compute_apps"],
                       "weights_ok": L.sha_file(weights) == "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec",
                       "lock_sha256": L.sha_file(ROOT / L.LOCK[0])}
    owner_sha, pin_sha = L.sha_file(ROOT / L.OWNER_PATH), L.sha_file(ROOT / L.PIN_PATH)
    exp_env = dict(L.ENV_EXPECTED, lock_sha256=L.LOCK[1])
    exp_gpu = {k: gpu["gpus"][0][k] for k in ("name", "uuid", "driver")}

    def fixture_and_run(tag, group, outroot, overrides=None):
        fx = out / "fixtures" / tag
        FX.build_fixture(fx, group=group, seed=42)
        args = ["--group", group, "--seed", "42", "--runtime-root", rt, "--fixture", str(fx), "--qualification-out",
                str(outroot), "--qualification-epochs", str(EPOCHS), "--qualification-warmup", str(WARMUP)]
        if overrides:
            args += ["--qualification-overrides", json.dumps(overrides)]
        r = run_cli(args)
        run_root = outroot / group / "seed_42"
        e = FX.expectations(fx, run_root=run_root, group=group, seed=42, epochs=EPOCHS, warmup_epochs=WARMUP,
                            env=exp_env, gpu=exp_gpu, code_sha256=code, source_commit=head,
                            owner_decisions_sha256=owner_sha, precision_pin_sha256=pin_sha,
                            qualification_overrides=overrides, grad_scaler=O1_SCALER)
        ep = out / "expect" / f"{tag}.json"
        ep.parent.mkdir(exist_ok=True)
        ep.write_text(json.dumps(e))
        return r, run_root, e, ep

    runs, audits = {}, {}
    (out / "audit").mkdir()
    for tag, group, root in (("E03_a", "E03", out / "runs_a"), ("E00_a", "E00", out / "runs_a"),
                             ("E06b_a", "E06b", out / "runs_a"), ("E03_b", "E03", out / "runs_b")):
        r, rr, e, ep = fixture_and_run(tag, group, root)
        runs[tag] = dict(r, run_root=str(rr))
        audits[tag] = audit_cli(rr, ep, out / "audit" / f"{tag}.json") if r["rc"] == 0 else {"status": "NOT_RUN"}
        if tag == "E03_a":
            good = (rr, e)
    doc["runs"] = runs
    doc["runs_pass"] = all(v["rc"] == 0 for v in runs.values())
    doc["audits"] = audits
    doc["audits_pass"] = all(v["status"] == "PASS" for v in audits.values())

    # F2 replay
    ra, rb = Path(runs["E03_a"]["run_root"]), Path(runs["E03_b"]["run_root"])

    def strip(p):
        return [{k: v for k, v in x.items() if k != "wall_seconds"} for x in json.loads(p.read_text())]
    sa, sb = (json.loads((r / "detector/selection.json").read_text()) for r in (ra, rb))
    doc["replay"] = {"steps_equal": (ra / "detector/steps.jsonl").read_bytes() == (rb / "detector/steps.jsonl").read_bytes(),
                     "epochs_equal": strip(ra / "detector/epochs.json") == strip(rb / "detector/epochs.json"),
                     "checkpoint_file_sha256_equal": sa["checkpoint_file_sha256"] == sb["checkpoint_file_sha256"],
                     "selected_epoch": sa["selected_epoch"], "checkpoint_file_sha256": sa["checkpoint_file_sha256"]}
    doc["replay"]["bitwise"] = all(doc["replay"][k] for k in ("steps_equal", "epochs_equal", "checkpoint_file_sha256_equal"))

    # F3 GradScaler skip / fail-closed
    r, rr, e, ep = fixture_and_run("E03_skip", "E03", out / "runs_skip", {"nonfinite_grad_at_step": 1})
    steps = [json.loads(x) for x in (rr / "detector/steps.jsonl").read_text().splitlines()] if r["rc"] == 0 else []
    au = audit_cli(rr, ep, out / "audit" / "E03_skip.json") if r["rc"] == 0 else {"status": "NOT_RUN"}
    doc["skip_recovery"] = {"rc": r["rc"], "audit": au,
                            "applied_first3": [s["optimizer_step_applied"] for s in steps[:3]],
                            "scale_step1": [steps[1]["grad_scale_before"], steps[1]["grad_scale"]] if steps else None}
    doc["skip_recovery"]["pass"] = r["rc"] == 0 and au["status"] == "PASS" and \
        doc["skip_recovery"]["applied_first3"] == [True, False, True] and doc["skip_recovery"]["scale_step1"] == [65536.0, 32768.0]
    fc = {}
    for tag, group, ov, needle in (("E00_unit_scale", "E00", {"grad_scaler_init_scale": 1.0, "nonfinite_grad_at_step": 0},
                                    "grad scale 1.0 <= 1"),
                                   ("E00_nan_loss", "E00", {"nonfinite_loss_at_step": 4}, "non-finite loss at step 4")):
        r, rr, e, ep = fixture_and_run(tag, group, out / ("runs_" + tag), ov)
        f = json.loads((rr / "detector/failure.json").read_text()) if (rr / "detector/failure.json").exists() else {}
        st = json.loads((rr / "run_status.json").read_text()) if (rr / "run_status.json").exists() else {}
        fc[tag] = {"rc": r["rc"], "error": f.get("error"), "status": st.get("status"),
                   "steps_kept": len((rr / "detector/steps.jsonl").read_text().splitlines())
                   if (rr / "detector/steps.jsonl").exists() else None,
                   "pass": r["rc"] == 1 and needle in str(f.get("error")) and st.get("status") == "FAILED"}
    fc["pass"] = all(v["pass"] for v in fc.values())
    doc["fail_closed"] = fc

    # F4 create-only refusal
    before = sorted((p.relative_to(ra).as_posix(), p.stat().st_mtime_ns) for p in ra.rglob("*"))
    r = run_cli(["--group", "E03", "--seed", "42", "--runtime-root", rt, "--fixture", str(out / "fixtures/E03_a"),
                 "--qualification-out", str(out / "runs_a"), "--qualification-epochs", str(EPOCHS),
                 "--qualification-warmup", str(WARMUP)])
    after = sorted((p.relative_to(ra).as_posix(), p.stat().st_mtime_ns) for p in ra.rglob("*"))
    doc["create_only_refusal"] = {"rc": r["rc"], "unchanged": before == after, "pass": r["rc"] == 2 and before == after}

    # F5 negatives
    doc["negatives"] = negatives(good[0], good[1], out / "negatives")

    # firewall evidence from the real launcher processes
    # the on-disk report is written by the audit hook itself (no 'installed' key); the run manifest records installed
    fws = {t: json.loads((Path(runs[t]["run_root"]) / "firewall_report.json").read_text()) for t in runs}
    inst = {t: json.loads((Path(runs[t]["run_root"]) / "run_manifest.json").read_text())["firewall"].get("installed")
            for t in runs}
    doc["firewall"] = {"reports": fws, "manifest_installed": inst,
                       "decoys_on_disk": json.loads((out / "fixtures/E03_a/fixture.json").read_text())["test_decoys"],
                       "pass": all(inst[t] is True and f.get("faces_root") and f["violation_count"] == 0
                                   and f["listing_attempts"] == 0 and f["opens_under_faces_root"] > 0
                                   and f["distinct_files_opened"] <= f["allowlist_size"] for t, f in fws.items())}

    # F7 resources
    sp = ST.preflight()
    doc["storage_preflight"] = {"ok": sp["ok"], "checks": sp["checks"], "filesystem": sp.get("filesystem"),
                                "unexpected_entries": sp.get("unexpected_entries")}
    det = Path(runs["E03_a"]["run_root"]) / "detector"
    ck = next(det.glob("checkpoint_*.pt")).stat().st_size
    pq_bytes = [p.stat().st_size for p in det.glob("predictions_val_epoch*.parquet")]
    val_rows = len(json.loads((out / "fixtures/E03_a/fixture.json").read_text())["pools"]["val"])
    steps_bytes = (det / "steps.jsonl").stat().st_size
    n_steps = len((det / "steps.jsonl").read_text().splitlines())
    per_run = ck + 40 * (max(pq_bytes) / val_rows * 3121 + 20000) + 7000 * steps_bytes / n_steps + 200000
    ep = json.loads((det / "epochs.json").read_text())
    train_s = [e["wall_seconds"] for e in ep]
    doc["resources"] = {"gpu_free_mib": gpu["gpus"][0]["free_mib"], "gpu_used_mib": gpu["gpus"][0]["used_mib"],
                        "gpu_utilization_pct": gpu["gpus"][0]["utilization_pct"],
                        "hdd_free_bytes": (sp.get("filesystem") or {}).get("free_bytes"),
                        "storage_estimate": {"checkpoint_bytes": ck, "per_run_bytes_estimate": int(per_run),
                                             "matrix_24_bytes_estimate": int(24 * per_run),
                                             "basis": "measured fixture checkpoint (same ResNet-18 state); VAL parquet "
                                                      "scaled per row to 3121 frames; steps.jsonl scaled to 7000 lines"},
                        "throughput_estimate": {"fixture_epoch_wall_seconds": train_s,
                                                "note": "fixture epochs are tiny (3 steps + 32 VAL frames); see the "
                                                        "benchmark for per-step cost"}}
    doc["resources"]["throughput_estimate"].update(benchmark(weights))
    doc["identity"]["gpu_after"] = L.gpu_identity()["gpus"][0]
    if a.dry_preflight:
        doc["dry_preflight"] = dry_preflight(rt, weights, None)
    doc["utc_end"] = L.utc()
    doc["pass"] = all([doc["runs_pass"], doc["audits_pass"], doc["replay"]["bitwise"], doc["skip_recovery"]["pass"],
                       doc["fail_closed"]["pass"], doc["create_only_refusal"]["pass"], doc["negatives"]["pass"],
                       doc["firewall"]["pass"], doc["storage_preflight"]["ok"],
                       doc.get("dry_preflight", {"pass": True})["pass"]])
    (out / "qualification.json").write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n")
    print(json.dumps({"pass": doc["pass"], "out": str(out)}))
    return 0 if doc["pass"] else 1


def benchmark(weights):
    """Provisional per-step cost on the shared GPU: O1 train step at batch 64 (random tensors), O1 eval batch, and the
    CPU decode + augmentation of 64 256x256 PNGs with the launcher's worker count."""
    import io
    import numpy as np
    import torch
    import torch.nn.functional as F
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image
    from gpatbench.downstream import resnet_runner as R
    from gpatbench.downstream import science_runner as SR
    pol = SR.POLICIES["CUDA_AMP_FP16_GRADSCALER_DETERMINISTIC"]
    SR.configure(pol)
    model = R.build_model(weights, 0).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    x = torch.randn(64, 3, 224, 224, device="cuda")
    y = (torch.arange(64, device="cuda") >= 32).float()
    ts = []
    for _ in range(40):
        torch.cuda.synchronize()
        t = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            lo = model(x).squeeze(1)
        loss = F.binary_cross_entropy_with_logits(lo.float(), y)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    peak = torch.cuda.max_memory_allocated() // 2 ** 20
    model.eval()
    with torch.no_grad():
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(10):
            with torch.autocast("cuda", dtype=torch.float16):
                model(x)
        torch.cuda.synchronize()
    ev = (time.perf_counter() - t) / 10
    rng = np.random.Generator(np.random.PCG64(3))
    pngs = []
    for _ in range(64):
        buf = io.BytesIO()
        Image.fromarray(np.clip(rng.normal(128, 40, size=(256, 256, 3)), 0, 255).astype(np.uint8)).save(buf, "PNG")
        pngs.append(buf.getvalue())
    g = torch.Generator().manual_seed(1)

    def one(j):
        with Image.open(io.BytesIO(j[0])) as im:
            img = im.copy()
        hashlib.sha256(j[0]).hexdigest()
        return SR.apply_aug(img, j[1])
    pool = ThreadPoolExecutor(L.WORKERS)
    cs = []
    for _ in range(5):
        jobs = [(r, SR.draw_aug_params(g)) for r in pngs]
        t = time.perf_counter()
        torch.stack(SR._map(pool, one, jobs))
        cs.append(time.perf_counter() - t)
    pool.shutdown()
    step = float(np.median(ts[10:]))
    cpu = float(np.median(cs))
    val = 49 * (ev + cpu)
    per_run = 7000 * (step + cpu) + 40 * val
    return {"gpu_train_step_seconds_median": step, "gpu_peak_allocated_mib": int(peak), "gpu_eval_batch64_seconds": ev,
            "cpu_decode_augment_batch64_seconds": cpu, "val_epoch_seconds_estimate": val,
            "per_run_seconds_estimate": per_run, "matrix_24_hours_estimate": 24 * per_run / 3600,
            "status": "PROVISIONAL (random tensors / noise PNGs on a shared GPU; real throughput is measured by the "
                      "first scientific run's audit)"}


if __name__ == "__main__":
    sys.exit(main())
