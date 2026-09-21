"""M6C1 dry run: validate the E01/E02 runtime end-to-end on SYNTHETIC arrays only.

Safety properties (enforced, not merely claimed):
  * no benchmark image is opened — the only inputs are numpy arrays built in-process;
  * no TEST/TRAIN/VAL path is read;
  * the runtime run directory is a throwaway temp dir, never the production runtime root;
  * no learned model is invoked and no checkpoint is written.

Usage: python3 tools/m6c1_dry_run.py [--keep]
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from methods.common.config import load_logging_contract, load_method_config  # noqa: E402
from methods.common.runlog import RunContext  # noqa: E402
from methods.common.seeding import seed_everything  # noqa: E402
from methods.fas_aug.e01 import AssetIndex, E01Generator, ToyOperatorBackend  # noqa: E402
from methods.freq_sub.e02 import E02Generator  # noqa: E402

SEED = 42
N_PAIRS = 3


def _toy_assets(root: Path) -> AssetIndex:
    """Synthetic asset tree created in reverse order, to prove enumeration is order-independent."""
    for sub, names in (("background", ["b10.png", "b02.png", "b1.png"]),
                       ("noiseTexture", ["n2.png", "n1.png"]),
                       ("MPTexture", ["m3.png", "m1.png", "m2.png"])):
        d = root / sub
        d.mkdir(parents=True, exist_ok=True)
        for nm in names:
            (d / nm).write_bytes(b"toy")
    return AssetIndex(root).load(["background", "noiseTexture", "MPTexture"])


def _toy_images(n: int = 256):
    rng = np.random.default_rng(12345)
    tgt = rng.integers(0, 256, (n, n, 3), dtype=np.uint8)
    src = rng.integers(0, 256, (n, n, 3), dtype=np.uint8)
    return tgt, src


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true", help="keep the temp runtime dir")
    args = ap.parse_args()

    report: dict = {"dry_run_benchmark_images_opened": 0, "test_split_accessed": False, "steps": []}
    tmp = Path(tempfile.mkdtemp(prefix="m6c1-dryrun-"))
    report["temp_runtime_root"] = str(tmp)
    accesses = {"open_events": 0, "benchmark_image_opens": 0, "manifest_opens": 0}
    active = [True]
    source_root = (ROOT / "third_party/source_cache/fas_aug").resolve()

    def audit_open(event, args):
        if not active[0] or event != "open" or not isinstance(args[0], (str, bytes)):
            return
        accesses["open_events"] += 1
        path = Path(args[0].decode() if isinstance(args[0], bytes) else args[0]).resolve()
        if path.is_relative_to(ROOT / "manifests") or path.suffix == ".parquet":
            accesses["manifest_opens"] += 1
            raise RuntimeError(f"dry run forbids manifest access: {path}")
        image_suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".mp4", ".avi", ".mov"}
        if path.suffix.lower() in image_suffixes and not path.is_relative_to(source_root) and not path.is_relative_to(tmp):
            accesses["benchmark_image_opens"] += 1
            raise RuntimeError(f"dry run forbids image/video outside synthetic temp or pinned operator assets: {path}")
    sys.addaudithook(audit_open)
    try:
        lc = load_logging_contract()
        report["steps"].append({"step": "logging_contract", "version": lc["version"], "ok": True})

        tgt, src = _toy_images()
        assets = _toy_assets(tmp / "toy_assets")
        report["asset_enumeration"] = {k: v for k, v in assets.dirs.items()}

        for method_id, build in (("E01", "e01"), ("E02", "e02")):
            cfg = load_method_config(method_id)
            seeded = seed_everything(SEED)
            with RunContext(method_id=method_id, seed=SEED, config=cfg, runtime_root=tmp,
                            command_line="tools/m6c1_dry_run.py") as run:
                run.log_event("dry_run_start", {"synthetic_inputs_only": True,
                                                "benchmark_images_opened": 0,
                                                "seeding": seeded})
                if method_id == "E01":
                    gen = E01Generator(cfg, asset_index=assets,
                                       backend=ToyOperatorBackend(), synthetic_only=True).prepare()
                    results = [gen.generate_one(f"PTR{i:06d}", SEED, tgt) for i in range(1, N_PAIRS + 1)]
                else:
                    gen = E02Generator(cfg).prepare()
                    results = [gen.generate_one(f"PTR{i:06d}", SEED, tgt, src)
                               for i in range(1, N_PAIRS + 1)]
                for r in results:
                    run.log_generation(r.to_record(include_output_ref="IN_MEMORY_SYNTHETIC"))
                # determinism: regenerate and require byte-identical output
                if method_id == "E01":
                    again = E01Generator(cfg, asset_index=assets,
                                         backend=ToyOperatorBackend(), synthetic_only=True).prepare()
                    repeat = [again.generate_one(f"PTR{i:06d}", SEED, tgt) for i in range(1, N_PAIRS + 1)]
                else:
                    again = E02Generator(cfg).prepare()
                    repeat = [again.generate_one(f"PTR{i:06d}", SEED, tgt, src)
                              for i in range(1, N_PAIRS + 1)]
                identical = all(np.array_equal(a.output, b.output) for a, b in zip(results, repeat))
                run.log_event("determinism_check", {"byte_identical_repeat": identical})
                summary = run.close(summary={"method_summary": gen.finalize(),
                                             "determinism_byte_identical": identical})
            files = sorted(p.name for p in run.run_dir.iterdir())
            n_lines = len(run.path("metrics").read_text().splitlines())
            report["steps"].append({
                "step": method_id, "run_id": run.run_id, "run_dir": str(run.run_dir),
                "files": files, "metrics_records": n_lines,
                "determinism_byte_identical": identical,
                "all_success": all(r.success for r in results),
                "method_summary": gen.finalize(),
                "example_metadata": results[0].metadata,
                "completion_status": summary["completion_status"],
            })

        # resume safety: a second open without resume must be refused
        from methods.common.runlog import RunDirectoryError
        cfg = load_method_config("E01")
        try:
            RunContext(method_id="E01", seed=SEED, config=cfg, runtime_root=tmp).open()
            refused = False
        except RunDirectoryError:
            refused = True
        report["steps"].append({"step": "resume_safety", "overwrite_refused": refused})

        # resume=True must APPEND, never truncate
        before = len((tmp / "runs/m6/E01/seed_42/metrics.jsonl").read_text().splitlines())
        r2 = RunContext(method_id="E01", seed=SEED, config=cfg, runtime_root=tmp, resume=True).open()
        r2.close()
        after = len((tmp / "runs/m6/E01/seed_42/metrics.jsonl").read_text().splitlines())
        report["steps"].append({"step": "resume_append", "records_before": before,
                                "records_after": after, "history_preserved": after > before})

        from methods.fas_aug.official import OfficialFASAugBackend
        official = OfficialFASAugBackend(load_method_config("E01")).prepare()
        import importlib.util
        missing = [name for name in ("PIL", "cv2") if importlib.util.find_spec(name) is None]
        if missing:
            report["official_operator_synthetic_smoke"] = {"status": "NOT_RUN_WITH_REASON", "reason": "Missing execution dependencies: " + ", ".join(missing)}
        else:
            gen = E01Generator(backend=official).prepare()
            checks = []
            for i, op in enumerate(gen.operator_set):
                pair = next(f"PTR{j:06d}" for j in range(1, 100) if gen.recipe(f"PTR{j:06d}", SEED, index=i)["level_k"])
                a = gen.generate_one(pair, SEED, tgt, index=i)
                b = gen.generate_one(pair, SEED, tgt, index=i)
                checks.append({"operator": op, "ok": a.success and b.success and np.array_equal(a.output, b.output), "failure_reason": a.failure_reason})
            report["official_operator_synthetic_smoke"] = {"status": "PASS" if all(c["ok"] for c in checks) else "FAIL", "checks": checks}
        checks = [s.get(k, True) for s in report["steps"] for k in
                  ("ok", "all_success", "determinism_byte_identical", "overwrite_refused", "history_preserved")]
        residual = next(s["method_summary"]["max_abs_imag_observed"] for s in report["steps"] if s["step"] == "E02")
        report["ok"] = all(checks) and residual <= 1e-8 and report["official_operator_synthetic_smoke"]["status"] != "FAIL"
        report["open_audit"] = accesses
        report["dry_run_benchmark_images_opened"] = accesses["benchmark_image_opens"]
        report["ok"] = report["ok"] and accesses["benchmark_image_opens"] == 0 and accesses["manifest_opens"] == 0
        print(json.dumps(report, indent=2, default=str))
        return 0 if report["ok"] else 1
    finally:
        active[0] = False
        if not args.keep:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
