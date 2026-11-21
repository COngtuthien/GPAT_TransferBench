#!/usr/bin/env python3
"""E01/E02 bank-writer qualification on a SYNTHETIC fixture (no benchmark image; no production bank).

    python tools/e01_e02_bank_qualification.py --work <empty scratch dir> --out outputs/audit/E01_E02_BANK_WRITER_QUALIFICATION.json

Gates (all must pass):
  G1 authorization: repository has no approved ruling; CLI exits 3 (E01_E02_BLOCKED_BY_MILESTONE_RULING) for both
     methods and seeds and writes nothing;
  G2 frozen TRAIN population (metadata only, empty faces root): 8838 TRAIN pairs, 12668 TRAIN samples, M2 COMPLETE;
  G3 E01 and E02 banks for seeds 42/1337/2026 on a 24-pair fixture (every E01 operator 3x per seed): completed,
     0 failures, file verification, full replay;
  G4 repeatability: an independent second root reproduces every bank manifest byte-for-byte;
  G5 method invariants: E01 round-robin = position mod 8, A2-05 worked seed; E02 k=40, n_eligible=159,
     imaginary residual <= 1e-8, A2-04 worked seed;
  G6 safety: rerun into an existing root refused with outputs unchanged; input corruption aborts the run;
  G7 firewall: audit hook saw 0 opens under any runtime root / faces_256, 0 non-fixture image decodes outside
     the pinned FAS-Aug assets, and only the three frozen manifests (G2) under manifests/.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OPENS: list[str] = []


def _hook(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        OPENS.append(os.fsdecode(args[0]))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    work = Path(a.work).resolve()
    work.mkdir(parents=True, exist_ok=False)
    sys.addaudithook(_hook)

    import numpy as np
    from methods.common import bank_writer as bw
    from methods.common.bank_fixture import build_fixture
    from methods.common.config import load_method_config
    from methods.common.runlog import RunDirectoryError
    import tools.e01_e02_bank as cli

    gates, ev = {}, {}
    t_start = time.time()

    # G1 ----------------------------------------------------------- authorization
    cli_root = work / "cli_runtime_root"
    cli_root.mkdir()
    g1 = []
    for m in bw.METHODS:
        for s in (42, 1337, 2026):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = cli.main(["--method", m, "--seed", str(s), "--runtime-root", str(cli_root)])
            g1.append({"method": m, "seed": s, "exit": rc, "status": json.loads(buf.getvalue())["status"]})
    gates["G1_authorization_blocked"] = (not (ROOT / bw.RULING_PATH).exists() and
                                         all(r["exit"] == 3 and r["status"] == bw.BLOCKED_STATUS for r in g1) and
                                         os.listdir(cli_root) == [])
    ev["G1"] = g1

    # G2 ----------------------------------------------------------- frozen population (metadata only)
    prt = work / "population_runtime_root"
    (prt / "data" / "processed" / "faces_256").mkdir(parents=True)
    pop = {}
    for m in bw.METHODS:
        inp = bw.production_inputs(prt, load_method_config(m))
        pop[m] = {**inp.provenance, "first": inp.pairs[0].pair_id, "last": inp.pairs[-1].pair_id,
                  "datasets": {d: sum(1 for r in inp.pairs if r.dataset == d) for d in sorted({r.dataset for r in inp.pairs})},
                  "distinct_targets": len({r.target_live_id for r in inp.pairs}), "images_read": len(inp.opened)}
    gates["G2_frozen_population"] = all(p["pairs"] == 8838 and p["referenced_samples"] == 12668 and
                                        p["images_read"] == 0 and (p["first"], p["last"]) == ("PTR000001", "PTR008838")
                                        for p in pop.values())
    ev["G2"] = pop

    # G3/G4/G5 ----------------------------------------------------- banks
    inputs = build_fixture(work / "fixture_faces", 24)
    qa = bw.qualification_authorization("m6_e01e02_bank_qualification")
    ops = load_method_config("E01")["generation"]["operator_set"]
    banks, g3, g4, g5 = {}, True, True, True
    for m in bw.METHODS:
        for s in (42, 1337, 2026):
            rec = {}
            for root in ("root_a", "root_b"):
                t = time.time()
                summ = bw.write_bank(m, s, inputs, qa, work / root)
                rd = work / root / "runs" / "m6" / m / f"seed_{s}"
                rec[root] = {"run_id": summ["run_id"], "status": summ["completion_status"],
                             "outputs": summ["output_count"], "failures": summ["failure_count"],
                             "bank_manifest_sha256": summ["bank_manifest_sha256"],
                             "files_ok": bw.verify_bank_files(rd, m)["ok"],
                             "replay_ok": bw.replay_bank(rd, m, s, inputs, qa)["ok"],
                             "seconds": round(time.time() - t, 3)}
                if m == "E01":
                    rec[root].update(operator_counts=summ["operator_counts"], level_zero_noops=summ["level_zero_noop_count"])
                else:
                    rec[root].update(max_abs_imag=summ["max_abs_imag"], k_values=summ["k_values"],
                                     n_eligible_values=summ["n_eligible_values"])
            rows = bw.read_manifest(work / "root_a" / "runs" / "m6" / m / f"seed_{s}")
            ra = rec["root_a"]
            g3 &= ra["status"] == "completed" and ra["outputs"] == 24 and ra["failures"] == 0 and ra["files_ok"] and ra["replay_ok"]
            g4 &= ra["bank_manifest_sha256"] == rec["root_b"]["bank_manifest_sha256"] and ra["run_id"] == rec["root_b"]["run_id"]
            if m == "E01":
                g5 &= all(r["selected_operator_name"] == ops[r["position"] % 8] for r in rows)
                g5 &= ra["operator_counts"] == {o: 3 for o in sorted(ops)}
                if s == 42:
                    g5 &= rows[0]["deterministic_pair_seed"] == "795981663"
            else:
                g5 &= ra["k_values"] == [40] and ra["n_eligible_values"] == [159] and ra["max_abs_imag"] <= 1e-8
                if s == 42:
                    g5 &= rows[0]["deterministic_pair_seed"] == \
                        "19354000309184887456743517179028239418116648094833629759175866349341962868907"
            banks[f"{m}_seed_{s}"] = rec
    gates["G3_banks_complete_verified_replayed"] = bool(g3)
    gates["G4_repeatability_independent_root"] = bool(g4)
    gates["G5_method_invariants"] = bool(g5)
    ev["banks"] = banks

    # G6 ----------------------------------------------------------- safety
    rd = work / "root_a" / "runs" / "m6" / "E02" / "seed_42"
    before = {str(p): sha(p) for p in rd.rglob("*") if p.is_file()}
    try:
        bw.write_bank("E02", 42, inputs, qa, work / "root_a")
        rerun_refused = False
    except RunDirectoryError:
        rerun_refused = True
    unchanged = before == {str(p): sha(p) for p in rd.rglob("*") if p.is_file()}
    bad = build_fixture(work / "fixture_corrupt", 6)
    victim = bad.face_path(bad.pairs[3].source_spoof_id)
    victim.write_bytes(victim.read_bytes()[:-1] + bytes([victim.read_bytes()[-1] ^ 1]))
    try:
        bw.write_bank("E02", 42, bad, qa, work / "root_corrupt")
        aborted = False
    except bw.BankError:
        aborted = True
    cs = json.loads((work / "root_corrupt/runs/m6/E02/seed_42/run_summary.json").read_text())
    gates["G6_collision_and_integrity_fail_closed"] = (rerun_refused and unchanged and aborted and
                                                       cs["completion_status"] == "failed" and cs["rows_before_abort"] == 3)
    ev["G6"] = {"rerun_refused": rerun_refused, "existing_outputs_unchanged": unchanged, "corruption_aborted": aborted,
                "corrupt_run_status": cs["completion_status"], "rows_before_abort": cs["rows_before_abort"]}

    # G7 ----------------------------------------------------------- firewall
    fas_assets = str((ROOT / "third_party" / "source_cache" / "fas_aug").resolve())
    opens = [str(Path(p).resolve()) if not p.startswith("/proc") else p for p in OPENS]
    runtime_opens = [p for p in opens if "GPAT_TransferBench_runtime" in p or "/faces_256/" in p or p.startswith("/media/")]
    images = [p for p in opens if p.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".avi", ".mov", ".mp4"))]
    foreign_images = [p for p in images if not (p.startswith(str(work)) or p.startswith(fas_assets))]
    manifests = sorted({str(Path(p).relative_to(ROOT)) for p in opens if p.startswith(str(ROOT / "manifests"))})
    gates["G7_firewall"] = (not runtime_opens and not foreign_images and
                            manifests == sorted([bw.PAIRS_PATH, bw.SPLIT_PATH, bw.M2_ACCOUNTING_PATH]))
    ev["G7"] = {"open_events": len(opens), "runtime_or_faces_256_opens": runtime_opens[:20],
                "image_opens_total": len(images), "image_opens_outside_fixture_and_pinned_fas_aug_assets": foreign_images[:20],
                "manifests_opened": manifests,
                "test_split_rows_used": 0,
                "note": "split_v1 is read as (sample_id, split, dataset) metadata only to prove every referenced sample is TRAIN"}

    import PIL
    import cv2
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"]).decode().strip()
    dirty = bool(subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"]).decode().strip())
    record = {
        "record": "E01_E02_BANK_WRITER_QUALIFICATION",
        "status": "E01_E02_BANK_WRITER_QUALIFIED_SYNTHETIC" if all(gates.values()) else "E01_E02_BANK_WRITER_QUALIFICATION_FAILED",
        "scope": "synthetic fixture only; no benchmark image decoded; no production bank; not FAS-Aug/FreqSub evidence",
        "source_commit": commit, "worktree_dirty_at_run": dirty,
        "host": platform.node(), "python_executable": sys.executable, "python_version": platform.python_version(),
        "numpy": np.__version__, "pillow": PIL.__version__, "opencv": cv2.__version__,
        "environment_note": ("laptop .venv: NumPy 2.5.3 / Python 3.12.3 differ from the M6D1 gpat-m5 lock "
                             "(NumPy 2.4.6 / Python 3.11.16); this qualifies the writer logic. The production CLI "
                             "refuses any interpreter other than the locked gpat-m5 (verify_locked_environment)."),
        "seed_use_disclosure": ("the generators accept only 42/1337/2026, so the frozen seeds were used on synthetic "
                                "images, exactly as in the M6C1/M6D1 synthetic qualifications; no benchmark data"),
        "fixture": {"pairs": 24, "fixture_seed": inputs.provenance["fixture_seed"], "generator": "PCG64 smooth+noise",
                    "e01_ordinal_rule_used": qa.e01_round_robin_ordinal,
                    "e01_ordinal_rule_note": "qualification choice only; production takes it from the owner ruling (Q2)"},
        "gates": gates, "evidence": ev,
        "file_sha256": {p: sha(ROOT / p) for p in ("methods/common/bank_writer.py", "methods/common/bank_fixture.py",
                                                   "tools/e01_e02_bank.py", "tools/e01_e02_bank_qualification.py",
                                                   "tests/test_e01_e02_bank_writer.py", "methods/fas_aug/e01.py",
                                                   "methods/fas_aug/official.py", "methods/freq_sub/e02.py",
                                                   "configs/methods/e01_fas_aug.yaml", "configs/methods/e02_freqsub.yaml")},
        "wall_seconds": round(time.time() - t_start, 2),
    }
    Path(a.out).write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"status": record["status"], "gates": gates}, indent=2))
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
