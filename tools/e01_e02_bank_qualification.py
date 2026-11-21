#!/usr/bin/env python3
"""E01/E02 bank-writer qualification on a SYNTHETIC fixture (no benchmark image; no production bank).

    python tools/e01_e02_bank_qualification.py --work <empty scratch dir> --out <new record.json> \
        --reference <028b5c1 record of the same environment>

Post-ruling version (owner ruling configs/amendments/e01_e02_bank_milestone_ruling.yaml, 2026-11-22). The
pre-ruling version that produced outputs/audit/E01_E02_BANK_WRITER_QUALIFICATION{,_GPAT_M5}.json is this file at
241321f; those records are not rewritten. Gates (all must pass):
  G1 authorization: the approved ruling (Q1 A_M8_SMOKE_GATED, Q2 MANIFEST_POSITION_ZERO_BASED, Q3
     CONFIRM_CLIP_THEN_RINT, Q4 SEPARATE_CLONE_CPU_CONCURRENT) is present with no smoke-test record; production
     authorization refuses both methods with E01_E02_BLOCKED_BY_M8_SMOKE_GATE; the CLI exits 3 with that status
     for both methods and seeds and writes nothing;
  G2 frozen TRAIN population (metadata only, empty faces root): 8838 TRAIN pairs, 12668 TRAIN samples, M2 COMPLETE;
  G3 E01 and E02 banks for seeds 42/1337/2026 on a 24-pair fixture (every E01 operator 3x per seed): completed,
     0 failures, file verification, full replay;
  G4 repeatability: an independent second root reproduces every bank manifest byte-for-byte;
  G5 method invariants: E01 round-robin = position mod 8, A2-05 worked seed; E02 k=40, n_eligible=159,
     imaginary residual <= 1e-8, A2-04 worked seed;
  G6 safety: rerun into an existing root refused with outputs unchanged; input corruption aborts the run;
  G7 firewall: audit hook saw 0 opens under any runtime root / faces_256, 0 non-fixture image decodes outside
     the pinned FAS-Aug assets, and only the three frozen manifests (G2) under manifests/.
  G8 Q2 exact assignment: on the full frozen population (metadata and pure recipes only, no image) for seeds
     42/1337/2026, operator = operator_set[position % 8] (PTR000001 -> Color_Diversity), never the pair_id suffix,
     no seed rotation, counts 6x1105 + 2x1104, operator_seed = SHA256(pair_id + str(seed)) mod 2^31 recomputed
     independently; fixture banks agree, level-0 outputs equal the live input; the production gate (temporary
     ruling copies in the work dir) refuses PAIR_ID_NUMERIC and option B and accepts only the approved decisions;
  G9 Q3 quantization: every E02 fixture output equals an independent float64 FFT recomputation (source spoof
     magnitude, target live phase, pixel-wise conjugate closure, real part, clip [0,255], numpy.rint, uint8)
     from the recorded block selection and the pair's own source face; edit maps equal the independent mask;
  G10 unchanged mapping: every bank manifest is byte-identical to the --reference (pre-ruling, same environment)
     record, so the ruling and the gate correction changed no output.
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
    ap.add_argument("--reference", required=True, help="pre-ruling qualification record of the same environment")
    a = ap.parse_args()
    if Path(a.out).exists():
        raise SystemExit(f"refusing to overwrite {a.out}")
    reference = json.loads(Path(a.reference).read_text())
    reference_sha256 = sha(a.reference)
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

    # G1 ----------------------------------------------------------- authorization (post-ruling)
    approved = {"Q1_milestone_owner": "A_M8_SMOKE_GATED", "Q2_e01_round_robin_ordinal": "MANIFEST_POSITION_ZERO_BASED",
                "Q3_e02_uint8_quantization": "CONFIRM_CLIP_THEN_RINT", "Q4_execution_host": "SEPARATE_CLONE_CPU_CONCURRENT"}
    ruling = json.loads((ROOT / bw.RULING_PATH).read_text())
    ruling_ok = (ruling.get("status") == "OWNER_APPROVED" and ruling.get("approved") is True and
                 ruling.get("decisions") == approved and ruling.get("smoke_test_record") is None and
                 set(ruling.get("authorizes", {}).values()) == {False})
    gate_status = {}
    for m in bw.METHODS:
        try:
            bw.load_production_authorization(m)
            gate_status[m] = "AUTHORIZED"
        except bw.BankBlocked as exc:
            gate_status[m] = exc.status
    cli_root = work / "cli_runtime_root"
    cli_root.mkdir()
    g1 = []
    for m in bw.METHODS:
        for s in (42, 1337, 2026):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = cli.main(["--method", m, "--seed", str(s), "--runtime-root", str(cli_root)])
            g1.append({"method": m, "seed": s, "exit": rc, "status": json.loads(buf.getvalue())["status"]})
    gates["G1_authorization_m8_smoke_gated"] = (
        ruling_ok and set(gate_status.values()) == {bw.M8_SMOKE_GATE_STATUS} and
        all(r["exit"] == 3 and r["status"] == bw.M8_SMOKE_GATE_STATUS for r in g1) and os.listdir(cli_root) == [])
    ev["G1"] = {"ruling_path": bw.RULING_PATH, "ruling_sha256": sha(ROOT / bw.RULING_PATH),
                "ruling_decisions": ruling.get("decisions"), "ruling_smoke_test_record": ruling.get("smoke_test_record"),
                "production_gate_status": gate_status, "cli": g1, "cli_runtime_root_entries": os.listdir(cli_root)}

    # G2 ----------------------------------------------------------- frozen population (metadata only)
    prt = work / "population_runtime_root"
    (prt / "data" / "processed" / "faces_256").mkdir(parents=True)
    pop, frozen = {}, {}
    for m in bw.METHODS:
        inp = bw.production_inputs(prt, load_method_config(m))
        frozen[m] = inp
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

    # G8 ----------------------------------------------------------- Q2 exact assignment (full population, no image)
    import random as _random
    e01_cfg = load_method_config("E01")
    gen01 = bw.make_generator("E01", e01_cfg)
    full = {}
    g8 = ops == ["Color_Diversity", "Color_Distortion", "Reflection", "BN_Halftone", "Moire_Pattern",
                 "SFC_Halftone", "Hand_Trembling", "Low_Resolution"]
    seq = None
    for s in (42, 1337, 2026):
        counts, lz, suffix_equal, seed_bad, op_bad = {}, 0, 0, 0, 0
        this_seq = []
        for row in frozen["E01"].pairs:
            rec = gen01.recipe(row.pair_id, s, index=row.position)
            this_seq.append(rec["operator"])
            op_bad += rec["operator"] != ops[row.position % 8] or rec["operator_index"] != row.position % 8
            suffix_equal += rec["operator"] == ops[int(row.pair_id[3:]) % 8]
            want = int.from_bytes(hashlib.sha256((row.pair_id + str(s)).encode("utf-8")).digest(), "big") % (2 ** 31)
            seed_bad += rec["operator_seed"] != want
            counts[rec["operator"]] = counts.get(rec["operator"], 0) + 1
            lz += bool(rec["is_level_zero_noop"])
            if rec["is_level_zero_noop"] != (_random.Random(want).randrange(10) == 0):
                seed_bad += 1
        seq = this_seq if seq is None else seq
        full[s] = {"pairs": len(this_seq), "operator_mismatch": op_bad, "pair_id_suffix_agreements": suffix_equal,
                   "operator_seed_mismatch": seed_bad, "operator_counts": counts,
                   "predicted_level_zero_noops": lz, "predicted_level_zero_fraction": lz / len(this_seq),
                   "same_sequence_as_seed_42": this_seq == seq, "first_8": this_seq[:8]}
        g8 &= (op_bad == 0 and suffix_equal == 0 and seed_bad == 0 and this_seq == seq and
               counts == {o: (1105 if i < 6 else 1104) for i, o in enumerate(ops)} and this_seq[:8] == ops)
    g8 &= gen01.recipe("PTR000001", 42, index=0)["operator_seed"] == 795981663
    lz_identity = []
    for s in (42, 1337, 2026):
        rd = work / "root_a" / "runs" / "m6" / "E01" / f"seed_{s}"
        for r, row in zip(bw.read_manifest(rd), inputs.pairs):
            md = json.loads(r["recipe_or_operator_metadata"])
            g8 &= md["operator_index"] == row.position % 8 and r["selected_operator_name"] == ops[row.position % 8]
            if r["is_level_zero_noop"]:
                same = bool(np.array_equal(bw.decode_png((rd / r["synthetic_output_path"]).read_bytes()),
                                           inputs.read(row.target_live_id)[0]))
                lz_identity.append({"seed": s, "pair_id": row.pair_id, "equals_live_input": same})
                g8 &= same
    gate_dir = work / "gate_rulings"
    gate_dir.mkdir()
    smoke = gate_dir / "temporary_smoke_record.json"
    smoke.write_text(json.dumps({"smoke_test_passed": True, "note": "temporary gate probe; not a smoke test"}))
    smoke_ref = {"path": str(smoke), "sha256": sha(smoke)}

    def probe(method, **change):
        doc = json.loads((ROOT / bw.RULING_PATH).read_text())
        doc["smoke_test_record"] = smoke_ref
        doc["decisions"].update(change)
        path = gate_dir / f"probe_{len(list(gate_dir.iterdir()))}.yaml"
        path.write_text(json.dumps(doc))
        try:
            a_ = bw.load_production_authorization(method, path)
            return {"status": "AUTHORIZED", "ordinal": a_.e01_round_robin_ordinal, "option": a_.milestone_option}
        except bw.BankBlocked as exc:
            return {"status": exc.status}
    probes = {"E01_approved": probe("E01"), "E02_approved": probe("E02"),
              "E01_pair_id_numeric": probe("E01", Q2_e01_round_robin_ordinal="PAIR_ID_NUMERIC"),
              "E01_option_B": probe("E01", Q1_milestone_owner="B_M6_EXECUTION_EXTENSION"),
              "E02_option_B": probe("E02", Q1_milestone_owner="B_M6_EXECUTION_EXTENSION"),
              "E02_quantization_other": probe("E02", Q3_e02_uint8_quantization="OTHER")}
    g8 &= (probes["E01_approved"] == {"status": "AUTHORIZED", "ordinal": "MANIFEST_POSITION_ZERO_BASED",
                                      "option": "A_M8_SMOKE_GATED"} and
           probes["E02_approved"]["status"] == "AUTHORIZED" and
           all(probes[k] == {"status": bw.BLOCKED_STATUS} for k in
               ("E01_pair_id_numeric", "E01_option_B", "E02_option_B", "E02_quantization_other")))
    gates["G8_q2_exact_operator_assignment"] = bool(g8)
    ev["G8"] = {"operator_set": ops, "full_population_by_seed": {str(k): v for k, v in full.items()},
                "fixture_level_zero_identity": lz_identity, "production_gate_probes": probes,
                "probe_note": "temporary ruling copies and a temporary smoke record in the work dir test the gate "
                              "logic only; nothing was generated and the repository ruling was not modified"}

    # G9 ----------------------------------------------------------- Q3 independent E02 recomputation
    def independent_e02(t, src, blocks_rc, n=256, b=16):
        sel = np.zeros((n, n), dtype=bool)
        for br, bc in blocks_rc:
            sel[br * b:(br + 1) * b, bc * b:(bc + 1) * b] = True
        yy, xx = np.nonzero(sel)
        edit = sel.copy()
        edit[(n - yy) % n, (n - xx) % n] = True
        real = np.empty((n, n, 3), dtype=np.float64)
        for c in range(3):
            ft = np.fft.fftshift(np.fft.fft2(t[:, :, c].astype(np.float64)))
            fs = np.fft.fftshift(np.fft.fft2(src[:, :, c].astype(np.float64)))
            mag = np.abs(ft)
            mag[edit] = np.abs(fs)[edit]
            real[:, :, c] = np.fft.ifft2(np.fft.ifftshift(mag * np.exp(1j * np.angle(ft)))).real
        return real, edit
    g9, g9_rows = True, {"rows": 0, "pixel_mismatch_rows": 0, "edit_map_mismatch_rows": 0,
                         "source_face_mismatch_rows": 0, "pixels_where_floor_differs": 0,
                         "pixels_where_trunc_after_clip_differs": 0, "pixels_clipped": 0}
    for s in (42, 1337, 2026):
        rd = work / "root_a" / "runs" / "m6" / "E02" / f"seed_{s}"
        for r, row in zip(bw.read_manifest(rd), inputs.pairs):
            md = json.loads(r["recipe_or_operator_metadata"])
            t, _ = inputs.read(row.target_live_id)
            src, src_sha = inputs.read(row.source_spoof_id)
            real, edit = independent_e02(t, src, md["selected_blocks_rc"])
            clipped = np.clip(real, 0.0, 255.0)
            want = np.rint(clipped).astype(np.uint8)
            got = bw.decode_png((rd / r["synthetic_output_path"]).read_bytes())
            em = bw.decode_png((rd / r["edit_map_path"]).read_bytes())
            g9_rows["rows"] += 1
            g9_rows["pixel_mismatch_rows"] += not np.array_equal(got, want)
            g9_rows["edit_map_mismatch_rows"] += not np.array_equal(em > 0, edit)
            g9_rows["source_face_mismatch_rows"] += r["source_face_sha256"] != src_sha or not r["source_spoof_used"]
            g9_rows["pixels_where_floor_differs"] += int((np.floor(clipped).astype(np.uint8) != want).sum())
            g9_rows["pixels_where_trunc_after_clip_differs"] += int((clipped.astype(np.uint8) != want).sum())
            g9_rows["pixels_clipped"] += int(((real < 0) | (real > 255)).sum())
    g9 = (g9_rows["rows"] == 72 and g9_rows["pixel_mismatch_rows"] == 0 and g9_rows["edit_map_mismatch_rows"] == 0
          and g9_rows["source_face_mismatch_rows"] == 0 and g9_rows["pixels_where_floor_differs"] > 0)
    gates["G9_q3_e02_quantization_independent"] = bool(g9)
    ev["G9"] = {**g9_rows, "rule": "real part -> clip [0,255] -> numpy.rint (half to even) -> uint8",
                "note": "floor/trunc counts show that the rounding rule decides output bytes; it is the M6D1 "
                        "implementation-defined rule confirmed by owner ruling Q3"}

    # G10 ---------------------------------------------------------- mapping unchanged vs pre-ruling record
    g10 = {k: {"now": v["root_a"]["bank_manifest_sha256"],
               "reference": reference["evidence"]["banks"][k]["root_a"]["bank_manifest_sha256"]}
           for k, v in banks.items()}
    gates["G10_bank_manifests_equal_pre_ruling_reference"] = all(v["now"] == v["reference"] for v in g10.values())
    ev["G10"] = {"reference_path": a.reference, "reference_sha256": reference_sha256,
                 "reference_source_commit": reference["source_commit"], "reference_python": reference["python_version"],
                 "reference_numpy": reference["numpy"], "reference_host": reference["host"], "banks": g10}

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
        "record": "E01_E02_BANK_WRITER_QUALIFICATION_POST_RULING",
        "status": ("E01_E02_BANK_WRITER_QUALIFIED_POST_RULING_SYNTHETIC" if all(gates.values())
                   else "E01_E02_BANK_WRITER_POST_RULING_QUALIFICATION_FAILED"),
        "scope": ("synthetic fixture only plus frozen-manifest metadata and pure recipes; no benchmark image decoded; "
                  "no production bank; not FAS-Aug/FreqSub evidence"),
        "owner_ruling": {"path": bw.RULING_PATH, "sha256": sha(ROOT / bw.RULING_PATH)},
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
                    "e01_ordinal_rule_note": "equals the owner-approved Q2 rule; production accepts only that rule"},
        "gates": gates, "evidence": ev,
        "file_sha256": {p: sha(ROOT / p) for p in (bw.RULING_PATH, "methods/common/bank_writer.py",
                                                   "methods/common/bank_fixture.py",
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
