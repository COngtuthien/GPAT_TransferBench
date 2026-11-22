#!/usr/bin/env python3
"""Independent completion audit of one E01/E02 production bank.

  collect (GPU host, locked gpat-m5, CPU-only, nice/ionice; writes only --out; the tool is fed on stdin and imports
  the writer from the production clone given by --repo):
    CUDA_VISIBLE_DEVICES= <gpat-m5 python> -I -B - collect --repo <production clone> --tool-sha256 <sha of this file> \
        --method E01 --seed 42 --runtime-root <rt> --evidence-dir <evidence>/E01_s42 \
        --out <evidence>/E01_s42/audit_collected.json < tools/e01_e02_production_bank_audit.py
  build (laptop; refuses to overwrite):
    python tools/e01_e02_production_bank_audit.py build --collected <json> --out-prefix outputs/audit/E01_PRODUCTION_BANK_SEED42

collect gathers facts without trusting the writer's own verification. It re-reads the frozen manifests (SHA-checked,
with a TRAIN/VAL row filter so no TEST row is parsed) and re-hashes every output and edit map from disk. It recomputes
E01 operator assignment (position % 8) and operator seeds, plus E02 pair seeds, band geometry, k, PCG64 block
selection, edit maps and the full float64 FFT substitution (clip -> numpy.rint). These are separate implementations,
not the writer's functions. It replays the whole bank once more in this separate process, and checks E01 level-zero
no-op outputs against their target faces. Every face read is O_NOATIME with an atime stat check, behind a TRAIN
allowlist audit hook. It classifies the before/after atime snapshots by split and records the E04 state.
build evaluates every gate from the collected facts against the committed production authority, writes
<prefix>.json/.md and prints the status. A bank passes only if every gate passes.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else None   # stdin: set by --repo

N_PAIRS = 8838
LAUNCH_COMMIT = "0ca57e3bcd77e1a0c54004c7fd4c21e5fe101822"
CANDIDATE_COMMIT = "d995f556a8adcc5a032ba266a022b470117d1c1c"
E04_CHECKOUT = "/home/student20261/workdir/GPAT_TransferBench"
E04_AUTHORITY = "18ba00e94813f005876c8d6235b0b848226b2c1d"
E04_RUN_ID = "e7100eb9a5ab0537"
PAIRS_SHA = "a5e4fdaef236f15730c7e3885b537e08e995faffbe654167fc940f44bbc75243"
SPLIT_SHA = "fb9aeb369a124fc96ba855ef2ce269236c4a743fe960e73ab739412c9cb5092d"
M2_SHA = "65bb83e69ea7cebcb37373971e5e8f5248609b7963fd9d32aa1c0ba4377c0662"
E02_NAMESPACE = "gpatbench.freqsub.block.v1"
RUN_DIR_FILES = {"bank", "bank_manifest.parquet", "bank_manifest.sha256", "checkpoint_index.json", "checkpoints",
                 "generation_log.jsonl", "metrics.jsonl", "resolved_config.yaml", "run_manifest.json", "run_summary.json",
                 "stderr.log", "stdout.log"}


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ===================================================================== collect
class NoAtimeFaces:
    def __init__(self, faces_root: Path, dataset: dict, face_sha: dict):
        self.root, self.dataset, self.face_sha = faces_root, dataset, face_sha
        self.reads = 0
        self.atime_changed = 0

    def read(self, sid):
        import numpy as np
        from PIL import Image
        p = self.root / self.dataset[sid] / f"{sid}.png"
        a0 = os.stat(p).st_atime_ns
        fd = os.open(p, os.O_RDONLY | os.O_NOATIME)
        try:
            chunks = []
            while True:
                b = os.read(fd, 1 << 20)
                if not b:
                    break
                chunks.append(b)
        finally:
            os.close(fd)
        a1 = os.stat(p).st_atime_ns
        self.reads += 1
        if a0 != a1:
            self.atime_changed += 1
            raise RuntimeError(f"atime changed reading {sid}")
        raw = b"".join(chunks)
        if sha_bytes(raw) != self.face_sha[sid]:
            raise RuntimeError(f"face sha mismatch {sid}")
        with Image.open(io.BytesIO(raw)) as im:
            assert im.mode == "RGB" and im.size == (256, 256)
            return np.array(im, dtype=np.uint8)


def pixel_identity(arr) -> str:
    import numpy as np
    a = np.ascontiguousarray(arr)
    return sha_bytes(f"{a.dtype.str}|{'x'.join(map(str, a.shape))}|".encode() + a.tobytes())


def e02_geometry(n=256, b=16, lo=0.20, hi=0.50, frac=0.75):
    """Independent band geometry: normalised radius about the fftshift DC at n//2; block eligible iff >= 75% in band."""
    import numpy as np
    idx = (np.arange(n) - n // 2) / n
    r = np.hypot(idx[:, None], idx[None, :])
    band = (r >= lo) & (r <= hi)
    elig = []
    for br in range(n // b):
        for bc in range(n // b):
            if band[br * b:(br + 1) * b, bc * b:(bc + 1) * b].mean() >= frac:
                elig.append((br, bc))
    return elig


def e02_recompute(t, s, blocks, n=256, b=16):
    import numpy as np
    sel = np.zeros((n, n), dtype=bool)
    for br, bc in blocks:
        sel[br * b:(br + 1) * b, bc * b:(bc + 1) * b] = True
    yy, xx = np.nonzero(sel)
    edit = sel.copy()
    edit[(n - yy) % n, (n - xx) % n] = True
    out = np.empty((n, n, 3), dtype=np.float64)
    imag = 0.0
    for c in range(3):
        ft = np.fft.fftshift(np.fft.fft2(t[:, :, c].astype(np.float64)))
        fs = np.fft.fftshift(np.fft.fft2(s[:, :, c].astype(np.float64)))
        mag = np.abs(ft)
        mag[edit] = np.abs(fs)[edit]
        z = np.fft.ifft2(np.fft.ifftshift(mag * np.exp(1j * np.angle(ft))))
        imag = max(imag, float(np.abs(z.imag).max()))
        out[:, :, c] = z.real
    return np.rint(np.clip(out, 0.0, 255.0)).astype(np.uint8), edit, imag


def read_snapshot(path: Path) -> dict:
    out = {}
    for line in path.read_text().splitlines():
        a, m, p = line.split("\t", 2)
        out[p] = (a, m)
    return out


def collect(a) -> int:
    import numpy as np
    import pyarrow.parquet as pq
    from PIL import Image
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    m, seed = a.method, a.seed
    rt, ev = Path(a.runtime_root), Path(a.evidence_dir)
    out_path = Path(a.out)
    if out_path.exists():
        raise SystemExit(f"refusing to overwrite {out_path}")
    rd = rt / "runs" / "m6" / m / f"seed_{seed}"
    faces = rt / "data" / "processed" / "faces_256"
    facts: dict = {"method": m, "seed": seed, "run_dir": str(rd), "tool_sha256": a.tool_sha256, "audit_git_commit":
                   subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
                   "audit_git_status": subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"], capture_output=True,
                                                      text=True).stdout}
    # -- process evidence
    proc = {k: (ev / k).read_text().strip() if (ev / k).is_file() else None
            for k in ("exit", "launch_time", "end_time", "launch_head", "launch_git_status", "atime_pre.time",
                      "atime_post.time")}
    lines = (ev / "stdout.log").read_text().splitlines()
    proc["stdout_progress_lines"] = sum(1 for x in lines if x.startswith(f"{m} s{seed} "))
    proc["stdout_last_progress"] = next((x for x in reversed(lines) if x.startswith(f"{m} s{seed} ")), None)
    try:
        proc["cli_result"] = json.loads(lines[-1])
    except (IndexError, ValueError):
        proc["cli_result"] = None
    proc["stderr_tail"] = (ev / "stderr.log").read_text()[-3000:]
    proc["tmux_or_process_alive"] = subprocess.run(["bash", "-c", f"ps -eo args | grep -q '[e]01_e02_bank.py --method {m} --seed {seed} '"]).returncode == 0
    facts["process"] = proc
    # -- run dir metadata
    facts["run_manifest"] = json.loads((rd / "run_manifest.json").read_text())
    summ = json.loads((rd / "run_summary.json").read_text())
    facts["run_summary"] = {k: summ.get(k) for k in ("completion_status", "run_id", "input_count", "output_count",
                                                       "failure_count", "failures", "bank_manifest_sha256",
                                                       "authorization", "environment_versions", "environment_lock",
                                                       "operator_counts", "level_zero_noop_count", "max_abs_imag",
                                                       "k_values", "n_eligible_values", "duplicate_output_pixel_groups",
                                                       "duplicate_output_pixel_rows", "generation_wall_time_seconds",
                                                       "post_write_integrity", "git_commit", "config_sha256")}
    facts["run_summary_raw_keys"] = sorted(summ)
    facts["run_dir_listing"] = sorted(os.listdir(rd))
    facts["checkpoints_dir_listing"] = sorted(os.listdir(rd / "checkpoints"))
    from methods.common.bank_writer import filesystem_identity
    facts["filesystem"] = filesystem_identity(rd)
    du = 0
    for dp, _, fs in os.walk(rd):
        for f in fs:
            du += os.lstat(os.path.join(dp, f)).st_blocks * 512
    facts["run_dir_disk_bytes"] = du
    # -- frozen inputs, independently (TRAIN/VAL row filter: no TEST row is parsed)
    frozen = {}
    for rel, want in (("manifests/pairs_train_v1.parquet", PAIRS_SHA), ("manifests/split_v1.parquet", SPLIT_SHA),
                      ("manifests/m2_sample_accounting.parquet", M2_SHA)):
        frozen[rel] = sha_file(ROOT / rel) == want
    facts["frozen_manifest_sha_ok"] = frozen
    pairs = pq.read_table(ROOT / "manifests/pairs_train_v1.parquet",
                          columns=["pair_id", "target_live_id", "source_spoof_id", "dataset", "split"]).to_pylist()
    split_rows = pq.read_table(ROOT / "manifests/split_v1.parquet", columns=["sample_id", "split", "dataset"],
                               filters=[("split", "in", ["TRAIN", "VAL"])]).to_pylist()
    train = {r["sample_id"]: r["dataset"] for r in split_rows if r["split"] == "TRAIN"}
    val = {r["sample_id"]: r["dataset"] for r in split_rows if r["split"] == "VAL"}
    facts["split_rows_read"] = {"TRAIN": len(train), "VAL": len(val), "TEST": 0,
                                "splits_seen": sorted({r["split"] for r in split_rows})}
    referenced = sorted({p["target_live_id"] for p in pairs} | {p["source_spoof_id"] for p in pairs})
    m2 = pq.read_table(ROOT / "manifests/m2_sample_accounting.parquet",
                       columns=["sample_id", "face_png_sha256", "final_status"],
                       filters=[("sample_id", "in", referenced)]).to_pylist()
    face_sha = {r["sample_id"]: r["face_png_sha256"] for r in m2 if r["final_status"] == "COMPLETE"}
    # -- firewall for this audit process: only referenced TRAIN faces may be opened; no listing of faces_256
    from gpatbench.downstream import audit_hook
    allow = [f"{train[s]}/{s}.png" for s in referenced if s in train]
    audit_hook.install(faces, allow, os.devnull)
    reader = NoAtimeFaces(faces, train, face_sha)
    # -- manifest and files
    man_path = rd / "bank_manifest.parquet"
    rows = pq.read_table(man_path).to_pylist()
    facts["manifest"] = {"sha256": sha_file(man_path), "sha256_file": (rd / "bank_manifest.sha256").read_text().split()[0],
                         "rows": len(rows), "columns": pq.read_schema(man_path).names}
    ops = None
    if m == "E01":
        import yaml
        ops = yaml.safe_load((ROOT / "configs/methods/e01_fas_aug.yaml").read_text())["generation"]["operator_set"]
    geometry = e02_geometry() if m == "E02" else None
    cnt = {k: 0 for k in ("population_mismatch", "non_train_sample", "face_sha_mismatch", "status_not_success",
                          "output_sha_mismatch", "pixel_sha_mismatch", "decode_bad", "file_mode_not_0444",
                          "editmap_sha_mismatch", "operator_mismatch", "operator_seed_mismatch", "metadata_identity_mismatch",
                          "level_out_of_range", "level_zero_noop_rows", "level_zero_noop_not_identical",
                          "level_zero_rows", "outputs_identical_to_target", "pair_seed_mismatch", "n_eligible_not_159",
                          "k_not_40", "selection_mismatch", "blocks_mismatch", "editmap_mismatch", "edited_pixels_mismatch",
                          "independent_pixels_mismatch", "imag_residual_over_1e-8", "recorded_imag_over_1e-8")}
    op_counts, level_hist = {}, {}
    digest = hashlib.sha256()
    for i, (r, p) in enumerate(zip(rows, pairs)):
        if (r["position"], r["pair_id"], r["target_live_id"], r["source_spoof_id"], r["dataset"]) != \
                (i, p["pair_id"], p["target_live_id"], p["source_spoof_id"], p["dataset"]) or p["split"] != "TRAIN":
            cnt["population_mismatch"] += 1
        for sid in (r["target_live_id"], r["source_spoof_id"]):
            if sid not in train:
                cnt["non_train_sample"] += 1
        if r["target_face_sha256"] != face_sha.get(r["target_live_id"]) or \
                (m == "E02" and r["source_face_sha256"] != face_sha.get(r["source_spoof_id"])) or \
                (m == "E01" and (r["source_face_sha256"] is not None or r["source_spoof_used"])):
            cnt["face_sha_mismatch"] += 1
        if r["generation_status"] != "success":
            cnt["status_not_success"] += 1
            continue
        fp = rd / r["synthetic_output_path"]
        raw = fp.read_bytes()
        digest.update(f"{r['pair_id']}\t{sha_bytes(raw)}\n".encode())
        if sha_bytes(raw) != r["output_sha256"]:
            cnt["output_sha_mismatch"] += 1
        if (os.stat(fp).st_mode & 0o777) != 0o444:
            cnt["file_mode_not_0444"] += 1
        with Image.open(io.BytesIO(raw)) as im:
            ok = im.format == "PNG" and im.mode == "RGB" and im.size == (256, 256)
            arr = np.array(im)
        if not ok or arr.dtype != np.uint8 or arr.shape != (256, 256, 3):
            cnt["decode_bad"] += 1
        if pixel_identity(arr) != r["output_pixel_sha256"]:
            cnt["pixel_sha_mismatch"] += 1
        md = json.loads(r["recipe_or_operator_metadata"])
        if md.get("pair_id") != r["pair_id"] or md.get("global_seed") != seed or r["experiment_seed"] != seed \
                or r["method_id"] != m:
            cnt["metadata_identity_mismatch"] += 1
        if m == "E01":
            want_op = ops[i % 8]
            if r["selected_operator_name"] != want_op or md.get("operator") != want_op or md.get("operator_index") != i % 8:
                cnt["operator_mismatch"] += 1
            want_seed = int.from_bytes(hashlib.sha256((r["pair_id"] + str(seed)).encode("utf-8")).digest(), "big") % (2 ** 31)
            if md.get("operator_seed") != want_seed or r["deterministic_pair_seed"] != str(want_seed):
                cnt["operator_seed_mismatch"] += 1
            lk = md.get("level_k")
            if not isinstance(lk, int) or not 0 <= lk <= 9:
                cnt["level_out_of_range"] += 1
            level_hist[lk] = level_hist.get(lk, 0) + 1
            op_counts[want_op] = op_counts.get(want_op, 0) + 1
            cnt["level_zero_rows"] += lk == 0
            if lk == 0 or md.get("is_level_zero_noop"):
                tgt = reader.read(r["target_live_id"])
                same = np.array_equal(tgt, arr)
                cnt["outputs_identical_to_target"] += same
                if md.get("is_level_zero_noop"):
                    cnt["level_zero_noop_rows"] += 1
                    cnt["level_zero_noop_not_identical"] += not same
        else:
            em_p = rd / r["edit_map_path"]
            em_raw = em_p.read_bytes()
            if sha_bytes(em_raw) != r["edit_map_sha256"]:
                cnt["editmap_sha_mismatch"] += 1
            if (os.stat(em_p).st_mode & 0o777) != 0o444:
                cnt["file_mode_not_0444"] += 1
            want_seed = int.from_bytes(hashlib.sha256(f"{E02_NAMESPACE}|{r['pair_id']}|{seed}".encode()).digest(), "big")
            if str(want_seed) != md.get("pair_seed") or r["deterministic_pair_seed"] != str(want_seed):
                cnt["pair_seed_mismatch"] += 1
            if md.get("n_eligible") != 159 or r["n_eligible"] != 159 or len(geometry) != 159:
                cnt["n_eligible_not_159"] += 1
            if md.get("k") != 40 or r["k_selected"] != 40:
                cnt["k_not_40"] += 1
            chosen = sorted(np.random.Generator(np.random.PCG64(want_seed)).choice(159, size=40, replace=False).tolist())
            if chosen != md.get("selected_indices") or json.dumps(chosen) != r["selected_block_indices_sorted_ascending"]:
                cnt["selection_mismatch"] += 1
            blocks = [list(geometry[j]) for j in chosen]
            if blocks != md.get("selected_blocks_rc"):
                cnt["blocks_mismatch"] += 1
            t = reader.read(r["target_live_id"])
            s = reader.read(r["source_spoof_id"])
            want_px, edit, imag = e02_recompute(t, s, blocks)
            with Image.open(io.BytesIO(em_raw)) as im:
                em = np.array(im)
            if not np.array_equal(em > 0, edit) or set(np.unique(em).tolist()) - {0, 255}:
                cnt["editmap_mismatch"] += 1
            if int(edit.sum()) != md.get("edited_pixels"):
                cnt["edited_pixels_mismatch"] += 1
            if not np.array_equal(want_px, arr):
                cnt["independent_pixels_mismatch"] += 1
            cnt["imag_residual_over_1e-8"] += imag > 1e-8
            cnt["recorded_imag_over_1e-8"] += not (md.get("max_abs_imag", 1) <= 1e-8 and r["max_abs_imag"] <= 1e-8)
    facts["row_checks"] = cnt
    facts["e01_operator_counts"] = op_counts
    facts["e01_level_histogram"] = {str(k): v for k, v in sorted(level_hist.items(), key=lambda kv: str(kv[0]))}
    facts["pair_output_sha_digest"] = digest.hexdigest()
    bank_files = sorted(os.listdir(rd / "bank"))
    facts["bank_files"] = {"count": len(bank_files),
                           "set_equal": set(bank_files) == {Path(r["synthetic_output_path"]).name for r in rows
                                                            if r["generation_status"] == "success"}}
    if m == "E02":
        em_files = sorted(os.listdir(rd / "edit_maps"))
        facts["edit_map_files"] = {"count": len(em_files),
                                   "set_equal": set(em_files) == {Path(r["edit_map_path"]).name for r in rows
                                                                  if r["generation_status"] == "success"}}
    facts["duplicates"] = {"pair_id": len(rows) - len({r["pair_id"] for r in rows}),
                           "output_path": len(rows) - len({r["synthetic_output_path"] for r in rows})}
    # -- independent-process full replay (writer generators, fresh inputs, O_NOATIME)
    from methods.common import bank_writer as bw
    from methods.common.config import load_method_config
    cfg = load_method_config(m)
    inputs = bw.production_inputs(rt, cfg)
    auth = bw.load_production_authorization(m, seed=seed)
    rep = bw.replay_bank(rd, m, seed, inputs, auth, config=cfg)
    facts["audit_replay"] = {"ok": rep["ok"], "rows_replayed": rep["rows_replayed"], "mismatches": rep["mismatches"],
                             "o_noatime_reads": inputs.atime_checked_reads}
    facts["audit_reader"] = {"o_noatime_reads": reader.reads, "atime_changed": reader.atime_changed}
    facts["audit_firewall"] = audit_hook.report()
    # -- atime snapshots
    pre, post = read_snapshot(ev / "atime_pre.tsv"), read_snapshot(ev / "atime_post.tsv")
    allow_set = {f"faces_256/{x}" for x in allow}
    val_set = {f"faces_256/{d}/{s}.png" for s, d in val.items()}
    train_set = {f"faces_256/{d}/{s}.png" for s, d in train.items()}
    changed_a = sorted(p for p in pre if p in post and pre[p][0] != post[p][0])
    changed_m = sorted(p for p in pre if p in post and pre[p][1] != post[p][1])
    facts["atime"] = {"pre_files": len(pre), "post_files": len(post), "same_file_set": set(pre) == set(post),
                      "atime_changed_total": len(changed_a), "mtime_changed_total": len(changed_m),
                      "atime_changed_val": sum(p in val_set for p in changed_a),
                      "atime_changed_train_allowlisted": sum(p in allow_set for p in changed_a),
                      "atime_changed_train_other": sum(p in train_set and p not in allow_set for p in changed_a),
                      "atime_changed_other_test_by_elimination": sum(p not in train_set and p not in val_set for p in changed_a),
                      "val_files": sum(p in val_set for p in pre), "train_files": sum(p in train_set for p in pre),
                      "other_files_test_by_elimination": sum(p not in train_set and p not in val_set for p in pre),
                      "changed_examples": changed_a[:20]}
    # -- E04 state (read-only)
    e04 = {"checkout_head": subprocess.run(["git", "-C", E04_CHECKOUT, "rev-parse", "HEAD"], capture_output=True,
                                           text=True).stdout.strip(),
           "checkout_status": subprocess.run(["git", "-C", E04_CHECKOUT, "status", "--porcelain"], capture_output=True,
                                             text=True).stdout,
           "process": subprocess.run(["bash", "-c", "ps -eo pid,etimes,args | grep '[p]ython.*tools/run_e04.py --seed 2026' || true"],
                                     capture_output=True, text=True).stdout.strip(),
           "exit_marker": (Path.home() / "workdir/e04_s2026.exit").read_text().strip()
           if (Path.home() / "workdir/e04_s2026.exit").is_file() else None}
    ck = rt / "runs" / "m6" / "E04" / "seed_2026" / "checkpoints"
    e04["latest_checkpoint"] = sorted(os.listdir(ck))[-1] if ck.is_dir() and os.listdir(ck) else None
    e04["run_manifest_run_id"] = json.loads((rt / "runs/m6/E04/seed_2026/run_manifest.json").read_text()).get("run_id")
    facts["e04"] = e04
    out_path.write_text(json.dumps(facts, indent=1, sort_keys=True, default=str) + "\n")
    print(json.dumps({"collected": str(out_path), "replay_ok": rep["ok"]}))
    return 0


# ===================================================================== build
def build(a) -> int:
    from methods.common import bank_writer as bw
    from methods.common.runlog import compute_run_id
    f = json.loads(Path(a.collected).read_text())
    prefix = Path(a.out_prefix)
    for suffix in (".json", ".md"):
        if prefix.with_suffix(suffix).exists():
            raise SystemExit(f"refusing to overwrite {prefix.with_suffix(suffix)}")
    auth_rec = json.loads((ROOT / bw.PRODUCTION_AUTHORITY_PATH).read_text())
    m, seed = f["method"], f["seed"]
    cfg_sha = bw.APPROVED_METHOD_CONFIGS[m][1]
    p, rm, rs, rc = f["process"], f["run_manifest"], f["run_summary"], f["row_checks"]
    cli = p.get("cli_result") or {}
    fw_cli = cli.get("firewall") or {}
    ledger = cli.get("atime_ledger") or {}
    reads_per_pass = N_PAIRS if m == "E01" else 2 * N_PAIRS
    g = {}
    # A. process and source authority
    g["A.exit_zero"] = p["exit"] == "0"
    g["A.cli_completed"] = cli.get("status") == "completed"
    g["A.progress_complete"] = p["stdout_last_progress"] == f"{m} s{seed} {N_PAIRS}/{N_PAIRS}"
    g["A.launch_head_is_launch_authority"] = p["launch_head"] == LAUNCH_COMMIT
    g["A.launch_worktree_clean"] = p["launch_git_status"] == ""
    g["A.process_exited"] = p["tmux_or_process_alive"] is False
    g["A.time_order"] = (p["atime_pre.time"] <= p["launch_time"] <= p["end_time"] <= p["atime_post.time"])
    g["A.authority_candidate_ancestor"] = subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor",
                                                          CANDIDATE_COMMIT, LAUNCH_COMMIT]).returncode == 0
    g["A.launch_gates_passed"] = (cli.get("gates") or {}).get("status") == "GATES_PASS" and \
        (cli["gates"]["launch"]["head_commit"], cli["gates"]["launch"]["candidate_commit"]) == (LAUNCH_COMMIT, CANDIDATE_COMMIT)
    # B. run metadata, identity and environment
    g["B.run_manifest_commit_clean"] = rm.get("git_commit") == LAUNCH_COMMIT and rm.get("git_dirty") is False
    g["B.config_identity"] = rm.get("config_sha256") == cfg_sha
    g["B.run_id_recomputed"] = rs["run_id"] == rm.get("run_id") == compute_run_id(m, seed, cfg_sha, LAUNCH_COMMIT) \
        == cli.get("run_id")
    g["B.summary_completed"] = (rs["completion_status"], rs["input_count"], rs["output_count"], rs["failure_count"]) == \
        ("completed", N_PAIRS, N_PAIRS, 0)
    authz = rs["authorization"] or {}
    g["B.production_authorization_bound"] = authz.get("mode") == "PRODUCTION" and \
        (authz.get("binding") or {}).get("binding_sha256") == auth_rec["binding_sha256"] and \
        (authz.get("binding") or {}).get("smoke_record_sha256") == bw.APPROVED_SMOKE_RECORD_SHA256 and \
        authz.get("ruling_sha256") == bw.APPROVED_RULING_SHA256
    g["B.locked_environment"] = all(rs["environment_versions"].get(k) == v for k, v in bw.LOCKED_ENVIRONMENT.items()) and \
        (rs["environment_lock"] or {}).get("verified") is True
    g["B.run_dir_listing_exact"] = set(f["run_dir_listing"]) == (RUN_DIR_FILES | ({"edit_maps"} if m == "E02" else set())) \
        and f["checkpoints_dir_listing"] == []
    # C. population and TEST isolation
    g["C.frozen_manifests_sha"] = all(f["frozen_manifest_sha_ok"].values())
    g["C.rows_8838"] = f["manifest"]["rows"] == N_PAIRS
    g["C.population_equals_frozen_order"] = rc["population_mismatch"] == 0
    g["C.all_samples_train"] = rc["non_train_sample"] == 0
    g["C.no_duplicates"] = f["duplicates"] == {"pair_id": 0, "output_path": 0}
    g["C.all_success"] = rc["status_not_success"] == 0
    g["C.no_test_rows_parsed"] = f["split_rows_read"]["TEST"] == 0 and f["split_rows_read"]["splits_seen"] == ["TRAIN", "VAL"]
    # D. files and hashes
    g["D.manifest_sha_matches"] = f["manifest"]["sha256"] == f["manifest"]["sha256_file"] == rs["bank_manifest_sha256"] \
        == cli.get("bank_manifest_sha256")
    g["D.bank_file_set_exact"] = f["bank_files"] == {"count": N_PAIRS, "set_equal": True}
    g["D.output_sha_and_pixels"] = rc["output_sha_mismatch"] == rc["pixel_sha_mismatch"] == rc["decode_bad"] == 0
    g["D.face_lineage_sha"] = rc["face_sha_mismatch"] == 0
    g["D.read_only_files"] = rc["file_mode_not_0444"] == 0
    g["D.metadata_identity"] = rc["metadata_identity_mismatch"] == 0
    if m == "E02":
        g["D.edit_map_file_set_exact"] = f["edit_map_files"] == {"count": N_PAIRS, "set_equal": True}
        g["D.edit_map_sha"] = rc["editmap_sha_mismatch"] == 0
    # E. frozen transformation semantics
    if m == "E01":
        ops = json.loads(json.dumps(f["e01_operator_counts"]))
        g["E.operator_is_position_mod_8"] = rc["operator_mismatch"] == 0
        g["E.operator_counts_1105_1104"] = sorted(ops.values()) == [1104, 1104, 1105, 1105, 1105, 1105, 1105, 1105] and \
            ops == rs["operator_counts"]
        g["E.operator_seed_sha256_mod_2_31"] = rc["operator_seed_mismatch"] == 0
        g["E.levels_in_grid_and_level0_retained"] = rc["level_out_of_range"] == 0 and rc["level_zero_rows"] > 0
        g["E.level_zero_noop_outputs_equal_target"] = rc["level_zero_noop_not_identical"] == 0 and \
            rc["level_zero_noop_rows"] == rs["level_zero_noop_count"]
    else:
        g["E.pair_seed_namespace_sha256"] = rc["pair_seed_mismatch"] == 0
        g["E.n_eligible_159_k_40"] = rc["n_eligible_not_159"] == rc["k_not_40"] == 0 and rs["k_values"] == [40] and \
            rs["n_eligible_values"] == [159]
        g["E.pcg64_selection_and_blocks"] = rc["selection_mismatch"] == rc["blocks_mismatch"] == 0
        g["E.edit_maps_conjugate_closed"] = rc["editmap_mismatch"] == rc["edited_pixels_mismatch"] == 0
        g["E.independent_fft_clip_rint_pixels"] = rc["independent_pixels_mismatch"] == 0
        g["E.imag_residual_le_1e-8"] = rc["imag_residual_over_1e-8"] == rc["recorded_imag_over_1e-8"] == 0 and \
            rs["max_abs_imag"] <= 1e-8
    # F. deterministic replay
    g["F.cli_full_replay"] = (cli.get("replay") or {}).get("ok") is True and cli["replay"]["rows_replayed"] == N_PAIRS
    g["F.cli_file_verification"] = (cli.get("file_verification") or {}).get("ok") is True and \
        cli["file_verification"]["files_checked"] == N_PAIRS
    g["F.independent_process_full_replay"] = f["audit_replay"]["ok"] is True and f["audit_replay"]["rows_replayed"] == N_PAIRS \
        and f["audit_replay"]["mismatches"] == []
    # G. firewall and no-atime reads
    g["G.cli_firewall_clean"] = fw_cli.get("violation_count") == 0 and fw_cli.get("listing_attempts") == 0 and \
        fw_cli.get("distinct_files_opened", 0) <= fw_cli.get("allowlist_size", -1)
    g["G.cli_o_noatime_reads"] = ledger.get("o_noatime_reads") == 1 + 2 * reads_per_pass and \
        ledger.get("atime_change_policy") == "abort_run"
    g["G.audit_firewall_clean"] = f["audit_firewall"]["violation_count"] == 0 and f["audit_firewall"]["listing_attempts"] == 0
    g["G.audit_reads_noatime"] = f["audit_reader"]["atime_changed"] == 0 and f["audit_replay"]["o_noatime_reads"] == reads_per_pass
    # H. access-time ledger (stat-only snapshots before/after the bank)
    at = f["atime"]
    g["H.snapshot_complete"] = at["same_file_set"] and at["pre_files"] == 20615
    g["H.val_atime_unchanged"] = at["atime_changed_val"] == 0
    g["H.test_atime_unchanged"] = at["atime_changed_other_test_by_elimination"] == 0
    g["H.train_atime_unchanged"] = at["atime_changed_train_allowlisted"] == 0 and at["atime_changed_train_other"] == 0
    g["H.no_mtime_change"] = at["mtime_changed_total"] == 0
    # I. storage identity
    fs = f["filesystem"]
    g["I.output_filesystem_approved"] = (fs["filesystem_uuid"], fs["mount_point"]) == \
        (auth_rec["storage"]["filesystem_uuid"], auth_rec["storage"]["mount_point"]) and \
        f["run_dir"] == f"{auth_rec['storage']['runtime_root']}/runs/m6/{m}/seed_{seed}"
    # J. E04 non-interference
    e04 = f["e04"]
    g["J.e04_checkout_untouched"] = e04["checkout_head"] == E04_AUTHORITY and e04["checkout_status"] == ""
    g["J.e04_running_or_exited_zero"] = (bool(e04["process"]) and e04["exit_marker"] is None) or e04["exit_marker"] == "0"
    g["J.e04_run_identity"] = e04["run_manifest_run_id"] == E04_RUN_ID
    g["J.cpu_only"] = g["A.launch_gates_passed"] and (rm.get("environment") or {}).get("gpu_count") == 0 and \
        (rm.get("environment") or {}).get("gpu_model") is None       # launch gate refuses unless CUDA is hidden
    g["K.audit_tool_identity"] = f["tool_sha256"] == sha_file(ROOT / "tools/e01_e02_production_bank_audit.py") and \
        f["audit_git_commit"] == LAUNCH_COMMIT and f["audit_git_status"] == ""
    failed = sorted(k for k, v in g.items() if not v)
    status = f"{m}_SEED{seed}_PRODUCTION_BANK_AUDIT_PASS" if not failed else f"{m}_SEED{seed}_PRODUCTION_BANK_AUDIT_FAIL"
    record = {"record": f"{m}_PRODUCTION_BANK_SEED{seed}_AUDIT", "status": status, "method": m, "seed": seed,
              "run_id": rs["run_id"], "run_dir": f["run_dir"], "output_count": rs["output_count"],
              "failure_count": rs["failure_count"], "bank_manifest_sha256": f["manifest"]["sha256"],
              "pair_output_sha_digest": f["pair_output_sha_digest"], "run_dir_disk_bytes": f["run_dir_disk_bytes"],
              "filesystem": {k: fs[k] for k in ("filesystem_uuid", "mount_point", "source", "fstype")},
              "launch_commit": LAUNCH_COMMIT, "candidate_commit": CANDIDATE_COMMIT,
              "audit_tool_commit": f["audit_git_commit"], "config_sha256": cfg_sha,
              "environment_versions": rs["environment_versions"], "authorization": authz,
              "process": {k: p[k] for k in ("exit", "launch_time", "end_time", "atime_pre.time", "atime_post.time")},
              "generation_wall_time_seconds": rs["generation_wall_time_seconds"],
              "e01_operator_counts": f["e01_operator_counts"] if m == "E01" else None,
              "e01_level_histogram": f["e01_level_histogram"] if m == "E01" else None,
              "e01_level_zero_noop_count": rs["level_zero_noop_count"] if m == "E01" else None,
              "e01_outputs_identical_to_target": rc["outputs_identical_to_target"] if m == "E01" else None,
              "duplicate_output_pixel_groups": rs["duplicate_output_pixel_groups"],
              "duplicate_output_pixel_rows": rs["duplicate_output_pixel_rows"],
              "row_checks": rc, "cli_firewall": fw_cli, "cli_atime_ledger": ledger, "audit_replay": f["audit_replay"],
              "audit_firewall": f["audit_firewall"], "atime": at, "e04": e04, "gates": g, "failed_gates": failed,
              "gate_count": len(g), "passed": len(g) - len(failed),
              "collected_sha256": sha_file(a.collected)}
    prefix.with_suffix(".json").write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    md = [f"# {m} production bank, seed {seed}: independent audit", "", f"Status: `{status}` ({len(g) - len(failed)}/{len(g)} gates)", "",
          f"- run_id `{rs['run_id']}`, run dir `{f['run_dir']}`",
          f"- outputs {rs['output_count']} / {N_PAIRS}, failures {rs['failure_count']}",
          f"- bank manifest SHA256 `{f['manifest']['sha256']}`",
          f"- filesystem UUID `{fs['filesystem_uuid']}` at `{fs['mount_point']}` ({fs['source']})",
          f"- launch commit `{LAUNCH_COMMIT}` (candidate `{CANDIDATE_COMMIT}`), gpat-m5, CPU-only",
          f"- replay: CLI in-process {N_PAIRS}/{N_PAIRS} identical; independent process {f['audit_replay']['rows_replayed']} rows, "
          f"{len(f['audit_replay']['mismatches'])} mismatches",
          f"- face atime: {at['atime_changed_total']} changed of {at['pre_files']} (VAL {at['atime_changed_val']}, "
          f"TEST by elimination {at['atime_changed_other_test_by_elimination']}); every read was O_NOATIME",
          f"- TEST firewall: CLI violations {fw_cli.get('violation_count')}, audit violations "
          f"{f['audit_firewall']['violation_count']}, TEST rows parsed 0",
          f"- E04: checkout `{e04['checkout_head'][:8]}` clean={e04['checkout_status'] == ''}, process "
          f"{'running' if e04['process'] else 'not running'}, exit marker {e04['exit_marker']}",
          "", "| gate | result |", "|---|---|"] + [f"| {k} | {'PASS' if v else 'FAIL'} |" for k, v in g.items()]
    prefix.with_suffix(".md").write_text("\n".join(md) + "\n")
    print(json.dumps({"status": status, "passed": len(g) - len(failed), "gates": len(g), "failed": failed}))
    return 0 if not failed else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--method", required=True, choices=("E01", "E02"))
    c.add_argument("--seed", required=True, type=int, choices=(42, 1337, 2026))
    c.add_argument("--runtime-root", required=True)
    c.add_argument("--evidence-dir", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--repo", required=True)
    c.add_argument("--tool-sha256", required=True)
    b = sub.add_parser("build")
    b.add_argument("--collected", required=True)
    b.add_argument("--out-prefix", required=True)
    a = ap.parse_args(argv)
    global ROOT
    if a.cmd == "collect":
        ROOT = Path(a.repo).resolve()
    sys.path.insert(0, str(ROOT))
    return collect(a) if a.cmd == "collect" else build(a)


if __name__ == "__main__":
    sys.exit(main())
