#!/usr/bin/env python3
"""Validate M6D1 evidence and firewall; never install or run scientific methods.

Default checks saved GPU evidence. --live-gpu adds public-key, read-only GPU
identity/source/version checks. Requires finalized ledger and artifact index.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEAD = "7387a25f62b50ad748edd1a69e21616af44501b9"
PIN = "0da1dd79bad00e225b8cb6977c7f3f06ee8f7517"
PREFIX_SHA = "a6cb6a8c6fef626cf4e8bd929b27d4f6634bc7829c77b511afe81dc70394820e"
ENV = "/home/student20261/miniconda3/envs/gpat-m5"
HOST = "student20261@100.121.84.44"
AUDIT = "outputs/audit/M6D1_CORE_ENVIRONMENT_QUALIFICATION.json"
LOCK = "environments/m6_core_gpu.lock.json"
LEDGER = "outputs/audit/EXECUTION_LEDGER.jsonl"
INDEX = "outputs/audit/ARTIFACT_INDEX.csv"
CAPTURES = {f"environments/m6_core_gpu.{suffix}" for suffix in
            ("conda-explicit.txt", "pip-freeze.txt", "runtime.json")}
ALLOWED = CAPTURES | {LOCK, AUDIT, AUDIT[:-5] + ".md", LEDGER, INDEX,
                       "tools/m6d1_core_environment_preflight.py"}
VERSIONS = {"python": "3.11.16", "numpy": "2.4.6", "pillow": "12.3.0",
            "opencv": "5.0.0", "pyyaml": "6.0.3", "imagecms_available": True}
PAIR_SEED = "19354000309184887456743517179028239418116648094833629759175866349341962868907"
FIRST_TEN = [2, 4, 16, 22, 25, 30, 33, 39, 42, 44]
DECISION = "QUALIFIED_EXECUTION_ENVIRONMENT_FOR_E01_E02_ONLY"


def require(condition: bool, label: str) -> None:
    if not condition:
        raise ValueError(label)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read(rel: str) -> bytes:
    return (ROOT / rel).read_bytes()


def load(rel: str) -> dict:
    return json.loads(read(rel))


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def canonical(data: dict) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode()


def check_gpu_record(gpu: dict) -> None:
    require(gpu["head"] == HEAD and gpu["branch"] == "m6-baselines"
            and gpu["worktree_clean"] is True, "GPU identity/clean state")
    require(gpu["versions"] == VERSIONS, "GPU versions")
    require(gpu["production_run_paths_exist"] == {"E01": False, "E02": False},
            "production runtime absent")
    source = gpu["source"]
    require(source["verified"] is True and source["pin"] == PIN, "source pin/deployment")
    pin = load("third_party/source_pins.json")["sources"]["fas_aug"]
    require(pin["pinned_commit"] == PIN and source["tree"] == pin["commit_tree"], "pin provenance")
    require(source["asset_counts"] == {"background": 90, "noiseTexture": 48, "MPTexture": 190},
            "asset counts")
    require(source["selected_icc_counts"] == {"rgb_profile_dict": 11, "cmyk_profile_dict": 7},
            "selected ICC counts")
    files = source["files"]
    require(source["files_count"] == len(files) == 470, "470 deployed files")
    by_path = {r["path"]: r for r in files}
    require(len(by_path) == 470, "unique source paths")
    for rel, row in by_path.items():
        require(not Path(rel).is_absolute() and ".." not in Path(rel).parts, "safe source path")
        require(re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) is not None
                and row["size_bytes"] >= 0, "source file evidence")
    for rel, spec in pin["cited_files"].items():
        require(by_path[rel]["sha256"] == spec["sha256"]
                and by_path[rel]["size_bytes"] == spec["size_bytes"], "cited source hashes")
    for rel, expected in gpu["reviewed_code_sha256"].items():
        require(sha(read(rel)) == expected == sha(git("show", f"{HEAD}:{rel}")), "reviewed code unchanged")
    freeze = {r["method_id"]: r["config_sha256"] for r in
              load("outputs/audit/M6B_CONFIG_FREEZE.json")["methods"]}
    for method, filename in (("E01", "e01_fas_aug.yaml"), ("E02", "e02_freqsub.yaml")):
        rel = "configs/methods/" + filename
        raw = read(rel)
        require(raw == read("frozen_config_snapshot/" + rel) == git("show", f"{HEAD}:{rel}"),
                "config/snapshot/commit integrity")
        require(sha(raw) == gpu["config_sha256"][rel] == freeze[method], "M6B config hash")


def check_qualification(q: dict) -> None:
    tests = q["focused_tests"]
    require([t["file"] for t in tests] == [f"tests/test_m6c1_{n}.py" for n in
            ("e01", "e02", "runtime")], "exact focused files")
    for t in tests:
        require(t["command"] == f'"$PY" {t["file"]} -v', "direct-file command")
        raw = t["stderr"]
        require(sha(raw.encode()) == t["stderr_sha256"] and raw.endswith("\nOK\n"), "test output/hash")
        count = int(re.search(r"Ran (\d+) tests in", raw)[1])
        require(t["run"] == t["pass"] == count == len(re.findall(r"\.\.\. ok$", raw, re.M)),
                "actual passing count")
        require(all(t[k] == 0 for k in ("fail", "error", "skip", "exit_code"))
                and t["skip_explanations"] == [], "no failure/error/skip")
    require([t["run"] for t in tests] == [26, 29, 28], "live per-file counts")
    synth = q["synthetic_contract"]
    e01, e02 = synth["E01"], synth["E02"]
    require(e01["passed"] is True and e01["source_bound_official_backend"] is True
            and e01["source_pin"] == PIN and e01["known_operator_seed"] == 795981663,
            "known E01 vector and official source")
    require(e01["seeds"] == [42, 1337, 2026] and e01["synthetic_shape"] == [256, 256, 3],
            "E01 synthetic scope")
    ops = ["Color_Diversity", "Color_Distortion", "Reflection", "BN_Halftone",
           "Moire_Pattern", "SFC_Halftone", "Hand_Trembling", "Low_Resolution"]
    expected = {(op, seed) for op in ops for seed in (42, 1337, 2026)}
    for key in ("checks", "level_zero_checks"):
        require(len(e01[key]) == 24 and {(r["operator"], r["seed"]) for r in e01[key]} == expected,
                "eight operators x three seeds")
    for r in e01["checks"]:
        require(r["byte_identical_repeat"] is True and r["metadata"]["backend_is_official"] is True
                and r["metadata"]["backend"] == "official_fas_aug"
                and r["metadata"]["sampled_operator_parameters"]["source_commit"] == PIN,
                "E01 official deterministic repeat")
    require(all(r["noop"] is True for r in e01["level_zero_checks"]), "level zero")
    require(e02["passed"] is True and e02["numpy_version"] == "2.4.6"
            and e02["n_eligible"] == 159 and e02["k"] == 40, "E02 geometry/version")
    require(e02["known_pair_seed"] == PAIR_SEED and e02["selected_indices"][:10] == FIRST_TEN,
            "known E02 seed/selection")
    require(len(e02["selected_indices"]) == len(set(e02["selected_indices"])) == 40,
            "E02 no replacement")
    require(len(e02["checks"]) == 6 and e02["tolerance"] == 1e-8, "E02 probes/tolerance")
    for r in e02["checks"]:
        require(r["byte_identical_repeat"] is True and r["hermitian_mask"] is True
                and 0 <= r["max_abs_imag"] < 1e-8 and r["output_dtype"] == "uint8", "E02 contract")
    require(e02["max_abs_imag"] == max(r["max_abs_imag"] for r in e02["checks"]), "residual maximum")
    require(synth["access"] == {"benchmark_image_opens": 0, "manifest_opens": 0,
                                 "test_scientific_execution": 0}, "synthetic access firewall")
    dry = q["dry_run"]
    require(dry["ok"] is True and dry["dry_run_benchmark_images_opened"] == 0
            and dry["open_audit"]["benchmark_image_opens"] == 0
            and dry["open_audit"]["manifest_opens"] == 0
            and dry["test_split_accessed"] is False, "dry-run safety/PASS")
    steps = {s["step"]: s for s in dry["steps"]}
    for method in ("E01", "E02"):
        require(steps[method]["determinism_byte_identical"] is True
                and steps[method]["all_success"] is True, "dry-run deterministic success")
    require(steps["resume_safety"]["overwrite_refused"] is True
            and steps["resume_append"]["history_preserved"] is True
            and steps["resume_append"]["records_after"] > steps["resume_append"]["records_before"],
            "resume contracts")
    smoke = dry["official_operator_synthetic_smoke"]
    require(smoke["status"] == "PASS" and len(smoke["checks"]) == 8
            and all(r["ok"] is True for r in smoke["checks"]), "official smoke")
    require(q["dry_run_temp_root_removed"] is True
            and q["production_run_paths_absent_before_and_after"] is True, "temporary cleanup")
    require(all(q[k] is False for k in ("benchmark_image_decoding", "test_scientific_execution",
                                        "scientific_bank_generated")), "scientific firewall")


def live_gpu(audit: dict) -> dict:
    # Fixed read-only script; audit strings are JSON data, never executable code.
    expected = {"source": audit["final_gpu"]["source"],
                "temp": audit["qualification_evidence"]["dry_run"]["temp_runtime_root"],
                "hashes": {**audit["final_gpu"]["config_sha256"],
                           **audit["final_gpu"]["reviewed_code_sha256"]}}
    script = r'''
import hashlib, json, platform, subprocess
from pathlib import Path
import numpy, PIL, PIL.ImageCms, cv2, yaml
root=Path('/home/student20261/workdir/GPAT_TransferBench')
def git(*args): return subprocess.check_output(['git','-C',str(root),*args]).decode().strip()
assert git('rev-parse','HEAD') == '7387a25f62b50ad748edd1a69e21616af44501b9'
assert git('branch','--show-current') == 'm6-baselines' and git('status','--porcelain') == ''
versions={'python':platform.python_version(),'numpy':numpy.__version__,'pillow':PIL.__version__,'opencv':cv2.__version__,'pyyaml':yaml.__version__,'imagecms_available':True}
assert versions == {'python':'3.11.16','numpy':'2.4.6','pillow':'12.3.0','opencv':'5.0.0','pyyaml':'6.0.3','imagecms_available':True}
source=root/'third_party/source_cache/fas_aug'
assert subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD']).decode().strip() == expected['source']['pin']
assert subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD^{tree}']).decode().strip() == expected['source']['tree']
actual={str(p.relative_to(source)) for p in source.rglob('*') if p.is_file() and '.git' not in p.relative_to(source).parts}
assert actual == {r['path'] for r in expected['source']['files']}
for row in expected['source']['files']:
    p=source/row['path']; assert not p.is_symlink()
    data=p.read_bytes(); assert len(data)==row['size_bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
for rel,digest in expected['hashes'].items():
    assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==digest
for method in ['E01','E02']:
    assert not (Path('/home/student20261/workdir/GPAT_TransferBench_runtime/runs/m6')/method).exists()
assert not Path(expected['temp']).exists()
print(json.dumps({'head':git('rev-parse','HEAD'),'worktree_clean':True,'source_files_verified':len(actual),'versions':versions,'production_paths_absent':True,'dry_run_temp_removed':True}))
'''
    payload = "expected = " + repr(expected) + "\n" + script
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "PreferredAuthentications=publickey",
                             "-o", "PasswordAuthentication=no", HOST, f"{ENV}/bin/python -B -"],
                            input=payload.encode(), capture_output=True, timeout=60)
    require(result.returncode == 0, "live GPU check: " + result.stderr.decode())
    return json.loads(result.stdout)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-gpu", action="store_true")
    args = parser.parse_args()
    require(git("rev-parse", "HEAD").decode().strip() == HEAD, "laptop HEAD")
    require(git("rev-parse", "origin/m6-baselines").decode().strip() == HEAD, "origin identity")
    require(git("branch", "--show-current").decode().strip() == "m6-baselines", "branch")
    require(git("rev-list", "--left-right", "--count", "HEAD...@{u}").split() == [b"0", b"0"], "divergence")
    audit, lock = load(AUDIT), load(LOCK)
    require(set(audit["allowed_changed_paths"]) == ALLOWED, "authorized paths")
    changed = set(git("diff", "--name-only", "HEAD", "-z").decode().split("\0")) - {""}
    changed |= set(git("ls-files", "--others", "--exclude-standard", "-z").decode().split("\0")) - {""}
    require(changed == ALLOWED, "changed paths differ from nine authorized artifacts: " + str(changed ^ ALLOWED))
    require(audit["starting_laptop"] == {"head": HEAD, "origin_m6_baselines": HEAD,
            "branch": "m6-baselines", "worktree_clean": True, "upstream_divergence": [0, 0]}, "starting laptop")
    require(audit["starting_gpu"] == audit["final_gpu"], "GPU before/after evidence")
    check_gpu_record(audit["final_gpu"])
    require(audit["invocation_incident"]["classification"] ==
            "NON_SCIENTIFIC_INVOCATION_ERROR_RESOLVED_BY_DIRECT_FILE_EXECUTION"
            and audit["invocation_incident"]["scientific_test_failure"] is False, "invocation-only incident")
    require(audit["focused_test_python"] == ENV + "/bin/python"
            and audit["focused_test_environment"] == {"PYTHONDONTWRITEBYTECODE": "1"}, "direct target Python")
    for probe in audit["probes"].values():
        require(sha(probe["source"].encode()) == probe["sha256"], "embedded probe hash")
    check_qualification(audit["qualification_evidence"])
    require(sha(canonical(audit["qualification_evidence"])) ==
            lock["qualification_evidence"]["canonical_json_sha256"], "lock evidence binding")
    runtime = load("environments/m6_core_gpu.runtime.json")
    for obj in (audit, lock, runtime):
        require(obj["qualified_methods"] == ["E01", "E02"], "qualification scope exactly E01/E02")
    require(audit["qualification_decision"] == lock["qualification"] == runtime["qualification"] == DECISION,
            "qualification decision")
    require(audit["method_status"] == lock["method_status"] ==
            {"E01": "IMPLEMENTED_NOT_EXECUTED", "E02": "IMPLEMENTED_NOT_EXECUTED"}, "not executed")
    require(audit["not_qualified"] == lock["not_qualified"] == ["E03", "E04", "E05", "E06c", "E07c"],
            "learned methods not qualified")
    require(lock["source_host"] == HOST and lock["source_environment_name"] == "gpat-m5"
            and lock["source_environment_absolute_path"] == ENV and lock["source_repository_commit"] == HEAD
            and lock["e01_source_pin"] == PIN and lock["numpy_execution_version"] == "2.4.6", "lock identity")
    require(runtime["environment_absolute_path"] == ENV and runtime["environment_name"] == "gpat-m5"
            and runtime["original_usage"] == "M5" and runtime["environment_preexisted_m6d1"] is True,
            "existing M5 environment")
    for key, expected in VERSIONS.items():
        require(runtime[key if key == "imagecms_available" else key + "_version"] == expected,
                "runtime version: " + key)
    require(runtime["pip_version"] == "26.2.1", "pip version")
    for obj in (lock, runtime, audit["environment_origin"]):
        for key in ("package_install_performed", "environment_created_by_m6d1", "environment_mutated"):
            require(obj[key] is False, key)
    safety_keys = {"package_install", "environment_creation", "environment_mutation",
                   "source_cache_mutation", "repository_synchronization", "scientific_config_change",
                   "benchmark_image_decoding", "test_scientific_execution", "scientific_bank_generated",
                   "learned_model_execution", "training", "optimizer", "backward", "scientific_checkpoint",
                   "benchmark_inference", "diffusion_sampling", "commit", "push"}
    require(set(audit["safety"]) == safety_keys and all(v is False for v in audit["safety"].values()),
            "prohibited action firewall")
    review = audit["static_review"]
    require(review["synthetic_arrays_only"] is True and review["benchmark_images_decoded"] is False
            and review["test_data_executed"] is False
            and review["reviewed_files_sha256"] == audit["final_gpu"]["reviewed_code_sha256"],
            "static safety review bound to executed code")
    require(all(audit["post_dry_run_verification"][k] is True for k in
                ("temp_root_absent", "production_E01_absent", "production_E02_absent", "gpu_worktree_clean"))
            and audit["post_dry_run_verification"]["exit_code"] == 0, "post-dry-run checks")
    for key in ("torch_version", "torchvision_version", "torch_cuda_runtime", "cudnn_version",
                "gpus", "system_nvcc_version", "required_by_e01_e02", "unrelated_package_availability"):
        require(key in runtime and runtime[key] is not None, "runtime inventory: " + key)
    require(runtime["unrelated_missing_packages_are_qualification_failures"] is False, "unrelated missing scope")
    require(set(lock["capture_sha256"]) == CAPTURES and lock["capture_sha256"] == audit["capture_sha256"],
            "three exact capture references")
    for rel, digest in lock["capture_sha256"].items():
        require(sha(read(rel)) == digest, "capture hash: " + rel)
    lock_sha = sha(read(LOCK))
    require(lock_sha == audit["environment_lock_sha256"], "environment lock SHA256")
    require(b"@EXPLICIT\n" in read("environments/m6_core_gpu.conda-explicit.txt"), "Conda explicit format")
    require(b"numpy==2.4.6\n" in read("environments/m6_core_gpu.pip-freeze.txt"), "NumPy freeze")
    raw = read(LEDGER); lines = raw.splitlines(keepends=True)
    committed = git("show", HEAD + ":" + LEDGER)
    require(len(lines) == 98 and b"".join(lines[:97]) == committed and sha(committed) == PREFIX_SHA,
            "97 -> 98 append-only ledger")
    require(audit["ledger"] == {"before_rows": 97, "after_rows": 98, "appended_records": 1,
            "prefix_bytes": len(committed), "prefix_sha256": PREFIX_SHA, "first_97_byte_identical": True},
            "ledger evidence")
    last = json.loads(lines[-1])
    require(last["classification"] == "M6D1_CORE_ENVIRONMENT_QUALIFICATION"
            and last["environment_lock_sha256"] == lock_sha, "one M6D1 record")
    require(sum(json.loads(line).get("classification") == "M6D1_CORE_ENVIRONMENT_QUALIFICATION"
                for line in lines) == 1, "exactly one M6D1 ledger record")
    require(set(last["file_sha256"]) == ALLOWED - {LEDGER, INDEX}, "ledger artifact coverage")
    for rel, digest in last["file_sha256"].items():
        require(sha(read(rel)) == digest, "ledger artifact hash: " + rel)
    index_raw = read(INDEX)
    require(index_raw.count(b"\n") == index_raw.count(b"\r\n"), "index CRLF")
    rows = list(csv.DictReader(io.StringIO(index_raw.decode())))
    # Same inclusion contract as the established builder, without modifying it.
    candidates = git("ls-files", "--cached", "--others", "--exclude-standard", "-z").decode().split("\0")
    expected_paths = sorted(rel for rel in candidates if rel and (ROOT / rel).is_file()
                            and rel not in {INDEX, LEDGER} and Path(rel).name != ".gitkeep"
                            and not {".git", "__pycache__", ".venv"}.intersection(Path(rel).parts)
                            and not rel.startswith(("data/raw/", "data/processed/", "cache/")))
    require([r["path"] for r in rows] == expected_paths, "artifact index path coverage/order")
    for row in rows:
        data = read(row["path"])
        require(len(data) == int(row["size_bytes"]) and sha(data) == row["sha256"],
                "artifact index entry: " + row["path"])
    gpu_live = live_gpu(audit) if args.live_gpu else {"mode": "RECORDED_EVIDENCE_ONLY"}
    print(json.dumps({"result": "M6D1_PREFLIGHT_PASS", "qualified_methods": ["E01", "E02"],
                      "focused_tests": {"run": 83, "pass": 83, "fail": 0, "error": 0, "skip": 0},
                      "environment_lock_sha256": lock_sha, "artifact_sha256": last["file_sha256"],
                      "ledger_rows": 98, "ledger_prefix_rows": 97, "ledger_prefix_sha256": PREFIX_SHA,
                      "ledger_sha256": sha(raw), "artifact_index_sha256": sha(index_raw),
                      "artifact_index_path_count": len(rows), "firewall": "PASS", "gpu": gpu_live},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
