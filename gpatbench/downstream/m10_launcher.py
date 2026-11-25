"""M10 ResNet-18 scientific launcher: one (group, detector seed) per invocation, never the whole matrix.

    env -i HOME=$HOME PATH=/usr/bin:/bin LANG=C.UTF-8 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 \
        PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
        ~/workdir/envs/gpat-m8-downstream/bin/python -B tools/run_m10_detector.py \
        --group E00 --seed 42 --runtime-root ~/workdir/GPAT_TransferBench_runtime

SCIENTIFIC mode needs the ACTIVE launch authority (configs/amendments/m10_resnet18_science_launch_authority.yaml) and
refuses unless every preflight gate passes (nothing is created when one fails):

  authority      active, TRAIN/VAL only, (group, seed) in the 24-run matrix, owner decisions / precision pin hashes
  source         clean worktree, the qualified source commit is an ancestor of HEAD, every bound code file hash equal
  frozen         evaluator contract, frozen detector config, spec hashes
  environment    sys.prefix, python, torch / torchvision / CUDA / cuDNN equal to the approved locked environment, lock
                 hash, CUBLAS_WORKSPACE_CONFIG, PYTHONHASHSEED
  weights        ImageNet ResNet-18 weight file SHA256
  splits         split_v1 / M2 hashes; TRAIN/VAL populations (counts, disjointness) rebuilt; TEST rows never parsed
  bank           E00 real-only; otherwise the seed-matched bank fully verified and equal to the authority's manifest hash
  storage        activated storage authority + live HDD preflight (UUID, mount, owner, 0700, rw, capacity, contents)
  gpu            UUID / driver / name equal to the authority, free VRAM >= floor (unrelated jobs recorded, never touched)
  concurrency    exclusive flock + no other detector process
  order (O5)     every earlier run of the seed-major order is complete with a committed PASS independent audit; the run
                 root does not exist

Then: create-only run root (0700), TEST firewall audit hook (only the TRAIN/VAL face files may be opened under the faces
root; no listing there), O1 precision configuration, frozen 40-epoch training (science_runner.train_detector), v2 run
manifest, exit status. Partial evidence is always kept. No TEST input exists in this code path.

NON_SCIENTIFIC_QUALIFICATION mode (--fixture) runs the same path on synthetic fixture images into a qualification
directory that must lie outside the detector root, the bank root and the runtime root; all outputs are labelled
non-scientific.
"""
from __future__ import annotations

import fcntl
import hashlib
import io
import json
import os
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

GROUPS = ("E00", "E01", "E02", "E03", "E04", "E05", "E06b", "E06c")
SEEDS = (42, 1337, 2026)
ORDER = tuple((g, s) for s in SEEDS for g in GROUPS)          # O5 A_SEED_MAJOR
AUTHORITY_PATH = "configs/amendments/m10_resnet18_science_launch_authority.yaml"
OWNER_PATH = "configs/amendments/m10_resnet18_owner_decisions.yaml"
PIN_PATH = "configs/amendments/m10_resnet18_science_precision.yaml"
CONTRACT = ("configs/amendments/m8_downstream_evaluator_contract.yaml",
            "67e4b62ddc6b81b8b9a8657e71b3979f6e6a186827183b052ae3d29b587fcbe0")
CONFIG = ("configs/frozen/downstream_resnet18.yaml", "f117c57380404bccfdc42370290c4bd2141b69346d3ee3f044d020f8b90a447b")
SPEC = ("docs/spec/GPAT_TransferBench_v1_0_Frozen_Specification_2026.docx",
        "f7d2371682ad9504f59f0c6984e4b1a741f855f1b1f60838f9d60b160281489e")
LOCK = ("environments/m8_downstream.lock.json", "70202661f1de3188aa7a6460a640982cea98074e1c41146ff516ac6fef3d7b9c")
ENV_EXPECTED = {"name": "gpat-m8-downstream", "python": "3.11.16", "torch": "2.12.1+cu130",
                "torchvision": "0.27.1+cu130", "cuda": "13.0", "cudnn": 92000,
                "sys_prefix": "/home/student20261/workdir/envs/gpat-m8-downstream"}
WORKERS = 8                    # decode/augment threads; parameters pre-drawn in sample order (qualified bitwise)
THREADS = 8
AUDIT_DIR = "outputs/audit/M10_RUNS"
AUDIT_RECORD = AUDIT_DIR + "/M10_{group}_SEED{seed}_AUDIT.json"
DEFAULT_LOCK_FILE = "~/workdir/m10_detector.lock"
CODE_FILES = ("gpatbench/downstream/science_runner.py", "gpatbench/downstream/resnet_runner.py",
              "gpatbench/downstream/metrics.py", "gpatbench/downstream/m10_selection.py",
              "gpatbench/downstream/m10_storage.py", "gpatbench/downstream/m10_launcher.py",
              "gpatbench/downstream/m10_run_manifest.py", "gpatbench/downstream/science_banks.py",
              "gpatbench/downstream/science_inputs.py", "gpatbench/downstream/firewall.py",
              "gpatbench/downstream/audit_hook.py", "gpatbench/downstream/bank_reader.py",
              "gpatbench/downstream/faceio.py", "methods/common/learned_bank.py", "methods/common/bank_writer.py",
              "tools/run_m10_detector.py", "tools/m10_run_auditor.py")
RUN_FILES = ("launch_preflight.json", "pools.json", "run_status.json", "firewall_report.json", "run_manifest.json",
             "exit_status.json", "detector")


class LaunchError(RuntimeError):
    pass


def sha_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _git(repo, *args) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


def code_sha256(repo) -> dict:
    return {f: sha_file(Path(repo) / f) for f in CODE_FILES}


def predecessors(group: str, seed: int) -> list:
    if (group, seed) not in ORDER:
        raise LaunchError(f"({group}, {seed}) is not in the M10 matrix")
    return list(ORDER[:ORDER.index((group, seed))])


def _atomic_json(path: Path, obj):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "x") as fh:
        fh.write(json.dumps(obj, indent=1, sort_keys=True, default=str) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    fd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


# ---------------------------------------------------------------------------------------------------- identities
def load_authority(repo) -> tuple[dict, str]:
    p = Path(repo) / AUTHORITY_PATH
    if not p.is_file():
        raise LaunchError(f"no active launch authority at {AUTHORITY_PATH}")
    raw = p.read_bytes()
    a = json.loads(raw)
    if a.get("status") != "OWNER_APPROVED_ACTIVE" or a.get("approved") is not True:
        raise LaunchError("launch authority is not OWNER_APPROVED_ACTIVE")
    auth = a.get("authorizes") or {}
    if auth.get("scientific_train_val_execution") is not True or auth.get("test_evaluation") is not False \
            or auth.get("gpat") is not False or auth.get("dinov3") is not False:
        raise LaunchError("launch authority scope is not TRAIN/VAL-only")
    return a, hashlib.sha256(raw).hexdigest()


def env_identity() -> dict:
    import torch
    import torchvision
    return {"name": Path(sys.prefix).name, "python": sys.version.split()[0], "sys_prefix": sys.prefix,
            "executable": sys.executable, "lock_path": LOCK[0], "torch": str(torch.__version__),
            "torchvision": str(torchvision.__version__), "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")}


def gpu_identity() -> dict:
    q = subprocess.run(["nvidia-smi", "--query-gpu=index,name,uuid,driver_version,memory.total,memory.used,"
                        "memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
                       check=True, capture_output=True, text=True).stdout.strip().splitlines()
    gpus = []
    for line in q:
        i, name, uuid, drv, tot, used, free, util = [x.strip() for x in line.split(",")]
        gpus.append({"index": int(i), "name": name, "uuid": uuid, "driver": drv, "total_mib": int(tot),
                     "used_mib": int(used), "free_mib": int(free), "utilization_pct": int(util)})
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory,gpu_uuid", "--format=csv,noheader,nounits"],
                          check=True, capture_output=True, text=True).stdout.strip().splitlines()
    procs = []
    for line in apps:
        if line.strip():
            pid, mem, uuid = [x.strip() for x in line.split(",")]
            procs.append({"pid": int(pid), "used_mib": int(mem), "gpu_uuid": uuid, "cmdline": _cmdline(int(pid))[:200]})
    return {"gpus": gpus, "compute_apps": procs, "utc": utc()}


def _cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
    except OSError:
        return ""


def other_detector_processes() -> list:
    me = os.getpid()
    ancestors, p = set(), me
    while p > 1:
        ancestors.add(p)
        try:
            p = int(Path(f"/proc/{p}/stat").read_text().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            break
    out = []
    for d in Path("/proc").iterdir():
        if not d.name.isdigit() or int(d.name) in ancestors:
            continue
        try:
            argv = [a.decode(errors="replace") for a in (d / "cmdline").read_bytes().split(b"\0") if a]
        except OSError:
            continue
        if argv and "python" in Path(argv[0]).name and any(a.endswith("run_m10_detector.py") for a in argv):
            out.append({"pid": int(d.name), "argv": argv[:8]})
    return out


class ExclusiveLock:
    def __init__(self, path):
        self.path = os.path.expanduser(path)
        self.fh = None

    def acquire(self) -> bool:
        self.fh = open(self.path, "a+")
        try:
            fcntl.flock(self.fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.fh.close()
            self.fh = None
            return False
        self.fh.seek(0)
        self.fh.truncate()
        self.fh.write(json.dumps({"pid": os.getpid(), "utc": utc()}) + "\n")
        self.fh.flush()
        return True

    def release(self):
        if self.fh is not None:
            fcntl.flock(self.fh.fileno(), fcntl.LOCK_UN)
            self.fh.close()
            self.fh = None


class VerifiedReader:
    """Thread-safe verified image reads: O_NOATIME, SHA256 checked before decoding, atime stat-verified unchanged,
    256x256 RGB PNG. A ledger counts distinct files and reads."""

    def __init__(self, root, *, noatime=True):
        self.root = Path(root)
        if not self.root.is_dir() or self.root.is_symlink():
            raise LaunchError(f"image root {self.root}")
        self.noatime = noatime
        self.lock = threading.Lock()
        self.files, self.reads, self.atime_changed = set(), 0, 0

    def read(self, rel: str, expected_sha256: str):
        from PIL import Image
        if ".." in Path(rel).parts or Path(rel).is_absolute():
            raise LaunchError(f"unsafe path {rel}")
        p = self.root / rel
        if p.is_symlink():
            raise LaunchError(f"symlink {rel}")
        a0 = os.stat(p).st_atime_ns
        fd = os.open(p, os.O_RDONLY | (os.O_NOATIME if self.noatime else 0))
        try:
            chunks = []
            while True:
                b = os.read(fd, 1 << 20)
                if not b:
                    break
                chunks.append(b)
        finally:
            os.close(fd)
        raw = b"".join(chunks)
        a1 = os.stat(p).st_atime_ns
        if hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise LaunchError(f"sha256 mismatch {rel}")
        with self.lock:
            self.files.add(rel)
            self.reads += 1
            self.atime_changed += a0 != a1
        if self.noatime and a0 != a1:
            raise LaunchError(f"atime changed on {rel}")
        with Image.open(io.BytesIO(raw)) as im:
            if im.format != "PNG" or im.mode != "RGB" or im.size != (256, 256):
                raise LaunchError(f"{rel} is not a 256x256 RGB PNG")
            return im.copy()

    def summary(self):
        with self.lock:
            return {"root": str(self.root), "distinct_files": len(self.files), "reads": self.reads,
                    "atime_changed_files": self.atime_changed}


# ---------------------------------------------------------------------------------------------------- inputs
def _ids_sha(items, key="sample_id", sha_key=None) -> str:
    return hashlib.sha256("\n".join(f"{i[key]}|{i[sha_key]}" if sha_key else i[key] for i in items).encode()).hexdigest()


def scientific_inputs(repo, runtime_root, group, seed) -> dict:
    from . import science_banks as SB
    from . import science_inputs as SI
    pops = SI.build_populations(repo)
    faces_root = Path(runtime_root) / "data" / "processed" / "faces_256"
    reader = VerifiedReader(faces_root)
    synthetic, bank, syn_reader = [], None, None
    if group != "E00":
        rb = SB.resolve_bank(repo, group, seed, verify_pixels=True)
        by_id = {}
        for s in rb["samples"]:
            synthetic.append({"sample_id": s.sample_id, "path": s.path, "output_sha256": s.output_sha256})
            by_id[s.sample_id] = s
        b = rb["binding"]
        bank = {"method_id": group, "seed": seed, "manifest_sha256": b["manifest_sha256"], "path": b["path"],
                "family": b["family"], "audit_record": b["audit_record"], "audit_record_sha256": b["audit_record_sha256"],
                "closure": b["closure"], "closure_sha256": b["closure_sha256"], "outputs": len(rb["samples"]),
                "failures": len(rb["failures"]), "samples_sha256": rb["samples_sha256"], "reader": rb["reader"]}

        def load_synthetic(item, _by=by_id):
            return SB.load_synthetic_image(_by[item["sample_id"]])
    else:
        load_synthetic = None
    allow = sorted({f["face_rel_path"] for k in ("train_live", "train_real_spoof", "val") for f in pops["pools"][k]})
    return {"pools": pops["pools"], "pool_sha256": pops["pool_sha256"], "counts": pops["counts"],
            "split_manifest_sha256": pops["split_manifest_sha256"], "m2_sha256": pops["m2_sha256"],
            "synthetic": synthetic, "bank": bank, "faces_root": str(faces_root), "firewall_allowlist": allow,
            "load_real": lambda f: reader.read(f["face_rel_path"], f["face_sha256"]),
            "load_synthetic": load_synthetic, "readers": {"faces": reader}}


def fixture_inputs(fixture_dir) -> dict:
    """NON_SCIENTIFIC fixture (tools/m10_launcher_qualification.py): synthetic PNGs; same verified reads/firewall."""
    fx = Path(fixture_dir)
    d = json.loads((fx / "fixture.json").read_text())
    if d.get("label") != "NON_SCIENTIFIC_FIXTURE":
        raise LaunchError("not a non-scientific fixture")
    reader = VerifiedReader(fx / "faces")
    syn_reader = VerifiedReader(fx / "bank")
    pools = d["pools"]
    synthetic = d["synthetic"]["samples"] if d["synthetic"] else []
    bank = None
    if d["synthetic"]:
        man = fx / "bank" / "manifest.json"
        bank = {"method_id": d["synthetic"]["method_id"], "seed": d["synthetic"]["seed"],
                "manifest_sha256": sha_file(man), "path": str(fx / "bank"), "family": "fixture",
                "outputs": len(synthetic), "failures": 0, "samples_sha256": _ids_sha(synthetic, sha_key="output_sha256")}
    pool_sha = {k: _ids_sha(v, sha_key="face_sha256") for k, v in pools.items()}
    allow = sorted({f["face_rel_path"] for v in pools.values() for f in v})
    return {"pools": pools, "pool_sha256": pool_sha,
            "counts": {**{k: len(v) for k, v in pools.items()},
                       "val_videos": len({(f["dataset"], f["video_id"]) for f in pools["val"]})},
            "split_manifest_sha256": d["split_manifest_sha256"], "m2_sha256": d["m2_sha256"],
            "synthetic": synthetic, "bank": bank, "faces_root": str(fx / "faces"), "firewall_allowlist": allow,
            "load_real": lambda f: reader.read(f["face_rel_path"], f["face_sha256"]),
            "load_synthetic": (lambda s: syn_reader.read(s["path"], s["output_sha256"])) if synthetic else None,
            "readers": {"faces": reader, "bank": syn_reader}}


# ---------------------------------------------------------------------------------------------------- preflight
def _gate(gates, name, ok, **detail):
    gates[name] = {"ok": bool(ok), **detail}
    return bool(ok)


def preflight(*, repo, group, seed, mode, authority=None, authority_sha=None, m10_root=None, out_root=None,
              runtime_root=None, weights_path, lock_file=DEFAULT_LOCK_FILE, gpu=None, env=None,
              storage_identity_fn=None) -> dict:
    """Every launch gate; returns {'gates', 'ok', 'context'}. Nothing is created here (the lock is not taken)."""
    from . import m10_storage as ST
    from . import science_runner as SR
    repo = Path(repo)
    g = {}
    ctx = {"repo": str(repo), "group": group, "seed": seed, "mode": mode}
    sci = mode == "SCIENTIFIC"
    _gate(g, "matrix", (group, seed) in ORDER and type(seed) is int, group=group, seed=seed)
    # authority -------------------------------------------------------------------------------------------
    owner_sha, pin_sha = sha_file(repo / OWNER_PATH), sha_file(repo / PIN_PATH)
    owner = json.loads((repo / OWNER_PATH).read_text())
    ctx.update(owner_decisions_sha256=owner_sha, precision_pin_sha256=pin_sha)
    _gate(g, "owner_decisions", owner.get("status") == "OWNER_APPROVED" and owner.get("approved") is True
          and owner["decisions"]["O1_science_precision"] == "A_CUDA_AMP_FP16_GRADSCALER_DETERMINISTIC"
          and owner["decisions"]["O2_threshold_tie_rule"] == "A_SMALLER_THRESHOLD", sha256=owner_sha)
    try:
        pol = SR.load_science_precision(repo)
        _gate(g, "precision_pin", pol.name == SR.O1_POLICY, policy=pol.name, sha256=pin_sha)
    except Exception as exc:                                       # noqa: BLE001
        _gate(g, "precision_pin", False, error=str(exc))
    if sci:
        ok = authority is not None and authority.get("status") == "OWNER_APPROVED_ACTIVE"
        b = (authority or {}).get("bindings", {})
        ok = ok and b.get("owner_decisions", {}).get("sha256") == owner_sha \
            and b.get("precision_pin", {}).get("sha256") == pin_sha \
            and [group, seed] in [list(x) for x in (authority.get("matrix") or {}).get("order", [])]
        _gate(g, "launch_authority", ok, sha256=authority_sha)
    else:
        _gate(g, "launch_authority_not_required", authority is None, note="NON_SCIENTIFIC_QUALIFICATION")
    # source ------------------------------------------------------------------------------------------------
    try:
        head = _git(repo, "rev-parse", "HEAD")
        dirty = _git(repo, "status", "--porcelain", "--untracked-files=normal")
    except Exception as exc:                                       # noqa: BLE001
        head, dirty = None, f"git error {exc}"
    codes = code_sha256(repo)
    ctx.update(execution_commit=head, code_sha256=codes)
    src_ok = head is not None and dirty == ""
    if sci:
        src = authority["bindings"]["source"]
        try:
            anc = subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", src["commit"], head]).returncode == 0
        except Exception:                                          # noqa: BLE001
            anc = False
        src_ok = src_ok and anc and codes == src["code_sha256"]
        ctx["source_commit"] = src["commit"]
    else:
        ctx["source_commit"] = head
    _gate(g, "source", src_ok, head=head, clean=dirty == "", porcelain=dirty[:500] if dirty else "")
    # frozen ------------------------------------------------------------------------------------------------
    fz = {p: (sha_file(repo / p), e) for p, e in (CONTRACT, CONFIG, SPEC, LOCK)}
    _gate(g, "frozen_hashes", all(a == e for a, e in fz.values()), files={p: a for p, (a, e) in fz.items()})
    # environment -------------------------------------------------------------------------------------------
    env = env or env_identity()
    lock = json.loads((repo / LOCK[0]).read_text())
    ctx["environment"] = dict(env, lock_sha256=fz[LOCK[0]][0])
    env_ok = all(env.get(k) == v for k, v in ENV_EXPECTED.items()) and lock.get("sys_prefix") == env["sys_prefix"] \
        and lock.get("python_vv", "").split()[0] == env["python"] and env.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8" \
        and env.get("PYTHONHASHSEED") == "0"
    _gate(g, "environment", env_ok, observed={k: env.get(k) for k in (*ENV_EXPECTED, "CUBLAS_WORKSPACE_CONFIG",
                                                                      "PYTHONHASHSEED")}, expected=ENV_EXPECTED)
    # weights -----------------------------------------------------------------------------------------------
    from . import resnet_runner as R
    wsha = sha_file(weights_path) if Path(weights_path).is_file() else None
    ctx["weights_sha256"] = wsha
    _gate(g, "weights", wsha == R.WEIGHTS_SHA256, path=str(weights_path), sha256=wsha)
    # gpu ---------------------------------------------------------------------------------------------------
    gpu = gpu or gpu_identity()
    ctx["gpu_snapshot"] = gpu
    want = (authority or {}).get("bindings", {}).get("gpu") if sci else None
    floor = (want or {}).get("free_floor_mib", 4096)
    vis = [x for x in gpu["gpus"] if x["index"] == 0]
    gok = len(vis) == 1 and vis[0]["name"] == "NVIDIA GeForce RTX 3090" and vis[0]["free_mib"] >= floor
    if want:
        gok = gok and vis[0]["uuid"] == want["uuid"] and vis[0]["driver"] == want["driver"]
    ctx["gpu"] = vis[0] if vis else None
    _gate(g, "gpu", gok, free_floor_mib=floor, observed=vis[0] if vis else None,
          unrelated_compute_apps=gpu["compute_apps"])
    # storage -----------------------------------------------------------------------------------------------
    if sci:
        try:
            st_auth = ST.load_storage_authority(repo)
            _gate(g, "storage_authority", st_auth["root"] == m10_root and authority["bindings"]["storage_authority"][
                "sha256"] == st_auth["_sha256"], sha256=st_auth["_sha256"])
            ctx["storage_authority"] = {"path": ST.STORAGE_AUTHORITY, "sha256": st_auth["_sha256"]}
        except Exception as exc:                                   # noqa: BLE001
            _gate(g, "storage_authority", False, error=str(exc))
        pre = ST.preflight(m10_root, identity_fn=storage_identity_fn)
        _gate(g, "storage_preflight", pre["ok"], **{k: pre.get(k) for k in ("checks", "filesystem", "unexpected_entries")})
        run_root = ST.run_root_path(m10_root, group, seed)
    else:
        bad = [r for r in (ST.EXPECTED["root"], ST.EXPECTED["bank_root"], str(runtime_root or "/nonexistent"))
               if ST._inside(os.path.realpath(out_root), r)]
        _gate(g, "qualification_root_outside_official_roots", out_root is not None and not bad, out_root=str(out_root))
        run_root = os.path.join(out_root, group, "seed_%d" % seed)
        ctx["storage_authority"] = None
    ctx["run_root"] = run_root
    _gate(g, "run_root_absent", not os.path.lexists(run_root), run_root=run_root)
    # concurrency -------------------------------------------------------------------------------------------
    others = other_detector_processes()
    held = False
    lp = os.path.expanduser(lock_file)
    if os.path.exists(lp):
        with open(lp) as fh:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            except BlockingIOError:
                held = True
    _gate(g, "no_concurrent_detector", not others and not held, processes=others, lock_held=held, lock_file=lp)
    # order (O5) --------------------------------------------------------------------------------------------
    if sci:
        missing = []
        for pg, ps in predecessors(group, seed):
            rel = AUDIT_RECORD.format(group=pg, seed=ps)
            tracked = subprocess.run(["git", "-C", str(repo), "ls-files", "--error-unmatch", rel],
                                     capture_output=True).returncode == 0
            rec = json.loads((repo / rel).read_text()) if tracked else {}
            proot = ST.run_root_path(m10_root, pg, ps)
            status = json.loads((Path(proot) / "run_status.json").read_text()).get("status") \
                if (Path(proot) / "run_status.json").is_file() else None
            if not (tracked and rec.get("status") == "PASS" and rec.get("label") == "SCIENTIFIC"
                    and rec.get("run_root") == proot and status == "COMPLETED"):
                missing.append(f"{pg}/seed_{ps}")
        later = [f"{lg}/seed_{ls}" for lg, ls in ORDER[ORDER.index((group, seed)) + 1:]
                 if os.path.lexists(ST.run_root_path(m10_root, lg, ls))]
        _gate(g, "order_predecessors_audited", not missing and not later, missing=missing, later_roots=later)
    return {"gates": g, "ok": all(v["ok"] for v in g.values()), "context": ctx, "utc": utc()}


def input_gates(pre, inputs, authority, group, seed, mode):
    """Bank and population identity of the built inputs (appended to the preflight gates)."""
    g = pre["gates"]
    if mode == "SCIENTIFIC":
        want = None if group == "E00" else authority["bank_families"][group]["banks"][str(seed)]
        got = (inputs["bank"] or {}).get("manifest_sha256")
        _gate(g, "bank_identity", got == want and (group == "E00") == (inputs["bank"] is None)
              and (group == "E00" or (inputs["bank"]["seed"] == seed and inputs["bank"]["method_id"] == group
                                      and inputs["bank"]["outputs"] + inputs["bank"]["failures"] == 8838
                                      and inputs["bank"]["reader"]["ok"] is True)),
              manifest_sha256=got, expected=want)
        d = authority["data"]
        _gate(g, "populations", inputs["counts"] == d["populations_counts"]
              and inputs["split_manifest_sha256"] == d["split_v1"] and inputs["m2_sha256"] == d["m2_accounting"],
              counts=inputs["counts"], pool_sha256=inputs["pool_sha256"])
    else:
        _gate(g, "bank_identity", (group == "E00") == (inputs["bank"] is None)
              and (group == "E00" or (inputs["bank"]["seed"] == seed and inputs["bank"]["method_id"] == group)),
              manifest_sha256=(inputs["bank"] or {}).get("manifest_sha256"), note="fixture bank")
        _gate(g, "populations", set(inputs["counts"]) == {"train_live", "train_real_spoof", "val", "val_videos"},
              counts=inputs["counts"], note="fixture populations")
    pre["ok"] = all(v["ok"] for v in g.values())


# ---------------------------------------------------------------------------------------------------- run
def _status(run_root: Path, status: str, **extra):
    p = run_root / "run_status.json"
    d = json.loads(p.read_text()) if p.exists() else {"events": []}
    d["events"].append({"status": status, "utc": utc(), **extra})
    d["status"] = status
    _atomic_json(p, d)


def execute(*, repo, group, seed, mode, inputs, pre, weights_path, epochs, warmup_epochs, argv, command,
            qualification_overrides=None, install_firewall=True, policy_override=None) -> int:
    """Create the run root (create-only), run, write evidence. Returns the process exit code (0 only on success)."""
    from . import audit_hook
    from . import m10_run_manifest as RM
    from . import m10_storage as ST
    from . import science_runner as SR
    ctx = pre["context"]
    sci = mode == "SCIENTIFIC"
    if sci and (policy_override is not None or qualification_overrides or not install_firewall
                or (epochs, warmup_epochs) != (40, 3)):
        raise LaunchError("SCIENTIFIC runs take no policy / qualification override and always install the firewall")
    if not pre["ok"]:
        raise LaunchError(f"preflight failed: {[k for k, v in pre['gates'].items() if not v['ok']]}")
    if sci:
        run_root = Path(ST.create_run_root(os.path.dirname(os.path.dirname(ctx["run_root"])), group, seed))
    else:
        rr = Path(ctx["run_root"])
        rr.parent.mkdir(parents=True, exist_ok=True)
        rr.mkdir(mode=0o700)
        run_root = rr
    if str(run_root) != ctx["run_root"]:
        raise LaunchError("run root path differs from the preflight")
    run_id = hashlib.sha256(f"{group}|{seed}|{ctx['execution_commit']}|{utc()}|{os.getpid()}".encode()).hexdigest()[:16]
    exit_code, err = 1, None
    try:
        _atomic_json(run_root / "launch_preflight.json", pre)
        _status(run_root, "STARTED", run_id=run_id, pid=os.getpid(), argv=argv)
        pools = inputs["pools"]
        _atomic_json(run_root / "pools.json", {
            "train_live": [f["sample_id"] for f in pools["train_live"]],
            "train_real_spoof": [f["sample_id"] for f in pools["train_real_spoof"]],
            "val": [f["sample_id"] for f in pools["val"]],
            "synthetic": [s["sample_id"] for s in inputs["synthetic"]],
            "pool_sha256": inputs["pool_sha256"], "counts": inputs["counts"],
            "synthetic_samples_sha256": (inputs["bank"] or {}).get("samples_sha256"),
            "firewall_allowlist_sha256": hashlib.sha256("\n".join(inputs["firewall_allowlist"]).encode()).hexdigest(),
            "firewall_allowlist_size": len(inputs["firewall_allowlist"])})
        if install_firewall:
            audit_hook.install(inputs["faces_root"], inputs["firewall_allowlist"], run_root / "firewall_report.json")
        policy = SR.load_science_precision(repo) if policy_override is None else SR.POLICIES[policy_override]
        SR.configure(policy, threads=THREADS)
        res = SR.train_detector(
            mode=mode, method_id=group, seed=seed, live=pools["train_live"], real_spoof=pools["train_real_spoof"],
            synthetic=inputs["synthetic"], val=pools["val"], load_real=inputs["load_real"],
            load_synthetic=inputs["load_synthetic"], weights_path=weights_path, out_dir=run_root / "detector",
            split_manifest_sha256=inputs["split_manifest_sha256"], policy=policy, epochs=epochs,
            warmup_epochs=warmup_epochs, retention="SELECTED_ONLY", workers=WORKERS,
            bank_seed=None if group == "E00" else seed, track_id=SR.TRACKS[group],
            qualification_overrides=qualification_overrides)
        if install_firewall:
            fw = dict(audit_hook.write_report(), installed=True)
        else:                                   # in-process NON_SCIENTIFIC tests only (refused in SCIENTIFIC mode)
            fw = {"installed": False, "allowlist_size": len(inputs["firewall_allowlist"]), "violation_count": 0,
                  "listing_attempts": 0, "opens_under_faces_root": None, "distinct_files_opened": 0}
            _atomic_json(run_root / "firewall_report.json", fw)
        if fw["violation_count"] or fw["listing_attempts"]:
            raise LaunchError("TEST firewall violation")
        det, sel, ep = res["record"], res["selection"], res["epochs"]
        d = run_root / "detector"
        steps_sha = sha_file(d / "steps.jsonl")
        manifest = {
            "schema": RM.SCHEMA, "label": sel["label"], "run_id": run_id, "method_id": group,
            "track_id": SR.TRACKS[group], "detector_seed": seed, "bank_seed": det["bank_seed"],
            "source_commit": ctx["source_commit"], "execution_commit": ctx["execution_commit"],
            "code_sha256": ctx["code_sha256"], "evaluator_contract_sha256": CONTRACT[1], "frozen_config_sha256": CONFIG[1],
            "owner_decisions_sha256": ctx["owner_decisions_sha256"], "precision_pin_sha256": ctx["precision_pin_sha256"],
            "launch_authority": {"path": AUTHORITY_PATH, "sha256": pre["gates"]["launch_authority"]["sha256"]} if sci else None,
            "storage_authority": ctx["storage_authority"],
            "split_manifest_sha256": inputs["split_manifest_sha256"], "m2_accounting_sha256": inputs["m2_sha256"],
            "pool_sha256": inputs["pool_sha256"], "pool_counts": inputs["counts"],
            "synthetic_bank": None if group == "E00" else {k: v for k, v in inputs["bank"].items() if k != "reader"},
            "weights_sha256": ctx["weights_sha256"], "environment": ctx["environment"],
            "gpu": {k: ctx["gpu"][k] for k in ("name", "uuid", "driver", "total_mib")},
            "precision_policy": det["precision_policy"], "grad_scaler": det["grad_scaler"],
            "determinism": det["determinism_before"], "amp_observed": det["amp_observed"],
            "optimizer": det["optimizer"], "scheduler": det["scheduler"],
            "rng": {"model": f"torch.manual_seed({seed}) before building the model (new fc head)",
                    "sampler": "numpy PCG64(SHA256('gpatbench.downstream.sampler.v1|resnet18|<method>|<seed>|<pool>|<pass>'))",
                    "augmentation": "torch.Generator().manual_seed(SHA256('gpatbench.downstream.augment.v1|resnet18|"
                                    "<method>|<seed>') mod 2^63), D14 parameters drawn in sample order",
                    "pool_passes": det["pool_passes"], "pool_discarded": det["pool_discarded"]},
            "command": command, "argv": argv, "working_directory": os.getcwd(), "run_root": str(run_root),
            "start_utc": det["start_utc"], "end_utc": det["end_utc"], "epochs": det["epochs"],
            "warmup_epochs": det["warmup_epochs"], "steps_per_epoch": det["steps_per_epoch"],
            "total_steps": det["total_steps"], "optimizer_steps_applied": det["optimizer_steps_applied"],
            "optimizer_steps_skipped": det["optimizer_steps_skipped"],
            "epoch_metrics": [{k: e[k] for k in ("epoch", "train_loss_mean", "train_loss_last", "optimizer_steps_applied",
                                                 "optimizer_steps_skipped", "grad_norm_pre_clip_mean",
                                                 "grad_norm_pre_clip_max", "grad_scale_end", "val_acer", "val_auc",
                                                 "threshold_hex", "threshold_tie_break", "state_sha256",
                                                 "predictions_sha256", "became_best", "val_videos", "val_frames")}
                              for e in ep],
            "checkpoint": {"path": "detector/" + sel["checkpoint_file"], "file_sha256": sel["checkpoint_file_sha256"],
                           "state_sha256": sel["checkpoint_state_sha256"], "epoch": sel["selected_epoch"],
                           "retention": "SELECTED_ONLY (O3)"},
            "val_predictions": [{"epoch": e["epoch"], "path": f"detector/predictions_val_epoch{e['epoch']:02d}.parquet",
                                 "sha256": e["predictions_sha256"], "frames": e["val_frames"], "videos": e["val_videos"]}
                                for e in ep],
            "selection": {k: sel[k] for k in ("selected_epoch", "checkpoint_state_sha256", "checkpoint_file_sha256",
                                              "val_acer", "val_auc", "frozen_val_threshold_hex", "selection_rule",
                                              "threshold_rule", "selection_source", "selection_reasoning",
                                              "threshold_tie_break")},
            "logs": {"steps": {"path": "detector/steps.jsonl", "sha256": steps_sha, "lines": det["total_steps"]},
                     "epochs": {"path": "detector/epochs.json", "sha256": sha_file(d / "epochs.json")},
                     "selection": {"path": "detector/selection.json", "sha256": sha_file(d / "selection.json")},
                     "detector": {"path": "detector/detector.json", "sha256": sha_file(d / "detector.json")},
                     "readers": {k: r.summary() for k, r in inputs["readers"].items()}},
            "firewall": fw, "exit_status": {"exit_code": 0, "source": "launcher in-process; see exit_status.json"},
            "audit": {"status": "PENDING_INDEPENDENT_AUDIT", "tool": "tools/m10_run_auditor.py"},
            "reproduction": [f"git checkout {ctx['execution_commit']} (code identical to source {ctx['source_commit']})",
                             f"environment {ENV_EXPECTED['name']} ({LOCK[0]} sha256 {LOCK[1]})",
                             f"command: {command}",
                             "inputs: split_v1 / M2 / faces_256 and the seed-matched bank as hashed in this manifest",
                             "expected: identical steps.jsonl, epochs.json and checkpoint SHA256 under the same GPU, "
                             "driver and environment (deterministic CUDA; not guaranteed across hardware)"],
            "test_accessed": False,
            "backup": "no independent copy: the single copy is under the HDD detector root; the checkpoint SHA256 "
                      "committed in the audit record is a checksum, not a backup"}
        RM.validate(manifest)
        _atomic_json(run_root / "run_manifest.json", manifest)
        exit_code = 0
    except BaseException as exc:                                   # noqa: BLE001
        err = {"type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()}
        if install_firewall and not (run_root / "firewall_report.json").exists():
            try:
                audit_hook.write_report()
            except Exception:                                      # noqa: BLE001
                pass
        if isinstance(exc, KeyboardInterrupt):
            exit_code = 130
    finally:
        _atomic_json(run_root / "exit_status.json", {"exit_code": exit_code, "utc": utc(), "run_id": run_id,
                                                     "error": err})
        _status(run_root, "COMPLETED" if exit_code == 0 else "FAILED", exit_code=exit_code)
    return exit_code


def main(argv=None) -> int:
    import argparse
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description="M10 ResNet-18 detector launcher (one group/seed)")
    ap.add_argument("--group", required=True, choices=GROUPS)
    ap.add_argument("--seed", required=True, type=int, choices=SEEDS)
    ap.add_argument("--runtime-root")
    ap.add_argument("--weights", default="~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth")
    ap.add_argument("--lock-file", default=DEFAULT_LOCK_FILE)
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--fixture", help="NON_SCIENTIFIC_QUALIFICATION: fixture directory")
    ap.add_argument("--qualification-out", help="NON_SCIENTIFIC_QUALIFICATION: output root (outside official roots)")
    ap.add_argument("--qualification-epochs", type=int)
    ap.add_argument("--qualification-warmup", type=int)
    ap.add_argument("--qualification-overrides", default="{}")
    ap.add_argument("--qualification-policy", help="NON_SCIENTIFIC_QUALIFICATION only (default: the O1 pin)")
    a = ap.parse_args(argv)
    repo = Path(__file__).resolve().parents[2]
    weights = os.path.expanduser(a.weights)
    command = " ".join([sys.executable, "-B", "tools/run_m10_detector.py", *argv])
    if a.fixture:
        mode, authority, asha = "NON_SCIENTIFIC_QUALIFICATION", None, None
        if not a.qualification_out or a.runtime_root is None:
            ap.error("--fixture needs --qualification-out and --runtime-root (to keep outputs outside it)")
        epochs, warm = a.qualification_epochs or 2, a.qualification_warmup or 1
        overrides = json.loads(a.qualification_overrides)
    else:
        mode = "SCIENTIFIC"
        if a.qualification_out or a.qualification_epochs or a.qualification_warmup or a.qualification_overrides != "{}" \
                or a.qualification_policy:
            ap.error("qualification options are refused in SCIENTIFIC mode (no hidden overrides)")
        if not a.runtime_root:
            ap.error("--runtime-root is required")
        authority, asha = load_authority(repo)
        epochs, warm, overrides = 40, 3, None
    from . import m10_storage as ST
    pre = preflight(repo=repo, group=a.group, seed=a.seed, mode=mode, authority=authority, authority_sha=asha,
                    m10_root=ST.M10_ROOT if mode == "SCIENTIFIC" else None,
                    out_root=a.qualification_out, runtime_root=a.runtime_root, weights_path=weights,
                    lock_file=a.lock_file)
    if not pre["ok"]:
        print(json.dumps({"preflight_ok": False, "failed": [k for k, v in pre["gates"].items() if not v["ok"]],
                          "gates": pre["gates"]}, indent=1, sort_keys=True, default=str))
        return 2
    lock = ExclusiveLock(a.lock_file)
    if not a.preflight_only and not lock.acquire():
        print(json.dumps({"preflight_ok": False, "failed": ["no_concurrent_detector (lock)"]}))
        return 2
    try:
        inputs = None
        try:
            inputs = fixture_inputs(a.fixture) if a.fixture else scientific_inputs(repo, a.runtime_root, a.group, a.seed)
            input_gates(pre, inputs, authority, a.group, a.seed, mode)
        except Exception as exc:                                   # noqa: BLE001
            _gate(pre["gates"], "inputs", False, error=f"{type(exc).__name__}: {exc}")
            pre["ok"] = False
        if a.preflight_only or not pre["ok"]:
            print(json.dumps({"preflight_ok": pre["ok"], "failed": [k for k, v in pre["gates"].items() if not v["ok"]],
                              "gates": pre["gates"]}, indent=1, sort_keys=True, default=str))
            return 0 if pre["ok"] else 2
        rc = execute(repo=repo, group=a.group, seed=a.seed, mode=mode, inputs=inputs, pre=pre, weights_path=weights,
                     epochs=epochs, warmup_epochs=warm, argv=argv, command=command, qualification_overrides=overrides,
                     policy_override=a.qualification_policy if a.fixture else None)
    finally:
        lock.release()
    print(json.dumps({"exit_code": rc, "run_root": pre["context"]["run_root"]}))
    return rc
