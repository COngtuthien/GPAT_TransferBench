"""Dedicated M10 detector-output root on the approved external HDD (owner decision O3 A_SELECTED_ONLY_HDD_NEW_ROOT).

    /media/ext_data/student20261_gpat_m10_detectors/<group>/seed_<seed>/

Same filesystem (UUID 608647d2-9cb0-45b0-b999-73c080ba909b, ext4, mounted at /media/ext_data) as the immutable bank root
/media/ext_data/student20261_gpat_m8_banks, but a separate sibling directory: no detector output is ever written
under the bank root, and the bank root is never inside this one. Checks are read-only except the two explicit
create functions: create_root (once, after every identity gate passes on the parent mount) and create_run_root
(create-only per run; an existing run root is refused, never reused or cleaned).

Never: sudo, mount/remount/format, chmod of a parent, symlinks, path traversal, touching unrelated data.
"""
from __future__ import annotations

import os
import stat
from pathlib import Path

M10_ROOT = "/media/ext_data/student20261_gpat_m10_detectors"
BANK_ROOT = "/media/ext_data/student20261_gpat_m8_banks"
EXPECTED = {"root": M10_ROOT, "mount_point": "/media/ext_data", "filesystem_uuid": "608647d2-9cb0-45b0-b999-73c080ba909b",
            "fstype": "ext4", "owner": "student20261", "mode": 0o700, "forbidden_prefix": "/media/student20261/",
            "bank_root": BANK_ROOT}
GROUPS = ("E00", "E01", "E02", "E03", "E04", "E05", "E06b", "E06c")
SEEDS = (42, 1337, 2026)
MIN_FREE_BYTES = 20 * 2 ** 30         # >> 24 runs x ~60 MB measured selected-only footprint
MIN_FREE_INODES = 100_000
STORAGE_AUTHORITY = "configs/amendments/m10_detector_storage_location.yaml"


class StorageError(RuntimeError):
    pass


def filesystem_identity(path: str) -> dict:
    """Device, mount point, fs type, mount options and UUID (stat + /proc/self/mountinfo + /dev/disk/by-uuid)."""
    from methods.common.learned_bank import filesystem_identity as fi
    d = fi(path)
    d["statvfs_readonly"] = bool(os.statvfs(path).f_flag & os.ST_RDONLY)
    return d


def _rw(options) -> bool:
    opts = (options or "").split(",")
    return "rw" in opts and "ro" not in opts


def _inside(child: str, parent: str) -> bool:
    child, parent = os.path.normpath(child), os.path.normpath(parent)
    return child == parent or child.startswith(parent.rstrip("/") + "/")


def contents_problems(root: str, uid: int) -> list[str]:
    """Only <group>/seed_<s>/ real directories owned by uid, mode 0700, are allowed (no files, links or other names)."""
    bad = []
    seeds = {"seed_%d" % s for s in SEEDS}
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        st = os.lstat(p)
        if name not in GROUPS or not stat.S_ISDIR(st.st_mode) or st.st_uid != uid or stat.S_IMODE(st.st_mode) != 0o700:
            bad.append(name)
            continue
        for sub in sorted(os.listdir(p)):
            sst = os.lstat(os.path.join(p, sub))
            if sub not in seeds or not stat.S_ISDIR(sst.st_mode) or sst.st_uid != uid \
                    or stat.S_IMODE(sst.st_mode) != 0o700:
                bad.append(name + "/" + sub)
    return bad


def preflight(root: str | None = None, expected: dict | None = None, *, identity_fn=None,
              min_free_bytes: int = MIN_FREE_BYTES, min_free_inodes: int = MIN_FREE_INODES,
              check_contents: bool = True) -> dict:
    """Read-only identity / safety checks of the detector-output root. `ok` is True only if every check passes."""
    import pwd
    exp = dict(EXPECTED, **(expected or {}))
    root = root or exp["root"]
    identity_fn = identity_fn or filesystem_identity
    checks = {"path_is_canonical": os.path.isabs(root) and os.path.normpath(root) == root and ".." not in root.split("/"),
              "not_udisks_duplicate": not root.startswith(exp["forbidden_prefix"]),
              "separate_from_bank_root": not _inside(root, exp["bank_root"]) and not _inside(exp["bank_root"], root)}
    try:
        lst = os.lstat(root)
    except FileNotFoundError:
        checks["exists"] = False
        return {"root": root, "checks": checks, "ok": False}
    checks["exists"] = True
    checks["not_symlink"] = not stat.S_ISLNK(lst.st_mode)
    checks["is_directory"] = stat.S_ISDIR(lst.st_mode)
    checks["realpath_equals_path"] = os.path.realpath(root) == root
    try:
        checks["owner"] = pwd.getpwuid(lst.st_uid).pw_name == exp["owner"]
    except KeyError:
        checks["owner"] = False
    checks["owner_is_process_user"] = lst.st_uid == os.getuid()
    checks["mode_exact"] = stat.S_IMODE(lst.st_mode) == exp["mode"]
    fs = identity_fn(root) if checks["is_directory"] and checks["not_symlink"] else {}
    checks["mount_point"] = fs.get("mount_point") == exp["mount_point"]
    checks["filesystem_uuid"] = fs.get("filesystem_uuid") == exp["filesystem_uuid"]
    checks["fstype"] = fs.get("fstype") == exp["fstype"]
    checks["mount_read_write"] = _rw(fs.get("mount_options")) and _rw(fs.get("super_options")) \
        and fs.get("statvfs_readonly") is False
    checks["user_rwx_access"] = bool(checks["is_directory"] and os.access(root, os.R_OK | os.W_OK | os.X_OK))
    checks["free_bytes"] = (fs.get("free_bytes") or 0) >= min_free_bytes
    checks["free_inodes"] = (fs.get("free_inodes") or 0) >= min_free_inodes
    problems = contents_problems(root, lst.st_uid) if check_contents and checks["is_directory"] \
        and checks["not_symlink"] else []
    checks["no_unexpected_contents"] = problems == []
    return {"root": root, "checks": checks, "ok": all(checks.values()), "filesystem": fs,
            "mode_octal": oct(stat.S_IMODE(lst.st_mode)), "unexpected_entries": problems}


def parent_preflight(expected: dict | None = None, *, identity_fn=None) -> dict:
    """Checks on the mount point before the one-time creation of the root (the root itself must not exist)."""
    exp = dict(EXPECTED, **(expected or {}))
    identity_fn = identity_fn or filesystem_identity
    mp = exp["mount_point"]
    fs = identity_fn(mp) if os.path.isdir(mp) else {}
    checks = {"root_absent": not os.path.lexists(exp["root"]),
              "root_parent_is_mount_point": os.path.dirname(exp["root"]) == mp,
              "mount_point_not_symlink": os.path.isdir(mp) and not os.path.islink(mp),
              "mount_point_identity": fs.get("mount_point") == mp,
              "filesystem_uuid": fs.get("filesystem_uuid") == exp["filesystem_uuid"],
              "fstype": fs.get("fstype") == exp["fstype"],
              "mount_read_write": _rw(fs.get("mount_options")) and _rw(fs.get("super_options"))
              and fs.get("statvfs_readonly") is False,
              "free_bytes": (fs.get("free_bytes") or 0) >= MIN_FREE_BYTES,
              "bank_root_present_and_separate": os.path.isdir(exp["bank_root"]) and not _inside(exp["root"], exp["bank_root"])}
    return {"checks": checks, "ok": all(checks.values()), "filesystem": fs}


def create_root(expected: dict | None = None, *, identity_fn=None) -> dict:
    """One-time creation of the private root (owner-authorized). mkdir (not recursive), then chmod 0700 of the new
    directory only (clears a setgid bit inherited from the mount point); then the full preflight must pass."""
    exp = dict(EXPECTED, **(expected or {}))
    pre = parent_preflight(exp, identity_fn=identity_fn)
    if not pre["ok"]:
        raise StorageError(f"parent preflight failed: {[k for k, v in pre['checks'].items() if not v]}")
    os.mkdir(exp["root"], 0o700)
    os.chmod(exp["root"], 0o700)
    post = preflight(exp["root"], exp, identity_fn=identity_fn)
    if not post["ok"]:
        raise StorageError(f"created root fails preflight: {[k for k, v in post['checks'].items() if not v]}")
    return {"parent_preflight": pre, "preflight": post}


def run_root_path(root: str, group: str, seed: int) -> str:
    if group not in GROUPS or seed not in SEEDS or type(seed) is not int:
        raise StorageError(f"no M10 run ({group!r}, {seed!r})")
    return os.path.join(root, group, "seed_%d" % seed)


def create_run_root(root: str, group: str, seed: int, *, expected: dict | None = None, identity_fn=None) -> str:
    """Create-only <root>/<group>/seed_<seed>/ (0700). The group directory is created if missing; an existing run root
    is refused (never reused, cleaned or overwritten)."""
    pre = preflight(root, expected, identity_fn=identity_fn)
    if not pre["ok"]:
        raise StorageError(f"storage preflight failed: {[k for k, v in pre['checks'].items() if not v]}")
    g = os.path.join(root, group)
    run = run_root_path(root, group, seed)
    if os.path.lexists(run):
        raise StorageError(f"run root {run} already exists (create-only)")
    if not os.path.lexists(g):
        os.mkdir(g, 0o700)
        os.chmod(g, 0o700)
    if os.path.islink(g) or not os.path.isdir(g) or os.path.realpath(g) != g:
        raise StorageError(f"group directory {g} is not a real directory")
    os.mkdir(run, 0o700)
    os.chmod(run, 0o700)
    if os.path.realpath(run) != run:
        raise StorageError("run root realpath differs")
    return run


def load_storage_authority(repo: str | Path) -> dict:
    """The activated storage-authority record of the detector root (OWNER_APPROVED, bound to EXPECTED)."""
    import hashlib
    import json
    p = Path(repo) / STORAGE_AUTHORITY
    if not p.is_file():
        raise StorageError(f"no storage authority at {STORAGE_AUTHORITY}")
    raw = p.read_bytes()
    d = json.loads(raw)
    reg = d.get("detector_root_registry") or {}
    if d.get("status") != "OWNER_APPROVED" or d.get("approved") is not True \
            or (d.get("activation_verification") or {}).get("ok") is not True:
        raise StorageError("storage authority is not activated")
    for k in ("root", "mount_point", "filesystem_uuid", "fstype", "owner", "bank_root"):
        if reg.get(k) != EXPECTED[k]:
            raise StorageError(f"storage authority {k} differs from the approved root")
    if reg.get("mode") != "0700" or reg.get("layout") != "<m10_root>/<group>/seed_<seed>/":
        raise StorageError("storage authority mode / layout")
    return dict(reg, _sha256=hashlib.sha256(raw).hexdigest())
