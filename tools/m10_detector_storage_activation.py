"""One-time creation and verification of the M10 detector-output root (owner decision O3 A_SELECTED_ONLY_HDD_NEW_ROOT).

GPU host, clean clone, approved env (stdlib + the repo only):
    python -I -B tools/m10_detector_storage_activation.py --repo <clone> [--create] > storage_verification.json

Without --create: read-only identity checks of the mount point and (if present) the root.
With --create: the root is created only if it does not exist and every parent gate passes (UUID, mount, ext4, rw,
capacity, bank root present and separate); mkdir (not recursive) + chmod 0700 of the new directory only; then the full
preflight must pass. No sudo, remount, parent permission change, symlink or other write.
"""
import argparse
import json
import os
import pwd
import sys
import time
from pathlib import Path


def device_info(fs: dict) -> dict:
    out = {"source": fs.get("source"), "device": fs.get("device")}
    try:
        maj, mi = (int(x) for x in fs["device"].split(":"))
        sysdev = Path(f"/sys/dev/block/{maj}:{mi}").resolve()
        out["sys_path"] = str(sysdev)
        disk = sysdev.parent
        for k in ("model", "vendor"):
            p = disk / "device" / k
            out[k] = p.read_text().strip() if p.is_file() else None
        out["partition"] = sysdev.name
        out["disk"] = disk.name
        out["disk_size_bytes"] = int((disk / "size").read_text()) * 512
        out["removable"] = (disk / "removable").read_text().strip()
    except Exception as exc:                                      # noqa: BLE001
        out["error"] = str(exc)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--create", action="store_true")
    a = ap.parse_args()
    sys.path.insert(0, str(Path(a.repo).resolve()))
    from gpatbench.downstream import m10_storage as ST
    doc = {"utc_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "user": pwd.getpwuid(os.getuid()).pw_name,
           "uid": os.getuid(), "expected": ST.EXPECTED}
    doc["parent_preflight"] = ST.parent_preflight() if not os.path.lexists(ST.M10_ROOT) else {"skipped": "root exists"}
    mp = ST.filesystem_identity(ST.EXPECTED["mount_point"])
    doc["mount_point_identity"] = mp
    doc["device"] = device_info(mp)
    doc["mount_point_stat"] = {"mode": oct(os.stat(ST.EXPECTED["mount_point"]).st_mode & 0o7777),
                               "owner": pwd.getpwuid(os.stat(ST.EXPECTED["mount_point"]).st_uid).pw_name}
    bst = os.lstat(ST.BANK_ROOT)
    doc["bank_root"] = {"path": ST.BANK_ROOT, "mode": oct(bst.st_mode & 0o7777), "owner": pwd.getpwuid(bst.st_uid).pw_name,
                        "untouched": True}
    created = False
    if a.create:
        if os.path.lexists(ST.M10_ROOT):
            doc["create"] = {"created": False, "reason": "root already exists; never re-created"}
        else:
            r = ST.create_root()
            created = True
            doc["create"] = {"created": True, "parent_preflight": r["parent_preflight"]}
    doc["preflight"] = ST.preflight() if os.path.lexists(ST.M10_ROOT) else {"ok": False, "reason": "absent"}
    bst2 = os.lstat(ST.BANK_ROOT)
    doc["bank_root"]["unchanged_mode_owner_mtime"] = (bst.st_mode, bst.st_uid, bst.st_mtime_ns) == \
        (bst2.st_mode, bst2.st_uid, bst2.st_mtime_ns)
    doc["created_in_this_invocation"] = created
    doc["ok"] = bool(doc["preflight"].get("ok")) and doc["bank_root"]["unchanged_mode_owner_mtime"]
    doc["utc_end"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(json.dumps(doc, indent=1, sort_keys=True, default=str))
    return 0 if doc["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
