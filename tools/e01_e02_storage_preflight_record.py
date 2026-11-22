#!/usr/bin/env python3
"""Build outputs/audit/E01_E02_STORAGE_PREFLIGHT.json from a read-only GPU-host storage collection.

    python tools/e01_e02_storage_preflight_record.py --raw <collected.json> --out outputs/audit/E01_E02_STORAGE_PREFLIGHT.json

The collection (lsblk/df/findmnt/fstab/udev/sysfs, filesystem identities via /proc/self/mountinfo and
/dev/disk/by-uuid, smoke-bank sizes and one create/read/hash/delete probe in a new user-owned directory) was taken with
read-only commands; nothing was mounted, formatted, repartitioned or re-permissioned. This tool only classifies it.
Refuses to overwrite --out.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

FULL_PAIRS = 8838
SMOKE_PAIRS = 186
GIB = 1 << 30
FLOOR_BYTES = 24 * GIB
FLOOR_INODES = 1_000_000
RT = "/home/student20261/workdir/GPAT_TransferBench_runtime"


def disks(raw):
    out = []
    for d in raw["lsblk"]["blockdevices"]:
        if d["type"] != "disk":
            continue
        out.append({"device": d["path"], "model": (d.get("model") or "").strip(), "size_bytes": d["size"],
                    "size_decimal_tb": round(d["size"] / 1e12, 3), "size_tib": round(d["size"] / (1 << 40), 3),
                    "rotational": d["rota"], "transport": d.get("tran"),
                    "partitions": [{"device": c["path"], "size_bytes": c["size"], "fstype": c.get("fstype"),
                                    "label": c.get("label"), "uuid": c.get("uuid"),
                                    "mountpoints": [m for m in (c.get("mountpoints") or []) if m]}
                                   for c in d.get("children") or []]})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    raw = json.loads(Path(a.raw).read_text())
    fs = raw["filesystems"]
    sizes = raw["smoke_bank_sizes"]
    per_bank = {m: max(v["disk_bytes"] for k, v in sizes.items() if k.startswith(m)) * FULL_PAIRS / SMOKE_PAIRS
                for m in ("E01", "E02")}
    files_per_bank = {m: max(v["files"] for k, v in sizes.items() if k.startswith(m)) * FULL_PAIRS // SMOKE_PAIRS
                      for m in ("E01", "E02")}
    need = 3 * (per_bank["E01"] + per_bank["E02"])
    e04 = raw["e04_run_sizes"]
    e04_remaining = max(e04["seed_42"]["disk_bytes"], e04["seed_1337"]["disk_bytes"]) - e04["seed_2026"]["disk_bytes"]
    root, ext = fs[RT], fs["/media/ext_data"]
    probe = raw["create_read_hash_delete_probe"]
    sda = next(d for d in disks(raw) if d["device"] == "/dev/sda")
    five_tb_ssd = [d for d in disks(raw) if not d["rotational"] and 4.5e12 <= d["size_bytes"] <= 5.6e12]
    candidates = [
        {"device": "/dev/nvme0n1p3", "disk_model": "Samsung SSD 980 1TB", "filesystem": root["fstype"],
         "filesystem_uuid": root["filesystem_uuid"], "mount_point": root["mount_point"], "mount_options": root["mount_options"],
         "total_bytes": root["total_bytes"], "free_bytes": root["free_bytes"], "free_inodes": root["free_inodes"],
         "ssd_nvme": True, "writable_by_scientific_user": probe["roundtrip_identical"] and probe["deleted"],
         "persistent": "fstab UUID 227ce2f6 at / (defaults)", "local": True,
         "overlaps_scientific_storage": "holds the runtime root: faces_256 inputs (read-only) and the E01-E07 run roots, "
                                        "including the running E04 seed 2026",
         "state": "MOUNTED_AND_WRITABLE", "suitable": True, "selected": True,
         "reason": "the only filesystem that satisfies the frozen execution contract (configs/execution/m5_gpu_3090.yaml: "
                   "runtime_root, 'containment: all physical roots must resolve inside runtime_root', symlinks_used false) "
                   "with no path change; the estimated need is ~13% of free space and a hard per-bank floor protects E04 "
                   "and other users"},
        {"device": "/dev/sda2", "disk_model": f"Seagate {sda['model']} (USB {sda['transport']}, rotational)",
         "filesystem": ext["fstype"], "filesystem_uuid": ext["filesystem_uuid"],
         "mount_point": ext["mount_point"] + " (also auto-mounted at /media/student20261/" + ext["filesystem_uuid"] + ")",
         "mount_options": ext["mount_options"], "total_bytes": ext["total_bytes"], "free_bytes": ext["free_bytes"],
         "free_inodes": ext["free_inodes"], "ssd_nvme": False, "writable_by_scientific_user": ext["user_writable"],
         "persistent": "fstab UUID 608647d2 at /media/ext_data (defaults); the /media/student20261 auto-mount is transient",
         "local": "local USB-attached disk", "overlaps_scientific_storage": False,
         "shared_with_other_users": raw["ext_data_top_level_owners"], "mode": raw["ext_data_mode"],
         "state": "MOUNTED_BUT_UNSUITABLE_FOR_THIS_PRODUCTION", "suitable": False, "selected": False,
         "reason": "not an SSD (rotational HDD over USB) and not ~5 TB (4.0 TB decimal / 3.64 TiB); a group-writable "
                   "setgid directory shared with other accounts; outside the runtime_root containment contract, so using "
                   "it would need a path-contract change, which the directive says needs further owner approval. No "
                   "write test was performed on this shared disk."},
        {"device": "/dev/sda1", "filesystem": "vfat", "label": "EFI", "state": "INSTALLED_BUT_UNMOUNTED", "suitable": False,
         "selected": False, "reason": "200 MiB EFI partition of the USB disk; unmounted (not mounted by this work)"},
        {"device": "/dev/nvme0n1p1", "filesystem": "vfat", "mount_point": "/boot/efi", "state": "MOUNTED_SYSTEM_PARTITION",
         "suitable": False, "selected": False, "reason": "512 MiB EFI system partition"},
        {"device": "/dev/loop*", "filesystem": "squashfs", "state": "MOUNTED_READ_ONLY", "suitable": False,
         "selected": False, "reason": "read-only snap images"},
    ]
    record = {
        "record": "E01_E02_STORAGE_PREFLIGHT", "collected_utc": raw["collected_utc"], "hostname": raw["hostname"],
        "machine_id": raw["machine_id"], "boot_id": raw["boot_id"],
        "method": "read-only lsblk/df/findmnt/fstab/udevadm/sysfs/mountinfo collection; sudo -n nvme list refused "
                  "(password required; not prompted); one 1 MiB create/read/hash/delete probe in a new user-owned "
                  "directory outside the runtime root",
        "forbidden_actions_performed": [],
        "physical_disks": disks(raw),
        "answer_approximately_5tb_ssd_present": bool(five_tb_ssd),
        "answer_detail": "No. The host has two disks: a Samsung SSD 980 1TB NVMe (system/root) and a Seagate One Touch "
                         "w/PW 4.0 TB USB disk that reports rotational=1 (HDD). No ~5 TB SSD is installed, mounted or "
                         "unmounted.",
        "candidates": candidates,
        "capacity_plan": {
            "basis": "on-disk bytes of the passing M8 smoke banks (186 pairs per bank, the same writer and encoding), "
                     "scaled by 8838/186; logs and manifests included",
            "estimated_bytes_per_bank": {m: int(v) for m, v in per_bank.items()},
            "estimated_files_per_bank": files_per_bank,
            "estimated_bytes_six_banks": int(need),
            "e02_edit_maps": "included (PNG L, ~0.9 KB each)",
            "deterministic_replay": "in memory; no extra disk",
            "temporary_space": "none in the runtime root (create-only, direct writes)",
            "e04_seed_2026_remaining_bytes_estimate": int(e04_remaining),
            "free_bytes_now": root["free_bytes"], "free_inodes_now": root["free_inodes"],
            "min_free_bytes_before_bank": FLOOR_BYTES, "min_free_inodes_before_bank": FLOOR_INODES,
            "projected_free_after_six_banks": int(root["free_bytes"] - need),
            "future_previous_work_banks": "learned-method banks (E03-E06) of similar size would fit above the floor; "
                                          "if not, the 4 TB USB disk would need an owner path-contract decision",
            "quota": "no quota tool installed; ext4 root without user quotas"},
        "probe": probe,
        "decision": {"status": "STORAGE_SELECTED_EXISTING_RUNTIME_FILESYSTEM",
                     "selected_runtime_root": RT, "filesystem_uuid": root["filesystem_uuid"],
                     "mount_point": root["mount_point"], "device": root["source"],
                     "bank_paths": f"{RT}/runs/m6/<E01|E02>/seed_<42|1337|2026>/ (frozen M6B layout; unchanged)",
                     "evidence_paths": "/home/student20261/workdir/e01_e02_production/ (outside the runtime root)",
                     "path_contract_changed": False, "symlinks_or_bind_mounts_introduced": False},
    }
    out.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"five_tb_ssd": bool(five_tb_ssd), "need_gib": round(need / GIB, 2),
                      "free_gib": round(root["free_bytes"] / GIB, 2)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
