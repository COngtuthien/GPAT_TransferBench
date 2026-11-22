"""Filesystem firewall for M8 smoke stages: a Python audit hook that allows file opens under the faces root only for an
explicit allowlist of face files and forbids any directory listing there (so TEST file names are never even enumerated).
Every decision is logged; a violation raises PermissionError and is recorded for the evidence."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_STATE = {"installed": False}


def install(faces_root: str | Path, allowed_rel_paths, log_path: str | Path) -> None:
    if _STATE["installed"]:
        raise RuntimeError("firewall already installed")
    root = os.path.realpath(str(faces_root))
    allowed = {os.path.realpath(os.path.join(root, p)) for p in allowed_rel_paths}
    st = {"faces_root": root, "allowlist": len(allowed), "opens_under_faces_root": 0, "distinct_files": set(),
          "violations": [], "listing_attempts": 0}
    _STATE.update(installed=True, st=st, log=str(log_path))

    def under(path) -> str | None:
        if isinstance(path, int):
            return None
        try:
            rp = os.path.realpath(os.fsdecode(path))
        except Exception:
            return None
        return rp if rp == root or rp.startswith(root + os.sep) else None

    def hook(event, args):
        if event == "open" and args:
            rp = under(args[0])
            if rp is not None:
                st["opens_under_faces_root"] += 1
                if rp not in allowed:
                    st["violations"].append({"event": event, "path": rp})
                    raise PermissionError(f"M8 firewall: {rp} is not on the smoke allowlist")
                st["distinct_files"].add(rp)
        elif event in ("os.listdir", "os.scandir", "glob.glob", "os.walk") and args:
            rp = under(args[0])
            if rp is not None:
                st["listing_attempts"] += 1
                st["violations"].append({"event": event, "path": rp})
                raise PermissionError(f"M8 firewall: listing {rp} is forbidden")

    sys.addaudithook(hook)


def report() -> dict:
    st = _STATE["st"]
    return {"faces_root": st["faces_root"], "allowlist_size": st["allowlist"],
            "opens_under_faces_root": st["opens_under_faces_root"], "distinct_files_opened": len(st["distinct_files"]),
            "violations": st["violations"][:50], "violation_count": len(st["violations"]),
            "listing_attempts": st["listing_attempts"]}


def write_report() -> dict:
    r = report()
    Path(_STATE["log"]).write_text(json.dumps(r, indent=2) + "\n")
    return r
