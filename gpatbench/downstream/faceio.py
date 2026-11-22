"""Face reads for the M8 smoke: bytes read with O_NOATIME, SHA256 checked against the M2 record before decoding,
atime stat-verified unchanged (a ledger proves the smoke never disturbed atime-based firewalls such as E04's)."""
from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path

CANONICAL = 256


class FaceReadError(RuntimeError):
    pass


class FaceReader:
    def __init__(self, faces_root: str | Path):
        self.faces_root = Path(faces_root)
        if not self.faces_root.is_dir() or self.faces_root.is_symlink():
            raise FaceReadError(f"faces root {self.faces_root}")
        self.ledger: dict[str, dict] = {}

    def read(self, rel_path: str, expected_sha256: str):
        """-> PIL RGB 256x256 image. rel_path is '<dataset>/<sample_id>.png' under the faces root."""
        from PIL import Image
        p = self.faces_root / rel_path
        if p.is_symlink() or ".." in Path(rel_path).parts:
            raise FaceReadError(f"unsafe face path {rel_path}")
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
        raw = b"".join(chunks)
        a1 = os.stat(p).st_atime_ns
        digest = hashlib.sha256(raw).hexdigest()
        if digest != expected_sha256:
            raise FaceReadError(f"face sha256 mismatch {rel_path}")
        e = self.ledger.setdefault(rel_path, {"reads": 0, "atime_unchanged": True})
        e["reads"] += 1
        e["atime_unchanged"] = e["atime_unchanged"] and a0 == a1
        if a0 != a1:
            raise FaceReadError(f"atime changed on {rel_path}")
        with Image.open(io.BytesIO(raw)) as im:
            if im.format != "PNG" or im.mode != "RGB" or im.size != (CANONICAL, CANONICAL):
                raise FaceReadError(f"{rel_path} is not a 256x256 RGB PNG")
            return im.copy()

    def summary(self) -> dict:
        return {"files": len(self.ledger), "reads": sum(e["reads"] for e in self.ledger.values()),
                "atime_changed_files": sum(1 for e in self.ledger.values() if not e["atime_unchanged"])}
