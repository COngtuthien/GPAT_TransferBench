"""Tiny synthetic dataset trees mirroring the local layouts (no real data is copied)."""
from __future__ import annotations

import zipfile
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]


def _img(seed: int, size: int = 16) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (size, size, 3), dtype=np.uint8)


def _png(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    path.write_bytes(buf.tobytes())


def _video(path: Path, n: int, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 32))
    assert w.isOpened(), path
    for i in range(n):
        w.write(np.full((32, 32, 3), (seed * 7 + i * 3) % 256, np.uint8))
    w.release()


def build(tmp: Path, casia_frames: int = 12) -> dict:
    casia, msu, siw = tmp / "casia", tmp / "msu", tmp / "siw"
    k = 0
    for part, subs in (("train", [1, 2]), ("test", [1])):
        for s in subs:
            for cls, codes in (("live", ["1"]), ("spoof", ["3", "HR_1"])):
                for code in codes:
                    for f in range(casia_frames):
                        k += 1
                        img = _img(k)
                        _png(casia / part / cls / f"s{s}v{code}f{f}.png", img)
                        if part == "train" and cls == "live":
                            _png(casia / part / cls / f"fs{s}v{code}f{f}.png", img[:, ::-1])
                            _png(casia / part / cls / f"bs{s}v{code}f{f}.png", np.clip(img.astype(int) + 10, 0, 255).astype(np.uint8))
    (msu / "scene01").mkdir(parents=True)
    (msu / "README.txt").write_text("fixture\n")
    (msu / "train_sub_list.txt").write_text("01\n")
    (msu / "test_sub_list.txt").write_text("02\n")
    (msu / "DecFrames.m").write_text("% fixture\n")
    (msu / "ffmpeg/bin").mkdir(parents=True)
    (msu / "ffmpeg/bin/ffmpeg.exe").write_bytes(b"MZ")
    for cid in ("001", "002"):
        for stem, ext in ((f"real/real_client{cid}_laptop_SD_scene01", "mov"),
                          (f"attack/attack_client{cid}_android_SD_printed_photo_scene01", "mp4"),
                          (f"attack/attack_client{cid}_laptop_SD_ipad_video_scene01", "mov")):
            _video(msu / "scene01" / f"{stem}.{ext}", 20, int(cid))
            (msu / "scene01" / f"{stem}.face").write_text("0,1,1,2,2,1,1,2,2\n")
    (siw).mkdir()
    (siw / "README.pdf").write_bytes(b"%PDF-fixture")
    (siw / "DRA.pdf").write_bytes(b"%PDF-fixture")
    _video(siw / "Live/Live_1.mov", 15, 3)
    _video(siw / "Live/Live_2.mp4", 5, 4)          # fewer than 8 frames
    _video(siw / "Spoof/Replay/Replay_1.mov", 30, 5)
    _video(siw / "Spoof/Paper/Paper_7.mov", 12, 6)  # unmapped token
    with zipfile.ZipFile(tmp / "siw.zip", "w") as z:
        for p in sorted(siw.rglob("*")):
            if p.is_file():
                z.write(p, "SiW-Mv2/" + p.relative_to(siw).as_posix())
    cfg = yaml.safe_load((REPO / "configs/frozen/data_v1.yaml").read_text())
    cfg["datasets"]["casia_fasd"].update(root=str(casia), archives_crosscheck=[])
    cfg["datasets"]["msu_mfsd"].update(root=str(msu), archives_crosscheck=[])
    cfg["datasets"]["siwmv2"].update(root=str(siw), archives_crosscheck=[{"path": str(tmp / "siw.zip"), "strip_prefix": "SiW-Mv2/"}])
    cfg["attack_map"] = str(REPO / "configs/frozen/attack_map_v1.yaml")
    out = tmp / "project"
    (out / "manifests").mkdir(parents=True)
    (out / "outputs/audit").mkdir(parents=True)
    cfg_path = tmp / "data_fixture.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return {"config": cfg_path, "project_root": out, "casia": casia, "msu": msu, "siw": siw}
