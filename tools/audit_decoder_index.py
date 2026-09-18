"""DEV-006 decoder index-continuity audit (read-only, diagnostic).

Question: after a failed OpenCV read() at original index f, are later successfully decoded
frames still assigned their correct original index (no backward shift)?

Method, per audited video:
  1. OpenCV (pinned opencv-python-headless, CAP_FFMPEG) sequential decode over [0, N_declared):
     record CAP_PROP_POS_FRAMES / CAP_PROP_POS_MSEC before and after each read(), status, and a
     content fingerprint (grayscale 64x48 INTER_AREA thumbnail) of each decoded frame.
  2. Independent reference: PyAV (separate FFmpeg build), single-threaded decode, one packet at a
     time in demux order. Each packet's presentation index = rank of its PTS among all video
     packets; decode errors are caught per packet. Fingerprints computed the same way.
  3. Alignment: for each OpenCV-valid index i, compare its fingerprint with the PyAV frames at
     presentation index i+d, d in [-3, 3]. Continuity holds at i if d=0 is the unique best match.

Outputs: outputs/audit/decoder_index_audit.csv (per-index rows around failures + per-video summary
rows) and a JSON summary printed to stdout. Nothing under the dataset roots is written.

Run with an audit venv built from environments/m1_decoder_audit_laptop.lock.txt (kept OUTSIDE the project root,
so its site-packages are never scanned as project artifacts):  <audit-venv>/bin/python tools/audit_decoder_index.py [--global]
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import av
import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "configs/frozen/data_v1.yaml").read_text())
OUT_CSV = ROOT / "outputs/audit/decoder_index_audit.csv"
WINDOW = 5
OFFSETS = range(-3, 4)

# (dataset, video rel path, reason)
TARGETS = [
    ("msu_mfsd", "scene01/attack/attack_client008_laptop_SD_ipad_video_scene01.mov", "mid-stream failure (M1)"),
    ("msu_mfsd", "scene01/attack/attack_client023_laptop_SD_iphone_video_scene01.mov", "mid-stream failure (M1)"),
    ("msu_mfsd", "scene01/attack/attack_client028_laptop_SD_printed_photo_scene01.mov", "FFmpeg error lines"),
    ("msu_mfsd", "scene01/attack/attack_client049_laptop_SD_printed_photo_scene01.mov", "FFmpeg error lines"),
    ("msu_mfsd", "scene01/attack/attack_client051_laptop_SD_printed_photo_scene01.mov", "FFmpeg error lines"),
    ("msu_mfsd", "scene01/attack/attack_client053_laptop_SD_ipad_video_scene01.mov", "FFmpeg error lines"),
    ("msu_mfsd", "scene01/real/real_client001_laptop_SD_scene01.mov", "control (no errors)"),
    ("siwmv2", "Live/Live_889.mp4", "trailing: declared 111, decoded 108"),
]


def fp(bgr: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, (64, 48), interpolation=cv2.INTER_AREA).astype(np.float32)


def opencv_pass(path: Path):
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rows, fps_ = [], {}
    for i in range(max(n, 0)):
        pb, mb = cap.get(cv2.CAP_PROP_POS_FRAMES), cap.get(cv2.CAP_PROP_POS_MSEC)
        ok, fr = cap.read()
        pa, ma = cap.get(cv2.CAP_PROP_POS_FRAMES), cap.get(cv2.CAP_PROP_POS_MSEC)
        valid = bool(ok and fr is not None and fr.size > 0)
        rows.append({"i": i, "ok": valid, "pos_before": pb, "pos_after": pa, "msec_before": mb, "msec_after": ma})
        if valid:
            fps_[i] = fp(fr)
    extra = bool(cap.read()[0])
    fps_val = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return n, rows, fps_, extra, fps_val


def pyav_pass(path: Path):
    """Presentation index = PTS rank among video packets; errors caught per packet."""
    with av.open(str(path)) as c:
        st = c.streams.video[0]
        st.thread_type = "NONE"
        pkts = [p for p in c.demux(st) if p.pts is not None]
    order = {p_pts: k for k, p_pts in enumerate(sorted(p.pts for p in pkts))}
    frames, errors = {}, {}
    with av.open(str(path)) as c:
        st = c.streams.video[0]
        st.thread_type = "NONE"
        time_base = float(st.time_base)
        start = st.start_time or 0
        for p in c.demux(st):
            if p.pts is None:
                continue
            k = order[p.pts]
            try:
                out = p.decode()
            except av.error.FFmpegError as e:  # decode failure for this packet
                errors[k] = f"{type(e).__name__}: {e}"
                continue
            for f in out:
                idx = order.get(f.pts, k)
                frames[idx] = (fp(f.to_ndarray(format="bgr24")), (f.pts - start) * time_base * 1000.0)
        # flush
        try:
            for f in st.codec_context.decode(None):
                idx = order.get(f.pts)
                if idx is not None:
                    frames[idx] = (fp(f.to_ndarray(format="bgr24")), (f.pts - start) * time_base * 1000.0)
        except av.error.FFmpegError:
            pass
    return len(pkts), frames, errors


def align(i, cv_fp, av_frames):
    diffs = {d: float(np.abs(cv_fp - av_frames[i + d][0]).mean()) for d in OFFSETS if (i + d) in av_frames}
    if 0 not in diffs:
        return None, diffs
    best = min(diffs, key=lambda d: (diffs[d], abs(d)))
    return best, diffs


def main() -> int:
    rows_out, summary = [], {"opencv": cv2.__version__, "pyav": av.__version__,
                             "pyav_libs": {k: ".".join(map(str, v)) for k, v in av.library_versions.items()},
                             "videos": []}
    for ds, rel, reason in TARGETS:
        path = Path(CFG["datasets"][ds]["root"]) / rel
        n, cv_rows, cv_fps, extra, fps_val = opencv_pass(path)
        n_pkts, av_frames, av_err = pyav_pass(path)
        fails = [r["i"] for r in cv_rows if not r["ok"]]
        # alignment over ALL OpenCV-valid indices
        aligned, misaligned, ambiguous = 0, [], []
        per_i = {}
        for i, f in cv_fps.items():
            best, diffs = align(i, f, av_frames)
            per_i[i] = (best, diffs)
            if best is None:
                ambiguous.append(i)
            elif best == 0:
                others = [v for d, v in diffs.items() if d != 0]
                if others and min(others) <= diffs[0] * 1.05 + 1e-6:
                    ambiguous.append(i)   # static content: offset not uniquely identifiable
                else:
                    aligned += 1
            else:
                misaligned.append(i)
        # after the first failure: any valid frame best-matching a shifted PyAV index?
        after_fail_misaligned = [i for i in misaligned if fails and i > fails[0]]
        focus = sorted({j for f in fails for j in range(f - WINDOW, f + WINDOW + 1) if 0 <= j < n})
        focus += [j for j in range(max(0, n - WINDOW), n) if j not in focus] if fails == [] and reason.startswith("trailing") else []
        for j in focus:
            r = cv_rows[j]
            best, diffs = per_i.get(j, (None, {}))
            rows_out.append({
                "row_type": "index", "dataset": ds, "video": rel, "declared_index": j,
                "opencv_status": "OK" if r["ok"] else "READ_FAILED",
                "reported_pos_before": r["pos_before"], "reported_pos_after": r["pos_after"],
                "reported_msec_before": round(r["msec_before"], 3), "reported_msec_after": round(r["msec_after"], 3),
                "pyav_status": "ERROR" if j in av_err else ("OK" if j in av_frames else "NO_FRAME"),
                "pyav_pts_msec": round(av_frames[j][1], 3) if j in av_frames else "",
                "best_offset": "" if best is None else best,
                "diff_at_0": "" if 0 not in diffs else round(diffs[0], 3),
                "min_diff_other": "" if len(diffs) < 2 else round(min(v for d, v in diffs.items() if d != 0), 3),
                "interpretation": ("failed read at this original index" if not r["ok"] else
                                   "aligned (unique best offset 0)" if best == 0 and j not in ambiguous else
                                   "ambiguous (static content)" if j in ambiguous else f"MISALIGNED offset {best}"),
            })
        v = {"dataset": ds, "video": rel, "reason": reason, "declared": n, "pyav_packets": n_pkts,
             "opencv_valid": len(cv_fps), "opencv_failed_indices": fails,
             "pyav_error_indices": sorted(av_err), "pyav_decoded": len(av_frames),
             "aligned_unique": aligned, "ambiguous_static": len(ambiguous), "misaligned": len(misaligned),
             "misaligned_after_first_failure": len(after_fail_misaligned), "frames_beyond_declared": extra,
             "opencv_fps": fps_val}
        v["verdict"] = ("CONTINUITY_PROVEN" if not misaligned and aligned > 0 else
                        "CONTINUITY_VIOLATED" if after_fail_misaligned else "INCONCLUSIVE")
        summary["videos"].append(v)
        rows_out.append({"row_type": "video_summary", "dataset": ds, "video": rel, "declared_index": "",
                         "opencv_status": f"valid={len(cv_fps)} failed={fails}",
                         "pyav_status": f"decoded={len(av_frames)} errors={sorted(av_err)} packets={n_pkts}",
                         "interpretation": f"{v['verdict']}: aligned={aligned} ambiguous={len(ambiguous)} misaligned={len(misaligned)}"})
    fields = ["row_type", "dataset", "video", "declared_index", "opencv_status", "reported_pos_before",
              "reported_pos_after", "reported_msec_before", "reported_msec_after", "pyav_status", "pyav_pts_msec",
              "best_offset", "diff_at_0", "min_diff_other", "interpretation"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for r in rows_out:
            w.writerow({k: r.get(k, "") for k in fields})
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__" and "--global" not in sys.argv:
    sys.exit(main())


# ----------------------------------------------------------------------------- global check
def _global_one(args):
    """For one video: OpenCV per-index POS_MSEC (after each successful read) vs PyAV packet PTS
    (ms) at the same presentation rank. Any disagreement > 1 ms = potential index shift."""
    ds, root, rel = args
    path = Path(root) / rel
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cv_ms, failed = {}, []
    for i in range(max(n, 0)):
        ok, fr = cap.read()
        if ok and fr is not None and fr.size > 0:
            cv_ms[i] = cap.get(cv2.CAP_PROP_POS_MSEC)
        else:
            failed.append(i)
    cap.release()
    with av.open(str(path)) as c:
        st = c.streams.video[0]
        tb, start = float(st.time_base), (st.start_time or 0)
        pts = sorted(p.pts for p in c.demux(st) if p.pts is not None)
    av_ms = {k: (p - start) * tb * 1000.0 for k, p in enumerate(pts)}
    mism = [i for i, m in cv_ms.items() if i not in av_ms or abs(m - av_ms[i]) > 1.0]
    return {"dataset": ds, "video": rel, "declared": n, "pyav_packets": len(pts), "opencv_valid": len(cv_ms),
            "opencv_failed_indices": json.dumps(failed[:20]), "n_failed": len(failed),
            "failed_inside_stream": sum(1 for f in failed if cv_ms and f < max(cv_ms)),
            "timestamp_mismatches": len(mism), "first_mismatch": mism[0] if mism else "",
            "verdict": "INDEX_TIMESTAMPS_CONSISTENT" if not mism else "INDEX_TIMESTAMP_MISMATCH"}


def main_global(workers: int = 8) -> int:
    import multiprocessing as mp
    import pyarrow.parquet as pq
    vids = [r for r in pq.read_table(ROOT / "manifests/inventory_videos.parquet").to_pylist()
            if r["source_kind"] == "video_file"]
    jobs = [(r["dataset"], CFG["datasets"][r["dataset"]]["root"], r["source_path"]) for r in vids]
    with mp.get_context("forkserver").Pool(workers) as pool:
        res = sorted(pool.imap_unordered(_global_one, jobs, chunksize=1), key=lambda r: (r["dataset"], r["video"]))
    out = ROOT / "outputs/audit/decoder_index_global.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(res)
    import collections
    print(json.dumps({"videos": len(res), "verdicts": collections.Counter(r["verdict"] for r in res),
                      "with_failed_inside_stream": sum(r["failed_inside_stream"] > 0 for r in res),
                      "declared_ne_packets": sum(r["declared"] != r["pyav_packets"] for r in res)}, default=dict))
    return 0


if __name__ == "__main__" and "--global" in sys.argv:
    sys.exit(main_global())
