"""CASIA-FASD source + semantics re-audit (read-only; does not trust generated manifests).

1. Walks the selected CASIA root directly and re-derives, from filenames alone:
   partitions, subject numbers, codes per subject, folder x code table, and the owner-approved
   identity mapping (train N -> N, test N -> 20+N) and code semantics (1,2,HR_1 live).
2. Reads the PNG IHDR header of every canonical and derived frame (width/height/bit depth/colour
   type) without decoding pixels.
3. Cross-checks the independent re-derivation against manifests/inventory_videos.parquet.

Writes outputs/audit/casia_source_audit.json. Run: .venv/bin/python tools/audit_casia_source.py
"""
from __future__ import annotations

import collections
import json
import os
import re
import struct
import sys
from pathlib import Path

import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "configs/frozen/data_v1.yaml").read_text())
CASIA = Path(CFG["datasets"]["casia_fasd"]["root"])
RX = re.compile(r"^(train|test)/(live|spoof)/([bf]?)s(\d+)v([1-8]|HR_[1-4])f(\d+)\.png$")
LIVE = {"1", "2", "HR_1"}
MACRO = {"3": "print", "4": "print", "HR_2": "print", "5": "print", "6": "print", "HR_3": "print",
         "7": "replay", "8": "replay", "HR_4": "replay"}
COLOR = {0: "gray", 2: "rgb", 3: "palette", 4: "gray+alpha", 6: "rgba"}


def ihdr(p: Path):
    with open(p, "rb") as f:
        h = f.read(33)
    if h[:8] != b"\x89PNG\r\n\x1a\n" or h[12:16] != b"IHDR":
        return None
    w, hgt, depth, ctype = struct.unpack(">IIBB", h[16:26])
    return w, hgt, depth, COLOR.get(ctype, str(ctype))


def main() -> int:
    seqs, dims, bad = collections.defaultdict(set), collections.Counter(), []
    for dp, _, fn in os.walk(CASIA):
        for f in fn:
            rel = (Path(dp) / f).relative_to(CASIA).as_posix()
            m = RX.match(rel)
            if not m:
                bad.append(rel)
                continue
            part, cls, aug, n, code, _fi = m.groups()
            kind = "derived" if aug else "canonical"
            dims[(kind, ihdr(Path(dp) / f))] += 1
            if not aug:
                seqs[(part, cls, int(n), code)].add(int(_fi))
    subj = {p: sorted({k[2] for k in seqs if k[0] == p}) for p in ("train", "test")}
    codes = collections.defaultdict(set)
    for part, cls, n, code in seqs:
        codes[(part, n)].add(code)
    ident = {(p, n): (n if p == "train" else 20 + n) for (p, n) in codes}
    folder_code = collections.Counter((cls, code) for (_, cls, _, code) in seqs)
    derived = {"n_sequences": len(seqs), "train_subject_numbers": subj["train"], "test_subject_numbers": subj["test"],
               "all_subjects_have_12_codes": all(len(c) == 12 for c in codes.values()),
               "identity_ids": sorted(set(ident.values())),
               "live_sequences": sum(1 for k in seqs if k[3] in LIVE),
               "spoof_sequences": sum(1 for k in seqs if k[3] not in LIVE),
               "spoof_macro_counts": dict(collections.Counter(MACRO[k[3]] for k in seqs if k[3] not in LIVE)),
               "folder_x_code": {f"{a}/{b}": v for (a, b), v in sorted(folder_code.items())},
               "folder_disagreements": sum(v for (cls, code), v in folder_code.items() if (cls == "live") != (code in LIVE)),
               "unparsed_files": bad[:10], "n_unparsed": len(bad),
               "png_header_summary": {f"{k}:{d}": v for (k, d), v in sorted(dims.items(), key=str)}}
    # cross-check vs manifest
    man = {r["video_id"]: r for r in pq.read_table(ROOT / "manifests/inventory_videos.parquet").to_pylist()
           if r["dataset"] == "casia_fasd"}
    mism = []
    for (part, cls, n, code) in seqs:
        vid = f"{part}/{cls}/s{n}v{code}"
        r = man.get(vid)
        exp = (str(ident[(part, n)]), 0 if code in LIVE else 1, None if code in LIVE else code,
               "live" if code in LIVE else MACRO[code])
        got = None if r is None else (r["subject_id_raw"], r["label_binary"], r["attack_raw"], r["attack_macro"])
        if got != exp:
            mism.append({"video_id": vid, "expected": exp, "manifest": got})
    derived["manifest_rows"] = len(man)
    derived["manifest_mismatches"] = mism[:20]
    derived["n_manifest_mismatches"] = len(mism) + len(set(man) - {f"{p}/{c}/s{n}v{k}" for p, c, n, k in seqs})
    (ROOT / "outputs/audit/casia_source_audit.json").write_text(json.dumps(derived, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in derived.items() if k not in ("folder_x_code",)}, indent=1))
    return 0 if not mism and not bad else 1


if __name__ == "__main__":
    sys.exit(main())
