"""SiW-Mv2 subject-mapping recovery + duplicate audit (DIAGNOSTIC; read-only on the dataset).

1. Fetches the official protocol/code files from github.com/CHELSEA234/Multi-domain-learning-FAS at a
   PINNED commit into outputs/audit/siw_subject_recovery/official_repo_cache/ (git-ignored) and writes
   a tracked manifest with each file's sha256.
2. Builds outputs/audit/siw_subject_mapping_candidate.csv: one row per local SiW-Mv2 video with its
   protocol name (= video stem, traced through preprocessing.py/config_siwm.py) and protocol-list
   membership. candidate_subject_id is filled ONLY from a source that states person identity;
   none exists at the pinned commit, so every row is NO_EVIDENCE (no pseudo IDs are created).
3. Builds outputs/audit/SIW_DUPLICATE_VIDEO_AUDIT.csv for exact-byte duplicate video groups.

Run: .venv/bin/python tools/audit_siw_resolution.py
"""
from __future__ import annotations

import collections
import csv
import hashlib
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
REPO = "https://github.com/CHELSEA234/Multi-domain-learning-FAS.git"
PINNED = "8667dbcd316b38141729c057adf7517fe0602608"
OUTDIR = ROOT / "outputs/audit/siw_subject_recovery"
CACHE = OUTDIR / "official_repo_cache"
LISTS = [f"source_SiW_Mv2/pro_3_text/{n}.txt" for n in (
    "trainlist_all", "trainlist_live", "testlist_all", "testlist_live", "train_A_pretrain", "test_A_pretrain",
    "train_B_spoof", "test_B_spoof", "train_C_race", "test_C_race", "train_D_age", "test_D_age",
    "train_E_ill", "test_E_ill")]
CODE = ["source_SiW_Mv2/README.md", "source_SiW_Mv2/config_siwm.py", "source_SiW_Mv2/preprocessing.py",
        "source_SiW_Mv2/csv_parser.py", "source_SiW_Mv2/dataset.py", "README.md",
        "source_multi_domain/FASMD/README.md", "source_multi_domain/combine_label_illu.csv"]


def fetch() -> dict:
    heads = subprocess.run(["git", "ls-remote", REPO], capture_output=True, text=True, timeout=60,
                           env={"GIT_TERMINAL_PROMPT": "0", "PATH": "/usr/bin:/bin"}).stdout
    CACHE.mkdir(parents=True, exist_ok=True)
    man = {"repo": REPO, "pinned_commit": PINNED, "ls_remote": heads.strip().splitlines(),
           "pinned_is_current_head": any(l.startswith(PINNED) and l.endswith("HEAD") for l in heads.splitlines()),
           "files": {}}
    for rel in LISTS + CODE:
        dst = CACHE / rel
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            url = f"https://raw.githubusercontent.com/CHELSEA234/Multi-domain-learning-FAS/{PINNED}/{rel}"
            with urllib.request.urlopen(url, timeout=60) as r:
                dst.write_bytes(r.read())
        man["files"][rel] = hashlib.sha256(dst.read_bytes()).hexdigest()
    (OUTDIR / "official_repo_manifest.json").write_text(json.dumps(man, indent=2, sort_keys=True) + "\n")
    return man


def main() -> int:
    man = fetch()
    membership = collections.defaultdict(list)
    counts = collections.defaultdict(collections.Counter)
    for rel in LISTS:
        name = Path(rel).stem
        for tok in (CACHE / rel).read_text().split():
            membership[tok].append(name)
            counts[tok][name] += 1
    videos = [r for r in pq.read_table(ROOT / "manifests/inventory_videos.parquet").to_pylist() if r["dataset"] == "siwmv2"]
    files = {r["rel_path"]: r for r in pq.read_table(ROOT / "manifests/raw_file_index.parquet").to_pylist() if r["dataset"] == "siwmv2"}
    rows = []
    for v in sorted(videos, key=lambda r: r["video_id"]):
        stem = Path(v["source_path"]).stem          # preprocessing.py L31: folder_name = basename without extension
        lists = sorted(set(membership.get(stem, [])))
        rows.append({
            "video_id": v["video_id"], "local_path": v["source_path"], "attack_raw": v["attack_raw"] or "",
            "protocol_name": stem, "protocol_lists": ";".join(lists),
            "protocol_max_repeats": max(counts[stem].values()) if stem in counts else 0,
            "candidate_subject_id": "", "mapping_source": "none",
            "mapping_source_commit": PINNED, "mapping_rule": "no official source maps a video to a person identity",
            "confidence_class": "NO_EVIDENCE",
            "notes": "protocol name = video stem (preprocessing.py L31, config_siwm.py L162-176); not a person id",
        })
    with open(ROOT / "outputs/audit/siw_subject_mapping_candidate.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    local_stems = {r["protocol_name"] for r in rows}
    listed = set(membership)
    live_listed = {t for t in listed if t.startswith("Live_")}
    cover = {"local_videos": len(rows), "local_in_any_list": sum(1 for r in rows if r["protocol_lists"]),
             "listed_names": len(listed), "listed_not_local": sorted(listed - local_stems)[:30],
             "n_listed_not_local": len(listed - local_stems), "live_listed": len(live_listed),
             "trainlist_live_testlist_live_overlap": len(set(t for t in listed if "trainlist_live" in membership[t])
                                                         & set(t for t in listed if "testlist_live" in membership[t])),
             "names_repeated_within_a_list": sum(1 for t in counts if max(counts[t].values()) > 1)}
    # duplicates
    by_hash = collections.defaultdict(list)
    for v in videos:
        by_hash[files[v["source_path"]]["sha256"]].append(v)
    drows = []
    for h, grp in sorted(by_hash.items()):
        if len(grp) < 2:
            continue
        a, b = sorted(grp, key=lambda r: r["video_id"])[:2]
        sa, sb = Path(a["source_path"]).stem, Path(b["source_path"]).stem
        la, lb = sorted(set(membership.get(sa, []))), sorted(set(membership.get(sb, [])))
        cross = sorted({(x, y) for x in la for y in lb if x.replace("train", "T").replace("test", "T") == y.replace("train", "T").replace("test", "T") and x != y})
        if a["attack_raw"] == b["attack_raw"]:
            cls = "SAME_CONTENT_SAME_ATTACK_UNKNOWN_IDENTITY"
        else:
            cls = "SAME_CONTENT_DIFFERENT_METADATA"
        drows.append({"path_A": a["source_path"], "path_B": b["source_path"], "sha256": h, "size_bytes": files[a["source_path"]]["size_bytes"],
                      "attack_raw_A": a["attack_raw"], "attack_raw_B": b["attack_raw"],
                      "protocol_lists_A": ";".join(la), "protocol_lists_B": ";".join(lb),
                      "official_protocol_train_test_conflict": ";".join(f"{x}|{y}" for x, y in cross),
                      "candidate_subject_A": "", "candidate_subject_B": "",
                      "classification": "UNRESOLVED_DUPLICATE", "sub_classification": cls,
                      "notes": "kept as two raw entries; identity unknown (Q-14); M3 must keep identical content in one split"})
    with open(ROOT / "outputs/audit/SIW_DUPLICATE_VIDEO_AUDIT.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(drows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(drows)
    print(json.dumps({"manifest_pinned_is_head": man["pinned_is_current_head"], "coverage": cover,
                      "confidence": collections.Counter(r["confidence_class"] for r in rows),
                      "duplicates": drows}, indent=1, default=list))
    return 0


if __name__ == "__main__":
    sys.exit(main())
