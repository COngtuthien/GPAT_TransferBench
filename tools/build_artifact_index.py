"""Rebuild outputs/audit/ARTIFACT_INDEX.csv: path, size, SHA256 of every project file.

Excluded: .git/, __pycache__/, .gitkeep placeholders, git-ignored large data dirs, the index
itself, and the append-only EXECUTION_LEDGER.jsonl (its hash changes on every append; its
integrity is protected by append-only policy + Git history instead).

Usage: python3 tools/build_artifact_index.py
"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gpatbench.audit.ledger import sha256_file  # noqa: E402

INDEX = ROOT / "outputs" / "audit" / "ARTIFACT_INDEX.csv"
EXCLUDE_FILES = {INDEX, ROOT / "outputs" / "audit" / "EXECUTION_LEDGER.jsonl"}
EXCLUDE_DIR_PARTS = {".git", "__pycache__", ".venv"}  # .venv is git-ignored; its lock is indexed instead
EXCLUDE_PREFIXES = ("data/raw/", "data/processed/", "cache/")


def iter_files():
    for p in sorted(ROOT.rglob("*")):
        rel = p.relative_to(ROOT).as_posix()
        if not p.is_file() or p in EXCLUDE_FILES or p.name == ".gitkeep":
            continue
        if EXCLUDE_DIR_PARTS & set(p.relative_to(ROOT).parts) or rel.startswith(EXCLUDE_PREFIXES):
            continue
        yield rel, p


def main():
    rows = [(rel, p.stat().st_size, sha256_file(p)) for rel, p in iter_files()]
    with open(INDEX, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "size_bytes", "sha256"])
        w.writerows(rows)
    print(f"indexed {len(rows)} files -> {INDEX.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
