"""Stage-aware helpers for the 'no later-milestone artifact' guards.

These guards exist to catch a milestone leaking its outputs before it has run. They must therefore
track the stage state rather than hard-code one moment in the project's history: once M3 has
legitimately produced `split_v1.parquet`, forbidding that file would only mean the test is stale,
while M4 pair manifests and training checkpoints must stay forbidden until M4 actually starts.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE_STATE = ROOT / "outputs/audit/STAGE_STATE.json"

M1_MANIFESTS = {"manifests/inventory.parquet", "manifests/inventory_videos.parquet",
                "manifests/raw_file_index.parquet"}
M2_MANIFESTS = {"manifests/m2_sample_accounting.parquet"}
M3_MANIFESTS = {"manifests/split_v1.parquet", "manifests/split_groups_v1.parquet"}

# Never allowed at any stage up to and including M3.
FORBIDDEN_NAME_FRAGMENTS = ("pairs_", "pair_manifest")


def statuses() -> dict:
    return {k: v["status"] for k, v in json.loads(STAGE_STATE.read_text())["milestones"].items()}


def started(milestone: str) -> bool:
    return statuses().get(milestone) in {"IN_PROGRESS", "BLOCKED", "COMPLETE"}


def allowed_manifests() -> set:
    """Manifest paths permitted by the milestones that have actually started."""
    allowed = set(M1_MANIFESTS)
    if started("M2"):
        allowed |= M2_MANIFESTS
    if started("M3"):
        allowed |= M3_MANIFESTS
    return allowed


def forbidden_parquet(paths) -> list:
    """Parquet paths that no started milestone accounts for."""
    allowed = allowed_manifests()
    bad = [p for p in paths if p not in allowed]
    bad += [p for p in paths if any(f in Path(p).name for f in FORBIDDEN_NAME_FRAGMENTS)]
    return sorted(set(bad))
