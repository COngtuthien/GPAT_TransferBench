"""Stage-aware helpers for the 'no later-milestone artifact' guards.

These guards exist to catch a milestone leaking its outputs before it has run. They must therefore
track the stage state rather than hard-code one moment in the project's history: once M3 has
legitimately produced `split_v1.parquet`, forbidding that file would only mean the test is stale,
while M4 pair manifests and training checkpoints must stay forbidden until M4 actually starts.

M4 is a partial case: the common pair manifests are legitimate once M4 has started, but the native
manifests stay forbidden while their construction is blocked by a source gap, because writing one
would mean a pairing rule was invented.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE_STATE = ROOT / "outputs/audit/STAGE_STATE.json"

M1_MANIFESTS = {"manifests/inventory.parquet", "manifests/inventory_videos.parquet",
                "manifests/raw_file_index.parquet"}
M2_MANIFESTS = {"manifests/m2_sample_accounting.parquet"}
M3_MANIFESTS = {"manifests/split_v1.parquet", "manifests/split_groups_v1.parquet"}
M4_COMMON_MANIFESTS = {"manifests/pairs_train_v1.parquet", "manifests/val_pairs_v1.parquet",
                       "manifests/pair_train_stats_v1.json"}
# Materialized only from pinned official semantics; see M4_NATIVE_PAIR_SOURCE_AUDIT.md.
M4_NATIVE_MANIFESTS = {"manifests/dsdg_identity_pairs_v1.parquet",
                       "manifests/difffas_recon_pairs_v1.parquet"}

# Never allowed until M4 has actually started.
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
    if started("M4"):
        allowed |= M4_COMMON_MANIFESTS
        if native_pairs_unblocked():
            allowed |= M4_NATIVE_MANIFESTS
    return allowed


def native_pairs_unblocked() -> bool:
    """True only once the native pair construction is no longer source-blocked."""
    m4 = json.loads(STAGE_STATE.read_text())["milestones"]["M4"]
    native = (m4.get("execution") or {}).get("native") or {}
    return native.get("status") not in (None, "BLOCKED_BY_NATIVE_PAIR_CONSTRUCTION_SOURCE_GAP")


def forbidden_parquet(paths) -> list:
    """Parquet paths that no started milestone accounts for."""
    allowed = allowed_manifests()
    bad = [p for p in paths if p not in allowed]
    if not started("M4"):
        bad += [p for p in paths if any(f in Path(p).name for f in FORBIDDEN_NAME_FRAGMENTS)]
    return sorted(set(bad))


# Data artifacts that hold pair membership. Configs, audit records and hash files are contracts and
# provenance, not membership, so they are never matched here.
PAIR_DATA_SUFFIXES = (".parquet", ".json", ".csv", ".feather", ".arrow")
PAIR_DATA_NAME_PREFIXES = ("pairs_", "pair_train_stats", "dsdg_identity_pairs",
                           "difffas_recon_pairs", "val_pairs", "pair_manifest")


def forbidden_pair_artifacts(paths) -> list:
    """Pair-membership data files that the current stage does not account for.

    Before M4 starts, every pair data file is forbidden. Once M4 has started the common manifests
    are legitimate, but the native ones stay forbidden while their construction is source-blocked --
    writing one then would mean a pairing rule was invented. Only files under `manifests/` are
    considered: the frozen config, its snapshot, the proposal and the `.sha256` records describe the
    contract and its provenance, and `membership_in_this_file: false` is asserted separately.
    """
    allowed = allowed_manifests()
    out = []
    for q in paths:
        pp = Path(q)
        if pp.parts[:1] != ("manifests",):
            continue
        if pp.suffix not in PAIR_DATA_SUFFIXES:
            continue
        if not any(pp.name.startswith(x) for x in PAIR_DATA_NAME_PREFIXES):
            continue
        if q not in allowed:
            out.append(q)
    return sorted(set(out))
