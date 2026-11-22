"""Validator for a future M8 deterministic smoke-test record (outputs/audit/M8_SMOKE_TEST_RECORD.json).

The validator never produces a record; it only decides whether a candidate record is admissible. A record may state
smoke_test_passed true only if:
- it binds an approved smoke contract (configs/amendments/m8_smoke_test_contract.yaml, status OWNER_APPROVED);
- each O3 stage (generator, bank, evaluator, metrics, plots) executed and passed;
- the independent re-run reproduced it, the TEST firewall held, and the source tree was clean;
- it declares that no scientific result comes from it (O3).
`smoke_test_passed` must equal the conjunction of those facts, so a passing flag cannot be asserted over a partial
pipeline or over simulated stages.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REQUIRED_STAGES = ("generator", "bank", "evaluator", "metrics", "plots")          # M6D6jR O3 must_exercise
CONTRACT_PATH = "configs/amendments/m8_smoke_test_contract.yaml"
SCHEMA = "m8_smoke_test_record_v1"


class SmokeRecordInvalid(ValueError):
    pass


def _approved_contract_sha256(root: Path) -> str | None:
    p = root / CONTRACT_PATH
    if not p.is_file():
        return None
    doc = json.loads(p.read_text())
    if doc.get("status") != "OWNER_APPROVED" or doc.get("approved") is not True:
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()


def validate_smoke_record(record: dict, root: str | Path) -> dict:
    """Return {"admissible": bool, "passed": bool, "problems": [...]}; raise on a self-contradictory record."""
    root = Path(root)
    problems = []
    if record.get("schema_version") != SCHEMA:
        problems.append("schema_version")
    contract = _approved_contract_sha256(root)
    if contract is None:
        problems.append(f"no owner-approved smoke contract at {CONTRACT_PATH}")
    elif record.get("contract_sha256") != contract:
        problems.append("contract_sha256 does not match the approved contract")
    stages = record.get("stages") or {}
    stage_ok = {}
    for s in REQUIRED_STAGES:
        st = stages.get(s) or {}
        ok = (st.get("executed") is True and st.get("simulated") is False and st.get("status") == "PASS"
              and isinstance(st.get("gates"), dict) and bool(st["gates"]) and all(v is True for v in st["gates"].values())
              and isinstance(st.get("commands"), list) and bool(st["commands"])
              and all(c.get("exit_code") == 0 for c in st["commands"]))
        stage_ok[s] = ok
        if not ok:
            problems.append(f"stage {s} not executed and passed (executed, not simulated, PASS, gates, exit codes)")
    extras = {"independent_replay_identical": record.get("independent_replay", {}).get("identical") is True,
              "test_firewall_held": record.get("test_firewall", {}).get("violations") == 0
              and record.get("test_firewall", {}).get("test_rows_used") == 0,
              "source_tree_clean": record.get("source", {}).get("worktree_dirty") is False,
              "no_scientific_result": record.get("scientific_results_produced") is False}
    for k, v in extras.items():
        if not v:
            problems.append(k)
    derived = not problems
    claimed = record.get("smoke_test_passed")
    if claimed is True and not derived:
        raise SmokeRecordInvalid(f"record claims smoke_test_passed true but: {problems}")
    if claimed not in (True, False):
        problems.append("smoke_test_passed must be a bool")
    return {"admissible": claimed is derived, "passed": derived and claimed is True, "problems": problems,
            "stages": stage_ok, "checks": extras}
