#!/usr/bin/env python3
"""M8 global smoke test: fail-closed readiness check (no execution path exists).

    python tools/m8_smoke.py --check

Prints every readiness gate and exits 3 (M8_SMOKE_BLOCKED) unless all hold. Even when all hold it executes nothing:
the toy selector, evaluator and plot renderer are not implemented, because the contract (S2-S10) and the evaluator
contract are not approved. It reads only repository files; it opens no image, no runtime root and no TEST data.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXIT_BLOCKED = 3
S1_RECORD = "configs/amendments/m8_smoke_test_s1_owner_decision.yaml"
CONTRACT = "configs/amendments/m8_smoke_test_contract.yaml"
EVALUATOR_CONTRACT = "configs/amendments/m8_downstream_evaluator_contract.yaml"
COMPONENTS = {"toy_selector": "gpatbench/downstream/toy_subset.py",
              "evaluator_runner": "gpatbench/downstream/resnet_runner.py",
              "plot_renderer": "gpatbench/downstream/render.py",
              "evaluator_environment_lock": "environments/m8_downstream.lock.json"}


def _approved(path: str) -> bool:
    p = ROOT / path
    if not p.is_file():
        return False
    d = json.loads(p.read_text())
    return d.get("approved") is True and str(d.get("status", "")).startswith("OWNER_APPROVED")


def gates() -> dict:
    s1 = json.loads((ROOT / S1_RECORD).read_text()) if (ROOT / S1_RECORD).is_file() else {}
    g = {"S1_global_end_to_end_recorded": s1.get("decision", {}).get("option") == "A_GLOBAL_END_TO_END"
         and s1.get("approved") is True,
         "smoke_contract_S2_S10_approved": _approved(CONTRACT),
         "downstream_evaluator_contract_approved": _approved(EVALUATOR_CONTRACT)}
    for name, path in COMPONENTS.items():
        g[f"{name}_present"] = (ROOT / path).is_file()
    return g


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", required=True)
    ap.parse_args(argv)
    g = gates()
    ready = all(g.values())
    print(json.dumps({"status": "M8_SMOKE_READINESS_GATES_PASS_NO_EXECUTION_PATH" if ready else "M8_SMOKE_BLOCKED",
                      "gates": g, "missing": sorted(k for k, v in g.items() if not v)}, indent=2))
    return EXIT_BLOCKED


if __name__ == "__main__":
    sys.exit(main())
