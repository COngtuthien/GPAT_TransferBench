# Execution Flow

Human-readable flow of milestones. Machine state: `STAGE_STATE.json`. Action-level trace:
`EXECUTION_LEDGER.jsonl` (append-only). Milestone gates from spec §25.

```
M0 Bootstrap ─▶ M1 Inventory ─▶ M2 Preprocess ─▶ M3 Split ─▶ M4 Pairs ─▶ M5 Probe
      ─▶ M6 Baselines ─▶ M7 GPAT ─▶ M8 Banks ─▶ M9 Generator eval ─▶ M10 ResNet downstream
      ─▶ M11 DINOv3 downstream ─▶ M12 Ablations ─▶ M13 Statistics ─▶ M14 Paper pack
```

Owner review is required between milestones (no automatic advance).

## M0 — Bootstrap (session 2026-09-18, laptop cong-ThinkBook-16-G7-AHP)

1. Located spec candidates (2), verified byte-identical SHA256, verified content markers, read full spec.
2. Created project root `/home/cong/GPAT_TransferBench` (DEV-001), skeleton per §0.2 (+DEV-002), `git init -b main`.
3. Copied spec read-only into `docs/spec/`; stdlib text extraction.
4. Extracted canonical YAMLs §23.1 / §23.2 verbatim from the DOCX; snapshot into `frozen_config_snapshot/`.
5. Laptop environment audit (read-only) → `environments/laptop_environment_initial.txt`, `LAPTOP_ENVIRONMENT_AUDIT.md`.
6. GPU connectivity: alias + port OK; login needs password → PENDING (`GPU_CONNECTIVITY_AUDIT.md`).
7. PRISM discovery read-only → `PRISM_REUSE_AUDIT.md`.
8. Dataset path discovery (names only) → `configs/data_source_registry.yaml`.
9. Registries: `third_party/registry.yaml` (E00–E11), `models/registry.yaml`.
10. Audit infrastructure, M0 tests (`tests/test_m0_bootstrap.py`), artifact index.
11. Commit — requires owner-provided Git identity (see STAGE_STATE).

Not done (by design): inventory, frame extraction, preprocessing, split, pairs, any training,
any TEST access, GPU deployment.

## M0 — Finalization (session 2, 2026-09-18)

1. Owner approvals recorded in `deviation_report.md`: DEV-001 APPROVED, DEV-002 APPROVED, DEV-004 APPROVED_WITH_CONDITIONS, DEV-003 UNRESOLVED (to be resolved before M7).
2. Spec page-count note added to `SPEC_PROVENANCE.md` (33 pages confirmed externally per the owner; not rendered locally).
3. PRISM integrity re-check: PASS.
4. Staged-content inspection: 1 binary (spec DOCX, 711 KB), no data, weights, archives or secrets.
5. M0 tests re-run.
6. M0 commit → push `main` to `origin` (https://github.com/C0ngtuthien/GPAT_TransferBench.git), with no force push.
7. Post-push provenance (commit SHA, remote SHA) recorded in the ledger and STAGE_STATE via a follow-up provenance commit.
8. M0 commit `1823abad4e501cd640369f93026508f10d3950f2` pushed interactively by the owner to `origin` = https://github.com/COngtuthien/GPAT_TransferBench.git (`main`). The remote `main` SHA was verified equal to local HEAD, both by the owner and by the agent's `ls-remote`.
9. M0 → **COMPLETE**. Finalization commit "M0: record remote publication provenance" records this. The owner pushes it and verifies it; no further provenance commit is made for that push.

Open carry-overs: DEV-003 UNRESOLVED (before M7). GPU audit PENDING (before any GPU execution). Q-01…Q-11 open for their milestones.
