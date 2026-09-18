# GPAT-TransferBench v1.0

Standalone benchmark of GPAT (Geometry-Preserving Artifact Transfer) against re-implemented
synthetic spoof generators for image-based Face Anti-Spoofing, on a pooled
CASIA-FASD + MSU-MFSD + SiW-Mv2 subject-disjoint split.

**Source of truth:** `docs/spec/GPAT_TransferBench_v1_0_Frozen_Specification_2026.docx`
(SHA256 `f7d2371682ad9504f59f0c6984e4b1a741f855f1b1f60838f9d60b160281489e`, read-only).
See `outputs/audit/SPEC_PROVENANCE.md`. The spec overrides any other instruction; conflicts go to
`outputs/audit/deviation_report.md`.

## Status

See `outputs/audit/STAGE_STATE.json`. Milestones M0–M14 follow spec §25; one milestone at a time.

## Hard rules (summary of spec §0.1 — read the spec itself)

- Never substitute named models, invent hyperparameters, alter split logic, or simplify baselines.
- Missing detail → `BLOCKED_BY_SOURCE_GAP` / `BLOCKED_BY_DEPENDENCY`, never a guess.
- TEST is never used for checkpoint/threshold/quality/hyperparameter selection.
- Every run saves code commit, config hash, manifest hash, checkpoint hash, seed, environment lock, per-sample predictions.
- Append-only evidence: no `git reset --hard`, `git clean`, force-push, history amendment, or deletion of scientific evidence. `outputs/audit/EXECUTION_LEDGER.jsonl` is append-only
  (`python3 -m gpatbench.audit.ledger append ...`).

## Layout

Directory contract of spec §0.2 plus: `models/` (auxiliary model registry), `environments/`
(environment captures/locks), `frozen_config_snapshot/`, `gpatbench/` (package; CLI per §24 later),
`docs/spec/`, `tools/`. Hosts: laptop `/home/cong/GPAT_TransferBench` (development, Git source of
truth, audit); GPU `/home/sparc/workdir/longnm/GPAT_TransferBench` (heavy execution; not yet deployed).

## Tests

```
.venv/bin/python -m unittest discover -s tests -v   # M1+ needs the project venv (environments/m1_inventory_laptop.lock.txt)
```

M1 inventory: `.venv/bin/python -m gpatbench.cli inventory --config configs/frozen/data_v1.yaml`
