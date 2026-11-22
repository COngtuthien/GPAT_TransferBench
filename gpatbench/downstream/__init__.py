"""Downstream evaluation infrastructure (spec §14-§16, §19-§22, §26). NON-EXECUTING.

Nothing in this package builds, trains or runs a detector, reads an image or touches TEST. It holds only the parts of
the downstream contract the frozen spec resolves:

- `metrics`: video aggregation (§15), APCER/BPCER/ACER, accuracy, balanced accuracy, AUC (§16), the VAL threshold
  sweep (§15), ROC/DET points, subgroups and the seed mean / sample std (§18);
- `schemas`: the §22 prediction / synthetic-manifest / run-metadata schemas and canonical hashing;
- `firewall`: split-membership checks that refuse TEST;
- `bank_reader`: a verified reader for E01/E02 bank directories;
- `plots`: the §26 rule that plot inputs must not hold unmeasured placeholders;
- `smoke_evidence`: the validator a future M8 smoke-test record must pass.

Unresolved choices are never defaulted. The decision operator at the threshold is a required argument. EER (D7),
spoof F1 (D8) and frame-level rates at the video threshold (D9) follow the owner-selected evaluator directions;
TPR@FPR and the SiW-Mv2 raw-type coverage rule (D10) wait for their exact rules to be approved
(outputs/audit/M8_CONTRACT_FINALIZATION.md). `toy_subset` plans the toy subset from TRAIN/VAL metadata only.
"""
