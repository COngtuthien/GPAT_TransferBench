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

Unresolved choices are never defaulted. The decision operator at the threshold is a required argument; EER, TPR@FPR,
F1, the frame-level threshold and the SiW-Mv2 raw-type coverage rule are not implemented because the spec does not
fix them (see outputs/audit/M8_DOWNSTREAM_READINESS_ASSESSMENT.md).
"""
