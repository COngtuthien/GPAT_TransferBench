# M6 Preflight — Stage-State Reconciliation

Date: 2026-09-21

During M6 preflight, outputs/audit/STAGE_STATE.json was found to
still report M5 as NOT_STARTED although authoritative M5 training,
completion recording, Git publication, and the frozen M5 tag had
already completed.

This reconciliation changes no scientific result, split, model,
checkpoint, metric, threshold, or protocol.

Authoritative M5 training commit:
ad27df2389041b0f008668b7936d5efffd148c35

M5 completion commit:
9b846026747263fa2656bdbb865270a70e5af38b

M5 tag:
m5-artifact-probe-v1

Checkpoint SHA256:
b5ace6c263d8473215ff2ab825b98541bdcf332251512216cbd93546f32e54ff

Selected epoch: 14
Best VAL macro-F1: 0.9693201750054498
Validation passes: 30
TEST used: no
Synthetic samples used: no

M5 is reconciled to COMPLETE / FINALIZED.
M6 remains NOT_STARTED because only preflight/source auditing has
occurred; no M6 baseline has been trained or generated.
