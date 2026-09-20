"""ArtifactProbeNet (spec §12.1) — M5. PRE-FLIGHT ONLY.

Nothing in this package trains a model or writes a checkpoint. `preprocess` implements the frozen
high-pass operator with every *unresolved* choice exposed as a required argument, so the contract
cannot be fixed by accident: a caller must state each open decision explicitly, and the frozen
config that will eventually supply them does not exist yet.
"""
