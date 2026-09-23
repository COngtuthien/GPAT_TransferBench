# M6D3c — E04 training graph qualification stopped in Phase A

Decision: **STOP_AND_REPORT**. Reason: **UNRESOLVED_SCIENTIFIC_GRAPH_CHOICES**.
The filesystem firewall was preserved in this attempt. M6D3b's distinct historical
filesystem incident remains unchanged; it is not reclassified or erased.

The [resolution addendum](../../docs/spec/amendments/GPAT_TransferBench_v1_0_E04_Training_Graph_Resolution_Addendum.md)
contains the complete 29-component evidence inventory, source references, frozen
requirements and unresolved implementation decisions. It is **evidence only, not
an executable graph freeze**. No graph-resolution YAML or graph implementation was
created. No scientific architecture was selected on the basis of synthetic results.

## Why implementation stopped

1. Fig.4 fixes encoder endpoints but does not specify internal convolution widths.
   The three-slab ConvBlock inset does not supply an ordered per-layer channel
   table. The pinned predecessor has different endpoints. Constant-width blocks
   and blocks retaining predecessor internal widths both fit the stated endpoints
   and yield materially different networks. Neither is selected.
2. The paper's inpainting loss needs P0, with attack-dependent priors in Fig.7.
   E04 forbids attack labels, and base/A3/A4 do not specify a binary-only P0.
   Removing the term, setting a zero denominator, or inventing a mask is not a
   resolved implementation. RGB trace magnitude reduction also needs definition.
3. Combined versus separate generator optimizer applications and the effective
   hard-sample trace target remain ambiguous. These affect Adam state, gradients,
   update ordering and the required number of qualification steps.

The owner's explicit ambiguity stop rule applies. Resuming requires an exact
architecture/graph specification or explicit controlled choices addressing these
items, followed by an executable additive freeze before any optimizer execution.

## Source and version findings

Retrieved [arXiv:2012.05185v1](https://arxiv.org/pdf/2012.05185) directly, plus the
exact [author PDF URL](http://cvlab.cse.msu.edu/pdfs/liu_liu_PAMI2022.pdf) already in
M6D3b and four source-text files from the pinned STDN commit
`c79f1f8c615d2b8471b3df29da881bb18dd54c90`. Hashes and locations are in JSON.
No local paper search was performed. The author PDF was retrieved via HTTP after
HTTPS certificate validation failed; no certificate validation bypass was used.

The two paper versions differ in printed frequency sizes, trace regularization
and coefficient placement. The owner's arXiv Eq.23/24 and frozen E04 objectives
remain authoritative. These discrepancies do not justify rewriting the config.

## Qualification state

| Item | State |
|---|---|
| Geometry | E04_GEOMETRY_RUNTIME_QUALIFIED retained from M6D3a |
| Training graph | NOT_YET_IMPLEMENTED; not qualified |
| Training runtime | NOT_YET_QUALIFIED |
| E04 method | IMPLEMENTED_NOT_EXECUTED |
| Fidelity | CONTROLLED_ADAPTATION |
| Benchmark execution | NOT_STARTED |
| Synthetic optimizer steps | 0 |
| Two-process repeatability | NOT_RUN |

All shape/state/parameter/op-count, initializer, finite-loss, gradient, LR-runtime,
weighted-objective, update-isolation and repeatability gates are NOT_RUN. Expected
LR values are contractual expectations, not measured results. No E04 training
environment lock was created. The exact requested E03 Python path returned ENOENT
on this host; this is not evidence that it is absent on the GPU host. No remote
environment was touched and no package was installed.

The input256/batch8/budget150000, Normal(0,.02), Adam(.9,.999,epsilon1e-8), zero
weight decay, base LR5e-5 /10 each45000, alpha=(100,5,1,1e-4,10,1), beta=.1,
lambda=1, K32, seeds42/1337/2026 and terminal150000 remain unchanged. TEST remains
forbidden. All frozen snapshot files and preserved E04/M6D3b inputs are checked
against the starting commit.

## Audit bookkeeping and access boundary

Starting commit: `28f17846a67279d9b196ef0e0792a998104327f1`, initially clean.
Ledger: **102 → 103**, exactly one `M6D3C_E04_TRAINING_GRAPH_QUALIFICATION` row.
The first 262935 bytes (102 rows) remain identical, SHA256
`86614988e10545656dc6aeff36817fe239dd9b0e4685823119cc7e8455d4101b`.

The new preflight validates this stopped audit only. Its `--rebuild-index` mode
is the final repository write: it carries committed index metadata for unchanged
files and hashes only the four explicitly named new artifacts. It preserves CRLF
and never opens manifest targets. The ordinary index builder was deliberately not
invoked because its untracked discovery and manifest hashing violate this task's
narrower access boundary. This is an incremental metadata rebuild, not a fresh
rehash of untouched artifacts. Read-only integrity verification follows.

No benchmark image filename enumerated; no image bytes opened or decoded; no
manifest payload or sample record opened; no runtime data directory enumerated,
counted or file-statted. Public paper illustrations were viewed as literature.
No TRAIN/VAL/TEST scientific execution, benchmark training, checkpoint, synthetic
bank, commit or push. These are command-audit claims, not an OS-wide syscall trace.
