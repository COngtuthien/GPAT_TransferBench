# M6D3d — E04 training graph owner resolution

Decision: **OWNER_RESOLUTION_RECORDED**.
Contract: **E04_TRAINING_GRAPH_CONTRACT_RESOLVED**.
Graph/runtime qualification: **NOT YET QUALIFIED**.

The [M6D3d owner-resolution addendum](../../docs/spec/amendments/GPAT_TransferBench_v1_0_E04_Training_Graph_Owner_Resolution_Addendum_M6D3d.md)
and [machine-readable overlay](../../configs/amendments/e04_training_graph_resolution.yaml)
record all five owner resolutions plus the complete 29-component disposition
table. The earlier graph-resolution document is retained byte-for-byte as
M6D3c historical evidence; this new additive document supplies current authority.

| Owner resolution | Recorded result | Provenance |
|---|---|---|
| Encoder | F1=128x128x64, F2=64x64x96, F3=32x32x128; ConvBlocks k3c64s2,k3c96s2,k3c128s2 | PAPER, resolved; not a controlled width choice |
| P0 | zeros_like(P) for every spoof; negative prior contributes zero without evaluating 0/0; primary mask term and inpainting stay active | CONTROLLED_RECONSTRUCTION_BINARY_ONLY_P0 |
| Hard targets | Exact zero float32 32x32 spoof depth; L_S uses warped/synthesized ground-truth trace with paper stop-gradient | PAPER |
| Optimization | G Eq.23 → D L_D → G Eq.24 every minibatch; D LR=G/2; one G Adam state shared by two separate applies, separate D state; one iteration increment after Step 3 | PAPER schedule/LR, A4 Adam, owner-approved controlled state/iteration ownership |
| STDN | Reference only where PhySTD is silent; G_D_RATIO=2 not imported | Owner instruction |

The zero-P0 contribution is a deliberate controlled change to the unavailable
spoof-type-conditioned prior. It is not original-author behavior. There is no
attack-type input, replacement dataset-specific prior, epsilon workaround or
disabled inpainting branch. The paper-versus-author-PDF equation-number crosswalk
is recorded without changing frozen Eq.23/Eq.24 objectives.

## Scope and checks

Only documentation, a declarative overlay and a standard-library static
preflight were created. No methods implementation was edited. The preflight
validates all owner decisions, immutable input hashes, complete component-table
coverage, ledger integrity and index metadata. Its optional self-test rejects
14 in-memory invalid contracts covering provenance, widths, zero-prior handling,
active inpainting, label prohibition, hard targets/stops, ordering, optimizer
state, iteration/LR ownership and STDN scheduling. These checks use dictionaries;
they do not implement a graph or evaluate losses, tensors or optimizers.

Commands:

```sh
python3 -B tools/m6d3d_e04_training_resolution_preflight.py --self-test --rebuild-index
python3 -B tools/m6d3d_e04_training_resolution_preflight.py --self-test
```

No TensorFlow import/run, graph implementation/execution, optimizer application,
environment creation/probe or package installation. No benchmark data, manifest
payload/sample record, image filename/bytes or real depth target was accessed.
No model/checkpoint or synthetic bank was created. No commit or push.

## Immutable authority and bookkeeping

Starting committed `m6-baselines` HEAD:
`80dd758475f985377170308351ed2cf99a809d7d`; initial tracked worktree clean.
All prior M6D3b/M6D3c evidence, including their preflights and M6D3c addendum,
remain byte-identical. The E04 base, A3/A4, fixed geometry evidence and entire
frozen_config_snapshot remain unchanged. Exact protected hashes are in JSON.

Ledger: **103 → 104**, exactly one
`M6D3D_E04_TRAINING_GRAPH_RESOLUTION` row. The first 265194 bytes/103 rows
are preserved with SHA256
`354d907ecfbfebb757682d075e8184b4289b4858d308445d678040a7ab57dbea`.

ARTIFACT_INDEX is rebuilt LAST with CRLF: carry all 508 committed unchanged
rows as opaque metadata and hash only the five named new artifacts, yielding
513 rows. Unchanged targets, including manifests, are not opened or statted by
the index operation. The standard builder's untracked discovery is not used.
Only read-only verification follows the final index write. This is an incremental
metadata rebuild, not a fresh hash audit of every unchanged artifact.

## Final status

- E04_GEOMETRY_RUNTIME_QUALIFIED — retained from M6D3a, not rerun.
- E04_TRAINING_GRAPH_CONTRACT_RESOLVED.
- E04_TRAINING_GRAPH_NOT_YET_QUALIFIED.
- E04_TRAINING_RUNTIME_NOT_YET_QUALIFIED.
- E04 IMPLEMENTED_NOT_EXECUTED.
- CONTROLLED_ADAPTATION.

Contract validation is not graph qualification. The access statements reflect
the recorded commands and narrowly scoped evidence reads, not an OS-wide syscall
trace. No prior historical STOP decision or incident disclosure was rewritten.
