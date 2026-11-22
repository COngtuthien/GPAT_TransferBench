# M8 global deterministic smoke test: execution report

Date: 2026-11-22. Branch `m6-baselines`.

**Status: `M8_GLOBAL_SMOKE_PASSED_BINDING_PENDING`.** The global smoke test ran end to end on real TRAIN/VAL toy data in
all five stages (generator → bank → evaluator → metrics → plots). An independent replay reproduced it byte for byte, and it
was re-derived twice from raw evidence.

**Nothing is bound.** `configs/amendments/e01_e02_bank_milestone_ruling.yaml` keeps `smoke_test_record: null`, O3 is not
marked completed, and no production bank or scientific training was run. Every output is `SMOKE_NON_SCIENTIFIC`, and no
number below is a benchmark result.

## Authority

| Item | Identity |
|---|---|
| Smoke contract `configs/amendments/m8_smoke_test_contract.yaml` (OWNER_APPROVED, `181aa87`) | `13704edde0fd56966144063b9f1f66401264eb0e49d6220297591f04bc914ebb` |
| Evaluator contract `configs/amendments/m8_downstream_evaluator_contract.yaml` (OWNER_APPROVED, `181aa87`) | `67e4b62ddc6b81b8b9a8657e71b3979f6e6a186827183b052ae3d29b587fcbe0` |
| Smoke source authority (implementation; clean isolated clone on the GPU host) | `dd893016ddb0e6e828d867a69c2d7c18ba363834` |
| Environment lock `environments/m8_downstream.lock.json` | `70202661f1de3188aa7a6460a640982cea98074e1c41146ff516ac6fef3d7b9c` |
| Smoke record `outputs/audit/M8_SMOKE_TEST_RECORD.json` | see `M8_SMOKE_EVIDENCE_SHA256.txt` |

## Environment (`gpat-m8-downstream`, A_VENV_OVERLAY)

The environment is `venv --system-site-packages --without-pip` over an unchanged `gpat-m5`. Only six wheels were
installed, using `pip --no-index --no-deps --require-hashes --only-binary=:all: --no-compile`. Each wheel's SHA256 equals
the plan:

| Wheel | Version |
|---|---|
| matplotlib | 3.11.2 |
| contourpy | 1.3.3 |
| cycler | 0.12.1 |
| fonttools | 4.68.0 |
| kiwisolver | 1.5.1 |
| pyparsing | 3.3.3 |

The environment takes 56 MB.

| Gate | Result |
|---|---|
| V1 `gpat-m5` unchanged | PASS: pip-freeze SHA256 `a6067c98…` and full file-tree (path, size, mtime) SHA256 `87317fc7…` are identical before installation, after installation and at qualification |
| V2 versions | PASS: Python 3.11.16, torch 2.12.1+cu130, torchvision 0.27.1+cu130, numpy 2.4.6, pillow 12.3.0, pyarrow 25.0.1; the six added packages resolve only inside the overlay |
| V3 weights | PASS: `resnet18-f37072fd.pth`, SHA256 `f37072fd…07ec`, loads on CPU; fc.in_features 512 |
| V4 CPU determinism (**hard gate**) | PASS. Two separate processes ran the real runner (E01-type mixed batches, 2 epochs × 3 steps, the D6 schedule from lr 0, sampler reshuffles, VAL selection and holdout). Step log, epoch records, VAL and holdout prediction files and detector record are byte-identical |
| V5 plot determinism | PASS: 8 PNGs byte-identical across two processes |
| V6 lock | PASS: `environments/m8_downstream.lock.json` |

## Synthetic qualification (before real data)

Two full five-stage runs ran on a synthetic world from the clean clone. The world was sized so the D1 remainder-discard and
reshuffle paths ran: E00 discarded 120 real-spoof items, and E01/E02 discarded 8 real-spoof and 20 synthetic items.

- **Audit:** every gate passed, including byte-identical replay, independent re-derivation, coverage, firewall and a clean
  tree.
- **Record:** the synthetic record is `smoke_test_passed: false` by design
  (`M8_SMOKE_EVIDENCE/qual_smoke_record_candidate.json`).

Unit tests:

- `test_m8_bank_smoke_mode` (6): SMOKE authorization, global positions, O_NOATIME, root separation, and a CLI that cannot
  reach SMOKE mode;
- `test_m8_smoke_pipeline_units` (7): D8 and D10 vectors, the face reader, the audit-hook firewall, the toy manifests and
  the tables.

## Real preflight (`M8_SMOKE_EVIDENCE/real_preflight.json`)

The toy plan was re-derived from frozen TRAIN/VAL metadata and **equals the approved feasibility record**:

| Item | Value |
|---|---|
| TRAIN live | 96 frames |
| TRAIN real spoof | 256 frames |
| Toy frozen pairs | 186 |
| VAL select | 12 live / 32 spoof videos |
| VAL holdout | 12 live / 33 spoof videos |
| Full batches per epoch | 3 |
| E01 operators represented | 8 |
| Select/holdout overlap | none |

All 1,233 allowlisted faces were present and SHA256-verified against M2: 1,063 toy-role frames plus 170 further pair
targets and sources. The reads used O_NOATIME, with 0 atime changes and a clean firewall. Resources: 41.3 GiB disk free,
20.2 GiB RAM available, 20 CPUs at load 9.6 while E04 trained.

## Real smoke execution

**Run A** used `~/workdir/m8_smoke/real_A` and **replay B** used a fresh `real_B`, both from the clean clone at `dd89301`.
Each ran under `nice 19` / `ionice -c3` with CUDA hidden.

| Stage | Interpreter | Wall time (A) | Result |
|---|---|---|---|
| select | gpat-m8-downstream | 0.5 s | toy manifest `50ac8fb8…`; plan equals the feasibility record; 0 TEST rows |
| banks | gpat-m5 (locked; `verify_locked_environment`) | 72 s | E01 and E02 × seeds 42/1337/2026: 186/186 outputs each, 0 failures; file verification and full replay PASS; E01 operator = global position % 8; E02 k 40, n_eligible 159, residual ≤ 1e-8 |
| evaluator | gpat-m8-downstream | 123 s | E00 (32 live / 32 spoof), E01 and E02 (32 / 16 / 16, bank seed 42) at detector seed 42; 6 real optimizer steps each (first at lr 0); VAL select after each epoch; D5 chose epoch 1 for all three; the holdout was predicted once at the frozen threshold |
| metrics | gpat-m8-downstream | 0.5 s | T06/T08/T13/T13b (D10)/T14, curves and training logs, from prediction files only |
| plots | gpat-m8-downstream | 1.1 s | fig14–fig20 and fig24 from the tables; no placeholders |

Smoke-only holdout values (the TEST stand-in, seed 42, 6 steps; **not results**):

| Detector | ACER | AUC | EER |
|---|---|---|---|
| E00 | 119/264 | 61/132 | 7/12 |
| E01 | 131/264 | 50/99 | 1/2 |
| E02 | 71/132 | 199/396 | 17/33 |

These are near chance, as expected after 6 steps.

## Replay, re-derivation, firewall

- **Replay (S9 BITWISE_ALL_STAGES):** 1,721 deterministic artifacts were compared between A and B, with **0 differences**.
  They cover the toy manifest, every bank PNG and edit map, the bank manifests, the step logs, epoch and detector records,
  all prediction files, all tables and all PNG figures.
- **Independent re-derivation:** `tools/m8_smoke_audit.py` does not import the metrics module. It recomputed video
  aggregation, the section-15 sweep, D5 selection, ACER/AUC/EER/F1/TPR@FPR and the D4 decisions from the prediction files.
  Every T06 value matched, on the host and again on the laptop (Python 3.12, NumPy 2.5.3).
- **Laptop toy manifest:** built from frozen metadata on the laptop, it is byte-identical to the host's.
- **TEST firewall:**
  - TRAIN/VAL parquet row filters, so 0 TEST rows reached Python;
  - an allowlist audit hook over file opens under `faces_256` (0 violations, 0 directory listings);
  - O_NOATIME reads with a stat-verified atime ledger (0 atime changes).

  The bank stage opened 247 distinct faces and the evaluator 1,035, all on the allowlist.
- **Validator:** `gpatbench.downstream.smoke_evidence.validate_smoke_record` accepts the record (admissible, passed) on
  the host and on the laptop.

## E04 isolation

E04 seed 2026 kept running throughout: PID 435565, global_step 31127 → 32657+ during this work, checkout `18ba00e` clean.
The smoke used a separate clone (`~/workdir/m8_smoke_clone_dd89301`), separate roots (`~/workdir/m8_smoke/*`), CPU only
and low priority.

The faces were read with O_NOATIME so that E04's completion-audit atime gate (`H.val_zero`, VAL atime unchanged over its
run) is not disturbed. The overlay venv and every write live outside `gpat-m5` and the E04 checkout and runtime.

## Proposed production binding (NOT applied; needs a separate owner approval)

In `configs/amendments/e01_e02_bank_milestone_ruling.yaml`, recorded additively with an addendum:

```
"smoke_test_record": {"path": "outputs/audit/M8_SMOKE_TEST_RECORD.json",
                      "sha256": "<SHA256 of outputs/audit/M8_SMOKE_TEST_RECORD.json at the binding commit>"}
```

The binding approval should also:

1. Name the approved **production authority commit**. It must contain the smoke source `dd89301` and the record.
2. Record O3 `smoke_test_completed: true` additively. The M6D6jR file stays unchanged.
3. Harden `methods/common/bank_writer.load_production_authorization`, in a separate implementation commit, so that it
   requires:
   - `validate_smoke_record(record)` passes against the current approved contract SHA256 (`13704edd…`);
   - all five stages are PASS;
   - the record's bank coverage contains exactly E01 and E02 at seeds 42/1337/2026;
   - `mode == "real"`;
   - the record's `source.git_commit` is an ancestor of the production authority;
   - the record file's SHA256 equals the bound value, which guards against a stale or unrelated record.

   Today the gate checks only the file hash and `smoke_test_passed`.

Full E01/E02 production banks remain blocked until that approval. Learned methods (E03–E06c) still need their own
bank-stage acceptance (S7). GPAT stays deferred.
