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

## M1 — Dataset inventory (2026-09-18)

1. Verified start state (HEAD = origin/main = af0b2b8, clean). M1 set IN_PROGRESS.
2. Re-read spec §3.1–3.5, §25, App. A. Read local docs: MSU README.txt + protocol lists; SiW README.pdf/DRA.pdf; CASIA has none.
3. Evidence gathering: filename grammars, CASIA derived copies (flip/brightness), CASIA subject identity (visual, scratchpad only), CASIA HR_1 conflict, SiW lowercase-dir misread (M0 correction), decode behaviour.
4. Created project `.venv` (pandas, pyarrow, numpy, opencv-python-headless, PyYAML) → `environments/m1_inventory_laptop.lock.txt`.
5. Implemented `gpatbench/data/{base,casia_fasd,msu_mfsd,siwmv2,frames,inventory,report}.py` and `gpatbench/cli.py inventory`, plus `configs/frozen/{data_v1,attack_map_v1}.yaml`.
6. Runs:
   - trial run (aborted);
   - run A (completed; superseded after it exposed mid-stream decode failures and duplicate files);
   - final run B (8 workers) and rerun C (6 workers): byte-identical outputs.
7. Outputs: `manifests/{inventory,inventory_videos,raw_file_index}.parquet`, `outputs/audit/dataset_*.csv`, `unmapped_attack_tokens.csv`, `dataset_report.html`, `m1_inventory_{facts,run}.json`.
8. M1 → COMPLETE, with owner-review items (DEV-005…009, Q-12…Q-16) and M3 blockers recorded in STAGE_STATE.

Not done (by design): frame extraction, face detection, crops, caches, split, pairs, training, GPU access.

## M1 — Correction pass after owner review (2026-09-18, on top of ce12a58, which was not pushed)

1. Owner decisions:
   - CASIA code semantics and identities: train N, test 20+N (DEV-010, supersedes DEV-008);
   - DEV-007 and DEV-009 APPROVED;
   - DEV-005 exact rule requested;
   - Q-04 revised to an index-bounded rule.
2. Verified the owner mapping against all 600 CASIA sequences: no contradiction.
3. Official SiW-Mv2 sources checked (repo @8667dbc, supplementary):
   - `Paper` → print traced to `config_siwm.py`;
   - no video→subject mapping exists (Q-14 BLOCKED).
4. CASIA original-video search (read-only): none found (Q-15 unresolved).
5. Code, config and tests changed; `attack_map_v1` rev 2 and `data_v1` rev 2 snapshotted.
6. Runs D and E: byte-identical outputs; raw index identical to ce12a58; sample set unchanged.
7. M1 → COMPLETE (corrected). Pending owner review: DEV-005, DEV-006 implementation, Q-14/15/16.

## M1 — Dataset-resolution / final freeze pass (2026-09-18/19, base 112506b, not pushed)

1. DEV-005 APPROVED (owner), with toy tests.
2. DEV-006: independent decoder audit (PyAV 18.1.0) on the targeted videos and a global timestamp check of all 1,980 videos → continuity proven → APPROVED.
3. CASIA:
   - independent disk re-audit (0 manifest mismatches, all frames 112×112);
   - final original-source search: none found;
   - resize frequency audit (diagnostic);
   - DEV-011 adaptation proposed.
4. SiW:
   - official repo @8667dbc re-verified;
   - protocol entries traced to video stems;
   - no video→person mapping (Q-14 CASE C);
   - AdaFace audit not applicable (0 candidate groups);
   - duplicate audit (Q-16);
   - Paper → print re-confirmed.
5. Readiness matrix and resolution report; frozen config rev1 preserved in history with diffs; manifests unchanged.

## Consolidated narrative M0 → dataset policy freeze (written 2026-09-19; each step says WHY)

1. **M0 bootstrap** (1823aba, af0b2b8; pushed).
   - Spec identified and hashed; skeleton, registries, audit infrastructure.
   - *Why:* nothing scientific may run before provenance exists.
2. **M1 initial inventory** (ce12a58, unpushed).
   - Indexed all 125,841 local files; built a deterministic 8-frame sample (20,640 rows).
   - *Why:* the spec makes the local inventory the source of truth.
   - Two runs were superseded because they revealed decoder behaviour: a trial run aborted, and run A.
3. **CASIA label/subject correction** (112506b, unpushed).
   - The owner review showed HR_1 is live and test subjects are 21–50.
   - *Why:* the first adapter trusted the packager's folder labels and used a partition-local subject key.
   - The fix was verified against all 600 sequences.
4. **Decoder-index investigation** (be428bf).
   - PyAV was used as an independent reference and all 1,980 videos were audited.
   - *Why:* the owner rejected the "stop after 5 failures" rule. Index continuity had to be proven, not assumed.
   - It was proven; OpenCV `POS_FRAMES` was found to lag after a failure.
5. **SiW authoritative-metadata investigation** (be428bf).
   - Traced the official repo @8667dbc.
   - *Why:* Q-14 blocked M3 for SiW.
   - Protocol names turned out to be video stems; no person mapping exists; AdaFace was not used to invent IDs.
6. **CASIA resize-frequency audit** (be428bf).
   - *Why:* only 112 px crops exist locally, and GPAT works on high-frequency detail. The impact of 112→256 had to be quantified before any decision.
   - Result: roughly 5–6× lower relative high-frequency power.
7. **Owner decision: accept the controlled CASIA preprocessing** (DEV-011, this pass).
   - *Why:* no original source exists locally, and the adaptation is applied identically to all methods and disclosed.
8. **Owner decision: SiW video-disjoint fallback** (DEV-012) **and different-video/content pairing** (DEV-013), this pass.
   - *Why:* blocking SiW would remove 1,700 videos and 14 attack types from the benchmark. The limitation is disclosed rather than hidden, and exact duplicates are grouped to avoid content leakage.
9. **Dataset policy freeze** (this pass).
   - Artifact: `configs/frozen/dataset_protocol_policy_v1.yaml`.
   - *Why:* M2–M4 must follow one explicit, versioned policy. The allocator objective is deliberately left undefined (Q-01) to avoid a hidden scientific choice.
10. **Future M2**, after owner review and push: CASIA via DEV-011; MSU and SiW via spec §4.
11. **Future M3:** first an owner-approved allocator objective, then per-dataset grouped 70/15/15 splits with the policy's distribution audit.

## M2A — Auxiliary model resolution, contracts, environment, smoke (2026-09-19; base bc947fd = origin/main)

1. **Verified state and re-read the spec §4, the policy, data_v1 and the registry.**
   - *Why:* M2 must follow the frozen artifacts, not memory.
2. **Traced official sources at pinned commits** (InsightFace, FaceXFormer, AdaFace, CVLFace) and the HF APIs, without downloading any weights.
   - *Why:* model identities must be authoritative (owner provenance hierarchy).
   - Result: FaceXFormer weights and code verified exactly. The AdaFace local file is identified exactly as CVLFace IR-50 WebFace4M. The two SCRFD files are identified only structurally.
3. **Built an isolated M2 environment** `~/.venvs/gpatbench-m2` (CPU torch 2.14.0; no CUDA on the laptop) and froze its lock before any smoke run.
   - *Why:* DEV-004 condition 3.
4. **Implemented `gpatbench/preprocess/`:** contracts, a SCRFD reimplementation, frame extraction, and the FaceXFormer/AdaFace adapters with code and weight hash guards.
5. **Ran the smoke:** 24-sample deterministic manifest; runs 1–2, then a code refactor, then runs 3–4 on the final code.
   - *Why:* validate the full path end-to-end and its determinism before any full run.
   - Result: 24/24 OK; 208/208 files byte-identical.
6. **Opened questions** Q-18…Q-23 and marked Q-02/Q-03 as owner decisions. M2 → **BLOCKED**.
   - *Why:* the spec and the official model contracts leave genuine gaps (AdaFace colour and alignment, crop borders, logits storage). Choosing silently would be a hidden scientific decision.
7. **Next:** the owner resolves the blockers; the contract is then frozen (`configs/frozen/preprocess_v1.yaml`), the M2B smoke is repeated on the chosen host, and only then does full M2B run.

## M2A — Owner decision resolution and contract freeze (2026-09-19; base 0688103a, additive, cwd `/home/cong/GPAT_TransferBench`)

1. **Verified the starting state** (HEAD `0688103a`, clean tree, 1 commit ahead of `origin/main`, M2 BLOCKED)
   and re-read every M2A artifact plus the relevant frozen-spec sections before touching code.
2. **Q-02 — verified the owner's SCRFD selection against the official source.** Downloaded the official
   InsightFace release pack `antelopev2.zip` into a non-Git cache and compared its member
   `antelopev2/scrfd_10g_bnkps.onnx` with the local file: **byte-identical**.
   - *Why:* the previous pass could only identify the file structurally. The owner's selection stands either
     way, but a byte identity removes the last inference from the detector's provenance.
   - `det_2.5g.onnx` stays in the registry as an unselected candidate and is now refused by hash in code.
3. **Q-03 — obtained the original AdaFace R50 / WebFace4M checkpoint** from the link published in the official
   README at the pinned commit, into the external model cache (never in Git). Hashed, strict-loaded into the
   official `ir_50`, recorded.
   - *Why:* the owner chose the original release so the spec's "official BGR" wording is followed literally.
   - **Finding:** the CVLFace export and this checkpoint are the *same trained model* — 466/467 tensors
     bit-identical, and `input_layer.0.weight` is an exact channel-axis reversal. So "original + BGR" and
     "CVLFace + RGB" are the same function; the decision restores spec compliance without changing the
     identity space. Verified end-to-end: cos = 1.0 (to float32) on every unchanged canonical face.
4. **Q-18 / Q-19 — froze the AdaFace adapter** as module constants, not parameters: canonical 256 → 112
   INTER_AREA → RGB→BGR → official `to_input` arithmetic → IR-50 → 512-D L2. No MTCNN, no alignment, no
   second detector (**DEV-014**), recorded transparently as an adapter deviation rather than a claim to run
   the full standalone AdaFace photo pipeline.
5. **Q-22 — implemented the owner's square/zero-padding semantics.** The requested square is built first and
   never shrunk or shifted; only the source read region is clamped; outside pixels are constant zero
   (**DEV-016**). Exhaustive synthetic unit tests plus real border evidence.
6. **Q-23 — froze the scientific representation and piloted the storage.** Full float32 11×224×224 logits and
   a uint8 mask derived from the *stored* logits; lossless deterministic `npy1+shuffle4+zstd` shards.
   - *Why:* the owner forbids any lossy reduction, so the size question had to be answered by compression,
     not by dtype. Measured on the existing 24 smoke logits: 45.57 GB → 34.74 GB, exact reconstruction 24/24.
   - `zstandard` 0.25.0 was added to the M2 environment for this and only this; lock and runtime facts regenerated.
7. **Promoted the contract**: `configs/proposed/preprocess_v1.proposed.yaml` (kept byte-unchanged as history)
   → `configs/frozen/preprocess_v1.yaml` + byte-identical snapshot. A test asserts the only remaining null is
   `facexformer.parsing_class_names` (Q-21, which blocks M9, not M2B).
8. **Re-ran the same 24-sample manifest** (deliberately not reselected) on the frozen contract, twice, from
   clean directories with the final code: `runC`/`runD`, 24/24 OK, **210/210 files byte-identical**.
   - Only 3 canonical faces differ from the previous pass — exactly the SiW crops that touch a frame border.
9. **M2 → IN_PROGRESS**, phase `M2A_COMPLETE_AWAITING_OWNER_REVIEW_FOR_M2B`. M2 is deliberately **not**
   COMPLETE and M3 remains NOT_STARTED; no full preprocessing, cache or pair work was started.

## M2B — Storage preflight and hard safety gate (2026-09-19; base d92239bd, cwd `/home/cong/GPAT_TransferBench`)

1. **Verified the starting state**: HEAD == `origin/main` == `d92239bd` (the owner pushed both M2A commits),
   clean tree, M2 `IN_PROGRESS` / `M2A_COMPLETE_AWAITING_OWNER_REVIEW_FOR_M2B`, M3 `NOT_STARTED`.
2. **Re-read the frozen contracts** (spec §0.2/§3.4/§4/App. A, `data_v1`, `dataset_protocol_policy_v1`,
   `preprocess_v1`, the registry and every M2A report) before any execution.
3. **Verified the frozen auxiliaries** against `models/registry.yaml`: SCRFD, FaceXFormer and AdaFace all
   hash-match. **Verified the frozen M1 inputs** are unchanged (3 manifests + 2 frozen configs).
   - *Why:* §3 and §25 require both before a full run is even considered.
4. **Ran the mandatory storage preflight** (`tools/m2b_storage_preflight.py`).
   - Basis: the exact frozen M1 counts (20,640 samples over 6 dataset/resolution classes, taken from
     `inventory_videos.frame_sizes`) plus per-component byte measurements from the existing deterministic
     24-sample smoke. No new scientific sample was processed for the estimate.
   - Every per-sample size uses the **maximum** observed value, with the mean reported alongside.
   - *Why the per-resolution split:* CASIA is a 112×112 image sequence, while 11,760 of the 13,600 SiW
     samples are 1920×1080 video frames. One pooled average would have hidden the dominant term.
5. **Established the CASIA frame policy from the contract, not by guessing**: CASIA's M1-selected source is
   already a lossless canonical PNG, the frozen route consumes it directly and records
   `frame_png_sha256 = source_file_sha256`, and the frame-extraction block describes a video decoder. So no
   duplicated CASIA frame artifact is produced, and only 15,840 MSU/SiW frames are persisted.
6. **Result: `BLOCKED_BY_STORAGE_CAPACITY`.** `data/processed` and `cache` resolve to the project
   filesystem (`/dev/nvme0n1p7`, ext4, `/home`) with **36.50 GiB** free. M2B needs **63.92 GiB**
   (51.92 persistent + 2.00 temporary peak + the owner's 10 GiB reserve) — a **27.42 GiB** shortfall.
   - The two terms missing from the earlier ≈37.5 GB figure: **16.27 GiB** of spec-required lossless
     MSU/SiW frame PNGs, and the fact that that figure was compared against the *external* volume.
   - Not marginal: even mean sizes, compressed masks and zero allowances still need 48.16 GiB.
7. **Stopped before full preprocessing**, per the owner's hard rule. The dataset was **not** partially
   processed, `data/processed` and `cache` remain empty (test-enforced), and no cache component was
   redirected to another filesystem. Remediation options are reported for the owner, not acted on.
8. **Added the tests that are meaningful at this gate** (preflight arithmetic, the 10 GiB reserve, the gate
   decision, the CASIA frame policy, bounded streaming shard construction, and the shard-index integrity
   audit that the M2B cache audit will reuse). Tests for resume semantics and full-cache completeness are
   deliberately not written as stubs: they belong with the M2B execution code, which is not implemented
   while the gate blocks.
9. **Next:** the owner chooses a storage remedy; then the preflight is re-run and must return `PASS`
   before M2B executes. If the outputs move to another filesystem, the M2A determinism smoke must be
   repeated on that path first.

## M2B — Storage relocation and full preprocessing (2026-09-19; base 1fdb73c8, cwd `/home/cong/GPAT_TransferBench`)

1. **Verified the starting state**: HEAD == `origin/main` == `1fdb73c8`, clean tree, M2 `IN_PROGRESS` /
   `M2B_BLOCKED_BY_STORAGE_CAPACITY`, M3 `NOT_STARTED`.
2. **Audited the external volume before writing anything** (`findmnt`, `df -B1`, `stat -f`) and proved it
   supports `fsync`, atomic `rename`, directory `fsync` and case-sensitive names. `/dev/nvme0n1p5`, ntfs3,
   65.11 GiB free, writable by the current user.
   - *Why:* the scientific outputs were about to move to a different filesystem type; correctness of the
     atomic-write protocol had to be established on it first, not assumed.
3. **Created the owner-named runtime root** and an execution config
   (`configs/execution/m2b_laptop_external_storage.yaml`, infrastructure only, recorded as **DEV-017**).
   The repository stayed in `/home`; no project directory was replaced by a symlink; the runner refuses a
   config whose roots escape the approved runtime root or whose frozen-contract hash has changed.
4. **Re-ran the storage preflight against the new physical roots** with the same conservative model and
   the same 10 GiB reserve — the reserve was not reduced to force a pass. Result **PASS**: 65.11 GiB free
   against 63.92 GiB required. The previous blocked preflight was left untouched, and
   `M2B_STORAGE_DECISION_TRAIL.md` records PREVIOUS → OWNER DECISION → NEW.
   - Flagged honestly: the margin is only ~1.19 GiB, so free space is checked at every chunk boundary.
5. **Proved relocation changes nothing scientific**: re-ran the frozen 24-sample smoke manifest
   (sha256 `138f5929…`, not reselected) onto the new filesystem and compared with the frozen `/home` run —
   **210/210 files byte-identical**, results equal except timing.
6. **Implemented the resumable execution engine** (`gpatbench/preprocess/m2b.py`, `tools/m2b_run.py`)
   *before* processing anything: deterministic ordinal → shard assignment that is a pure function of the
   frozen manifest, per-sample states, append-only crash-safe state logs, hash-verified artifact reuse,
   atomic writes everywhere, and bounded streaming shards (one in-flight shard per field; no uncompressed
   whole-cache copy ever exists).
   - Grouping a video's eight samples into one sequential decode pass (`read_video_frames`) keeps the
     DEV-006 loop-counter identity while cutting decode work ~4x; proven equal to the per-sample decoder.
7. **Validated the engine on a 2-shard pilot before the full run**, then validated resume for real:
   an unchanged chunk skipped in 0.0 s; a deliberately corrupted face was detected on resume, the chunk was
   rebuilt, and both the face and the shard came back **byte-identical** with the event recorded.
   - *Why:* "resume never silently accepts a corrupted output" had to be demonstrated, not asserted.
   - This exposed a real gap first: the fast skip trusted finalized chunks without re-verifying face
     hashes. Closed before the full run.
8. **Ran the full frozen inventory**, 2 independent worker processes over disjoint shards, each with the
   frozen determinism settings. **20,640/20,640 accounted**: 20,615 COMPLETE, 25 FAILED — all
   `SCRFD_NO_FACE` (SiW), the correct no-fallback outcome. ~2.4 h wall clock.
9. **Audited and validated**: full block-level cache integrity **PASS** across all 7 fields (567 shards,
   no duplicate, orphan, overlapping, mis-shaped, non-finite or hash-failing entry); deterministic final
   validation re-computed 72 samples (the 24 frozen smoke plus 48 hash-selected, none hand-picked) from
   raw sources and found **max abs difference 0.0** against the stored artifacts.
10. **M2 → COMPLETE / FINALIZED.** M3 remains `NOT_STARTED`; no split, pair or training work was started.
    Q-21 stays deferred to M9 as agreed.

## PRE-M3 — M2 failure audit and Q-01 allocator freeze (2026-09-19; base 493d6ed1, no split created)

1. **Verified the starting state** (HEAD == `origin/main` == `493d6ed1`, clean tree, M2 COMPLETE/
   FINALIZED, M3 NOT_STARTED) and confirmed no M2 scientific process was still writing — only stale
   watcher shells from the previous pass, which are not workers.
2. **Audited M2's 25 failures at canonical-video level** (read-only; nothing in the external runtime
   volume was touched). All 25 are `SCRFD_NO_FACE` on SiW, spread over **23** distinct videos, at most
   **2** frames each, leaving at least **6** usable frames per affected video.
   - **Zero-usable-video hard gate: PASS** — 0 of the 2,580 canonical videos lost all their samples,
     so no video silently leaves the benchmark and no deletion/substitution policy is needed.
   - *Why this mattered first:* if any video had ended with zero usable frames, Q-01 could not be
     frozen — it would have been an owner policy question, not an allocator question.
3. **Froze the M3 population contract**: the allocator balances **canonical videos**, each weighing
   exactly one video regardless of how many frames survived; only M2 COMPLETE rows become usable
   image rows; FAILED rows stay provenance-only.
   - *Why:* balancing on surviving frame counts would let a detector failure quietly reweight the
     benchmark, making a 7-frame video "lighter" than an 8-frame one.
4. **Resolved Q-01 as a lexicographic contract** (P0 constraints → P1 video ratio → P2 binary → P3
   attack_macro → P4 attack_raw → P5 seeded tie-break), each level pinned as an exact equality before
   the next is minimized, so no later priority can worsen an earlier one.
   - *Why lexicographic rather than one weighted sum:* the spec names the terms but not their
     weights; a single weighted objective would have buried an unchosen trade-off in a number.
5. **Made the objective exactly representable.** The idealised per-category normalisation is rational
   and its exact integer form needs `lcm(N_c) ≈ 1.2e17` at the SiW raw level — unrepresentable in the
   float64 arithmetic any MILP backend uses. The frozen objective is therefore a fixed-point integer
   form (`W_c = 10**6 // N_c`) declared as *the definition*, with its divergence bound stated rather
   than hidden.
6. **Found the structure that makes an exact optimum cheap:** groups with an identical category
   profile are interchangeable, collapsing 50/35/1694 allocation groups to **1/1/16** profile classes.
   The real problems become 3–48 integer variables, and HiGHS proves a global optimum at every level
   in under 3 s.
7. **Pinned both sources of arbitrariness.** The optimal *count vector* is made unique by minimising
   each column in a seeded order (iterating splits naturally would have systematically starved VAL);
   the *concrete groups* are then ordered by the frozen seeded hash. Tested against permuted inputs,
   repeated runs and three `PYTHONHASHSEED` values in fresh interpreters.
8. **Measured feasibility on the real metadata without creating a split.** Achieved shapes: CASIA
   70.000/16.000/14.000, MSU 71.429/14.286/14.286, SiW 70.000/15.000/15.000 (group granularity, not
   allocator weakness, explains CASIA/MSU). No category at any level has fewer than three allocation
   groups. The assignment produced during the measurement was **discarded**; no membership was
   persisted and `manifests/split_v1.parquet` does not exist.
9. **M3 stays NOT_STARTED.** Allocator code and a frozen config do not start the milestone; M3 begins
   when the authoritative split is created.
