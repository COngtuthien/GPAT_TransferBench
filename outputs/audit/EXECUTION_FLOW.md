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
   *clarification appended 2026-09-20: "567 shards" means 567 **data shard files**, i.e. 7 fields ×
   81 shard groups; see `M2_CACHE_COUNT_RECONCILIATION.md` for every unit*,
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

## M3 — Authoritative split execution (2026-09-19; base d0111c3a)

1. **Verified the starting state** (HEAD == `origin/main` == `d0111c3a`, clean, M2 COMPLETE/FINALIZED,
   M3 NOT_STARTED, `split_v1.yaml` sha `9dc04ede…`, 245 tests PASS) and **recomputed the M2 population
   from the manifests** rather than trusting the earlier report: 20,640 sampled, 20,615 COMPLETE,
   25 FAILED (all `SCRFD_NO_FACE`, SiW), 0 zero-usable canonical videos.
2. **Resolved the `sha256` lineage without inventing a meaning.** M1 filled the column for CASIA
   (`original_frame_bytes`) and left MSU/SiW as `PENDING_M2_CANONICAL_PNG`; M2 produced exactly that
   canonical frame PNG, so the placeholder is resolved with M2's `frame_png_sha256` and `sha256_kind`
   records which lineage applies. For CASIA the two agree on all 4,800 rows (asserted at build time).
3. **Proved the profile-class reduction before using it.** For every class, the per-group contribution
   vector — canonical-video count plus binary/macro/raw counts — must be identical, since the solver
   optimises per-class counts. Group size is part of the signature because it enters P1 directly. All
   three datasets pass; a test injects a forced collision to show the checker actually refuses one.
4. **Ran the frozen allocator per dataset** (never pooled) through `python -m gpatbench.cli split`,
   which is the same code path as the library call — there is no execution route that bypasses the
   frozen allocator, and the CLI refuses a config that is not the frozen one.
5. **Wrote `manifests/split_v1.parquet`** with 20,615 rows: only M2 COMPLETE samples, spec columns
   first, canonical order `dataset > video_id > frame_index > sample_id`, and frozen Parquet writer
   settings. A small `split_groups_v1.parquet` records the group-level assignment.
6. **Audited leakage from the written artifact**, not from memory: CASIA and MSU subject/video/sample
   intersections empty; SiW content-group/video/sample intersections empty and all six exact-byte
   duplicate groups intact. SiW subject leakage is reported as **N/A** with its reason — the split is
   canonical-video / content-group-disjoint and is never described as subject-disjoint.
7. **Re-ran the whole allocator three times in fresh interpreters**, twice with deliberately permuted
   input metadata and under three different `PYTHONHASHSEED` values. All four runs produced a
   **byte-identical** manifest, so the hash was frozen rather than merely the assignment.
   - *Why the permuted runs matter:* an assignment can be stable while serialization is not; requiring
     identical bytes is what makes the manifest hash citable.
8. **Froze `outputs/audit/split_v1.sha256`** with the manifest hash, the schema signature, the writer
   settings, the split-config hash and the hashes of every frozen input the split was derived from.
9. **M3 → COMPLETE / FINALIZED.** M4 remains NOT_STARTED; no pairs were built and nothing was trained.

## M4 — Pre-flight: pair contract analysis (2026-09-19; base b0c59e1e, no pairs written)

1. **Verified the starting state** (HEAD == `origin/main` == `b0c59e1e`, clean, M3 COMPLETE/FINALIZED,
   split manifest sha `fb9aeb36…`, 20,615 rows, 278 tests PASS) and re-read the spec's §6 pairing
   contract, §8.6 DSDG and §8.7 DiffFAS method cards rather than working from memory.
2. **Separated what the spec settles from what it does not.** Settled: build from TRAIN *and* VAL
   (App. A `pairs_val = build_common_pairs(split.VAL)`; §19 needs `val_pairs_v1.parquet`), no TEST
   pairs, one pair per spoof source, the 64 cap, the 0.50/0.30/0.20 weights, minimum `d_pair`, the
   lexical tie-break and the seed. Target reuse is left unconstrained, so no one-to-one matching was
   invented.
3. **Measured feasibility before designing anything.** TRAIN has 8,838 spoof sources and VAL 1,905;
   **zero** sources in any dataset or split lack an eligible target, so nothing would have to be
   skipped. Every source has at least 64 eligible targets (tightest: MSU VAL at exactly 64), which
   turns the candidate sampler from a corner case into something that decides the evaluated set of
   *every* pair.
4. **Found that three of the four minimised quantities are undefined**, and quantified each rather
   than asserting it mattered: the pose norm changes the 0.50-weighted term by ≈1.47× (L1/L2 median
   ratio) and per-dataset std differs sharply from pooled; BT.601 vs BT.709 luminance differs by up
   to 0.0132 per image on a ≈0.55 range.
5. **Hit a harder problem in the scale term.** "log face-box area ratio" presumes a face box, but
   CASIA has **none** — the approved DEV-011 route runs no SCRFD, so 0 of 4,800 CASIA rows carry a
   bbox while CASIA supplies 2,520 of the 8,838 TRAIN sources. That is a missing quantity, not a
   choice between conventions, and every candidate resolution changes the metric for a large share of
   the benchmark.
   - *Why it was not patched:* treating the 112×112 frame as the box would silently make `d_scale`
     identically zero for every CASIA pair, which is a scientific change disguised as a default.
6. **Built the machinery so that a silent default is impossible.** `gpatbench/pairs/common.py` takes
   every contested convention as a *required* policy field with no default, so no pair can be
   computed without stating which convention produced it, and `PairMetricPolicy.frozen` stays False.
7. **Audited the native methods.** CASIA and MSU have full TRAIN identity coverage (35/35 and 25/25
   identities with both live and spoof); SiW has zero, so DSDG identity pairing and DiffFAS
   same-identity reconstruction are not instantiable there. Binary variants do not rescue it — the
   missing information is identity, not style — and DEV-013 was explicitly not stretched from common
   pairing to a same-identity requirement.
8. **Recorded six questions** (Q-24..Q-27 blocking, Q-28/Q-29 native scope) with proposals, put the
   contract in `configs/proposed/pairs_v1.proposed.yaml` rather than `configs/frozen/`, and **created
   no pair manifest**. M4 remains NOT_STARTED.

## M4 — Pre-flight owner decisions: Q-24 … Q-27 resolved and frozen (2026-09-19; base 255f70d4)

1. **Verified the starting state** (HEAD `255f70d4`, one commit ahead of `origin/main`, clean, M4
   NOT_STARTED, split sha `fb9aeb36…`, 321 tests PASS) and re-read the preflight evidence rather
   than working from memory.
2. **Q-24 — candidate selection.** Frozen as a two-stage SHA-256: a per-source seed digest, then a
   per-candidate digest over the raw seed bytes, ranked as an unsigned big-endian 256-bit integer
   with lexical `target_sample_id` as a defensive collision tie-break. Eligible targets are put in
   lexical order *before* any hashing, so the ranking cannot inherit input order. No PRNG.
   - `pair_id` was frozen with it (`PTR%06d` / `PVA%06d` over dataset then source id) and is assigned
     **after** membership is final. *Why that ordering matters:* spec §8.1 seeds FAS-Aug from
     `SHA256(pair_id + global_seed)`, so letting selection see `pair_id` would close a loop.
3. **Q-25 — pose.** Per-dataset z-score on TRAIN rows only, population std (`ddof=0`), Euclidean L2
   with no √3 division; a degenerate std is a hard error, VAL reuses TRAIN statistics, TEST never
   contributes. Pooling was rejected on measured grounds: SiW yaw std 0.283 against CASIA 0.052.
4. **Q-26 — scale.** The visible (frame-clipped) SCRFD box as a fraction of the original frame area,
   taken *before* the 1.25× expansion, padding and 256 resize, with `d_scale = |ln f_t − ln f_s|`.
   Normalising by frame area is what makes MSU (640×480) and SiW (1920×1080) comparable.
   - **CASIA has no detector box at all**, so the owner fixed `face_area_fraction = 1.0` and hence
     `d_scale = 0` exactly, recorded as **DEV-018**. SCRFD was not rerun, no bbox invented, no
     landmark-derived pseudo-box. The weights are deliberately **not** renormalised, and the honest
     consequence — CASIA pairs ranked by pose and luminance alone — is written into the deviation
     rather than left implicit.
5. **Q-27 — luminance.** BT.601 `Y` on the frozen canonical 256×256 RGB face, channels in [0,1],
   full-image mean including the Q-22 zero-padded pixels, absolute difference. No OpenCV YCrCb path
   and no BT.709; "normalized" fixes the range and nothing else.
6. **Made the decisions checkable rather than asserted.** The tests now pin analytically known
   values: pure-red/green/blue Y means of exactly 0.299/0.587/0.114, a z-score of exactly 1, a
   `d_pose` of exactly √12, a `d_scale` of exactly `ln 2`, identical area fractions at different
   resolutions, and the frozen digest construction byte for byte.
7. **Ran the frozen contract on real data** (deterministic sample, all candidates evaluated): every
   component finite and non-negative in every dataset × split cell, and CASIA `d_scale` unique values
   = `[0.0]` exactly, confirming DEV-018 end to end. No formula was changed from these distributions
   and TEST was never inspected.
8. **Froze `configs/frozen/pairs_v1.yaml`** with a byte-identical snapshot, preserving
   `configs/proposed/pairs_v1.proposed.yaml` unchanged as pre-decision history. No common-pair field
   remains null/TODO/OWNER_DECISION_REQUIRED.
9. **Created no pair manifest and no train-stats artifact.** M4 remains NOT_STARTED; Q-28/Q-29
   (native DSDG/DiffFAS scope for SiW) stay open and explicitly non-blocking for common pairs.

## M4 pre-execution correction (2026-09-20) — still NOT_STARTED

1. **Stated the Q-24 candidate preimage in byte terms.** The frozen rule hashes the **RAW 32-byte**
   source-seed digest concatenated with `UTF8("|gpatbench.pair.candidate.v1|" + target_sample_id)`.
   The hex-text variant is now named as forbidden in the config and guarded by
   `TestQ24Preimage`. Describing the rule merely as "two-stage SHA-256" is no longer sufficient.
2. **Found the implementation was already correct.** `gpatbench/pairs/common.py` needed no change.
   `tools/m4_q24_preimage_audit.py` imports the module *as committed at `e4d167b`* and re-runs
   selection on the same deterministic sample: **0 of 240** candidate-64 sets changed. The
   counterfactual hex variant would have changed **200 of 240 (83.33%)**. The defect was in the
   config prose and in §3 of the 2026-09-19 owner report, not in executable code.
3. **Classified it honestly** as `PRE_EXECUTION_CONTRACT_IMPLEMENTATION_CORRECTION`, not a new
   scientific decision: the owner contract did not change, and no pair manifest, train-stats artifact
   or native manifest existed to invalidate.
4. **Resolved Q-28 and Q-29 for M4 manifest scope.** Both native manifests contain **CASIA + MSU
   only**; SiW is `NOT_INSTANTIABLE_MISSING_SUBJECT_ID` with `native_identity_pair_coverage = 0`,
   recorded in the audit coverage table and never as manifest rows. DIFFFAS-BIN does not rescue SiW.
   The later M6 adaptation strategy is explicitly **not** settled by this.
5. **Reasserted the DEV-013 boundary.** It governs the common SiW pairing rule only — different video
   AND different exact-content group — and is never a same-person claim or a same-identity
   substitute for native pairing.
6. **Reconciled the M2 cache counts** read-only: 81 shard groups × 7 fields, 20,615 entries per
   field, 567 data shard files + 567 index files = 1,134 physical cache files. The earlier "567
   shards" and the M4 report's "972 / 162" counted different units and were both arithmetically
   right; `M2_CACHE_COUNT_RECONCILIATION.md` defines every unit. No M2 artifact was modified.
7. **Created no manifest of any kind.** M4 is still NOT_STARTED.

## M4 execution (2026-09-20) — common pairs COMPLETE, native pairs BLOCKED

1. **Verified the starting state before touching anything**: HEAD `443614b`, local `main` ==
   `origin/main`, clean tree, M4 NOT_STARTED, 359 tests PASS, and the population recomputed from
   `split_v1.parquet` rather than trusted from the preflight report (20,615 rows; TRAIN spoof
   2,520 + 1,200 + 5,118 = 8,838; VAL spoof 576 + 240 + 1,089 = 1,905 — all matching).
2. **Fitted the TRAIN pair statistics** per dataset on TRAIN rows only, population std `ddof=0`,
   float64, and froze them as `manifests/pair_train_stats_v1.json` under an explicit canonical-JSON
   policy (UTF-8, LF, sorted keys, ASCII-escaped, NaN/Inf rejected, shortest round-trip float repr,
   one trailing newline). VAL reuses them; TEST never contributed and was never read.
3. **Built the authoritative common manifests** with the frozen Q-24 raw-byte candidate ranking and
   the Q-25/Q-26/Q-27 distances: `pairs_train_v1.parquet` (8,838 rows) and `val_pairs_v1.parquet`
   (1,905 rows), one row per spoof source, written atomically with the frozen Parquet writer.
4. **Audited exhaustively** (`tools/m4_audit.py`, PASS, 0 failures over all 10,743 pairs): structure,
   split boundaries, TEST absence, M2 FAILED exclusion, CASIA/MSU different-subject, SiW DEV-013
   different-video AND different-content-group, candidate cap, distance recomputation, and
   winner-is-the-minimum re-derived independently. CASIA `d_scale` unique values = `[0.0]`.
5. **Caught a real determinism defect at the rerun gate.** Shuffling the input changed membership
   because `fit_pose_stats` stacked TRAIN rows in arrival order, so the population std varied in the
   last ulp and could flip a near-tied target. Fixed by stacking in canonical `sample_id` order; no
   Q-25 decision changed and nothing was invalidated, because no manifest had been frozen yet. The
   artifacts were regenerated with the corrected fit.
6. **Proved determinism**: four fresh processes (canonical and shuffled input × `PYTHONHASHSEED`
   0/1/424242) reproduced all three files byte-identically, with rows, schema, membership and
   `pair_id` compared separately.
7. **Wired `python -m gpatbench.cli build-pairs`** onto exactly the same code path, with an
   `--audit-only` mode that cannot change membership and a hard refusal of any non-frozen config.
8. **Stopped before the native manifests.** `third_party/registry.yaml` pins no commit for DSDG or
   DiffFAS (`pinned_commit: null`, `url_verification: UNVERIFIED`) and `methods/dsdg` /
   `methods/difffas` hold only `.gitkeep`, so the official pair-construction semantics cannot be
   inspected. No sampling rule was invented and no native manifest was written, not even an empty
   one. CASIA (35/35 identities) and MSU (25/25) are supported-but-blocked; SiW-Mv2 stays
   `NOT_INSTANTIABLE_MISSING_SUBJECT_ID` with coverage 0.
9. **Left M4 at IN_PROGRESS**, phase `COMMON_PAIRS_COMPLETE_NATIVE_PAIR_SOURCE_BLOCKED`. The
   milestone is not marked complete merely because the common pairs succeeded. M5 remains
   NOT_STARTED; nothing was trained and no synthetic image was generated.

Artifacts: `pair_train_stats_v1.json` `a7ccabb0…`, `pairs_train_v1.parquet` `a5e4fdae…`,
`val_pairs_v1.parquet` `84d12491…`.

## Owner Protocol Amendment A1 (2026-09-20) — M4 COMPLETE under the amended main-track rule

1. **Verified the starting state**: HEAD `f15cd443`, clean tree, M4 `IN_PROGRESS`
   (`COMMON_PAIRS_COMPLETE_NATIVE_PAIR_SOURCE_BLOCKED`), M5 NOT_STARTED, and the three frozen common
   artifacts byte-unchanged (`a7ccabb0…`, `a5e4fdae…`, `84d12491…`).
2. **Recorded the owner's scientific decision as a versioned, hashed amendment** rather than editing
   the frozen specification, which stays byte-identical. Amendment A1 sha256
   `03828716def5e535d82445974972bf71a5c8ecc60392fac4b884bcbe060e3472`.
3. **Pinned the two official sources** the owner authorized — FaceX-Zoo `16b793a7…` (sparse, DSDG
   only) and murphytju/DiffFAS `23f40519…` — into a git-ignored cache, source only. One stray binary
   checkpoint that arrived with the sparse checkout was removed; **no model weights were
   downloaded**. Exact commits, trees and per-file sha256 + blob ids are in
   `third_party/source_pins.json`. Pinning does not start M6.
4. **Read the DSDG relation and every loss from the pinned code.** `GenDataset_s` indexes spoof
   frames and draws the live partner with `random.choice` over the same OULU **user id**
   (`make_train_list.py: label = video_name[4:6]`) — same-subject, online, random. Of the seven loss
   terms, exactly one, `loss_pair`, mathematically requires the two images to be the same person;
   `loss_ip` compares each reconstruction with its own input and does not.
5. **Froze DSDG-BIN-IDFREE (E06c, DEV-020)**: the common fair pair becomes the training relation and
   `lambda_pair = 0`; everything else is retained. The binary collapse makes `loss_cls`
   mathematically degenerate (`CrossEntropyLoss` over one logit), which is disclosed rather than
   patched, with no substituted supervision. The relation is `pairs_train_v1.parquet` itself — an
   explicit decision, because a separate manifest would be a redundant projection plus three
   constants and a second copy could drift.
6. **Traced the DiffFAS `use_pair` branch and proved it numerically.** Under `use_pair=false` the
   content image is absent from the model input, is not the conditioning, is not the target, does not
   affect the noise and does not enter the variational term. Running the official `training_losses`
   at the pinned commit with everything fixed but the content tensor gives bit-identical model input
   and losses (`max_abs_loss_diff = 0.0`), while the same probe detects the dependency under
   `use_pair=true`. `content_training_role = INERT_API_PLACEHOLDER`.
7. **Froze DIFFFAS-BIN-IDFREE (E07c, DEV-021)** and built its Track-A manifest: 8,838 rows
   (CASIA 2,520 · MSU 1,200 · SiW 5,118), one deterministic binary style guide per TRAIN spoof GT by
   raw-byte SHA-256 ranking, `style_id = SPOOF_BINARY`, `use_pair = false`. Self-guides are inside
   the official support, so the 3 that occur are reported and not adjusted away.
8. **Proved determinism** for both Track-A relations across four fresh processes (canonical and
   shuffled input × `PYTHONHASHSEED` 0/1/424242): byte-identical manifest, identical rows, schema,
   membership and `track_pair_id`, identical DSDG adapter relation.
9. **Audited exhaustively** (`tools/m4_idfree_audit.py`, PASS, 0 failures) and built the Track-A
   fairness matrix (8 methods, **0 violations**): every Track-A method trains on CASIA + MSU + SiW,
   consumes no subject identity and no attack type, and carries `N_syn = 8,838`.
10. **Kept Track B alive and clearly secondary.** DSDG-NATIVE and DiffFAS-NATIVE are not deleted;
    they are relabelled `B_NATIVE_FULL_SECONDARY`, coverage CASIA + MSU with SiW
    `NOT_INSTANTIABLE_MISSING_SUBJECT_ID`, deferred to M6, and barred from any main table without a
    track column. No native manifest exists.
11. **Marked M4 COMPLETE under the AMENDED rule**, with the audit trail stating plainly that the
    originally specified native manifests were not created. M5 and M6 remain NOT_STARTED; nothing was
    trained, nothing was generated and TEST performance has never been observed.

## M5 pre-flight (2026-09-20) — contract audited, M5 still NOT_STARTED

1. **Verified the starting state**: HEAD `f0d54b7`, local `main` == `origin/main`, clean tree,
   M4 COMPLETE / `FINALIZED_UNDER_AMENDMENT_A1`, M5 and M6 NOT_STARTED, 464 tests PASS, and the four
   frozen M2/M3/M4 hashes unchanged.
2. **Read §12.1 verbatim** and separated what the spec fixes (backbone, weight enum, 224×224
   high-pass RGB, `k=9`/`σ=1.5`, `attack_macro` over real TRAIN, clipped inverse-frequency weighted
   CE, AdamW 1e-4/1e-4, batch 64, 30 epochs, cosine, AMP, seed 42, max VAL macro-F1, 512-D
   L2-normalised embedding) from what it does not.
3. **Recorded the Amendment A1 firewall.** Track A bans attack-type supervision *for generators*;
   §12.1 explicitly permits fine-grained labels in this measurement probe "because the labels are
   not fed back to the generator or downstream detector". The seven conditions that make that true
   are written into the proposed config and test-guarded.
4. **Searched for Q-07 instead of assuming.** It **already exists** in `deviation_report.md` and
   `configs/CONFIG_STATUS.md` (resize vs crop, 256→224). It was extended with three further
   geometry choices rather than duplicated, and no historical ID was invented.
5. **Audited the population.** TRAIN 14,467 (5,629 live / 8,838 spoof), VAL 3,121 (1,216 / 1,905).
   `attack_macro` is a §3.2 enum that **includes `live`** and is populated on every row, so
   "SPOOF classes only" (K=6) and "LIVE + spoof" (K=7) are both readable — D-M5-01, recommendation
   B, not chosen here.
6. **Showed the class-weight degeneracy explicitly.** Under the literal `w_c = 1/n_c`, **every**
   class clips to exactly 0.5 in both populations, so the weighting disappears. Four non-degenerate
   alternatives were computed in full; `N/(K·n_c)` is recommended and left to the owner (D-M5-02).
7. **Measured the high-pass pipeline** on 36 real canonical faces instead of arguing about it:
   Gaussian implementation (OpenCV / torchvision / explicit conv) and value domain differ by
   < 7e-07 and are **not** execution-affecting; border mode (0.26), resize order (0.084) and
   interpolation (0.13) **are**, against a typical residual peak of 0.245. The signed residual spans
   ≈[−0.47, +0.59], so clipping it would destroy half the signal, and ImageNet normalisation after
   HP maps the input to ≈[−4.17, +0.81].
8. **Resolved the backbone weight provenance.** Fetched the official
   `ResNet18_Weights.IMAGENET1K_V1` file and verified it by torchvision's own filename convention:
   the digest `f37072fd47e89c…` begins with the `f37072fd` in `resnet18-f37072fd.pth`. Recorded in
   `models/registry.yaml`; the binary is git-ignored.
9. **Found a hard environment blocker (E-M5-01).** This machine is torch 2.14.0+cpu with no CUDA and
   no `nvidia-smi`, while §12.1 requires AMP. M5 **execution** cannot run here; the contract freeze
   is not blocked.
10. **Built only refusing scaffolding.** `gpatbench/probe/preprocess.py` exposes the four unresolved
    choices as required keyword arguments with no defaults, so no contract can be fixed by accident;
    `python -m gpatbench.cli train-probe` exists and refuses, and a test asserts there is no hidden
    trainer behind it.
11. **Left M5 NOT_STARTED.** No frozen config, no model, no checkpoint, no training log,
    `models/artifact_probe/` absent. Analysis and scaffolding do not start a milestone.

## M5 owner resolution (2026-09-20) — contract frozen, M5 still NOT_STARTED

1. **Verified the starting state**: HEAD `98315be`, clean tree, M4 COMPLETE /
   `FINALIZED_UNDER_AMENDMENT_A1`, M5 and M6 NOT_STARTED, and the five frozen M2/M3/M4 hashes
   unchanged.
2. **Recomputed the population from the manifest rather than trusting the pre-flight report.**
   TRAIN 14,467 / VAL 3,121 and all fourteen per-class counts were asserted against the expected
   values; a mismatch would have stopped the pass.
3. **Froze the seven-class contract** (`live, makeup, mask_2d, mask_3d, partial, print, replay`),
   excluding `other_spoof` because it has zero observations — no unused output logit is created.
4. **Recomputed the class weights in float64** as `clip(N/(K·n_c), 0.5, 3.0)`: `live` pins to 0.5,
   `mask_2d` to 3.0, and the other five carry genuine inverse-frequency weighting. The trainer
   recomputes them at runtime and refuses on any mismatch.
5. **Froze the input pipeline**: whole 256×256 canonical face, no crop, `uint8/255`, OpenCV
   `INTER_AREA` 256→224, per-channel `GaussianBlur(9×9, σ1.5, BORDER_REFLECT_101)`, signed
   residual, **no post-high-pass normalisation**. `frozen_probe_input` is the single entry point;
   the candidate API keeps its no-default arguments so the two cannot be confused.
6. **Froze the backbone as a full fine-tune**: `resnet18(IMAGENET1K_V1)` with `fc → Linear(512, 7)`
   and all 11,180,103 parameters trainable, with the seed set before the classifier is built. The
   weight hash `f37072fd47e89c…` was re-verified.
7. **Implemented the single trainer** (`gpatbench.probe.train.run`) plus contract, data, metrics and
   model modules, and wired `python -m gpatbench.cli train-probe`. It refuses rather than adapts:
   non-frozen config, changed hashes, CPU authoritative run, TEST split, synthetic data, count or
   weight mismatch, wrong weight hash.
8. **Ran the CPU-safe smoke — PASS, 22/22.** Real faces through the frozen transform (signed,
   `3×224×224`, float32, deterministic), one forward pass (7 logits, 512-D unit-norm embedding),
   the weighted loss in frozen class order, the fixed-class macro-F1 with its zero-division and
   tie rules, the cosine sequence (epoch 1 = 1e-4, LR → 0 after 30 steps), and every refusal.
   **No `optimizer.step()` on real data, no checkpoint, no performance claim.**
9. **Resolved E-M5-01 to GPU_REQUIRED** and wrote the GPU execution plan: repository sync, a
   minimal ≈1.7 GB faces bundle (geometry/identity caches, frames and raw data are *not* needed),
   a resume-safe rsync, integrity verification against the frozen M2 provenance, and the full list
   of remote environment checks. **Nothing was transferred and no remote fact was measured.**
10. **Left M5 NOT_STARTED**, pre-flight status `READY_FOR_GPU_EXECUTION_PREFLIGHT`. A frozen
    contract and a refusing trainer do not start a milestone.

