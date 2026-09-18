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
