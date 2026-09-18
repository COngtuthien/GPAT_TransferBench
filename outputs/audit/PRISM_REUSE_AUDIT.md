# PRISM Reuse Audit — M0 DISCOVERY LEVEL (read-only)

Scope: existence + surface-level discovery only. **No reuse decision is made at M0.**
No old project was modified: only `ls`, `grep -l`, `cat .git` and `git` read commands with
`GIT_OPTIONAL_LOCKS=0` (prevents `git status` from refreshing the index) were run.
No code was copied into GPAT-TransferBench.

Spec constraint to respect in any later reuse decision: the spec title page states
**"Standalone GPAT study — no PRISM-FAS-B components"** and §0 "evaluated as a standalone
generator, not as part of PRISM-FAS-B". Any reuse must be scientifically re-validated against
this spec (e.g. PRISM GPAT code may implement a different architecture than spec §9).

## 1. Laptop: `/home/cong/PRISM_FAS_C_LLM_Project_E10_DEV`

| Field | Value |
|---|---|
| Exists | YES |
| Type | Git **linked worktree** of `/home/cong/PRISM_FAS_C_LLM_Project` (`.git` file → `.../PRISM_FAS_C_LLM_Project/.git/worktrees/PRISM_FAS_C_LLM_Project_E10_DEV`) |
| HEAD | `528120f8915a1f4a28da2b412ecdc737b3a72b84` |
| Branch | `gpu-work/c-ext-final-closeout` |
| Last commit | 2026-09-08 19:39:28 +0700 "docs(c-ext): final EXT-Q1Q2 scientific closeout" |
| Dirty | YES — 25 porcelain entries (untracked `src/prism_fas/evaluation/c_ext_gpat_only_*.py` etc.) |
| Top-level | AGENTS.md, CLAUDE.md, DECISIONS.md, MILESTONES.md, configs/, src/prism_fas/, scripts/, tests/ (61 entries), reports/, runs/, state/, requirements*, pyproject.toml, modal_*.py |

## 2. Laptop (additional, found during discovery): `/home/cong/PRISM_FAS_C_LLM_Project`

| Field | Value |
|---|---|
| Exists | YES (main worktree) |
| HEAD | `6f0642a1d05c35e4c1778d329fb547f1b22a4115`, branch `gpu-work/e7-gpat-bank-prep` |
| Last commit | 2026-09-08 10:40:31 +0700 "fix(c-ext): harden E8 evaluation provenance and descriptive multi-seed scoring" |
| Dirty | YES — 64 entries (incl. ` M .gitignore`, ` M src/prism_fas/detector/dataset.py`) |
| Worktrees listed | main, `_E10_DEV`, `_E9_DEV_6f0642a` (prunable), `_E9_EXEC_c3e4ca` (prunable) |

Other PRISM-related directories seen in `/home/cong` (not inspected): `PRISM_FAS_B_Project/` (git repo; **PRISM-FAS-B — excluded by spec**), `PRISM_FAS_C_FLOW2_FINAL_SNAPSHOT/`, `PRISM_FAS_C_GPU_EVIDENCE_ARCHIVE_20260908/`, `PRISM_FAS_C_GPU_FINAL_SNAPSHOT/`, `EXT_E1_E10_BACKUP_AUDIT/`, `e9_runtime_evidence/`.

## 3. GPU: `/home/sparc/workdir/longnm/PRISM_FAS_C_LLM_Project`

**NOT AUDITED — PENDING.** SSH requires an interactive password (see `GPU_CONNECTIVITY_AUDIT.md`).

## 4. Relevant components seen (in `_E10_DEV`, filename/keyword level only)

Keyword counts = number of `*.py` files under `src/ scripts/ tests/` containing the term (case-insensitive `grep -l`).

| Topic | Hits | Modules observed (examples) | Preliminary reuse candidate? |
|---|---|---|---|
| SCRFD | 45 | `synthesis/quality_models.py`, `synthesis/identity_calibration.py`, `data/preprocess_m2.py`, `data/m2_runner.py` (grep-confirmed) | CANDIDATE for M2 detector wrapper — must match spec §4 (input 320, thr 0.50, largest face, 1.25× square crop) |
| FaceXFormer | 18 | `synthesis/quality_models.py`, `synthesis/structural_calibration.py` | CANDIDATE for M2 geometry cache wrapper |
| AdaFace | 34 | `synthesis/gpat_losses.py`, `synthesis/identity_calibration.py` | CANDIDATE for M2 identity cache wrapper (IR-50, official BGR normalization) |
| Preprocessing / crop | 111 (crop) | `data/preprocess_m2.py`, `data/m2_runner.py`, `data/media/`, `synthesis/m8_pipeline.py`; tests `test_m2_*_routing.py`, `test_crop_source_media_type.py` | CANDIDATE (failure-routing patterns); crop params must be re-checked vs spec |
| Dataset adapters | — | `data/adapters/{adapters,base}.py`, `data/schemas/`, test `test_casia_record_identity.py` | CANDIDATE for M1 CASIA/MSU/SiW-Mv2 adapters — subject-ID recovery logic especially |
| Manifests | 189 | `data/manifests/`, many | CANDIDATE (patterns only) |
| GPAT / physics synthesis | 184 (gpat) | `synthesis/gpat_model.py`, `gpat_losses.py`, `gpat_trainer.py`, `gpat_checkpoint.py`, `dwt.py`, `physics.py`, `operators/` | **NOT a direct candidate** — architecture must be rebuilt from spec §9–§10; at most reference for DWT/test patterns after line-by-line comparison |
| Provenance / hashing | 162 / 383 | across `synthesis/`, `search/`, `pipeline/` | CANDIDATE (patterns) |
| GPU orchestration | ssh 10, rsync 2 | `pipeline/{gpu_preflight,handoff,portability,portable_paths,orchestrator}.py`, `cloud/remote_verify.py`, `evaluation/c_ext_gpat_only_a3_gpu_runtime.py` | CANDIDATE for laptop→GPU handoff tooling |
| Tests | 61 entries | `tests/` | CANDIDATE (test patterns) |

## 5. Local auxiliary model files (outside PRISM repos)

`/media/cong/Data/AI on IOT/Anti_spoofing/model_cache/` (names and sizes only; not hashed, not loaded):

| File | Size (bytes) | Likely role (UNVERIFIED) |
|---|---|---|
| `face_detectors/scrfd_10g_bnkps.onnx` | 16923827 | SCRFD candidate |
| `face_detectors/det_2.5g.onnx` | 3292009 | SCRFD-2.5G candidate |
| `face_geometry/ckpts/model.pt` | 1104869851 | FaceXFormer candidate (dir name only) |
| `face_identity/pretrained_model/model.pt` | 174611121 | AdaFace candidate (dir name only) |
| `backbones/model.safetensors` | 14847232 | unknown backbone |
| `pretrained/m9/siglip2/model.safetensors` | 1500800904 | SigLIP2 — not used by this spec |
| `code/facexformer/`, `code/adaface/` | dirs | vendored code candidates |

## 6. Decision status

All items: **DISCOVERY ONLY — reuse NOT decided.** Decisions belong to M1 (adapters) and
M2 (auxiliaries) with explicit provenance (source repo, commit, file hash) for anything reused.
