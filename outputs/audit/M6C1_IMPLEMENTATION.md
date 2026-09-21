# M6C1 — Common runtime, E01 FAS-Aug and E02 Frequency Substitution

Updated 2026-09-22 from interrupted dirty handoff on `m6-baselines`.
Starting committed HEAD: `9fd2188a885ffbd2807bbf2708c31dc9e69c18a0`.

**IMPLEMENTED · DRY-RUN VERIFIED · NOT BENCHMARK-EXECUTED.**
E01/E02 current status is `IMPLEMENTED_NOT_EXECUTED`; E03–E07c remain `CONFIG_FROZEN`.

## Handoff preservation and findings

The handoff contained 3 modified tracked metadata files and 16 untracked files:
14 Python implementation/test/tool files and the two M6C1 audit files. No reset, stash,
checkout, clean, commit or push was performed. Safety patch:
`/tmp/GPAT_M6C1_HANDOFF_PRE_CODEX.patch`, SHA256
`0829f146764006b6ffab865f26cf9216ce3f292208004901a0683f37679205c7`.
The exact initial status and historical per-file hashes are in the JSON audit.

No duplicate CONFIG_STATUS current rows or ARTIFACT_INDEX paths existed at handoff.
Stale current-row “M6B not started” wording did exist and was removed. E03–E07c
scientific details/statuses are unchanged. Historical milestone prose remains.
The runlog metrics reader already used a context manager; no pre-fix duplicate remained.
Three unclosed dry-run readers and its double-close summary overwrite were corrected.
The E02 Decimal rounding test was already clean and was retained unchanged.

## Common runtime

The retained runtime now verifies both frozen snapshot byte identity and independent
M6B recorded SHA256s. It validates method IDs, the exact seed set and all TEST prohibition
flags, including reconstruction choice. Generators reject mutated config dictionaries,
nonfrozen seeds and explicit TEST inputs; no benchmark loader was added.

Run IDs retain the frozen SHA256 formula and directory layout. JSON/YAML writes are atomic;
metrics and generation logs are append-only and fsynced. Resume refuses incomplete artifacts,
partial JSONL, changed run identity/source/environment or altered resolved config. An exclusive
writer lock prevents simultaneous appenders. Resume preserves UUID/start time and records the
prior manifest/summary in its reconciliation event. Close is idempotent.

The manifest supplies required top-level identity/environment fields and UUID4. Unavailable
fields have explicit null reasons. E01/E02 records also go to `generation_log.jsonl` with frozen
method-specific fields. Epoch logging is refused for these non-learned methods. Pixel output is
excluded from `GeneratorResult.to_record`; arrays/binary/image payloads are rejected by the logger.
Checkpoint index support is metadata-only in tests: no checkpoint file is created.
Environment lock/fingerprint remain null with an explicit pending-freeze reason.

## E01 recipe and official backend

The existing seed, magnitude and round-robin recipe layer was retained. The previously
insufficient level-zero test now checks byte identity, and generation returns a true no-op.
The default is now `OfficialFASAugBackend`; a toy backend requires `synthetic_only=True` and
cannot produce a scientific E01 result through the production default.

**official_operator_backend = IMPLEMENTED** in `methods/fas_aug/official.py`.
The adapter verifies `RizhaoCai/FAS-Aug@0da1dd79bad00e225b8cb6977c7f3f06ee8f7517`
against frozen config/provenance and actual Git HEAD/tree. It checks cited source SHA256s,
source/asset git blobs, texture directory counts 90/48/190, and required ICC profiles.
Missing source, module, asset, altered bytes or provenance mismatch fails closed; no toy fallback.

It compiles unchanged upstream class/helper function AST nodes from the verified cache.
This excludes unused torch/Compose imports and eager module-level asset loading, without
rewriting operator bodies or vendoring source. Lazy texture decoding uses the same RGB conversion;
profile paths are resolved against the source root without changing cwd. Every call receives a
private RNG namespace. Asset directory order is ascending bytewise lexicographic (A2-01), while
authored ICC/direction/interpolation dictionary order stays upstream-defined. The recipe consumes
one magnitude draw; official pixel functions then consume their original choice/crop draws,
which are captured in provenance. Level zero bypasses all operator draws. Names, ranges,
probabilities, magnitude distribution and eight-operator set are unchanged.

**official_operator_synthetic_smoke = PASS (M6C1_OFFICIAL_E01_SMOKE_CLOSURE).**
Pinned upstream `requirements.txt` lists `pillow` without a version constraint.
Installed Pillow **12.3.0** in the current development `.venv` only. Observed Python
**3.12.3**, NumPy **2.5.3**. These are the **M6C1 synthetic smoke environment only**;
observed dependency versions are **NOT YET the final frozen execution environment**.

All eight official operators executed successfully on a synthetic in-memory 256x256 RGB
uint8 array: Color_Diversity, Color_Distortion, Reflection, BN_Halftone, Moire_Pattern,
SFC_Halftone, Hand_Trembling and Low_Resolution. Each nonzero recipe changed pixels,
each deterministic repeat was byte-identical, and each level-zero recipe returned the
input byte-identically. Source pin verification and asset discovery passed (90 backgrounds,
48 noise textures, 190 moire textures). Production rejected ToyOperatorBackend; no fallback.
No implementation fix or scientific behavior change was needed.

The official-only smoke used the same `sys.addaudithook('open')` path firewall as the
existing dry run. It recorded **920 open events**, separately:

```text
official_e01_smoke_benchmark_images_opened = 0
official_e01_smoke_manifest_opens = 0
```

Only verified upstream operator assets were permitted as file-backed images; the input and
outputs remained in memory. No benchmark image output or bank was written. Full per-operator
recipes, sampled choices and synthetic array hashes are in the JSON audit's
`official_e01_smoke_closure`. Input generation: `Generator(PCG64(7))`, uniform uint8 RGB.
Operator indices 0..7 exercise the frozen set; no benchmark outputs informed selection.

E01 vectors for `PTR000001`:

| Global seed | UTF-8 preimage | Operator seed |
|---|---|---|
| 42 | `PTR00000142` | 795981663 |
| 1337 | `PTR0000011337` | 924982993 |
| 2026 | `PTR0000012026` | 1531213468 |

Seed = unsigned big-endian SHA256 integer modulo 2^31, with no separator.

## E02 verification

**implementation_verified = true; synthetic_transform_tested = true.**
No scientific redesign: 43,186 eligible pixels; 159 eligible blocks in row-major order;
`k = (n_eligible + 2) // 4 = 40`. The full 32-byte SHA256 digest, unsigned big-endian,
seeds `Generator(PCG64(pair_seed))`; choice is without replacement. The selected SET
is scientific; indices are sorted before application.

Preimage: `gpatbench.freqsub.block.v1|PTR000001|<decimal global seed>`.
All three full seed and 40-index selection vectors are in the JSON audit.
For seed 42, pair seed is
`19354000309184887456743517179028239418116648094833629759175866349341962868907`,
and the full sorted set is:

```text
2, 4, 16, 22, 25, 30, 33, 39, 42, 44, 46, 47, 53, 54, 55, 61, 64, 69, 76, 78,
80, 90, 93, 95, 96, 97, 98, 115, 117, 123, 125, 129, 130, 131, 134, 136, 138, 145, 149, 152
```

Hermitian mapping remains pixel-wise `p -> (256-p) mod 256`, with 0 and 128
self-conjugate; no block-grid mirror. Target phase is preserved; no spatial post-filter.
Current measured inverse-FFT residuals on the focused test fixtures:
noise **1.0999371638972949e-13**, smooth **3.2291627502247798e-12**;
both below **1e-8**. Previous-agent reported values (~1.20e-13/~5.32e-12) remain
historical observations in JSON. NumPy **2.5.3** is observed, not an authoritative runtime pin.

## Validation and data-access disclosure

- Focused command: `python -m unittest tests.test_m6c1_runtime tests.test_m6c1_e01 tests.test_m6c1_e02 -v`:
  **83/83 tests pass, 0 skipped**, including the previously Pillow-dependent test. ResourceWarning checks found no leaked handle.
- `python tools/m6b_validate_configs.py`: **PASS, 0 failures**.
- Historical bootstrap: **16/17 pass**, only `test_manifests_are_stage_appropriate`
  fails on `manifests/artifact_probe_classes_v1.json`; left unchanged.
- `python tools/m6c1_dry_run.py`: **PASS**, synthetic in-memory arrays, toy E01 explicitly
  non-scientific, E02 deterministic/Hermitian, config snapshots verified, deterministic run IDs,
  overwrite refused, resume append preserved. Temporary runtime removed. No production runtime write.
- Handoff dry-run open audit (before Pillow installation): **454 opens, 0 benchmark image/video opens, 0 manifest opens**.
  Verified upstream operator source/assets are permitted; the hook refuses other image/video paths.

The previous agent's **266 opens / 0 benchmark image opens / 0 manifest opens** applies
**only to its audited synthetic dry run**. It subsequently launched an inadvertent full
regression command, reported completed with exit 0 in the handoff. Its actual benchmark-file
access cannot be proven from retained evidence:

```text
dry_run_benchmark_images_opened = 0
inadvertent_full_regression_command_launched = true
full_regression_benchmark_data_access = UNVERIFIED
```

No whole-milestone zero-access claim is made. The old uncommitted ledger record is preserved
as historical bytes; its overbroad claims are corrected by exactly one appended
`M6C1_HANDOFF_COMPLETION` entry. The separate official-smoke closure appends exactly one
`M6C1_OFFICIAL_E01_SMOKE_CLOSURE` entry; neither record rewrites history. That accidental command provides no benchmark evidence,
scientific decision, retained experimental bank or model result. The full suite was not rerun.

## Final integrity

Current implementation/test/tool hashes are refreshed in the JSON audit. The canonical
`tools/build_artifact_index.py` rebuilds the unique-path index with current sizes/SHA256s and
CRLF. The index excludes itself and the append-only ledger by repository convention.
Final validation checks every indexed row and all recorded current hashes; historical ledger
hashes are never rewritten. All protected path diffs are empty; no weight-like dirty file exists.

NO LEARNED BASELINE TRAINING. NO FULL BANK GENERATION. NO TEST DATA USED FOR SCIENTIFIC
DECISIONS. NO GPU JOB. NO FROZEN CONFIG CHANGE. NO CHECKPOINT CREATED. NO COMMIT. NO PUSH.
