# M6D4c — E05 training-runner owner resolution

**Contract resolved only. No training runner implemented or qualified.**
E05 remains **PCGAN (controlled architecture resolution)**, **CONTROLLED_ADAPTATION**,
**IMPLEMENTED_NOT_EXECUTED**. M6D4b remains byte-identical historical
**STOP_AND_REPORT** evidence. M6D4a architecture/runtime qualification is retained,
not rerun. These explicit owner decisions close the three normative gaps without
claiming numerical or runtime validation.

## Starting authority

Branch `m6-baselines`; HEAD = local `origin/m6-baselines` =
`5adb76285b9b9c4dc25bb4a33f3be7d8575a8abc`. Initial worktree clean; divergence **0 0**.
No fetch, GPU SSH, environment mutation, commit or push was needed or performed.
All eight required authority files were read completely.

## Exact decisions and preserved semantics

Let `(z_pat_src,z_con_src)=E(x_src)`, `(z_pat_tgt,z_con_tgt)=E(x_tgt)`,
`r_src=G(z_pat_src,z_con_src)` and `m=G(z_pat_src,z_con_tgt)`.
For each sample, `norm2(a-b)=sqrt(sum_C,H,W((a-b)**2))`.

1. **Source-only reconstruction:** `L_rec=mean_n norm2(x_src[n]-r_src[n])`.
   At frozen batch one, one unsquared Euclidean source-image norm. No target
   reconstruction, source/target average, MSE, RMS, L1 or pixel-count normalization.
   Distance semantics retain PAPER / frozen-evidence provenance. The explicit
   estimator is **CONTROLLED_RECONSTRUCTION_OWNER_RESOLUTION**. Eq.2 leaves the
   estimator open; Eq.4 and resolved L_advrec identify source reconstruction, so
   the owner selects the minimal estimator without another target contribution.
2. **Image D:** `d_src=D(x_src)`, `d_tgt=D(x_tgt)`,
   `d_rec=D(r_src.detach())`, `d_mix=D(m.detach())`.
   `L_D_real=0.5*(mean softplus(-d_src)+mean softplus(-d_tgt))`;
   `L_D_rec=mean softplus(d_rec)`; `L_D_mix=mean softplus(d_mix)`;
   `L_D_image=1.0*L_D_real+0.5*L_D_rec+0.5*L_D_mix`.
   Both real images contribute equally: **CONTROLLED_RECONSTRUCTION_OWNER_RESOLUTION**,
   not a PCGAN paper fact. Logistic construction and weights 1/.5/.5 are
   **CONTROLLED_RECONSTRUCTION_PREDECESSOR_DERIVED**.
3. **Patch D:** three independent pinned random draws: reference from source,
   another positive draw from source, and fake from `m.detach()`.
   `ref_feat=extract_features(crops_ref,aggregate=True)`;
   `pos_feat=extract_features(crops_pos,aggregate=False)`;
   `fake_feat=extract_features(crops_fake,aggregate=False)`.
   `p_real=discriminate_features(ref_feat,pos_feat)`;
   `p_fake=discriminate_features(ref_feat,fake_feat)`.
   `L_D_patch_real=mean softplus(-p_real)`;
   `L_D_patch_fake=mean softplus(p_fake)`;
   `L_D_patch=1.0*L_D_patch_real+1.0*L_D_patch_fake`.
   Interface is A2 / M6D4a qualified architecture. Positive/fake construction and
   unit weights are **CONTROLLED_RECONSTRUCTION_PREDECESSOR_DERIVED**.
4. **Combined D:** `L_D_total=L_D_image+L_D_patch`, unit combination is
   **CONTROLLED_RECONSTRUCTION_OWNER_RESOLUTION**. No R1, patch R1, lazy-R1 scaling,
   gradient penalty, extra patch regularizer, VGG/perceptual, GPAT identity or landmark term.
5. **Complete iteration:** one D Adam application, then one G Adam application,
   same explicit source/target pair, exactly two applications per complete cycle.
   D fakes use current E/G and detach; E/G must not update in D. After D, recompute
   E/G paths `r_src_G,m_G`; no stale pre-D fake reuse. All five G losses use the
   updated discriminators. Freeze D/PatchD parameters for G updates while retaining
   gradients through discriminator computations to generated images and E/G.
   D/PatchD must not update in G. D-first is predecessor-derived; grouping both
   updates into one benchmark iteration is **CONTROLLED_RECONSTRUCTION_OWNER_RESOLUTION**.
6. **Counter:** `benchmark_iteration=0` initially; both steps share `t`; increment
   once after successful G completion, `t -> t+1`. Individual Adam applications do
   not increment it. Terminal **4000 complete cycles** implies **4000 D + 4000 G**
   applications in an uninterrupted scientific run. This count consequence is
   **CONTROLLED_RECONSTRUCTION_OWNER_RESOLUTION**, not a paper optimizer-state fact.

Preserved generator terms: `L_recblur=mean_n norm2(B(x_tgt[n])-B(m[n]))`,
`L_advrec=mean softplus(-D(r_src))`, `L_advmix=mean softplus(-D(m))`, and
`L_pat=mean softplus(-PatchD(reference_features,mixed_features))`.
All five terms sum with weights exactly **1**. `alpha=.2`, `beta=1e-6` retain PMN scope.
Image and patch scores are raw logits. A5 blur is exactly avg_pool2d with kernel/stride
2, padding 0, ceil_mode=False, count_include_pad=False, independently applied with
no detach. L_pat uses independent random source-reference/mixed-candidate crops;
reference features aggregate through the qualified interface, candidates do not.

Preserved disjoint optimizer groups: **Encoder+Generator**, **ImageD+PatchD**.
Both: Adam **lr=1e-6, betas=(.9,.999)**; no new weight decay or upstream scaling.
StyleGAN noise stays enabled with fresh pinned normal draws, no fix_noise; random
crops remain pinned and independent where specified, no center substitution.
No draws occur in this milestone. FP32 and existing precision/seeds remain unchanged.

Checkpoint policy remains **BASELINE_FINAL_STATE_V1**, terminal counter 4000,
no VAL/TEST selection, no best seed, and zero extra optimizer applications for a
later terminal save. No checkpoint is produced here.

## Full M6D4b A–J disposition

| Item | Component | Status |
| --- | --- | --- |
| A | L_rec | CONTRACT_RESOLVED |
| B | L_recblur | CONTRACT_RESOLVED |
| C | L_advrec | CONTRACT_RESOLVED |
| D | L_advmix | CONTRACT_RESOLVED |
| E | L_pat | CONTRACT_RESOLVED |
| F | discriminator | CONTRACT_RESOLVED |
| G | parameter ownership | CONTRACT_RESOLVED |
| H | update schedule | CONTRACT_RESOLVED |
| I | StyleGAN noise | CONTRACT_RESOLVED |
| J | batch-one explicit pair | CONTRACT_RESOLVED |

This table is normative contract resolution, not numerical/runtime qualification.
The addendum and overlay retain exact formulas, sampling, provenance and boundaries.
New decisions must not be called PAPER, OFFICIAL_PCGAN, AUTHOR_SPECIFIED, native PCGAN,
faithful PCGAN, or official reproduction. The pinned Swapping Autoencoder predecessor
is `6baa180f1184ee79a6b967f9d80ee0e02a979ac7`, not official PCGAN code.

## Immutable input SHA256

| Input | SHA256 |
| --- | --- |
| `configs/methods/e05_pcgan.yaml` | `478756e150c427832315800acba954cedf778bac520eee3bde8cabf7359e71de` |
| `configs/amendments/e05_a5_blur_operator_resolution.yaml` | `af28a48b43828652a7423235aba6be239325e3fbfeeaf8fded9bbb57c11a7ca1` |
| `outputs/audit/M6D4B_E05_TRAINING_RUNNER_QUALIFICATION.json` | `d35247075e38b3a4d1cc2156faa030bb04058bd1d8b906a242d4e7a88b25519f` |
| `outputs/audit/M6D4B_E05_TRAINING_RUNNER_QUALIFICATION.md` | `37589973b502e71743b665828c88804b43fcd325ca6a8e82e3f04dcf87b81bf6` |
| `docs/spec/amendments/GPAT_TransferBench_v1_0_Amendment_A2_M6_Baseline_Execution_Contracts.md` | `b4fa7bfa75e5a977348c468d1bfe3e0004c3920293ecd9178700f9f4869fca8d` |
| `docs/spec/amendments/GPAT_TransferBench_v1_0_Amendment_A5_E05_Blur_Operator_Resolution.md` | `05d6c6c271bf23aece3b2fcf9cdef5d94bd32517421a0ac83be26c5a9174d4dd` |
| `outputs/audit/M6D4A_E05_RUNTIME_ARCHITECTURE_QUALIFICATION.md` | `f57d4e99c42a633b02ff41d4af0b017c0970260242743bc8752798731411bbc6` |
| `methods/pcgan/source_traceability.md` | `61645843f65a51e7858787cfbc7cf30f1ff75763c8ded6da41d96410d4c5c1bd` |

All match the authoritative committed bytes. Four additional historical hashes
cover the M6D4b process files, test and preflight. Historical STOP evidence, M6D4a,
frozen config and PCGAN implementation files remain unchanged.

## Created artifacts

- `configs/amendments/e05_training_runner_resolution.yaml`
- `docs/spec/amendments/GPAT_TransferBench_v1_0_E05_Training_Runner_Owner_Resolution_Addendum_M6D4c.md`
- `tools/m6d4c_e05_training_resolution_preflight.py`
- `outputs/audit/M6D4C_E05_TRAINING_RUNNER_RESOLUTION.md`
- `outputs/audit/M6D4C_E05_TRAINING_RUNNER_RESOLUTION.json`

## Static validation and finalization

**Preflight PASS; 996 in-memory invalid-contract rejections**.
Checks cover immutable hashes, estimator/norm, five unit G terms, exact D formulas,
no R1, groups/Adam, D→G ordering, detach/recomputation/same pair, two applications,
counter after G, terminal 4000, TEST prohibition and CONTROLLED_ADAPTATION disclosure.
Missing/extra fields, altered leaves/list entries, invalid types and duplicate keys
are rejected. The audit JSON retains every rejection case. Standard-library-only
preflight, invoked using `python3 -I -S -B`; no project/framework imports, no formula
execution. Runtime/numerical tests and the full repository suite are not run.

Ledger **107 → 108**, exactly one `M6D4C_E05_TRAINING_RUNNER_RESOLUTION` row.
First 107 rows remain byte-identical: **280377 bytes**, SHA256
`e8c46615e36cd087901cb757ee9f2b3a216b7910a842ae50dbf44880f97cc65d`. Artifact index rebuilt **last**, **550 → 555 rows**, **CRLF preserved**,
five explicit new artifact paths. Historical index metadata is carried without
opening targets; ledger and index themselves have no historical index rows.
Final preflight verifies ledger artifact hashes and the exact index bytes.

Expected final Git changes: five created artifacts above and only two tracked
modifications, `EXECUTION_LEDGER.jsonl` and `ARTIFACT_INDEX.csv`. HEAD/origin remain
unchanged, divergence 0 0. No commit or push.

## Execution and final status

**E05_ARCHITECTURE_RUNTIME_QUALIFIED** (prior evidence retained)

**E05_TRAINING_RUNNER_CONTRACT_RESOLVED**

**E05_TRAINING_RUNNER_NOT_YET_QUALIFIED**

**E05 REMAINS IMPLEMENTED_NOT_EXECUTED — CONTROLLED_ADAPTATION PRESERVED**

No runner implemented; no runtime or numerical qualification; no optimizer instantiated.
**OPTIMIZER APPLICATIONS = 0. BENCHMARK TRAINING = false.**
No Torch/TensorFlow import, CUDA/model execution, environment mutation, benchmark data
access, TRAIN/VAL/TEST sample access, manifest sample access, image decoding or benchmark
image filename enumeration. No scientific checkpoint or synthetic bank. **NO COMMIT.
NO PUSH.**
