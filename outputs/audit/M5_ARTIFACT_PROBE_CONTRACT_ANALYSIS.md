# M5 — ArtifactProbeNet Contract Analysis (pre-flight)

**Date:** 2026-09-20 · **Milestone state: M5 NOT_STARTED** · Nothing was trained; no checkpoint exists.
**Verdict: READY_FOR_OWNER_DECISIONS** — 9 execution-affecting items are unresolved, plus one
environment blocker. Proposed contract: `configs/proposed/artifact_probe.proposed.yaml`.

## 1. What spec §12.1 fixes

Read verbatim from the frozen specification (`f7d23716…`), §12.1:

| item | frozen value |
|---|---|
| purpose | independent evaluation of whether synthetic samples preserve source attack-family evidence |
| identity | **never** the GPAT `E_art` network |
| backbone | torchvision ResNet-18 `IMAGENET1K_V1` |
| input | 224×224 high-pass RGB, Gaussian residual `k=9`, `σ=1.5` |
| labels | `attack_macro` over real TRAIN only |
| loss | class-weighted cross-entropy, inverse-frequency weights clipped to `[0.5, 3.0]` |
| optimizer | AdamW, lr `1e-4`, weight decay `1e-4`, batch 64, 30 epochs, cosine decay, AMP |
| seeds | `[42]` only, "because this is a frozen measurement tool" |
| checkpoint | maximum VAL macro-F1, freeze thereafter |
| embedding | penultimate 512-D, L2-normalized |

§4's preprocessing table supplies the operator itself: `HP(x) = x − GaussianBlur(x, kernel=9, sigma=1.5)`.

## 2. Amendment A1 firewall — COMPLIANT

Amendment A1 forbids attack-type supervision **for generators**; spec §5.1 restricts
generator-visible supervision to image + `label_binary`. ArtifactProbeNet is neither a generator nor
a downstream detector — it is an independent measurement tool, and §12.1 says so explicitly:
*"Using fine-grained labels in this evaluation probe is permitted even for Binary Track because the
labels are not fed back to the generator or downstream detector."*

The firewall conditions, recorded in the proposed config and test-guarded:

| condition | required |
|---|---|
| probe labels enter generator optimization | **false** |
| probe predictions select or modify Track-A training samples | **false** |
| probe predictions decide common pair membership | **false** |
| probe predictions decide generator checkpoint selection | **false** |
| probe gradients touch GPAT / DSDG / DiffFAS parameters | **false** |
| probe frozen before synthetic evaluation | **true** |
| `attack_macro` exported as generator input | **false** |

Permitted later use is exactly the three measurement roles the spec names: artifact embedding cosine
(§13), attack consistency (§13) and the fingerprint probe (§12.2).

## 3. Q-07 exists — it was not invented and must not be duplicated

A repository search found an **authoritative Q-07** already on record:

> `outputs/audit/deviation_report.md`: `| Q-07 | M5 | 12.1 | ArtifactProbeNet input is 224×224 from
> 256×256 faces; resize vs crop method not stated. |`
> `configs/CONFIG_STATUS.md`: `| configs/frozen/artifact_probe.yaml | §12.1 | NOT CREATED (M5) | Q-07 |`

So **`NO_AUTHORITATIVE_Q07_FOUND` is NOT the finding here** — Q-07 is real, it is scoped to the
256→224 geometry, and this pass extends it with three further geometry choices in the same pipeline
rather than opening a duplicate ID. The other eight items get new `D-M5-xx` ids because no existing
record covers them. Q-07 does not appear in the frozen specification itself; it is a project record
of a specification gap, which is what it should be.

## 4. The nine open items

| id | field | why it is execution-affecting |
|---|---|---|
| **D-M5-01** | classification population | changes K, every weight, the macro-F1 denominator and therefore which epoch is selected |
| **D-M5-02** | class-weight normalization | the literal reading makes every weight clip to 0.5, i.e. no weighting at all |
| **Q-07** | probe input geometry (resize/crop, resize order, interpolation, border mode) | measured shifts up to 0.26 against a residual whose typical peak is 0.245 |
| **D-M5-03** | post-high-pass normalization and value domain | ImageNet normalization maps the residual to ≈[−4.17, +0.81] |
| **D-M5-04** | backbone fine-tune scope | full / partial / classifier-only give different probes |
| **D-M5-05** | VAL macro-F1 definition | class set and `zero_division` change the selection metric |
| **D-M5-06** | checkpoint tie-break | undefined for equal macro-F1 |
| **D-M5-07** | cosine scheduler details | implementation, stepping, `T_max`, `eta_min`, warmup |
| **D-M5-08** | augmentation and dataloader | `NO_AUGMENTATION_SPECIFIED`; §14.1 is scoped to the downstream evaluator |
| **E-M5-01** | training device | environment, not contract: this machine is CPU-only and §12.1 requires AMP |

Details, candidates, measured impact and recommendations are in
`configs/proposed/artifact_probe.proposed.yaml` and the four companion analyses.

## 5. A structural limitation worth stating before training

Measured on the frozen split: **4 of the 6 spoof `attack_macro` classes occur in SiW-Mv2 only**
(`makeup`, `mask_2d`, `mask_3d`, `partial`). Only `print` and `replay` span all three datasets.

A classifier can therefore reach a high macro-F1 partly by recognising **dataset-specific imaging
characteristics** rather than attack-family evidence. That is not a defect to fix here — it is a
property of the pooled population — but it bounds what the probe's outputs can support:

- `artifact embedding cosine` and `attack consistency` (§13) are computed **within a dataset**
  (source and synthetic share a dataset through the common pair contract), so the confound does not
  invalidate them;
- any cross-dataset interpretation of probe classes would be unsafe and must not be claimed.

This must be disclosed wherever probe-derived metrics are reported.

## 6. What this pass produced and did not

Produced: the analyses, the measured impact of every contested choice, a candidate high-pass module
whose unresolved options are **required arguments with no defaults**, a `train-probe` CLI that
refuses to run, and the proposed config.

Not produced: any frozen config, any model, any checkpoint, any training log. `models/artifact_probe/`
does not exist. M5 stays **NOT_STARTED** — scaffolding and analysis do not start a milestone.
