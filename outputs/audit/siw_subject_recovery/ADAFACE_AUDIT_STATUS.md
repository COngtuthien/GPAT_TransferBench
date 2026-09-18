# SiW-Mv2 AdaFace/SCRFD Identity Consistency Audit — STATUS: NOT RUN (NOT_APPLICABLE)

DIAGNOSTIC_ONLY area. Nothing here is a scientific cache and nothing may be reused as M2 output.

## Why it was not run

- Per the owner's rules (§14–§18 of the resolution brief), model-based identity may only **SUPPORT / CONTRADICT / FLAG AMBIGUITY** for a *source-derived candidate mapping*.
- The investigation found **no** source-derived candidate mapping: `siw_subject_mapping_candidate.csv` has 1,700 × `NO_EVIDENCE`. There are therefore **0 candidate groups** to validate.
- Running AdaFace without candidate groups could only produce clusters, i.e. pseudo-subject IDs. That is explicitly forbidden, and it would be an "attempt to determine the identity of a subject", which SiW-Mv2 DRA §4 prohibits.
- **AdaFace was NOT used as ground truth, and it was not used at all.**

## Model resources found (read-only; hashed for provenance, not loaded)

Location: `/media/cong/Data/AI on IOT/Anti_spoofing/model_cache/`

| File | Size (B) | sha256 | Provenance available | Status |
|---|---|---|---|---|
| `face_detectors/scrfd_10g_bnkps.onnx` | 16,923,827 | `5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91` | none (no source metadata) | SCRFD candidate; variant = 10G bnkps by filename (Q-02 open) |
| `face_detectors/det_2.5g.onnx` | 3,292,009 | `041f73f47371333d1d17a6fee6c8ab4e6aecabefe398ff32cca4e2d5eaee0af9` | none | SCRFD-2.5G candidate by filename |
| `face_identity/pretrained_model/model.pt` | 174,611,121 | `43bd2d570584d95d4a17ce81f26449034c45dbeed750afcab651872abc0e1496` | HF download metadata: revision `60a65befbcf7e19284c4f3ac730f56867ed29594`, etag = sha256 above; **repo id not recorded** | AdaFace candidate; IR-50 vs other backbone **UNVERIFIED** (Q-03 open) |
| `code/adaface/adaface_net.py` | – | – | vendored code, origin unrecorded | not used |

Runtime dependencies (torch, onnxruntime) are not installed on the laptop, and none were installed for this
pass. If a future owner-approved source mapping appears, the audit would first need these
dependencies plus verification of the exact AdaFace IR-50 checkpoint (models/registry.yaml).
