"""Provenance verification of the owner-selected M2 auxiliary models against official sources.

Writes outputs/audit/M2A_OFFICIAL_VERIFICATION.json. Read-only with respect to the project; the
official artifacts it inspects live outside Git in the external model cache:

  * SCRFD (Q-02): the local scrfd_10g_bnkps.onnx is compared BYTE-FOR-BYTE with the member
    `antelopev2/scrfd_10g_bnkps.onnx` of the official InsightFace release pack antelopev2.zip
    (github.com/deepinsight/insightface, release v0.7). The zip is never committed.
  * AdaFace (Q-03): the checkpoint downloaded from the link published in the official
    mk-minchul/AdaFace README (row "R50 | WebFace4M") is hashed, strict-loaded into the official
    net.py build_model('ir_50'), and compared tensor-by-tensor with the CVLFace export that the
    owner did NOT select, to document the exact relationship between the two official releases.

Run with the M2 environment:
    /home/cong/.venvs/gpatbench-m2/bin/python tools/m2a_verify_official.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CACHE = Path("/media/cong/Data/AI on IOT/Anti_spoofing/model_cache")
OUT = ROOT / "outputs/audit/M2A_OFFICIAL_VERIFICATION.json"

ANTELOPEV2_ZIP = CACHE / "_provenance_tmp/antelopev2.zip"
ANTELOPEV2_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip"
ANTELOPEV2_MEMBER = "antelopev2/scrfd_10g_bnkps.onnx"
SCRFD_LOCAL = CACHE / "face_detectors/scrfd_10g_bnkps.onnx"
SCRFD_NOT_SELECTED = CACHE / "face_detectors/det_2.5g.onnx"

ADAFACE_CKPT = CACHE / "face_identity/adaface_original/adaface_ir50_webface4m.ckpt"
ADAFACE_URL = "https://drive.google.com/file/d/1BmDRrhPsHSbXcWZoYFPJg2KJn1sd3QpN/view?usp=sharing"
ADAFACE_DIRECT = "https://drive.usercontent.google.com/download?id=1BmDRrhPsHSbXcWZoYFPJg2KJn1sd3QpN&export=download&confirm=t"
ADAFACE_README = "https://github.com/mk-minchul/AdaFace/blob/c60eaa786a42c03444f3df7096dbaf9d57ae010d/README.md"
ADAFACE_CODE_COMMIT = "c60eaa786a42c03444f3df7096dbaf9d57ae010d"
ADAFACE_CVLFACE = CACHE / "face_identity/pretrained_model/model.pt"


def sha256_file(p: Path) -> str | None:
    if not Path(p).is_file():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def verify_scrfd() -> dict:
    d = {"question": "Q-02", "owner_decision": "candidate A scrfd_10g_bnkps.onnx (SCRFD_10G_KPS)",
         "official_source": ANTELOPEV2_URL, "official_member": ANTELOPEV2_MEMBER,
         "local_path": str(SCRFD_LOCAL), "local_sha256": sha256_file(SCRFD_LOCAL),
         "local_size_bytes": SCRFD_LOCAL.stat().st_size if SCRFD_LOCAL.is_file() else None,
         "not_selected_candidate": {"path": str(SCRFD_NOT_SELECTED), "sha256": sha256_file(SCRFD_NOT_SELECTED),
                                    "status": "NOT_SELECTED_FOR_FINAL_M2"}}
    if not ANTELOPEV2_ZIP.is_file():
        d.update(official_package_available=False, match="OFFICIAL_BYTE_HASH_NOT_INDEPENDENTLY_CONFIRMED")
        return d
    with zipfile.ZipFile(ANTELOPEV2_ZIP) as z:
        blob = z.read(ANTELOPEV2_MEMBER)
        members = [{"name": i.filename, "size": i.file_size} for i in z.infolist()]
    official = hashlib.sha256(blob).hexdigest()
    d.update(official_package_available=True, official_package_path=str(ANTELOPEV2_ZIP),
             official_package_sha256=sha256_file(ANTELOPEV2_ZIP),
             official_package_size_bytes=ANTELOPEV2_ZIP.stat().st_size, official_package_members=members,
             official_member_sha256=official, official_member_size_bytes=len(blob),
             match="OFFICIAL_BYTE_HASH_CONFIRMED" if official == d["local_sha256"] else "MISMATCH")
    return d


def verify_adaface() -> dict:
    import torch
    d = {"question": "Q-03/Q-18/Q-19", "owner_decision": "original mk-minchul/AdaFace R50 / WebFace4M",
         "official_readme": ADAFACE_README, "official_link": ADAFACE_URL, "direct_url": ADAFACE_DIRECT,
         "code_commit": ADAFACE_CODE_COMMIT, "local_path": str(ADAFACE_CKPT),
         "size_bytes": ADAFACE_CKPT.stat().st_size if ADAFACE_CKPT.is_file() else None,
         "sha256": sha256_file(ADAFACE_CKPT)}
    if not ADAFACE_CKPT.is_file():
        d["status"] = "MISSING"
        return d
    sys.path.insert(0, str(CACHE / "code/adaface"))
    import adaface_net
    ck = torch.load(ADAFACE_CKPT, map_location="cpu", weights_only=False)
    sd = {k[6:]: v for k, v in ck["state_dict"].items() if k.startswith("model.")}
    model = adaface_net.build_model("ir_50")
    res = model.load_state_dict(sd, strict=True)
    model.eval()
    with torch.inference_mode():
        feat, norm = model(torch.zeros(1, 3, 112, 112))
    d.update(checkpoint_top_keys=sorted(ck.keys()), n_state_dict_tensors=len(ck["state_dict"]),
             n_model_tensors=len(sd), strict_load="OK",
             strict_load_detail=f"missing={list(res.missing_keys)} unexpected={list(res.unexpected_keys)}",
             architecture="ir_50 (IR-50 / R50)", training_dataset="WebFace4M",
             embedding_dim=int(feat.shape[1]), embedding_l2=float(torch.linalg.norm(feat)))

    # Relationship to the CVLFace export that was NOT selected (documented, not used).
    cvl = torch.load(ADAFACE_CVLFACE, map_location="cpu", weights_only=False)
    c = {k[4:]: v for k, v in cvl.items() if k.startswith("net.")}
    diff = {k: float((sd[k].float() - c[k].float()).abs().max()) for k in sd if k in c and sd[k].shape == c[k].shape}
    differing = sorted(k for k, v in diff.items() if v != 0.0)
    flip_key = "input_layer.0.weight"
    d["cvlface_comparison"] = {
        "cvlface_path": str(ADAFACE_CVLFACE), "cvlface_sha256": sha256_file(ADAFACE_CVLFACE),
        "same_key_set": sorted(sd) == sorted(c), "n_tensors": len(sd),
        "n_tensors_differing": len(differing), "differing_keys": differing,
        "first_conv_is_exact_channel_reversal":
            bool(torch.equal(sd[flip_key].flip(1).contiguous(), c[flip_key])) if flip_key in c else None,
        "conclusion": ("The two official releases hold the SAME trained AdaFace IR-50 WebFace4M weights; the CVLFace "
                       "export differs only by an exact reversal of the input convolution's channel axis, i.e. it is "
                       "the BGR model re-expressed for RGB input. Selecting the original release + BGR input therefore "
                       "follows the spec wording literally without changing the trained model."),
        "status": "NOT_SELECTED_FOR_FINAL_M2 (kept as historical candidate evidence)"}
    return d


def main() -> int:
    from gpatbench.preprocess import logit_store as LS  # noqa: E402
    rep = {"generated_for": "M2A owner decision resolution", "cwd": str(ROOT),
           "scrfd": verify_scrfd(), "adaface": verify_adaface(), "logit_codec": LS.codec_provenance()}
    OUT.write_text(json.dumps(rep, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(json.dumps({"scrfd_match": rep["scrfd"].get("match"),
                      "adaface_sha256": rep["adaface"].get("sha256"),
                      "adaface_strict_load": rep["adaface"].get("strict_load"),
                      "cvlface_differing_tensors": rep["adaface"].get("cvlface_comparison", {}).get("n_tensors_differing")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
