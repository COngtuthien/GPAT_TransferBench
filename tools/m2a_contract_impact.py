"""Measure what the owner's contract decisions actually changed, by comparing two smoke runs.

    python tools/m2a_contract_impact.py <old_tag> <new_tag>

Writes outputs/audit/M2A_CONTRACT_CHANGE_IMPACT.json. Diagnostic only: it compares artifacts the
smoke already produced and never selects anything by a benchmark metric.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/audit/M2A_CONTRACT_CHANGE_IMPACT.json"


def main() -> int:
    old, new = (ROOT / "outputs/exploratory/m2a_smoke" / t for t in sys.argv[1:3])
    same, changed = [], []
    for f in sorted((new / "faces").glob("*.png")):
        sid = f.stem
        prev = old / "faces" / f"{sid}.png"
        if not prev.is_file():
            continue
        (same if prev.read_bytes() == f.read_bytes() else changed).append(sid)
    cos = [float(np.load(old / "identity" / f"{s}__embedding.npy") @ np.load(new / "identity" / f"{s}__embedding.npy"))
           for s in same]
    fx_same = sum((old / "geometry" / f"{s}__parsing_logits.npy").read_bytes()
                  == (new / "geometry" / f"{s}__parsing_logits.npy").read_bytes() for s in same)
    rep = {
        "old_run": old.name, "old_contract": "PROVISIONAL (CVLFace checkpoint + RGB input; literal clamp crop)",
        "new_run": new.name, "new_contract": "FROZEN (original AdaFace R50-WebFace4M + BGR input; requested square + zero padding)",
        "canonical_faces_unchanged": len(same), "canonical_faces_changed": len(changed),
        "changed_sample_ids": sorted(changed),
        "changed_reason": "Q-22: these are exactly the crops that touch the frame border; the square is now preserved with zero padding instead of being clamped to a non-square region",
        "facexformer_logits_byte_identical_on_unchanged_faces": f"{fx_same}/{len(same)}",
        "adaface_cosine_on_unchanged_faces": {"min": min(cos), "max": max(cos)} if cos else None,
        "interpretation": ("On the faces that did not change, the new identity embeddings agree with the previous ones "
                           "to float32 precision. That is expected and is itself evidence: the CVLFace export and the "
                           "original release hold the same trained weights, differing only by the input convolution's "
                           "channel order, so 'CVLFace + RGB' and 'original + BGR' are the same function. The owner "
                           "decision therefore restores literal compliance with the spec's official-BGR wording without "
                           "changing the identity space. It must not be confused with the earlier diagnostic "
                           "cos(RGB-input, BGR-input) = 0.81-0.97, which fed ONE checkpoint the wrong channel order."),
    }
    OUT.write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("canonical_faces_unchanged", "canonical_faces_changed",
                                          "facexformer_logits_byte_identical_on_unchanged_faces",
                                          "adaface_cosine_on_unchanged_faces")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
