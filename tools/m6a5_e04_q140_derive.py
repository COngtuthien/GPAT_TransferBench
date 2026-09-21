"""Derive and freeze the E04 controlled-reconstruction Q=140 vertex set.

CONTROLLED_ADAPTATION. The original PhySTD/TPAMI-2022 Q=140 vertex indices were never
published (only the count and a qualitative rationale). This script defines a deterministic,
algorithmic substitute over the CANONICAL NEUTRAL mesh of the pinned 3DDFA_V2 BFM model.

Determinism contract (frozen, see Amendment A3):
  canonical mesh   : bfm_noneck_v3.pkl mean shape u, alpha_shp = alpha_exp = 0
                     vertices = u.reshape(3, -1, order='F').T            -> (38365, 3)
  semantic anchors : the model's own 68 iBUG landmark vertices,
                     vertex_id = keypoints.reshape(68,3)[:,0] // 3       (x-component rule)
  candidate mask   : mesh vertices whose canonical (x, y) lies inside or on the 2D convex
                     hull of the 68 anchor vertices (monotone-chain hull, pure numpy)
  seed             : all 68 anchors, in canonical iBUG order 0..67
  extension        : farthest-point sampling to 140, maximizing the minimum EUCLIDEAN
                     distance in canonical 3D to the already-selected set
  tie-break        : smallest vertex index wins
  ordering         : [68 anchors in iBUG order] ++ [72 FPS picks in selection order]

The output list is IMAGE-INDEPENDENT: it is computed once from the canonical neutral mesh and
is identical for every image. Nothing here touches benchmark data, VAL or TEST.

Usage: python3 tools/m6a5_e04_q140_derive.py
"""
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

BFM_DIR = Path("/media/cong/Data/GPAT_TransferBench_runtime/third_party_weights/tddfa_v2/configs")
OUT = Path(__file__).resolve().parents[1] / "outputs" / "audit" / "M6A5_E04_Q140_VERTEX_SET.json"
Q = 140


def convex_hull_2d(pts):
    """Monotone-chain convex hull. Returns hull vertices CCW. Deterministic, pure numpy."""
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    p = pts[order]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for q in p:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], q) <= 0:
            lower.pop()
        lower.append(q)
    upper = []
    for q in p[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], q) <= 0:
            upper.pop()
        upper.append(q)
    return np.array(lower[:-1] + upper[:-1], dtype=np.float64)


def inside_hull(pts, hull):
    """True where pts are inside or on the CCW hull (all left-of-or-on every edge)."""
    inside = np.ones(len(pts), dtype=bool)
    n = len(hull)
    for i in range(n):
        a, b = hull[i], hull[(i + 1) % n]
        side = (b[0] - a[0]) * (pts[:, 1] - a[1]) - (b[1] - a[1]) * (pts[:, 0] - a[0])
        inside &= side >= 0
    return inside


def main():
    bfm = pickle.load(open(BFM_DIR / "bfm_noneck_v3.pkl", "rb"))
    u = bfm["u"].astype(np.float64)
    V = u.reshape(3, -1, order="F").T                      # canonical NEUTRAL mesh
    K = np.asarray(bfm["keypoints"]).ravel().astype(np.int64).reshape(68, 3)
    anchors = (K[:, 0] // 3).astype(np.int64)              # x-component rule
    assert len(set(anchors.tolist())) == 68

    hull = convex_hull_2d(V[anchors][:, :2])
    mask = inside_hull(V[:, :2], hull)
    cand = np.flatnonzero(mask).astype(np.int64)
    # anchors are hull-defining, so they are inside by construction
    assert np.all(mask[anchors]), "anchors must lie inside their own hull"

    # farthest-point sampling, seeded by all 68 anchors
    selected = list(anchors)
    d = np.full(cand.shape[0], np.inf)
    P = V[cand]
    for s in selected:
        d = np.minimum(d, np.linalg.norm(P - V[s], axis=1))
    while len(selected) < Q:
        m = d.max()
        # tie-break: smallest vertex index among all argmax candidates
        nxt = int(cand[np.flatnonzero(d == m)].min())
        selected.append(nxt)
        d = np.minimum(d, np.linalg.norm(P - V[nxt], axis=1))

    idx = np.asarray(selected, dtype=np.int64)
    assert idx.shape == (Q,) and len(set(idx.tolist())) == Q

    payload = {
        "schema_version": "m6a5-e04-q140-vertex-set-v1",
        "status": "CONTROLLED_ADAPTATION",
        "original_requirement": "PhySTD/TPAMI-2022 Q=140 facial vertices (exact indices never published)",
        "substitute": "deterministic 140-vertex coverage set over the canonical neutral BFM mesh",
        "num_vertices_total": int(V.shape[0]),
        "num_candidates_in_face_region": int(cand.size),
        "num_anchors": 68,
        "num_fps_extension": Q - 68,
        "Q": Q,
        "canonical_model": {
            "file": "bfm_noneck_v3.pkl",
            "sha256": "89ac96480eddc331120f2c8401737c55a5fcc97b10f8a64574e63976c3d246ca",
            "mean_shape_only": True, "alpha_shp": 0, "alpha_exp": 0,
            "vertex_layout": "u.reshape(3, -1, order='F').T",
        },
        "anchor_rule": "keypoints.reshape(68,3)[:,0] // 3 (x-component rule)",
        "candidate_mask": "canonical (x,y) inside-or-on the 2D convex hull of the 68 anchor vertices",
        "distance_metric": "euclidean_3d_canonical",
        "initial_anchor": "all 68 iBUG landmark vertices (seeded, not a single point)",
        "tie_break": "smallest vertex index",
        "ordering": "68 anchors in iBUG order 0..67, then 72 FPS picks in selection order",
        "image_independent": True,
        "vertex_indices": idx.tolist(),
    }
    body = json.dumps(payload["vertex_indices"], separators=(",", ":")).encode()
    payload["vertex_indices_sha256"] = hashlib.sha256(body).hexdigest()

    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"candidates in face region : {cand.size} / {V.shape[0]}")
    print(f"selected                  : {idx.size} (68 anchors + {Q-68} FPS)")
    print(f"vertex_indices_sha256     : {payload['vertex_indices_sha256']}")
    print(f"wrote {OUT}")
    # coverage diagnostics
    S = V[idx]
    dm = np.linalg.norm(S[:, None, :] - S[None, :, :], axis=2)
    np.fill_diagonal(dm, np.inf)
    print(f"min pairwise dist         : {dm.min():.1f}")
    cover = np.linalg.norm(V[cand][:, None, :] - S[None, :, :], axis=2).min(axis=1)
    print(f"face-region coverage: max dist to nearest selected = {cover.max():.1f}, mean = {cover.mean():.1f}")


if __name__ == "__main__":
    main()
