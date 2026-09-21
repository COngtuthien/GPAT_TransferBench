"""E02 — Frequency Substitution, deterministic implementation.

Fidelity: SPEC_DEFINED. The algorithm is frozen spec §8.2; Amendment A2 (§A2-02/03/04) completed
the spec-silent execution details. Every scientific constant is READ FROM THE CONFIG.

Spec §8.2, verbatim contract implemented here:
  * per-channel FFT2 at 256x256, fftshift; A_t, phi_t from the target, A_s from the source;
  * eligible region: radial normalised frequency r in [0.20, 0.50], r=0 at centre, r=0.5 at Nyquist;
  * non-overlapping 16x16 blocks; a block is eligible if >= 75% of its pixels lie in the region;
  * select exactly 25% of eligible blocks with a pair-specific RNG seed;
  * replace target magnitude with source magnitude in the selected blocks AND their
    conjugate-symmetric counterparts; preserve target phase;
  * reconstruct A_mix*exp(j*phi_t), ifftshift, ifft2, real part, clip to [0,255]; no post-filter.
"""
from __future__ import annotations

import hashlib
import time

import numpy as np

from ..common.config import load_method_config
from ..common.interface import GeneratorResult, MethodGenerator

METHOD_ID = "E02"
NAMESPACE = "gpatbench.freqsub.block.v1"
ELIGIBLE_PIXEL_FRACTION = 0.75   # spec §8.2: "at least 75% of its pixels"


# ---------------------------------------------------------------------- seed (A2-04)
def pair_seed_preimage(pair_id: str, global_seed: int) -> bytes:
    """UTF-8("gpatbench.freqsub.block.v1|" + pair_id + "|" + decimal_string(global_seed))."""
    if not isinstance(pair_id, str) or not pair_id:
        raise ValueError("pair_id must be a non-empty string")
    if isinstance(global_seed, bool) or not isinstance(global_seed, int):
        raise TypeError("global_seed must be an int")
    return f"{NAMESPACE}|{pair_id}|{global_seed}".encode("utf-8")


def pair_seed(pair_id: str, global_seed: int) -> int:
    """Unsigned big-endian integer of the FULL 32-byte digest. No truncation. No modulo."""
    return int.from_bytes(hashlib.sha256(pair_seed_preimage(pair_id, global_seed)).digest(), "big")


# ---------------------------------------------------------------------- geometry (spec §8.2)
def eligible_blocks(n: int = 256, block: int = 16, r_lo: float = 0.20,
                    r_hi: float = 0.50) -> tuple[np.ndarray, np.ndarray]:
    """Return (eligible_rc, eligible_grid).

    `eligible_rc` is an (n_eligible, 2) int array of (block_row, block_col) in CANONICAL ROW-MAJOR
    order: block_row ascending first, block_col ascending within each row. `eligible_grid` is the
    (G, G) boolean grid.
    """
    if n % block:
        raise ValueError(f"block size {block} does not divide {n}")
    g = n // block
    centre = n // 2                       # fftshift places DC at index n//2
    f = (np.arange(n) - centre) / n       # normalised frequency; p=0 -> -0.5 (Nyquist edge)
    fy, fx = np.meshgrid(f, f, indexing="ij")
    r = np.sqrt(fy ** 2 + fx ** 2)
    in_band = (r >= r_lo) & (r <= r_hi)
    per_block = in_band.reshape(g, block, g, block).transpose(0, 2, 1, 3).reshape(g, g, block * block)
    grid = per_block.mean(axis=2) >= ELIGIBLE_PIXEL_FRACTION
    rc = np.argwhere(grid)                # argwhere is row-major: row asc, then col asc
    return rc.astype(np.int64), grid


def block_count(n_eligible: int) -> int:
    """A2-02 round-half-up, float-free: k = (n_eligible + 2) // 4."""
    if not isinstance(n_eligible, (int, np.integer)) or n_eligible < 0:
        raise ValueError("n_eligible must be a non-negative integer")
    return (int(n_eligible) + 2) // 4


def select_block_indices(n_eligible: int, k: int, seed: int) -> np.ndarray:
    """A2-03: PCG64, sampling WITHOUT replacement; the selected SET is the result.

    Returned indices are sorted ascending, which is the frozen canonical processing order. The
    order in which `rng.choice` happened to return them carries no scientific meaning.
    """
    rng = np.random.Generator(np.random.PCG64(seed))
    chosen = rng.choice(n_eligible, size=k, replace=False)
    return np.sort(chosen)


# ---------------------------------------------------------------------- Hermitian handling (A2)
def conjugate_index(p: int, n: int = 256) -> int:
    """Pixel-wise conjugate counterpart in the SHIFTED spectrum: p -> (n - p) mod n.

    Self-conjugate exactly at p = 0 (Nyquist) and p = n//2 (DC). A block-grid mirror is FORBIDDEN.
    """
    return (n - p) % n


def conjugate_mask(mask: np.ndarray) -> np.ndarray:
    """Pixel-wise conjugate image of a boolean mask over the shifted spectrum."""
    n = mask.shape[0]
    if mask.shape != (n, n):
        raise ValueError("mask must be square")
    yy, xx = np.nonzero(mask)
    out = np.zeros_like(mask)
    out[(n - yy) % n, (n - xx) % n] = True
    return out


def _edit_mask(selected_rc: np.ndarray, n: int, block: int) -> tuple[np.ndarray, np.ndarray]:
    sel = np.zeros((n, n), dtype=bool)
    for br, bc in selected_rc:
        sel[br * block:(br + 1) * block, bc * block:(bc + 1) * block] = True
    return sel, sel | conjugate_mask(sel)


# ---------------------------------------------------------------------- substitution
def substitute(target: np.ndarray, source: np.ndarray, selected_rc: np.ndarray, *,
               block: int = 16) -> tuple[np.ndarray, dict]:
    """Spec §8.2 substitution. `target`/`source` are HxWx C or HxW uint8/float arrays.

    Returns (image_uint8, diagnostics). `diagnostics["max_abs_imag"]` is the largest imaginary
    residual of the inverse transform across channels, which must be ~0 by Hermitian symmetry.
    """
    t = np.asarray(target)
    s = np.asarray(source)
    if t.shape != s.shape:
        raise ValueError(f"target/source shape mismatch: {t.shape} vs {s.shape}")
    squeeze = t.ndim == 2
    if squeeze:
        t, s = t[:, :, None], s[:, :, None]
    n = t.shape[0]
    if t.shape[1] != n:
        raise ValueError("expected a square image")

    sel_mask, edit = _edit_mask(selected_rc, n, block)
    out = np.empty(t.shape, dtype=np.float64)
    max_imag = 0.0
    for c in range(t.shape[2]):
        ft = np.fft.fftshift(np.fft.fft2(t[:, :, c].astype(np.float64)))
        fs = np.fft.fftshift(np.fft.fft2(s[:, :, c].astype(np.float64)))
        amp = np.where(edit, np.abs(fs), np.abs(ft))      # magnitude substituted
        phase = np.angle(ft)                              # target phase PRESERVED
        rec = np.fft.ifft2(np.fft.ifftshift(amp * np.exp(1j * phase)))
        max_imag = max(max_imag, float(np.abs(rec.imag).max()))
        out[:, :, c] = rec.real                           # real part; no spatial post-filter
    img = np.clip(out, 0.0, 255.0)
    img = np.rint(img).astype(np.uint8)
    if squeeze:
        img = img[:, :, 0]
    diag = {
        "edited_pixels": int(edit.sum()),
        "selected_pixels": int(sel_mask.sum()),
        "conjugate_only_pixels": int((edit & ~sel_mask).sum()),
        "max_abs_imag": max_imag,
        "edit_mask": edit,
    }
    return img, diag


# ---------------------------------------------------------------------- generator
class E02Generator(MethodGenerator):
    """E02 generator driven entirely by the frozen config."""

    method_id = METHOD_ID

    def __init__(self, config: dict | None = None):
        canonical = load_method_config(METHOD_ID)
        self.config = config if config is not None else canonical
        if ({k: v for k, v in self.config.items() if k != "_runtime"} !=
                {k: v for k, v in canonical.items() if k != "_runtime"} or
                self.config.get("_runtime", {}).get("config_sha256") != canonical["_runtime"]["config_sha256"]):
            raise ValueError("generator config differs from verified M6B config")
        if self.config["method_id"] != METHOD_ID:
            raise ValueError("config is not the E02 contract")
        g = self.config["generation"]
        self.n = int(g["input_resolution"])
        gr = g["block_grid"]
        if gr[0] != gr[1]:
            raise ValueError("non-square block grid is not supported by the frozen contract")
        self.block = self.n // int(gr[0])
        self.declared_n_eligible = int(g["n_eligible_frozen_geometry"])
        self.declared_k = int(self.config["block_count_rule"]["k_frozen_geometry"])
        self.eligible_rc: np.ndarray | None = None
        self.eligible_grid: np.ndarray | None = None
        self.n_eligible = 0
        self.k = 0
        self._count = 0
        self._max_imag = 0.0

    def prepare(self) -> "E02Generator":
        if self.config["block_selection"]["rng"] != \
                "numpy.random.Generator(numpy.random.PCG64(pair_seed))":
            raise ValueError("frozen RNG contract changed; refusing")
        if self.config["conjugate_symmetry"]["rule"] != "PIXELWISE_MIRROR_P_TO_256_MINUS_P":
            raise ValueError("frozen conjugate rule changed; refusing")
        self.eligible_rc, self.eligible_grid = eligible_blocks(self.n, self.block)
        self.n_eligible = int(self.eligible_rc.shape[0])
        if self.n_eligible != self.declared_n_eligible:
            raise ValueError(f"computed n_eligible {self.n_eligible} disagrees with the frozen "
                             f"config value {self.declared_n_eligible}. REFUSED.")
        self.k = block_count(self.n_eligible)
        if self.k != self.declared_k:
            raise ValueError(f"computed k {self.k} disagrees with the frozen config value "
                             f"{self.declared_k}. REFUSED.")
        return self

    def selection(self, pair_id: str, global_seed: int) -> dict:
        """The deterministic selection for one pair. Pure: opens nothing."""
        if self.eligible_rc is None:
            raise RuntimeError("prepare() must be called first")
        if type(global_seed) is not int or global_seed not in self.config["seeds"]["experiment_seeds"]:
            raise ValueError("experiment seed must be one of [42, 1337, 2026]")
        if pair_id.upper().startswith(("PTE", "TEST")):
            raise ValueError("TEST pairs are forbidden")
        seed = pair_seed(pair_id, global_seed)
        idx = select_block_indices(self.n_eligible, self.k, seed)
        rc = self.eligible_rc[idx]
        return {
            "pair_id": pair_id,
            "global_seed": global_seed,
            "pair_seed": str(seed),                 # 256-bit: string-encoded for JSON safety
            "pair_seed_preimage": pair_seed_preimage(pair_id, global_seed).decode("utf-8"),
            "n_eligible": self.n_eligible,
            "k": self.k,
            "selected_indices": [int(i) for i in idx],
            "selected_blocks_rc": [[int(r), int(c)] for r, c in rc],
        }

    def generate_one(self, pair_id: str, global_seed: int, target_live: np.ndarray,
                     source_spoof: np.ndarray, *, source_spoof_ref: str | None = None,
                     target_live_ref: str | None = None, split: str = "TRAIN") -> GeneratorResult:
        t0 = time.perf_counter()
        out, failure = None, None
        try:
            if split not in ("TRAIN", "VAL"):
                raise ValueError("TEST/unknown split forbidden")
            if np.shape(target_live) != (self.n, self.n, 3) or np.shape(source_spoof) != (self.n, self.n, 3):
                raise ValueError("E02 requires canonical 256x256 RGB inputs")
            if not np.isfinite(target_live).all() or not np.isfinite(source_spoof).all():
                raise ValueError("nonfinite input")
            sel = self.selection(pair_id, global_seed)
            rc = np.asarray(sel["selected_blocks_rc"], dtype=np.int64)
            out, diag = substitute(target_live, source_spoof, rc, block=self.block)
            sel.update({k: v for k, v in diag.items() if k != "edit_mask"})
            if diag["max_abs_imag"] > 1e-8:
                raise ValueError("inverse FFT imaginary residual exceeds documented 1e-8 tolerance")
            ok = True
            self._count += 1
            self._max_imag = max(self._max_imag, diag["max_abs_imag"])
        except Exception as exc:
            out = None
            sel = {"pair_id": pair_id, "global_seed": global_seed}
            ok = False
            failure = f"{type(exc).__name__}: {exc}"
        return GeneratorResult(
            method_id=METHOD_ID, global_seed=global_seed, pair_id=pair_id,
            inputs={"target_live_shape": list(np.shape(target_live)),
                    "source_spoof_shape": list(np.shape(source_spoof)),
                    "target_live_ref": target_live_ref, "source_spoof_ref": source_spoof_ref},
            output=out, metadata=sel, success=ok, failure_reason=failure,
            elapsed_seconds=time.perf_counter() - t0)

    def finalize(self) -> dict:
        return {"method_id": METHOD_ID, "generated": self._count,
                "n_eligible": self.n_eligible, "k": self.k,
                "max_abs_imag_observed": self._max_imag,
                "numpy_version": np.__version__}
