"""E01 — FAS-Aug, deterministic benchmark adapter.

Fidelity: FAITHFUL_OFFICIAL_WITH_DETERMINISM_CLARIFICATION (Amendment A2 §A2-01/A2-05).

Scope of this module. It implements the *frozen deterministic contract* of E01 — which operator
runs, at which magnitude, with which asset, under which seed — exactly as
`configs/methods/e01_fas_aug.yaml` declares it. Every scientific constant is READ FROM THE CONFIG;
none is re-declared here.

The pixel transforms themselves belong to the pinned upstream
(`RizhaoCai/FAS-Aug@0da1dd79…`) and are reached through the `OperatorBackend` adapter seam below.
The official adapter selectively loads unchanged pinned operator bodies; no code is vendored.
`ToyOperatorBackend` exists ONLY for structural dry-run/tests on synthetic arrays; its output is
never FAS-Aug output and must never be reported as such.
"""
from __future__ import annotations

import hashlib
import random
import time
from pathlib import Path

import numpy as np

from ..common.config import load_method_config
from ..common.interface import GeneratorResult, MethodGenerator

METHOD_ID = "E01"

#: Which official asset directory each operator draws from, per the pinned upstream helper
#: (`data/fas_aug_helper.py`). Operators absent from this map use no asset.
OPERATOR_ASSET_DIR = {
    "Reflection": "background",
    "BN_Halftone": "noiseTexture",
    "Moire_Pattern": "MPTexture",
}


# ---------------------------------------------------------------------- seed (A2-05)
def operator_seed_preimage(pair_id: str, global_seed: int) -> bytes:
    """UTF-8(pair_id + decimal_string(global_seed)). There is NO separator."""
    if not isinstance(pair_id, str) or not pair_id:
        raise ValueError("pair_id must be a non-empty string")
    if isinstance(global_seed, bool) or not isinstance(global_seed, int):
        raise TypeError("global_seed must be an int")
    return (pair_id + str(global_seed)).encode("utf-8")


def operator_seed(pair_id: str, global_seed: int) -> int:
    """unsigned_big_endian_integer(SHA256(preimage)) mod 2**31 — frozen spec §8.1 + A2-05."""
    digest = hashlib.sha256(operator_seed_preimage(pair_id, global_seed)).digest()
    return int.from_bytes(digest, "big") % (2 ** 31)


# ---------------------------------------------------------------------- operator / magnitude
def select_operator(pair_id: str, operator_set: list[str], *, index: int | None = None) -> tuple[int, str]:
    """Deterministic round-robin by pair_id (spec §8.1 overrides the official random policy).

    `index` is the pair's position in the frozen canonical pair order. When it is None the trailing
    decimal digits of `pair_id` are used, which is the stable ordinal for the frozen `PTR%06d` /
    `PVA%06d` format.
    """
    if not operator_set:
        raise ValueError("operator_set is empty")
    if index is None:
        digits = "".join(ch for ch in pair_id if ch.isdigit())
        if not digits:
            raise ValueError(f"cannot derive a round-robin index from pair_id {pair_id!r}; "
                             f"pass index= explicitly")
        index = int(digits)
    i = index % len(operator_set)
    return i, operator_set[i]


def magnitude_for_level(level_k: int, num_mag: int, lo: float, hi: float) -> tuple[float, float]:
    """Official quantisation: level = k/(NUM_MAG-1); mag = level*(hi-lo) + lo.

    Returns (level, magnitude). k = 0 is RETAINED: it yields level 0.0 and magnitude == lo, the
    official no-op end of the grid. It is never skipped (A2-01 / M6A2 E01-2).
    """
    if not 0 <= level_k <= num_mag - 1:
        raise ValueError(f"level_k must be in [0, {num_mag - 1}], got {level_k}")
    level = level_k / (num_mag - 1)
    return level, level * (hi - lo) + lo


# ---------------------------------------------------------------------- assets (A2-01)
def enumerate_assets(directory: str | Path) -> list[str]:
    """ASCENDING LEXICOGRAPHIC (bytewise) order by filename — A2-01.

    Host-dependent `os.listdir` order is never the scientific contract. Sorting is on the UTF-8
    bytes of the filename so the result is locale-independent and matches git tree order.
    """
    d = Path(directory)
    if not d.is_dir():
        raise FileNotFoundError(f"asset directory not found: {d}")
    names = [p.name for p in d.iterdir() if p.is_file()]
    return sorted(names, key=lambda n: n.encode("utf-8"))


class AssetIndex:
    """Deterministic, lexicographically ordered view of the official asset directories."""

    def __init__(self, asset_root: str | Path, expected_counts: dict | None = None):
        self.root = Path(asset_root)
        self.dirs: dict[str, list[str]] = {}
        self.expected_counts = expected_counts or {}

    def load(self, subdirs: list[str]) -> "AssetIndex":
        for sub in subdirs:
            self.dirs[sub] = enumerate_assets(self.root / sub)
            exp = self.expected_counts.get(sub)
            if exp is not None and len(self.dirs[sub]) != exp:
                raise ValueError(f"asset directory {sub!r} has {len(self.dirs[sub])} files, "
                                 f"frozen config declares {exp}. REFUSED.")
        return self

    def pick(self, subdir: str, rng: random.Random) -> tuple[int, str]:
        names = self.dirs[subdir]
        i = rng.randrange(len(names))
        return i, names[i]


# ---------------------------------------------------------------------- operator backend seam
class OperatorBackend:
    """Adapter seam to the pinned upstream FAS-Aug operator implementations.

    The official backend verifies and binds `third_party/source_cache/fas_aug`.
    """

    name = "abstract"
    is_official = False

    def apply(self, image: np.ndarray, operator: str, magnitude: float,
              asset: str | None, rng: random.Random) -> np.ndarray:
        raise NotImplementedError


class ToyOperatorBackend(OperatorBackend):
    """STRUCTURAL ONLY. Deterministic synthetic stand-in for dry-run and unit tests.

    This is NOT FAS-Aug and its output must never be reported as an E01 result. It exists so the
    deterministic contract (seed -> operator -> magnitude -> asset -> byte-identical output) can be
    exercised without opening a benchmark image or vendoring upstream code.
    """

    name = "toy"
    is_official = False

    def apply(self, image, operator, magnitude, asset, rng):
        arr = np.asarray(image)
        if arr.dtype != np.uint8:
            raise TypeError("ToyOperatorBackend expects a uint8 array")
        if magnitude == 0.0:
            return arr.copy()
        delta = int(round(magnitude)) % 256
        return ((arr.astype(np.int32) + delta) % 256).astype(np.uint8)


# ---------------------------------------------------------------------- generator
class E01Generator(MethodGenerator):
    """E01 FAS-Aug generator driven entirely by the frozen config."""

    method_id = METHOD_ID

    def __init__(self, config: dict | None = None, *, asset_index: AssetIndex | None = None,
                 backend: OperatorBackend | None = None, synthetic_only: bool = False):
        canonical = load_method_config(METHOD_ID)
        self.config = config if config is not None else canonical
        if ({k: v for k, v in self.config.items() if k != "_runtime"} !=
                {k: v for k, v in canonical.items() if k != "_runtime"} or
                self.config.get("_runtime", {}).get("config_sha256") != canonical["_runtime"]["config_sha256"]):
            raise ValueError("generator config differs from verified M6B config")
        if self.config["method_id"] != METHOD_ID:
            raise ValueError("config is not the E01 contract")
        self.asset_index = asset_index
        self.synthetic_only = synthetic_only
        if backend is None:
            from .official import OfficialFASAugBackend
            backend = OfficialFASAugBackend(self.config)
        self.backend = backend
        if not synthetic_only and not self.backend.is_official:
            raise ValueError("scientific E01 generation requires OfficialFASAugBackend; toy is synthetic-only")
        self._gen = self.config["generation"]
        self.operator_set = list(self._gen["operator_set"])
        self.operator_ranges = dict(self._gen["operator_ranges"])
        self.num_mag = int(self._gen["num_mag"])
        self.input_resolution = int(self._gen["input_resolution"])
        self._count = 0
        self._noop = 0

    def prepare(self) -> "E01Generator":
        if self.backend.is_official:
            self.backend.prepare()
            self.asset_index = self.backend.asset_index
        if len(self.operator_set) != int(self._gen["operator_count"]):
            raise ValueError("operator_set length disagrees with the frozen operator_count")
        missing = [o for o in self.operator_set if o not in self.operator_ranges]
        if missing:
            raise ValueError(f"frozen config lacks ranges for {missing}")
        if self.config["asset_enumeration"]["rule"] != "ASCENDING_LEXICOGRAPHIC_BYTEWISE_BY_FILENAME":
            raise ValueError("frozen asset enumeration rule changed; refusing")
        return self

    def recipe(self, pair_id: str, global_seed: int, *, index: int | None = None) -> dict:
        """The full deterministic recipe for one pair. Pure: opens nothing."""
        if global_seed not in self.config["seeds"]["experiment_seeds"] or isinstance(global_seed, bool):
            raise ValueError("experiment seed must be one of [42, 1337, 2026]")
        if pair_id.upper().startswith(("PTE", "TEST")):
            raise ValueError("TEST pairs are forbidden")
        seed = operator_seed(pair_id, global_seed)
        op_i, op = select_operator(pair_id, self.operator_set, index=index)
        rng = random.Random(seed)
        level_k = rng.randrange(self.num_mag)
        lo, hi = self.operator_ranges[op]
        level, mag = magnitude_for_level(level_k, self.num_mag, float(lo), float(hi))
        asset_dir = OPERATOR_ASSET_DIR.get(op)
        asset_i, asset = None, None
        if level_k != 0 and asset_dir and self.asset_index is not None:
            asset_i, asset = self.asset_index.pick(asset_dir, rng)
        return {
            "pair_id": pair_id,
            "global_seed": global_seed,
            "operator_seed": seed,
            "operator_seed_preimage": operator_seed_preimage(pair_id, global_seed).decode("utf-8"),
            "operator_index": op_i,
            "operator": op,
            "level_k": level_k,
            "level": level,
            "magnitude": mag,
            "magnitude_range": [float(lo), float(hi)],
            "is_level_zero_noop": level_k == 0,
            "asset_dir": asset_dir,
            "asset_index": asset_i,
            "asset": asset,
            "backend": self.backend.name,
            "backend_is_official": self.backend.is_official,
        }

    def generate_one(self, pair_id: str, global_seed: int, target_live: np.ndarray, *,
                     index: int | None = None, source_spoof_ref: str | None = None, split: str = "TRAIN") -> GeneratorResult:
        t0 = time.perf_counter()
        failure = None
        out = None
        try:
            if split not in ("TRAIN", "VAL"):
                raise ValueError("TEST/unknown split forbidden")
            rec = self.recipe(pair_id, global_seed, index=index)
            if self.backend.is_official:
                out, parameters = self.backend.apply_recipe(target_live, rec)
                rec["sampled_operator_parameters"] = parameters
            elif rec["is_level_zero_noop"]:
                out = np.asarray(target_live).copy()
                rec["sampled_operator_parameters"] = {"random_draws": []}
            else:
                rng = random.Random(rec["operator_seed"])
                rng.randrange(self.num_mag)
                if rec["asset"] is not None:
                    self.asset_index.pick(rec["asset_dir"], rng)
                out = self.backend.apply(target_live, rec["operator"], rec["magnitude"], rec["asset"], rng)
                rec["sampled_operator_parameters"] = {"synthetic_only": True}
            ok = True
            self._count += 1
            self._noop += int(rec["is_level_zero_noop"])
        except Exception as exc:
            out = None
            rec = {"pair_id": pair_id, "global_seed": global_seed}
            ok = False
            failure = f"{type(exc).__name__}: {exc}"
        return GeneratorResult(
            method_id=METHOD_ID, global_seed=global_seed, pair_id=pair_id,
            inputs={"target_live_shape": list(np.shape(target_live)),
                    "source_spoof_used": False, "source_spoof_ref": source_spoof_ref},
            output=out, metadata=rec, success=ok, failure_reason=failure,
            elapsed_seconds=time.perf_counter() - t0)

    def finalize(self) -> dict:
        return {"method_id": METHOD_ID, "generated": self._count,
                "level_zero_noop_count": self._noop,
                "level_zero_noop_fraction": (self._noop / self._count) if self._count else None,
                "backend": self.backend.name, "backend_is_official": self.backend.is_official}
