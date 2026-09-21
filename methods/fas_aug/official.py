"""Bind unchanged operator bodies from the verified FAS-Aug source checkout.

Selective loading compiles the upstream class and helper function AST nodes unchanged.
It avoids module-level eager texture decoding, cwd-relative IO, and unused torch/Compose
imports. Pixel math, profile dictionary order, probabilities and RNG call order stay
upstream-defined. Textures are decoded lazily; only directory lists are sorted (A2-01).
No upstream code or asset bytes are vendored here.
"""
from __future__ import annotations

import ast
import hashlib
import io
import json
import random
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ..common.config import ROOT, load_method_config
from .e01 import AssetIndex, OperatorBackend


class OfficialBackendError(RuntimeError):
    """Pinned source, asset or execution dependency is unavailable or inconsistent."""


class _RecordingRandom:
    def __init__(self, rng):
        self.rng, self.draws = rng, []

    def choice(self, values):
        value = self.rng.choice(values)
        self.draws.append({"draw": "choice", "value": value})
        return value

    def randint(self, low, high):
        value = self.rng.randint(low, high)
        self.draws.append({"draw": "randint", "low": low, "high": high, "value": value})
        return value


class OfficialFASAugBackend(OperatorBackend):
    name = "official_fas_aug"
    is_official = True

    def __init__(self, config=None, *, source_root=None, provenance_path=None):
        self.config = config if config is not None else load_method_config("E01")
        self.root = Path(source_root or ROOT / self.config["source"]["local_source_path"]).resolve()
        self.provenance_path = Path(provenance_path or ROOT / self.config["source"]["provenance"])
        self.asset_index = None
        self._blobs = {}
        self._code = []

    def _git(self, *args):
        try:
            return subprocess.check_output(["git", "-C", str(self.root), *args], stderr=subprocess.PIPE)
        except (OSError, subprocess.CalledProcessError) as exc:
            raise OfficialBackendError(f"pinned source unavailable: {self.root}: {exc}") from exc

    def _verified_bytes(self, rel):
        path = self.root / rel
        if not path.is_file() or path.is_symlink():
            raise OfficialBackendError(f"required upstream source/operator asset unavailable: {rel}")
        data = path.read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if blob != self._blobs.get(rel):
            raise OfficialBackendError(f"pinned source/operator asset mismatch: {rel}")
        return data

    def prepare(self):
        if not (self.root / ".git").exists():
            raise OfficialBackendError(f"pinned source unavailable: {self.root}")
        try:
            pin = json.loads(self.provenance_path.read_text())["sources"]["fas_aug"]
        except (OSError, ValueError, KeyError) as exc:
            raise OfficialBackendError(f"source provenance unavailable: {exc}") from exc
        source = self.config["source"]
        if (pin["pinned_commit"] != source["pinned_commit"] or
                pin["repository"] != source["repository"] or
                self._git("rev-parse", "HEAD").decode().strip() != source["pinned_commit"] or
                self._git("rev-parse", "HEAD^{tree}").decode().strip() != pin["commit_tree"]):
            raise OfficialBackendError("official source pin/commit/provenance mismatch")
        self.source_commit = source["pinned_commit"]
        for row in self._git("ls-tree", "-r", "-z", "HEAD", "--", "data").split(b"\0"):
            if row:
                meta, rel = row.split(b"\t", 1)
                self._blobs[rel.decode()] = meta.split()[2].decode()
        self._code = []
        for rel, kind in (("data/fas_aug_helper.py", ast.FunctionDef),
                          ("data/FAS_Augmentations.py", ast.ClassDef)):
            data = self._verified_bytes(rel)
            if hashlib.sha256(data).hexdigest() != pin["cited_files"][rel]["sha256"]:
                raise OfficialBackendError(f"source provenance SHA256 mismatch: {rel}")
            tree = ast.parse(data, filename=str(self.root / rel))
            nodes = [node for node in tree.body if isinstance(node, kind)]
            self._code.append(compile(ast.Module(body=nodes, type_ignores=[]), str(self.root / rel), "exec"))
        self.asset_index = AssetIndex(self.root / "data", self.config["asset_enumeration"]["asset_counts"])
        try:
            self.asset_index.load(list(self.config["asset_enumeration"]["asset_counts"]))
        except (OSError, ValueError) as exc:
            raise OfficialBackendError(f"required operator asset unavailable: {exc}") from exc
        for sub, names in self.asset_index.dirs.items():
            expected = [Path(p).name for p in self._blobs if p.startswith(f"data/{sub}/")]
            if names != sorted(expected, key=lambda s: s.encode()):
                raise OfficialBackendError(f"operator asset list differs from pinned tree: {sub}")
            for name in names:
                self._verified_bytes(f"data/{sub}/{name}")
        # Profile dictionaries are authored choices, not filesystem enumerations. Preserve order.
        self.profiles = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in ("rgb_profile_dict", "cmyk_profile_dict"):
                        self.profiles[target.id] = list(ast.literal_eval(node.value))
        for kind, directory in (("rgb", "RGB Profiles"), ("cmyk", "CMYK Profiles")):
            names = self.profiles[f"{kind}_profile_dict"]
            if len(names) != self.config["asset_enumeration"]["icc_profiles"][kind]:
                raise OfficialBackendError(f"upstream {kind} profile choice count mismatch")
            for name in names:
                self._verified_bytes(f"data/profile/{directory}/{name}")
        # Verify every frozen dispatch name and range without importing execution dependencies.
        ns = self._namespace(None, None, None, None)
        aug = self._augmenter(ns)
        if [r[0] for r in aug.al] != self.config["generation"]["operator_set"]:
            raise OfficialBackendError("official operator set mismatch")
        for name, low, high, _ in aug.al:
            if [low, high] != self.config["generation"]["operator_ranges"][name]:
                raise OfficialBackendError(f"official magnitude range mismatch: {name}")
        return self

    def _namespace(self, Image, ImageCms, cv2, rng):
        ns = {"Image": Image, "ImageCms": ImageCms, "cv2": cv2, "np": np, "random": rng}
        for code in self._code:
            exec(code, ns)
        return ns

    @staticmethod
    def _augmenter(ns):
        return ns["FAS_Augmentations"](SimpleNamespace(TRAIN=SimpleNamespace(AUG=SimpleNamespace(SAVE=False))))

    def apply_recipe(self, image, recipe):
        if not self._code or self.asset_index is None:
            raise OfficialBackendError("prepare() must verify pinned source first")
        try:
            from PIL import Image, ImageCms
            import cv2
        except ImportError as exc:
            raise OfficialBackendError(f"official pixel dependencies required: Pillow with ImageCms and OpenCV: {exc}") from exc
        arr = np.asarray(image)
        n = self.config["generation"]["input_resolution"]
        if arr.shape != (n, n, 3) or arr.dtype != np.uint8:
            raise ValueError(f"official E01 requires {n}x{n} RGB uint8")
        rng = random.Random(recipe["operator_seed"])
        if rng.randrange(self.config["generation"]["num_mag"]) != recipe["level_k"]:
            raise OfficialBackendError("recipe magnitude does not match pair RNG")
        recorder = _RecordingRandom(rng)
        backend = self

        class Profiles:
            @staticmethod
            def profile(path):
                return ImageCms.ImageCmsProfile(io.BytesIO(backend._verified_bytes(path)))

            def profileToProfile(self, img, src, dst):
                return ImageCms.profileToProfile(img, self.profile(src), self.profile(dst))

            def buildTransformFromOpenProfiles(self, src, dst, *args):
                return ImageCms.buildTransformFromOpenProfiles(self.profile(src), self.profile(dst), *args)

            applyTransform = staticmethod(ImageCms.applyTransform)

        class Textures:
            def __init__(self, sub):
                self.sub = sub

            def __getitem__(self, name):
                data = backend._verified_bytes(f"data/{self.sub}/{name}")
                with Image.open(io.BytesIO(data)) as img:
                    return img.convert("RGB")

        ns = self._namespace(Image, Profiles(), cv2, recorder)
        for prefix, sub in (("R", "background"), ("BN", "noiseTexture"), ("MP", "MPTexture")):
            ns[f"{prefix}_NAME_LIST"] = self.asset_index.dirs[sub]
            ns[f"{prefix}_DICT"] = Textures(sub)
        aug = self._augmenter(ns)
        output, _ = aug.apply_augment(Image.fromarray(arr), recipe["operator"], recipe["level"], 0)
        # Texture choice is the first upstream draw, after the magnitude draw.
        if recipe["asset"] is not None and recorder.draws[0]["value"] != recipe["asset"]:
            raise OfficialBackendError("recipe asset and official dispatch disagree")
        return np.asarray(output).copy(), {"random_draws": recorder.draws,
                                          "source_commit": self.source_commit}
