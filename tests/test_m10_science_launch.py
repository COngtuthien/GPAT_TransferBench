"""M10 scientific launcher, O1-O7 rules and the independent per-run auditor.

Tiers (each skips cleanly where its requirement is absent):
  - stdlib / pyarrow only (laptop .venv): O2 threshold + D5 selection vs brute force, storage fail-closed, run-manifest
    v2, launch order / predecessor / preflight gates with injected environment and GPU identities, populations;
  - torch CPU (gpat-m7-cpu / host env): fixture runs through the real launcher (non-scientific, CPU policy), retention
    of the best-so-far checkpoint, fail-closed paths with partial evidence, the auditor's positive and injected
    negative cases, the TEST firewall in a subprocess;
  - CUDA (GPU host, gpat-m8-downstream): the O1 policy -- float16 autocast, GradScaler lifecycle / skipped steps /
    genuine non-finite gradients, deterministic configuration and bitwise replay.
All fixture outputs are NON_SCIENTIFIC; nothing here reads a real face, a bank image or TEST.
"""
import copy
import importlib.util
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gpatbench.downstream import m10_launcher as L  # noqa: E402
from gpatbench.downstream import m10_run_manifest as RM  # noqa: E402
from gpatbench.downstream import m10_selection as S10  # noqa: E402
from gpatbench.downstream import m10_storage as ST  # noqa: E402
from gpatbench.downstream import metrics as M  # noqa: E402

_spec = importlib.util.spec_from_file_location("m10_run_auditor", str(ROOT / "tools/m10_run_auditor.py"))
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False
CUDA = HAVE_TORCH and torch.cuda.is_available()
WEIGHTS = os.path.expanduser("~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth")
HAVE_WEIGHTS = os.path.isfile(WEIGHTS)
GE = "SPOOF_IF_SCORE_GE_THRESHOLD"


# ------------------------------------------------------------------------------------------ independent reference
def ref_threshold(y, s):
    """Brute force written from the owner text: min ACER; then nearest 0.5 (float, as the approved implementation);
    then the smallest threshold."""
    u = sorted(set(s))
    cands = sorted(set([(a + b) / 2 for a, b in zip(u, u[1:])] + [0.0, 1.0]))
    nsp, nlv = y.count(1), y.count(0)

    def acer(t):
        return (Fraction(sum(1 for a, b in zip(y, s) if a == 1 and b < t), nsp)
                + Fraction(sum(1 for a, b in zip(y, s) if a == 0 and b >= t), nlv)) / 2
    best = min(acer(t) for t in cands)
    tied = [t for t in cands if acer(t) == best]
    dist = min(abs(t - 0.5) for t in tied)
    return min(t for t in tied if abs(t - 0.5) == dist), best


class ThresholdO2(unittest.TestCase):
    Y8 = [0, 0, 0, 0, 1, 1, 1, 1]

    def test_perfect_classification(self):
        r = S10.select_threshold_o2(self.Y8, [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
        self.assertEqual((r["threshold"], r["val_acer"], r["tie_break"]), (0.5, 0, "NONE"))

    def test_completely_inverted(self):
        r = S10.select_threshold_o2(self.Y8, [0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1])
        self.assertEqual((r["threshold"], r["val_acer"], r["tie_break"]), (0.0, Fraction(1, 2), "O2_SMALLER_THRESHOLD"))
        with self.assertRaises(M.UnresolvedThresholdTie):         # frozen metrics.py still documents the gap
            M.select_threshold(self.Y8, [0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1], rule=GE)

    def test_constant_scores(self):
        r = S10.select_threshold_o2(self.Y8, [0.5] * 8)
        self.assertEqual((r["threshold"], r["val_acer"]), (0.0, Fraction(1, 2)))
        self.assertEqual(r["nearest_thresholds"], [0.0, 1.0])

    def test_two_equally_close_thresholds(self):
        # minimum ACER at 0.25 and 0.75 (both exactly 1/4 from 0.5): the smaller wins
        y, s = [0, 1, 0, 1], [0.2, 0.3, 0.7, 0.8]
        ref, _ = ref_threshold(y, s)
        r = S10.select_threshold_o2(y, s)
        self.assertEqual(r["threshold"], ref)
        self.assertIn(0.25, r["min_acer_thresholds"])
        self.assertIn(0.75, r["min_acer_thresholds"])
        self.assertEqual((r["threshold"], r["tie_break"]), (0.25, "O2_SMALLER_THRESHOLD"))

    def test_floating_point_ties(self):
        # float midpoints 0.025 and 0.975: equidistant from 0.5 as real numbers; their float distances both round to
        # 0.475 (the approved implementation's comparison) although the stored floats differ by ~2e-17 exactly
        y, s = [0, 1, 1, 0], [0.95, 1.0, 0.05, 0.0]
        self.assertEqual(0.5 - 0.025, 0.975 - 0.5)
        self.assertNotEqual(abs(Fraction(0.025) - Fraction(1, 2)), abs(Fraction(0.975) - Fraction(1, 2)))
        with self.assertRaises(M.UnresolvedThresholdTie):
            M.select_threshold(y, s, rule=GE)
        r = S10.select_threshold_o2(y, s)
        self.assertEqual(r["nearest_thresholds"], [0.025, 0.975])
        self.assertEqual((r["threshold"], r["tie_break"]), (0.025, "O2_SMALLER_THRESHOLD"))
        self.assertEqual(r["threshold"], ref_threshold(y, s)[0])
        self.assertEqual(A.brute_threshold(y, s)[0], 0.025)
        # a near-tie that is not a float tie is decided by distance, not by size
        y, s = [1, 0, 1, 0], [0.05, 0.15, 0.85, 0.95]
        r = S10.select_threshold_o2(y, s)
        self.assertEqual(r["threshold"], ref_threshold(y, s)[0])

    def test_boundary_thresholds_and_ge_comparison(self):
        r = S10.select_threshold_o2([0, 1], [0.0, 1.0])
        self.assertEqual((r["threshold"], r["val_acer"]), (0.5, 0))
        r = S10.select_threshold_o2([0, 1], [1.0, 1.0])            # only candidates 0 and 1: both ACER 1/2
        self.assertEqual(r["threshold"], 0.0)
        self.assertEqual(M.predict_spoof([0.5, 0.5 - 2 ** -53], 0.5, rule=GE), [1, 0])
        # a spoof video exactly at the threshold is predicted spoof
        y, s = [0, 0, 1, 1], [0.2, 0.4, 0.6, 0.8]
        t = S10.select_threshold_o2(y, s)["threshold"]
        self.assertEqual(t, 0.5)
        self.assertEqual(M.predict_spoof([t], t, rule=GE), [1])

    def test_brute_force_agreement_and_consistency_with_frozen_rule(self):
        rng = random.Random(20261126)
        for _ in range(400):
            n = rng.randint(2, 14)
            y = [rng.randint(0, 1) for _ in range(n)]
            y[0], y[1] = 0, 1
            pool = [rng.choice([0.0, 0.1, 0.25, 0.3, 0.5, 0.7, 0.75, 0.9, 1.0]) if rng.random() < 0.5 else rng.random()
                    for _ in range(n)]
            ref, acer = ref_threshold(y, pool)
            r = S10.select_threshold_o2(y, pool)
            self.assertEqual((r["threshold"], r["val_acer"]), (ref, acer))
            self.assertEqual(A.brute_threshold(y, pool), (ref, acer))           # the auditor's own implementation
            try:
                frozen = M.select_threshold(y, pool, rule=GE)["threshold"]
                self.assertEqual(frozen, r["threshold"])                      # O2 only completes exact ties
            except M.UnresolvedThresholdTie:
                self.assertEqual(r["tie_break"], "O2_SMALLER_THRESHOLD")

    def test_invalid_input_refused(self):
        for y, s in (([0, 1], [0.2, float("nan")]), ([0, 1], [0.2, float("inf")]), ([0, 1], [0.2, 1.5]),
                     ([0, 1], [0.2, -0.1]), ([0, 0], [0.2, 0.3]), ([0, 2], [0.2, 0.3]), ([0, 1], [0.2]),
                     ([0, True], [0.1, 0.2])):
            with self.assertRaises(M.MetricError):
                S10.select_threshold_o2(y, s)


class CheckpointSelectionD5(unittest.TestCase):
    def test_acer_then_auc_then_earliest(self):
        es = [{"epoch": 0, "val_acer": "1/4", "val_auc": "9/10"}, {"epoch": 1, "val_acer": "1/8", "val_auc": "4/5"},
              {"epoch": 2, "val_acer": "1/8", "val_auc": "9/10"}, {"epoch": 3, "val_acer": "1/8", "val_auc": "9/10"}]
        self.assertEqual(S10.select_epoch(es)["epoch"], 2)

    def test_tied_acer_different_auc(self):
        es = [{"epoch": 0, "val_acer": "1/8", "val_auc": "1/2"}, {"epoch": 1, "val_acer": "1/8", "val_auc": "3/4"}]
        self.assertEqual(S10.select_epoch(es)["epoch"], 1)

    def test_identical_acer_and_auc(self):
        es = [{"epoch": e, "val_acer": "1/8", "val_auc": "3/4"} for e in range(5)]
        self.assertEqual(S10.select_epoch(es)["epoch"], 0)

    def test_selection_over_all_40_epochs_never_final_by_default(self):
        rng = random.Random(7)
        es = [{"epoch": e, "val_acer": str(Fraction(rng.randint(5, 40), 100)), "val_auc": str(Fraction(rng.randint(50, 99), 100))}
              for e in range(40)]
        es[17]["val_acer"], es[39]["val_acer"] = "1/100", "1/2"
        self.assertEqual(S10.select_epoch(es)["epoch"], 17)
        brute = min(es, key=lambda e: (Fraction(e["val_acer"]), -Fraction(e["val_auc"]), e["epoch"]))
        self.assertEqual(brute["epoch"], 17)

    def test_duplicate_or_empty_refused(self):
        with self.assertRaises(M.MetricError):
            S10.select_epoch([])
        with self.assertRaises(M.MetricError):
            S10.select_epoch([{"epoch": 0, "val_acer": "0", "val_auc": "1"}] * 2)


# ------------------------------------------------------------------------------------------ storage
class Storage(unittest.TestCase):
    def setUp(self):
        self.t = Path(tempfile.mkdtemp())
        self.root = str(self.t / "m10")
        os.mkdir(self.root, 0o700)
        os.chmod(self.root, 0o700)
        import pwd
        self.exp = {"root": self.root, "mount_point": str(self.t), "bank_root": str(self.t / "banks"),
                    "owner": pwd.getpwuid(os.getuid()).pw_name, "forbidden_prefix": "/media/student20261/"}
        self.fs = {"mount_point": str(self.t), "filesystem_uuid": ST.EXPECTED["filesystem_uuid"], "fstype": "ext4",
                   "mount_options": "rw,relatime", "super_options": "rw,errors=remount-ro", "statvfs_readonly": False,
                   "free_bytes": 3 * 2 ** 40, "free_inodes": 10 ** 8}

    def tearDown(self):
        shutil.rmtree(self.t, ignore_errors=True)

    def pf(self, **fs):
        return ST.preflight(self.root, self.exp, identity_fn=lambda p: dict(self.fs, **fs))

    def test_ok(self):
        r = self.pf()
        self.assertTrue(r["ok"], r["checks"])

    def test_wrong_uuid(self):
        self.assertFalse(self.pf(filesystem_uuid="00000000-0000-0000-0000-000000000000")["checks"]["filesystem_uuid"])

    def test_wrong_owner(self):
        self.exp["owner"] = "root"
        r = self.pf()
        self.assertFalse(r["ok"])
        self.assertFalse(r["checks"]["owner"])

    def test_wrong_mode(self):
        for mode in (0o750, 0o755, 0o2700, 0o777):
            os.chmod(self.root, mode)
            self.assertFalse(self.pf()["checks"]["mode_exact"], oct(mode))

    def test_missing_mount(self):
        self.assertFalse(self.pf(mount_point=None, filesystem_uuid=None)["ok"])
        shutil.rmtree(self.root)
        self.assertFalse(self.pf()["checks"]["exists"])

    def test_read_only_filesystem(self):
        for kw in ({"mount_options": "ro,relatime"}, {"super_options": "ro"}, {"statvfs_readonly": True}):
            self.assertFalse(self.pf(**kw)["checks"]["mount_read_write"], kw)

    def test_symlink_escape(self):
        real = self.t / "elsewhere"
        real.mkdir(mode=0o700)
        os.rmdir(self.root)
        os.symlink(real, self.root)
        r = self.pf()
        self.assertFalse(r["checks"]["not_symlink"])
        self.assertFalse(r["ok"])
        os.unlink(self.root)
        os.mkdir(self.root, 0o700)
        os.symlink(real, os.path.join(self.root, "E00"))          # symlinked group directory inside the root
        self.assertFalse(self.pf()["checks"]["no_unexpected_contents"])

    def test_path_traversal_and_bank_root_overlap(self):
        r = ST.preflight(self.root + "/../m10", self.exp, identity_fn=lambda p: self.fs)
        self.assertFalse(r["checks"]["path_is_canonical"])
        self.exp["bank_root"] = self.root + "/banks"
        self.assertFalse(self.pf()["checks"]["separate_from_bank_root"])
        self.assertFalse(ST.preflight(None, None, identity_fn=lambda p: self.fs)["ok"]
                         and os.path.exists("/nonexistent-m10"))
        self.assertTrue(ST.preflight(ST.M10_ROOT, None, identity_fn=lambda p: {})["checks"]["separate_from_bank_root"])

    def test_unexpected_existing_output(self):
        Path(self.root, "stray.txt").write_text("x")
        self.assertFalse(self.pf()["checks"]["no_unexpected_contents"])
        os.unlink(Path(self.root, "stray.txt"))
        os.makedirs(Path(self.root, "E00", "seed_7"), mode=0o700)
        os.chmod(Path(self.root, "E00"), 0o700)
        self.assertEqual(self.pf()["unexpected_entries"], ["E00/seed_7"])

    def test_insufficient_capacity(self):
        self.assertFalse(self.pf(free_bytes=2 ** 30)["checks"]["free_bytes"])
        self.assertFalse(self.pf(free_inodes=10)["checks"]["free_inodes"])

    def test_create_run_root_is_create_only(self):
        fn = lambda p: self.fs                                       # noqa: E731
        run = ST.create_run_root(self.root, "E03", 42, expected=self.exp, identity_fn=fn)
        self.assertEqual(run, os.path.join(self.root, "E03", "seed_42"))
        self.assertEqual(oct(os.stat(run).st_mode & 0o7777), "0o700")
        with self.assertRaises(ST.StorageError):
            ST.create_run_root(self.root, "E03", 42, expected=self.exp, identity_fn=fn)
        with self.assertRaises(ST.StorageError):
            ST.create_run_root(self.root, "E07c", 42, expected=self.exp, identity_fn=fn)
        with self.assertRaises(ST.StorageError):
            ST.create_run_root(self.root, "E03", 7, expected=self.exp, identity_fn=fn)
        self.assertTrue(self.pf()["ok"])

    def test_create_root_refuses_bad_parent(self):
        os.rmdir(self.root)
        bad = dict(self.fs, filesystem_uuid="x")
        with self.assertRaises(ST.StorageError):
            ST.create_root(self.exp, identity_fn=lambda p: bad)
        self.assertFalse(os.path.exists(self.root))
        os.mkdir(self.exp["bank_root"])
        r = ST.create_root(self.exp, identity_fn=lambda p: self.fs)
        self.assertTrue(r["preflight"]["ok"])

    def test_storage_authority_record(self):
        p = ROOT / ST.STORAGE_AUTHORITY
        if not p.exists():
            self.skipTest("storage authority not yet written")
        reg = ST.load_storage_authority(ROOT)
        self.assertEqual((reg["root"], reg["filesystem_uuid"], reg["bank_root"]),
                         (ST.M10_ROOT, "608647d2-9cb0-45b0-b999-73c080ba909b", ST.BANK_ROOT))


# ------------------------------------------------------------------------------------------ manifest / order / preflight
def good_manifest(group="E03", seed=42):
    return {"schema": RM.SCHEMA, "label": "SCIENTIFIC", "run_id": "r", "method_id": group, "track_id": RM.TRACKS[group],
            "detector_seed": seed, "bank_seed": None if group == "E00" else seed, "source_commit": "a" * 40,
            "execution_commit": "b" * 40, "code_sha256": {"x": "c" * 64}, "evaluator_contract_sha256": "d" * 64,
            "frozen_config_sha256": "e" * 64, "owner_decisions_sha256": "f" * 64, "precision_pin_sha256": "1" * 64,
            "launch_authority": {"sha256": "2" * 64}, "storage_authority": {"sha256": "3" * 64},
            "split_manifest_sha256": "4" * 64, "m2_accounting_sha256": "5" * 64, "pool_sha256": {}, "pool_counts": {},
            "synthetic_bank": None if group == "E00" else {"method_id": group, "seed": seed, "manifest_sha256": "6" * 64},
            "weights_sha256": "7" * 64,
            "environment": {k: "x" for k in RM.ENV_KEYS}, "gpu": {k: "x" for k in RM.GPU_KEYS},
            "precision_policy": {"name": "CUDA_AMP_FP16_GRADSCALER_DETERMINISTIC"}, "grad_scaler": {}, "determinism": {},
            "amp_observed": {}, "optimizer": {}, "scheduler": {}, "rng": {}, "command": "c", "argv": [],
            "working_directory": "/", "run_root": "/r", "start_utc": "a", "end_utc": "b", "epochs": 40,
            "warmup_epochs": 3, "steps_per_epoch": 175, "total_steps": 7000, "optimizer_steps_applied": 6990,
            "optimizer_steps_skipped": 10, "epoch_metrics": [{k: 0 for k in RM.EPOCH_KEYS} for _ in range(40)],
            "checkpoint": {"path": "p", "file_sha256": "8" * 64},
            "val_predictions": [{"sha256": "9" * 64} for _ in range(40)],
            "selection": dict({k: "x" for k in RM.SELECTION_KEYS}, selected_epoch=3, selection_source="VAL only"),
            "logs": {}, "firewall": {"violation_count": 0, "listing_attempts": 0}, "exit_status": {"exit_code": 0},
            "audit": {"status": "PENDING_INDEPENDENT_AUDIT"}, "reproduction": ["x"], "test_accessed": False,
            "backup": "no independent copy"}


class RunManifestV2(unittest.TestCase):
    def test_valid(self):
        for g in RM.GROUPS:
            self.assertTrue(RM.validate(good_manifest(g))["ok"], g)

    def test_refusals(self):
        cases = {"wrong track (E06b as common track)": ("E06b", {"track_id": "A_FAIR_COMMON_IDENTITY_FREE"}),
                 "E00 with bank": ("E00", {"bank_seed": 42}),
                 "wrong bank seed": ("E03", {"bank_seed": 1337}),
                 "BF16 policy": ("E03", {"precision_policy": {"name": "CUDA_AMP_BF16_DETERMINISTIC"}}),
                 "39 epochs": ("E03", {"epochs": 39}),
                 "test accessed": ("E03", {"test_accessed": True}),
                 "self-certified audit": ("E03", {"audit": {"status": "PASS"}}),
                 "nonzero exit": ("E03", {"exit_status": {"exit_code": 1}}),
                 "firewall violation": ("E03", {"firewall": {"violation_count": 1, "listing_attempts": 0}}),
                 "step accounting": ("E03", {"optimizer_steps_skipped": 11}),
                 "no authority": ("E03", {"launch_authority": None}),
                 "short commit": ("E03", {"source_commit": "abc"})}
        for name, (g, patch) in cases.items():
            d = good_manifest(g)
            d.update(patch)
            with self.assertRaises(RM.ManifestError, msg=name):
                RM.validate(d)
        d = good_manifest()
        del d["epoch_metrics"]
        with self.assertRaises(RM.ManifestError):
            RM.validate(d)


class OrderAndPreflight(unittest.TestCase):
    def test_seed_major_order(self):
        self.assertEqual(len(L.ORDER), 24)
        self.assertEqual(L.ORDER[:8], tuple((g, 42) for g in L.GROUPS))
        self.assertEqual(L.ORDER[8], ("E00", 1337))
        self.assertEqual(L.ORDER[-1], ("E06c", 2026))
        self.assertEqual(L.predecessors("E00", 42), [])
        self.assertEqual(L.predecessors("E01", 1337)[-1], ("E00", 1337))
        with self.assertRaises(L.LaunchError):
            L.predecessors("E07c", 42)

    def test_tracks_and_owner_record(self):
        o = json.loads((ROOT / L.OWNER_PATH).read_text())
        self.assertEqual(o["decisions"], {"O1_science_precision": "A_CUDA_AMP_FP16_GRADSCALER_DETERMINISTIC",
                                          "O2_threshold_tie_rule": "A_SMALLER_THRESHOLD",
                                          "O3_checkpoint_retention_and_run_root": "A_SELECTED_ONLY_HDD_NEW_ROOT",
                                          "O4_science_environment": "A_APPROVE_UNCHANGED",
                                          "O5_execution_order_and_gpu_sharing": "A_SEED_MAJOR",
                                          "O6_primary_table_view": "A_RAW_MATCHED_PRIMARY",
                                          "O7_track_id_labels": "A_CONFIRM"})
        self.assertEqual(o["O7_track_labels"]["mapping"], RM.TRACKS)
        self.assertEqual(o["O7_track_labels"]["mapping"], A.TRACKS)
        import csv
        st = {r["experiment_id"]: r["track"] for r in csv.DictReader(open(ROOT / "outputs/audit/method_status.csv"))}
        for m, t in RM.TRACKS.items():
            if m != "E00":
                self.assertEqual(st[m], t)
        self.assertEqual(RM.TRACKS["E06b"], "B_NATIVE_FULL_SECONDARY")
        self.assertFalse(o["authorizes"]["test_evaluation"] or o["authorizes"]["gpat"] or o["authorizes"]["dinov3"])
        self.assertEqual(o["based_on"]["configs/amendments/m8_downstream_evaluator_contract.yaml"], L.CONTRACT[1])
        self.assertEqual(o["based_on"]["configs/frozen/downstream_resnet18.yaml"], L.CONFIG[1])

    def test_frozen_files_unchanged(self):
        for p, want in (L.CONTRACT, L.CONFIG, L.LOCK, L.SPEC):
            self.assertEqual(L.sha_file(ROOT / p), want, p)

    def _pre(self, **kw):
        good_env = dict(L.ENV_EXPECTED, CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONHASHSEED="0", lock_path=L.LOCK[0],
                        executable="x", CUDA_VISIBLE_DEVICES="0")
        gpu = {"gpus": [{"index": 0, "name": "NVIDIA GeForce RTX 3090", "uuid": "GPU-b722cd9d-c9fa-223c-915e-cc09d9b862b7",
                         "driver": "580.178.04", "total_mib": 24576, "used_mib": 1000, "free_mib": 20000,
                         "utilization_pct": 50}], "compute_apps": [], "utc": "x"}
        env = kw.pop("env", good_env)
        gpu = kw.pop("gpu", gpu)
        t = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, t, True)
        args = dict(repo=ROOT, group="E00", seed=42, mode="NON_SCIENTIFIC_QUALIFICATION", out_root=t,
                    runtime_root="/nonexistent-rt", weights_path=WEIGHTS, lock_file=os.path.join(t, "lock"),
                    env=env, gpu=gpu)
        args.update(kw)
        return L.preflight(**args)

    def test_preflight_environment_and_gpu_gates(self):
        g = self._pre()["gates"]
        self.assertTrue(g["environment"]["ok"])
        self.assertTrue(g["gpu"]["ok"])
        self.assertTrue(g["frozen_hashes"]["ok"])
        self.assertTrue(g["owner_decisions"]["ok"] and g["precision_pin"]["ok"])
        bad = dict(L.ENV_EXPECTED, torch="2.13.0+cu130", CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONHASHSEED="0")
        self.assertFalse(self._pre(env=bad)["gates"]["environment"]["ok"])
        bad = dict(L.ENV_EXPECTED, CUBLAS_WORKSPACE_CONFIG=None, PYTHONHASHSEED="0")
        self.assertFalse(self._pre(env=bad)["gates"]["environment"]["ok"])
        low = {"gpus": [{"index": 0, "name": "NVIDIA GeForce RTX 3090", "uuid": "u", "driver": "d", "total_mib": 24576,
                         "used_mib": 23000, "free_mib": 1500, "utilization_pct": 99}], "compute_apps": []}
        self.assertFalse(self._pre(gpu=low)["gates"]["gpu"]["ok"])
        other = {"gpus": [{"index": 0, "name": "NVIDIA A100", "uuid": "u", "driver": "d", "total_mib": 81920,
                           "used_mib": 0, "free_mib": 81920, "utilization_pct": 0}], "compute_apps": []}
        self.assertFalse(self._pre(gpu=other)["gates"]["gpu"]["ok"])

    def test_preflight_qualification_root_must_be_outside_official_roots(self):
        for bad in (ST.M10_ROOT, ST.BANK_ROOT + "/q"):
            self.assertFalse(self._pre(out_root=bad)["gates"]["qualification_root_outside_official_roots"]["ok"], bad)

    def test_preflight_held_lock_refuses(self):
        t = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, t, True)
        lk = L.ExclusiveLock(os.path.join(t, "lock"))
        self.assertTrue(lk.acquire())
        try:
            self.assertFalse(self._pre(lock_file=os.path.join(t, "lock"))["gates"]["no_concurrent_detector"]["ok"])
            self.assertFalse(L.ExclusiveLock(os.path.join(t, "lock")).acquire())
        finally:
            lk.release()

    def test_scientific_preflight_needs_active_authority(self):
        draft = json.loads((ROOT / "configs/amendments/m10_resnet18_science_launch_authority.DRAFT.yaml").read_text())
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / L.AUTHORITY_PATH
            p.parent.mkdir(parents=True)
            p.write_text(json.dumps(draft))
            with self.assertRaises(L.LaunchError):
                L.load_authority(t)
            p.write_text(json.dumps(dict(draft, status="OWNER_APPROVED_ACTIVE", approved=True,
                                         authorizes={"scientific_train_val_execution": True, "test_evaluation": True,
                                                     "gpat": False, "dinov3": False})))
            with self.assertRaises(L.LaunchError):                    # TEST in scope is refused
                L.load_authority(t)

    def test_cli_refuses_hidden_overrides_in_scientific_mode(self):
        for extra in (["--qualification-epochs", "2"], ["--qualification-policy", "CUDA_AMP_BF16_DETERMINISTIC"],
                      ["--qualification-overrides", '{"grad_scaler_init_scale": 1}']):
            with self.assertRaises(SystemExit):
                L.main(["--group", "E00", "--seed", "42", "--runtime-root", "/x", *extra])

    def test_launcher_has_no_test_input(self):
        src = (ROOT / "gpatbench/downstream/m10_launcher.py").read_text() + \
            (ROOT / "gpatbench/downstream/science_runner.py").read_text()
        self.assertNotIn("holdout", src.replace("no TEST / holdout", ""))
        self.assertNotIn('"TEST"', src)

    def test_unscale_before_clip_order_in_source(self):
        src = (ROOT / "gpatbench/downstream/science_runner.py").read_text()
        i = [src.index(x) for x in ("scaler.scale(loss).backward()", "scaler.unscale_(opt)",
                                    "torch.nn.utils.clip_grad_norm_(model.parameters(), R.FROZEN[\"grad_clip_norm\"])",
                                    "scaler.step(opt)", "scaler.update()")]
        self.assertEqual(i, sorted(i))


class Populations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.pop = A.rebuild_populations(ROOT)
        except Exception as exc:                                       # noqa: BLE001
            raise unittest.SkipTest(f"split manifests unavailable: {exc}")

    def test_counts_disjointness_and_allowlist(self):
        self.assertEqual({k: len(v) for k, v in self.pop["ids"].items()},
                         {"train_live": 5629, "train_real_spoof": 8838, "val": 3121})
        self.assertEqual(self.pop["val_videos"], 391)
        self.assertTrue(self.pop["train_val_disjoint"])
        self.assertEqual(self.pop["allowlist_size"], 5629 + 8838 + 3121)
        ids = set().union(*map(set, self.pop["ids"].values()))
        self.assertEqual(len(ids), 5629 + 8838 + 3121)

    def test_auditor_rebuild_equals_launcher_inputs(self):
        try:
            from gpatbench.downstream import science_inputs as SI
        except ImportError as exc:
            self.skipTest(str(exc))
        p = SI.build_populations(ROOT)
        self.assertEqual(p["pool_sha256"], self.pop["pool_sha256"])
        self.assertEqual({k: [f["sample_id"] for f in v] for k, v in p["pools"].items()}, self.pop["ids"])
        self.assertTrue(all(f["split"] in ("TRAIN", "VAL") for v in p["pools"].values() for f in v))

    def test_raw_matched_all_banks_include_e04_e05(self):
        from gpatbench.downstream import science_banks as SB
        for m in SB.METHODS:
            for s in SB.SEEDS:
                b = SB.bank_binding(ROOT, m, s)
                self.assertEqual((b["output_count"], b["failure_count"]), (8838, 0), (m, s))


# ------------------------------------------------------------------------------------------ torch: runs + auditor
def _fixture_run(base, group, *, epochs=2, warmup=1, policy="CPU_FP32_DETERMINISTIC", overrides=None, n_live=64,
                 n_val_videos=8, seed=42, install_firewall=False):
    from gpatbench.downstream import m10_fixture as FX
    fx = Path(base) / ("fx_" + group)
    FX.build_fixture(fx, group=group, seed=seed, n_live=n_live, n_spoof=48, n_syn=48, n_val_videos=n_val_videos)
    inputs = L.fixture_inputs(fx)
    env = {"name": "fixture", "python": sys.version.split()[0], "sys_prefix": sys.prefix, "lock_path": L.LOCK[0],
           "lock_sha256": L.LOCK[1], "torch": "t", "torchvision": "tv", "cuda": None, "cudnn": 0}
    gpu = {"name": "cpu", "uuid": "none", "driver": "none", "total_mib": 0}
    run_root = str(Path(base) / "out" / group / ("seed_%d" % seed))
    code = L.code_sha256(ROOT)
    pre = {"ok": True, "gates": {}, "context": {
        "run_root": run_root, "execution_commit": "0" * 40, "source_commit": "0" * 40, "code_sha256": code,
        "owner_decisions_sha256": L.sha_file(ROOT / L.OWNER_PATH), "precision_pin_sha256": L.sha_file(ROOT / L.PIN_PATH),
        "environment": env, "gpu": gpu, "weights_sha256": L.sha_file(WEIGHTS), "storage_authority": None}}
    rc = L.execute(repo=ROOT, group=group, seed=seed, mode="NON_SCIENTIFIC_QUALIFICATION", inputs=inputs, pre=pre,
                   weights_path=WEIGHTS, epochs=epochs, warmup_epochs=warmup, argv=["test"], command="test",
                   install_firewall=install_firewall, policy_override=policy, qualification_overrides=overrides)
    exp = FX.expectations(fx, run_root=run_root, group=group, seed=seed, epochs=epochs, warmup_epochs=warmup, env=env,
                          gpu={k: gpu[k] for k in ("name", "uuid", "driver")}, code_sha256=code, source_commit="0" * 40,
                          owner_decisions_sha256=pre["context"]["owner_decisions_sha256"],
                          precision_pin_sha256=pre["context"]["precision_pin_sha256"], policy_name=policy,
                          qualification_overrides=overrides, firewall_required=install_firewall)
    return rc, Path(run_root), exp


@unittest.skipUnless(HAVE_TORCH and HAVE_WEIGHTS, "torch + ResNet-18 weights")
class AuditorOnFixtureRuns(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t = Path(tempfile.mkdtemp())
        cls.runs = {}
        for g in ("E03", "E00", "E06b"):
            rc, rr, exp = _fixture_run(cls.t / g, g)
            assert rc == 0, (g, (rr / "exit_status.json").read_text())
            cls.runs[g] = (rr, exp)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.t, ignore_errors=True)

    def copy(self, g="E03"):
        rr, exp = self.runs[g]
        d = Path(tempfile.mkdtemp(dir=self.t)) / g / "seed_42"
        shutil.copytree(rr, d)
        os.chmod(d, 0o700)
        e = copy.deepcopy(exp)
        e["run_root"] = str(d)
        if e["bank"]:
            bd = Path(tempfile.mkdtemp(dir=self.t))
            shutil.copy(e["bank"]["manifest_file"], bd / "manifest.json")
            e["bank"]["manifest_file"] = str(bd / "manifest.json")
        m = json.loads((d / "run_manifest.json").read_text())
        m["run_root"] = str(d)
        (d / "run_manifest.json").write_text(json.dumps(m))
        return d, e

    def audit(self, d, e, **kw):
        return A.audit(d, e, **kw)

    def assertFails(self, d, e, gate, **kw):
        r = self.audit(d, e, **kw)
        self.assertEqual(r["status"], "FAIL")
        self.assertIn(gate, r["failed_gates"], r["failed_gates"])
        return r

    # -------------------------------------------------------------------------------- positives
    def test_good_runs_pass(self):
        for g in self.runs:
            d, e = self.copy(g)
            r = self.audit(d, e)
            self.assertEqual(r["status"], "PASS", (g, r["failed_gates"], {k: r["gates"][k] for k in r["failed_gates"]}))

    def test_run_evidence_and_selected_only_retention(self):
        rr, _ = self.runs["E03"]
        det = rr / "detector"
        sel = json.loads((det / "selection.json").read_text())
        ep = json.loads((det / "epochs.json").read_text())
        self.assertEqual(sorted(p.name for p in det.glob("checkpoint_*.pt")), [sel["checkpoint_file"]])
        self.assertEqual(json.loads((det / "best_so_far.json").read_text())["epoch"], sel["selected_epoch"])
        self.assertEqual(sel["selected_epoch"], S10.select_epoch(ep)["epoch"])
        import torch
        from gpatbench.downstream import resnet_runner as R
        st = torch.load(det / sel["checkpoint_file"], map_location="cpu", weights_only=True)
        self.assertEqual(R.state_sha256(st), sel["checkpoint_state_sha256"])
        self.assertEqual(A.state_digest(st), sel["checkpoint_state_sha256"])
        self.assertEqual(L.sha_file(det / sel["checkpoint_file"]), sel["checkpoint_file_sha256"])
        steps = [json.loads(x) for x in (det / "steps.jsonl").read_text().splitlines()]
        self.assertTrue(all((s["n_live"], s["n_real_spoof"], s["n_synthetic"]) == (32, 16, 16) for s in steps))
        m = json.loads((rr / "run_manifest.json").read_text())
        RM.validate(m)
        self.assertEqual(m["track_id"], "A_FAIR_COMMON_IDENTITY_FREE")
        self.assertFalse(any(p.name.endswith(".tmp") for p in det.iterdir()))

    def test_e00_real_only_and_e06b_track(self):
        steps = [json.loads(x) for x in (self.runs["E00"][0] / "detector/steps.jsonl").read_text().splitlines()]
        self.assertTrue(all((s["n_live"], s["n_real_spoof"], s["n_synthetic"]) == (32, 32, 0) for s in steps))
        import pyarrow.parquet as pq
        rr = self.runs["E06b"][0]
        tr = set(pq.read_table(rr / "detector/predictions_val_epoch00.parquet", columns=["track_id"]).column(0).to_pylist())
        self.assertEqual(tr, {"B_NATIVE_FULL_SECONDARY"})

    # -------------------------------------------------------------------------------- injected failures
    def test_tampered_checkpoint(self):
        d, e = self.copy()
        ck = next((d / "detector").glob("checkpoint_*.pt"))
        b = bytearray(ck.read_bytes())
        b[len(b) // 2] ^= 0xFF
        ck.write_bytes(bytes(b))
        self.assertFails(d, e, "A27_selected_checkpoint_file_and_reload")

    def test_missing_run_manifest(self):
        d, e = self.copy()
        os.unlink(d / "run_manifest.json")
        self.assertFails(d, e, "A00_required_evidence_present")

    def test_altered_threshold(self):
        d, e = self.copy()
        p = d / "detector/selection.json"
        s = json.loads(p.read_text())
        s["frozen_val_threshold_hex"] = (0.123).hex()
        p.write_text(json.dumps(s))
        self.assertFails(d, e, "A26_checkpoint_selection_D5_and_frozen_threshold")

    def test_altered_val_predictions_consistently_rehashed(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        d, e = self.copy()
        det = d / "detector"
        p = det / "predictions_val_epoch00.parquet"
        rows = pq.read_table(p).to_pylist()
        rows[0]["p_frame"] = min(1.0, rows[0]["p_frame"] + 0.01)
        os.unlink(p)
        pq.write_table(pa.Table.from_pylist(rows), p, compression="zstd")
        new = L.sha_file(p)
        ep = json.loads((det / "epochs.json").read_text())
        ep[0]["predictions_sha256"] = new
        (det / "epochs.json").write_text(json.dumps(ep))
        m = json.loads((d / "run_manifest.json").read_text())
        m["val_predictions"][0]["sha256"] = new
        (d / "run_manifest.json").write_text(json.dumps(m))
        r = self.assertFails(d, e, "A23_val_predictions_complete")
        self.assertIn("p_frame != sigmoid(logit)", json.dumps(r["gates"]["A23_val_predictions_complete"]))

    def test_incorrect_epoch_count(self):
        d, e = self.copy()
        p = d / "detector/epochs.json"
        ep = json.loads(p.read_text())
        p.write_text(json.dumps(ep[:-1]))
        self.assertFails(d, e, "A15_epochs_and_step_count")

    def test_incorrect_optimizer_steps(self):
        d, e = self.copy()
        p = d / "detector/steps.jsonl"
        lines = p.read_text().splitlines()
        p.write_text("\n".join(lines[:-1]) + "\n")
        self.assertFails(d, e, "A15_epochs_and_step_count")

    def test_wrong_amp_policy(self):
        d, e = self.copy()
        m = json.loads((d / "run_manifest.json").read_text())
        m["precision_policy"]["name"] = "CUDA_AMP_BF16_DETERMINISTIC"
        (d / "run_manifest.json").write_text(json.dumps(m))
        self.assertFails(d, e, "A19_amp_gradscaler_policy_O1")
        d, e = self.copy()
        e["expect_o1_policy"] = True                                  # a CPU fp32 run is not the O1 policy
        self.assertFails(d, e, "A19_amp_gradscaler_policy_O1")

    def _patch_manifest(self, d, **kw):
        m = json.loads((d / "run_manifest.json").read_text())
        for k, v in kw.items():
            if isinstance(v, dict) and isinstance(m.get(k), dict):
                m[k].update(v)
            else:
                m[k] = v
        (d / "run_manifest.json").write_text(json.dumps(m))

    def test_wrong_environment(self):
        d, e = self.copy()
        self._patch_manifest(d, environment={"torch": "2.13.0"})
        self.assertFails(d, e, "A08_environment")

    def test_wrong_bank_seed(self):
        d, e = self.copy()
        self._patch_manifest(d, bank_seed=1337)
        self.assertFails(d, e, "A13_seed_matched_bank_mapping")

    def test_wrong_track_label(self):
        d, e = self.copy("E06b")
        self._patch_manifest(d, track_id="A_FAIR_COMMON_IDENTITY_FREE")
        self.assertFails(d, e, "A11_track_label_O7")

    def test_wrong_gpu(self):
        d, e = self.copy()
        self._patch_manifest(d, gpu={"uuid": "GPU-00000000"})
        self.assertFails(d, e, "A09_gpu_identity")

    def test_missing_exit_marker(self):
        d, e = self.copy()
        os.unlink(d / "exit_status.json")
        self.assertFails(d, e, "A03_exit_status_and_completion")
        d, e = self.copy()
        e["exit_marker_required"] = True
        self.assertFails(d, e, "A03_exit_status_and_completion")
        d, e = self.copy()
        mk = Path(tempfile.mkdtemp(dir=self.t)) / "x.exit"
        mk.write_text("1\n")
        self.assertFails(d, e, "A03_exit_status_and_completion", exit_marker=str(mk))
        mk.write_text("0\n")
        self.assertEqual(self.audit(d, e, exit_marker=str(mk))["status"], "PASS")

    def test_partial_run(self):
        d, e = self.copy()
        p = d / "run_status.json"
        s = json.loads(p.read_text())
        s["events"] = s["events"][:1]
        s["status"] = "STARTED"
        p.write_text(json.dumps(s))
        self.assertFails(d, e, "A03_exit_status_and_completion")
        d, e = self.copy()
        (d / "detector/failure.json").write_text("{}")
        self.assertFails(d, e, "A01_run_root_layout_selected_only")

    def test_corrupted_output(self):
        d, e = self.copy()
        p = d / "detector/predictions_val_epoch01.parquet"
        p.write_bytes(p.read_bytes()[:100])
        self.assertFails(d, e, "A23_val_predictions_complete")
        d, e = self.copy()
        (d / "detector/epochs.json").write_text("{not json")
        self.assertFails(d, e, "A00_required_evidence_present")

    def test_altered_bank_manifest(self):
        d, e = self.copy()
        p = Path(e["bank"]["manifest_file"])
        p.write_text(p.read_text().replace("Y0000", "Y9999"))
        self.assertFails(d, e, "A13_seed_matched_bank_mapping")

    def test_unauthorized_test_read(self):
        d, e = self.copy()
        p = d / "firewall_report.json"
        fw = json.loads(p.read_text())
        fw["violation_count"] = 1
        fw["violations"] = [{"event": "open", "path": "faces/fixture/T0000.png"}]
        p.write_text(json.dumps(fw))
        self.assertFails(d, e, "A29_test_firewall_zero_access")
        d, e = self.copy()
        e["firewall_required"] = True                                 # an unfirewalled run fails a firewalled profile
        self.assertFails(d, e, "A29_test_firewall_zero_access")

    def test_duplicate_or_extra_run_root_content(self):
        d, e = self.copy()
        ck = next((d / "detector").glob("checkpoint_*.pt"))
        shutil.copy(ck, d / "detector" / "checkpoint_epoch00.pt" if ck.name != "checkpoint_epoch00.pt"
                    else d / "detector" / "checkpoint_epoch01.pt")
        self.assertFails(d, e, "A01_run_root_layout_selected_only")
        d, e = self.copy()
        (d / "detector.rerun").mkdir()
        self.assertFails(d, e, "A01_run_root_layout_selected_only")
        d, e = self.copy()
        e["run_root"] = str(d.parent / "seed_1337")
        self.assertFails(d, e, "A01_run_root_layout_selected_only")

    def test_altered_lr_and_batch_and_scaler(self):
        d, e = self.copy()
        p = d / "detector/steps.jsonl"
        steps = [json.loads(x) for x in p.read_text().splitlines()]
        steps[1]["lr_head_hex"] = (1e-3).hex()
        steps[2]["batch_ids_sha256"] = "0" * 64
        steps[0]["n_synthetic"], steps[0]["n_real_spoof"] = 32, 0
        p.write_text("".join(json.dumps(s) + "\n" for s in steps))
        r = self.assertFails(d, e, "A17_lr_schedule_D6")
        self.assertIn("A16_sampler_batches_rederived", r["failed_gates"])
        self.assertIn("A14_batch_composition", r["failed_gates"])

    def test_wrong_pools_and_test_ids(self):
        d, e = self.copy()
        p = d / "pools.json"
        pj = json.loads(p.read_text())
        pj["val"][0] = "T0000"
        p.write_text(json.dumps(pj))
        self.assertFails(d, e, "A12_train_val_split_identity")


@unittest.skipUnless(HAVE_TORCH and HAVE_WEIGHTS, "torch + ResNet-18 weights")
class RunnerFailClosedAndRetention(unittest.TestCase):
    def setUp(self):
        self.t = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.t, ignore_errors=True)

    def test_nonfinite_loss_stops_with_partial_evidence(self):
        rc, rr, _ = _fixture_run(self.t, "E00", overrides={"nonfinite_loss_at_step": 3})
        self.assertEqual(rc, 1)
        f = json.loads((rr / "detector/failure.json").read_text())
        self.assertIn("non-finite loss at step 3", f["error"])
        self.assertEqual(len((rr / "detector/steps.jsonl").read_text().splitlines()), 3)
        self.assertEqual(len(json.loads((rr / "detector/epochs.json").read_text())), 1)   # epoch 0 kept
        self.assertEqual(len(list((rr / "detector").glob("checkpoint_*.pt"))), 1)          # best-so-far kept
        self.assertEqual(json.loads((rr / "run_status.json").read_text())["status"], "FAILED")
        self.assertEqual(json.loads((rr / "exit_status.json").read_text())["exit_code"], 1)

    def test_nonfinite_gradient_without_scaler_refused(self):
        rc, rr, _ = _fixture_run(self.t, "E00", overrides={"nonfinite_grad_at_step": 0})
        self.assertEqual(rc, 1)
        self.assertIn("non-finite gradient", json.loads((rr / "detector/failure.json").read_text())["error"])

    def test_best_so_far_retention_across_epochs(self):
        """Scripted VAL outcomes force a best-so-far trajectory 0 -> 1 -> 3 with ties; files follow it exactly."""
        from gpatbench.downstream import science_runner as SR
        script = iter([("1/2", "1/2"), ("1/4", "3/5"), ("1/4", "3/5"), ("1/8", "1/2"), ("1/8", "1/2"), ("1/4", "1")])
        real_sel, real_auc = SR.S10.select_threshold_o2, SR.M.auc
        cur = {}

        def fake_sel(y, s):
            a, u = next(script)
            cur["auc"] = Fraction(u)
            r = real_sel(y, s)
            r["val_acer"] = Fraction(a)
            return r
        SR.S10.select_threshold_o2, SR.M.auc = fake_sel, (lambda y, s: cur["auc"])
        try:
            rc, rr, _ = _fixture_run(self.t, "E00", epochs=6, warmup=1, n_live=32)
        finally:
            SR.S10.select_threshold_o2, SR.M.auc = real_sel, real_auc
        self.assertEqual(rc, 0, (rr / "exit_status.json").read_text())
        ep = json.loads((rr / "detector/epochs.json").read_text())
        self.assertEqual([e["became_best"] for e in ep], [True, True, False, True, False, False])
        self.assertEqual(ep[3]["checkpoint_event"]["removed_previous_best"], "checkpoint_epoch01.pt")
        self.assertEqual(sorted(p.name for p in (rr / "detector").glob("checkpoint_*.pt")), ["checkpoint_epoch03.pt"])
        self.assertEqual(json.loads((rr / "detector/selection.json").read_text())["selected_epoch"], 3)

    def test_failed_persist_keeps_previous_best(self):
        from gpatbench.downstream import science_runner as SR
        real = SR._persist_checkpoint
        n = {"calls": 0}
        script = iter([("1/2", "1/2"), ("1/4", "1/2"), ("1/8", "1/2")])
        real_sel = SR.S10.select_threshold_o2

        def fake_sel(y, s):
            r = real_sel(y, s)
            r["val_acer"] = Fraction(next(script)[0])
            return r

        def flaky(state, ssha, epoch, d):
            n["calls"] += 1
            if n["calls"] == 3:
                raise OSError("simulated write failure")
            return real(state, ssha, epoch, d)
        SR._persist_checkpoint, SR.S10.select_threshold_o2 = flaky, fake_sel
        try:
            rc, rr, _ = _fixture_run(self.t, "E00", epochs=3, warmup=1, n_live=32)
        finally:
            SR._persist_checkpoint, SR.S10.select_threshold_o2 = real, real_sel
        self.assertEqual(rc, 1)
        det = rr / "detector"
        self.assertEqual(sorted(p.name for p in det.glob("checkpoint_*.pt")), ["checkpoint_epoch01.pt"])
        self.assertEqual(json.loads((det / "best_so_far.json").read_text())["epoch"], 1)
        self.assertIn("simulated write failure", json.loads((det / "failure.json").read_text())["error"])

    def test_persist_checkpoint_atomic_and_verified(self):
        import torch
        from gpatbench.downstream import resnet_runner as R
        from gpatbench.downstream import science_runner as SR
        st = {"w": torch.arange(6.0).reshape(2, 3)}
        ptr = SR._persist_checkpoint(st, R.state_sha256(st), 4, self.t)
        self.assertEqual(ptr["checkpoint_file"], "checkpoint_epoch04.pt")
        self.assertEqual(L.sha_file(self.t / "checkpoint_epoch04.pt"), ptr["checkpoint_file_sha256"])
        self.assertFalse((self.t / "checkpoint_epoch04.pt.tmp").exists())
        with self.assertRaises(SR.ScienceRunError):
            SR._persist_checkpoint(st, R.state_sha256(st), 4, self.t)              # no overwrite
        with self.assertRaises(SR.ScienceRunError):
            SR._persist_checkpoint(st, "0" * 64, 5, self.t)                        # reload must verify

    def test_scientific_mode_refusals(self):
        from gpatbench.downstream import science_runner as SR
        kw = dict(method_id="E03", seed=42, live=[{"split": "TRAIN"}], real_spoof=[], synthetic=[{}], val=[],
                  load_real=None, load_synthetic=None, weights_path=WEIGHTS, out_dir=self.t / "x",
                  split_manifest_sha256="0", bank_seed=42, track_id="A_FAIR_COMMON_IDENTITY_FREE")
        for pol, extra in (("CUDA_AMP_BF16_DETERMINISTIC", {}), ("CUDA_FP32_DETERMINISTIC", {}),
                           ("CPU_FP32_DETERMINISTIC", {}),
                           ("CUDA_AMP_FP16_GRADSCALER_DETERMINISTIC", {"epochs": 2}),
                           ("CUDA_AMP_FP16_GRADSCALER_DETERMINISTIC", {"retention": "ALL_EPOCHS"}),
                           ("CUDA_AMP_FP16_GRADSCALER_DETERMINISTIC", {"qualification_overrides": {"grad_scaler_init_scale": 1.0}})):
            with self.assertRaises(SR.ScienceRunError, msg=(pol, extra)):
                SR.train_detector(mode="SCIENTIFIC", policy=SR.POLICIES[pol], **dict(kw, **extra))
        self.assertFalse((self.t / "x").exists())
        with self.assertRaises(SR.ScienceRunError):                     # E00 is real-only
            SR.train_detector(mode="NON_SCIENTIFIC_QUALIFICATION", policy=SR.POLICIES["CPU_FP32_DETERMINISTIC"],
                              **dict(kw, method_id="E00", bank_seed=None))
        with self.assertRaises(SR.ScienceRunError):                     # D12 seed-matched bank
            SR.train_detector(mode="NON_SCIENTIFIC_QUALIFICATION", policy=SR.POLICIES["CPU_FP32_DETERMINISTIC"],
                              **dict(kw, bank_seed=1337))


@unittest.skipUnless(HAVE_TORCH and HAVE_WEIGHTS, "torch + ResNet-18 weights")
class SmokeRunnerEquivalence(unittest.TestCase):
    """The O1/O2/O3 changes keep the science runner bitwise equal to the qualified smoke runner on CPU (every step's
    lr / loss / grad norm / batch ids, every epoch's state sha256, logits and threshold, the selected epoch)."""

    def test_cpu_bitwise_equal_to_smoke_runner(self):
        spec = importlib.util.spec_from_file_location("q", str(ROOT / "tools/m10_downstream_qualification.py"))
        q = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(q)
        from gpatbench.downstream import resnet_runner as R
        from gpatbench.downstream import science_runner as SR
        SR.configure(SR.POLICIES["CPU_FP32_DETERMINISTIC"])
        t = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, t, True)
        keys = ("step", "epoch", "lr_backbone_hex", "lr_head_hex", "loss_hex", "grad_norm_hex", "n_live",
                "n_real_spoof", "n_synthetic", "batch_ids_sha256")
        for m in ("E00", "E03"):
            for fseed in range(20261125, 20261135):
                store, live, spoof, syn, val = q.fixture(seed=fseed)
                load = q.loader(store)
                sy = [] if m == "E00" else syn
                try:
                    R.train_detector(method_id=m, seed=42, live=live, real_spoof=spoof, synthetic=sy, val_select=val,
                                     holdout=val, load_real=load, load_synthetic=load, weights_path=WEIGHTS, epochs=2,
                                     warmup_epochs=1, out_dir=t / ("smoke_%s_%d" % (m, fseed)),
                                     split_manifest_sha256="0" * 64)
                    break
                except M.UnresolvedThresholdTie:
                    continue
            sd = t / ("smoke_%s_%d" % (m, fseed))
            smoke_steps = [json.loads(x) for x in (sd / "steps.jsonl").read_text().splitlines()]
            smoke_ep = json.loads((sd / "epochs.json").read_text())
            smoke_sel = json.loads((sd / "detector.json").read_text())["selected_epoch"]
            for w in (0, 4):
                d = t / ("sci_%s_%d" % (m, w))
                r = SR.train_detector(mode="NON_SCIENTIFIC_QUALIFICATION", method_id=m, seed=42, live=live,
                                      real_spoof=spoof, synthetic=sy, val=val, load_real=load, load_synthetic=load,
                                      weights_path=WEIGHTS, out_dir=d, split_manifest_sha256="0" * 64,
                                      policy=SR.POLICIES["CPU_FP32_DETERMINISTIC"], epochs=2, warmup_epochs=1,
                                      workers=w, bank_seed=None if m == "E00" else 42)
                sci = [json.loads(x) for x in (d / "steps.jsonl").read_text().splitlines()]
                self.assertEqual([{k: s[k] for k in keys} for s in sci], [{k: s[k] for k in keys} for s in smoke_steps])
                for k in ("state_sha256", "threshold_hex", "logits_sha256", "val_acer", "val_auc"):
                    self.assertEqual([e[k] for e in r["epochs"]], [e[k] for e in smoke_ep], (m, w, k))
                self.assertEqual(r["selection"]["selected_epoch"], smoke_sel)


@unittest.skipUnless(HAVE_TORCH, "torch")
class TestFirewallSubprocess(unittest.TestCase):
    def test_decoy_open_and_listing_refused(self):
        t = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, t, True)
        (t / "faces" / "fixture").mkdir(parents=True)
        for n in ("TRAIN.png", "TEST.png"):
            (t / "faces" / "fixture" / n).write_bytes(b"x")
        code = f"""
import os, sys, json
sys.path.insert(0, {str(ROOT)!r})
from gpatbench.downstream import audit_hook
audit_hook.install({str(t / 'faces')!r}, ['fixture/TRAIN.png'], {str(t / 'fw.json')!r})
open({str(t / 'faces/fixture/TRAIN.png')!r}, 'rb').read()
fd = os.open({str(t / 'faces/fixture/TRAIN.png')!r}, os.O_RDONLY); os.close(fd)
out = {{}}
for name, fn in (('open', lambda: open({str(t / 'faces/fixture/TEST.png')!r}, 'rb')),
                 ('os_open', lambda: os.open({str(t / 'faces/fixture/TEST.png')!r}, os.O_RDONLY)),
                 ('listdir', lambda: os.listdir({str(t / 'faces/fixture')!r})),
                 ('scandir', lambda: list(os.scandir({str(t / 'faces')!r})))):
    try:
        fn(); out[name] = 'allowed'
    except PermissionError:
        out[name] = 'refused'
print(json.dumps(out)); print(json.dumps(audit_hook.write_report()))
"""
        r = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stderr)
        out, rep = (json.loads(x) for x in r.stdout.splitlines()[:2])
        self.assertEqual(out, {"open": "refused", "os_open": "refused", "listdir": "refused", "scandir": "refused"})
        self.assertEqual((rep["violation_count"], rep["distinct_files_opened"]), (4, 1))


# ------------------------------------------------------------------------------------------ CUDA (O1)
@unittest.skipUnless(CUDA and HAVE_WEIGHTS and os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8",
                     "CUDA + CUBLAS_WORKSPACE_CONFIG=:4096:8 (GPU host)")
class CudaO1Policy(unittest.TestCase):
    O1 = "CUDA_AMP_FP16_GRADSCALER_DETERMINISTIC"

    def setUp(self):
        self.t = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.t, ignore_errors=True)

    def test_fp16_autocast_deterministic_config_and_audit(self):
        from gpatbench.downstream import science_runner as SR
        rc, rr, exp = _fixture_run(self.t, "E03", policy=self.O1)
        self.assertEqual(rc, 0, (rr / "exit_status.json").read_text())
        m = json.loads((rr / "run_manifest.json").read_text())
        self.assertEqual(m["amp_observed"], {"train_logits_dtype": "torch.float16", "val_logits_dtype": "torch.float16"})
        self.assertEqual(m["determinism"], SR.O1_DETERMINISM)
        self.assertEqual(m["grad_scaler"], SR.O1_SCALER)
        r = A.audit(rr, exp)
        self.assertEqual(r["status"], "PASS", {k: r["gates"][k] for k in r["failed_gates"]})

    def test_gradscaler_skip_recorded_then_recovers(self):
        rc, rr, exp = _fixture_run(self.t, "E03", policy=self.O1, overrides={"nonfinite_grad_at_step": 1})
        self.assertEqual(rc, 0, (rr / "exit_status.json").read_text())
        steps = [json.loads(x) for x in (rr / "detector/steps.jsonl").read_text().splitlines()]
        self.assertEqual([s["optimizer_step_applied"] for s in steps[:3]], [True, False, True])
        self.assertEqual((steps[1]["grad_scale_before"], steps[1]["grad_scale"]), (65536.0, 32768.0))
        self.assertEqual(steps[1]["grad_norm_post_clip_hex"], None)
        self.assertFalse(math.isfinite(float.fromhex(steps[1]["grad_norm_hex"])))
        m = json.loads((rr / "run_manifest.json").read_text())
        self.assertEqual((m["optimizer_steps_skipped"], m["optimizer_steps_applied"]), (1, m["total_steps"] - 1))
        r = A.audit(rr, exp)
        self.assertEqual(r["status"], "PASS", {k: r["gates"][k] for k in r["failed_gates"]})

    def test_skipped_step_leaves_parameters_unchanged(self):
        import torch
        from gpatbench.downstream import resnet_runner as R
        from gpatbench.downstream import science_runner as SR
        SR.configure(SR.POLICIES[self.O1])
        model = R.build_model(WEIGHTS, 0).cuda()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler("cuda")
        x = torch.randn(4, 3, 224, 224, device="cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(x).squeeze(1)
        self.assertEqual(out.dtype, torch.float16)
        loss = out.float().mean()
        before = {k: v.detach().clone() for k, v in model.state_dict().items()}
        scaler.scale(loss).backward()
        next(iter(model.parameters())).grad.view(-1)[0] = float("inf")
        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        self.assertFalse(math.isfinite(float(gn)))
        self.assertEqual(scaler.get_scale(), 32768.0)
        self.assertTrue(all(torch.equal(before[k], v) for k, v in model.state_dict().items()))

    def test_genuine_nonfinite_gradient_at_unit_scale_stops(self):
        rc, rr, _ = _fixture_run(self.t, "E00", policy=self.O1,
                                 overrides={"grad_scaler_init_scale": 1.0, "nonfinite_grad_at_step": 0})
        self.assertEqual(rc, 1)
        self.assertIn("grad scale 1.0 <= 1", json.loads((rr / "detector/failure.json").read_text())["error"])

    def test_bitwise_replay(self):
        a = _fixture_run(self.t / "a", "E06c", policy=self.O1)
        b = _fixture_run(self.t / "b", "E06c", policy=self.O1)
        for f in ("steps.jsonl", "epochs.json"):
            ja = (a[1] / "detector" / f).read_text()
            jb = (b[1] / "detector" / f).read_text()
            if f == "epochs.json":
                ja, jb = (json.dumps([{k: v for k, v in e.items() if k != "wall_seconds"} for e in json.loads(x)])
                          for x in (ja, jb))
            self.assertEqual(ja, jb, f)
        sa = json.loads((a[1] / "detector/selection.json").read_text())
        sb = json.loads((b[1] / "detector/selection.json").read_text())
        self.assertEqual(sa["checkpoint_file_sha256"], sb["checkpoint_file_sha256"])

    def test_missing_cublas_config_refused(self):
        code = ("import sys; sys.path.insert(0, %r)\n"
                "from gpatbench.downstream import science_runner as SR\n"
                "try:\n    SR.configure(SR.POLICIES['CUDA_AMP_FP16_GRADSCALER_DETERMINISTIC'])\n    print('configured')\n"
                "except SR.ScienceRunError as e:\n    print('refused', e)\n") % str(ROOT)
        env = {k: v for k, v in os.environ.items() if k != "CUBLAS_WORKSPACE_CONFIG"}
        r = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, env=env, timeout=300)
        self.assertTrue(r.stdout.startswith("refused"), r.stdout + r.stderr)


if __name__ == "__main__":
    unittest.main()
