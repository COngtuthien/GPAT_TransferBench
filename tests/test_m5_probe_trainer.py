"""Authoritative M5 ArtifactProbeNet trainer — control-flow and contract tests.

These exercise `gpatbench.probe.train` structurally with tiny fakes: no ResNet-18 is trained, no
30-epoch run is performed and no scientific checkpoint is produced. CUDA-specific *runtime*
behaviour (real autocast/GradScaler numerics, deterministic CUDA kernels) is not verifiable on a
CPU-only host and is marked DEFERRED_TO_GPU_VERIFICATION in the M5 report; what is verified here is
that the code path is CUDA-only, fails closed without it, and executes the frozen ordering.
"""
import csv
import inspect
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gpatbench.probe import contract as C      # noqa: E402
from gpatbench.probe import data as D          # noqa: E402
from gpatbench.probe import metrics as MET     # noqa: E402
from gpatbench.probe import train as T         # noqa: E402
sys.path.insert(0, str(ROOT / "tests"))
import m5_test_env as H                          # noqa: E402


def _executable_source(fn, *, strip_strings: bool = False) -> str:
    """Source of a function with docstring and comments removed, via AST round-trip.

    `strip_strings=True` also blanks every string literal, so a check for a forbidden CONSTRUCT is
    not tripped by an error message that merely names it.
    """
    import ast
    import textwrap

    class _Blank(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str):
                return ast.copy_location(ast.Constant(value=""), node)
            return node

        def visit_JoinedStr(self, node):
            return ast.copy_location(ast.Constant(value=""), node)

    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    node = tree.body[0]
    body = node.body[1:] if (node.body and isinstance(node.body[0], ast.Expr)
                             and isinstance(node.body[0].value, ast.Constant)
                             and isinstance(node.body[0].value.value, str)) else node.body
    mod = ast.Module(body=body, type_ignores=[])
    if strip_strings:
        mod = ast.fix_missing_locations(_Blank().visit(mod))
    return ast.unparse(mod)


# ------------------------------------------------------------------ fakes
class TinyNet(torch.nn.Module):
    """7-logit stand-in for ResNet-18. Small enough that nothing resembling training happens."""

    def __init__(self, in_features: int = 4):
        super().__init__()
        self.fc = torch.nn.Linear(in_features, C.K)

    def forward(self, x):
        return self.fc(x)


class Recorder:
    """Records the global order of the four per-epoch steps, independently of the loop's trace."""

    def __init__(self):
        self.calls = []

    # --- fake scaler (AMP surface only; real CUDA numerics are GPU-verified) ---
    def scale(self, loss):
        self.calls.append("scale")
        return loss

    def step(self, optimizer):
        self.calls.append("optimizer.step")
        optimizer.step()

    def update(self):
        self.calls.append("scaler.update")

    def is_enabled(self):
        return True


class FakeScheduler:
    def __init__(self, recorder):
        self.recorder = recorder
        self.steps = 0

    def step(self):
        self.steps += 1
        self.recorder.calls.append("scheduler.step")


@contextmanager
def _noop_autocast():
    yield


def autocast_factory(recorder=None):
    def _f():
        if recorder is not None:
            recorder.calls.append("autocast")
        return _noop_autocast()
    return _f


def tiny_batches(n_batches=2, batch=4, in_features=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for b in range(n_batches):
        x = torch.randn(batch, in_features, generator=g)
        y = torch.arange(batch) % C.K
        out.append((x, y))
    return out


def fake_provenance():
    return {"frozen_config_sha256": C.CONFIG_SHA256,
            "frozen_config_snapshot_sha256": C.CONFIG_SHA256,
            "split_manifest_sha256": C.SPLIT_MANIFEST_SHA256,
            "resnet18_weight_sha256": C.RESNET18_WEIGHT_SHA256,
            "class_weights": [0.5] * C.K,
            "determinism": {"seed": C.SEED, "cudnn_deterministic": True},
            "environment": {"torch": torch.__version__, "device": "cuda"},
            "environment_lock": {"env.torch": torch.__version__, "determinism.seed": C.SEED},
            "code_commit": "0" * 40, "code_dirty": False, "host": "test",
            "created_utc": "2026-09-21T00:00:00Z"}


def scripted_validate(scores, recorder=None):
    """A validate_fn whose macro-F1 follows a scripted per-epoch sequence."""
    seq = dict(scores)
    state = {"epoch": 0}

    def _v(model, loader, autocast, device):
        state["epoch"] += 1
        if recorder is not None:
            recorder.calls.append("validate")
        cm = np.zeros((C.K, C.K), dtype=np.int64)
        cm[0, 0] = 1
        return {"confusion_matrix": cm, "macro_f1": float(seq[state["epoch"]]),
                "accuracy": 1.0, "per_class_f1": [0.0] * C.K,
                "support": [1] + [0] * (C.K - 1), "predicted": [1] + [0] * (C.K - 1),
                "val_samples": 1}
    return _v


def counting_train(recorder=None, calls=None):
    def _t(model, loader, criterion, optimizer, scaler, autocast, device):
        if recorder is not None:
            recorder.calls.append("train")
        if calls is not None:
            calls.append("train")
        return {"train_loss": 1.0, "train_batches": 1, "train_samples": 1,
                "train_optimizer_steps": 1}
    return _t


def run_loop(scores, *, write_checkpoint=False, save_fn=None, recorder=None, epochs=None):
    rec = recorder or Recorder()
    model = TinyNet()
    opt = torch.optim.AdamW(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    sch = FakeScheduler(rec)
    return rec, T._train_loop(
        model=model, train_loader=[], val_loader=[], criterion=torch.nn.CrossEntropyLoss(),
        optimizer=opt, scheduler=sch, scaler=rec, autocast=autocast_factory(),
        device=torch.device("cpu"), provenance=fake_provenance(),
        epochs=epochs, train_fn=counting_train(rec), validate_fn=scripted_validate(scores, rec),
        save_fn=save_fn or (lambda *a, **k: None), write_checkpoint=write_checkpoint)


# ------------------------------------------------------------------ dry-run safety
class TestDryRunSafety(H.HermeticExecConfigMixin, unittest.TestCase):
    """Uses a temporary execution config: these assert dry-run safety, not where faces live."""

    def test_dry_run_never_calls_optimizer_step(self):
        seen = {"steps": 0, "loop": 0}
        real_build = T.build_optimizer_and_scheduler

        def spy(model):
            opt, sch = real_build(model)
            orig = opt.step

            def step(*a, **k):
                seen["steps"] += 1
                return orig(*a, **k)
            opt.step = step
            return opt, sch

        def loop_spy(**kw):
            seen["loop"] += 1
            raise AssertionError("dry_run must not enter the training loop")

        T.build_optimizer_and_scheduler = spy
        real_loop, T._train_loop = T._train_loop, loop_spy
        try:
            out = T.run(dry_run=True)
        finally:
            T.build_optimizer_and_scheduler = real_build
            T._train_loop = real_loop
        self.assertIs(out["dry_run"], True)
        self.assertEqual(seen["steps"], 0)
        self.assertEqual(seen["loop"], 0)

    def test_dry_run_writes_no_checkpoint_and_no_audit_file(self):
        before = {p: p.exists() for p in (T.CHECKPOINT_PATH, T.TRAINING_LOG, T.EPOCH_METRICS,
                                          T.SELECTION_REPORT, T.CLASS_MAPPING,
                                          T.CHECKPOINT_SHA_RECORD)}
        out = T.run(dry_run=True)
        self.assertIs(out["checkpoint_written"], False)
        for p, existed in before.items():
            self.assertEqual(p.exists(), existed, f"{p} appeared during a dry run")
        self.assertFalse(T.CHECKPOINT_DIR.exists())

    def test_dry_run_does_not_construct_data_loaders(self):
        called = {"n": 0}
        real = D.make_loaders

        def spy(*a, **k):
            called["n"] += 1
            raise AssertionError("dry_run must not build loaders")

        D.make_loaders = spy
        try:
            T.run(dry_run=True)
        finally:
            D.make_loaders = real
        self.assertEqual(called["n"], 0)


# ------------------------------------------------------------------ authoritative gate
class TestAuthoritativeGate(H.HermeticExecConfigMixin, unittest.TestCase):
    def test_scaffold_only_error_is_gone(self):
        src = (ROOT / "gpatbench/probe/train.py").read_text()
        self.assertNotIn("training loop is NOT implemented", src)
        self.assertIn("_train_loop", src)
        self.assertTrue(hasattr(T, "_train_loop"))

    def test_authoritative_requires_cuda_and_fails_closed(self):
        if torch.cuda.is_available():
            self.skipTest("this guard is only meaningful without CUDA")
        with self.assertRaises(C.ProbeContractViolation) as cm:
            T.run(dry_run=False)
        self.assertIn("CUDA", str(cm.exception))
        self.assertFalse(T.CHECKPOINT_PATH.exists())

    def test_resolve_device_has_no_cpu_fallback_for_authoritative(self):
        if not torch.cuda.is_available():
            with self.assertRaises(C.ProbeContractViolation):
                T.resolve_device(require_cuda=True)
        self.assertEqual(T.resolve_device(require_cuda=False).type, "cpu")

    def test_build_amp_refuses_non_cuda_devices(self):
        for dev in ("cpu", "meta"):
            with self.assertRaises(C.ProbeContractViolation, msg=dev):
                T.build_amp(torch.device(dev))

    def test_amp_contract_is_cuda_float16_with_enabled_scaler(self):
        body = _executable_source(T.build_amp)
        self.assertIn("device_type='cuda'", body)
        self.assertIn("dtype=torch.float16", body)
        self.assertIn("enabled=True", body)
        self.assertIn("is_enabled()", body)
        code = _executable_source(T.build_amp, strip_strings=True)   # messages excluded
        for bad in ("bfloat16", "device_type='cpu'"):
            self.assertNotIn(bad, code, bad)

    def test_no_cpu_fallback_anywhere_in_the_authoritative_path(self):
        """Executable code only: an error message that NAMES the forbidden fallbacks is fine."""
        for fn in (T.build_amp, T.resolve_device, T.run, T._train_loop, T.train_one_epoch):
            code = _executable_source(fn, strip_strings=True)
            for bad in ("bfloat16", "enabled=False", "allow_cpu_authoritative",
                        "fallback_to_cpu"):
                self.assertNotIn(bad, code, f"{fn.__name__}: {bad}")


# ------------------------------------------------------------------ epoch control flow
class TestEpochControlFlow(unittest.TestCase):
    def test_validation_epochs_are_exactly_1_to_30(self):
        rec, loop = run_loop({e: 0.0 for e in range(1, 31)})
        self.assertEqual(loop["epochs_run"], list(range(1, 31)))
        self.assertEqual(loop["validation_passes"], 30)
        self.assertEqual(tuple(loop["epochs_run"]), C.validation_epochs())

    def test_validation_runs_exactly_once_per_epoch(self):
        rec, loop = run_loop({e: 0.0 for e in range(1, 31)})
        vals = [e for e, _ in loop["events"] if e == T.STEP_VAL]
        self.assertEqual(len(vals), 30)
        self.assertEqual(rec.calls.count("validate"), 30)
        self.assertEqual([ep for st, ep in loop["events"] if st == T.STEP_VAL],
                         list(range(1, 31)))

    def test_per_epoch_order_is_train_val_select_scheduler(self):
        rec, loop = run_loop({e: 0.0 for e in range(1, 31)})
        expected = []
        for ep in range(1, 31):
            expected += [(T.STEP_TRAIN, ep), (T.STEP_VAL, ep), (T.STEP_SELECT, ep),
                         (T.STEP_SCHEDULER, ep)]
        self.assertEqual(loop["events"], expected)
        # independent evidence: the order the fakes actually observed
        observed = [c for c in rec.calls if c in ("train", "validate", "scheduler.step")]
        self.assertEqual(observed, ["train", "validate", "scheduler.step"] * 30)

    def test_scheduler_steps_exactly_once_per_epoch_and_always_last(self):
        rec, loop = run_loop({e: 0.0 for e in range(1, 31)})
        self.assertEqual(loop["scheduler_steps"], 30)
        for ep in range(1, 31):
            block = [st for st, e in loop["events"] if e == ep]
            self.assertEqual(block, [T.STEP_TRAIN, T.STEP_VAL, T.STEP_SELECT, T.STEP_SCHEDULER])
            self.assertEqual(block[-1], T.STEP_SCHEDULER)

    def test_checkpoint_selection_happens_after_validation_and_before_scheduler(self):
        rec, loop = run_loop({e: 0.0 for e in range(1, 31)})
        for ep in range(1, 31):
            idx = {st: i for i, (st, e) in enumerate(loop["events"]) if e == ep}
            self.assertLess(idx[T.STEP_TRAIN], idx[T.STEP_VAL])
            self.assertLess(idx[T.STEP_VAL], idx[T.STEP_SELECT])
            self.assertLess(idx[T.STEP_SELECT], idx[T.STEP_SCHEDULER])

    def test_scheduler_is_not_stepped_per_batch(self):
        src = inspect.getsource(T.train_one_epoch)
        self.assertNotIn("scheduler", src)


# ------------------------------------------------------------------ selection rule
class TestSelection(unittest.TestCase):
    def test_strict_rule_keeps_the_earlier_epoch_on_an_exact_tie(self):
        scores = {e: 0.4 for e in range(1, 31)}
        scores[5] = 0.9
        scores[12] = 0.9          # exact tie with epoch 5
        rec, loop = run_loop(scores)
        self.assertEqual(loop["best_epoch"], 5)
        self.assertEqual(loop["best_macro_f1"], 0.9)

    def test_any_epoch_in_range_may_win(self):
        for peak in (1, 17, 30):
            scores = {e: 0.1 for e in range(1, 31)}
            scores[peak] = 0.99
            rec, loop = run_loop(scores)
            self.assertEqual(loop["best_epoch"], peak)

    def test_training_loss_is_never_a_tie_break(self):
        src = inspect.getsource(T._train_loop)
        sel = src.split("3. best-checkpoint selection")[1].split("4. scheduler")[0]
        self.assertNotIn("train_loss", sel)
        self.assertIn("MET.is_better", sel)

    def test_loop_cross_checks_itself_against_the_pure_selection_rule(self):
        rec, loop = run_loop({e: e / 100.0 for e in range(1, 31)})
        self.assertEqual(loop["best_epoch"], 30)
        self.assertEqual(C.select_best_epoch([(r["epoch"], r["val_macro_f1"])
                                              for r in loop["epoch_rows"]])[0], 30)

    def test_checkpoint_is_written_only_when_the_score_strictly_improves(self):
        writes = []
        scores = {e: 0.4 for e in range(1, 31)}
        scores[3] = 0.9
        scores[9] = 0.9            # tie -> must NOT rewrite
        scores[20] = 0.95          # improvement -> must write
        rec, loop = run_loop(scores, write_checkpoint=True,
                             save_fn=lambda model, **k: writes.append(k["epoch"]))
        self.assertEqual(writes, [1, 3, 20])
        self.assertEqual(loop["checkpoint_written_at"], [1, 3, 20])
        self.assertNotIn(9, writes)


# ------------------------------------------------------------------ population firewall
class TestPopulationFirewall(unittest.TestCase):
    def test_test_split_cannot_be_loaded(self):
        with self.assertRaises(C.ProbeContractViolation):
            D.load_rows("TEST")
        with self.assertRaises(C.ProbeContractViolation):
            D.ProbeDataset("TEST")
        self.assertEqual(D.ALLOWED_SPLITS, ("TRAIN", "VAL"))

    def test_trainer_never_references_test(self):
        src = (ROOT / "gpatbench/probe/train.py").read_text()
        self.assertNotIn('"TEST"', src)
        self.assertNotIn("'TEST'", src)
        self.assertIn("test_split_used", src)

    def test_synthetic_samples_cannot_enter(self):
        src = (ROOT / "gpatbench/probe/data.py").read_text()
        for bad in ("synthetic_root", "banks/", "generated_image", "bank_path"):
            self.assertNotIn(bad, src, bad)
        self.assertEqual(D.ProbeDataset("TRAIN", rows=[]).synthetic_samples, 0)

    def test_preflight_rejects_a_weakened_population_or_augmentation_contract(self):
        import copy
        base = C.load_config()
        for patch, where in (({"any": True}, "augmentation"),
                             ({"test_split_usage": "SOMETIMES"}, "population"),
                             ({"synthetic_samples": "SOME"}, "population")):
            bad = copy.deepcopy(base)
            bad[where].update(patch)
            with self.assertRaises(C.ProbeContractViolation, msg=str(patch)):
                T.preflight.__wrapped__(bad) if hasattr(T.preflight, "__wrapped__") else \
                    self._preflight_with(bad)

    def _preflight_with(self, cfg):
        """Drive the same checks preflight applies, on an injected config."""
        if cfg["augmentation"]["any"]:
            raise C.ProbeContractViolation("augmentation is forbidden")
        if cfg["population"]["test_split_usage"] != "NEVER":
            raise C.ProbeContractViolation("TEST must never be used")
        if cfg["population"]["synthetic_samples"] != "NONE":
            raise C.ProbeContractViolation("synthetic samples forbidden")

    def test_frozen_class_order_is_unchanged(self):
        self.assertEqual(C.CLASSES,
                         ("live", "makeup", "mask_2d", "mask_3d", "partial", "print", "replay"))
        self.assertEqual(C.K, 7)


# ------------------------------------------------------------------ loaders
class FakeProbeDataset:
    def __init__(self, split, rows=None, root=None):
        self.split = split
        self.rows = [{"sample_id": f"s{i:03d}", "label": i % C.K} for i in range(8)]
        self.synthetic_samples = 0

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return np.zeros((3, 4, 4), dtype=np.float32), int(self.rows[i]["label"])


class TestLoaders(unittest.TestCase):
    def _loaders(self, **kw):
        real = D.ProbeDataset
        D.ProbeDataset = FakeProbeDataset
        try:
            return D.make_loaders(**kw)
        finally:
            D.ProbeDataset = real

    def test_loader_configuration_matches_the_frozen_contract(self):
        cfg = C.load_config()["dataloader"]
        gpu = cfg["gpu_defaults"]
        tr, va, tr_ds, va_ds = self._loaders(
            num_workers=int(gpu["num_workers"]), pin_memory=bool(gpu["pin_memory"]),
            persistent_workers=bool(gpu["persistent_workers"]),
            prefetch_factor=int(gpu["prefetch_factor"]),
            batch_size=int(cfg["train"]["batch_size"]))
        for dl in (tr, va):
            self.assertEqual(dl.batch_size, 64)
            self.assertIs(dl.drop_last, False)
            self.assertEqual(dl.num_workers, 4)
            self.assertIs(dl.pin_memory, True)
            self.assertIs(dl.persistent_workers, True)
            self.assertEqual(dl.prefetch_factor, 2)
            self.assertIs(dl.worker_init_fn, D.worker_init_fn)
        self.assertEqual((tr_ds.split, va_ds.split), ("TRAIN", "VAL"))

    def test_train_shuffles_with_a_seeded_generator_and_val_does_not(self):
        tr, va, _, _ = self._loaders(num_workers=0, persistent_workers=False, prefetch_factor=None)
        self.assertIsInstance(tr.sampler, torch.utils.data.RandomSampler)
        self.assertIsInstance(va.sampler, torch.utils.data.SequentialSampler)
        self.assertIsNotNone(tr.generator)
        g = torch.Generator()
        g.manual_seed(C.SEED)
        self.assertEqual(tr.generator.initial_seed(), g.initial_seed())

    def test_val_preserves_canonical_sample_id_order(self):
        src = inspect.getsource(D.load_rows)
        self.assertIn('out.sort(key=lambda r: r["sample_id"])', src)
        rows = D.load_rows("VAL")
        self.assertEqual([r["sample_id"] for r in rows], sorted(r["sample_id"] for r in rows))

    def test_worker_init_fn_is_deterministic(self):
        import random as _r
        D.worker_init_fn(2)
        a = (_r.random(), float(np.random.rand()), float(torch.rand(1)))
        D.worker_init_fn(2)
        b = (_r.random(), float(np.random.rand()), float(torch.rand(1)))
        self.assertEqual(a, b)
        D.worker_init_fn(3)
        c = (_r.random(), float(np.random.rand()), float(torch.rand(1)))
        self.assertNotEqual(a, c, "different workers must not share a stream")


# ------------------------------------------------------------------ determinism
class TestDeterminism(unittest.TestCase):
    def test_determinism_settings_are_applied_and_reported(self):
        det = T.apply_determinism()
        self.assertEqual(det["seed"], 42)
        for k in ("cudnn_benchmark", "tf32_matmul", "tf32_cudnn"):
            self.assertIs(det[k], False, k)
        for k in ("cudnn_deterministic", "use_deterministic_algorithms"):
            self.assertIs(det[k], True, k)
        self.assertEqual(det["float32_matmul_precision"], "highest")
        self.assertEqual(det["cublas_workspace_config"], ":4096:8")
        self.assertIs(torch.backends.cudnn.benchmark, False)
        self.assertIs(torch.backends.cuda.matmul.allow_tf32, False)

    def test_seeds_cover_every_required_source(self):
        src = inspect.getsource(T.seed_everything)
        for needed in ("random.seed", "np.random.seed", "torch.manual_seed",
                       "torch.cuda.manual_seed_all"):
            self.assertIn(needed, src, needed)

    def test_determinism_is_not_silently_relaxed(self):
        src = (ROOT / "gpatbench/probe/train.py").read_text()
        for bad in ("use_deterministic_algorithms(False)", "benchmark = True",
                    "allow_tf32 = True", "deterministic = False"):
            self.assertNotIn(bad, src, bad)


# ------------------------------------------------------------------ checkpoint + audit schemas
class TestCheckpointAndAudit(unittest.TestCase):
    def _loop_result(self, scores=None):
        scores = scores or {e: (0.9 if e == 7 else 0.1) for e in range(1, 31)}
        rec, loop = run_loop(scores)
        return loop

    def test_checkpoint_payload_contains_the_required_provenance(self):
        payload = T.checkpoint_payload(
            TinyNet(), epoch=7, macro_f1=0.9,
            val={"confusion_matrix": np.zeros((C.K, C.K), dtype=np.int64),
                 "per_class_f1": [0.0] * C.K, "support": [0] * C.K, "predicted": [0] * C.K,
                 "val_samples": 3121},
            provenance=fake_provenance())
        for key in ("schema_version", "model_state_dict", "selected_epoch", "best_val_macro_f1",
                    "classes", "class_index", "seed", "frozen_config_sha256",
                    "frozen_config_snapshot_sha256", "split_manifest_sha256",
                    "resnet18_weight_sha256", "class_weights", "optimizer", "scheduler",
                    "epochs", "batch_size", "validation_epochs", "selection_rule", "amp",
                    "determinism", "environment", "code_commit", "host", "created_utc"):
            self.assertIn(key, payload, key)
        self.assertEqual(payload["classes"], list(C.CLASSES))
        self.assertEqual(payload["selected_epoch"], 7)
        self.assertEqual(payload["seed"], 42)
        self.assertEqual(payload["validation_epochs"], list(range(1, 31)))
        self.assertEqual(payload["optimizer"]["name"], "AdamW")
        self.assertEqual(payload["scheduler"]["T_max"], 30)
        self.assertEqual(payload["amp"], {"autocast_device_type": "cuda",
                                          "autocast_dtype": "float16",
                                          "grad_scaler_enabled": True,
                                          "val_autocast": T.VAL_AUTOCAST})
        self.assertIs(payload["test_split_used"], False)
        self.assertIs(payload["synthetic_samples_used"], False)
        self.assertIs(payload["augmentation"], False)

    def test_checkpoint_payload_embeds_no_raw_data(self):
        payload = T.checkpoint_payload(
            TinyNet(), epoch=1, macro_f1=0.5,
            val={"confusion_matrix": np.zeros((C.K, C.K), dtype=np.int64),
                 "per_class_f1": [0.0] * C.K, "support": [0] * C.K, "predicted": [0] * C.K,
                 "val_samples": 1},
            provenance=fake_provenance())
        for bad in ("images", "pixels", "faces", "sample_ids", "dataset_rows"):
            self.assertNotIn(bad, payload, bad)

    def test_checkpoint_is_written_atomically_and_only_for_a_real_selection(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "artifact_probe_v1.pt"
            T.save_checkpoint(TinyNet(), epoch=3, macro_f1=0.8,
                              val={"confusion_matrix": np.zeros((C.K, C.K), dtype=np.int64),
                                   "per_class_f1": [0.0] * C.K, "support": [0] * C.K,
                                   "predicted": [0] * C.K, "val_samples": 1},
                              provenance=fake_provenance(), path=path)
            self.assertTrue(path.is_file())
            self.assertFalse(path.with_name(path.name + ".tmp").exists())
            got = torch.load(path, weights_only=False)
            self.assertEqual(got["selected_epoch"], 3)
            self.assertEqual(got["classes"], list(C.CLASSES))

    def test_audit_schemas_are_deterministic_and_one_row_per_epoch(self):
        loop = self._loop_result()
        with tempfile.TemporaryDirectory() as td:
            t = Path(td)
            paths = {"training_log": t / "log.jsonl", "epoch_metrics": t / "metrics.csv",
                     "selection_report": t / "sel.md", "class_mapping": t / "classes.json",
                     "environment_lock": t / "env.txt", "checkpoint_sha": t / "ckpt.sha256"}
            res = T.write_audit_outputs(loop, fake_provenance(),
                                        checkpoint_path=t / "absent.pt", paths=paths)
            rows = list(csv.DictReader(paths["epoch_metrics"].open()))
            self.assertEqual(len(rows), 30)
            self.assertEqual(list(rows[0].keys()), list(T.EPOCH_METRICS_COLUMNS))
            self.assertEqual([int(r["epoch"]) for r in rows], list(range(1, 31)))
            log = [json.loads(x) for x in paths["training_log"].read_text().splitlines()]
            self.assertEqual(len(log), 30)
            self.assertEqual(sorted(log[0]), sorted(log[29]), "JSONL schema must be stable")
            cm = json.loads(paths["class_mapping"].read_text())
            self.assertEqual(cm["classes"], list(C.CLASSES))
            self.assertEqual(cm["excluded"], ["other_spoof"])
            self.assertIn("**Selected epoch:** 7", paths["selection_report"].read_text())
            self.assertIsNone(res["checkpoint_sha256"])
            self.assertFalse(paths["checkpoint_sha"].exists(),
                             "the SHA record must not exist without a checkpoint")

    def test_checkpoint_sha_audit_only_after_the_selected_checkpoint_exists(self):
        loop = self._loop_result()
        with tempfile.TemporaryDirectory() as td:
            t = Path(td)
            ckpt = t / "artifact_probe_v1.pt"
            paths = {"training_log": t / "log.jsonl", "epoch_metrics": t / "metrics.csv",
                     "selection_report": t / "sel.md", "class_mapping": t / "classes.json",
                     "environment_lock": t / "env.txt", "checkpoint_sha": t / "ckpt.sha256"}
            # 1. no checkpoint -> no SHA record
            self.assertIsNone(T.write_audit_outputs(loop, fake_provenance(),
                                                    checkpoint_path=ckpt,
                                                    paths=paths)["checkpoint_sha256"])
            self.assertFalse(paths["checkpoint_sha"].exists())
            # 2. checkpoint exists -> SHA recorded, and it is the SHA of that file
            T.save_checkpoint(TinyNet(), epoch=7, macro_f1=0.9,
                              val={"confusion_matrix": np.zeros((C.K, C.K), dtype=np.int64),
                                   "per_class_f1": [0.0] * C.K, "support": [0] * C.K,
                                   "predicted": [0] * C.K, "val_samples": 1},
                              provenance=fake_provenance(), path=ckpt)
            res = T.write_audit_outputs(loop, fake_provenance(), checkpoint_path=ckpt,
                                        paths=paths)
            self.assertEqual(res["checkpoint_sha256"], C.sha256_file(ckpt))
            rec = json.loads(paths["checkpoint_sha"].read_text())
            self.assertEqual(rec["sha256"], C.sha256_file(ckpt))
            self.assertEqual(rec["selected_epoch"], 7)
            self.assertEqual(rec["validation_passes"], 30)
            self.assertIs(rec["committed_to_git"], False)

    def test_audit_paths_match_the_frozen_config(self):
        out = C.load_config()["outputs"]
        rel = lambda p: p.relative_to(C.ROOT).as_posix()
        self.assertEqual(rel(T.CHECKPOINT_PATH), out["checkpoint"])
        self.assertEqual(rel(T.TRAINING_LOG), out["training_log"])
        self.assertEqual(rel(T.EPOCH_METRICS), out["epoch_metrics"])
        self.assertEqual(rel(T.SELECTION_REPORT), out["selection_report"])
        self.assertEqual(rel(T.CLASS_MAPPING), out["class_mapping"])
        self.assertEqual(rel(T.ENVIRONMENT_LOCK), out["environment_lock"])
        self.assertEqual(rel(T.CHECKPOINT_SHA_RECORD), out["checkpoint_sha_record"])

    def test_checkpoint_stays_gitignored(self):
        self.assertIn("*.pt", (ROOT / ".gitignore").read_text())


# ------------------------------------------------------------------ execution-storage portability
def _write_exec_config(tmp: Path, *, runtime_root: Path, faces: Path, extra=None) -> Path:
    body = {"version": "m5_execution_v1", "kind": "EXECUTION_INFRASTRUCTURE",
            "storage": {"runtime_root": str(runtime_root),
                        "roots": {"faces_256_root": str(faces)},
                        "runstate_root": str(runtime_root / "runstate"),
                        "tmp_root": str(runtime_root / "tmp")}}
    if extra:
        body["storage"].update(extra)
    p = tmp / "exec.yaml"
    p.write_text(yaml.safe_dump(body, sort_keys=True))
    return p


class TestExecutionStoragePortability(unittest.TestCase):
    """GPAT_M5_EXEC_CONFIG relocates PHYSICAL storage only; containment is still enforced."""

    def setUp(self):
        import os
        self._saved = os.environ.get(D.EXEC_CONFIG_ENV)
        os.environ.pop(D.EXEC_CONFIG_ENV, None)

    def tearDown(self):
        import os
        os.environ.pop(D.EXEC_CONFIG_ENV, None)
        if self._saved is not None:
            os.environ[D.EXEC_CONFIG_ENV] = self._saved

    def _set(self, value):
        import os
        os.environ[D.EXEC_CONFIG_ENV] = str(value)

    def test_default_selects_the_historical_laptop_config(self):
        """Selector semantics only: the historical faces DIRECTORY need not exist here."""
        self.assertEqual(D.resolve_exec_config(), D.DEFAULT_EXEC_CONFIG)
        self.assertEqual(D.DEFAULT_EXEC_CONFIG,
                         ROOT / "configs/execution/m2b_laptop_external_storage.yaml")
        self.assertTrue(D.DEFAULT_EXEC_CONFIG.is_file())
        s = D.resolve_storage(require_faces=False)          # structure, not physical readiness
        self.assertEqual(s["exec_config_path"], D.DEFAULT_EXEC_CONFIG)
        self.assertEqual(D.execution_config_provenance()["m5_execution_config_path"],
                         "configs/execution/m2b_laptop_external_storage.yaml")

    def test_empty_env_value_is_treated_as_unset(self):
        self._set("")
        self.assertEqual(D.resolve_exec_config(), D.DEFAULT_EXEC_CONFIG)

    def test_env_selects_an_alternate_execution_yaml(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            faces = tmp / "rt" / "data" / "processed" / "faces_256"
            faces.mkdir(parents=True)
            cfg = _write_exec_config(tmp, runtime_root=tmp / "rt", faces=faces)
            self._set(cfg)
            self.assertEqual(D.resolve_exec_config(), cfg)
            s = D.resolve_storage()
            self.assertEqual(s["faces_256_root"], faces.resolve())
            self.assertEqual(D.faces_root(), faces.resolve())
            prov = D.execution_config_provenance()
            self.assertEqual(prov["faces_256_root_resolved"], faces.resolve().as_posix())
            self.assertIs(prov["m5_execution_config_selected_via_env"], True)

    def test_relative_env_value_resolves_against_the_repository_root(self):
        self._set("configs/execution/m5_gpu_3090.yaml")
        self.assertEqual(D.resolve_exec_config(),
                         ROOT / "configs/execution/m5_gpu_3090.yaml")

    def test_explicitly_set_but_nonexistent_config_fails_closed(self):
        self._set("/definitely/not/here/exec.yaml")
        with self.assertRaises(C.ProbeContractViolation) as cm:
            D.resolve_exec_config()
        self.assertIn("no fallback", str(cm.exception))
        with self.assertRaises(C.ProbeContractViolation):
            D.faces_root()

    def test_invalid_yaml_or_missing_storage_block_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            bad = Path(td) / "bad.yaml"
            bad.write_text("storage: [this is not a mapping\n")      # unparseable
            self._set(bad)
            with self.assertRaises(C.ProbeContractViolation):
                D.load_exec_config()
            bad.write_text("version: x\n")                            # parses, no storage block
            with self.assertRaises(C.ProbeContractViolation) as cm:
                D.load_exec_config()
            self.assertIn("storage", str(cm.exception))

    def test_missing_required_storage_key_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            p = tmp / "exec.yaml"
            p.write_text(yaml.safe_dump({"storage": {"runtime_root": str(tmp),
                                                     "roots": {"faces_256_root": str(tmp)}}}))
            self._set(p)
            with self.assertRaises(C.ProbeContractViolation):
                D.resolve_storage(require_faces=False)

    def test_a_faces_root_outside_runtime_root_is_rejected_by_containment(self):
        from gpatbench.preprocess.m2b import RootContainmentError
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            outside = tmp / "elsewhere" / "faces_256"
            outside.mkdir(parents=True)
            (tmp / "rt").mkdir()
            cfg = _write_exec_config(tmp, runtime_root=tmp / "rt", faces=outside)
            self._set(cfg)
            with self.assertRaises(RootContainmentError):
                D.resolve_storage()

    def test_a_dotdot_escape_is_rejected_by_containment(self):
        from gpatbench.preprocess.m2b import RootContainmentError
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            (tmp / "rt").mkdir()
            (tmp / "outside").mkdir()
            cfg = _write_exec_config(tmp, runtime_root=tmp / "rt",
                                     faces=tmp / "rt" / ".." / "outside")
            self._set(cfg)
            with self.assertRaises(RootContainmentError):
                D.resolve_storage()

    def test_a_symlink_escape_cannot_bypass_containment(self):
        from gpatbench.preprocess.m2b import RootContainmentError
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            rt = tmp / "rt"
            rt.mkdir()
            real = tmp / "outside_faces"
            real.mkdir()
            link = rt / "faces_256"
            link.symlink_to(real, target_is_directory=True)
            cfg = _write_exec_config(tmp, runtime_root=rt, faces=link)
            self._set(cfg)
            with self.assertRaises(RootContainmentError):
                D.resolve_storage()

    def test_selected_faces_root_must_exist_for_loader_use(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            (tmp / "rt").mkdir()
            missing = tmp / "rt" / "data" / "processed" / "faces_256"      # never created
            cfg = _write_exec_config(tmp, runtime_root=tmp / "rt", faces=missing)
            self._set(cfg)
            with self.assertRaises(C.ProbeContractViolation) as cm:
                D.resolve_storage()
            self.assertIn("does not exist", str(cm.exception))
            # structure is still checkable without the directory
            self.assertEqual(D.resolve_storage(require_faces=False)["faces_256_root"],
                             missing.resolve())

    def test_provenance_does_not_require_the_faces_directory_but_faces_root_does(self):
        """Provenance reports selection; readiness is a separate, still-strict check."""
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = H.write_exec_config(tmp, create_faces=False)     # config valid, directory absent
            self._set(cfg)
            prov = D.execution_config_provenance()                 # must NOT raise
            self.assertTrue(prov["faces_256_root_resolved"].endswith("faces_256"))
            self.assertEqual(prov["m5_execution_config_sha256"], C.sha256_file(cfg))
            with self.assertRaises(C.ProbeContractViolation):      # readiness still fails closed
                D.faces_root()
            with self.assertRaises(C.ProbeContractViolation):
                D.resolve_storage(require_faces=True)

    def test_env_is_restored_after_every_portability_test(self):
        import os
        sentinel = "/tmp/__m5_sentinel__.yaml"
        os.environ[D.EXEC_CONFIG_ENV] = sentinel
        try:
            self._saved = sentinel                    # emulate a caller-provided value
            self.tearDown()
            self.assertEqual(os.environ.get(D.EXEC_CONFIG_ENV), sentinel)
        finally:
            os.environ.pop(D.EXEC_CONFIG_ENV, None)
            self._saved = None
            self.setUp()

    def test_there_is_no_direct_faces_root_override(self):
        src = (ROOT / "gpatbench/probe/data.py").read_text()
        self.assertNotIn("GPAT_M5_FACES_ROOT", src)
        self.assertIn("m2b.resolve_roots", src)          # relocation always goes through it
        self.assertEqual(D.EXEC_CONFIG_ENV, "GPAT_M5_EXEC_CONFIG")

    def test_no_test_path_appears_in_any_execution_config(self):
        for p in sorted((ROOT / "configs/execution").glob("*.yaml")):
            cfg = yaml.safe_load(p.read_text())
            blob = json.dumps(cfg)
            self.assertNotIn("/test/", blob.lower(), p.name)
            self.assertNotIn("test_root", blob, p.name)

    def test_gpu_execution_config_is_structurally_valid_without_the_gpu_filesystem(self):
        from gpatbench.preprocess import m2b
        p = ROOT / "configs/execution/m5_gpu_3090.yaml"
        cfg = yaml.safe_load(p.read_text())
        self.assertEqual(cfg["kind"], "EXECUTION_INFRASTRUCTURE")
        self.assertIs(cfg["scientific_contract_modified_by_this_file"], False)
        self.assertEqual(cfg["scientific_contract_sha256"], C.CONFIG_SHA256)
        self.assertIs(cfg["storage"]["symlinks_used"], False)
        r = m2b.resolve_roots(cfg)                       # containment passes
        self.assertEqual(
            str(r["roots"]["faces_256_root"]),
            "/home/student20261/workdir/GPAT_TransferBench_runtime/data/processed/faces_256")
        self.assertEqual(str(r["runtime_root"]),
                         "/home/student20261/workdir/GPAT_TransferBench_runtime")
        self.assertEqual(list(r["roots"]), ["faces_256_root"],
                         "M5 must not claim M2 cache roots it does not read")

    def test_provenance_carries_the_three_required_execution_fields(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = H.write_exec_config(Path(td))
            self._set(cfg)
            self._assert_provenance_roundtrip()

    def _assert_provenance_roundtrip(self):
        prov = D.execution_config_provenance()
        for k in ("m5_execution_config_path", "m5_execution_config_sha256",
                  "faces_256_root_resolved"):
            self.assertIn(k, prov)
            self.assertTrue(prov[k])
        out = T.run(dry_run=True)
        for k in ("m5_execution_config_path", "m5_execution_config_sha256",
                  "faces_256_root_resolved"):
            self.assertEqual(out[k], prov[k], k)

    def test_checkpoint_and_sha_record_carry_the_execution_fields(self):
        with tempfile.TemporaryDirectory() as td:
            self._set(H.write_exec_config(Path(td)))
            self._assert_checkpoint_and_sha_carry_fields()

    def _assert_checkpoint_and_sha_carry_fields(self):
        prov = dict(fake_provenance())
        prov.update(D.execution_config_provenance())
        payload = T.checkpoint_payload(
            TinyNet(), epoch=1, macro_f1=0.5,
            val={"confusion_matrix": np.zeros((C.K, C.K), dtype=np.int64),
                 "per_class_f1": [0.0] * C.K, "support": [0] * C.K, "predicted": [0] * C.K,
                 "val_samples": 1},
            provenance=prov)
        for k in ("m5_execution_config_path", "m5_execution_config_sha256",
                  "faces_256_root_resolved"):
            self.assertEqual(payload[k], prov[k], k)
        rec, loop = run_loop({e: 0.1 for e in range(1, 31)})
        with tempfile.TemporaryDirectory() as td:
            t = Path(td)
            ckpt = t / "c.pt"
            ckpt.write_bytes(b"x")
            paths = {"training_log": t / "l.jsonl", "epoch_metrics": t / "m.csv",
                     "selection_report": t / "s.md", "class_mapping": t / "c.json",
                     "environment_lock": t / "e.txt", "checkpoint_sha": t / "k.sha256"}
            T.write_audit_outputs(loop, prov, checkpoint_path=ckpt, paths=paths)
            sha_rec = json.loads(paths["checkpoint_sha"].read_text())
            for k in ("m5_execution_config_path", "m5_execution_config_sha256",
                      "faces_256_root_resolved"):
                self.assertEqual(sha_rec[k], prov[k], k)
            lock = paths["environment_lock"].read_text()
            self.assertIn("m5_execution_config_sha256=", lock)
            self.assertIn("faces_256_root_resolved=", lock)

    def test_frozen_artifact_probe_config_is_untouched_by_relocation(self):
        self.assertEqual(C.sha256_file(C.FROZEN_CONFIG), C.CONFIG_SHA256)
        self.assertEqual(C.FROZEN_SNAPSHOT.read_bytes(), C.FROZEN_CONFIG.read_bytes())
        gpu = yaml.safe_load((ROOT / "configs/execution/m5_gpu_3090.yaml").read_text())
        for forbidden in ("classes", "class_weights", "input", "backbone", "optimization",
                          "validation", "population", "augmentation"):
            self.assertNotIn(forbidden, gpu, f"execution config must not carry {forbidden}")

    def test_test_split_remains_impossible_under_any_execution_config(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            faces = tmp / "rt" / "faces"
            faces.mkdir(parents=True)
            self._set(_write_exec_config(tmp, runtime_root=tmp / "rt", faces=faces))
            with self.assertRaises(C.ProbeContractViolation):
                D.load_rows("TEST")
            with self.assertRaises(C.ProbeContractViolation):
                D.ProbeDataset("TEST")


# ------------------------------------------------------------------ CLI audit visibility
EXEC_PROVENANCE_FIELDS = ("m5_execution_config_path", "m5_execution_config_sha256",
                          "faces_256_root_resolved", "m5_execution_config_selected_via_env",
                          "m5_execution_config_env_var")


class TestCliExecutionProvenance(H.HermeticExecConfigMixin, unittest.TestCase):
    """The CLI must SHOW what the run resolved; a dry-run that hides its face store is unauditable."""

    def _cli_dry_run(self, cfg=None):
        r = subprocess.run([sys.executable, "-m", "gpatbench.cli", "train-probe", "--config",
                            "configs/frozen/artifact_probe.yaml", "--dry-run"],
                           cwd=ROOT, capture_output=True, text=True,
                           env=H.subprocess_env(cfg or self.exec_config))
        self.assertEqual(r.returncode, 0, r.stderr[-800:])
        return json.loads(r.stdout)

    def test_dry_run_json_contains_all_five_execution_provenance_fields(self):
        rec = self._cli_dry_run()
        for k in EXEC_PROVENANCE_FIELDS:
            self.assertIn(k, rec, k)
            self.assertIsNotNone(rec[k], k)
        self.assertEqual(rec["m5_execution_config_env_var"], "GPAT_M5_EXEC_CONFIG")
        self.assertIs(rec["m5_execution_config_selected_via_env"], True)

    def test_printed_values_are_the_ones_the_run_resolved(self):
        """Not recomputed by the CLI: the JSON must equal T.run()'s own fields."""
        rec = self._cli_dry_run()
        out = T.run(dry_run=True)                       # same env, installed by the mixin
        for k in EXEC_PROVENANCE_FIELDS:
            self.assertEqual(rec[k], out[k], k)
        self.assertEqual(rec["m5_execution_config_path"], str(self.exec_config))
        self.assertEqual(rec["m5_execution_config_sha256"], C.sha256_file(self.exec_config))

    def test_cli_does_not_recompute_the_provenance(self):
        """cmd_train_probe must copy the fields out of `out`, never derive them itself."""
        import ast
        fn = next(n for n in ast.walk(ast.parse((ROOT / "gpatbench/cli.py").read_text()))
                  if isinstance(n, ast.FunctionDef) and n.name == "cmd_train_probe")
        body = ast.unparse(fn)
        for k in EXEC_PROVENANCE_FIELDS:
            self.assertIn(f"out['{k}']", body, k)        # direct indexing, not .get()
            self.assertNotIn(f"out.get('{k}'", body, k)  # a missing field must fail loudly
        for bad in ("execution_config_provenance", "resolve_storage", "resolve_exec_config"):
            self.assertNotIn(bad, body, f"CLI must not recompute via {bad}")

    def test_a_different_execution_config_changes_the_printed_values(self):
        """Proves the fields track the selected config rather than being constants."""
        first = self._cli_dry_run()
        with tempfile.TemporaryDirectory() as td:
            other = H.write_exec_config(Path(td))
            second = self._cli_dry_run(other)
            other_sha = C.sha256_file(other)          # hash while the temp file still exists
        self.assertNotEqual(first["m5_execution_config_path"],
                            second["m5_execution_config_path"])
        self.assertNotEqual(first["faces_256_root_resolved"],
                            second["faces_256_root_resolved"])
        self.assertEqual(second["m5_execution_config_sha256"], other_sha)

    def test_cli_dry_run_remains_checkpoint_free(self):
        rec = self._cli_dry_run()
        self.assertIs(rec["dry_run"], True)
        self.assertIs(rec["checkpoint_written"], False)
        self.assertFalse(T.CHECKPOINT_DIR.exists())
        self.assertFalse(T.CHECKPOINT_PATH.exists())
        for p in (T.TRAINING_LOG, T.EPOCH_METRICS, T.SELECTION_REPORT, T.CLASS_MAPPING,
                  T.CHECKPOINT_SHA_RECORD):
            self.assertFalse(p.exists(), p)

    def test_no_wording_claims_cpu_only_execution(self):
        """The verified dry-run may run on an RTX 3090; nothing may imply CPU-only."""
        rec = self._cli_dry_run()
        self.assertEqual(rec["note"], "Dry-run preflight only: no optimizer.step(), "
                                      "no checkpoint, no scientific claim.")
        self.assertNotIn("CPU", rec["note"])
        for f in ("gpatbench/cli.py", "gpatbench/probe/train.py"):
            src = (ROOT / f).read_text()
            for bad in ("CPU-safe", "CPU safe", "CPU-only"):
                self.assertNotIn(bad, src, f"{f}: {bad}")

    def test_help_text_is_device_neutral(self):
        r = subprocess.run([sys.executable, "-m", "gpatbench.cli", "train-probe", "--help"],
                           cwd=ROOT, capture_output=True, text=True,
                           env=H.subprocess_env(self.exec_config))
        self.assertEqual(r.returncode, 0)
        self.assertIn("contract preflight", r.stdout)
        self.assertNotIn("CPU", r.stdout)

    def test_execution_storage_contract_is_unchanged_by_this_pass(self):
        gpu = ROOT / "configs/execution/m5_gpu_3090.yaml"
        self.assertEqual(C.sha256_file(gpu),
                         "c9d22abf1b5e284593bfed21500e8113df7f14ff564b7434e3eda43cb7e7eb70")
        self.assertEqual(C.sha256_file(C.FROZEN_CONFIG), C.CONFIG_SHA256)
        self.assertEqual(C.FROZEN_SNAPSHOT.read_bytes(), C.FROZEN_CONFIG.read_bytes())
        self.assertEqual(D.EXEC_CONFIG_ENV, "GPAT_M5_EXEC_CONFIG")
        # fail-closed behaviour still intact
        with tempfile.TemporaryDirectory() as td:
            import os
            saved = os.environ.get(D.EXEC_CONFIG_ENV)
            os.environ[D.EXEC_CONFIG_ENV] = str(H.write_exec_config(Path(td),
                                                                    create_faces=False))
            try:
                with self.assertRaises(C.ProbeContractViolation):
                    D.faces_root()
                with self.assertRaises(C.ProbeContractViolation):
                    T.run(dry_run=True)
            finally:
                if saved is None:
                    os.environ.pop(D.EXEC_CONFIG_ENV, None)
                else:
                    os.environ[D.EXEC_CONFIG_ENV] = saved


# ------------------------------------------------------------------ single code path
class TestSingleCodePath(unittest.TestCase):
    def test_there_is_exactly_one_trainer_entry_point(self):
        import ast
        tree = ast.parse((ROOT / "gpatbench/probe/train.py").read_text())
        public = [n.name for n in tree.body
                  if isinstance(n, ast.FunctionDef) and not n.name.startswith("_")]
        self.assertIn("run", public)
        for name in public:
            self.assertNotIn(name, ("train", "fit", "main", "train_probe"))
        other = [p.name for p in (ROOT / "gpatbench/probe").glob("*.py")
                 if p.name not in ("__init__.py", "train.py")]
        for f in other:
            src = (ROOT / "gpatbench/probe" / f).read_text()
            self.assertNotIn("optimizer.step", src, f)
            self.assertNotIn("GradScaler", src, f)

    def test_cli_delegates_to_the_single_path(self):
        import ast
        fn = next(n for n in ast.walk(ast.parse((ROOT / "gpatbench/cli.py").read_text()))
                  if isinstance(n, ast.FunctionDef) and n.name == "cmd_train_probe")
        called = {n.func.attr for n in ast.walk(fn)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        self.assertIn("run", called)
        for bad in ("_train_loop", "train_one_epoch", "validate", "save_checkpoint",
                    "write_audit_outputs"):
            self.assertNotIn(bad, called, bad)


if __name__ == "__main__":
    unittest.main()
