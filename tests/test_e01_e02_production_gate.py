"""M8 production binding and the hardened E01/E02 production gate (owner approval 2026-11-22).

Covers: the byte-unchanged ruling and the additive binding / O3 completion records; the governance gate
(load_production_authorization) positive at the repository and negative for every binding field and every smoke-record
identity it pins (via a temporary copy of the governance files); the launch gate (verify_production_launch) for the
production authority commit, host, environment, CUDA, runtime root, filesystem identity, free-space floor and run-dir
collision; O_NOATIME face reads in every mode; and the CLI refusing on a non-production host without touching the
runtime root. No benchmark image is read.
"""
import contextlib
import copy
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from methods.common import bank_writer as bw  # noqa: E402
from methods.common import runlog  # noqa: E402
from methods.common.bank_fixture import build_fixture  # noqa: E402

M8 = bw.M8_SMOKE_GATE_STATUS
AUTH = bw.PRODUCTION_AUTHORITY_STATUS
PRE = bw.PRODUCTION_PREFLIGHT_STATUS
O3_HISTORICAL = "configs/amendments/e07c_m6d6jr_science_launch_decision.yaml"
O3_COMPLETION = "configs/amendments/m8_o3_smoke_completion.yaml"
GOVERNANCE_FILES = (bw.RULING_PATH, bw.SMOKE_BINDING_PATH, bw.APPROVED_SMOKE_RECORD_PATH, bw.SMOKE_CONTRACT_PATH,
                    bw.EVALUATOR_CONTRACT_PATH, *(p for p, _ in bw.APPROVED_METHOD_CONFIGS.values()))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def binding():
    return json.loads((ROOT / bw.SMOKE_BINDING_PATH).read_text())


class Base(unittest.TestCase):
    def assertBlocked(self, status, fn, *a, **k):
        with self.assertRaises(bw.BankBlocked) as cm:
            fn(*a, **k)
        self.assertEqual(cm.exception.status, status, str(cm.exception))
        return str(cm.exception)


class GovernanceRecords(Base):
    def test_01_ruling_byte_unchanged_and_never_inline_bound(self):
        self.assertEqual(sha(ROOT / bw.RULING_PATH), bw.APPROVED_RULING_SHA256)
        self.assertIsNone(json.loads((ROOT / bw.RULING_PATH).read_text())["smoke_test_record"])

    def test_02_binding_pins_equal_code_and_files(self):
        b = binding()
        self.assertEqual((b["status"], b["approved"], b["schema_version"]), ("OWNER_APPROVED", True, bw.SMOKE_BINDING_SCHEMA))
        self.assertEqual(b["ruling"], {"path": bw.RULING_PATH, "sha256": bw.APPROVED_RULING_SHA256})
        self.assertEqual((b["smoke_test_record"]["path"], b["smoke_test_record"]["sha256"]),
                         (bw.APPROVED_SMOKE_RECORD_PATH, bw.APPROVED_SMOKE_RECORD_SHA256))
        self.assertEqual(sha(ROOT / bw.APPROVED_SMOKE_RECORD_PATH), bw.APPROVED_SMOKE_RECORD_SHA256)
        self.assertEqual(sha(ROOT / bw.SMOKE_CONTRACT_PATH), bw.APPROVED_SMOKE_CONTRACT_SHA256)
        self.assertEqual(sha(ROOT / bw.EVALUATOR_CONTRACT_PATH), bw.APPROVED_EVALUATOR_CONTRACT_SHA256)
        self.assertEqual(b["smoke_source_commit"], bw.SMOKE_SOURCE_COMMIT)
        self.assertEqual(b["scope"]["methods"], ["E01", "E02"])
        self.assertEqual(b["scope"]["seeds"], [42, 1337, 2026])
        self.assertEqual({k for k, v in b["authorizes"].items() if v}, {"e01_e02_production_bank_generation"})
        for m, (p, d) in bw.APPROVED_METHOD_CONFIGS.items():
            self.assertEqual(sha(ROOT / p), d, m)

    def test_03_o3_completion_is_additive(self):
        c = json.loads((ROOT / O3_COMPLETION).read_text())
        self.assertIs(c["smoke_test_completed"], True)
        self.assertEqual(c["smoke_test_record"]["sha256"], bw.APPROVED_SMOKE_RECORD_SHA256)
        self.assertEqual(c["smoke_test_record"]["smoke_source_commit"], bw.SMOKE_SOURCE_COMMIT)
        self.assertEqual(c["historical_o3_record"]["sha256"], sha(ROOT / O3_HISTORICAL))
        hist = json.loads((ROOT / O3_HISTORICAL).read_text())["O3_smoke_test"]
        self.assertIs(hist["smoke_test_completed"], False)       # historical record not rewritten


class GovernanceGate(Base):
    def test_04_bound_gate_admits_e01_e02_at_each_seed(self):
        for m in bw.METHODS:
            for s in bw.PRODUCTION_SEEDS:
                a = bw.load_production_authorization(m, seed=s)
                self.assertEqual((a.mode, a.label, a.ruling_sha256), ("PRODUCTION", "production", bw.APPROVED_RULING_SHA256))
                r = a.record()
                self.assertEqual(r["binding"]["binding_sha256"], sha(ROOT / bw.SMOKE_BINDING_PATH))
                self.assertEqual(r["binding"]["smoke_record_sha256"], bw.APPROVED_SMOKE_RECORD_SHA256)
                self.assertEqual(r["binding"]["method_config_sha256"], bw.APPROVED_METHOD_CONFIGS[m][1])
            self.assertEqual(a.e01_round_robin_ordinal, "MANIFEST_POSITION_ZERO_BASED" if m == "E01" else None)

    def test_05_record_shapes_of_other_modes_unchanged(self):
        self.assertNotIn("binding", bw.qualification_authorization("u").record())
        self.assertNotIn("binding", bw.smoke_authorization().record())

    def test_06_binding_field_negatives(self):
        cases = {
            "status": lambda b: b.update(status="DRAFT"),
            "approved": lambda b: b.update(approved=False),
            "schema": lambda b: b.update(schema_version="x"),
            "ruling_sha": lambda b: b["ruling"].update(sha256="0" * 64),
            "ruling_path": lambda b: b["ruling"].update(path="configs/amendments/other.yaml"),
            "record_path": lambda b: b["smoke_test_record"].update(path="outputs/audit/M8_SMOKE_TEST_RECORD_copy.json"),
            "record_sha": lambda b: b["smoke_test_record"].update(sha256="0" * 64),
            "contract": lambda b: b["smoke_contract"].update(sha256="1" * 64),
            "evaluator": lambda b: b["evaluator_contract"].update(sha256="2" * 64),
            "smoke_source": lambda b: b.update(smoke_source_commit="3" * 40),
            "seeds": lambda b: b["scope"].update(seeds=[42, 1337]),
            "methods": lambda b: b["scope"].update(methods=["E01"]),
            "split": lambda b: b["scope"].update(split="VAL"),
            "test_access": lambda b: b["authorizes"].update(test_split_access=True),
            "gpat": lambda b: b["authorizes"].update(gpat=True),
            "no_production": lambda b: b["authorizes"].update(e01_e02_production_bank_generation=False),
            "config": lambda b: b["method_configs"]["E01"].update(sha256="4" * 64),
        }
        with tempfile.TemporaryDirectory() as td:
            for name, mutate in cases.items():
                b = binding()
                mutate(b)
                p = Path(td) / f"{name}.yaml"
                p.write_text(json.dumps(b))
                for m in bw.METHODS:
                    if name == "config" and m == "E02":
                        continue
                    self.assertBlocked(M8, bw.load_production_authorization, m, binding_path=p)
            draft = Path(td) / "e01_e02_m8_smoke_binding.DRAFT.yaml"
            draft.write_text(json.dumps(binding()))
            self.assertBlocked(M8, bw.load_production_authorization, "E01", binding_path=draft)
            self.assertBlocked(M8, bw.load_production_authorization, "E01", binding_path=Path(td) / "absent.yaml")
            ok = Path(td) / "ok.yaml"
            ok.write_text(json.dumps(binding()))
            self.assertEqual(bw.load_production_authorization("E02", binding_path=ok).mode, "PRODUCTION")

    def test_07_inline_record_in_a_ruling_is_refused(self):
        with tempfile.TemporaryDirectory() as td:
            r = json.loads((ROOT / bw.RULING_PATH).read_text())
            r["smoke_test_record"] = {"path": bw.APPROVED_SMOKE_RECORD_PATH, "sha256": bw.APPROVED_SMOKE_RECORD_SHA256}
            p = Path(td) / "e01_e02_bank_milestone_ruling.yaml"
            p.write_text(json.dumps(r))
            self.assertIn("inside the ruling", self.assertBlocked(M8, bw.load_production_authorization, "E01", p))

    def test_08_seed_scope_and_smoke_ancestry(self):
        self.assertBlocked(AUTH, bw.load_production_authorization, "E01", seed=7)
        with mock.patch.object(bw, "_is_ancestor", return_value=False):
            self.assertBlocked(AUTH, bw.load_production_authorization, "E02", seed=42)
        with self.assertRaises(bw.BankError):
            bw.load_production_authorization("E03")


@contextlib.contextmanager
def fake_root(mutate_record):
    """A temporary copy of the governance files with a mutated smoke record; the binding and the pinned record SHA follow
    the mutation, so only the record's own content can make the gate refuse."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        for rel in GOVERNANCE_FILES:
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / rel, root / rel)
        rec = json.loads((ROOT / bw.APPROVED_SMOKE_RECORD_PATH).read_text())
        mutate_record(rec)
        (root / bw.APPROVED_SMOKE_RECORD_PATH).write_text(json.dumps(rec))
        digest = sha(root / bw.APPROVED_SMOKE_RECORD_PATH)
        b = binding()
        b["smoke_test_record"]["sha256"] = digest
        (root / bw.SMOKE_BINDING_PATH).write_text(json.dumps(b))
        with mock.patch.object(bw, "ROOT", root), mock.patch.object(bw, "APPROVED_SMOKE_RECORD_SHA256", digest), \
                mock.patch.object(bw, "_is_ancestor", return_value=True):
            yield root


class SmokeRecordIdentity(Base):
    def run_case(self, mutate, blocked=("E01", "E02")):
        with fake_root(mutate):
            for m in bw.METHODS:
                if m in blocked:
                    self.assertBlocked(M8, bw.load_production_authorization, m)
                else:
                    self.assertEqual(bw.load_production_authorization(m).mode, "PRODUCTION")

    def test_09_unmutated_copy_passes(self):
        self.run_case(lambda r: None, blocked=())

    def test_10_record_tampering_refused(self):
        def stage_gate_false(r):
            r["stages"]["metrics"]["gates"][next(iter(r["stages"]["metrics"]["gates"]))] = False

        def drop_seed(r):
            del r["coverage"]["banks"]["E02.seed_2026"]
        cases = [
            lambda r: r.update(mode="synthetic"),
            lambda r: r.update(label="SCIENTIFIC"),
            lambda r: r.update(smoke_test_passed=False),
            lambda r: r.update(scientific_results_produced=True),
            stage_gate_false,
            lambda r: r["stages"]["plots"].update(simulated=True),
            lambda r: r["stages"].pop("plots"),
            drop_seed,
            lambda r: r["coverage"]["banks"]["E01.seed_42"].update(outputs=185),
            lambda r: r["coverage"]["banks"]["E01.seed_42"].update(failures=1),
            lambda r: r["independent_replay"].update(artifacts_compared=1720),
            lambda r: r["independent_replay"].update(differences=["x"]),
            lambda r: r["test_firewall"].update(violations=1),
            lambda r: r["test_firewall"].update(atime_changed_files=1),
            lambda r: r["source"].update(git_commit="5" * 40),
            lambda r: r["source"].update(worktree_dirty=True),
            lambda r: r.update(contract_sha256="6" * 64),
            lambda r: r.update(evaluator_contract_sha256="7" * 64),
        ]
        for i, mutate in enumerate(cases):
            with self.subTest(case=i):
                self.run_case(mutate)

    def test_11_per_method_bank_stage_gates(self):
        def e02_gate(r):
            r["stages"]["bank"]["gates"]["B.banks.E02.seed_1337.full_replay"] = False
        with fake_root(e02_gate):
            with self.assertRaises(Exception):   # the validator rejects any false stage gate for both methods
                bw.load_production_authorization("E02")

        def drop(key):
            def mutate(r):
                held = [st for st in ("generator", "bank") if key in r["stages"][st]["gates"]]
                self.assertEqual(len(held), 1, key)
                del r["stages"][held[0]]["gates"][key]
            return mutate
        self.run_case(drop("A.banks.E02.seed_42.k40_n159_residual"), blocked=("E02",))
        self.run_case(drop("B.banks.E02.seed_1337.full_replay"), blocked=("E02",))
        self.run_case(drop("B.banks.E01.seed_2026.operator_is_position_mod_8"), blocked=("E01",))
        self.run_case(drop("A.banks.E01.seed_42.file_verification"), blocked=("E01",))


def authority_doc(cand, files, rt, fs, machine_id, **over):
    doc = {"schema_version": bw.PRODUCTION_AUTHORITY_SCHEMA, "status": "OWNER_APPROVED", "approved": True,
           "candidate_commit": cand, "binding_sha256": sha(ROOT / bw.SMOKE_BINDING_PATH),
           "ruling_sha256": bw.APPROVED_RULING_SHA256, "smoke_record_sha256": bw.APPROVED_SMOKE_RECORD_SHA256,
           "scope": {"methods": ["E01", "E02"], "seeds": [42, 1337, 2026]}, "authority_files": files,
           "execution": {"machine_id": machine_id},
           "storage": {"runtime_root": str(rt), "filesystem_uuid": fs["filesystem_uuid"], "mount_point": fs["mount_point"],
                       "min_free_bytes_before_bank": 1, "min_free_inodes_before_bank": 1}}
    doc.update(over)
    return doc


class LaunchGate(Base):
    def setUp(self):
        self.auth = bw.load_production_authorization("E01", seed=42)
        self.head = runlog.git_commit()
        self.files = {rel: sha(ROOT / rel) for rel in bw.AUTHORITY_FILES}
        self._td = tempfile.TemporaryDirectory()
        self.td = Path(self._td.name)
        self.rt = (self.td / "runtime").resolve()
        (self.rt / "runs" / "m6").mkdir(parents=True)
        self.fs = bw.filesystem_identity(self.rt)
        self.machine = Path("/etc/machine-id").read_text().strip() if Path("/etc/machine-id").is_file() else "none"
        self.exec_cfg = self.td / "exec.yaml"
        self.exec_cfg.write_text(json.dumps({"storage": {"runtime_root": str(self.rt)}}))

    def tearDown(self):
        self._td.cleanup()

    def write(self, doc):
        p = self.td / f"authority_{time.monotonic_ns()}.yaml"
        p.write_text(json.dumps(doc))
        return p

    @contextlib.contextmanager
    def host(self, *, cuda="", dirty=False, files_ok=True):
        env = {"CUDA_VISIBLE_DEVICES": cuda} if cuda is not None else {}
        with mock.patch.object(bw, "EXECUTION_CONFIG_PATH", str(self.exec_cfg)), \
                mock.patch.object(runlog, "git_dirty", return_value=dirty), \
                mock.patch.object(bw, "verify_locked_environment", return_value={"python_version": "3.11.16"}), \
                mock.patch.dict(os.environ, env, clear=False):
            if cuda is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            if files_ok:
                with mock.patch.object(bw, "_git", side_effect=self.fake_git), \
                        mock.patch.object(bw, "_is_ancestor", return_value=True):
                    yield
            else:
                yield

    def fake_git(self, *args):
        rel = args[1].split(":", 1)[1]
        return mock.Mock(returncode=0, stdout=(ROOT / rel).read_bytes())

    def launch(self, doc, seed=42, auth=None, rt=None):
        return bw.verify_production_launch("E01", seed, auth or self.auth, rt or self.rt, authority_path=self.write(doc))

    def good(self, **over):
        return authority_doc(self.head, dict(self.files), self.rt, self.fs, self.machine, **over)

    def test_12_authority_record_negatives(self):
        qa = bw.qualification_authorization("u")
        self.assertBlocked(AUTH, self.launch, self.good(), auth=qa)
        self.assertBlocked(AUTH, self.launch, self.good(), seed=7)
        self.assertBlocked(AUTH, bw.verify_production_launch, "E01", 42, self.auth, self.rt,
                           authority_path=self.td / "absent.yaml")
        self.assertBlocked(AUTH, self.launch, self.good(status="DRAFT"))
        self.assertBlocked(AUTH, self.launch, self.good(approved=False))
        self.assertBlocked(AUTH, self.launch, self.good(binding_sha256="0" * 64))
        self.assertBlocked(AUTH, self.launch, self.good(scope={"methods": ["E01"], "seeds": [42]}))
        self.assertBlocked(AUTH, self.launch, self.good(candidate_commit="8" * 40))      # not an ancestor
        short = dict(self.files)
        short.pop("methods/common/bank_writer.py")
        self.assertBlocked(AUTH, self.launch, self.good(authority_files=short))
        with self.host():
            bad = dict(self.files, **{"tools/e01_e02_bank.py": "9" * 64})
            self.assertBlocked(AUTH, self.launch, self.good(authority_files=bad))
        with self.host(dirty=True):
            self.assertIn("dirty", self.assertBlocked(AUTH, self.launch, self.good()))

    def test_13_preflight_negatives_and_pass(self):
        with self.host():
            self.assertBlocked(PRE, self.launch, self.good(execution={"machine_id": "not-this-host"}))
            self.assertBlocked(PRE, self.launch, self.good(storage={**self.good()["storage"], "runtime_root": "/elsewhere"}))
            self.assertBlocked(PRE, self.launch, self.good(storage={**self.good()["storage"], "filesystem_uuid": "x-y"}))
            self.assertBlocked(PRE, self.launch, self.good(storage={**self.good()["storage"], "mount_point": "/nowhere"}))
            self.assertBlocked(PRE, self.launch,
                               self.good(storage={**self.good()["storage"], "min_free_bytes_before_bank": 1 << 62}))
            self.assertBlocked(PRE, self.launch,
                               self.good(storage={**self.good()["storage"], "min_free_inodes_before_bank": 1 << 62}))
            (self.rt / "runs" / "m6" / "E01" / "seed_42").mkdir(parents=True)
            self.assertIn("run directory exists", self.assertBlocked(PRE, self.launch, self.good()))
            if self.fs["filesystem_uuid"] is None:
                self.assertBlocked(PRE, self.launch, self.good(), seed=1337)
            else:
                ok = self.launch(self.good(), seed=1337)
                self.assertEqual(ok["candidate_commit"], self.head)
                self.assertEqual(ok["filesystem"]["filesystem_uuid"], self.fs["filesystem_uuid"])
                self.assertTrue(ok["run_dir"].endswith("runs/m6/E01/seed_1337"))
        with self.host(cuda="0"):
            self.assertIn("CUDA", self.assertBlocked(PRE, self.launch, self.good(), seed=2026))
        with self.host(cuda=None):
            self.assertIn("CUDA", self.assertBlocked(PRE, self.launch, self.good(), seed=2026))
        with self.host():
            with mock.patch.object(bw, "verify_locked_environment", side_effect=bw.BankError("not gpat-m5")):
                self.assertIn("environment", self.assertBlocked(PRE, self.launch, self.good(), seed=2026))

    def test_14_symlinked_runtime_root_refused(self):
        link = self.td / "link"
        link.symlink_to(self.rt)
        with self.host():
            self.assertBlocked(PRE, self.launch, self.good(storage={**self.good()["storage"], "runtime_root": str(link)}),
                               rt=link)


class NoAtimeReads(Base):
    def test_15_every_mode_reads_with_o_noatime(self):
        with tempfile.TemporaryDirectory() as td:
            inp = build_fixture(Path(td) / "faces", 3)
            old = time.time() - 5 * 86400
            files = [inp.face_path(s) for s in inp.face_sha256]
            for f in files:
                os.utime(f, (old, f.stat().st_mtime))
            before = {f: f.stat().st_atime_ns for f in files}
            flags = []
            real_open = os.open

            def spy(path, fl, *a, **k):
                if str(path).endswith(".png"):
                    flags.append(fl)
                return real_open(path, fl, *a, **k)
            with mock.patch.object(bw.os, "open", spy):
                for s in inp.face_sha256:
                    inp.read(s)
            self.assertTrue(flags and all(fl & os.O_NOATIME for fl in flags))
            self.assertEqual({f: f.stat().st_atime_ns for f in files}, before)
            self.assertEqual(inp.atime_checked_reads, len(files))
            self.assertNotIn("read_bytes()", Path(bw.__file__).read_text().split("def read(self, sample_id")[1]
                             .split("def _read_parquet_checked")[0])

    def test_16_atime_change_during_read_aborts(self):
        with tempfile.TemporaryDirectory() as td:
            inp = build_fixture(Path(td) / "faces", 2)
            sid = next(iter(inp.face_sha256))
            real = bw.read_bytes_noatime

            def touching(p):
                raw = real(p)
                os.utime(p, (time.time() - 3600, Path(p).stat().st_mtime))
                return raw
            with mock.patch.object(bw, "read_bytes_noatime", touching):
                with self.assertRaises(bw.BankError):
                    inp.read(sid)


class Cli(Base):
    def test_17_cli_refuses_off_host_and_touches_nothing(self):
        import tools.e01_e02_bank as cli
        with tempfile.TemporaryDirectory() as td:
            for m in bw.METHODS:
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    rc = cli.main(["--method", m, "--seed", "1337", "--runtime-root", td])
                self.assertEqual(rc, cli.EXIT_BLOCKED)
                self.assertIn(json.loads(out.getvalue())["status"], {AUTH, PRE})
            self.assertEqual(os.listdir(td), [])

    def test_18_cli_has_no_test_subset_or_resume_option(self):
        import tools.e01_e02_bank as cli
        src = Path(cli.__file__).read_text()
        opts = {line.split('"')[1] for line in src.splitlines() if "ap.add_argument(" in line}
        self.assertEqual(opts, {"--method", "--seed", "--runtime-root", "--check-only"})
        for token in ("TEST", "--resume", "--limit", "--subset", "smoke_authorization"):
            self.assertNotIn(f'"{token}', src)
        self.assertLess(src.index("verify_production_launch"), src.index("production_inputs"))
        self.assertLess(src.index("audit_hook.install"), src.index("inputs.read("))


if __name__ == "__main__":
    unittest.main()
