"""M6D6f E07c auxiliary exact epoch-boundary resume (Amendment A8): authority, sidecar, CLI, reconciliation, evidence.

Static checks run everywhere and never claim CUDA execution; they use synthetic run roots in
temporary directories only (no canonical face, no benchmark image, no manifest row beyond the
authorized TRAIN reader already exercised by M6D6e). The optional live test (GPU host) runs a
synthetic tiny model on CUDA through the real A8 transaction, verified load and restore. The
real-TRAIN bitwise resume result is validated from the recorded process evidence.
"""
import ast
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from methods.common.config import sha256_file
from methods.common.learned import PreparationError
from methods.difffas import aux_resume as ar
from methods.difffas import aux_runner_io as aio

ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = 'a00aba9f998fb0d91122617012af0dec7794c007'
A8_DOC = 'docs/spec/amendments/GPAT_TransferBench_v1_0_Amendment_A8_E07c_Aux_Resume_Policy.md'
A8_DOC_SHA = 'deb11b5cd80fa873bde1ef9e87160fed3a8065903cd8c81c9463b087fbf73b2f'
A8_OVERLAY_SHA = 'a098ce509156151ee6ce7ead1e9afa5711264b676a597e2514b1058ed46f471a'
ENGINE = ROOT / 'methods/difffas/aux_runner.py'
RESUME = ROOT / 'methods/difffas/aux_resume.py'
HARNESS = ROOT / 'methods/difffas/aux_resume_qualification.py'
CLI = ROOT / 'tools/run_e07c_aux.py'
EVIDENCE = {k: ROOT / f'outputs/audit/M6D6F_E07C_AUX_RESUME_{k.upper()}.json'
            for k in ('reference', 'interrupted', 'restored', 'epoch0_interrupted', 'epoch0_restored')}
HAS_TORCH = importlib.util.find_spec('torch') is not None


def fn_source(path, name, cls=None):
    tree = ast.parse(Path(path).read_text())
    body = tree.body
    if cls:
        body = next(n for n in body if isinstance(n, ast.ClassDef) and n.name == cls).body
    return ast.unparse(next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == name))


def load_cli():
    spec = importlib.util.spec_from_file_location('run_e07c_aux_m6d6f_under_test', CLI)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evidence():
    return {k: json.loads(p.read_text()) for k, p in EVIDENCE.items()} if all(p.is_file() for p in EVIDENCE.values()) \
        else None


def synthetic_root(d, epoch=1, payload=b'sidecar-bytes'):
    """A run root with a committed sidecar entry (bytes are opaque; nothing is deserialized)."""
    run = Path(d) / 'run'
    rdir = run / 'checkpoints' / 'resume'
    rdir.mkdir(parents=True)
    side = rdir / ar.sidecar_name(epoch)
    side.write_bytes(payload)
    entry = {'path': ar.sidecar_rel(epoch), 'completed_epoch': epoch, 'global_step': epoch * 56,
             'size_bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest(), 'status': 'COMMITTED_CURRENT',
             'bytes_pruned': False}
    (run / ar.INDEX_NAME).write_text(json.dumps({'kind': ar.RESUME_KIND, 'schema_version': 1, 'committed': entry,
                                                 'entries': [entry]}))
    return run, side, entry


class Ctx:
    """Minimal torch-free context for metrics reconciliation tests."""
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)

    def path(self, key):
        return self.run_dir / {'metrics': 'metrics.jsonl', 'checkpoint_index': 'checkpoint_index.json'}[key]


def step(g, epoch):
    return {'record_type': 'epoch', 'epoch': epoch, 'global_step': g, 'loss_hex': '0x1p+0'}


def write_metrics(path, records):
    raw = b''.join(json.dumps(r).encode() + b'\n' for r in records)
    Path(path).write_bytes(raw)
    return raw


class TestM6D6fStatic(unittest.TestCase):
    # ---------------------------------------------------------------- 1-5 authority
    def test_01_current_authority(self):
        r = subprocess.run(['git', '-C', str(ROOT), 'merge-base', '--is-ancestor', AUTHORITY, 'HEAD'])
        self.assertEqual(r.returncode, 0)
        self.assertEqual(ar.load_policy()['policy']['authority_commit'], AUTHORITY)

    def test_02_a8_authority_identity(self):
        self.assertEqual(sha256_file(ROOT / A8_DOC), A8_DOC_SHA)
        self.assertEqual(sha256_file(ROOT / ar.A8_OVERLAY), A8_OVERLAY_SHA)
        self.assertEqual(ar.A8_OVERLAY_SHA256, A8_OVERLAY_SHA)
        p = ar.load_policy()['policy']
        self.assertEqual((p['amendment_id'], p['classification'], p['milestone'], p['method_id']),
                         ('A8_E07C_AUX_RESUME_POLICY', 'DETERMINISTIC_IMPLEMENTATION_CLARIFICATION', 'M6D6f', 'E07c'))
        self.assertEqual(p['amendment_document'], {'path': A8_DOC, 'sha256': A8_DOC_SHA})
        for rel, digest in p['bound_authority_sha256'].items():
            self.assertEqual(sha256_file(ROOT / rel), digest, rel)
        with tempfile.TemporaryDirectory() as d:
            bad = Path(d) / 'a8.yaml'
            bad.write_bytes((ROOT / ar.A8_OVERLAY).read_bytes() + b' ')
            with self.assertRaises(PreparationError):
                ar.load_policy(bad)

    def test_03_no_new_fidelity_or_deviation(self):
        p = ar.load_policy()['policy']
        self.assertEqual(p['fidelity'], {**p['fidelity'], 'fidelity_class': 'CONTROLLED_ADAPTATION', 'deviation': 'DEV-021',
                                         'new_fidelity_class': False, 'new_deviation': False})
        self.assertEqual(p['logical_run']['scientific_auxiliary_runs'], 1)
        cfg = json.loads((ROOT / aio.CONTRACT_PATH).read_text())
        self.assertEqual((cfg['fidelity_class'], cfg['deviation'], cfg['new_deviation']), ('CONTROLLED_ADAPTATION', 'DEV-021', False))
        self.assertEqual(cfg['resume']['status'], 'AUX_RESUME_NOT_QUALIFIED')          # M6D6e record untouched
        text = (ROOT / A8_DOC).read_text()
        self.assertIn('no new fidelity class\nand no new scientific deviation', text)

    def test_04_source_has_no_resume(self):
        src = (ROOT / 'third_party/source_cache/difffas/models/pretrain_classifier.py').read_text()
        self.assertEqual(hashlib.sha256(src.encode()).hexdigest(),
                         '3444691f6f2c59b41a07f06db9388b134e849bb889c5f6b03489d47c7e037b16')
        for token in ('load_state_dict', 'torch.load', 'resume', 'start_epoch', 'optimizer.state_dict'):
            self.assertNotIn(token, src)
        self.assertIn('for epoch in range(0,num_epochs)', src)
        self.assertEqual(ar.load_policy()['policy']['source']['source_resume_capability'], 'NONE')

    def test_05_run_logging_resume_requirements(self):
        text = (ROOT / 'configs/run_logging_v1.yaml').read_text()
        for line in ('resume_policy: APPEND_OR_EXPLICITLY_RECONCILE', 'resume_overwrite: FORBIDDEN',
                     'resume_reconciliation_record_required: true'):
            self.assertIn(line, text)
        p = ar.load_policy()['policy']['run_logging_v1_requirements']
        self.assertEqual(p, {'resume_policy': 'APPEND_OR_EXPLICITLY_RECONCILE', 'resume_overwrite': 'FORBIDDEN',
                             'resume_reconciliation_record_required': True})

    # ---------------------------------------------------------------- 6-12 sidecar
    def test_06_epoch0_sidecar_before_first_iterator(self):
        src = fn_source(ENGINE, 'run_scientific')
        fresh = src.index('run.begin_fresh()')
        self.assertLess(src.index('seed_process(torch, mode, seed, config)'), src.index('production_components('))
        self.assertLess(src.index('production_components('), fresh)
        self.assertLess(fresh, src.index('trainer.run_epoch(epoch, ctx, t0)'))
        self.assertIn('require(t.completed_epoch == t.global_step == t.optimizer_applications == 0',
                      fn_source(RESUME, 'begin_fresh', 'LogicalRun'))
        self.assertEqual(ar.load_policy()['policy']['epoch_0_boundary']['before'],
                         ['iter(loader)', 'first batch', 'first backward', 'first optimizer.step'])

    def test_07_sidecar_self_contained_model_state(self):
        src = fn_source(RESUME, 'build_state')
        self.assertIn("'model': model_state", src)
        self.assertIn('trainer.model.state_dict()', src)
        self.assertIn("'scientific_checkpoint_file': checkpoint", src)
        self.assertNotIn('encoder_final', fn_source(RESUME, 'restore'))           # restore never reads the .pkl

    def test_08_09_10_11_optimizer_and_rng_captured(self):
        src = fn_source(RESUME, 'build_state') + fn_source(RESUME, 'rng_state')
        for token in ('trainer.optimizer.state_dict()', 'torch.get_rng_state()', 'torch.cuda.get_rng_state(DEVICE)',
                      'torch.cuda.get_rng_state_all()', 'random.getstate()', 'np.random.get_state()',
                      "torch.tensor(internal, dtype=torch.int64)", 'torch.from_numpy(keys.astype(np.int64))'):
            self.assertIn(token, src)
        restore = fn_source(RESUME, 'restore')
        order = [restore.index(t) for t in ("model.load_state_dict(state['model'], strict=True)",
                                            "optimizer.load_state_dict(state['optimizer'])", "optimizer.state[p]['momentum_buffer']",
                                            'random.setstate(', 'np.random.set_state(', "torch.set_rng_state(r['torch_cpu'])",
                                            "torch.cuda.set_rng_state_all(r['torch_cuda_all'])",
                                            'ep.precision_state() == ep.EXPECTED_STATE')]
        self.assertEqual(order, sorted(order))

    def test_12_weights_only_true_never_false(self):
        text = RESUME.read_text()
        self.assertIn("torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)", text)
        self.assertNotIn('weights_only=False', text)
        loads = [ast.unparse(n) for n in ast.walk(ast.parse(text)) if isinstance(n, ast.Call) and
                 ast.unparse(n.func).endswith('torch.load')]
        self.assertEqual(loads, ["torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)"])
        self.assertNotIn('weights_only=False', HARNESS.read_text())

    # ---------------------------------------------------------------- 13-18 SHA / containment / explicit path
    def test_13_14_sha_before_load_and_wrong_sha_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            run, side, entry = synthetic_root(d)
            self.assertEqual(ar.read_verified(side, entry), b'sidecar-bytes')
            side.write_bytes(b'sidecar-bytez')                               # same size, wrong SHA256
            torch_stub = mock.MagicMock()
            with self.assertRaises(ar.ResumeStop):
                ar.load_sidecar(torch_stub, ar.read_verified(side, entry))
            torch_stub.load.assert_not_called()
        v = fn_source(RESUME, 'verify_before_open', 'LogicalRun')
        self.assertLess(v.index('read_verified(path, entry)'), v.index('load_sidecar(self.torch, raw)'))

    def test_13b_malformed_sha_rejected_before_open(self):
        with tempfile.TemporaryDirectory() as d:
            run, side, entry = synthetic_root(d)
            idx = json.loads((run / ar.INDEX_NAME).read_text())
            idx['committed']['sha256'] = idx['entries'][0]['sha256'] = 'NOT-A-SHA'
            (run / ar.INDEX_NAME).write_text(json.dumps(idx))
            with mock.patch('os.open', side_effect=AssertionError('file opened')) as opened:
                with self.assertRaises(ar.ResumeStop):
                    ar.resolve_committed(run, str(side))
                with self.assertRaises(ar.ResumeStop):
                    ar.read_verified(side, idx['committed'])
            opened.assert_not_called()

    def test_15_sidecar_containment(self):
        with tempfile.TemporaryDirectory() as d:
            run, side, entry = synthetic_root(d)
            outside = Path(d) / ar.sidecar_name(1)
            outside.write_bytes(b'sidecar-bytes')
            link = run / 'checkpoints' / 'resume' / 'runner_state_epoch_002.pth'
            link.symlink_to(side)
            for bad in (outside, link, run / 'checkpoints' / ar.sidecar_name(1), Path(ar.sidecar_rel(1)),
                        run / 'checkpoints' / 'resume' / '..' / 'resume' / ar.sidecar_name(1)):
                with self.assertRaises(ar.ResumeStop, msg=str(bad)):
                    ar.resolve_committed(run, str(bad))
            self.assertEqual(ar.resolve_committed(run, str(side))[0], side.resolve())

    def test_16_exact_indexed_path_only(self):
        with tempfile.TemporaryDirectory() as d:
            run, side, entry = synthetic_root(d)
            older = run / 'checkpoints' / 'resume' / ar.sidecar_name(0)
            older.write_bytes(b'x')
            with self.assertRaises(ar.ResumeStop):
                ar.resolve_committed(run, str(older))                         # exists, but not the committed entry
            idx = json.loads((run / ar.INDEX_NAME).read_text())
            idx['entries'].append(dict(entry, path=ar.sidecar_rel(0)))          # two COMMITTED_CURRENT entries
            (run / ar.INDEX_NAME).write_text(json.dumps(idx))
            with self.assertRaises(ar.ResumeStop):
                ar.resolve_committed(run, str(side))

    def test_17_18_no_resume_latest_no_discovery(self):
        cli = load_cli()
        for flag in ('--resume-latest', '--resume', '--auto-resume', '--resume-from', '--resume-epoch', '--resume-dir'):
            with self.assertRaises(SystemExit) as cm, mock.patch('sys.stderr', io.StringIO()):
                cli.parse(['--seed', '42', '--execution-config', aio.EXEC_CONFIG, flag])
            self.assertEqual(cm.exception.code, 2, flag)
        args = cli.parse(['--seed', '42', '--execution-config', aio.EXEC_CONFIG, '--resume-state', '/x/y.pth'])
        self.assertEqual(args.resume_state, '/x/y.pth')
        text = RESUME.read_text() + CLI.read_text()
        for token in ('glob(', 'max(', 'sorted(os.listdir', 'latest'):
            self.assertNotIn(token, fn_source(RESUME, 'resolve_committed') + fn_source(CLI, 'preconditions'))
        self.assertIn("require(raw.is_absolute(), '--resume-state must be an absolute path", text)

    # ---------------------------------------------------------------- 19-21 identity
    def test_19_20_same_logical_run_and_seed(self):
        v = fn_source(RESUME, 'validate_state')
        for token in ("lr['run_id']", "lr['run_uuid']", "lr['start_utc']", "state['seed']", 'ctx.aux_seed'):
            self.assertIn(token, v)
        self.assertEqual(aio.run_root('/rt', aio.SCIENTIFIC, 42), Path('/rt/runs/m6/E07c/aux_encoder/seed_42'))
        self.assertIn('self.run_uuid = old["run_uuid"]', (ROOT / 'methods/common/runlog.py').read_text())
        rid = aio.run_id(60606, 'c' * 64, AUTHORITY)
        roots = {aio.run_root('/rt', aio.QUALIFICATION, 60606, rid, lab) for lab in aio.RESUME_QUALIFICATION_LABELS}
        self.assertEqual(len(roots), 3)
        self.assertTrue(all('/qualification/m6d6f/E07c_aux/q60606-' in str(r) for r in roots))
        with self.assertRaises(PreparationError):
            aio.run_root('/rt', aio.QUALIFICATION, 60606, rid)                   # M6D6f roots need a role label
        with self.assertRaises(PreparationError):
            aio.run_root('/rt', aio.QUALIFICATION, 60605, rid, 'reference')      # M6D6e roots carry none
        for seed in (42, 1337, 2026, 0, 60607):
            with self.assertRaises(PreparationError):
                aio.validate_mode_seed(aio.QUALIFICATION, seed)
        with self.assertRaises(PreparationError):
            aio.validate_mode_seed(aio.SCIENTIFIC, 60606)

    def test_21_identity_gates(self):
        ident = fn_source(RESUME, 'process_identity') + fn_source(RESUME, 'git_identity')
        for token in ("'rev-parse', 'HEAD'", "'branch', '--show-current'", "'status', '--porcelain'",
                      'python_executable=sys.executable', 'environment=dict(ctx._environment)',
                      "exec_config_sha256=storage['exec_config_sha256']", "faces_256_root=storage['faces_256_root']",
                      'code_sha256=code_hashes()', 'precision_state=dict(precision_state)',
                      "require(not ident['git_dirty'], 'SCIENTIFIC resume state requires a clean worktree')"):
            self.assertIn(token, ident)
        ids = aio.identities(aio.load_contract())
        for key in ('config_sha256', 'a1_adaptation_sha256', 'a3_sha256', 'a6_overlay_sha256', 'a7_overlay_sha256',
                    'm6d6e_contract_sha256', 'environment_lock_sha256', 'split_manifest_sha256', 'class_map_sha256',
                    'source_commit', 'source_tree'):
            self.assertIn(key, ids)
        self.assertEqual(set(ar.a8_identities()), {'a8_overlay_sha256', 'a8_document_sha256'})
        state = {'kind': ar.RESUME_KIND, 'schema_version': 1, **ar.ENGINEERING_LABELS, 'method_id': 'E07c',
                 'runner_mode': 'QUALIFICATION', 'seed': 60606, 'seed_field': 'qualification_seed',
                 'logical_run': {'identity': 'ONE_LOGICAL_RUN', 'run_id': 'r', 'run_uuid': 'u', 'start_utc': 's',
                                 'run_dir': '/d'},
                 'completed_epoch': 1, 'global_step': 56, 'identities': {'git_commit': 'A', 'a7_overlay_sha256': 'x'},
                 'counters': {'logical_authoritative_optimizer_steps': 56, 'physical_optimizer_steps_executed': 57,
                              'superseded_optimizer_steps': 1}}
        ctx = mock.Mock(mode='QUALIFICATION', aux_seed=60606, seed_field='qualification_seed', run_id='r', run_uuid='u',
                        start_utc='s', run_dir=Path('/d'))
        entry = {'completed_epoch': 1, 'global_step': 56}
        ar.validate_state(state, identity={'git_commit': 'A', 'a7_overlay_sha256': 'x'}, ctx=ctx, entry=entry)
        for drift in ({'git_commit': 'B', 'a7_overlay_sha256': 'x'}, {'git_commit': 'A', 'a7_overlay_sha256': 'y'},
                      {'git_commit': 'A', 'a7_overlay_sha256': 'x', 'extra': 1}):
            with self.assertRaises(ar.ResumeStop):
                ar.validate_state(state, identity=drift, ctx=ctx, entry=entry)
        for field, value in (('seed', 42), ('runner_mode', 'SCIENTIFIC'), ('scientific_checkpoint', True)):
            with self.assertRaises(ar.ResumeStop):
                ar.validate_state(dict(state, **{field: value}), identity=state['identities'], ctx=ctx, entry=entry)
        with self.assertRaises(ar.ResumeStop):
            ar.validate_state(dict(state, logical_run=dict(state['logical_run'], run_uuid='v')),
                              identity=state['identities'], ctx=ctx, entry=entry)

    # ---------------------------------------------------------------- 22-25 metrics
    def _reconcile_fixture(self, d, post):
        run = Path(d)
        pre = [{'record_type': 'event', 'event': 'e07c_aux_run_start', 'payload': {}}] + [step(g, 1) for g in range(1, 57)]
        raw = write_metrics(run / 'metrics.jsonl', pre)
        state = {'completed_epoch': 1, 'global_step': 56,
                 'metrics_boundary': {'bytes': len(raw), 'prefix_sha256': hashlib.sha256(raw).hexdigest()}}
        with open(run / 'metrics.jsonl', 'ab') as fh:
            fh.write(b''.join(json.dumps(r).encode() + b'\n' for r in post))
        return Ctx(run), state

    def test_22_23_24_partial_epoch_superseded_never_truncated(self):
        with tempfile.TemporaryDirectory() as d:
            ctx, state = self._reconcile_fixture(d, [{'record_type': 'event', 'event': 'e07c_aux_resume_state_committed',
                                                      'payload': {}}, step(57, 2)])
            before = ctx.path('metrics').read_bytes()
            out = ar.reconcile_metrics(ctx, state)
            self.assertEqual(ctx.path('metrics').read_bytes(), before)                 # read-only analysis
            s = out['superseded']
            self.assertEqual((s['classification'], s['step_records'], s['global_step_range'], s['epochs']),
                             (ar.SUPERSEDED, 1, [57, 57], [2]))
            self.assertEqual((out['overwrite'], out['truncated'], out['replay']),
                             (False, False, {'from_epoch': 2, 'from_global_step': 57}))
        for code in (fn_source(RESUME, 'reconcile_metrics'), fn_source(RESUME, 'begin_resume', 'LogicalRun')):
            for token in ('truncate(', "open(ctx.path('metrics'), 'w'", 'write_text(', 'write_bytes('):
                self.assertNotIn(token, code)
        self.assertIn("ctx.log_event('e07c_aux_resume_reconciliation', payload)", fn_source(RESUME, 'begin_resume', 'LogicalRun'))

    def test_23b_logical_view_and_repeated_resume(self):
        recs = [step(g, 1) for g in range(1, 57)] + [step(57, 2), step(58, 2)]
        recs.append({'record_type': 'event', 'event': 'e07c_aux_resume_reconciliation',
                     'payload': {'committed_boundary': {'global_step': 56}}})
        recs += [step(57, 2)]
        logical = ar.logical_trajectory(recs)
        self.assertEqual([r['global_step'] for r in logical], list(range(1, 58)))
        with tempfile.TemporaryDirectory() as d:
            ctx, state = self._reconcile_fixture(d, recs[56:])
            out = ar.reconcile_metrics(ctx, state)
            self.assertEqual((out['superseded']['step_records'], out['superseded']['previous_reconciliations_after_boundary']),
                             (3, 1))

    def test_24b_reconciliation_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            ctx, state = self._reconcile_fixture(d, [step(57, 2)])
            with open(ctx.path('metrics'), 'ab') as fh:
                fh.write(b'{"record_type": "epo')                                  # torn record
            with self.assertRaises(ar.ResumeStop):
                ar.reconcile_metrics(ctx, state)
        with tempfile.TemporaryDirectory() as d:
            ctx, state = self._reconcile_fixture(d, [step(57, 2)])
            raw = bytearray(ctx.path('metrics').read_bytes())
            raw[5] = ord('X')                                                       # rewritten prefix
            ctx.path('metrics').write_bytes(bytes(raw))
            with self.assertRaises(ar.ResumeStop):
                ar.reconcile_metrics(ctx, state)
        with tempfile.TemporaryDirectory() as d:
            ctx, state = self._reconcile_fixture(d, [step(113, 3)])                  # a record past epoch E+1
            with self.assertRaises(ar.ResumeStop):
                ar.reconcile_metrics(ctx, state)

    def test_25_physical_logical_superseded_separated(self):
        src = fn_source(RESUME, 'summary', 'LogicalRun') + fn_source(RESUME, '_counters_now', 'LogicalRun')
        for key in ('logical_authoritative_optimizer_steps', 'physical_optimizer_steps_executed', 'superseded_optimizer_steps',
                    'physical = logical + superseded'):
            self.assertIn(key, src)
        b = fn_source(RESUME, 'begin_resume', 'LogicalRun')
        self.assertIn("'physical_optimizer_steps_executed': c['physical_optimizer_steps_executed'] + new_superseded", b)
        self.assertIn("'superseded_optimizer_steps': c['superseded_optimizer_steps'] + new_superseded", b)
        self.assertEqual(ar.load_policy()['policy']['step_accounting']['logical_steps_completed_run'], 200 * 56)
        self.assertIn("'11200 logical optimizer steps'", fn_source(ENGINE, 'run_scientific'))

    # ---------------------------------------------------------------- transaction
    def test_commit_transaction_order(self):
        src = fn_source(RESUME, '_commit', 'LogicalRun')
        order = [src.index(t) for t in ("open(partial, 'xb')", 'os.fsync(fh.fileno())', 'sha256_file(partial)',
                                        'os.replace(partial, final)', 'atomic_write_json(index_path(ctx.run_dir), index)',
                                        'os.remove(old)')]
        self.assertEqual(order, sorted(order))
        self.assertIn("index['pruning_log'].append(", src)
        self.assertIn('bytes_pruned=True', src)
        self.assertEqual(ar.sidecar_name(0), 'runner_state_epoch_000.pth')
        self.assertEqual(ar.sidecar_rel(137), 'checkpoints/resume/runner_state_epoch_137.pth')
        cb = fn_source(RESUME, 'commit_boundary', 'LogicalRun')
        self.assertLess(cb.index('t.checkpoint_event(epoch, self.ctx, self.config)'), cb.index('self._commit('))

    def test_stale_files_recorded_then_removed(self):
        with tempfile.TemporaryDirectory() as d:
            run, side, entry = synthetic_root(d)
            rdir = run / 'checkpoints' / 'resume'
            (rdir / 'runner_state_epoch_002.pth.partial').write_bytes(b'p')
            (rdir / 'runner_state_epoch_002.pth').write_bytes(b'q')
            (run / 'checkpoints' / 'encoder_final.pkl.partial').write_bytes(b'r')
            index = json.loads((run / ar.INDEX_NAME).read_text())
            removed, pruned = ar.reconcile_files(Ctx(run), {}, entry, index)
            self.assertEqual(sorted(r['path'] for r in removed),
                             ['checkpoints/encoder_final.pkl.partial', 'checkpoints/resume/runner_state_epoch_002.pth',
                              'checkpoints/resume/runner_state_epoch_002.pth.partial'])
            self.assertTrue(all(len(r['sha256']) == 64 and r['size_bytes'] == 1 for r in removed))
            self.assertEqual(sorted(os.listdir(rdir)), [ar.sidecar_name(1)])             # committed bytes untouched
            (rdir / ar.sidecar_name(0)).write_bytes(b'z')                              # unindexed, before boundary
            with self.assertRaises(ar.ResumeStop):
                ar.reconcile_files(Ctx(run), {}, entry, index)

    # ---------------------------------------------------------------- CLI
    def test_cli_refusals_existing_and_missing_roots(self):
        cli = load_cli()
        env = {'PYTHONHASHSEED': '42', 'NVIDIA_TF32_OVERRIDE': '0', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}
        with tempfile.TemporaryDirectory() as d:
            rt = Path(d) / 'rt'
            storage = {'runtime_root': str(rt), 'faces_256_root': str(rt / 'data/processed/faces_256'),
                       'exec_config': aio.EXEC_CONFIG,
                       'exec_config_sha256': aio.load_contract()['bound_inputs_sha256'][aio.EXEC_CONFIG]}
            with mock.patch.object(aio, 'faces_root_from_exec_config', return_value=storage):
                plain = cli.parse(['--seed', '42', '--execution-config', aio.EXEC_CONFIG])
                side = rt / 'runs/m6/E07c/aux_encoder/seed_42/checkpoints/resume/runner_state_epoch_001.pth'
                resumed = cli.parse(['--seed', '42', '--execution-config', aio.EXEC_CONFIG, '--resume-state', str(side)])
                plan = cli.preconditions(plain, environ=env, dirty=False, branch='m6-baselines', check_interpreter=False)
                self.assertIsNone(plan['resume'])
                with self.assertRaises(SystemExit), mock.patch('sys.stderr', io.StringIO()):
                    cli.preconditions(resumed, environ=env, dirty=False, branch='m6-baselines', check_interpreter=False)
                run, _, entry = synthetic_root(Path(d) / 'x')
                seed_root = rt / 'runs/m6/E07c/aux_encoder/seed_42'
                seed_root.parent.mkdir(parents=True)
                run.rename(seed_root)
                with self.assertRaises(SystemExit), mock.patch('sys.stderr', io.StringIO()):
                    cli.preconditions(plain, environ=env, dirty=False, branch='m6-baselines', check_interpreter=False)
                plan = cli.preconditions(resumed, environ=env, dirty=False, branch='m6-baselines', check_interpreter=False)
                self.assertEqual((plan['resume']['completed_epoch'], plan['resume']['resumes_at_epoch']), (1, 2))
                self.assertEqual(plan['resume_policy'], 'A8_ONE_LOGICAL_RUN_EXACT_EPOCH_BOUNDARY')
                for k in ({'dirty': True}, {'branch': 'main'}):
                    with self.assertRaises(SystemExit), mock.patch('sys.stderr', io.StringIO()):
                        cli.preconditions(resumed, environ=env, **{'dirty': False, 'branch': 'm6-baselines',
                                                                   'check_interpreter': False, **k})
        self.assertNotIn('import torch', CLI.read_text().split('def main')[0])

    # ---------------------------------------------------------------- 34-36
    def test_34_scientific_checkpoint_format_unchanged(self):
        src = fn_source(ENGINE, 'checkpoint_event', 'Trainer')
        self.assertIn('save_whole_module(self.model, partial, config)', src)
        self.assertIn('os.replace(partial, final)', src)
        for token in ('state_dict', 'aux_resume'):
            self.assertNotIn(token, src)
        self.assertEqual(aio.CHECKPOINT_NAME, 'encoder_final.pkl')
        self.assertEqual(aio.CHECKPOINT_RULE, 'FINAL_STATE_AFTER_EPOCH_200')
        p = ar.load_policy()['policy']['scientific_checkpoint_unchanged']
        self.assertEqual((p['path'], p['format']), ('checkpoints/encoder_final.pkl', 'torch.save of the WHOLE nn.Module'))
        for token in ('resume_state', 'load_state_dict', 'torch.load('):          # engine still restores nothing itself
            self.assertNotIn(token, ENGINE.read_text())

    def test_35_sidecar_not_consumable_by_main(self):
        self.assertEqual(ar.ENGINEERING_LABELS, {'scientific_checkpoint': False, 'official_difffas_checkpoint': False,
                                                 'selection_candidate': False, 'consumable_by_main_difffas': False,
                                                 'reportable_as_trained_model': False})
        for rel in ('methods/difffas/aux_checkpoint.py', 'methods/difffas/execution_policy.py',
                    'methods/difffas/encoder.py', 'methods/difffas/adapter.py'):
            text = (ROOT / rel).read_text()
            self.assertNotIn('aux_resume', text)
            self.assertNotIn('runner_state_epoch', text)
        self.assertNotIn('.pth', aio.CHECKPOINT_NAME)

    def test_36_no_scientific_launch_code_in_harness(self):
        text = HARNESS.read_text()
        self.assertNotIn('run_scientific', text)
        self.assertNotIn('SCIENTIFIC_SEED', text)
        self.assertIn('SEED = aio.RESUME_QUALIFICATION_SEED', text)
        self.assertEqual(aio.RESUME_QUALIFICATION_SEED, 60606)


@unittest.skipUnless(evidence() is not None, 'M6D6f process evidence not yet recorded')
class TestM6D6fEvidence(unittest.TestCase):
    """Validated from outputs/audit/M6D6F_E07C_AUX_RESUME_*.json (the five fresh GPU qualification processes)."""

    @classmethod
    def setUpClass(cls):
        cls.ev = evidence()
        cls.R, cls.I, cls.S = cls.ev['reference'], cls.ev['interrupted'], cls.ev['restored']
        cls.EI, cls.ES = cls.ev['epoch0_interrupted'], cls.ev['epoch0_restored']

    def test_all_processes_pass_qualification_only(self):
        for k, e in self.ev.items():
            self.assertEqual(e['status'], 'PASS', k)
            self.assertEqual((e['qualification_seed'], e['auxiliary_encoder_training_seed'], e['experiment_seed'],
                              e['scientific_seed_consumed'], e['scientific_attempt'], e['label']),
                             (60606, None, None, False, False, 'QUALIFICATION_ONLY'), k)
            self.assertEqual(e['launch_environment']['PYTHONHASHSEED'], '60606')
            self.assertIn('/qualification/m6d6f/E07c_aux/q60606-', e['run_dir'])
            self.assertNotIn('/runs/', e['run_dir'])
            self.assertEqual(e['code_sha256'], self.R['code_sha256'])            # one code identity for all five
        self.assertEqual(len({e['run_id'] for e in self.ev.values()}), 1)
        self.assertEqual((self.S['run_dir'], self.ES['run_dir']), (self.I['run_dir'], self.EI['run_dir']))
        self.assertEqual(len({e['pid'] for e in self.ev.values()}), 5)                # five fresh OS processes
        for rel, digest in self.R['code_sha256'].items():
            self.assertEqual(sha256_file(ROOT / rel), digest, 'executed code unchanged: ' + rel)

    def test_06_epoch0_state_committed_before_iterator(self):
        for e in (self.R, self.I, self.EI):
            s0 = e['epoch0_sidecar']
            self.assertEqual((s0['completed_epoch'], s0['global_step'], s0['path'], s0['scientific_checkpoint_file']),
                             (0, 0, 'checkpoints/resume/runner_state_epoch_000.pth', None))
            self.assertEqual(e['rng_before_epoch0_sidecar'], e['rng_after_epoch0_sidecar'])      # no RNG consumed
        self.assertEqual(self.EI['rng_before_iterator'], self.R['rng_after_epoch0_sidecar'])

    def test_08_09_10_restore_reaches_committed_boundary_state(self):
        rec = self.S['reconciliation']
        self.assertEqual(rec['restore']['model_sha256'], self.I['epoch1_sidecar']['model_sha256'])
        self.assertEqual(rec['restore']['optimizer_digest'], self.I['epoch1_sidecar']['optimizer_digest'])
        self.assertEqual(rec['restore']['optimizer_digest']['momentum_buffers'], 110)
        self.assertEqual(self.S['rng_after_restore'], self.R['rng_at_epoch1_boundary'])
        self.assertEqual(self.S['rng_before_iterator'], self.R['rng_at_epoch1_boundary'])
        self.assertEqual(self.ES['rng_before_iterator'], self.R['rng_after_epoch0_sidecar'])
        self.assertEqual(self.R['epoch1_sidecar']['model_sha256'], self.I['epoch1_sidecar']['model_sha256'])
        self.assertEqual(rec['restore']['order'], ['model', 'optimizer', 'optimizer tensors/devices', 'python+numpy RNG',
                                                   'torch CPU RNG', 'torch CUDA RNG', 'A7 state'])

    def test_12_13_weights_only_and_sha_first(self):
        for e in (self.S, self.ES):
            self.assertEqual(e['load_calls'], [{'map_location': "'cpu'", 'weights_only': True}] * 3)
            v = e['verified_sidecar']
            self.assertTrue(v['sha256_verified_before_load'] and v['weights_only'])
            self.assertEqual(v['torch_load_calls_so_far'], [{'map_location': "'cpu'", 'weights_only': True}])
            self.assertEqual(e['reconciliation']['restored_sidecar']['sha256'], v['entry']['sha256'])
            self.assertTrue(e['fresh_context_refuses_existing_root'])
            self.assertEqual(e['explicit_resume_path'], e['run_dir'] + '/' + v['entry']['path'])
        self.assertEqual(self.S['verified_sidecar']['entry']['sha256'], self.I['epoch1_sidecar']['sha256'])
        self.assertEqual(self.ES['verified_sidecar']['entry']['sha256'], self.EI['epoch0_sidecar']['sha256'])

    def test_19_20_same_logical_run(self):
        rec = self.S['reconciliation']
        self.assertEqual(rec['run_id'], self.I['run_id'])
        self.assertEqual(rec['logical_run'], 'ONE_LOGICAL_RUN')
        self.assertEqual(self.S['index_after']['run_uuid'], self.I['index_after']['run_uuid'])
        sessions = self.S['index_after']['sessions']
        self.assertEqual([x['kind'] for x in sessions], ['FRESH', 'RESUME'])
        self.assertNotEqual(sessions[0]['process_session_id'], sessions[1]['process_session_id'])
        self.assertEqual(sessions[1]['restored_from'], 'checkpoints/resume/runner_state_epoch_001.pth')
        for field in ('git_commit', 'git_branch', 'code_sha256', 'environment', 'python_executable', 'seed', 'mode',
                      'split_manifest_sha256', 'class_map_sha256', 'a7_overlay_sha256', 'a8_overlay_sha256',
                      'precision_state', 'faces_256_root', 'exec_config_sha256'):
            self.assertIn(field, rec['identity_fields_checked'])

    def test_22_23_24_append_only_reconciliation(self):
        for e, g in ((self.S, 57), (self.ES, 1)):
            rec = e['reconciliation']
            self.assertEqual((rec['overwrite'], rec['truncated'], rec['policy']),
                             (False, False, 'APPEND_OR_EXPLICITLY_RECONCILE'))
            sup = rec['superseded']
            self.assertEqual((sup['classification'], sup['step_records'], sup['global_step_range']),
                             ('SUPERSEDED_BY_RESUME_ROLLBACK', 1, [g, g]))
            self.assertTrue(e['metrics_append_only']['prefix_sha256_unchanged'])
            self.assertGreater(e['metrics_append_only']['grew_by_bytes'], 0)
            m = e['metrics_after']
            self.assertEqual((m['physical_step_records'], m['logical_step_records']), (g + 1, g))
            self.assertEqual(m['physical_global_steps_tail'][-2:], [g, g])
            self.assertEqual(m['events'][-2:], ['resume_reconciliation', 'e07c_aux_resume_reconciliation'])
        self.assertEqual(self.S['reconciliation']['superseded']['loss_hex'], [self.I['steps']['57']['loss_hex']])
        self.assertEqual(self.S['metrics_before_resume']['sha256'], self.I['metrics_after']['sha256'])

    def test_25_step_accounting(self):
        self.assertEqual(self.S['reconciliation']['counters_after_reconciliation'],
                         {'logical_authoritative_optimizer_steps': 56, 'physical_optimizer_steps_executed': 57,
                          'superseded_optimizer_steps': 1, 'process_sessions': 2})
        s = self.S['step_accounting']
        self.assertEqual((s['logical_authoritative_optimizer_steps'], s['physical_optimizer_steps_executed'],
                          s['superseded_optimizer_steps'], s['committed_boundary']['global_step'],
                          s['uncommitted_in_progress_steps']), (57, 58, 1, 56, 1))
        e0 = self.ES['step_accounting']
        self.assertEqual((e0['logical_authoritative_optimizer_steps'], e0['physical_optimizer_steps_executed'],
                          e0['superseded_optimizer_steps']), (1, 2, 1))
        self.assertEqual({k: e['physical_optimizer_steps_this_process'] for k, e in self.ev.items()},
                         {'reference': 57, 'interrupted': 57, 'restored': 1, 'epoch0_interrupted': 1, 'epoch0_restored': 1})

    def _bitwise(self, cmp, g):
        self.assertEqual(cmp['global_step'], g)
        self.assertTrue(cmp['fields']['all_bitwise_equal'])
        self.assertEqual(cmp['fields']['tolerance'], 'NONE (bitwise)')
        t = cmp['tensors']
        self.assertTrue(t['all_bitwise_equal'])
        for key, n in (('gradients', 110), ('parameters_after', 112), ('buffers_after', 111), ('momentum_after', 110)):
            self.assertEqual((t[key]['tensors'], t[key]['bitwise_equal_tensors'], t[key]['max_abs_diff']), (n, n, 0.0), key)
        self.assertEqual(t['rng_after'], {'torch_cpu': True, 'torch_cuda_device': True})

    def test_26_to_32_resumed_epoch2_first_step_bitwise(self):
        for other in ('reference', 'interrupted'):
            self._bitwise(self.S['comparisons'][other], 57)
        from methods.difffas.aux_resume_qualification import compare_steps
        mine = self.S['steps']['57']
        for ref in (self.R['steps']['57'], self.I['steps']['57']):
            self.assertTrue(compare_steps(ref, mine)['all_bitwise_equal'])
            for f in ('sample_ids', 'input_sha256', 'labels', 'loss_hex', 'gradients_sha256', 'parameters_after_sha256',
                      'momentum_after_sha256', 'rng_after'):
                self.assertEqual(ref[f], mine[f], f)
        self.assertEqual((mine['epoch'], mine['iteration'], mine['batch_size'], mine['gradient_tensors'],
                          mine['without_grad']), (2, 0, 256, 110, ['norm.bias', 'norm.weight']))
        self.assertNotEqual(mine['sample_ids_sha256'], self.R['steps']['1']['sample_ids_sha256'])   # a new epoch order

    def test_33_epoch0_first_step_reproduced(self):
        for other in ('reference', 'e0interrupted'):
            self._bitwise(self.ES['comparisons'][other], 1)
        self.assertEqual(self.ES['steps']['1']['loss_hex'], self.R['steps']['1']['loss_hex'])
        self.assertEqual(self.ES['reconciliation']['scientific_checkpoint_on_disk'], {'state': 'ABSENT_AT_EPOCH_0'})

    def test_uninterrupted_epoch1_deterministic_across_processes(self):
        self.assertEqual(self.R['epoch1'], self.I['epoch1'])
        self.assertEqual(self.R['epoch1_losses_hex'], self.I['epoch1_losses_hex'])
        self.assertEqual(self.R['epoch1_checkpoint_event']['sha256'], self.I['epoch1_checkpoint_event']['sha256'])
        self.assertEqual((self.R['epoch1']['optimizer_steps'], self.R['epoch1']['consumed_examples'],
                          self.R['epoch1']['dropped_examples'], self.R['epoch1']['epoch_loss_denominator']),
                         (56, 14336, 131, 14467))

    def test_34_whole_module_checkpoint_and_sidecar_are_distinct(self):
        for e in (self.R, self.I):
            ck, sc = e['epoch1_checkpoint_event'], e['epoch1_sidecar']
            self.assertEqual((ck['path'], ck['serialization_api'], ck['format'], ck['checkpoint_type'], ck['selected_for_final']),
                             ('checkpoints/encoder_final.pkl', 'torch.save(model, path)',
                              'torch.save of the WHOLE nn.Module (matches torch.load(path).cuda())', 'periodic', False))
            self.assertEqual(sc['scientific_checkpoint_file']['sha256'], ck['sha256'])
            self.assertEqual(sc['path'], 'checkpoints/resume/runner_state_epoch_001.pth')
            idx = e['index_after']
            self.assertEqual([(x['completed_epoch'], x['status'], x['bytes_pruned']) for x in idx['entries']],
                             [(0, 'SUPERSEDED_BY_NEWER_BOUNDARY', True), (1, 'COMMITTED_CURRENT', False)])
            self.assertEqual(len(idx['pruning_log']), 1)
            for key, value in ar.ENGINEERING_LABELS.items():
                self.assertIs(idx[key], value)

    def test_36_zero_scientific_runs(self):
        for k, e in self.ev.items():
            self.assertFalse(e['scientific_training'] or e['scientific_checkpoint_created'] or
                             e['scientific_run_root_written'] or e['main_difffas_training'] or e['synthetic_bank'], k)
            self.assertEqual(e['benchmark_access_audit']['scientific_run_root_accesses'], 0)
        log = (ROOT / 'outputs/audit/M6D6F_E07C_AUX_RESUME_RUNTIME_LOG.txt').read_text()
        after = [ln for ln in log.splitlines() if ln.startswith('scientific_paths_after:')]
        self.assertEqual(len(after), 5)
        self.assertEqual(set(after), {'scientific_paths_after: runs=ABSENT e07c_root=ABSENT aux_seed_42=ABSENT '
                                      'encoder_final=ABSENT'})

    def test_37_zero_val_test_access_per_process(self):
        expected = {'reference': 17664, 'interrupted': 17664, 'restored': 3328, 'epoch0_interrupted': 3328,
                    'epoch0_restored': 3328}
        for k, e in self.ev.items():
            a = e['benchmark_access_audit']
            self.assertEqual((a['VAL_image_reads'], a['TEST_image_reads'], a['non_train_face_accesses'],
                              a['raw_or_other_benchmark_data_accesses'], a['unauthorized_manifest_accesses'],
                              a['faces_enumeration_or_write'], a['foreign_weight_accesses'], a['denied_events']),
                             (0, 0, 0, 0, 0, 0, 0, 0), k)
            self.assertTrue(a['train_faces_all_in_population'])
            self.assertEqual(a['train_face_opens'], expected[k], k)
            s = e['access_summary']
            self.assertEqual((s['TRAIN_rows_exposed_to_dataset'], s['VAL_rows_exposed_to_dataset'],
                              s['TEST_rows_exposed_to_dataset']), (14467, 0, 0))
            self.assertEqual([c['api'] for c in e['pyarrow_calls']], ['pyarrow.parquet.read_metadata',
                                                                    'pyarrow.parquet.read_schema',
                                                                    'pyarrow.parquet.read_table'])
            self.assertFalse(e['population']['main_relation_opened'])
            self.assertFalse(e['VAL_access'] or e['TEST_access'])


@unittest.skipUnless(HAS_TORCH, 'Torch unavailable on this host; live synthetic test runs on the GPU host')
class TestM6D6fLiveSynthetic(unittest.TestCase):
    """Synthetic CUDA tiny model through the real A8 transaction, verified load, restore and reconciliation."""

    def setUp(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('CUDA unavailable')

    def test_live_commit_verify_restore_bitwise(self):
        import random
        import numpy as np
        import torch
        from methods.difffas import aux_runner as engine
        from methods.difffas import execution_policy as ep
        from methods.difffas import DiffFASAdapter
        config = DiffFASAdapter().config
        with mock.patch.dict(os.environ, {'NVIDIA_TF32_OVERRIDE': '0', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}):
            precision = engine.configure_precision(torch, config)
        ids = dict(aio.identities(aio.load_contract()), **ar.a8_identities())
        env, reasons = engine.environment_record(torch)

        def components(seed):
            torch.manual_seed(seed)
            model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 7)).cuda()
            opt = torch.optim.SGD(model.parameters(), lr=aio.LR, momentum=aio.MOMENTUM, weight_decay=aio.WEIGHT_DECAY)
            t = mock.Mock(model=model, optimizer=opt, named=list(model.named_parameters()),
                          completed_epoch=0, global_step=0, optimizer_applications=0)
            return t

        def draw_and_step(t, ctx, g, epoch):
            x = torch.randn(4, 8).cuda()                     # CPU RNG draw (stands in for the sampler seed)
            y = torch.randint(0, 7, (4,)).cuda()
            t.optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(t.model(x), y)
            loss.backward()
            t.optimizer.step()
            t.global_step, t.optimizer_applications = g, t.optimizer_applications + 1
            ctx.log_epoch(aio.step_record(mode=aio.QUALIFICATION, seed=60606, epoch=epoch, global_step=g, iteration=0,
                                          learning_rate=aio.LR, loss=float(loss), running_loss=float(loss),
                                          batch_size=4, batch_sample_sha256='0' * 64, batch_class_histogram=[0] * 7,
                                          wall_clock_seconds=0.0, gpu_memory_bytes=1))
            return float(loss).hex(), {n: p.detach().cpu().clone() for n, p in t.model.named_parameters()}

        with tempfile.TemporaryDirectory() as d:
            rt = Path(d) / 'rt'
            storage = {'runtime_root': str(rt), 'faces_256_root': str(rt / 'f'), 'exec_config_sha256': 'e' * 64}
            with mock.patch.dict(os.environ, {'PYTHONHASHSEED': '60606'}):
                ctx = engine.E07cAuxRunContext(mode=aio.QUALIFICATION, seed=60606, runtime_root=rt, environment=env,
                                               missing_environment_reasons=reasons, identities=ids,
                                               label='interrupted')
                t = components(1)
                run = ar.LogicalRun(torch, t, ctx, config, storage=storage, precision=precision)
                ctx.open()
                e0 = run.begin_fresh()
                self.assertEqual((e0['completed_epoch'], e0['global_step'], e0['path']),
                                 (0, 0, 'checkpoints/resume/runner_state_epoch_000.pth'))
                for g in range(1, 57):
                    draw_and_step(t, ctx, g, 1)
                t.completed_epoch = 1
                ck = aio.checkpoint_path(ctx.run_dir)
                ck.write_bytes(b'whole-module-stand-in')
                e1 = run._commit(checkpoint={'path': 'checkpoints/encoder_final.pkl', 'epoch': 1,
                                             'sha256': sha256_file(ck), 'size_bytes': ck.stat().st_size, 'format': 'x'})
                idx = ar.read_index(ctx.run_dir)
                self.assertEqual(idx['committed']['path'], e1['path'])
                self.assertEqual([(e['completed_epoch'], e['status'], e['bytes_pruned']) for e in idx['entries']],
                                 [(0, 'SUPERSEDED_BY_NEWER_BOUNDARY', True), (1, 'COMMITTED_CURRENT', False)])
                self.assertEqual(sorted(os.listdir(ar.resume_dir(ctx.run_dir))), [ar.sidecar_name(1)])
                ref_loss, ref_params = draw_and_step(t, ctx, 57, 2)             # the interrupted, logged step
                ref_rng = torch.get_rng_state().clone()
                ctx.close(completion_status='interrupted')
                metrics_before = ctx.path('metrics').read_bytes()

                # ---- fresh "process": different init, then verified restore
                ctx2 = engine.E07cAuxRunContext(mode=aio.QUALIFICATION, seed=60606, runtime_root=rt, environment=env,
                                                missing_environment_reasons=reasons, identities=ids, resume=True,
                                                label='interrupted')
                t2 = components(999)
                torch.randn(100)
                random.random()
                np.random.rand(3)
                run2 = ar.LogicalRun(torch, t2, ctx2, config, storage=storage, precision=precision)
                with mock.patch.object(torch, 'load', wraps=torch.load) as spy:
                    run2.verify_before_open(str(ctx.run_dir / e1['path']))
                self.assertEqual(spy.call_args.kwargs['weights_only'], True)
                ctx2.open()
                rec = run2.begin_resume()
                self.assertEqual(rec['superseded']['global_step_range'], [57, 57])
                self.assertEqual(rec['superseded']['classification'], 'SUPERSEDED_BY_RESUME_ROLLBACK')
                self.assertEqual(rec['scientific_checkpoint_on_disk']['state'], 'MATCHES_COMMITTED_BOUNDARY')
                self.assertEqual((rec['counters_after_reconciliation']['physical_optimizer_steps_executed'],
                                  rec['counters_after_reconciliation']['superseded_optimizer_steps']), (57, 1))
                self.assertEqual((t2.completed_epoch, t2.global_step), (1, 56))
                loss, params = draw_and_step(t2, ctx2, 57, 2)
                self.assertEqual(loss, ref_loss)
                self.assertTrue(all(torch.equal(params[k], ref_params[k]) for k in ref_params))
                self.assertTrue(torch.equal(torch.get_rng_state(), ref_rng))
                self.assertEqual(ep.precision_state(), ep.EXPECTED_STATE)
                s = run2.summary()
                self.assertEqual((s['logical_authoritative_optimizer_steps'], s['physical_optimizer_steps_executed'],
                                  s['superseded_optimizer_steps'], s['process_sessions']), (57, 58, 1, 2))
                ctx2.close(completion_status='interrupted')
                after = ctx.path('metrics').read_bytes()
                self.assertTrue(after.startswith(metrics_before))
                logical = ar.logical_trajectory([json.loads(x) for x in after.splitlines()])
                self.assertEqual([r['global_step'] for r in logical], list(range(1, 58)))
                self.assertEqual(json.loads(ctx2.path('run_manifest').read_text())['run_uuid'], ctx.run_uuid)

                # ---- wrong SHA: refused before torch.load
                side = ctx.run_dir / e1['path']
                raw = bytearray(side.read_bytes())
                raw[-1] ^= 1
                side.write_bytes(bytes(raw))
                ctx3 = engine.E07cAuxRunContext(mode=aio.QUALIFICATION, seed=60606, runtime_root=rt, environment=env,
                                                missing_environment_reasons=reasons, identities=ids, resume=True,
                                                label='interrupted')
                run3 = ar.LogicalRun(torch, components(3), ctx3, config, storage=storage, precision=precision)
                with mock.patch.object(torch, 'load') as never:
                    with self.assertRaises(ar.ResumeStop):
                        run3.verify_before_open(str(side))
                never.assert_not_called()
