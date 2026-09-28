"""M6D6jF: E07c SCIENTIFIC RunContext seed-metadata hotfix (implementation bugfix; no training, GPU or data).

The first seed-42 launch at the M6D6jR authority died in E07cMainRunContext.open() with
`dict.update() got multiple values for keyword argument 'experiment_seed'`: SCIENTIFIC seed_field IS 'experiment_seed'.
These tests open/close the REAL E07cMainRunContext in a temporary runtime root in both modes (no Torch). QUALIFICATION
output is compared with the M6D6jR authority implementation (volatile fields normalized). History assertions compare
the state at the commit that ADDED this test (candidate: the worktree) with the authority (no B1 HEAD lock).
"""
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from methods.common.runlog import RunDirectoryError  # noqa: E402
from methods.difffas import main_runner as mr  # noqa: E402
from methods.difffas import main_runner_io as mio  # noqa: E402

AUTHORITY = '8358d8b6fe468478ad86a715616becb81bb9b339'
THIS = 'tests/test_m6d6jf_e07c_scientific_context_hotfix.py'
RUNNER = 'methods/difffas/main_runner.py'
CHANGED_METHODS = {'_seed_metadata', '_open_locked', '_resolved_config', '_manifest', 'log_event', 'close'}
VOLATILE = {'run_uuid', 'start_utc', 'end_utc', 'utc', 'wall_clock_seconds', 'git_dirty'}
ENV = {'host': 'm6d6jf-fixture', 'user': 'm6d6jf-fixture', 'platform': 'fixture', 'python_version': '3',
       'numpy_version': 'fixture', 'gpu_model': None, 'gpu_count': 0, 'cuda_version': None, 'cudnn_version': None,
       'framework': 'torch', 'framework_version': 'fixture', 'pytorch_version': 'fixture', 'tensorflow_version': None,
       'dependency_fingerprint': 'fixture', 'environment_lock_path': 'environments/e07c.lock.json'}
REASONS = {k: 'M6D6jF static fixture: no GPU contacted' for k, v in ENV.items() if v is None}


def git(*args):
    return subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True)


def m6d6jf_commit():
    return git('log', '--diff-filter=A', '--format=%H', '-1', '--', THIS).stdout.decode().strip() or None


def at_m6d6jf(rel):
    commit = m6d6jf_commit()
    return git('show', f'{commit}:{rel}').stdout if commit else (ROOT / rel).read_bytes()


def at_authority(rel):
    return git('show', f'{AUTHORITY}:{rel}').stdout


def authority_runner_module():
    """The buggy M6D6jR-authority main_runner.py, loaded from Git into an isolated module."""
    tmp = Path(tempfile.mkdtemp(prefix='m6d6jf-authority-'))
    path = tmp / 'main_runner_at_authority.py'
    path.write_bytes(at_authority(RUNNER))
    spec = importlib.util.spec_from_file_location('m6d6jf_main_runner_at_authority', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def context(module, mode, seed, rt):
    ids = mio.identities(mio.load_contract())
    return module.E07cMainRunContext(mode=mode, seed=seed, runtime_root=rt, environment=dict(ENV),
                                     missing_environment_reasons=dict(REASONS), identities=ids,
                                     command_line='m6d6jf-static-fixture')


def lifecycle(module, mode, seed, rt, summary=None):
    ctx = context(module, mode, seed, rt)
    ctx.open()
    ctx.log_event('m6d6jf_probe', {'k': 1})
    out = ctx.close(summary=summary or {'test_split_accessed': False})
    return ctx, out


def artifacts(ctx):
    return {'run_manifest': json.loads(ctx.path('run_manifest').read_text()),
            'resolved_config': yaml.safe_load(ctx.path('resolved_config').read_text()),
            'checkpoint_index': json.loads(ctx.path('checkpoint_index').read_text()),
            'run_summary': json.loads(ctx.path('run_summary').read_text()),
            'events': [json.loads(x) for x in ctx.path('metrics').read_text().splitlines()]}


def normalized_bytes(ctx, rt):
    """Every run file, with volatile values removed and the temp root replaced (key ORDER is kept)."""
    def scrub(node):
        if isinstance(node, dict):
            return {k: scrub(v) for k, v in node.items() if k not in VOLATILE}
        if isinstance(node, list):
            return [scrub(v) for v in node]
        return node
    out = {}
    for key in ('run_manifest', 'checkpoint_index', 'run_summary'):
        out[key] = json.dumps(scrub(json.loads(ctx.path(key).read_text())))
    out['resolved_config'] = json.dumps(scrub(yaml.safe_load(ctx.path('resolved_config').read_text())))
    out['metrics'] = [json.dumps(scrub(json.loads(x))) for x in ctx.path('metrics').read_text().splitlines()]
    return json.loads(json.dumps(out).replace(str(rt), '<RT>'))


class ScientificContextOpen(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix='m6d6jf-')
        self.rt = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_01_authority_reproduces_the_launch_failure(self):
        old = authority_runner_module()
        ctx = context(old, mio.SCIENTIFIC, 42, self.rt)
        with self.assertRaisesRegex(TypeError, "multiple values for keyword argument 'experiment_seed'"):
            ctx.open()

    def test_01b_authority_close_would_refuse_scientific_summary(self):
        """Third latent site: the authority close() put experiment_seed into the summary, which RunContext refuses."""
        old = authority_runner_module()
        ctx = context(mr, mio.SCIENTIFIC, 42, self.rt)
        ctx.open()
        with self.assertRaisesRegex(RunDirectoryError, 'summary may not overwrite run identity'):
            old.E07cMainRunContext.close(ctx, summary={'test_split_accessed': False})
        self.assertFalse(ctx.path('run_summary').exists())
        self.assertEqual(ctx.close(summary={'test_split_accessed': False})['experiment_seed'], 42)

    def test_02_scientific_seed_42_opens_and_closes(self):
        ctx, summary = lifecycle(mr, mio.SCIENTIFIC, 42, self.rt)
        self.assertEqual(ctx.run_dir, self.rt / 'runs' / 'm6' / 'E07c' / 'seed_42')
        a = artifacts(ctx)
        m = a['run_manifest']
        self.assertEqual((m['experiment_seed'], m['runner_mode'], m['scientific_run'], m['qualification_only'],
                          m['resume_policy'], m['completion_status']),
                         (42, mio.SCIENTIFIC, True, False, 'NONE', 'completed'))
        self.assertNotIn('qualification_seed', m)
        self.assertNotIn('experiment_seed', m['missing_field_reasons'])
        self.assertIs(ctx.resume, False)
        self.assertEqual((summary['experiment_seed'], summary['runner_mode']), (42, mio.SCIENTIFIC))
        self.assertNotIn('qualification_seed', summary)
        self.assertEqual(a['run_summary'], json.loads(json.dumps(summary)))

    def test_03_scientific_checkpoint_index(self):
        ctx, _ = lifecycle(mr, mio.SCIENTIFIC, 42, self.rt)
        idx = artifacts(ctx)['checkpoint_index']
        self.assertEqual((idx['experiment_seed'], idx['runner_mode'], idx['checkpoints'], idx['checkpoint_rule']),
                         (42, mio.SCIENTIFIC, [], 'BASELINE_FINAL_STATE_V1'))
        self.assertNotIn('qualification_seed', idx)
        self.assertNotIn('labels', idx)

    def test_04_scientific_resolved_config(self):
        ctx, _ = lifecycle(mr, mio.SCIENTIFIC, 42, self.rt)
        r = artifacts(ctx)['resolved_config']['_resolved']
        self.assertEqual((r['experiment_seed'], r['runner_mode'], r['scientific_run'], r['qualification_only'],
                          r['labels']), (42, mio.SCIENTIFIC, True, False, []))
        self.assertNotIn('qualification_seed', r)
        self.assertEqual(ctx._seed_metadata(), {'experiment_seed': 42})

    def test_05_scientific_events(self):
        ctx, _ = lifecycle(mr, mio.SCIENTIFIC, 42, self.rt)
        (event,) = artifacts(ctx)['events']
        self.assertEqual((event['event'], event['experiment_seed'], event['payload']), ('m6d6jf_probe', 42, {'k': 1}))
        self.assertNotIn('qualification_seed', event)

    def test_06_scientific_failure_close_path(self):
        ctx = context(mr, mio.SCIENTIFIC, 42, self.rt)
        with self.assertRaises(RuntimeError):
            with ctx:
                raise RuntimeError('m6d6jf probe')
        s = artifacts(ctx)['run_summary']
        self.assertEqual((s['experiment_seed'], s['completion_status'], s['failure_reason']),
                         (42, 'failed', 'RuntimeError: m6d6jf probe'))

    def test_07_every_scientific_seed_opens(self):
        for seed in (42, 1337, 2026):
            ctx, summary = lifecycle(mr, mio.SCIENTIFIC, seed, self.rt)
            self.assertEqual((ctx._seed_metadata(), summary['experiment_seed']), ({'experiment_seed': seed}, seed))

    def test_08_existing_root_refused_fresh_only(self):
        ctx, _ = lifecycle(mr, mio.SCIENTIFIC, 42, self.rt)
        before = {p: p.read_bytes() for p in ctx.run_dir.rglob('*') if p.is_file()}
        again = context(mr, mio.SCIENTIFIC, 42, self.rt)
        with self.assertRaisesRegex(RunDirectoryError, 'already exists.*MAIN_CHECKPOINT_RESUME is unqualified'):
            again.open()
        self.assertEqual({p: p.read_bytes() for p in ctx.run_dir.rglob('*') if p.is_file()}, before)
        with self.assertRaisesRegex(RunDirectoryError, 'opened only once'):
            ctx.open()


class QualificationUnchanged(unittest.TestCase):
    def setUp(self):
        self._tmp = [tempfile.TemporaryDirectory(prefix='m6d6jf-q-') for _ in range(2)]
        self.rts = [Path(t.name) for t in self._tmp]

    def tearDown(self):
        for t in self._tmp:
            t.cleanup()

    def test_09_qualification_metadata(self):
        ctx, summary = lifecycle(mr, mio.QUALIFICATION, 60608, self.rts[0])
        self.assertEqual(ctx._seed_metadata(), {'experiment_seed': None, 'qualification_seed': 60608})
        a = artifacts(ctx)
        for node in (a['run_manifest'], a['checkpoint_index'], a['resolved_config']['_resolved'], a['events'][0]):
            self.assertEqual((node['experiment_seed'], node['qualification_seed']), (None, 60608))
        self.assertEqual((summary['experiment_seed'], summary['qualification_seed']), (None, 60608))
        self.assertEqual(a['run_manifest']['missing_field_reasons']['experiment_seed'],
                         'QUALIFICATION_ONLY_NOT_A_SCIENTIFIC_RUN')
        self.assertEqual(summary['missing_field_reasons']['experiment_seed'], 'QUALIFICATION_ONLY_NOT_A_SCIENTIFIC_RUN')
        self.assertEqual(a['checkpoint_index']['labels'], list(mio.QUALIFICATION_LABELS))

    def test_10_qualification_output_identical_to_authority(self):
        old = authority_runner_module()
        new_ctx, _ = lifecycle(mr, mio.QUALIFICATION, 60608, self.rts[0])
        old_ctx, _ = lifecycle(old, mio.QUALIFICATION, 60608, self.rts[1])
        self.assertEqual(normalized_bytes(new_ctx, self.rts[0]), normalized_bytes(old_ctx, self.rts[1]))


class ScopeAndHistory(unittest.TestCase):
    def test_11_runner_change_limited_to_seed_metadata(self):
        new, old = ast.parse(at_m6d6jf(RUNNER)), ast.parse(at_authority(RUNNER))
        rest = lambda t: [ast.dump(n) for n in t.body if getattr(n, 'name', None) != 'E07cMainRunContext']  # noqa: E731
        self.assertEqual(rest(new), rest(old))
        cls = lambda t: {n.name: ast.dump(n) for n in next(c for c in t.body if getattr(c, 'name', None) ==  # noqa: E731
                                                           'E07cMainRunContext').body if hasattr(n, 'name')}
        cn, co = cls(new), cls(old)
        self.assertEqual({k for k in cn.keys() | co.keys() if cn.get(k) != co.get(k)}, CHANGED_METHODS)
        self.assertNotIn('_seed_metadata', co)

    def test_12_no_seed_field_spliced_beside_experiment_seed(self):
        cls = next(n for n in ast.parse(at_m6d6jf(RUNNER)).body if getattr(n, 'name', None) == 'E07cMainRunContext')
        for fn in (n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name != '_seed_metadata'):
            src = ast.unparse(fn)
            self.assertNotIn('self.seed_field', src, fn.name)
            self.assertNotIn('experiment_seed=self.seed', src, fn.name)
            self.assertNotIn("'experiment_seed': self.seed", src, fn.name)

    def test_13_historical_and_science_files_unchanged(self):
        paths = [p for p in git('ls-tree', '-r', '--name-only', AUTHORITY).stdout.decode().splitlines()
                 if p != RUNNER and (p.startswith(('configs/', 'docs/spec/', 'environments/', 'tests/', 'tools/'))
                                     or p.startswith('methods/') or
                                     (p.startswith('outputs/audit/M6D6') and Path(p).suffix in ('.json', '.md')))]
        self.assertGreater(len(paths), 100)
        for rel in paths:
            self.assertEqual(at_m6d6jf(rel), at_authority(rel), rel)

    def test_14_ledger_prefix(self):
        ledger = 'outputs/audit/EXECUTION_LEDGER.jsonl'
        prefix = at_authority(ledger)
        self.assertEqual(len(prefix.splitlines()), 127)
        self.assertTrue(at_m6d6jf(ledger).startswith(prefix))

    def test_15_record_and_evidence_via_static_preflight(self):
        spec = importlib.util.spec_from_file_location('m6d6jf_preflight_under_test',
                                                      ROOT / 'tools/m6d6jf_e07c_scientific_context_preflight.py')
        pf = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pf)
        ev = pf.check_evidence(reader=at_m6d6jf)
        self.assertEqual((ev['status'], ev['authority_commit'], ev['failed_attempt']['failure_phase']),
                         ('PASS', AUTHORITY, 'RUN_CONTEXT_OPEN'))
        self.assertEqual(ev['statuses']['MAIN_CHECKPOINT_RESUME'], 'UNQUALIFIED')


if __name__ == '__main__':
    unittest.main()
