"""E04 D12 activation in the qualification evidence tool: build-d12 writes a NEW record (never the original 41/42
record, which `build` now refuses to overwrite); the 41 retained gates and D12 R1-R10 plus the evidence check gate;
K.real_data_gradient_repeatability is kept as a superseded, non-gating, never-counted diagnostic; any failed retained
gate, R criterion, evidence check or approval pin leaves E04_PRODUCTION_QUALIFICATION_INCOMPLETE; the evidence check
fails closed on a missing or altered document, the original run id, a dirty or non-descendant GPU checkout and a
fresh pre-pass that differs from the frozen one. Stdlib only.
"""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))

import e04_d12_procedural_gate as g  # noqa: E402
import e04_production_qualification_evidence as ev  # noqa: E402

QUAL = json.loads((ROOT / ev.OUT['record']).read_text())
APPROVAL = json.loads((ROOT / g.APPROVED_AMENDMENT).read_text())
HEAD = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def fresh_gates(k_pass=False):
    return [dict(x, **{'pass': True if x['gate'] != g.SUPERSEDED_GATE else k_pass}) for x in QUAL['gates']]


def d12_result(**fail):
    c = {k: k not in fail for k in g.CRITERIA}
    return {'criteria_sha256': g.CRITERIA_SHA256, 'criteria': c, 'failures': {k: [v] for k, v in fail.items()},
            'pass': all(c.values()), 'diagnostics_not_gating': {'functional_gradient_ratio': {'t0 step1': 1e-7},
                                                                'forward_varying_tensors': [3],
                                                                'cross_process_post_state_distinct': [3]},
            'superseded_gate': {'gate': g.SUPERSEDED_GATE, 'recorded_pass': False, 'counted_as_passed': False}}


def compose(gates=None, d12=None, evidence=True, approval=None):
    doc = {'gates': gates if gates is not None else fresh_gates(), 'status': 'x', 'failed_gates': []}
    return ev.compose_d12(doc, d12 or d12_result(), {'pass': evidence, 'problems': []}, approval or APPROVAL,
                          ev.ORIGINAL_RECORD_SHA256)


class E04D12Activation(unittest.TestCase):
    def test_01_new_record_paths_and_original_refused(self):
        self.assertFalse(set(ev.OUT_D12.values()) & set(ev.OUT.values()))
        self.assertEqual(hashlib.sha256((ROOT / ev.OUT['record']).read_bytes()).hexdigest(), ev.ORIGINAL_RECORD_SHA256)
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(SystemExit):
                ev.main(['build', '--collected', d, '--repo-root', str(ROOT)])
        self.assertEqual(hashlib.sha256((ROOT / ev.OUT['record']).read_bytes()).hexdigest(), ev.ORIGINAL_RECORD_SHA256)
        self.assertTrue(subprocess.run(['git', 'merge-base', '--is-ancestor', ev.D12_APPROVAL_COMMIT, HEAD],
                                       cwd=ROOT).returncode == 0)

    def test_02_pass_requires_everything_and_k_never_counted(self):
        out = compose()
        self.assertEqual(out['status'], ev.STATUS_D12_OK)
        self.assertEqual(out['failed_gates'], [])
        names = [x['gate'] for x in out['gates']]
        self.assertNotIn(g.SUPERSEDED_GATE, names)
        self.assertEqual(sorted(n for n in names if not n.startswith('D12.')), sorted(g.RETAINED_GATES))
        self.assertEqual(sorted(n for n in names if n.startswith('D12.R')), sorted('D12.' + k for k in g.CRITERIA))
        self.assertIn('D12.evidence_complete_and_hashed', names)
        self.assertEqual((out['gates_passed'], out['gates_total']), (52, 52))
        k = out['superseded_gates']
        self.assertEqual([(x['gate'], x['pass'], x['gating'], x['counted_as_passed']) for x in k],
                         [(g.SUPERSEDED_GATE, False, False, False)])
        self.assertEqual((out['original_record']['modified'], out['original_record']['retroactively_passed']),
                         (False, False))
        self.assertFalse(out['scientific_seed_execution_authorized'])
        self.assertEqual(out['scientific_seeds_launched'], [])
        self.assertIsNone(out['e04_scientific_authority'])
        self.assertTrue(out['production_qualification_authority'])
        self.assertEqual(out['d12']['acceptance_criteria_sha256'], g.CRITERIA_SHA256)

    def test_03_any_failure_is_incomplete(self):
        for name in g.RETAINED_GATES:
            gates = fresh_gates()
            next(x for x in gates if x['gate'] == name)['pass'] = False
            out = compose(gates=gates)
            self.assertEqual((out['status'], out['failed_gates']), (ev.STATUS_FAIL, [name]), name)
            self.assertFalse(out['production_qualification_authority'])
        for k in g.CRITERIA:
            out = compose(d12=d12_result(**{k: 'defect'}))
            self.assertEqual((out['status'], out['failed_gates']), (ev.STATUS_FAIL, ['D12.' + k]), k)
        self.assertEqual(compose(evidence=False)['failed_gates'], ['D12.evidence_complete_and_hashed'])
        bad = dict(APPROVAL, acceptance_criteria_sha256='0' * 64)
        self.assertEqual(compose(approval=bad)['status'], ev.STATUS_FAIL)
        counted = d12_result()
        counted['superseded_gate']['counted_as_passed'] = True
        self.assertEqual(compose(d12=counted)['failed_gates'], ['D12.approval_pin'])

    def test_04_k_pass_on_requalification_still_not_counted(self):
        out = compose(gates=fresh_gates(k_pass=True))
        self.assertEqual(out['superseded_gates'][0]['pass'], True)
        self.assertFalse(out['superseded_gates'][0]['counted_as_passed'])
        self.assertEqual(out['gates_total'], 52)

    def test_05_evidence_check_fails_closed(self):
        with tempfile.TemporaryDirectory() as d:
            col = Path(d)
            for name in ev.D12_COLLECTED:
                (col / name).parent.mkdir(parents=True, exist_ok=True)
                body = {'run_id': 'fresh000000000000'} if name[:-5] in ev.CASES else {'name': name}
                if name.startswith('prepass_compare'):
                    body = {'identical': True, 'differing': []}
                (col / name).write_text(json.dumps(body))
            (col / 'host_state.json').write_text(json.dumps({'git_head': HEAD, 'git_status': ''}))

            def sums():
                (col / 'raw_SHA256SUMS').write_text(''.join('%s  ./%s\n' % (hashlib.sha256(
                    (col / n).read_bytes()).hexdigest(), n) for n in ev.D12_COLLECTED))
            sums()
            ok = ev.evidence_check(ROOT, col)
            self.assertTrue(ok['pass'], ok['problems'])
            (col / 'P/P2/summary.json').write_text('{"altered": 1}')
            self.assertFalse(ev.evidence_check(ROOT, col)['pass'])
            sums()
            (col / 'C/d10_trace2/P1/d12_extra.json').unlink()
            self.assertFalse(ev.evidence_check(ROOT, col)['pass'])
            (col / 'C/d10_trace2/P1/d12_extra.json').write_text('{}')
            sums()
            self.assertTrue(ev.evidence_check(ROOT, col)['pass'])
            (col / 'det_b.json').write_text(json.dumps({'run_id': ev.ORIGINAL_RUN_ID}))
            sums()
            self.assertFalse(ev.evidence_check(ROOT, col)['pass'])
            (col / 'det_b.json').write_text(json.dumps({'run_id': 'fresh000000000000'}))
            sums()
            for host in ({'git_head': HEAD, 'git_status': ' M x'}, {'git_head': '57bd1f9', 'git_status': ''}):
                (col / 'host_state.json').write_text(json.dumps(host))
                self.assertFalse(ev.evidence_check(ROOT, col)['pass'], host)
            (col / 'host_state.json').write_text(json.dumps({'git_head': HEAD, 'git_status': ''}))
            (col / 'prepass_compare_frozen.json').write_text(json.dumps({'identical': False, 'differing': ['x']}))
            sums()
            self.assertFalse(ev.evidence_check(ROOT, col)['pass'])

    def test_06_collect_atime_default_prefix_unchanged(self):
        import inspect
        sig = inspect.signature(ev.collect_atime)
        self.assertEqual((sig.parameters['post_name'].default, sig.parameters['prefix'].default), ('post', 'e04_prodq'))


if __name__ == '__main__':
    unittest.main()
