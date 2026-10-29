"""E04 D12 procedural gate: owner-approved and active (used only by the qualification evidence tool's build-d12;
refuses without an OWNER_APPROVED amendment pinning its criteria), criteria identical to the DRAFT amendment, the superseded K gate never
counted as passed, and every criterion fails closed on its defect (graph change, input / initial-state difference,
missing gradient, wrong optimizer application, wrong state transition, non-finite state, forward corruption, gradient
above the integrity bound, a failed or missing retained gate, firewall). Numerical (non-bitwise) variation within the
bounds passes. Stdlib only.
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

import e04_d11_analysis as d11a  # noqa: E402
import e04_d12_procedural_gate as g  # noqa: E402

DRAFT = json.loads((ROOT / 'configs/amendments/e04_d12_procedural_reproducibility.DRAFT.yaml').read_text())
QUAL = json.loads((ROOT / 'outputs/audit/E04_PRODUCTION_QUALIFICATION.json').read_text())
PARTS = [{'nodes': [{'name': 'x', 'op': 'Const', 'inputs': [], 'attrs': {}, 'device': '/device:GPU:0'},
                    {'name': 'y', 'op': 'AddN', 'inputs': ['x', 'x'], 'attrs': {}, 'device': '/device:GPU:0'},
                    {'name': 'r', 'op': '_Retval', 'inputs': ['y'], 'attrs': {}, 'device': '/device:CPU:0'}]}]


def canon(parts):
    memo, nodes = d11a.canonical(parts)
    return (hashlib.sha256('\n'.join(sorted(set(memo.values()))).encode()).hexdigest(),
            hashlib.sha256('\n'.join(sorted(memo[n] for n, v in nodes.items() if v['op'] == '_Retval'))
                           .encode()).hexdigest())


def semantic():
    return {'scope_ok': True, 'adam_ok': True, 'beta_powers_ok': True, 'counters_ok': True, 'finite': True}


def iteration(t):
    grad = {'per_variable_l2_ref': {'G/a/kernel:0': 3.0, 'G/a/bias:0': 1e-6, 'G/a/BN/beta:0': 2.0,
                                    'D/D3/conv1/kernel:0': 4.0},
            'differing_variables': [{'variable': 'G/a/bias:0', 'l2_diff': 1e-6},       # BN-cancelled: excluded
                                    {'variable': 'D/D3/conv1/kernel:0', 'l2_diff': 1e-6}]}
    step = {'production_restore_exact': True, 'semantic_restore_exact': True, 'semantic': [semantic(), semantic()],
            'gradient_repeat': grad, 'production_losses': [[1.0, 2.0]]}
    fwd = {'tensor': 'D/D3/conv1/leaky_relu:0', 'type': 'LeakyRelu', 'gradient': False, 'forward': True,
           'data_movement': False, 'chunk_local_root': False, 'l2_diff': 1e-6, 'l2_ref': 1.0, 'finite': True}
    return {'iteration': t, 'forward_state_unchanged': True, 'state_finite': True, 'forward': {'varying': [fwd]},
            'pair_rows': [t, t + 1], 'input_sha256': {'rgb': 'r%d' % t}, 'state_sha256_after': 's%d' % t,
            'steps': {s: copy.deepcopy(step) for s in g.STEPS}}


def replay():
    conn = {k: {'present_equals_reachable': True, 'all_present_finite_shape': True, 'present': 72, 'reachable': 72,
                'variables': 72} for k in ('L_D', 'L_G') + g.STEPS}
    app = {k: {'apply_adam_ops': 72, 'present_gradients': 72, 'targets_equal': True,
               'apply_inputs_are_step_gradients': True, 'random_ops_on_path': [], 'random_ops_on_bn_path': []}
           for k in '123'}
    return {'summary': {'static': {'construction': list(g.QUALIFIED_CONSTRUCTION), 'gradient_connectivity': conn,
                                   'optimizer_application': app},
                        'graph_operations_before_session': 11828, 'graph_operations_after_session': 11828,
                        'qualification_seed': 60904, 'initial_state_sha256': g.INITIAL_STATE_SHA256, 'iterations': 3,
                        'firewall': {'denied': [], 'runtime_writes': []}},
            'iterations': [iteration(t) for t in range(3)]}


def case():
    trace = {'partitions': copy.deepcopy(PARTS), 'first_trace_place': {'x': '/device:GPU:0', 'y': '/device:GPU:0'}}
    return [replay() for _ in range(3)], [copy.deepcopy(trace) for _ in range(2)], copy.deepcopy(QUAL['gates'])


class D12ProceduralGate(unittest.TestCase):
    def setUp(self):
        self._pins = g.CANONICAL_SET_SHA256, g.RETVAL_CANONICAL_SHA256
        g.CANONICAL_SET_SHA256, g.RETVAL_CANONICAL_SHA256 = canon(PARTS)

    def tearDown(self):
        g.CANONICAL_SET_SHA256, g.RETVAL_CANONICAL_SHA256 = self._pins

    def failed(self, mutate):
        r, t, q = case()
        mutate(r, t, q)
        out = g.evaluate(r, t, q)
        return sorted(k for k, v in out['criteria'].items() if not v)

    def test_01_active_only_through_the_approved_amendment(self):
        self.assertTrue(g.ACTIVE)
        self.assertEqual(g.require_approval(ROOT)['acceptance_criteria_sha256'], g.CRITERIA_SHA256)
        users = sorted(p.name for p in list((ROOT / 'tools').glob('*.py')) + list((ROOT / 'methods').glob('**/*.py'))
                       if p.name != 'e04_d12_procedural_gate.py' and 'e04_d12_procedural_gate' in p.read_text())
        self.assertEqual(users, ['e04_production_qualification_evidence.py'])
        draft = subprocess.run(['git', 'show', '57bd1f9:tools/e04_production_qualification_evidence.py'], cwd=ROOT,
                               capture_output=True, text=True, check=True)
        self.assertNotIn('d12', draft.stdout.lower())
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(SystemExit):
                g.require_approval(d)
            with self.assertRaises(SystemExit):
                g.gate(d, [], [], [])

    def test_02_guard_requires_owner_approved_pin(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / g.APPROVED_AMENDMENT
            p.parent.mkdir(parents=True)
            for doc in (dict(DRAFT), dict(DRAFT, status='OWNER_APPROVED'),
                        dict(DRAFT, status='OWNER_APPROVED', approved=True, acceptance_criteria_sha256='0' * 64)):
                p.write_text(json.dumps(doc))
                with self.assertRaises(SystemExit):
                    g.require_approval(d)
            p.write_text(json.dumps(dict(DRAFT, status='OWNER_APPROVED', approved=True)))
            self.assertEqual(g.require_approval(d)['status'], 'OWNER_APPROVED')

    def test_03_draft_pins_the_gate_criteria(self):
        self.assertEqual(DRAFT['status'], 'DRAFT_FOR_OWNER_FINAL_APPROVAL')
        self.assertIs(DRAFT['approved'], False)
        self.assertIs(DRAFT['execution_authorized'], False)
        self.assertIs(DRAFT['activates_gate'], False)
        self.assertIs(DRAFT['creates_scientific_authority'], False)
        self.assertEqual(DRAFT['replacement_acceptance_criteria'], g.CRITERIA)
        self.assertEqual(DRAFT['acceptance_criteria_sha256'], g.CRITERIA_SHA256)
        self.assertEqual(DRAFT['superseded_gate']['gate'], g.SUPERSEDED_GATE)
        self.assertIs(DRAFT['superseded_gate']['retroactively_passed'], False)
        self.assertEqual(DRAFT['integrity_bound']['relative_l2'], g.BOUND)
        self.assertEqual(DRAFT['retained_gates'], list(g.RETAINED_GATES))
        self.assertEqual(DRAFT['performance_criteria'], [])

    def test_04_retained_gates_are_the_original_41(self):
        names = [x['gate'] for x in QUAL['gates']]
        self.assertEqual(QUAL['failed_gates'], [g.SUPERSEDED_GATE])
        self.assertEqual(sorted(set(names) - {g.SUPERSEDED_GATE}), sorted(g.RETAINED_GATES))
        self.assertEqual(len(g.RETAINED_GATES), 41)

    def test_05_valid_non_bitwise_case_passes_and_k_never_counted(self):
        r, t, q = case()
        out = g.evaluate(r, t, q)
        self.assertTrue(out['pass'], out['failures'])
        self.assertEqual(out['superseded_gate'], {'gate': g.SUPERSEDED_GATE, 'recorded_pass': False,
                                                  'counted_as_passed': False})
        self.assertTrue(all(v > 0 for v in out['diagnostics_not_gating']['bitwise_gradient_variables_differing']
                            .values()))                                              # non-bitwise, diagnostic only
        for k in q:
            if k['gate'] == g.SUPERSEDED_GATE:
                k['pass'] = True
        self.assertFalse(g.evaluate(r, t, q)['superseded_gate']['counted_as_passed'])

    def test_06_graph_identity(self):
        self.assertEqual(self.failed(lambda r, t, q: r[0]['summary']['static'].__setitem__(
            'construction', ['0' * 64, 11315])), ['R1_graph_identity'])
        self.assertEqual(self.failed(lambda r, t, q: r[1]['summary'].__setitem__(
            'graph_operations_after_session', 11829)), ['R1_graph_identity'])
        self.assertEqual(self.failed(lambda r, t, q: t[1]['partitions'][0]['nodes'][1].__setitem__('op', 'AddV2')),
                         ['R1_graph_identity'])
        self.assertEqual(self.failed(lambda r, t, q: t[1]['first_trace_place'].__setitem__('y', '/device:CPU:0')),
                         ['R1_graph_identity'])
        self.assertEqual(self.failed(lambda r, t, q: t.pop()), ['R1_graph_identity'])

    def test_07_inputs_and_initial_state(self):
        self.assertEqual(self.failed(lambda r, t, q: r[2]['iterations'][1].__setitem__('pair_rows', [9, 9])),
                         ['R2_deterministic_inputs_and_initial_state'])
        self.assertEqual(self.failed(lambda r, t, q: r[2]['iterations'][1]['input_sha256'].__setitem__('rgb', 'z')),
                         ['R2_deterministic_inputs_and_initial_state'])
        self.assertEqual(self.failed(lambda r, t, q: r[0]['summary'].__setitem__('initial_state_sha256', 'x')),
                         ['R2_deterministic_inputs_and_initial_state'])
        self.assertEqual(self.failed(lambda r, t, q: r.pop()), ['R2_deterministic_inputs_and_initial_state'])

    def test_08_connectivity_and_optimizer(self):
        self.assertEqual(self.failed(lambda r, t, q: r[0]['summary']['static']['gradient_connectivity']['L_G']
                                     .__setitem__('present_equals_reachable', False)), ['R3_gradient_connectivity'])
        self.assertEqual(self.failed(lambda r, t, q: r[0]['summary']['static']['gradient_connectivity']['step2']
                                     .__setitem__('present', 71)), ['R3_gradient_connectivity'])
        self.assertEqual(self.failed(lambda r, t, q: r[0]['summary']['static']['optimizer_application']['3']
                                     .__setitem__('apply_adam_ops', 71)), ['R4_optimizer_application'])
        self.assertEqual(self.failed(lambda r, t, q: r[0]['summary']['static']['optimizer_application']['1']
                                     .__setitem__('random_ops_on_path', ['RandomUniform'])),
                         ['R4_optimizer_application'])
        self.assertEqual(self.failed(lambda r, t, q: r[0]['summary']['static']['optimizer_application'].pop('2')),
                         ['R4_optimizer_application'])

    def test_09_state_transitions_and_finiteness(self):
        for k in ('scope_ok', 'adam_ok', 'beta_powers_ok', 'counters_ok'):
            self.assertEqual(self.failed(lambda r, t, q: r[1]['iterations'][2]['steps']['step3']['semantic'][1]
                                         .__setitem__(k, False)), ['R5_state_transitions'])
        self.assertEqual(self.failed(lambda r, t, q: r[1]['iterations'][0]['steps']['step1']
                                     .__setitem__('semantic_restore_exact', False)), ['R5_state_transitions'])
        self.assertEqual(self.failed(lambda r, t, q: r[1]['iterations'][0]
                                     .__setitem__('forward_state_unchanged', False)), ['R5_state_transitions'])
        self.assertEqual(self.failed(lambda r, t, q: r[1]['iterations'][0]['steps'].pop('step2')),
                         ['R5_state_transitions'])
        self.assertEqual(self.failed(lambda r, t, q: r[1]['iterations'][0].__setitem__('state_finite', False)),
                         ['R6_finite'])
        self.assertEqual(self.failed(lambda r, t, q: r[1]['iterations'][0]['steps']['step1']['semantic'][0]
                                     .__setitem__('finite', False)), ['R6_finite'])

    def test_10_forward_and_gradient_integrity(self):
        self.assertEqual(self.failed(lambda r, t, q: r[0]['iterations'][0]['forward']['varying'][0]
                                     .__setitem__('l2_diff', 2e-3)), ['R7_forward_integrity'])
        self.assertEqual(self.failed(lambda r, t, q: r[0]['iterations'][0]['forward']['varying'].append(dict(
            r[0]['iterations'][0]['forward']['varying'][0], tensor='Placeholder:0', type='Placeholder',
            data_movement=True))), ['R7_forward_integrity'])
        self.assertEqual(self.failed(lambda r, t, q: r[0]['iterations'][0]['forward']['varying'].append(dict(
            r[0]['iterations'][0]['forward']['varying'][0], tensor='concat:0', type='ConcatV2', data_movement=True,
            chunk_local_root=True))), ['R7_forward_integrity'])
        self.assertEqual(self.failed(lambda r, t, q: r[0]['iterations'][0]['forward']['varying'][0]
                                     .__setitem__('finite', False)), ['R6_finite', 'R7_forward_integrity'])
        self.assertEqual(self.failed(lambda r, t, q: r[0]['iterations'][1]['steps']['step2']['gradient_repeat'][
            'differing_variables'].append({'variable': 'G/a/kernel:0', 'l2_diff': 0.1})), ['R8_gradient_integrity'])
        self.assertEqual(self.failed(lambda r, t, q: r[0]['iterations'][1]['steps']['step2']['gradient_repeat'][
            'differing_variables'].append({'variable': 'G/a/bias:0', 'l2_diff': 5.0})), [])   # BN-cancelled bias

    def test_11_retained_gates_and_firewall(self):
        self.assertEqual(self.failed(lambda r, t, q: q[0].__setitem__('pass', False)), ['R9_retained_gates'])
        self.assertEqual(self.failed(lambda r, t, q: q.pop(5)), ['R9_retained_gates'])
        self.assertEqual(self.failed(lambda r, t, q: q.append(dict(q[0]))), ['R9_retained_gates'])
        self.assertEqual(self.failed(lambda r, t, q: q.append({'gate': 'K.new_performance_gate', 'pass': True})),
                         ['R9_retained_gates'])
        self.assertEqual(self.failed(lambda r, t, q: r[2]['summary']['firewall']['denied'].append('TEST/x.png')),
                         ['R10_firewall'])


if __name__ == '__main__':
    unittest.main()
