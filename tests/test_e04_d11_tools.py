"""E04 D11 investigation tooling (tools/e04_d11_analysis.py): the placement-scope check fails closed on any change
outside the 32 ScatterNd targets, on a target left on GPU, or on a non-target device change; the canonical
partition-graph hash ignores op names (classic-optimizer CSE representatives) but not values, op types, devices or
inputs. The D11 protocol is fixed, approves nothing and reuses the D10 design file byte-for-byte. Synthetic, stdlib.
"""
import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))

import e04_d11_analysis as an  # noqa: E402

PROTOCOL = json.loads((ROOT / 'outputs/audit/E04_D11_INVESTIGATION_PROTOCOL.json').read_text())
ORIGINS = ['gradients_1/warp_B/GatherNd_grad/ScatterNd', 'gradients_7/warp_B/GatherNd_grad/ScatterNd']
CONSTRUCTION = ['61f1ed58', 11315]
CPU, GPU = '/job:localhost/replica:0/task:0/device:CPU:0', '/job:localhost/replica:0/task:0/device:GPU:0'


def proc(arm, process='P1'):
    target_dev = CPU if arm == 'cpu_scatternd' else GPU
    place = {o: target_dev for o in ORIGINS}
    place.update({'G/conv/Conv2D': GPU, 'warp_B/GatherNd': CPU, 'gradients_1/AddN_5': GPU})
    changed = [{'node': o, 'device_before': '/device:GPU:0', 'device_after': '/device:CPU:0', 'fields': ['device'],
                'only_device_differs': True} for o in ORIGINS] if arm == 'cpu_scatternd' else []
    refs = [{'op': 'gradients_1/AddN_5', 'type': 'AddN', 'requested_device': '/device:GPU:0',
             'class_before': ['loc:@' + ORIGINS[0]], 'class_after': []}]
    if arm == 'cpu_scatternd':
        changed.insert(0, {'node': 'gradients_1/AddN_5', 'device_before': '/device:GPU:0',
                           'device_after': '/device:GPU:0', 'fields': ['attr:_class'], 'only_device_differs': False})
    return {'extra': {
        'arm': arm, 'process': process,
        'placement': {'target_names': list(ORIGINS),
                      'targets': [{'op': o, 'forward_requested_device': '/cpu:0'.upper()} for o in ORIGINS],
                      'node_diff': {'changed': changed, 'added': [], 'removed': []},
                      'colocation_references': refs,
                      'released_colocation': ['gradients_1/AddN_5'] if arm == 'cpu_scatternd' else [],
                      'construction_before': list(CONSTRUCTION)},
        'trace': {'first_trace_place': place, 'target_executed_device': {o: [target_dev, target_dev] for o in ORIGINS}}}}


def scope(arm, p, controls=None):
    return an.scope_checks({'P1': p}, arm, ORIGINS, CONSTRUCTION, controls or {})[0]


class D11Tools(unittest.TestCase):
    def test_01_scope_passes_for_both_arms(self):
        o, c = proc('original'), proc('cpu_scatternd')
        self.assertTrue(scope('original', o))
        self.assertTrue(scope('cpu_scatternd', c, {'original/P1': o}))

    def test_02_scope_fails_on_extra_changed_node(self):
        c = proc('cpu_scatternd')
        c['extra']['placement']['node_diff']['changed'].append(
            {'node': 'gradients_1/G/x/ResizeBilinear_grad/ResizeBilinearGrad', 'device_before': '/device:GPU:0',
             'device_after': '/device:CPU:0', 'fields': ['device'], 'only_device_differs': True})
        self.assertFalse(scope('cpu_scatternd', c))

    def test_02b_colocation_release_is_scoped(self):
        c = proc('cpu_scatternd')                        # the released AddN must change only its _class
        c['extra']['placement']['node_diff']['changed'][0]['fields'] = ['device', 'attr:_class']
        self.assertFalse(scope('cpu_scatternd', c))
        c = proc('cpu_scatternd')                        # still colocated with a target after the release
        c['extra']['placement']['colocation_references'][0]['class_after'] = ['loc:@' + ORIGINS[0]]
        self.assertFalse(scope('cpu_scatternd', c))
        c = proc('cpu_scatternd')                        # a release not backed by a recorded reference
        c['extra']['placement']['released_colocation'] = []
        self.assertFalse(scope('cpu_scatternd', c))
        o, c = proc('original'), proc('cpu_scatternd')   # the released AddN must stay on the control's device
        c['extra']['trace']['first_trace_place']['gradients_1/AddN_5'] = CPU
        self.assertFalse(scope('cpu_scatternd', c, {'original/P1': o}))

    def test_03_scope_fails_on_non_device_change_or_added_node(self):
        c = proc('cpu_scatternd')
        c['extra']['placement']['node_diff']['changed'][1]['fields'] = ['device', 'attr:T']   # a target
        self.assertFalse(scope('cpu_scatternd', c))
        c = proc('cpu_scatternd')
        c['extra']['placement']['node_diff']['added'] = ['new_op']
        self.assertFalse(scope('cpu_scatternd', c))
        o = proc('original')
        o['extra']['placement']['node_diff']['changed'] = [{'node': 'x', 'fields': ['device'],
                                                            'only_device_differs': True}]
        self.assertFalse(scope('original', o))

    def test_04_scope_fails_when_target_not_enforced_on_cpu(self):
        c = proc('cpu_scatternd')
        c['extra']['trace']['target_executed_device'][ORIGINS[0]] = [CPU, GPU]
        self.assertFalse(scope('cpu_scatternd', c))

    def test_05_scope_fails_on_non_target_device_difference_against_control(self):
        o, c = proc('original'), proc('cpu_scatternd')
        c['extra']['trace']['first_trace_place']['G/conv/Conv2D'] = CPU
        self.assertFalse(scope('cpu_scatternd', c, {'original/P1': o}))

    def test_06_scope_fails_on_wrong_targets_or_construction(self):
        c = proc('cpu_scatternd')
        c['extra']['placement']['target_names'] = ORIGINS[:1]
        self.assertFalse(scope('cpu_scatternd', c))
        c = proc('cpu_scatternd')
        c['extra']['placement']['construction_before'] = ['other', 11315]
        self.assertFalse(scope('cpu_scatternd', c))

    def test_07_canonical_hash_ignores_names_only(self):
        def part(name, value='v1', op='Mul', device=GPU):
            return [{'nodes': [
                {'name': 'c', 'op': 'Const', 'device': GPU, 'inputs': [], 'attrs': {'dtype': 'f'}, 'value': [value]},
                {'name': 'arg', 'op': '_Arg', 'device': GPU, 'inputs': [], 'attrs': {'index': '0'}},
                {'name': name, 'op': op, 'device': device, 'inputs': ['arg', 'c'], 'attrs': {'T': 'f'}},
                {'name': 'ret', 'op': '_Retval', 'device': GPU, 'inputs': [name], 'attrs': {'index': '0'}}]}]
        h = lambda p: an.canonical(p)[0]['ret']  # noqa: E731
        self.assertEqual(h(part('mul_a')), h(part('mul_b')))
        self.assertNotEqual(h(part('mul_a')), h(part('mul_a', value='v2')))
        self.assertNotEqual(h(part('mul_a')), h(part('mul_a', op='Add')))
        self.assertNotEqual(h(part('mul_a')), h(part('mul_a', device=CPU)))

    def test_08_canonical_resolves_transfers(self):
        def parts(src_value):
            return [{'nodes': [
                {'name': 'c', 'op': 'Const', 'device': GPU, 'inputs': [], 'attrs': {}, 'value': [src_value]},
                {'name': 's', 'op': '_Send', 'device': GPU, 'inputs': ['c'], 'attrs': {'tensor_name': 'e1'},
                 'transfer': {'tensor_name': 'edge_1'}}]},
                {'nodes': [{'name': 'r', 'op': '_Recv', 'device': CPU, 'inputs': [], 'attrs': {'tensor_name': 'e1'},
                            'transfer': {'tensor_name': 'edge_1'}}]}]
        self.assertNotEqual(an.canonical(parts('a'))[0]['r'], an.canonical(parts('b'))[0]['r'])

    def test_09_protocol_fixed_and_reuses_d10_design(self):
        self.assertEqual(PROTOCOL['status'], 'FIXED_BEFORE_EXECUTION')
        self.assertFalse(PROTOCOL['approves_d11'])
        self.assertFalse(PROTOCOL['creates_scientific_authority'])
        d10 = hashlib.sha256((ROOT / 'outputs/audit/E04_D10_INVESTIGATION_PROTOCOL.json').read_bytes()).hexdigest()
        self.assertIn(d10, PROTOCOL['inputs_fixed']['design_file'])
        self.assertEqual(set(PROTOCOL['decision_rule']) - {'no_post_hoc_change'},
                         {'E04_D11_UNSUPPORTED', 'E04_D11_INCONCLUSIVE', 'E04_D11_READY_FOR_OWNER_APPROVAL'})
        self.assertEqual(an.SHORT_ITERATIONS, PROTOCOL['stages']['S_short']['iterations'])
        self.assertEqual(an.C4_THRESHOLD, json.loads((ROOT / 'outputs/audit/E04_D10_INVESTIGATION_PROTOCOL.json')
                                                     .read_text())['criteria']['C4_fixed_state_magnitude']['threshold'])


if __name__ == '__main__':
    unittest.main()
