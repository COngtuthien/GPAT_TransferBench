"""E04 D12 investigation tooling: the integrity classification fails closed on varying fed inputs, data-movement roots,
non-finite or out-of-bound forward tensors; the BN-cancelled set is the BN-sibling set; the numpy Adam reference and
tolerance accept a float32 ApplyAdam step and reject wrong semantics (missing bias correction, wrong learning rate,
stale moments); the session-proxy comparison records magnitudes. The protocol is fixed and approves nothing. numpy.
"""
import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))

import e04_d12_analysis as an  # noqa: E402
import e04_d12_production_replay as rp  # noqa: E402
import e04_d12_confound_probe as cp  # noqa: E402

PROTOCOL = json.loads((ROOT / 'outputs/audit/E04_D12_INVESTIGATION_PROTOCOL.json').read_text())


def row(**k):
    base = {'tensor': 'D/D3/conv1/LeakyRelu:0', 'type': 'LeakyRelu', 'gradient': False, 'forward': True,
            'data_movement': False, 'chunk_local_root': False, 'l2_diff': 1e-6, 'l2_ref': 1.0, 'finite': True}
    base.update(k)
    return base


def tf_adam32(var, m, v, g, lr, b1p, b2p):
    """float32 emulation of the TF ApplyAdam kernel."""
    f = np.float32
    lr_t = f(lr) * np.sqrt(f(1) - f(b2p)) / (f(1) - f(b1p))
    m1 = m + (g - m) * (f(1) - f(.9))
    v1 = v + (g * g - v) * (f(1) - f(.999))
    return (var - lr_t * m1 / (np.sqrt(v1) + f(1e-8))).astype(f), m1.astype(f), v1.astype(f)


class D12Tools(unittest.TestCase):
    def test_01_forward_integrity_classification(self):
        self.assertEqual(an.forward_violations([row()]), [])
        self.assertTrue(an.forward_violations([row(type='Placeholder', data_movement=True)]))
        self.assertTrue(an.forward_violations([row(type='ConcatV2', data_movement=True, chunk_local_root=True)]))
        self.assertTrue(an.forward_violations([row(l2_diff=2e-3)]))
        self.assertTrue(an.forward_violations([row(finite=False)]))
        self.assertTrue(an.forward_violations([row(l2_ref=0.0)]))
        self.assertTrue(an.forward_violations([row(l2_ref=None, l2_diff=None)]))
        self.assertEqual(an.forward_violations([row(gradient=True, l2_diff=5.0)]), [])   # gradients: not forward
        self.assertEqual(an.forward_violations([row(type='ConcatV2', data_movement=True)]), [])  # input varied too

    def test_02_cancelled_set_is_bn_sibling(self):
        names = ['G/a/bias:0', 'G/a/BN/beta:0', 'G/a/kernel:0', 'G/out/bias:0', 'D/D3/conv1/bias:0',
                 'D/D3/conv1/BN/beta:0']
        self.assertEqual(an.cancelled(names), {'G/a/bias:0', 'D/D3/conv1/bias:0'})

    def test_03_adam_reference_accepts_float32_kernel(self):
        rng = np.random.RandomState(0)
        var, g = rng.randn(1000).astype(np.float32), (rng.randn(1000) * 1e-3).astype(np.float32)
        m, v = (rng.randn(1000) * 1e-4).astype(np.float32), (rng.rand(1000) * 1e-6).astype(np.float32)
        lr, b1p, b2p = np.float32(5e-5), np.float32(.9 ** 3), np.float32(.999 ** 3)
        a_var, a_m, a_v = tf_adam32(var, m, v, g, lr, b1p, b2p)
        e_var, e_m, e_v = rp.adam_expected(var, m, v, g, lr, b1p, b2p)
        self.assertTrue(rp.within(a_var, e_var, var)[0])
        self.assertTrue(rp.within(a_m, e_m, m)[0])
        self.assertTrue(rp.within(a_v, e_v, v)[0])
        z = np.zeros(1000, np.float32)                       # first step from zero moments
        a_var, a_m, a_v = tf_adam32(var, z, z, g, lr, np.float32(.9), np.float32(.999))
        self.assertTrue(rp.within(a_var, rp.adam_expected(var, z, z, g, lr, .9, .999)[0], var)[0])

    def test_04_adam_reference_rejects_wrong_semantics(self):
        rng = np.random.RandomState(1)
        var, g = rng.randn(1000).astype(np.float32), (rng.randn(1000) * 1e-3).astype(np.float32)
        m, v = (rng.randn(1000) * 1e-4).astype(np.float32), (rng.rand(1000) * 1e-6).astype(np.float32)
        lr, b1p, b2p = np.float32(5e-5), np.float32(.9 ** 3), np.float32(.999 ** 3)
        e_var = rp.adam_expected(var, m, v, g, lr, b1p, b2p)[0]
        self.assertFalse(rp.within(tf_adam32(var, m, v, g, lr * 2, b1p, b2p)[0], e_var, var)[0])
        self.assertFalse(rp.within(tf_adam32(var, m, v, g, lr, np.float32(0), np.float32(0))[0], e_var, var)[0])
        self.assertFalse(rp.within(var, e_var, var)[0])                  # no update applied
        self.assertFalse(rp.within(tf_adam32(var, m, v, -g, lr, b1p, b2p)[0], e_var, var)[0])

    def test_05_compare_magnitudes(self):
        a = np.ones(4, np.float32)
        self.assertIsNone(cp.compare(a, a.copy()))
        b = a.copy()
        b[1] = np.nextafter(np.float32(1), np.float32(2))
        r = cp.compare(a, b)
        self.assertEqual((r['differing'], r['max_ulp']), (1, 1))
        self.assertAlmostEqual(r['l2_ref'], 2.0)
        self.assertEqual(cp.compare(np.array([1, 2]), np.array([1, 3]))['differing'], 1)

    def test_06_protocol_fixed_and_inert(self):
        self.assertEqual(PROTOCOL['status'], 'FIXED_BEFORE_EXECUTION')
        self.assertFalse(PROTOCOL['approves_d12'])
        self.assertFalse(PROTOCOL['activates_any_gate'])
        self.assertFalse(PROTOCOL['creates_scientific_authority'])
        self.assertEqual(set(PROTOCOL['decision_rule']) - {'descriptive_only', 'no_post_hoc_change'},
                         {'E04_D12_INCONCLUSIVE', 'E04_D12_BLOCKED_BY_SEMANTIC_DEFECT', 'NO_SEMANTIC_DEFECT_FOUND'})
        self.assertEqual(an.BOUND, 1e-3)
        self.assertEqual(set(an.FACTORS), set(PROTOCOL['stages']['C_confound']['factors']))
        self.assertEqual(len(an.C_PROCESSES), 4)
        self.assertEqual(len(an.P_PROCESSES), PROTOCOL['stages']['P_production_path']['processes'])
        self.assertEqual(rp.ITERATIONS, PROTOCOL['stages']['P_production_path']['iterations'])
        self.assertEqual(sorted(cp.DATA_MOVEMENT), sorted(x.strip() for x in
                                                          PROTOCOL['definitions']['data_movement_ops'].split(',')))


if __name__ == '__main__':
    unittest.main()
