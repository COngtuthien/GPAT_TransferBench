"""E04 production I/O (owner resolution D1-D9): framework-free checks of the production constants, RNG streams,
checkpoint semantics, CLI refusals and the D9 environment tolerance. Stdlib + numpy (+ the static E04 adapter for the
checkpoint-rule cross-check); no TensorFlow, no benchmark data.
"""
import ast
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from methods.physics_std import production_io as pio  # noqa: E402

VECTORS = json.loads((ROOT / 'outputs/audit/E04_RNG_TEST_VECTORS.json').read_text())


class E04ProductionIO(unittest.TestCase):
    def test_01_contract_and_ruling(self):
        cfg = pio.verify_contract()
        self.assertEqual((cfg['method_id'], cfg['fidelity_class']), ('E04', 'CONTROLLED_ADAPTATION'))
        ruling = json.loads((ROOT / pio.RULING_YAML[0]).read_text())
        self.assertEqual(ruling['approved_draft']['commit'], '4112d164861bffa9adbe042f97833bc8ebaa697a')
        self.assertFalse(ruling['scientific_seed_execution_authorized'])

    def test_02_seed_table_and_modes(self):
        table = {42: (679869707, 320493565), 1337: (1349475396, 35554372), 2026: (692890966, 1053992267)}
        for s, (p, h) in table.items():
            self.assertEqual((pio.derive_seed('PAIR_ORDER', s), pio.derive_seed('HARD_SAMPLE', s)), (p, h))
            self.assertEqual((VECTORS['seeds'][str(s)]['pair_seed'], VECTORS['seeds'][str(s)]['hard_seed']), (p, h))
        pio.validate_mode_seed(pio.QUALIFICATION, 60904)
        for bad in ((pio.SCIENTIFIC, 60904), (pio.QUALIFICATION, 42), (pio.SCIENTIFIC, 314159)):
            with self.assertRaises(pio.E04Stop):
                pio.validate_mode_seed(*bad)

    def test_03_full_streams_equal_approved_vectors(self):
        for s in (42, 1337, 2026):
            h = pio.stream_hashes(s)
            e = VECTORS['seeds'][str(s)]
            self.assertEqual(h['pair_sha256_int32_le'], e['pair_stream']['sha256_int32_le'])
            self.assertEqual(h['hard_sha256'], e['hard_stream']['sha256_float32_scales_then_keep_per_iteration'])
            self.assertEqual(h['pair_passes_generated'], 68)

    def test_04_stream_semantics(self):
        p, h = pio.PairStream(42), pio.HardStream(42)
        rows, passes = p.next_rows()
        self.assertEqual(rows, VECTORS['seeds']['42']['pair_stream']['iteration0_rows'])
        self.assertEqual(passes, [0, 0, 0, 0])
        s, k = h.next()
        first = VECTORS['seeds']['42']['hard_stream']['first_iterations'][0]
        self.assertEqual([[float(x) for x in r.ravel()] for r in s], first['hard_scales'])
        self.assertEqual([float(x) for x in k.ravel()], first['hard_keep'])
        self.assertIsNot(p.rng, h.rng)                       # separate RNG objects (D7)
        for _ in range(2208):
            p.next_rows()
        rows, passes = p.next_rows()                         # draws 8836..8839 straddle pass 0 -> 1
        self.assertEqual(passes, [0, 0, 1, 1])

    def test_05_checkpoint_semantics_equal_adapter(self):
        from methods.physics_std.adapter import PhysicsSTDAdapter
        a = PhysicsSTDAdapter()
        self.assertEqual(pio.scientific_checkpoints(), list(range(15000, 150001, 15000)))
        for it in pio.scientific_checkpoints():
            kind, selected, reason = pio.checkpoint_kind(it)
            ev = a.validate_checkpoint_event(42, iteration=it, terminal=selected, selected=selected)
            self.assertEqual((ev['selected'], ev['selection_reason']), (selected, reason))
            self.assertEqual(kind, 'selected' if it == 150000 else 'periodic')
        self.assertEqual(pio.checkpoint_prefix(150000), 'iteration_150000')

    def test_06_learning_rates(self):
        for t, g in ((0, 5e-5), (44999, 5e-5), (45000, 5e-6), (90000, 5e-7), (135000, 5e-8), (149999, 5e-8)):
            G, D = pio.expected_learning_rates(t)
            self.assertAlmostEqual(G, g, delta=g * 1e-12)
            self.assertEqual(D, G * .5)

    def test_07_cli_refusals(self):
        self.assertEqual(pio.refused_arguments(['--seed', '42', '--execution-config', 'x']), [])
        for flag in ('--iterations=10', '--resume', '--qualification-seed', '--lr', '--val', '--checkpoint-every',
                     '--anything-else'):
            self.assertTrue(pio.refused_arguments([flag]), flag)
        sys.path.insert(0, str(ROOT / 'tools'))
        import run_e04
        with self.assertRaises(SystemExit) as cm:
            run_e04.parse(['--seed', '42', '--execution-config', 'x', '--iterations', '5'])
        self.assertEqual(cm.exception.code, 2)
        args = run_e04.parse(['--seed', '60904', '--execution-config', 'x'])
        with self.assertRaises(SystemExit):
            run_e04.preconditions(args, environ={}, dirty=False)

    def test_08_d9_environment_tolerance(self):
        lock = json.loads((ROOT / pio.LOCK[0]).read_text())
        record = json.loads((ROOT / pio.RUNTIME_RECORD[0]).read_text())
        live = {'python': lock['python'], 'python_executable_sha256': lock['python_executable_sha256'],
                'environment_path': lock['environment_path'], 'package_versions': dict(lock['package_versions']),
                'native_runtime': dict(lock['native_runtime']), 'precision_env': dict(lock['precision_env']),
                'session_config': record['session_config'],
                'gpu': 'NVIDIA GeForce RTX 3090, 580.178.04, GPU-b722cd9d-c9fa-223c-915e-cc09d9b862b7'}
        self.assertEqual(pio.environment_differences(lock, record, live)[0], [])
        for change in ({'gpu': 'NVIDIA GeForce RTX 3090, 570.00, GPU-b722cd9d-c9fa-223c-915e-cc09d9b862b7'},
                       {'gpu': 'NVIDIA GeForce RTX 3090, 580.178.04, GPU-other'},
                       {'native_runtime': {'cudaRuntimeGetVersion': 12000, 'cuda_query_rc': 0, 'cudnnGetVersion': 8907}},
                       {'package_versions': dict(lock['package_versions'], scipy='1.11.0')},
                       {'session_config': 'x'}, {'precision_env': {}}):
            self.assertTrue(pio.environment_differences(lock, record, dict(live, **change))[0], change)

    def test_09_no_restore_path_in_controller(self):
        tree = ast.parse((ROOT / 'methods/physics_std/production_runner.py').read_text())
        attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        self.assertNotIn('restore', attrs)
        self.assertNotIn('latest_checkpoint', attrs)


if __name__ == '__main__':
    unittest.main()
