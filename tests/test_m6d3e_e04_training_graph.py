"""Focused static and measured-evidence gates; never launches extra optimizers."""
import copy
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = None


def load_file(name):
    path = ROOT/'methods/physics_std'/(name+'.py')
    spec = importlib.util.spec_from_file_location('m6d3e_'+name,path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ContractTests(unittest.TestCase):
    def test_committed_contract_hash(self):
        c = load_file('training_contract').load_contract(ROOT)
        self.assertEqual(c['status'],'E04_TRAINING_GRAPH_CONTRACT_RESOLVED')

    def test_changed_contract_rejected(self):
        m = load_file('training_contract')
        with tempfile.TemporaryDirectory(prefix='m6d3e_contract_') as directory:
            path = Path(directory)/m.OVERLAY
            path.parent.mkdir(parents=True)
            path.write_text('{}')
            with self.assertRaisesRegex(ValueError,'authority changed'):
                m.load_contract(directory)

    def test_test_manifest_firewall(self):
        f = load_file('training_runtime').forbidden
        self.assertTrue(f('/home/example/GPAT_TransferBench/manifests/split_v1.parquet'))
        self.assertTrue(f('/home/example/GPAT_TransferBench_runtime/data'))
        self.assertTrue(f('/home/example/GPAT_TransferBench/data/processed/faces_256'))

    def test_checkpoint_and_bank_firewall(self):
        f = load_file('training_runtime').forbidden
        self.assertTrue(f('/tmp/forbidden_fixture.ckpt'))
        self.assertTrue(f('/home/example/GPAT_TransferBench_runtime/banks'))
        self.assertTrue(f('/home/example/GPAT_TransferBench/runs/m6/E04'))

    def test_code_and_audit_paths_allowed(self):
        f = load_file('training_runtime').forbidden
        self.assertFalse(f(str(ROOT/'methods/physics_std/network.py')))
        self.assertFalse(f('/tmp/m6d3e_qualification.json'))

    def test_no_data_or_image_loader_in_graph(self):
        import ast
        for name in ['network','losses','trace_transfer','training_graph']:
            tree = ast.parse((ROOT/'methods/physics_std'/(name+'.py')).read_text())
            calls = [n.func.attr if isinstance(n.func,ast.Attribute) else n.func.id
                     for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,(ast.Attribute,ast.Name))]
            self.assertFalse(set(calls)&{'open','read_bytes','read_text','imread','decode_jpeg','decode_png','listdir','scandir','rglob','walk'})


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        if EVIDENCE is None:
            self.skipTest('Run through synthetic worker; evidence tests never launch a graph themselves')
        self.e = EVIDENCE

    def test_synthetic_input(self):
        self.assertEqual(self.e['fixture_spec']['RGB'],[8,256,256,3])
        self.assertEqual((self.e['fixture_spec']['live'],self.e['fixture_spec']['spoof']),(4,4))

    def test_frequency_decomposition(self):
        self.assertLessEqual(max(self.e['frequency_oracle_max_errors'].values()),4e-6)
        self.assertEqual(self.e['outputs']['original/frequency_input']['shape'],[8,256,256,9])

    def test_paper_encoder_shapes(self):
        for key,shape in [('F1',[8,128,128,64]),('F2',[8,64,64,96]),('F3',[8,32,32,128])]:
            self.assertEqual(self.e['outputs']['original/'+key]['shape'],shape)

    def test_decoder_layout(self):
        self.assertEqual(self.e['outputs']['original/decoder_raw']['shape'],[8,256,256,13])
        for name,width in [('B',3),('C',3),('T',3),('P',1),('I_P',3)]:
            self.assertEqual(self.e['outputs']['original/'+name]['shape'],[8,256,256,width])

    def test_four_independent_discriminators(self):
        names=self.e['groups']['D']
        sets=[set(n for n in names if n.startswith('D/D%d/'%i)) for i in range(1,5)]
        self.assertTrue(all(sets))
        self.assertEqual(sum(map(len,sets)),len(set.union(*sets)))
        for i,size in enumerate([4,12,32,32],1):
            self.assertEqual(self.e['outputs']['D%d'%i]['shape'],[8,size,size,1])

    def test_depth_shape_and_range(self):
        self.assertEqual(self.e['outputs']['original/depth']['shape'],[8,32,32,1])
        r=self.e['forward']['original/depth']
        self.assertGreaterEqual(r['min'],0); self.assertLessEqual(r['max'],1)

    def test_spoof_depth_zeros(self):
        self.assertTrue(self.e['fixture_spec']['spoof_depth_exact_zero'])
        self.assertTrue(self.e['fixture_spec']['hard_depth_exact_zero'])

    def test_p0_zero(self):
        self.assertTrue(self.e['inpainting']['P0_exact_zero'])
        self.assertEqual(self.e['forward']['P0']['max'],0)

    def test_negative_prior_short_circuit(self):
        self.assertEqual(self.e['inpainting']['negative_prior_op_type'],'Const')
        self.assertEqual(self.e['inpainting']['negative_prior_inputs'],0)
        self.assertEqual(self.e['initial_losses']['P0_negative_prior'],0)

    def test_active_primary_mask(self):
        self.assertGreater(self.e['initial_losses']['P_positive'],0)
        self.assertGreater(self.e['inpainting']['P_primary_gradient']['l2'],0)

    def test_inpainting_participates(self):
        self.assertGreater(self.e['forward']['original/P']['max'],0)
        self.assertGreater(self.e['inpainting']['I_P_adversarial_gradient']['l2'],0)

    def test_weighted_objectives(self):
        self.assertTrue(self.e['weighted_initial']['exact'])
        self.assertTrue(all(s['weighted_recomputation']['exact'] for s in self.e['steps']))

    def test_variable_groups(self):
        g,d=set(self.e['groups']['G']),set(self.e['groups']['D'])
        self.assertFalse(g&d)
        self.assertEqual(g|d,{v['name'] for v in self.e['variables'] if v['trainable']})

    def test_initialization(self):
        self.assertEqual(len(self.e['initialization']),len(self.e['groups']['G'])+len(self.e['groups']['D']))
        for name,r in self.e['initialization'].items():
            if name.endswith('/kernel:0'):
                self.assertEqual((r['distribution'],r['mean'],r['stddev']),('Normal',0.,.02))

    def test_lr_boundaries(self):
        self.assertEqual([r['iteration'] for r in self.e['lr_boundaries']],[0,44999,45000,89999,90000,135000])
        for r in self.e['lr_boundaries']:
            self.assertLess(abs(r['G']/r['expected_G']-1),2e-7)

    def test_half_discriminator_lr(self):
        for r in self.e['lr_boundaries']:
            self.assertEqual(r['D'],r['G']*.5)

    def test_shared_generator_adam(self):
        slots=self.e['optimizer_state']['step_slot_references']
        self.assertEqual(slots[1],slots[3])
        self.assertTrue(self.e['optimizer_state']['shared_G_slots'])

    def test_separate_discriminator_adam(self):
        s=self.e['optimizer_state']
        self.assertFalse(set(s['G_beta_powers'])&set(s['D_beta_powers']))
        self.assertEqual((s['generator_instances'],s['discriminator_instances']),(1,1))

    def test_three_sequential_steps(self):
        self.assertEqual([s['objective'] for s in self.e['steps']],['loss_G_step1','loss_D','loss_G_step3'])
        self.assertEqual(self.e['optimizer_applications'],3)

    def test_global_iteration_once(self):
        self.assertEqual([s['iteration_before'] for s in self.e['steps']],[0,0,0])
        self.assertEqual([s['iteration_after'] for s in self.e['steps']],[0,0,1])
        self.assertEqual(self.e['final_iteration'],1)

    def test_same_lr_index_all_steps(self):
        self.assertEqual(len({s['lr_G'] for s in self.e['steps']}),1)
        self.assertEqual(len({s['lr_D'] for s in self.e['steps']}),1)

    def test_gradients_connected(self):
        self.assertEqual(set(self.e['gradient_inventory']),{'L_depth','L_G','L_P','L_R','L_S','L_H','L_D'})
        for loss,r in self.e['gradient_inventory'].items():
            self.assertTrue(r['connected'],loss)
            self.assertTrue(any(g['l2']>0 for g in r['connected'].values()),loss)

    def test_unrelated_trainables_unchanged(self):
        self.assertTrue(all(s['unrelated_trainables_unchanged'] for s in self.e['steps']))
        self.assertTrue(all(s['changed_trainables'] for s in self.e['steps']))

    def test_native_bn_changes_recorded(self):
        for s in self.e['steps']:
            self.assertEqual(len(s['bn_changes_by_forward_collection']),2)
            self.assertTrue(all(s['bn_changes_by_forward_collection']))
            self.assertTrue(all('/moving_' in n for g in s['bn_changes_by_forward_collection'] for n in g))

    def test_supervision_detached(self):
        self.assertTrue(self.e['stop_gradient_verified'])

    def test_fp32_precision(self):
        self.assertTrue(all(v['dtype'] in ('float32','int32','int64') for v in self.e['variables']))
        self.assertEqual(self.e['precision_env']['NVIDIA_TF32_OVERRIDE'],'0')
        self.assertEqual(self.e['precision_env']['TF_ENABLE_AUTO_MIXED_PRECISION'],'0')

    def test_gpu_convolutions(self):
        self.assertTrue(any('gpu' in (p or '').lower() for p in self.e['convolution_placements'].values()))


def run_evidence_tests(evidence):
    global EVIDENCE
    EVIDENCE=evidence
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(ContractTests),
                             unittest.defaultTestLoader.loadTestsFromTestCase(EvidenceTests)])
    stream=io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    return {'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
            'skipped':len(result.skipped),'transcript':stream.getvalue()}


if __name__=='__main__':
    unittest.main()
