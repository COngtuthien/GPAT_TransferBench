"""Focused E03 runtime/controller tests; no benchmark pixels or model execution."""
import argparse
import ast
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('e03_worker',ROOT/'methods/stdn/runtime_qualification.py')
worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
spec=importlib.util.spec_from_file_location('e03_preflight',ROOT/'tools/m6d2b_e03_runtime_preflight.py')
preflight=importlib.util.module_from_spec(spec);spec.loader.exec_module(preflight)
EVIDENCE=ROOT/'outputs/audit/M6D2B_E03_RUNTIME_QUALIFICATION.json'
REPORT=json.loads(EVIDENCE.read_text())['qualification'] if EVIDENCE.exists() else None


class RuntimeContract(unittest.TestCase):
    def test_frozen_contract_and_source(self):
        c,h=worker.verify_contract(ROOT,ROOT/'third_party/source_cache/stdn')
        self.assertEqual(c['fidelity_class'],'FAITHFUL_OFFICIAL')
        self.assertEqual(c['optimizer']['lr_decay'],{'type':'exponential','rate':0.9,'every_steps':20000,'staircase':True})
        self.assertIn('train.py',h)

    def test_variable_oracle(self):
        v=worker.expected_variables()
        self.assertEqual(len(v),170)
        self.assertEqual(sum(k.startswith('STDN/') for k in v),98)
        self.assertEqual(v['STDN/up2/weights:0'],[3,3,64,128])
        self.assertEqual(v['STDN/conv10/weights:0'],[3,3,192,64])
        self.assertEqual(v['Disc/d3/conv9/weights:0'],[3,3,96,1])

    def test_python38_containment(self):
        self.assertTrue(worker.under(ROOT/'methods/stdn',ROOT))
        self.assertFalse(worker.under(ROOT.parent/(ROOT.name+'_other'),ROOT))

    def test_containment_symlink_escape(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'inside').mkdir();(p/'inside/link').symlink_to(p)
            self.assertFalse(worker.under(p/'inside/link/outside',p/'inside'))

    def test_firewall(self):
        f=worker.Firewall(ROOT,ROOT/'third_party/source_cache/stdn')
        for p in [ROOT/'manifests/split_v1.parquet',ROOT/'data/processed/image.png',ROOT/'third_party/source_cache/stdn/data/pretrain/checkpoint',Path('/home/student20261/workdir/GPAT_TransferBench_runtime/data/face.png')]:
            with self.assertRaises(RuntimeError): f('open',(str(p),'r',0))

    def test_firewall_allows_source(self):
        f=worker.Firewall(ROOT,ROOT/'third_party/source_cache/stdn')
        f('open',(str(ROOT/'third_party/source_cache/stdn/model/model.py'),'r',0))
        self.assertEqual(f.denied,[])

    def test_precision_controls(self):
        for k in ['NVIDIA_TF32_OVERRIDE','TF_ENABLE_AUTO_MIXED_PRECISION','TF_ENABLE_CUBLAS_TENSOR_OP_MATH_FP32','TF_ENABLE_CUDNN_TENSOR_OP_MATH_FP32']:
            self.assertEqual(worker.PRECISION[k],'0')
        self.assertEqual(worker.PRECISION['TF_XLA_FLAGS'],'--tf_xla_auto_jit=0')

    def test_original_schedule_checkpoint_and_loss(self):
        text=(ROOT/'third_party/source_cache/stdn/train.py').read_text()
        for fragment in ['if step%config.G_D_RATIO ==0:', 'sess.run(losses+[g_op, d_op, fig])','sess.run(losses+[g_op, fig])',"saver.save(sess, config.LOG_DIR+'/ckpt', global_step=epoch+1)",'g_loss = esr_loss*50 + gan_loss + reg_loss','pixel_loss*0.0','pixel_loss*0.1']:
            self.assertIn(fragment,text)

    def test_worker_no_checkpoint_writer(self):
        tree=ast.parse((ROOT/'methods/stdn/runtime_qualification.py').read_text())
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
        self.assertFalse(any(isinstance(n.func,ast.Attribute) and n.func.attr in ['save','restore','imread','imwrite','load_model'] for n in calls))

    def test_seed_enforcement(self):
        text=(ROOT/'methods/stdn/runtime_qualification.py').read_text()
        self.assertIn('assert args.seed in [42,1337,2026]',text)
        self.assertIn('tf.set_random_seed(args.seed)',text)
        self.assertIn("os.environ.get('PYTHONHASHSEED')==str(args.seed)",text)

    def test_topology_control_permutations(self):
        a={'name':'op','controls':['b','a'],'inputs':['x','y']}
        b=dict(a,controls=['a','b'])
        self.assertEqual(preflight.normalize_topology([a]),preflight.normalize_topology([b]))

    def test_topology_preserves_data_operand_order(self):
        a={'name':'op','controls':['a','b'],'inputs':['x','y']}
        b=dict(a,inputs=['y','x'])
        self.assertNotEqual(preflight.normalize_topology([a]),preflight.normalize_topology([b]))


class ExecutedEvidence(unittest.TestCase):
    def setUp(self):
        if REPORT is None:
            self.skipTest('Runtime evidence has not been captured; pass --qualification-json')

    def test_stages_and_graph(self):
        self.assertIsNotNone(REPORT,'Executed report required for final runtime evidence tests')
        for run in REPORT['repeat_runs']:
            self.assertTrue(run['A']['contrib_unchanged'])
            self.assertEqual(run['B']['graph_ops_created'],0)
            self.assertEqual(run['C']['trainable_variables'],worker.expected_variables())
            self.assertEqual(run['C']['adam_ops'],170)
            self.assertEqual(run['C']['weight_ema_count'],170)
            self.assertEqual(run['C']['loss_ema_count'],2)

    def test_forward_step_optimizer_firewall(self):
        for run in REPORT['repeat_runs']:
            self.assertTrue(run['D']['finite'])
            self.assertGreater(run['D']['gpu_executed_nodes'],0)
            self.assertTrue(run['E']['all_G_and_D_gradients_finite'])
            self.assertEqual(run['E']['optimizer_applications'],1)
            self.assertEqual(run['E']['global_step_after'],1)
            self.assertFalse(run['E']['discriminator_update_executed'])
            self.assertEqual(run['firewall']['denied'],[])
            for k,v in [('beta1',0.9),('beta2',0.999),('epsilon',1e-8),('initial_learning_rate',6e-5)]:
                self.assertAlmostEqual(run['optimizer_validation'][k],v,places=7)

    def test_repeatability(self):
        self.assertTrue(REPORT['repeatability']['structure_identical'])
        self.assertTrue(REPORT['repeatability']['initial_weights_identical'])
        self.assertTrue(REPORT['repeatability']['diagnostics_within_tolerance'])

    def test_captured_topology_equivalence(self):
        preflight.validate_topology(REPORT)

    def test_d_optimizer_executed_and_repeated(self):
        preflight.validate_d_optimizer(REPORT)

    def test_missing_d_execution_rejected(self):
        q=copy.deepcopy(REPORT);q.pop('d_optimizer_qualification')
        with self.assertRaisesRegex(RuntimeError,'D optimizer path'):
            preflight.validate_d_optimizer(q)

    def test_generator_weight_change_rejected(self):
        q=copy.deepcopy(REPORT);p=q['d_optimizer_qualification']['probes'][0]
        r=json.loads(p['stdout'].split('E03_RESULT=',1)[1]);r['D_optimizer']['generator_after_sha256']='changed'
        p['stdout']='E03_RESULT='+json.dumps(r)
        with self.assertRaisesRegex(RuntimeError,'unchanged G trainable'):
            preflight.validate_d_optimizer(q)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--qualification-json');args,rest=ap.parse_known_args()
    if args.qualification_json:
        REPORT=json.loads(Path(args.qualification_json).read_text())
        suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(c) for c in [RuntimeContract,ExecutedEvidence]])
    else:
        suite=unittest.defaultTestLoader.loadTestsFromTestCase(RuntimeContract)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(not result.wasSuccessful())
