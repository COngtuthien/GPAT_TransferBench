"""Validate real retained GPU evidence and fail-closed qualification guards.

These tests do not pretend CPU/static checks execute CUDA. The two separately
launched runtime_qualification processes produce the CUDA/autograd evidence.
"""
import hashlib
import json
from pathlib import Path
import unittest

from methods.pcgan.runtime_qualification import Firewall, PIN, ROOT
from methods.pcgan import PCGANAdapter
from methods.pcgan.architecture import architecture_mapping


class TestE05Runtime(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runs = [json.loads((ROOT/f'outputs/audit/M6D4A_E05_SYNTHETIC_PROCESS_{i}.json').read_text()) for i in (1,2)]
        cls.patch = json.loads((ROOT/'environments/e05.compatibility_patch_manifest.json').read_text())

    def test_environment_identity(self):
        for r in self.runs:
            self.assertEqual(r['environment_before'],r['environment_after'])
            self.assertIn('/gpat-m6-e05/',r['environment_before']['executable'])
        self.assertEqual(self.runs[0]['environment_before'],self.runs[1]['environment_before'])

    def test_source_identity(self):
        local = PCGANAdapter().validate_source()
        for r in self.runs:
            self.assertEqual(r['source_before'],r['source_after'])
            self.assertEqual(r['source_before']['commit'],PIN)
            self.assertEqual(r['source_before']['files_sha256'],local['files_sha256'])
            self.assertEqual(r['source_before']['repository'],local['repository'])
            self.assertEqual(r['source_before']['worktree_status'],'')

    def test_compatibility_patch_closure(self):
        patches = {p['file']:p for p in self.patch['patches']}
        self.assertEqual(set(patches),{'util/util.py'})
        root = ROOT/'third_party/source_cache/swapping_autoencoder'
        for rel, identity in self.patch['source']['code_files'].items():
            raw = (root/rel).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(),identity['sha256'])
            new = raw
            if rel=='util/util.py':
                new=raw.replace(b'return int(major) >= 10 and int(minor) >= 1',b'return (int(major), int(minor)) >= (10, 1)')
            self.assertEqual(hashlib.sha256(new).hexdigest(),self.patch['generated_files_sha256'][rel])
            if rel in patches:
                self.assertEqual(patches[rel]['original_sha256'],hashlib.sha256(raw).hexdigest())
                self.assertEqual(patches[rel]['generated_sha256'],hashlib.sha256(new).hexdigest())
                self.assertTrue(patches[rel]['diff'])

    def test_custom_ops_real_cuda(self):
        for r in self.runs:
            for op in r['custom_ops'].values():
                self.assertTrue(op['real_cuda'])
                self.assertTrue(op['output']['finite'])
                self.assertEqual(op['output']['device'],'cuda:0')
                self.assertIn('/builds/e05_pcgan/torch_extensions/',op['extension']['path'])
                self.assertEqual(len(op['extension']['sha256']),64)
                self.assertLess(op['reference_max_abs_error'],1e-5)
                self.assertTrue(all(g['finite'] for g in op['gradients'].values()))

    def check_model(self,key,cls):
        adapter=PCGANAdapter()
        arch=architecture_mapping(adapter.config,adapter.validate_source())
        options={k:v['value'] for k,v in arch['inherited_source_defaults'].items()}
        options.update({v['target'].removeprefix('opt.'):v['value'] for v in arch['settings']})
        for r in self.runs:
            m=r['models'][key]
            self.assertEqual(m['class'],cls)
            self.assertEqual(m['options'],options)
            self.assertGreater(m['trainable_parameters'],0)
            import math
            self.assertEqual(sum(math.prod(s) for s in m['parameters'].values()),m['trainable_parameters'])
            self.assertIn(cls,m['module_tree'])

    def test_encoder_construction(self): self.check_model('encoder','StyleGAN2ResnetEncoder')
    def test_generator_construction(self): self.check_model('generator','StyleGAN2ResnetGenerator')
    def test_image_D_construction(self): self.check_model('image_discriminator','StyleGAN2Discriminator')
    def test_patch_D_construction(self): self.check_model('patch_discriminator','StyleGAN2PatchDiscriminator')

    def test_measured_encoder_shapes(self):
        for r in self.runs:
            for branch in ('src','tgt'):
                self.assertEqual(r['forward'][branch+'_spatial']['shape'],[1,8,128,128])
                self.assertEqual(r['forward'][branch+'_global']['shape'],[1,2048])

    def test_rgb_shapes(self):
        for r in self.runs:
            for branch in ('src_reconstruction','tgt_reconstruction','mixed'):
                self.assertEqual(r['forward'][branch]['shape'],[1,3,256,256])

    def test_discriminator_shapes(self):
        for r in self.runs:
            for branch in ('real','reconstruction','mixed'):
                self.assertEqual(r['forward']['image_D_'+branch]['shape'],[1,1])

    def test_pinned_patch_extraction(self):
        for r in self.runs:
            self.assertIn('util.apply_random_crop',r['patch_interface'])
            for name in ('patch_source','patch_mixed'):
                self.assertEqual(r['forward'][name]['shape'],[1,8,3,128,128])
            self.assertEqual(r['forward']['patch_D']['shape'],[8,1])

    def test_a5_real_torch(self):
        for r in self.runs:
            for name in ('blur_target','blur_mixed'):
                self.assertEqual(r['forward'][name]['shape'],[1,3,128,128])
            g=r['gradients']['blur_target']['target_input']
            self.assertEqual(g['min'],.25)
            self.assertEqual(g['max'],.25)

    def test_blur_gradient_continuity(self):
        for r in self.runs:
            g=r['gradients']['blur_mixed']
            for n in ('source_input','target_input','spatial_code','global_code'):
                self.assertGreater(g[n]['l2'],0)
            self.assertTrue(any(n.startswith('G.') and x['l2']>0 for n,x in g.items()))

    def test_all_forward_and_backward_finite(self):
        for r in self.runs:
            for x in r['forward'].values():self.assertTrue(x['finite'])
            for branch in r['gradients'].values():
                self.assertTrue(branch)
                for x in branch.values():self.assertTrue(x['finite'])

    def test_no_optimizer_or_parameter_update(self):
        for r in self.runs:
            self.assertEqual(r['optimizer_steps'],0)
            self.assertEqual(r['parameter_sha256_before'],r['parameter_sha256_after'])

    def test_no_pretrained_checkpoint_or_bank(self):
        for r in self.runs:
            self.assertEqual(r['pretrained_loads'],0)
            self.assertEqual(r['checkpoints'],0)
            self.assertFalse(r['synthetic_bank'])

    def test_no_benchmark_access(self):
        for r in self.runs:
            self.assertFalse(r['benchmark_data_access'])
            self.assertEqual(r['firewall']['denied'],[])

    def test_firewall_rejects_data_and_weights_before_io(self):
        firewall=Firewall(Path('/tmp/builds/e05_pcgan'))
        for p in (ROOT/'manifests/forbidden.parquet',ROOT/'data/forbidden',
                  Path(str(ROOT)+'_runtime/data'),Path('/tmp/forbidden.pth')):
            with self.assertRaisesRegex(RuntimeError,'FIREWALL'):
                firewall('open',(str(p),'r',0))
        with self.assertRaisesRegex(RuntimeError,'FIREWALL'):
            firewall('os.scandir',(str(ROOT/'data'),))

    def test_fp32_controls(self):
        for r in self.runs:
            p=r['environment_before']['precision']
            self.assertFalse(p['matmul_tf32'] or p['cudnn_tf32'] or p['autocast_cuda'] or p['autocast_cpu'])
            self.assertEqual(p['default_dtype'],'torch.float32')
            self.assertTrue(all(x['dtype']=='torch.float32' for x in r['forward'].values()))

    def test_repeatability_structural_identity(self):
        p,q=self.runs
        for key in ('diagnostic_seed','source_before','models','parameter_sha256_before'):
            self.assertEqual(p[key],q[key])
        for n in p['custom_ops']:
            self.assertEqual(p['custom_ops'][n]['extension'],q['custom_ops'][n]['extension'])

    def test_scope_and_status(self):
        for r in self.runs:
            self.assertEqual(r['status'],'PASS')
            self.assertFalse(r['five_loss_runner_qualified'])
            self.assertEqual(r['fidelity'],'CONTROLLED_ADAPTATION')


if __name__=='__main__':unittest.main()
