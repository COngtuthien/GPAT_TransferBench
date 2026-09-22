"""A6 static source/contract tests; no framework import or model execution."""
import ast
import hashlib
from pathlib import Path
import subprocess
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'third_party/source_cache/difffas'
OVERLAY = ROOT / 'configs/amendments/e07c_a6_feature_interface_source_correction.yaml'
BASE = 'configs/methods/e07c_difffas_bin_idfree.yaml'
BASE_SHA = 'dba34a9222b80ed66a73b8662c10cd348ab46e4f00c2e8166d0a4fe6d552bc1c'


def member(tree, kind, name):
    return next(n for n in tree.body if isinstance(n, kind) and n.name == name)


class TestE07cFeatureContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.overlay = yaml.safe_load(OVERLAY.read_text())
        cls.config = yaml.safe_load((ROOT / BASE).read_text())
        cls.tree = ast.parse((SOURCE / 'models/custom_rn.py').read_bytes())
        cls.resnet = member(cls.tree, ast.ClassDef, 'ResNet')
        cls.init = member(cls.resnet, ast.FunctionDef, '__init__')
        cls.assignments = {t.attr: n.value for n in cls.init.body if isinstance(n, ast.Assign)
                           for t in n.targets if isinstance(t, ast.Attribute)}

    def test_source_repository_and_commit(self):
        source = self.overlay['source']
        for args, expected in [(['rev-parse', 'HEAD'], source['pinned_commit']),
                               (['remote', 'get-url', 'origin'], source['repository'])]:
            self.assertEqual(subprocess.check_output(['git', '-C', str(SOURCE), *args],
                                                     text=True).strip(), expected)
        self.assertEqual(source['pinned_commit'], '23f40519ec25a833ebc06842aa6fbab74fad4d15')
        self.assertEqual(source['repository'], 'https://github.com/murphytju/DiffFAS')

    def test_exact_source_hashes_and_git_bytes(self):
        source = self.overlay['source']
        self.assertEqual(source['sha256'], 'fa788c4d2453b40585b295a8292eaf1afce7a0c622eb8ecc2adec5453a9a64c3')
        for path_key, hash_key in [('file', 'sha256'), ('consumer_file', 'consumer_sha256')]:
            p = ROOT / source[path_key]
            raw = p.read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), source[hash_key])
            pinned = subprocess.check_output(['git', '-C', str(SOURCE), 'show',
                source['pinned_commit'] + ':' + p.relative_to(SOURCE).as_posix()])
            self.assertEqual(raw, pinned)

    def test_basicblock_expansion(self):
        block = member(self.tree, ast.ClassDef, 'BasicBlock')
        expansion = next(n.value for n in block.body if isinstance(n, ast.AnnAssign)
                         and isinstance(n.target, ast.Name) and n.target.id == 'expansion')
        self.assertEqual(ast.literal_eval(expansion), 1)

    def test_stage_channels(self):
        self.assertEqual([ast.literal_eval(self.assignments[name].args[1])
                          for name in ('layer2', 'layer3', 'layer4')], [256, 512, 512])

    def test_direct_stage_returns(self):
        forward = member(self.resnet, ast.FunctionDef, '_forward_impl')
        body = forward.body
        for stage, output in [('layer2', 'o32x32'), ('layer3', 'o16x16'), ('layer4', 'o8x8')]:
            i = next(i for i, n in enumerate(body) if isinstance(n, ast.Assign)
                     and ast.unparse(n.value) == 'self.' + stage + '(x)')
            self.assertEqual(ast.unparse(body[i + 1]), output + ' = x')
        self.assertEqual(ast.unparse(body[-1]), 'return (o32x32, o16x16, o8x8, x)')

    def test_corrected_spatial_channel_interface(self):
        # Propagate pinned stem/stage strides; no model or tensor construction.
        resolution = self.config['conditioning_encoder']['training_contract']['input_resolution']
        for name in ('conv1', 'maxpool'):
            kwargs = {k.arg: ast.literal_eval(k.value) for k in self.assignments[name].keywords}
            resolution = (resolution + 2 * kwargs['padding'] - kwargs['kernel_size']) // kwargs['stride'] + 1
        for name, output in [('layer2', 'x32x32'), ('layer3', 'x16x16'), ('layer4', 'x8x8')]:
            call = self.assignments[name]
            stride = next(ast.literal_eval(k.value) for k in call.keywords if k.arg == 'stride')
            resolution //= stride
            channels = ast.literal_eval(call.args[1])
            self.assertEqual(self.overlay['correction'][output], [resolution, resolution, channels])
        self.assertEqual(self.overlay['correction'], {
            'x32x32': [32, 32, 256], 'x16x16': [16, 16, 512],
            'x8x8': [8, 8, 512], 'embg': ['B', 7]})

    def test_old_descriptions_explicitly_superseded(self):
        self.assertEqual(self.overlay['superseded_descriptions'], {
            'x32x32': [32, 32, 128], 'x16x16': [16, 16, 256]})
        for name, shape in self.overlay['superseded_descriptions'].items():
            self.assertEqual(self.config['conditioning_encoder']['feature_interface_at_256'][name]['shape'], shape)
            self.assertNotEqual(self.overlay['correction'][name], shape)

    def test_no_channel_adapter_or_architecture_substitution(self):
        for key in ('projection_layers_added', 'official_source_modified', 'architecture_substituted'):
            self.assertIs(self.overlay[key], False)
        text = (ROOT / self.overlay['amendment_document']).read_text()
        self.assertIn('No additional learned or fixed', text)
        self.assertIn('channel adapter is authorized', text)

    def test_custom_constructor_and_block_topology(self):
        fn = member(self.tree, ast.FunctionDef, 'resnet18')
        call = next(n.value for n in fn.body if isinstance(n, ast.Return))
        self.assertEqual(ast.unparse(call.func), '_resnet')
        self.assertEqual(ast.literal_eval(call.args[0]), 'resnet18')
        self.assertEqual(ast.unparse(call.args[1]), 'BasicBlock')
        self.assertEqual(ast.literal_eval(call.args[2]), [3, 4, 6, 3])
        self.assertEqual(self.overlay['source']['architecture'], 'models/custom_rn.py::resnet18')

    def test_preserved_head_input_and_k7(self):
        call = self.assignments['fc']
        self.assertEqual(ast.unparse(call.func), 'nn.Linear')
        self.assertEqual(ast.unparse(call.args[0]), '512 * block.expansion')
        encoder = self.config['conditioning_encoder']
        self.assertEqual(encoder['training_contract']['head'], 'fc = nn.Linear(512, 7)')
        self.assertEqual(encoder['substitute_objective']['K'], 7)
        self.assertEqual(self.overlay['correction']['embg'], ['B', 7])

    def test_unet_consumes_three_and_discards_fourth(self):
        tree = ast.parse((SOURCE / 'models/unet_autoenc.py').read_bytes())
        model = member(tree, ast.ClassDef, 'BeatGANsAutoencModel')
        encode = member(model, ast.FunctionDef, 'encode')
        self.assertEqual(ast.unparse(encode.body[-1]), 'return (x32x32, x16x16, x8x8, embg)')
        forward = member(model, ast.FunctionDef, 'forward')
        unpack = next(n for n in ast.walk(forward) if isinstance(n, ast.Assign)
                      and ast.unparse(n.value) == 'self.encode(x_cond, encoder)')
        self.assertEqual(ast.unparse(unpack.targets[0]), '(x32x32, x16x16, x8x8, _)')
        cond = next(n.value for n in ast.walk(forward) if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == 'cond' for t in n.targets)
                    and isinstance(n.value, ast.BinOp))
        consumed = {n.id for n in ast.walk(cond) if isinstance(n, ast.Name)}
        self.assertTrue({'x32x32', 'x16x16', 'x8x8'} <= consumed)
        self.assertFalse({'embg', '_'} & consumed)

    def test_frozen_configs_and_a3_unchanged(self):
        hashes = {BASE: BASE_SHA,
                  'frozen_config_snapshot/' + BASE: BASE_SHA,
                  'configs/frozen/difffas_bin_idfree_v1.yaml':
                      'aa9e984166db3854bba4221098afaef1898474e2e3f1f08a3f80cab0035cf3eb',
                  self.overlay['amendment_a3']['path']:
                      'b12451537bcc3bc14e96e5bcd2ce390b5fc0c8a4b5c60bff7f40a2333665a67a'}
        for p, expected in hashes.items():
            with self.subTest(path=p):
                self.assertEqual(hashlib.sha256((ROOT / p).read_bytes()).hexdigest(), expected)
        self.assertEqual(self.overlay['base_config_path'], BASE)
        self.assertEqual(self.overlay['base_config_sha256'], BASE_SHA)
        self.assertEqual(self.overlay['amendment_a3']['sha256'], self.config['amendment_a3_sha256'])

    def test_overlay_scope_and_fidelity(self):
        self.assertEqual(set(self.overlay), {'amendment_id', 'status', 'provenance',
            'base_config_path', 'base_config_sha256', 'amendment_document', 'amendment_a3',
            'source', 'correction', 'superseded_descriptions', 'projection_layers_added',
            'official_source_modified', 'architecture_substituted'})
        self.assertEqual(self.overlay['amendment_id'], 'A6_E07C_FEATURE_INTERFACE_SOURCE_CORRECTION')
        self.assertEqual(self.overlay['status'], 'OWNER_APPROVED_ADDITIVE_NON_DESTRUCTIVE')
        self.assertEqual(self.overlay['provenance'], 'SOURCE_GROUNDED_CONTRACT_CORRECTION')
        self.assertEqual(self.config['fidelity_class'], 'CONTROLLED_ADAPTATION')
        self.assertTrue((ROOT / self.overlay['amendment_document']).is_file())


if __name__ == '__main__':
    unittest.main()
