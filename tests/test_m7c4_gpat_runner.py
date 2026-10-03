"""M7C4: GPAT production runner + real-TRAIN data path + checkpoint/resume qualification.

Tiers:
  StaticM7C4 (stdlib + numpy/pyarrow/yaml; any host): relation authority, TRAIN-only firewall, order determinism,
      layout, identity map, seeds/roots, schedules, CLI refusal, no VAL/TEST/selector path, evidence + preflight.
  LiveCPUM7C4 (torch; gpat-m7-cpu): worker-count order equality (stub reader, no image), RNG/atomic checkpoint
      round trip, recovery/candidate schema, optimizer ownership B0-B3 + warmup.
  LiveGPUM7C4 (CUDA; gpat-m7-gpu): GeneratorStep with stub teachers on synthetic tensors: D-before-G, one pre-update
      forward, scaler topology, LR/curriculum index, DEV-022 denominator, EMA timing.
History assertions read the commit that ADDED this file (candidate: the worktree).
"""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
AUTHORITY = 'cff688dd35524bf7f01c39916ef87cd4987b6470'
THIS = 'tests/test_m7c4_gpat_runner.py'
EVIDENCE = 'outputs/audit/M7C4_GPAT_RUNNER_QUALIFICATION.json'
RUNNER_FILES = ('methods/gpat/runner.py', 'methods/gpat/runner_io.py', 'methods/gpat/runner_data.py',
                'methods/gpat/runner_checkpoint.py', 'methods/gpat/runner_cli.py')


def git(*args):
    return subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True)


def m7c4_commit():
    return git('log', '--diff-filter=A', '--format=%H', '-1', '--', THIS).stdout.decode().strip() or None


def at_m7c4(rel):
    commit = m7c4_commit()
    return git('show', f'{commit}:{rel}').stdout if commit else (ROOT / rel).read_bytes()


def at_authority(rel):
    return git('show', f'{AUTHORITY}:{rel}').stdout


def has(*mods):
    return all(importlib.util.find_spec(m) is not None for m in mods)


def cuda_available():
    if not has('torch', 'ptwt'):
        return False
    import torch
    return torch.cuda.is_available()


def load_tool(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(has('numpy', 'pyarrow', 'yaml'), 'static tier needs numpy + pyarrow + yaml')
class StaticM7C4(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from methods.gpat import runner_io as rio
        cls.rio = rio
        cls.records = rio.read_relation()

    def test_01_relation_authority_and_allowlist(self):
        rio = self.rio
        self.assertEqual(len(self.records), 8838)
        self.assertEqual(rio.RELATION, 'manifests/pairs_train_v1.parquet')
        self.assertEqual(rio.RELATION_COLUMNS, ('pair_id', 'dataset', 'source_spoof_id', 'target_live_id',
                                                'attack_macro', 'source_subject', 'split', 'split_manifest_sha256'))
        self.assertFalse(set(rio.RELATION_COLUMNS) & set(rio.RELATION_NEVER_READ))
        self.assertEqual({d: sum(r['dataset'] == d for r in self.records) for d in rio.DATASETS},
                         {'casia_fasd': 2520, 'msu_mfsd': 1200, 'siwmv2': 5118})
        self.assertTrue(all(r['source_subject'] is None for r in self.records if r['dataset'] == 'siwmv2'))
        cfg = json.loads(at_m7c4('configs/amendments/gpat_a10_m7_contract_resolution.yaml'))
        self.assertEqual(cfg['bound_authority_sha256'][rio.RELATION], rio.RELATION_SHA256)

    def test_02_train_only_firewall(self):
        rio = self.rio
        log = rio.AccessLog(self.records)
        log.check(self.records[0]['source_spoof_id'], 'source_spoof_id')
        with self.assertRaises(Exception):
            log.check('0' * 64, 'target_live_id')
        rep = log.report()
        self.assertEqual((rep['train_images'], rep['non_train_images'], rep['val_images'], rep['test_images']),
                         (1, 1, 0, 0))
        for rel in RUNNER_FILES + ('tools/m7c4_gpat_runner_qualification.py',):
            src = at_m7c4(rel).decode()
            for forbidden in ('val_pairs_v1', "'VAL'", 'artifact_probe_v1', 'ArtifactProbe(', 'select_checkpoint',
                              'generate_bank', 'test_pairs'):
                self.assertFalse(forbidden in src, f'{rel}: {forbidden}')
            self.assertFalse("read_table(SPLIT" in src, rel)
        io_src = at_m7c4('methods/gpat/runner_io.py').decode()
        self.assertNotIn("pq.read_table(ROOT / 'manifests/split_v1.parquet'", io_src)

    def test_03_epoch_order_determinism(self):
        rio = self.rio
        a = rio.epoch_permutation('generator', rio.SCIENTIFIC, 42, 1)
        self.assertEqual(a, rio.epoch_permutation('generator', rio.SCIENTIFIC, 42, 1))
        self.assertNotEqual(a, rio.epoch_permutation('generator', rio.SCIENTIFIC, 42, 2))
        self.assertNotEqual(a, rio.epoch_permutation('generator', rio.SCIENTIFIC, 1337, 1))
        self.assertNotEqual(a, rio.epoch_permutation('warmup', rio.SCIENTIFIC, 42, 1))
        self.assertEqual(sorted(a), list(range(8838)))
        code = ('import sys; sys.path.insert(0, %r); from methods.gpat import runner_io as r; '
                'rec = r.read_relation(); print(r.order_sha256(rec, r.epoch_permutation("generator", r.SCIENTIFIC, '
                '42, 1)))' % str(ROOT))
        outs = {subprocess.run([sys.executable, '-c', code], env=dict(os.environ, PYTHONHASHSEED=h),
                               capture_output=True, text=True).stdout.strip() for h in ('0', '1', '12345')}
        self.assertEqual(outs, {rio.order_sha256(self.records, a)})                  # PYTHONHASHSEED-independent

    def test_03b_frozen_epoch_order_preimage(self):
        """OWNER_IMPLEMENTATION_CLARIFICATION (M7C4): payload 'GPAT-M7|{stage}|{MODE}|{seed}|{epoch}', MODE in
        {SCIENTIFIC, QUALIFICATION} (upper case, owner-confirmed), UTF-8; seed64 = first 16 HEX chars (64 bits) of the
        SHA-256 hex digest; order = numpy Generator(PCG64(seed64)).permutation(8838)."""
        import numpy as np
        rio = self.rio
        payload = 'GPAT-M7|generator|SCIENTIFIC|42|1'.encode('utf-8')
        self.assertEqual(payload, b'GPAT-M7|generator|SCIENTIFIC|42|1')
        hexd = hashlib.sha256(payload).hexdigest()
        seed64 = int(hexd[:16], 16)
        self.assertEqual(seed64, int.from_bytes(hashlib.sha256(payload).digest()[:8], 'big'))     # 16 hex = 8 bytes
        self.assertLess(seed64, 2 ** 64)
        self.assertEqual(rio.order_key('generator', rio.SCIENTIFIC, 42, 1), seed64)
        perm = [int(i) for i in np.random.Generator(np.random.PCG64(seed64)).permutation(8838)]
        self.assertEqual(perm, rio.epoch_permutation('generator', rio.SCIENTIFIC, 42, 1))
        self.assertEqual((rio.SCIENTIFIC, rio.QUALIFICATION), ('SCIENTIFIC', 'QUALIFICATION'))
        frozen = {('generator', 'SCIENTIFIC', 42): '2b8961cea6327719bc7ae7768d1fd6cfa8caf4d0bb25b23ec8a4f6cbc3396fda',
                  ('generator', 'SCIENTIFIC', 1337): '4591f8e1c18ab3e4a41cbcdb6a2506b639f31060858d08d87785322a28d1b444',
                  ('generator', 'SCIENTIFIC', 2026): '11d4b830b4af9badb0f0e4d7758b1d678f33a143666e2e5dc6f8f62cbbc3aad7',
                  ('generator', 'QUALIFICATION', 70404):
                      '118216183a328a9373cd2b66446d9dc567c56ce7428228f45b58f0df7a2304d9'}
        for (stage, mode, seed), want in frozen.items():
            key = int(hashlib.sha256(f'GPAT-M7|{stage}|{mode}|{seed}|1'.encode('utf-8')).hexdigest()[:16], 16)
            order = [int(i) for i in np.random.Generator(np.random.PCG64(key)).permutation(8838)]
            self.assertEqual(rio.order_sha256(self.records, order), want, (stage, mode, seed))
        rec = json.loads(at_m7c4('configs/amendments/gpat_m7c4_runner_resolution.yaml'))
        self.assertEqual(rec['resolutions']['EPOCH_ORDER']['payload'], 'GPAT-M7|{stage}|{MODE}|{seed}|{epoch}')
        self.assertEqual({k: v for k, v in rec['resolutions']['EPOCH_ORDER']['order_sha256'].items()},
                         {'generator|SCIENTIFIC|42|1': frozen[('generator', 'SCIENTIFIC', 42)],
                          'generator|SCIENTIFIC|1337|1': frozen[('generator', 'SCIENTIFIC', 1337)],
                          'generator|SCIENTIFIC|2026|1': frozen[('generator', 'SCIENTIFIC', 2026)],
                          'generator|QUALIFICATION|70404|1': frozen[('generator', 'QUALIFICATION', 70404)],
                          'warmup|SCIENTIFIC|42|1': '7f34f5fb1dc2ae7330fc390ccb704ee6a588369f9dda75474abcd3527720986e',
                          'warmup|SCIENTIFIC|1337|1': '3075a74cf9209c4483520704dee97f17b2c01ab0b149943af1d1221548770aba',
                          'warmup|SCIENTIFIC|2026|1': 'f33dde05e4ff82b606874ffc75c9a02918735bef40e5a602b19a472e96c2636f',
                          'warmup|QUALIFICATION|70404|1':
                              '23df90b9546eccb684620615c90f2f0891dd240ee40187af0ea3337d0be6db84'})

    def test_04_layout_and_accounting(self):
        rio = self.rio
        rep = rio.layout_report(rio.epoch_permutation('generator', rio.QUALIFICATION, 70404, 1))
        self.assertEqual((rep['microbatches'], rep['optimizer_groups'], rep['regular_group'], rep['tail_group']),
                         (2210, 1105, [4, 4], [4, 2]))
        self.assertEqual((rep['regular_weights'], rep['tail_weights']), ([0.5, 0.5], [4 / 6, 2 / 6]))
        self.assertEqual((rep['warmup_batches'], rep['warmup_tail']), (139, 6))
        self.assertTrue(rep['covered_exactly_once'])
        self.assertEqual((rio.TOTAL_UPDATES, rio.WARMUP_TOTAL_STEPS), (66300, 1390))
        self.assertEqual(rio.CANDIDATE_EPOCHS, tuple(range(10, 61)))
        self.assertEqual(len(rio.CANDIDATE_EPOCHS), 51)
        self.assertEqual((rio.update_index(1, 1), rio.update_index(5, 1105), rio.update_index(6, 1),
                          rio.update_index(15, 1105) + 1, rio.update_index(60, 1105)), (1, 5525, 5526, 16576, 66300))

    def test_05_identity_map(self):
        rio = self.rio
        m = rio.identity_map(self.records)
        self.assertEqual(len(m['classes']), 60)
        labels, valid = rio.identity_labels_for(self.records, m)
        self.assertEqual(sum(valid), 2520 + 1200)
        self.assertTrue(all(l == -1 for l, v in zip(labels, valid) if not v))
        self.assertEqual(m['source_manifest_sha256'], rio.RELATION_SHA256)

    def test_06_seed_allowlist_and_isolated_roots(self):
        rio = self.rio
        for s in (42, 1337, 2026):
            rio.validate_mode_seed(rio.SCIENTIFIC, s)
        for bad in (70404, 0, 43, True):
            with self.assertRaises(Exception):
                rio.validate_mode_seed(rio.SCIENTIFIC, bad)
        with self.assertRaises(Exception):
            rio.validate_mode_seed(rio.QUALIFICATION, 42)
        rt = '/rt'
        self.assertEqual(str(rio.run_root(rt, rio.SCIENTIFIC, 'GPAT-B3', 42)), '/rt/runs/m7/E11/seed_42')
        q = str(rio.run_root(rt, rio.QUALIFICATION, 'GPAT-B3', 70404, 'resume_ref'))
        self.assertTrue(q.startswith('/rt/qualification/m7/M7C4/') and 'runs/m7' not in q)
        self.assertEqual({v[1] for v in rio.VARIANTS.values()}, {'E08', 'E09', 'E10', 'E11'})

    def test_07_schedule_indexing(self):
        from methods.gpat import runtime_contract as rc
        self.assertEqual(rc.main_lr(1), 0.0)
        self.assertAlmostEqual(rc.main_lr(5525), 2e-4, places=15)
        self.assertAlmostEqual(rc.main_lr(66300), 2e-6, places=15)
        self.assertAlmostEqual(rc.curriculum(1)['s_hf'], 0.02)
        self.assertAlmostEqual(rc.curriculum(5525)['s_hf'], 0.05)
        self.assertEqual((rc.curriculum(5526)['s_hf'], rc.curriculum(5526)['lambda_adv']), (0.10, 0.05))
        self.assertEqual((rc.curriculum(16576)['s_hf'], rc.curriculum(16576)['lambda_adv']), (0.15, 0.10))
        self.assertEqual(rc.EMA_START_UPDATE, 5525)
        self.assertEqual(self.rio.EMA_START_EPOCH * self.rio.GROUPS_PER_EPOCH, rc.EMA_START_UPDATE)

    def test_08_cli_refusals(self):
        env = dict(os.environ, PYTHONHASHSEED='70404')
        r = subprocess.run([sys.executable, '-m', 'gpatbench.cli', 'train-generator', '--method', 'GPAT-B0', '--seed',
                            '70404', '--preflight-only'], cwd=ROOT, env=env, capture_output=True, text=True)
        self.assertEqual(r.returncode, 2)
        self.assertIn('never scientific', r.stderr)
        r = subprocess.run([sys.executable, '-m', 'gpatbench.cli', 'train-generator', '--method', 'GPAT-B9', '--seed',
                            '42'], cwd=ROOT, capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)
        from methods.gpat import runner_cli
        with self.assertRaises(SystemExit) as cm:
            runner_cli.preconditions('GPAT-B0', 42, environ={'PYTHONHASHSEED': '42'}, dirty=True,
                                     branch='m6-baselines', check_interpreter=False, verify_assets=False)
        self.assertEqual(cm.exception.code, 2)
        with self.assertRaises(SystemExit):
            runner_cli.preconditions('GPAT-B0', 42, environ={'PYTHONHASHSEED': '41'}, dirty=False,
                                     branch='m6-baselines', check_interpreter=False, verify_assets=False)
        cli = at_m7c4('gpatbench/cli.py').decode()
        self.assertIn('runner_cli.add_parser(sub)', cli)

    def test_09_asset_config(self):
        import yaml
        rio = self.rio
        cfg = yaml.safe_load(at_m7c4(rio.ASSET_CONFIG))
        self.assertEqual({k: cfg['assets'][k]['sha256'] for k in rio.TEACHER_SHA256}, rio.TEACHER_SHA256)
        self.assertNotIn('artifact_probe', json.dumps(cfg))
        self.assertIs(cfg['scientific_values'], False)
        self.assertEqual(cfg['classification'], 'EXECUTION_ONLY_HOST_BINDING')
        self.assertTrue(all(e.get('path') and e.get('identity') and e.get('role') for e in cfg['assets'].values()))
        for rel in ('configs/methods/gpat_b0.yaml', 'configs/methods/gpat_b3.yaml'):
            self.assertNotIn('/home/student20261', at_m7c4(rel).decode())
        rec = json.loads(at_m7c4(rio.M7C3_RECORD))
        self.assertEqual(rec['teacher_assets']['adaface']['checkpoint_sha256'], rio.TEACHER_SHA256['adaface_weight'])
        self.assertEqual(rec['teacher_assets']['facexformer']['checkpoint_sha256'],
                         rio.TEACHER_SHA256['facexformer_weight'])
        self.assertEqual(hashlib.sha256(at_m7c4(rio.GPU_LOCK)).hexdigest(), rio.GPU_LOCK_SHA256)
        self.assertEqual(hashlib.sha256(at_m7c4(rio.M7C3_RECORD)).hexdigest(), rio.M7C3_RECORD_SHA256)

    def test_10_runner_reuses_static_core(self):
        import ast
        src = at_m7c4('methods/gpat/runner.py').decode()
        calls = {ast.unparse(n.func) for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Call)}
        for name in ('losses.l_id', 'losses.l_lm', 'losses.l_parse', 'losses.l_low', 'losses.l_artcon',
                     'spectral.spec_loss', 'losses.l_gadv', 'losses.l_type', 'losses.l_budget', 'losses.l_tv',
                     'losses.l_bg', 'losses.face_mask_dilated', 'losses.idadv_microbatch', 'losses.idadv_share',
                     'losses.assemble_generator_loss', 'losses.l_d', 'rc.main_lr', 'rc.curriculum',
                     'rc.attack_warmup_lr', 'tp.facexformer_input_exact', 'tp.adaface_input'):
            self.assertIn(name, calls, name)
        for forbidden in ('F.cross_entropy', 'F.cosine_similarity', 'torch.fft.fft2', 'F.binary_cross_entropy_with_logits',
                          'tp.facexformer_input', 'losses.lferr_selection'):
            self.assertNotIn(forbidden, calls, forbidden)
        self.assertNotIn('torch.save(', src)                       # checkpoints only via runner_checkpoint

    def test_11_static_core_unchanged(self):
        for rel in ('methods/gpat/' + n for n in ('artifact_encoder.py', 'composition.py', 'config.py',
                                                  'discriminator.py', 'ema.py', 'generator.py', 'losses.py',
                                                  'model.py', 'spectral.py', 'teacher_preprocess.py', 'wavelet.py')):
            self.assertEqual(at_m7c4(rel), at_authority(rel), rel)

    def test_12_evidence_and_preflight(self):
        pf = load_tool('m7c4_pf', 'tools/m7c4_gpat_runner_preflight.py')
        ev = pf.check_evidence(at_m7c4)
        self.assertEqual(ev['status'], 'PASS')
        rep = ev['firewall']
        self.assertEqual((rep['val_images'], rep['test_images'], rep['non_train_images'], rep['val_metadata'],
                          rep['test_metadata']), (0, 0, 0, 0, 0))
        orders = ev['scenarios']['orders']
        for key, value in orders['generator_epoch1'].items():
            mode, seed = key.split('_')
            self.assertEqual(self.rio.order_sha256(self.records, self.rio.epoch_permutation(
                'generator', mode, int(seed), 1)), value, key)
        bad = copy.deepcopy(ev)
        bad['firewall']['val_images'] = 1
        raw = json.dumps(bad).encode()
        with self.assertRaises(ValueError):
            pf.check_evidence(lambda rel: raw if rel == EVIDENCE else at_m7c4(rel))


class _StubReader:
    """No image is read: a constant canonical-size RGB image (metadata-only loader tests)."""

    def __call__(self, sample_id):
        from PIL import Image
        return Image.new('RGB', (256, 256), (int(sample_id[:2], 16), 0, 0))


@unittest.skipUnless(has('torch', 'torchvision', 'PIL', 'pyarrow'), 'CPU tier needs torch (gpat-m7-cpu)')
class LiveCPUM7C4(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from methods.gpat import runner_io as rio, runner_data as rd, runner_checkpoint as ck
        cls.torch, cls.rio, cls.rd, cls.ck = torch, rio, rd, ck
        cls.records = rio.read_relation()

    def test_20_worker_count_order_equality(self):
        rio, rd = self.rio, self.rd
        plan = [mb for g in rio.groups(rio.epoch_permutation('generator', rio.QUALIFICATION, 70404, 1))[:6] for mb in g]
        orders = {}
        for workers in (0, 4):
            sampler = rd.PlanSampler()
            sampler.set_plan(plan)
            loader = rd.make_loader(rd.GeneratorPairs(self.records, _StubReader()), sampler, 70404,
                                    num_workers=workers, pin_memory=False)
            orders[workers] = [b['index'].tolist() for b in loader]
        self.assertEqual(orders[0], orders[4])
        self.assertEqual(orders[0], plan)

    def test_21_decode_contract(self):
        torch = self.torch
        x = self.rd.decode(_StubReader()('ff' + '0' * 62))
        self.assertEqual((tuple(x.shape), x.dtype), ((3, 256, 256), torch.float32))
        self.assertEqual(float(x[0, 0, 0]), 255 / 127.5 - 1)
        self.assertEqual(float(x[1, 0, 0]), -1.0)

    def test_22_rng_and_atomic_checkpoint(self):
        import random
        import tempfile
        import numpy as np
        torch, ck = self.torch, self.ck
        random.seed(1)
        np.random.seed(2)
        torch.manual_seed(3)
        state = ck.rng_state()
        a = (random.random(), float(np.random.rand()), float(torch.rand(1)))
        with tempfile.TemporaryDirectory() as d:
            info = ck.atomic_save({'rng': state, 'w': torch.arange(4)}, Path(d) / 'r.pt')
            loaded = ck.load(info['path'])                                     # weights_only=True
            self.assertEqual(info['sha256'], ck.file_sha256(info['path']))
            self.assertEqual([p.name for p in Path(d).iterdir()], ['r.pt'])
        ck.restore_rng(loaded['rng'])
        self.assertEqual(a, (random.random(), float(np.random.rand()), float(torch.rand(1))))

    def test_23_recovery_and_candidate_schema(self):
        import tempfile
        import torchvision
        torch, ck = self.torch, self.ck
        from methods.gpat.config import load_config
        from methods.gpat.ema import ModelEMA
        from methods.gpat.model import GPATCore
        from methods.gpat import runner as R
        core = GPATCore(load_config('B3'), pretrained_state=torchvision.models.resnet18(weights=None).state_dict(),
                        with_discriminator=True)
        g, d = R.generator_optimizer(core), R.discriminator_optimizer(core)
        p = ck.recovery_payload(modules={'e_art': core.e_art, 'g_res': core.g_res, 'discriminator': core.discriminator,
                                         'attack_head': core.attack_head, 'identity_head': core.identity_head},
                                optimizers={'G_OPT': g, 'D_OPT': d},
                                scalers={'G_SCALER': torch.amp.GradScaler('cuda', enabled=False),
                                         'D_SCALER': torch.amp.GradScaler('cuda', enabled=False)},
                                ema=None, position={'epoch': 1, 'next_group': 2, 'global_update': 1},
                                provenance={'method': 'GPAT-B3'}, identity_map_sha256='x', order={'seed': 70404})
        self.assertEqual(p['kind'], ck.RECOVERY_KIND)
        self.assertIn('NOT_ELIGIBLE_FOR_VAL_SELECTION', p['labels'])
        self.assertEqual(set(p['modules']), {'e_art', 'g_res', 'discriminator', 'attack_head', 'identity_head'})
        ema = {'e_art': ModelEMA(core.e_art), 'g_res': ModelEMA(core.g_res)}
        meta = {k: 'x' for k in ('method', 'seed', 'epoch', 'global_update', 'config_sha256', 'code_commit',
                                 'gpu_env_lock_sha256', 'source_manifest_sha256', 'teacher_sha256', 'ema_decay',
                                 'gamma')}
        cand = ck.candidate_payload(e_art_ema=ema['e_art'], g_res_ema=ema['g_res'], metadata=meta)
        self.assertIs(cand['metadata']['selected'], False)
        self.assertEqual(set(cand), {'kind', 'e_art_ema', 'g_res_ema', 'metadata'})
        with self.assertRaises(Exception):
            ck.candidate_payload(e_art_ema=ema['e_art'], g_res_ema=ema['g_res'], metadata=dict(meta, val_score=1))
        with tempfile.TemporaryDirectory() as dd:
            cand['metadata']['global_update'] = 11050
            ck.write_candidate(dd, 10, cand)
            with self.assertRaises(Exception):
                ck.write_candidate(dd, 10, cand)                                   # immutable
            with self.assertRaises(Exception):
                ck.write_candidate(dd, 9, cand)                                    # 10..60 only

    def test_24_optimizer_ownership(self):
        import torchvision
        from methods.gpat.config import load_config
        from methods.gpat.model import GPATCore
        from methods.gpat import runner as R
        expect = {'B0': ['e_art', 'g_res'], 'B1': ['attack_head', 'e_art', 'g_res'],
                  'B2': ['e_art', 'g_res', 'identity_head'], 'B3': ['attack_head', 'e_art', 'g_res', 'identity_head']}
        for v, mods in expect.items():
            core = GPATCore(load_config(v), pretrained_state=torchvision.models.resnet18(weights=None).state_dict(),
                            with_discriminator=True)
            self.assertEqual(sorted(core.generator_modules()), mods, v)
            g = R.generator_optimizer(core)
            ids = {id(p) for p in g.param_groups[0]['params']}
            self.assertEqual(ids, {id(p) for m in core.generator_modules().values() for p in m.parameters()})
            self.assertFalse(ids & {id(p) for p in core.discriminator.parameters()})
            self.assertEqual((g.defaults['betas'], g.defaults['weight_decay'], type(g).__name__), ((0.5, 0.999), 0.0, 'Adam'))
            d = R.discriminator_optimizer(core)
            self.assertEqual({id(p) for p in d.param_groups[0]['params']}, {id(p) for p in core.discriminator.parameters()})
            if v in ('B1', 'B3'):
                w = R.warmup_optimizer(core)
                self.assertEqual(type(w).__name__, 'AdamW')
                self.assertEqual((w.defaults['lr'], w.defaults['weight_decay']), (1e-4, 1e-4))
                self.assertEqual({id(p) for p in w.param_groups[0]['params']},
                                 {id(p) for m in (core.e_art, core.attack_head) for p in m.parameters()})
            else:
                with self.assertRaises(Exception):
                    R.warmup_optimizer(core)


class _StubTeachers:
    """Differentiable stand-ins with the real teacher interfaces (GPU tier, synthetic tensors only)."""

    def __init__(self, torch):
        self.torch = torch
        g = torch.Generator().manual_seed(5)
        self.w = torch.randn(3, 512, generator=g).cuda()

    def modules(self):
        return {}

    def facexformer(self, x):
        import torch.nn.functional as F
        y = F.interpolate(x.float(), size=(224, 224), mode='bilinear', align_corners=False)
        seg = y.mean(1, keepdim=True).repeat(1, 11, 1, 1) * self.torch.arange(11., device=x.device).view(1, 11, 1, 1)
        lm = self.torch.tanh(y[:, :2, :68, 0].permute(0, 2, 1))
        return seg, lm

    def adaface(self, x):
        import torch.nn.functional as F
        return F.normalize(x.float().mean((-2, -1)) @ self.w, dim=1)

    def art(self, x):
        return x.float().mean((-2, -1)) @ self.w


@unittest.skipUnless(cuda_available(), 'GPU tier: needs CUDA (gpat-m7-gpu)')
class LiveGPUM7C4(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        import torchvision
        from methods.gpat import runner as R, runtime_contract as rc
        from methods.gpat.config import load_config
        from methods.gpat.model import GPATCore
        rc.apply_qualification_determinism(70404, gpu=True)
        cls.torch, cls.R, cls.rc = torch, R, rc
        cls.cfg = load_config('B3')
        torch.manual_seed(1)
        cls.core = GPATCore(cls.cfg, pretrained_state=torchvision.models.resnet18(weights=None).state_dict(),
                            with_discriminator=True).cuda().train()

    def batch(self, n, seed, valid):
        torch = self.torch
        g = torch.Generator().manual_seed(seed)
        return {'index': torch.arange(n), 'x_source': (torch.rand(n, 3, 256, 256, generator=g) * 2 - 1).cuda(),
                'x_target': (torch.rand(n, 3, 256, 256, generator=g) * 2 - 1).cuda(),
                'attack': torch.randint(0, 6, (n,), generator=g).cuda(),
                'identity': torch.where(torch.tensor(valid), torch.randint(0, 60, (n,), generator=g), -1).cuda(),
                'identity_valid': torch.tensor(valid).cuda()}

    def make_step(self):
        torch, R = self.torch, self.R
        core = copy.deepcopy(self.core)
        g_opt, d_opt = R.generator_optimizer(core), R.discriminator_optimizer(core)
        step = R.GeneratorStep(core, _StubTeachers(torch), self.cfg, g_opt, d_opt, torch.amp.GradScaler('cuda'),
                               torch.amp.GradScaler('cuda'))
        return core, step

    def test_30_d_before_g_single_forward_and_scalers(self):
        core, step = self.make_step()
        calls = []
        for name, scaler in (('D', step.d_scaler), ('G', step.g_scaler)):
            orig = scaler.step
            scaler.step = (lambda o, *a, _n=name, _f=orig, **k: (calls.append(_n), _f(o, *a, **k))[1])
        fwd = []
        h = core.g_res.register_forward_hook(lambda m, i, o: fwd.append(1))
        group = [self.batch(4, 1, [True, False, True, False]), self.batch(2, 2, [False, True])]
        step.capture = []
        rec = step(group, 5526)
        h.remove()
        self.assertEqual(calls, ['D', 'G'])                                      # D first, once each, at the boundary
        self.assertEqual(len(fwd), 2)                                             # one GPAT forward per microbatch
        self.assertIsNot(step.g_scaler, step.d_scaler)
        self.assertEqual(rec['sample_weights'], [4 / 6, 2 / 6])
        self.assertEqual(rec['labelled_identity_count'], 3)
        self.assertEqual(rec['learning_rate'], self.rc.main_lr(5526))
        self.assertEqual((rec['scale_hf'], rec['lambda_adv']), (0.10, 0.05))
        self.assertTrue(all(p.grad is None for p in core.parameters()))         # zeroed after the boundary

    def test_31_dev022_share_not_sample_weighted(self):
        torch = self.torch
        core, step = self.make_step()
        step.capture = []
        group = [self.batch(4, 3, [True, False, False, False]), self.batch(4, 4, [True, True, False, False])]
        step(group, 1)
        shares = [c['components']['idadv_share'] for c in step.capture]
        self.assertEqual(len(shares), 2)
        zero = [self.batch(4, 5, [False] * 4), self.batch(4, 6, [False] * 4)]
        step.capture = []
        rec = step(zero, 2)
        self.assertEqual(rec['labelled_identity_count'], 0)
        self.assertEqual([c['components']['idadv_share'] for c in step.capture], [0.0, 0.0])

    def test_32_same_pre_update_gradients(self):
        """G and D gradients both come from the pre-update parameters: D's step does not precede any G backward."""
        torch = self.torch
        core, step = self.make_step()
        d_before = {k: v.clone() for k, v in core.discriminator.state_dict().items()}
        seen = []
        orig_backward = torch.Tensor.backward
        step.inspect = lambda stage, s: seen.append((stage, all(torch.equal(core.discriminator.state_dict()[k], v)
                                                                for k, v in d_before.items()),
                                                     any(p.grad is not None for p in core.g_res.parameters())))
        step([self.batch(4, 7, [True] * 4), self.batch(4, 8, [False] * 4)], 6000)
        # both unscales happen before either step: every G backward already happened and D is still pre-update
        self.assertEqual(seen, [('D', True, True), ('G', True, True)])
        self.assertIs(torch.Tensor.backward, orig_backward)

    def test_33_ema_timing_and_lr(self):
        R, rc = self.R, self.rc
        core, step = self.make_step()
        from methods.gpat.ema import ModelEMA
        step.ema = {'e_art': ModelEMA(core.e_art), 'g_res': ModelEMA(core.g_res)}
        rec = step([self.batch(4, 9, [False] * 4), self.batch(4, 10, [False] * 4)], 5525)
        self.assertFalse(rec['ema_updated'])                                     # u <= 5525: not yet
        rec = step([self.batch(4, 11, [False] * 4), self.batch(4, 12, [False] * 4)], 5526)
        self.assertTrue(rec['ema_updated'])
        self.assertEqual(step.ema['g_res'].updates, 1)


    # ------------------------------------------------------------- FAIL_CLOSED_AMP_OVERFLOW (owner M7C4)
    def armed(self, *, with_ema=False):
        torch = self.torch
        core, step = self.make_step()
        counts = {'D': 0, 'G': 0}
        for name, opt in (('D', step.d_opt), ('G', step.g_opt)):
            orig = opt.step
            opt.step = (lambda *a, _n=name, _f=orig, **k: (counts.__setitem__(_n, counts[_n] + 1), _f(*a, **k))[1])
        if with_ema:
            from methods.gpat.ema import ModelEMA
            step.ema = {'e_art': ModelEMA(core.e_art), 'g_res': ModelEMA(core.g_res)}
        params = {k: v.detach().clone() for k, v in core.state_dict().items()}
        return core, step, counts, params

    def group(self):
        return [self.batch(4, 21, [True, False, True, False]), self.batch(4, 22, [False] * 4)]

    def assert_untouched(self, core, step, counts, params, scales):
        torch = self.torch
        self.assertEqual(counts, {'D': 0, 'G': 0})                              # neither optimizer stepped
        self.assertEqual((step.d_scaler.get_scale(), step.g_scaler.get_scale()), scales)   # no update()
        self.assertTrue(all(torch.equal(core.state_dict()[k], v) for k, v in params.items()
                            if 'running' not in k and 'num_batches' not in k))
        if step.ema is not None:
            self.assertEqual((step.ema['e_art'].updates, step.ema['g_res'].updates), (0, 0))

    def test_34_finite_group_steps_each_optimizer_once(self):
        core, step, counts, params = self.armed()
        rec = step(self.group(), 6000)
        self.assertEqual(counts, {'D': 1, 'G': 1})
        self.assertEqual(rec['group_status'], 'COMPLETE')

    def test_35_inf_in_d_gradient_fails_closed(self):
        torch, R = self.torch, self.R
        core, step, counts, params = self.armed(with_ema=True)
        first = next(core.discriminator.parameters())
        step.inspect = lambda stage, s: first.grad.view(-1)[0].fill_(float('inf')) if stage == 'D' else None
        with self.assertRaises(R.AmpOverflowStop) as cm:
            step(self.group(), 6000)
        self.assert_untouched(core, step, counts, params, (65536.0, 65536.0))
        self.assertEqual(cm.exception.record['offending_optimizers'], ['D_OPT'])
        self.assertEqual(cm.exception.record['optimizer_steps_taken'], 0)

    def test_36_inf_in_g_gradient_fails_closed(self):
        torch, R = self.torch, self.R
        core, step, counts, params = self.armed(with_ema=True)
        p = core.g_res.ending.weight
        step.inspect = lambda stage, s: p.grad.view(-1)[0].fill_(float('nan')) if stage == 'G' else None
        with self.assertRaises(R.AmpOverflowStop) as cm:
            step(self.group(), 6000)
        self.assert_untouched(core, step, counts, params, (65536.0, 65536.0))
        self.assertEqual(cm.exception.record['offending_optimizers'], ['G_OPT'])
        self.assertIn('g_res.ending.weight', cm.exception.record['offending_parameters']['G_OPT'])

    def test_37_nan_loss_prohibits_backward(self):
        torch, R = self.torch, self.R
        core, step, counts, params = self.armed()
        orig = step.t.art
        step.t.art = lambda x: orig(x) * float('nan')
        with self.assertRaises(R.NonFiniteLossStop):
            step(self.group(), 6000)
        self.assertTrue(all(p.grad is None for p in core.parameters()))          # backward never ran
        self.assert_untouched(core, step, counts, params, (65536.0, 65536.0))

    def test_38_warmup_gradient_inf_fails_closed(self):
        torch, R = self.torch, self.R
        core = copy.deepcopy(self.core)
        opt, scaler = R.warmup_optimizer(core), torch.amp.GradScaler('cuda')
        w = R.WarmupStep(core, opt, scaler)
        steps = []
        orig = opt.step
        opt.step = lambda *a, **k: (steps.append(1), orig(*a, **k))[1]
        before = {k: v.clone() for k, v in core.attack_head.state_dict().items()}
        w.inspect = lambda s: core.attack_head.fc.weight.grad.view(-1)[0].fill_(float('inf'))
        b = self.batch(6, 30, [False] * 6)
        with self.assertRaises(R.AmpOverflowStop):
            w(b, 5)
        self.assertEqual(steps, [])
        self.assertTrue(all(torch.equal(core.attack_head.state_dict()[k], v) for k, v in before.items()))

    def test_39_post_step_nonfinite_parameter(self):
        torch, R = self.torch, self.R
        core, step, counts, params = self.armed(with_ema=True)
        orig = step.g_opt.step

        def corrupting_step(*a, **k):
            out = orig(*a, **k)
            with torch.no_grad():
                core.g_res.intro.weight.view(-1)[0] = float('inf')
            return out
        step.g_opt.step = corrupting_step
        with self.assertRaises(R.NumericalPostStepStop) as cm:
            step(self.group(), 6000)
        self.assertEqual(cm.exception.failure_type, 'FAILED_NUMERICAL_POST_STEP')
        self.assertEqual((step.ema['e_art'].updates, step.ema['g_res'].updates), (0, 0))   # EMA never advanced


class _FakeCtx:
    def __init__(self, index_path):
        self.index_path, self.logs = index_path, []

    def path(self, key):
        return self.index_path

    def log(self, kind, record):
        self.logs.append((kind, record))


@unittest.skipUnless(has('torch', 'pyarrow'), 'loop tier needs torch (gpat-m7-cpu)')
class LiveCPUFailClosedLoopM7C4(unittest.TestCase):
    """Safe recovery C at boundary k; group k+1 fails: no checkpoint for k+1, C stays the resumable authority with
    next_group = k+1, nothing advances, the failed TRAIN group is not skipped and not retried."""

    def test_40_failed_group_keeps_previous_safe_recovery(self):
        import tempfile
        import numpy as np
        import torch
        from methods.gpat import runner as R, runner_checkpoint as ck, runner_io as rio
        k = 3
        with tempfile.TemporaryDirectory() as d:
            c = ck.atomic_save({'kind': ck.RECOVERY_KIND, 'position': {'epoch': 1, 'next_group': k + 1,
                                                                       'global_update': k}}, Path(d) / 'latest.pt')
            index = Path(d) / 'checkpoint_index.json'
            index.write_text(json.dumps({'checkpoints': [{'path': 'checkpoints/recovery/latest.pt', 'role': 'recovery',
                                                          'sha256': c['sha256'], 'global_step': k}]}))

            class Fake(R.Trainer):
                def __init__(self):                                           # no data, no model: loop logic only
                    self.position = {'stage': 'generator', 'epoch': 1, 'next_group': k + 1, 'global_update': k}
                    self.ctx, self.saves, self.ends = _FakeCtx(index), 0, 0
                    self.plan = [[[2 * g], [2 * g + 1]] for g in range(rio.GROUPS_PER_EPOCH)]
                    self.gen_sampler = type('S', (), {'set_plan': lambda s, p: setattr(s, 'p', p)})()
                    sampler = self.gen_sampler
                    self.gen_loader = type('L', (), {'__iter__': lambda s: iter(
                        [{'index': np.array(mb)} for mb in sampler.p])})()
                    self.seen = []

                    def step(group, u):
                        self.seen.append(u)
                        raise R.AmpOverflowStop('injected', {'failure_type': 'FAIL_CLOSED_AMP_OVERFLOW',
                                                             'global_update_attempted': u})
                    self.step = step

                def generator_plan(self, epoch):
                    return self.plan

                def log_access(self, batch, roles):
                    pass

                def save_recovery(self):
                    self.saves += 1

                def end_epoch(self, e):
                    self.ends += 1

            tr = Fake()
            with self.assertRaises(R.AmpOverflowStop):
                tr.run_generator_groups(R.StopFlag())
            self.assertEqual(tr.seen, [k + 1])                                  # the failed group was attempted once
            self.assertEqual(tr.position, {'stage': 'generator', 'epoch': 1, 'next_group': k + 1, 'global_update': k})
            self.assertEqual((tr.saves, tr.ends), (0, 0))                       # no checkpoint for k+1
            self.assertEqual(ck.file_sha256(c['path']), c['sha256'])            # C unchanged and still authoritative
            self.assertEqual(ck.load(c['path'])['position']['next_group'], k + 1)
            kind, record = tr.ctx.logs[-1]
            self.assertEqual(kind, 'numerical_failure')
            self.assertEqual(record['last_safe_recovery']['sha256'], c['sha256'])
            self.assertEqual(record['attempted_group_or_batch'], k + 1)


if __name__ == '__main__':
    unittest.main()
