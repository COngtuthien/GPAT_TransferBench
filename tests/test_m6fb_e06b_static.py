"""M6FB: E06b DSDG-NATIVE frozen config + static adapter. No model, GPU, image or TEST access.

History assertions compare the state at the commit that ADDED this test (candidate: the worktree) with the M6H
authority, so later milestones never break them (no B1 HEAD lock). The real-population test reads only the
TRAIN-filtered, allowlisted split metadata (no image, no TEST row materialized).
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import subprocess
import sys
import unittest

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from methods.common.config import FrozenConfigError, METHOD_FILES, load_method_config  # noqa: E402
from methods.common.learned import PreparationError, checkpoint_plan  # noqa: E402
from methods.dsdg import native  # noqa: E402

AUTHORITY = '4ee7ac29ca16687aa4eb2a492d2a25554ce29253'
THIS = 'tests/test_m6fb_e06b_static.py'
PREFLIGHT = 'tools/m6fb_e06b_static_preflight.py'
CONFIG = 'configs/methods/e06b_dsdg_native.yaml'
SNAPSHOT = 'frozen_config_snapshot/configs/methods/e06b_dsdg_native.yaml'
LEDGER = 'outputs/audit/EXECUTION_LEDGER.jsonl'
EVIDENCE = ROOT / 'outputs/audit/M6FB_E06B_STATIC_IMPLEMENTATION.json'


def load_preflight():
    spec = importlib.util.spec_from_file_location('m6fb_preflight_under_test', ROOT / PREFLIGHT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git(*args):
    return subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True)


def m6fb_commit():
    return git('log', '--diff-filter=A', '--format=%H', '-1', '--', THIS).stdout.decode().strip() or None


def at_m6fb(rel):
    commit = m6fb_commit()
    return git('show', f'{commit}:{rel}').stdout if commit else (ROOT / rel).read_bytes()


def at_authority(rel):
    return git('show', f'{AUTHORITY}:{rel}').stdout


def row(sid, ds, subject, label, macro, split='TRAIN', m2='COMPLETE'):
    return {'sample_id': sid, 'dataset': ds, 'subject_id_global': subject, 'label_binary': label,
            'attack_macro': macro, 'split': split, 'm2_status': m2}


def synthetic_rows():
    rows = []
    for ds, subjects in (('casia_fasd', 3), ('msu_mfsd', 2)):
        for s in range(subjects):
            subj = f'{ds}::{s}'
            rows += [row(f'{ds}-{s}-live-{i}', ds, subj, 0, 'live') for i in range(3 + s)]
            rows += [row(f'{ds}-{s}-print-{i}', ds, subj, 1, 'print') for i in range(2)]
            rows += [row(f'{ds}-{s}-replay-{i}', ds, subj, 1, 'replay') for i in range(2)]
    return rows


def real_train_rows():
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        return None
    if not (ROOT / 'manifests/split_v1.parquet').is_file():
        return None
    return native.load_train_rows(load_method_config('E06b'))


class TestM6FBConfig(unittest.TestCase):
    def test_00_config_snapshot_byte_equality_and_freeze_record(self):
        raw = at_m6fb(CONFIG)
        self.assertEqual(raw, at_m6fb(SNAPSHOT))
        self.assertEqual(hashlib.sha256(raw).hexdigest(), load_preflight().CONFIG_SHA)
        cfg = load_method_config('E06b')
        self.assertEqual(cfg['_runtime']['config_sha256'], hashlib.sha256(raw).hexdigest())
        self.assertNotIn('E06b', METHOD_FILES)  # the M6B set stays exactly as M6B froze it

    def test_01_frozen_values(self):
        c = yaml.safe_load(at_m6fb(CONFIG))
        self.assertEqual((c['method_id'], c['track'], c['status'], c['milestone_status']),
                         ('E06b', 'B_NATIVE_FULL_SECONDARY', 'CONFIG_FROZEN', 'NOT_TRAINED'))
        self.assertEqual(c['native_population']['train_datasets'], ['casia_fasd', 'msu_mfsd'])
        self.assertEqual(c['native_population']['non_instantiable']['siwmv2']['status'],
                         'NOT_INSTANTIABLE_MISSING_SUBJECT_ID')
        self.assertEqual((c['spoof_type_supervision']['vocabulary'], c['spoof_type_supervision']['class_index'],
                          c['spoof_type_supervision']['num_spoof_classes'], c['training']['attack_type']),
                         (['print', 'replay'], {'print': 0, 'replay': 1}, 2, 2))
        lo = c['losses']
        self.assertEqual((lo['lambda_pair'], lo['lambda_mmd'], lo['lambda_ip'], lo['lambda_type'], lo['lambda_ort']),
                         (5, 50, 1000, 10, 1))
        self.assertTrue(lo['loss_cls_active'] and lo['loss_pair_active'])
        self.assertEqual(lo['removed'], [])
        self.assertEqual((c['training']['all_epochs'], c['training']['effective_batch_size'], c['training']['hdim'],
                          c['training']['workers'], c['optimizer']['learning_rate']), (200, 240, 128, 8, 2e-4))
        self.assertEqual(c['checkpoint']['rule'], 'OFFICIAL_GENERATOR_EPOCH_200')
        self.assertEqual(c['seeds']['experiment_seeds'], [42, 1337, 2026])
        self.assertEqual(c['external_assets'][0]['sha256'],
                         'd0750746622270548137b092700811b00dc64d67ca8df042c6cf48cb3b1b9964')
        self.assertEqual(c['generation']['n_syn_intended'], 'DEFERRED_TO_M8')
        self.assertEqual((c['native_relation']['materialized_pair_list'], c['native_relation']['native_manifest']),
                         (False, 'NOT_CREATED'))

    def test_02_fidelity_target_vs_pending(self):
        c = load_method_config('E06b')
        self.assertEqual((c['target_fidelity'], c['final_execution_fidelity'], c['fidelity_class']),
                         ('FAITHFUL_OFFICIAL', 'PENDING_M6F_C_RUNTIME_QUALIFICATION',
                          'PENDING_M6F_C_RUNTIME_QUALIFICATION'))
        self.assertNotEqual(c['fidelity_class'], 'CONTROLLED_ADAPTATION')

    def test_03_no_test_access_firewall(self):
        c = load_method_config('E06b')
        t = c['data']['splits']['TEST']
        self.assertEqual((t['allowed'], t['used_for'], t['code_path_present']), (False, [], False))
        self.assertIs(c['data']['splits']['VAL']['may_select_checkpoint'], False)
        with self.assertRaises(PreparationError):
            native.NativeRelation([row('x', 'casia_fasd', 'casia_fasd::1', 1, 'print', split='TEST')])
        with self.assertRaises(PreparationError):
            native.NativeRelation([row('x', 'casia_fasd', 'casia_fasd::1', 1, 'print', split='VAL')])

    def test_04_native_semantics_match_contract(self):
        sem = native.native_semantics(load_method_config('E06b'))
        self.assertEqual((sem['datasets'], sem['vocabulary'], sem['class_index']),
                         (('casia_fasd', 'msu_mfsd'), ('print', 'replay'), {'print': 0, 'replay': 1}))
        adapter = native.DSDGNativeAdapter()
        self.assertEqual(adapter.lambdas()['lambda_pair'], 5)
        self.assertEqual((adapter.spoof_class('print'), adapter.spoof_class('replay')), (0, 1))
        with self.assertRaises(PreparationError):
            adapter.spoof_class('mask_3d')


class TestM6FBRelation(unittest.TestCase):
    def test_05_same_subject_pools_and_online_draw(self):
        rel = native.NativeRelation(synthetic_rows())
        rng = random.Random(7)
        for i, (sid, key, cls) in enumerate(rel.spoof):
            for _ in range(5):
                live = rel.draw_live(i, rng)
                self.assertIn(live, rel.pools[key])
                self.assertTrue(live.startswith(sid.split('-')[0] + '-' + sid.split('-')[1] + '-live'))
        draws = {rel.draw_live(0, rng) for _ in range(50)}
        self.assertGreater(len(draws), 1, 'the live partner is redrawn on every load')

    def test_06_siw_and_pseudo_identity_rejected(self):
        with self.assertRaises(PreparationError):
            native.NativeRelation([row('s', 'siwmv2', None, 1, 'print')])
        with self.assertRaises(PreparationError):
            native.NativeRelation([row('s', 'casia_fasd', None, 1, 'print')])
        with self.assertRaises(PreparationError):
            native.NativeRelation([row('s', 'casia_fasd', 'content_group::7', 1, 'print')])

    def test_07_unexpected_macro_rejected_not_remapped(self):
        rows = synthetic_rows() + [row('odd', 'casia_fasd', 'casia_fasd::0', 1, 'mask_3d')]
        with self.assertRaises(PreparationError):
            native.NativeRelation(rows)
        with self.assertRaises(PreparationError):
            native.NativeRelation(synthetic_rows() + [row('bad', 'casia_fasd', 'casia_fasd::0', 0, 'print')])

    def test_08_every_spoof_needs_a_same_subject_live_pool(self):
        rows = [r for r in synthetic_rows() if not (r['subject_id_global'] == 'msu_mfsd::1' and r['label_binary'] == 0)]
        with self.assertRaises(PreparationError):
            native.NativeRelation(rows)

    def test_09_deterministic_order_independent_of_input_order(self):
        rows = synthetic_rows()
        shuffled = list(rows)
        random.Random(3).shuffle(shuffled)
        a, b = native.NativeRelation(rows), native.NativeRelation(shuffled)
        self.assertEqual((a.spoof, a.pools), (b.spoof, b.pools))
        self.assertEqual([s[0] for s in a.spoof], sorted((s[0] for s in a.spoof), key=lambda x: x.encode()))

    def test_10_dataset_item_keys_and_type(self):
        rel = native.NativeRelation(synthetic_rows())
        reader = lambda sid: np.full((256, 256, 3), 7, dtype=np.uint8)  # noqa: E731
        item = native.NativePairDataset(rel, reader)[0]
        self.assertEqual(sorted(item), ['0', '1', 'type'])
        self.assertEqual((item['0'].shape, item['0'].dtype, item['type'].dtype), ((3, 256, 256), np.float32, np.int64))
        self.assertEqual(int(item['type']), rel.spoof[0][2])

    def test_11_worker_seed_contract(self):
        rel = native.NativeRelation(synthetic_rows())
        batches = [[i % len(rel) for i in range(b * 3, b * 3 + 3)] for b in range(10)]
        seeds = [(12345 + w) % 2 ** 32 for w in range(8)]
        self.assertEqual(native.simulate_live_draws(rel, batches, seeds), native.simulate_live_draws(rel, batches, seeds))
        self.assertNotEqual(native.simulate_live_draws(rel, batches, seeds),
                            native.simulate_live_draws(rel, batches, [(12345 + w) % 2 ** 32 for w in range(4)]))
        self.assertEqual(native.worker_batch_assignment(16)['assignment'][0], [0, 8])
        lc = native.loader_contract(load_method_config('E06b'))
        self.assertEqual((lc['num_workers'], lc['persistent_workers'], lc['drop_last'], lc['batch_size'],
                          lc['num_workers_affects_live_draws']), (8, False, False, 240, True))


class TestM6FBRealPopulation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = real_train_rows()
        if cls.rows is None:
            raise unittest.SkipTest('split metadata / pyarrow unavailable')
        cls.rel = native.NativeRelation(cls.rows)

    def test_12_exact_population(self):
        self.assertTrue(all(r['split'] == 'TRAIN' and r['dataset'] in native.DATASETS for r in self.rows))
        s = self.rel.summary()
        self.assertEqual(s['casia_fasd'], {'spoof': 2520, 'live': 840, 'subjects': 35, 'print': 1680, 'replay': 840})
        self.assertEqual(s['msu_mfsd'], {'spoof': 1200, 'live': 400, 'subjects': 25, 'print': 400, 'replay': 800})
        self.assertEqual(s['total'], {'spoof': 3720, 'live': 1240, 'subjects': 60, 'print': 2080, 'replay': 1640})
        self.assertEqual(len(self.rel), 3720)
        self.assertEqual(len(self.rel.pools), 60)
        self.assertEqual({len(v) for k, v in self.rel.pools.items() if k[0] == 'casia_fasd'}, {24})
        self.assertEqual({len(v) for k, v in self.rel.pools.items() if k[0] == 'msu_mfsd'}, {16})
        self.assertEqual({m for _, _, m in self.rel.spoof}, {0, 1})

    def test_13_real_draws_stay_within_subject(self):
        rng = random.Random(42)
        for i in range(0, len(self.rel), 97):
            self.assertIn(self.rel.draw_live(i, rng), self.rel.pools[self.rel.spoof[i][1]])


class TestM6FBPlanAndHistory(unittest.TestCase):
    def test_14_batch_and_microbatch_plan(self):
        p = native.epoch_plan(3720)
        self.assertEqual((p['batch_sizes'], p['optimizer_steps_per_epoch'], p['chunks_full_batch'],
                          p['chunks_tail_batch']), ([240] * 15 + [120], 16, 12, 6))
        self.assertEqual(p['chunks_per_batch'], [12] * 15 + [6])

    def test_15_no_native_manifest_and_no_writer(self):
        commit = m6fb_commit()
        path = 'manifests/dsdg_identity_pairs_v1.parquet'
        self.assertFalse(git('cat-file', '-e', f'{commit}:{path}').returncode == 0 if commit else (ROOT / path).exists())
        src = at_m6fb('methods/dsdg/native.py').decode()
        for token in ('to_parquet', 'write_table', 'open(', 'FROZEN_LAMBDAS'):
            self.assertNotIn(token, src)

    def test_16_e06c_unchanged(self):
        for rel in ('configs/methods/e06c_dsdg_bin_idfree.yaml', 'configs/frozen/dsdg_bin_idfree_v1.yaml',
                    'methods/dsdg/adapter.py', 'methods/dsdg/training_graph.py', 'methods/dsdg/runner.py',
                    'methods/dsdg/runner_io.py', 'methods/dsdg/microbatch_execution.py',
                    'methods/dsdg/microbatch_execution_v2.py', 'outputs/audit/M6B_CONFIG_FREEZE.json',
                    'tools/m6b_validate_configs.py'):
            self.assertEqual(at_m6fb(rel), at_authority(rel), rel)
        cfg = load_method_config('E06c')
        plan = checkpoint_plan(cfg, 42)
        self.assertEqual((plan['rule'], plan['final_epoch'], plan['authoritative_path_basename']),
                         ('OFFICIAL_GENERATOR_EPOCH_200', 200, 'netG_model_epoch_200_iter_0.pth'))
        with self.assertRaises(FrozenConfigError):
            load_method_config('E06d')

    def test_17_a9_difffas_exclusion_unchanged(self):
        rel = 'configs/amendments/difffas_a9_resource_constrained_scope_exclusion.yaml'
        self.assertEqual(at_m6fb(rel), at_authority(rel))
        a9 = json.loads(at_m6fb(rel))
        self.assertEqual((a9['scope']['methods'], a9['E06b']['status'], a9['m6']['M6_CLOSED']),
                         (['E07c', 'E07b'], 'ACTIVE_M6_WORK', False))

    def test_18_evidence_and_ledger_prefix(self):
        pf = load_preflight()
        if EVIDENCE.is_file():
            ev = pf.check_evidence(at_m6fb, at_authority)
            self.assertEqual((ev['statuses'], ev['m6']), (pf.STATUSES, {'M6_CLOSED': False, 'M7_started': False}))
        prefix, now = at_authority(LEDGER), at_m6fb(LEDGER)
        self.assertEqual(hashlib.sha256(prefix).hexdigest(), pf.LEDGER_PREFIX_SHA)
        self.assertTrue(now.startswith(prefix))
        appended = now[len(prefix):].splitlines()
        self.assertLessEqual(len(appended), 1)
        if appended:
            r = json.loads(appended[0])
            for k, v in pf.LEDGER_EXPECTED.items():
                self.assertEqual(r[k], v, k)

    def test_19_static_only(self):
        self.assertNotIn('torch', sys.modules)


if __name__ == '__main__':
    unittest.main()
