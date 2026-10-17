"""M7D1-N6C: VAL-only best-EMA selection for M7D1-A1 GPAT-B0 / E08 / seed 42 (run 92e91ddb78f1e242).

Static over the committed evidence (no torch, no GPU, no image, no split file): the frozen A10 D14 rule on synthetic
inputs, the 51-row VAL metrics CSV, an independent re-ranking of that CSV, the selection record, the TEST firewall, and
the N6B EMA manifest membership of the selected checkpoint.
"""
import csv
import hashlib
import importlib.util
import io
import json
import math
from fractions import Fraction
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SEL_TOOL = ROOT / 'tools' / 'm7d1_n6c_a1_seed42_selection.py'
EVAL_TOOL = ROOT / 'tools' / 'm7d1_n6c_a1_seed42_val_eval.py'
CSV_PATH = ROOT / 'outputs' / 'audit' / 'M7D1_A1_E08_SEED42_VAL_METRICS.csv'
SEL_PATH = ROOT / 'outputs' / 'audit' / 'M7D1_A1_E08_SEED42_SELECTION.json'
SHA_PATH = ROOT / 'outputs' / 'audit' / 'M7D1_A1_E08_SEED42_N6C_EVIDENCE_SHA256.txt'
EMA_MANIFEST = ROOT / 'outputs' / 'audit' / 'M7D1_A1_E08_SEED42_EMA_MANIFEST.txt'
A10 = ROOT / 'configs' / 'amendments' / 'gpat_a10_m7_contract_resolution.yaml'
EPOCHS = list(range(10, 61))
METRICS = ('ID', 'Dice', 'ArtSim', 'NME', 'LFErr')


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sel = load(SEL_TOOL, 'm7d1_n6c_selection')
ev = load(EVAL_TOOL, 'm7d1_n6c_val_eval')


def synthetic(**over):
    """51 candidates with distinct, monotone-in-epoch metrics (later epoch better on ID only)."""
    rows = []
    for e in EPOCHS:
        r = {'epoch': e, 'checkpoint': f'checkpoints/ema_candidates/ema_epoch_{e:02d}.pt', 'evaluated_val_pairs': 1905,
             'ID': 0.5 + e / 1000, 'Dice': 0.9, 'ArtSim': 0.4, 'NME': 0.01, 'LFErr': 0.0}
        r.update(over.get(e, {}))
        rows.append(r)
    return rows


def independent_ranking(rows):
    """Second implementation: rank_i = #{v_j < v_i} + (#{v_j == v_i} + 1) / 2 (pairwise counting, no sorting)."""
    n = len(rows)
    weights = {'ID': Fraction(1, 4), 'Dice': Fraction(3, 20), 'ArtSim': Fraction(1, 4), 'NME': Fraction(1, 5),
               'LFErr': Fraction(3, 20)}
    sign = {'ID': 1, 'Dice': 1, 'ArtSim': 1, 'NME': -1, 'LFErr': -1}
    scores = {}
    for r in rows:
        s = Fraction(0)
        for m in METRICS:
            v = sign[m] * r[m]
            less = sum(1 for q in rows if sign[m] * q[m] < v)
            eq = sum(1 for q in rows if sign[m] * q[m] == v)
            rank = less + Fraction(eq + 1, 2)
            s += weights[m] * (rank - 1) / (n - 1)
        scores[r['epoch']] = s
    best = max(scores.values())
    tied = [r for r in rows if scores[r['epoch']] == best]
    top_art = max(r['ArtSim'] for r in tied)
    return min(r['epoch'] for r in tied if r['ArtSim'] == top_art), scores


class TestFrozenRule(unittest.TestCase):
    def test_01_weights_frozen_and_sum_to_one(self):
        self.assertEqual({k: float(v) for k, v in sel.WEIGHTS.items()},
                         {'ID': 0.25, 'Dice': 0.15, 'ArtSim': 0.25, 'NME': 0.20, 'LFErr': 0.15})
        self.assertEqual(sum(sel.WEIGHTS.values()), Fraction(1))
        def find(node):
            if isinstance(node, dict):
                if 'D14' in node:
                    return node['D14']
                for v in node.values():
                    hit = find(v)
                    if hit is not None:
                        return hit
            return None

        d14 = find(json.loads(A10.read_text()))
        self.assertEqual(d14['ties'], 'average_rank')
        self.assertEqual(d14['percentile'], 'p = (rank - 1) / (N - 1)')
        self.assertEqual(d14['tie_break'], ['higher_raw_ArtSim', 'earlier_epoch'])
        self.assertEqual(d14['ranked_as'], ['ID', 'Dice', 'ArtSim', 'neg_NME', 'neg_LFErr'])
        self.assertEqual(d14['candidates'], {'ema_epochs': [10, 60], 'inclusive': True, 'N': 51})
        self.assertFalse(d14['test_participates'])

    def test_02_rank_direction_each_metric(self):
        for m, better_high in (('ID', True), ('Dice', True), ('ArtSim', True), ('NME', False), ('LFErr', False)):
            rows = synthetic()
            for r in rows:
                r['ID'] = 0.5                                   # neutralize the default ID ordering
                r[m] = (0.001 * r['epoch']) if better_high else (1.0 - 0.001 * r['epoch'])
            ordered, _ = sel.rank_candidates(rows)
            by = {r['epoch']: r for r in ordered}
            key = f'p_{sel.RANK_NAMES[m]}'
            self.assertEqual(by[60][key], 1, m)                 # epoch 60 is best on m in both directions
            self.assertEqual(by[10][key], 0, m)
            self.assertEqual(ordered[0]['epoch'], 60, m)

    def test_03_nme_lferr_inverted(self):
        rows = synthetic()
        for r in rows:
            r['ID'] = 0.5
            r['NME'] = 0.001 * r['epoch']                       # larger NME at later epochs = worse
            r['LFErr'] = 0.002 * r['epoch']
        ordered, _ = sel.rank_candidates(rows)
        by = {r['epoch']: r for r in ordered}
        self.assertEqual(by[10]['p_neg_NME'], 1)
        self.assertEqual(by[60]['p_neg_NME'], 0)
        self.assertEqual(by[10]['p_neg_LFErr'], 1)
        self.assertEqual(ordered[0]['epoch'], 10)
        self.assertEqual(sel.RANK_NAMES['NME'], 'neg_NME')
        self.assertEqual(sel.ORIENTATION, {'ID': 1, 'Dice': 1, 'ArtSim': 1, 'NME': -1, 'LFErr': -1})

    def test_04_average_rank_ties(self):
        self.assertEqual(sel.average_ranks([3.0, 1.0, 3.0, 2.0]), [Fraction(7, 2), 1, Fraction(7, 2), 2])
        ordered, _ = sel.rank_candidates(synthetic())           # LFErr all 0 -> average rank 26 -> p = 0.5
        self.assertTrue(all(r['rank_neg_LFErr'] == 26 and r['p_neg_LFErr'] == Fraction(1, 2) for r in ordered))

    def test_05_artsim_first_tie_break(self):
        rows = synthetic()
        for r in rows:
            r['ID'] = 0.5
        # epochs 20 and 30 get identical rank profiles except nothing: make them tie on score but differ on raw ArtSim
        # via a swap that keeps every rank multiset equal: ID best for 20, Dice best for 30 with equal weights 0.15?
        # Use ID (0.25) vs ArtSim (0.25): 20 best on ID, 30 best on ArtSim -> equal scores; 30 has higher raw ArtSim.
        for r in rows:
            r['ArtSim'] = 0.4
        rows[EPOCHS.index(20)]['ID'] = 0.9
        rows[EPOCHS.index(30)]['ArtSim'] = 0.8
        ordered, tie = sel.rank_candidates(rows)
        self.assertEqual(ordered[0]['score_exact'], ordered[1]['score_exact'])
        self.assertEqual(ordered[0]['epoch'], 30)
        self.assertTrue(tie['tie_break_required'])
        self.assertEqual(tie['tie_break_stage'], 'higher_raw_ArtSim')
        self.assertEqual(independent_ranking(rows)[0], 30)

    def test_06_earlier_epoch_second_tie_break(self):
        rows = synthetic()
        for r in rows:
            r['ID'] = 0.5
        rows[EPOCHS.index(40)]['Dice'] = 0.95
        rows[EPOCHS.index(25)]['Dice'] = 0.95                   # identical metrics for 25 and 40
        ordered, tie = sel.rank_candidates(rows)
        self.assertEqual(ordered[0]['epoch'], 25)
        self.assertEqual(tie['tie_break_stage'], 'earlier_epoch')
        self.assertEqual(tie['max_score_shared_by_epochs'], [25, 40])
        self.assertEqual(independent_ranking(rows)[0], 25)

    def test_07_rejections(self):
        rows = synthetic()
        with self.assertRaises(sel.SelectionError):
            sel.rank_candidates(rows[:-1])                      # 50 rows
        bad = synthetic()
        bad[0]['epoch'] = 61
        with self.assertRaises(sel.SelectionError):
            sel.rank_candidates(bad)
        dup = synthetic()
        dup[1]['epoch'] = 10
        with self.assertRaises(sel.SelectionError):
            sel.rank_candidates(dup)
        for v in (math.nan, math.inf):
            nf = synthetic()
            nf[5]['NME'] = v
            with self.assertRaises(sel.SelectionError):
                sel.rank_candidates(nf)
        rec = synthetic()
        rec[0]['checkpoint'] = 'checkpoints/recovery/latest.pt'
        with self.assertRaises(sel.SelectionError):
            sel.rank_candidates(rec)
        cnt = synthetic()
        cnt[3]['evaluated_val_pairs'] = 1904
        with self.assertRaises(sel.SelectionError):
            sel.rank_candidates(cnt)


class TestEvaluatorContract(unittest.TestCase):
    def test_08_evaluator_constants(self):
        self.assertEqual(ev.EPOCHS, tuple(EPOCHS))
        self.assertEqual(ev.EVAL_S_HF, 0.15)
        self.assertEqual(ev.GENERATOR_PRECISION, 'R06_FP16_AUTOCAST_E_ART_G_RES')
        self.assertEqual(ev.VAL_PAIRS_SHA256, hashlib.sha256((ROOT / ev.VAL_PAIRS).read_bytes()).hexdigest())
        self.assertEqual(ev.EMA_MANIFEST_DIGEST, hashlib.sha256(EMA_MANIFEST.read_bytes()).hexdigest())
        self.assertEqual(ev.RUN_ROOT, sel.RUN_ROOT)
        self.assertNotEqual(ev.EVAL_ROOT, ev.RUN_ROOT)
        self.assertFalse(ev.EVAL_ROOT.startswith(ev.RUN_ROOT))
        import ast
        tree = ast.parse(EVAL_TOOL.read_text())
        doc = tree.body[0].value if isinstance(tree.body[0], ast.Expr) else None
        consts = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)
                  and n is not doc]
        self.assertFalse([c for c in consts if 'split_v1' in c or 'test_' in c.lower() and 'accessed' not in c])
        self.assertFalse([c for c in consts if 'pairs_train' in c])
        self.assertEqual([c for c in consts if 'latest.pt' in c], ['latest.pt'])     # only the exclusion guard

    def test_09_firewall_denies_non_val(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / 'faces' / 'casia_fasd').mkdir(parents=True)
            (d / 'manifests').mkdir()
            fw = ev.Firewall(d, d / 'faces', {'a' * 64: 'casia_fasd'})
            fw.hook('open', (str(d / 'faces' / 'casia_fasd' / ('a' * 64 + '.png')), 'rb', 0))
            with self.assertRaises(PermissionError):
                fw.hook('open', (str(d / 'faces' / 'casia_fasd' / ('b' * 64 + '.png')), 'rb', 0))
            with self.assertRaises(PermissionError):
                fw.hook('open', (str(d / 'manifests' / 'split_v1.parquet'), 'rb', 0))
            fw.hook('open', (str(d / 'manifests' / 'val_pairs_v1.parquet'), 'rb', 0))
            with self.assertRaises(PermissionError):
                fw.hook('open', (ev.RUN_ROOT + '/metrics.jsonl', 'wb', 0))
            rep = fw.report()
            self.assertEqual(rep['val_image_opens'], 1)
            self.assertEqual(rep['non_val_image_opens'], 1)
            self.assertEqual(rep['val_metadata_opens'], 1)
            self.assertIsNone(rep['test_images_accessed'])      # any denial voids the zero-access claim


@unittest.skipUnless(SEL_PATH.exists(), 'N6C evidence not yet built')
class TestEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.csv_text = CSV_PATH.read_text()
        cls.rows = list(csv.DictReader(io.StringIO(cls.csv_text)))
        cls.sel = json.loads(SEL_PATH.read_text())

    def test_10_exactly_51_rows_epochs_10_60(self):
        self.assertEqual(len(self.rows), 51)
        self.assertEqual([int(r['epoch']) for r in self.rows], EPOCHS)
        self.assertEqual(tuple(self.rows[0].keys()), sel.CSV_COLUMNS)

    def test_11_metrics_finite_complete(self):
        for r in self.rows:
            for m in METRICS:
                self.assertTrue(math.isfinite(float(r[m])), (r['epoch'], m))
            self.assertEqual(r['status'], 'COMPLETE')
            self.assertEqual(int(r['evaluated_val_pairs']), 1905)
        self.assertEqual({float(r['LFErr']) for r in self.rows}, {0.0})       # N-08, gamma = 0

    def test_12_recovery_excluded(self):
        for r in self.rows:
            self.assertEqual(r['checkpoint'], f"checkpoints/ema_candidates/ema_epoch_{int(r['epoch']):02d}.pt")
        self.assertNotIn('recovery', self.csv_text)
        self.assertFalse(self.sel['ema_candidates']['recovery_latest_selectable'])

    def test_13_deterministic_winner_from_csv(self):
        raw = sel.read_csv_rows(self.csv_text)
        ordered, tie = sel.rank_candidates(raw)
        win_ind, scores = independent_ranking(raw)
        self.assertEqual(ordered[0]['epoch'], win_ind)
        self.assertEqual(self.sel['winner']['epoch'], win_ind)
        for r in self.rows:
            e = int(r['epoch'])
            self.assertEqual(r['selection_score_exact'], f'{scores[e].numerator}/{scores[e].denominator}')
            self.assertEqual(float(r['selection_score']), float(scores[e]))
        self.assertEqual(self.sel['tie_break'], tie)
        self.assertEqual([t['epoch'] for t in self.sel['top_5']], [r['epoch'] for r in ordered[:5]])
        self.assertEqual(self.sel['runner_up']['epoch'], ordered[1]['epoch'])
        self.assertEqual(Fraction(self.sel['margin_to_runner_up_exact']),
                         ordered[0]['score_exact'] - ordered[1]['score_exact'])

    def test_14_selected_sha_in_ema_manifest(self):
        w = self.sel['winner']
        lines = [l.split(' ') for l in EMA_MANIFEST.read_text().splitlines() if not l.startswith('#')]
        entry = {int(e): (sha, rel) for e, _, sha, rel in lines}
        self.assertEqual(entry[w['epoch']], (w['checkpoint_sha256'], w['checkpoint']))
        self.assertEqual(self.sel['ema_candidates']['manifest_digest'],
                         hashlib.sha256(EMA_MANIFEST.read_bytes()).hexdigest())
        for r in self.rows:
            self.assertEqual(entry[int(r['epoch'])][0], r['checkpoint_sha256'])

    def test_15_test_access_zero(self):
        fw = self.sel['firewall']
        self.assertEqual((fw['test_images_accessed'], fw['test_metadata_accessed']), (0, 0))
        self.assertEqual(fw['train_images_accessed'], 0)
        for s in fw['sessions']:
            self.assertEqual(s['denied'], [])
            self.assertEqual((s['non_val_image_opens'], s['other_manifest_opens'], s['run_root_write_opens']),
                             (0, 0, 0))
            self.assertEqual((s['test_images_accessed'], s['test_metadata_accessed']), (0, 0))
        self.assertFalse(self.sel['selection_rule']['test_used'])
        self.assertFalse(self.sel['confirmations']['test_accessed'])

    def test_16_reproducibility_and_identity(self):
        rep = self.sel['reproducibility']
        self.assertTrue(rep['csv_reranking_same_winner'])
        for m in METRICS:
            self.assertTrue(rep['winner_full_val_reevaluation'][m]['per_pair_bitwise_equal'], m)
        self.assertEqual(self.sel['run']['run_id'], '92e91ddb78f1e242')
        self.assertEqual(self.sel['evaluators']['tool_sha256'], hashlib.sha256(EVAL_TOOL.read_bytes()).hexdigest())
        self.assertEqual(self.sel['owner_decisions_n6c']['eval_s_hf']['value'], 0.15)
        self.assertEqual(self.sel['evaluators']['config']['eval_s_hf'], 0.15)
        self.assertEqual(self.sel['post_selection_sanity']['label'], 'POST_SELECTION_SANITY_ONLY')
        self.assertFalse(self.sel['post_selection_sanity']['affects_selection'])
        for name, f in self.sel['launcher_logs'].items():
            if f.get('EXIT_STATUS') == '0':
                self.assertEqual(f['run_root_listing_pre'], sel.RUN_ROOT_LISTING_DIGEST, name)
                self.assertEqual(f['run_root_listing_post'], sel.RUN_ROOT_LISTING_DIGEST, name)

    def test_17_evidence_sha256(self):
        for line in SHA_PATH.read_text().splitlines():
            digest, rel = line.split('  ')
            self.assertEqual(hashlib.sha256((ROOT / rel).read_bytes()).hexdigest(), digest, rel)

    def test_18_no_new_training_started(self):
        c = self.sel['confirmations']
        for k in ('training_started', 'seeds_1337_2026_started', 'b1_b2_b3_started', 'bank_generation',
                  'downstream_fas', 'scientific_run_root_modified', 'checkpoint_copied_or_renamed', 'ensemble'):
            self.assertFalse(c[k], k)


if __name__ == '__main__':
    unittest.main()
