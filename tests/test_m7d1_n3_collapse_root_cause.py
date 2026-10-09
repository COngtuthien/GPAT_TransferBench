"""M7D1-N3: evidence checks for the GPAT-B0 / E08 / seed 42 generator identity collapse root-cause localization
(DIAGNOSTIC_ONLY).

Stdlib on any host: validates the committed collected document and re-derives its diagnosis. If torch is importable,
also checks the exact linear Adam-step decomposition used by the attribution against torch.optim.Adam.
"""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'outputs' / 'audit'
COLLECTED = AUDIT / 'M7D1_N3_SEED42_COLLAPSE_COLLECTED.json'
CLOSE = AUDIT / 'M7D1_E08_SEED42_COMPLETION.json'
ROOTCAUSE = AUDIT / 'M7D1_E08_SEED42_COLLAPSE_ROOT_CAUSE.json'
GRADS = AUDIT / 'M7D1_E08_SEED42_COLLAPSE_GRADIENTS.csv'
FORKS = AUDIT / 'M7D1_E08_SEED42_COLLAPSE_FORKS.csv'
REPORT = AUDIT / 'M7D1_N3_SEED42_COLLAPSE_ROOT_CAUSE.md'
SHAFILE = AUDIT / 'M7D1_E08_SEED42_COLLAPSE_EVIDENCE_SHA256.txt'
REQUIRED = (1, 1105, 1971, 5525, 5526, 5700, 5750, 5758, 5759, 5777, 5778, 5779, 5780, 5781, 5800)


def tool():
    spec = importlib.util.spec_from_file_location('m7d1_n3', ROOT / 'tools' / 'm7d1_n3_collapse_root_cause.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class StaticM7D1N3(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = json.loads(COLLECTED.read_text())
        cls.t = tool()

    def test_01_identity_and_labels(self):
        c = self.c
        self.assertEqual((c['method'], c['experiment'], c['seed'], c['scientific_run_id']),
                         ('GPAT-B0', 'E08', 42, '7b799fbd6d0426da'))
        self.assertEqual(c['labels'], self.t.LABELS)
        self.assertIn('DIAGNOSTIC_ONLY', c['labels'])
        self.assertIn('NOT_ELIGIBLE_FOR_VAL_SELECTION', c['labels'])

    def test_02_diagnosis_rederives(self):
        self.assertEqual(self.t.diagnose(self.c), self.c['diagnosis'])
        self.assertEqual(self.c['diagnosis']['status'], 'ROOT_CAUSE_LOCALIZED')

    def test_03_replay_parity(self):
        rp = self.c['replay']
        self.assertEqual(rp['records_compared'], 5850)
        # only the EMA flag differs (harness did not forward step.ema; EMA is write-only), exactly u5526..u5850
        self.assertEqual(rp['differing_fields'], {'ema_updated': [5526, 5850, 325]})
        self.assertEqual(rp['amp_retry_updates_replay'], rp['amp_retry_updates_scientific'])
        self.assertEqual(rp['amp_retry_event_mismatch'], [])
        for u in REQUIRED:
            self.assertTrue(rp['representative'][str(u)]['bitwise_equal_training_fields'], u)
        self.assertEqual(self.c['diagnosis']['parity']['status'], 'PARITY_PASS_TRAINING_STATE')

    def test_04_fork_A_is_the_scientific_trajectory(self):
        p = self.c['forks']['A']['parity_vs_scientific']
        self.assertEqual((p['updates_compared'], p['bitwise_equal_updates']), (325, 325))
        self.assertEqual((p['mismatched_updates'], p['amp_retry_event_mismatch']), ([], []))
        self.assertEqual(self.c['forks']['A']['first_collapse_update'], 5781)

    def test_05_fork_table(self):
        f = self.c['forks']
        for name in ('A', 'B', 'C', 'D', 'E', 'F', 'BDE'):
            self.assertEqual(f[name]['values'], self.t.FORKS[name], name)
            self.assertEqual(f[name]['end'], 5850)
        got = {n: f[n]['first_collapse_update'] for n in f}
        self.assertEqual(got, {'A': 5781, 'B': None, 'C': 5716, 'D': None, 'E': None, 'F': None, 'BDE': None})
        d = self.c['diagnosis']
        self.assertEqual(d['sufficient_single_changes'], ['lambda_adv'])
        self.assertEqual(d['not_sufficient_single_changes'], ['lambda_con', 'lambda_spec', 's_hf'])

    def test_06_attribution_self_checks(self):
        for src, a in self.c['attribution'].items():
            for u, s in a['snapshots'].items():
                if s['attribution_vs_recorded_G_grad_norm_rel'] is not None:
                    self.assertLess(s['attribution_vs_recorded_G_grad_norm_rel'], 1e-2, (src, u))
                if s['next_snapshot_pred_vs_actual_rel_err'] is not None:
                    self.assertLess(s['next_snapshot_pred_vs_actual_rel_err'], 1e-2, (src, u))
                ad = s['additivity']
                self.assertLess(abs(ad['lin_sum_of_parts'] - ad['lin_TOTAL']), 0.05 * abs(ad['lin_TOTAL']) + 0.01,
                                (src, u))
        flip = self.c['diagnosis']['attribution_windows_fork_A']['5777-5780']
        self.assertEqual(flip['largest_closing_part'], 'gadv')
        self.assertGreater(flip['gadv_share_of_closing'], 0.5)

    def test_07_firewall_and_access(self):
        blocks = [self.c['replay']] + list(self.c['attribution'].values()) + list(self.c['forks'].values())
        for b in blocks:
            self.assertEqual(b['write_firewall']['denied'], [])
            self.assertEqual(b['write_firewall']['outside_allowed'], [])
            for k in ('val_images', 'test_images', 'val_metadata', 'test_metadata', 'non_train_images'):
                self.assertEqual(b['access'][k], 0, k)

    def test_08_scientific_metrics_unchanged_reference(self):
        close = json.loads(CLOSE.read_text())
        self.assertIn('5f4aa9a004516fe2297b39e0a847f1d12d786e6f6c54f0bc0c401b0f476d632f', json.dumps(close))


class FinalEvidenceM7D1N3(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t = tool()
        cls.doc = json.loads(ROOTCAUSE.read_text())
        cls.c = json.loads(COLLECTED.read_text())

    def test_01_derived_files_rederive_bytewise(self):
        ver = self.doc['access_and_safety']['scientific_run_root_unchanged']
        j, g, f = self.t.finalize(COLLECTED, ver)
        self.assertEqual(j, ROOTCAUSE.read_text())
        self.assertEqual(g, GRADS.read_text())
        self.assertEqual(f, FORKS.read_text())
        self.assertEqual((j, g, f), self.t.finalize(COLLECTED, ver))        # deterministic

    def test_02_sha256_manifest(self):
        import hashlib
        want = {}
        for line in SHAFILE.read_text().splitlines():
            h, name = line.split('  ')
            want[name] = h
        files = (ROOTCAUSE, GRADS, FORKS, COLLECTED, REPORT)
        self.assertEqual(set(want), {f'outputs/audit/{p.name}' for p in files})
        for p in files:
            self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), want[f'outputs/audit/{p.name}'], p.name)
        self.assertEqual(self.doc['source_collected_sha256'], want[f'outputs/audit/{COLLECTED.name}'])

    def test_03_root_cause_content(self):
        d = self.doc
        self.assertEqual(d['status'], 'ROOT_CAUSE_LOCALIZED')
        self.assertEqual(d['scientific_run']['run_id'], '7b799fbd6d0426da')
        self.assertEqual(d['scientific_run']['authority_commit'], json.loads(CLOSE.read_text())['authority_commit'])
        self.assertEqual(d['classification']['primary'], 'ADVERSARIAL_IMBALANCE / CURRICULUM_TRANSITION_INSTABILITY')
        self.assertEqual(d['classification']['mechanism'], 'MASK_SIGMOID_SATURATION through shared G_res trunk')
        self.assertIn('ONLY within the tested diagnostic window', d['classification'][
            'lambda_adv_sufficient_and_necessary_scope'])
        self.assertEqual((d['timeline']['stage2_transition_update'], d['timeline']['collapse_onset_update']),
                         (5526, 5781))
        self.assertEqual(d['parity']['status'], 'PARITY_PASS_TRAINING_STATE')
        fa = d['parity']['fork_A_corrected_parity']
        self.assertEqual((fa['bitwise_equal_updates'], fa['updates_compared']), (325, 325))
        flip = d['attribution']['flip_5777_5780']
        self.assertLess(flip['gadv_lin_dlogit'], flip['momentum_lin_dlogit'])
        self.assertLess(flip['momentum_lin_dlogit'], flip['fidelity_losses_lin_dlogit_sum'])
        self.assertLess(flip['fidelity_losses_lin_dlogit_sum'], 0)
        self.assertTrue(all(v > 0.8 for v in d['hf_residual_active']['tanh_hf_absmean'].values()))
        a = d['access_and_safety']
        self.assertTrue(a['val_test_access_zero_all_processes'] and a['write_firewall_clean_all_processes'])
        self.assertTrue(a['scientific_run_root_unchanged']['unchanged'])
        close = json.dumps(json.loads(CLOSE.read_text()))
        for h in a['scientific_run_root_unchanged']['sha256'].values():
            self.assertIn(h, close)
        self.assertEqual((a['bank'], a['seed1337_2026'], a['B1_B2_B3'], a['val_selection'], a['fix']),
                         ('NOT_TOUCHED', 'NOT_LAUNCHED', 'NOT_LAUNCHED', 'NOT_RUN', 'NOT_IMPLEMENTED'))
        self.assertGreaterEqual(len(d['limitations']), 8)

    def test_04_gradients_csv_schema_and_coverage(self):
        import csv
        rows = list(csv.DictReader(GRADS.open()))
        self.assertEqual(tuple(rows[0].keys()), self.t.GRAD_FIELDS)
        keys = {(r['source'], int(r['update']), r['loss_term']) for r in rows}
        self.assertEqual(len(keys), len(rows))
        for src, a in self.c['attribution'].items():
            for u in a['snapshots']:
                for k in list(self.t.TERMS) + list(self.t.ADAM_PARTS):
                    self.assertIn((src, int(u), k), keys)
        for u in range(5757, 5783):
            self.assertIn(('fork_A', u, 'gadv'), keys)
        dirs = {r['mask_direction'] for r in rows}
        self.assertLessEqual(dirs, {'OPEN_MASK', 'CLOSE_MASK', 'ZERO_WEIGHT', ''})

    def test_05_forks_csv(self):
        import csv
        rows = {r['fork_id']: r for r in csv.DictReader(FORKS.open())}
        self.assertEqual(tuple(next(iter(rows.values())).keys()), self.t.FORK_FIELDS)
        self.assertEqual(set(rows), {'A', 'B', 'C', 'D', 'E', 'F', 'BDE'})
        self.assertEqual({k: (r['collapse'], r['collapse_update']) for k, r in rows.items()},
                         {'A': ('true', '5781'), 'B': ('false', ''), 'C': ('true', '5716'), 'D': ('false', ''),
                          'E': ('false', ''), 'F': ('false', ''), 'BDE': ('false', '')})
        self.assertEqual(rows['A']['parity_vs_scientific'], '325/325')
        for r in rows.values():
            self.assertEqual((r['start_update'], r['end_update'], r['records']), ('5526', '5850', '325'))


class AdamDecomposition(unittest.TestCase):
    def test_parts_sum_to_torch_adam_step(self):
        try:
            import torch
        except ImportError:
            self.skipTest('torch not available')
        t = tool()
        from methods.gpat import runtime_contract as rc
        torch.manual_seed(0)
        e = torch.nn.Linear(3, 2)
        end = torch.nn.Conv2d(4, 13, 3)
        named = [('e_art.lin.weight', e.weight), ('e_art.lin.bias', e.bias),
                 ('g_res.ending.weight', end.weight), ('g_res.ending.bias', end.bias)]
        opt = torch.optim.Adam([p for _, p in named], lr=1e-3, betas=(0.5, 0.999), weight_decay=0.0)
        for _ in range(3):
            for _, p in named:
                p.grad = torch.randn_like(p) * 1e-3
            opt.step()
        u = 5600
        for g in opt.param_groups:
            g['lr'] = rc.main_lr(u)
        gsl = [(n, p.numel()) for n, p in named]
        offs, o = {}, 0
        for n, k in gsl:
            offs[n] = (o, o + k)
            o += k
        unit = {k: torch.randn(o) * 0.1 for k in t.TERMS}
        W = {k: (0.0 if k == 'low' else 0.05 * (i + 1)) for i, k in enumerate(t.TERMS)}
        clip = 0.3
        tr = SimpleNamespace(step=SimpleNamespace(g_named=named), g_opt=opt)
        parts, pred, info = t.adam_parts(tr, u, unit, W, clip, [True] * len(named), offs, gsl)
        g = clip * sum(W[k] * unit[k] for k in t.TERMS)
        theta0 = torch.cat([p.detach().reshape(-1) for _, p in named]).clone()
        for (n, p) in named:
            a, b = offs[n]
            p.grad = g[a:b].view_as(p).clone()
        opt.step()
        actual = torch.cat([p.detach().reshape(-1) for _, p in named]) - theta0
        self.assertTrue(torch.allclose(parts['TOTAL'], actual, rtol=1e-4, atol=1e-7))   # theta1 - theta0 in fp32: ulp(0.5) ~ 6e-8
        self.assertTrue(torch.allclose(pred, actual, rtol=1e-4, atol=1e-7))
        lin = sum(v for k, v in parts.items() if not k.startswith('TOTAL'))
        self.assertTrue(torch.allclose(lin, parts['TOTAL'], rtol=1e-6, atol=1e-12))
        split = parts['TOTAL_mask_head_row_only'] + parts['TOTAL_g_res_except_mask_row'] + parts['TOTAL_e_art_only']
        self.assertTrue(torch.allclose(split, parts['TOTAL'], atol=1e-12))
        self.assertEqual(info['adam_t'], 4.0)


if __name__ == '__main__':
    unittest.main()
