"""E03 seed-completion audit, owner ruling 2026-10-29 (OWNER_APPROVED_POST_COMPLETION_REBOOT_EVIDENCE_SUBSTITUTION):
regression tests of the post-completion-reboot evidence path of B.process_exited_zero and J.single_e03_session.

Stdlib only, any host. Uses the committed seed1337 (live tmux pane) and seed2026 (tmux removed by a clean reboot after
completion) collections and mutates copies of them; every mutation that breaks one durable-evidence condition must make
both substituted gates fail closed, and no other gate may change behaviour.
"""
import copy
from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'outputs' / 'audit'
ALT = ['B.process_exited_zero', 'J.single_e03_session']
PASS = 'PASS_BY_APPROVED_POST_COMPLETION_REBOOT_EVIDENCE'


def tool():
    spec = importlib.util.spec_from_file_location('e03_seed_audit', ROOT / 'tools' / 'e03_seed_completion_audit.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def wtmp_time(ts):
    return datetime.fromtimestamp(ts, timezone(timedelta(hours=7))).strftime('%Y-%m-%dT%H:%M:%S+07:00')


class RebootAmendment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = tool()
        cls.docs = {s: json.loads((AUDIT / ('E03_SEED%d_COMPLETION.json' % s)).read_text()) for s in (1337, 2026)}
        cls.end = cls.mod.epoch_seconds(cls.docs[2026]['collected']['run_manifest']['end_utc'])

    def collections(self, seed=2026):
        d = self.docs[seed]
        return copy.deepcopy(d['collected']), d['collected_tf']

    def build(self, c, t, root=ROOT):
        doc, _ = self.mod.build(c, t, root)
        return doc, {g['gate']: g for g in doc['gates']}

    def assert_rejected(self, c, t, check, root=ROOT):
        doc, g = self.build(c, t, root)
        for name in ALT:
            self.assertFalse(g[name]['pass'], name)
            self.assertEqual(g[name]['detail']['result'], 'FAIL_POST_COMPLETION_REBOOT_EVIDENCE')
            if check in g[name]['detail']['checks']:
                self.assertFalse(g[name]['detail']['checks'][check]['pass'], (name, check))
        self.assertEqual(doc['status'], 'E03_SEED2026_NEEDS_REVIEW')
        self.assertEqual(doc['gates_passed_by_approved_alternative'], [])
        self.assertNotIn('post_completion_reboot', doc.get('disclosures', {}))
        return doc, g

    def test_01_live_tmux_path_unchanged(self):
        doc, g = self.build(*self.collections(1337))
        self.assertEqual((doc['gates_passed'], doc['gates_total']), (88, 88))
        self.assertEqual(g['B.process_exited_zero'], {'gate': 'B.process_exited_zero', 'pass': True})
        self.assertEqual(g['J.single_e03_session'], {'gate': 'J.single_e03_session', 'pass': True})
        self.assertNotIn('gates_passed_by_approved_alternative', doc)
        self.assertNotIn('post_completion_reboot', doc['disclosures'])
        self.assertEqual(json.loads(json.dumps(doc)), self.docs[1337])

    def test_02_valid_completion_then_verified_reboot(self):
        doc, g = self.build(*self.collections())
        self.assertEqual((doc['status'], doc['gates_passed'], doc['gates_total']),
                         ('E03_SEED2026_COMPLETED_CLEANLY', 88, 88))
        self.assertEqual(doc['gates_passed_by_approved_alternative'], ALT)
        for name in ALT:
            self.assertEqual(g[name]['detail']['result'], PASS)
            self.assertEqual(sorted(g[name]['detail']['checks']), sorted(self.mod.REBOOT_GATE_CHECKS[name]))
        self.assertEqual(json.loads(json.dumps(doc)), self.docs[2026])

    def test_03_reboot_before_or_during_completion(self):
        c, t = self.collections()
        rp = c['host']['reboot_provenance']
        boots = {b['index']: b for b in rp['journal_boots']}
        boots[-1]['last_entry'] = int((self.end - 1000) * 1e6)          # run boot ends before the run did
        boots[0]['first_entry'] = int((self.end - 900) * 1e6)
        rp['btime'] = int(self.end - 900)
        rp['wtmp'][1] = rp['wtmp'][1].replace('2026-10-29T09:44:06+07:00', wtmp_time(self.end - 1000))
        self.assert_rejected(c, t, 'reboot_after_completion')
        c, t = self.collections()                                         # kernel boot time alone before completion
        c['host']['reboot_provenance']['btime'] = int(self.end - 10)
        self.assert_rejected(c, t, 'reboot_after_completion')

    def test_04_missing_reboot_provenance(self):
        for key in ('journal_boots', 'wtmp', 'journal_tmux_scopes'):
            c, t = self.collections()
            c['host']['reboot_provenance'][key] = None
            self.assert_rejected(c, t, {'journal_boots': 'reboot_after_completion', 'wtmp': 'clean_shutdown_recorded',
                                        'journal_tmux_scopes': 'pane_scope_exited_at_completion'}[key])
        c, t = self.collections()
        del c['host']['reboot_provenance']
        self.assert_rejected(c, t, 'reboot_after_completion')
        c, t = self.collections()                                          # no clean `shutdown system down` record
        rp = c['host']['reboot_provenance']
        rp['wtmp'] = [x for x in rp['wtmp'] if not x.startswith('shutdown')]
        self.assert_rejected(c, t, 'clean_shutdown_recorded')

    def test_05_nonzero_or_malformed_exit_marker(self):
        for raw in ('1\n', '0', '00\n', ''):
            c, t = self.collections()
            c['host']['exit_file_raw'], c['host']['exit_file'] = raw, raw.strip()
            self.assert_rejected(c, t, 'exit_marker_zero')

    def test_06_missing_or_inconsistent_end_timestamp(self):
        for value in ('', 'not-a-time', '2026-10-28T18:52:20Z', '2026-10-28T19:52:21Z'):
            c, t = self.collections()
            c['host']['end_time'] = value
            self.assert_rejected(c, t, 'end_marker_valid')
        c, t = self.collections()
        c['host']['marker_files']['end_time'] = None
        self.assert_rejected(c, t, 'marker_files_durable')

    def test_07_invalid_completion_record(self):
        root = self.docs[2026]['run_root']
        for last in (['{"status": "failed", "run_dir": "%s"}' % root], ['{"status": "completed", "run_dir": "/x"}'],
                     ['^[[Z{"status": "completed", "run_dir": "%s"}' % root], []):
            c, t = self.collections()
            c['host']['pane_log_last'] = last
            self.assert_rejected(c, t, 'completion_record')
        c, t = self.collections()
        c['host']['pane_log_fail_hits']['Traceback'] = 1
        self.assert_rejected(c, t, 'completion_record')

    def test_08_run_id_mismatch(self):
        for where in ('run_summary', 'run_manifest'):
            c, t = self.collections()
            c[where]['run_id'] = 'e8c85cc378e778f5'
            self.assert_rejected(c, t, 'run_identity_consistent')

    def test_09_active_process_after_completion(self):
        c, t = self.collections()
        c['host']['runner_processes'] = ['4242 /home/student20261/miniconda3/envs/gpat-m6-e03/bin/python -u -B '
                                         'tools/run_e03.py --seed 2026 --execution-config configs/execution/'
                                         'm5_gpu_3090.yaml']
        self.assert_rejected(c, t, 'no_runner_process')

    def test_10_missing_or_modified_raw_evidence(self):
        after_reboot = self.mod.epoch_seconds('2026-10-29T03:00:00Z')
        for key, field, value in (('pane_log', 'mtime', after_reboot), ('exit', 'mtime', after_reboot),
                                  ('command', 'sha256', '0' * 64), ('command', 'mtime', after_reboot),
                                  ('exit', 'size', 3)):
            c, t = self.collections()
            c['host']['marker_files'][key][field] = value
            self.assert_rejected(c, t, 'marker_files_durable')
        for key in ('pane_log', 'exit', 'command', 'launch_time'):
            c, t = self.collections()
            c['host']['marker_files'][key] = None
            self.assert_rejected(c, t, 'marker_files_durable')
        c, t = self.collections()                                       # the run's pane scope never stopped
        sc = c['host']['reboot_provenance']['journal_tmux_scopes']
        c['host']['reboot_provenance']['journal_tmux_scopes'] = [x for x in sc if 'c2f56e5d' not in x['message'] or
                                                                 x['message'].startswith('Started')]
        self.assert_rejected(c, t, 'pane_scope_exited_at_completion')
        with tempfile.TemporaryDirectory() as tmp:                      # the first NEEDS_REVIEW build is not kept
            self.assert_rejected(*self.collections(), 'prior_diagnostic_preserved', root=Path(tmp))

    def test_11_absent_session_without_verified_reboot(self):
        c, t = self.collections()                       # one boot spans everything: the session vanished, no reboot
        rp = c['host']['reboot_provenance']
        run_boot = next(b for b in rp['journal_boots'] if b['index'] == -1)
        rp['journal_boots'] = [dict(run_boot, index=0, last_entry=int(self.end * 1e6) + 10 ** 11)]
        rp['boot_id'] = run_boot['boot_id']
        rp['btime'] = int(run_boot['first_entry'] / 1e6)
        rp['wtmp'] = [x for x in rp['wtmp'] if not x.startswith('shutdown system down  7.0.0-38-generic 2026-10-29')]
        self.assert_rejected(c, t, 'reboot_after_completion')
        for sessions, other in ((['e03_stdn_seed42'], ['e03_stdn_seed42']), (['e03_stdn_seed2026_x'], [])):
            c, t = self.collections()                   # some E03 tmux session exists
            c['host']['reboot_provenance']['tmux_ls'] = {'rc': 0, 'sessions': sessions, 'stderr': ''}
            c['host']['other_e03_tmux'] = other
            self.assert_rejected(c, t, 'live_tmux_state_absent')
        c, t = self.collections()                       # a second tmux pane launched inside the run window
        sc = c['host']['reboot_provenance']['journal_tmux_scopes']
        own = next(x for x in sc if x['message'].startswith('Started tmux-spawn-c2f56e5d'))
        sc.append(dict(own, realtime=own['realtime'] + 3600,
                       message=own['message'].replace('c2f56e5d', 'deadbeef')))
        self.assert_rejected(c, t, 'single_pane_scope_in_run_window')

    def test_12_scope_restricted_to_two_gates(self):
        self.assertEqual(self.mod.REBOOT_RULING['gates'], ALT)
        self.assertEqual(sorted(self.mod.REBOOT_GATE_CHECKS), ALT)
        reboot_doc, rg = self.build(*self.collections())
        alt = [n for n, g in rg.items() if isinstance(g.get('detail'), dict) and 'result' in g['detail']]
        self.assertEqual(alt, ALT)
        # the same run with live tmux evidence: live path, and every other gate is byte-identical
        c, t = self.collections()
        c['host'].update(tmux_pane_dead=True, tmux_pane_dead_status=0, other_e03_tmux=['e03_stdn_seed2026'])
        live_doc, lg = self.build(c, t)
        self.assertEqual(live_doc['gates_passed'], 88)
        self.assertNotIn('gates_passed_by_approved_alternative', live_doc)
        self.assertEqual(list(lg), list(rg))
        self.assertEqual({n: g for n, g in lg.items() if n not in ALT}, {n: g for n, g in rg.items() if n not in ALT})
        # a pane that exists but is alive or failed never falls back to the reboot path
        for dead, status in ((False, None), (True, 1)):
            c, t = self.collections()
            c['host'].update(tmux_pane_dead=dead, tmux_pane_dead_status=status, other_e03_tmux=['e03_stdn_seed2026'])
            doc, g = self.build(c, t)
            self.assertEqual(g['B.process_exited_zero'], {'gate': 'B.process_exited_zero', 'pass': False})
            self.assertNotIn('gates_passed_by_approved_alternative', doc)
        # any other (scientific) gate failure blocks the substitution
        c, t = self.collections()
        c['atime_firewall']['by_class']['casia_fasd|TEST']['atime_changed'] = 1
        doc, g = self.assert_rejected(c, t, 'other_gates_pass')
        self.assertEqual(doc['failed_gates'], ['B.process_exited_zero', 'H.test_zero', 'J.single_e03_session'])
        self.assertEqual(g['B.process_exited_zero']['detail']['checks']['other_gates_pass']['failed'], ['H.test_zero'])


if __name__ == '__main__':
    unittest.main()
