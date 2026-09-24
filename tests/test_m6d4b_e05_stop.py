"""M6D4b STOP integrity and rejection tests; no training evidence is simulated."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from tools import m6d4b_e05_training_preflight as pre


class TestM6D4bStop(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit = json.loads(pre.read(pre.BASE+'.json'))

    def test_complete_stop_evidence(self):
        pre.validate_stop(self.audit)

    def test_false_qualification_rejected(self):
        a = deepcopy(self.audit)
        a['final_status'][1] = 'E05_TRAINING_RUNNER_QUALIFIED'
        with self.assertRaisesRegex(ValueError, 'qualification status'):
            pre.validate_stop(a)

    def test_missing_scientific_blocker_rejected(self):
        a = deepcopy(self.audit)
        a['phase_a']['blockers'].remove('UPDATE_SCHEDULE')
        with self.assertRaisesRegex(ValueError, 'blockers retained'):
            pre.validate_stop(a)

    def test_incomplete_phase_a_rejected(self):
        a = deepcopy(self.audit)
        a['phase_a']['evidence_table'].pop()
        with self.assertRaisesRegex(ValueError, 'A-J evidence'):
            pre.validate_stop(a)

    def test_fabricated_optimizer_execution_rejected(self):
        a = deepcopy(self.audit)
        a['execution']['optimizer_applications'] = 1
        with self.assertRaisesRegex(ValueError, 'optimizer execution'):
            pre.validate_stop(a)

    def test_fabricated_loss_measurement_rejected(self):
        a = deepcopy(self.audit)
        a['execution']['initial_five_losses'] = {'L_rec': 0.0}
        with self.assertRaisesRegex(ValueError, 'unexecuted evidence'):
            pre.validate_stop(a)

    def test_process_records_explicitly_not_run(self):
        for number, path in enumerate(pre.PROCESSES, 1):
            record = json.loads(pre.read(path))
            pre.validate_process(record, number)
            record['launched'] = True
            with self.assertRaisesRegex(ValueError, 'fabricated process'):
                pre.validate_process(record, number)

    def test_archive_hash_corruption_rejected(self):
        sync = deepcopy(self.audit['gpu_reconciliation'])
        path = next(iter(sync['files']))
        sync['files'][path]['archive_sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'archive/worktree/commit'):
            pre.validate_sync(sync)

    def test_data_paths_rejected_before_open(self):
        with patch.object(Path, 'read_bytes', side_effect=AssertionError('must not open')):
            for path in ('manifests/split_v1.parquet', 'manifests/pairs_train_v1.parquet',
                         'outputs/data/forbidden.json', 'outputs/faces_256/forbidden.jpg',
                         '/home/forbidden', '../forbidden', 'outputs/forbidden.pt'):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    pre.read(path)

    def test_index_only_opens_explicit_new_artifacts(self):
        seen = []
        original = pre.read
        def checked(path):
            self.assertIn(path, pre.ARTIFACTS)
            seen.append(path)
            return original(path)
        with patch.object(pre, 'read', side_effect=checked):
            raw, count = pre.expected_index()
        self.assertEqual(set(seen), set(pre.ARTIFACTS))
        self.assertEqual(count, 550)
        self.assertEqual(raw.count(b'\n'), raw.count(b'\r\n'))


if __name__ == '__main__':
    unittest.main()
