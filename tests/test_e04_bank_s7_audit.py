"""E04 Physics-STD S7 audit qualification (tools/learned_bank_s7_audit.py E04 branch).

1. Builder: a synthetic collected document with the exact host-collector schema, built from the real committed
   records (contract, family closure, final-checkpoint CSV, toy manifest, fixture-qualified graph identity, runtime
   session record), must PASS every gate; each of the owner's negative cases must fail the named gate(s) and never
   yield E04_S7_ALL_THREE_SEEDS_QUALIFIED.
2. Collector helpers on real data: PNG structure / CRC / IHDR / zlib checks and the pixel identity on real PNG bytes,
   the A/B byte + decoded-array comparison on real directories, the split_v1 / M2 cross-check on real parquet rows,
   and the code-change scope on real git history.
3. The owner-ruled adapter fix (feed-pack TRAIN allowlist, no pyarrow in the inference process).
Laptop .venv (numpy, Pillow, pyarrow, git); no TensorFlow, no GPU, no faces.
"""
import copy
import csv
import hashlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from methods.common import learned_bank as lb  # noqa: E402


def tool():
    spec = importlib.util.spec_from_file_location('s7_audit_e04', ROOT / 'tools' / 'learned_bank_s7_audit.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = tool()
FAKE = 'feedface' * 5                       # synthetic execution commit (the overlay view serves the working tree)
S7 = '/home/student20261/workdir/m8_learned_banks/s7/' + FAKE
REPO = '/home/student20261/workdir/m8_learned_banks/clones/s7e04_' + FAKE
APPROVAL = {'owner_authorized': True, 'scope': 'E04 S7 toy-bank acceptance', 'date': '2026-11-25',
            'instruction': 'test approval record'}


def sha(b):
    return hashlib.sha256(b if isinstance(b, bytes) else b.encode()).hexdigest()


def authority_bytes(approval=None):
    real = ROOT / M.E04_AUTHORITY_PATH
    if approval is None and real.is_file():
        approval = json.loads(real.read_text())['approval']
    draft = (ROOT / M.E04_AUTHORITY_DRAFT_PATH).read_bytes()
    return (json.dumps(M.derive_e04_s7_authority(draft, approval or APPROVAL), indent=1, sort_keys=True) + '\n').encode()


class View(M.GitView):
    """Working tree + overlay; the synthetic execution commit FAKE serves the same bytes."""

    def __init__(self, overlay=None):
        super().__init__(ROOT)
        self.overlay = {M.E04_AUTHORITY_PATH: authority_bytes()}
        self.overlay.update(overlay or {})

    def read(self, rel):
        return self.overlay[rel] if rel in self.overlay else super().read(rel)

    def show(self, commit, rel):
        return self.read(rel) if commit == FAKE else super().show(commit, rel)

    def is_ancestor(self, commit):
        return commit == FAKE or super().is_ancestor(commit)


def iso(minutes):
    h, m = divmod(20 * 60 + minutes, 60)
    return '2026-11-25T%02d:%02d:00Z' % (h, m)


def synthetic_collected(view=None):
    v = view or View()
    auth_sha = sha(v.read(M.E04_AUTHORITY_PATH))
    contract_sha = sha(v.read(M.E04_CONTRACT_PATH))
    closure_sha = sha(v.read(M.E04_CLOSURE_PATH))
    records = {M.E04_CONTRACT_PATH: contract_sha, M.E04_AUTHORITY_PATH: auth_sha, M.E04_CLOSURE_PATH: closure_sha}
    ck = {int(r['seed']): {'data-00000-of-00001': r['final_data_sha256'], 'index': r['final_index_sha256'],
                           'meta': r['final_meta_sha256']}
          for r in csv.DictReader(io.StringIO((ROOT / M.E04_FINAL_CKPT_CSV).read_text()))}
    toy = json.loads((ROOT / M.TOY_MANIFEST[0]).read_text())['toy_pairs']
    fixture = json.loads((ROOT / M.E04_FIXTURE_RECORD).read_text())
    session = json.loads((ROOT / M.E04_RUNTIME_RECORD).read_text())['session_config']
    stdn = json.loads((ROOT / lb.SOURCES_PATH).read_text())['families']['STDN']['commit']
    face = {}
    for t in toy:
        face[t['target_live_id']] = sha('face|' + t['target_live_id'])
        face[t['source_spoof_id']] = sha('face|' + t['source_spoof_id'])

    def ck_files(s):
        return {k: {'path': '%s/runs/m6/E04/seed_%d/%s' % (M.RUNTIME, s, M.E04_CKPT_FILES[k]), 'sha256': h,
                    'size': 1000 + len(k), 'mtime_ns': 1764000000000000000 + s} for k, h in ck[s].items()}

    def ck_record(s):
        return {'ok': True, 'identity': M.checkpoint_identity(ck[s]), 'run_id': M.E04_RUN_IDS[s], 'files': ck_files(s)}
    hdd = ['E01', 'E02', 'E03', 'E05', 'E06b', 'E06c']
    t0, wmin = 0, {}
    for s, L in M.E04_PROCS:
        wmin[(s, L)] = (t0 + 2, t0 + 4)         # launch, end (minutes); preflight one minute before launch
        t0 += 5
    windows = {k: (iso(a), iso(b)) for k, (a, b) in wmin.items()}
    struct = M.canonical(dict(M.E04_PNG_STRUCTURE, scanline_filters_ok=True))
    c = {'phase': 'collect', 'method': 'E04', 'layout': M.E04_PROCESS_LAYOUT, 'utc': iso(t0 + 10), 'repo': REPO,
         'git': {'head': FAKE, 'branch': 'm6-baselines', 'status_porcelain': ''},
         'collector': {'python': '3.11.16', 'executable': '/x/gpat-m6-e05/bin/python'},
         's7_root': S7, 'out_root': S7 + '/E04', 'out_root_entries': ['seed_1337', 'seed_2026', 'seed_42'],
         'seed_root_entries': {str(s): ['run_A', 'run_B'] for s in M.SEEDS},
         'runs': {'A': {}, 'B': {}}, 'replay': {}, 'library_reader': {}, 'markers': {}, 'preflights': {},
         'preflights_sha256': {}, 'cli_compare': {}, 'atime': {}, 'checkpoints_after': {},
         'hdd_bank_root_entries': hdd, 'processes_after': [], 'gpu_after': {}, 'run_roots_unchanged_during_collect': True}
    samples = {}
    for s in M.SEEDS:
        rows = []
        for t in toy:
            pid = t['pair_id']
            samples[t['target_live_id']] = ['TRAIN', t['dataset'], face[t['target_live_id']], 'COMPLETE']
            samples[t['source_spoof_id']] = ['TRAIN', t['dataset'], face[t['source_spoof_id']], 'COMPLETE']
            r = {'schema': 'gpat_learned_bank_v1', 'row_index': t['position'], 'row_id': pid, 'method_id': 'E04',
                 'bank_seed': s, 'generator_seed': s, 'mode': 'S7_TOY', 'label': 'spoof', 'synthetic_id': None,
                 'pair_id': pid, 'target_live_id': t['target_live_id'], 'source_spoof_id': t['source_spoof_id'],
                 'dataset': t['dataset'], 'target_face_sha256': face[t['target_live_id']],
                 'source_face_sha256': face[t['source_spoof_id']], 'deterministic_noise_seed': None,
                 'dsdg_batch_index': None, 'dsdg_batch_offset': None,
                 'checkpoint_identity_sha256': M.checkpoint_identity(ck[s]), 'config_sha256': M.E04_CONFIG[1],
                 'contract_sha256': M.E04_CONTRACT_SHA256, 'conversion_rule': M.E04_CONVERSION_RULE,
                 'generation_status': 'success', 'failure_reason': None, 'output_path': 'bank/%s.png' % pid,
                 'output_sha256': sha('png|%d|%s' % (s, pid)), 'output_pixel_sha256': sha('px|%d|%s' % (s, pid)),
                 'raw_output_finite': True, 'raw_output_min': -0.01, 'raw_output_max': 1.02}
            rows.append(r)
        body = ''.join(M.canonical(r) + '\n' for r in rows).encode('ascii')
        man_sha = sha(body)
        png_files = {r['pair_id'] + '.png': [r['output_sha256'], r['output_pixel_sha256'], 50000] for r in rows}
        c['markers'][str(s)], c['preflights'][str(s)], c['preflights_sha256'][str(s)] = {}, {}, {}
        for L in M.LABELS:
            run_dir = str(M.e04_run_dir(S7, s, L))
            listing = [['bank', 'dir', None, '0o555']] + [['bank/' + n, v[2], v[0], '0o444'] for n, v in
                                                          sorted(png_files.items())] + [
                ['bank_manifest.jsonl', len(body), man_sha, '0o444'],
                ['bank_manifest.sha256', 85, sha('x'), '0o444'], ['bank_provenance.json', 9000, sha(L), '0o444']]
            prov = {
                'schema': 'gpat_learned_bank_v1', 'record': 'bank_provenance', 'method_id': 'E04', 'bank_seed': s,
                'mode': 'S7_TOY', 'git_head': FAKE, 'run_dir': run_dir, 'start_utc': windows[(s, L)][0],
                'end_utc': windows[(s, L)][1], 'wall_time_seconds': 120.0 + (L == 'B'),
                'authorization': {'mode': 'S7_TOY', 'methods': ['E04'], 'seeds': [42, 1337, 2026],
                                  'contract_sha256': M.E04_CONTRACT_SHA256, 'label': 's7_toy', 'records': records},
                'population': {'relation_source': 'e04_feed_pack_v1 pairs_train.json',
                               'pack_manifest_sha256': M.E04_PACK_MANIFEST_SHA256,
                               'allowlist_source': M.E04_ALLOWLIST_SOURCE, 'pairs_val_opened': False,
                               'toy_manifest': M.TOY_MANIFEST[0], 'toy_manifest_sha256': M.TOY_MANIFEST[1],
                               'method_id': 'E04', 'seed': s, 'kind': 'S7_TOY_POPULATION', 'rows': 186,
                               'row_ids_sha256': sha('\n'.join(t['pair_id'] for t in toy)),
                               'first_index': toy[0]['position'], 'last_index': toy[-1]['position']},
                'adapter': {'checkpoint': {'method_id': 'E04', 'seed': s, 'run_id': M.E04_RUN_IDS[s],
                                           'rule': 'BASELINE_FINAL_STATE_V1', 'files': ck_files(s),
                                           'checkpoint_identity_sha256': M.checkpoint_identity(ck[s]),
                                           'seed_closure_commit': 'x', 'family_closure_commit': 'eeff32d'},
                            'source': {'files': dict(M.E04_FROZEN_FILES), 'synthesis_lines': [32, 36, 37, 39],
                                       'predecessor_reference': {'commit': stdn,
                                                                 'lines': ['model/utils.py:24', 'model/utils.py:37']}},
                            'coverage': copy.deepcopy(fixture['real_checkpoints'][str(s)]['coverage']),
                            'moving_statistics': copy.deepcopy(fixture['real_checkpoints'][str(s)]['moving_statistics']),
                            'driver': {'lock_driver_version': '595.84', 'observed_driver_version': M.DRIVER,
                                       'driver_policy': 'OWNER_APPROVED_DRIVER_VERSION_ONLY_RUNTIME_COMPATIBILITY',
                                       'gpu_model': M.GPU_NAME, 'gpu_uuid': M.GPU_UUID},
                            'environment_session_config': session, 'framework': 'tensorflow 1.15.5',
                            'graph': copy.deepcopy(fixture['key_results']['graph']), 'method_id': 'E04',
                            'bank_seed': s, 'generator_seed': s, 'is_fixture': False,
                            'checkpoint_identity_sha256': M.checkpoint_identity(ck[s]), 'config_sha256': M.E04_CONFIG[1],
                            'contract_sha256': M.E04_CONTRACT_SHA256, 'decisions': dict(M.E04_DECISIONS),
                            'conversion_rule': M.E04_CONVERSION_RULE},
                'environment': {'python_executable': M.E04_PYTHON, 'python_version': '3.8.20', 'numpy': '1.23.5',
                                'pillow': '10.4.0', 'host': 'nd-System-Product-Name', 'cuda_visible_devices': '0'},
                'status': 'completed', 'rows': 186, 'outputs': 186, 'failure_count': 0, 'failures': [],
                'manifest_sha256': man_sha,
                'adapter_finalize': {'forwards': 186, 'batch_size': 1, 'stochastic_at_inference': False,
                                     'auxiliary_outputs_written': False, 'bn_mode': 'inference_moving_statistics'},
                'reader': {'faces_root': M.E04_FACES_ROOT, 'allowlisted_samples': 300, 'reads': 372,
                           'atime_checked_reads': 372, 'distinct_samples_read': 300, 'splits_allowlisted': ['TRAIN'],
                           'o_noatime': True},
                'post_write_verification': {'ok': True, 'rows': 186, 'files_checked': 186, 'problems': []}}
            c['runs'][L][str(s)] = {
                'dir': run_dir, 'provenance': prov, 'provenance_sha256': sha(L + str(s)), 'aborted': False,
                'listing': listing, 'listing_sha256': sha(json.dumps(listing)), 'dir_mode': '0o555', 'bank_dir_mode': '0o555',
                'manifest': {'sha256': man_sha, 'bytes': len(body), 'lines': 186, 'ends_with_newline': True,
                             'canonical_lines': True, 'fields_ok': True,
                             'sha_file': '%s  bank_manifest.jsonl\n' % man_sha},
                'manifest_rows': [[r[k] for k in M.MANIFEST_FIELDS_E04] for r in rows],
                'png_structures': {struct: 186}, 'png_files': copy.deepcopy(png_files)}
            a, e = windows[(s, L)]
            line = M.e04_command_line(REPO, S7, s, L)
            c['markers'][str(s)][L] = {
                'command': '#!/bin/bash\n# M8 S7 E04 seed %d run %s\ndate\n%secho $?\n' % (s, L, line),
                'command_sha256': sha('cmd%d%s' % (s, L)), 'launch_time': a, 'exit_raw': '0\n', 'end_time': e,
                'pane': {'lines': 40, 'sha256': sha('pane'), 'fail_hits': {k: 0 for k in M.FAIL_PATTERNS}, 'tail': [],
                         's7_result': {'command': 's7', 'method': 'E04', 'label': L, 'seed': s, 'process_seeds': [s],
                                       'results': {str(s): {'status': 'completed', 'rows': 186, 'outputs': 186,
                                                            'failure_count': 0, 'manifest_sha256': man_sha,
                                                            'checkpoint_unchanged_after': True,
                                                            'peak_cuda_memory_allocated_bytes': None}}}},
                'tmux_pane': '1 0', 'gpu_samples': {'samples': 60, 'peak_used_mib_by_pid': {'4242': 1500, '612400': 4680},
                                                    'sha256': sha('g')}}
            seed_root = str(M.e04_seed_root(S7, s))
            pf_utc = iso(wmin[(s, L)][0] - 1)
            c['preflights'][str(s)][L] = {
                'phase': 'preflight', 'method': 'E04', 'utc': pf_utc, 'repo': REPO,
                'git': {'head': FAKE, 'branch': 'm6-baselines', 'status_porcelain': ''}, 'processes': [],
                'm8s7_panes': {'m8s7_E05_a2_s42_A': '1 0'}, 'seed': s, 'label': L,
                'gpu': {'name': M.GPU_NAME, 'uuid': M.GPU_UUID, 'driver': M.DRIVER, 'total_mib': 24576,
                        'used_mib': 9700, 'free_mib': 14000,
                        'compute_apps': [{'pid': 612400, 'name': 'python', 'used_mib': 4680, 'ps': 'unrelated'}]},
                'disk': {'free_bytes': 30 << 30}, 'memory': {'mem_available_bytes': 24 << 30},
                's7_root': {'path': S7, 'exists': True, 'within_runtime_root': False, 'within_hdd_bank_root': False},
                'per_seed': {'seed': s, 'label': L, 'seed_root': seed_root, 'seed_root_exists': L == 'B',
                             'run_dir_exists': False, 'run_A_completed': L == 'B'},
                'checkpoints': {str(x): ck_record(x) for x in M.SEEDS},
                'authority': {'ok': True, 'record': {'mode': 'S7_TOY', 'methods': ['E04'], 'seeds': [42, 1337, 2026],
                                                     'contract_sha256': M.E04_CONTRACT_SHA256, 'label': 's7_toy',
                                                     'records': records}},
                'inputs': {'toy_manifest_sha256': M.TOY_MANIFEST[1], 'feed_pack_manifest_sha256': M.E04_PACK_MANIFEST_SHA256,
                           'faces_root_is_dir': True},
                'hdd_bank_root_entries': hdd,
                'environment': {'python': M.E04_PYTHON, 'present': True, 'probe_rc': 0, 'probe_stderr_tail': [],
                                'probe': {'python': '3.8.20', 'tensorflow': '1.15.5', 'precision_errors': [],
                                          'environment_lock': {'ok': True, 'driver': {'gpu_uuid': M.GPU_UUID,
                                                                                      'observed_driver_version': M.DRIVER},
                                                               'native_runtime': {}, 'package_versions': {},
                                                               'session_config': session},
                                          'frozen_sources': {'files': dict(M.E04_FROZEN_FILES)},
                                          'bn_only_difference': True, 'contract_sha256': M.E04_CONTRACT_SHA256,
                                          's7_authority': {'records': records},
                                          'config': {'path': M.E04_CONFIG[0], 'sha256': M.E04_CONFIG[1]},
                                          'checkpoint_counters': {str(x): {'global_iteration': 150000, 'minibatch_phase': 0}
                                                                  for x in M.SEEDS}}}}
            c['preflights_sha256'][str(s)][L] = sha('pf%d%s' % (s, L))
            c['library_reader'].setdefault(str(s), {})[L] = {'ok': True, 'problems': [], 'files_checked': 186,
                                                              'failures': 0, 'provenance_ok': True, 'manifest_sha256': man_sha}
        c['library_reader'][str(s)]['compare_runs'] = {'manifest_equal': True, 'file_set_equal': True,
                                                       'files_compared': 186, 'files_differing': [],
                                                       'bitwise_identical': True}
        c['replay'][str(s)] = {'files_A': 186, 'files_B': 186, 'file_set_equal': True, 'bytes_differing': [],
                               'pixels_differing': [], 'pixel_arrays_compared_equal': 186, 'manifest_bytes_equal': True,
                               'manifest_sha_file_equal': True,
                               'provenance_differing_keys': ['end_utc', 'run_dir', 'start_utc', 'wall_time_seconds']}
        cli = {'command': 's7-compare', 'method': 'E04', 'seed': s, 'seeds': {str(s): {
            'run_A_reader_ok': True, 'run_B_reader_ok': True, 'failures': [0, 0],
            'replay': {'bitwise_identical': True}}}}
        c['cli_compare'][str(s)] = {'stdout': json.dumps(cli) + '\n', 'rc': '0\n',
                                    'command': '#!/bin/bash\n' + M.e04_compare_line(REPO, S7, s)}
        c['atime'][str(s)] = {'pre_utc': iso(wmin[(s, 'A')][0] - 1),
                              'post_utc': windows[(s, 'B')][1], 'pre_sha256': sha('pre'), 'post_sha256': sha('post'),
                              'files_pre': 20627, 'files_post': 20627, 'same_paths': True, 'changed_paths': [],
                              'by_split': {sp: {'files': n, 'atime_changed': 0, 'mtime_changed': 0, 'missing_post': 0}
                                           for sp, n in (('TRAIN', 12668), ('VAL', 3000), ('TEST', 4959))}}
        c['checkpoints_after'][str(s)] = ck_record(s)
    rel = {t['pair_id']: [t['position'], t['target_live_id'], t['source_spoof_id'], t['dataset']] for t in toy}
    c['lineage'] = {'files_sha256': dict((M.PAIRS_PARQUET, M.SPLIT_PARQUET, M.M2_PARQUET)), 'pairs_total': 8838,
                    'pairs_unique': True, 'pairs_stored_in_ascending_order': True, 'pairs_splits': ['TRAIN'],
                    'toy_relation': rel, 'samples': samples,
                    'feed_pack_crosscheck': {'manifest_sha256': M.E04_PACK_MANIFEST_SHA256, 'payload_sha_ok': True,
                                             'samples': 12668, 'samples_mismatching': [], 'samples_mismatching_count': 0,
                                             'pairs': 8838, 'pairs_equal_pairs_train_v1': True, 'pairs_split': 'TRAIN'}}
    return c


BASE = synthetic_collected()


class Builder(unittest.TestCase):
    def run_build(self, mutate=None, view=None):
        c = copy.deepcopy(BASE)
        if mutate:
            mutate(c)
        return M.build_e04(c, ROOT, view or View())

    def failed(self, mutate=None, view=None):
        doc = self.run_build(mutate, view)
        self.assertNotEqual(doc['status'], 'E04_S7_ALL_THREE_SEEDS_QUALIFIED')
        self.assertFalse(doc['s7_acceptance_passed'])
        return doc['failed_gates']

    def assertFails(self, gates, mutate=None, view=None):
        f = self.failed(mutate, view)
        for g in gates:
            self.assertIn(g, f)
        return f

    # ---------------------------------------------------------------- positive
    def test_00_synthetic_collection_passes_every_gate(self):
        doc = self.run_build()
        self.assertEqual(doc['failed_gates'], [], [g for g in doc['gates'] if not g['pass']][:3])
        self.assertEqual(doc['status'], 'E04_S7_ALL_THREE_SEEDS_QUALIFIED')
        self.assertTrue(all(v['qualified'] for v in doc['per_seed'].values()))
        self.assertGreaterEqual(doc['gates_total'], 150)
        names = {g['gate'] for g in doc['gates']}
        for s in M.SEEDS:
            for L in M.LABELS:
                for g in ('P.seed%d.run%s.resources', 'R.seed%d.run%s.exit_marker_zero', 'S.seed%d.run%s.inference_graph_bn_mode',
                          'S.seed%d.run%s.png_independent_verification', 'S.seed%d.run%s.train_only_o_noatime_reader'):
                    self.assertIn(g % (s, L), names)
        md = M.report_e04(doc)
        self.assertIn('E04_S7_ALL_THREE_SEEDS_QUALIFIED', md)
        self.assertIn('| 1337 | PASS | PASS | identical |', md)
        json.dumps(doc, sort_keys=True)

    def test_01_partial_status_when_one_seed_fails(self):
        def m(c):
            c['replay']['2026']['bytes_differing'] = ['PTR000028.png']
        doc = self.run_build(m)
        self.assertEqual(doc['status'], 'E04_S7_PARTIALLY_QUALIFIED')
        self.assertEqual([s for s, v in sorted(doc['per_seed'].items()) if v['qualified']], ['1337', '42'])

    # ---------------------------------------------------------------- owner negative cases
    def test_10_missing_launch_authority(self):
        v = View({M.E04_AUTHORITY_PATH: None})
        f = self.assertFails(['P.e04_authority_activated', 'P.e04_authority_derived_from_draft',
                              'P.e04_authority_in_execution_commit'], view=v)
        self.assertEqual(self.run_build(view=v)['status'], 'E04_S7_BLOCKED_BY_AUTHORITY')
        self.assertTrue(f)

    def test_11_authority_not_approved_or_widened(self):
        a = json.loads(authority_bytes())
        a['authorizes']['production_bank_generation'] = True
        self.assertFails(['P.e04_authority_activated'], view=View({M.E04_AUTHORITY_PATH: json.dumps(a).encode()}))
        a = json.loads(authority_bytes())
        a['status'] = 'DRAFT_PENDING_OWNER_DECISION'
        self.assertFails(['P.e04_authority_activated'], view=View({M.E04_AUTHORITY_PATH: json.dumps(a).encode()}))

    def test_12_incorrect_contract_sha256(self):
        raw = (ROOT / M.E04_CONTRACT_PATH).read_bytes() + b' '
        self.assertFails(['P.e04_contract', 'P.e04_authority_activated'], view=View({M.E04_CONTRACT_PATH: raw}))

        def m(c):
            c['runs']['B']['1337']['provenance']['adapter']['contract_sha256'] = '0' * 64
        self.assertFails(['S.seed1337.runB.config_contract_decisions'], m)

    def test_13_incorrect_generator_checkpoint(self):
        def m(c):
            c['runs']['A']['42']['provenance']['adapter']['checkpoint']['files']['data-00000-of-00001']['sha256'] = '1' * 64
        self.assertFails(['S.seed42.runA.checkpoint_before'], m)

        def other_iteration(c):
            f = c['runs']['A']['42']['provenance']['adapter']['checkpoint']['files']['index']
            f['path'] = f['path'].replace('iteration_150000', 'iteration_135000')
        self.assertFails(['S.seed42.runA.checkpoint_before'], other_iteration)

        def counter(c):
            c['preflights']['1337']['A']['environment']['probe']['checkpoint_counters']['1337']['global_iteration'] = 135000
        self.assertFails(['P.seed1337.runA.environment_lock'], counter)

    def test_14_incorrect_seed_and_checkpoint_seed_mapping(self):
        def m(c):
            p = c['runs']['A']['1337']['provenance']
            p['bank_seed'] = 42
            p['adapter']['generator_seed'] = 42
        self.assertFails(['S.seed1337.runA.completed', 'S.seed1337.runA.checkpoint_before'], m)

        def swapped(c):
            c['runs']['B']['2026']['provenance']['adapter']['checkpoint'] = copy.deepcopy(
                c['runs']['B']['42']['provenance']['adapter']['checkpoint'])
        self.assertFails(['S.seed2026.runB.checkpoint_before', 'X.no_cross_method_substitution'], swapped)

    def test_15_altered_source_identity(self):
        def spoof(c):
            r = c['runs']['A']['42']
            i = M.MANIFEST_FIELDS_E04.index('source_spoof_id')
            r['manifest_rows'][10][i] = r['manifest_rows'][11][i]
        self.assertFails(['S.seed42.toy_population', 'L.seed42.pair_lineage_pairs_train_v1'], spoof)

        def frozen(c):
            c['runs']['A']['42']['provenance']['adapter']['source']['files']['methods/physics_std/training_graph.py'] = '2' * 64
        self.assertFails(['S.seed42.runA.frozen_synthesis_sources'], frozen)

    def test_16_wrong_toy_population(self):
        def m(c):
            rows = c['runs']['A']['1337']['manifest_rows']
            i = M.MANIFEST_FIELDS_E04.index
            rows[5][i('pair_id')] = rows[5][i('row_id')] = 'PTR999999'
        self.assertFails(['S.seed1337.toy_population'], m)

        def kind(c):
            c['runs']['A']['42']['provenance']['population']['toy_manifest_sha256'] = '3' * 64
        self.assertFails(['S.seed42.toy_population'], kind)

    def test_17_wrong_pair_lineage(self):
        def m(c):
            pid = next(iter(c['lineage']['toy_relation']))
            c['lineage']['toy_relation'][pid][0] += 1
        self.assertFails(['L.seed42.pair_lineage_pairs_train_v1', 'L.seed2026.pair_lineage_pairs_train_v1'], m)

        def face(c):
            sid = next(iter(c['lineage']['samples']))
            c['lineage']['samples'][sid][2] = '4' * 64
        self.assertFails(['L.seed42.face_lineage_split_v1_m2'], face)

    def test_18_missing_output(self):
        def m(c):
            r = c['runs']['B']['42']
            name = sorted(r['png_files'])[0]
            del r['png_files'][name]
            r['listing'] = [x for x in r['listing'] if x[0] != 'bank/' + name]
            r['png_structures'] = {k: 185 for k in r['png_structures']}
        self.assertFails(['S.seed42.runB.outputs_complete_immutable', 'S.seed42.runB.png_independent_verification',
                          'S.seed42.file_and_pixel_hashes', 'S.seed42.bitwise_replay_A_B'], m)

    def test_19_altered_png_bytes(self):
        def m(c):
            r = c['runs']['B']['2026']
            name = sorted(r['png_files'])[7]
            r['png_files'][name][0] = '5' * 64
            for x in r['listing']:
                if x[0] == 'bank/' + name:
                    x[2] = '5' * 64
            c['replay']['2026']['bytes_differing'] = [name]
        self.assertFails(['S.seed2026.file_and_pixel_hashes', 'S.seed2026.bitwise_replay_A_B'], m)

        def structure(c):
            r = c['runs']['A']['42']
            k = next(iter(r['png_structures']))
            bad = json.loads(k)
            bad['chunk_types'] = ['IHDR', 'tEXt', 'IDAT', 'IEND']
            r['png_structures'] = {k: 185, M.canonical(bad): 1}
        self.assertFails(['S.seed42.runA.png_independent_verification'], structure)

    def test_20_altered_output_manifest(self):
        def m(c):
            c['runs']['A']['1337']['manifest_rows'][3][M.MANIFEST_FIELDS_E04.index('output_sha256')] = '6' * 64
        self.assertFails(['S.seed1337.manifest_rederives_from_rows', 'S.seed1337.file_and_pixel_hashes'], m)

        def sha_file(c):
            c['runs']['B']['42']['manifest']['sha_file'] = '0' * 64 + '  bank_manifest.jsonl\n'
        self.assertFails(['S.seed42.runB.manifest'], sha_file)

    def test_21_mismatched_ab_replay(self):
        def m(c):
            c['replay']['1337']['pixels_differing'] = ['PTR000028.png']
            c['replay']['1337']['pixel_arrays_compared_equal'] = 185
        self.assertFails(['S.seed1337.bitwise_replay_A_B'], m)

        def prov(c):
            c['replay']['42']['provenance_differing_keys'].append('adapter')
        self.assertFails(['S.seed42.bitwise_replay_A_B'], prov)

        def cli(c):
            c['cli_compare']['2026']['rc'] = '1\n'
        self.assertFails(['S.seed2026.cli_s7_compare'], cli)

    def test_22_wrong_bn_inference_mode(self):
        def m(c):
            c['runs']['A']['42']['provenance']['adapter']['graph']['fused_batch_norm_is_training'] = True
        self.assertFails(['S.seed42.runA.inference_graph_bn_mode'], m)

        def batch_stats(c):
            c['runs']['B']['1337']['provenance']['adapter_finalize']['bn_mode'] = 'training_batch_statistics'
        self.assertFails(['S.seed1337.runB.inference_graph_bn_mode'], batch_stats)

        def moving(c):
            c['runs']['A']['2026']['provenance']['adapter']['moving_statistics']['values_sha256'] = '7' * 64
        self.assertFails(['S.seed2026.runA.stored_bn_statistics'], moving)

    def test_23_wrong_output_conversion(self):
        def m(c):
            rows = c['runs']['A']['42']['manifest_rows']
            for r in rows:
                r[M.MANIFEST_FIELDS_E04.index('conversion_rule')] = 'np.round(out * 255).astype(np.uint8)'
        self.assertFails(['S.seed42.manifest_rows'], m)

        def adapter(c):
            c['runs']['B']['42']['provenance']['adapter']['conversion_rule'] = 'np.round'
        self.assertFails(['S.seed42.runB.config_contract_decisions'], adapter)

    def test_24_unexpected_optimizer_activity(self):
        def m(c):
            g = c['runs']['A']['1337']['provenance']['adapter']['graph']
            g['gradient_ops'] = 12
            g['mutating_reachable_ops'] = ['ApplyAdam']
        self.assertFails(['S.seed1337.runA.no_training_ops', 'S.seed1337.runA.inference_graph_bn_mode'], m)

        def log(c):
            c['markers']['42']['B']['pane']['fail_hits']['TrainingForbidden'] = 1
        self.assertFails(['R.seed42.runB.result_line_and_log'], log)

    def test_25_missing_exit_markers(self):
        def m(c):
            c['markers']['2026']['B']['exit_raw'] = None
            c['markers']['2026']['B']['end_time'] = ''
        self.assertFails(['R.seed2026.runB.exit_marker_zero', 'R.sequential_independent_processes'], m)

        def nonzero(c):
            c['markers']['42']['A']['exit_raw'] = '1\n'
        self.assertFails(['R.seed42.runA.exit_marker_zero'], nonzero)

    def test_26_val_test_access(self):
        def atime(c):
            c['atime']['1337']['by_split']['TEST']['atime_changed'] = 1
        self.assertFails(['F.seed1337.atime_ledger_zero_changes'], atime)

        def val_input(c):
            sid = c['runs']['A']['42']['manifest_rows'][0][M.MANIFEST_FIELDS_E04.index('target_live_id')]
            c['lineage']['samples'][sid][0] = 'VAL'
        self.assertFails(['F.seed42.no_val_test_inputs', 'L.seed42.face_lineage_split_v1_m2'], val_input)

        def pack(c):
            c['lineage']['feed_pack_crosscheck']['samples_mismatching_count'] = 1
        self.assertFails(['F.feed_pack_allowlist_crosscheck'], pack)

        def opened(c):
            c['runs']['B']['2026']['provenance']['population']['pairs_val_opened'] = True
        self.assertFails(['F.seed2026.no_val_test_inputs', 'S.seed2026.runB.train_only_o_noatime_reader'], opened)

    def test_27_unauthorized_face_atime_modification(self):
        def m(c):
            c['atime']['42']['by_split']['TRAIN']['atime_changed'] = 2
            c['atime']['42']['changed_paths'] = ['faces_256/casia_fasd/x.png']
        self.assertFails(['F.seed42.atime_ledger_zero_changes'], m)

        def reader(c):
            c['runs']['A']['42']['provenance']['reader']['o_noatime'] = False
        self.assertFails(['S.seed42.runA.train_only_o_noatime_reader'], reader)

        def late(c):
            c['atime']['2026']['post_utc'] = '2026-11-25T20:00:00Z'
        self.assertFails(['F.seed2026.atime_ledger_zero_changes'], late)

    def test_28_modified_checkpoint(self):
        def m(c):
            c['checkpoints_after']['1337']['files']['meta']['sha256'] = '8' * 64
        self.assertFails(['S.seed1337.checkpoint_unchanged_before_after'], m)

        def mtime(c):
            c['checkpoints_after']['42']['files']['index']['mtime_ns'] += 1
        self.assertFails(['S.seed42.checkpoint_unchanged_before_after'], mtime)

    def test_29_wrong_gpu_or_environment_identity(self):
        def gpu(c):
            c['preflights']['42']['A']['gpu']['uuid'] = 'GPU-00000000-0000-0000-0000-000000000000'
        self.assertFails(['P.seed42.runA.resources'], gpu)

        def py(c):
            c['runs']['B']['1337']['provenance']['environment']['python_executable'] = '/usr/bin/python3'
        self.assertFails(['S.seed1337.runB.environment_and_gpu'], py)

        def driver(c):
            c['runs']['A']['2026']['provenance']['adapter']['driver']['gpu_uuid'] = 'GPU-other'
        self.assertFails(['S.seed2026.runA.environment_and_gpu'], driver)

        def lock(c):
            c['preflights']['42']['B']['environment']['probe']['precision_errors'] = ['TF_DETERMINISTIC_OPS']
        self.assertFails(['P.seed42.runB.environment_lock'], lock)

        def busy(c):
            c['preflights']['1337']['A']['processes'] = ['123 python tools/learned_bank.py production --method E03']
        self.assertFails(['P.seed1337.runA.resources'], busy)

    def test_30_cross_method_evidence_substitution(self):
        def m(c):
            p = c['runs']['A']['42']['provenance']
            p['method_id'] = 'E03'
            p['authorization']['records'] = {'configs/amendments/learned_bank_s7_launch_authority.yaml': '0' * 64}
        self.assertFails(['X.no_cross_method_substitution', 'S.seed42.runA.completed', 'S.seed42.runA.authorization'], m)

        def other_dir(c):
            c['runs']['B']['1337']['dir'] = c['runs']['B']['1337']['dir'].replace('/E04/', '/E03/')
        self.assertFails(['X.no_cross_method_substitution'], other_dir)

        def other_record(c):
            c['method'] = 'E03'
        self.assertFails(['X.no_cross_method_substitution'], other_record)

    def test_31_execution_checkout_and_process_isolation(self):
        def m(c):
            c['git']['head'] = '18ba00e94813f005876c8d6235b0b848226b2c1d'
        self.assertFails(['P.execution_checkout'], m)

        def dirty(c):
            c['preflights']['2026']['A']['git']['status_porcelain'] = ' M methods/physics_std/bank_adapter.py'
        self.assertFails(['P.execution_checkout'], dirty)

        def two_seeds(c):
            c['markers']['42']['A']['pane']['s7_result']['process_seeds'] = [42, 1337]
        self.assertFails(['R.seed42.runA.result_line_and_log'], two_seeds)

        def overlap(c):
            c['markers']['1337']['A']['launch_time'] = c['markers']['42']['B']['launch_time']
        self.assertFails(['R.sequential_independent_processes'], overlap)

        def reused_root(c):
            c['preflights']['42']['A']['per_seed']['seed_root_exists'] = True
        self.assertFails(['P.seed42.runA.fresh_run_root'], reused_root)

        def command(c):
            c['markers']['2026']['A']['command'] = c['markers']['2026']['A']['command'].replace(
                'TF_DETERMINISTIC_OPS=1 ', '')
        self.assertFails(['R.seed2026.runA.command'], command)

    def test_32_storage_never_created(self):
        def m(c):
            c['hdd_bank_root_entries'] = c['hdd_bank_root_entries'] + ['E04']
        self.assertFails(['F.no_e04_storage_created'], m)

    def test_33_missing_seed_evidence_fails_closed(self):
        def m(c):
            c['runs']['B']['2026'] = None
            c['replay']['2026'] = None
            c['preflights']['2026']['B'] = None
        doc = self.run_build(m)
        self.assertIn('S.seed2026.runB.completed', doc['failed_gates'])
        self.assertIn('P.execution_checkout', doc['failed_gates'])
        self.assertNotEqual(doc['status'], 'E04_S7_ALL_THREE_SEEDS_QUALIFIED')

    def test_33b_empty_or_truncated_collection_never_passes(self):
        for c in ({}, {'method': 'E04'}, {'git': {'head': FAKE}, 'method': 'E04', 'runs': {'A': {}, 'B': {}}}):
            doc = M.build_e04(c, ROOT, View())
            self.assertFalse(doc['s7_acceptance_passed'])
            self.assertEqual(doc['gates_total'], 180)
            self.assertTrue(all(g['gate'].startswith('P.e04_') for g in doc['gates'] if g['pass']))
            M.report_e04(doc)

    def test_34_code_change_scope_on_real_history(self):
        doc = self.run_build()
        g = {x['gate']: x for x in doc['gates']}['P.e04_code_change_scope']
        self.assertTrue(g['pass'])
        bad = (ROOT / M.E04_ADAPTER_PATH).read_text().replace(
            "y = np.clip(x, 0.0, 1.0) * 255.0", "y = np.clip(x, 0.0, 1.0) * 255.0 + 0.5")
        self.assertFails(['P.e04_code_change_scope'], view=View({M.E04_ADAPTER_PATH: bad.encode()}))


class CollectorHelpers(unittest.TestCase):
    def png(self, seed=0):
        rng = np.random.default_rng(seed)
        arr = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
        return arr, lb.encode_png(arr)

    def test_png_inspect_real_bytes(self):
        arr, raw = self.png()
        st, px, dec = M.png_inspect(raw)
        self.assertEqual(st, dict(M.E04_PNG_STRUCTURE, scanline_filters_ok=True))
        self.assertEqual(px, lb.pixel_sha256(arr))
        self.assertTrue(np.array_equal(dec, arr))

    def test_png_inspect_detects_tampering(self):
        _, raw = self.png(1)
        flipped = bytearray(raw)
        flipped[100] ^= 0xFF
        st, _, _ = M.png_inspect(bytes(flipped))
        self.assertFalse(st['crc_ok'])
        import zlib
        import struct
        text = b'tEXtComment\x00x'
        chunk = struct.pack('>I', len(text) - 4) + text + struct.pack('>I', zlib.crc32(text) & 0xffffffff)
        st, _, _ = M.png_inspect(raw[:33] + chunk + raw[33:])
        self.assertEqual(st['chunk_types'], ['IHDR', 'tEXt', 'IDAT', 'IEND'])
        from PIL import Image
        buf = io.BytesIO()
        Image.fromarray(np.zeros((256, 256, 4), np.uint8)).save(buf, format='PNG')
        st, _, _ = M.png_inspect(buf.getvalue())
        self.assertNotEqual(st, dict(M.E04_PNG_STRUCTURE, scanline_filters_ok=True))
        st, px, _ = M.png_inspect(raw + b'junk')
        self.assertEqual(st['trailing_bytes'], 4)

    def make_run(self, root, pngs, start):
        d = Path(root)
        (d / 'bank').mkdir(parents=True)
        for n, raw in pngs.items():
            (d / 'bank' / n).write_bytes(raw)
        (d / 'bank_manifest.jsonl').write_bytes(b'{"a":1}\n')
        (d / 'bank_manifest.sha256').write_text('%s  bank_manifest.jsonl\n' % sha(b'{"a":1}\n'))
        (d / 'bank_provenance.json').write_text(json.dumps({'start_utc': start, 'adapter': {'x': 1}}))
        return d

    def test_independent_compare_real_dirs(self):
        _, r1 = self.png(2)
        _, r2 = self.png(3)
        with tempfile.TemporaryDirectory() as t:
            a = self.make_run(Path(t) / 'A', {'P1.png': r1, 'P2.png': r2}, '1')
            b = self.make_run(Path(t) / 'B', {'P1.png': r1, 'P2.png': r2}, '2')
            out = M.independent_compare(a, b)
            self.assertEqual((out['bytes_differing'], out['pixels_differing'], out['pixel_arrays_compared_equal'],
                              out['manifest_bytes_equal'], out['provenance_differing_keys']), ([], [], 2, True, ['start_utc']))
            os.chmod(str(b / 'bank' / 'P2.png'), 0o644)
            (b / 'bank' / 'P2.png').write_bytes(r1)
            out = M.independent_compare(a, b)
            self.assertEqual((out['bytes_differing'], out['pixels_differing']), (['P2.png'], ['P2.png']))
            ev = M.run_dir_evidence(a)
            self.assertEqual(ev['png_structures'], {M.canonical(dict(M.E04_PNG_STRUCTURE, scanline_filters_ok=True)): 2})
            self.assertEqual(ev['png_files']['P1.png'][0], sha(r1))

    def test_lineage_and_pack_crosscheck_on_real_parquet(self):
        import pyarrow.parquet as pq
        toy = json.loads((ROOT / M.TOY_MANIFEST[0]).read_text())['toy_pairs'][:2]
        sids = sorted({t['target_live_id'] for t in toy} | {t['source_spoof_id'] for t in toy})
        m2 = {r['sample_id']: r for r in pq.read_table(str(ROOT / M.M2_PARQUET[0]), columns=[
            'sample_id', 'face_png_sha256'], filters=[('sample_id', 'in', sids)]).to_pylist()}
        sp = {r['sample_id']: r for r in pq.read_table(str(ROOT / M.SPLIT_PARQUET[0]), columns=[
            'sample_id', 'dataset'], filters=[('sample_id', 'in', sids)]).to_pylist()}
        cols = ['sample_id', 'role', 'dataset', 'split', 'label_binary', 'face_png_sha256', 'q140_row', 'm0_row',
                'q140_xy_sha256', 'm0_sha256']
        rows = [[s, 'x', sp[s]['dataset'], 'TRAIN', 0, m2[s]['face_png_sha256'], 0, None, None, None] for s in sids]
        f = M.MANIFEST_FIELDS_E04
        mrows = {42: [[{'pair_id': t['pair_id'], 'target_live_id': t['target_live_id'],
                        'source_spoof_id': t['source_spoof_id']}.get(k) for k in f] for t in toy]}
        with tempfile.TemporaryDirectory() as t:
            def pack(sample_rows):
                for n in ('pack_manifest.json', 'samples.json', 'pairs_train.json'):
                    p = Path(t) / n
                    if p.exists():
                        p.unlink()
                s_raw = json.dumps({'columns': cols, 'rows': sample_rows}).encode()
                p_raw = json.dumps({'columns': ['pair_id', 'target_live_id', 'source_spoof_id', 'dataset'],
                                    'split': 'TRAIN', 'rows': []}).encode()
                (Path(t) / 'samples.json').write_bytes(s_raw)
                (Path(t) / 'pairs_train.json').write_bytes(p_raw)
                (Path(t) / 'pack_manifest.json').write_text(json.dumps({'payloads': {
                    'samples.json': {'sha256': sha(s_raw)}, 'pairs_train.json': {'sha256': sha(p_raw)}}}))
                with mock.patch.object(M, 'E04_PACK_DIR', t):
                    return M.e04_lineage(ROOT, mrows)
            out = pack(rows)
            self.assertEqual(out['files_sha256'], dict((M.PAIRS_PARQUET, M.SPLIT_PARQUET, M.M2_PARQUET)))
            self.assertEqual((out['pairs_total'], out['pairs_unique'], out['pairs_stored_in_ascending_order']),
                             (8838, True, True))
            for tt in toy:
                self.assertEqual(out['toy_relation'][tt['pair_id']], [tt['position'], tt['target_live_id'],
                                                                     tt['source_spoof_id'], tt['dataset']])
            self.assertTrue(all(v[0] == 'TRAIN' and v[3] == 'COMPLETE' for v in out['samples'].values()))
            fp = out['feed_pack_crosscheck']
            self.assertEqual((fp['samples'], fp['samples_mismatching_count'], fp['payload_sha_ok']), (len(sids), 0, True))
            bad = copy.deepcopy(rows)
            bad[0][5] = '0' * 64
            self.assertEqual(pack(bad)['feed_pack_crosscheck']['samples_mismatching'], [bad[0][0]])

    def test_derived_authority_keeps_every_draft_field(self):
        draft_raw = (ROOT / M.E04_AUTHORITY_DRAFT_PATH).read_bytes()
        draft = json.loads(draft_raw)
        out = M.derive_e04_s7_authority(draft_raw, APPROVAL)
        activation = {'status', 'approved', 'launch_preconditions'}
        for k, v in draft.items():
            if k not in activation:
                self.assertEqual(out[k], v, k)
        self.assertEqual((out['status'], out['approved']), ('OWNER_APPROVED', True))
        self.assertEqual(out['derived_from'], {'path': M.E04_AUTHORITY_DRAFT_PATH, 'sha256': sha(draft_raw)})
        real = ROOT / M.E04_AUTHORITY_PATH
        if real.is_file():
            self.assertEqual(real.read_bytes(), authority_bytes())

    def test_command_lines(self):
        line = M.e04_command_line(REPO, S7, 1337, 'B')
        for k, val in M.E04_PRECISION_ENV:
            self.assertIn('%s=%s ' % (k, val), line)
        self.assertTrue(line.endswith('--out-root %s/E04/seed_1337 --runtime-root %s --seed 1337\n' % (S7, M.RUNTIME)))
        self.assertIn('CUDA_VISIBLE_DEVICES= ', M.e04_compare_line(REPO, S7, 42))
        self.assertNotIn('CUDA_VISIBLE_DEVICES=0', M.e04_compare_line(REPO, S7, 42))


class ExistingMethodsUntouched(unittest.TestCase):
    def test_existing_records_still_dispatch_to_original_builder(self):
        for m in ('E03', 'E05', 'E06b', 'E06c'):
            d = json.loads((ROOT / 'outputs' / 'audit' / ('LEARNED_BANK_S7_%s.json' % m)).read_text())
            self.assertEqual(d['schema'], 'learned_bank_s7_acceptance_record_v1')
            self.assertNotEqual(d['collected']['method'], 'E04')
        self.assertEqual(M.ESTIMATE_GIB, {'E06c': 4, 'E06b': 4, 'E05': 3, 'E03': 2})


class AdapterAllowlistFix(unittest.TestCase):
    """Owner ruling e04_s7_input_allowlist_ruling.yaml: TRAIN allowlist from the pinned feed pack, no parquet."""

    def fake_pack(self, split='TRAIN'):
        rows = [('PTR%06d' % i, 'L%05d' % i, 'S%05d' % i, 'casia_fasd') for i in range(lb.N_SYN)]
        samples = {}
        for _, li, spf, ds in rows:
            samples[li] = {'split': 'TRAIN', 'dataset': ds, 'face_png_sha256': sha(li), 'role': 'live'}
            samples[spf] = {'split': 'TRAIN', 'dataset': ds, 'face_png_sha256': sha(spf), 'role': 'spoof'}
        samples['S00007']['split'] = split

        class Pack:
            pass
        p = Pack()
        p.rows, p.manifest_sha256 = rows, 'f' * 64
        p.sample = lambda sid: dict(samples[sid])
        p.q140_xy = lambda sid: None
        return p

    def test_allowlist_from_pack_without_parquet(self):
        from methods.physics_std import bank_adapter as ba
        from methods.physics_std import feed_pack
        with mock.patch.object(feed_pack, 'load', return_value=self.fake_pack()), \
                mock.patch.object(lb, 'train_face_allowlist_parquet', side_effect=AssertionError('parquet used')):
            out = ba.load_e04_inputs('/nonexistent/runtime')
        self.assertEqual(len(out['allowlist']), 2 * lb.N_SYN)
        self.assertEqual(out['allowlist']['L00003'], {'split': 'TRAIN', 'dataset': 'casia_fasd',
                                                      'face_png_sha256': sha('L00003')})
        self.assertEqual(out['allowlist_source'], M.E04_ALLOWLIST_SOURCE)
        self.assertEqual(len(out['relation']), lb.N_SYN)

    def test_non_train_pack_sample_refused(self):
        from methods.physics_std import bank_adapter as ba
        from methods.physics_std import feed_pack
        with mock.patch.object(feed_pack, 'load', return_value=self.fake_pack('VAL')):
            with self.assertRaises(lb.LearnedBankError):
                ba.load_e04_inputs('/nonexistent/runtime')

    def test_ruling_record(self):
        r = json.loads((ROOT / M.E04_RULING_PATH).read_text())
        self.assertEqual((r['status'], r['approved'], r['decision']), ('OWNER_APPROVED', True, M.E04_RULING_DECISION))
        self.assertEqual(r['allowlist_source'], M.E04_ALLOWLIST_SOURCE)
        self.assertFalse(r['defect']['pixel_affecting'])

    def test_cli_seed_selection(self):
        sys.path.insert(0, str(ROOT / 'tools'))
        spec = importlib.util.spec_from_file_location('lbcli', ROOT / 'tools' / 'learned_bank.py')
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        self.assertEqual(cli.SEED_SELECTABLE, ('E05', 'E04'))
        for m in ('E03', 'E06b', 'E06c'):
            with self.assertRaises(SystemExit) as cm, mock.patch('sys.stderr', io.StringIO()):
                cli.main(['s7', '--method', m, '--label', 'A', '--out-root', '/x', '--runtime-root', '/y', '--seed', '42'])
            self.assertEqual(cm.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
