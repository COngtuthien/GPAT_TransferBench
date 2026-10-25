"""E06b three-seed family closure: gates the committed per-seed closures of the E06b DSDG-NATIVE scientific training
family (successful seeds 42, 1337, 2026; training authority d9ecbef, FAITHFUL_OFFICIAL_WITH_RUNTIME_COMPATIBILITY,
track B_NATIVE_FULL_SECONDARY) and writes outputs/audit/E06B_THREE_SEED_{TRAINING_CLOSURE.json,FINAL_CHECKPOINTS.csv}.
Modelled on tools/e06c_three_seed_training_closure.py.

Inputs:
  * the committed per-seed evidence outputs/audit/E06B_SEED<s>_{COMPLETION.json,CHECKPOINT_MANIFEST.txt,
    EVIDENCE_SHA256.txt}; every per-seed COMPLETION.json is re-derived with its own closure tool and compared with
    the bytes committed by its closure commit;
  * a read-only host recheck document (final netG sha256/size, run-root listing digests, resolved_config.yaml
    text, the seed42 attempt-1 archive digest), collected with tools/e06b_family_host_recheck.py and embedded in the
    output so the build re-derives.

The seed42 attempt 1 (d3c0ddd5dc82a4a1, 24f7049, FAILED_TECHNICAL_CHECKPOINT_WRITER_BUG) is failed technical history
only: it is not a member of the family and is only checked to stay archived, untouched and outside runs/.

This closes generator TRAINING only. No selection, VAL, TEST, SiW, bank, downstream or training code is run, and no
performance metric is computed: the frozen primary result rule MEAN_AND_STD_OVER_ALL_THREE_SEEDS applies to future
evaluation metrics.
"""
import argparse
import csv
import hashlib
import importlib.util
import io
import json
import subprocess
import sys
from pathlib import Path

CLASSIFICATION = 'E06B_THREE_SEED_SCIENTIFIC_TRAINING_CLOSURE'
STATUS_OK = 'SCIENTIFIC_TRAINING_COMPLETED_3_OF_3_SEEDS'
STATUS_BLOCKED = 'E06B_FAMILY_NEEDS_REVIEW'
SEEDS = (42, 1337, 2026)
AUTHORITY = 'd9ecbef7295fc6b41ced6a4966b49810d36a8c87'
CONFIG_SHA256 = '9d665dc2c909d421b8e54964407e27bb40d133f11bceec2e2cb29810f268417f'
FIDELITY = 'FAITHFUL_OFFICIAL_WITH_RUNTIME_COMPATIBILITY'
RUN_BASE = '/home/student20261/workdir/GPAT_TransferBench_runtime/runs/m6/E06b'
FINAL_NETG = 'checkpoints/netG_model_epoch_200_iter_0.pth'
CHECKPOINT_RULE = 'OFFICIAL_GENERATOR_EPOCH_200'
CHECKPOINT_EPOCHS = [1] + list(range(10, 201, 10))
KNOWN_FINAL_NETG = {42: 'd56122e0cf363a6c870e29b394998732f4bcdaf35a96a6110e937905b64ff8aa',
                    1337: '645538e866a6d835a1bc35240c2f15c27296742850df96d722fd4d38609b4f53'}
RUN_IDS = {42: 'aa96129289d080ba', 1337: '9ffda3ea3f6cc767', 2026: 'ec71778b28479de5'}
ATTEMPTS = {42: 2, 1337: 1, 2026: 1}
CLOSURE_COMMITS = {42: '1f546c9fab1e284efd9deaef18ab7d415ff2c2ee',
                   1337: 'a8a7b5dd9a1fd5cd8e82b870162732163b2859f9',
                   2026: '1729333197c5caab7a0d4d9302f1d3274ac90d98'}
ATTEMPT1 = {'seed': 42, 'run_id': 'd3c0ddd5dc82a4a1', 'authority': '24f7049b894330574b84d11baeb40ed4a7fb3c6b',
            'classification': 'FAILED_TECHNICAL_CHECKPOINT_WRITER_BUG',
            'listing_digest_path_size_sha256': '8ce8d5a8be3ec6179ef4299a51ab7903383bb963307e809d50c7e2dd9d91f6ca',
            'file_count': 14}
# run_manifest fields that differ by design between seeds (everything else must be byte-equal)
MANIFEST_PER_SEED = ('command_line', 'end_utc', 'experiment_seed', 'run_id', 'run_uuid', 'start_utc')
# resolved_config.yaml lines allowed to differ, as functions of the seed
RESOLVED_PER_SEED = (lambda s: f'  experiment_seed: {s}', lambda s: f'  run_dir: {RUN_BASE}/seed_{s}')
# run-start blocks that must be identical across seeds (seeding differs only by its seed value)
RUN_START_COMMON = ('binding', 'execution_mode', 'identities', 'lightcnn', 'loader', 'population', 'precision',
                    'runtime_fidelity_assessment')
# per-seed gate names that embed the seed set / attempt naming; normalised before comparing gate semantics
GATE_ALIASES = {'B.not_attempt1': 'B.not_seed42_attempt1',
                'H.attempt1_archived_untouched_nonselectable': 'H.seed42_attempt1_archived_untouched_nonselectable'}
CSV_COLUMNS = ('seed', 'run_id', 'authority', 'final_netG_path', 'final_netG_sha256', 'checkpoint_size_bytes',
               'metrics_sha256', 'run_root_sha_listing_digest', 'run_root_mtime_listing_digest', 'completion_status',
               'completed_epoch', 'global_step', 'val_access', 'test_access', 'siw_access', 'seed_closure_commit')
EVIDENCE = {'closure': 'outputs/audit/E06B_THREE_SEED_TRAINING_CLOSURE.json',
            'csv': 'outputs/audit/E06B_THREE_SEED_FINAL_CHECKPOINTS.csv',
            'report': 'outputs/audit/E06B_THREE_SEED_TRAINING_CLOSURE.md',
            'sha256': 'outputs/audit/E06B_THREE_SEED_EVIDENCE_SHA256.txt'}


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def seed_paths(s):
    a = f'outputs/audit/E06B_SEED{s}_'
    return {'completion': a + 'COMPLETION.json', 'manifest': a + 'CHECKPOINT_MANIFEST.txt',
            'report': a + 'COMPLETION.md', 'sha256': a + 'EVIDENCE_SHA256.txt',
            'tool': f'tools/e06b_seed{s}_completion_audit.py'}


def seed_tool(repo, s):
    spec = importlib.util.spec_from_file_location(f'e06b_s{s}', repo / seed_paths(s)['tool'])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def git_show(repo, commit, path):
    r = subprocess.run(['git', 'show', f'{commit}:{path}'], cwd=repo, capture_output=True)
    return r.stdout if r.returncode == 0 else None


class Gates:
    def __init__(self):
        self.results = []

    def __call__(self, name, ok, detail=None):
        self.results.append({'gate': name, 'pass': bool(ok), **({'detail': detail} if detail is not None else {})})
        return ok

    @property
    def failed(self):
        return [r['gate'] for r in self.results if not r['pass']]


def seed_row(s, d):
    f, ri, fw = d['checkpoints']['final_netG'], d['run_root_integrity'], d['firewall']
    return {'seed': s, 'run_id': d['run_id'], 'authority': d['training_authority_commit'],
            'final_netG_path': f['path'], 'final_netG_sha256': f['sha256'], 'checkpoint_size_bytes': f['size_bytes'],
            'metrics_sha256': ri['metrics_jsonl_sha256'], 'run_root_sha_listing_digest': ri['sha_listing_digest'],
            'run_root_mtime_listing_digest': ri['listing_digest'],
            'completion_status': d['collected']['run_summary']['completion_status'],
            'completed_epoch': d['training']['completed_epoch'], 'global_step': d['training']['global_step'],
            'val_access': fw['val_access'], 'test_access': fw['test_access'], 'siw_access': fw['siw_access'],
            'seed_closure_commit': CLOSURE_COMMITS[s]}


def csv_text(rows):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=CSV_COLUMNS, lineterminator='\n')
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def build(repo_root, host):
    repo = Path(repo_root)
    G = Gates()
    docs, rows = {}, []

    # A. per-seed evidence: SHA manifests, deterministic re-derivation, bytes equal to each closure commit
    present = sorted(p.name for p in (repo / 'outputs/audit').glob('E06B_SEED*_COMPLETION.json'))
    G('A.exact_seed_evidence_set', present == sorted(f'E06B_SEED{s}_COMPLETION.json' for s in SEEDS), present)
    for s in SEEDS:
        p = seed_paths(s)
        lines = (repo / p['sha256']).read_text().splitlines()
        listed = {name: digest for digest, name in (x.split('  ') for x in lines)}
        G(f'A.seed{s}.evidence_sha256', sorted(listed) == sorted([p['completion'], p['manifest'], p['report']])
          and all(sha256_bytes((repo / n).read_bytes()) == h for n, h in listed.items()))
        d = json.loads((repo / p['completion']).read_bytes())
        docs[s] = d
        rebuilt, text = seed_tool(repo, s).build(d['collected'], repo)
        G(f'A.seed{s}.rederives', json.loads(json.dumps(rebuilt)) == d
          and text == (repo / p['manifest']).read_text())
        G(f'A.seed{s}.equals_closure_commit', all(git_show(repo, CLOSURE_COMMITS[s], p[k]) == (repo / p[k]).read_bytes()
                                                  for k in ('completion', 'manifest', 'report', 'sha256', 'tool')),
          CLOSURE_COMMITS[s])

    # B. per-seed closure status and identical gate semantics
    for s, d in docs.items():
        G(f'B.seed{s}.closed_clean', d['status'] == f'E06B_SEED{s}_COMPLETED_CLEANLY' and d['failed_gates'] == []
          and all(g['pass'] for g in d['gates']) and d['seed'] == s and d['attempt'] == ATTEMPTS[s], d['status'])
    names = {s: [g['gate'] for g in d['gates']] for s, d in docs.items()}
    G('B.same_gate_count', {len(v) for v in names.values()} == {84}, {s: len(v) for s, v in names.items()})
    norm = {s: [GATE_ALIASES.get(n, n) for n in v if not n.startswith('H.parent_only')] for s, v in names.items()}
    G('B.same_gate_semantics', norm[42] == norm[1337] == norm[2026])

    # C. authority / identity (common by design)
    mans = {s: d['collected']['run_manifest'] for s, d in docs.items()}
    starts = {s: d['collected']['run_start_event'] for s, d in docs.items()}
    for s, d in docs.items():
        m, git = mans[s], d['collected']['git']
        G(f'C.seed{s}.authority', d['training_authority_commit'] == m['git_commit'] == git['head'] == AUTHORITY
          and m['git_dirty'] is False and git['status_porcelain'] == '' and git['branch'] == 'm6-baselines')
        rid = sha256_bytes(f'E06b|{s}|{CONFIG_SHA256}|{AUTHORITY}'.encode())[:16]
        G(f'C.seed{s}.run_id_rule', rid == d['run_id'] == m['run_id'] == RUN_IDS[s], rid)
    G('C.run_ids_distinct', len({d['run_id'] for d in docs.values()}) == 3)
    G('C.method', all((d['method']['id'], d['method']['name'], d['method']['fidelity'], d['method']['track'],
                       d['method']['execution_mode']) == ('E06b', 'DSDG-NATIVE', FIDELITY, 'B_NATIVE_FULL_SECONDARY',
                                                          'GLOBAL_STATISTIC_PRESERVING_MICROBATCH_V2')
                      for d in docs.values()))
    G('C.frozen_config_sha', all(d['frozen']['config_sha256'] == CONFIG_SHA256 == mans[s]['config_sha256']
                                 == d['collected']['config_file_sha256'] for s, d in docs.items()))
    common = {s: {k: v for k, v in m.items() if k not in MANIFEST_PER_SEED} for s, m in mans.items()}
    diff_keys = sorted(k for k in set().union(*common.values()) if len({json.dumps(c.get(k), sort_keys=True)
                                                                         for c in common.values()}) > 1)
    G('C.manifest_common_fields_equal', diff_keys == [] and common[42].keys() == common[1337].keys()
      == common[2026].keys(), diff_keys)
    G('C.manifest_seed_fields', all(m['experiment_seed'] == s and m['run_id'] == docs[s]['run_id']
                                    for s, m in mans.items()))
    G('C.command_line_differs_only_by_seed', len({m['command_line'].replace(f'--seed {s} ', '--seed <S> ')
                                                  for s, m in mans.items()}) == 1
      and all(f'--seed {s} ' in m['command_line'] and '--resume' not in m['command_line'] for s, m in mans.items()))
    ids = starts[42]['identities']
    G('C.identities_equal', all(r['identities'] == ids for r in starts.values())
      and all(d['collected']['identities_recomputed'] == ids for d in docs.values())
      and ids['runtime_fidelity_assessment'] == FIDELITY and ids['config_sha256'] == CONFIG_SHA256, sorted(ids))
    rs_diff = sorted(k for k in RUN_START_COMMON
                     if len({json.dumps(r[k], sort_keys=True) for r in starts.values()}) > 1)
    G('C.run_start_blocks_equal', rs_diff == [] and all(set(r) == set(RUN_START_COMMON) | {'seeding'}
                                                        for r in starts.values()), rs_diff)
    G('C.seeding_differs_only_by_seed', all(r['seeding']['seed'] == s and r['seeding']['role'] == 'EXPERIMENT_SEED'
                                            for s, r in starts.items())
      and len({json.dumps(dict(r['seeding'], seed=None), sort_keys=True) for r in starts.values()}) == 1)
    G('C.loader_workers8', all({k: r['loader'][k] for k in ('batch_size', 'num_workers', 'persistent_workers',
                                                            'drop_last', 'shuffle')}
                               == {'batch_size': 240, 'num_workers': 8, 'persistent_workers': False,
                                   'drop_last': False, 'shuffle': True} for r in starts.values()))
    G('C.lightcnn_environment_equal', len({(d['collected']['lightcnn']['sha256'],
                                            d['collected']['collector']['torch_version'],
                                            mans[s]['dependency_fingerprint'], mans[s]['environment_name'])
                                           for s, d in docs.items()}) == 1)
    G('C.source_and_split_equal', len({(d['collected']['source_recomputed']['commit'],
                                        d['collected']['split_manifest_sha256']) for d in docs.values()}) == 1)
    G('C.population_rederived_equal', len({json.dumps(d['collected']['population_rederived'], sort_keys=True)
                                           for d in docs.values()}) == 1)
    hs = host['seeds']
    rc_ok, rc_diff = True, {}
    base = hs['42']['resolved_config_yaml'].splitlines()
    for s in SEEDS:
        y = hs[str(s)]['resolved_config_yaml']
        rc_ok &= sha256_bytes(y.encode()) == docs[s]['frozen']['resolved_config_sha256'] \
            == hs[str(s)]['resolved_config_sha256']
        lines = y.splitlines()
        rc_ok &= len(lines) == len(base)
        changed = [i for i, (a, b) in enumerate(zip(base, lines)) if a != b]
        rc_diff[s] = [lines[i] for i in changed]
        rc_ok &= all(lines[i] in [f(s) for f in RESOLVED_PER_SEED] and base[i] in [f(42) for f in RESOLVED_PER_SEED]
                     for i in changed)
        rc_ok &= all(sum(f(s) == x for x in lines) == 1 for f in RESOLVED_PER_SEED)
    G('C.resolved_config_differs_only_by_seed', rc_ok, rc_diff)

    # D. training completion / native relation (identical protocol)
    for s, d in docs.items():
        t, m = d['training'], d['collected']['metrics']
        o, pa = t['order_prediction'], t['pairing_audit']
        G(f'D.seed{s}.completed_200_3200', (t['completed_epoch'], t['global_step'], t['step_records'],
                                            t['optimizer_applications'], t['steps_per_epoch'])
          == (200, 3200, 3200, 3200, 16) and m['contiguous'] and m['epoch_rows'] == [3720]
          and d['collected']['run_summary']['completion_status'] == 'completed')
        G(f'D.seed{s}.order_and_pairs_independently_predicted',
          (o['pair_hash_match'], o['epoch_order_match'], o['permutations'], o['epoch_complete_events_ok'])
          == (3200, 200, 200, 200))
        G(f'D.seed{s}.native_pairing', pa['pairs'] == 744000 and all(
            pa[k] == 744000 for k in ('same_subject', 'partner_live', 'spoof_is_spoof', 'train_split', 'casia_msu'))
          and (pa['class_0'], pa['class_1']) == (416000, 328000))
        G(f'D.seed{s}.fresh_run', d['collected']['run_start_events'] == 1 and d['confirmations']['resume_used'] is False
          and not d['collected']['resume_index_present'])
    G('D.same_epoch_plan', len({d['training']['epoch_plan'] for d in docs.values()}) == 1)
    G('D.seed_streams_differ', len({d['run_root_integrity']['metrics_jsonl_sha256'] for d in docs.values()}) == 3)

    # E. numerical health
    for s, d in docs.items():
        n = d['numerical_health']
        G(f'E.seed{s}.finite_active', n['nonfinite_steps'] == 0 and n['corrupted_lines'] == 0
          and n['loss_ranges']['loss_cls'][0] > 0.0 and n['loss_ranges']['loss_pair'][0] > 0.0
          and n['netCls_grad_nonzero_min'] > 0
          and all(r['storages']['nonfinite_elements'] == 0 and r['storages']['float_elements'] > 0
                  for r in d['collected']['official_checkpoints']))

    # F. checkpoints: one selected official epoch-200 netG per seed, independently
    for s, d in docs.items():
        k = d['checkpoints']
        f = k['final_netG']
        summ = d['collected']['run_summary']
        sel = [r for r in d['collected']['official_checkpoints'] if r['selected_for_final']]
        G(f'F.seed{s}.single_final_netG', (k['events'], k['epochs'], k['official_files'], k['resume_sidecars'])
          == (21, CHECKPOINT_EPOCHS, 63, 0)
          and [(r['path'], r['epoch'], r['global_step'], r['checkpoint_type'], r['selection_reason']) for r in sel]
          == [(FINAL_NETG, 200, 3200, 'selected', CHECKPOINT_RULE)]
          and f['sha256'] == sel[0]['sha256'] == sel[0]['sha256_actual'] == summ['final_or_selected_checkpoint_sha256']
          and summ['final_or_selected_checkpoint_path'] == f'{RUN_BASE}/seed_{s}/{FINAL_NETG}'
          and summ['checkpoint_selection_rule'] == CHECKPOINT_RULE)
        if s in KNOWN_FINAL_NETG:
            G(f'F.seed{s}.known_final_sha', f['sha256'] == KNOWN_FINAL_NETG[s])
        rows.append(seed_row(s, d))
    G('F.final_netG_distinct', len({r['final_netG_sha256'] for r in rows}) == 3)
    G('F.run_roots_distinct', sorted(d['run_root'] for d in docs.values())
      == sorted(f'{RUN_BASE}/seed_{s}' for s in SEEDS))

    # G. firewall
    for s, d in docs.items():
        fw = d['firewall']
        by = fw['atime']['by_class']
        cm = {c: v for c, v in by.items() if not c.startswith('siw') and c.split('|')[1] in ('TEST', 'VAL')}
        siw = {c: v for c, v in by.items() if c.startswith('siw')}
        G(f'G.seed{s}.val_0_test_0_siw_0', (fw['val_access'], fw['test_access'], fw['siw_access']) == (0, 0, 0)
          and fw['test_split_accessed'] is False and len(cm) == 4 and len(siw) == 3
          and all(v['atime_changed'] == 0 and v['sensitive_whole_window'] == v['files'] for v in cm.values())
          and all(v['atime_changed'] == 0 for v in siw.values()))
        G(f'G.seed{s}.no_best_or_downstream', not any(d['confirmations'].values()))

    # H. seed42 attempt 1 exclusion
    G('H.attempt1_not_a_member', ATTEMPT1['run_id'] not in {r['run_id'] for r in rows}
      and all(d['run_id'] != ATTEMPT1['run_id'] and mans[s]['git_commit'] != ATTEMPT1['authority']
              for s, d in docs.items()))
    G('H.attempt1_classified_nonselectable', all(
        (d['attempt1']['run_id'], d['attempt1']['classification'], d['attempt1']['selectable'],
         d['attempt1']['eligible_for_bank'], d['attempt1']['eligible_for_downstream'],
         d['confirmations']['attempt1_selected'])
        == (ATTEMPT1['run_id'], ATTEMPT1['classification'], False, False, False, False) for d in docs.values()))
    a1 = host['attempt1_archive']
    G('H.attempt1_archived_untouched', a1['exists'] and not a1['under_runs']
      and a1['sha_listing_digest'] == ATTEMPT1['listing_digest_path_size_sha256'] == docs[42]['collected'][
          'attempt1_archive']['listing_digest_path_size_sha256'] and a1['file_count'] == ATTEMPT1['file_count'], a1)

    # I. read-only host recheck at family closure (run roots still immutable)
    G('I.host_parent_exactly_three_seeds', [p for p in host['parent_listing'] if not p.startswith('.')]
      == sorted(f'seed_{s}' for s in SEEDS), host['parent_listing'])
    for s, d in docs.items():
        h, ri = hs[str(s)], d['run_root_integrity']
        G(f'I.seed{s}.host_matches_evidence', h['final_netG_sha256'] == d['checkpoints']['final_netG']['sha256']
          and h['final_netG_bytes'] == d['checkpoints']['final_netG']['size_bytes']
          and h['metrics_jsonl_sha256'] == ri['metrics_jsonl_sha256'] and h['listing_digest'] == ri['listing_digest']
          and h['sha_listing_digest'] == ri['sha_listing_digest'] and h['file_count'] == ri['file_count']
          == h['sha_listing_count'] and h['listing_stable_during_recheck'])

    text = csv_text(rows)
    doc = {
        'classification': CLASSIFICATION, 'status': STATUS_BLOCKED if G.failed else STATUS_OK,
        'failed_gates': G.failed, 'gates': G.results,
        'method': {'id': 'E06b', 'name': 'DSDG-NATIVE', 'fidelity': FIDELITY, 'track': 'B_NATIVE_FULL_SECONDARY',
                   'execution_mode': 'GLOBAL_STATISTIC_PRESERVING_MICROBATCH_V2'},
        'family': {'seeds': list(SEEDS), 'completed_seeds': sum(r['completion_status'] == 'completed' for r in rows),
                   'scientific_runs': len(rows), 'run_ids': {str(s): RUN_IDS[s] for s in SEEDS},
                   'successful_attempt_per_seed': {str(s): ATTEMPTS[s] for s in SEEDS},
                   'dropped_seeds': [], 'replacement_seeds': [],
                   'composition_note': 'exactly the three frozen experiment seeds; no fourth, replacement, preferred '
                                       'or dropped seed, and no cross-seed choice'},
        'excluded_history': dict(ATTEMPT1, member=False, counted=False, selectable=False, eligible_for_bank=False,
                                 eligible_for_downstream=False,
                                 role='seed42 failed technical history only (epoch 1 / step 16, 0 checkpoints); '
                                      'archived under diagnostics/m6, outside runs/'),
        'scientific_training_authority': AUTHORITY,
        'authority_note': 'Training authority for all three successful seeds is d9ecbef (checkpoint-context fix); the '
                          'seed closure commits and this family closure are evidence/reporting only.',
        'frozen': {'config_sha256': CONFIG_SHA256, 'identities': ids,
                   'freeze_record_sha256': ids['freeze_record_sha256'], 'contract_sha256': ids['contract_sha256'],
                   'environment_lock_sha256': ids['environment_lock_sha256'], 'lightcnn_sha256': ids['lightcnn_sha256'],
                   'source_commit': ids['source_commit'], 'split_manifest_sha256': ids['split_manifest_sha256'],
                   'loader': starts[42]['loader'], 'population': starts[42]['population']},
        'protocol': {'epochs': 200, 'optimizer_steps': 3200, 'steps_per_epoch': 16,
                     'epoch_plan': '15 x 240 + 1 x 120 = 3720 CASIA+MSU TRAIN spoof rows',
                     'native_relation': 'online same-subject TRAIN live partner drawn per spoof row (random.choice in '
                                        'the 8 loader workers); print=0 / replay=1',
                     'nominal_global_batch': 240, 'max_microbatch': 20, 'drop_last': False, 'workers': 8,
                     'checkpoint_epochs': CHECKPOINT_EPOCHS, 'checkpoint_rule': CHECKPOINT_RULE,
                     'final_checkpoint': FINAL_NETG,
                     'final_rule_scope': 'applied independently within each seed; no cross-seed choice'},
        'different_by_design': {'manifest_fields': list(MANIFEST_PER_SEED),
                                'resolved_config_lines': {str(s): v for s, v in rc_diff.items()},
                                'run_start_seeding': {str(s): r['seeding']['seed'] for s, r in starts.items()},
                                'also': 'seed-derived shuffle / partner / noise draws, learned model and checkpoint '
                                        'bytes'},
        'final_checkpoints': rows, 'final_checkpoints_csv': EVIDENCE['csv'],
        'final_checkpoints_csv_sha256': sha256_bytes(text.encode()),
        'aggregation': {'primary_result_rule': 'MEAN_AND_STD_OVER_ALL_THREE_SEEDS',
                        'applies_to': 'future downstream evaluation metrics (none computed yet)',
                        'training_metadata': {'seeds_completed': '3/3', 'epochs_each': 200,
                                              'optimizer_steps_each': 3200, 'checkpoint_events_each': 21,
                                              'official_files_each': 63, 'runner_state_sidecars_each': 0,
                                              'final_checkpoint_exists_each': True, 'numerically_finite_each': True,
                                              'val_access_total': 0, 'test_access_total': 0, 'siw_access_total': 0},
                        'descriptive_training_losses': {
                            'note': 'per-seed epoch-mean training losses; descriptive only, NOT a performance metric',
                            'by_seed': {str(s): {e: {k: d['numerical_health']['loss_mean_at_checkpoint_epochs'][e][k]
                                                     for k in ('total_loss', 'loss_rec')} for e in ('1', '200')}
                                        for s, d in docs.items()}},
                        'wall_clock_note': 'training speed is not an acceptance gate and not a comparison metric'},
        'firewall': {'val_access': 0 if not G.failed else None, 'test_access': 0 if not G.failed else None,
                     'siw_access': 0 if not G.failed else None,
                     'note': 'per-seed stat-only relatime atime snapshots: 992/992 CASIA+MSU TEST and 1088/1088 VAL '
                             'faces sensitive and unchanged in every run; SiW unchanged in every run (SiW TEST/VAL '
                             'fully sensitive); no VAL pass; upstream "test" block = TRAIN visualization, not '
                             'benchmark TEST'},
        'not_completed': {'synthetic_bank': 'NOT_STARTED', 'downstream_fas_evaluation': 'NOT_STARTED',
                          'whole_benchmark': 'NOT_COMPLETED'},
        'confirmations': {'best_seed_selection': False, 'preferred_seed_selection': False, 'seed_dropped': False,
                          'replacement_seed': False, 'additional_seed_launched': False, 'attempt1_counted': False,
                          'val_checkpoint_selection': False, 'test_used': False, 'siw_used': False,
                          'bank_generation': False, 'downstream_evaluation': False, 'e05_started': False,
                          'other_method_started': False, 'gpat_started': False, 'run_root_mutated': False},
        'per_seed_evidence': {str(s): dict(seed_paths(s), closure_commit=CLOSURE_COMMITS[s]) for s in SEEDS},
        'host_recheck': host,
    }
    return doc, text


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--host-recheck', required=True)
    ap.add_argument('--repo-root', default=str(Path(__file__).resolve().parents[1]))
    args = ap.parse_args(argv)
    doc, text = build(args.repo_root, json.loads(Path(args.host_recheck).read_text()))
    root = Path(args.repo_root)
    (root / EVIDENCE['csv']).write_text(text)
    (root / EVIDENCE['closure']).write_text(json.dumps(doc, indent=2, sort_keys=True) + '\n')
    print(doc['status'], doc['failed_gates'])
    return 0 if doc['status'] == STATUS_OK else 1


if __name__ == '__main__':
    sys.exit(main())
