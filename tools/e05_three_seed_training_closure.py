"""E05 three-seed family closure: gates the committed per-seed closures of the E05 PCGAN (controlled architecture
resolution) scientific training family (successful seeds 42, 1337, 2026; training authority efdf753,
CONTROLLED_ADAPTATION, track A_FAIR_COMMON_IDENTITY_FREE) and writes
outputs/audit/E05_THREE_SEED_{TRAINING_CLOSURE.json,FINAL_CHECKPOINTS.csv}. Modelled on
tools/e06b_three_seed_training_closure.py.

Inputs:
  * the committed per-seed evidence outputs/audit/E05_SEED<s>_{COMPLETION.json,CHECKPOINT_MANIFEST.txt,
    COMPLETION.md,EVIDENCE_SHA256.txt}; every per-seed COMPLETION.json is re-derived both with the seed-generic
    tools/e05_seed_completion_audit.py of this checkout and with the copy of that tool in its own closure commit, and
    the evidence bytes are compared with that closure commit;
  * a read-only host recheck document (checkpoint sha256/size, metrics sha256, run-root listing digests,
    resolved_config.yaml text, final host state), collected with tools/e05_family_host_recheck.py and embedded in the
    output so the build re-derives.

This closes generator TRAINING only. No selection, VAL, TEST, bank, downstream or training code is run, and no
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
import types
from pathlib import Path

CLASSIFICATION = 'E05_THREE_SEED_SCIENTIFIC_TRAINING_CLOSURE'
STATUS_OK = 'SCIENTIFIC_TRAINING_COMPLETED_3_OF_3_SEEDS'
STATUS_BLOCKED = 'E05_FAMILY_NEEDS_REVIEW'
SEEDS = (42, 1337, 2026)
AUTHORITY = 'efdf753156045dd2bed52229d9eac869b18d72f8'
CONFIG_SHA256 = '478756e150c427832315800acba954cedf778bac520eee3bde8cabf7359e71de'
METHOD = ('E05', 'PCGAN (controlled architecture resolution)', 'CONTROLLED_ADAPTATION', 'A_FAIR_COMMON_IDENTITY_FREE')
RUN_BASE = '/home/student20261/workdir/GPAT_TransferBench_runtime/runs/m6/E05'
FINAL = 'checkpoints/iteration_4000.pth'
FIRST = 'checkpoints/iteration_0001.pth'
CHECKPOINT_RULE = 'BASELINE_FINAL_STATE_V1'
ITERATIONS, PAIR_ROWS = 4000, 8838
SEED_TOOL = 'tools/e05_seed_completion_audit.py'
RUN_IDS = {42: '18463bf5cd471e55', 1337: '036c806bb2f339b7', 2026: 'b739351daf9779bc'}
CLOSURE_COMMITS = {42: '072ce7d68cbdb0fab7f6d6d85ebce5b4ac1f4cfa',
                   1337: '023beb4d69c63a29e504f4497945ac1da1bb6985',
                   2026: '21d47c24fbe827d296c6b67db1f28393f819c9c9'}
KNOWN_FINAL_SHA = {42: '4ad80b35e759154f741d249db8c6e43622cab916e13dc785ad9273bd65113cb1',
                   1337: 'b6a4cb2c61b1b582f1acd1eb3514f6a15ea49e544e1dac48415004c6ce4f7b34',
                   2026: 'b7f0575f1e061936175ac04d2ac333e1008577bdd3654566c009a2f72bf0b166'}
DRIVER_RULING = {'lock_driver_version': '595.84', 'observed_driver_version': '580.178.04',
                 'driver_policy': 'OWNER_APPROVED_DRIVER_VERSION_ONLY_RUNTIME_COMPATIBILITY'}
# run_manifest fields that differ by design between seeds (everything else must be byte-equal)
MANIFEST_PER_SEED = ('command_line', 'end_utc', 'experiment_seed', 'run_id', 'run_uuid', 'start_utc')
# resolved_config.yaml lines allowed to differ, as functions of the seed
RESOLVED_PER_SEED = (lambda s: f'  experiment_seed: {s}', lambda s: f'  run_dir: {RUN_BASE}/seed_{s}',
                     lambda s: f'    generator_seed: {s}')
# resolved_config.yaml lines that pin the shared protocol (each must appear in every seed's file)
PROTOCOL_LINES = ('  total_iterations: 4000', '  batch_size: 1', '  name: Adam', '  learning_rate: 1.0e-06',
                  '  - 0.9', '  - 0.999', '  total: L_rec + L_recblur + L_advrec + L_advmix + L_pat',
                  '  weighting: ALL_UNIT_WEIGHTED', '    shuffle: true', '    num_workers: 0', '    drop_last: false',
                  '    rule: BASELINE_FINAL_STATE_V1', '    resume: NOT_QUALIFIED_FRESH_ONLY',
                  '    iteration_unit: one D Adam application then one G Adam application')
# run-start fields that differ by design (the seed value only)
RUN_START_PER_SEED = (('loader', 'generator_seed'), ('seeding', 'seed'))
CSV_COLUMNS = ('seed', 'run_id', 'scientific_authority', 'final_checkpoint_path', 'final_checkpoint_sha256',
               'checkpoint_size', 'metrics_sha256', 'run_root_listing_digest', 'completion_status',
               'distinct_train_pairs', 'VAL_access', 'TEST_access', 'observed_driver', 'seed_closure_commit')
EVIDENCE = {'closure': 'outputs/audit/E05_THREE_SEED_TRAINING_CLOSURE.json',
            'csv': 'outputs/audit/E05_THREE_SEED_FINAL_CHECKPOINTS.csv',
            'report': 'outputs/audit/E05_THREE_SEED_TRAINING_CLOSURE.md',
            'sha256': 'outputs/audit/E05_THREE_SEED_EVIDENCE_SHA256.txt'}


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def seed_paths(s):
    a = f'outputs/audit/E05_SEED{s}_'
    return {'completion': a + 'COMPLETION.json', 'manifest': a + 'CHECKPOINT_MANIFEST.txt',
            'report': a + 'COMPLETION.md', 'sha256': a + 'EVIDENCE_SHA256.txt'}


def git_show(repo, commit, path):
    r = subprocess.run(['git', 'show', f'{commit}:{path}'], cwd=repo, capture_output=True)
    return r.stdout if r.returncode == 0 else None


def current_tool(repo):
    spec = importlib.util.spec_from_file_location('e05_seed_audit', repo / SEED_TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def committed_tool(repo, commit):
    """The seed tool exactly as committed in a closure commit (source from git, executed as a fresh module)."""
    src = git_show(repo, commit, SEED_TOOL).decode()
    mod = types.ModuleType(f'e05_seed_audit_{commit[:7]}')
    exec(compile(src, f'{commit[:7]}:{SEED_TOOL}', 'exec'), mod.__dict__)
    return mod, src


class Gates:
    def __init__(self):
        self.results = []

    def __call__(self, name, ok, detail=None):
        self.results.append({'gate': name, 'pass': bool(ok), **({'detail': detail} if detail is not None else {})})
        return ok

    @property
    def failed(self):
        return [r['gate'] for r in self.results if not r['pass']]


def final_entry(d):
    return next(x for x in d['checkpoints']['files'] if x['selected_for_final'])


def seed_row(s, d):
    f, ri, fw = final_entry(d), d['run_root_integrity'], d['firewall']
    return {'seed': s, 'run_id': d['run_id'], 'scientific_authority': d['scientific_authority'],
            'final_checkpoint_path': f['path'], 'final_checkpoint_sha256': f['sha256'],
            'checkpoint_size': f['file_size_bytes'], 'metrics_sha256': ri['metrics_jsonl_sha256'],
            'run_root_listing_digest': ri['sha_listing_digest'],
            'completion_status': d['collected']['run_summary']['completion_status'],
            'distinct_train_pairs': d['distinct_train_pairs'], 'VAL_access': fw['val_access'],
            'TEST_access': fw['test_access'],
            'observed_driver': d['collected']['run_start']['gpu_driver']['live_driver_version'],
            'seed_closure_commit': CLOSURE_COMMITS[s]}


def csv_text(rows):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=CSV_COLUMNS, lineterminator='\n')
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def without(d, path):
    out = json.loads(json.dumps(d))
    out[path[0]].pop(path[1], None)
    return out


def build(repo_root, host):
    repo = Path(repo_root)
    G = Gates()
    docs, rows = {}, []
    tool = current_tool(repo)
    tool_src = (repo / SEED_TOOL).read_text()

    # A. per-seed evidence: SHA manifests, deterministic re-derivation, bytes equal to each closure commit
    present = sorted(p.name for p in (repo / 'outputs/audit').glob('E05_SEED*_COMPLETION.json'))
    G('A.exact_seed_evidence_set', present == sorted(f'E05_SEED{s}_COMPLETION.json' for s in SEEDS), present)
    for s in SEEDS:
        p = seed_paths(s)
        lines = (repo / p['sha256']).read_text().splitlines()
        listed = {name: digest for digest, name in (x.split('  ') for x in lines)}
        G(f'A.seed{s}.evidence_sha256', sorted(listed) == sorted([p['completion'], p['manifest'], p['report']])
          and all(sha256_bytes((repo / n).read_bytes()) == h for n, h in listed.items()))
        d = json.loads((repo / p['completion']).read_bytes())
        docs[s] = d
        rebuilt, text = tool.build(d['collected'], repo)
        G(f'A.seed{s}.rederives_current_tool', json.loads(json.dumps(rebuilt)) == d
          and text == (repo / p['manifest']).read_text())
        old, old_src = committed_tool(repo, CLOSURE_COMMITS[s])
        rebuilt, text = old.build(d['collected'], repo)
        G(f'A.seed{s}.rederives_closure_commit_tool', json.loads(json.dumps(rebuilt)) == d
          and text == (repo / p['manifest']).read_text())
        changed = [(a, b) for a, b in zip(old_src.splitlines(), tool_src.splitlines()) if a != b]
        G(f'A.seed{s}.tool_differs_only_in_run_id_table', len(old_src.splitlines()) == len(tool_src.splitlines())
          and all(a.startswith('RUN_IDS = ') and b.startswith('RUN_IDS = ') for a, b in changed), len(changed))
        G(f'A.seed{s}.equals_closure_commit', all(git_show(repo, CLOSURE_COMMITS[s], p[k]) == (repo / p[k]).read_bytes()
                                                  for k in ('completion', 'manifest', 'report', 'sha256')),
          CLOSURE_COMMITS[s])

    # B. per-seed closure status and identical gate semantics
    for s, d in docs.items():
        G(f'B.seed{s}.closed_clean', d['status'] == f'E05_SEED{s}_COMPLETED_CLEANLY' and d['failed_gates'] == []
          and all(g['pass'] for g in d['gates']) and d['seed'] == s, d['status'])
    names = {s: [g['gate'] for g in d['gates']] for s, d in docs.items()}
    G('B.same_gate_semantics', names[42] == names[1337] == names[2026] and len(names[42]) == 65,
      {s: len(v) for s, v in names.items()})

    # C. authority / identity (common by design)
    mans = {s: d['collected']['run_manifest'] for s, d in docs.items()}
    starts = {s: d['collected']['run_start'] for s, d in docs.items()}
    for s, d in docs.items():
        m, git = mans[s], d['collected']['git']
        G(f'C.seed{s}.authority', d['scientific_authority'] == m['git_commit'] == git['head'] == AUTHORITY
          and m['git_dirty'] is False and git['status_porcelain'] == '' and git['branch'] == 'm6-baselines')
        rid = sha256_bytes(f'E05|{s}|{CONFIG_SHA256}|{AUTHORITY}'.encode())[:16]
        G(f'C.seed{s}.run_id_rule', rid == d['run_id'] == m['run_id'] == RUN_IDS[s], rid)
    G('C.run_ids_distinct', len({d['run_id'] for d in docs.values()}) == 3)
    G('C.method', all((d['method']['id'], d['method']['name'], d['method']['fidelity'], d['method']['track'])
                      == METHOD and mans[s]['fidelity_class'] == METHOD[2] for s, d in docs.items()))
    G('C.frozen_config_sha', all(d['config_sha256'] == CONFIG_SHA256 == mans[s]['config_sha256']
                                 == d['collected']['files_sha256']['config']
                                 == d['collected']['files_sha256']['config_snapshot'] for s, d in docs.items()))
    common = {s: {k: v for k, v in m.items() if k not in MANIFEST_PER_SEED} for s, m in mans.items()}
    diff_keys = sorted(k for k in set().union(*common.values()) if len({json.dumps(c.get(k), sort_keys=True)
                                                                         for c in common.values()}) > 1)
    G('C.manifest_common_fields_equal', diff_keys == [] and common[42].keys() == common[1337].keys()
      == common[2026].keys(), diff_keys)
    G('C.command_line_differs_only_by_seed', len({m['command_line'].replace(f'--seed {s} ', '--seed <S> ')
                                                  for s, m in mans.items()}) == 1
      and all(f'--seed {s} ' in m['command_line'] and '--resume' not in m['command_line'] for s, m in mans.items()))
    ids = starts[42]['identities']
    G('C.identities_equal', all(r['identities'] == ids for r in starts.values())
      and ids['config_sha256'] == CONFIG_SHA256 and ids['fidelity_class'] == METHOD[2], sorted(ids))
    norm = {s: r for s, r in starts.items()}
    for path in RUN_START_PER_SEED:
        norm = {s: without(r, path) for s, r in norm.items()}
    rs_diff = sorted(k for k in norm[42] if len({json.dumps(r.get(k), sort_keys=True) for r in norm.values()}) > 1)
    G('C.run_start_equal_except_seed', rs_diff == [] and all(r['loader']['generator_seed'] == r['seeding']['seed'] == s
                                                              for s, r in starts.items()), rs_diff)
    G('C.files_identities_equal', len({json.dumps(d['collected']['files_sha256'], sort_keys=True)
                                       for d in docs.values()}) == 1)
    G('C.environment_equal', len({json.dumps(d['collected']['environment_recheck'], sort_keys=True)
                                  for d in docs.values()}) == 1
      and len({json.dumps(d['environment'], sort_keys=True) for d in docs.values()}) == 1
      and all(d['collected']['environment_recheck']['differences_other_than_driver'] == [] for d in docs.values()))
    G('C.extensions_equal_no_rebuild', len({json.dumps(d['collected']['extensions'], sort_keys=True)
                                            for d in docs.values()}) == 1
      and all(d['collected']['run_start_jit_build_invoked'] is False for d in docs.values()))
    G('C.driver_ruling_consistent', all({k: d['driver_ruling'][k] for k in DRIVER_RULING} == DRIVER_RULING
                                        and starts[s]['gpu_driver']['live_driver_version']
                                        == DRIVER_RULING['observed_driver_version']
                                        and starts[s]['gpu_driver']['lock_driver_version']
                                        == DRIVER_RULING['lock_driver_version'] for s, d in docs.items())
      and len({json.dumps(r['gpu_driver'], sort_keys=True) for r in starts.values()}) == 1)
    hs = host['seeds']
    rc_ok, rc_diff = True, {}
    base = hs['42']['resolved_config_yaml'].splitlines()
    for s in SEEDS:
        y = hs[str(s)]['resolved_config_yaml']
        rc_ok &= sha256_bytes(y.encode()) == docs[s]['run_root_integrity']['small_file_sha256']['resolved_config.yaml'] \
            == hs[str(s)]['resolved_config_sha256']
        lines = y.splitlines()
        rc_ok &= len(lines) == len(base)
        changed = [i for i, (a, b) in enumerate(zip(base, lines)) if a != b]
        rc_diff[s] = [lines[i] for i in changed]
        rc_ok &= all(lines[i] in [f(s) for f in RESOLVED_PER_SEED] and base[i] in [f(42) for f in RESOLVED_PER_SEED]
                     for i in changed)
        rc_ok &= all(sum(f(s) == x for x in lines) == 1 for f in RESOLVED_PER_SEED)
        rc_ok &= all(x in lines for x in PROTOCOL_LINES)
    G('C.resolved_config_differs_only_by_seed', rc_ok, rc_diff)

    # D. training completion / data protocol (identical protocol, seed-derived order)
    for s, d in docs.items():
        m = d['collected']['metrics']
        o, pa = m['order'], m['pairs']
        G(f'D.seed{s}.completed_4000', d['iterations'] == m['records'] == ITERATIONS and m['global_step_is_1_to_n']
          and d['collected']['run_summary']['completion_status'] == 'completed'
          and d['collected']['run_summary']['benchmark_iterations'] == ITERATIONS)
        G(f'D.seed{s}.applications', d['optimizer_applications'] == {'D': ITERATIONS, 'G': ITERATIONS,
                                                                    'hook_D': ITERATIONS, 'hook_G': ITERATIONS,
                                                                    'total': 2 * ITERATIONS}
          and m['applications_exact'] == ITERATIONS)
        G(f'D.seed{s}.order_independently_predicted', (o['match'], o['predicted_positions']) == (ITERATIONS, ITERATIONS)
          and o['sampler_order_sha256'] == o['predicted_order_sha256'])
        G(f'D.seed{s}.distinct_train_pairs', d['distinct_train_pairs'] == o['distinct_pair_ids']
          == o['distinct_sampler_indices'] == ITERATIONS
          and all(pa[k] == ITERATIONS for k in ('manifest_row_match', 'pair_sha256_match', 'source_spoof_target_live',
                                               'both_faces_train_same_dataset',
                                               'face_png_sha256_match_m2_accounting'))
          and m['consumed_splits'] == ['TRAIN'])
        pm = starts[s]['pair_manifest']
        G(f'D.seed{s}.source_target_disjoint', pm['rows'] == PAIR_ROWS and pm['unique_source_ids'] == PAIR_ROWS
          and pm['unique_sample_ids'] == pm['unique_source_ids'] + pm['unique_target_ids'] and pm['splits'] == ['TRAIN'])
        G(f'D.seed{s}.input_contract', m['input_contract'] == ITERATIONS)
        G(f'D.seed{s}.fresh_run', d['resume'] == 'NOT_QUALIFIED_FRESH_ONLY' and mans[s]['resume'] == d['resume']
          and d['collected']['event_counts'].get('e05_run_start') == 1 and d['confirmations']['resume_used'] is False
          and d['collected']['resume_files'] == [] and d['collected']['runner_state_files'] == [])
    G('D.seed_streams_differ', len({d['collected']['metrics']['order']['sampler_order_sha256'] for d in docs.values()})
      == 3 and len({d['run_root_integrity']['metrics_jsonl_sha256'] for d in docs.values()}) == 3)

    # E. numerical health
    for s, d in docs.items():
        m = d['collected']['metrics']
        G(f'E.seed{s}.finite_active', m['losses_finite'] == m['g_terms_active'] == m['gradients_exact'] == ITERATIONS
          and m['nonfinite_values'] == 0 and m['g_total_unit_sum_max_relative_error'] <= 1e-5
          and m['d_losses_finite_positive'] == ITERATIONS
          and all(x['load']['nonfinite_floating_tensors'] == 0 for x in d['collected']['checkpoints'])
          and d['collected']['logs']['stderr.log']['bytes'] == 0)

    # F. checkpoints: one selected iteration-4000 state per seed, independently
    for s, d in docs.items():
        summ = d['collected']['run_summary']
        files = d['checkpoints']['files']
        sel = [x for x in files if x['selected_for_final']]
        G(f'F.seed{s}.two_events_one_selected', len(d['collected']['checkpoint_events']) == 2
          and [(x['path'], x['checkpoint_type'], x['selected_for_final']) for x in files]
          == [(FIRST, 'periodic', False), (FINAL, 'selected', True)]
          and [(x['path'], x['global_step'], x['selection_reason']) for x in sel] == [(FINAL, ITERATIONS,
                                                                                      CHECKPOINT_RULE)]
          and sel[0]['sha256'] == summ['final_or_selected_checkpoint_sha256'] == KNOWN_FINAL_SHA[s]
          and summ['final_or_selected_checkpoint_path'] == f'{RUN_BASE}/seed_{s}/{FINAL}'
          and summ['checkpoint_selection_rule'] == CHECKPOINT_RULE
          and d['checkpoints']['final_selected'] == {'path': FINAL, 'sha256': KNOWN_FINAL_SHA[s], 'rule': CHECKPOINT_RULE}
          and d['checkpoints']['resume_sidecars'] == 0)
        ck = {x['global_step']: x for x in d['collected']['checkpoints']}
        G(f'F.seed{s}.final_provenance', ck[ITERATIONS]['load']['provenance']['optimizer_applications']
          == {'D': ITERATIONS, 'G': ITERATIONS, 'hook_D': ITERATIONS, 'hook_G': ITERATIONS, 'total': 2 * ITERATIONS}
          and ck[ITERATIONS]['load']['provenance']['selection_reason'] == CHECKPOINT_RULE)
        rows.append(seed_row(s, d))
    G('F.final_checkpoints_distinct', len({r['final_checkpoint_sha256'] for r in rows}) == 3)
    G('F.run_roots_distinct', sorted(d['run_root'] for d in docs.values())
      == sorted(f'{RUN_BASE}/seed_{s}' for s in SEEDS))

    # G. firewall / confirmations
    for s, d in docs.items():
        fw = d['firewall']
        vt = {c: v for c, v in fw['atime']['by_class'].items() if c.split('|')[1] in ('TEST', 'VAL')}
        G(f'G.seed{s}.val_0_test_0', (fw['val_access'], fw['test_access'], fw['val_changed'], fw['test_changed'])
          == (0, 0, 0, 0) and (fw['val_files'], fw['test_files']) == (3121, 3027) and len(vt) == 6
          and all(v['atime_changed'] == 0 and v['sensitive_whole_window'] == v['files'] and v['consumed'] == 0
                  for v in vt.values())
          and mans[s]['test_split_accessed'] is False and mans[s]['val_split_accessed'] is False)
        G(f'G.seed{s}.no_best_or_downstream', not any(d['confirmations'].values()))

    # H. read-only host recheck at family closure (run roots still immutable, nothing running)
    G('H.host_parent_exactly_three_seeds', [p for p in host['parent_listing'] if not p.startswith('.')]
      == sorted(f'seed_{s}' for s in SEEDS), host['parent_listing'])
    for s, d in docs.items():
        h, ri = hs[str(s)], d['run_root_integrity']
        G(f'H.seed{s}.host_matches_evidence', all(h['checkpoints'][x['path']] == {'sha256': x['sha256'],
                                                                                'bytes': x['file_size_bytes']}
                                                  for x in d['checkpoints']['files'])
          and h['metrics_jsonl_sha256'] == ri['metrics_jsonl_sha256'] and h['sha_listing_digest']
          == ri['sha_listing_digest'] and h['file_count'] == ri['file_count'] and h['listing_stable_during_recheck'])
    st = host['host_state']
    G('H.host_final_state', st['hostname'] == 'nd-System-Product-Name'
      and st['gpu'] == 'NVIDIA GeForce RTX 3090, 580.178.04, GPU-b722cd9d-c9fa-223c-915e-cc09d9b862b7'
      and st['method_processes'] == [] and st['gpu_compute_apps'] == []
      and all(v == '1 0' for v in st['e05_tmux_panes'].values()), st['method_processes'])
    G('H.gpu_checkout_authority_clean', st['git_head'] == AUTHORITY and st['git_branch'] == 'm6-baselines'
      and st['git_status_porcelain'] == '')

    text = csv_text(rows)
    doc = {
        'classification': CLASSIFICATION, 'status': STATUS_BLOCKED if G.failed else STATUS_OK,
        'failed_gates': G.failed, 'gates': G.results,
        'method': dict(zip(('id', 'name', 'fidelity', 'track'), METHOD)),
        'family': {'seeds': list(SEEDS), 'completed_seeds': sum(r['completion_status'] == 'completed' for r in rows),
                   'scientific_runs': len(rows), 'run_ids': {str(s): RUN_IDS[s] for s in SEEDS},
                   'dropped_seeds': [], 'replacement_seeds': [], 'preferred_seed': None, 'best_seed': None,
                   'composition_note': 'exactly the three frozen experiment seeds; no fourth, replacement, preferred '
                                       'or dropped seed, and no cross-seed choice'},
        'scientific_training_authority': AUTHORITY,
        'authority_note': 'Training authority for all three successful seeds is efdf753; the seed closure commits and '
                          'this family closure are evidence/reporting only.',
        'frozen': {'config_sha256': CONFIG_SHA256, 'identities': ids, 'files_sha256': docs[42]['collected'][
            'files_sha256'], 'loader_except_seed': without(starts[42], ('loader', 'generator_seed'))['loader'],
            'precision': starts[42]['precision'], 'pair_manifest': starts[42]['pair_manifest']},
        'protocol': {'benchmark_iterations': ITERATIONS, 'iteration_unit': 'one D Adam application then one G Adam '
                                                                          'application',
                     'optimizer': 'Adam lr 1e-6 betas (0.9, 0.999), D = ImageD+PatchD, G = Encoder+Generator',
                     'g_losses': ['L_rec', 'L_recblur', 'L_advrec', 'L_advmix', 'L_pat'],
                     'g_total': 'unit-weighted sum', 'batch_size': 1, 'num_workers': 0, 'drop_last': False,
                     'pair_order': 'seeded shuffle: torch.Generator().manual_seed(seed), one int64 draw, then '
                                   'randperm(8838); iteration t uses sampler position t-1; 4000 distinct pairs',
                     'pair_roles': 'x_src = source_spoof_id face (pattern/artifact), x_tgt = target_live_id face '
                                   '(content)',
                     'input_mapping': 'canonical RGB256 uint8 -> float32 (x/255-0.5)/0.5 in [-1,1]; no resize, crop, '
                                      'flip or augmentation',
                     'checkpoints': {'1': 'periodic, not selected', '4000': 'selected, BASELINE_FINAL_STATE_V1'},
                     'checkpoint_rule': CHECKPOINT_RULE, 'final_checkpoint': FINAL,
                     'final_rule_scope': 'applied independently within each seed; no cross-seed choice',
                     'resume': 'NOT_QUALIFIED_FRESH_ONLY',
                     'resolution_wording': ('owner-approved controlled implementation resolutions; not native, '
                                            'faithful or official PCGAN choices')},
        'different_by_design': {'manifest_fields': list(MANIFEST_PER_SEED),
                                'resolved_config_lines': {str(s): v for s, v in rc_diff.items()},
                                'run_start_fields': ['.'.join(p) for p in RUN_START_PER_SEED],
                                'also': 'seed-derived pair order and model initialisation/noise; learned model and '
                                        'checkpoint bytes'},
        'driver_ruling': dict(DRIVER_RULING, scope='only the NVIDIA driver version differs from the lock; GPU model, '
                                                   'GPU UUID and every other lock field are exact in all three runs'),
        'final_checkpoints': rows, 'final_checkpoints_csv': EVIDENCE['csv'],
        'final_checkpoints_csv_sha256': sha256_bytes(text.encode()),
        'aggregation': {'primary_result_rule': 'MEAN_AND_STD_OVER_ALL_THREE_SEEDS',
                        'applies_to': 'future downstream evaluation metrics (none computed yet)',
                        'training_metadata': {'seeds_completed': '3/3', 'iterations_each': ITERATIONS,
                                              'optimizer_applications_each': {'D': ITERATIONS, 'G': ITERATIONS,
                                                                              'total': 2 * ITERATIONS},
                                              'distinct_train_pairs_each': ITERATIONS, 'checkpoint_events_each': 2,
                                              'selected_checkpoints_each': 1, 'resume_sidecars_each': 0,
                                              'numerically_finite_each': True, 'val_access_total': 0,
                                              'test_access_total': 0},
                        'descriptive_training_losses': {
                            'note': 'per-seed last-100-iteration mean training losses; descriptive only, NOT a '
                                    'performance metric and not used for any choice',
                            'by_seed': {str(s): {k: d['losses']['stats'][k]['mean_last_100']
                                                 for k in ('L_G_total', 'L_D_total')} for s, d in docs.items()}},
                        'wall_clock_note': 'training speed is not an acceptance gate and not a comparison metric'},
        'firewall': {'val_access': 0 if not G.failed else None, 'test_access': 0 if not G.failed else None,
                     'note': 'per-seed stat-only relatime atime snapshots: 3121/3121 VAL and 3027/3027 TEST faces '
                             'atime-sensitive and unchanged in every run; TRAIN scope proven by pair manifest, '
                             'independent sampler order and M2 face digests'},
        'not_completed': {'synthetic_bank': 'NOT_STARTED', 'downstream_fas_evaluation': 'NOT_STARTED',
                          'whole_benchmark': 'NOT_COMPLETED'},
        'confirmations': {'best_seed_selection': False, 'preferred_seed_selection': False, 'seed_dropped': False,
                          'replacement_seed': False, 'additional_seed_launched': False,
                          'val_checkpoint_selection': False, 'test_used': False, 'bank_generation': False,
                          'downstream_evaluation': False, 'e03_started': False, 'other_method_started': False,
                          'gpat_started': False, 'run_root_mutated': False},
        'per_seed_evidence': {str(s): dict(seed_paths(s), tool=SEED_TOOL, closure_commit=CLOSURE_COMMITS[s])
                              for s in SEEDS},
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
