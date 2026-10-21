"""E06c three-seed family closure: gates the committed per-seed closures of the E06c DSDG-BIN-IDFREE scientific
training family (seeds 42, 1337, 2026; launch authority 84809868, CONTROLLED_ADAPTATION, DEV-020) and writes
outputs/audit/E06C_THREE_SEED_{TRAINING_CLOSURE.json,FINAL_CHECKPOINTS.csv}.

Inputs:
  * the committed per-seed evidence outputs/audit/E06C_SEED<s>_{COMPLETION.json,CHECKPOINT_MANIFEST.txt,
    EVIDENCE_SHA256.txt}; every per-seed COMPLETION.json is re-derived with its own closure tool and compared with
    the bytes committed by its closure commit;
  * a read-only host recheck document (final netG sha256/size, run-root listing digests, resolved_config.yaml
    text), collected with tools/e06c_family_host_recheck.py and embedded in the output so the build re-derives.

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
from pathlib import Path

CLASSIFICATION = 'E06C_THREE_SEED_SCIENTIFIC_TRAINING_CLOSURE'
STATUS_OK = 'SCIENTIFIC_TRAINING_COMPLETED_3_OF_3_SEEDS'
STATUS_BLOCKED = 'E06C_FAMILY_NEEDS_REVIEW'
SEEDS = (42, 1337, 2026)
AUTHORITY = '84809868acf472af5ba8cf6ab47165ae038e1012'
CONFIG_SHA256 = '7176dd4cd49007320ab0c44519e7efde60568098503077303218bd913b6b5002'
TRAIN_RELATION_SHA256 = 'a5e4fdaef236f15730c7e3885b537e08e995faffbe654167fc940f44bbc75243'
SPEC_SHA256 = 'f7d2371682ad9504f59f0c6984e4b1a741f855f1b1f60838f9d60b160281489e'
RUN_BASE = '/home/student20261/workdir/GPAT_TransferBench_runtime/runs/m6/E06c'
FINAL_NETG = 'checkpoints/netG_model_epoch_200_iter_0.pth'
CHECKPOINT_RULE = 'OFFICIAL_GENERATOR_EPOCH_200'
CHECKPOINT_EPOCHS = [1] + list(range(10, 201, 10))
KNOWN_FINAL_NETG = {42: '665684fddf11ec28b95e2a98122c85bfc5352bb8ea69ecb60caac9418c0871d9',
                    1337: '43ffa30dc7eaa568bc0b573a34978f897f46156232ee1a7e71d7f994beb8f336'}
CLOSURE_COMMITS = {42: 'cf5ed6cbd4fbc7f989136e33577ee298619fe6a4',
                   1337: '9281c79d1c1f8e2b34199ff5434a6731d6e8bc9a',
                   2026: '81a8ef6d1e1dec97452b32955c17d4e8718abcb7'}
# run_manifest fields that differ by design between seeds (everything else must be byte-equal)
MANIFEST_PER_SEED = ('command_line', 'end_utc', 'experiment_seed', 'resolved_config_sha256', 'run_id', 'run_uuid',
                     'start_utc')
# resolved_config.yaml lines allowed to differ, as functions of the seed
RESOLVED_PER_SEED = (lambda s: f'  experiment_seed: {s}', lambda s: f'  run_dir: {RUN_BASE}/seed_{s}',
                     lambda s: f'    generator_seed: {s}')
CSV_COLUMNS = ('seed', 'run_id', 'launch_authority', 'final_checkpoint_path', 'final_checkpoint_sha256',
               'checkpoint_size_bytes', 'metrics_sha256', 'run_root_listing_digest', 'completion_status',
               'completed_epoch', 'global_step', 'test_access', 'val_selection', 'seed_closure_commit')
EVIDENCE = {'closure': 'outputs/audit/E06C_THREE_SEED_TRAINING_CLOSURE.json',
            'csv': 'outputs/audit/E06C_THREE_SEED_FINAL_CHECKPOINTS.csv',
            'report': 'outputs/audit/E06C_THREE_SEED_TRAINING_CLOSURE.md',
            'sha256': 'outputs/audit/E06C_THREE_SEED_EVIDENCE_SHA256.txt'}


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def seed_paths(s):
    a = f'outputs/audit/E06C_SEED{s}_'
    return {'completion': a + 'COMPLETION.json', 'manifest': a + 'CHECKPOINT_MANIFEST.txt',
            'report': a + 'COMPLETION.md', 'sha256': a + 'EVIDENCE_SHA256.txt',
            'tool': f'tools/e06c_seed{s}_completion_audit.py'}


def seed_tool(repo, s):
    spec = importlib.util.spec_from_file_location(f'e06c_s{s}', repo / seed_paths(s)['tool'])
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
    f = d['checkpoints']['final_netG']
    return {'seed': s, 'run_id': d['run_id'], 'launch_authority': d['launch_authority_commit'],
            'final_checkpoint_path': f['path'], 'final_checkpoint_sha256': f['sha256'],
            'checkpoint_size_bytes': f['size_bytes'], 'metrics_sha256': d['run_root_integrity']['metrics_jsonl_sha256'],
            'run_root_listing_digest': d['run_root_integrity']['listing_digest'],
            'completion_status': d['collected']['run_summary']['completion_status'],
            'completed_epoch': d['training']['completed_epoch'], 'global_step': d['training']['global_step'],
            'test_access': d['firewall']['test_benchmark_access'],
            'val_selection': d['firewall']['val_checkpoint_selection_access'],
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
    present = sorted(p.name for p in (repo / 'outputs/audit').glob('E06C_SEED*_COMPLETION.json'))
    G('A.exact_seed_evidence_set', present == sorted(f'E06C_SEED{s}_COMPLETION.json' for s in SEEDS), present)
    for s in SEEDS:
        p = seed_paths(s)
        lines = (repo / p['sha256']).read_text().splitlines()
        listed = {name: digest for digest, name in (x.split('  ') for x in lines)}
        G(f'A.seed{s}.evidence_sha256', sorted(listed) == sorted([p['completion'], p['manifest'], p['report']])
          and all(sha256_bytes((repo / n).read_bytes()) == h for n, h in listed.items()))
        raw = (repo / p['completion']).read_bytes()
        d = json.loads(raw)
        docs[s] = d
        rebuilt, text = seed_tool(repo, s).build(d['collected'], repo)
        G(f'A.seed{s}.rederives', json.loads(json.dumps(rebuilt)) == d
          and text == (repo / p['manifest']).read_text())
        G(f'A.seed{s}.equals_closure_commit', all(git_show(repo, CLOSURE_COMMITS[s], p[k]) == (repo / p[k]).read_bytes()
                                                  for k in ('completion', 'manifest', 'report', 'sha256', 'tool')),
          CLOSURE_COMMITS[s])

    # B. per-seed closure status and identical gate semantics
    for s, d in docs.items():
        G(f'B.seed{s}.closed_clean', d['status'] == f'E06C_SEED{s}_COMPLETED_CLEANLY' and d['failed_gates'] == []
          and all(g['pass'] for g in d['gates']) and d['seed'] == s, d['status'])
    names = {s: [g['gate'] for g in d['gates']] for s, d in docs.items()}
    G('B.same_gate_count', {len(v) for v in names.values()} == {86}, {s: len(v) for s, v in names.items()})
    norm = {s: [n for n in v if not n.startswith('H.parent_only')] for s, v in names.items()}
    G('B.same_gate_semantics', norm[42] == norm[1337] == norm[2026])

    # C. authority / identity (common by design)
    mans = {s: d['collected']['run_manifest'] for s, d in docs.items()}
    for s, d in docs.items():
        m, git = mans[s], d['collected']['git']
        G(f'C.seed{s}.authority', d['launch_authority_commit'] == m['git_commit'] == git['head'] == AUTHORITY
          and m['git_dirty'] is False and git['status_porcelain'] == '' and git['branch'] == 'm6-baselines')
        rid = sha256_bytes(f'E06c|{s}|{CONFIG_SHA256}|{AUTHORITY}'.encode())[:16]
        G(f'C.seed{s}.run_id_rule', rid == d['run_id'] == m['run_id'], rid)
    G('C.run_ids_distinct', len({d['run_id'] for d in docs.values()}) == 3)
    G('C.method', all((d['method']['id'], d['method']['name'], d['method']['fidelity_class'], d['method']['deviation'],
                       d['method']['execution_mode']) == ('E06c', 'DSDG-BIN-IDFREE', 'CONTROLLED_ADAPTATION', 'DEV-020',
                                                          'GLOBAL_STATISTIC_PRESERVING_MICROBATCH_V2')
                      for d in docs.values()))
    G('C.frozen_shas', all((d['frozen']['config_sha256'], d['frozen']['spec_sha256'],
                            d['frozen']['train_relation_sha256']) == (CONFIG_SHA256, SPEC_SHA256, TRAIN_RELATION_SHA256)
                           for d in docs.values()))
    common = {s: {k: v for k, v in m.items() if k not in MANIFEST_PER_SEED} for s, m in mans.items()}
    diff_keys = sorted(k for k in set().union(*common.values()) if len({json.dumps(c.get(k), sort_keys=True)
                                                                         for c in common.values()}) > 1)
    G('C.manifest_common_fields_equal', diff_keys == [] and common[42].keys() == common[1337].keys()
      == common[2026].keys(), diff_keys)
    G('C.manifest_seed_fields', all(m['experiment_seed'] == s and m['run_id'] == docs[s]['run_id']
                                    for s, m in mans.items()))
    G('C.command_line_differs_only_by_seed', len({m['command_line'].replace(f'--seed {s} ', '--seed <S> ')
                                                  for s, m in mans.items()}) == 1
      and all(f'--seed {s} ' in m['command_line'] and '--resume-state' not in m['command_line']
              for s, m in mans.items()))
    ids = mans[42]['identities']
    G('C.identities_equal', all(m['identities'] == ids for m in mans.values())
      and all(d['collected']['identities_recomputed'] == ids for d in docs.values()), sorted(ids))
    G('C.lightcnn_environment_equal', len({(d['collected']['lightcnn']['sha256'],
                                            d['collected']['collector']['torch_version'],
                                            mans[s]['dependency_fingerprint'], mans[s]['environment_name'])
                                           for s, d in docs.items()}) == 1)
    G('C.m6d5e_files_equal', all(d['collected']['m6d5e_files_now'] == mans[42]['m6d5e_file_sha256']
                                 for d in docs.values()))
    hs = host['seeds']
    rc_ok, rc_diff = True, {}
    base = hs['42']['resolved_config_yaml'].splitlines()
    for s in SEEDS:
        y = hs[str(s)]['resolved_config_yaml']
        rc_ok &= sha256_bytes(y.encode()) == mans[s]['resolved_config_sha256'] == hs[str(s)]['resolved_config_sha256']
        lines = y.splitlines()
        rc_ok &= len(lines) == len(base)
        changed = [i for i, (a, b) in enumerate(zip(base, lines)) if a != b]
        rc_diff[s] = [lines[i] for i in changed]
        rc_ok &= all(lines[i] in [f(s) for f in RESOLVED_PER_SEED] and base[i] in [f(42) for f in RESOLVED_PER_SEED]
                     for i in changed)
        rc_ok &= all(sum(f(s) == x for x in lines) == 1 for f in RESOLVED_PER_SEED)
    G('C.resolved_config_differs_only_by_seed', rc_ok, rc_diff)

    # D. training completion (identical protocol)
    for s, d in docs.items():
        t, m = d['training'], d['collected']['metrics']
        o = t['order_prediction']
        G(f'D.seed{s}.completed_200_7400', (t['completed_epoch'], t['global_step'], t['step_records'],
                                            t['optimizer_applications'], t['steps_per_epoch'])
          == (200, 7400, 7400, 7400, 37) and m['contiguous'] and m['epoch_rows'] == [8838]
          and d['collected']['run_summary']['completion_status'] == 'completed')
        G(f'D.seed{s}.order_independently_predicted', (o['batch_hash_match'], o['epoch_order_match'],
                                                       o['permutations'], o['epoch_complete_events_ok'])
          == (7400, 200, 200, 200))
        G(f'D.seed{s}.fresh_run', d['collected']['resume_reconciliation_events'] == 0)
    G('D.same_epoch_plan', len({d['training']['epoch_plan'] for d in docs.values()}) == 1)
    G('D.seed_streams_differ', len({d['run_root_integrity']['metrics_jsonl_sha256'] for d in docs.values()}) == 3)

    # E. numerical health
    for s, d in docs.items():
        n = d['numerical_health']
        G(f'E.seed{s}.finite', n['nonfinite_steps'] == 0 and n['corrupted_lines'] == 0 and n['sidecars_checked'] == 21
          and n['loss_ranges']['loss_cls'] == [0.0, 0.0] and n['loss_ranges']['loss_pair'] == [0.0, 0.0]
          and all(v['equal_multiset'] for v in n['final_sidecar']['official_storage_match'].values()))

    # F. checkpoints: one selected official epoch-200 netG per seed, independently
    for s, d in docs.items():
        k = d['checkpoints']
        f = k['final_netG']
        sel = [r for r in d['collected']['official_checkpoints'] if r['selected_for_final']]
        G(f'F.seed{s}.single_final_netG', (k['events'], k['epochs'], k['official_files']) == (21, CHECKPOINT_EPOCHS, 63)
          and [(r['path'], r['epoch'], r['global_step'], r['checkpoint_type'], r['selection_reason']) for r in sel]
          == [(FINAL_NETG, 200, 7400, 'selected', CHECKPOINT_RULE)]
          and f['sha256'] == sel[0]['sha256'] == sel[0]['sha256_actual']
          == d['collected']['run_summary']['final_or_selected_checkpoint_sha256']
          and d['collected']['run_summary']['final_or_selected_checkpoint_path'] == f'{RUN_BASE}/seed_{s}/{FINAL_NETG}'
          and d['collected']['run_summary']['checkpoint_selection_rule'] == CHECKPOINT_RULE)
        if s in KNOWN_FINAL_NETG:
            G(f'F.seed{s}.known_final_sha', f['sha256'] == KNOWN_FINAL_NETG[s])
        rows.append(seed_row(s, d))
    G('F.final_netG_distinct', len({r['final_checkpoint_sha256'] for r in rows}) == 3)
    G('F.run_roots_distinct', sorted(d['run_root'] for d in docs.values())
      == sorted(f'{RUN_BASE}/seed_{s}' for s in SEEDS))

    # G. firewall
    for s, d in docs.items():
        fw = d['firewall']
        test = {c: v for c, v in fw['atime']['by_class'].items() if c.split('_')[0] == 'TEST'}
        G(f'G.seed{s}.test_0_val_selection_0', fw['test_benchmark_access'] == 0
          and fw['val_checkpoint_selection_access'] == 0 and fw['val_split_accessed'] is False
          and fw['test_split_accessed'] is False and test
          and all(v['atime_changed'] == 0 and v['sensitive_whole_window'] == v['files'] for v in test.values()))
        G(f'G.seed{s}.no_best_or_downstream', not any(d['confirmations'].values()))

    # H. read-only host recheck at family closure (run roots still immutable)
    G('H.host_parent_exactly_three_seeds', [p for p in host['parent_listing'] if not p.startswith('.')]
      == sorted(f'seed_{s}' for s in SEEDS), host['parent_listing'])
    for s, d in docs.items():
        h = hs[str(s)]
        G(f'H.seed{s}.host_matches_evidence', h['final_netG_sha256'] == d['checkpoints']['final_netG']['sha256']
          and h['final_netG_bytes'] == d['checkpoints']['final_netG']['size_bytes']
          and h['metrics_jsonl_sha256'] == d['run_root_integrity']['metrics_jsonl_sha256']
          and h['listing_digest'] == d['run_root_integrity']['listing_digest']
          and h['file_count'] == d['run_root_integrity']['file_count'] and h['listing_stable_during_recheck'])

    text = csv_text(rows)
    doc = {
        'classification': CLASSIFICATION, 'status': STATUS_BLOCKED if G.failed else STATUS_OK,
        'failed_gates': G.failed, 'gates': G.results,
        'method': {'id': 'E06c', 'name': 'DSDG-BIN-IDFREE', 'fidelity_class': 'CONTROLLED_ADAPTATION',
                   'deviation': 'DEV-020', 'execution_mode': 'GLOBAL_STATISTIC_PRESERVING_MICROBATCH_V2'},
        'family': {'seeds': list(SEEDS), 'completed_seeds': sum(r['completion_status'] == 'completed' for r in rows),
                   'scientific_runs': len(rows), 'dropped_seeds': [], 'replacement_seeds': [],
                   'composition_note': 'exactly the three frozen experiment seeds; no fourth, replacement, best, '
                                       'preferred or dropped seed'},
        'scientific_training_authority': AUTHORITY,
        'authority_note': 'Launch authority for all three seeds is 84809868; the seed closure commits and this family '
                          'closure are evidence/reporting only.',
        'frozen': {'config_sha256': CONFIG_SHA256, 'spec_sha256': SPEC_SHA256,
                   'train_relation_sha256': TRAIN_RELATION_SHA256, 'identities': ids,
                   'environment_lock_sha256': ids['environment_lock_sha256'], 'lightcnn_sha256': ids['lightcnn_sha256'],
                   'source_commit': ids['source_commit']},
        'protocol': {'epochs': 200, 'optimizer_steps': 7400, 'steps_per_epoch': 37,
                     'epoch_plan': '36 x 240 + 1 x 198 = 8838 TRAIN pairs', 'nominal_global_batch': 240,
                     'max_microbatch': 20, 'drop_last': False, 'checkpoint_epochs': CHECKPOINT_EPOCHS,
                     'checkpoint_rule': CHECKPOINT_RULE, 'final_checkpoint': FINAL_NETG,
                     'final_rule_scope': 'applied independently within each seed; no cross-seed choice'},
        'different_by_design': {'manifest_fields': list(MANIFEST_PER_SEED),
                                'resolved_config_lines': {str(s): v for s, v in rc_diff.items()},
                                'also': 'data shuffle / random draws, learned model and checkpoint bytes'},
        'final_checkpoints': rows, 'final_checkpoints_csv': EVIDENCE['csv'],
        'final_checkpoints_csv_sha256': sha256_bytes(text.encode()),
        'aggregation': {'primary_result_rule': 'MEAN_AND_STD_OVER_ALL_THREE_SEEDS',
                        'applies_to': 'future downstream evaluation metrics (none computed yet)',
                        'training_metadata': {'seeds_completed': '3/3', 'epochs_each': 200,
                                              'optimizer_steps_each': 7400, 'checkpoint_events_each': 21,
                                              'official_files_each': 63,
                                              'final_checkpoint_exists_each': True, 'numerically_finite_each': True,
                                              'test_access_total': 0, 'val_checkpoint_selection_total': 0},
                        'descriptive_training_losses': {
                            'note': 'per-seed epoch-mean training losses; descriptive only, NOT a performance metric',
                            'by_seed': {str(s): {e: {k: d['numerical_health']['loss_mean_at_checkpoint_epochs'][e][k]
                                                     for k in ('total_loss', 'loss_rec')} for e in ('1', '200')}
                                        for s, d in docs.items()}},
                        'wall_clock_note': 'training speed is not an acceptance gate and not a comparison metric'},
        'firewall': {'test_benchmark_access': 0 if not G.failed else None, 'val_checkpoint_selection_access': 0,
                     'note': 'per-seed relatime atime snapshots: 3027/3027 TEST faces sensitive and unchanged in every '
                             'run; no VAL pass; upstream "test" block = TRAIN visualization, not benchmark TEST'},
        'not_completed': {'synthetic_bank': 'NOT_STARTED', 'downstream_fas_evaluation': 'NOT_STARTED',
                          'whole_benchmark': 'NOT_COMPLETED'},
        'confirmations': {'best_seed_selection': False, 'seed_dropped': False, 'additional_seed_launched': False,
                          'val_checkpoint_selection': False, 'test_used': False, 'bank_generation': False,
                          'downstream_evaluation': False, 'e06b_started': False, 'other_method_started': False,
                          'gpat_started': False, 'run_root_mutated': False},
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
