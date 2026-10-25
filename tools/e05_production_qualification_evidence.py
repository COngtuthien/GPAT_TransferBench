"""E05 production-runner qualification evidence: read-only host collect + deterministic laptop build.

  collect  (GPU host, gpat-m6-e05, `python -I -B - collect --repo <gpu repo> < this file`) -- read-only; prints one JSON
           document: the qualification process result (/tmp), the small files of both qualification roots, the
           scientific-preflight outputs and refusal probes, the GPU checkout state, and a stat-only before/after atime
           comparison of faces_256 + manifests (split_v1 is read here, by this audit process only, to classify the
           snapshot paths; the qualification process never reads it).
  build    (repository) -- gates the collected document and writes outputs/audit/E05_PRODUCTION_RUNNER_QUALIFICATION.json
           (which embeds the collected document so the build re-derives).

No training, selection, VAL, TEST, bank or downstream code is run; nothing on the host is written.
"""
import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import sys

STATUS_OK = 'E05_PRODUCTION_RUNNER_QUALIFIED'
STATUS_BLOCKED = 'E05_PRODUCTION_QUALIFICATION_NEEDS_REVIEW'
WORKDIR = '/home/student20261/workdir'
RUNTIME = f'{WORKDIR}/GPAT_TransferBench_runtime'
RESULT = '/tmp/gpat_e05_prodq/result.json'
ATIME_PRE, ATIME_POST = f'{WORKDIR}/e05_prodq_atime_pre.tsv', f'{WORKDIR}/e05_prodq_atime_post.tsv'
PREFLIGHT_SEEDS = (42, 1337, 2026)
REFUSAL_PROBES = ('seed60901', 'iterations', 'resume', 'dirty_env')
EVIDENCE = {'json': 'outputs/audit/E05_PRODUCTION_RUNNER_QUALIFICATION.json',
            'report': 'outputs/audit/E05_PRODUCTION_RUNNER_QUALIFICATION.md',
            'sha256': 'outputs/audit/E05_PRODUCTION_RUNNER_EVIDENCE_SHA256.txt'}
SMALL = ('run_manifest.json', 'run_summary.json', 'checkpoint_index.json')
LOSS_KEYS = ('L_rec', 'L_recblur', 'L_advrec', 'L_advmix', 'L_pat', 'L_G_total', 'L_D_real', 'L_D_rec', 'L_D_mix',
             'L_D_patch_real', 'L_D_patch_fake', 'L_D_image', 'L_D_patch', 'L_D_total')
G_TERMS = LOSS_KEYS[:5]


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


def utc(ts):
    return datetime.fromtimestamp(ts, timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def epoch_seconds(iso):
    return datetime.strptime(iso, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc).timestamp()


# ============================================================================ collect (GPU host, read-only)
def run_root_files(root):
    root = Path(root)
    out = {'exists': root.is_dir(), 'files': {}}
    if not root.is_dir():
        return out
    for p in sorted(root.rglob('*')):
        if p.is_file():
            out['files'][str(p.relative_to(root))] = {'bytes': p.stat().st_size, 'sha256': sha256_file(p)}
    for name in SMALL:
        out[name] = json.loads((root / name).read_text())
    out['metrics'] = [json.loads(line) for line in (root / 'metrics.jsonl').read_text().splitlines()]
    out['stdout'] = (root / 'stdout.log').read_text(errors='replace')
    out['stderr'] = (root / 'stderr.log').read_text(errors='replace')
    out['resolved_config_sha256'] = sha256_file(root / 'resolved_config.yaml')
    return out


def collect_atime(repo, window):
    import pyarrow.parquet as pq
    t = pq.read_table(Path(repo) / 'manifests/split_v1.parquet', columns=['sample_id', 'dataset', 'split']).to_pylist()
    meta = {r['sample_id']: (r['dataset'], r['split']) for r in t}

    def load(p):
        out = {}
        for line in open(p):
            a, m, path = line.rstrip('\n').split('\t')
            out[path] = (float(a), float(m))
        return out
    pre, post = load(ATIME_PRE), load(ATIME_POST)
    lo, hi = epoch_seconds(window[0]), epoch_seconds(window[1])
    stats, manifests, changed = {}, [], []
    for path, (a, m) in sorted(pre.items()):
        if '/manifests/' in path:
            a2 = post[path][0]
            manifests.append({'file': os.path.basename(path), 'changed': a2 != a, 'mtime_unchanged': post[path][1] == m,
                              'atime_post_utc': utc(a2)})
            continue
        sid = os.path.basename(path)[:-4]
        ds, sp = meta.get(sid, (os.path.basename(os.path.dirname(path)), 'UNMAPPED'))
        s = stats.setdefault(f'{ds}|{sp}', {'files': 0, 'sensitive_whole_window': 0, 'atime_changed': 0,
                                           'mtime_changed': 0, 'missing_post': 0})
        s['files'] += 1
        s['sensitive_whole_window'] += (a <= m) or (lo - a >= 86400)   # relatime: any read in the window updates
        if path not in post:
            s['missing_post'] += 1
            continue
        a2, m2 = post[path]
        s['mtime_changed'] += m2 != m
        if a2 != a:
            s['atime_changed'] += 1
            changed.append({'path': path, 'class': f'{ds}|{sp}', 'inside_window': lo - 60 <= a2 <= hi + 60})
    return {'pre_snapshot_utc': Path(ATIME_PRE[:-4] + '.time').read_text().strip(),
            'post_snapshot_utc': Path(ATIME_POST[:-4] + '.time').read_text().strip(),
            'files_pre': len(pre), 'files_post': len(post), 'window_utc': list(window), 'by_class': stats,
            'manifests': manifests, 'changed_faces': changed, 'pre_sha256': sha256_file(ATIME_PRE),
            'post_sha256': sha256_file(ATIME_POST), 'mount': 'ext4 rw,relatime (/)'}


def collect(repo):
    import subprocess
    result = json.loads(Path(RESULT).read_text())
    git = {k: subprocess.run(['git', '-C', repo, *a], capture_output=True, text=True).stdout.strip()
           for k, a in (('head', ['rev-parse', 'HEAD']), ('branch', ['branch', '--show-current']),
                        ('status_porcelain', ['status', '--porcelain']))}
    preflight = {}
    for s in PREFLIGHT_SEEDS:
        p = Path(f'{WORKDIR}/e05_prodq_preflight_{s}')
        preflight[str(s)] = {'exit': p.with_suffix('.exit').read_text().strip(),
                             'plan': json.loads(p.with_suffix('.json').read_text() or 'null'),
                             'stderr': p.with_suffix('.err').read_text()}
    refusals = {}
    for name in REFUSAL_PROBES:
        p = Path(f'{WORKDIR}/e05_prodq_refusal_{name}')
        refusals[name] = {'exit': p.with_suffix('.exit').read_text().strip(), 'stderr': p.with_suffix('.err').read_text(),
                          'stdout': p.with_suffix('.out').read_text()}
    qparent = Path(RUNTIME, 'qualification', 'e05_production', 'E05')
    access = Path(result['access_log'])
    return {'result': result, 'git': git, 'preflight': preflight, 'refusals': refusals,
            'train_root': run_root_files(result['train_run']['run_dir']),
            'probe_root': run_root_files(result['probe_run']['run_dir']),
            'qualification_parent_listing': sorted(os.listdir(qparent)),
            'scientific_e05_root_exists': Path(RUNTIME, 'runs', 'm6', 'E05').exists(),
            'access_log': {'sha256': sha256_file(access), 'lines': len(access.read_text().splitlines())},
            'atime': collect_atime(repo, (result['utc_start'], result['utc_end'])),
            'collector_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}


# ============================================================================ build (repository)
class Gates:
    def __init__(self):
        self.results = []

    def __call__(self, name, ok, detail=None):
        self.results.append({'gate': name, 'pass': bool(ok), **({'detail': detail} if detail is not None else {})})
        return ok

    @property
    def failed(self):
        return [r['gate'] for r in self.results if not r['pass']]


def build(c, repo_root):
    sys.path.insert(0, str(repo_root))
    from methods.pcgan import production_io as pio
    repo = Path(repo_root)
    G = Gates()
    r, tr, pb = c['result'], c['train_root'], c['probe_root']
    lock = json.loads((repo / pio.LOCK_PATH).read_text())

    # A. authority / code identity
    G('A.result_collected', r['status'] == 'COLLECTED', r.get('error'))
    G('A.gpu_checkout_clean_at_candidate', c['git']['head'] == r['git_commit'] and c['git']['status_porcelain'] == ''
      and c['git']['branch'] == 'm6-baselines', c['git'])
    files = {rel: hashlib.sha256((repo / rel).read_bytes()).hexdigest() for rel in pio.PRODUCTION_FILES}
    G('A.executed_code_equals_repository', r['identities']['production_files_sha256'] == files ==
      r['production_files_sha256_after'], files)
    harness = 'methods/pcgan/production_qualification.py'
    G('A.qualification_seed_not_experiment', r['qualification_seed'] == pio.QUALIFICATION_SEED and
      r['experiment_seed'] is None and pio.QUALIFICATION_SEED not in pio.SCIENTIFIC_SEEDS)

    # B. frozen identities / environment / runtime closure
    ids = r['identities']
    G('B.identities', (ids['config_sha256'], ids['spec_sha256'], ids['environment_lock_sha256'],
                       ids['compatibility_manifest_sha256'], ids['pair_manifest_sha256'], ids['m6d4c_overlay_sha256'],
                       ids['a5_overlay_sha256'], ids['source_commit'], ids['source_tree'], ids['fidelity_class']) ==
      (pio.CONFIG_SHA, pio.SPEC_SHA, pio.LOCK_SHA, pio.COMPAT_MANIFEST_SHA, pio.PAIR_MANIFEST_SHA,
       pio.M6D4C_OVERLAY_SHA, pio.A5_OVERLAY_SHA, pio.SOURCE_COMMIT, pio.SOURCE_TREE, 'CONTROLLED_ADAPTATION'))
    env_diff, driver = pio.environment_differences(lock['identity'], r['environment'], pio.QUALIFICATION_SEED)
    G('B.environment_equals_lock', r['environment_identity_equals_lock'] and env_diff == [] and
      r['gpu_driver'] == driver, env_diff)
    G('B.precision', r['precision'] == {'default_dtype': 'torch.float32', 'matmul_tf32': False, 'cudnn_tf32': False,
                                        'cudnn_benchmark': False, 'cudnn_deterministic': True, 'autocast_cuda': False,
                                        'autocast_cpu': False, 'matmul_precision': 'highest', 'amp': False})
    bc = r['runtime']['build_closure']
    G('B.existing_cuda_binaries_no_rebuild', bc['jit_build_invoked'] is False and
      {k: v['extension']['sha256'] for k, v in bc['extensions'].items()} ==
      {k: v['extension']['sha256'] for k, v in lock['custom_ops'].items()} and
      bc['manifest_sha256'] == pio.COMPAT_MANIFEST_SHA)
    G('B.source_pin_before_after', r['source'] == {'commit': pio.SOURCE_COMMIT, 'tree': pio.SOURCE_TREE} and
      r['source_after'] == pio.SOURCE_COMMIT)
    G('B.models_and_optimizers', {k: v['parameter_count'] for k, v in r['runtime']['models'].items()} ==
      {'Encoder': 929960, 'Generator': 3699233, 'ImageD': 28859521, 'PatchD': 24522145} and
      {g: (m['tensors'], m['parameters'], m['lr'], m['betas'], m['weight_decay']) for g, m in r['optimizers'].items()}
      == {'G': (59, 4629193, 1e-6, [0.9, 0.999], 0), 'D': (80, 53381666, 1e-6, [0.9, 0.999], 0)})

    # C. scientific CLI preflight (no torch, no root) and refusals
    for s, p in c['preflight'].items():
        plan = p['plan'] or {}
        G(f'C.preflight_seed{s}', p['exit'] == '0' and plan.get('experiment_seed') == int(s) and
          plan.get('torch_imported') is False and plan.get('run_dir_exists') is False and
          plan.get('git_commit') == r['git_commit'] and plan.get('images_read') == 0 and
          plan.get('model_executed') is False and plan.get('total_iterations') == 4000 and
          plan.get('checkpoint_iterations') == [1, 4000] and plan.get('resume') == 'NOT_QUALIFIED_FRESH_ONLY', p['exit'])
    G('C.refusals_exit_2', all(v['exit'] == '2' and 'REFUSED' in v['stderr'] for v in c['refusals'].values()),
      {k: v['exit'] for k, v in c['refusals'].items()})
    G('C.no_scientific_e05_root', c['scientific_e05_root_exists'] is False)

    # D. real TRAIN data path
    m = r['pair_manifest']
    G('D.pair_manifest', (m['sha256'], m['rows'], m['splits'], m['columns_read'], m['subject_columns_read'],
                          m['sha256_checked_before_parse']) ==
      (pio.PAIR_MANIFEST_SHA, 8838, ['TRAIN'], list(pio.ALLOWLIST), [], True))
    recs = r['records']
    pred = r['predicted']
    G('D.order_equals_independent_prediction', [x['sampler_index'] for x in recs] == pred['sampler_indices'] and
      len(recs) == pio.QUALIFICATION_ITERATIONS)
    G('D.pairs_equal_frozen_manifest_rows', [{k: x[k] for k in ('pair_id', 'source_spoof_id', 'target_live_id',
                                                                  'dataset')} for x in recs] ==
      [{k: p[k] for k in ('pair_id', 'source_spoof_id', 'target_live_id', 'dataset')} for p in pred['pairs']] and
      all(p['split'] == 'TRAIN' and p['source_spoof_id'] != p['target_live_id'] for p in pred['pairs']))
    G('D.input_contract', all(x['input_shape'] == [1, 3, 256, 256] and x['input_dtype'] == 'torch.float32' and
                              -1.0 <= x['input_range'][0] <= x['input_range'][1] <= 1.0 for x in recs))
    G('D.faces_from_canonical_root', r['access']['face_paths_opened_in_order'] == r['expected_face_paths'] and
      all(p.startswith(c['result']['storage']['faces_256_root'] + '/') for p in r['expected_face_paths']) and
      all(len(x['source_png_sha256']) == len(x['target_png_sha256']) == 64 for x in recs))

    # E. iteration accounting / numerical health
    G('E.global_steps_1_to_3', [x['global_step'] for x in recs] == [1, 2, 3] and
      all(x['benchmark_iteration'] == x['global_step'] for x in recs))
    G('E.one_D_one_G_per_iteration', [x['optimizer_applications'] for x in recs] ==
      [{'D': t, 'G': t, 'hook_D': t, 'hook_G': t, 'total': 2 * t} for t in (1, 2, 3)] and
      r['applications_end'] == {'D': 3, 'G': 3, 'hook_D': 3, 'hook_G': 3, 'total': 6})
    G('E.losses_finite_G_terms_active', all(set(x['train_losses']) == set(LOSS_KEYS) and
                                            all(v == v and abs(v) != float('inf') for v in x['train_losses'].values())
                                            and all(x['train_losses'][k] > 0 for k in G_TERMS) for x in recs))
    G('E.gradients_finite_isolated', all(x['gradients']['D']['finite'] == x['gradients']['D']['non_none'] == 80 and
                                         x['gradients']['G']['finite'] == x['gradients']['G']['non_none'] == 59 and
                                         x['gradients']['D']['inactive_group_non_none'] == 0 and
                                         x['gradients']['G']['inactive_group_non_none'] == 0 for x in recs))
    G('E.adam_state_steps', r['adam_steps_after_each_iteration'] == [{'D': [t, 80], 'G': [t, 59]} for t in (1, 2, 3)])
    iso = r['isolation_iteration_1']
    G('E.parameter_isolation_iteration1', set(iso) == {'before_D', 'after_D', 'after_G'} and
      all(iso['before_D'][k] != iso['after_D'][k] for k in ('ImageD', 'PatchD')) and
      all(iso['before_D'][k] == iso['after_D'][k] for k in ('Encoder', 'Generator')) and
      all(iso['after_D'][k] != iso['after_G'][k] for k in ('Encoder', 'Generator')) and
      all(iso['after_D'][k] == iso['after_G'][k] for k in ('ImageD', 'PatchD')))
    G('E.learning_rates', all(x['learning_rates'] == {'D': 1e-6, 'G': 1e-6} and x['learning_rate'] == 1e-6
                              for x in recs))

    # F. checkpoint writer
    tw, pw = r['train_written'], r['probe_written']
    G('F.periodic_iteration1', [(w['path'], w['global_step'], w['checkpoint_type'], w['selected_for_final'],
                                 w['selection_reason'], w['actual_benchmark_iteration'], w['terminal_semantics_probe'])
                                for w in tw] ==
      [('checkpoints/iteration_0001.pth', 1, 'periodic', False, pio.PERIODIC_REASON, 1, False)] and
      tw[0]['tensors_round_trip_equal'] > 0)
    G('F.train_index_matches_files', [(e['path'], e['sha256'], e['file_size_bytes']) for e in
                                      tr['checkpoint_index.json']['checkpoints']] ==
      [(w['path'], w['sha256'], w['file_size_bytes']) for w in tw] and
      all(tr['files'][w['path']]['sha256'] == w['sha256'] for w in tw))
    G('F.terminal_semantics_probe', (pw['path'], pw['global_step'], pw['checkpoint_type'], pw['selected_for_final'],
                                     pw['selection_reason'], pw['actual_benchmark_iteration'],
                                     pw['terminal_semantics_probe']) ==
      ('checkpoints/iteration_4000.pth', 4000, 'selected', True, 'BASELINE_FINAL_STATE_V1', 3, True) and
      [(e['path'], e['sha256'], e['selected_for_final']) for e in pb['checkpoint_index.json']['checkpoints']] ==
      [(pw['path'], pw['sha256'], True)] and pb['files'][pw['path']]['sha256'] == pw['sha256'])
    G('F.refusals', set(r['refusals']) == {'qualification_cannot_run_4000', 'terminal_without_4000_iterations',
                                           'probe_in_train_case', 'rewrite_periodic', 'periodic_in_probe_case'},
      r['refusals'])
    G('F.no_selected_in_train_root', not any(e['selected_for_final'] for e in tr['checkpoint_index.json']['checkpoints']))

    # G. lifecycle
    tm = [x for x in tr['metrics'] if x['record_type'] == 'epoch']
    ev = [x['event'] for x in tr['metrics'] if x['record_type'] == 'event']
    G('G.train_lifecycle', tr['run_manifest.json']['completion_status'] == 'interrupted' and
      tr['run_summary.json']['completion_status'] == 'interrupted' and
      tr['run_summary.json']['failure_reason'].startswith('QUALIFICATION_STOP_AFTER_3_ITERATIONS') and
      [x['global_step'] for x in tm] == [1, 2, 3] and ev == ['e05_run_start', 'e05_checkpoint_event'] and
      tr['run_manifest.json']['experiment_seed'] is None and tr['run_manifest.json']['qualification_seed'] == 60901
      and tr['run_manifest.json']['scientific_run'] is False and tr['stderr'] == '' and
      [{k: v for k, v in x.items() if k not in ('record_type', 'utc')} for x in tm] == recs)
    G('G.probe_lifecycle', pb['run_summary.json']['completion_status'] == 'interrupted' and
      pio.TERMINAL_PROBE_LABEL in pb['run_manifest.json']['labels'] and
      pb['run_summary.json']['final_or_selected_checkpoint_path'] is None and
      [x for x in pb['metrics'] if x['record_type'] == 'epoch'] == [] and
      [x['event'] for x in pb['metrics'] if x['record_type'] == 'event'] == ['e05_checkpoint_event'])
    G('G.roots_outside_runs', all('/qualification/e05_production/E05/' in x[k]['run_dir'] and '/runs/' not in
                                  x[k]['run_dir'] for x in (r,) for k in ('train_run', 'probe_run')))

    # H. firewall
    fw, acc = r['firewall'], r['access']
    counts = acc['counts_by_category']
    G('H.no_denials', fw['denied'] == [] and acc['denied_events'] == 0 and acc['processes_logged'] == 1)
    G('H.only_pair_manifest_and_authorized_faces', counts['AUTHORIZED_TRAIN_FACE'] == 6 and
      counts['PAIR_MANIFEST'] >= 1 and all(v == 0 for k, v in counts.items()
                                           if k not in ('AUTHORIZED_TRAIN_FACE', 'PAIR_MANIFEST')), counts)
    G('H.val_test_zero_in_process', (acc['VAL_image_accesses'], acc['TEST_image_accesses'],
                                     acc['VAL_metadata_accesses'], acc['TEST_metadata_accesses']) == (0, 0, 0, 0))
    a = c['atime']
    by = a['by_class']
    G('H.atime_snapshot_complete', a['files_pre'] == a['files_post'] and 'UNMAPPED' not in ''.join(by))
    G('H.atime_test_val_unchanged_sensitive', all(v['atime_changed'] == 0 and v['sensitive_whole_window'] == v['files']
                                                  for k, v in by.items() if k.split('|')[1] in ('TEST', 'VAL')),
      {k: v for k, v in by.items() if k.split('|')[1] in ('TEST', 'VAL')})
    G('H.atime_only_authorized_faces_changed', {x['path'] for x in a['changed_faces']} <=
      set(r['expected_face_paths']), a['changed_faces'])
    G('H.atime_no_other_manifest_or_mtime', all(not x['changed'] for x in a['manifests']
                                                if x['file'] != 'pairs_train_v1.parquet') and
      all(x['mtime_unchanged'] for x in a['manifests']) and all(v['mtime_changed'] == 0 for v in by.values()))

    doc = {'status': STATUS_BLOCKED if G.failed else STATUS_OK, 'failed_gates': G.failed, 'gates': G.results,
           'method': {'id': 'E05', 'reporting_label': 'PCGAN (controlled architecture resolution)',
                      'track': 'A_FAIR_COMMON_IDENTITY_FREE', 'fidelity': 'CONTROLLED_ADAPTATION',
                      'fidelity_authority': 'A2-07 architecture resolution; A5 blur operator; M6D4c runner resolution',
                      'forbidden_wording': ['native PCGAN', 'faithful PCGAN', 'official reproduction'],
                      'executable_architecture_basis': 'taesungp/swapping-autoencoder-pytorch@' + pio.SOURCE_COMMIT},
           'candidate_commit': r['git_commit'], 'harness': harness,
           'qualification_seed': pio.QUALIFICATION_SEED, 'qualification_iterations': pio.QUALIFICATION_ITERATIONS,
           'execution_choices': pio.resolved_overlays(),
           'blockers_closed': {
               'A_real_train_pair_path_and_firewall': not [g for g in G.failed if g[0] in 'DH'],
               'B_production_loop_terminal_writer_lifecycle': not [g for g in G.failed if g[0] in 'EFG'],
               'C_fail_closed_scientific_cli': not [g for g in G.failed if g[0] == 'C'],
               'D_real_train_production_qualification': not G.failed},
           'confirmations': {'scientific_seed_launched': False, 'scientific_e05_root_created': c['scientific_e05_root_exists'],
                             'resume_implemented': False, 'val_pass': False, 'test_access': False,
                             'best_checkpoint_or_seed': False, 'bank_generation': False, 'downstream_evaluation': False,
                             'other_method_started': False, 'gpat_started': False},
           'collected': c}
    return doc


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('collect')
    a.add_argument('--repo', required=True)
    b = sub.add_parser('build')
    b.add_argument('--collected', required=True)
    b.add_argument('--repo-root', default=str(Path(globals().get('__file__', 'tools/x')).resolve().parents[1]))
    args = ap.parse_args(argv)
    if args.cmd == 'collect':
        json.dump(collect(args.repo), sys.stdout, sort_keys=True)
        return 0
    doc = build(json.loads(Path(args.collected).read_text()), args.repo_root)
    (Path(args.repo_root) / EVIDENCE['json']).write_text(json.dumps(doc, indent=2, sort_keys=True) + '\n')
    print(doc['status'], doc['failed_gates'])
    return 0 if doc['status'] == STATUS_OK else 1


if __name__ == '__main__':
    sys.exit(main())
