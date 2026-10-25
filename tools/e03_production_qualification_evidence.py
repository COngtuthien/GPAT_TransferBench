"""E03 STDN production qualification evidence: collect (laptop, read-only) + build (stdlib, deterministic).

  collect --results <round-2 dir> --round1 <round-1 dir> --host <host_recheck.json> --out <collected.json>
      Compacts the GPU qualification results of the candidate (methods/stdn/production_qualification.py cases val_a,
      val_b, no_val, terminal_probe, failure_probe, predict), the CLI preflight/refusal probes, the stat-only atime
      snapshots (classified with manifests/split_v1.parquet) and the read-only host recheck. Parser logs are mapped
      to sample ids with the frozen pair manifests so atime changes can be compared with the faces actually parsed.
  build --collected <collected.json>
      Gates everything and writes outputs/audit/E03_PRODUCTION_RUNNER_QUALIFICATION.json (embedding the collection).
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from methods.stdn import production_io as pio  # noqa: E402

EVIDENCE = 'outputs/audit/E03_PRODUCTION_RUNNER_QUALIFICATION.json'
CANDIDATE = 'c07ecdd72c227c7b8fcace15e12dc23b28e9c8c8'
ROUND1_CANDIDATE = '68a42c2bb1a177a69da665928092fc4c78be7307'
CASES = ('val_a', 'val_b', 'no_val', 'terminal_probe', 'failure_probe')
PLANS = {'val_a': [[201, 'checkpoint', 51], [8, None, 0]], 'val_b': [[201, 'checkpoint', 51], [8, None, 0]],
         'no_val': [[201, 'checkpoint', 0], [8, None, 0]], 'terminal_probe': [[2, 'probe50', 0]],
         'failure_probe': [[2, None, 0]]}
REFUSALS = ('seed60903', 'resume', 'epochs', 'valsteps', 'test', 'dirty_env')


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(p):
    return sha256_bytes(Path(p).read_bytes())


def canon(obj):
    return sha256_bytes(json.dumps(obj, sort_keys=True).encode())


# ============================================================================ collect
def compact(d, rows):
    recs = d['metrics_records']
    train = [r for r in recs if r.get('record_kind') == 'TRAIN_STEP']
    val = [r for r in recs if r.get('record_kind') == 'VAL_STEP']
    events = [r for r in recs if r.get('record_type') == 'event']
    out = {k: d[k] for k in ('case', 'run_id', 'run_dir', 'qualification_seed', 'experiment_seed', 'plan', 'structure',
                             'driver', 'seeding', 'live_environment', 'live_packages_sha256', 'pack',
                             'session_counts', 'face_reads', 'access', 'steps', 'scientific_execution')}
    out['identities'] = d['identities']
    out['final'] = ({k: v for k, v in d['final'].items() if k != 'per_variable'} if 'final' in d else None)
    out['points'] = {}
    for name, p in d['points'].items():
        out['points'][name] = {k: v for k, v in p.items() if k != 'per_variable'}
    if 'before_val' in d['points'] and 'after_val' in d['points']:
        a, b = d['points']['before_val']['per_variable'], d['points']['after_val']['per_variable']
        out['val_variable_diff'] = {'compared': len(a), 'changed': sorted(n for n in a if a[n] != b.get(n)),
                                    'missing': sorted(set(a) ^ set(b))}
    out['checkpoints'] = d.get('checkpoints', [])
    out['checkpoint_index'] = d['checkpoint_index']
    out['run_manifest'] = {k: d['run_manifest'].get(k) for k in (
        'method_id', 'run_id', 'git_commit', 'git_dirty', 'config_sha256', 'completion_status', 'runner_mode',
        'scientific_run', 'qualification_seed', 'qualification_case', 'fidelity', 'historical_fidelity_field',
        'labels', 'resume', 'test_split_accessed', 'val_split_accessed', 'test_code_path_present', 'experiment_seed')}
    out['run_summary'] = {k: d['run_summary'].get(k) for k in ('completion_status', 'failure_reason', 'records_written')}
    out['root_listing'] = d['root_listing']
    out['root_listing_digest'] = sha256_bytes(''.join('%s %d %s\n' % tuple(x) for x in d['root_listing']).encode())
    out['events'] = sorted({e['event'] for e in events})
    out['train_records'] = [{'e': r['epoch'], 's': r['train_step_in_epoch'], 'gs': r['global_step'],
                             'gs_before': r['shared_global_step_before'], 'ops': r['optimizer_ops_applied'],
                             'apps': r['applications'], 'lr': r['learning_rates'], 'loss': r['train_losses'],
                             'rows': r['pair_rows'], 'ids': r['pair_ids'], 'flips': r['flips_live_spoof'],
                             'bright': r['brightness_delta_live_spoof'], 'fig': r['figure']} for r in train]
    out['val_records'] = [{'e': r['epoch'], 's': r['val_step_in_epoch'], 'gs': r['global_step'], 'loss': r['val_losses'],
                           'rows': r['pair_rows'], 'ids': r['pair_ids'], 'flips': r['flips_live_spoof'],
                           'fig': r['figure']} for r in val]
    out['parsed_samples'] = {s: sorted({x for i, _, _ in d['parser_log'][s] for x in rows[s][i][1:3]})
                             for s in ('TRAIN', 'VAL')}
    out['parser_log_sha256'] = {s: canon(d['parser_log'][s]) for s in ('TRAIN', 'VAL')}
    out['parser_log'] = {s: d['parser_log'][s][:430] for s in ('TRAIN', 'VAL')}
    if 'failure' in d:
        out['failure'] = d['failure']
    return out


def atime_classes(pre, post, split, parsed):
    def load(f):
        out = {}
        for line in Path(f).read_text().splitlines():
            a, m, p = line.split('\t')
            out[p] = (float(a), float(m))
        return out
    from datetime import datetime, timezone
    A, B = load(pre), load(post)
    t_pre = datetime.strptime(Path(str(pre)[:-4] + '.time').read_text().strip(),
                              '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc).timestamp()
    by, manifests = {}, []
    for path, (a, m) in sorted(A.items()):
        if '/manifests/' in path:
            manifests.append({'file': Path(path).name, 'atime_changed': B[path][0] != a, 'mtime_unchanged': B[path][1] == m})
            continue
        sid = Path(path).stem
        meta = split.get(sid)
        key = '%s|%s' % (meta['dataset'], meta['split']) if meta else 'UNMAPPED'
        c = by.setdefault(key, {'files': 0, 'sensitive': 0, 'changed': 0, 'changed_not_parsed': 0, 'missing': 0,
                                'mtime_changed': 0})
        c['files'] += 1
        c['sensitive'] += (a <= m) or (t_pre - a >= 86400)
        if path not in B:
            c['missing'] += 1
            continue
        c['mtime_changed'] += B[path][1] != m
        if B[path][0] != a:
            c['changed'] += 1
            c['changed_not_parsed'] += sid not in parsed
    return {'pre_sha256': sha256_file(pre), 'post_sha256': sha256_file(post), 'files': len(A),
            'same_paths': sorted(A) == sorted(B), 'by_class': by, 'manifests': manifests,
            'pre_utc': Path(str(pre)[:-4] + '.time').read_text().strip(),
            'post_utc': Path(str(post)[:-4] + '.time').read_text().strip()}


def collect(results, round1, host):
    import pyarrow.parquet as pq
    results, round1 = Path(results), Path(round1)
    rows = {s: [[r['pair_id'], r['target_live_id'], r['source_spoof_id'], r['dataset']] for r in
                pq.read_table(ROOT / v[0], columns=['pair_id', 'target_live_id', 'source_spoof_id', 'dataset'])
                .to_pylist()] for s, v in pio.PAIR_MANIFESTS.items()}
    split = {r['sample_id']: r for r in pq.read_table(ROOT / pio.SPLIT_MANIFEST[0],
                                                      columns=['sample_id', 'dataset', 'split']).to_pylist()}
    cases = {}
    parsed = set()
    for c in CASES:
        d = json.loads((results / (c + '.json')).read_text())
        cases[c] = compact(d, rows)
        cases[c]['exit'] = (results / (c + '.exit')).read_text().strip()
        cases[c]['result_sha256'] = sha256_file(results / (c + '.json'))
        parsed |= set(cases[c]['parsed_samples']['TRAIN']) | set(cases[c]['parsed_samples']['VAL'])
    round1_cases = {}
    for c in ('val_a', 'val_b', 'no_val', 'terminal_probe'):
        d = json.loads((round1 / (c + '.json')).read_text())
        round1_cases[c] = {'run_id': d['run_id'], 'run_dir': d['run_dir'], 'result_sha256': sha256_file(round1 / (c + '.json')),
                           'final_fingerprint_sha256': d['final']['fingerprint_sha256'],
                           'in_process_val_unchanged': (d['points']['before_val']['fingerprint_sha256'] ==
                                                        d['points']['after_val']['fingerprint_sha256']
                                                        if 'after_val' in d['points'] and 'before_val' in d['points']
                                                        else None)}
        for s in ('TRAIN', 'VAL'):
            parsed |= {x for i, _, _ in d['parser_log'][s] for x in rows[s][i][1:3]}
    pred = json.loads((results / 'predict.json').read_text())
    cli = {'preflight': {}, 'refusals': {}}
    for s in pio.SCIENTIFIC_SEEDS:
        p = json.loads((results / ('preflight_%d.json' % s)).read_text())
        cli['preflight'][str(s)] = {'exit': (results / ('preflight_%d.exit' % s)).read_text().strip(),
                                    'stderr': (results / ('preflight_%d.err' % s)).read_text()[-300:],
                                    'plan': {k: p[k] for k in ('experiment_seed', 'git_commit', 'git_branch', 'git_dirty',
                                                               'run_dir', 'run_dir_exists', 'tensorflow_imported',
                                                               'graph_executed', 'images_decoded',
                                                               'landmark_arrays_loaded', 'pack_manifest_sha256',
                                                               'fidelity', 'rng', 'resume', 'max_epoch',
                                                               'steps_per_epoch', 'val_steps', 'final_global_step',
                                                               'checkpoint_rule')},
                                    'json_sha256': sha256_file(results / ('preflight_%d.json' % s))}
    for r in REFUSALS:
        cli['refusals'][r] = {'exit': (results / ('refusal_%s.exit' % r)).read_text().strip(),
                              'stderr': (results / ('refusal_%s.err' % r)).read_text().strip()[-240:],
                              'stdout_bytes': len((results / ('refusal_%s.out' % r)).read_bytes())}
    lock = json.loads((ROOT / pio.LOCK[0]).read_text())
    atime = atime_classes(results / 'e03_prodq_atime_pre.tsv', results / 'e03_prodq_atime_post2.tsv', split, parsed)
    atime['post_round1_sha256'] = sha256_file(results / 'e03_prodq_atime_post.tsv')
    return {'candidate_commit': CANDIDATE, 'round1_candidate_commit': ROUND1_CANDIDATE, 'cases': cases,
            'round1': round1_cases, 'prediction': pred['prediction'], 'prediction_sha256': sha256_file(results / 'predict.json'),
            'cli': cli, 'atime': atime, 'host': json.loads(Path(host).read_text()),
            'lock': {'python': lock['python'], 'tensorflow': lock['tensorflow'], 'environment_path': lock['environment_path'],
                     'cuda_cudnn_native': lock['cuda_cudnn_native'], 'precision_env': lock['precision_env'],
                     'session_config': lock['session_config'], 'gpu_driver': lock['gpu_driver'],
                     'packages_sha256': canon(lock['packages'])},
            'parsed_sample_count': len(parsed)}


# ============================================================================ build
class Gates:
    def __init__(self):
        self.results = []

    def __call__(self, name, ok, detail=None):
        self.results.append({'gate': name, 'pass': bool(ok), **({'detail': detail} if detail is not None else {})})
        return ok

    @property
    def failed(self):
        return [r['gate'] for r in self.results if not r['pass']]


def f32_power(n):
    """TF Adam beta1_power after n applications: float32 0.9, multiplied by float32 0.9 n times (stdlib emulation)."""
    import struct
    f = lambda x: struct.unpack('<f', struct.pack('<f', x))[0]  # noqa: E731
    b, x = f(0.9), f(0.9)
    for _ in range(n):
        x = f(x * b)
    return x


def rng_state_sha(seed, draws):
    import numpy as np
    rng = np.random.RandomState(seed)
    rng.rand(draws) if draws else None
    name, keys, pos, has_gauss, cached = rng.get_state()
    return sha256_bytes(keys.tobytes() + repr((name, pos, has_gauss, cached)).encode())


def expected_other_runs(case):
    plan = PLANS[case]
    n_train = sum(p[0] for p in plan)
    n_val = sum(p[2] for p in plan)
    e0 = plan[0][0] if case != 'failure_probe' else 2
    ckpts = sum(1 for p in plan if p[1])
    points = 0 if case == 'failure_probe' else (1 if plan[0][1] else 0) + 2
    if case == 'failure_probe':
        return 2 + 1 + 2 * 2                         # 2 counters, 1 start, on_step at steps 0 and 1 (then injected)
    return n_train + n_val + 1 + 2 * min(4, e0) + 4 * ckpts + 2 * points


def build(c, repo_root=ROOT):
    G = Gates()
    cs, host, pred = c['cases'], c['host'], c['prediction']
    a, b, n, tp, fp_ = (cs[k] for k in CASES)
    ids0 = a['identities']

    # A. identity / authority
    G('A.candidate_on_gpu', host['git_head'] == c['candidate_commit'] and host['git_branch'] == 'm6-baselines' and
      host['git_status'] == '' and host['hostname'] == 'nd-System-Product-Name')
    G('A.all_cases_same_commit_identity', all(x['run_manifest']['git_commit'] == c['candidate_commit'] and
                                              x['run_manifest']['git_dirty'] is False and x['identities'] == ids0
                                              for x in cs.values()) and len({x['run_id'] for x in cs.values()}) == 1)
    rid = sha256_bytes(('E03|%d|%s|%s' % (pio.QUALIFICATION_SEED, pio.CONFIG_SHA, c['candidate_commit'])).encode())[:16]
    G('A.qualification_run_id', a['run_id'] == rid, rid)
    G('A.frozen_identities', ids0['config_sha256'] == pio.CONFIG_SHA and ids0['spec_sha256'] == pio.SPEC_SHA and
      ids0['ruling_addendum_sha256'] == pio.RULING_ADDENDUM[1] and ids0['ruling_yaml_sha256'] == pio.RULING_YAML[1] and
      ids0['runtime_addendum_sha256'] == pio.RUNTIME_ADDENDUM[1] and ids0['environment_lock_sha256'] == pio.LOCK[1] and
      ids0['source_commit'] == pio.SOURCE_PIN and ids0['pack_manifest_sha256'] == pio.PACK_MANIFEST_SHA256 and
      ids0['pair_manifest_sha256'] == {s: v[1] for s, v in pio.PAIR_MANIFESTS.items()} and ids0['fidelity'] == pio.FIDELITY)
    G('A.qualification_not_scientific', all(x['qualification_seed'] == 60903 and x['experiment_seed'] is None and
                                            x['scientific_execution'] is False and 'runs' not in x['run_dir'].split('/')
                                            and '/qualification/e03_production/E03/' in x['run_dir']
                                            for x in cs.values()) and 'E03' not in host['runs_m6'])

    # B. environment (driver-version-only)
    lock, live = c['lock'], a['live_environment']
    G('B.environment_equals_lock_except_driver', all(live[k] == lock[k] for k in (
        'python', 'tensorflow', 'environment_path', 'cuda_cudnn_native', 'precision_env', 'session_config')) and
      a['live_packages_sha256'] == lock['packages_sha256'] and all(x['live_packages_sha256'] == lock['packages_sha256']
                                                                    for x in cs.values()))
    G('B.driver_ruling', a['driver'] == {'lock_driver_version': '595.84', 'observed_driver_version': '580.178.04',
                                         'driver_policy': pio.DRIVER_POLICY, 'gpu_model': pio.GPU_MODEL,
                                         'gpu_uuid': pio.GPU_UUID} and
      live['gpu_driver'].split(', ')[0::3] == lock['gpu_driver'].split(', ')[0::3] and
      live['gpu_driver'].split(', ')[1] == lock['gpu_driver'].split(', ')[1] and live['gpu_uuid'] == pio.GPU_UUID)
    G('B.python_executable', live['executable'] in (pio.PYTHON_EXECUTABLE, pio.PYTHON_EXECUTABLE + '3.8'))

    # C. feed pack
    G('C.pack_pinned_and_unchanged', a['pack']['manifest_sha256'] == pio.PACK_MANIFEST_SHA256 ==
      host['pack']['pack_manifest.json'])
    G('C.pack_counts', a['pack']['counts']['pairs'] == {'TRAIN': 8838, 'VAL': 1905} and
      a['pack']['counts']['unique_samples'] == 15321 and a['pack']['counts']['samples_by_split'] ==
      {'TRAIN': 12668, 'VAL': 2653})

    # D. unchanged official graph
    s = a['structure']
    G('D.graph_structure', s['trainable_equals_m6d2b_oracle'] and (s['global_variables'], s['trainable_variables'],
      s['trainable_G'], s['trainable_D'], s['adam_ops'], s['bn_moving'], s['weight_ema'], s['loss_ema']) ==
      (747, 170, 98, 72, 170, 60, 170, 2) and s['variables_created_by_feed'] == s['variables_created_by_val_graph'] == 0)
    G('D.val_graph_nonmutating', s['val_step_returns_no_train_op'] and s['val_reachable_state_mutating_ops'] == [] and
      s['val_fused_bn_is_training_true'] == 0 and s['val_fused_bn_is_training_false'] == 48 and
      not s['val_reads_train_iterator'])
    G('D.same_structure_all_cases', all(x['structure'] == s for x in cs.values()))

    # E. official schedule / shared global step / LR
    def sched_ok(x):
        recs, gs = x['train_records'], 0
        for r in recs:
            gd = r['s'] % 2 == 0
            if r['ops'] != (['g_op', 'd_op'] if gd else ['g_op']) or r['gs_before'] != gs or r['gs'] - gs != (2 if gd else 1):
                return False
            gs = r['gs']
        return True
    G('E.one_run_per_step_schedule', all(sched_ok(x) for x in (a, b, n)) and
      a['session_counts']['train_gd'] == 105 and a['session_counts']['train_g'] == 104 and
      a['session_counts']['val'] == 51 and 'val' not in n['session_counts'])
    G('E.final_counts', all(x['final']['global_step'] == 314 and x['final']['applications'] == {'G': 209, 'D': 105}
                            for x in (a, b, n)))
    G('E.beta1_power_counts', all(abs(x['final']['beta1_power']['beta1_power:0'] - f32_power(209)) == 0 and
                                  abs(x['final']['beta1_power']['beta1_power_1:0'] - f32_power(105)) == 0
                                  for x in (a, b, n)), [f32_power(209), f32_power(105)])
    st = a['steps']
    G('E.step_state_classes', [x['gd'] for x in st] == [True, False, True, False] and
      'trainable_D' not in st[1]['changed_by_class_vs_previous_step'] and
      'trainable_D' not in st[3]['changed_by_class_vs_previous_step'] and
      st[2]['changed_by_class_vs_previous_step'].get('trainable_D') == 72 and
      all(x['changed_by_class_vs_previous_step'].get('trainable_G') == 98 for x in st[1:]))
    lr32 = 5.999999848427251e-05
    G('E.learning_rate_shared_step', all(r['lr'] == {'G': lr32, 'D': lr32} for x in (a, b, n) for r in x['train_records'])
      and pio.expected_learning_rate(314) == pio.LEARNING_RATE)

    # F. data: independent prediction, mapping, determinism of decisions
    rows_a = [x for r in a['train_records'] for x in r['rows']]
    flips_a = [x for r in a['train_records'] for x in r['flips']]
    vrows = [x for r in a['val_records'] for x in r['rows']]
    vflips = [x for r in a['val_records'] for x in r['flips']]
    G('F.train_order_predicted', rows_a == pred['TRAIN']['order'][:len(rows_a)] and len(rows_a) == 418)
    G('F.train_flips_predicted', flips_a == pred['TRAIN']['flips'][:len(flips_a)])
    G('F.val_order_and_flips_predicted', vrows == pred['VAL']['order'][:len(vrows)] and len(vrows) == 102 and
      vflips == pred['VAL']['flips'][:len(vflips)])
    G('F.reshuffle_each_pass', all(pred[x]['pass1_is_permutation'] and pred[x]['pass2_is_permutation'] and
                                   pred[x]['reshuffled_each_pass'] for x in ('TRAIN', 'VAL')))
    G('F.split_seeds', pred['TRAIN']['parser_seed'] == pred['TRAIN']['shuffle_seed'] == 60903 and
      pred['VAL']['parser_seed'] == pred['VAL']['shuffle_seed'] == pio.val_parser_seed(60903) == 2050698081)
    key = lambda x: [(r['rows'], r['flips'], r['bright']) for r in x['train_records']]  # noqa: E731
    G('F.data_decisions_identical_across_processes', key(a) == key(b) == key(n))
    G('F.train_brightness_val_none', all(any(abs(v) > 0 for p in r['bright'] for v in p) for r in a['train_records']) and
      all(abs(v) <= 0.25 + 1e-6 for r in a['train_records'] for p in r['bright'] for v in p))
    G('F.parser_rng_is_pure_function_of_parse_count', all(
        x['final']['train_parser']['rng_state_sha256'] == rng_state_sha(60903, x['final']['train_parser']['draws']) and
        x['final']['train_parser']['draws'] == 2 * x['final']['train_parser']['calls'] for x in (a, b, n, tp)))

    # G. VAL non-influence
    pb, pa = a['points']['before_val'], a['points']['after_val']
    G('G.in_process_all_variables_byte_identical', a['val_variable_diff'] == {'compared': 747, 'changed': [],
                                                                               'missing': []} and
      pb['fingerprint_sha256'] == pa['fingerprint_sha256'] and b['val_variable_diff']['changed'] == [])
    G('G.in_process_counters_unchanged', pb['counters'] == pa['counters'] and b['points']['before_val']['counters'] ==
      b['points']['after_val']['counters'])
    G('G.train_parser_untouched_by_val', pb['train_parser'] == pa['train_parser'] and
      b['points']['before_val']['train_parser'] == b['points']['after_val']['train_parser'])
    G('G.val_parser_advanced_only', pa['val_parser']['calls'] > pb['val_parser']['calls'] == 0 and
      pa['val_parser']['rng_state_sha256'] == rng_state_sha(pio.val_parser_seed(60903), pa['val_parser']['draws']))
    G('G.val_never_changes_global_step', all(r['gs'] == pb['counters']['global_step'] for r in a['val_records']))
    G('G.subsequent_train_decisions_identical', key(a)[201:] == key(n)[201:] == key(b)[201:] and len(key(a)) == 209)
    div = {}
    for name, other in (('val_a_vs_val_b', b), ('val_a_vs_no_val', n)):
        first = next((i for i, (x, y) in enumerate(zip(a['train_records'], other['train_records'])) if x['loss'] != y['loss']),
                     None)
        rel = max(abs(x['loss'][k] - y['loss'][k]) / abs(x['loss'][k]) for x, y in zip(a['train_records'], other['train_records'])
                  for k in ('g_loss', 'd_loss', 'a_loss'))
        div[name] = {'first_differing_train_step': first, 'max_relative_loss_difference': rel,
                     'before_val_fingerprint_equal': a['points']['before_val']['fingerprint_sha256'] ==
                     other['points']['before_val']['fingerprint_sha256']}
    G('G.cross_process_numeric_difference_precedes_val', div['val_a_vs_val_b']['first_differing_train_step'] ==
      div['val_a_vs_no_val']['first_differing_train_step'] == 1 and not div['val_a_vs_val_b']['before_val_fingerprint_equal']
      and a['points']['init'] == b['points']['init'] == n['points']['init'], div)

    # H. checkpoints
    ck = a['checkpoints'][0]
    listing = {x[0]: (x[1], x[2]) for x in a['root_listing']}
    G('H.ckpt1_official_before_val', ck['epoch'] == 1 and ck['checkpoint_type'] == 'official' and
      ck['selected_for_final'] is False and ck['global_step'] == 302 and ck['round_trip_equal'] == 747 and
      ck['all_finite'] and ck['state_unchanged_by_save'] and
      a['points']['before_checkpoint']['fingerprint_sha256'] == pb['fingerprint_sha256'])
    G('H.ckpt1_files_indexed', sorted(f['path'] for f in a['checkpoint_index']['checkpoints']) ==
      sorted('checkpoints/ckpt-1' + x for x in pio.CHECKPOINT_SUFFIXES) and all(
          listing[f['path']] == (f['file_size_bytes'], f['sha256']) for f in a['checkpoint_index']['checkpoints']))
    pc = tp['checkpoints'][0]
    G('H.ckpt50_semantics_probe', pc['epoch'] == 50 and pc['checkpoint_type'] == 'selected' and pc['selected_for_final']
      and pc['terminal_semantics_probe'] and pc['global_step'] == 3 and pio.TERMINAL_PROBE_LABEL in pc['labels'] and
      all(f['selection_reason'] == pio.CHECKPOINT_RULE and listing_ok for f in tp['checkpoint_index']['checkpoints']
          for listing_ok in [{x[0]: (x[1], x[2]) for x in tp['root_listing']}.get(f['path']) ==
                             (f['file_size_bytes'], f['sha256'])]))
    G('H.no_resume_sidecar', all(x['checkpoint_index']['resume_sidecars'] is None and
                                 not any('runner_state' in p[0] or p[0].endswith('.partial') for p in x['root_listing'])
                                 for x in cs.values()))

    # I. figures (R3)
    figs = sorted(r['fig'] for r in a['train_records'] + a['val_records'] if r['fig'])
    G('I.figure_cadence', figs == sorted(['diagnostics/figures/Epoch-1-1.png', 'diagnostics/figures/Epoch-1-201.png',
                                          'diagnostics/figures/Epoch-2-1.png', 'diagnostics/figures/Epoch-1-Val-1.png',
                                          'diagnostics/figures/Epoch-1-Val-51.png']) and all(f in listing for f in figs))
    G('I.figures_add_no_session_run', all(x['session_counts']['other'] == expected_other_runs(k) for k, x in cs.items()),
      {k: [x['session_counts']['other'], expected_other_runs(k)] for k, x in cs.items()})

    # J. lifecycle
    G('J.qualification_lifecycle', all(x['run_summary']['completion_status'] == 'interrupted' and x['exit'] == '0'
                                       for k, x in cs.items() if k != 'failure_probe'))
    f = fp_['failure']
    G('J.failure_cleanup_fresh_only', f['completion_status'] == 'failed' and 'INJECTED' in f['failure_reason'] and
      f['checkpoint_files'] == [] and f['partial_files'] == [] and 'fresh only' in (f['second_open_refused'] or ''))
    G('J.roots_unchanged_since_case', all(host['qualification_roots'][Path(x['run_dir']).name] ==
                                          [x['root_listing_digest'], len(x['root_listing'])] for x in cs.values()))

    # K. firewall
    at = c['atime']
    test = {k: v for k, v in at['by_class'].items() if k.endswith('|TEST')}
    G('K.no_denied_access', all(x['access']['denied'] == [] and x['access']['runtime_writes_outside_root'] == []
                                for x in cs.values()))
    G('K.test_zero', len(test) == 3 and all(v['changed'] == 0 and v['sensitive'] == v['files'] for v in test.values())
      and sum(v['files'] for v in test.values()) == 3027)
    G('K.train_val_changes_only_parsed', all(v['changed_not_parsed'] == 0 for k, v in at['by_class'].items()) and
      'UNMAPPED' not in at['by_class'] and at['same_paths'] and all(m['mtime_unchanged'] for m in at['manifests']))
    G('K.face_reads_by_split', a['face_reads']['VAL'] > 0 and n['face_reads']['VAL'] == 0 and
      all(x['access']['faces_by_split']['TRAIN'] + x['access']['faces_by_split']['VAL'] == x['access']['face_opens']
          for x in cs.values()))

    # L. CLI
    G('L.preflight_static', all(v['exit'] == '0' and v['plan']['tensorflow_imported'] is False and
                                v['plan']['run_dir_exists'] is False and v['plan']['images_decoded'] == 0 and
                                v['plan']['landmark_arrays_loaded'] == 0 and v['plan']['git_commit'] == c['candidate_commit']
                                and v['plan']['experiment_seed'] == int(s_) and v['plan']['rng']['VAL']['parser_seed'] ==
                                pio.val_parser_seed(int(s_)) for s_, v in c['cli']['preflight'].items()))
    G('L.refusals', all(v['exit'] == '2' and v['stdout_bytes'] == 0 and 'REFUSED' in v['stderr']
                        for v in c['cli']['refusals'].values()))

    # M. production code identity (candidate == this checkout)
    files = pio.production_file_hashes(repo_root)
    G('M.production_files_equal_qualified', files == ids0['production_files_sha256'], files)

    failed = G.failed
    status = 'E03_PRODUCTION_RUNNER_QUALIFIED' if not failed else 'E03_PRODUCTION_QUALIFICATION_NEEDS_REVIEW'
    doc = {
        'milestone': 'E03-STDN-PRODUCTION-QUALIFICATION', 'status': status, 'failed_gates': failed, 'gates': G.results,
        'method': {'id': 'E03', 'name': 'STDN', 'track': pio.TRACK, 'fidelity': pio.FIDELITY,
                   'fidelity_layers': pio.FIDELITY_LAYERS, 'historical_fidelity_field': 'FAITHFUL_OFFICIAL'},
        'candidate_commit': c['candidate_commit'], 'round1_candidate_commit': c['round1_candidate_commit'],
        'qualification_seed': 60903, 'qualification_run_id': rid,
        'owner_policy': {'rulings': {'V': 'SEPARATE_DETERMINISTIC_SPLIT_RNG_STREAMS', 'R1': 'pairs_train_v1',
                                     'R2': 'val_pairs_v1', 'R3': 'RETAIN_OFFICIAL_DIAGNOSTIC_FIGURES'},
                         'decisions': {'D1': 'minimal hash-verified E03 feed pack (no TEST)',
                                       'D2': '1 + (u32 % 2147483647)', 'D3': 'reshuffle_each_iteration=True explicit'},
                         'driver': {'lock': '595.84', 'observed': '580.178.04', 'policy': pio.DRIVER_POLICY}},
        'determinism': {'data_decisions': 'bitwise identical across processes and equal to the independent predictor',
                        'numeric_training': 'NOT bitwise reproducible across processes: identical init and step-0 '
                                            'losses, first float difference after the first optimizer update (step 1), '
                                            'chaotic growth; the unchanged official graph contains atomic-accumulation '
                                            'GPU kernels', 'atomic_accumulation_op_types':
                                                s.get('atomic_accumulation_op_types'), 'divergence': div,
                        'val_non_influence_basis': 'in-process byte identity of all 747 TF variables (sha256 of each '
                                                   'variable value bytes), shared global_step/lr, TRAIN parser call/draw '
                                                   'counts and RandomState state across the real VAL block; plus '
                                                   'cross-process identity of all subsequent TRAIN data decisions'},
        'collected': c,
    }
    return doc


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('collect')
    a.add_argument('--results', required=True)
    a.add_argument('--round1', required=True)
    a.add_argument('--host', required=True)
    a.add_argument('--out', required=True)
    b = sub.add_parser('build')
    b.add_argument('--collected', required=True)
    args = ap.parse_args(argv)
    if args.cmd == 'collect':
        Path(args.out).write_text(json.dumps(collect(args.results, args.round1, args.host), sort_keys=True))
        return 0
    doc = build(json.loads(Path(args.collected).read_text()))
    (ROOT / EVIDENCE).write_text(json.dumps(doc, indent=1, sort_keys=True) + '\n')
    print(doc['status'], doc['failed_gates'])
    return 0 if not doc['failed_gates'] else 1


if __name__ == '__main__':
    sys.exit(main())
