"""E03 three-seed family closure: gates the committed per-seed closures of the E03 STDN scientific training family
(seeds 42, 1337, 2026; training authority d547fc9,
FAITHFUL_OFFICIAL_CORE_WITH_TRACK_A_CONTROLLED_DATA_ADAPTATION_AND_RUNTIME_COMPATIBILITY, track
A_FAIR_COMMON_IDENTITY_FREE) and writes outputs/audit/E03_THREE_SEED_{TRAINING_CLOSURE.json,
FINAL_CHECKPOINTS.csv, TRAINING_CLOSURE.md, EVIDENCE_SHA256.txt}. Modelled on tools/e05_three_seed_training_closure.py.

Inputs:
  * the committed per-seed evidence outputs/audit/E03_SEED<s>_{COMPLETION.json,CHECKPOINT_MANIFEST.txt,
    COMPLETION.md,EVIDENCE_SHA256.txt}; every per-seed COMPLETION.json is re-derived both with the seed-generic
    tools/e03_seed_completion_audit.py of this checkout and with the copy of that tool in its own closure commit, and
    the evidence bytes are compared with that closure commit;
  * a read-only host recheck document (ckpt-50 sha256/size, metrics sha256, run-root listing digests,
    resolved_config.yaml text, final host state), collected with tools/e03_family_host_recheck.py and embedded in the
    output so the build re-derives.

This closes STDN TRAINING only. No selection, TEST, bank, downstream or training code is run, and no performance
metric is computed: the frozen primary result rule MEAN_AND_STD_OVER_ALL_THREE_SEEDS applies to future evaluation
metrics. VAL losses are diagnostics only and are reported per seed, never aggregated into a result.
"""
import argparse
import csv
import difflib
import hashlib
import importlib.util
import io
import json
import subprocess
import sys
import types
from datetime import datetime, timezone
from pathlib import Path

CLASSIFICATION = 'E03_THREE_SEED_SCIENTIFIC_TRAINING_CLOSURE'
STATUS_OK = 'SCIENTIFIC_TRAINING_COMPLETED_3_OF_3_SEEDS'
STATUS_BLOCKED = 'E03_FAMILY_NEEDS_REVIEW'
SEEDS = (42, 1337, 2026)
AUTHORITY = 'd547fc9f8932a273d719ae589aade85a9d989b85'
CONFIG_SHA256 = '08dde851bc9e8ac688c794a6a6999204a7e5b75fb244fcd97e46b22dcb302fe0'
FIDELITY = 'FAITHFUL_OFFICIAL_CORE_WITH_TRACK_A_CONTROLLED_DATA_ADAPTATION_AND_RUNTIME_COMPATIBILITY'
METHOD = ('E03', 'STDN', FIDELITY, 'A_FAIR_COMMON_IDENTITY_FREE')
RUN_BASE = '/home/student20261/workdir/GPAT_TransferBench_runtime/runs/m6/E03'
SUFFIXES = ('.data-00000-of-00001', '.index', '.meta')
FINAL_PREFIX = 'checkpoints/ckpt-50'
CHECKPOINT_RULE = 'OFFICIAL_LATEST_FINAL_CKPT_50'
SEED_TOOL = 'tools/e03_seed_completion_audit.py'
RUN_IDS = {42: 'a565b76bc7349532', 1337: 'e8c85cc378e778f5', 2026: '237887eebab18637'}
VAL_SEEDS = {42: 1355875380, 1337: 1050417158, 2026: 1242808125}
CLOSURE_COMMITS = {42: 'b95c127e394d9b2ea644573d317b44405652039e', 1337: '5cdb65db0a75fcdcddb64b1b5af040364a057723',
                   2026: '15bfc84dd4cf1d68427d85d867188b702a847b65'}
KNOWN_FINAL_DATA_SHA = {42: '5b337a63db63e405bf35bceefdacc101060cb7d1625b5a4d2d389b963e311753',
                        1337: '316c94c066c32314c6819bc105d0043e1ddc7b27329afd48ddcfb8bc8d181551',
                        2026: '9dbd463cb3eb56c9ecc578dd275e0c2507a1e6d5f8b9b4876406d5011bd52cb3'}
# the disclosed changes of the seed tool besides RUN_IDS, in order: (tool before, tool after, id, what changed). Neither
# changes the trained model or the scientific procedure; each earlier seed re-derives byte-identically afterwards.
TOOL_AMENDMENTS = (
    (CLOSURE_COMMITS[42], CLOSURE_COMMITS[1337], 'seed1337_pane_log_keyboard_echo',
     'B.pane_log_clean strips only a leading ^[[A..^[[D cursor-key tty echo prefix from the last pane-log line and '
     'discloses it (seed1337: keys pressed in an attached tmux client); the gate set is unchanged and seed42 re-derives '
     'byte-identically with both tool versions'),
    (CLOSURE_COMMITS[1337], CLOSURE_COMMITS[2026], 'seed2026_post_completion_reboot_evidence',
     'owner ruling 2026-10-29 (OWNER_APPROVED_POST_COMPLETION_REBOOT_EVIDENCE_SUBSTITUTION): B.process_exited_zero and '
     'J.single_e03_session may pass as PASS_BY_APPROVED_POST_COMPLETION_REBOOT_EVIDENCE on durable evidence when a '
     'clean host reboot after the run completed removed the tmux server (seed2026); the live tmux path and every other '
     'gate are unchanged and seed42/seed1337 re-derive byte-identically with all three tool versions'))
REBOOT_GATES = ['B.process_exited_zero', 'J.single_e03_session']
REBOOT_RESULT = 'PASS_BY_APPROVED_POST_COMPLETION_REBOOT_EVIDENCE'
# seeds whose two live-tmux gates passed by the reboot ruling -> their preserved first (NEEDS_REVIEW) build
REBOOT_SEEDS = {2026: ('outputs/audit/history/E03_SEED2026_AUDIT1_NEEDS_REVIEW/E03_SEED2026_COMPLETION.json',
                       '62bbc549c838f96be97cfc093fa0b6be66a3a52d0925cbb12858c1583b079ceb')}
SEED_GATES = 88
DRIVER_RULING = {'lock_driver_version': '595.84', 'observed_driver_version': '580.178.04',
                 'driver_policy': 'OWNER_APPROVED_DRIVER_VERSION_ONLY_RUNTIME_COMPATIBILITY'}
# run_manifest fields that differ by design between seeds (everything else must be byte-equal)
MANIFEST_PER_SEED = ('command_line', 'end_utc', 'experiment_seed', 'run_id', 'run_uuid', 'start_utc')
# resolved_config.yaml lines allowed to differ, as functions of the seed
RESOLVED_PER_SEED = (lambda s: f'  experiment_seed: {s}', lambda s: f'  run_dir: {RUN_BASE}/seed_{s}',
                     lambda s: f'    tf_graph_seed: {s}', lambda s: f'      parser_seed: {s}',
                     lambda s: f'      shuffle_seed: {s}', lambda s: f'      parser_seed: {VAL_SEEDS[s]}',
                     lambda s: f'      shuffle_seed: {VAL_SEEDS[s]}')
# resolved_config.yaml lines that pin the shared protocol (each must appear in every seed's file)
PROTOCOL_LINES = ('  batch_size: 2', '  g_d_ratio: 2', '  max_epoch: 50', '  steps_per_epoch: 2000',
                  '  val_steps: 500', '  learning_rate: 6.0e-05', '    every_steps: 20000', '  beta1: 0.9',
                  '  beta2: 0.999', '  epsilon: 1.0e-08', '  weight_ema_decay: 0.9999', '  loss_ema_decay: 0.9',
                  '  rule: OFFICIAL_LATEST_FINAL_CKPT_50', '  resume: NOT_QUALIFIED_FRESH_ONLY',
                  '    global_step_per_epoch: 3000', '    decay_steps_shared_global: 20000')
# run-start fields that differ by design (the seed values only)
RUN_START_PER_SEED = ('seeding', '_seed')
CSV_COLUMNS = ('seed', 'run_id', 'scientific_authority', 'final_checkpoint_prefix', 'final_data_sha256',
               'final_index_sha256', 'final_meta_sha256', 'final_global_step', 'g_applications', 'd_applications',
               'metrics_sha256', 'run_root_listing_digest', 'completion_status', 'train_parser_seed',
               'val_parser_seed', 'TEST_access', 'observed_driver', 'seed_closure_commit')
EVIDENCE = {'closure': 'outputs/audit/E03_THREE_SEED_TRAINING_CLOSURE.json',
            'csv': 'outputs/audit/E03_THREE_SEED_FINAL_CHECKPOINTS.csv',
            'report': 'outputs/audit/E03_THREE_SEED_TRAINING_CLOSURE.md',
            'sha256': 'outputs/audit/E03_THREE_SEED_EVIDENCE_SHA256.txt'}


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def seed_paths(s):
    a = f'outputs/audit/E03_SEED{s}_'
    return {'completion': a + 'COMPLETION.json', 'manifest': a + 'CHECKPOINT_MANIFEST.txt',
            'report': a + 'COMPLETION.md', 'sha256': a + 'EVIDENCE_SHA256.txt'}


def git_show(repo, commit, path):
    r = subprocess.run(['git', 'show', f'{commit}:{path}'], cwd=repo, capture_output=True)
    return r.stdout if r.returncode == 0 else None


def current_tool(repo):
    spec = importlib.util.spec_from_file_location('e03_seed_audit', repo / SEED_TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def committed_tool(repo, commit):
    """The seed tool exactly as committed in a closure commit (source from git, executed as a fresh module)."""
    src = git_show(repo, commit, SEED_TOOL).decode()
    mod = types.ModuleType(f'e03_seed_audit_{commit[:7]}')
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


def seed_row(s, d):
    f, ri, n = d['checkpoints']['final_selected']['sha256'], d['run_root_integrity'], d['counts']
    return {'seed': s, 'run_id': d['run_id'], 'scientific_authority': d['scientific_authority'],
            'final_checkpoint_prefix': d['checkpoints']['final_selected']['prefix'],
            'final_data_sha256': f['.data-00000-of-00001'], 'final_index_sha256': f['.index'],
            'final_meta_sha256': f['.meta'], 'final_global_step': n['final_global_step'],
            'g_applications': n['g_applications'], 'd_applications': n['d_applications'],
            'metrics_sha256': ri['metrics_jsonl_sha256'], 'run_root_listing_digest': ri['sha_listing_digest'],
            'completion_status': d['collected']['run_summary']['completion_status'],
            'train_parser_seed': d['rng']['TRAIN']['parser_seed'], 'val_parser_seed': d['rng']['VAL']['parser_seed'],
            'TEST_access': d['firewall']['test_access'],
            'observed_driver': d['collected']['run_start']['gpu_driver']['observed_driver_version'],
            'seed_closure_commit': CLOSURE_COMMITS[s]}


def csv_text(rows):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=CSV_COLUMNS, lineterminator='\n')
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def strip_keys(d, keys):
    return {k: v for k, v in d.items() if k not in keys}


def without_run_ids(src):
    return [x for x in src.splitlines() if not x.startswith('RUN_IDS = ')]


def build(repo_root, host):
    repo = Path(repo_root)
    G = Gates()
    docs, rows = {}, []
    tool = current_tool(repo)
    tool_src = (repo / SEED_TOOL).read_text()
    chain = [CLOSURE_COMMITS[s] for s in SEEDS]                   # seed tool versions, in closure order
    versions = {c: without_run_ids(committed_tool(repo, c)[1]) for c in chain}
    amendments = [{'id': key, 'from': a, 'to': b, 'note': note,
                   'diff_excluding_run_ids': list(difflib.unified_diff(versions[a], versions[b], a[:7], b[:7], n=0,
                                                                       lineterm=''))}
                  for a, b, key, note in TOOL_AMENDMENTS]
    G('A.seed_tool_current_equals_latest_closure_except_run_ids', without_run_ids(tool_src) == versions[chain[-1]],
      [len(a['diff_excluding_run_ids']) for a in amendments])
    G('A.seed_tool_amendment_chain', [(a['from'], a['to']) for a in amendments] == list(zip(chain, chain[1:]))
      and all(a['diff_excluding_run_ids'] for a in amendments), [a['id'] for a in amendments])

    # A. per-seed evidence: SHA manifests, deterministic re-derivation, bytes equal to each closure commit
    present = sorted(p.name for p in (repo / 'outputs/audit').glob('E03_SEED*_COMPLETION.json'))
    G('A.exact_seed_evidence_set', present == sorted(f'E03_SEED{s}_COMPLETION.json' for s in SEEDS), present)
    for s in SEEDS:
        p = seed_paths(s)
        lines = (repo / p['sha256']).read_text().splitlines()
        listed = {name: digest for digest, name in (x.split('  ') for x in lines)}
        G(f'A.seed{s}.evidence_sha256', sorted(listed) == sorted([p['completion'], p['manifest'], p['report']])
          and all(sha256_bytes((repo / n).read_bytes()) == h for n, h in listed.items()))
        d = json.loads((repo / p['completion']).read_bytes())
        docs[s] = d
        for label, mod in (('current_tool', tool), ('closure_commit_tool', committed_tool(repo, CLOSURE_COMMITS[s])[0])):
            rebuilt, text = mod.build(d['collected'], d['collected_tf'], repo)
            G(f'A.seed{s}.rederives_{label}', json.loads(json.dumps(rebuilt)) == d
              and text == (repo / p['manifest']).read_text() and mod.report(rebuilt) == (repo / p['report']).read_text())
        later = [a['id'] for a in amendments if chain.index(a['from']) >= chain.index(CLOSURE_COMMITS[s])]
        G(f'A.seed{s}.tool_differs_only_in_run_ids_or_disclosed_amendments',
          versions[CLOSURE_COMMITS[s]] == without_run_ids(tool_src) if not later else
          [a['id'] for a in amendments[chain.index(CLOSURE_COMMITS[s]):]] == later, later or 'identical')
        G(f'A.seed{s}.equals_closure_commit', all(git_show(repo, CLOSURE_COMMITS[s], p[k]) == (repo / p[k]).read_bytes()
                                                  for k in ('completion', 'manifest', 'report', 'sha256')),
          CLOSURE_COMMITS[s])

    # B. per-seed closure status and identical gate semantics
    for s, d in docs.items():
        G(f'B.seed{s}.closed_clean', d['status'] == f'E03_SEED{s}_COMPLETED_CLEANLY' and d['failed_gates'] == []
          and all(g['pass'] for g in d['gates']) and d['seed'] == s, d['status'])
    for s, d in docs.items():
        want = REBOOT_GATES if s in REBOOT_SEEDS else []
        alt = [g['gate'] for g in d['gates'] if isinstance(g.get('detail'), dict) and 'result' in g['detail']]
        G(f'B.seed{s}.approved_alternative_gates', alt == want == d.get('gates_passed_by_approved_alternative', [])
          and all(g['detail']['result'] == REBOOT_RESULT for g in d['gates'] if g['gate'] in alt)
          and ('post_completion_reboot' in d.get('disclosures', {})) == bool(want), alt)
    for s, (path, digest) in sorted(REBOOT_SEEDS.items()):
        old = json.loads((repo / path).read_bytes())
        G(f'B.seed{s}.prior_needs_review_build_preserved', sha256_bytes((repo / path).read_bytes()) == digest
          and old['status'] == f'E03_SEED{s}_NEEDS_REVIEW' and old['failed_gates'] == REBOOT_GATES
          and old['collected']['run_manifest']['run_id'] == RUN_IDS[s], path)
    names = {s: [g['gate'] for g in d['gates']] for s, d in docs.items()}
    G('B.same_gate_semantics', names[42] == names[1337] == names[2026] and len(names[42]) == SEED_GATES,
      {s: len(v) for s, v in names.items()})

    # C. authority / identity (common by design)
    mans = {s: d['collected']['run_manifest'] for s, d in docs.items()}
    starts = {s: d['collected']['run_start'] for s, d in docs.items()}
    for s, d in docs.items():
        m, git, tgit = mans[s], d['collected']['git'], d['collected_tf']['git']
        G(f'C.seed{s}.authority', d['scientific_authority'] == m['git_commit'] == git['head'] == tgit['head'] ==
          AUTHORITY and m['git_dirty'] is False and git['status_porcelain'] == '' and git['branch'] == 'm6-baselines')
        rid = sha256_bytes(f'E03|{s}|{CONFIG_SHA256}|{AUTHORITY}'.encode())[:16]
        G(f'C.seed{s}.run_id_rule', rid == d['run_id'] == m['run_id'] == RUN_IDS[s], rid)
    G('C.run_ids_distinct', len({d['run_id'] for d in docs.values()}) == 3)
    G('C.method', all((d['method']['id'], d['method']['name'], d['method']['fidelity'], d['method']['track'])
                      == METHOD and mans[s]['fidelity'] == FIDELITY for s, d in docs.items()))
    G('C.frozen_config_sha', all(d['config_sha256'] == CONFIG_SHA256 == mans[s]['config_sha256']
                                 == d['collected']['files_sha256']['configs/methods/e03_stdn.yaml']
                                 == d['collected']['files_sha256']['frozen_config_snapshot/configs/methods/e03_stdn.yaml']
                                 for s, d in docs.items()))
    common = {s: strip_keys(m, MANIFEST_PER_SEED) for s, m in mans.items()}
    diff_keys = sorted(k for k in set().union(*common.values()) if len({json.dumps(c.get(k), sort_keys=True)
                                                                         for c in common.values()}) > 1)
    G('C.manifest_common_fields_equal', diff_keys == [] and common[42].keys() == common[1337].keys()
      == common[2026].keys(), diff_keys)
    G('C.command_line_differs_only_by_seed', len({m['command_line'].replace(f'--seed {s} ', '--seed <S> ')
                                                  for s, m in mans.items()}) == 1
      and all(f'--seed {s} ' in m['command_line'] and '--resume' not in m['command_line'] for s, m in mans.items()))
    ids = starts[42]['identities']
    G('C.identities_equal', all(r['identities'] == ids for r in starts.values())
      and ids['config_sha256'] == CONFIG_SHA256 and ids['fidelity'] == FIDELITY, sorted(ids))
    norm = {s: strip_keys(r, RUN_START_PER_SEED) for s, r in starts.items()}
    rs_diff = sorted(k for k in norm[42] if len({json.dumps(r.get(k), sort_keys=True) for r in norm.values()}) > 1)
    G('C.run_start_equal_except_seed', rs_diff == [] and all(
        r['seeding']['TRAIN'] == {'parser_seed': s, 'shuffle_seed': s} and r['seeding']['tf_graph_seed'] == s and
        r['seeding']['VAL'] == {'parser_seed': VAL_SEEDS[s], 'shuffle_seed': VAL_SEEDS[s]} and r['_seed'] == s
        for s, r in starts.items()), rs_diff)
    G('C.files_identities_equal', len({json.dumps(d['collected']['files_sha256'], sort_keys=True)
                                       for d in docs.values()}) == 1
      and len({json.dumps(d['collected_tf']['identities_recomputed'], sort_keys=True) for d in docs.values()}) == 1)
    G('C.environment_equal', len({json.dumps(d['collected_tf']['environment_recheck'], sort_keys=True)
                                  for d in docs.values()}) == 1
      and len({json.dumps(d['environment'], sort_keys=True) for d in docs.values()}) == 1
      and all(d['collected_tf']['environment_recheck']['differences_other_than_driver'] == [] for d in docs.values()))
    G('C.driver_ruling_consistent', all({k: d['driver_ruling'][k] for k in DRIVER_RULING} == DRIVER_RULING
                                        and starts[s]['gpu_driver']['observed_driver_version']
                                        == DRIVER_RULING['observed_driver_version']
                                        and starts[s]['gpu_driver']['lock_driver_version']
                                        == DRIVER_RULING['lock_driver_version'] for s, d in docs.items())
      and len({json.dumps(r['gpu_driver'], sort_keys=True) for r in starts.values()}) == 1)
    G('C.official_graph_structure_equal', len({json.dumps(r['structure'], sort_keys=True)
                                               for r in starts.values()}) == 1)
    hs = host['seeds']
    rc_ok, rc_diff = True, {}
    base = hs['42']['resolved_config_yaml'].splitlines()
    for s in SEEDS:
        y = hs[str(s)]['resolved_config_yaml']
        rc_ok &= sha256_bytes(y.encode()) == hs[str(s)]['resolved_config_sha256']
        lines = y.splitlines()
        rc_ok &= len(lines) == len(base)
        changed = [i for i, (a, b) in enumerate(zip(base, lines)) if a != b]
        rc_diff[s] = [lines[i] for i in changed]
        rc_ok &= all(lines[i] in [f(s) for f in RESOLVED_PER_SEED] and base[i] in [f(42) for f in RESOLVED_PER_SEED]
                     for i in changed)
        rc_ok &= all(sum(f(s) == x for x in lines) >= 1 for f in RESOLVED_PER_SEED)
        rc_ok &= all(x in lines for x in PROTOCOL_LINES)
    G('C.resolved_config_differs_only_by_seed', rc_ok, rc_diff)

    # D. training completion / data protocol (identical protocol, seed-derived streams)
    for s, d in docs.items():
        n, m = d['counts'], d['collected']['metrics']
        G(f'D.seed{s}.completed_50_epochs', n == {'epochs': 50, 'train_steps': 100000, 'val_steps': 25000,
                                                  'val_blocks': 50, 'g_applications': 100000,
                                                  'd_applications': 50000, 'final_global_step': 150000,
                                                  'checkpoints': 50, 'checkpoint_files': 150}
          and d['collected']['run_summary']['completion_status'] == 'completed')
        G(f'D.seed{s}.streams_independently_replayed', all(
            m['streams'][sp]['sha256'] == d['collected_tf']['predicted_streams'][sp]['sha256'] and
            d['rng']['flips'][sp]['match'] == d['rng']['flips'][sp]['pairs'] for sp in ('TRAIN', 'VAL'))
          and d['rng']['VAL']['parser_seed'] == VAL_SEEDS[s] and d['rng']['TRAIN']['parser_seed'] == s)
        G(f'D.seed{s}.fresh_run', d['resume'] == 'NOT_QUALIFIED_FRESH_ONLY' and mans[s]['resume'] == d['resume']
          and d['collected']['event_counts'].get('e03_run_start') == 1 and d['confirmations']['resume_used'] is False
          and d['collected']['resume_files'] == [] and d['collected']['runner_state_files'] == [])
    G('D.seed_streams_differ', all(len({d['rng'][k] for d in docs.values()}) == 3
                                   for k in ('train_stream_sha256', 'val_stream_sha256'))
      and len({d['run_root_integrity']['metrics_jsonl_sha256'] for d in docs.values()}) == 3)

    # E. numerical health
    for s, d in docs.items():
        m = d['collected']['metrics']
        G(f'E.seed{s}.finite', m['train_checks']['losses'] == 100000 and m['val_checks']['losses'] == 25000
          and all(r['nonfinite_variables'] == 0 and r['variables'] == 747 for r in d['collected_tf']['checkpoints'])
          and all(v['bytes'] == 0 for v in d['collected']['logs'].values()))

    # F. checkpoints: ckpt-50 is the final state of each seed, independently
    for s, d in docs.items():
        summ, fs = d['collected']['run_summary'], d['checkpoints']['final_selected']
        G(f'F.seed{s}.ckpt50_final', fs['prefix'] == FINAL_PREFIX and fs['rule'] == CHECKPOINT_RULE
          and fs['sha256']['.data-00000-of-00001'] == summ['final_or_selected_checkpoint_sha256']
          == KNOWN_FINAL_DATA_SHA[s] and summ['final_or_selected_checkpoint_path'] == f'{RUN_BASE}/seed_{s}/'
          f'{FINAL_PREFIX}' and summ['checkpoint_selection_rule'] == CHECKPOINT_RULE
          and [x['epoch'] for x in d['checkpoints']['files']] == list(range(1, 51))
          and d['checkpoints']['resume_sidecars'] == 0)
        rows.append(seed_row(s, d))
    G('F.final_checkpoints_distinct', len({r['final_data_sha256'] for r in rows}) == 3)
    G('F.run_roots_distinct', sorted(d['run_root'] for d in docs.values())
      == sorted(f'{RUN_BASE}/seed_{s}' for s in SEEDS))

    # G. firewall / confirmations
    for s, d in docs.items():
        fw = d['firewall']
        tests = {c: v for c, v in fw['atime']['by_class'].items() if c.endswith('|TEST')}
        G(f'G.seed{s}.test_0', (fw['test_access'], fw['test_changed'], fw['test_files']) == (0, 0, 3027)
          and len(tests) == 3 and all(v['atime_changed'] == 0 and v['sensitive_whole_window'] == v['files']
                                      and v['consumed'] == 0 for v in tests.values())
          and all(v['changed_not_consumed'] == 0 for v in fw['atime']['by_class'].values())
          and mans[s]['test_split_accessed'] is False and mans[s]['val_role'] == 'DIAGNOSTICS_ONLY_NEVER_SELECTS')
        G(f'G.seed{s}.no_best_or_downstream', not any(d['confirmations'].values()))

    # H. read-only host recheck at family closure (run roots still immutable, nothing running)
    G('H.host_parent_exactly_three_seeds', [p for p in host['parent_listing'] if not p.startswith('.')]
      == sorted(f'seed_{s}' for s in SEEDS), host['parent_listing'])
    for s, d in docs.items():
        h, ri, fs = hs[str(s)], d['run_root_integrity'], d['checkpoints']['final_selected']['sha256']
        sizes = {sf: next(x for x in d['checkpoints']['files'] if x['epoch'] == 50)['size'][sf] for sf in SUFFIXES}
        G(f'H.seed{s}.host_matches_evidence', all(h['final_checkpoint'][FINAL_PREFIX + sf] ==
                                                  {'sha256': fs[sf], 'bytes': sizes[sf]} for sf in SUFFIXES)
          and h['metrics_jsonl_sha256'] == ri['metrics_jsonl_sha256'] and h['sha_listing_digest']
          == ri['sha_listing_digest'] and h['file_count'] == ri['file_count'] and h['listing_stable_during_recheck'])
    st = host['host_state']
    # E03 tmux panes: dead with status 0, or (seed2026 reboot ruling) all gone because the host is still in the boot that
    # followed the verified post-completion reboot, which started after every seed's run had ended
    reb = docs[2026].get('disclosures', {}).get('post_completion_reboot', {})
    last_end = max(datetime.strptime(d['end_utc'], '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc).timestamp()
                   for d in docs.values())
    panes_live = all(v == '1 0' for v in st['e03_tmux_panes'].values())
    panes_rebooted = (all(v == '' for v in st['e03_tmux_panes'].values()) and 'tmux_sessions' in st
                      and not any(x.startswith('e03_stdn_seed') for x in st['tmux_sessions'])
                      and reb.get('current_boot_id') is not None and st.get('boot_id') == reb['current_boot_id']
                      and isinstance(st.get('btime'), int) and st['btime'] > last_end)
    h_index = len(G.results)
    G('H.host_final_state', st['hostname'] == 'nd-System-Product-Name'
      and st['gpu'] == 'NVIDIA GeForce RTX 3090, 580.178.04, GPU-b722cd9d-c9fa-223c-915e-cc09d9b862b7'
      and st['method_processes'] == [] and (panes_live or panes_rebooted),
      {'method_processes': st['method_processes'],
       'e03_tmux': 'LIVE_PANES_DEAD_STATUS_0' if panes_live else REBOOT_RESULT if panes_rebooted else 'FAIL',
       'boot_id': st.get('boot_id'), 'seed2026_verified_reboot_boot_id': reb.get('current_boot_id')})
    G('H.gpu_checkout_clean_on_branch', st['git_branch'] == 'm6-baselines' and st['git_status_porcelain'] == '',
      st['git_head'])
    # owner ruling 2026-10-29 (FAMILY_HOST_STATE_RULING, condition 7): the reboot path of H.host_final_state passes only
    # if every other family gate passed; the live-pane path is unchanged
    others_failed = [g for g in G.failed if g != 'H.host_final_state']
    if not panes_live and panes_rebooted and others_failed:
        G.results[h_index]['pass'] = False
        G.results[h_index]['detail'] = dict(G.results[h_index]['detail'], e03_tmux='FAIL_POST_COMPLETION_REBOOT_EVIDENCE',
                                            blocked_by=others_failed)

    text = csv_text(rows)
    lm = {str(s): d['loss_diagnostics'] for s, d in docs.items()}
    doc = {
        'classification': CLASSIFICATION, 'status': STATUS_BLOCKED if G.failed else STATUS_OK,
        'failed_gates': G.failed, 'gates': G.results, 'gates_passed': len(G.results) - len(G.failed),
        'gates_total': len(G.results),
        'method': dict(zip(('id', 'name', 'fidelity', 'track'), METHOD)),
        'family': {'seeds': list(SEEDS), 'completed_seeds': sum(r['completion_status'] == 'completed' for r in rows),
                   'scientific_runs': len(rows), 'run_ids': {str(s): RUN_IDS[s] for s in SEEDS},
                   'dropped_seeds': [], 'replacement_seeds': [], 'preferred_seed': None, 'best_seed': None,
                   'composition_note': 'exactly the three frozen experiment seeds; no fourth, replacement, preferred '
                                       'or dropped seed, and no cross-seed choice'},
        'scientific_training_authority': AUTHORITY,
        'authority_note': 'Training authority for all three seeds is d547fc9; the seed closure commits and this family '
                          'closure are evidence/reporting only.',
        'frozen': {'config_sha256': CONFIG_SHA256, 'identities': ids,
                   'files_sha256': docs[42]['collected']['files_sha256'], 'structure': starts[42]['structure'],
                   'session_config': starts[42]['session_config'], 'pack': starts[42]['pack']},
        'protocol': {'epochs': 50, 'train_steps_per_epoch': 2000, 'val_steps_per_epoch': 500, 'batch_size': 2,
                     'g_d_ratio': 2, 'schedule': 'step % 2 == 0: g_op + d_op (+2 shared global_step); else g_op (+1)',
                     'final_global_step': 150000, 'g_applications': 100000, 'd_applications': 50000,
                     'optimizer': 'Adam 6e-5, betas (0.9, 0.999), eps 1e-8, exponential decay 0.9 every 20000 shared '
                                  'global steps (staircase); weight EMA 0.9999, loss EMA 0.9',
                     'feed': 'pairs_train_v1 / val_pairs_v1 via e03_feed_pack_v1; independent TRAIN/VAL RNG (ruling V)',
                     'checkpoints': 'ckpt-1..ckpt-50, each after its TRAIN block and before its VAL block',
                     'checkpoint_rule': CHECKPOINT_RULE, 'final_checkpoint': FINAL_PREFIX,
                     'final_rule_scope': 'applied independently within each seed; no cross-seed choice',
                     'val_role': 'DIAGNOSTICS_ONLY_NEVER_SELECTS', 'resume': 'NOT_QUALIFIED_FRESH_ONLY',
                     'fidelity_note': 'official STDN core unchanged with Track-A controlled data adaptation and '
                                      'runtime compatibility; not a fully native reproduction'},
        'different_by_design': {'manifest_fields': list(MANIFEST_PER_SEED),
                                'resolved_config_lines': {str(s): v for s, v in rc_diff.items()},
                                'run_start_fields': list(RUN_START_PER_SEED),
                                'also': 'seed-derived TRAIN/VAL streams, flips, brightness and model initialisation; '
                                        'learned weights and checkpoint bytes'},
        'driver_ruling': dict(DRIVER_RULING, scope='only the NVIDIA driver version differs from the lock; GPU model, '
                                                   'GPU UUID and every other lock field are exact in all three runs'),
        'final_checkpoints': rows, 'final_checkpoints_csv': EVIDENCE['csv'],
        'final_checkpoints_csv_sha256': sha256_bytes(text.encode()),
        'aggregation': {'primary_result_rule': 'MEAN_AND_STD_OVER_ALL_THREE_SEEDS',
                        'applies_to': 'future downstream evaluation metrics (none computed yet)',
                        'training_metadata': {'seeds_completed': '3/3', 'epochs_each': 50, 'train_steps_each': 100000,
                                              'g_applications_each': 100000, 'd_applications_each': 50000,
                                              'final_global_step_each': 150000, 'checkpoints_each': 50,
                                              'val_blocks_each': 50, 'resume_sidecars_each': 0,
                                              'numerically_finite_each': True, 'test_access_total': 0},
                        'descriptive_losses': {
                            'note': 'per-seed epoch-1 / epoch-50 mean TRAIN and VAL (diagnostic) losses; descriptive '
                                    'only, NOT a performance metric, not aggregated and not used for any choice',
                            'by_seed': lm},
                        'wall_clock_note': 'training speed is not an acceptance gate and not a comparison metric'},
        'firewall': {'test_access': 0 if not G.failed else None,
                     'note': 'per-seed stat-only relatime atime snapshots: 3027/3027 TEST faces atime-sensitive and '
                             'unchanged in every run; every changed face was a consumed TRAIN/VAL pack face'},
        'not_completed': {'synthetic_bank': 'NOT_STARTED', 'downstream_fas_evaluation': 'NOT_STARTED',
                          'whole_benchmark': 'NOT_COMPLETED'},
        'confirmations': {'best_seed_selection': False, 'preferred_seed_selection': False, 'seed_dropped': False,
                          'replacement_seed': False, 'additional_seed_launched': False,
                          'val_checkpoint_selection': False, 'test_used': False, 'bank_generation': False,
                          'downstream_evaluation': False, 'other_method_started': False, 'gpat_started': False,
                          'run_root_mutated': False},
        'per_seed_evidence': {str(s): dict(seed_paths(s), tool=SEED_TOOL, closure_commit=CLOSURE_COMMITS[s])
                              for s in SEEDS},
        'per_seed_disclosures': {str(s): d['disclosures'] for s, d in docs.items() if 'disclosures' in d},
        'seed_tool_amendments': amendments,
        'approved_alternative_gates': {str(s): d.get('gates_passed_by_approved_alternative', [])
                                       for s, d in docs.items()},
        'preserved_diagnostics': {str(s): {'path': p, 'sha256': h} for s, (p, h) in REBOOT_SEEDS.items()},
        'host_recheck': host,
    }
    return doc, text


def report(doc):
    rows = doc['final_checkpoints']
    lm = doc['aggregation']['descriptive_losses']['by_seed']
    fmt = lambda xs: ' / '.join('%.4g' % x for x in xs)  # noqa: E731
    out = [
        '# E03 STDN: three-seed scientific training closure', '',
        '**Status:** `%s`, %d/%d gates passing (`E03_THREE_SEED_TRAINING_CLOSURE.json`).'
        % (doc['status'], doc['gates_passed'], doc['gates_total']), '',
        'Training authority for all three seeds: **`%s`**. The per-seed closure commits and this closure are evidence '
        'only.' % AUTHORITY, '',
        'Execution fidelity: `%s` (not a fully native reproduction).' % FIDELITY, '',
        '| seed | run_id | G / D | global_step | ckpt-50 data sha256 | run-root listing | TEST | closure commit |',
        '|---|---|---|---|---|---|---|---|']
    out += ['| %d | `%s` | %d / %d | %d | `%s` | `%s…` | %d | `%s` |' % (
        r['seed'], r['run_id'], r['g_applications'], r['d_applications'], r['final_global_step'],
        r['final_data_sha256'], r['run_root_listing_digest'][:16], r['TEST_access'], r['seed_closure_commit'][:7])
        for r in rows]
    out += ['', '## What is common and what differs', '',
            '- Common to all three seeds:',
            '  - frozen config `08dde851…` and every identity;',
            '  - the official graph structure and session configuration;',
            '  - the feed pack `c8c9850e…` and the environment re-check;',
            '  - the driver ruling (580.178.04 vs lock 595.84, version only);',
            '  - 50 × (2000 TRAIN, ckpt-e, 500 VAL);',
            '  - G 100000 / D 50000 / global_step 150000.',
            '- Differs by design only:',
            '  - the seed, the run_id and the timestamps;',
            '  - the TRAIN seed = experiment seed and the VAL seed from ruling V;',
            '  - the resulting streams and learned weights.',
            '- Every seed\'s TRAIN/VAL streams equal an independent shuffle replay and its flips equal a numpy '
            'prediction.',
            '- ckpt-50 is the final checkpoint of each seed by `OFFICIAL_LATEST_FINAL_CKPT_50`. There is no '
            'cross-seed choice and no best seed, and no seed was dropped or replaced.',
            '- TEST: 0 accesses in every seed.',
            '', '## Disclosures', '']
    out += ['- seed%s: %s' % (s, '; '.join('`%s`: %s' % (k, v['note']) for k, v in sorted(x.items())))
            for s, x in sorted(doc['per_seed_disclosures'].items(), key=lambda x: int(x[0]))]
    out += ['- Seed tool amendment %d, `%s` (`%s` → `%s`): %s.' % (i, a['id'], a['from'][:7], a['to'][:7], a['note'])
            for i, a in enumerate(doc['seed_tool_amendments'], 1)]
    out += ['- Gates passed by the owner-approved post-completion-reboot evidence: %s. The first seed2026 build '
            '(86/88, NEEDS_REVIEW) is kept unchanged at `%s`. At this family recheck, the E03 tmux panes are gone '
            'because the host is still in the boot that followed the verified reboot (`H.host_final_state`: `%s`).'
            % ('; '.join('seed%s `%s`' % (s, '`, `'.join(v)) for s, v in sorted(
                doc['approved_alternative_gates'].items(), key=lambda x: int(x[0])) if v) or 'none',
               doc['preserved_diagnostics']['2026']['path'],
               next(g for g in doc['gates'] if g['gate'] == 'H.host_final_state')['detail']['e03_tmux']),
            '', '## Aggregation', '',
            'The frozen primary result rule is `MEAN_AND_STD_OVER_ALL_THREE_SEEDS`. It applies to downstream evaluation '
            'metrics, and none has been computed. The losses below are descriptive only: VAL is diagnostics, and '
            'nothing here is aggregated or ranked.', '',
            '| seed | TRAIN epoch 50 (a / d / g) | VAL epoch 50 (a / d / g) |', '|---|---|---|']
    out += ['| %s | %s | %s |' % (s, fmt(v['train_epoch_means']['50']), fmt(v['val_epoch_means']['50']))
            for s, v in sorted(lm.items(), key=lambda x: int(x[0]))]
    out += ['', '## Not completed', '',
            'The E03 synthetic bank, downstream FAS evaluation and the whole benchmark have not started. This closes '
            'STDN training only.', '']
    return '\n'.join(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--host-recheck', required=True)
    ap.add_argument('--repo-root', default=str(Path(__file__).resolve().parents[1]))
    args = ap.parse_args(argv)
    doc, text = build(args.repo_root, json.loads(Path(args.host_recheck).read_text()))
    root = Path(args.repo_root)
    (root / EVIDENCE['csv']).write_text(text)
    (root / EVIDENCE['closure']).write_text(json.dumps(doc, indent=2, sort_keys=True) + '\n')
    (root / EVIDENCE['report']).write_text(report(doc))
    (root / EVIDENCE['sha256']).write_text(''.join('%s  %s\n' % (sha256_bytes((root / EVIDENCE[k]).read_bytes()),
                                                                 EVIDENCE[k]) for k in ('closure', 'csv', 'report')))
    print(doc['status'], doc['gates_passed'], '/', doc['gates_total'], doc['failed_gates'])
    return 0 if doc['status'] == STATUS_OK else 1


if __name__ == '__main__':
    sys.exit(main())
