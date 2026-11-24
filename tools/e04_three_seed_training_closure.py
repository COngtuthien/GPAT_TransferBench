"""E04 three-seed family closure: gates the committed per-seed closures of the E04 Physics-STD scientific training family
(seeds 42, 1337, 2026; launch authority 18ba00e, production qualification b0f0ffd under the owner-approved D12
procedural-reproducibility ruling c60481b, fidelity CONTROLLED_ADAPTATION, track A_FAIR_COMMON_IDENTITY_FREE) and writes
outputs/audit/E04_THREE_SEED_{TRAINING_CLOSURE.json, FINAL_CHECKPOINTS.csv, CHECKPOINT_INVENTORY.csv,
REPRODUCTION_MANIFEST.json, TRAINING_CLOSURE.md, EVIDENCE_SHA256.txt}. Modelled on tools/e03_three_seed_training_closure.py.

Inputs:
  * the committed per-seed evidence outputs/audit/E04_SEED<s>_{COMPLETION.json,CHECKPOINT_MANIFEST.txt,
    COMPLETION.md,EVIDENCE_SHA256.txt}; every per-seed COMPLETION.json is re-derived both with the seed-generic
    tools/e04_seed_completion_audit.py of this checkout and with the copy of that tool in its own closure commit, and
    the evidence bytes are compared with that closure commit;
  * a read-only host recheck document (sha256/size of all 30 checkpoint files per seed, metrics sha256, run-root
    listing digests, resolved_config.yaml text, wrapper markers, final host state), collected with
    tools/e04_family_host_recheck.py and embedded in the output so the build re-derives.

This closes Physics-STD TRAINING only. No selection, TEST, bank, downstream or training code is run, and no performance
metric is computed: the frozen primary result rule MEAN_AND_STD_OVER_ALL_THREE_SEEDS applies to future evaluation
metrics. D12: the three runs are independent GPU executions that are not bitwise reproducible; no run is compared
bitwise to another and no seed is preferred.
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

CLASSIFICATION = 'E04_THREE_SEED_SCIENTIFIC_TRAINING_CLOSURE'
STATUS_OK = 'SCIENTIFIC_TRAINING_COMPLETED_3_OF_3_SEEDS'
STATUS_BLOCKED = 'E04_FAMILY_NEEDS_REVIEW'
SEEDS = (42, 1337, 2026)
AUTHORITY = '18ba00e94813f005876c8d6235b0b848226b2c1d'
QUALIFICATION_AUTHORITY = 'b0f0ffd06b26e1c2d542df9e09101417ee3b687c'
D12_RULING_COMMIT = 'c60481b91922c1c72c73a603219585ec4a78c1e4'
D12_CRITERIA_SHA = '9f1a2471f0c9e6ad24b9cc4727779361b763ec84b7e43dc221fc6b411c199d0d'
CONFIG_SHA256 = '418617942ebd87c45cb92bb7dc9ad96a6500a104489a657f9ca3345c3686576a'
FIDELITY = 'CONTROLLED_ADAPTATION'
METHOD = ('E04', 'Physics-STD', FIDELITY, 'A_FAIR_COMMON_IDENTITY_FREE')
RUN_BASE = '/home/student20261/workdir/GPAT_TransferBench_runtime/runs/m6/E04'
SUFFIXES = ('.data-00000-of-00001', '.index', '.meta')
ITERATIONS, EVERY = 150000, 15000
CHECKPOINT_ITERATIONS = list(range(EVERY, ITERATIONS + 1, EVERY))
FINAL_PREFIX = 'checkpoints/iteration_150000'
CHECKPOINT_RULE = 'BASELINE_FINAL_STATE_V1'
SEED_TOOL = 'tools/e04_seed_completion_audit.py'
RUN_IDS = {42: '40befa6004aee85c', 1337: 'a42fdd2a2accaf0a', 2026: 'e7100eb9a5ab0537'}
DERIVED = {42: (679869707, 320493565), 1337: (1349475396, 35554372), 2026: (692890966, 1053992267)}
CLOSURE_COMMITS = {42: 'ea533db43dc3b9d0a4108364d68b2e8217c7c835', 1337: '6da67199b32710c728b9f16f89c32b0a26cad84f',
                   2026: '0e3f1d1a373d3e60130fd472a4e1799e2a622487'}
KNOWN_FINAL_DATA_SHA = {42: '55891ee1fd3a46ea3c451680d9a1a67998ede8246d754b6ea92aaf4bc45d71c6',
                        1337: 'fffa5f252c6dde2e1c5a49b380cec0bffe5d13e27ab711cbb3983188897ae0c5',
                        2026: '089c5e01da28c086e3660d4ba1ae4248cbf80cae907a163a0282f4af6812e0b2'}
# the disclosed changes of the seed tool besides RUN_IDS, in order: (tool before, tool after, id, what changed). Neither
# changes the trained model or the scientific procedure; each earlier seed re-derives byte-identically afterwards.
TOOL_AMENDMENTS = (
    (CLOSURE_COMMITS[42], CLOSURE_COMMITS[1337], 'seed1337_uninterrupted_boot_branch',
     'B.no_reboot_during_run also accepts an uninterrupted boot (no wtmp boot after launch, run boot "still running", '
     '/proc btime before launch and equal to that record within 120 s, same-boot_id journal boot spanning launch..end); '
     'a reboot, shutdown or journal boot inside the window still fails and seed42 re-derives byte-identically'),
    (CLOSURE_COMMITS[1337], CLOSURE_COMMITS[2026], 'seed2026_finished_sibling_session',
     'J.no_other_e04_runner accepts a leftover e04_* tmux session only when it is another registered seed\'s session '
     'with one dead pane (status 0), exit marker "0" and an end marker before this seed\'s launch (the finished seed-1337 '
     'session); no runner process is still required, and seed42/seed1337 re-derive byte-identically'))
SEED_GATES = 90
DRIVER_RULING = {'lock_driver_version': '595.84', 'observed_driver_version': '580.178.04',
                 'driver_policy': 'OWNER_APPROVED_DRIVER_VERSION_ONLY_RUNTIME_COMPATIBILITY (E04 D9, conditional '
                                  'narrow scope)'}
# run_manifest fields that differ by design between seeds (everything else must be byte-equal), plus the disclosed
# host kernel (not part of the E04 lock): seed 42 ran on 7.0.0-38, seeds 1337/2026 on 7.0.0-43
MANIFEST_PER_SEED = ('command_line', 'end_utc', 'experiment_seed', 'run_id', 'run_uuid', 'start_utc')
KERNEL_FIELDS = ('platform',)
KERNELS = {42: 'Linux-7.0.0-38-generic-x86_64-with-glibc2.17', 1337: 'Linux-7.0.0-43-generic-x86_64-with-glibc2.17',
           2026: 'Linux-7.0.0-43-generic-x86_64-with-glibc2.17'}
RUN_START_PER_SEED = ('_seed', '_utc', 'seeding')
COUNTS = {'iterations': 150000, 'final_global_iteration': 150000, 'g_applications': 300000, 'd_applications': 150000,
          'checkpoints': 10, 'checkpoint_files': 30, 'metrics_records': 150011, 'face_reads': 1200000,
          'pair_rows_consumed': 600000}
CSV_COLUMNS = ('seed', 'run_id', 'scientific_authority', 'final_checkpoint_prefix', 'final_data_sha256',
               'final_index_sha256', 'final_meta_sha256', 'final_state_sha256', 'final_global_iteration',
               'g_applications', 'd_applications', 'metrics_sha256', 'run_root_listing_digest', 'completion_status',
               'pair_order_seed', 'hard_sample_seed', 'TEST_access', 'VAL_atime_changed', 'observed_driver',
               'seed_closure_commit')
INVENTORY_COLUMNS = ('seed', 'run_id', 'iteration', 'file', 'bytes', 'sha256', 'state_sha256', 'final_selected')
EVIDENCE = {'closure': 'outputs/audit/E04_THREE_SEED_TRAINING_CLOSURE.json',
            'csv': 'outputs/audit/E04_THREE_SEED_FINAL_CHECKPOINTS.csv',
            'inventory': 'outputs/audit/E04_THREE_SEED_CHECKPOINT_INVENTORY.csv',
            'reproduction': 'outputs/audit/E04_THREE_SEED_REPRODUCTION_MANIFEST.json',
            'report': 'outputs/audit/E04_THREE_SEED_TRAINING_CLOSURE.md',
            'sha256': 'outputs/audit/E04_THREE_SEED_EVIDENCE_SHA256.txt'}
LAUNCH_ENV = ('PATH=/home/student20261/miniconda3/envs/gpat-m6-e03/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:'
              '/usr/bin:/sbin:/bin CUDA_VISIBLE_DEVICES=0 NVIDIA_TF32_OVERRIDE=0 TF_ENABLE_AUTO_MIXED_PRECISION=0 '
              'TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE=0 TF_ENABLE_CUBLAS_TENSOR_OP_MATH_FP32=0 '
              'TF_ENABLE_CUDNN_TENSOR_OP_MATH_FP32=0 TF_ENABLE_CUDNN_RNN_TENSOR_OP_MATH_FP32=0 '
              'TF_XLA_FLAGS=--tf_xla_auto_jit=0 TF_DETERMINISTIC_OPS=1 TF_CUDNN_DETERMINISTIC=1 '
              'CUBLAS_WORKSPACE_CONFIG=:4096:8 TF_ENABLE_ONEDNN_OPTS=0 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 '
              'PYTHONHASHSEED=<SEED>')


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def seed_paths(s):
    a = f'outputs/audit/E04_SEED{s}_'
    return {'completion': a + 'COMPLETION.json', 'manifest': a + 'CHECKPOINT_MANIFEST.txt',
            'report': a + 'COMPLETION.md', 'sha256': a + 'EVIDENCE_SHA256.txt'}


def git_show(repo, commit, path):
    r = subprocess.run(['git', 'show', f'{commit}:{path}'], cwd=repo, capture_output=True)
    return r.stdout if r.returncode == 0 else None


def current_tool(repo):
    spec = importlib.util.spec_from_file_location('e04_seed_audit', repo / SEED_TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def committed_tool(repo, commit):
    """The seed tool exactly as committed in a closure commit (source from git, executed as a fresh module)."""
    src = git_show(repo, commit, SEED_TOOL).decode()
    mod = types.ModuleType(f'e04_seed_audit_{commit[:7]}')
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
    fs, ri, n = d['checkpoints']['final_selected'], d['run_root_integrity'], d['counts']
    f = fs['sha256']
    return {'seed': s, 'run_id': d['run_id'], 'scientific_authority': d['scientific_authority'],
            'final_checkpoint_prefix': fs['prefix'], 'final_data_sha256': f['.data-00000-of-00001'],
            'final_index_sha256': f['.index'], 'final_meta_sha256': f['.meta'], 'final_state_sha256': fs['state_sha256'],
            'final_global_iteration': n['final_global_iteration'], 'g_applications': n['g_applications'],
            'd_applications': n['d_applications'], 'metrics_sha256': ri['metrics_jsonl_sha256'],
            'run_root_listing_digest': ri['sha_listing_digest'],
            'completion_status': d['collected']['run_summary']['completion_status'],
            'pair_order_seed': d['rng']['pair_order_seed'], 'hard_sample_seed': d['rng']['hard_sample_seed'],
            'TEST_access': d['firewall']['test_access'], 'VAL_atime_changed': d['firewall']['by_split']['VAL']['atime_changed'],
            'observed_driver': d['collected']['run_start']['gpu_driver']['observed_driver_version'],
            'seed_closure_commit': CLOSURE_COMMITS[s]}


def inventory_rows(s, d):
    out = []
    for r in d['checkpoints']['files']:
        prefix = 'checkpoints/iteration_%06d' % r['iteration']
        for sf in SUFFIXES:
            out.append({'seed': s, 'run_id': d['run_id'], 'iteration': r['iteration'], 'file': prefix + sf,
                        'bytes': r['size'][sf], 'sha256': r['sha256'][sf], 'state_sha256': r['state_sha256'],
                        'final_selected': prefix == FINAL_PREFIX})
    return out


def csv_text(rows, columns):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=columns, lineterminator='\n')
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def strip_keys(d, keys):
    return {k: v for k, v in d.items() if k not in keys}


def without_run_ids(src):
    return [x for x in src.splitlines() if not x.startswith('RUN_IDS = ')]


def reproduction(docs, host):
    """Per-seed reproduction manifest of the training procedure (D12: procedural, not bitwise)."""
    out = {}
    for s, d in sorted(docs.items()):
        c, t = d['collected'], d['collected_tf']
        m, h = c['run_manifest'], c['host']
        fs = d['checkpoints']['final_selected']
        out[str(s)] = {
            'method_id': 'E04', 'seed': s, 'run_id': d['run_id'],
            'source_commit': AUTHORITY, 'branch': 'm6-baselines',
            'qualification_authority': QUALIFICATION_AUTHORITY, 'd12_ruling_commit': D12_RULING_COMMIT,
            'd12_acceptance_criteria_sha256': D12_CRITERIA_SHA,
            'frozen_config': {'path': 'configs/methods/e04_physics_std.yaml', 'sha256': CONFIG_SHA256},
            'identities': m['identities'],
            'input_population': {'pairs_train_v1_sha256': m['identities']['pairs_train_v1_sha256'],
                                 'feed_pack_manifest_sha256': m['identities']['pack_manifest_sha256'],
                                 'prepass_manifest_sha256': m['identities']['prepass_manifest_sha256'],
                                 'split_manifest_sha256': m['identities']['split_manifest_sha256'],
                                 'train_samples_consumed': c['consumed_faces']['samples'],
                                 'val_rows': c['pack']['val_rows'], 'test_rows': c['pack']['test_rows']},
            'environment': {'name': m['environment_name'], 'python_executable': m['python_executable'],
                            'python_version': m['python_version'], 'tensorflow': m['tensorflow_version'],
                            'numpy': m['numpy_version'], 'cuda_runtime': m['cuda_version'],
                            'cudnn': m['cudnn_version'], 'environment_lock': m['environment_lock_path'],
                            'environment_lock_sha256': m['identities']['environment_lock_sha256'],
                            'gpu_model': m['gpu_model'], 'gpu_uuid': d['collected_tf']['environment_recheck']['driver']['gpu_uuid'],
                            'driver_ruling': d['driver_ruling'], 'run_platform': d['environment']['run_platform']},
            'rng': {'experiment_seed': s, 'pair_order_seed': d['rng']['pair_order_seed'],
                    'hard_sample_seed': d['rng']['hard_sample_seed'], 'tf_graph_seed': d['rng']['tf_graph_seed'],
                    'derivation': '1 + (u32_be(SHA256("GPAT_TRANSFERBENCH|E04|<PAIR_ORDER|HARD_SAMPLE>|<seed>")[0:4]) '
                                  'mod 2147483647)',
                    'pair_stream_sha256_int32_le': d['rng']['pair_sha256_int32_le'],
                    'hard_stream_sha256': d['rng']['hard_sha256']},
            'command': {'working_directory': '/home/student20261/workdir/GPAT_TransferBench',
                        'environment': LAUNCH_ENV.replace('<SEED>', str(s)),
                        'invocation': 'python -u -B tools/run_e04.py --seed %d --execution-config '
                                      'configs/execution/m5_gpu_3090.yaml' % s,
                        'runner_command_line': m['command_line'],
                        'wrapper_file': '~/workdir/e04_s%d.command' % s,
                        'wrapper_sha256': h['command_file_sha256'],
                        'prerequisite': 'fresh run root (NOT_QUALIFIED_FRESH_ONLY); GPU checkout at the source commit, '
                                        'clean; run_id = sha256("E04|<seed>|<config sha256>|<source commit>")[:16]'},
            'window': {'start_utc': m['start_utc'], 'end_utc': m['end_utc'], 'exit_status': int(h['exit_file']),
                       'exit_marker_utc': h['end_time']},
            'outputs': {'run_root': d['run_root'], 'run_root_listing_digest': d['run_root_integrity']['sha_listing_digest'],
                        'metrics_jsonl_sha256': d['run_root_integrity']['metrics_jsonl_sha256'],
                        'final_checkpoint': {'path': fs['path'], 'rule': fs['rule'], 'sha256': fs['sha256'],
                                             'state_sha256': fs['state_sha256']},
                        'checkpoint_manifest': seed_paths(s)['manifest']},
            'audit': {'completion_record': seed_paths(s)['completion'], 'status': d['status'],
                      'gates': '%d/%d' % (d['gates_passed'], d['gates_total']), 'closure_commit': CLOSURE_COMMITS[s],
                      'test_access': d['firewall']['test_access'],
                      'val_atime_changed': d['firewall']['by_split']['VAL']['atime_changed'],
                      'host_recheck_listing_digest': host['seeds'][str(s)]['sha_listing_digest']},
            'reproducibility': 'D12 procedural: re-running the command under the same source, config, environment and '
                               'seed reproduces the same data streams, schedule and optimizer accounting; GPU '
                               'numerical results (weights, checkpoint bytes) are not bitwise reproducible',
            'known_limitations': ['GPU training not bitwise reproducible (D12)',
                                  'NVIDIA driver 580.178.04 vs lock 595.84 (D9 owner-approved, version only)',
                                  'checkpoint bytes live only in the run root on the GPU host; git holds hashes only']}
    return out


def build(repo_root, host):
    repo = Path(repo_root)
    G = Gates()
    docs, rows, inv = {}, [], []
    tool = current_tool(repo)
    tool_src = (repo / SEED_TOOL).read_text()
    chain = [CLOSURE_COMMITS[s] for s in SEEDS]                   # seed tool versions, in closure order
    versions = {c: without_run_ids(committed_tool(repo, c)[1]) for c in chain}
    amendments = [{'id': key, 'from': a, 'to': b, 'note': note,
                   'diff_excluding_run_ids': list(difflib.unified_diff(versions[a], versions[b], a[:7], b[:7], n=0,
                                                                       lineterm=''))}
                  for a, b, key, note in TOOL_AMENDMENTS]
    G('A.seed_tool_current_equals_latest_closure', tool_src.splitlines() == committed_tool(repo, chain[-1])[1]
      .splitlines(), [len(a['diff_excluding_run_ids']) for a in amendments])
    G('A.seed_tool_amendment_chain', [(a['from'], a['to']) for a in amendments] == list(zip(chain, chain[1:]))
      and all(a['diff_excluding_run_ids'] for a in amendments), [a['id'] for a in amendments])
    G('A.seed_tool_run_ids', tool.RUN_IDS == RUN_IDS and tool.AUTHORITY == AUTHORITY and tool.CONFIG_SHA == CONFIG_SHA256
      and all(tool.DERIVED[s] == DERIVED[s] for s in SEEDS))

    # A. per-seed evidence: SHA manifests, deterministic re-derivation, bytes equal to each closure commit
    present = sorted(p.name for p in (repo / 'outputs/audit').glob('E04_SEED*_COMPLETION.json'))
    G('A.exact_seed_evidence_set', present == sorted(f'E04_SEED{s}_COMPLETION.json' for s in SEEDS), present)
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
        G(f'A.seed{s}.equals_closure_commit', all(git_show(repo, CLOSURE_COMMITS[s], p[k]) == (repo / p[k]).read_bytes()
                                                  for k in ('completion', 'manifest', 'report', 'sha256')),
          CLOSURE_COMMITS[s])

    # B. per-seed closure status and identical gate semantics
    for s, d in docs.items():
        G(f'B.seed{s}.closed_clean', d['status'] == f'E04_SEED{s}_COMPLETED_CLEANLY' and d['failed_gates'] == []
          and all(g['pass'] for g in d['gates']) and d['seed'] == s and d['gates_total'] == SEED_GATES, d['status'])
    names = {s: [g['gate'] for g in d['gates']] for s, d in docs.items()}
    G('B.same_gate_semantics', names[42] == names[1337] == names[2026] and len(names[42]) == SEED_GATES,
      {s: len(v) for s, v in names.items()})

    # C. authority / identity (common by design)
    mans = {s: d['collected']['run_manifest'] for s, d in docs.items()}
    starts = {s: d['collected']['run_start'] for s, d in docs.items()}
    for s, d in docs.items():
        m, git, tgit = mans[s], d['collected']['git'], d['collected_tf']['git']
        G(f'C.seed{s}.authority', d['scientific_authority'] == m['git_commit'] == git['head'] == tgit['head'] ==
          AUTHORITY and m['git_dirty'] is False and git['status_porcelain'] == '' and git['branch'] == 'm6-baselines'
          and d['qualification_authority'] == QUALIFICATION_AUTHORITY and d['d12']['ruling_commit'] == D12_RULING_COMMIT
          and d['d12']['acceptance_criteria_sha256'] == D12_CRITERIA_SHA)
        rid = sha256_bytes(f'E04|{s}|{CONFIG_SHA256}|{AUTHORITY}'.encode())[:16]
        G(f'C.seed{s}.run_id_rule', rid == d['run_id'] == m['run_id'] == RUN_IDS[s], rid)
    G('C.run_ids_distinct', len({d['run_id'] for d in docs.values()}) == 3)
    G('C.method', all((d['method']['id'], d['method']['name'], d['method']['fidelity'], d['method']['track'])
                      == METHOD and mans[s]['fidelity'] == FIDELITY for s, d in docs.items()))
    G('C.frozen_config_sha', all(d['config_sha256'] == CONFIG_SHA256 == mans[s]['config_sha256']
                                 == d['collected']['files_sha256']['configs/methods/e04_physics_std.yaml']
                                 == d['collected']['files_sha256']['frozen_config_snapshot/configs/methods/'
                                                                   'e04_physics_std.yaml']
                                 for s, d in docs.items()))
    common = {s: strip_keys(m, MANIFEST_PER_SEED + KERNEL_FIELDS + ('environment',)) for s, m in mans.items()}
    diff_keys = sorted(k for k in set().union(*common.values()) if len({json.dumps(c.get(k), sort_keys=True)
                                                                         for c in common.values()}) > 1)
    env_common = {s: strip_keys(m['environment'], KERNEL_FIELDS) for s, m in mans.items()}
    env_diff = sorted(k for k in env_common[42] if len({json.dumps(e.get(k), sort_keys=True)
                                                         for e in env_common.values()}) > 1)
    G('C.manifest_common_fields_equal', diff_keys == [] and env_diff == [] and common[42].keys() == common[1337].keys()
      == common[2026].keys(), diff_keys + env_diff)
    G('C.kernel_disclosed_not_locked', all(mans[s]['platform'] == mans[s]['environment']['platform'] == KERNELS[s]
                                           == docs[s]['environment']['run_platform'] for s in SEEDS),
      {str(s): KERNELS[s] for s in SEEDS})
    G('C.command_line_differs_only_by_seed', len({m['command_line'].replace(f'--seed {s} ', '--seed <S> ')
                                                  for s, m in mans.items()}) == 1
      and all(f'--seed {s} ' in m['command_line'] and '--resume' not in m['command_line'] for s, m in mans.items()))
    ids = starts[42]['identities']
    G('C.identities_equal', all(r['identities'] == ids == mans[s]['identities'] for s, r in starts.items())
      and ids['config_sha256'] == CONFIG_SHA256 and ids['fidelity'] == FIDELITY, sorted(ids))
    norm = {s: strip_keys(r, RUN_START_PER_SEED) for s, r in starts.items()}
    rs_diff = sorted(k for k in norm[42] if len({json.dumps(r.get(k), sort_keys=True) for r in norm.values()}) > 1)
    seed_norm = {s: strip_keys(r['seeding'], ('hard_sample_seed', 'pair_order_seed', 'python_numpy_global',
                                              'tf_graph_seed')) for s, r in starts.items()}
    G('C.run_start_equal_except_seed', rs_diff == [] and len({json.dumps(v, sort_keys=True)
                                                              for v in seed_norm.values()}) == 1 and all(
        (r['seeding']['pair_order_seed'], r['seeding']['hard_sample_seed']) == DERIVED[s]
        and r['seeding']['tf_graph_seed'] == r['seeding']['python_numpy_global'] == r['_seed'] == s
        for s, r in starts.items()), rs_diff)
    G('C.files_identities_equal', len({json.dumps(d['collected']['files_sha256'], sort_keys=True)
                                       for d in docs.values()}) == 1
      and len({json.dumps(d['collected_tf']['identities_recomputed'], sort_keys=True) for d in docs.values()}) == 1)
    G('C.environment_equal', len({json.dumps(d['collected_tf']['environment_recheck'], sort_keys=True)
                                  for d in docs.values()}) == 1
      and len({json.dumps(strip_keys(d['environment'], ('run_platform',)), sort_keys=True)
               for d in docs.values()}) == 1
      and all(d['collected_tf']['environment_recheck']['differences_other_than_driver'] == [] for d in docs.values()))
    G('C.driver_ruling_consistent', all({k: d['driver_ruling'][k] for k in DRIVER_RULING} == DRIVER_RULING
                                        and starts[s]['gpu_driver']['observed_driver_version']
                                        == DRIVER_RULING['observed_driver_version']
                                        and starts[s]['gpu_driver']['lock_driver_version']
                                        == DRIVER_RULING['lock_driver_version'] for s, d in docs.items())
      and len({json.dumps(r['gpu_driver'], sort_keys=True) for r in starts.values()}) == 1)
    G('C.graph_structure_equal', len({json.dumps(r['structure'], sort_keys=True) for r in starts.values()}) == 1
      and len({json.dumps(d['collected_tf']['structure'], sort_keys=True) for d in docs.values()}) == 1
      and len({json.dumps(r['session_config'], sort_keys=True) for r in starts.values()}) == 1)
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
        # every differing line must mention this seed's run dir, experiment seed or derived seeds, and the 42 line 42's
        keys = (str(s), RUN_IDS[s]) + tuple(str(x) for x in DERIVED[s])
        keys42 = ('42', RUN_IDS[42]) + tuple(str(x) for x in DERIVED[42])
        rc_ok &= all(any(k in lines[i] for k in keys) and any(k in base[i] for k in keys42) for i in changed)
    G('C.resolved_config_differs_only_by_seed', rc_ok, rc_diff)

    # D. training completion / data protocol (identical protocol, seed-derived streams)
    for s, d in docs.items():
        G(f'D.seed{s}.completed_150000_iterations', d['counts'] == COUNTS
          and d['collected']['run_summary']['completion_status'] == 'completed'
          and d['collected']['run_summary']['global_iteration'] == ITERATIONS)
        r = d['rng']
        G(f'D.seed{s}.streams_independently_replayed', (r['pair_order_seed'], r['hard_sample_seed']) == DERIVED[s]
          and all(v == ITERATIONS for v in r['record_by_record_replay'].values())
          and r['offsets_sample']['checked'] == r['offsets_sample']['match'] > 0)
        G(f'D.seed{s}.fresh_run', d['resume'] == 'NOT_QUALIFIED_FRESH_ONLY' and mans[s]['resume'] == d['resume']
          and d['collected']['event_counts'] == {'e04_checkpoint_event': 10, 'e04_run_start': 1}
          and d['confirmations']['resume_used'] is False and d['collected']['resume_files'] == []
          and d['checkpoints']['resume_sidecars'] == 0)
        G(f'D.seed{s}.exit_zero_no_reboot', d['collected']['host']['exit_file_raw'] == '0\n'
          and next(g for g in d['gates'] if g['gate'] == 'B.no_reboot_during_run')['pass']
          and d['host_continuity']['records_inside_window'] == []
          and d['host_continuity']['journal_boots_starting_inside_window'] == [])
    G('D.seed_streams_differ', all(len({d['rng'][k] for d in docs.values()}) == 3
                                   for k in ('pair_sha256_int32_le', 'hard_sha256'))
      and len({d['run_root_integrity']['metrics_jsonl_sha256'] for d in docs.values()}) == 3)
    G('D.family_totals', sum(d['counts']['iterations'] for d in docs.values()) == 450000
      and sum(d['counts']['g_applications'] for d in docs.values()) == 900000
      and sum(d['counts']['d_applications'] for d in docs.values()) == 450000
      and sum(d['counts']['checkpoints'] for d in docs.values()) == 30)

    # E. numerical health
    for s, d in docs.items():
        t = d['collected_tf']
        G(f'E.seed{s}.finite', all(r['nonfinite_variables'] == 0 and r['variables'] == 500 for r in t['checkpoints'])
          and all(v['bytes'] == 0 for v in d['collected']['logs'].values())
          and next(g for g in d['gates'] if g['gate'] == 'F.losses_finite')['pass'])

    # F. checkpoints: iteration_150000 is the final state of each seed, independently
    for s, d in docs.items():
        summ, fs = d['collected']['run_summary'], d['checkpoints']['final_selected']
        G(f'F.seed{s}.final_state', fs['prefix'] == FINAL_PREFIX and fs['rule'] == CHECKPOINT_RULE
          and fs['sha256']['.data-00000-of-00001'] == summ['final_or_selected_checkpoint_sha256']
          == KNOWN_FINAL_DATA_SHA[s] and fs['path'] == f'{RUN_BASE}/seed_{s}/{FINAL_PREFIX}'
          and summ['checkpoint_selection_rule'] == CHECKPOINT_RULE
          and [x['iteration'] for x in d['checkpoints']['files']] == CHECKPOINT_ITERATIONS)
        rows.append(seed_row(s, d))
        inv.extend(inventory_rows(s, d))
    G('F.final_checkpoints_distinct', len({r['final_data_sha256'] for r in rows}) == 3)
    G('F.inventory_30_events_90_files', len(inv) == 90 and len({(r['seed'], r['iteration']) for r in inv}) == 30
      and sum(r['final_selected'] for r in inv) == 9)
    G('F.run_roots_distinct', sorted(d['run_root'] for d in docs.values())
      == sorted(f'{RUN_BASE}/seed_{s}' for s in SEEDS))

    # G. firewall / confirmations
    for s, d in docs.items():
        fw = d['firewall']
        G(f'G.seed{s}.test_0_val_0', (fw['test_access'], fw['test_changed'], fw['test_files']) == (0, 0, 3027)
          and fw['by_split']['TEST']['sensitive_whole_window'] == 3027 and fw['by_split']['TEST']['consumed'] == 0
          and fw['by_split']['VAL']['atime_changed'] == 0 and fw['by_split']['VAL']['consumed'] == 0
          and all(v['changed_not_consumed'] == 0 for v in fw['by_split'].values())
          and d['collected']['pack']['test_rows'] == 0 and d['collected']['pack']['val_rows'] == 0
          and mans[s]['test_split_accessed'] is False and mans[s]['val_split_accessed'] is False
          and all(next(g for g in d['gates'] if g['gate'] == k)['pass'] for k in ('H.test_zero', 'H.val_zero')),
          {'VAL_sensitive': fw['by_split']['VAL']['sensitive_whole_window'], 'VAL_files': fw['by_split']['VAL']['files']})
        G(f'G.seed{s}.no_best_or_downstream', not any(v for k, v in d['confirmations'].items()))

    # H. read-only host recheck at family closure (run roots still immutable, nothing running)
    G('H.host_parent_exactly_three_seeds', [p for p in host['parent_listing'] if not p.startswith('.')]
      == sorted(f'seed_{s}' for s in SEEDS), host['parent_listing'])
    for s, d in docs.items():
        h, ri = hs[str(s)], d['run_root_integrity']
        files = {('checkpoints/iteration_%06d' % r['iteration']) + sf: {'sha256': r['sha256'][sf], 'bytes': r['size'][sf]}
                 for r in d['checkpoints']['files'] for sf in SUFFIXES}
        G(f'H.seed{s}.host_matches_evidence', h['checkpoint_files'] == files
          and h['metrics_jsonl_sha256'] == ri['metrics_jsonl_sha256'] and h['sha_listing_digest']
          == ri['sha_listing_digest'] and h['file_count'] == ri['file_count'] and h['listing_stable_during_recheck']
          and h['exit_marker'] == '0\n' and h['end_marker'].strip() == d['collected']['host']['end_time'])
    st = host['host_state']
    # E04 tmux panes: dead with status 0, or (seed 42) absent because later host reboots after its completion removed
    # the tmux server; its completion was already proven by the durable exit marker + boot continuity (ea533db)
    last_end = max(datetime.strptime(d['end_utc'], '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc).timestamp()
                   for d in docs.values())
    s42_end = datetime.strptime(docs[42]['end_utc'], '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc).timestamp()
    panes = st['e04_tmux_panes']
    panes_ok = all(v == '1 0' or (k == 'e04_physics_std_seed42' and v == '' and k not in st['tmux_sessions']
                                  and st['btime'] > s42_end) for k, v in panes.items())
    G('H.host_final_state', st['hostname'] == 'nd-System-Product-Name'
      and st['gpu'] == 'NVIDIA GeForce RTX 3090, 580.178.04, GPU-b722cd9d-c9fa-223c-915e-cc09d9b862b7'
      and st['method_processes'] == [] and panes_ok and st['btime'] < min(
          datetime.strptime(docs[s]['start_utc'], '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc).timestamp()
          for s in (1337, 2026)) and last_end > 0,
      {'method_processes': st['method_processes'], 'e04_tmux_panes': panes, 'boot_id': st['boot_id'],
       'gpu_compute_apps_unrelated': st['gpu_compute_apps']})
    G('H.gpu_checkout_at_authority_clean', st['git_head'] == AUTHORITY and st['git_branch'] == 'm6-baselines'
      and st['git_status_porcelain'] == '', st['git_head'])

    text = csv_text(rows, CSV_COLUMNS)
    inv_text = csv_text(inv, INVENTORY_COLUMNS)
    repro = reproduction(docs, host)
    lm = {str(s): {'first_block': d['loss_diagnostics']['first_block']['loss_means'],
                   'last_block': d['loss_diagnostics']['last_block']['loss_means']} for s, d in docs.items()}
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
        'totals': {'iterations': 450000, 'g_applications': 900000, 'd_applications': 450000, 'checkpoint_events': 30,
                   'checkpoint_files': 90},
        'scientific_training_authority': AUTHORITY, 'qualification_authority': QUALIFICATION_AUTHORITY,
        'd12': {'ruling_commit': D12_RULING_COMMIT, 'acceptance_criteria_sha256': D12_CRITERIA_SHA,
                'standard': 'procedural reproducibility; GPU training is not bitwise reproducible and no seed is '
                            'compared bitwise to another'},
        'authority_note': 'Training authority for all three seeds is 18ba00e; the seed closure commits and this family '
                          'closure are evidence/reporting only.',
        'frozen': {'config_sha256': CONFIG_SHA256, 'identities': ids,
                   'files_sha256': docs[42]['collected']['files_sha256'], 'structure': starts[42]['structure'],
                   'session_config': starts[42]['session_config'], 'pack': starts[42]['pack']},
        'protocol': {'iterations': ITERATIONS, 'batch': 8, 'pairs_per_iteration': 4,
                     'optimizer_sequence': 'per iteration: G (Step 1, Eq.23) -> D (Step 2, L_D) -> G (Step 3, Eq.24); '
                                           'one shared G Adam',
                     'g_applications': 300000, 'd_applications': 150000,
                     'learning_rate': 'G 5e-5 * 10^-floor(t/45000), D half', 'checkpoint_every': EVERY,
                     'checkpoints': CHECKPOINT_ITERATIONS, 'checkpoint_rule': CHECKPOINT_RULE,
                     'final_checkpoint': FINAL_PREFIX,
                     'final_rule_scope': 'applied independently within each seed; no cross-seed choice',
                     'feed': 'pairs_train_v1 via e04_feed_pack_v1 (TRAIN only); D4/D7 seeded pair/hard streams',
                     'resume': 'NOT_QUALIFIED_FRESH_ONLY'},
        'different_by_design': {'manifest_fields': list(MANIFEST_PER_SEED),
                                'resolved_config_lines': {str(s): v for s, v in rc_diff.items()},
                                'run_start_fields': list(RUN_START_PER_SEED),
                                'disclosed_not_locked': {'host_kernel': {str(s): KERNELS[s] for s in SEEDS}},
                                'also': 'seed-derived pair/hard streams and model initialisation; learned weights and '
                                        'checkpoint bytes'},
        'driver_ruling': dict(DRIVER_RULING, scope='only the NVIDIA driver version differs from the lock; GPU model, '
                                                   'GPU UUID and every other lock field are exact in all three runs'),
        'final_checkpoints': rows, 'final_checkpoints_csv': EVIDENCE['csv'],
        'final_checkpoints_csv_sha256': sha256_bytes(text.encode()),
        'checkpoint_inventory_csv': EVIDENCE['inventory'], 'checkpoint_inventory_csv_sha256': sha256_bytes(
            inv_text.encode()),
        'reproduction_manifest': EVIDENCE['reproduction'],
        'aggregation': {'primary_result_rule': 'MEAN_AND_STD_OVER_ALL_THREE_SEEDS',
                        'applies_to': 'future downstream evaluation metrics (none computed yet)',
                        'descriptive_losses': {
                            'note': 'per-seed first/last 1500-iteration block mean TRAIN losses; descriptive only, NOT '
                                    'a performance metric, not aggregated and not used for any choice',
                            'by_seed': lm},
                        'wall_clock_note': 'training speed is not an acceptance gate and not a comparison metric'},
        'firewall': {'test_access': 0 if not G.failed else None, 'val_atime_changed': 0 if not G.failed else None,
                     'note': 'per-seed stat-only relatime atime snapshots: 3027/3027 TEST and 3121/3121 VAL faces '
                             'atime-sensitive over each run window and unchanged; every changed face was a consumed '
                             'TRAIN pack face'},
        'not_completed': {'synthetic_bank': 'NOT_STARTED', 'downstream_fas_evaluation': 'NOT_STARTED',
                          'whole_benchmark': 'NOT_COMPLETED'},
        'confirmations': {'best_seed_selection': False, 'preferred_seed_selection': False, 'seed_dropped': False,
                          'replacement_seed': False, 'additional_seed_launched': False,
                          'val_checkpoint_selection': False, 'test_used': False, 'bank_generation': False,
                          'downstream_evaluation': False, 'other_method_started': False, 'gpat_started': False,
                          'run_root_mutated': False, 'retraining': False},
        'per_seed_evidence': {str(s): dict(seed_paths(s), tool=SEED_TOOL, closure_commit=CLOSURE_COMMITS[s])
                              for s in SEEDS},
        'seed_tool_amendments': amendments,
        'host_recheck': host,
    }
    return doc, text, inv_text, repro


def report(doc):
    rows = doc['final_checkpoints']
    lm = doc['aggregation']['descriptive_losses']['by_seed']
    keys = ['step1/loss_G_step1', 'step1/L_depth', 'step2/loss_D', 'step3/loss_G_step3']
    out = [
        '# E04 Physics-STD: three-seed scientific training closure', '',
        '**Status:** `%s`, %d/%d gates passing (`E04_THREE_SEED_TRAINING_CLOSURE.json`).'
        % (doc['status'], doc['gates_passed'], doc['gates_total']), '',
        'Training authority for all three seeds: **`%s`** (production qualification `%s`, D12 ruling `%s`, criteria '
        '`%s…`). The per-seed closure commits and this closure are evidence only.'
        % (AUTHORITY, QUALIFICATION_AUTHORITY[:7], D12_RULING_COMMIT[:7], D12_CRITERIA_SHA[:8]), '',
        'Execution fidelity: `%s` (reporting label "Physics-STD (controlled geometry/depth reconstruction)").'
        % FIDELITY, '',
        '| seed | run_id | iterations | G / D | final iteration_150000 data sha256 | run-root listing | TEST | VAL | '
        'closure commit |', '|---|---|---|---|---|---|---|---|---|']
    out += ['| %d | `%s` | %d | %d / %d | `%s` | `%s…` | %d | %d | `%s` |' % (
        r['seed'], r['run_id'], r['final_global_iteration'], r['g_applications'], r['d_applications'],
        r['final_data_sha256'], r['run_root_listing_digest'][:16], r['TEST_access'], r['VAL_atime_changed'],
        r['seed_closure_commit'][:7]) for r in rows]
    t = doc['totals']
    out += ['', '**Family totals:** %d iterations, %d G and %d D optimizer applications, %d checkpoint events (%d '
            'files; inventory `E04_THREE_SEED_CHECKPOINT_INVENTORY.csv`).' % (
                t['iterations'], t['g_applications'], t['d_applications'], t['checkpoint_events'],
                t['checkpoint_files']),
            '', '## What is common and what differs', '',
            '- Common to all three seeds:',
            '  - frozen config `41861794…` and every identity (pre-pass, feed pack, pairs, split, ruling, lock);',
            '  - the qualified graph structure and session configuration;',
            '  - the environment re-check and the D9 driver ruling (580.178.04 vs lock 595.84, version only);',
            '  - 150000 iterations of G → D → G, batch 8 (4 live targets + 4 spoof sources), 10 checkpoints every '
            '15000 iterations.',
            '- Differs by design only:',
            '  - the seed, the run_id, the timestamps and the wrapper command;',
            '  - the derived PAIR_ORDER / HARD_SAMPLE seeds and the TF graph seed;',
            '  - the resulting streams and learned weights.',
            '- Disclosed, not part of the E04 lock: seed 42 ran on host kernel 7.0.0-38, seeds 1337 and 2026 on '
            '7.0.0-43.',
            '- Each seed\'s pair and hard streams equal an independent record-by-record numpy replay.',
            '- iteration_150000 is the final checkpoint of each seed by `BASELINE_FINAL_STATE_V1`. There is no '
            'cross-seed choice and no best seed, and no seed was dropped, replaced or retrained.',
            '- TEST: 0 accesses in every seed. VAL: 0 atime changes in every seed (all VAL faces atime-sensitive).',
            '- D12: the runs are independent GPU executions that are not bitwise reproducible; none is compared '
            'bitwise to another.',
            '', '## Seed-tool amendments (disclosed)', '']
    out += ['%d. `%s` (`%s` → `%s`): %s.' % (i, a['id'], a['from'][:7], a['to'][:7], a['note'])
            for i, a in enumerate(doc['seed_tool_amendments'], 1)]
    out += ['', '## Host recheck', '',
            'Read-only at %s: all 90 checkpoint files, metrics.jsonl and the run-root listing digests equal the per-seed '
            'evidence. The three exit markers are `0`, no E04 or other method process is running, and the GPU checkout '
            'is at `%s`, clean. The seed-42 tmux session is absent because the host rebooted after that run had ended. '
            'The seed-1337 and seed-2026 panes are dead with status 0.' % (doc['host_recheck']['utc'], AUTHORITY[:7]),
            '', '## Descriptive training losses', '',
            'Block means of 1500 iterations, for orientation only. Nothing here is aggregated, ranked or used for a '
            'choice.', '', '| seed | ' + ' | '.join('%s first → last' % k for k in keys) + ' |',
            '|---|' + '---|' * len(keys)]
    out += ['| %s | %s |' % (s, ' | '.join('%.4g → %.4g' % (v['first_block'][k], v['last_block'][k]) for k in keys))
            for s, v in sorted(lm.items(), key=lambda x: int(x[0]))]
    out += ['', '## Reproduction', '',
            '`E04_THREE_SEED_REPRODUCTION_MANIFEST.json` records, per seed, the source commit, config, identities, '
            'environment, RNG derivation, exact launch environment and command, window, exit status, outputs and audit '
            'record. Checkpoint bytes stay in the run roots on the GPU host; git holds their hashes only.',
            '', '## Not completed', '',
            'The E04 synthetic bank, downstream FAS evaluation and the whole benchmark have not started. This closes '
            'Physics-STD training only.', '']
    return '\n'.join(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--host-recheck', required=True)
    ap.add_argument('--repo-root', default=str(Path(__file__).resolve().parents[1]))
    args = ap.parse_args(argv)
    doc, text, inv_text, repro = build(args.repo_root, json.loads(Path(args.host_recheck).read_text()))
    root = Path(args.repo_root)
    (root / EVIDENCE['csv']).write_text(text)
    (root / EVIDENCE['inventory']).write_text(inv_text)
    (root / EVIDENCE['reproduction']).write_text(json.dumps(repro, indent=2, sort_keys=True) + '\n')
    (root / EVIDENCE['closure']).write_text(json.dumps(doc, indent=2, sort_keys=True) + '\n')
    (root / EVIDENCE['report']).write_text(report(doc))
    (root / EVIDENCE['sha256']).write_text(''.join(
        '%s  %s\n' % (sha256_bytes((root / EVIDENCE[k]).read_bytes()), EVIDENCE[k])
        for k in ('closure', 'csv', 'inventory', 'reproduction', 'report')))
    print(doc['status'], doc['gates_passed'], '/', doc['gates_total'], doc['failed_gates'])
    return 0 if doc['status'] == STATUS_OK else 1


if __name__ == '__main__':
    sys.exit(main())
