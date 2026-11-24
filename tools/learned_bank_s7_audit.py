"""M8 learned-bank S7 toy acceptance audit (E03 / E05 / E06b / E06c), per method, under the owner S7 launch authority
configs/amendments/learned_bank_s7_launch_authority.yaml and the plan outputs/audit/LEARNED_BANK_S7_ACCEPTANCE_PLAN.json.

Phases (preflight and collect never write anything but their stdout; run on the GPU host from the clean execution
clone with a Python that has numpy, Pillow and pyarrow -- gpat-m6-e05 -- and CUDA_VISIBLE_DEVICES=""):

  preflight --method M --repo <clone> --s7-root <S7>     resource / authority preflight immediately before run A:
            no E04 or learned-bank process, no live m8s7_* pane, GPU UUID, free GPU memory >= estimate + 2 GiB
            (unrelated processes are recorded, never touched), >= 2 GiB free on the S7 filesystem, fresh per-method
            out root, all 12 approved checkpoints re-hashed, S7 authority loads, method environment present.
  collect   --method M --repo <clone> --s7-root <S7>     after run B: both runs' provenance, independent reader
            verification, file listings, manifest rows (run A), bitwise A/B comparison, checkpoints re-hashed after
            run B, wrapper markers and pane logs, the stat-only atime ledgers mapped through split_v1, GPU state.
  build     --method M --collected <json>              (laptop, repository) gates everything and writes
            outputs/audit/LEARNED_BANK_S7_<M>.{json,md} and LEARNED_BANK_S7_<M>_SHA256.txt. The JSON embeds the collected
            document, so the build re-derives; `s7_acceptance_passed` is what load_production_authority requires.

The S7 outputs are S7_TOY (non-scientific) acceptance banks: never production banks and never benchmark evidence.
Layout note: tools/learned_bank.py s7 creates <out-root>/run_<label>/ exclusively, so every method uses its own out root
<S7>/<M>/ inside the authorized S7 root (<S7>/<M>/run_{A,B}/<M>/seed_<s>/).
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

WORKDIR = '/home/student20261/workdir'
RUNTIME = WORKDIR + '/GPAT_TransferBench_runtime'
GPU_UUID = 'GPU-b722cd9d-c9fa-223c-915e-cc09d9b862b7'
GPU_NAME = 'NVIDIA GeForce RTX 3090'
DRIVER = '580.178.04'
SEEDS = (42, 1337, 2026)
LABELS = ('A', 'B')
TOY_N = 186
ESTIMATE_GIB = {'E06c': 4, 'E06b': 4, 'E05': 3, 'E03': 2}
MARGIN_GIB = 2
DISK_FLOOR_GIB = 2
AUTHORITY_PATH = 'configs/amendments/learned_bank_s7_launch_authority.yaml'
PLAN_PATH = 'outputs/audit/LEARNED_BANK_S7_ACCEPTANCE_PLAN.json'
FAIL_PATTERNS = ('Traceback', 'Error', 'error:', 'out of memory', 'Killed', 'Segmentation fault', 'core dumped',
                 'TrainingForbidden', 'LearnedBank')
FACE_READS = {'E03': 2 * TOY_N, 'E05': 2 * TOY_N, 'E06b': 0, 'E06c': 0}


def utc_now():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def epoch(iso):
    return datetime.strptime(iso.strip(), '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc).timestamp()


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sh(*a, cwd=None):
    r = subprocess.run(list(a), capture_output=True, text=True, cwd=cwd)
    return r.stdout.strip()


def marker_paths(method, label=None, tag=None):
    base = '%s/m8s7_%s' % (WORKDIR, tag or method)
    if label is None:
        return {'preflight': base + '_preflight.json', 'atime_pre': base + '_atime_pre.tsv',
                'atime_post': base + '_atime_post.tsv', 'compare': base + '_compare.json',
                'compare_rc': base + '_compare.rc'}
    b = '%s_%s' % (base, label)
    return {k: '%s.%s' % (b, k) for k in ('command', 'launch_time', 'exit', 'end_time', 'pane.log')}


def lb_module(repo):
    sys.path.insert(0, str(repo))
    os.chdir(str(repo))
    from methods.common import learned_bank as lb
    return lb


def git_state(repo):
    return {'head': sh('git', 'rev-parse', 'HEAD', cwd=repo), 'branch': sh('git', 'branch', '--show-current', cwd=repo),
            'status_porcelain': sh('git', 'status', '--porcelain', cwd=repo)}


def gpu_state():
    q = sh('nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total,memory.used,memory.free',
           '--format=csv,noheader,nounits').split(', ')
    apps = []
    for line in sh('nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                   '--format=csv,noheader,nounits').splitlines():
        pid, name, mem = [x.strip() for x in line.split(',')]
        apps.append({'pid': int(pid), 'name': name, 'used_mib': int(mem),
                     'ps': sh('ps', '-o', 'user=,etime=,args=', '-p', pid)[:200]})
    return {'name': q[0], 'uuid': q[1], 'driver': q[2], 'total_mib': int(q[3]), 'used_mib': int(q[4]),
            'free_mib': int(q[5]), 'compute_apps': apps}


def processes():
    me = str(os.getpid())
    out = []
    for line in sh('ps', '-eo', 'pid,args').splitlines()[1:]:
        pid, _, args = line.strip().partition(' ')
        if pid != me and ('tools/run_e04.py' in args or 'tools/learned_bank.py' in args or 'run_e0' in args):
            out.append(line.strip()[:200])
    return out


TAG = None                       # attempt tag (e.g. E05_a2): marker/tmux prefix m8s7_<tag>; None = m8s7_<method>
RULING_PATH = 'configs/amendments/e05_source_relocation_ruling.yaml'
E05_FAILURE = 'outputs/audit/LEARNED_BANK_S7_E05_ATTEMPT1_FAILURE.json'


def m8s7_panes():
    out = {}
    for s in sh('tmux', 'ls', '-F', '#{session_name}').splitlines():
        if s.startswith('m8s7_'):
            out[s] = sh('tmux', 'list-panes', '-t', s, '-F', '#{pane_dead} #{pane_dead_status}')
    return out


# ------------------------------------------------------------------------------------------------------- preflight
def preflight(method, repo, s7_root):
    lb = lb_module(repo)
    out_root = Path(s7_root) / method
    doc = {'phase': 'preflight', 'method': method, 'utc': utc_now(), 'repo': str(repo), 'git': git_state(repo),
           **({'tag': TAG} if TAG else {}),
           'collector': {'python': sys.version.split()[0], 'executable': sys.executable},
           'processes': processes(), 'm8s7_panes': m8s7_panes(), 'gpu': gpu_state(),
           'estimate_gib': ESTIMATE_GIB[method], 'margin_gib': MARGIN_GIB}
    du = shutil.disk_usage(str(Path(s7_root).parent))
    doc['disk'] = {'path': str(Path(s7_root).parent), 'free_bytes': du.free, 'floor_bytes': DISK_FLOOR_GIB << 30}
    doc['s7_root'] = {'path': str(s7_root), 'exists': Path(s7_root).is_dir(),
                      'entries': sorted(os.listdir(s7_root)) if Path(s7_root).is_dir() else None,
                      'method_out_root': str(out_root), 'method_out_root_exists': out_root.exists(),
                      'within_runtime_root': str(Path(s7_root).resolve()).startswith(RUNTIME),
                      'within_hdd_bank_root': str(Path(s7_root).resolve()).startswith('/media/')}
    ck = {}
    for (m, s) in sorted(lb.CHECKPOINTS):
        try:
            r = lb.verify_checkpoint(m, s, Path(RUNTIME) / 'runs' / 'm6' / m / ('seed_%d' % s))
            ck['%s|%d' % (m, s)] = {'ok': True, 'identity': r['checkpoint_identity_sha256'],
                                     'files': {k: v['sha256'] for k, v in r['files'].items()}}
        except Exception as exc:                 # recorded, gated in build
            ck['%s|%d' % (m, s)] = {'ok': False, 'error': '%s: %s' % (type(exc).__name__, exc)}
    doc['checkpoints'] = ck
    try:
        a = lb.load_s7_authority(method)
        doc['authority'] = {'ok': True, 'record': a.record()}
    except Exception as exc:
        doc['authority'] = {'ok': False, 'error': '%s: %s' % (type(exc).__name__, exc)}
    env = lb.ENVIRONMENTS[method]
    probe = ('import sys; import importlib; m = "tensorflow" if %r == "E03" else "torch"; t = importlib.import_module(m);'
             ' print(sys.version.split()[0], m, t.__version__)' % method)
    r = subprocess.run(['env', '-i', 'HOME=' + os.environ.get('HOME', ''), 'PATH=/usr/bin:/bin', 'CUDA_VISIBLE_DEVICES=',
                        'PYTHONNOUSERSITE=1', 'TF_CPP_MIN_LOG_LEVEL=3', env['python'], '-I', '-B', '-c', probe],
                       capture_output=True, text=True)
    doc['environment'] = {'name': env['name'], 'python': env['python'], 'present': Path(env['python']).is_file(),
                          'probe_rc': r.returncode, 'probe': r.stdout.strip()}
    doc['hdd_storage_preflight'] = lb.storage_preflight()
    print(json.dumps(doc, sort_keys=True, default=str))


# ------------------------------------------------------------------------------------------------------- collect
def listing(root):
    out = []
    for p in sorted(Path(root).rglob('*')):
        st = p.lstat()
        rel = str(p.relative_to(root))
        if p.is_symlink():
            out.append([rel, 'symlink', None, None])
        elif p.is_dir():
            out.append([rel, 'dir', None, oct(st.st_mode & 0o7777)])
        else:
            out.append([rel, st.st_size, sha256_file(p), oct(st.st_mode & 0o7777)])
    return out


def pane_summary(path):
    p = Path(path)
    if not p.is_file():
        return None
    lines = p.read_bytes().decode('utf-8', errors='replace').replace('\r', '\n').splitlines()
    result = None
    for x in lines:
        x = x.strip()
        if x.startswith('{') and '"command"' in x:
            try:
                d = json.loads(x)
            except ValueError:
                continue
            if d.get('command') == 's7':
                result = d
    return {'lines': len(lines), 'sha256': sha256_file(p), 's7_result': result,
            'fail_hits': {k: sum(k in x for x in lines) for k in FAIL_PATTERNS},
            'tail': [x for x in lines if x.strip()][-3:]}


def read_text(path):
    p = Path(path)
    return p.read_text() if p.is_file() else None


def atime_ledger(method, repo):
    import pyarrow.parquet as pq
    p = marker_paths(method, tag=TAG)

    def load(f):
        out = {}
        with open(f) as h:
            for line in h:
                a, m, path = line.rstrip('\n').split('\t')
                out[path] = (a, m)
        return out
    if not (Path(p['atime_pre']).is_file() and Path(p['atime_post']).is_file()):
        return None
    pre, post = load(p['atime_pre']), load(p['atime_post'])
    split = {r['sample_id']: r['split'] for r in pq.read_table(str(Path(repo) / 'manifests/split_v1.parquet'),
                                                               columns=['sample_id', 'split']).to_pylist()}
    by = {}
    for path, (a, m) in pre.items():
        sid = os.path.basename(path)[:-4]
        sp = split.get(sid, 'UNMAPPED')
        s = by.setdefault(sp, {'files': 0, 'atime_changed': 0, 'mtime_changed': 0, 'missing_post': 0})
        s['files'] += 1
        if path not in post:
            s['missing_post'] += 1
            continue
        s['atime_changed'] += post[path][0] != a
        s['mtime_changed'] += post[path][1] != m
    return {'pre_utc': read_text(p['atime_pre'][:-4] + '.time').strip(),
            'post_utc': read_text(p['atime_post'][:-4] + '.time').strip(),
            'pre_sha256': sha256_file(p['atime_pre']), 'post_sha256': sha256_file(p['atime_post']),
            'files_pre': len(pre), 'files_post': len(post), 'same_paths': sorted(pre) == sorted(post), 'by_split': by}


def collect(method, repo, s7_root):
    lb = lb_module(repo)
    out_root = Path(s7_root) / method
    doc = {'phase': 'collect', 'method': method, 'utc': utc_now(), 'repo': str(repo), 'git': git_state(repo),
           'collector': {'python': sys.version.split()[0], 'executable': sys.executable},
           'out_root': str(out_root), 'out_root_entries': sorted(os.listdir(out_root)), 'runs': {}}
    if TAG:
        doc['tag'] = TAG
    before = {L: listing(out_root / ('run_' + L)) for L in LABELS}
    for L in LABELS:
        doc['runs'][L] = {}
        for s in SEEDS:
            d = out_root / ('run_' + L) / method / ('seed_%d' % s)
            prov_p = d / lb.PROVENANCE
            rb = lb.read_bank(d, expect_method=method, expect_seed=s, expect_mode='S7_TOY', expect_rows=TOY_N,
                              expect_checkpoint_identity=lb.checkpoint_identity_sha256(method, s))
            rows = rb.pop('rows')
            lst = listing(d)
            rec = {'dir': str(d), 'provenance': json.loads(prov_p.read_text()) if prov_p.is_file() else None,
                   'aborted': (d / ('ABORTED_' + lb.PROVENANCE)).exists(), 'read_bank': rb,
                   'listing': lst, 'listing_sha256': sha256_bytes(json.dumps(lst).encode()),
                   'dir_mode': oct(d.stat().st_mode & 0o7777), 'bank_dir_mode': oct((d / lb.BANK_DIR).stat().st_mode
                                                                                     & 0o7777)}
            if L == 'A':
                rec['manifest_fields'] = list(lb.MANIFEST_FIELDS)
                rec['manifest_rows'] = [[r[k] for k in lb.MANIFEST_FIELDS] for r in rows]
            doc['runs'][L][str(s)] = rec
    doc['compare'] = {str(s): lb.compare_runs(out_root / 'run_A' / method / ('seed_%d' % s),
                                              out_root / 'run_B' / method / ('seed_%d' % s)) for s in SEEDS}
    after = {}
    for s in SEEDS:
        try:
            r = lb.verify_checkpoint(method, s, Path(RUNTIME) / 'runs' / 'm6' / method / ('seed_%d' % s))
            after[str(s)] = {'ok': True, 'identity': r['checkpoint_identity_sha256'],
                             'files': {k: {'sha256': v['sha256'], 'size': v['size'], 'mtime_ns': v['mtime_ns']}
                                       for k, v in r['files'].items()}}
        except Exception as exc:
            after[str(s)] = {'ok': False, 'error': '%s: %s' % (type(exc).__name__, exc)}
    doc['checkpoints_after_run_B'] = after
    doc['markers'] = {}
    for L in LABELS:
        mp = marker_paths(method, L, TAG)
        doc['markers'][L] = {'command': read_text(mp['command']),
                             'command_sha256': sha256_file(mp['command']) if Path(mp['command']).is_file() else None,
                             'launch_time': (read_text(mp['launch_time']) or '').strip(),
                             'exit_raw': read_text(mp['exit']), 'end_time': (read_text(mp['end_time']) or '').strip(),
                             'pane': pane_summary(mp['pane.log']),
                             'tmux_pane': sh('tmux', 'list-panes', '-t', 'm8s7_%s_%s' % (TAG or method, L), '-F',
                                             '#{pane_dead} #{pane_dead_status}')}
    pf = marker_paths(method, tag=TAG)['preflight']
    doc['preflight'] = json.loads(read_text(pf)) if Path(pf).is_file() else None
    doc['preflight_sha256'] = sha256_file(pf) if Path(pf).is_file() else None
    mp = marker_paths(method, tag=TAG)
    doc['cli_compare'] = {'stdout': read_text(mp['compare']), 'rc': read_text(mp['compare_rc'])}
    doc['atime'] = atime_ledger(method, repo)
    gs = '%s/m8s7_%s_A.gpu_samples.tsv' % (WORKDIR, TAG or method)
    doc['gpu_samples_run_A'] = read_text(gs).splitlines() if Path(gs).is_file() else None
    doc['processes_after'] = processes()
    doc['gpu_after'] = gpu_state()
    doc['run_roots_unchanged_during_collect'] = all(listing(out_root / ('run_' + L)) == before[L] for L in LABELS)
    print(json.dumps(doc, sort_keys=True, default=str))


# ------------------------------------------------------------------------------------------------------- build
class Gates:
    def __init__(self):
        self.results = []

    def __call__(self, name, ok, detail=None):
        self.results.append({'gate': name, 'pass': bool(ok), **({'detail': detail} if detail is not None else {})})
        return ok

    @property
    def failed(self):
        return [r['gate'] for r in self.results if not r['pass']]


def evidence_paths(method):
    a = 'outputs/audit/LEARNED_BANK_S7_%s' % method
    return {'record': a + '.json', 'report': a + '.md', 'sha256': a + '_SHA256.txt'}


def build(c, repo_root):
    sys.path.insert(0, str(repo_root))
    from methods.common import learned_bank as lb
    repo = Path(repo_root)
    M = c['method']
    G = Gates()
    auth_raw = (repo / AUTHORITY_PATH).read_bytes()
    auth = json.loads(auth_raw)
    auth_sha = sha256_bytes(auth_raw)
    contract_sha = sha256_bytes((repo / lb.CONTRACT_PATH).read_bytes())
    e04_sha = sha256_bytes((repo / lb.E04_COMPLETION_RECORD).read_bytes())
    plan = json.loads((repo / PLAN_PATH).read_text())
    commit = c['git']['head']
    pf = c['preflight'] or {}
    toy = lb.load_toy_manifest(repo)
    sources = json.loads((repo / lb.SOURCES_PATH).read_text())['families']
    env = lb.ENVIRONMENTS[M]

    # P. authority, execution checkout, preflight
    G('P.authority_covers_method', auth['status'] == 'OWNER_APPROVED' and auth['approved'] is True and M in auth['methods']
      and auth['seeds'] == list(SEEDS) and auth['contract_sha256'] == contract_sha
      and auth['e04_seed_2026_completion_sha256'] == e04_sha and auth['authorizes']['production_bank_generation'] is False
      and auth['resource_preflight']['required'] is True)
    G('P.execution_checkout', commit == pf.get('git', {}).get('head') and c['git']['status_porcelain'] == ''
      and pf.get('git', {}).get('status_porcelain') == '' and c['git']['branch'] == 'm6-baselines'
      and subprocess.run(['git', 'merge-base', '--is-ancestor', commit, 'HEAD'], cwd=repo).returncode == 0
      and subprocess.run(['git', 'show', '%s:%s' % (commit, AUTHORITY_PATH)], cwd=repo,
                         capture_output=True).stdout == auth_raw, commit)
    est = ESTIMATE_GIB[M] + MARGIN_GIB
    g = pf.get('gpu', {})
    G('P.preflight_resources', pf.get('processes') == [] and all(v == '1 0' for v in pf.get('m8s7_panes', {}).values())
      and g.get('uuid') == GPU_UUID and g.get('name') == GPU_NAME and g.get('free_mib', 0) >= est * 1024
      and pf.get('disk', {}).get('free_bytes', 0) >= DISK_FLOOR_GIB << 30,
      {'gpu_free_mib': g.get('free_mib'), 'required_mib': est * 1024, 'unrelated_gpu_processes': g.get('compute_apps'),
       'disk_free_gib': round(pf.get('disk', {}).get('free_bytes', 0) / 2 ** 30, 1)})
    G('P.preflight_fresh_roots', pf.get('s7_root', {}).get('method_out_root_exists') is False
      and pf['s7_root']['within_runtime_root'] is False and pf['s7_root']['within_hdd_bank_root'] is False
      and pf.get('utc', '9') <= c['markers']['A']['launch_time'])
    G('P.preflight_12_checkpoints', len(pf.get('checkpoints', {})) == 12 and all(
        v['ok'] and v['identity'] == lb.checkpoint_identity_sha256(k.split('|')[0], int(k.split('|')[1]))
        for k, v in pf['checkpoints'].items()))
    G('P.preflight_authority_loads', pf.get('authority', {}).get('ok') is True
      and pf['authority']['record']['records'].get(AUTHORITY_PATH) == auth_sha)
    G('P.preflight_environment', pf.get('environment', {}).get('present') is True
      and pf['environment']['probe_rc'] == 0 and pf['environment']['python'] == env['python'], pf.get('environment'))

    # R. run lifecycle: A then B, separate processes, exit 0
    mk = c['markers']
    for L in LABELS:
        m = mk[L]
        res = (m['pane'] or {}).get('s7_result') or {}
        G('R.run%s.exit_zero' % L, m['exit_raw'] == '0\n' and m['tmux_pane'] in ('1 0', ''), m['exit_raw'])
        G('R.run%s.command' % L, m['command'] is not None and (
            'tools/learned_bank.py s7 --method %s --label %s --out-root %s --runtime-root %s'
            % (M, L, c['out_root'], RUNTIME)) in m['command'] and env['python'] in m['command']
          and ('cd %s ' % c['repo']) in m['command'] and 'env -i ' in m['command']
          and 'CUDA_VISIBLE_DEVICES=0' in m['command'])
        G('R.run%s.result_line' % L, res.get('label') == L and res.get('method') == M and all(
            res['results'][str(s)]['status'] == 'completed' and res['results'][str(s)]['rows'] == TOY_N
            and res['results'][str(s)]['outputs'] == TOY_N and res['results'][str(s)]['failure_count'] == 0
            and res['results'][str(s)]['checkpoint_unchanged_after'] is True for s in SEEDS)
          and not any(m['pane']['fail_hits'].values()), (m['pane'] or {}).get('fail_hits'))
    G('R.B_after_A', epoch(mk['A']['launch_time']) <= epoch(mk['A']['end_time']) <= epoch(mk['B']['launch_time'])
      <= epoch(mk['B']['end_time']), [mk[L][k] for L in LABELS for k in ('launch_time', 'end_time')])
    G('R.out_root_exactly_two_runs', c['out_root_entries'] == ['run_A', 'run_B']
      and c['out_root'] == '%s/%s' % (pf.get('s7_root', {}).get('path'), M))
    G('R.no_process_left', c['processes_after'] == [])

    # S. per seed and run
    rows_by_seed = {}
    for s in SEEDS:
        auth_ck = lb.CHECKPOINTS[(M, s)]
        ident = lb.checkpoint_identity_sha256(M, s)
        for L in LABELS:
            r = c['runs'][L][str(s)]
            p = r['provenance'] or {}
            a = p.get('adapter', {})
            pre = 'S.seed%d.run%s.' % (s, L)
            G(pre + 'completed', p.get('status') == 'completed' and not r['aborted'] and p.get('rows') == TOY_N
              and p.get('outputs') == TOY_N and p.get('failure_count') == 0 and p.get('failures') == []
              and p.get('mode') == 'S7_TOY' and p.get('method_id') == M and p.get('bank_seed') == s
              and p.get('post_write_verification', {}).get('ok') is True, p.get('status'))
            ar = p.get('authorization', {})
            G(pre + 'authorization', ar.get('mode') == 'S7_TOY' and ar.get('methods') == [M]
              and ar.get('seeds') == list(SEEDS) and ar.get('contract_sha256') == contract_sha
              and ar.get('records') == {lb.CONTRACT_PATH: contract_sha, AUTHORITY_PATH: auth_sha,
                                        lb.E04_COMPLETION_RECORD: e04_sha})
            ck = a.get('checkpoint', {})
            G(pre + 'checkpoint_before', a.get('is_fixture') is False and a.get('checkpoint_identity_sha256') == ident
              and ck.get('checkpoint_identity_sha256') == ident and ck.get('run_id') == auth_ck['run_id']
              and {k: v['sha256'] for k, v in ck.get('files', {}).items()} == auth_ck['sha256']
              and a.get('generator_seed') == a.get('bank_seed') == s)
            G(pre + 'config_contract', a.get('config_sha256') == lb.METHOD_CONFIGS[M][1]
              and a.get('contract_sha256') == contract_sha)
            src = a.get('source', {})
            fam = sources[{'E03': 'STDN', 'E05': 'PCGAN'}.get(M, 'DSDG')]
            src_ok = src.get('files') == fam['files'] and src.get('commit') == fam['commit'] \
                and src.get('root') == '%s/%s' % (c['repo'], fam['root'])
            G(pre + 'pinned_source', src_ok)
            if M == 'E03':
                gr, cov = a.get('graph', {}), a.get('coverage', {})
                graph_ok = (gr.get('training_nn') is False and gr.get('update_ops') == 0 and gr.get('ema_variables') == 0
                            and gr.get('mutating_reachable_ops') in (0, []) and gr.get('random_reachable_ops') in (0, [])
                            and gr.get('batch_size') == 1 and cov.get('graph_variables_covered') == gr.get('global_variables')
                            and cov.get('ema_restored') == 0)
                guard_ok = p.get('adapter_finalize', {}).get('forwards') == TOY_N
                gdetail = {'graph': gr, 'coverage': cov}
            else:
                gv = a.get('training_guards', {})
                guard_ok = gv == {'optimizer_constructions': 0, 'optimizer_steps': 0, 'backward_passes': 0,
                                  'torch_saves': 0}
                if M == 'E05':
                    graph_ok = bool(a.get('runtime_closure', {}).get('build_closure')) and bool(a.get('precision')) \
                        and a.get('tensor2im_segment_sha256') is not None \
                        and p.get('adapter_finalize', {}).get('generator_forwards') == TOY_N \
                        and p['adapter_finalize'].get('encoder_forwards') == 2 * TOY_N \
                        and p['adapter_finalize'].get('reconstruction_forwards') == 0
                else:
                    fin = p.get('adapter_finalize', {})
                    graph_ok = bool(a.get('load')) and bool(a.get('determinism')) and fin.get('batches_run') == 2 \
                        and fin.get('forward_calls') == 2 and fin.get('discarded_final_batch_samples') == 14 \
                        and fin.get('stream_seed') == lb.dsdg_bank_seed(M, s) and fin.get('live_half_stored') is False
                gdetail = {k: a.get(k) for k in ('netG_parameters', 'device', 'framework')}
            G(pre + 'inference_graph', graph_ok, gdetail)
            G(pre + 'no_training', guard_ok)
            rd = p.get('reader')
            if FACE_READS[M]:
                rd_ok = rd is not None and rd['reads'] == rd['atime_checked_reads'] == FACE_READS[M] \
                    and rd['splits_allowlisted'] == ['TRAIN'] and rd['o_noatime'] is True \
                    and rd['faces_root'] == RUNTIME + '/data/processed/faces_256'
            else:
                rd_ok = rd is None
            G(pre + 'train_only_reader', rd_ok, rd)
            rb = r['read_bank']
            G(pre + 'reader_verification', rb['ok'] and rb['problems'] == [] and rb['files_checked'] == TOY_N
              and rb['failures'] == 0 and rb['provenance_ok'] is True)
            pngs = [x for x in r['listing'] if x[0].startswith('bank/') and x[0].endswith('.png')]
            G(pre + 'outputs_immutable', len(pngs) == TOY_N and all(x[3] == '0o444' for x in pngs)
              and r['dir_mode'] == '0o555' and r['bank_dir_mode'] == '0o555'
              and sorted(x[0] for x in r['listing'] if x[1] != 'dir') == sorted(
                  ['bank_manifest.jsonl', 'bank_manifest.sha256', 'bank_provenance.json'] + [x[0] for x in pngs]))
            G(pre + 'environment', p.get('environment', {}).get('python_executable') == env['python']
              and p['environment'].get('cuda_visible_devices') == '0' and p.get('git_head') == commit
              and (M == 'E03' or a.get('device') == 'cuda'), p.get('environment'))
        # population (run A manifest rows; B's manifest is byte-equal by the replay gate)
        ra = c['runs']['A'][str(s)]
        rows = [dict(zip(ra['manifest_fields'], x)) for x in ra['manifest_rows']]
        rows_by_seed[s] = rows
        pre = 'S.seed%d.' % s
        if M in lb.PAIR_METHODS:
            want = [(t['position'], t['pair_id'], t['target_live_id'], t['source_spoof_id'], t['dataset']) for t in toy]
            got = [(r['row_index'], r['pair_id'], r['target_live_id'], r['source_spoof_id'], r['dataset']) for r in rows]
            pop_ok = got == want and all(r['synthetic_id'] is None for r in rows)
            if M == 'E05':
                pop_ok = pop_ok and all(r['deterministic_noise_seed'] == lb.e05_pair_seed(s, r['pair_id']) for r in rows)
            else:
                pop_ok = pop_ok and all(r['deterministic_noise_seed'] is None for r in rows)
            pop_ok = pop_ok and all(r['target_face_sha256'] and r['source_face_sha256'] for r in rows)
        else:
            pop_ok = [r['row_index'] for r in rows] == list(range(TOY_N)) and all(
                r['row_id'] == r['synthetic_id'] == lb.synthetic_id(M, s, r['row_index']) and r['pair_id'] is None
                and r['target_live_id'] is None and r['source_spoof_id'] is None and r['dataset'] is None
                and r['deterministic_noise_seed'] == lb.dsdg_bank_seed(M, s)
                and (r['dsdg_batch_index'], r['dsdg_batch_offset']) == lb.dsdg_location(r['row_index'])
                for r in rows)
        G(pre + 'population', pop_ok)
        G(pre + 'rows_finite_rgb_uint8', len(rows) == TOY_N and all(
            r['generation_status'] == 'success' and r['raw_output_finite'] is True and r['mode'] == 'S7_TOY'
            and r['label'] == 'spoof' and r['conversion_rule'] == lb.CONVERSION_RULES[M]
            and r['checkpoint_identity_sha256'] == ident and r['generator_seed'] == r['bank_seed'] == s
            and r['output_path'] == 'bank/%s.png' % r['row_id'] for r in rows))
        cmp_ = c['compare'][str(s)]
        lA, lB = c['runs']['A'][str(s)], c['runs']['B'][str(s)]
        G(pre + 'bitwise_replay_A_B', cmp_['bitwise_identical'] is True and cmp_['manifest_equal'] is True
          and cmp_['file_set_equal'] is True and cmp_['files_compared'] == TOY_N and cmp_['files_differing'] == []
          and lA['read_bank']['manifest_sha256'] == lB['read_bank']['manifest_sha256']
          and [x for x in lA['listing'] if x[0].startswith('bank/')] == [x for x in lB['listing'] if x[0].startswith('bank/')],
          {'manifest_sha256': lA['read_bank']['manifest_sha256'], 'files_compared': cmp_['files_compared']})
        af = c['checkpoints_after_run_B'][str(s)]
        G(pre + 'checkpoint_after_run_B', af['ok'] and af['identity'] == ident and all(
            af['files'][k]['sha256'] == auth_ck['sha256'][k] for k in auth_ck['sha256']) and all(
            c['runs'][L][str(s)]['provenance']['adapter']['checkpoint']['files'][k]['mtime_ns']
            == af['files'][k]['mtime_ns'] for L in LABELS for k in auth_ck['sha256']))

    if M == 'E05':
        rul_raw = (repo / RULING_PATH).read_bytes()
        rul = json.loads(rul_raw)
        fail_raw = (repo / E05_FAILURE).read_bytes()
        G('P.e05_o2_ruling_in_execution_commit', rul['status'] == 'OWNER_APPROVED' and rul['approved'] is True
          and rul['decision'] == 'O2_PATH_INDEPENDENT_SOURCE_IDENTITY' and rul['relocatable_fields'] == ['root']
          and subprocess.run(['git', 'show', '%s:%s' % (commit, RULING_PATH)], cwd=repo,
                             capture_output=True).stdout == rul_raw
          and rul['resolves']['failure_record_sha256'] == sha256_bytes(fail_raw) and c.get('tag') == 'E05_a2'
          and c['out_root'] != json.loads(fail_raw)['raw']['out_root'])
        src_root = '%s/third_party/source_cache/swapping_autoencoder' % c['repo']
        for s in SEEDS:
            for L in LABELS:
                bc = c['runs'][L][str(s)]['provenance']['adapter'].get('runtime_closure', {}).get('build_closure', {})
                si, imp = bc.get('source_identity') or {}, bc.get('import_resolution') or {}
                G('S.seed%d.run%s.e05_source_identity' % (s, L), si.get('relocated') is True and si.get('root') == src_root
                  and si.get('differing_fields') == ['root'] and si.get('historical_root') == rul['historical_root']
                  and si.get('commit') == '6baa180f1184ee79a6b967f9d80ee0e02a979ac7'
                  and si.get('tree') == '25a434f1282a0177d6afdcc68ebfcf40bd805621'
                  and si.get('files_verified') == 23 and si.get('code_files_verified') == 67
                  and si.get('unexpected_executable_files') == 0
                  and si.get('ruling') == {'path': RULING_PATH, 'sha256': sha256_bytes(rul_raw)}
                  and len(imp) == 7 and bc.get('jit_build_invoked') is False
                  and len(bc.get('extensions', {})) == 2, {'source_identity': si, 'imports': imp})

    cc = c.get('cli_compare') or {}
    try:
        cli = json.loads((cc.get('stdout') or '').strip().splitlines()[-1])
    except (ValueError, IndexError):
        cli = {}
    G('S.cli_s7_compare', cc.get('rc') == '0\n' and cli.get('command') == 's7-compare' and cli.get('method') == M
      and all(cli['seeds'][str(s)]['replay']['bitwise_identical'] is True and cli['seeds'][str(s)]['failures'] == [0, 0]
              and cli['seeds'][str(s)]['run_A_reader_ok'] and cli['seeds'][str(s)]['run_B_reader_ok'] for s in SEEDS),
      cc.get('rc'))

    # F. firewall
    at = c['atime']
    if at is None:
        G('F.atime_ledger', False, 'missing')
    else:
        by = at['by_split']
        G('F.atime_ledger', at['same_paths'] and at['files_pre'] == at['files_post'] > 0
          and at['pre_utc'] <= c['markers']['A']['launch_time'] and at['post_utc'] >= c['markers']['B']['end_time']
          and all(v['atime_changed'] == 0 and v['mtime_changed'] == 0 and v['missing_post'] == 0 for v in by.values())
          and {'TRAIN', 'VAL', 'TEST'} <= set(by), by)
    G('F.no_val_test_inputs', all(
        p['provenance']['population'].get('pairs_val_opened', False) is False for L in LABELS
        for p in c['runs'][L].values()) and (M not in lb.PAIR_METHODS or all(
            r['target_live_id'] in {t['target_live_id'] for t in toy} for rows in rows_by_seed.values() for r in rows)))
    G('F.collect_did_not_change_roots', c['run_roots_unchanged_during_collect'] is True)

    failed = G.failed
    status = 'S7_ACCEPTANCE_PASSED' if not failed else 'S7_FAILED_NEEDS_OWNER_REVIEW'
    peak = {L: {s: ((c['markers'][L]['pane'] or {}).get('s7_result') or {}).get('results', {}).get(str(s), {}).get(
        'peak_cuda_memory_allocated_bytes') for s in SEEDS} for L in LABELS}
    doc = {
        'schema': 'learned_bank_s7_acceptance_record_v1', 'method': M, 'seeds': list(SEEDS), 'status': status,
        's7_acceptance_passed': not failed, 'failed_gates': failed, 'gates': G.results,
        'gates_passed': len(G.results) - len(failed), 'gates_total': len(G.results),
        'label': 'S7_TOY (S7_NON_SCIENTIFIC toy acceptance; never a production bank or benchmark evidence)',
        'authority': {'path': AUTHORITY_PATH, 'sha256': auth_sha}, 'contract_sha256': contract_sha,
        'e04_seed_2026_completion_sha256': e04_sha, 'execution_commit': commit,
        'checkpoints': {str(s): {'run_id': lb.CHECKPOINTS[(M, s)]['run_id'], 'sha256': lb.CHECKPOINTS[(M, s)]['sha256'],
                                 'checkpoint_identity_sha256': lb.checkpoint_identity_sha256(M, s),
                                 'unchanged_before_and_after': not any(g for g in failed if 'checkpoint' in g)}
                        for s in SEEDS},
        'outputs': {str(s): {'rows': TOY_N, 'manifest_sha256': c['runs']['A'][str(s)]['read_bank']['manifest_sha256'],
                             'bank_listing_sha256': sha256_bytes(json.dumps(
                                 [x for x in c['runs']['A'][str(s)]['listing'] if x[0].startswith('bank/')]).encode()),
                             'run_A_dir': c['runs']['A'][str(s)]['dir'], 'run_B_dir': c['runs']['B'][str(s)]['dir'],
                             'bitwise_identical': c['compare'][str(s)]['bitwise_identical']} for s in SEEDS},
        'profiling': {'peak_cuda_memory_allocated_bytes': peak,
                      **({'nvidia_smi_samples_run_A': c['gpu_samples_run_A']} if c.get('gpu_samples_run_A') else {}),
                      'note': 'torch.cuda.max_memory_allocated per process after each seed (cumulative within the run); '
                              'E03 (TF1) reports none'},
        'reproduction': {
            'method_id': M, 'seeds': list(SEEDS), 'execution_commit': commit, 'branch': 'm6-baselines',
            'authority': AUTHORITY_PATH, 'config': {'path': lb.METHOD_CONFIGS[M][0], 'sha256': lb.METHOD_CONFIGS[M][1]},
            'environment': {'name': env['name'], 'python': env['python'],
                            'lock': {'path': env['lock'][0], 'sha256': env['lock'][1]},
                            'probe': pf.get('environment', {}).get('probe'),
                            'run_environment': c['runs']['A']['42']['provenance']['environment'],
                            'framework': c['runs']['A']['42']['provenance']['adapter'].get('framework')},
            'gpu': {'uuid': pf.get('gpu', {}).get('uuid'), 'name': pf.get('gpu', {}).get('name'),
                    'driver': pf.get('gpu', {}).get('driver')},
            'population': plan['methods'][M]['population'],
            'rng': ('E05: torch.manual_seed(e05_pair_seed(seed, pair_id)) before each mix forward' if M == 'E05' else
                    'E03: deterministic (no sampling at inference)' if M == 'E03' else
                    'DSDG: torch.Generator().manual_seed(dsdg_bank_seed(method, seed)); per batch of 100 randn noise, '
                    'randn noise_s, z = cat(noise_s, noise, noise); 2 batches, 14 discarded'),
            'conversion_rule': lb.CONVERSION_RULES[M],
            'working_directory': c['repo'],
            'commands': {L: c['markers'][L]['command'] for L in LABELS},
            'compare_command': '%s -B tools/learned_bank.py s7-compare --method %s --out-root %s'
                               % (env['python'], M, c['out_root']),
            'windows': {L: [c['markers'][L]['launch_time'], c['markers'][L]['end_time']] for L in LABELS},
            'exit_status': {L: c['markers'][L]['exit_raw'] for L in LABELS},
            'output_root': c['out_root'],
            'atime_ledger': {k: (at or {}).get(k) for k in ('pre_utc', 'post_utc', 'pre_sha256', 'post_sha256')},
            'known_limitations': ['S7 outputs live only on the GPU host S7 root; git holds hashes, not image bytes',
                                  'toy population (186 rows per seed); production equivalence is checked later']},
        'collected': c,
    }
    if c.get('tag'):
        doc['execution_tag'] = c['tag']
    if M == 'E05':
        doc['source_relocation_ruling'] = {'path': RULING_PATH, 'sha256': sha256_bytes((repo / RULING_PATH).read_bytes())}
        doc['prior_attempts'] = [{'attempt': 1, 'record': E05_FAILURE, 'sha256': sha256_bytes((repo / E05_FAILURE).read_bytes()),
                                  'status': 'S7_FAILED_NEEDS_OWNER_REVIEW (source-root path binding; resolved by O2)'}]
    return doc


def report(doc):
    M = doc['method']
    rows = ''.join('| %s | %s | %s | %s | `%s…` |\n' % (
        s, 'PASS' if all(g['pass'] for g in doc['gates'] if g['gate'].startswith('S.seed%s.runA' % s)) else 'FAIL',
        'PASS' if all(g['pass'] for g in doc['gates'] if g['gate'].startswith('S.seed%s.runB' % s)) else 'FAIL',
        'identical' if v['bitwise_identical'] else 'DIFFERENT', v['manifest_sha256'][:16])
        for s, v in sorted(doc['outputs'].items(), key=lambda x: int(x[0])))
    failed = doc['failed_gates']
    r = doc['reproduction']
    return '\n'.join([
        '# M8 learned-bank S7 toy acceptance: %s' % M, '',
        '**Status:** `%s`, %d/%d gates passing; `s7_acceptance_passed` = %s.' % (
            doc['status'], doc['gates_passed'], doc['gates_total'], str(doc['s7_acceptance_passed']).lower()), '',
        'Label: %s.' % doc['label'], '',
        'Authority `%s` (`%s…`), contract `%s…`, E04 seed-2026 record `%s…`, execution commit `%s`.' % (
            doc['authority']['path'], doc['authority']['sha256'][:12], doc['contract_sha256'][:12],
            doc['e04_seed_2026_completion_sha256'][:12], doc['execution_commit']), '',
        '| seed | run A | run B | A/B bytes | manifest sha256 |', '|---|---|---|---|---|', rows.rstrip('\n'), '',
        'Failed gates: %s' % (', '.join('`%s`' % g for g in failed) if failed else 'none.'), '',
        '## Execution', '',
        '- Environment `%s` (`%s`), framework %s, GPU %s `%s`, driver %s.' % (
            r['environment']['name'], r['environment']['python'], r['environment']['framework'], r['gpu']['name'],
            r['gpu']['uuid'], r['gpu']['driver']),
        '- Run A %s → %s; run B %s → %s; exit status A %s, B %s.' % (
            r['windows']['A'][0], r['windows']['A'][1], r['windows']['B'][0], r['windows']['B'][1],
            r['exit_status']['A'].strip(), r['exit_status']['B'].strip()),
        '- Population: %s.' % r['population'],
        '- RNG: %s.' % r['rng'],
        '- Conversion: `%s`.' % r['conversion_rule'],
        '- Output root (GPU host): `%s`.' % r['output_root'],
        '- Peak CUDA memory per seed (bytes): %s.' % json.dumps(doc['profiling']['peak_cuda_memory_allocated_bytes']),
        '- atime ledger: pre %s, post %s; no TRAIN/VAL/TEST face atime or mtime changed.' % (
            r['atime_ledger']['pre_utc'], r['atime_ledger']['post_utc']) if not any('F.atime' in g for g in failed)
        else '- atime ledger gate FAILED; see the JSON record.',
        '', '## Scope', '',
        'S7 establishes method-specific readiness for a separately authorized production launch only. No production '
        'bank, retraining, downstream training, TEST evaluation or GPAT work was performed.', ''])


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('preflight', 'collect'):
        a = sub.add_parser(name)
        a.add_argument('--method', required=True, choices=sorted(ESTIMATE_GIB))
        a.add_argument('--repo', required=True)
        a.add_argument('--s7-root', required=True)
        a.add_argument('--tag', default=None)
    b = sub.add_parser('build')
    b.add_argument('--collected', required=True)
    b.add_argument('--repo-root', default=str(Path(globals().get('__file__', 'tools/x')).resolve().parents[1]))
    args = ap.parse_args(argv)
    global TAG
    TAG = getattr(args, 'tag', None)
    if args.cmd == 'preflight':
        return preflight(args.method, args.repo, args.s7_root)
    if args.cmd == 'collect':
        return collect(args.method, args.repo, args.s7_root)
    c = json.loads(Path(args.collected).read_text())
    doc = build(c, args.repo_root)
    root = Path(args.repo_root)
    ev = evidence_paths(doc['method'])
    (root / ev['record']).write_text(json.dumps(doc, indent=1, sort_keys=True) + '\n')
    (root / ev['report']).write_text(report(doc))
    (root / ev['sha256']).write_text(''.join('%s  %s\n' % (sha256_file(root / ev[k]), ev[k]) for k in ('record', 'report')))
    print(doc['status'], doc['gates_passed'], '/', doc['gates_total'], doc['failed_gates'])
    return 0 if doc['s7_acceptance_passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
