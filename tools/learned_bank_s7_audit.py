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
PER_SEED = False                 # owner ruling B2 P2: one OS process per (seed, run); <S7>/<M>/seed_<s>/run_<L>/<M>/seed_<s>
B1B2_RULING_PATH = 'configs/amendments/e05_s7_b1_b2_ruling.yaml'
MEM_FLOOR_GIB = 4
RULING_PATH = 'configs/amendments/e05_source_relocation_ruling.yaml'
E05_FAILURE = 'outputs/audit/LEARNED_BANK_S7_E05_ATTEMPT1_FAILURE.json'


def m8s7_panes():
    out = {}
    for s in sh('tmux', 'ls', '-F', '#{session_name}').splitlines():
        if s.startswith('m8s7_'):
            out[s] = sh('tmux', 'list-panes', '-t', s, '-F', '#{pane_dead} #{pane_dead_status}')
    return out


# ------------------------------------------------------------------------------------------------------- preflight
def mem_available_bytes():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    return None


def preflight(method, repo, s7_root, seed=None, label=None):
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
    if seed is not None:
        seed_root = out_root / ('seed_%d' % seed)
        doc['memory'] = {'mem_available_bytes': mem_available_bytes(), 'floor_bytes': MEM_FLOOR_GIB << 30}
        doc['per_seed'] = {'seed': seed, 'label': label, 'seed_root': str(seed_root),
                           'seed_root_exists': seed_root.exists(),
                           'run_dir_exists': (seed_root / ('run_' + label)).exists(),
                           'run_A_completed': (seed_root / 'run_A' / method / ('seed_%d' % seed) /
                                               lb.PROVENANCE).is_file()}
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
    if PER_SEED:
        doc['layout'] = 'per_seed_process'
        doc['s7_root'] = str(s7_root)
        doc['seed_root_entries'] = {str(s): sorted(os.listdir(out_root / ('seed_%d' % s))) for s in SEEDS}

    def label_root(L, s):
        return (out_root / ('seed_%d' % s) if PER_SEED else out_root) / ('run_' + L)

    def run_dir(L, s):
        return label_root(L, s) / method / ('seed_%d' % s)
    watched = sorted({str(label_root(L, s)) for L in LABELS for s in SEEDS})
    before = {w: listing(w) for w in watched}
    for L in LABELS:
        doc['runs'][L] = {}
        for s in SEEDS:
            d = run_dir(L, s)
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
    doc['compare'] = {str(s): lb.compare_runs(run_dir('A', s), run_dir('B', s)) for s in SEEDS}
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
    def markers(tag, L):
        mp = marker_paths(method, L, tag)
        return {'command': read_text(mp['command']),
                'command_sha256': sha256_file(mp['command']) if Path(mp['command']).is_file() else None,
                'launch_time': (read_text(mp['launch_time']) or '').strip(),
                'exit_raw': read_text(mp['exit']), 'end_time': (read_text(mp['end_time']) or '').strip(),
                'pane': pane_summary(mp['pane.log']),
                'tmux_pane': sh('tmux', 'list-panes', '-t', 'm8s7_%s_%s' % (tag or method, L), '-F',
                                '#{pane_dead} #{pane_dead_status}')}

    def json_file(path):
        return (json.loads(read_text(path)) if Path(path).is_file() else None,
                sha256_file(path) if Path(path).is_file() else None)
    if PER_SEED:
        doc['markers_per_seed'], doc['preflights'], doc['preflights_sha256'], doc['cli_compare_per_seed'] = {}, {}, {}, {}
        for s in SEEDS:
            t = '%s_s%d' % (TAG, s)
            doc['markers_per_seed'][str(s)] = {L: markers(t, L) for L in LABELS}
            pfs = {L: json_file(marker_paths(method, tag='%s_%s' % (t, L))['preflight']) for L in LABELS}
            doc['preflights'][str(s)] = {L: v[0] for L, v in pfs.items()}
            doc['preflights_sha256'][str(s)] = {L: v[1] for L, v in pfs.items()}
            mp = marker_paths(method, tag=t)
            doc['cli_compare_per_seed'][str(s)] = {'stdout': read_text(mp['compare']), 'rc': read_text(mp['compare_rc'])}
    else:
        doc['markers'] = {L: markers(TAG, L) for L in LABELS}
        pf = marker_paths(method, tag=TAG)['preflight']
        doc['preflight'], doc['preflight_sha256'] = json_file(pf)
        mp = marker_paths(method, tag=TAG)
        doc['cli_compare'] = {'stdout': read_text(mp['compare']), 'rc': read_text(mp['compare_rc'])}
    doc['atime'] = atime_ledger(method, repo)
    gs = '%s/m8s7_%s_A.gpu_samples.tsv' % (WORKDIR, TAG or method)
    doc['gpu_samples_run_A'] = read_text(gs).splitlines() if Path(gs).is_file() else None
    doc['processes_after'] = processes()
    doc['gpu_after'] = gpu_state()
    doc['run_roots_unchanged_during_collect'] = all(listing(w) == before[w] for w in watched)
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


def per_seed_process_gates(G, c, M, commit, ckout_ok, auth_sha, env, lb):
    """Owner ruling B2 (P2_ONE_PROCESS_PER_SEED): six OS processes (seed x run), each with its own preflight, wrapper,
    tmux session, exit marker and `s7 --seed S` result line, strictly sequential (one GPU job at a time)."""
    est = ESTIMATE_GIB[M] + MARGIN_GIB
    procs = [(s, L) for s in SEEDS for L in LABELS]
    mks, pfs = c['markers_per_seed'], c['preflights']
    G('P.execution_checkout', ckout_ok and all((pfs[str(s)][L] or {}).get('git', {}).get('head') == commit
                                               and pfs[str(s)][L]['git']['status_porcelain'] == '' for s, L in procs),
      commit)
    for s, L in procs:
        pf, m = pfs[str(s)][L] or {}, mks[str(s)][L]
        g, ps = pf.get('gpu', {}), pf.get('per_seed', {})
        pre = 'P.seed%d.run%s.' % (s, L)
        G(pre + 'resources', pf.get('processes') == [] and all(v == '1 0' for v in pf.get('m8s7_panes', {}).values())
          and g.get('uuid') == GPU_UUID and g.get('name') == GPU_NAME and g.get('driver') == DRIVER
          and g.get('free_mib', 0) >= est * 1024 and pf.get('disk', {}).get('free_bytes', 0) >= DISK_FLOOR_GIB << 30
          and (pf.get('memory', {}).get('mem_available_bytes') or 0) >= MEM_FLOOR_GIB << 30,
          {'gpu_free_mib': g.get('free_mib'), 'required_mib': est * 1024, 'unrelated_gpu_processes': g.get('compute_apps'),
           'disk_free_gib': round(pf.get('disk', {}).get('free_bytes', 0) / 2 ** 30, 1),
           'mem_available_gib': round((pf.get('memory', {}).get('mem_available_bytes') or 0) / 2 ** 30, 1)})
        G(pre + 'fresh_run_root', ps.get('seed') == s and ps.get('label') == L and ps.get('run_dir_exists') is False
          and ps.get('seed_root') == '%s/seed_%d' % (c['out_root'], s)
          and ((L == 'A' and ps.get('seed_root_exists') is False) or
               (L == 'B' and ps.get('seed_root_exists') is True and ps.get('run_A_completed') is True))
          and pf['s7_root']['path'] == c['s7_root'] and pf['s7_root']['within_runtime_root'] is False
          and pf['s7_root']['within_hdd_bank_root'] is False and pf.get('utc', '9') <= m['launch_time'], ps)
        G(pre + 'checkpoints_and_authority', len(pf.get('checkpoints', {})) == 12 and all(
            v['ok'] and v['identity'] == lb.checkpoint_identity_sha256(k.split('|')[0], int(k.split('|')[1]))
            for k, v in pf['checkpoints'].items()) and pf.get('authority', {}).get('ok') is True
          and pf['authority']['record']['records'].get(AUTHORITY_PATH) == auth_sha)
        G(pre + 'environment', pf.get('environment', {}).get('present') is True and pf['environment']['probe_rc'] == 0
          and pf['environment']['python'] == env['python'], pf.get('environment'))
        res = (m['pane'] or {}).get('s7_result') or {}
        pre = 'R.seed%d.run%s.' % (s, L)
        G(pre + 'exit_zero', m['exit_raw'] == '0\n' and m['tmux_pane'] in ('1 0', ''), m['exit_raw'])
        G(pre + 'command', m['command'] is not None and (
            'tools/learned_bank.py s7 --method %s --label %s --out-root %s/seed_%d --runtime-root %s --seed %d\n'
            % (M, L, c['out_root'], s, RUNTIME, s)) in m['command'] and m['command'].count('--seed ') == 1
          and env['python'] in m['command'] and ('cd %s ' % c['repo']) in m['command'] and 'env -i ' in m['command']
          and 'CUDA_VISIBLE_DEVICES=0' in m['command'])
        r = (res.get('results') or {}).get(str(s), {})
        G(pre + 'result_line', res.get('label') == L and res.get('method') == M and res.get('seed') == s
          and res.get('process_seeds') == [s] and sorted(res.get('results', {})) == [str(s)]
          and r.get('status') == 'completed' and r.get('rows') == TOY_N and r.get('outputs') == TOY_N
          and r.get('failure_count') == 0 and r.get('checkpoint_unchanged_after') is True
          and not any(m['pane']['fail_hits'].values()), (m['pane'] or {}).get('fail_hits'))
    win = [(epoch(mks[str(s)][L]['launch_time']), epoch(mks[str(s)][L]['end_time'])) for s, L in procs]
    G('R.one_process_per_seed_run_sequential', all(a <= b for a, b in win) and all(
        win[i][1] <= win[i + 1][0] for i in range(len(win) - 1))
      and len({mks[str(s)][L]['command_sha256'] for s, L in procs}) == len(procs),
      [[mks[str(s)][L][k] for k in ('launch_time', 'end_time')] for s, L in procs])
    G('R.out_root_layout', c['out_root_entries'] == sorted('seed_%d' % s for s in SEEDS)
      and all(c['seed_root_entries'][str(s)] == ['run_A', 'run_B'] for s in SEEDS)
      and c['out_root'] == '%s/%s' % (c['s7_root'], M))
    G('R.no_process_left', c['processes_after'] == [])
    for s in SEEDS:
        cc = c['cli_compare_per_seed'][str(s)]
        try:
            cli = json.loads((cc.get('stdout') or '').strip().splitlines()[-1])
        except (ValueError, IndexError):
            cli = {}
        v = (cli.get('seeds') or {}).get(str(s), {})
        G('S.seed%d.cli_s7_compare' % s, cc.get('rc') == '0\n' and cli.get('command') == 's7-compare'
          and cli.get('method') == M and cli.get('seed') == s and sorted(cli.get('seeds', {})) == [str(s)]
          and v.get('replay', {}).get('bitwise_identical') is True and v.get('failures') == [0, 0]
          and v.get('run_A_reader_ok') is True and v.get('run_B_reader_ok') is True, cc.get('rc'))


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
    per_seed = c.get('layout') == 'per_seed_process'
    pf = (c['preflights']['42']['A'] if per_seed else c['preflight']) or {}
    if per_seed:
        procs = [(s, L) for s in SEEDS for L in LABELS]
        proc_markers = [c['markers_per_seed'][str(s)][L] for s, L in procs]
    else:
        proc_markers = [c['markers'][L] for L in LABELS]
    first_launch, last_end = proc_markers[0]['launch_time'], proc_markers[-1]['end_time']
    toy = lb.load_toy_manifest(repo)
    sources = json.loads((repo / lb.SOURCES_PATH).read_text())['families']
    env = lb.ENVIRONMENTS[M]

    # P. authority, execution checkout, preflight
    G('P.authority_covers_method', auth['status'] == 'OWNER_APPROVED' and auth['approved'] is True and M in auth['methods']
      and auth['seeds'] == list(SEEDS) and auth['contract_sha256'] == contract_sha
      and auth['e04_seed_2026_completion_sha256'] == e04_sha and auth['authorizes']['production_bank_generation'] is False
      and auth['resource_preflight']['required'] is True)
    ckout_ok = (c['git']['status_porcelain'] == '' and c['git']['branch'] == 'm6-baselines'
                and subprocess.run(['git', 'merge-base', '--is-ancestor', commit, 'HEAD'], cwd=repo).returncode == 0
                and subprocess.run(['git', 'show', '%s:%s' % (commit, AUTHORITY_PATH)], cwd=repo,
                                   capture_output=True).stdout == auth_raw)
    if per_seed:
        per_seed_process_gates(G, c, M, commit, ckout_ok, auth_sha, env, lb)
    else:
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
        if per_seed:
            b12_raw = (repo / B1B2_RULING_PATH).read_bytes()
            b12 = json.loads(b12_raw)
            G('P.e05_b1_b2_ruling_in_execution_commit', b12['status'] == 'OWNER_APPROVED' and b12['approved'] is True
              and b12['decisions'] == {'B1': 'TENSOR2IM_CHECK_AT_DEFINING_MODULE', 'B2': 'P2_ONE_PROCESS_PER_SEED'}
              and subprocess.run(['git', 'show', '%s:%s' % (commit, B1B2_RULING_PATH)], cwd=repo,
                                 capture_output=True).stdout == b12_raw)
            manifest = json.loads((repo / 'environments/e05.compatibility_patch_manifest.json').read_text())
            t2i_file = manifest['generated_files_sha256']['util/util.py']
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
                if per_seed:
                    ad = c['runs'][L][str(s)]['provenance']['adapter']
                    t2 = ad.get('tensor2im_runtime') or {}
                    G('S.seed%d.run%s.e05_tensor2im_runtime' % (s, L), t2 == {
                        'module': 'util.util', 'file': 'util/util.py', 'file_sha256': t2i_file,
                        'segment_sha256': ad.get('tensor2im_segment_sha256')}
                      and ad.get('tensor2im_segment_sha256') == b12['tensor2im']['pinned_segment_sha256'], t2)

    cc = c.get('cli_compare') or {}
    try:
        cli = json.loads((cc.get('stdout') or '').strip().splitlines()[-1])
    except (ValueError, IndexError):
        cli = {}
    per_seed or G('S.cli_s7_compare', cc.get('rc') == '0\n' and cli.get('command') == 's7-compare' and cli.get('method') == M
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
          and at['pre_utc'] <= first_launch and at['post_utc'] >= last_end
          and all(v['atime_changed'] == 0 and v['mtime_changed'] == 0 and v['missing_post'] == 0 for v in by.values())
          and {'TRAIN', 'VAL', 'TEST'} <= set(by), by)
    G('F.no_val_test_inputs', all(
        p['provenance']['population'].get('pairs_val_opened', False) is False for L in LABELS
        for p in c['runs'][L].values()) and (M not in lb.PAIR_METHODS or all(
            r['target_live_id'] in {t['target_live_id'] for t in toy} for rows in rows_by_seed.values() for r in rows)))
    G('F.collect_did_not_change_roots', c['run_roots_unchanged_during_collect'] is True)

    failed = G.failed
    status = 'S7_ACCEPTANCE_PASSED' if not failed else 'S7_FAILED_NEEDS_OWNER_REVIEW'
    def mk_of(L, s):
        return c['markers_per_seed'][str(s)][L] if per_seed else c['markers'][L]
    peak = {L: {s: ((mk_of(L, s)['pane'] or {}).get('s7_result') or {}).get('results', {}).get(str(s), {}).get(
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
            **({'commands': {L: c['markers'][L]['command'] for L in LABELS},
                'compare_command': '%s -B tools/learned_bank.py s7-compare --method %s --out-root %s'
                                   % (env['python'], M, c['out_root']),
                'windows': {L: [c['markers'][L]['launch_time'], c['markers'][L]['end_time']] for L in LABELS},
                'exit_status': {L: c['markers'][L]['exit_raw'] for L in LABELS}} if not per_seed else {
                'layout': 'P2_ONE_PROCESS_PER_SEED: one OS process per (seed, run), strictly sequential',
                'commands': {str(s): {L: mk_of(L, s)['command'] for L in LABELS} for s in SEEDS},
                'compare_commands': {str(s): '%s -B tools/learned_bank.py s7-compare --method %s --out-root %s/seed_%d '
                                             '--seed %d' % (env['python'], M, c['out_root'], s, s) for s in SEEDS},
                'windows': {str(s): {L: [mk_of(L, s)['launch_time'], mk_of(L, s)['end_time']] for L in LABELS}
                            for s in SEEDS},
                'exit_status': {str(s): {L: mk_of(L, s)['exit_raw'] for L in LABELS} for s in SEEDS},
                'preflights_sha256': c['preflights_sha256']}),
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
        ('- Run A %s → %s; run B %s → %s; exit status A %s, B %s.' % (
            r['windows']['A'][0], r['windows']['A'][1], r['windows']['B'][0], r['windows']['B'][1],
            r['exit_status']['A'].strip(), r['exit_status']['B'].strip()) if 'layout' not in r else
         '- One OS process per seed and run (owner ruling B2, P2), strictly sequential: %s.' % '; '.join(
             'seed %s run %s %s → %s exit %s' % (s, L, r['windows'][s][L][0], r['windows'][s][L][1],
                                                 r['exit_status'][s][L].strip())
             for s in sorted(r['windows'], key=int) for L in ('A', 'B'))),
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


# =================================================================================================================
# E04 Physics-STD branch (additive; the E03 / E05 / E06b / E06c paths above are unchanged).
#
# Authority: configs/amendments/learned_bank_s7_launch_authority_E04.yaml (derived from its DRAFT) under the approved E04
# contract (45240cd2...), the E04 family closure and owner ruling configs/amendments/e04_s7_input_allowlist_ruling.yaml.
# Layout (one OS process per seed and run, seeds 42 -> 1337 -> 2026, A before B):
#   <S7>/E04/seed_<s>/run_<L>/E04/seed_<s>/{bank/,bank_manifest.jsonl,bank_manifest.sha256,bank_provenance.json}
# Host markers ~/workdir/m8s7_E04_s<s>_<L>.{command,launch_time,exit,end_time,pane.log,gpu_samples.tsv},
#   preflight m8s7_E04_s<s>_<L>_preflight.json, per-seed atime ledgers m8s7_E04_s<s>_atime_{pre,post}.{tsv,time},
#   per-seed CLI comparison m8s7_E04_s<s>_compare.{json,rc}.
# The collector re-derives everything it gates from the raw run directories (it does not trust the runner's status):
# PNG structure / CRC / IHDR / zlib stream and a PIL decode in a different interpreter and Pillow than the writer,
# file and pixel SHA256 per output, byte and decoded-array A/B comparison, manifest re-derivation, the pair and face
# lineage against pairs_train_v1 / split_v1 / M2 accounting (pyarrow, row-filtered to referenced ids) and the whole
# feed-pack allowlist cross-check the owner ruling moved out of the py3.8 inference process.
# =================================================================================================================
import ast
import struct
import zlib

E04 = 'E04'
E04_AUTHORITY_PATH = 'configs/amendments/learned_bank_s7_launch_authority_E04.yaml'
E04_AUTHORITY_DRAFT_PATH = 'configs/amendments/learned_bank_s7_launch_authority_E04.DRAFT.yaml'
E04_AUTHORITY_SCHEMA = 'learned_bank_s7_launch_authority_e04_v1'
E04_CONTRACT_PATH = 'configs/amendments/e04_bank_generation_contract.yaml'
E04_CONTRACT_SHA256 = '45240cd25e840c16a11256639cab52c2f67c73d2a5a853284d7d9aa96b6df13f'
E04_DECISIONS = {
    'E04-Q1_normalization_statistics': 'A_MOVING_STATISTICS_PER_PAIR',
    'E04-Q2_synthesis_variant': 'A_PAPER_SPOOF_SYNTHESIS', 'E04-Q3_float_to_uint8': 'A_PREDECESSOR_CLIP_TRUNCATE',
    'E04-Q4_auxiliary_outputs': 'A_NONE', 'E04-Q5_device_and_reproducibility': 'A_GPU_FP32_BITWISE_REPLAY',
    'E04-Q6_governance_inclusion': 'A_ADDITIVE_E04_EXTENSION'}
E04_CLOSURE_PATH = 'outputs/audit/E04_THREE_SEED_TRAINING_CLOSURE.json'
E04_CLOSURE_STATUS = 'SCIENTIFIC_TRAINING_COMPLETED_3_OF_3_SEEDS'
E04_FINAL_CKPT_CSV = 'outputs/audit/E04_THREE_SEED_FINAL_CHECKPOINTS.csv'
E04_RUN_IDS = {42: '40befa6004aee85c', 1337: 'a42fdd2a2accaf0a', 2026: 'e7100eb9a5ab0537'}
E04_FINAL_ITERATION = 150000
E04_CKPT_FILES = {'data-00000-of-00001': 'checkpoints/iteration_150000.data-00000-of-00001',
                  'index': 'checkpoints/iteration_150000.index', 'meta': 'checkpoints/iteration_150000.meta'}
E04_FIXTURE_RECORD = 'outputs/audit/E04_BANK_FIXTURE_QUALIFICATION.json'
E04_RULING_PATH = 'configs/amendments/e04_s7_input_allowlist_ruling.yaml'
E04_RULING_DECISION = 'FEED_PACK_ALLOWLIST_WITH_INDEPENDENT_AUDITOR_CROSSCHECK'
E04_ALLOWLIST_SOURCE = ('e04_feed_pack_v1 samples.json (manifest-pinned TRAIN closure; owner ruling '
                        'configs/amendments/e04_s7_input_allowlist_ruling.yaml)')
E04_ADAPTER_PATH = 'methods/physics_std/bank_adapter.py'
E04_CLI_PATH = 'tools/learned_bank.py'
E04_LB_PATH = 'methods/common/learned_bank.py'
E04_QUALIFIED_ADAPTER_COMMIT = '92ef74c99765109158432fb713a68b3ad238f431'     # CPU fixture-qualified adapter
E04_PREP_COMMIT = '2caff0d166d2b347e31e96b9ecf96294aa4ae779'                  # session start (S7 prep)
E04_ALLOWED_ADAPTER_CHANGES = ['ALLOWLIST_SOURCE', 'load_e04_inputs']          # owner ruling scope
E04_ALLOWED_CLI_CHANGES = ['SEED_SELECTABLE', 'e04_inputs']                     # per-seed processes + provenance
E04_PYTHON = '/home/student20261/miniconda3/envs/gpat-m6-e03/bin/python'
E04_PY_VERSION = '3.8.20'
E04_TF = '1.15.5'
E04_ENV_LOCK = ('environments/e04_training.lock.json', 'a9629d46c2464f011fb68e0f071ecd4cc21cf0ede5af68faf9ab0d4595fefbdd')
E04_RUNTIME_RECORD = 'environments/e04_training.runtime.json'
E04_CONFIG = ('configs/methods/e04_physics_std.yaml', '418617942ebd87c45cb92bb7dc9ad96a6500a104489a657f9ca3345c3686576a')
E04_PRECISION_ENV = (           # order of the launch command (E04 training lock precision_env)
    ('CUBLAS_WORKSPACE_CONFIG', ':4096:8'), ('CUDA_VISIBLE_DEVICES', '0'), ('NVIDIA_TF32_OVERRIDE', '0'),
    ('PYTHONDONTWRITEBYTECODE', '1'), ('PYTHONNOUSERSITE', '1'), ('TF_CUDNN_DETERMINISTIC', '1'),
    ('TF_DETERMINISTIC_OPS', '1'), ('TF_ENABLE_AUTO_MIXED_PRECISION', '0'),
    ('TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE', '0'), ('TF_ENABLE_CUBLAS_TENSOR_OP_MATH_FP32', '0'),
    ('TF_ENABLE_CUDNN_RNN_TENSOR_OP_MATH_FP32', '0'), ('TF_ENABLE_CUDNN_TENSOR_OP_MATH_FP32', '0'),
    ('TF_ENABLE_ONEDNN_OPTS', '0'), ('TF_XLA_FLAGS', '--tf_xla_auto_jit=0'))
E04_FROZEN_FILES = {
    'methods/physics_std/network.py': '9d691809d487c5c9c27b5c9e08389225f606f8d72f1017d2991cd71e6214da21',
    'methods/physics_std/trace_transfer.py': '8152a7643858cb523344dcfa8fbe1a84168082d545d6ca68f7435cf2a06882cd',
    'methods/physics_std/training_graph.py': 'af17ab2789f203834d9fd9121b357dd3e44e49689fccac6be25024926bf69f51'}
E04_SYNTHESIS_LINES = [32, 36, 37, 39]
E04_PREDECESSOR = {'commit': 'c79f1f8c61', 'lines': ['model/utils.py:24', 'model/utils.py:37']}
E04_CONVERSION_RULE = ('(np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8) float32 [E04-Q3 A_PREDECESSOR_CLIP_TRUNCATE; '
                       'reference stdn model/utils.py:24,37]')
E04_PACK_MANIFEST_SHA256 = 'db1908d56efb23758fee49f88d2158374b85572a213a269419881bd02037520b'
E04_PACK_DIR = RUNTIME + '/data/processed/e04_feed_pack_v1'
E04_PACK_COUNTS = {'pairs': 8838, 'samples': 12668}
E04_FACES_ROOT = RUNTIME + '/data/processed/faces_256'
E04_FACE_READS = 2 * TOY_N
E04_GPU_FLOOR_MIB = 4 * 1024          # authority: free GPU memory >= 4 GiB (provisional; run A samples the peak)
E04_MEM_FLOOR_GIB = 4
E04_RUN_INSTANCE_FIELDS = ['end_utc', 'run_dir', 'start_utc', 'wall_time_seconds']   # the only provenance keys A/B may differ in
E04_PNG_STRUCTURE = {'signature_ok': True, 'chunk_types': ['IHDR', 'IDAT', 'IEND'], 'crc_ok': True, 'trailing_bytes': 0,
                     'ihdr': [256, 256, 8, 2, 0, 0, 0], 'zlib_header': '789c', 'decompressed_bytes': 256 * (1 + 256 * 3),
                     'pil_format': 'PNG', 'pil_mode': 'RGB', 'shape': [256, 256, 3], 'dtype': 'uint8'}
PAIRS_PARQUET = ('manifests/pairs_train_v1.parquet', 'a5e4fdaef236f15730c7e3885b537e08e995faffbe654167fc940f44bbc75243')
SPLIT_PARQUET = ('manifests/split_v1.parquet', 'fb9aeb369a124fc96ba855ef2ce269236c4a743fe960e73ab739412c9cb5092d')
M2_PARQUET = ('manifests/m2_sample_accounting.parquet', '65bb83e69ea7cebcb37373971e5e8f5248609b7963fd9d32aa1c0ba4377c0662')
TOY_MANIFEST = ('outputs/audit/M8_SMOKE_EVIDENCE/run_a/select/toy_manifest.json',
                '50ac8fb8b9fc6938ba630d0853523d358250137c8fbbc62704d2a73830f7f690')
HDD_BANK_ROOT = '/media/ext_data/student20261_gpat_m8_banks'
MANIFEST_FIELDS_E04 = (
    'schema', 'row_index', 'row_id', 'method_id', 'bank_seed', 'generator_seed', 'mode', 'label', 'synthetic_id',
    'pair_id', 'target_live_id', 'source_spoof_id', 'dataset', 'target_face_sha256', 'source_face_sha256',
    'deterministic_noise_seed', 'dsdg_batch_index', 'dsdg_batch_offset', 'checkpoint_identity_sha256',
    'config_sha256', 'contract_sha256', 'conversion_rule', 'generation_status', 'failure_reason', 'output_path',
    'output_sha256', 'output_pixel_sha256', 'raw_output_finite', 'raw_output_min', 'raw_output_max')
E04_PROCESS_LAYOUT = 'ONE_PROCESS_PER_SEED_AND_RUN'
E04_PROCS = [(s, L) for s in SEEDS for L in LABELS]


def e04_tag(seed, label=None):
    return 'E04_s%d' % seed if label is None else 'E04_s%d_%s' % (seed, label)


def e04_seed_root(s7_root, seed):
    return Path(s7_root) / E04 / ('seed_%d' % seed)


def e04_run_dir(s7_root, seed, label):
    return e04_seed_root(s7_root, seed) / ('run_' + label) / E04 / ('seed_%d' % seed)


def e04_command_line(repo, s7_root, seed, label):
    """The exact launch line (inside the wrapper script) the auditor expects for (seed, run)."""
    envs = ' '.join('%s=%s' % kv for kv in E04_PRECISION_ENV)
    return ('cd %s && env -i HOME=/home/student20261 PATH=/usr/bin:/bin LANG=C.UTF-8 %s nice -n 10 %s -B '
            'tools/learned_bank.py s7 --method E04 --label %s --out-root %s --runtime-root %s --seed %d\n'
            % (repo, envs, E04_PYTHON, label, e04_seed_root(s7_root, seed), RUNTIME, seed))


def e04_compare_line(repo, s7_root, seed):
    envs = ' '.join('%s=%s' % kv for kv in E04_PRECISION_ENV if kv[0] != 'CUDA_VISIBLE_DEVICES')
    return ('cd %s && env -i HOME=/home/student20261 PATH=/usr/bin:/bin LANG=C.UTF-8 CUDA_VISIBLE_DEVICES= %s %s -B '
            'tools/learned_bank.py s7-compare --method E04 --out-root %s --seed %d\n'
            % (repo, envs, E04_PYTHON, e04_seed_root(s7_root, seed), seed))


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False)


def checkpoint_identity(files):
    """Independent re-statement of the approved checkpoint identity: sha256 of canonical JSON {file key: sha256}."""
    return sha256_bytes(canonical(files).encode())


# ------------------------------------------------------------------------------------------------- E04 authority
def derive_e04_s7_authority(draft_raw, owner_record):
    """The activated E04 S7 launch authority = the committed DRAFT with only the activation fields set. Every DRAFT
    scientific field (contract, closure, checkpoint identities, toy manifest, scope, exclusions, reproducibility
    standard, resource preflight) is carried over byte-for-byte in value."""
    d = json.loads(draft_raw)
    out = dict(d)
    out.update({
        'status': 'OWNER_APPROVED', 'approved': True,
        'approval': owner_record,
        'derived_from': {'path': E04_AUTHORITY_DRAFT_PATH, 'sha256': sha256_bytes(draft_raw)},
        'launch_preconditions': [
            'E04 branch of tools/learned_bank_s7_audit.py implemented and qualified (positive, negative and '
            'regression tests) before this record was committed',
            'owner ruling configs/amendments/e04_s7_input_allowlist_ruling.yaml (feed-pack TRAIN allowlist; split_v1 / '
            'M2 cross-check in the independent auditor)'],
        'process_layout': E04_PROCESS_LAYOUT,
        'process_layout_detail': 'seeds 42 -> 1337 -> 2026 sequentially; for each seed run A then run B, each in its '
                                 'own OS process (tools/learned_bank.py s7 --seed S), one GPU job at a time; A/B '
                                 'compared per seed before the next seed starts',
        'execution': {'environment': 'gpat-m6-e03', 'python': E04_PYTHON, 'tensorflow': E04_TF,
                      'environment_lock': {'path': E04_ENV_LOCK[0], 'sha256': E04_ENV_LOCK[1]},
                      'precision_env': dict(E04_PRECISION_ENV), 'gpu_name': GPU_NAME, 'gpu_uuid': GPU_UUID,
                      'driver': DRIVER, 'driver_policy': 'E04 D9 OWNER_APPROVED_DRIVER_VERSION_ONLY_RUNTIME_COMPATIBILITY',
                      'bn_mode': 'inference (training=False), stored moving statistics, batch 1',
                      'conversion_rule': E04_CONVERSION_RULE},
        'inputs': {'splits': ['TRAIN'], 'face_reads': 'O_NOATIME with atime check, M2 face SHA256 before decoding',
                   'feed_pack_manifest_sha256': E04_PACK_MANIFEST_SHA256, 'val_test_access': False},
        'authorizes': {'s7_toy_acceptance': True, 'production_bank_generation': False,
                       'storage_directory_creation': False, 'storage_extension_activation': False,
                       'generator_retraining': False, 'val_test_access': False, 'downstream_evaluation': False},
        'independent_audit': {'tool': 'tools/learned_bank_s7_audit.py (E04 branch)',
                              'record': 'outputs/audit/LEARNED_BANK_S7_E04.json', 'required': True},
        'fail_closed': 'any failed gate, missing evidence or non-bitwise A/B replay stops E04 for owner review; '
                       'outputs and logs are preserved; no automatic retry, no tolerance or CPU fallback, no change '
                       'of BN mode, conversion, checkpoint or graph',
        'source_commit_note': 'this record is committed before any E04 GPU inference; the execution commit is the '
                              'commit that contains it (or a descendant with no change to any E04 path)'})
    return out


class GitView:
    """Repository access for the builder (tests substitute an overlay)."""

    def __init__(self, repo):
        self.repo = Path(repo)

    def read(self, rel):
        p = self.repo / rel
        return p.read_bytes() if p.is_file() else None

    def show(self, commit, rel):
        r = subprocess.run(['git', 'show', '%s:%s' % (commit, rel)], cwd=str(self.repo), capture_output=True)
        return r.stdout if r.returncode == 0 else None

    def is_ancestor(self, commit):
        return subprocess.run(['git', 'merge-base', '--is-ancestor', commit, 'HEAD'], cwd=str(self.repo),
                              capture_output=True).returncode == 0


def top_level_changes(old_src, new_src):
    """Names of module-level definitions/assignments whose source differs (added, removed or changed) between two
    versions of a Python module; unnamed statements are compared as a multiset of source segments."""
    def nodes(src):
        named, unnamed = {}, []
        tree = ast.parse(src)
        for node in tree.body:
            seg = ast.get_source_segment(src, node)
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                named[node.name] = seg
            elif isinstance(node, ast.Assign) and all(isinstance(t, ast.Name) for t in node.targets):
                named[','.join(t.id for t in node.targets)] = seg
            else:
                unnamed.append(seg)
        return named, sorted(unnamed)
    (a, ua), (b, ub) = nodes(old_src), nodes(new_src)
    changed = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    if ua != ub:
        changed.append('<unnamed statements>')
    return changed


# ------------------------------------------------------------------------------------------------- E04 preflight
E04_PROBE = r'''
import json, os, sys
repo = sys.argv[1]
sys.path.insert(0, repo)
os.chdir(repo)
out = {'python': sys.version.split()[0], 'executable': sys.executable}
def attempt(key, fn):
    try:
        out[key] = fn()
    except BaseException as exc:
        out[key] = {'error': '%s: %s' % (type(exc).__name__, exc)}
import tensorflow as tf
out['tensorflow'] = tf.__version__
from methods.common import learned_bank as lb
from methods.physics_std import bank_adapter as ba
from methods.physics_std.production_runner import verify_environment
out['precision_errors'] = ba.precision_environment_errors(os.environ, gpu=True)
def env():
    live, drv = verify_environment(tf)
    return {'ok': True, 'driver': drv, 'native_runtime': live['native_runtime'], 'gpu': live['gpu'],
            'package_versions': live['package_versions'], 'session_config': live['session_config']}
attempt('environment_lock', env)
attempt('frozen_sources', ba.verify_frozen_sources)
attempt('bn_only_difference', ba.inference_layers_differ_only_in_bn_mode)
attempt('contract_sha256', lambda: lb.load_e04_generation_contract()[1])
attempt('s7_authority', lambda: lb.load_s7_authority('E04').record())
attempt('config', lambda: lb.verify_config_identity('E04'))
def counters():
    res = {}
    for s in (42, 1337, 2026):
        prefix = '%s/runs/m6/E04/seed_%d/checkpoints/iteration_150000' % (sys.argv[2], s)
        r = tf.train.NewCheckpointReader(prefix)
        names = sorted(k for k in r.get_variable_to_shape_map() if not k.startswith(('G/', 'D/')) and 'Adam' not in k)
        res[str(s)] = {k: r.get_tensor(k).tolist() for k in names if r.get_tensor(k).size <= 1}
    return res
attempt('checkpoint_counters', counters)
print(json.dumps(out, sort_keys=True, default=str))
'''


def e04_checkpoint_hashes(lb, seed):
    r = lb.verify_checkpoint(E04, seed, Path(RUNTIME) / 'runs' / 'm6' / E04 / ('seed_%d' % seed))
    return {'ok': True, 'identity': r['checkpoint_identity_sha256'], 'run_id': r['run_id'],
            'files': {k: {'path': v['path'], 'sha256': v['sha256'], 'size': v['size'], 'mtime_ns': v['mtime_ns']}
                      for k, v in r['files'].items()}}


def preflight_e04(repo, s7_root, seed, label):
    lb = lb_module(repo)
    out_root = Path(s7_root) / E04
    seed_root = e04_seed_root(s7_root, seed)
    doc = {'phase': 'preflight', 'method': E04, 'utc': utc_now(), 'repo': str(repo), 'git': git_state(repo),
           'collector': {'python': sys.version.split()[0], 'executable': sys.executable},
           'processes': processes(), 'm8s7_panes': m8s7_panes(), 'gpu': gpu_state(), 'seed': seed, 'label': label}
    du = shutil.disk_usage(str(Path(s7_root).parent))
    doc['disk'] = {'path': str(Path(s7_root).parent), 'free_bytes': du.free, 'floor_bytes': DISK_FLOOR_GIB << 30}
    doc['memory'] = {'mem_available_bytes': mem_available_bytes(), 'floor_bytes': E04_MEM_FLOOR_GIB << 30}
    doc['s7_root'] = {'path': str(s7_root), 'exists': Path(s7_root).is_dir(), 'method_out_root': str(out_root),
                      'method_out_root_exists': out_root.exists(),
                      'within_runtime_root': str(Path(s7_root).resolve()).startswith(RUNTIME),
                      'within_hdd_bank_root': str(Path(s7_root).resolve()).startswith('/media/')}
    doc['per_seed'] = {'seed': seed, 'label': label, 'seed_root': str(seed_root), 'seed_root_exists': seed_root.exists(),
                       'run_dir_exists': (seed_root / ('run_' + label)).exists(),
                       'run_A_completed': (e04_run_dir(s7_root, seed, 'A') / lb.PROVENANCE).is_file()}
    ck = {}
    for s in SEEDS:
        try:
            ck[str(s)] = e04_checkpoint_hashes(lb, s)
        except Exception as exc:
            ck[str(s)] = {'ok': False, 'error': '%s: %s' % (type(exc).__name__, exc)}
    doc['checkpoints'] = ck
    try:
        doc['authority'] = {'ok': True, 'record': lb.load_s7_authority(E04).record()}
    except Exception as exc:
        doc['authority'] = {'ok': False, 'error': '%s: %s' % (type(exc).__name__, exc)}
    toy = Path(repo) / TOY_MANIFEST[0]
    pack_m = Path(E04_PACK_DIR) / 'pack_manifest.json'
    doc['inputs'] = {'toy_manifest_sha256': sha256_file(toy) if toy.is_file() else None,
                     'feed_pack_manifest_sha256': sha256_file(pack_m) if pack_m.is_file() else None,
                     'faces_root_is_dir': Path(E04_FACES_ROOT).is_dir()}
    doc['hdd_bank_root_entries'] = sorted(os.listdir(HDD_BANK_ROOT)) if Path(HDD_BANK_ROOT).is_dir() else None
    env = ['env', '-i', 'HOME=' + os.environ.get('HOME', '/home/student20261'), 'PATH=/usr/bin:/bin', 'LANG=C.UTF-8',
           'TF_CPP_MIN_LOG_LEVEL=3'] + ['%s=%s' % kv for kv in E04_PRECISION_ENV]
    r = subprocess.run(env + [E04_PYTHON, '-I', '-B', '-c', E04_PROBE, str(repo), RUNTIME], capture_output=True,
                       text=True, cwd=str(repo))
    try:
        probe = json.loads(r.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        probe = None
    doc['environment'] = {'python': E04_PYTHON, 'present': Path(E04_PYTHON).is_file(), 'probe_rc': r.returncode,
                          'probe': probe, 'probe_stderr_tail': r.stderr.strip().splitlines()[-3:]}
    print(json.dumps(doc, sort_keys=True, default=str))


# ------------------------------------------------------------------------------------------------- E04 collect
def pixel_identity(arr):
    """Encoder-independent image identity (definition of the learned-bank manifest), recomputed here."""
    import numpy as np
    a = np.ascontiguousarray(arr)
    return sha256_bytes(('%s|%s|' % (a.dtype.str, 'x'.join(str(n) for n in a.shape))).encode() + a.tobytes())


def png_inspect(raw):
    """Independent PNG check: signature, chunk sequence and CRCs, IHDR, zlib stream and scanline filters, then a PIL
    decode (collector interpreter) and the pixel identity. Returns (structure dict, pixel sha256, decoded array)."""
    import io
    import numpy as np
    from PIL import Image
    st = {'signature_ok': raw[:8] == b'\x89PNG\r\n\x1a\n'}
    pos, types, crc_ok, idat, ihdr = 8, [], True, b'', None
    try:
        while pos + 12 <= len(raw):
            n = struct.unpack('>I', raw[pos:pos + 4])[0]
            typ, data = raw[pos + 4:pos + 8], raw[pos + 8:pos + 8 + n]
            crc = struct.unpack('>I', raw[pos + 8 + n:pos + 12 + n])[0]
            crc_ok = crc_ok and (zlib.crc32(typ + data) & 0xffffffff) == crc
            t = typ.decode('latin-1')
            if not types or types[-1] != t:
                types.append(t)
            if t == 'IHDR':
                ihdr = list(struct.unpack('>IIBBBBB', data))
            elif t == 'IDAT':
                idat += data
            pos += 12 + n
            if t == 'IEND':
                break
    except struct.error:
        crc_ok = False
    st.update(chunk_types=types, crc_ok=crc_ok, trailing_bytes=len(raw) - pos, ihdr=ihdr,
              zlib_header=idat[:2].hex())
    try:
        dec = zlib.decompress(idat)
        st['decompressed_bytes'] = len(dec)
        st['scanline_filters_ok'] = len(dec) % 769 == 0 and set(dec[0::769]) <= {0, 1, 2, 3, 4}
    except zlib.error:
        st['decompressed_bytes'], st['scanline_filters_ok'] = None, False
    try:
        with Image.open(io.BytesIO(raw)) as im:
            st['pil_format'], st['pil_mode'] = im.format, im.mode
            arr = np.array(im)
        st['shape'], st['dtype'] = list(arr.shape), str(arr.dtype)
        return st, pixel_identity(arr), arr
    except Exception as exc:
        st['decode_error'] = '%s: %s' % (type(exc).__name__, exc)
        return st, None, None


def manifest_inspect(d):
    p, ps = Path(d) / 'bank_manifest.jsonl', Path(d) / 'bank_manifest.sha256'
    if not p.is_file():
        return None
    body = p.read_bytes()
    lines = body.decode('ascii', errors='replace').splitlines()
    rows, canon, fields = [], True, True
    for x in lines:
        try:
            r = json.loads(x)
        except ValueError:
            canon = False
            continue
        canon = canon and canonical(r) == x
        fields = fields and sorted(r) == sorted(MANIFEST_FIELDS_E04)
        rows.append(r)
    return {'sha256': sha256_bytes(body), 'bytes': len(body), 'lines': len(lines),
            'ends_with_newline': body.endswith(b'\n'), 'canonical_lines': canon, 'fields_ok': fields,
            'sha_file': ps.read_text() if ps.is_file() else None, 'rows': rows}


def run_dir_evidence(d):
    """Everything the builder gates for one run directory, re-derived from the raw files."""
    d = Path(d)
    if not d.is_dir():
        return None
    prov_p = d / 'bank_provenance.json'
    lst = listing(d)
    man = manifest_inspect(d)
    structures, files = {}, {}
    bank = d / 'bank'
    for p in sorted(bank.iterdir()) if bank.is_dir() else []:
        raw = p.read_bytes()
        st, px, _ = png_inspect(raw)
        key = canonical(st)
        structures[key] = structures.get(key, 0) + 1
        files[p.name] = [sha256_bytes(raw), px, len(raw)]
    return {'dir': str(d), 'provenance': json.loads(prov_p.read_text()) if prov_p.is_file() else None,
            'provenance_sha256': sha256_file(prov_p) if prov_p.is_file() else None,
            'aborted': (d / 'ABORTED_bank_provenance.json').exists(), 'listing': lst,
            'listing_sha256': sha256_bytes(json.dumps(lst).encode()), 'dir_mode': oct(d.stat().st_mode & 0o7777),
            'bank_dir_mode': oct(bank.stat().st_mode & 0o7777) if bank.is_dir() else None,
            'manifest': {k: v for k, v in (man or {}).items() if k != 'rows'} if man else None,
            'manifest_rows': [[r.get(k) for k in MANIFEST_FIELDS_E04] for r in man['rows']] if man else None,
            'png_structures': structures, 'png_files': files}


def independent_compare(da, db):
    """A/B replay re-derived from the raw directories: file sets, every byte, every decoded pixel array, manifest
    bytes, manifest sha files and the provenance keys that differ (declared run-instance fields only)."""
    import numpy as np
    da, db = Path(da), Path(db)
    if not (da.is_dir() and db.is_dir()):
        return None
    fa = sorted(os.listdir(da / 'bank'))
    fb = sorted(os.listdir(db / 'bank'))
    bytes_diff, px_diff, decoded = [], [], 0
    for n in sorted(set(fa) & set(fb)):
        ra, rb = (da / 'bank' / n).read_bytes(), (db / 'bank' / n).read_bytes()
        if ra != rb:
            bytes_diff.append(n)
        _, _, aa = png_inspect(ra)
        _, _, ab = png_inspect(rb)
        if aa is None or ab is None or aa.shape != ab.shape or aa.dtype != ab.dtype or not np.array_equal(aa, ab):
            px_diff.append(n)
        else:
            decoded += 1
    pa, pb = (json.loads((x / 'bank_provenance.json').read_text()) if (x / 'bank_provenance.json').is_file() else {}
              for x in (da, db))
    return {'files_A': len(fa), 'files_B': len(fb), 'file_set_equal': fa == fb, 'bytes_differing': bytes_diff[:50],
            'pixels_differing': px_diff[:50], 'pixel_arrays_compared_equal': decoded,
            'manifest_bytes_equal': (da / 'bank_manifest.jsonl').read_bytes() == (db / 'bank_manifest.jsonl').read_bytes(),
            'manifest_sha_file_equal': (da / 'bank_manifest.sha256').read_bytes() == (db / 'bank_manifest.sha256').read_bytes(),
            'provenance_differing_keys': sorted(k for k in set(pa) | set(pb) if pa.get(k) != pb.get(k))}


def e04_lineage(repo, rows_by_seed):
    """pairs_train_v1 / split_v1 / M2 accounting (hash-checked, row-filtered to referenced ids) for every toy pair and
    face the runs used, plus the whole e04_feed_pack_v1 allowlist cross-check (owner ruling)."""
    import pyarrow.parquet as pq
    repo = Path(repo)
    out = {'files_sha256': {rel: sha256_file(repo / rel) for rel, _ in (PAIRS_PARQUET, SPLIT_PARQUET, M2_PARQUET)}}
    pairs = pq.read_table(str(repo / PAIRS_PARQUET[0]),
                          columns=['pair_id', 'target_live_id', 'source_spoof_id', 'dataset', 'split']).to_pylist()
    ids = [r['pair_id'] for r in pairs]
    order = sorted(ids)
    pos = {p: i for i, p in enumerate(order)}
    by_id = {r['pair_id']: r for r in pairs}
    out['pairs_total'] = len(pairs)
    out['pairs_unique'] = len(set(ids)) == len(ids)
    out['pairs_stored_in_ascending_order'] = ids == order
    out['pairs_splits'] = sorted({r['split'] for r in pairs})
    used_pairs = sorted({r[MANIFEST_FIELDS_E04.index('pair_id')] for rows in rows_by_seed.values() for r in rows})
    out['toy_relation'] = {p: [pos.get(p), by_id[p]['target_live_id'], by_id[p]['source_spoof_id'], by_id[p]['dataset']]
                           if p in by_id else None for p in used_pairs}
    used = sorted({r[MANIFEST_FIELDS_E04.index(k)] for rows in rows_by_seed.values() for r in rows
                   for k in ('target_live_id', 'source_spoof_id')})

    def sample_table(sids):
        sp = pq.read_table(str(repo / SPLIT_PARQUET[0]), columns=['sample_id', 'split', 'dataset'],
                           filters=[('sample_id', 'in', sids)]).to_pylist()
        m2 = pq.read_table(str(repo / M2_PARQUET[0]), columns=['sample_id', 'face_png_sha256', 'final_status'],
                           filters=[('sample_id', 'in', sids)]).to_pylist()
        a, b = {r['sample_id']: r for r in sp}, {r['sample_id']: r for r in m2}
        return {s: [a[s]['split'], a[s]['dataset'], b[s]['face_png_sha256'], b[s]['final_status']]
                if s in a and s in b else None for s in sids}
    out['samples'] = sample_table(used) if used else {}
    # whole-pack cross-check (the allowlist the inference process used)
    pk = Path(E04_PACK_DIR)
    m_raw = (pk / 'pack_manifest.json').read_bytes()
    pack = {'manifest_sha256': sha256_bytes(m_raw)}
    s_raw = (pk / 'samples.json').read_bytes()
    p_raw = (pk / 'pairs_train.json').read_bytes()
    man = json.loads(m_raw)
    pack['payload_sha_ok'] = (sha256_bytes(s_raw) == man['payloads']['samples.json']['sha256']
                              and sha256_bytes(p_raw) == man['payloads']['pairs_train.json']['sha256'])
    sj, pj = json.loads(s_raw), json.loads(p_raw)
    cols = sj['columns']
    samples = [dict(zip(cols, r)) for r in sj['rows']]
    ref = sample_table(sorted(r['sample_id'] for r in samples))
    bad = [r['sample_id'] for r in samples if ref.get(r['sample_id']) is None
           or ref[r['sample_id']][0] != 'TRAIN' or r['split'] != 'TRAIN' or ref[r['sample_id']][1] != r['dataset']
           or ref[r['sample_id']][2] != r['face_png_sha256'] or ref[r['sample_id']][3] != 'COMPLETE']
    pack.update(samples=len(samples), samples_mismatching=bad[:20], samples_mismatching_count=len(bad),
                pairs=len(pj['rows']),
                pairs_equal_pairs_train_v1=[list(r) for r in pj['rows']] == [
                    [by_id[p]['pair_id'], by_id[p]['target_live_id'], by_id[p]['source_spoof_id'], by_id[p]['dataset']]
                    for p in order],
                pairs_split=pj.get('split'))
    out['feed_pack_crosscheck'] = pack
    return out


def gpu_samples_summary(path):
    p = Path(path)
    if not p.is_file():
        return None
    peak, lines = {}, p.read_text().splitlines()
    for x in lines:
        parts = [y.strip() for y in x.split(',')]
        if len(parts) >= 4 and parts[-1].isdigit():
            peak[parts[1]] = max(peak.get(parts[1], 0), int(parts[-1]))
    return {'samples': len(lines), 'peak_used_mib_by_pid': peak, 'sha256': sha256_file(p)}


def e04_markers(s, L):
    t = e04_tag(s)
    mp = marker_paths(E04, L, t)
    return {'command': read_text(mp['command']),
            'command_sha256': sha256_file(mp['command']) if Path(mp['command']).is_file() else None,
            'launch_time': (read_text(mp['launch_time']) or '').strip(), 'exit_raw': read_text(mp['exit']),
            'end_time': (read_text(mp['end_time']) or '').strip(), 'pane': pane_summary(mp['pane.log']),
            'tmux_pane': sh('tmux', 'list-panes', '-t', 'm8s7_%s_%s' % (t, L), '-F', '#{pane_dead} #{pane_dead_status}'),
            'gpu_samples': gpu_samples_summary('%s/m8s7_%s_%s.gpu_samples.tsv' % (WORKDIR, t, L))}


def e04_atime(repo, s):
    import pyarrow.parquet as pq
    p = marker_paths(E04, tag=e04_tag(s))

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
    split = {r['sample_id']: r['split'] for r in pq.read_table(str(Path(repo) / SPLIT_PARQUET[0]),
                                                               columns=['sample_id', 'split']).to_pylist()}
    by, changed = {}, []
    for path, (a, m) in pre.items():
        sp = split.get(os.path.basename(path)[:-4], 'UNMAPPED')
        b = by.setdefault(sp, {'files': 0, 'atime_changed': 0, 'mtime_changed': 0, 'missing_post': 0})
        b['files'] += 1
        if path not in post:
            b['missing_post'] += 1
            continue
        if post[path] != (a, m):
            changed.append(path)
        b['atime_changed'] += post[path][0] != a
        b['mtime_changed'] += post[path][1] != m
    return {'pre_utc': (read_text(p['atime_pre'][:-4] + '.time') or '').strip(),
            'post_utc': (read_text(p['atime_post'][:-4] + '.time') or '').strip(),
            'pre_sha256': sha256_file(p['atime_pre']), 'post_sha256': sha256_file(p['atime_post']),
            'files_pre': len(pre), 'files_post': len(post), 'same_paths': sorted(pre) == sorted(post), 'by_split': by,
            'changed_paths': changed[:50]}


def collect_e04(repo, s7_root):
    lb = lb_module(repo)
    out_root = Path(s7_root) / E04
    doc = {'phase': 'collect', 'method': E04, 'layout': E04_PROCESS_LAYOUT, 'utc': utc_now(), 'repo': str(repo),
           'git': git_state(repo), 'collector': {'python': sys.version.split()[0], 'executable': sys.executable},
           's7_root': str(s7_root), 'out_root': str(out_root),
           'out_root_entries': sorted(os.listdir(out_root)) if out_root.is_dir() else None,
           'seed_root_entries': {str(s): sorted(os.listdir(e04_seed_root(s7_root, s)))
                                 if e04_seed_root(s7_root, s).is_dir() else None for s in SEEDS}}
    watched = [str(e04_seed_root(s7_root, s)) for s in SEEDS if e04_seed_root(s7_root, s).is_dir()]
    before = {w: listing(w) for w in watched}
    doc['runs'] = {L: {str(s): run_dir_evidence(e04_run_dir(s7_root, s, L)) for s in SEEDS} for L in LABELS}
    doc['replay'] = {str(s): independent_compare(e04_run_dir(s7_root, s, 'A'), e04_run_dir(s7_root, s, 'B'))
                     for s in SEEDS}
    rb = {}
    for s in SEEDS:
        rb[str(s)] = {}
        for L in LABELS:
            d = e04_run_dir(s7_root, s, L)
            if d.is_dir():
                r = lb.read_bank(d, expect_method=E04, expect_seed=s, expect_mode='S7_TOY', expect_rows=TOY_N,
                                 expect_checkpoint_identity=lb.checkpoint_identity_sha256(E04, s))
                r.pop('rows')
                rb[str(s)][L] = r
        if all(e04_run_dir(s7_root, s, L).is_dir() for L in LABELS):
            rb[str(s)]['compare_runs'] = lb.compare_runs(e04_run_dir(s7_root, s, 'A'), e04_run_dir(s7_root, s, 'B'))
    doc['library_reader'] = rb
    rows = {s: doc['runs'][L][s]['manifest_rows'] for L in LABELS for s in doc['runs'][L]
            if doc['runs'][L][s] and doc['runs'][L][s]['manifest_rows']}
    doc['lineage'] = e04_lineage(repo, rows)
    after = {}
    for s in SEEDS:
        try:
            after[str(s)] = e04_checkpoint_hashes(lb, s)
        except Exception as exc:
            after[str(s)] = {'ok': False, 'error': '%s: %s' % (type(exc).__name__, exc)}
    doc['checkpoints_after'] = after
    doc['markers'] = {str(s): {L: e04_markers(s, L) for L in LABELS} for s in SEEDS}
    doc['preflights'], doc['preflights_sha256'] = {}, {}
    for s in SEEDS:
        doc['preflights'][str(s)], doc['preflights_sha256'][str(s)] = {}, {}
        for L in LABELS:
            f = marker_paths(E04, tag=e04_tag(s, L))['preflight']
            doc['preflights'][str(s)][L] = json.loads(read_text(f)) if Path(f).is_file() else None
            doc['preflights_sha256'][str(s)][L] = sha256_file(f) if Path(f).is_file() else None
    doc['cli_compare'] = {}
    for s in SEEDS:
        mp = marker_paths(E04, tag=e04_tag(s))
        cmd = '%s/m8s7_%s_compare.command' % (WORKDIR, e04_tag(s))
        doc['cli_compare'][str(s)] = {'stdout': read_text(mp['compare']), 'rc': read_text(mp['compare_rc']),
                                      'command': read_text(cmd)}
    doc['atime'] = {str(s): e04_atime(repo, s) for s in SEEDS}
    doc['hdd_bank_root_entries'] = sorted(os.listdir(HDD_BANK_ROOT)) if Path(HDD_BANK_ROOT).is_dir() else None
    doc['processes_after'] = processes()
    doc['gpu_after'] = gpu_state()
    doc['run_roots_unchanged_during_collect'] = all(listing(w) == before[w] for w in watched)
    print(json.dumps(doc, sort_keys=True, default=str))


# ------------------------------------------------------------------------------------------------- E04 build
class SafeGates(Gates):
    """Gates whose predicate may raise on missing / malformed evidence: an exception is a failed gate."""

    def check(self, name, fn, detail=None):
        try:
            ok = bool(fn())
        except Exception as exc:
            return self(name, False, {'exception': '%s: %s' % (type(exc).__name__, exc)})
        return self(name, ok, detail)


def _csv_checkpoints(raw):
    import csv
    import io
    rows = list(csv.DictReader(io.StringIO(raw.decode('utf-8'))))
    out = {}
    for r in rows:
        out[int(r['seed'])] = r
    return out


def build_e04(c, repo_root, view=None):
    repo = Path(repo_root)
    v = view or GitView(repo)
    G = SafeGates()
    commit = (c.get('git') or {}).get('head')

    def jload(raw):
        return json.loads(raw) if raw else {}
    auth_raw = v.read(E04_AUTHORITY_PATH)
    auth, auth_sha = jload(auth_raw), sha256_bytes(auth_raw) if auth_raw else None
    draft_raw = v.read(E04_AUTHORITY_DRAFT_PATH)
    contract_raw = v.read(E04_CONTRACT_PATH)
    contract, contract_sha = jload(contract_raw), sha256_bytes(contract_raw) if contract_raw else None
    closure_raw = v.read(E04_CLOSURE_PATH)
    closure, closure_sha = jload(closure_raw), sha256_bytes(closure_raw) if closure_raw else None
    ruling_raw = v.read(E04_RULING_PATH)
    ruling = jload(ruling_raw)
    csv_raw = v.read(E04_FINAL_CKPT_CSV) or b''
    csv_ck = _csv_checkpoints(csv_raw) if csv_raw else {}
    ck_table = {s: {'data-00000-of-00001': csv_ck[s]['final_data_sha256'], 'index': csv_ck[s]['final_index_sha256'],
                    'meta': csv_ck[s]['final_meta_sha256']} for s in SEEDS if s in csv_ck}
    idents = {str(s): checkpoint_identity(ck_table[s]) for s in ck_table}
    toy_raw = v.read(TOY_MANIFEST[0])
    toy = json.loads(toy_raw)['toy_pairs'] if toy_raw else []
    fixture = jload(v.read(E04_FIXTURE_RECORD))
    graph_id = fixture.get('key_results', {}).get('graph')
    runtime_rec = jload(v.read(E04_RUNTIME_RECORD))
    expected_records = {E04_CONTRACT_PATH: contract_sha, E04_AUTHORITY_PATH: auth_sha, E04_CLOSURE_PATH: closure_sha}
    repo_s7 = c.get('repo')
    runs_all = c.get('runs') or {}

    # ---------------- P: governance and authority
    G.check('P.e04_contract', lambda: contract_sha == E04_CONTRACT_SHA256 and contract['status'] == 'OWNER_APPROVED'
            and contract['approved'] is True and contract['decisions'] == E04_DECISIONS
            and contract['method_id'] == E04 and v.show(commit, E04_CONTRACT_PATH) == contract_raw, contract_sha)
    G.check('P.e04_family_closure', lambda: closure['status'] == E04_CLOSURE_STATUS and closure['failed_gates'] == []
            and closure['final_checkpoints_csv_sha256'] == sha256_bytes(csv_raw) and sorted(csv_ck) == list(SEEDS)
            and all(csv_ck[s]['run_id'] == E04_RUN_IDS[s] and csv_ck[s]['final_checkpoint_prefix']
                    == 'checkpoints/iteration_150000' and int(csv_ck[s]['final_global_iteration']) == E04_FINAL_ITERATION
                    and csv_ck[s]['completion_status'] == 'completed' for s in SEEDS)
            and v.show(commit, E04_CLOSURE_PATH) == closure_raw, closure_sha)
    G.check('P.e04_authority_activated', lambda: auth_raw is not None and auth['schema_version'] == E04_AUTHORITY_SCHEMA
            and auth['record'] == 'E04_S7_LAUNCH_AUTHORITY' and auth['status'] == 'OWNER_APPROVED'
            and auth['approved'] is True and auth['methods'] == [E04] and auth['seeds'] == list(SEEDS)
            and auth['contract_sha256'] == E04_CONTRACT_SHA256 == contract_sha
            and auth['e04_family_closure_sha256'] == closure_sha and auth['checkpoint_identities'] == idents
            and auth['toy_manifest'] == {'pairs': TOY_N, 'path': TOY_MANIFEST[0], 'sha256': TOY_MANIFEST[1]}
            and auth['resource_preflight']['required'] is True and auth['process_layout'] == E04_PROCESS_LAYOUT
            and auth['authorizes'] == {'s7_toy_acceptance': True, 'production_bank_generation': False,
                                       'storage_directory_creation': False, 'storage_extension_activation': False,
                                       'generator_retraining': False, 'val_test_access': False,
                                       'downstream_evaluation': False}
            and 'production banks' in auth['not_authorized'] and auth['independent_audit']['required'] is True
            and auth['execution']['gpu_uuid'] == GPU_UUID and auth['execution']['python'] == E04_PYTHON,
            {'path': E04_AUTHORITY_PATH, 'sha256': auth_sha})
    G.check('P.e04_authority_derived_from_draft', lambda: draft_raw is not None
            and v.show(E04_PREP_COMMIT, E04_AUTHORITY_DRAFT_PATH) == draft_raw
            and auth == derive_e04_s7_authority(draft_raw, auth['approval'])
            and auth['approval']['owner_authorized'] is True and auth['approval']['scope'] == 'E04 S7 toy-bank acceptance')
    G.check('P.e04_authority_in_execution_commit', lambda: auth_raw is not None
            and v.show(commit, E04_AUTHORITY_PATH) == auth_raw)
    G.check('P.e04_input_allowlist_ruling', lambda: ruling['status'] == 'OWNER_APPROVED' and ruling['approved'] is True
            and ruling['decision'] == E04_RULING_DECISION and v.show(commit, E04_RULING_PATH) == ruling_raw
            and ruling['allowlist_source'] == E04_ALLOWLIST_SOURCE and ruling_raw is not None)
    G.check('P.e04_code_change_scope', lambda: top_level_changes(
        v.show(E04_QUALIFIED_ADAPTER_COMMIT, E04_ADAPTER_PATH).decode(),
        v.show(commit, E04_ADAPTER_PATH).decode()) == E04_ALLOWED_ADAPTER_CHANGES
        and top_level_changes(v.show(E04_PREP_COMMIT, E04_CLI_PATH).decode(), v.show(commit, E04_CLI_PATH).decode())
        == E04_ALLOWED_CLI_CHANGES
        and v.show(E04_PREP_COMMIT, E04_LB_PATH) == v.show(commit, E04_LB_PATH)
        and all(sha256_bytes(v.show(commit, rel)) == h for rel, h in E04_FROZEN_FILES.items()),
        {'adapter_changes': E04_ALLOWED_ADAPTER_CHANGES, 'cli_changes': E04_ALLOWED_CLI_CHANGES})
    G.check('P.execution_checkout', lambda: c['git']['status_porcelain'] == '' and c['git']['branch'] == 'm6-baselines'
            and v.is_ancestor(commit) and all(c['preflights'][str(s)][L]['git']['head'] == commit
                                              and c['preflights'][str(s)][L]['git']['status_porcelain'] == ''
                                              for s, L in E04_PROCS), commit)

    # ---------------- per (seed, run): preflight, lifecycle, provenance, outputs
    for s, L in E04_PROCS:
        pf = (c.get('preflights', {}).get(str(s)) or {}).get(L) or {}
        mk = (c.get('markers', {}).get(str(s)) or {}).get(L) or {}
        run = (runs_all.get(L) or {}).get(str(s)) or {}
        p = run.get('provenance') or {}
        a = p.get('adapter') or {}
        pre = 'P.seed%d.run%s.' % (s, L)
        G.check(pre + 'resources', lambda: pf['processes'] == [] and all(x == '1 0' for x in pf['m8s7_panes'].values())
                and pf['gpu']['uuid'] == GPU_UUID and pf['gpu']['name'] == GPU_NAME and pf['gpu']['driver'] == DRIVER
                and pf['gpu']['free_mib'] >= E04_GPU_FLOOR_MIB and pf['disk']['free_bytes'] >= DISK_FLOOR_GIB << 30
                and pf['memory']['mem_available_bytes'] >= E04_MEM_FLOOR_GIB << 30,
                {'gpu_free_mib': (pf.get('gpu') or {}).get('free_mib'), 'floor_mib': E04_GPU_FLOOR_MIB,
                 'unrelated_gpu_processes': (pf.get('gpu') or {}).get('compute_apps')})
        G.check(pre + 'fresh_run_root', lambda: pf['seed'] == s and pf['label'] == L
                and pf['per_seed']['seed_root'] == str(e04_seed_root(c['s7_root'], s))
                and pf['per_seed']['run_dir_exists'] is False
                and ((L == 'A' and pf['per_seed']['seed_root_exists'] is False)
                     or (L == 'B' and pf['per_seed']['seed_root_exists'] is True
                         and pf['per_seed']['run_A_completed'] is True))
                and pf['s7_root']['path'] == c['s7_root'] and pf['s7_root']['within_runtime_root'] is False
                and pf['s7_root']['within_hdd_bank_root'] is False and pf['utc'] <= mk['launch_time'])
        G.check(pre + 'checkpoints', lambda: all(
            pf['checkpoints'][str(x)]['ok'] and pf['checkpoints'][str(x)]['identity'] == idents[str(x)]
            and pf['checkpoints'][str(x)]['run_id'] == E04_RUN_IDS[x]
            and {k: f['sha256'] for k, f in pf['checkpoints'][str(x)]['files'].items()} == ck_table[x] for x in SEEDS))
        G.check(pre + 'authority_and_inputs', lambda: pf['authority']['ok'] is True
                and pf['authority']['record']['records'] == expected_records
                and pf['authority']['record']['methods'] == [E04] and pf['authority']['record']['mode'] == 'S7_TOY'
                and pf['inputs'] == {'toy_manifest_sha256': TOY_MANIFEST[1],
                                     'feed_pack_manifest_sha256': E04_PACK_MANIFEST_SHA256, 'faces_root_is_dir': True}
                and E04 not in (pf['hdd_bank_root_entries'] or []))
        pr = (pf.get('environment') or {}).get('probe') or {}
        G.check(pre + 'environment_lock', lambda: pf['environment']['present'] is True and pf['environment']['probe_rc'] == 0
                and pr['python'] == E04_PY_VERSION and pr['tensorflow'] == E04_TF and pr['precision_errors'] == []
                and pr['environment_lock']['ok'] is True and pr['environment_lock']['driver']['gpu_uuid'] == GPU_UUID
                and pr['environment_lock']['driver']['observed_driver_version'] == DRIVER
                and pr['environment_lock']['session_config'] == runtime_rec['session_config']
                and pr['frozen_sources']['files'] == E04_FROZEN_FILES and pr['bn_only_difference'] is True
                and pr['contract_sha256'] == E04_CONTRACT_SHA256 and pr['s7_authority']['records'] == expected_records
                and pr['config'] == {'path': E04_CONFIG[0], 'sha256': E04_CONFIG[1]}
                and all(pr['checkpoint_counters'][str(x)]['global_iteration'] == E04_FINAL_ITERATION for x in SEEDS),
                {'probe': {k: pr.get(k) for k in ('python', 'tensorflow', 'precision_errors', 'checkpoint_counters')}})

        pre = 'R.seed%d.run%s.' % (s, L)
        res = (mk.get('pane') or {}).get('s7_result') or {}
        G.check(pre + 'exit_marker_zero', lambda: mk['exit_raw'] == '0\n' and mk['end_time'] != ''
                and mk['launch_time'] != '' and mk['tmux_pane'] in ('1 0', ''), mk.get('exit_raw'))
        G.check(pre + 'command', lambda: mk['command'] is not None
                and e04_command_line(repo_s7, c['s7_root'], s, L) in mk['command']
                and mk['command'].count('tools/learned_bank.py') == 1)
        G.check(pre + 'result_line_and_log', lambda: res['command'] == 's7' and res['method'] == E04
                and res['label'] == L and res['seed'] == s and res['process_seeds'] == [s]
                and sorted(res['results']) == [str(s)] and res['results'][str(s)]['status'] == 'completed'
                and res['results'][str(s)]['rows'] == TOY_N and res['results'][str(s)]['outputs'] == TOY_N
                and res['results'][str(s)]['failure_count'] == 0
                and res['results'][str(s)]['checkpoint_unchanged_after'] is True
                and res['results'][str(s)]['manifest_sha256'] == run['manifest']['sha256']
                and not any(mk['pane']['fail_hits'].values()), (mk.get('pane') or {}).get('fail_hits'))

        pre = 'S.seed%d.run%s.' % (s, L)
        G.check(pre + 'completed', lambda: p['status'] == 'completed' and run['aborted'] is False
                and p['rows'] == TOY_N and p['outputs'] == TOY_N and p['failure_count'] == 0 and p['failures'] == []
                and p['mode'] == 'S7_TOY' and p['method_id'] == E04 and p['bank_seed'] == s
                and p['post_write_verification']['ok'] is True and p['post_write_verification']['files_checked'] == TOY_N
                and p['git_head'] == commit, p.get('status'))
        G.check(pre + 'authorization', lambda: p['authorization'] == {
            'mode': 'S7_TOY', 'methods': [E04], 'seeds': list(SEEDS), 'contract_sha256': E04_CONTRACT_SHA256,
            'label': 's7_toy', 'records': expected_records})
        ck = a.get('checkpoint') or {}
        G.check(pre + 'checkpoint_before', lambda: a['is_fixture'] is False and a['checkpoint_identity_sha256'] == idents[str(s)]
                and ck['checkpoint_identity_sha256'] == idents[str(s)] and ck['method_id'] == E04 and ck['seed'] == s
                and ck['run_id'] == E04_RUN_IDS[s] and ck['rule'] == 'BASELINE_FINAL_STATE_V1'
                and {k: f['sha256'] for k, f in ck['files'].items()} == ck_table[s]
                and all(f['path'] == '%s/runs/m6/E04/seed_%d/%s' % (RUNTIME, s, E04_CKPT_FILES[k])
                        for k, f in ck['files'].items())
                and a['generator_seed'] == a['bank_seed'] == s, a.get('checkpoint_identity_sha256'))
        G.check(pre + 'config_contract_decisions', lambda: a['config_sha256'] == E04_CONFIG[1]
                and a['contract_sha256'] == E04_CONTRACT_SHA256 and a['decisions'] == E04_DECISIONS
                and a['conversion_rule'] == E04_CONVERSION_RULE and a['method_id'] == E04)
        G.check(pre + 'frozen_synthesis_sources', lambda: a['source']['files'] == E04_FROZEN_FILES
                and a['source']['synthesis_lines'] == E04_SYNTHESIS_LINES
                and a['source']['predecessor_reference']['lines'] == E04_PREDECESSOR['lines']
                and a['source']['predecessor_reference']['commit'].startswith(E04_PREDECESSOR['commit']))
        fin = p.get('adapter_finalize') or {}
        G.check(pre + 'inference_graph_bn_mode', lambda: graph_id is not None and a['graph'] == graph_id
                and a['graph']['fused_batch_norm'] == 15 and a['graph']['fused_batch_norm_is_training'] is False
                and a['graph']['update_ops'] == 0 and a['graph']['batch_size'] == 1
                and fin == {'forwards': TOY_N, 'batch_size': 1, 'stochastic_at_inference': False,
                            'auxiliary_outputs_written': False, 'bn_mode': 'inference_moving_statistics'},
                {'graph_equals_fixture_qualified': a.get('graph') == graph_id})
        G.check(pre + 'stored_bn_statistics', lambda: a['coverage'] == fixture['real_checkpoints'][str(s)]['coverage']
                and a['coverage']['restored_variables'] == 102 and a['coverage']['checkpoint_variables'] == 500
                and a['coverage']['restored_names_sha256'] == a['graph']['variable_names_sha256']
                and a['graph']['trainable_variables'] == 72 and a['graph']['bn_moving_statistics'] == 30
                and a['moving_statistics'] == fixture['real_checkpoints'][str(s)]['moving_statistics']
                and a['moving_statistics']['ok'] is True)
        G.check(pre + 'no_training_ops', lambda: a['graph']['gradient_ops'] == 0 and a['graph']['mutating_reachable_ops'] == []
                and a['graph']['random_reachable_ops'] == [] and a['coverage']['not_restored_classes']['optimizer_slots'] == 288
                and fin['forwards'] == TOY_N)
        rd = p.get('reader') or {}
        pop = p.get('population') or {}
        G.check(pre + 'train_only_o_noatime_reader', lambda: rd['reads'] == rd['atime_checked_reads'] == E04_FACE_READS
                and rd['splits_allowlisted'] == ['TRAIN'] and rd['o_noatime'] is True
                and rd['faces_root'] == E04_FACES_ROOT and pop['allowlist_source'] == E04_ALLOWLIST_SOURCE
                and pop['pairs_val_opened'] is False and pop['pack_manifest_sha256'] == E04_PACK_MANIFEST_SHA256
                and pop['relation_source'] == 'e04_feed_pack_v1 pairs_train.json', rd)
        env = p.get('environment') or {}
        G.check(pre + 'environment_and_gpu', lambda: env['python_executable'] == E04_PYTHON
                and env['python_version'] == E04_PY_VERSION and env['cuda_visible_devices'] == '0'
                and env['host'] == 'nd-System-Product-Name' and a['framework'] == 'tensorflow ' + E04_TF
                and a['driver']['gpu_uuid'] == GPU_UUID and a['driver']['observed_driver_version'] == DRIVER
                and a['environment_session_config'] == runtime_rec['session_config'], env)
        man = run.get('manifest') or {}
        G.check(pre + 'manifest', lambda: man['lines'] == TOY_N and man['canonical_lines'] is True
                and man['fields_ok'] is True and man['ends_with_newline'] is True
                and man['sha_file'] == '%s  bank_manifest.jsonl\n' % man['sha256'] and p['manifest_sha256'] == man['sha256'])
        pngs = [x for x in run.get('listing') or [] if x[0].startswith('bank/') and x[0].endswith('.png')]
        G.check(pre + 'outputs_complete_immutable', lambda: len(pngs) == TOY_N and all(x[3] == '0o444' for x in pngs)
                and run['dir_mode'] == '0o555' and run['bank_dir_mode'] == '0o555'
                and sorted(x[0] for x in run['listing'] if x[1] != 'dir') == sorted(
                    ['bank_manifest.jsonl', 'bank_manifest.sha256', 'bank_provenance.json'] + [x[0] for x in pngs])
                and len(run['png_files']) == TOY_N, len(pngs))
        G.check(pre + 'png_independent_verification', lambda: run['png_structures'] == {
            canonical(dict(E04_PNG_STRUCTURE, scanline_filters_ok=True)): TOY_N}
            and all(run['png_files'][x[0][5:]][0] == x[2] and run['png_files'][x[0][5:]][2] == x[1] for x in pngs),
            {'structures': run.get('png_structures')})
        lr = ((c.get('library_reader') or {}).get(str(s)) or {}).get(L) or {}
        G.check(pre + 'library_reader_agrees', lambda: lr['ok'] is True and lr['problems'] == []
                and lr['files_checked'] == TOY_N and lr['failures'] == 0 and lr['provenance_ok'] is True
                and lr['manifest_sha256'] == man['sha256'])

    # ---------------- R: process sequence (one process per seed and run, strictly sequential)
    G.check('R.sequential_independent_processes', lambda: all(
        epoch(c['markers'][str(s)][L]['launch_time']) <= epoch(c['markers'][str(s)][L]['end_time']) for s, L in E04_PROCS)
        and all(epoch(c['markers'][str(x[0])][x[1]]['end_time']) <= epoch(c['markers'][str(y[0])][y[1]]['launch_time'])
                for x, y in zip(E04_PROCS, E04_PROCS[1:]))
        and len({c['markers'][str(s)][L]['command_sha256'] for s, L in E04_PROCS}) == len(E04_PROCS)
        and len({c['preflights_sha256'][str(s)][L] for s, L in E04_PROCS}) == len(E04_PROCS),
        [[c.get('markers', {}).get(str(s), {}).get(L, {}).get(k) for k in ('launch_time', 'end_time')]
         for s, L in E04_PROCS])
    G.check('R.out_root_layout', lambda: c['out_root'] == '%s/%s' % (c['s7_root'], E04)
            and c['out_root_entries'] == ['seed_%d' % s for s in sorted(SEEDS, key=str)]
            and all(c['seed_root_entries'][str(s)] == ['run_A', 'run_B'] for s in SEEDS))
    G.check('R.no_process_left', lambda: c['processes_after'] == [])

    # ---------------- S / L / F per seed
    for s in SEEDS:
        ra, rb_ = ((runs_all.get(L) or {}).get(str(s)) or {} for L in LABELS)
        rows = [dict(zip(MANIFEST_FIELDS_E04, x)) for x in (ra.get('manifest_rows') or [])]
        pre = 'S.seed%d.' % s
        want = [(t['position'], t['pair_id'], t['target_live_id'], t['source_spoof_id'], t['dataset']) for t in toy]
        pa = ((ra.get('provenance') or {}).get('population')) or {}
        G.check(pre + 'toy_population', lambda: sha256_bytes(toy_raw) == TOY_MANIFEST[1] and len(toy) == TOY_N
                and [(r['row_index'], r['pair_id'], r['target_live_id'], r['source_spoof_id'], r['dataset'])
                     for r in rows] == want and len({r['pair_id'] for r in rows}) == TOY_N
                and pa['kind'] == 'S7_TOY_POPULATION' and pa['rows'] == TOY_N and pa['method_id'] == E04 and pa['seed'] == s
                and pa['toy_manifest'] == TOY_MANIFEST[0] and pa['toy_manifest_sha256'] == TOY_MANIFEST[1]
                and pa['row_ids_sha256'] == sha256_bytes('\n'.join(t['pair_id'] for t in toy).encode())
                and pa['first_index'] == toy[0]['position'] and pa['last_index'] == toy[-1]['position'])
        G.check(pre + 'manifest_rows', lambda: len(rows) == TOY_N and all(
            r['schema'] == 'gpat_learned_bank_v1' and r['method_id'] == E04 and r['row_id'] == r['pair_id']
            and r['bank_seed'] == r['generator_seed'] == s and r['mode'] == 'S7_TOY' and r['label'] == 'spoof'
            and r['synthetic_id'] is None and r['deterministic_noise_seed'] is None and r['dsdg_batch_index'] is None
            and r['dsdg_batch_offset'] is None and r['checkpoint_identity_sha256'] == idents[str(s)]
            and r['config_sha256'] == E04_CONFIG[1] and r['contract_sha256'] == E04_CONTRACT_SHA256
            and r['conversion_rule'] == E04_CONVERSION_RULE and r['generation_status'] == 'success'
            and r['failure_reason'] is None and r['output_path'] == 'bank/%s.png' % r['pair_id']
            and r['raw_output_finite'] is True and isinstance(r['raw_output_min'], float)
            and isinstance(r['raw_output_max'], float) and r['raw_output_min'] <= r['raw_output_max'] for r in rows))
        G.check(pre + 'manifest_rederives_from_rows', lambda: sha256_bytes(
            ''.join(canonical(r) + '\n' for r in rows).encode('ascii')) == ra['manifest']['sha256'])
        G.check(pre + 'file_and_pixel_hashes', lambda: all(
            ra['png_files'][r['pair_id'] + '.png'][:2] == [r['output_sha256'], r['output_pixel_sha256']]
            and rb_['png_files'][r['pair_id'] + '.png'][:2] == [r['output_sha256'], r['output_pixel_sha256']]
            for r in rows) and len(rows) == TOY_N)
        rp = (c.get('replay') or {}).get(str(s)) or {}
        lbc = ((c.get('library_reader') or {}).get(str(s)) or {}).get('compare_runs') or {}
        G.check(pre + 'bitwise_replay_A_B', lambda: rp['files_A'] == rp['files_B'] == TOY_N and rp['file_set_equal'] is True
                and rp['bytes_differing'] == [] and rp['pixels_differing'] == []
                and rp['pixel_arrays_compared_equal'] == TOY_N and rp['manifest_bytes_equal'] is True
                and rp['manifest_sha_file_equal'] is True
                and set(rp['provenance_differing_keys']) <= set(E04_RUN_INSTANCE_FIELDS)
                and ra['manifest']['sha256'] == rb_['manifest']['sha256'] and ra['png_files'] == rb_['png_files']
                and [x for x in ra['listing'] if x[0].startswith('bank/')] == [x for x in rb_['listing'] if x[0].startswith('bank/')]
                and lbc['bitwise_identical'] is True and lbc['files_differing'] == [],
                {'provenance_differing_keys': rp.get('provenance_differing_keys'),
                 'pixel_arrays_compared_equal': rp.get('pixel_arrays_compared_equal')})
        cc = (c.get('cli_compare') or {}).get(str(s)) or {}

        def cli_ok():
            cli = json.loads(cc['stdout'].strip().splitlines()[-1])
            v_ = cli['seeds'][str(s)]
            return (cc['rc'] == '0\n' and cli['command'] == 's7-compare' and cli['method'] == E04 and cli['seed'] == s
                    and sorted(cli['seeds']) == [str(s)] and v_['replay']['bitwise_identical'] is True
                    and v_['failures'] == [0, 0] and v_['run_A_reader_ok'] is True and v_['run_B_reader_ok'] is True
                    and cc['command'] is not None and e04_compare_line(repo_s7, c['s7_root'], s) in cc['command'])
        G.check(pre + 'cli_s7_compare', cli_ok, cc.get('rc'))
        af = (c.get('checkpoints_after') or {}).get(str(s)) or {}
        G.check(pre + 'checkpoint_unchanged_before_after', lambda: af['ok'] is True and af['identity'] == idents[str(s)]
                and {k: f['sha256'] for k, f in af['files'].items()} == ck_table[s] and all(
                    c['runs'][L][str(s)]['provenance']['adapter']['checkpoint']['files'][k]['mtime_ns']
                    == af['files'][k]['mtime_ns'] == c['preflights'][str(s)][L]['checkpoints'][str(s)]['files'][k]['mtime_ns']
                    for L in LABELS for k in E04_CKPT_FILES))
        lin = c.get('lineage') or {}
        pre = 'L.seed%d.' % s
        G.check(pre + 'pair_lineage_pairs_train_v1', lambda: lin['files_sha256'] == dict(
            (PAIRS_PARQUET, SPLIT_PARQUET, M2_PARQUET)) and lin['pairs_total'] == 8838 and lin['pairs_unique'] is True
            and lin['pairs_stored_in_ascending_order'] is True and lin['pairs_splits'] == ['TRAIN']
            and all(lin['toy_relation'][r['pair_id']] == [r['row_index'], r['target_live_id'], r['source_spoof_id'],
                                                          r['dataset']] for r in rows) and len(rows) == TOY_N)
        G.check(pre + 'face_lineage_split_v1_m2', lambda: len(rows) == TOY_N and all(
            lin['samples'][r[k]] == ['TRAIN', r['dataset'], r[h], 'COMPLETE']
            for r in rows for k, h in (('target_live_id', 'target_face_sha256'), ('source_spoof_id', 'source_face_sha256'))))
        at = (c.get('atime') or {}).get(str(s))
        mA, mB = (c.get('markers', {}).get(str(s)) or {}).get('A') or {}, (c.get('markers', {}).get(str(s)) or {}).get('B') or {}
        pre = 'F.seed%d.' % s
        G.check(pre + 'atime_ledger_zero_changes', lambda: at['same_paths'] is True and at['files_pre'] == at['files_post'] > 0
                and at['pre_utc'] <= mA['launch_time'] and at['post_utc'] >= mB['end_time']
                and all(b['atime_changed'] == 0 and b['mtime_changed'] == 0 and b['missing_post'] == 0
                        for b in at['by_split'].values())
                and {'TRAIN', 'VAL', 'TEST'} <= set(at['by_split']) and at['changed_paths'] == [],
                (at or {}).get('by_split'))
        G.check(pre + 'no_val_test_inputs', lambda: len(rows) == TOY_N and all(
            lin['samples'][r[k]][0] == 'TRAIN' for r in rows for k in ('target_live_id', 'source_spoof_id'))
            and {r['pair_id'] for r in rows} <= {t['pair_id'] for t in toy}
            and all(c['runs'][L][str(s)]['provenance']['population']['pairs_val_opened'] is False for L in LABELS))

    # ---------------- F / X: global firewall, allowlist cross-check, cross-method substitution
    fp = (c.get('lineage') or {}).get('feed_pack_crosscheck') or {}
    G.check('F.feed_pack_allowlist_crosscheck', lambda: fp['manifest_sha256'] == E04_PACK_MANIFEST_SHA256
            and fp['payload_sha_ok'] is True and fp['samples'] == E04_PACK_COUNTS['samples']
            and fp['samples_mismatching_count'] == 0 and fp['pairs'] == E04_PACK_COUNTS['pairs']
            and fp['pairs_equal_pairs_train_v1'] is True and fp['pairs_split'] == 'TRAIN', fp)
    G.check('F.no_e04_storage_created', lambda: E04 not in (c['hdd_bank_root_entries'] or [])
            and all(E04 not in (c['preflights'][str(s)][L]['hdd_bank_root_entries'] or []) for s, L in E04_PROCS))
    G.check('F.collect_did_not_change_roots', lambda: c['run_roots_unchanged_during_collect'] is True)
    G.check('X.no_cross_method_substitution', lambda: c['method'] == E04 and c['layout'] == E04_PROCESS_LAYOUT and all(
        c['runs'][L][str(s)]['provenance']['method_id'] == E04
        and c['runs'][L][str(s)]['dir'] == str(e04_run_dir(c['s7_root'], s, L))
        and c['runs'][L][str(s)]['provenance']['run_dir'] == str(e04_run_dir(c['s7_root'], s, L))
        and 'configs/amendments/learned_bank_s7_launch_authority.yaml'
        not in c['runs'][L][str(s)]['provenance']['authorization']['records']
        and all('/E04/seed_%d/' % s in f['path'] for f in c['runs'][L][str(s)]['provenance']['adapter']['checkpoint']['files'].values())
        for s, L in E04_PROCS))

    failed = G.failed
    global_failed = [g for g in failed if not g.split('.')[1].startswith('seed')]
    per_seed = {}
    for s in SEEDS:
        mine = [g for g in G.results if '.seed%d.' % s in g['gate'] or g['gate'].startswith('S.seed%d.' % s)]
        per_seed[str(s)] = {'gates': len(mine), 'failed': [g['gate'] for g in mine if not g['pass']],
                            'qualified': not global_failed and all(g['pass'] for g in mine)}
    nq = sum(v['qualified'] for v in per_seed.values())
    if not failed:
        status = 'E04_S7_ALL_THREE_SEEDS_QUALIFIED'
    elif 'P.e04_authority_activated' in global_failed:
        status = 'E04_S7_BLOCKED_BY_AUTHORITY'
    elif nq:
        status = 'E04_S7_PARTIALLY_QUALIFIED'
    else:
        status = 'E04_S7_FAILED_NEEDS_OWNER_REVIEW'

    def peak(s, L):
        g = ((c.get('markers', {}).get(str(s)) or {}).get(L) or {}).get('gpu_samples') or {}
        base = {str(x['pid']) for x in (((c.get('preflights', {}).get(str(s)) or {}).get(L) or {}).get('gpu') or {}).get(
            'compute_apps', [])}
        own = {pid: m for pid, m in (g.get('peak_used_mib_by_pid') or {}).items() if pid not in base}
        return {'peak_used_mib_new_pids': own, 'samples': g.get('samples')}

    pf0 = ((c.get('preflights', {}).get('42') or {}).get('A') or {})
    pr0 = ((pf0.get('environment') or {}).get('probe')) or {}
    prov0 = ((runs_all.get('A') or {}).get('42') or {}).get('provenance') or {}
    doc = {
        'schema': 'learned_bank_s7_acceptance_record_e04_v1', 'method': E04, 'seeds': list(SEEDS), 'status': status,
        's7_acceptance_passed': not failed, 'failed_gates': failed, 'gates': G.results,
        'gates_passed': len(G.results) - len(failed), 'gates_total': len(G.results), 'per_seed': per_seed,
        'label': 'S7_TOY (S7_NON_SCIENTIFIC toy acceptance; never a production bank or benchmark evidence)',
        'fidelity': 'CONTROLLED_ADAPTATION (not an official Physics-STD inference reproduction)',
        'authority': {'path': E04_AUTHORITY_PATH, 'sha256': auth_sha}, 'contract': {'path': E04_CONTRACT_PATH,
                                                                                    'sha256': contract_sha},
        'family_closure': {'path': E04_CLOSURE_PATH, 'sha256': closure_sha},
        'input_allowlist_ruling': {'path': E04_RULING_PATH,
                                   'sha256': sha256_bytes(ruling_raw) if ruling_raw else None},
        'execution_commit': commit,
        'checkpoints': {str(s): {'run_id': E04_RUN_IDS[s], 'iteration': E04_FINAL_ITERATION,
                                 'files': {E04_CKPT_FILES[k]: h for k, h in ck_table.get(s, {}).items()},
                                 'checkpoint_identity_sha256': idents.get(str(s))} for s in SEEDS},
        'outputs': {str(s): {
            'rows': TOY_N,
            'manifest_sha256': {L: (((runs_all.get(L) or {}).get(str(s)) or {}).get('manifest') or {}).get('sha256') for L in LABELS},
            'png_files_sha256': {L: sha256_bytes(canonical(((runs_all.get(L) or {}).get(str(s)) or {}).get('png_files')).encode())
                                 for L in LABELS},
            'run_dirs': {L: ((runs_all.get(L) or {}).get(str(s)) or {}).get('dir') for L in LABELS},
            'bitwise_identical': per_seed[str(s)]['qualified'] and not any('bitwise' in g for g in failed)}
            for s in SEEDS},
        'profiling': {'gpu_memory': {str(s): {L: peak(s, L) for L in LABELS} for s in SEEDS},
                      'wall_time_seconds': {str(s): {L: ((((runs_all.get(L) or {}).get(str(s)) or {}).get('provenance') or {})
                                                         .get('wall_time_seconds')) for L in LABELS} for s in SEEDS},
                      'note': 'TF1 reports no allocator peak; nvidia-smi samples every 2 s per process, new PIDs only'},
        'reproduction': {
            'method_id': E04, 'seeds': list(SEEDS), 'process_layout': E04_PROCESS_LAYOUT,
            'execution_commit': commit, 'branch': 'm6-baselines', 'working_directory': repo_s7,
            'authority': E04_AUTHORITY_PATH, 'config': {'path': E04_CONFIG[0], 'sha256': E04_CONFIG[1]},
            'environment': {'name': 'gpat-m6-e03', 'python': E04_PYTHON, 'python_version': pr0.get('python'),
                            'tensorflow': pr0.get('tensorflow'),
                            'lock': {'path': E04_ENV_LOCK[0], 'sha256': E04_ENV_LOCK[1]},
                            'native_runtime': (pr0.get('environment_lock') or {}).get('native_runtime'),
                            'package_versions': (pr0.get('environment_lock') or {}).get('package_versions'),
                            'precision_env': dict(E04_PRECISION_ENV),
                            'run_environment': prov0.get('environment')},
            'gpu': {'uuid': (pf0.get('gpu') or {}).get('uuid'), 'name': (pf0.get('gpu') or {}).get('name'),
                    'driver': (pf0.get('gpu') or {}).get('driver'),
                    'driver_policy': ((pr0.get('environment_lock') or {}).get('driver') or {}).get('driver_policy')},
            'bn_mode': 'tf.layers.batch_normalization(training=False): restored moving statistics, batch 1 (E04-Q1)',
            'geometry': {'feed_pack_manifest_sha256': E04_PACK_MANIFEST_SHA256,
                         'offsets': 'dense_offsets(source_xy=spoof Q140, target_xy=live Q140), scipy 1.10.1 (D2)'},
            'warp': {'implementation': 'methods/physics_std/trace_transfer.py warp (CPU, bilinear, border clamp)',
                     'sha256': E04_FROZEN_FILES['methods/physics_std/trace_transfer.py']},
            'synthesis': {'equation': 'out = (1-P_w)*(I_live + B~_w + C~_w + T_w) + P_w*I_source_w (E04-Q2)',
                          'training_graph_lines': E04_SYNTHESIS_LINES,
                          'training_graph_sha256': E04_FROZEN_FILES['methods/physics_std/training_graph.py']},
            'conversion_rule': E04_CONVERSION_RULE,
            'png': 'lossless PNG, PIL compress_level 6, optimize False, no metadata chunks (IHDR/IDAT/IEND only)',
            'population': {'toy_manifest': TOY_MANIFEST[0], 'sha256': TOY_MANIFEST[1], 'pairs': TOY_N,
                           'images': TOY_N * len(LABELS) * len(SEEDS)},
            'commands': {str(s): {L: ((c.get('markers', {}).get(str(s)) or {}).get(L) or {}).get('command') for L in LABELS}
                         for s in SEEDS},
            'compare_commands': {str(s): ((c.get('cli_compare') or {}).get(str(s)) or {}).get('command') for s in SEEDS},
            'windows': {str(s): {L: [((c.get('markers', {}).get(str(s)) or {}).get(L) or {}).get(k)
                                     for k in ('launch_time', 'end_time')] for L in LABELS} for s in SEEDS},
            'exit_status': {str(s): {L: ((c.get('markers', {}).get(str(s)) or {}).get(L) or {}).get('exit_raw')
                                     for L in LABELS} for s in SEEDS},
            'preflights_sha256': c.get('preflights_sha256'),
            'output_root': c.get('out_root'), 's7_root': c.get('s7_root'),
            'atime_ledgers': {str(s): {k: ((c.get('atime') or {}).get(str(s)) or {}).get(k)
                                       for k in ('pre_utc', 'post_utc', 'pre_sha256', 'post_sha256')} for s in SEEDS},
            'host_scripts': 'tools/e04_bank_scripts/e04_s7_{preflight,launch,post,collect}.sh',
            'known_disclosures': [
                'CONTROLLED_ADAPTATION: bank inference uses the stored BN moving statistics (batch 1); training-time '
                'synthesis used 8-image batch statistics; not official Physics-STD test-time behaviour',
                'clip-truncate float->uint8 follows the ECCV20-STDN predecessor (reference only)',
                'TRAIN allowlist from the pinned e04_feed_pack_v1 in the inference process (owner ruling '
                'e04_s7_input_allowlist_ruling.yaml); split_v1 / M2 cross-check performed by this auditor',
                'S7 outputs live only on the GPU host S7 root; git holds hashes, not image or checkpoint bytes',
                'toy population (186 pairs per seed); production equivalence is checked by a later production audit']},
        'collected': c,
    }
    return doc


def report_e04(doc):
    failed = doc['failed_gates']
    r = doc['reproduction']

    def seed_runs(s, L):
        return 'PASS' if all(g['pass'] for g in doc['gates'] if ('.seed%s.run%s.' % (s, L)) in g['gate']) else 'FAIL'
    rows = ''.join('| %s | %s | %s | %s | %s | %s |\n' % (
        s, seed_runs(s, 'A'), seed_runs(s, 'B'),
        'identical' if doc['outputs'][s]['bitwise_identical'] else 'NOT VERIFIED',
        '%d gates, %d failed' % (v['gates'], len(v['failed'])), 'QUALIFIED' if v['qualified'] else 'NOT QUALIFIED')
        for s, v in sorted(doc['per_seed'].items(), key=lambda x: int(x[0])))
    win = '; '.join('seed %s run %s %s → %s exit %s' % (s, L, r['windows'][s][L][0], r['windows'][s][L][1],
                                                         (r['exit_status'][s][L] or 'missing').strip())
                    for s in sorted(r['windows'], key=int) for L in ('A', 'B'))
    return '\n'.join([
        '# M8 learned-bank S7 toy acceptance: E04 Physics-STD', '',
        '**Status:** `%s`, %d/%d gates passing; `s7_acceptance_passed` = %s.' % (
            doc['status'], doc['gates_passed'], doc['gates_total'], str(doc['s7_acceptance_passed']).lower()), '',
        'Label: %s. Fidelity: %s.' % (doc['label'], doc['fidelity']), '',
        'Authority `%s` (`%s`), contract `%s`, family closure `%s`, input-allowlist ruling `%s`, execution commit `%s`.'
        % (doc['authority']['path'], doc['authority']['sha256'], doc['contract']['sha256'],
           doc['family_closure']['sha256'], doc['input_allowlist_ruling']['sha256'], doc['execution_commit']), '',
        '| seed | run A | run B | A/B replay | audit | final |', '|---|---|---|---|---|---|', rows.rstrip('\n'), '',
        'Failed gates: %s' % (', '.join('`%s`' % g for g in failed) if failed else 'none.'), '',
        '## Execution', '',
        '- Environment `%s` (`%s`, Python %s, TensorFlow %s), GPU %s `%s`, driver %s (%s).' % (
            r['environment']['name'], r['environment']['python'], r['environment']['python_version'],
            r['environment']['tensorflow'], r['gpu']['name'], r['gpu']['uuid'], r['gpu']['driver'],
            r['gpu']['driver_policy']),
        '- One OS process per seed and run, strictly sequential: %s.' % win,
        '- Population: %d toy TRAIN pairs (`%s`) per seed and run; %d images in total.' % (
            r['population']['pairs'], r['population']['sha256'], r['population']['images']),
        '- BN: %s.' % r['bn_mode'],
        '- Synthesis: %s; conversion `%s`; %s.' % (r['synthesis']['equation'], r['conversion_rule'], r['png']),
        '- Output root (GPU host): `%s`.' % r['output_root'],
        '- Manifests (run A = run B when replay passed): %s.' % '; '.join(
            'seed %s `%s`' % (s, (v['manifest_sha256'].get('A') or 'missing')) for s, v in sorted(
                doc['outputs'].items(), key=lambda x: int(x[0]))),
        '- GPU memory (nvidia-smi, new PIDs): %s.' % json.dumps(doc['profiling']['gpu_memory'], sort_keys=True),
        '', '## Disclosures', ''] + ['- %s' % x for x in r['known_disclosures']] + [
        '', '## Scope', '',
        'S7 establishes E04 readiness for a separately authorized production launch only. No production bank, storage '
        'directory, retraining, downstream training, TEST evaluation, DINOv3 or GPAT work was performed.', ''])


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('preflight', 'collect'):
        a = sub.add_parser(name)
        a.add_argument('--method', required=True, choices=sorted(ESTIMATE_GIB) + [E04])
        a.add_argument('--repo', required=True)
        a.add_argument('--s7-root', required=True)
        a.add_argument('--tag', default=None)
        a.add_argument('--per-seed', action='store_true')
        if name == 'preflight':
            a.add_argument('--seed', type=int, choices=SEEDS, default=None)
            a.add_argument('--label', choices=LABELS, default=None)
    b = sub.add_parser('build')
    b.add_argument('--collected', required=True)
    b.add_argument('--repo-root', default=str(Path(globals().get('__file__', 'tools/x')).resolve().parents[1]))
    args = ap.parse_args(argv)
    global TAG, PER_SEED
    TAG = getattr(args, 'tag', None)
    PER_SEED = bool(getattr(args, 'per_seed', False))
    if getattr(args, 'method', None) == E04:          # E04: fixed per-seed layout and marker names, no --tag
        if TAG or PER_SEED:
            ap.error('E04 uses its own per-seed layout; --tag / --per-seed are not accepted')
        if args.cmd == 'preflight':
            if args.seed is None or args.label is None:
                ap.error('E04 preflight needs --seed and --label')
            return preflight_e04(args.repo, args.s7_root, args.seed, args.label)
        return collect_e04(args.repo, args.s7_root)
    if PER_SEED and not TAG:
        ap.error('--per-seed needs --tag')
    if args.cmd == 'preflight':
        if PER_SEED and (args.seed is None or args.label is None):
            ap.error('--per-seed preflight needs --seed and --label')
        return preflight(args.method, args.repo, args.s7_root, args.seed if PER_SEED else None,
                         args.label if PER_SEED else None)
    if args.cmd == 'collect':
        return collect(args.method, args.repo, args.s7_root)
    c = json.loads(Path(args.collected).read_text())
    is_e04 = c.get('method') == E04
    doc = build_e04(c, args.repo_root) if is_e04 else build(c, args.repo_root)
    root = Path(args.repo_root)
    ev = evidence_paths(doc['method'])
    (root / ev['record']).write_text(json.dumps(doc, indent=1, sort_keys=True) + '\n')
    (root / ev['report']).write_text(report_e04(doc) if is_e04 else report(doc))
    (root / ev['sha256']).write_text(''.join('%s  %s\n' % (sha256_file(root / ev[k]), ev[k]) for k in ('record', 'report')))
    print(doc['status'], doc['gates_passed'], '/', doc['gates_total'], doc['failed_gates'])
    return 0 if doc['s7_acceptance_passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
