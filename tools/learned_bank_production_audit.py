"""M8 learned-bank PRODUCTION audit (E06c / E03 / E06b / E05), one bank per (method, seed), under the per-method owner
authority configs/amendments/learned_bank_production_<M>.yaml, the generation contract and the HDD storage authority.

Phases (preflight and collect write nothing but stdout; GPU host, fed on stdin, library imported from the clean
execution clone --repo, a Python with numpy, Pillow and pyarrow -- gpat-m6-e05 -- and CUDA_VISIBLE_DEVICES=""):

  preflight --method M --seed S --repo <clone>   immediately before launch: no E04 / learned-bank process and no live
            m8prod_* / m8s7_* pane, GPU UUID/driver, free GPU memory >= estimate + 2 GiB (unrelated processes recorded,
            never touched), RAM, live HDD storage preflight (UUID, mount, ext4, rw, owner, 0700, no unexpected entries),
            free space and inodes, the registered bank directory absent, the state of every existing learned bank, all
            12 approved checkpoints re-hashed, the production authority loads (live storage), method environment.
  collect   --method M --seed S --repo <clone>   after the process exited and the post atime ledger was taken:
            provenance, the library reader (lb.read_bank, full pixel verification), an INDEPENDENT re-verification of
            every output (own file walk, sha256, PIL decode, geometry, pixel sha256, modes), descriptive image statistics
            (no threshold), S7-to-production byte equivalence of the 186 approved toy rows (runs A and B), checkpoints
            re-hashed, wrapper markers and pane log, the stat-only atime ledger by split, storage after, every other
            learned bank and the six E01/E02 production manifests unchanged.
  build     --collected <evidence>/collected.json   (laptop) gates everything against the committed authority and the
            committed gzip of the bank manifest (<evidence>/bank_manifest.jsonl.gz), writes
            outputs/audit/LEARNED_BANK_PRODUCTION_<M>_SEED<S>.{json,md} + _SHA256.txt.
  close     --method M                            (laptop) family closure from the three passed bank records:
            outputs/audit/LEARNED_BANK_PRODUCTION_<M>_FAMILY_CLOSURE.{json,md} + _SHA256.txt.

Evidence directory per bank: outputs/audit/LEARNED_BANK_PRODUCTION/<M>_s<S>/ (collected.json, bank_manifest.jsonl.gz,
preflight.json). Host markers: ~/workdir/m8prod_<M>_s<S>.{command,launch_time,exit,end_time,pane.log},
~/workdir/m8prod_<M>_s<S>_preflight.json, ~/workdir/m8prod_<M>_s<S>_atime_{pre,post}.{tsv,time}.
"""
import argparse
import gzip
import hashlib
import io
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

WORKDIR = '/home/student20261/workdir'
RUNTIME = WORKDIR + '/GPAT_TransferBench_runtime'
FACES = RUNTIME + '/data/processed/faces_256'
BANK_ROOT = '/media/ext_data/student20261_gpat_m8_banks'
HDD_UUID = '608647d2-9cb0-45b0-b999-73c080ba909b'
GPU_UUID = 'GPU-b722cd9d-c9fa-223c-915e-cc09d9b862b7'
GPU_NAME = 'NVIDIA GeForce RTX 3090'
DRIVER = '580.178.04'
SEEDS = (42, 1337, 2026)
METHOD_ORDER = ('E06c', 'E03', 'E06b', 'E05')
N = 8838
TOY_N = 186
ESTIMATE_GIB = {'E06c': 4, 'E06b': 4, 'E05': 3, 'E03': 2}
MARGIN_GIB = 2
MEM_FLOOR_GIB = 4
DISK_FLOOR_GIB = 20
INODE_FLOOR = 100000
FACE_READS = {'E03': 2 * N, 'E05': 2 * N, 'E06b': 0, 'E06c': 0}
AUTHORITY = 'configs/amendments/learned_bank_production_{M}.yaml'
AUTHORITY_SCHEMA = 'learned_bank_production_authority_v1'
EVIDENCE = 'outputs/audit/LEARNED_BANK_PRODUCTION/{M}_s{S}'
RECORD = 'outputs/audit/LEARNED_BANK_PRODUCTION_{M}_SEED{S}'
CLOSURE = 'outputs/audit/LEARNED_BANK_PRODUCTION_{M}_FAMILY_CLOSURE'
# the only paths an execution commit may change relative to the authority's bound source commit
ALLOWED_EXEC_DIFF = (re.compile(r'^configs/amendments/learned_bank_production_(E03|E05|E06b|E06c)\.yaml$'),
                     re.compile(r'^outputs/audit/'))
FAIL_PATTERNS = ('Traceback', 'Error', 'error:', 'out of memory', 'Killed', 'Segmentation fault', 'core dumped',
                 'TrainingForbidden', 'LearnedBank')
PY_ENV = ('PATH=/usr/bin:/bin LANG=C.UTF-8 CUDA_VISIBLE_DEVICES=0 NVIDIA_TF32_OVERRIDE=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 '
          'PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1')
E03_ENV = ('PATH=/usr/bin:/bin LANG=C.UTF-8 CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0 NVIDIA_TF32_OVERRIDE=0 '
           'PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 TF_CUDNN_DETERMINISTIC=1 TF_DETERMINISTIC_OPS=1 '
           'TF_ENABLE_AUTO_MIXED_PRECISION=0 TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE=0 '
           'TF_ENABLE_CUBLAS_TENSOR_OP_MATH_FP32=0 TF_ENABLE_CUDNN_RNN_TENSOR_OP_MATH_FP32=0 '
           'TF_ENABLE_CUDNN_TENSOR_OP_MATH_FP32=0 TF_ENABLE_ONEDNN_OPTS=0 TF_XLA_FLAGS=--tf_xla_auto_jit=0')


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


def tag(M, S):
    return 'm8prod_%s_s%d' % (M, S)


def marker_paths(M, S):
    b = '%s/%s' % (WORKDIR, tag(M, S))
    out = {k: '%s.%s' % (b, k) for k in ('command', 'launch_time', 'exit', 'end_time', 'pane.log')}
    out.update(preflight=b + '_preflight.json', atime_pre=b + '_atime_pre.tsv', atime_post=b + '_atime_post.tsv')
    return out


def run_id(M, S, exec_commit, authority_sha):
    return sha256_bytes(('GPAT_TRANSFERBENCH|LEARNED_BANK_PRODUCTION|%s|%d|%s|%s' % (M, S, exec_commit, authority_sha))
                        .encode())[:16]


def e01_e02_expected(repo):
    out = {}
    for m in ('E01', 'E02'):
        d = json.loads((Path(repo) / ('outputs/audit/%s_PRODUCTION_FAMILY_CLOSURE.json' % m)).read_text())
        for s, v in d['per_seed'].items():
            out['%s|%s' % (m, s)] = v['bank_manifest_sha256']
    return out


def lb_module(repo):
    sys.path.insert(0, str(repo))
    os.chdir(str(repo))
    from methods.common import learned_bank as lb
    return lb


def git_state(repo):
    return {'head': sh('git', 'rev-parse', 'HEAD', cwd=repo), 'branch': sh('git', 'branch', '--show-current', cwd=repo),
            'status_porcelain': sh('git', 'status', '--porcelain', '--untracked-files=all', cwd=repo)}


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


def bank_panes():
    out = {}
    for s in sh('tmux', 'ls', '-F', '#{session_name}').splitlines():
        if s.startswith('m8prod_') or s.startswith('m8s7_'):
            out[s] = sh('tmux', 'list-panes', '-t', s, '-F', '#{pane_dead} #{pane_dead_status}')
    return out


def mem_available_bytes():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    return None


def read_text(path):
    p = Path(path)
    return p.read_text() if p.is_file() else None


def bank_registry_state(lb):
    """Every learned bank directory under the HDD root: manifest sha256, provenance/aborted flags (read-only)."""
    out = {}
    root = Path(BANK_ROOT)
    for m in sorted(os.listdir(root)):
        for s in sorted(os.listdir(root / m)):
            d = root / m / s
            man = d / lb.MANIFEST
            out['%s/%s' % (m, s)] = {'manifest_sha256': sha256_file(man) if man.is_file() else None,
                                     'provenance': (d / lb.PROVENANCE).is_file(),
                                     'aborted': (d / ('ABORTED_' + lb.PROVENANCE)).exists(),
                                     'mode': oct(d.stat().st_mode & 0o7777)}
    return out


def e01_e02_state(repo):
    out = {}
    for key in e01_e02_expected(repo):
        m, s = key.split('|')
        p = Path(RUNTIME) / 'runs' / 'm6' / m / ('seed_%s' % s) / 'bank_manifest.parquet'
        out[key] = sha256_file(p) if p.is_file() else None
    return out


def storage_detail(lb):
    pf = lb.storage_preflight()
    vfs = os.statvfs(BANK_ROOT) if pf.get('exists') else None
    pf['capacity'] = None if vfs is None else {
        'total_bytes': vfs.f_blocks * vfs.f_frsize, 'free_bytes': vfs.f_bavail * vfs.f_frsize,
        'used_bytes': (vfs.f_blocks - vfs.f_bfree) * vfs.f_frsize, 'free_inodes': vfs.f_favail,
        'total_inodes': vfs.f_files}
    return pf


def checkpoints_all(lb):
    ck = {}
    for (m, s) in sorted(lb.CHECKPOINTS):
        try:
            r = lb.verify_checkpoint(m, s, Path(RUNTIME) / 'runs' / 'm6' / m / ('seed_%d' % s))
            ck['%s|%d' % (m, s)] = {'ok': True, 'identity': r['checkpoint_identity_sha256'],
                                     'files': {k: {'sha256': v['sha256'], 'size': v['size'], 'mtime_ns': v['mtime_ns']}
                                               for k, v in r['files'].items()}}
        except Exception as exc:                 # recorded, gated in build
            ck['%s|%d' % (m, s)] = {'ok': False, 'error': '%s: %s' % (type(exc).__name__, exc)}
    return ck


# ------------------------------------------------------------------------------------------------------- preflight
def preflight(M, S, repo):
    lb = lb_module(repo)
    bank = Path(BANK_ROOT) / M / ('seed_%d' % S)
    doc = {'phase': 'preflight', 'method': M, 'seed': S, 'utc': utc_now(), 'repo': str(repo), 'git': git_state(repo),
           'collector': {'python': sys.version.split()[0], 'executable': sys.executable},
           'processes': processes(), 'bank_panes': bank_panes(), 'gpu': gpu_state(),
           'memory': {'mem_available_bytes': mem_available_bytes(), 'floor_bytes': MEM_FLOOR_GIB << 30},
           'estimate_gib': ESTIMATE_GIB[M], 'margin_gib': MARGIN_GIB,
           'storage': storage_detail(lb), 'bank_dir': str(bank), 'bank_dir_exists': os.path.lexists(str(bank)),
           'method_dir_exists': (Path(BANK_ROOT) / M).is_dir(), 'registry': bank_registry_state(lb),
           'e01_e02_manifests': e01_e02_state(repo), 'checkpoints': checkpoints_all(lb),
           'markers_absent': {k: not os.path.lexists(v) for k, v in marker_paths(M, S).items() if k != 'preflight'},
           'tmux_session_absent': subprocess.run(['tmux', 'has-session', '-t', tag(M, S)],
                                                 capture_output=True).returncode != 0}
    try:
        a = lb.load_production_authority(M, S)
        doc['authority'] = {'ok': True, 'record': a.record()}
    except Exception as exc:
        doc['authority'] = {'ok': False, 'error': '%s: %s' % (type(exc).__name__, exc)}
    env = lb.ENVIRONMENTS[M]
    probe = ('import sys; import importlib; m = "tensorflow" if %r == "E03" else "torch"; t = importlib.import_module(m);'
             ' print(sys.version.split()[0], m, t.__version__)' % M)
    r = subprocess.run(['env', '-i', 'HOME=' + os.environ.get('HOME', ''), 'PATH=/usr/bin:/bin', 'CUDA_VISIBLE_DEVICES=',
                        'PYTHONNOUSERSITE=1', 'TF_CPP_MIN_LOG_LEVEL=3', env['python'], '-I', '-B', '-c', probe],
                       capture_output=True, text=True)
    doc['environment'] = {'name': env['name'], 'python': env['python'], 'present': Path(env['python']).is_file(),
                          'probe_rc': r.returncode, 'probe': r.stdout.strip()}
    print(json.dumps(doc, sort_keys=True, default=str))


# ------------------------------------------------------------------------------------------------------- collect
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
            if d.get('command') == 'production':
                result = d
    return {'lines': len(lines), 'sha256': sha256_file(p), 'production_result': result,
            'fail_hits': {k: sum(k in x for x in lines) for k in FAIL_PATTERNS},
            'tail': [x for x in lines if x.strip()][-3:]}


def atime_ledger(M, S, repo):
    import pyarrow.parquet as pq
    p = marker_paths(M, S)

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
        sp = split.get(os.path.basename(path)[:-4], 'UNMAPPED')
        s = by.setdefault(sp, {'files': 0, 'atime_changed': 0, 'mtime_changed': 0, 'missing_post': 0})
        s['files'] += 1
        if path not in post:
            s['missing_post'] += 1
            continue
        s['atime_changed'] += post[path][0] != a
        s['mtime_changed'] += post[path][1] != m
    return {'pre_utc': (read_text(p['atime_pre'][:-4] + '.time') or '').strip(),
            'post_utc': (read_text(p['atime_post'][:-4] + '.time') or '').strip(),
            'pre_sha256': sha256_file(p['atime_pre']), 'post_sha256': sha256_file(p['atime_post']),
            'files_pre': len(pre), 'files_post': len(post), 'same_paths': sorted(pre) == sorted(post), 'by_split': by}


def independent_verification(lb, d, rows):
    """Own walk of the bank directory (not lb.read_bank): every manifest file exists once, is a regular 0444 file, its
    bytes hash to output_sha256, PIL decodes a 256x256 RGB PNG whose uint8 pixels hash to output_pixel_sha256; nothing
    else is present. Also descriptive statistics (no threshold)."""
    import numpy as np
    from PIL import Image
    bank = d / lb.BANK_DIR
    names = sorted(os.listdir(bank))
    want = {r['output_path'].split('/', 1)[1]: r for r in rows if r['generation_status'] == 'success'}
    problems, stds, means, mins, maxs = [], [], [], [], []
    pix_seen, coarse_seen = {}, {}
    total_bytes = 0
    modes = set()
    for name in names:
        p = bank / name
        st = os.lstat(p)
        modes.add(oct(st.st_mode & 0o7777))
        if name not in want or not p.is_file() or p.is_symlink():
            problems.append('unexpected entry %s' % name)
            continue
        r = want[name]
        raw = p.read_bytes()
        total_bytes += len(raw)
        if sha256_bytes(raw) != r['output_sha256']:
            problems.append('%s: file sha256' % name)
        with Image.open(io.BytesIO(raw)) as im:
            fmt, mode, size = im.format, im.mode, im.size
            a = np.array(im)
        if (fmt, mode, size) != ('PNG', 'RGB', (256, 256)) or a.dtype != np.uint8 or a.shape != (256, 256, 3):
            problems.append('%s: geometry %s %s %s' % (name, fmt, mode, size))
        ph = sha256_bytes(('%s|%s|' % (a.dtype.str, 'x'.join(map(str, a.shape)))).encode() + a.tobytes())
        if ph != r['output_pixel_sha256']:
            problems.append('%s: pixel sha256' % name)
        f = a.astype(np.float64)
        stds.append(float(f.std()))
        means.append(float(f.mean()))
        mins.append(int(a.min()))
        maxs.append(int(a.max()))
        pix_seen.setdefault(ph, []).append(r['row_id'])
        lum = (0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]).reshape(8, 32, 8, 32).mean(axis=(1, 3))
        coarse_seen.setdefault(np.rint(lum).astype(np.int16).tobytes(), []).append(r['row_id'])
    missing = sorted(set(want) - set(names))
    if missing:
        problems.append('missing %d outputs, e.g. %s' % (len(missing), missing[:3]))

    def q(xs, k):
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(round((len(xs) - 1) * k)))] if xs else None
    dup = [v for v in pix_seen.values() if len(v) > 1]
    coarse = [v for v in coarse_seen.values() if len(v) > 1]
    return {'files': len(names), 'files_verified': len(names) - len([x for x in problems if 'unexpected' in x]),
            'problems': problems[:50], 'problem_count': len(problems), 'modes': sorted(modes),
            'total_png_bytes': total_bytes,
            'stats': {'per_image_std': {k: q(stds, v) for k, v in (('min', 0), ('p05', .05), ('median', .5),
                                                                     ('p95', .95), ('max', 1))},
                      'per_image_mean': {k: q(means, v) for k, v in (('min', 0), ('median', .5), ('max', 1))},
                      'per_image_min_median': q(mins, .5), 'per_image_max_median': q(maxs, .5),
                      'distinct_pixel_sha256': len(pix_seen), 'exact_duplicate_groups': len(dup),
                      'exact_duplicate_rows': sum(len(v) for v in dup),
                      'coarse_8x8_luma_signature_groups': len(coarse),
                      'coarse_8x8_luma_signature_rows': sum(len(v) for v in coarse),
                      'note': 'descriptive only; coarse signature = 8x8 block-mean luminance rounded to integers'}}


def s7_equivalence(lb, repo, M, S, d, rows):
    rec = json.loads((Path(repo) / ('outputs/audit/LEARNED_BANK_S7_%s.json' % M)).read_text())
    o = rec['outputs'][str(S)]
    by_id = {r['row_id']: r for r in rows}
    out = {'s7_record_sha256': sha256_file(Path(repo) / ('outputs/audit/LEARNED_BANK_S7_%s.json' % M)),
           's7_manifest_sha256_recorded': o['manifest_sha256'], 'runs': {}}
    for L, key in (('A', 'run_A_dir'), ('B', 'run_B_dir')):
        sd = Path(o[key])
        body = (sd / lb.MANIFEST).read_bytes()
        srows = [json.loads(x) for x in body.decode('ascii').splitlines()]
        field_mm, byte_mm, missing = [], [], []
        for r in srows:
            pr = by_id.get(r['row_id'])
            if pr is None:
                missing.append(r['row_id'])
                continue
            diff = sorted(k for k in r if k != 'mode' and r[k] != pr.get(k))
            if diff or r['mode'] != 'S7_TOY' or pr['mode'] != 'PRODUCTION' or set(r) != set(pr):
                field_mm.append([r['row_id'], diff])
            if (sd / r['output_path']).read_bytes() != (d / pr['output_path']).read_bytes():
                byte_mm.append(r['row_id'])
        out['runs'][L] = {'dir': str(sd), 'manifest_sha256': sha256_bytes(body), 'rows': len(srows),
                          'rows_matched': len(srows) - len(missing), 'missing_in_production': missing[:20],
                          'field_mismatches': field_mm[:20], 'field_mismatch_count': len(field_mm),
                          'byte_mismatches': byte_mm[:20], 'byte_mismatch_count': len(byte_mm),
                          'row_indices': [r['row_index'] for r in srows]}
    return out


def listing_summary(d, lb):
    top = sorted(os.listdir(d))
    return {'entries': top, 'dir_mode': oct(d.stat().st_mode & 0o7777),
            'bank_dir_mode': oct((d / lb.BANK_DIR).stat().st_mode & 0o7777),
            'top_files': {n: {'size': (d / n).stat().st_size, 'sha256': sha256_file(d / n),
                              'mode': oct((d / n).stat().st_mode & 0o7777)} for n in top if (d / n).is_file()}}


def collect(M, S, repo):
    lb = lb_module(repo)
    d = Path(BANK_ROOT) / M / ('seed_%d' % S)
    before = {'listing': listing_summary(d, lb), 'bank_names_sha256': sha256_bytes(
        '\n'.join(sorted(os.listdir(d / lb.BANK_DIR))).encode())}
    doc = {'phase': 'collect', 'method': M, 'seed': S, 'utc': utc_now(), 'repo': str(repo), 'git': git_state(repo),
           'collector': {'python': sys.version.split()[0], 'executable': sys.executable}, 'bank_dir': str(d)}
    prov_p = d / lb.PROVENANCE
    doc['provenance'] = json.loads(prov_p.read_text()) if prov_p.is_file() else None
    doc['aborted'] = (d / ('ABORTED_' + lb.PROVENANCE)).exists()
    rb = lb.read_bank(d, expect_method=M, expect_seed=S, expect_mode='PRODUCTION', expect_rows=N,
                      expect_checkpoint_identity=lb.checkpoint_identity_sha256(M, S))
    rows = rb.pop('rows')
    doc['read_bank'] = rb
    doc['listing'] = before['listing']
    doc['independent'] = independent_verification(lb, d, rows)
    doc['s7_equivalence'] = s7_equivalence(lb, repo, M, S, d, rows)
    doc['checkpoints_after'] = checkpoints_all(lb)
    mp = marker_paths(M, S)
    doc['markers'] = {'command': read_text(mp['command']),
                      'command_sha256': sha256_file(mp['command']) if Path(mp['command']).is_file() else None,
                      'launch_time': (read_text(mp['launch_time']) or '').strip(), 'exit_raw': read_text(mp['exit']),
                      'end_time': (read_text(mp['end_time']) or '').strip(), 'pane': pane_summary(mp['pane.log']),
                      'tmux_pane': sh('tmux', 'list-panes', '-t', tag(M, S), '-F', '#{pane_dead} #{pane_dead_status}')}
    pf = Path(mp['preflight'])
    doc['preflight'] = json.loads(pf.read_text()) if pf.is_file() else None
    doc['preflight_sha256'] = sha256_file(pf) if pf.is_file() else None
    doc['atime'] = atime_ledger(M, S, repo)
    doc['storage_after'] = storage_detail(lb)
    doc['registry_after'] = bank_registry_state(lb)
    doc['e01_e02_manifests_after'] = e01_e02_state(repo)
    doc['processes_after'] = processes()
    doc['gpu_after'] = gpu_state()
    doc['bank_unchanged_during_collect'] = (listing_summary(d, lb) == before['listing'] and sha256_bytes(
        '\n'.join(sorted(os.listdir(d / lb.BANK_DIR))).encode()) == before['bank_names_sha256'])
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


def git(repo, *a):
    return subprocess.run(['git'] + list(a), cwd=repo, capture_output=True)


def exec_commit_of(repo, M):
    """The commit that added the method's production authority (exactly one, on this branch)."""
    out = git(repo, 'log', '--format=%H', '--diff-filter=A', '--', AUTHORITY.format(M=M)).stdout.decode().split()
    return out[0] if len(out) == 1 else None


def pair_relation(repo):
    import pyarrow.parquet as pq
    p = Path(repo) / 'manifests/pairs_train_v1.parquet'
    assert sha256_file(p) == 'a5e4fdaef236f15730c7e3885b537e08e995faffbe654167fc940f44bbc75243'
    t = pq.read_table(str(p), columns=['pair_id', 'target_live_id', 'source_spoof_id', 'dataset']).to_pylist()
    return [(r['pair_id'], r['target_live_id'], r['source_spoof_id'], r['dataset']) for r in t]


def face_sha(repo, ids):
    import pyarrow.parquet as pq
    p = Path(repo) / 'manifests/m2_sample_accounting.parquet'
    assert sha256_file(p) == '65bb83e69ea7cebcb37373971e5e8f5248609b7963fd9d32aa1c0ba4377c0662'
    t = pq.read_table(str(p), columns=['sample_id', 'face_png_sha256'], filters=[('sample_id', 'in', sorted(ids))])
    return dict(zip(t.column('sample_id').to_pylist(), t.column('face_png_sha256').to_pylist()))


def seed31(text):
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], 'big') % (2 ** 31 - 1)


def build(c, repo_root):
    sys.path.insert(0, str(repo_root))
    from methods.common import learned_bank as lb
    repo = Path(repo_root)
    M, S = c['method'], c['seed']
    G = Gates()
    arel = AUTHORITY.format(M=M)
    auth_raw = (repo / arel).read_bytes()
    auth = json.loads(auth_raw)
    auth_sha = sha256_bytes(auth_raw)
    contract_sha = sha256_bytes((repo / lb.CONTRACT_PATH).read_bytes())
    storage_raw = (repo / lb.STORAGE_AUTHORITY_PATH).read_bytes()
    e04_sha = sha256_bytes((repo / lb.E04_COMPLETION_RECORD).read_bytes())
    s7rel = 'outputs/audit/LEARNED_BANK_S7_%s.json' % M
    s7_raw = (repo / s7rel).read_bytes()
    s7 = json.loads(s7_raw)
    commit = c['git']['head']
    exec_commit = exec_commit_of(repo, M)
    src_commit = auth['code_authority']['source_commit']
    pf = c.get('preflight') or {}
    mk = c['markers']
    prov = c['provenance'] or {}
    ad = prov.get('adapter', {})
    env = lb.ENVIRONMENTS[M]
    ident = lb.checkpoint_identity_sha256(M, S)
    auth_ck = lb.CHECKPOINTS[(M, S)]
    ev = EVIDENCE.format(M=M, S=S)
    rid = run_id(M, S, commit, auth_sha)

    # ---- A. authority and execution identity
    G('A.authority_record', auth.get('schema_version') == AUTHORITY_SCHEMA and auth.get('status') == 'OWNER_APPROVED'
      and auth.get('approved') is True and auth.get('method') == M and auth.get('seeds') == list(SEEDS)
      and S in auth['seeds'] and auth.get('contract_sha256') == contract_sha
      and auth['s7_acceptance_record'] == {'path': s7rel, 'sha256': sha256_bytes(s7_raw),
                                           'status': 'S7_ACCEPTANCE_PASSED', 'execution_commit': s7['execution_commit'],
                                           'gates': '%d/%d' % (s7['gates_passed'], s7['gates_total'])}
      and s7['s7_acceptance_passed'] is True and s7['failed_gates'] == [] and s7['seeds'] == list(SEEDS)
      and auth['storage_authority']['sha256'] == sha256_bytes(storage_raw)
      and auth['frozen_config'] == {'path': lb.METHOD_CONFIGS[M][0], 'sha256': lb.METHOD_CONFIGS[M][1]}
      and auth['checkpoints'][str(S)]['checkpoint_identity_sha256'] == ident
      and auth['checkpoints'][str(S)]['sha256'] == auth_ck['sha256']
      and auth['authorizes']['production_bank_generation'] is True and auth['authorizes']['generator_retraining'] is False
      and auth['authorizes']['test_evaluation'] is False and auth['authorizes']['gpat'] is False, arel)
    diff = git(repo, 'diff', '--name-only', src_commit, commit).stdout.decode().split()
    G('A.execution_checkout', commit == exec_commit and c['git']['status_porcelain'] == ''
      and c['git']['branch'] == 'm6-baselines' and pf.get('git', {}).get('head') == commit
      and pf.get('git', {}).get('status_porcelain') == ''
      and git(repo, 'merge-base', '--is-ancestor', src_commit, commit).returncode == 0
      and git(repo, 'merge-base', '--is-ancestor', commit, 'HEAD').returncode == 0
      and git(repo, 'show', '%s:%s' % (commit, arel)).stdout == auth_raw
      and all(any(p.match(x) for p in ALLOWED_EXEC_DIFF) for x in diff) and arel in diff
      and c['repo'] == '%s/m8_learned_banks/clones/prod_%s_%s' % (WORKDIR, M, commit),
      {'execution_commit': commit, 'source_commit': src_commit, 'diff_vs_source': diff})
    G('A.authority_committed_before_launch', git(repo, 'log', '-1', '--format=%ct', commit).returncode == 0
      and int(git(repo, 'log', '-1', '--format=%ct', commit).stdout.decode().strip() or 9e12) <= epoch(mk['launch_time']))

    # ---- P. preflight (immediately before launch)
    g = pf.get('gpu', {})
    st = pf.get('storage', {})
    cap = st.get('capacity') or {}
    G('P.resources', pf.get('method') == M and pf.get('seed') == S and pf.get('processes') == []
      and all(v == '1 0' for v in pf.get('bank_panes', {}).values()) and g.get('uuid') == GPU_UUID
      and g.get('name') == GPU_NAME and g.get('driver') == DRIVER
      and g.get('free_mib', 0) >= (ESTIMATE_GIB[M] + MARGIN_GIB) * 1024
      and (pf.get('memory', {}).get('mem_available_bytes') or 0) >= MEM_FLOOR_GIB << 30,
      {'gpu_free_mib': g.get('free_mib'), 'unrelated_gpu_processes': g.get('compute_apps'),
       'mem_available_gib': round((pf.get('memory', {}).get('mem_available_bytes') or 0) / 2 ** 30, 1)})
    G('P.storage', st.get('ok') is True and st.get('bank_root') == BANK_ROOT
      and st.get('filesystem', {}).get('filesystem_uuid') == HDD_UUID
      and st.get('filesystem', {}).get('mount_point') == '/media/ext_data' and st.get('mode_octal') == '0o700'
      and cap.get('free_bytes', 0) >= DISK_FLOOR_GIB << 30 and cap.get('free_inodes', 0) >= INODE_FLOOR,
      {'checks': st.get('checks'), 'capacity': cap})
    G('P.fresh_bank_dir', pf.get('bank_dir') == '%s/%s/seed_%d' % (BANK_ROOT, M, S) and pf.get('bank_dir_exists') is False
      and all(pf.get('markers_absent', {}).values()) and len(pf.get('markers_absent', {})) == 7
      and pf.get('tmux_session_absent') is True and pf.get('utc', '9') <= mk['launch_time'])
    G('P.checkpoints_and_authority', len(pf.get('checkpoints', {})) == 12 and all(
        v['ok'] and v['identity'] == lb.checkpoint_identity_sha256(k.split('|')[0], int(k.split('|')[1]))
        for k, v in pf['checkpoints'].items()) and pf.get('authority', {}).get('ok') is True
      and pf['authority']['record']['records'].get(arel) == auth_sha
      and pf['authority']['record']['mode'] == 'PRODUCTION' and pf['authority']['record']['seeds'] == [S])
    G('P.environment', pf.get('environment', {}).get('present') is True and pf['environment']['probe_rc'] == 0
      and pf['environment']['python'] == env['python'], pf.get('environment'))
    e0102 = e01_e02_expected(repo)
    G('P.e01_e02_banks_unchanged', pf.get('e01_e02_manifests') == e0102 and c.get('e01_e02_manifests_after') == e0102)

    # ---- R. run lifecycle
    res = ((mk['pane'] or {}).get('production_result') or {})
    rr = res.get('result') or {}
    G('R.exit_zero', mk['exit_raw'] == '0\n' and mk['tmux_pane'] in ('1 0', ''), mk['exit_raw'])
    G('R.command', mk['command'] is not None and (
        'tools/learned_bank.py production --method %s --seed %d --runtime-root %s\n' % (M, S, RUNTIME)) in mk['command']
      and mk['command'].count('tools/learned_bank.py') == 1 and env['python'] in mk['command']
      and ('cd %s ' % c['repo']) in mk['command'] and 'env -i ' in mk['command']
      and (E03_ENV if M == 'E03' else PY_ENV) in mk['command'])
    G('R.result_line', res.get('method') == M and res.get('seed') == S and rr.get('status') == 'completed'
      and rr.get('rows') == N and rr.get('outputs') == N and rr.get('failure_count') == 0
      and rr.get('manifest_sha256') == c['read_bank'].get('manifest_sha256')
      and not any((mk['pane'] or {}).get('fail_hits', {'x': 1}).values()), (mk['pane'] or {}).get('fail_hits'))
    G('R.window', epoch(mk['launch_time']) <= epoch(mk['end_time']) and prov.get('start_utc', '0') >= mk['launch_time']
      and prov.get('end_utc', '9') <= mk['end_time'], [mk['launch_time'], prov.get('start_utc'), prov.get('end_utc'),
                                                        mk['end_time']])
    G('R.no_process_left', c['processes_after'] == [])

    # ---- S. bank content
    G('S.provenance_completed', prov.get('status') == 'completed' and not c['aborted'] and prov.get('rows') == N
      and prov.get('outputs') == N and prov.get('failure_count') == 0 and prov.get('failures') == []
      and prov.get('mode') == 'PRODUCTION' and prov.get('method_id') == M and prov.get('bank_seed') == S
      and prov.get('run_dir') == c['bank_dir'] == '%s/%s/seed_%d' % (BANK_ROOT, M, S)
      and prov.get('post_write_verification', {}).get('ok') is True
      and prov.get('post_write_verification', {}).get('files_checked') == N, prov.get('status'))
    ar = prov.get('authorization', {})
    G('S.authorization', ar.get('mode') == 'PRODUCTION' and ar.get('methods') == [M] and ar.get('seeds') == [S]
      and ar.get('contract_sha256') == contract_sha and ar.get('label') == 'production'
      and ar.get('records') == {lb.CONTRACT_PATH: contract_sha, arel: auth_sha, s7rel: sha256_bytes(s7_raw),
                                lb.STORAGE_AUTHORITY_PATH: sha256_bytes(storage_raw), lb.E04_COMPLETION_RECORD: e04_sha})
    ck = ad.get('checkpoint', {})
    G('S.checkpoint_before', ad.get('is_fixture') is False and ad.get('checkpoint_identity_sha256') == ident
      and ck.get('checkpoint_identity_sha256') == ident and ck.get('run_id') == auth_ck['run_id']
      and {k: v['sha256'] for k, v in ck.get('files', {}).items()} == auth_ck['sha256']
      and ad.get('generator_seed') == ad.get('bank_seed') == S)
    G('S.config_contract', ad.get('config_sha256') == lb.METHOD_CONFIGS[M][1] and ad.get('contract_sha256') == contract_sha)
    sources = json.loads((repo / lb.SOURCES_PATH).read_text())['families']
    fam = sources[{'E03': 'STDN', 'E05': 'PCGAN'}.get(M, 'DSDG')]
    src = ad.get('source', {})
    G('S.pinned_source', src.get('files') == fam['files'] and src.get('commit') == fam['commit']
      and src.get('root') == '%s/%s' % (c['repo'], fam['root']))
    fin = prov.get('adapter_finalize', {})
    if M == 'E03':
        gr, cov = ad.get('graph', {}), ad.get('coverage', {})
        graph_ok = (gr.get('training_nn') is False and gr.get('update_ops') == 0 and gr.get('ema_variables') == 0
                    and gr.get('mutating_reachable_ops') in (0, []) and gr.get('random_reachable_ops') in (0, [])
                    and gr.get('batch_size') == 1 and cov.get('graph_variables_covered') == gr.get('global_variables')
                    and cov.get('ema_restored') == 0 and fin.get('forwards') == N and fin.get('batch_size') == 1
                    and fin.get('stochastic_at_inference') is False and fin.get('trace_sidecar_written') is False)
        guard_ok = fin.get('forwards') == N
        gdetail = {'graph': gr, 'coverage': cov, 'finalize': fin}
    else:
        guard_ok = ad.get('training_guards') == {'optimizer_constructions': 0, 'optimizer_steps': 0,
                                                 'backward_passes': 0, 'torch_saves': 0}
        if M == 'E05':
            graph_ok = bool(ad.get('runtime_closure', {}).get('build_closure')) and bool(ad.get('precision')) \
                and fin.get('generator_forwards') == N and fin.get('encoder_forwards') == 2 * N \
                and fin.get('reconstruction_forwards') == 0
            gdetail = {'finalize': fin}
        else:
            graph_ok = bool(ad.get('load')) and bool(ad.get('determinism')) and fin.get('batches_run') == 89 \
                and fin.get('forward_calls') == 89 and fin.get('discarded_final_batch_samples') == 62 \
                and fin.get('stream_seed') == lb.dsdg_bank_seed(M, S) and fin.get('live_half_stored') is False \
                and fin.get('stream_plan', {}).get('kept_in_final_batch') == 38 \
                and fin.get('stream_plan', {}).get('samples_drawn') == 8900
            gdetail = {'finalize': {k: v for k, v in fin.items() if k != 'stream_plan'}, 'device': ad.get('device')}
    G('S.inference_graph', graph_ok, gdetail)
    G('S.no_training', guard_ok)
    rd = prov.get('reader')
    if FACE_READS[M]:
        rd_ok = rd is not None and rd['reads'] == rd['atime_checked_reads'] == FACE_READS[M] \
            and rd['splits_allowlisted'] == ['TRAIN'] and rd['o_noatime'] is True and rd['faces_root'] == FACES
    else:
        rd_ok = rd is None
    G('S.train_only_reader', rd_ok, rd)
    G('S.environment', prov.get('environment', {}).get('python_executable') == env['python']
      and prov['environment'].get('cuda_visible_devices') == '0' and prov.get('git_head') == commit
      and (M == 'E03' or ad.get('device') == 'cuda'), prov.get('environment'))
    rb = c['read_bank']
    G('S.reader_verification', rb['ok'] and rb['problems'] == [] and rb['files_checked'] == N and rb['failures'] == 0
      and rb['provenance_ok'] is True)
    iv = c['independent']
    lst = c['listing']
    G('S.independent_verification', iv['files'] == N and iv['files_verified'] == N and iv['problem_count'] == 0
      and iv['modes'] == ['0o444'] and lst['dir_mode'] == '0o555' and lst['bank_dir_mode'] == '0o555'
      and lst['entries'] == sorted([lb.BANK_DIR, lb.MANIFEST, lb.MANIFEST_SHA, lb.PROVENANCE])
      and all(v['mode'] == '0o444' for v in lst['top_files'].values()) and c['bank_unchanged_during_collect'] is True,
      {k: iv[k] for k in ('files', 'files_verified', 'problem_count', 'modes', 'total_png_bytes')})
    # the committed full manifest (gzip) -> every row re-derived here
    gz = repo / ev / 'bank_manifest.jsonl.gz'
    body = gzip.decompress(gz.read_bytes()) if gz.is_file() else b''
    msha = sha256_bytes(body)
    G('S.manifest_committed', msha == rb.get('manifest_sha256') == prov.get('manifest_sha256')
      == lst['top_files'].get(lb.MANIFEST, {}).get('sha256'), msha)
    rows = [json.loads(x) for x in body.decode('ascii').splitlines()] if body else []
    canon_ok = all(lb.canonical_json(r) == x for r, x in zip(rows, body.decode('ascii').splitlines()))
    if M in lb.PAIR_METHODS:
        rel = pair_relation(repo)
        got = [(r['row_index'], r['pair_id'], r['target_live_id'], r['source_spoof_id'], r['dataset']) for r in rows]
        pop_ok = got == [(i,) + tuple(x) for i, x in enumerate(rel)] and all(r['row_id'] == r['pair_id']
                                                                            and r['synthetic_id'] is None for r in rows)
        if M == 'E05':
            pop_ok = pop_ok and all(r['deterministic_noise_seed'] == seed31(
                'GPAT_TRANSFERBENCH|E05|BANK|%d|%s' % (S, r['pair_id'])) for r in rows)
        else:
            pop_ok = pop_ok and all(r['deterministic_noise_seed'] is None for r in rows)
        fs = face_sha(repo, {r['target_live_id'] for r in rows} | {r['source_spoof_id'] for r in rows})
        pop_ok = pop_ok and all(r['target_face_sha256'] == fs.get(r['target_live_id'])
                                and r['source_face_sha256'] == fs.get(r['source_spoof_id']) for r in rows)
        rng = ('torch.manual_seed(e05_pair_seed(seed, pair_id)) immediately before the single mix forward of each pair'
               if M == 'E05' else 'none (E03 inference is deterministic; no sampling)')
    else:
        bseed = seed31('GPAT_TRANSFERBENCH|%s|BANK|%d' % (M, S))
        pop_ok = [r['row_index'] for r in rows] == list(range(N)) and all(
            r['row_id'] == r['synthetic_id'] == '%s_S%d_%06d' % (M, S, r['row_index']) and r['pair_id'] is None
            and r['target_live_id'] is None and r['source_spoof_id'] is None and r['dataset'] is None
            and r['target_face_sha256'] is None and r['source_face_sha256'] is None
            and r['deterministic_noise_seed'] == bseed
            and (r['dsdg_batch_index'], r['dsdg_batch_offset']) == divmod(r['row_index'], 100) for r in rows) \
            and bseed == lb.dsdg_bank_seed(M, S)
        rng = ('torch.Generator().manual_seed(%d) (dsdg_bank_seed); per batch of 100: noise = randn(100,128), '
               'noise_s = randn(100,128), z = cat(noise_s, noise, noise); 89 batches, 8900 drawn, 8838 kept, '
               'last 62 of batch 88 discarded' % bseed)
    G('S.population', len(rows) == N and canon_ok and pop_ok)
    G('S.rows_finite_rgb_uint8', len(rows) == N and all(
        r['schema'] == lb.SCHEMA and r['generation_status'] == 'success' and r['raw_output_finite'] is True
        and r['failure_reason'] is None and r['mode'] == 'PRODUCTION' and r['label'] == 'spoof' and r['method_id'] == M
        and r['conversion_rule'] == lb.CONVERSION_RULES[M] and r['checkpoint_identity_sha256'] == ident
        and r['config_sha256'] == lb.METHOD_CONFIGS[M][1] and r['contract_sha256'] == contract_sha
        and r['generator_seed'] == r['bank_seed'] == S and r['output_path'] == 'bank/%s.png' % r['row_id']
        and isinstance(r['raw_output_min'], float) and isinstance(r['raw_output_max'], float)
        and r['raw_output_min'] <= r['raw_output_max'] for r in rows)
      and len({r['row_id'] for r in rows}) == N and len({r['output_path'] for r in rows}) == N)
    eq = c['s7_equivalence']
    toy = lb.load_toy_manifest(repo)
    want_idx = [t['position'] for t in toy] if M in lb.PAIR_METHODS else list(range(TOY_N))
    G('S.s7_production_equivalence', eq['s7_record_sha256'] == sha256_bytes(s7_raw)
      and eq['s7_manifest_sha256_recorded'] == s7['outputs'][str(S)]['manifest_sha256'] and all(
        v['manifest_sha256'] == s7['outputs'][str(S)]['manifest_sha256'] and v['rows'] == TOY_N
        and v['rows_matched'] == TOY_N and v['field_mismatch_count'] == 0 and v['byte_mismatch_count'] == 0
        and v['row_indices'] == want_idx for v in eq['runs'].values()) and sorted(eq['runs']) == ['A', 'B'],
      {L: {k: v[k] for k in ('rows_matched', 'field_mismatch_count', 'byte_mismatch_count')}
       for L, v in eq['runs'].items()})
    af = c['checkpoints_after']
    G('S.checkpoint_after', len(af) == 12 and all(
        v['ok'] and v['identity'] == lb.checkpoint_identity_sha256(k.split('|')[0], int(k.split('|')[1]))
        for k, v in af.items()) and all(
        af['%s|%d' % (M, S)]['files'][k]['mtime_ns'] == ck['files'][k]['mtime_ns']
        and af['%s|%d' % (M, S)]['files'][k]['sha256'] == auth_ck['sha256'][k] for k in auth_ck['sha256'])
      and all(af[k]['files'] == pf['checkpoints'][k]['files'] for k in af))
    if M == 'E05':
        rul_raw = (repo / 'configs/amendments/e05_source_relocation_ruling.yaml').read_bytes()
        b12_raw = (repo / 'configs/amendments/e05_s7_b1_b2_ruling.yaml').read_bytes()
        rul, b12 = json.loads(rul_raw), json.loads(b12_raw)
        patch = json.loads((repo / 'environments/e05.compatibility_patch_manifest.json').read_text())
        bc = ad.get('runtime_closure', {}).get('build_closure', {})
        si, imp = bc.get('source_identity') or {}, bc.get('import_resolution') or {}
        G('S.e05_source_identity', si.get('relocated') is True
          and si.get('root') == '%s/third_party/source_cache/swapping_autoencoder' % c['repo']
          and si.get('differing_fields') == ['root'] and si.get('historical_root') == rul['historical_root']
          and si.get('commit') == '6baa180f1184ee79a6b967f9d80ee0e02a979ac7'
          and si.get('tree') == '25a434f1282a0177d6afdcc68ebfcf40bd805621' and si.get('files_verified') == 23
          and si.get('code_files_verified') == 67 and si.get('unexpected_executable_files') == 0
          and si.get('ruling') == {'path': 'configs/amendments/e05_source_relocation_ruling.yaml',
                                   'sha256': sha256_bytes(rul_raw)}
          and len(imp) == 7 and bc.get('jit_build_invoked') is False and len(bc.get('extensions', {})) == 2
          and auth['e05']['o2_ruling']['sha256'] == sha256_bytes(rul_raw), {'source_identity': si})
        G('S.e05_tensor2im_runtime', ad.get('tensor2im_runtime') == {
            'module': 'util.util', 'file': 'util/util.py', 'file_sha256': patch['generated_files_sha256']['util/util.py'],
            'segment_sha256': b12['tensor2im']['pinned_segment_sha256']}
          and ad.get('tensor2im_segment_sha256') == b12['tensor2im']['pinned_segment_sha256']
          and auth['e05']['b1_b2_ruling']['sha256'] == sha256_bytes(b12_raw))
        G('S.e05_one_process_one_seed', res.get('seed') == S and mk['command'].count('--seed %d' % S) == 1
          and auth['e05']['process_isolation'] == 'P2_ONE_PROCESS_PER_SEED')

    # ---- F. firewall and storage
    at = c['atime']
    if at is None:
        G('F.atime_ledger', False, 'missing')
    else:
        by = at['by_split']
        G('F.atime_ledger', at['same_paths'] and at['files_pre'] == at['files_post'] > 0
          and at['pre_utc'] <= mk['launch_time'] and at['post_utc'] >= mk['end_time']
          and all(v['atime_changed'] == 0 and v['mtime_changed'] == 0 and v['missing_post'] == 0 for v in by.values())
          and {'TRAIN', 'VAL', 'TEST'} <= set(by), by)
    G('F.no_val_test_inputs', prov.get('population', {}).get('pairs_val_opened', False) is False
      and prov.get('population', {}).get('kind') == 'PRODUCTION_FROZEN_TRAIN_POPULATION'
      and prov['population'].get('rows') == N and all(
          r['pair_id'] is None or re.match(r'^PTR\d{6}$', r['pair_id']) for r in rows))
    sa = c['storage_after']
    G('F.storage_after', sa.get('ok') is True and sa.get('filesystem', {}).get('filesystem_uuid') == HDD_UUID
      and sa.get('mode_octal') == '0o700' and sa.get('unexpected_entries') == [], sa.get('checks'))
    reg_before = pf.get('registry', {})
    key = '%s/seed_%d' % (M, S)
    G('F.other_banks_unchanged', all(c['registry_after'].get(k) == v for k, v in reg_before.items())
      and sorted(c['registry_after']) == sorted(list(reg_before) + [key])
      and c['registry_after'][key]['manifest_sha256'] == msha and c['registry_after'][key]['provenance'] is True
      and all(v['provenance'] and not v['aborted'] for v in c['registry_after'].values()),
      sorted(c['registry_after']))

    failed = G.failed
    status = 'LEARNED_BANK_PRODUCTION_PASSED' if not failed else 'LEARNED_BANK_PRODUCTION_FAILED_NEEDS_OWNER_REVIEW'
    doc = {
        'schema': 'learned_bank_production_audit_record_v1', 'method': M, 'seed': S, 'run_id': rid,
        'status': status, 'production_audit_passed': not failed, 'failed_gates': failed, 'gates': G.results,
        'gates_passed': len(G.results) - len(failed), 'gates_total': len(G.results),
        'authority': {'path': arel, 'sha256': auth_sha}, 'contract_sha256': contract_sha, 'execution_commit': commit,
        'source_commit': src_commit, 's7_record': {'path': s7rel, 'sha256': sha256_bytes(s7_raw)},
        'bank': {'path': c['bank_dir'], 'rows': N, 'outputs': rb.get('files_checked'), 'failures': rb.get('failures'),
                 'manifest_sha256': msha, 'manifest_evidence': ev + '/bank_manifest.jsonl.gz',
                 'provenance_sha256': lst['top_files'].get(lb.PROVENANCE, {}).get('sha256'),
                 'total_png_bytes': iv.get('total_png_bytes'),
                 'output_sha256_list_sha256': sha256_bytes('\n'.join(r['output_sha256'] for r in rows).encode()),
                 'pixel_sha256_list_sha256': sha256_bytes('\n'.join(r['output_pixel_sha256'] for r in rows).encode())},
        'checkpoint': {'run_id': auth_ck['run_id'], 'sha256': auth_ck['sha256'], 'checkpoint_identity_sha256': ident},
        'descriptive_statistics': iv.get('stats'),
        'storage': {'bank_root': BANK_ROOT, 'filesystem_uuid': sa.get('filesystem', {}).get('filesystem_uuid'),
                    'mount_point': sa.get('filesystem', {}).get('mount_point'), 'capacity_after': sa.get('capacity')},
        'window': {'launch': mk['launch_time'], 'end': mk['end_time'], 'wall_time_seconds': prov.get('wall_time_seconds')},
        'reproduction': {
            'method_id': M, 'seed': S, 'execution_commit': commit, 'branch': 'm6-baselines',
            'clone': c['repo'], 'command': mk['command'], 'command_sha256': mk['command_sha256'],
            'launch_script': 'tools/learned_bank_production_scripts/m8prod_launch.sh %s %d %s' % (M, S, commit),
            'post_script': 'tools/learned_bank_production_scripts/m8prod_post.sh %s %d' % (M, S),
            'exit_status': mk['exit_raw'], 'config': {'path': lb.METHOD_CONFIGS[M][0], 'sha256': lb.METHOD_CONFIGS[M][1]},
            'environment': {'name': env['name'], 'python': env['python'],
                            'lock': {'path': env['lock'][0], 'sha256': env['lock'][1]},
                            'probe': pf.get('environment', {}).get('probe'), 'run_environment': prov.get('environment'),
                            'framework': ad.get('framework')},
            'gpu': {k: g.get(k) for k in ('name', 'uuid', 'driver')},
            'population': ('all %d frozen pairs_train_v1 rows in canonical 0-based order' % N if M in lb.PAIR_METHODS
                           else 'synthetic stream indices 0..%d' % (N - 1)),
            'rng': rng, 'conversion_rule': lb.CONVERSION_RULES[M],
            'preflight_sha256': c.get('preflight_sha256'),
            'atime_ledger': {k: (at or {}).get(k) for k in ('pre_utc', 'post_utc', 'pre_sha256', 'post_sha256')},
            'audit_commands': {
                'collect': 'CUDA_VISIBLE_DEVICES= /home/student20261/miniconda3/envs/gpat-m6-e05/bin/python -I -B - collect '
                           '--method %s --seed %d --repo %s < tools/learned_bank_production_audit.py' % (M, S, c['repo']),
                'build': 'python3 tools/learned_bank_production_audit.py build --collected %s/collected.json' % ev},
            'replay_policy': 'C-Q2/C-Q3: bitwise A/B replay is established by the passed S7 record; the production audit '
                             'requires the 186 S7 toy rows to equal their production rows byte for byte (runs A and B). '
                             'A second full production bank is not required by the contract and was not generated.',
            'backup': 'the image bank exists only on the approved HDD; git holds the manifest (gzip), per-image file and '
                      'pixel SHA256 and this audit, not the image bytes; no independent binary backup exists'},
        'collected': {'path': ev + '/collected.json', 'sha256': sha256_file(repo / ev / 'collected.json')},
        'audit_tool': {'path': 'tools/learned_bank_production_audit.py',
                       'sha256': sha256_file(repo / 'tools/learned_bank_production_audit.py')},
    }
    return doc


def report(doc):
    r = doc['reproduction']
    st = doc['descriptive_statistics'] or {}
    cap = doc['storage']['capacity_after'] or {}
    failed = doc['failed_gates']
    return '\n'.join([
        '# M8 learned-bank production: %s seed %d' % (doc['method'], doc['seed']), '',
        '**Status:** `%s`, %d/%d gates passing. Run ID `%s`.' % (doc['status'], doc['gates_passed'],
                                                                  doc['gates_total'], doc['run_id']), '',
        'Failed gates: %s' % (', '.join('`%s`' % g for g in failed) if failed else 'none.'), '',
        '| | |', '|---|---|',
        '| bank | `%s` |' % doc['bank']['path'],
        '| rows / outputs / failures | %d / %s / %s |' % (doc['bank']['rows'], doc['bank']['outputs'], doc['bank']['failures']),
        '| manifest sha256 | `%s` |' % doc['bank']['manifest_sha256'],
        '| checkpoint | run `%s`, identity `%s…` |' % (doc['checkpoint']['run_id'],
                                                     doc['checkpoint']['checkpoint_identity_sha256'][:16]),
        '| authority | `%s` (`%s…`) |' % (doc['authority']['path'], doc['authority']['sha256'][:16]),
        '| execution commit | `%s` (source `%s`) |' % (doc['execution_commit'], doc['source_commit']),
        '| S7 record | `%s` (`%s…`) |' % (doc['s7_record']['path'], doc['s7_record']['sha256'][:16]),
        '| window | %s → %s (%.0f s generation) |' % (doc['window']['launch'], doc['window']['end'],
                                                     doc['window']['wall_time_seconds'] or 0),
        '| environment | `%s`, %s, GPU %s `%s` driver %s |' % (r['environment']['name'], r['environment']['framework'],
                                                            r['gpu']['name'], r['gpu']['uuid'], r['gpu']['driver']),
        '| storage | HDD UUID `%s`, mount `%s`; free after %.1f GiB |' % (
            doc['storage']['filesystem_uuid'], doc['storage']['mount_point'], (cap.get('free_bytes') or 0) / 2 ** 30),
        '', '## Scientific content', '',
        '- Population: %s.' % r['population'],
        '- RNG: %s.' % r['rng'],
        '- Conversion: `%s`.' % r['conversion_rule'],
        '- S7-to-production equivalence: the 186 approved S7 toy rows equal their production rows byte for byte '
        '(manifest fields other than `mode`, and PNG bytes), for S7 runs A and B.' if 'S.s7_production_equivalence'
        not in failed else '- S7-to-production equivalence FAILED.',
        '- Descriptive statistics (no threshold): per-image std median %s [p05 %s, p95 %s]; distinct pixel hashes %s of '
        '%d; exact duplicate rows %s; rows sharing a coarse 8x8 luminance signature %s.' % (
            '%.1f' % st['per_image_std']['median'], '%.1f' % st['per_image_std']['p05'],
            '%.1f' % st['per_image_std']['p95'], st['distinct_pixel_sha256'], doc['bank']['rows'],
            st['exact_duplicate_rows'], st['coarse_8x8_luma_signature_rows']) if st else '- Statistics missing.',
        '', '## Reproduction', '',
        '- Command (`%s…`): see the JSON record; launch script `%s`.' % ((r['command_sha256'] or '')[:16],
                                                                        r['launch_script']),
        '- %s' % r['replay_policy'],
        '- Backup: %s.' % r['backup'], ''])


def build_closure(repo_root, M):
    repo = Path(repo_root)
    G = Gates()
    recs = {}
    for S in SEEDS:
        p = repo / (RECORD.format(M=M, S=S) + '.json')
        recs[S] = (json.loads(p.read_text()), sha256_file(p)) if p.is_file() else (None, None)
    G('C.three_banks_passed', all(d and d['production_audit_passed'] is True and d['failed_gates'] == []
                                  for d, _ in recs.values()))
    ds = {S: d for S, (d, _) in recs.items()}
    if not all(ds.values()):
        return {'schema': 'learned_bank_production_family_closure_v1', 'method': M,
                'status': 'LEARNED_BANK_PRODUCTION_FAMILY_INCOMPLETE', 'failed_gates': G.failed, 'gates': G.results}
    G('C.distinct_run_ids', len({d['run_id'] for d in ds.values()}) == 3)
    G('C.same_authority_and_execution', len({(d['authority']['sha256'], d['execution_commit']) for d in ds.values()}) == 1)
    G('C.seeds_and_checkpoints', [d['seed'] for d in ds.values()] == list(SEEDS)
      and len({d['checkpoint']['checkpoint_identity_sha256'] for d in ds.values()}) == 3)
    win = [(epoch(ds[S]['window']['launch']), epoch(ds[S]['window']['end'])) for S in SEEDS]
    G('C.sequential_seed_order', all(a <= b for a, b in win) and all(win[i][1] <= win[i + 1][0] for i in range(2)),
      [[ds[S]['window']['launch'], ds[S]['window']['end']] for S in SEEDS])
    commit_times = {}
    for S in SEEDS:
        rel = RECORD.format(M=M, S=S) + '.json'
        out = git(repo, 'log', '--format=%H %ct', '--diff-filter=A', '--', rel).stdout.decode().split()
        commit_times[S] = (out[0], int(out[1])) if len(out) == 2 else (None, None)
    G('C.audit_committed_before_next_launch', all(
        commit_times[a][1] is not None and commit_times[a][1] <= epoch(ds[b]['window']['launch'])
        for a, b in ((42, 1337), (1337, 2026))) and commit_times[2026][0] is not None,
      {str(S): commit_times[S][0] for S in SEEDS})
    G('C.total_outputs', sum(d['bank']['outputs'] for d in ds.values()) == 3 * N
      and all(d['bank']['failures'] == 0 for d in ds.values()))
    G('C.registered_paths', all(d['bank']['path'] == '%s/%s/seed_%d' % (BANK_ROOT, M, S) for S, d in ds.items())
      and {d['storage']['filesystem_uuid'] for d in ds.values()} == {HDD_UUID})
    gate = lambda d, name: next(g['pass'] for g in d['gates'] if g['gate'] == name)
    for name in ('S.s7_production_equivalence', 'F.atime_ledger', 'F.no_val_test_inputs', 'S.checkpoint_after',
                 'S.no_training', 'S.population', 'P.e01_e02_banks_unchanged'):
        G('C.' + name.split('.', 1)[1], all(gate(d, name) for d in ds.values()))
    failed = G.failed
    a = ds[42]
    diag = repo / 'outputs/audit/E05_S7_CONTRAST_DIAGNOSTIC.json'
    disclosures = {
        'E06c': ['DSDG-BIN-IDFREE adapted binary variant (spoof-type labels collapsed); spoof RGB half only; synthetic IDs '
                 'without pair lineage'],
        'E06b': ['DSDG-NATIVE trained on CASIA+MSU TRAIN spoof coverage (3720 rows; SiW-Mv2 NOT_INSTANTIABLE_MISSING_SUBJECT'
                 '_ID); N_syn = 8838 pooled budget (E06b-Q0); spoof RGB half only; synthetic IDs without pair lineage'],
        'E03': ['official test-mode inference with raw (non-EMA) variables, no inference augmentation; saturation at 0 and '
                '255 is the source writer\'s clip-then-cast'],
        'E05': ['PCGAN controlled architecture resolution (not a native PCGAN reproduction)',
                'low-contrast outputs; seed 1337 close to collapse onto a near-constant grey image (see '
                'outputs/audit/E05_S7_CONTRAST_DIAGNOSTIC.md); retained under RAW-MATCHED; frozen paper budget 4000 '
                'iterations, lr 1e-6, batch 1']}[M]
    doc = {
        'schema': 'learned_bank_production_family_closure_v1', 'method': M, 'seeds': list(SEEDS),
        'status': ('LEARNED_BANK_PRODUCTION_FAMILY_COMPLETED_AND_AUDITED' if not failed
                   else 'LEARNED_BANK_PRODUCTION_FAMILY_FAILED_NEEDS_OWNER_REVIEW'),
        'failed_gates': failed, 'gates': G.results, 'gates_passed': len(G.results) - len(failed),
        'gates_total': len(G.results), 'total_outputs': sum(d['bank']['outputs'] for d in ds.values()),
        'authority': a['authority'], 'contract_sha256': a['contract_sha256'], 'execution_commit': a['execution_commit'],
        'source_commit': a['source_commit'], 's7_record': a['s7_record'],
        'config': a['reproduction']['config'], 'environment': {k: a['reproduction']['environment'][k]
                                                               for k in ('name', 'python', 'lock', 'framework')},
        'gpu': a['reproduction']['gpu'], 'storage': {'bank_root': BANK_ROOT, 'filesystem_uuid': HDD_UUID,
                                                     'mount_point': '/media/ext_data', 'fstype': 'ext4'},
        'per_seed': {str(S): {'run_id': d['run_id'], 'audit_record': RECORD.format(M=M, S=S) + '.json',
                              'audit_record_sha256': recs[S][1], 'audit_commit': commit_times[S][0],
                              'gates': '%d/%d' % (d['gates_passed'], d['gates_total']), 'bank': d['bank'],
                              'checkpoint': d['checkpoint'], 'window': d['window'],
                              'atime_ledger': d['reproduction']['atime_ledger'],
                              'descriptive_statistics': d['descriptive_statistics'],
                              'command_sha256': d['reproduction']['command_sha256']} for S, d in ds.items()},
        'rng': {str(S): d['reproduction']['rng'] for S, d in ds.items()},
        'conversion_rule': a['reproduction']['conversion_rule'],
        'population': a['reproduction']['population'],
        'test_firewall': 'TRAIN-only inputs; zero VAL/TEST face atime or mtime change in every bank window',
        'disclosures': disclosures,
        'e05_contrast_diagnostic': ({'path': 'outputs/audit/E05_S7_CONTRAST_DIAGNOSTIC.json',
                                     'sha256': sha256_file(diag)} if M == 'E05' else None),
        'reproduction_commands': {str(S): {'launch_script': d['reproduction']['launch_script'],
                                           'post_script': d['reproduction']['post_script'],
                                           'command': d['reproduction']['command'],
                                           'audit': d['reproduction']['audit_commands']} for S, d in ds.items()},
        'backup': a['reproduction']['backup'],
        'not_done': ['no downstream training, TEST evaluation, DINOv3 or GPAT', 'no E04 bank generation',
                     'outputs/audit/method_status.csv unchanged'],
    }
    return doc


def closure_report(doc):
    rows = ''.join('| %s | `%s` | %d | `%s…` | %s | %s → %s |\n' % (
        S, v['run_id'], v['bank']['outputs'], v['bank']['manifest_sha256'][:16], v['gates'], v['window']['launch'],
        v['window']['end']) for S, v in sorted(doc.get('per_seed', {}).items(), key=lambda x: int(x[0])))
    return '\n'.join([
        '# M8 learned-bank production family closure: %s' % doc['method'], '',
        '**Status:** `%s`, %d/%d closure gates passing; %d outputs.' % (
            doc['status'], doc.get('gates_passed', 0), doc.get('gates_total', 0), doc.get('total_outputs', 0)), '',
        'Failed gates: %s' % (', '.join('`%s`' % g for g in doc['failed_gates']) or 'none.'), '',
        '| seed | run ID | outputs | manifest sha256 | audit gates | window |', '|---|---|---|---|---|---|',
        rows.rstrip('\n'), '',
        '- Authority `%s` (`%s…`); execution commit `%s`; source commit `%s`.' % (
            doc['authority']['path'], doc['authority']['sha256'][:16], doc['execution_commit'], doc['source_commit']),
        '- Config `%s` (`%s…`); environment `%s` (%s).' % (doc['config']['path'], doc['config']['sha256'][:16],
                                                          doc['environment']['name'], doc['environment']['framework']),
        '- Storage: HDD `%s`, UUID `%s`, layout `<bank_root>/%s/seed_<seed>/`.' % (
            doc['storage']['bank_root'], doc['storage']['filesystem_uuid'], doc['method']),
        '- Population: %s.' % doc['population'],
        '- RNG per seed: %s.' % '; '.join('seed %s: %s' % (s, r) for s, r in sorted(doc['rng'].items(),
                                                                                   key=lambda x: int(x[0]))),
        '- Conversion: `%s`.' % doc['conversion_rule'],
        '- TEST firewall: %s.' % doc['test_firewall'],
        '- Disclosures: %s.' % '; '.join(doc['disclosures']),
        '- Backup: %s.' % doc['backup'], ''])


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('preflight', 'collect'):
        a = sub.add_parser(name)
        a.add_argument('--method', required=True, choices=METHOD_ORDER)
        a.add_argument('--seed', required=True, type=int, choices=SEEDS)
        a.add_argument('--repo', required=True)
    b = sub.add_parser('build')
    b.add_argument('--collected', required=True)
    b.add_argument('--repo-root', default=str(Path(globals().get('__file__', 'tools/x')).resolve().parents[1]))
    cl = sub.add_parser('close')
    cl.add_argument('--method', required=True, choices=METHOD_ORDER)
    cl.add_argument('--repo-root', default=str(Path(globals().get('__file__', 'tools/x')).resolve().parents[1]))
    args = ap.parse_args(argv)
    if args.cmd == 'preflight':
        return preflight(args.method, args.seed, args.repo)
    if args.cmd == 'collect':
        return collect(args.method, args.seed, args.repo)
    root = Path(args.repo_root)
    if args.cmd == 'build':
        c = json.loads(Path(args.collected).read_text())
        ev = root / EVIDENCE.format(M=c['method'], S=c['seed'])
        if Path(args.collected).resolve() != (ev / 'collected.json').resolve():
            ap.error('build reads the committed evidence %s/collected.json' % ev)
        doc = build(c, root)
        out = RECORD.format(M=doc['method'], S=doc['seed'])
        print(doc['status'], doc['gates_passed'], '/', doc['gates_total'], doc['failed_gates'])
        ok = doc['production_audit_passed']
    else:
        doc = build_closure(root, args.method)
        out = CLOSURE.format(M=args.method)
        print(doc['status'], doc.get('gates_passed'), '/', doc.get('gates_total'), doc['failed_gates'])
        ok = doc['status'] == 'LEARNED_BANK_PRODUCTION_FAMILY_COMPLETED_AND_AUDITED'
    (root / (out + '.json')).write_text(json.dumps(doc, indent=1, sort_keys=True) + '\n')
    (root / (out + '.md')).write_text(report(doc) if args.cmd == 'build' else closure_report(doc))
    (root / (out + '_SHA256.txt')).write_text(''.join('%s  %s\n' % (sha256_file(root / p), p)
                                                    for p in (out + '.json', out + '.md')))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
