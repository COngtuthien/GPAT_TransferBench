#!/usr/bin/env python3
"""Read-only M6D2a evidence validator; never imports STDN or a framework.

--draft checks finalized content before the ledger append/index rebuild.
--live-gpu adds fixed read-only remote identity/source/environment/host checks.
Default validates the finalized ledger and canonical CRLF artifact index.
No package manager, model, dataset, checkpoint or training entry point is called.
"""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
HEAD = '372d456dddb635d840e35c0659951f7d3716fc79'
OLD = '7387a25f62b50ad748edd1a69e21616af44501b9'
PIN = 'c79f1f8c615d2b8471b3df29da881bb18dd54c90'
HOST = 'student20261@100.121.84.44'
AUDIT = 'outputs/audit/M6D2A_E03_ENVIRONMENT_RESOLUTION.json'
REPORT = 'outputs/audit/M6D2A_E03_ENVIRONMENT_RESOLUTION.md'
PLAN = 'environments/e03_stdn_compatibility_plan.md'
TOOL = 'tools/m6d2a_e03_environment_preflight.py'
LEDGER = 'outputs/audit/EXECUTION_LEDGER.jsonl'
INDEX = 'outputs/audit/ARTIFACT_INDEX.csv'
DELIVERABLES = {AUDIT, REPORT, PLAN, TOOL}
ALLOWED = DELIVERABLES | {LEDGER, INDEX}
PREFIX_SHA = '47735be1b3abd85c05927a6b2b54685619f04fc143b4d8c94e6ce77f19943102'
PROHIBITED = ('environment_creation', 'package_install', 'environment_mutation',
              'source_patch', 'model_construction', 'weight_load', 'benchmark_image_decoding',
              'benchmark_manifest_execution', 'test_scientific_execution', 'training',
              'optimizer_step', 'backward', 'checkpoint', 'synthetic_bank',
              'scientific_config_change', 'amendment_change', 'commit', 'push')


def require(ok, message):
    if not ok:
        raise RuntimeError('STOP_AND_REPORT: ' + message)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def read(rel):
    return (ROOT / rel).read_bytes()


def git(*args, root=ROOT):
    return subprocess.check_output(['git', '-C', str(root), *args],
                                   env=dict(os.environ, GIT_NO_LAZY_FETCH='1',
                                            GIT_OPTIONAL_LOCKS='0'))


def source_check(root, recorded, pins):
    require(git('rev-parse', '--show-toplevel', root=root).decode().strip() == str(root), 'source root')
    require(git('rev-parse', 'HEAD', root=root).decode().strip() == PIN == recorded['pin'], 'source pin')
    require(git('rev-parse', 'HEAD^{tree}', root=root).decode().strip() == pins['commit_tree'] == recorded['tree'], 'source tree')
    require(git('remote', 'get-url', 'origin', root=root).decode().strip().removesuffix('.git') == pins['repository'], 'source origin')
    deleted = [r['path'] for r in pins['binary_weights_removed_after_checkout']]
    require(recorded['deleted'] == deleted, 'deletion provenance')
    require(git('status', '--porcelain', root=root).decode() == recorded['status'] ==
            ''.join(' D ' + p + '\n' for p in sorted(deleted)), 'exact four source deletions')
    actual = {str(p.relative_to(root)) for p in root.rglob('*')
              if p.is_file() and '.git' not in p.relative_to(root).parts}
    require(actual == set(recorded['files']), 'source-only file inventory; no unexpected weights/files')
    for line in git('ls-tree', '-r', 'HEAD', root=root).decode().splitlines():
        meta, rel = line.split('\t')
        mode, typ, blob = meta.split()
        if rel in deleted:
            require(not (root / rel).exists(), 'restored upstream weight: ' + rel)
            continue
        p = root / rel
        require(not p.is_symlink() and typ == 'blob', 'regular source file')
        b = p.read_bytes()  # Opaque upstream example bytes may be hashed; never decoded.
        require(recorded['files'][rel] == {'sha256': sha(b), 'size_bytes': len(b), 'git_blob': blob}, 'source digest: ' + rel)
        require(hashlib.sha1(b'blob ' + str(len(b)).encode() + b'\0' + b).hexdigest() == blob, 'pinned blob: ' + rel)
    for rel, row in pins['cited_files'].items():
        require(recorded['files'][rel]['sha256'] == row['sha256'], 'source provenance: ' + rel)


def api_inventory(root):
    inventory = {}
    for p in sorted(root.rglob('*.py')):
        for n, line in enumerate(p.read_text().splitlines(), 1):
            for symbol in re.findall(r'\b(?:tf|layers)\.[A-Za-z_][\w.]*', line):
                if symbol.startswith('layers.'):
                    symbol = symbol.replace('layers.', 'tf.contrib.layers.', 1)
                inventory.setdefault(symbol, []).append({'file': str(p.relative_to(root)), 'line': n, 'text': line.strip()})
    return inventory


def live_gpu(a):
    # Only this fixed program runs remotely. JSON values are never executed as code.
    program = r'''
import json, pathlib, subprocess, hashlib, os
r=pathlib.Path('/home/student20261/workdir/GPAT_TransferBench')
s=r/'third_party/source_cache/stdn'
expected=json.loads(input())
def g(root,*a): return subprocess.check_output(['git','-C',str(root),*a],env=dict(os.environ,GIT_OPTIONAL_LOCKS='0',GIT_NO_LAZY_FETCH='1')).decode().strip()
assert g(r,'rev-parse','HEAD') == expected['head'] == g(r,'rev-parse','origin/m6-baselines')
assert g(r,'branch','--show-current')=='m6-baselines' and not g(r,'status','--porcelain')
assert g(r,'rev-list','--left-right','--count','HEAD...origin/m6-baselines').split()==['0','0']
x=expected['source']
assert g(s,'rev-parse','HEAD')==x['pin'] and g(s,'rev-parse','HEAD^{tree}')==x['tree']
assert subprocess.check_output(['git','-C',str(s),'status','--porcelain']).decode()==x['status']
assert {str(p.relative_to(s)) for p in s.rglob('*') if p.is_file() and '.git' not in p.relative_to(s).parts}==set(x['files'])
for rel,v in x['files'].items():
 p=s/rel; assert not p.is_symlink(); b=p.read_bytes()
 assert hashlib.sha256(b).hexdigest()==v['sha256'] and len(b)==v['size_bytes']
for rel in x['deleted']: assert not (s/rel).exists()
for blob in x['weight_blobs_absent']: assert subprocess.run(['git','-C',str(s),'cat-file','-e',blob],capture_output=True).returncode!=0
for rel in ['configs/methods/e03_stdn.yaml','frozen_config_snapshot/configs/methods/e03_stdn.yaml']:
 assert hashlib.sha256((r/rel).read_bytes()).hexdigest()==expected['config_sha256']
e=pathlib.Path('/home/student20261/miniconda3/envs/stdn'); rows=[]; meta={}
for base,dirs,files in os.walk(e):
 for name in sorted(dirs+files):
  p=pathlib.Path(base)/name; st=p.lstat(); rel=str(p.relative_to(e))
  rows.append([rel,st.st_mode,st.st_size,st.st_mtime_ns,os.readlink(p) if p.is_symlink() else None])
  if p.is_file() and (rel.startswith('conda-meta/') or ('.dist-info/' in rel and name in ['METADATA','RECORD','INSTALLER','direct_url.json'])): meta[rel]=hashlib.sha256(p.read_bytes()).hexdigest()
f={'entries':len(rows),'metadata_tree_sha256':hashlib.sha256(json.dumps(sorted(rows),separators=(',',':')).encode()).hexdigest(),'package_metadata_sha256':hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest(),'package_metadata_files':len(meta)}
assert f==expected['environment_fingerprint']
for key,v in expected['host'].items():
 if isinstance(v,dict) and 'argv' in v:
  p=subprocess.run(v['argv'],capture_output=True,text=True); assert p.returncode==v['exit_code']==0 and p.stdout==v['stdout']
assert pathlib.Path('/etc/os-release').read_text()==expected['host']['os_release']
print(json.dumps({'head':expected['head'],'source_files_verified':len(x['files']),'environment_unchanged':True,'host_matches':True,'gpu_worktree_clean':True}))
'''
    expected = {**a['gpu_final'], 'environment_fingerprint': a['existing_environment']['environment_after'],
                'host': a['existing_environment']['host']}
    # Supply program via -c and JSON via stdin; shlex quotes fixed code only.
    import shlex
    cmd = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', HOST,
           'python3 -B -c ' + shlex.quote(program)]
    p = subprocess.run(cmd, input=json.dumps(expected), text=True, capture_output=True, timeout=60)
    require(p.returncode == 0, 'live GPU: ' + p.stderr)
    return json.loads(p.stdout)


def check_index():
    raw = read(INDEX)
    require(b'\r\n' in raw and b'\n' not in raw.replace(b'\r\n', b''), 'index CRLF')
    rows = list(csv.DictReader(io.StringIO(raw.decode())))
    indexed = {r['path'] for r in rows}
    require(len(indexed) == len(rows), 'unique index paths')
    expected = set()
    for rel in git('ls-files', '--cached', '--others', '--exclude-standard', '-z').decode().split('\0'):
        if not rel:
            continue
        p = ROOT / rel
        if (p.is_file() and rel not in {INDEX, LEDGER} and p.name != '.gitkeep'
                and not {'.git', '__pycache__', '.venv'} & set(Path(rel).parts)
                and not rel.startswith(('data/raw/', 'data/processed/', 'cache/'))):
            expected.add(rel)
    require(indexed == expected, 'canonical index path coverage')
    for row in rows:
        b = read(row['path'])  # Hash only, including opaque manifest bytes; no parsing/decoding.
        require(len(b) == int(row['size_bytes']) and sha(b) == row['sha256'], 'index hash: ' + row['path'])
    return {'path_count': len(rows), 'sha256': sha(raw), 'crlf': True}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--draft', action='store_true')
    ap.add_argument('--live-gpu', action='store_true')
    args = ap.parse_args()
    require(git('rev-parse', 'HEAD').decode().strip() == HEAD == git('rev-parse', 'origin/m6-baselines').decode().strip(), 'laptop HEAD/origin')
    require(git('branch', '--show-current').decode().strip() == 'm6-baselines', 'laptop branch')
    require(git('rev-list', '--left-right', '--count', 'HEAD...origin/m6-baselines').split() == [b'0', b'0'], 'laptop divergence')
    changed = set(git('diff', '--name-only', 'HEAD', '-z').decode().split('\0')) - {''}
    changed |= set(git('ls-files', '--others', '--exclude-standard', '-z').decode().split('\0')) - {''}
    require(changed <= ALLOWED and DELIVERABLES <= changed, 'six-path firewall')
    a = json.loads(read(AUDIT))
    require(a['starting_laptop'] == {'head': HEAD, 'origin': HEAD, 'branch': 'm6-baselines', 'clean': True, 'divergence': [0, 0]}, 'initial laptop')
    sync = a['gpu_sync']
    require(sync['pre_head'] == OLD and sync['post_head'] == HEAD and sync['remote_target'] == HEAD and sync['distance'] == 1 and sync['ancestry'] is True and sync['pre_clean'] and sync['post_clean'], 'GPU sync evidence')
    require(git('rev-list', '--count', OLD + '..' + HEAD).strip() == b'1', 'one commit distance')
    require(subprocess.run(['git', '-C', str(ROOT), 'merge-base', '--is-ancestor', OLD, HEAD]).returncode == 0, 'ancestry')
    gpu = a['gpu_final']
    require(gpu['head'] == gpu['origin'] == HEAD and gpu['clean'] and gpu['divergence'] == [0, 0] and gpu['branch'] == 'm6-baselines', 'final GPU')
    cfg = 'configs/methods/e03_stdn.yaml'
    b = read(cfg)
    require(b == read('frozen_config_snapshot/' + cfg) == git('show', HEAD + ':' + cfg), 'frozen config byte identity')
    require(sha(b) == gpu['config_sha256'] == a['frozen_contract']['config_sha256'], 'config SHA')
    for fragment in ['method_id: E03', 'fidelity_class: FAITHFUL_OFFICIAL', 'learned_method: true',
                     'input_resolution: 256', 'map_size: 32', 'batch_size: 2', 'g_d_ratio: 2',
                     'max_epoch: 50', 'steps_per_epoch: 2000', 'val_steps: 500',
                     'experiment_seeds: [42, 1337, 2026]', 'rule: OFFICIAL_LATEST_FINAL_CKPT_50', PIN,
                     'TEST:\n      used_for: []\n      allowed: false']:
        require(fragment in b.decode(), 'frozen field: ' + fragment)
    pins = json.loads(read('third_party/source_pins.json'))['sources']['stdn']
    require(pins['pinned_commit'] == PIN, 'source provenance pin')
    source_check(ROOT / 'third_party/source_cache/stdn', a['laptop_source'], pins)
    require(all(gpu['source'][k] == a['laptop_source'][k] for k in ['pin', 'tree', 'files', 'deleted', 'status']), 'GPU/laptop cache match')
    require(len(gpu['source']['weight_blobs_absent']) == 4 and a['source_deployment']['weight_objects_transferred'] == 0, 'no weight objects deployed')
    require(a['source_evidence']['api_inventory'] == api_inventory(ROOT / 'third_party/source_cache/stdn'), 'complete static TF API inventory')
    readme = (ROOT / 'third_party/source_cache/stdn/README.md').read_text().splitlines()[9]
    require(a['source_evidence']['readme_line_10'] == readme and a['source_evidence']['mathematical_conflict'] is True, 'README conflict retained')
    env = a['existing_environment']
    require(env['versions']['python'] == '3.10.21' and env['versions']['tensorflow'] == '2.21.0' and env['versions']['torch'] == '2.5.1+cu121' and env['versions']['torchvision'] == '0.20.1+cu121', 'live package versions')
    require(env['environment_before'] == env['environment_after'] and env['environment_unchanged'] and env['guard']['blocked'] == [], 'environment mutation gate')
    require(env['probe']['tensorflow'] == {'available': True, 'version': '2.21.0'}, 'TF import succeeded')
    for symbol in ['tensorflow.contrib', 'Session', 'ConfigProto', 'py_func', 'image.resize_images', 'train.AdamOptimizer']:
        require(env['probe'][symbol]['available'] is False, 'live incompatible API: ' + symbol)
    require(a['existing_environment_classification'] == 'EXISTING_ENV_NOT_SOURCE_COMPATIBLE', 'environment classification')
    h = env['host']
    require(h['gpu']['stdout'].strip() == 'NVIDIA GeForce RTX 3090, 595.84, 8.6, 24576 MiB', 'GPU/driver/capability')
    require('V12.0.140' in h['nvcc']['stdout'] and h['glibc']['stdout'].strip() == 'glibc 2.39' and '7.0.0-29-generic' in h['kernel']['stdout'] and 'Ubuntu 24.04.4 LTS' in h['os_release'] and h['conda']['stdout'].strip() == 'conda 26.7.2', 'host runtime facts')
    required = {'id', 'python', 'tensorflow', 'source_evidence_strength', 'package_availability', 'gpu_runtime', 'cuda', 'cudnn', 'rtx3090_viability', 'unchanged_source', 'tf_contrib', 'patching_required', 'fidelity_implications', 'ready', 'evidence_refs'}
    matrix = a['candidates']
    require({r['id'] for r in matrix} == {'A', 'B_OLD', 'B_112', 'C_1131', 'C_115', 'C_NVIDIA', 'D_TF2'}, 'candidate coverage')
    for row in matrix:
        require(required <= row.keys() and all(row[k] not in (None, '') for k in required - {'ready'}), 'candidate completeness')
        require(row['ready'] is False, 'no unauthorized ready candidate')
    require(a['decision'] == 'BLOCKED_BY_MULTIPLE_REASONS' and a['recommended_m6d2b_candidate'] is None, 'decision consistency')
    require(a['method_status'] == 'IMPLEMENTED_NOT_EXECUTED' and a['environment_status'] == 'ENVIRONMENT_NOT_QUALIFIED', 'E03 not qualified')
    require(set(a['safety']) == set(PROHIBITED) and all(v is False for v in a['safety'].values()), 'scientific/environment firewall')
    for rel, digest in a['reviewed_inputs_sha256'].items():
        require(sha(read(rel)) == digest == sha(git('show', HEAD + ':' + rel)), 'reviewed input unchanged: ' + rel)
    prefix = git('show', HEAD + ':' + LEDGER)
    require(len(prefix.splitlines()) == 98 and len(prefix) == 248107 and sha(prefix) == PREFIX_SHA, 'committed ledger prefix')
    ledger = read(LEDGER)
    require(ledger[:len(prefix)] == prefix, 'first 98 rows byte identical')
    if args.draft:
        require(ledger == prefix, 'draft before append')
    else:
        require(len(ledger.splitlines()) == 99, 'exactly 99 ledger rows')
        row = json.loads(ledger[len(prefix):])
        require(row['classification'] == 'M6D2A_E03_ENVIRONMENT_RESOLUTION' and row['records_appended'] == 1 and row['committed_prefix_sha256'] == PREFIX_SHA, 'single ledger record')
        require(row['decision'] == a['decision'] and set(row['file_sha256']) == DELIVERABLES, 'ledger evidence linkage')
        for rel, digest in row['file_sha256'].items():
            require(sha(read(rel)) == digest, 'finalized artifact hash: ' + rel)
    result = {'result': 'M6D2A_AUDIT_VALIDATION_PASS', 'decision': a['decision'], 'scientific_execution': False,
              'ledger_rows': len(ledger.splitlines()), 'prefix_sha256': PREFIX_SHA}
    if args.live_gpu:
        result['live_gpu'] = live_gpu(a)
    if not args.draft:
        result['artifact_index'] = check_index()
        require(changed == ALLOWED, 'exact final six paths')
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
