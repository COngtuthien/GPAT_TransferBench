#!/usr/bin/env python3
"""Validate E03 M6D2b evidence without constructing a model or running a step."""
import argparse
import base64
import csv
import gzip
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
HEAD='6aebf148bceb5e4d3692326dbabc6b3e2390fb9d'
PRE='372d456dddb635d840e35c0659951f7d3716fc79'
PREFIX='bd59b8d596a9977fd7cc25bc2dd1e5bdd215e3737168e2d52a2116e96c014df9'
BASE='outputs/audit/M6D2B_E03_RUNTIME_QUALIFICATION'
LOCK='environments/e03_stdn.lock.json'
ADDENDUM='docs/spec/amendments/GPAT_TransferBench_v1_0_E03_Runtime_Compatibility_Addendum.md'
LEDGER='outputs/audit/EXECUTION_LEDGER.jsonl'
INDEX='outputs/audit/ARTIFACT_INDEX.csv'
ALLOWED={BASE+'.json',BASE+'.md',LOCK,ADDENDUM,LEDGER,INDEX,
         'methods/stdn/runtime_qualification.py','tests/test_m6d2b_stdn_runtime.py',
         'tools/m6d2b_e03_runtime_preflight.py','environments/e03_stdn.conda-explicit.txt',
         'environments/e03_stdn.pip-freeze.txt','environments/e03_stdn.runtime.json',
         'environments/e03_stdn.compatibility_patch_manifest.json'}


def sha(b):return hashlib.sha256(b).hexdigest()
def read(p):return (ROOT/p).read_bytes()
def load(p):return json.loads(read(p))
def require(ok,label):
    if not ok:raise RuntimeError('STOP_AND_REPORT: '+label)
def git(*a):return subprocess.check_output(['git','-C',str(ROOT),*a],env=dict(os.environ,GIT_OPTIONAL_LOCKS='0'))


def normalize_topology(nodes):
    """Control inputs are prerequisite sets; data input order remains significant."""
    return sorted([dict(n,controls=sorted(n['controls'])) for n in nodes],key=lambda n:n['name'])


def historical_g_worker(q):
    p=q['g_worker_provenance'];b=p['source_utf8'].encode()
    require(sha(b)==p['sha256']=='c0326227f9f627ff5aaa2873d4259b96a61de553fc4208fdceb05586a761c3a9','original G worker provenance')
    # The new CLI/D observer must not change the already-qualified graph path.
    start=b'    sys.path.insert(0,';end=b"        if args.stage=='E'"
    require(b.split(start,1)[1].split(end,1)[0]==read('methods/stdn/runtime_qualification.py').split(start,1)[1].split(end,1)[0],'unchanged graph/forward construction path')
    return b


def validate_topology(q):
    normalized=[]
    original=historical_g_worker(q)
    observed=original.replace(b"result['graph_topology_sha256']=digest(canonical(graph_topology))",
                              b"result['graph_topology_sha256']=digest(canonical(graph_topology)); result['graph_topology_diagnostic']=graph_topology")
    require(len(q['topology_diagnostics'])==2,'two graph-only topology captures')
    for p in q['topology_diagnostics']:
        require(p['rc']==0 and p['worker_sha256']==sha(observed),'diagnostic worker is original plus output observer only')
        r=p['result_without_graph'];require(r['stage']=='C' and 'E' not in r,'no additional optimizer application')
        b=gzip.decompress(base64.b64decode(p['graph_json_gzip_base64']))
        require(sha(b)==p['raw_graph_sha256']==r['graph_topology_sha256'],'raw captured topology')
        n=normalize_topology(json.loads(b));nb=json.dumps(n,sort_keys=True,separators=(',',':')).encode()
        require(sha(nb)==p['canonical_graph_sha256'],'canonical topology hash')
        require(r['C']['structure_sha256']==q['repeat_runs'][0]['C']['structure_sha256'],'same graph/variable structure as executed trials')
        normalized.append(n)
    require(normalized[0]==normalized[1],'identical graph prerequisite sets and ordered data inputs')


def validate_d_optimizer(q):
    d=q.get('d_optimizer_qualification',{})
    require(d.get('status')=='PASS' and len(d.get('probes',[]))==2,'D optimizer path executed and repeated')
    require(d['environment_before']==d['environment_after'],'E03 environment/source bytes and metadata not mutated')
    runs=[]
    for p in d['probes']:
        require(p['rc']==0 and p['worker_sha256']==sha(read('methods/stdn/runtime_qualification.py')),'D worker execution provenance')
        r=json.loads(p['stdout'].split('E03_RESULT=',1)[1]);e=r['D_optimizer'];runs.append(r)
        require('E' not in r and e['optimizer_applications']==1 and e['generator_optimizer_applications']==0 and e['completed'],'exactly one D and zero G applications')
        require(e['loss_finite'] and math.isfinite(e['loss_before_application']) and e['gradient_count']==72 and e['all_gradients_finite'],'D loss and gradients finite')
        require(e['all_global_variables_finite'] and e['generator_trainable_variables_unchanged'] and e['generator_before_sha256']==e['generator_after_sha256'],'finite state and unchanged G trainable weights')
        changed=e['changed_discriminator_trainable_names']
        expected={n for n in r['C']['trainable_variables'] if n.startswith('Disc/')}
        require(set(changed)==expected and len(changed)==72,'all 72 expected D variables changed')
        require(all('/BN/moving_' in n and n.startswith('STDN/') for n in e['generator_nontrainable_changed_names']),'G state changes limited to native inline BN updates')
        require(e['global_step_before']==0 and e['global_step_after']==1,'source shared global-step increment')
        require(e['graph_before_sha256']==e['graph_after_sha256'],'graph unchanged by D application')
        require(e['canonical_topology_sha256']==q['repeatability']['supplemental_canonical_graph_sha256'] and r['C']['structure_sha256']==q['repeat_runs'][0]['C']['structure_sha256'],'same qualified topology/variables')
        require(len(e['adam_kernel_inputs_all_72'])==72 and all(all(math.isclose(x,y,rel_tol=1e-6,abs_tol=1e-12) for x,y in zip(row,[6e-5,0.9,0.999,1e-8])) for row in e['adam_kernel_inputs_all_72']),'all D Adam kernel settings')
        require(r['seed']==42 and r['precision_env']==q['repeat_runs'][0]['precision_env'] and r['session_config']==q['repeat_runs'][0]['session_config'],'same seed/precision controls')
        require(r['D']['input_shape']==[2,2,256,256,3] and r['D']['landmarks_shape']==[68,2] and r['D']['gpu_executed_nodes']>0,'synthetic paired batch and GPU execution')
        require(r['firewall']=={'denied':[],'benchmark_images_decoded':0,'test_data_accessed':False},'D scientific firewall')
    x,y=[r['D_optimizer'] for r in runs]
    for k in ['loss_before_application','gradient_global_norm']:
        require(math.isclose(x[k],y[k],rel_tol=q['repeatability']['rtol'],abs_tol=q['repeatability']['atol']),'D repeat diagnostic '+k)
    require(x['discriminator_after_sha256']==y['discriminator_after_sha256'] and x['post_trainable_weights_sha256']==y['post_trainable_weights_sha256'],'D repeat updated weights byte-identical')
    require(runs[0]['C']==runs[1]['C'],'D repeat variable/graph structure')


def validate():
    a=load(BASE+'.json');lock=load(LOCK);runtime=load('environments/e03_stdn.runtime.json')
    require(git('rev-parse','HEAD').decode().strip()==HEAD==git('rev-parse','origin/m6-baselines').decode().strip(),'laptop HEAD/origin')
    require(git('branch','--show-current').strip()==b'm6-baselines' and git('rev-list','--left-right','--count','HEAD...origin/m6-baselines').split()==[b'0',b'0'],'branch/divergence')
    changed=set(git('diff','--name-only','HEAD','-z').decode().split('\0'))-{''}
    changed|=set(git('ls-files','--others','--exclude-standard','-z').decode().split('\0'))-{''}
    require(changed<=ALLOWED,'allowed-path firewall')
    sync=a['gpu_sync'];require(sync['pre_head']==PRE and sync['post_head']==HEAD and sync['distance']==1 and sync['ancestry'] and sync['clean'],'one-commit GPU sync')
    require(a['decision']=='E03_ENVIRONMENT_QUALIFIED_WITH_COMPATIBILITY_RUNTIME','qualification decision')
    require(a['method_status']=='IMPLEMENTED_NOT_EXECUTED' and a['fidelity_disclosure']=='FAITHFUL_OFFICIAL_WITH_RUNTIME_COMPATIBILITY','method/fidelity status')
    require(a['environment_lock_sha256']==sha(read(LOCK)) and a['owner_addendum_sha256']==sha(read(ADDENDUM)),'lock/addendum hashes')
    for rel,digest in lock['artifact_sha256'].items():require(sha(read(rel))==digest,'locked artifact '+rel)
    require(lock['qualified_method']=='E03' and lock['source_pin']=='c79f1f8c615d2b8471b3df29da881bb18dd54c90','E03 lock identity')
    manifest=load('environments/e03_stdn.compatibility_patch_manifest.json')
    require(manifest['source_patches']==[] and manifest['tensorflow_api_shims']==[] and manifest['contrib_replacement'] is False,'original TF1/contrib preservation')
    for p in ['configs/methods/e03_stdn.yaml','frozen_config_snapshot/configs/methods/e03_stdn.yaml','third_party/source_pins.json']:
        require(read(p)==git('show',HEAD+':'+p),'frozen source authority '+p)
    require(read('configs/methods/e03_stdn.yaml')==read('frozen_config_snapshot/configs/methods/e03_stdn.yaml'),'snapshot')
    source=ROOT/'third_party/source_cache/stdn';prior=load('outputs/audit/M6D2A_E03_ENVIRONMENT_RESOLUTION.json')['laptop_source']
    require(runtime['source']['hashes']=={p:r['sha256'] for p,r in prior['files'].items()},'GPU source file hashes match authoritative cache')
    require(runtime['source']['head']['stdout'].strip()==lock['source_pin'] and runtime['source']['status']['stdout']==prior['status'],'GPU pin and four source-only deletions')
    for rel,row in prior['files'].items():require(sha((source/rel).read_bytes())==row['sha256'],'pinned cache '+rel)
    for rel in prior['deleted']:require(not (source/rel).exists(),'excluded weight '+rel)
    require({str(p.relative_to(source)) for p in source.rglob('*') if p.is_file() and '.git' not in p.relative_to(source).parts}==set(prior['files']),'no unexpected cache files or weights')
    require(subprocess.check_output(['git','-C',str(source),'status','--porcelain']).decode()==prior['status'],'source four deletions only')
    spec=importlib.util.spec_from_file_location('e03_worker',ROOT/'methods/stdn/runtime_qualification.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    g_worker=historical_g_worker(a['qualification'])
    for p in a['qualification']['stage_probes']:
        require(p['rc']==0 and p['worker_sha256']==sha(g_worker),'original G-stage process evidence')
    require([p['stage'] for p in a['qualification']['stage_probes']]==['A','B','C','D','E','E'],'stage order and clean repeat')
    runs=a['qualification']['repeat_runs'];require(len(runs)==2,'two clean qualification processes')
    for run in runs:
        require(run['A']['contrib_unchanged'] and run['B']['graph_ops_created']==0,'source imports')
        c,d,e=run['C'],run['D'],run['E']
        require(c['trainable_variables']==m.expected_variables() and c['adam_ops']==170 and c['weight_ema_count']==170 and c['loss_ema_count']==2,'model/optimizer/EMA structure')
        require(c['batch_norm_moving_count']==60 and c['regularization_losses']==0 and c['parameter_dtype']=='float32','normalization/regularization/precision')
        require(d['finite'] and d['gpu_executed_nodes']>0 and d['input_shape']==[2,2,256,256,3],'paired batch GPU forward')
        require(e['optimizer_applications']==1 and e['global_step_before']==0 and e['global_step_after']==1 and e['all_G_and_D_gradients_finite'] and e['all_weights_finite'],'one synthetic optimizer application')
        require(e['discriminator_update_executed'] is False and all(n.startswith('STDN/') for n in e['changed_trainable_names']),'G-only source schedule branch')
        require(run['precision_env']==m.PRECISION and 'disable_meta_optimizer: true' in run['session_config'],'precision controls')
        require(run['firewall']=={'denied':[],'benchmark_images_decoded':0,'test_data_accessed':False},'execution firewall')
    repeat=a['qualification']['repeatability']
    require(all(repeat[k] for k in ['structure_identical','supplemental_graph_topology_identical','initial_weights_identical','diagnostics_within_tolerance']),'repeatability')
    validate_topology(a['qualification'])
    validate_d_optimizer(a['qualification'])
    require(a['optimizer_path_coverage']=={'G':'PASS','D':'PASS'},'both optimizer paths qualified')
    d_evidence=json.dumps(a['qualification']['d_optimizer_qualification'],sort_keys=True,separators=(',',':')).encode()
    require(lock['qualification']['D_optimizer_application']=='PASS' and lock['qualification']['d_evidence_sha256']==sha(d_evidence),'D qualification locked')
    require(a['protected_environments_before']==a['protected_environments_after'],'protected environments unchanged')
    require(all(v is False for v in a['prohibited_actions'].values()),'scientific firewall')
    require(a['tests']['failures']==0 and a['tests']['errors']==0 and a['tests']['skipped']==0 and a['tests']['run']>=31,'focused tests')
    require(runtime['environment_path']=='/home/student20261/miniconda3/envs/gpat-m6-e03','isolated runtime')
    return a,changed


def check_index():
    b=read(INDEX);require(b'\r\n' in b and b'\n' not in b.replace(b'\r\n',b''),'CRLF')
    rows=list(csv.DictReader(io.StringIO(b.decode())));require(len({r['path'] for r in rows})==len(rows),'unique paths')
    for r in rows:require(sha(read(r['path']))==r['sha256'] and len(read(r['path']))==int(r['size_bytes']),'index hash '+r['path'])
    expected=set()
    for rel in git('ls-files','--cached','--others','--exclude-standard','-z').decode().split('\0'):
        p=ROOT/rel
        if rel and p.is_file() and rel not in {INDEX,LEDGER} and p.name!='.gitkeep' and not {'.git','__pycache__','.venv'}&set(p.parts) and not rel.startswith(('data/raw/','data/processed/','cache/')):expected.add(rel)
    require({r['path'] for r in rows}==expected,'index completeness')
    return {'path_count':len(rows),'sha256':sha(b),'crlf':True}


def live_gpu(a):
    """Read identities, package metadata and source hashes; never import TensorFlow."""
    code = '''import os,sys,json,hashlib,subprocess,pathlib,importlib.metadata as md
r=pathlib.Path('/home/student20261/workdir/GPAT_TransferBench')
def git(*args):return subprocess.check_output(['git','-C',str(r)]+list(args)).decode().strip()
def fingerprint(root):
 rows=[]
 for parent,dirs,files in os.walk(root):
  for name in dirs+files:
   p=pathlib.Path(parent)/name;s=p.lstat();rows.append([str(p.relative_to(root)),s.st_mode,s.st_size,s.st_mtime_ns,os.readlink(p) if p.is_symlink() else None])
 return {'entries':len(rows),'stat_sha256':hashlib.sha256(json.dumps(sorted(rows),separators=(',',':')).encode()).hexdigest()}
prior=json.loads((r/'outputs/audit/M6D2A_E03_ENVIRONMENT_RESOLUTION.json').read_text())['laptop_source'];s=r/'third_party/source_cache/stdn'
print(json.dumps({'head':git('rev-parse','HEAD'),'origin':git('rev-parse','origin/m6-baselines'),'branch':git('branch','--show-current'),'status':git('status','--porcelain'),'divergence':git('rev-list','--left-right','--count','HEAD...origin/m6-baselines').split(),'source_status':subprocess.check_output(['git','-C',str(s),'status','--porcelain']).decode(),'source_files':sorted(str(p.relative_to(s)) for p in s.rglob('*') if p.is_file() and '.git' not in p.relative_to(s).parts),'source_hashes':{p:hashlib.sha256((s/p).read_bytes()).hexdigest() for p in prior['files']},'deleted_absent':{p:not (s/p).exists() for p in prior['deleted']},'packages':sorted([{'name':d.metadata['Name'],'version':d.version} for d in md.distributions()],key=lambda d:d['name'].lower()),'protected_environments':{n:fingerprint(pathlib.Path('/home/student20261/miniconda3/envs')/n) for n in ['stdn','gpat-m5']},'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,compute_cap,driver_version,memory.total','--format=csv,noheader']).decode()}))
'''
    command=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','student20261@100.121.84.44',
             'env PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 /home/student20261/miniconda3/envs/gpat-m6-e03/bin/python -B -']
    p=subprocess.run(command,input=code,text=True,capture_output=True)
    require(p.returncode==0,'live GPU metadata: '+p.stderr)
    v=json.loads(p.stdout);runtime=load('environments/e03_stdn.runtime.json')
    require(v['head']==v['origin']==HEAD and v['branch']=='m6-baselines' and v['status']=='' and v['divergence']==['0','0'],'live GPU repository')
    require(v['source_status']==runtime['source']['status']['stdout'] and v['source_files']==sorted(runtime['source']['hashes']),'live source-only inventory and deletions')
    require(v['source_hashes']==runtime['source']['hashes'] and all(v['deleted_absent'].values()),'live source-only cache')
    require(v['packages']==runtime['packages'],'live isolated package lock')
    require(v['protected_environments']==a['protected_environments_before'],'live protected environments')
    require(v['gpu']==runtime['host']['nvidia_smi']['stdout'],'live GPU host facts')
    return {'result':'PASS','command':command,'probe_sha256':sha(code.encode()),'model_or_data_execution':False}


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--draft',action='store_true');ap.add_argument('--live-gpu',action='store_true');args=ap.parse_args()
    a,changed=validate()
    b=git('show',HEAD+':'+LEDGER);require(len(b.splitlines())==99 and sha(b)==PREFIX,'committed ledger prefix')
    now=read(LEDGER);require(now[:len(b)]==b,'byte-identical prefix')
    if args.draft:require(now==b,'draft before append')
    else:
        require(len(now.splitlines())==100,'exactly one appended record')
        row=json.loads(now[len(b):]);require(row['classification']=='M6D2B_E03_RUNTIME_QUALIFICATION','ledger milestone')
        for rel,digest in row['file_sha256'].items():require(sha(read(rel))==digest,'ledger hash '+rel)
        require(set(row['file_sha256'])==ALLOWED-{INDEX,LEDGER},'ledger artifact coverage')
        require(changed==ALLOWED,'final file scope')
    result={'result':'M6D2B_PREFLIGHT_PASS','decision':a['decision'],'G_optimizer_application':'PASS','D_optimizer_application':'PASS','ledger_rows':len(now.splitlines()),'prefix_sha256':PREFIX,'environment_lock_sha256':sha(read(LOCK))}
    if not args.draft:result['artifact_index']=check_index()
    if args.live_gpu:result['live_gpu']=live_gpu(a)
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
