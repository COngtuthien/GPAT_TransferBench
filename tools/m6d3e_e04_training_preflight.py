#!/usr/bin/env python3
"""Verify retained synthetic evidence without importing TensorFlow or data.

Index rebuilding uses committed metadata plus an explicit additive allowlist.
It never opens unchanged index targets, manifests, runtime directories or images.
No mode launches a graph, optimizer, benchmark run, commit or push.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = '90263169f0731b397dfa459d7ac94eaafd2c69ec'
BASE = 'outputs/audit/M6D3E_E04_TRAINING_GRAPH_QUALIFICATION'
LEDGER = 'outputs/audit/EXECUTION_LEDGER.jsonl'
INDEX = 'outputs/audit/ARTIFACT_INDEX.csv'
PREFIX_SHA = '5525fca5c6f3ee2d3ddb4bb9f02e0c50abfffc5e36493ab80bdf3a2b65e01eea'
SOURCES = tuple('methods/physics_std/'+n+'.py' for n in
                ('training_contract','network','trace_transfer','losses','training_graph','training_runtime'))
DOCUMENT = 'docs/spec/amendments/GPAT_TransferBench_v1_0_E04_Training_Graph_Implementation_M6D3e.md'
PROCESSES = tuple('outputs/audit/M6D3E_E04_SYNTHETIC_PROCESS_%d.json'%i for i in (1,2))
ENVIRONMENTS = tuple('environments/e04_training.'+s+'.json' for s in
                     ('runtime','lock','compatibility_patch_manifest'))
TOOL = 'tools/m6d3e_e04_training_preflight.py'
LOG = 'outputs/audit/M6D3E_E04_SYNTHETIC_RUNTIME_LOG.txt'
ARTIFACTS = SOURCES + (DOCUMENT,'tests/test_m6d3e_e04_training_graph.py',TOOL) + PROCESSES + ENVIRONMENTS + (LOG,BASE+'.md',BASE+'.json')
FINAL_STATUS = ['E04_GEOMETRY_RUNTIME_QUALIFIED','E04_TRAINING_GRAPH_CONTRACT_RESOLVED',
                'E04_TRAINING_GRAPH_QUALIFIED','E04_TRAINING_RUNTIME_QUALIFIED',
                'IMPLEMENTED_NOT_EXECUTED','CONTROLLED_ADAPTATION']


def require(ok, message):
    if not ok: raise ValueError('M6D3e: '+message)


def sha(raw): return hashlib.sha256(raw).hexdigest()


def git(*args):
    return subprocess.check_output(['git','-C',str(ROOT),*args])


def read(path):
    p=Path(path)
    require(not p.is_absolute() and '..' not in p.parts,'relative artifact path')
    require(p.parts[0] in {'methods','tools','tests','docs','configs','frozen_config_snapshot','environments','outputs'},'evidence root')
    require(not {'data','manifests','faces_256','runs'} & set(p.parts),'data firewall')
    require(p.suffix in {'.py','.md','.yaml','.json','.jsonl','.csv','.txt'} or p.name=='.gitkeep','text evidence only')
    return (ROOT/p).read_bytes()


def compare(p,q):
    exact_keys=['graph_def_sha256','operation_count','variables','parameters','outputs','groups',
                'optimizer_state','initialization','initial_state','forward','initial_losses',
                'final_losses','weighted_initial','lr_boundaries','gradient_inventory','fixture_sha256',
                'precision_env','session_config','native_runtime']
    exact={k:p[k]==q[k] for k in exact_keys}
    require(all(exact.values()),'required repeatability fields differ: '+str([k for k,v in exact.items() if not v]))
    steps=[]
    for a,b in zip(p['steps'],q['steps']):
        same={k:a[k]==b[k] for k in a if k!='post_state'}
        require(all(same.values()),'step losses/gradient or isolation behavior differs')
        names=[n for n in a['post_state'] if a['post_state'][n]!=b['post_state'][n]]
        # Measured native variance is retained, not hidden by rounding/tolerance.
        differences={n:{m:abs(a['post_state'][n][m]-b['post_state'][n][m])
                        for m in ('min','max','mean','l2')} for n in names}
        maxima={m:max((d[m] for d in differences.values()),default=0.) for m in ('min','max','mean','l2')}
        steps.append({'step':a['step'],'diagnostics_exact':same,'state_hashes_exact':not names,
                      'different_state_tensors':names,'summary_absolute_differences':differences,
                      'maximum_summary_absolute_differences':maxima})
    return {'independent_processes':2,'exact_fields':exact,'steps':steps,
        'raw_python_operation_list_hash_equal':p['operation_structure_sha256']==q['operation_structure_sha256'],
        'serialized_GraphDef_exact':True,
        'structure_note':'Serialized GraphDef bytes match exactly. The auxiliary Python operation-list hash is order-sensitive to native control-input enumeration and is not used to contradict identical serialized graphs.',
        'numerics_note':'Initial outputs/losses, all checked loss/step gradients, Step1 state, all G post-state and final losses are exact. Step2 re-evaluation yields native FP32 variance confined to D3 weights/Adam/BN state, retained through Step3. GPU convolution/reduction execution is the inferred source; the exact kernel is not established because CUPTI was unavailable. No scientific operator or extra optimizer application was used to force identity.',
        'measurement_limit':'Per-tensor hash and scalar-statistic differences are measured; these are not elementwise maximum-error bounds. Full tensors were not persisted.',
        'bitwise_post_state_repeatable':all(s['state_hashes_exact'] for s in steps)}


def expected_index():
    reader=csv.DictReader(io.StringIO(git('show',AUTHORITY+':'+INDEX).decode()))
    require(reader.fieldnames==['path','size_bytes','sha256'],'index schema')
    baseline=list(reader); rows={r['path']:r for r in baseline}
    require(len(rows)==len(baseline)==513,'baseline index')
    for p in ARTIFACTS:
        require(p not in rows,'additive artifact '+p)
        raw=read(p); rows[p]={'path':p,'size_bytes':str(len(raw)),'sha256':sha(raw)}
    out=io.StringIO(newline='')
    w=csv.DictWriter(out,fieldnames=reader.fieldnames,lineterminator='\r\n')
    w.writeheader(); w.writerows(rows[k] for k in sorted(rows))
    return out.getvalue().encode(),len(rows)


def verify(check_index=True):
    require(git('rev-parse','HEAD').decode().strip()==AUTHORITY,'HEAD changed')
    require(git('rev-parse','origin/m6-baselines').decode().strip()==AUTHORITY,'origin changed')
    audit=json.loads(read(BASE+'.json'))
    require(audit['decision']=='E04_TRAINING_GRAPH_AND_RUNTIME_QUALIFIED','qualification decision')
    require(audit['final_status']==FINAL_STATUS,'final status')
    for p,h in audit['artifacts_sha256'].items():require(sha(read(p))==h,'artifact hash '+p)
    for p,h in audit['preserved_inputs_sha256'].items():
        raw=read(p)
        require(sha(raw)==h and raw==git('show',AUTHORITY+':'+p),'preserved input '+p)
    p,q=[json.loads(read(n)) for n in PROCESSES]
    for r in (p,q):
        require(r['status']=='PASS' and r['optimizer_applications']==3,'one complete minibatch')
        require(not r['firewall']['denied'],'firewall')
        require(r['tests']['tests']==33 and not (r['tests']['errors']+r['tests']['failures']+r['tests']['skipped']),'measured tests')
        require([s['iteration_after'] for s in r['steps']]==[0,0,1],'iteration ownership')
    require(compare(p,q)==audit['repeatability'],'repeatability evidence')
    runtime=json.loads(read(ENVIRONMENTS[0]))
    require(runtime['environment_before']==runtime['environment_after'],'environment mutation')
    require(runtime['environment_owner']=='E03' and not runtime['created_by_E04'],'runtime ownership')
    require(sha(read(ENVIRONMENTS[1]))==audit['environment_lock_sha256'],'E04 reference lock')
    prefix=git('show',AUTHORITY+':'+LEDGER); current=read(LEDGER)
    require(len(prefix)==267781 and len(prefix.splitlines())==104 and sha(prefix)==PREFIX_SHA,'ledger authority')
    require(current.startswith(prefix) and len(current.splitlines())==105,'one append, unchanged first104')
    row=json.loads(current[len(prefix):])
    require(row['classification']=='M6D3E_E04_TRAINING_GRAPH_QUALIFICATION','ledger classification')
    require(row['decision']==audit['decision'],'ledger decision')
    for path,h in row['artifacts_sha256'].items():require(sha(read(path))==h,'ledger artifact '+path)
    index,count=expected_index()
    if check_index:require(read(INDEX)==index,'index exact CRLF metadata rebuild')
    return {'status':'PASS','ledger_rows':105,'prefix_byte_identical':True,'artifact_index_rows':count,
            'synthetic_optimizer_applications_total':6,'independent_processes':2,
            'graph_execution_in_preflight':False,'bitwise_post_state_repeatable':False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rebuild-index',action='store_true')
    args=p.parse_args()
    result=verify(check_index=not args.rebuild_index)
    if args.rebuild_index:
        (ROOT/INDEX).write_bytes(expected_index()[0])
        result=verify()
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
