#!/usr/bin/env python3
"""Verify retained M6D4a evidence; rebuild the index from committed metadata.

No model launch, data traversal, manifest payload read or image I/O occurs here.
Only the explicit new artifact allowlist is hashed; historical index rows are
carried from the authoritative commit without opening their targets.
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
AUTHORITY = '05e90efa6c36e11405822c79fed8511c2179f043'
BASE = 'outputs/audit/M6D4A_E05_RUNTIME_ARCHITECTURE_QUALIFICATION'
INDEX = 'outputs/audit/ARTIFACT_INDEX.csv'
LEDGER = 'outputs/audit/EXECUTION_LEDGER.jsonl'
PROCESSES = tuple(f'outputs/audit/M6D4A_E05_SYNTHETIC_PROCESS_{i}.json' for i in (1,2))
ENVIRONMENTS = tuple('environments/e05.'+s for s in
    ('runtime.json','lock.json','compatibility_patch_manifest.json','conda-explicit.txt','pip-freeze.txt'))
ARTIFACTS = ('methods/pcgan/runtime_qualification.py','tests/test_m6d4a_e05_runtime.py',
    'tools/m6d4a_e05_runtime_preflight.py',*ENVIRONMENTS,*PROCESSES,
    'outputs/audit/M6D4A_E05_RUNTIME_LOG.txt','outputs/audit/M6D4A_GPU_SYNC.json',BASE+'.md',BASE+'.json')
FINAL_STATUS = ['E05_ARCHITECTURE_RUNTIME_QUALIFIED','E05_TRAINING_RUNNER_NOT_YET_QUALIFIED',
                'IMPLEMENTED_NOT_EXECUTED','CONTROLLED_ADAPTATION']


def require(ok, message):
    if not ok: raise ValueError('M6D4a: '+message)


def sha(raw): return hashlib.sha256(raw).hexdigest()


def git(*args): return subprocess.check_output(['git','-C',str(ROOT),*args])


def read(path):
    p=Path(path)
    require(not p.is_absolute() and '..' not in p.parts,'relative evidence path')
    require(p.parts[0] in {'methods','tools','tests','docs','configs','frozen_config_snapshot','environments','outputs'},'evidence root')
    require(not {'data','manifests','faces_256','runs'} & set(p.parts),'data firewall')
    return (ROOT/p).read_bytes()


def compare(p,q):
    structural = {k:p[k]==q[k] for k in ('diagnostic_seed','source_before','models',
        'environment_before','parameter_sha256_before','generated_copy_manifest_sha256')}
    structural['extensions'] = all(p['custom_ops'][n]['extension']==q['custom_ops'][n]['extension'] for n in p['custom_ops'])
    require(all(structural.values()),'structural repeatability')
    fields = {'forward': (p['forward'],q['forward'])}
    fields.update({'gradient/'+n:(v,q['gradients'][n]) for n,v in p['gradients'].items()})
    measurements = {}
    for key,(a,b) in fields.items():
        require(a.keys()==b.keys(),'same diagnostic tensor inventory')
        for name,x in a.items():
            y=b[name]
            require(x['shape']==y['shape'] and x['dtype']==y['dtype'],'diagnostic shape/dtype')
            measurements[key+'/'+name]={'bitwise_equal':x['sha256']==y['sha256'],
                'summary_absolute_differences':{s:abs(x[s]-y[s]) for s in ('min','max','mean','l2')}}
    return {'independent_processes':2,'structural_identity':structural,
        'all_forward_bitwise_equal':all(v['bitwise_equal'] for k,v in measurements.items() if k.startswith('forward/')),
        'all_gradient_bitwise_equal':all(v['bitwise_equal'] for k,v in measurements.items() if k.startswith('gradient/')),
        'measurements':measurements,
        'measurement_limit':'Tensor hashes and scalar-statistic differences; scalar differences are not elementwise error bounds. No output images/tensor bank persisted.',
        'nondeterminism_note':'Backward variance is consistent with native CUDA accumulation, including bilinear interpolation backward in the generator skip and grid_sample backward in the patch-crop path. Kernel-level attribution is inferred, not directly profiled. Scientific operations were not altered.'}


def expected_index():
    reader=csv.DictReader(io.StringIO(git('show',AUTHORITY+':'+INDEX).decode()))
    baseline=list(reader);rows={r['path']:r for r in baseline}
    require(reader.fieldnames==['path','size_bytes','sha256'] and len(rows)==len(baseline)==530,'baseline index')
    for p in ARTIFACTS:
        require(p not in rows,'additive artifact '+p)
        raw=read(p);rows[p]={'path':p,'size_bytes':str(len(raw)),'sha256':sha(raw)}
    out=io.StringIO(newline='')
    writer=csv.DictWriter(out,fieldnames=reader.fieldnames,lineterminator='\r\n')
    writer.writeheader();writer.writerows(rows[k] for k in sorted(rows))
    return out.getvalue().encode(),len(rows)


def verify(check_index=True):
    for ref in ('HEAD','origin/m6-baselines'):
        require(git('rev-parse',ref).decode().strip()==AUTHORITY,ref+' authority')
    audit=json.loads(read(BASE+'.json'))
    require(audit['final_status']==FINAL_STATUS,'qualification status')
    for path,h in audit['artifacts_sha256'].items():require(sha(read(path))==h,'artifact '+path)
    for path,h in audit['preserved_inputs_sha256'].items():
        raw=read(path)
        require(sha(raw)==h and raw==git('show',AUTHORITY+':'+path),'immutable input '+path)
    p,q=[json.loads(read(n)) for n in PROCESSES]
    for r in (p,q):
        require(r['status']=='PASS','GPU process pass')
        require(r['source_before']==r['source_after'],'source mutation')
        require(r['environment_before']==r['environment_after'],'environment mutation')
        require(r['parameter_sha256_before']==r['parameter_sha256_after'],'parameter mutation')
        require(not r['firewall']['denied'] and not r['benchmark_data_access'],'data access')
        require(r['optimizer_steps']==r['checkpoints']==r['pretrained_loads']==0,'execution scope')
        require(not r['synthetic_bank'] and not r['five_loss_runner_qualified'],'runner scope')
    require(compare(p,q)==audit['repeatability'],'repeatability evidence')
    lock=read('environments/e05.lock.json')
    require(sha(lock)==audit['environment_lock_sha256'],'environment lock digest')
    patch=read('environments/e05.compatibility_patch_manifest.json')
    require(sha(patch)==p['generated_copy_manifest_sha256']==q['generated_copy_manifest_sha256'],'patch identity')
    runtime=json.loads(read('environments/e05.runtime.json'))
    require(runtime['created_by_E05'] and not runtime['reused_existing_environment'],'ownership')
    require(runtime['gpat_m5_unchanged'],'gpat-m5 identity')
    sync=json.loads(read('outputs/audit/M6D4A_GPU_SYNC.json'))
    require(sync['post_merge_hash_matches']==sync['archive_verified']==19,'recovery SHA256 equality')
    require(sync['ancestor_distance']==1 and sync['divergence']==[0,0] and sync['clean_after_sync'],'GPU sync')
    prefix=git('show',AUTHORITY+':'+LEDGER);current=read(LEDGER)
    require(len(prefix.splitlines())==105 and sha(prefix)=='10d52a4eb584d0dc200fe0e890794073861e484971f8131e023b9b5f13e6540f','ledger authority')
    require(current.startswith(prefix) and len(current.splitlines())==106,'exactly one append')
    row=json.loads(current[len(prefix):])
    require(row['classification']=='M6D4A_E05_RUNTIME_ARCHITECTURE_QUALIFICATION','ledger classification')
    for path,h in row['artifacts_sha256'].items():require(sha(read(path))==h,'ledger artifact '+path)
    expected,count=expected_index()
    if check_index:require(read(INDEX)==expected,'CRLF artifact index')
    return {'status':'PASS','ledger_rows':106,'first_105_rows_byte_identical':True,
            'artifact_index_rows':count,'independent_gpu_processes':2,'optimizer_steps':0,
            'benchmark_data_access':False,'model_execution_in_preflight':False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rebuild-index',action='store_true')
    args=p.parse_args()
    result=verify(check_index=not args.rebuild_index)
    if args.rebuild_index:
        (ROOT/INDEX).write_bytes(expected_index()[0]);result=verify()
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
