#!/usr/bin/env python3
"""Verify retained M6D6a E07c evidence; rebuild the index from committed metadata.

STATIC: no Torch import, no CUDA, no model construction, no optimizer, no data
traversal, no manifest payload read and no image I/O. Only the explicit new
artifact allowlist is hashed; historical index rows are carried from the
authoritative commit without opening their targets.
"""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = '596d8459c9f2e54a5f36e2463099b662b800eb7f'
PIN = '23f40519ec25a833ebc06842aa6fbab74fad4d15'
TREE = 'd190a5fb4a94cb9e423215f9a24a7863aa17ea65'
CUSTOM_RN_SHA = 'fa788c4d2453b40585b295a8292eaf1afce7a0c622eb8ecc2adec5453a9a64c3'
CONSUMER_SHA = '127ecd59fbbbd0d191a713aae62b11ea3de7058d42c2889100ff6a44199d2963'
BASE = 'outputs/audit/M6D6A_E07C_RUNTIME_QUALIFICATION'
INDEX = 'outputs/audit/ARTIFACT_INDEX.csv'
LEDGER = 'outputs/audit/EXECUTION_LEDGER.jsonl'
LEDGER_PREFIX_ROWS = 115
LEDGER_PREFIX_SHA = '856d5d629e3733d6bc2e1f39c7b4df9d19ee6d10b5f7cb857cb835b5a7f29f82'
INDEX_BASELINE_ROWS = 639
CLASSIFICATION = 'M6D6A_E07C_RUNTIME_QUALIFICATION'
QUALIFICATION_SEED = 60601
PROCESSES = tuple(f'outputs/audit/M6D6A_E07C_SYNTHETIC_PROCESS_{i}.json' for i in (1, 2))
ENVIRONMENTS = tuple('environments/e07c.' + s for s in
                     ('runtime.json', 'lock.json', 'conda-explicit.txt', 'pip-freeze.txt', 'pip-requirements.txt'))
ARTIFACTS = ('methods/difffas/runtime_qualification.py', 'tests/test_m6d6a_e07c_runtime.py',
             'tools/m6d6a_e07c_runtime_preflight.py', *ENVIRONMENTS, *PROCESSES,
             'outputs/audit/M6D6A_E07C_RUNTIME_LOG.txt', BASE + '.md', BASE + '.json')
PRESERVED = ('configs/methods/e07c_difffas_bin_idfree.yaml', 'configs/frozen/difffas_bin_idfree_v1.yaml',
             'frozen_config_snapshot/configs/methods/e07c_difffas_bin_idfree.yaml',
             'frozen_config_snapshot/configs/frozen/difffas_bin_idfree_v1.yaml',
             'configs/amendments/e07c_a6_feature_interface_source_correction.yaml',
             'docs/spec/amendments/GPAT_TransferBench_v1_0_Amendment_A1_Fair_IDFree_Main_Track.md',
             'docs/spec/amendments/GPAT_TransferBench_v1_0_Amendment_A2_M6_Baseline_Execution_Contracts.md',
             'docs/spec/amendments/GPAT_TransferBench_v1_0_Amendment_A3_Controlled_Reconstruction_E04_E07c.md',
             'docs/spec/amendments/GPAT_TransferBench_v1_0_Amendment_A6_E07c_Feature_Interface_Source_Correction.md',
             'docs/spec/GPAT_TransferBench_v1_0_Frozen_Specification_2026.docx',
             'configs/run_logging_v1.yaml', 'configs/CONFIG_STATUS.md', 'outputs/audit/STAGE_STATE.json',
             'outputs/audit/M6C2B3_IMPLEMENTATION.md', 'outputs/audit/M6C2B3_IMPLEMENTATION.json',
             'third_party/source_pins.json', 'methods/difffas/__init__.py', 'methods/difffas/adapter.py',
             'methods/difffas/contract.py', 'methods/difffas/encoder.py', 'methods/difffas/source.py',
             'methods/difffas/sampler.py', 'methods/difffas/seed_adapter.py',
             'methods/difffas/source_traceability.md', 'methods/common/learned.py', 'methods/common/upstream.py',
             'methods/common/config.py', 'tests/test_m6c2b3_contract.py', 'tests/test_m6c2b3_encoder.py',
             'tests/test_m6c2b3_runtime.py', 'tools/m6c2b3_preflight.py')
SPEC_SHA = 'f7d2371682ad9504f59f0c6984e4b1a741f855f1b1f60838f9d60b160281489e'
FINAL_STATUS = ['E07c_EXECUTION_ENVIRONMENT_QUALIFIED', 'E07c_CONDITIONING_ENCODER_RUNTIME_QUALIFIED',
                'E07c_MAIN_ARCHITECTURE_RUNTIME_QUALIFIED', 'E07c_SYNTHETIC_FORWARD_RUNTIME_QUALIFIED',
                'IMPLEMENTED_NOT_EXECUTED', 'CONTROLLED_ADAPTATION']
NOT_QUALIFIED = ['REAL_TRAIN_PATH', 'TRAINING_GRAPH_BACKWARD', 'TRAINING_MEMORY', 'AUXILIARY_ENCODER_TRAINING',
                 'CHECKPOINT_WRITER', 'RESUME', 'PRODUCTION_RUNNER', 'SCIENTIFIC_TRAINING', 'M8_BANK']
SCOPE = {'training_launched': False, 'optimizer_constructed': False, 'optimizer_applications': 0,
         'backward_passes': 0, 'checkpoint_created': False, 'checkpoint_loaded': False,
         'auxiliary_encoder_trained': False, 'benchmark_data_access': False, 'TRAIN_access': False,
         'VAL_access': False, 'TEST_access': False, 'synthetic_bank': False, 'diffusion_sampling': False,
         'training_graph_qualified': False}
FEATURES = {'x32x32': [256, 32, 32], 'x16x16': [512, 16, 16], 'x8x8': [512, 8, 8]}


def require(ok, message):
    if not ok:
        raise ValueError('M6D6a: ' + message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def git(*args, root=ROOT):
    return subprocess.check_output(['git', '-C', str(root), *args])


def read(path):
    p = Path(path)
    require(not p.is_absolute() and '..' not in p.parts, 'relative evidence path')
    require(p.parts[0] in {'methods', 'tools', 'tests', 'configs', 'environments', 'outputs', 'third_party',
                           'docs', 'frozen_config_snapshot'}, 'evidence root')
    require(not {'data', 'manifests', 'faces_256', 'runs'} & set(p.parts), 'data firewall')
    return (ROOT / p).read_bytes()


def source_cache():
    """Git metadata and the pinned closure only; no weights exist in this repository."""
    root = ROOT / 'third_party/source_cache/difffas'
    require(git('rev-parse', 'HEAD', root=root).decode().strip() == PIN, 'source commit')
    require(git('rev-parse', 'HEAD^{tree}', root=root).decode().strip() == TREE, 'source tree')
    tracked = git('status', '--porcelain', '--untracked-files=no', root=root).decode().strip()
    require(tracked == '', 'tracked pinned source unmodified')
    return root


def repeatability(p, q):
    fields = ('qualification_seed', 'contract', 'source_before', 'source_after', 'encoder', 'main_model',
              'diffusion', 'forward', 'connectivity', 'forward_loss_path', 'cuda_memory', 'environment_before',
              'counters', 'firewall', 'feature_interface_live_CHW', 'warnings')
    identity = {k: p[k] == q[k] for k in fields}
    bitwise = {n: p['forward'][n]['sha256'] == q['forward'][n]['sha256'] for n in p['forward']}
    internal = {n: v['sha256'] == q['connectivity']['internal_activations'][n]['sha256']
                for n, v in p['connectivity']['internal_activations'].items()}
    return {'independent_processes': 2, 'field_identity': identity,
            'forward_tensors_compared': len(bitwise), 'forward_bitwise_equal': sum(bitwise.values()),
            'internal_stage_tensors_compared': len(internal), 'internal_stage_bitwise_equal': sum(internal.values()),
            'all_bitwise_equal': all(bitwise.values()) and all(internal.values()),
            'initial_encoder_parameters_bitwise_equal':
                p['encoder']['state']['aggregate_parameter_sha256'] == q['encoder']['state']['aggregate_parameter_sha256'],
            'initial_main_parameters_bitwise_equal':
                p['main_model']['state_initial']['aggregate_parameter_sha256'] ==
                q['main_model']['state_initial']['aggregate_parameter_sha256'],
            'process_json_bytes_identical': sha(json.dumps(p, sort_keys=True).encode()) ==
                                            sha(json.dumps(q, sort_keys=True).encode()),
            'determinism_note': ('Fixed qualification seed, frozen cudnn benchmark=False/deterministic=True, TF32 off, '
                                 'CUBLAS_WORKSPACE_CONFIG=:4096:8; torch deterministic algorithms were NOT forced. '
                                 'Equality is observed, not manufactured.')}


def check_process(r):
    require(r['status'] == 'PASS' and r['label'] == 'SYNTHETIC_FORWARD_RUNTIME_ONLY', 'GPU process pass')
    require(r['qualification_seed'] == QUALIFICATION_SEED and r['experiment_seed'] is None and
            QUALIFICATION_SEED not in (42, 1337, 2026), 'qualification seed')
    require(r['source_before'] == r['source_after'] and r['source_before']['worktree_status'] == '', 'source mutation')
    s = r['source_before']
    require((s['commit'], s['tree']) == (PIN, TREE), 'pinned source identity')
    require(s['files_sha256']['models/custom_rn.py'] == CUSTOM_RN_SHA and
            s['files_sha256']['models/unet_autoenc.py'] == CONSUMER_SHA, 'A6 source digests')
    require(r['environment_before'] == r['environment_after'], 'environment mutation')
    e = r['encoder']
    require((e['module'], e['topology'], e['stage_widths'], e['structural_difference_vs_pinned_resnet18']) ==
            ('custom_rn', [3, 4, 6, 3], [64, 256, 512, 512], ['fc']), 'pinned custom_rn encoder')
    require(e['head'] == {'type': 'Linear', 'in_features': 512, 'out_features': 7} and
            not e['torchvision_substitute'] and not e['projection_or_adapter_modules_added'] and
            not e['weights_loaded'], 'A3 head, no substitute/projection/weights')
    for name, chw in FEATURES.items():
        require(r['forward']['encoder_' + name]['shape'] == [4, *chw], 'live A6 shape ' + name)
    require(r['forward']['encoder_embg']['shape'] == [4, 7], 'embg [B,7]')
    m = r['main_model']
    require(m['class'] == 'BeatGANsAutoencModel' and m['first_conv_in_channels'] == 3 and m['use_pair'] is False,
            'main model / use_pair=false / 3-channel')
    for st in (e['state'], m['state_initial']):
        require(st['parameters'] == sum(math.prod(v) for v in st['shapes'].values()), 'parameter arithmetic')
    c = r['connectivity']
    require(c['fourth_output_consumers'] == 0 and c['feature_consumer_blocks']['embg'] == [], 'fourth output discarded')
    require(all(c['feature_consumer_blocks'][k] for k in FEATURES), 'three features consumed')
    require(all(v > 0 for k in FEATURES for v in c['zeroing_sensitivity_max_abs_at_consumer_attention_blocks'][k].values()),
            'consumer sensitivity')
    require(r['finite_all'] and all(v['finite'] for v in c['internal_activations'].values()), 'finite outputs')
    require(r['forward_loss_path']['label'] == 'FORWARD_LOSS_PATH_ONLY' and
            not r['forward_loss_path']['backward_executed'], 'forward loss path only')
    require(r['counters'] == {'optimizer_constructions': 0, 'optimizer_applications': 0, 'backward_passes': 0,
                              'autograd_grad_calls': 0, 'checkpoint_saves': 0, 'checkpoint_loads': 0},
            'zero training/checkpoint calls')
    require(all(r[k] == v for k, v in SCOPE.items()), 'scope flags')
    fw = r['firewall']
    require(not fw['denied'] and not any(fw['attempts'].values()), 'firewall zero denials')
    require(r['compatibility_patch'] == 'NONE' and r['fidelity'] == 'CONTROLLED_ADAPTATION', 'patch/fidelity')
    mem = r['cuda_memory']
    require(mem['label'] == 'SYNTHETIC_FORWARD_MEMORY_OBSERVATION' and not mem['qualifies_training_memory'] and
            not mem['qualifies_scientific_batch_size'] and not mem['qualifies_400_epoch_feasibility'], 'memory label')


def expected_index():
    reader = csv.DictReader(io.StringIO(git('show', AUTHORITY + ':' + INDEX).decode()))
    baseline = list(reader)
    rows = {r['path']: r for r in baseline}
    require(reader.fieldnames == ['path', 'size_bytes', 'sha256'] and
            len(rows) == len(baseline) == INDEX_BASELINE_ROWS, 'baseline index')
    for p in ARTIFACTS:
        require(p not in rows, 'additive artifact ' + p)
        raw = read(p)
        rows[p] = {'path': p, 'size_bytes': str(len(raw)), 'sha256': sha(raw)}
    out = io.StringIO(newline='')
    writer = csv.DictWriter(out, fieldnames=reader.fieldnames, lineterminator='\r\n')
    writer.writeheader()
    writer.writerows(rows[k] for k in sorted(rows))
    return out.getvalue().encode(), len(rows), baseline


def worktree():
    status = sorted(git('status', '--porcelain', '--untracked-files=all').decode().splitlines())
    expected = sorted([' M ' + INDEX, ' M ' + LEDGER] + ['?? ' + p for p in ARTIFACTS])
    require(status == expected, 'worktree holds exactly the intended M6D6a changes: ' + json.dumps(status))
    return status


def verify(check_index=True, check_ledger=True):
    for ref in ('HEAD', 'origin/m6-baselines'):
        require(git('rev-parse', ref).decode().strip() == AUTHORITY, ref + ' authority')
    require(sha(read('docs/spec/GPAT_TransferBench_v1_0_Frozen_Specification_2026.docx')) == SPEC_SHA, 'spec SHA256')
    for path in PRESERVED:
        require(read(path) == git('show', AUTHORITY + ':' + path), 'immutable input ' + path)
    root = source_cache()
    audit = json.loads(read(BASE + '.json'))
    require(audit['final_status'] == FINAL_STATUS and audit['not_qualified'] == NOT_QUALIFIED, 'qualification status')
    require(audit['method_status'] == 'IMPLEMENTED_NOT_EXECUTED', 'method status unchanged')
    for path, h in audit['artifacts_sha256'].items():
        require(sha(read(path)) == h, 'artifact ' + path)
    for path, h in audit['preserved_inputs_sha256'].items():
        require(sha(read(path)) == h, 'preserved input ' + path)
    p, q = [json.loads(read(n)) for n in PROCESSES]
    for r in (p, q):
        check_process(r)
    for rel, h in p['source_before']['files_sha256'].items():
        require(sha((root / rel).read_bytes()) == h, 'laptop source closure ' + rel)
    rep = repeatability(p, q)
    require(rep == audit['repeatability'] and rep['all_bitwise_equal'] and rep['process_json_bytes_identical'],
            'repeatability evidence')
    lock = read('environments/e07c.lock.json')
    require(sha(lock) == audit['environment_lock_sha256'], 'environment lock digest')
    lockj = json.loads(lock)
    require(lockj['compatibility_patch'] == 'NONE' and
            lockj['runtime_json_sha256'] == sha(read('environments/e07c.runtime.json')), 'lock linkage')
    require(lockj['runtime_harness_sha256'] == sha(read('methods/difffas/runtime_qualification.py')), 'harness identity')
    for name, h in lockj['exports_sha256'].items():
        require(sha(read('environments/e07c.' + name)) == h, 'export ' + name)
    require(lockj['identity'] == p['environment_before'], 'lock identity = executed identity')
    runtime = json.loads(read('environments/e07c.runtime.json'))
    require(runtime['created_by_E07c'] and not runtime['reused_existing_environment'] and runtime['gpat_m5_unchanged']
            and runtime['protected_environment_fingerprints']['unchanged'], 'environment ownership')
    require(runtime['identity'] == p['environment_before'], 'runtime identity')
    log = read('outputs/audit/M6D6A_E07C_RUNTIME_LOG.txt').decode()
    require(log.count('scientific_paths_after: runs=ABSENT e07c_root=ABSENT encoder_final=ABSENT') ==
            log.count('scientific_paths_before:') >= 6, 'no scientific run root or auxiliary checkpoint created')
    prefix = git('show', AUTHORITY + ':' + LEDGER)
    current = read(LEDGER)
    require(len(prefix.splitlines()) == LEDGER_PREFIX_ROWS and sha(prefix) == LEDGER_PREFIX_SHA, 'ledger authority')
    require(current.startswith(prefix), 'ledger prefix byte-identical')
    if check_ledger:
        require(len(current.splitlines()) == LEDGER_PREFIX_ROWS + 1, 'exactly one append')
        row = json.loads(current[len(prefix):])
        require(row['classification'] == CLASSIFICATION and row['final_status'] == FINAL_STATUS and
                row['not_qualified'] == NOT_QUALIFIED and row['method_status'] == 'IMPLEMENTED_NOT_EXECUTED',
                'ledger row')
        require(all(row[k] == v for k, v in SCOPE.items()), 'ledger scope flags')
        for path, h in row['artifacts_sha256'].items():
            require(sha(read(path)) == h, 'ledger artifact ' + path)
    expected, count, _ = expected_index()
    if check_index:
        require(read(INDEX) == expected, 'CRLF artifact index')
        worktree()
    require('torch' not in sys.modules, 'static preflight must not import torch')
    return {'status': 'PASS', 'ledger_rows': len(current.splitlines()),
            f'first_{LEDGER_PREFIX_ROWS}_rows_byte_identical': True,
            'artifact_index_rows_before': INDEX_BASELINE_ROWS, 'artifact_index_rows': count,
            'independent_gpu_processes': 2, 'all_bitwise_equal': rep['all_bitwise_equal'],
            'optimizer_constructed': False, 'backward_passes': 0, 'torch_imported': False,
            'model_execution_in_preflight': False, 'benchmark_data_access': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rebuild-index', action='store_true')
    parser.add_argument('--before-ledger', action='store_true', help='evidence-only check before the ledger append')
    args = parser.parse_args()
    if args.before_ledger:
        result = verify(check_index=False, check_ledger=False)
    else:
        result = verify(check_index=not args.rebuild_index)
        if args.rebuild_index:
            (ROOT / INDEX).write_bytes(expected_index()[0])
            result = verify()
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
