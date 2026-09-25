#!/usr/bin/env python3
"""Verify M6D6a (initial deviation preserved) + M6D6a-r clean corrective evidence; rebuild the index.

STATIC: no Torch import, no CUDA, no model construction, no optimizer, no data
traversal, no manifest or image I/O. Only explicit allowlisted paths are read.
Historical index rows are carried from the authoritative commit without opening
their targets; only the explicit new artifacts are hashed.
Supersedes the never-run initial tools/m6d6a_e07c_runtime_preflight.py (kept unchanged).
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
SPEC_SHA = 'f7d2371682ad9504f59f0c6984e4b1a741f855f1b1f60838f9d60b160281489e'
PIN = '23f40519ec25a833ebc06842aa6fbab74fad4d15'
TREE = 'd190a5fb4a94cb9e423215f9a24a7863aa17ea65'
CUSTOM_RN_SHA = 'fa788c4d2453b40585b295a8292eaf1afce7a0c622eb8ecc2adec5453a9a64c3'
CONSUMER_SHA = '127ecd59fbbbd0d191a713aae62b11ea3de7058d42c2889100ff6a44199d2963'
A6_OVERLAY_SHA = 'dd3f29aa8ff96de9c0e2d504e6d07f4788d8fb8d030ffa26ef51443104ca971b'
LOCK_SHA = '0c909de1e3e3e8c129e0d9f4aab6386cee8e4eac0ca45d79423f795cced5f450'
QUALIFICATION_SEED = 60601
INDEX = 'outputs/audit/ARTIFACT_INDEX.csv'
LEDGER = 'outputs/audit/EXECUTION_LEDGER.jsonl'
LEDGER_PREFIX_ROWS = 115
LEDGER_PREFIX_SHA = '856d5d629e3733d6bc2e1f39c7b4df9d19ee6d10b5f7cb857cb835b5a7f29f82'
INDEX_BASELINE_ROWS = 639
CLASSIFICATION = 'M6D6A_E07C_RUNTIME_QUALIFICATION'
# Initial M6D6a candidate: recorded BEFORE the corrective rerun; must stay byte-identical.
INITIAL = {
    'outputs/audit/M6D6A_E07C_RUNTIME_QUALIFICATION.md': '8ce6b02cad29f448c6a369003f8f95d7002367a8263837820d00e41c94c6f611',
    'outputs/audit/M6D6A_E07C_SYNTHETIC_PROCESS_1.json': '0ccc5a963e4eeb277478ecdde725ce9c39151ed96dafb29de8825f21afa4dab2',
    'outputs/audit/M6D6A_E07C_SYNTHETIC_PROCESS_2.json': '0ccc5a963e4eeb277478ecdde725ce9c39151ed96dafb29de8825f21afa4dab2',
    'methods/difffas/runtime_qualification.py': '1eeb7c037dbb25876eb6257a4eb0dee40bc3a998d5d20c03a2b4b937abe12541',
    'tests/test_m6d6a_e07c_runtime.py': '15e3fb28455c77094f039b1851e136da35b8bceab13def6f830b7f10a76496dc',
    'tools/m6d6a_e07c_runtime_preflight.py': 'b7f02e2cafcdabf418aeadb63fc45c73025b0f6d1081b51e620840cc1008be7d',
    'environments/e07c.runtime.json': '0855943bb1aa00d12bf61f8145bdfad4598b467f8226b2cc05842902796d2d61',
    'environments/e07c.lock.json': LOCK_SHA,
    'environments/e07c.conda-explicit.txt': '2c5a2bbcacf767c54eb2f21ce2648a8bd265dbb57f6d0e0cdc01839d8b42b864',
    'environments/e07c.pip-freeze.txt': '42cebd80a0df9986b33fb09486f6dc619dd3695b1cbe0978cc67062cba9ea285',
    'environments/e07c.pip-requirements.txt': '9ea38ed8b8eb4de00dff577cfb7a1360201fa66dd6c28bf174a8bfb01de26f8f',
}
BASE = 'outputs/audit/M6D6A_R_E07C_CLEAN_REQUALIFICATION'
PROCESSES = tuple(f'outputs/audit/M6D6A_R_E07C_SYNTHETIC_PROCESS_{i}.json' for i in (1, 2))
CORRECTIVE = (*PROCESSES, 'outputs/audit/M6D6A_R_E07C_RUNTIME_LOG.txt',
              'tools/m6d6a_r_e07c_clean_requalification_preflight.py', BASE + '.md', BASE + '.json')
ARTIFACTS = tuple(sorted((*INITIAL, *CORRECTIVE)))
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
QUALIFIED = ['E07c_EXECUTION_ENVIRONMENT_QUALIFIED', 'E07c_CONDITIONING_ENCODER_RUNTIME_QUALIFIED',
             'E07c_MAIN_ARCHITECTURE_RUNTIME_QUALIFIED', 'E07c_SYNTHETIC_FORWARD_RUNTIME_QUALIFIED',
             'IMPLEMENTED_NOT_EXECUTED', 'CONTROLLED_ADAPTATION']
NOT_QUALIFIED = ['REAL_TRAIN_PATH', 'TRAINING_GRAPH_BACKWARD', 'TRAINING_MEMORY', 'AUXILIARY_ENCODER_TRAINING',
                 'CHECKPOINT_WRITER', 'RESUME', 'PRODUCTION_RUNNER', 'SCIENTIFIC_TRAINING', 'M8_BANK']
SCOPE = {'training_launched': False, 'optimizer_constructed': False, 'optimizer_applications': 0,
         'backward_passes': 0, 'checkpoint_created': False, 'checkpoint_loaded': False,
         'auxiliary_encoder_trained': False, 'benchmark_data_access': False, 'TRAIN_access': False,
         'VAL_access': False, 'TEST_access': False, 'synthetic_bank': False, 'diffusion_sampling': False,
         'training_graph_qualified': False}
SESSION_ZERO = {'benchmark_manifest_reads': 0, 'benchmark_image_reads': 0, 'TRAIN_reads': 0, 'VAL_reads': 0,
                'TEST_reads': 0, 'firewall_denials': 0, 'backward_calls': 0, 'optimizer_step_calls': 0,
                'scientific_checkpoints_created': 0, 'auxiliary_encoder_training_runs': 0,
                'scientific_seed_runs_completed': 0}
FEATURES = {'x32x32': [256, 32, 32], 'x16x16': [512, 16, 16], 'x8x8': [512, 8, 8]}


def require(ok, message):
    if not ok:
        raise ValueError('M6D6a-r: ' + message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def git(*args, root=ROOT):
    return subprocess.check_output(['git', '-C', str(root), *args])


def read(path):
    p = Path(path)
    require(not p.is_absolute() and '..' not in p.parts, 'relative evidence path')
    require(p.parts[0] in {'methods', 'tools', 'tests', 'configs', 'environments', 'outputs', 'docs',
                           'frozen_config_snapshot', 'third_party'}, 'evidence root')
    require(not {'data', 'manifests', 'faces_256', 'runs', 'cache'} & set(p.parts), 'data firewall')
    return (ROOT / p).read_bytes()


def source_cache():
    root = ROOT / 'third_party/source_cache/difffas'
    require(git('rev-parse', 'HEAD', root=root).decode().strip() == PIN, 'source commit')
    require(git('rev-parse', 'HEAD^{tree}', root=root).decode().strip() == TREE, 'source tree')
    require(git('status', '--porcelain', '--untracked-files=no', root=root).decode().strip() == '',
            'tracked pinned source unmodified')
    return root


def check_process(r):
    require(r['status'] == 'PASS' and r['label'] == 'SYNTHETIC_FORWARD_RUNTIME_ONLY', 'GPU process pass')
    require(r['qualification_seed'] == QUALIFICATION_SEED and r['experiment_seed'] is None and
            QUALIFICATION_SEED not in (42, 1337, 2026), 'qualification seed')
    s = r['source_before']
    require(r['source_before'] == r['source_after'] and s['worktree_status'] == '', 'source mutation')
    require((s['commit'], s['tree'], s['a6_overlay_sha256']) == (PIN, TREE, A6_OVERLAY_SHA), 'pinned source / A6 overlay')
    require(s['files_sha256']['models/custom_rn.py'] == CUSTOM_RN_SHA and
            s['files_sha256']['models/unet_autoenc.py'] == CONSUMER_SHA, 'A6 source digests')
    require(r['environment_before'] == r['environment_after'], 'environment mutation')
    e = r['encoder']
    require((e['module'], e['topology'], e['stage_widths'], e['structural_difference_vs_pinned_resnet18']) ==
            ('custom_rn', [3, 4, 6, 3], [64, 256, 512, 512], ['fc']), 'pinned custom_rn encoder')
    require(e['head'] == {'type': 'Linear', 'in_features': 512, 'out_features': 7} and
            not e['torchvision_substitute'] and not e['projection_or_adapter_modules_added'] and
            not e['weights_loaded'], 'A3 head; no substitute/projection/weights')
    require(r['feature_interface_live_CHW'] == {**FEATURES, 'embg': [7]}, 'live A6 interface')
    for name, chw in FEATURES.items():
        require(r['forward']['encoder_' + name]['shape'] == [4, *chw], 'live A6 shape ' + name)
    hist = r['contract']['feature_interface_historical_frozen_HWC']
    require(hist['x32x32'] == [32, 32, 128] and hist['x16x16'] == [16, 16, 256], 'historical pre-A6 text retained')
    m = r['main_model']
    require(m['class'] == 'BeatGANsAutoencModel' and m['first_conv_in_channels'] == 3 and m['use_pair'] is False,
            'main model / use_pair=false / 3-channel')
    for st in (e['state'], m['state_initial']):
        require(st['parameters'] == sum(math.prod(v) for v in st['shapes'].values()), 'parameter arithmetic')
    c = r['connectivity']
    require(c['fourth_output_consumers'] == 0 and c['feature_consumer_blocks']['embg'] == [], 'fourth output discarded')
    require(all(c['feature_consumer_blocks'][k] for k in FEATURES) and
            all(v > 0 for k in FEATURES for v in c['zeroing_sensitivity_max_abs_at_consumer_attention_blocks'][k].values()),
            'three features consumed')
    require(c['final_output_identically_zero'] and 'resnet_use_zero_module=True' in c['final_output_zero_reason'],
            'zero-initialised projection disclosed')
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


def repeatability(p, q, raw_p, raw_q):
    internal = {n: v['sha256'] == q['connectivity']['internal_activations'][n]['sha256']
                for n, v in p['connectivity']['internal_activations'].items()}
    forward = {n: p['forward'][n]['sha256'] == q['forward'][n]['sha256'] for n in p['forward']}
    initial = read('outputs/audit/M6D6A_E07C_SYNTHETIC_PROCESS_1.json')
    return {'independent_processes': 2, 'process_json_bytes_identical': raw_p == raw_q,
            'forward_tensors_compared': len(forward), 'forward_bitwise_equal': sum(forward.values()),
            'internal_stage_tensors_compared': len(internal), 'internal_stage_bitwise_equal': sum(internal.values()),
            'all_bitwise_equal': all(forward.values()) and all(internal.values()),
            'corrective_equals_initial_evidence_bytes': raw_p == initial,
            'process_sha256': sha(raw_p),
            'determinism_note': ('Fixed qualification seed 60601, frozen cudnn benchmark=False/deterministic=True, '
                                 'TF32 off, CUBLAS_WORKSPACE_CONFIG=:4096:8; torch deterministic algorithms NOT forced. '
                                 'Equality observed, not manufactured.')}


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
    return out.getvalue().encode(), len(rows)


def worktree(ledger_and_index_modified):
    status = sorted(git('status', '--porcelain', '--untracked-files=all').decode().splitlines())
    expected = ['?? ' + p for p in ARTIFACTS]
    if ledger_and_index_modified:
        expected += [' M ' + INDEX, ' M ' + LEDGER]
    require(status == sorted(expected), 'worktree holds exactly the intended M6D6a changes: ' + json.dumps(status))
    require(git('diff', '--cached', '--name-only').decode().strip() == '', 'nothing staged')
    return status


def verify(stage):
    """stage: 'before_ledger' (evidence only), 'before_index' (ledger appended), 'final'."""
    for ref in ('HEAD', 'origin/m6-baselines'):
        require(git('rev-parse', ref).decode().strip() == AUTHORITY, ref + ' authority')
    require(sha(read('docs/spec/GPAT_TransferBench_v1_0_Frozen_Specification_2026.docx')) == SPEC_SHA, 'spec SHA256')
    for path in PRESERVED:
        require(read(path) == git('show', AUTHORITY + ':' + path), 'immutable input ' + path)
    for path, h in INITIAL.items():
        require(sha(read(path)) == h, 'initial M6D6a evidence preserved ' + path)
    root = source_cache()
    audit = json.loads(read(BASE + '.json'))
    require(audit['initial_status'] == 'PROCEDURAL_FIREWALL_DEVIATION' and
            audit['final_status'] == 'CLEAN_REQUALIFICATION_PASS' and audit['qualified_statuses'] == QUALIFIED and
            audit['not_qualified'] == NOT_QUALIFIED and audit['method_status'] == 'IMPLEMENTED_NOT_EXECUTED',
            'status fields')
    require(audit['initial_candidate_sha256'] == INITIAL, 'initial hashes recorded in audit JSON')
    require(audit['session_access_audit']['counts'] == SESSION_ZERO and
            audit['session_access_audit']['scope'] == 'COMPLETE_CORRECTIVE_SESSION_LAPTOP_AND_GPU', 'session access audit')
    for path, h in audit['artifacts_sha256'].items():
        require(sha(read(path)) == h, 'artifact ' + path)
    for path, h in audit['preserved_inputs_sha256'].items():
        require(sha(read(path)) == h, 'preserved input ' + path)
    for run in audit['tests']:
        require(run['failures'] == run['errors'] == 0 and run['audit_denied'] == run['audit_manifest'] ==
                run['audit_image'] == run['audit_data_or_runs'] == 0, 'test run ' + run['suite'])
    raw_p, raw_q = [read(n) for n in PROCESSES]
    p, q = json.loads(raw_p), json.loads(raw_q)
    for r in (p, q):
        check_process(r)
    for rel, h in p['source_before']['files_sha256'].items():
        require(sha((root / rel).read_bytes()) == h, 'laptop source closure ' + rel)
    rep = repeatability(p, q, raw_p, raw_q)
    require(rep == audit['repeatability'] and rep['all_bitwise_equal'] and rep['process_json_bytes_identical'],
            'repeatability evidence')
    lock = read('environments/e07c.lock.json')
    require(sha(lock) == LOCK_SHA == audit['environment_lock_sha256'], 'environment lock digest')
    lockj = json.loads(lock)
    require(lockj['runtime_json_sha256'] == sha(read('environments/e07c.runtime.json')) and
            lockj['runtime_harness_sha256'] == sha(read('methods/difffas/runtime_qualification.py')), 'lock linkage')
    for name, h in lockj['exports_sha256'].items():
        require(sha(read('environments/e07c.' + name)) == h, 'export ' + name)
    require(lockj['identity'] == p['environment_before'] == q['environment_before'], 'executed identity = lock')
    ev = audit['environment_reverification']
    require(ev['conda_explicit_sha256'] == lockj['exports_sha256']['conda-explicit.txt'] and
            ev['pip_freeze_sha256'] == lockj['exports_sha256']['pip-freeze.txt'] and
            ev['python_sha256'] == lockj['identity']['python_sha256'] and ev['pip_check_pass'] and
            ev['protected_environments_unchanged'] and not ev['environment_rebuilt_or_mutated'], 'environment re-verified')
    log = read('outputs/audit/M6D6A_R_E07C_RUNTIME_LOG.txt').decode()
    part_a = log.split('PART B')[0]
    require(part_a.count('scientific_paths_after: runs=ABSENT e07c_root=ABSENT encoder_final=ABSENT') ==
            part_a.count('scientific_paths_before:') == 2, 'no scientific run root or auxiliary checkpoint created')
    require('PROCEDURAL_FIREWALL_DEVIATION' in log and 'xargs sha256sum' in log, 'initial deviation disclosed in log')
    prefix = git('show', AUTHORITY + ':' + LEDGER)
    current = read(LEDGER)
    require(len(prefix.splitlines()) == LEDGER_PREFIX_ROWS and sha(prefix) == LEDGER_PREFIX_SHA, 'ledger authority')
    require(current.startswith(prefix), 'first 115 ledger rows byte-identical')
    if stage == 'before_ledger':
        require(current == prefix, 'ledger not yet appended')
        worktree(False)
    else:
        require(len(current.splitlines()) == LEDGER_PREFIX_ROWS + 1, 'exactly one append')
        row = json.loads(current[len(prefix):])
        require((row['classification'], row['milestone'], row['corrective_requalification']) ==
                (CLASSIFICATION, 'M6D6a', 'M6D6a-r'), 'ledger row identity')
        require(row['initial_status'] == 'PROCEDURAL_FIREWALL_DEVIATION' and
                row['final_status'] == 'CLEAN_REQUALIFICATION_PASS' and row['qualified_statuses'] == QUALIFIED and
                row['not_qualified'] == NOT_QUALIFIED and row['method_status'] == 'IMPLEMENTED_NOT_EXECUTED',
                'ledger status fields')
        require(all(row[k] == v for k, v in SCOPE.items()) and row['session_access_counts'] == SESSION_ZERO,
                'ledger scope/access')
        for path, h in row['artifacts_sha256'].items():
            require(sha(read(path)) == h, 'ledger artifact ' + path)
    expected, count = expected_index()
    if stage == 'final':
        require(read(INDEX) == expected, 'CRLF artifact index')
        worktree(True)
    require('torch' not in sys.modules, 'static preflight must not import torch')
    return {'status': 'PASS', 'stage': stage, 'ledger_rows': len(current.splitlines()),
            f'first_{LEDGER_PREFIX_ROWS}_rows_byte_identical': True, 'initial_evidence_preserved': True,
            'artifact_index_rows_before': INDEX_BASELINE_ROWS, 'artifact_index_rows_expected': count,
            'corrective_processes': 2, 'all_bitwise_equal': rep['all_bitwise_equal'],
            'corrective_equals_initial_evidence_bytes': rep['corrective_equals_initial_evidence_bytes'],
            'session_access_counts': audit['session_access_audit']['counts'],
            'torch_imported': False, 'model_execution_in_preflight': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before-ledger', action='store_true', help='evidence-only check before the ledger append')
    parser.add_argument('--rebuild-index', action='store_true', help='after the ledger append: write the index LAST')
    args = parser.parse_args()
    if args.before_ledger:
        result = verify('before_ledger')
    elif args.rebuild_index:
        verify('before_index')
        (ROOT / INDEX).write_bytes(expected_index()[0])
        result = verify('final')
    else:
        result = verify('final')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
