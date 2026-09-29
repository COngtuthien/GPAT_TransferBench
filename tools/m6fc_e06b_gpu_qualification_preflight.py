#!/usr/bin/env python3
"""Verify the M6FC candidate: E06b DSDG-NATIVE GPU synthetic graph + execution qualification; ledger/index LAST.

The summary evidence is DERIVED from the four GPU-produced JSONs (copied back byte-exact); no measured value is
written on the laptop. QUALIFICATION_ONLY_SYNTHETIC / NON_SCIENTIFIC: no face image, VAL, TEST, checkpoint or bank.

STATIC: stdlib only (no Torch, YAML parser, numpy, PIL or pyarrow). check_evidence takes readers (no moving HEAD).

  --write-evidence-json / --before-ledger / --append-ledger --tests JSON / --rebuild-index / (default final)
"""
import argparse
import csv
import datetime as dt
import getpass
import hashlib
import io
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = '91d28a72f78d8f424da23cf07a015782e532ba0e'
BRANCH = 'm6-baselines'
MILESTONE = 'M6FC'
CLASSIFICATION = 'M6FC_E06B_GPU_QUALIFICATION'
INDEX = 'outputs/audit/ARTIFACT_INDEX.csv'
LEDGER = 'outputs/audit/EXECUTION_LEDGER.jsonl'
LEDGER_PREFIX_ROWS = 132
LEDGER_PREFIX_SHA = '615ae53366165cc60249d9c5dbda35c240b4dcf43068adc8d1a557c727f1146c'
INDEX_BASELINE_ROWS = 784
CONFIG_STATUS = 'configs/CONFIG_STATUS.md'
HARNESS = 'methods/dsdg/native_qualification.py'
HARNESS_SHA = '2c16a247199a735232c845f0bbe93b6b551909948967c9cd34bd7f6d2b07c718'   # bytes that ran all four PASS modes
GPU = {'workers': 'outputs/audit/M6FC_E06B_WORKER_DETERMINISM.json',
       'reference': 'outputs/audit/M6FC_E06B_REFERENCE_EQUIVALENCE.json',
       'b240': 'outputs/audit/M6FC_E06B_LOGICAL_B240.json',
       'tail120': 'outputs/audit/M6FC_E06B_LOGICAL_TAIL120.json'}
GPU_SHA = {'workers': '01978a0d34b37bcd4031e4b4e4a9af758055f209c13d3125cedb4b8902e3e08d',
           'reference': '9162fff449e6b1b7afc155edeb5500ba2f7945e62eccfdfeccab612931c8b394',
           'b240': 'a3cc3a88ce541631d62c0d9445c3d6e2235488a4d65744931685c8ae70a5c38a',
           'tail120': '265714bc28fe2fc59b28d4cc1fc050605cac1f8b2011d1036325177bc59ea494'}
EV_JSON = 'outputs/audit/M6FC_E06B_GPU_QUALIFICATION.json'
EV_MD = 'outputs/audit/M6FC_E06B_GPU_QUALIFICATION.md'
TESTS = 'tests/test_m6fc_e06b_gpu_qualification.py'
PREFLIGHT = 'tools/m6fc_e06b_gpu_qualification_preflight.py'
NEW = tuple(sorted((HARNESS, *GPU.values(), EV_JSON, EV_MD, TESTS, PREFLIGHT)))
CONFIG = 'configs/methods/e06b_dsdg_native.yaml'
CONFIG_SHA = '9d665dc2c909d421b8e54964407e27bb40d133f11bceec2e2cb29810f268417f'
LOCK = 'environments/e06c.lock.json'
LOCK_SHA = '91416a20fef6eb4bbe550dc0ccdc703163f51d8df9168c1418f7a2de48e64e95'
LIGHTCNN_SHA = 'd0750746622270548137b092700811b00dc64d67ca8df042c6cf48cb3b1b9964'
SEED = 60701
LAMBDAS = {'lambda_ip': 1000, 'lambda_mmd': 50, 'lambda_ort': 1, 'lambda_pair': 5, 'lambda_type': 10}
GATES = {'scalar_losses_finite': True, 'total_loss_relative_error_max': 1e-4, 'aggregate_gradient_cosine_min': 0.999,
         'aggregate_gradient_relative_l2_max': 0.01, 'post_step_parameter_delta_cosine_min': 0.99}
ABORTED_ATTEMPT = {
    'root': '<runtime_root>/builds/e06b_dsdg_m6fc/workers.aborted-attempt-1',
    'retained_on_gpu_host': True, 'mode': 'workers', 'status': 'STOP_WORKER_GATE',
    'cause': 'harness expected-sequence builder omitted torch RandomSampler\'s second randperm(n)[:n % n] drawn at '
             'sampler exhaustion; epoch 1 matched exactly, epoch 2 mispredicted',
    'fix': 'predictor consumes the remainder draw (verified in torch 2.12.1 source); the E06b model '
           '(native.simulate_live_draws) and the loader contract were NOT changed',
    'scientific_impact': 'none (mock CPU relation, no GPU, no data)',
    'classification': 'QUALIFICATION_HARNESS_ISSUE_ONLY', 'e06b_model_or_loader_failure': False}


def require(ok, message):
    if not ok:
        raise ValueError('M6FC: ' + message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args])


def read(path):
    p = Path(path)
    require(not p.is_absolute() and '..' not in p.parts, 'relative path')
    require(not {'data', 'faces_256', 'runs', 'cache', 'manifests'} & set(p.parts), 'data firewall ' + path)
    require(p.suffix not in {'.pkl', '.pt', '.pth', '.ckpt', '.parquet', '.png', '.jpg'}, 'no weight/manifest/image')
    return (ROOT / p).read_bytes()


def worktree_reader(rel):
    return read(rel)


def at_authority(rel):
    return git('show', f'{AUTHORITY}:{rel}')


def common_checks(name, r):
    require((r['milestone'], r['label'], r['scientific'], r['qualification_seed'], r['method_id']) ==
            ('M6F-C', 'QUALIFICATION_ONLY_SYNTHETIC', 'NON_SCIENTIFIC', SEED, 'E06b'), name + ' identity/label/seed')
    c = r['contract']
    require((c['config_sha256'], c['attack_type'], c['hdim'], c['lambdas'], c['vocabulary'], c['class_index'],
             c['effective_batch_size'], c['workers'], c['e06c_guards_used']) ==
            (CONFIG_SHA, 2, 128, LAMBDAS, ['print', 'replay'], {'print': 0, 'replay': 1}, 240, 8, False),
            name + ' live contract')
    require(r['environment_lock_sha256'] == LOCK_SHA and r['environment_lock'] == LOCK, name + ' DSDG env lock')
    require(not r['firewall']['denied'] and r['checkpoint_created'] is False and r['scientific_training_completed']
            is False and r['TEST_access'] is False and r['VAL_access'] is False and r['benchmark_data_access'] is False
            and r['benchmark_image_reads'] == 0 and r['synthetic_bank'] is False and r['native_manifest_created'] is False,
            name + ' firewall / no science / no data')
    cnt = r['counters']
    require(cnt['autograd_grad_calls'] == cnt['checkpoint_saves'] == cnt['activation_checkpoint_calls'] == 0,
            name + ' no forbidden calls')
    require(r['status'] == 'PASS', name + ' PASS')


def derive(reader=worktree_reader):
    runs = {}
    for name, rel in GPU.items():
        raw = reader(rel)
        require(sha(raw) == GPU_SHA[name], 'GPU evidence bytes ' + rel)
        r = json.loads(raw)
        common_checks(name, r)
        runs[name] = r
    require(sha(reader(HARNESS)) == HARNESS_SHA, 'harness bytes == the bytes that ran on the GPU')
    require(sha(reader(CONFIG)) == CONFIG_SHA, 'frozen E06b config unchanged')
    ref, w = runs['reference'], runs['workers']
    require(ref['pre_registered_gates'] == GATES and ref['per_term_relative_error_max'] == 1e-4, 'pre-registered gates')
    require(all(all(c['gates'].values()) for c in ref['reference_cases']) and all(ref['term_isolation']['gates'].values()),
            'reference gates')
    require(ref['binding']['netCls_live'] == {'bias_shape': [2], 'in_features': 128, 'out_features': 2,
                                              'weight_shape': [2, 128]}, 'live Cls(128, 2)')
    require((ref['lightcnn']['matched'], ref['asset_before']['sha256']) == (60, LIGHTCNN_SHA), 'LightCNN 60/60 + SHA')
    require(all(w['gates'].values()) and w['loader_contract']['num_workers'] == 8, 'worker gates')
    for name, B in (('b240', 240), ('tail120', 120)):
        r = runs[name]
        require(all(r['gates'].values()) and (r['logical_batch_size'], r['physical_microbatch'], r['chunks']) ==
                (B, 20, B // 20), name + ' gates / plan')
    env = ref['environment_before']
    cases = [{'global_batch': c['global_batch'], 'max_microbatch': c['max_microbatch'], 'chunk_sizes': c['chunk_sizes'],
              'epoch': c['epoch'], 'labels': c['path_a_full_batch']['labels'],
              'path_a_losses_raw_and_weighted': c['path_a_full_batch']['raw_and_weighted'],
              'path_a_total': c['path_a_full_batch']['total'], 'path_b_total': c['path_b_microbatch']['total'],
              'max_term_relative_error': max(c['comparison']['loss_rel_diff'].values()),
              'total_relative_error': c['comparison']['total_rel_error'],
              'gradient_cosine': c['comparison']['gradient']['cosine'],
              'gradient_relative_l2': c['comparison']['gradient']['relative_l2'],
              'gradient_max_abs_diff': c['comparison']['gradient_max_abs_diff'],
              'gradient_l2_full_batch': c['comparison']['gradient']['l2_a'],
              'update_cosine': c['comparison']['update']['cosine'],
              'update_relative_l2': c['comparison']['update']['relative_l2'],
              'update_max_abs_diff': c['comparison']['update_max_abs_diff'], 'gates': c['gates']}
             for c in ref['reference_cases']]
    logical = {name: {'logical_batch_size': r['logical_batch_size'], 'physical_microbatch': r['physical_microbatch'],
                      'chunks': r['chunks'], 'chunk_sizes': r['chunk_sizes'], 'epoch_branch': r['epoch_branch'],
                      'labels_count': r['inputs']['labels_count'], 'losses': r['losses'], 'total_loss': r['total_loss'],
                      'gradients': {n: {k: g[k] for k in ('tensors', 'non_none', 'finite', 'nonzero_elements')}
                                    for n, g in r['gradients'].items()},
                      'optimizer_applications': r['counters']['optimizer_applications'],
                      'backward_calls': r['counters']['backward_calls'],
                      'memory_before': r['memory']['before'], 'memory_after_step': r['memory']['after_step'],
                      'peak_allocated_bytes': r['peak_allocated_bytes'], 'peak_reserved_bytes': r['peak_reserved_bytes'],
                      'wall_clock_seconds': r['wall_clock_seconds'], 'gates': r['gates'], 'oom': False}
               for name, r in runs.items() if name in ('b240', 'tail120')}
    return {'environment': {'gpu': env['gpu_driver'], 'python': env['python'], 'torch': env['torch'],
                            'torchvision': env['torchvision'], 'cuda': env['cuda'], 'cudnn': env['cudnn'],
                            'numpy': env['numpy'], 'executable': env['executable'], 'precision': env['precision'],
                            'environment_lock': LOCK, 'environment_lock_sha256': LOCK_SHA},
            'source': {'commit': ref['source_before']['commit'], 'tree': ref['source_before']['tree'],
                       'files_verified': len(ref['source_before']['files_sha256'])},
            'lightcnn': {'sha256': ref['asset_before']['sha256'], 'bytes': ref['asset_before']['bytes'],
                         'matched_tensors': ref['lightcnn']['matched']},
            'graph': {'netCls_live': ref['binding']['netCls_live'], 'parameters_initial':
                      {n: {'parameters': p['parameters'], 'parameter_tensors': p['parameter_tensors']}
                       for n, p in ref['parameters_initial'].items()},
                      'batch_coupling': {k: ref['batch_coupling'][k] for k in ('batchnorm_modules',
                                                                                'instancenorm_modules')},
                      'pinned_statements_verified': ref['pinned_statements_verified'], 'lambdas': LAMBDAS},
            'reference_cases': cases, 'term_isolation': ref['term_isolation'],
            'reference_peak_allocated_bytes': ref['peak_memory']['peak_allocated_bytes'],
            'logical': logical,
            'workers': {'gates': w['gates'], 'epochs': w['epochs'], 'loader_contract': w['loader_contract'],
                        'relation_summary': w['relation']['summary']},
            'started_utc': {n: r['started_utc'] for n, r in runs.items()}}


def build_evidence_from(d):
    worst = {k: (max if k.endswith(('error', 'l2', 'diff')) else min)(c[k] for c in d['reference_cases'])
             for k in ('max_term_relative_error', 'total_relative_error', 'gradient_cosine', 'gradient_relative_l2',
                       'update_cosine', 'update_relative_l2', 'gradient_max_abs_diff', 'update_max_abs_diff')}
    return {
        'milestone': MILESTONE, 'classification': CLASSIFICATION, 'status': 'PASS', 'method_id': 'E06b',
        'authority_commit': AUTHORITY, 'gpu_authority_commit': AUTHORITY, 'label': 'QUALIFICATION_ONLY_SYNTHETIC',
        'scientific': 'NON_SCIENTIFIC', 'qualification_seed': SEED, 'config_sha256': CONFIG_SHA,
        'config_unchanged': True, 'environment_decision': 'owner (M6F-C): E06b binds the qualified DSDG environment '
                                                          'gpat-m6-e06c (environments/e06c.lock.json)',
        'harness': {'path': HARNESS, 'sha256': HARNESS_SHA}, 'gpu_evidence_sha256': {GPU[k]: v for k, v in GPU_SHA.items()},
        'pre_registered_gates': GATES, 'per_term_relative_error_max': 1e-4,
        'gate_provenance': 'owner-approved M6D5c reference gates (verified equal at runtime) + per-term bound equal '
                           'to the registered total bound; fixed in code before the first E06b GPU run',
        'worst_case_reference': worst, 'derived': d, 'aborted_attempt': ABORTED_ATTEMPT,
        'fidelity': {
            'target_source_fidelity': 'FAITHFUL_OFFICIAL',
            'execution_classification': 'EXECUTION_RUNTIME_COMPATIBILITY',
            'runtime_fidelity_assessment': 'FAITHFUL_OFFICIAL_WITH_RUNTIME_COMPATIBILITY',
            'runtime_fidelity_decision': 'OWNER_DECISION_AT_M6F_C_REVIEW',
            'final_method_fidelity': 'PENDING_M6F_D_PRODUCTION_QUALIFICATION',
            'final_method_fidelity_rule': 'if M6F-D passes without a new scientific deviation, the final E06b row '
                                          'may use FAITHFUL_OFFICIAL_WITH_RUNTIME_COMPATIBILITY',
            'controlled_adaptation': False,
            'frozen_config_field': 'final_execution_fidelity stays PENDING_M6F_C_RUNTIME_QUALIFICATION in the frozen '
                                   'YAML (never rewritten; E03 precedent)',
            'owner_rationale': ['effective batch remains exactly 240', 'architecture unchanged',
                                'all seven official DSDG losses and coefficients unchanged', 'lambda_pair remains 5',
                                'K remains 2', 'optimizer semantics unchanged',
                                'direct vs microbatch equivalence passes all pre-declared gates',
                                'full logical batch 240 and tail 120 execute successfully',
                                'microbatching only changes floating-point accumulation order',
                                'single-RTX3090 execution replaces upstream multi-GPU execution without changing '
                                'the scientific method'],
            'observable_deviations': ['not bitwise: FP32 accumulation order differs (term rel <= worst_case_reference)',
                                      'first Adam step normalizes g/sqrt(v): near-zero gradient elements can flip '
                                      'sign under FP32 noise, bounded by 2 x lr = 4e-4 per element; update cosine '
                                      'stays >= the registered 0.99',
                                      'single RTX 3090 instead of upstream 4-GPU DataParallel (both see the global '
                                      'batch of 240)'],
            'not_claimed': 'final method-level fidelity (production runner unqualified); plain FAITHFUL_OFFICIAL; '
                           'bitwise reproduction of a physical-240 step'},
        'statuses': ['CONFIG_FROZEN', 'STATIC_ADAPTER_IMPLEMENTED', 'GPU_GRAPH_QUALIFIED',
                     'EXECUTION_MAPPING_QUALIFIED', 'WORKER_DETERMINISM_QUALIFIED',
                     'PRODUCTION_RUNNER_NOT_YET_QUALIFIED', 'SCIENTIFIC_TRAINING_NOT_EXECUTED'],
        'scientific_training_completed': False, 'checkpoint_created': False, 'bank_created': False,
        'benchmark_image_reads': 0, 'VAL_access': False, 'TEST_access': False,
        'm6': {'M6_CLOSED': False, 'M7_started': False}}


def check_evidence(reader=worktree_reader):
    d = derive(reader)
    ev = json.loads(reader(EV_JSON))
    require(ev == build_evidence_from(d), 'summary evidence == derivation from the GPU JSONs')
    md = reader(EV_MD).decode()
    for token in ('M6FC', 'QUALIFICATION_ONLY_SYNTHETIC', 'NON_SCIENTIFIC', CONFIG_SHA, 'EXECUTION_RUNTIME_COMPATIBILITY',
                  'FAITHFUL_OFFICIAL_WITH_RUNTIME_COMPATIBILITY', 'PENDING_M6F_D_PRODUCTION_QUALIFICATION',
                  'QUALIFICATION_HARNESS_ISSUE_ONLY', 'scientific_training_completed = false',
                  'workers.aborted-attempt-1', 'M6_CLOSED = false', 'M7 HAS NOT STARTED'):
        require(token in md, 'report token ' + token)
    return ev


# ----------------------------------------------------------------- candidate-time checks
def authority():
    require(git('branch', '--show-current').decode().strip() == BRANCH, 'branch')
    require(git('rev-parse', 'HEAD').decode().strip() == AUTHORITY, 'uncommitted candidate on the M6F-B authority')


def tracked_unchanged(bookkeeping):
    changed = sorted(git('diff', '--name-only', AUTHORITY).decode().split())
    require(changed == sorted([CONFIG_STATUS, *bookkeeping]), 'only CONFIG_STATUS + ledger/index differ: ' + json.dumps(changed))
    require(git('diff', '--cached', '--name-only').decode().strip() == '', 'nothing staged')
    require(read(CONFIG) == at_authority(CONFIG), 'frozen E06b config byte-identical')
    base, now = at_authority(CONFIG_STATUS), read(CONFIG_STATUS)
    require(now.startswith(base) and len(now) > len(base), 'CONFIG_STATUS append-only')
    added = now[len(base):].decode()
    for token in ('M6FC', 'E06b', 'GPU_GRAPH_QUALIFIED', 'FAITHFUL_OFFICIAL_WITH_RUNTIME_COMPATIBILITY',
                  'PENDING_M6F_D_PRODUCTION_QUALIFICATION', 'M6_CLOSED = false',
                  'M7 HAS NOT STARTED'):
        require(token in added, 'CONFIG_STATUS M6FC note token ' + token)
    for p in ('manifests/dsdg_identity_pairs_v1.parquet', 'outputs/audit/method_status.csv'):
        require(not (ROOT / p).exists(), 'must not exist ' + p)


def worktree(bookkeeping):
    status = sorted(git('status', '--porcelain', '--untracked-files=all').decode().splitlines())
    require(status == sorted(['?? ' + p for p in NEW] + [' M ' + p for p in (CONFIG_STATUS, *bookkeeping)]),
            'worktree holds exactly the M6FC candidate: ' + json.dumps(status))


def whitespace():
    for rel in (*NEW, CONFIG_STATUS):
        raw = read(rel)
        require(not re.search(rb'[ \t]\r?\n', raw) and b'\r' not in raw, 'LF, no trailing whitespace ' + rel)


def expected_index():
    reader = csv.DictReader(io.StringIO(at_authority(INDEX).decode()))
    baseline = list(reader)
    rows = {r['path']: r for r in baseline}
    require(reader.fieldnames == ['path', 'size_bytes', 'sha256'] and len(rows) == len(baseline) == INDEX_BASELINE_ROWS,
            'baseline index')
    for p in NEW:
        require(p not in rows, 'additive artifact ' + p)
    for p in (*NEW, CONFIG_STATUS):
        raw = read(p)
        rows[p] = {'path': p, 'size_bytes': str(len(raw)), 'sha256': sha(raw)}
    out = io.StringIO(newline='')
    writer = csv.DictWriter(out, fieldnames=reader.fieldnames, lineterminator='\r\n')
    writer.writeheader()
    writer.writerows(rows[k] for k in sorted(rows))
    return out.getvalue().encode(), len(rows)


LEDGER_EXPECTED = {
    'milestone': MILESTONE, 'classification': CLASSIFICATION, 'status': 'PASS', 'final_status': 'PASS',
    'record_kind': 'GPU_SYNTHETIC_QUALIFICATION', 'method_id': 'E06b', 'label': 'QUALIFICATION_ONLY_SYNTHETIC',
    'scientific': 'NON_SCIENTIFIC', 'qualification_seed': SEED, 'config_sha256': CONFIG_SHA,
    'environment_lock_sha256': LOCK_SHA, 'harness_sha256': HARNESS_SHA, 'gpu_modes_pass': ['workers', 'reference',
                                                                                            'b240', 'tail120'],
    'execution_classification': 'EXECUTION_RUNTIME_COMPATIBILITY',
    'runtime_fidelity_assessment': 'FAITHFUL_OFFICIAL_WITH_RUNTIME_COMPATIBILITY',
    'final_method_fidelity': 'PENDING_M6F_D_PRODUCTION_QUALIFICATION', 'controlled_adaptation': False, 'oom': False,
    'aborted_attempts': 1, 'scientific_training_completed': False, 'checkpoint_writes': 0, 'banks_generated': 0,
    'benchmark_image_reads': 0, 'VAL_access': False, 'TEST_access': False, 'GPU_contacted': True,
    'production_runner_qualified': False, 'M6_closed': False, 'M7_started': False, 'commit': False, 'push': False,
    'authority_commit': AUTHORITY, 'git_commit': AUTHORITY, 'committed_prefix_rows': LEDGER_PREFIX_ROWS,
    'committed_prefix_sha256': LEDGER_PREFIX_SHA, 'artifact_index_rows_before': INDEX_BASELINE_ROWS,
    'artifact_index_rows_after': INDEX_BASELINE_ROWS + len(NEW), 'modified_existing_files': [CONFIG_STATUS]}


def check_ledger_row(row, prefix):
    for k, v in LEDGER_EXPECTED.items():
        require(row[k] == v, 'ledger field ' + k)
    require(sha(prefix) == row['committed_prefix_sha256'], 'ledger prefix binding')
    require(sorted(row['artifacts_sha256']) == sorted((*NEW, CONFIG_STATUS)), 'ledger artifact list')
    for path, h in row['artifacts_sha256'].items():
        require(sha(read(path)) == h, 'ledger artifact ' + path)
    require(row['tests'] and all(t['failures'] == 0 and (t['errors'] == 0 or t.get('errors_preexisting_environmental'))
                                 for t in row['tests']), 'ledger tests: no failures; errors only if pre-existing environmental')


def ledger_row(tests):
    return {**LEDGER_EXPECTED,
            'purpose': 'M6F-C: E06b DSDG-NATIVE GPU synthetic graph + microbatch execution + worker determinism',
            'notes': ('QUALIFICATION_ONLY_SYNTHETIC on one RTX 3090 in gpat-m6-e06c (owner decision). Live Cls(128,2), '
                      'lambda_pair=5, loss_cls/loss_pair active; direct-vs-microbatch equivalence met the pre-registered '
                      'M6D5c gates in 3 reference cases; logical 240 (12x20) and tail 120 (6x20) one Adam step each, '
                      'no OOM; 8-worker torch draws equal the pure-Python model over 2 epochs. Owner decision: runtime '
                      'fidelity FAITHFUL_OFFICIAL_WITH_RUNTIME_COMPATIBILITY (not CONTROLLED_ADAPTATION); final method '
                      'fidelity PENDING_M6F_D_PRODUCTION_QUALIFICATION. One aborted workers attempt (harness predictor '
                      'omitted the RandomSampler remainder draw; not an E06b failure) retained. No data, checkpoint or bank.'),
            'command': ('GPU: gpat-m6-e06c python -B methods/dsdg/native_qualification.py --mode '
                        'workers|reference|b240|tail120; laptop: tools/m6fc_e06b_gpu_qualification_preflight.py'),
            'artifacts_sha256': {p: sha(read(p)) for p in sorted((*NEW, CONFIG_STATUS))},
            'tests': tests, 'cwd': str(ROOT), 'host': socket.gethostname(), 'user': getpass.getuser(),
            'git_dirty': True, 'timestamp_utc': dt.datetime.now(dt.timezone.utc).isoformat()}


def verify(stage):
    authority()
    bookkeeping = {'before_ledger': [], 'before_index': [LEDGER], 'final': [INDEX, LEDGER]}[stage]
    tracked_unchanged(bookkeeping)
    whitespace()
    check_evidence()
    prefix = at_authority(LEDGER)
    current = (ROOT / LEDGER).read_bytes()
    require(len(prefix.splitlines()) == LEDGER_PREFIX_ROWS and sha(prefix) == LEDGER_PREFIX_SHA and
            current.startswith(prefix), f'first {LEDGER_PREFIX_ROWS} ledger rows byte-identical')
    if stage == 'before_ledger':
        require(current == prefix, 'ledger not yet appended')
    else:
        require(len(current.splitlines()) == LEDGER_PREFIX_ROWS + 1 and current.endswith(b'\n'), 'exactly one append')
        check_ledger_row(json.loads(current[len(prefix):]), prefix)
    expected, count = expected_index()
    if stage == 'final':
        require((ROOT / INDEX).read_bytes() == expected, 'CRLF sorted artifact index')
    worktree(bookkeeping)
    require(not {'torch', 'yaml', 'pyarrow', 'numpy', 'PIL'} & set(sys.modules), 'static preflight')
    return {'status': 'PASS', 'stage': stage, 'ledger_rows': len(current.splitlines()),
            f'first_{LEDGER_PREFIX_ROWS}_rows_byte_identical': True, 'ledger_prefix_sha256': sha(prefix),
            'artifact_index_rows_before': INDEX_BASELINE_ROWS, 'artifact_index_rows_expected': count,
            'artifact_rows_added': len(NEW), 'modified_existing_files': [CONFIG_STATUS, *bookkeeping]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-evidence-json', action='store_true')
    parser.add_argument('--before-ledger', action='store_true')
    parser.add_argument('--append-ledger', action='store_true')
    parser.add_argument('--tests')
    parser.add_argument('--rebuild-index', action='store_true')
    args = parser.parse_args()
    if args.write_evidence_json:
        require(not (ROOT / EV_JSON).exists(), 'fresh evidence')
        (ROOT / EV_JSON).write_text(json.dumps(build_evidence_from(derive()), indent=2, sort_keys=True) + '\n')
        print(json.dumps({'status': 'WRITTEN', 'path': EV_JSON}))
        return
    if args.before_ledger:
        result = verify('before_ledger')
    elif args.append_ledger:
        verify('before_ledger')
        row = ledger_row(json.loads(args.tests))
        with open(ROOT / LEDGER, 'ab') as fh:
            fh.write((json.dumps(row, sort_keys=True, ensure_ascii=False) + '\n').encode())
            fh.flush()
            os.fsync(fh.fileno())
        result = verify('before_index')
    elif args.rebuild_index:
        verify('before_index')
        (ROOT / INDEX).write_bytes(expected_index()[0])
        result = verify('final')
    else:
        result = verify('final')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
