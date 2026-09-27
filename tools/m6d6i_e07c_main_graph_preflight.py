#!/usr/bin/env python3
"""Verify the M6D6i E07c MAIN encoder-load integration + training-graph QUALIFICATION candidate; ledger/index LAST.

STATIC: stdlib only (no Torch, YAML, numpy, PIL or pyarrow); never opens a weight file, a manifest, an image or any
runtime path. Candidate-time checks (authority, branch, exact worktree, every tracked file vs the M6D6h authority) run
only here, once. check_evidence(reader) takes a reader so tests can validate the evidence at the M6D6i commit
instead of the moving HEAD.

  --before-ledger   evidence + candidate checks, ledger not yet appended
  --rebuild-index   after the one-row ledger append: rebuild ARTIFACT_INDEX.csv LAST (CRLF, sorted)
  (default)         final check
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = '84023aed3959d7ffda380b7612b860652579b443'
BRANCH = 'm6-baselines'
MILESTONE = 'M6D6i'
CLASSIFICATION = 'M6D6I_E07C_MAIN_ENCODER_LOAD_AND_TRAINING_GRAPH'
SEED = 60607
FROZEN_SHA = '49a24a3a7288782c144eaf4f58e83c76c37c1feeceea6de8a04b96a8fcfe107c'
FROZEN_BYTES = 185136819
FROZEN_TEMPLATE = '<runtime_root>/runs/m6/E07c/aux_encoder/seed_42/checkpoints/encoder_final.pkl'
FROZEN_GPU_PATH = ('/home/student20261/workdir/GPAT_TransferBench_runtime/runs/m6/E07c/aux_encoder/seed_42/'
                   'checkpoints/encoder_final.pkl')
FREEZE_RECORD_SHA = '6230e0b9531d47664f9eb14a7f87b24f67b0bf231a995adaba1294f944ec7693'
INDEX = 'outputs/audit/ARTIFACT_INDEX.csv'
LEDGER = 'outputs/audit/EXECUTION_LEDGER.jsonl'
LEDGER_PREFIX_ROWS = 123
INDEX_BASELINE_ROWS = 726

GRAPH = 'methods/difffas/main_graph.py'
HARNESS = 'methods/difffas/main_graph_qualification.py'
TESTS = 'tests/test_m6d6i_e07c_main_graph.py'
PREFLIGHT = 'tools/m6d6i_e07c_main_graph_preflight.py'
BASE = 'outputs/audit/M6D6I_E07C_'
P1, P2 = BASE + 'PROCESS_1.json', BASE + 'PROCESS_2.json'
AGG, REPORT, LOG = BASE + 'MAIN_GRAPH_QUALIFICATION.json', BASE + 'MAIN_GRAPH_QUALIFICATION.md', BASE + 'RUNTIME_LOG.txt'
NEW = tuple(sorted((GRAPH, HARNESS, TESTS, PREFLIGHT, P1, P2, AGG, REPORT, LOG)))
BOOKKEEPING = (INDEX, LEDGER)
PROTECTED = ('methods/difffas/execution_policy.py', 'methods/difffas/aux_checkpoint.py', 'methods/difffas/aux_runner.py',
             'methods/difffas/aux_runner_io.py', 'methods/difffas/aux_resume.py', 'tools/run_e07c_aux.py',
             'methods/difffas/encoder.py', 'methods/difffas/source.py', 'methods/difffas/contract.py',
             'methods/difffas/seed_adapter.py', 'methods/difffas/runtime_qualification.py',
             'methods/difffas/__init__.py', 'methods/common/upstream.py', 'methods/common/learned.py',
             'methods/common/config.py', 'configs/methods/e07c_difffas_bin_idfree.yaml',
             'configs/frozen/difffas_bin_idfree_v1.yaml', 'configs/amendments/e07c_a6_feature_interface_source_correction.yaml',
             'configs/amendments/e07c_a7_execution_policy.yaml', 'configs/amendments/e07c_a8_aux_resume_policy.yaml',
             'configs/amendments/e07c_m6d6e_aux_production_runner_contract.yaml',
             'configs/amendments/e07c_m6d6h_aux_encoder_freeze.yaml', 'configs/run_logging_v1.yaml',
             'environments/e07c.lock.json', 'tests/test_m6d6g_e07c_aux_scientific_run.py',
             'tests/test_m6d6h_e07c_aux_encoder_freeze.py', 'tools/m6d6h_e07c_aux_encoder_freeze_preflight.py',
             'docs/spec/GPAT_TransferBench_v1_0_Frozen_Specification_2026.docx')
EXPECTED_COUNTERS = {
    'throwaway_constructor': 1, 'resnet_init_calls': 1, 'sha_verified_event': 1, 'sha_rejected_event': 0,
    'torch_load': 1, 'checkpoint_read_opens': 2, 'checkpoint_write_attempts': 0, 'AdamW_construction': 1,
    'optimizer_constructions': 1, 'scheduler_construction': 1, 'DataLoader_iterator': 1, 'batch': 1,
    'randint_in_iteration': 1, 'training_losses': 1, 'encoder_forward': 1, 'model_forward': 1, 'zero_grad': 1,
    'backward': 1, 'scheduler_step': 1, 'optimizer_step': 1, 'accumulate': 1, 'torch_save': 0, 'sampling': 0,
    'autograd_grad': 0, 'autocast_entries': 0, 'grad_scaler_constructions': 0, 'torch_lr_scheduler_constructions': 0,
    'deterministic_algorithms_calls': 0, 'TRAIN_reads': 0, 'VAL_reads': 0, 'TEST_reads': 0, 'manifest_opens': 0,
    'M8_outputs': 0, 'scientific_runs': 0}
QUALIFICATION_COUNTERS = {'qualification_processes': 2, 'qualification_checkpoint_deserializations': 2,
                          'qualification_backward_calls': 2, 'qualification_optimizer_steps': 2,
                          'qualification_scheduler_steps': 2, 'qualification_ema_updates': 2}
SCIENTIFIC_COUNTERS = {'scientific_main_runs': 0, 'scientific_optimizer_steps': 0,
                       'scientific_checkpoint_deserializations': 0, 'scientific_checkpoint_writes': 0,
                       'experiment_seed_runs': 0}
QUALIFIED = ['E07c_MAIN_RUNNER_ENCODER_LOAD_INTEGRATION_QUALIFIED', 'MAIN_RUNNER_ENCODER_LOAD_INTEGRATION',
             'E07c_MAIN_DIFFFAS_TRAINING_GRAPH_QUALIFIED', 'MAIN_DIFFFAS_TRAINING_GRAPH',
             'E07c_MAIN_B4_TRAINING_MEMORY_QUALIFIED', 'E07c_MAIN_A7_ORDER_RNG_COMPATIBILITY_QUALIFIED']
NOT_QUALIFIED = ['MAIN_PRODUCTION_RUNNER', 'MAIN_CHECKPOINT_RESUME', 'MAIN_DIFFFAS_SCIENTIFIC_TRAINING', 'M8_BANK']
EXPECTED_SEAM = ['encoder_helper_call', 'resnet18_call', 'ResNet_init_call', 'resnet18_return', 'encoder_helper_return',
                 'load_frozen_aux_encoder_call', 'load_freeze_record_call', 'verify_future_checkpoint_call',
                 'checkpoint_open', 'load_verified_whole_module_call', 'read_verified_call', 'checkpoint_open',
                 'sha256_verified', 'torch_load', 'find_class', 'encoder_identity_call', 'encoder_eval']
ENCODER_SHAPES = [[4, 256, 32, 32], [4, 512, 16, 16], [4, 512, 8, 8], [4, 7]]
LOG_TOKENS = ('M6D6I_GPU_HEAD=' + AUTHORITY, 'M6D6I_FF_ONLY=7642b23e60a1d89fcd481b03d4c9cb361f4c6e20..' + AUTHORITY,
              f'{FROZEN_SHA}  {FROZEN_GPU_PATH}', 'M6D6I_PRE_CHECKPOINT', 'M6D6I_POST_CHECKPOINT',
              'M6D6I_LAUNCH_EXIT=0', 'M6D6I_GPU_EXIT=0')


def require(ok, message):
    if not ok:
        raise ValueError('M6D6i: ' + message)


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


def at_authority(rel):
    return git('show', f'{AUTHORITY}:{rel}')


# ----------------------------------------------------------------- candidate-time checks (run once, here only)
def authority():
    require(git('branch', '--show-current').decode().strip() == BRANCH, 'branch')
    require(subprocess.run(['git', '-C', str(ROOT), 'merge-base', '--is-ancestor', AUTHORITY, 'HEAD']).returncode == 0,
            'authority is an ancestor of HEAD')
    require(git('rev-parse', 'HEAD').decode().strip() == AUTHORITY, 'uncommitted candidate on the M6D6h authority')


def tracked_unchanged(expected_changed):
    changed = sorted(git('diff', '--name-only', AUTHORITY).decode().split())
    require(changed == sorted(expected_changed), 'only ledger/index may differ from the authority: ' + json.dumps(changed))
    for rel in PROTECTED:
        require(read(rel) == at_authority(rel), 'protected byte-identical ' + rel)
    require(git('ls-files', '*.pkl', '*.pt', '*.pth', '*.ckpt', '*.safetensors').decode().strip() == '', 'no weights')
    require(git('diff', '--cached', '--name-only').decode().strip() == '', 'nothing staged')


def worktree(bookkeeping_modified):
    status = sorted(git('status', '--porcelain', '--untracked-files=all').decode().splitlines())
    expected = ['?? ' + p for p in NEW] + ([' M ' + p for p in BOOKKEEPING] if bookkeeping_modified else [])
    require(status == sorted(expected), 'worktree holds exactly the M6D6i candidate: ' + json.dumps(status))


def whitespace():
    for rel in NEW:
        raw = read(rel)
        require(not re.search(rb'[ \t]\r?\n', raw), 'no whitespace before line endings ' + rel)
        require(b'\r' not in raw or rel.endswith('.csv'), 'LF line endings ' + rel)


# ----------------------------------------------------------------- evidence (reader-scoped; no HEAD lock)
def worktree_reader(rel):
    return read(rel)


def check_process(ev, n, code):
    require((ev['milestone'], ev['classification'], ev['status'], ev['process'], ev['qualification_seed'],
             ev['experiment_seed'], ev['experiment_seed_reason'], ev['auxiliary_encoder_training_seed'],
             ev['scientific_seed_consumed'], ev['authority_commit'], ev['repository_head']) ==
            (MILESTONE, CLASSIFICATION, 'PASS', n, SEED, None, 'QUALIFICATION_ONLY_NOT_A_SCIENTIFIC_RUN', None, False,
             AUTHORITY, AUTHORITY), f'process {n} identity')
    require(ev['candidate_sha256'] == code, f'process {n} ran the candidate code')
    require(ev['counters'] == EXPECTED_COUNTERS, f'process {n} counters')
    require(ev['qualification_counters'] == {k: 1 for k in QUALIFICATION_COUNTERS} and
            ev['scientific_counters'] == SCIENTIFIC_COUNTERS, f'process {n} qualification vs scientific counters')
    f = ev['frozen_checkpoint']
    require((f['sha256'], f['bytes'], f['path'], f['path_template'], f['freeze_record_sha256']) ==
            (FROZEN_SHA, FROZEN_BYTES, FROZEN_GPU_PATH, FROZEN_TEMPLATE, FREEZE_RECORD_SHA), f'process {n} frozen asset')
    s = ev['encoder_load']
    require(s['ordered_events'] == EXPECTED_SEAM and s['sha256_rejected_events'] == 0 and
            len(s['sha256_verified_events']) == 1 and s['sha256_verified_events'][0]['sha256'] == FROZEN_SHA and
            len(s['checkpoint_read_opens']) == 2 and all(o['read_only'] and o['path'] == FROZEN_GPU_PATH
                                                          for o in s['checkpoint_read_opens']) and
            s['torch_load_call'] == {'argument_type': '_io.BytesIO', 'extra_positional': 0, 'kwargs': ['weights_only'],
                                     'weights_only': False} and s['find_class_globals_equal_m6d6c'] and
            s['harness_direct_loader_calls'] == 0, f'process {n} secure seam')
    e = ev['encoder']
    require((e['class'], e['module'], e['topology'], e['block'], e['fc'], e['parameters'], e['dtype'], e['device'],
             e['training'], e['submodules_training_false'], e['grads_all_none'], e['unchanged_after_step']) ==
            ('custom_rn.ResNet', 'custom_rn', [3, 4, 6, 3], 'BasicBlock', {'in_features': 512, 'out_features': 7,
             'bias': True}, 46233707, 'torch.float32', 'cuda:0', False, True, True, {'parameters': True, 'buffers': True}),
            f'process {n} encoder identity / immutability')
    require([o['shapes'] for o in e['forward_in_loss_path']] == [ENCODER_SHAPES], f'process {n} A6 shapes (one forward)')
    r = ev['a7_rng']
    require((r['constructor_calls'], r['throwaway_object']['fc'], r['throwaway_object']['devices'],
             r['throwaway_returned'], r['cpu_changed_by_constructor'], r['cuda_unchanged_by_helper'],
             r['secure_load_changed_cpu_rng'], r['secure_load_changed_cuda_rng'], r['surviving_17_way_objects'],
             r['direct_replay']['equal_to_helper'], r['step_cpu_rng_consumption'], r['manual_rng_draw_substitute']) ==
            (1, [512, 17], ['cpu'], False, True, True, False, False, 0, True, 0, False), f'process {n} A7 RNG proof')
    d = r['dataloader_order']
    require(d['post_eval_replay_indices'] == d['training_first_batch_indices'] and
            len(d['training_first_batch_indices']) == 4 and r['rng_at_dataloader_iter'] == r['rng_post_eval'],
            f'process {n} DataLoader RNG/order proof')
    it = ev['iteration']
    require(it['iters'] == 1 and it['finite'] and not it['save_branch_taken'] and not it['visualization_branch_taken'],
            f'process {n} one iteration')
    g = ev['gradients']
    require(g['all_finite'] and all(g['region_tensors_with_grad'].get(k, 0) > 0 for k in g['required_regions']) and
            g['region_tensors_with_nonzero_grad'].get('out', 0) > 0 and g['attention_tensors_with_grad'] > 0,
            f'process {n} gradient reachability')
    sc = ev['scheduler']
    require(sc['steps'] == 1 and sc['applied_lr_at_optimizer_step'] == [1e-5 * 4e-2] and
            sc['lr_before_scheduler_step'] == [1e-5] and sc['state_dict']['phase'] == 0 and
            sc['state_dict']['phase_step'] == 1, f'process {n} scheduler')
    o = ev['optimizer']
    h = o['hyperparameters_after']
    require(o['class'] == 'torch.optim.AdamW' and o['steps'] == 1 and o['state_step_values'] == [1.0] and
            (h['betas'], h['eps'], h['weight_decay'], h['amsgrad'], h['lr']) == ([0.9, 0.999], 1e-8, 0, False, 1e-5 * 4e-2),
            f'process {n} optimizer')
    em = ev['ema']
    require(em['decay_applied'] == 0 and em['parameters_equal_main_after'] and em['buffers_unchanged'] and
            em['accumulate_calls'] == [{'decay': 0, 'model1_is_ema': True, 'model2_is_model': True}], f'process {n} EMA')
    p = ev['parameters']
    require(p['changed_tensors'] > 0 and not p['changed_tensors_with_all_zero_grad'] and
            any(c.startswith('out.') for c in p['changed']), f'process {n} parameter update')
    m = ev['cuda_memory']
    require(m['batch_size'] == 4 and not m['oom'] and m['peak_allocated_bytes'] > 0 and not m['amp'] and
            not m['microbatch'] and not m['activation_checkpointing'], f'process {n} B=4 memory')
    require(ev['synthetic_input']['label'] == 'SYNTHETIC_STRUCTURAL_QUALIFICATION_SUBSTITUTE' and
            ev['synthetic_input']['label_tensor'] is False and ev['synthetic_input']['n_items'] == 8,
            f'process {n} synthetic input')
    require(not ev['firewall']['denied'] and not any(ev['firewall']['attempts'].values()) and
            len(ev['firewall']['frozen_checkpoint_read_opens']) == 2, f'process {n} firewall')
    require((ev['torch_save_calls'], ev['main_checkpoints_written'], ev['qualification_checkpoints_written'],
             ev['TRAIN_access'], ev['VAL_access'], ev['TEST_access'], ev['manifest_access'], ev['M8_outputs'],
             ev['production_runner_claimed'], ev['resume_executed'], ev['visualization_branch_executed'],
             ev['sampling_executed'], ev['method_status'], ev['fidelity'], ev['deviation']) ==
            (0, 0, 0, False, False, False, False, 0, False, False, False, False, 'IMPLEMENTED_NOT_EXECUTED',
             'CONTROLLED_ADAPTATION', 'DEV-021'), f'process {n} scope')
    require(ev['source_before'] == ev['source_after'] and ev['source_before']['worktree_status'] == '' and
            ev['environment_before'] == ev['environment_after'], f'process {n} source/environment stable')
    require(ev['precision']['applied_state'] == ev['precision']['expected_state'] and
            ev['precision']['use_deterministic_algorithms'] == 'NOT_SET', f'process {n} A7 precision')


def check_evidence(reader=worktree_reader):
    code = {rel: sha(reader(rel)) for rel in (GRAPH, HARNESS)}
    procs = [json.loads(reader(rel)) for rel in (P1, P2)]
    for n, ev in enumerate(procs, 1):
        check_process(ev, n, code)
    require(procs[0]['pre_backward'] == procs[1]['pre_backward'], 'pre-backward repeatability')
    agg = json.loads(reader(AGG))
    require((agg['milestone'], agg['classification'], agg['status'], agg['authority_commit'], agg['qualification_seed'],
             agg['experiment_seed'], agg['experiment_seed_reason'], agg['batch_size'], agg['candidate_sha256']) ==
            (MILESTONE, CLASSIFICATION, 'PASS', AUTHORITY, SEED, None, 'QUALIFICATION_ONLY_NOT_A_SCIENTIFIC_RUN', 4, code),
            'aggregate identity')
    require(agg['qualification_counters'] == QUALIFICATION_COUNTERS and agg['scientific_counters'] == SCIENTIFIC_COUNTERS,
            'aggregate qualification vs scientific counters')
    f = agg['frozen_checkpoint']
    require(f['before_process_1'] == f['after_process_2'] and f['unchanged'] and not f['chmod_performed'] and
            (f['before_process_1']['sha256'], f['before_process_1']['size_bytes'], f['path']) ==
            (FROZEN_SHA, FROZEN_BYTES, FROZEN_GPU_PATH), 'frozen checkpoint unchanged')
    require(agg['pre_backward_repeatability']['equal'] is True, 'aggregate pre-backward repeatability')
    pb = agg['post_backward_characterization']
    require(pb['label'] == 'CHARACTERIZATION_ONLY_NOT_AN_ACCEPTANCE_GATE' and pb['tolerance_applied'] is None and
            pb['gradient_presence_identical'] is True, 'post-backward characterization only')
    for k in ('1', '2'):
        require(agg['processes'][k]['status'] == 'PASS' and agg['processes'][k]['counters'] == EXPECTED_COUNTERS and
                agg['processes'][k]['run']['returncode'] == 0, 'aggregate process ' + k)
    require(all(g['compute_apps'] == [] and g['memory_used_mib'] <= 512 for g in agg['gpu_prelaunch_gates'].values()),
            'GPU pre-launch gates')
    require((agg['torch_save_calls'], agg['main_checkpoints_written'], agg['qualification_checkpoints_written'],
             agg['TRAIN_reads'], agg['VAL_reads'], agg['TEST_reads'], agg['manifest_opens'], agg['M8_outputs'],
             agg['M8_bank'], agg['firewall_denials'], agg['production_runner_claimed'],
             agg['bitwise_deterministic_main_training_claimed']) == (0, 0, 0, 0, 0, 0, 0, 0, False, 0, False, False),
            'aggregate scope')
    require(agg['qualified_statuses'] == QUALIFIED and agg['not_qualified'] == NOT_QUALIFIED and
            agg['method_status'] == 'IMPLEMENTED_NOT_EXECUTED', 'statuses')
    log = reader(LOG).decode()
    for token in LOG_TOKENS:
        require(token in log, 'runtime log token ' + token)
    report = reader(REPORT).decode()
    for token in (*QUALIFIED, *NOT_QUALIFIED, 'QUALIFICATION TRAINING-GRAPH EXECUTION', 'SCIENTIFIC TRAINING',
                  'qualification_optimizer_steps = 2', 'scientific_optimizer_steps = 0', FROZEN_SHA, '60607',
                  'BLOCKED_BY_MAIN_B4_MEMORY', 'CHARACTERIZATION ONLY', 'IMPLEMENTED_NOT_EXECUTED', 'DEV-021', 'M6D6j'):
        require(token in report, 'report token ' + token)
    return agg


# ----------------------------------------------------------------- ledger / index
def expected_index():
    reader = csv.DictReader(io.StringIO(at_authority(INDEX).decode()))
    baseline = list(reader)
    rows = {r['path']: r for r in baseline}
    require(reader.fieldnames == ['path', 'size_bytes', 'sha256'] and len(rows) == len(baseline) == INDEX_BASELINE_ROWS,
            'baseline index')
    for p in NEW:
        require(p not in rows, 'additive artifact ' + p)
        raw = read(p)
        rows[p] = {'path': p, 'size_bytes': str(len(raw)), 'sha256': sha(raw)}
    out = io.StringIO(newline='')
    writer = csv.DictWriter(out, fieldnames=reader.fieldnames, lineterminator='\r\n')
    writer.writeheader()
    writer.writerows(rows[k] for k in sorted(rows))
    return out.getvalue().encode(), len(rows)


def check_ledger_row(row, agg):
    require((row['milestone'], row['classification'], row['method_id'], row['status'], row['qualification_seed'],
             row['experiment_seed'], row['experiment_seed_reason'], row['authority_commit'], row['git_commit'],
             row['commit'], row['push'], row['TRAIN_access'], row['VAL_access'], row['TEST_access'], row['M8_bank'],
             row['fidelity_class'], row['deviation'], row['method_status'], row['amendment_created']) ==
            (MILESTONE, CLASSIFICATION, 'E07c', 'PASS', SEED, None, 'QUALIFICATION_ONLY_NOT_A_SCIENTIFIC_RUN', AUTHORITY,
             AUTHORITY, False, False, False, False, False, False, 'CONTROLLED_ADAPTATION', 'DEV-021',
             'IMPLEMENTED_NOT_EXECUTED', False), 'ledger identity')
    for k, v in {**QUALIFICATION_COUNTERS, **SCIENTIFIC_COUNTERS}.items():
        require(row[k] == v, 'ledger counter ' + k)
    require(row['qualified_statuses'] == QUALIFIED and row['not_qualified'] == NOT_QUALIFIED, 'ledger statuses')
    require((row['frozen_checkpoint_sha256'], row['frozen_checkpoint_bytes']) == (FROZEN_SHA, FROZEN_BYTES),
            'ledger frozen asset')
    require(row['artifact_index_rows_before'] == INDEX_BASELINE_ROWS and
            row['artifact_index_rows_after'] == INDEX_BASELINE_ROWS + len(NEW), 'ledger index rows')
    require(sorted(row['artifacts_sha256']) == sorted(NEW), 'ledger artifact list')
    for path, h in row['artifacts_sha256'].items():
        require(sha(read(path)) == h, 'ledger artifact ' + path)
    require(row['peak_allocated_bytes'] == {k: v['peak_allocated_bytes'] for k, v in agg['processes'].items()},
            'ledger peak memory')


def verify(stage):
    """stage: 'before_ledger', 'before_index' (ledger appended), 'final'."""
    authority()
    appended = stage != 'before_ledger'
    tracked_unchanged({'before_ledger': [], 'before_index': [LEDGER], 'final': [INDEX, LEDGER]}[stage])
    whitespace()
    agg = check_evidence()
    prefix = at_authority(LEDGER)
    current = (ROOT / LEDGER).read_bytes()
    require(len(prefix.splitlines()) == LEDGER_PREFIX_ROWS and current.startswith(prefix),
            f'first {LEDGER_PREFIX_ROWS} ledger rows byte-identical')
    if not appended:
        require(current == prefix, 'ledger not yet appended')
        worktree(False)
    else:
        require(len(current.splitlines()) == LEDGER_PREFIX_ROWS + 1 and current.endswith(b'\n'), 'exactly one append')
        row = json.loads(current[len(prefix):])
        require(row['committed_prefix_rows'] == LEDGER_PREFIX_ROWS and row['committed_prefix_sha256'] == sha(prefix),
                'ledger prefix binding')
        check_ledger_row(row, agg)
    expected, count = expected_index()
    if stage == 'final':
        require((ROOT / INDEX).read_bytes() == expected, 'CRLF sorted artifact index')
        worktree(True)
    require(not {'torch', 'yaml', 'pyarrow', 'numpy', 'PIL'} & set(sys.modules), 'static preflight')
    return {'status': 'PASS', 'stage': stage, 'ledger_rows': len(current.splitlines()),
            f'first_{LEDGER_PREFIX_ROWS}_rows_byte_identical': True, 'artifact_index_rows_before': INDEX_BASELINE_ROWS,
            'artifact_index_rows_expected': count, 'artifact_rows_added': len(NEW), 'new_files': list(NEW),
            'modified_existing_files': list(BOOKKEEPING) if appended else [],
            'qualification_counters': QUALIFICATION_COUNTERS, 'scientific_counters': SCIENTIFIC_COUNTERS,
            'torch_imported': False, 'checkpoint_bytes_opened': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before-ledger', action='store_true')
    parser.add_argument('--rebuild-index', action='store_true')
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
