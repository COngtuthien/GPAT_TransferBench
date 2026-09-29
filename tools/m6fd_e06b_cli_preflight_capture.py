import json
import os
import sys
sys.path.insert(0, '.')
import tools.run_e06b as cli  # noqa: E402

Q = '/home/student20261/workdir/GPAT_TransferBench_runtime/qualification/m6fd/E06b/cli_preflight'
env = {'PYTHONHASHSEED': '42', 'NVIDIA_TF32_OVERRIDE': '0', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}
cfg = 'configs/execution/m5_gpu_3090.yaml'
plan = cli.preconditions(cli.parse(['--seed', '42', '--execution-config', cfg, '--preflight-only']), environ=env, dirty=False)
out = {'milestone': 'M6F-D', 'label': 'PRODUCTION_PATH_QUALIFICATION_ONLY',
       'check': 'tools/run_e06b.py preconditions (no Torch, no run)',
       'dirty_injected_false_reason': 'the uncommitted M6F-D runner files are staged for qualification; the real '
                                      'clean-worktree gate is exercised by the dirty_worktree refusal below',
       'preflight_pass': True, 'run_dir': plan['run_dir'], 'run_dir_exists': os.path.exists(plan['run_dir']),
       'torch_imported': plan['torch_imported'], 'resume': plan['resume'], 'source_commit': plan['source_commit'],
       'lightcnn_sha256': plan['lightcnn']['sha256'], 'faces_256_root': plan['storage']['faces_256_root'],
       'exec_config_sha256': plan['storage']['exec_config_sha256'],
       'runtime_fidelity_assessment': plan['runtime_fidelity_assessment'], 'identities': plan['identities']}
cases = (('dirty_worktree', ['--seed', '42', '--execution-config', cfg], env, True),
         ('qualification_seed_60801', ['--seed', '60801', '--execution-config', cfg], dict(env, PYTHONHASHSEED='60801'), False),
         ('wrong_pythonhashseed', ['--seed', '42', '--execution-config', cfg], dict(env, PYTHONHASHSEED='1'), False),
         ('tf32_not_disabled', ['--seed', '42', '--execution-config', cfg], dict(env, NVIDIA_TF32_OVERRIDE='1'), False),
         ('resume_state_flag', ['--seed', '42', '--execution-config', cfg, '--resume-state', 'y'], env, False),
         ('qualification_limit_flag', ['--seed', '42', '--execution-config', cfg,
                                       '--qualification-max-logical-batches', '1'], env, False),
         ('lambda_pair_override', ['--seed', '42', '--execution-config', cfg, '--lambda-pair', '0'], env, False),
         ('include_siw_flag', ['--seed', '42', '--execution-config', cfg, '--include-siw'], env, False))
ref = {}
for name, argv, e2, d in cases:
    try:
        cli.preconditions(cli.parse(argv), environ=e2, dirty=d)
        ref[name] = 'NOT_REFUSED'
    except SystemExit as e:
        ref[name] = 'REFUSED_EXIT_%s' % e.code
out['refusals'] = ref
out['all_refused'] = all(v == 'REFUSED_EXIT_2' for v in ref.values())
out['torch_imported_after_checks'] = 'torch' in sys.modules
with open(Q + '/M6FD_E06B_CLI_PREFLIGHT.json', 'w') as fh:
    fh.write(json.dumps(out, indent=2, sort_keys=True) + '\n')
print(json.dumps({'all_refused': out['all_refused'], 'run_dir_exists': out['run_dir_exists'],
                  'torch': out['torch_imported_after_checks']}))
