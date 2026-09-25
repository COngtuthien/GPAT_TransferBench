#!/usr/bin/env python3
"""Verify M6D6d E07c A7 execution-policy evidence; append-check the ledger; rebuild the index LAST.

STATIC: no Torch, no YAML parser, no CUDA, no model construction, no checkpoint I/O. Only
explicit allowlisted paths are read. The A7 overlay is checked by its pinned SHA256 and exact
lines. The pinned upstream order is re-derived here by stdlib AST, independently of the harness.
Historical index rows are carried from the authoritative commit without opening their targets
(tools/build_artifact_index.py is NOT used, because it hashes every file); only the explicit new
M6D6d artifacts are hashed.
"""
import argparse
import ast
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = 'e70c221464b4a2db4ea3d45176a0cee52a7747cf'
SPEC_SHA = 'f7d2371682ad9504f59f0c6984e4b1a741f855f1b1f60838f9d60b160281489e'
PIN = '23f40519ec25a833ebc06842aa6fbab74fad4d15'
TREE = 'd190a5fb4a94cb9e423215f9a24a7863aa17ea65'
SOURCE_FILES = {'FAS_train.py': 'c5eb42a1193f7708b9e972db0faac602c82ffe2dc10578b6c9013d81e26e4f84',
                'FAS_sample.py': '8b9787a6e3ed7d539ebd845605f8a4299c5795eaffde38066b00d79bb03060f0',
                'models/unet_autoenc.py': '127ecd59fbbbd0d191a713aae62b11ea3de7058d42c2889100ff6a44199d2963',
                'models/custom_rn.py': 'fa788c4d2453b40585b295a8292eaf1afce7a0c622eb8ecc2adec5453a9a64c3',
                'models/pretrain_classifier.py': '3444691f6f2c59b41a07f06db9388b134e849bb889c5f6b03489d47c7e037b16'}
A6_OVERLAY_SHA = 'dd3f29aa8ff96de9c0e2d504e6d07f4788d8fb8d030ffa26ef51443104ca971b'
LOCK_SHA = '0c909de1e3e3e8c129e0d9f4aab6386cee8e4eac0ca45d79423f795cced5f450'
M6D6A_HELPERS_SHA = '1eeb7c037dbb25876eb6257a4eb0dee40bc3a998d5d20c03a2b4b937abe12541'
M6D6C_SEAM_SHA = 'a794eac24b3572fc28158c1ca087d7487c1d9682be87ca232ce24a6295a4b944'
A7_DOC = 'docs/spec/amendments/GPAT_TransferBench_v1_0_Amendment_A7_E07c_Execution_Policy.md'
A7_DOC_SHA = '0a46c3b06e277ad38b17a6e85fe2924441d8d13b1de887bf54903dda9aa292ba'
A7_OVERLAY = 'configs/amendments/e07c_a7_execution_policy.yaml'
A7_OVERLAY_SHA = '3e4c758c1421aac6e993724a8a80b99e8b0754e75b983cec5ffca41479f492a3'
QUALIFICATION_SEED = 60604
PROCESS_SHA = '4c608129bc5356c9aa8e04f09d68bf2fc961bb9bd5b9985a0496ddb961d2ba60'
INITIAL_RNG = 'c73af325596cd62bc1ba2ed2e6be7a49ffd5ea6da044ac565f01cf8bfc0e8d01'
AFTER_RNG = '530629ef50c480dd347ccd9ccccf3fdd4858f40074879f773893e45d91c06b71'
NEXT_RAND = 'ad0e952d042755ac13e21c5fe8dbb3880191b565ced7a01c517f5fcf25636464'
BYPASS_RAND = '65ddce509a9c14c7382ecca332e26f81f88c4ec1f44b667e9743b5aa2d2890f9'
INDEX = 'outputs/audit/ARTIFACT_INDEX.csv'
LEDGER = 'outputs/audit/EXECUTION_LEDGER.jsonl'
LEDGER_PREFIX_ROWS = 118
LEDGER_PREFIX_SHA = '2fd718672fe123ba7ec1e729129f285b5546b244b7e7db19060f4e382f836295'
INDEX_BASELINE_ROWS = 674
CLASSIFICATION = 'M6D6D_E07C_EXECUTION_POLICY_QUALIFICATION'
BASE = 'outputs/audit/M6D6D_E07C_EXECUTION_POLICY'
PROCESSES = tuple(f'outputs/audit/M6D6D_E07C_RNG_PROCESS_{i}.json' for i in (1, 2))
LOG = 'outputs/audit/M6D6D_E07C_RUNTIME_LOG.txt'
POLICY = 'methods/difffas/execution_policy.py'
HARNESS = 'methods/difffas/execution_policy_qualification.py'
TESTS = 'tests/test_m6d6d_e07c_execution_policy.py'
PREFLIGHT = 'tools/m6d6d_e07c_execution_policy_preflight.py'
ARTIFACTS = tuple(sorted((A7_DOC, A7_OVERLAY, POLICY, HARNESS, TESTS, PREFLIGHT, *PROCESSES, LOG,
                          BASE + '.md', BASE + '.json')))
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
             'third_party/source_pins.json', 'methods/difffas/__init__.py', 'methods/difffas/adapter.py',
             'methods/difffas/contract.py', 'methods/difffas/encoder.py', 'methods/difffas/source.py',
             'methods/difffas/sampler.py', 'methods/difffas/seed_adapter.py',
             'methods/difffas/runtime_qualification.py', 'methods/difffas/aux_training_qualification.py',
             'methods/difffas/aux_checkpoint.py', 'methods/difffas/aux_checkpoint_qualification.py',
             'methods/common/learned.py', 'methods/common/upstream.py', 'methods/common/config.py',
             'environments/e07c.lock.json', 'environments/e07c.runtime.json',
             'environments/e07c.conda-explicit.txt', 'environments/e07c.pip-freeze.txt',
             'environments/e07c.pip-requirements.txt',
             'outputs/audit/M6D6C_E07C_AUX_CHECKPOINT_COMPATIBILITY.md',
             'outputs/audit/M6D6C_E07C_AUX_CHECKPOINT_COMPATIBILITY.json',
             'outputs/audit/M6D6C_E07C_AUX_CHECKPOINT_WRITER.json',
             'outputs/audit/M6D6C_E07C_AUX_CHECKPOINT_LOADER_PROCESS_1.json',
             'outputs/audit/M6D6C_E07C_AUX_CHECKPOINT_LOADER_PROCESS_2.json',
             'outputs/audit/M6D6C_E07C_AUX_CHECKPOINT_RUNTIME_LOG.txt',
             'tests/test_m6d6c_e07c_aux_checkpoint.py', 'tests/test_m6d6b_e07c_aux_training_graph.py',
             'tests/test_m6d6a_e07c_runtime.py', 'tests/test_m6c2b3_contract.py', 'tests/test_m6c2b3_encoder.py',
             'tests/test_m6c2b3_runtime.py', 'tools/m6d6c_e07c_aux_checkpoint_preflight.py',
             'tools/build_artifact_index.py')
QUALIFIED = ['E07c_PRODUCTION_PRECISION_POLICY_FROZEN', 'E07c_MAIN_ENCODER_LOADER_RNG_POLICY_FROZEN',
             'E07c_THROWAWAY_ENCODER_RNG_COMPATIBILITY_QUALIFIED', 'E07c_EXECUTION_POLICY_RUNTIME_QUALIFIED',
             'IMPLEMENTED_NOT_EXECUTED', 'CONTROLLED_ADAPTATION']
RETAINED = ['E07c_EXECUTION_ENVIRONMENT_QUALIFIED', 'E07c_CONDITIONING_ENCODER_RUNTIME_QUALIFIED',
            'E07c_MAIN_ARCHITECTURE_RUNTIME_QUALIFIED', 'E07c_SYNTHETIC_FORWARD_RUNTIME_QUALIFIED',
            'E07c_AUX_ENCODER_TRAINING_GRAPH_QUALIFIED', 'E07c_AUX_ENCODER_OPTIMIZER_STEP_QUALIFIED',
            'E07c_AUX_CHECKPOINT_WHOLE_MODULE_SERIALIZATION_QUALIFIED',
            'E07c_AUX_CHECKPOINT_LOADER_COMPATIBILITY_QUALIFIED', 'E07c_AUX_CHECKPOINT_SHA_BEFORE_DESERIALIZE_QUALIFIED']
NOT_QUALIFIED = ['REAL_TRAIN_PATH', 'AUXILIARY_ENCODER_200_EPOCH_TRAINING', 'AUX_B256_TRAINING_MEMORY',
                 'AUX_PRODUCTION_RUNNER', 'MAIN_DIFFFAS_TRAINING_GRAPH', 'MAIN_CHECKPOINT_RESUME',
                 'MAIN_RUNNER_ENCODER_LOAD_INTEGRATION', 'MAIN_PRODUCTION_RUNNER', 'SCIENTIFIC_TRAINING', 'M8_BANK']
RESOLVED = ['PRODUCTION_PRECISION_POLICY', 'THROWAWAY_RESNET18_RNG_DECISION']
EXPECTED_STATE = {'default_dtype': 'torch.float32', 'matmul_tf32': False, 'cudnn_tf32': False,
                  'cudnn_benchmark': False, 'cudnn_deterministic': True, 'float32_matmul_precision': 'highest',
                  'autocast_cuda': False, 'autocast_cpu': False, 'deterministic_algorithms_forced': False}
COUNTERS = {'optimizer_constructions': 0, 'optimizer_step_calls': 0, 'backward_calls': 0, 'autograd_grad_calls': 0,
            'torch_save_calls': 0, 'torch_load_calls': 0, 'torch_load_intercepted_before_io': 1,
            'autocast_entries': 0, 'grad_scaler_constructions': 0}
SCOPE = {'backward_calls': 0, 'optimizer_step_calls': 0, 'epochs': 0, 'torch_save_calls': 0, 'torch_load_calls': 0,
         'auxiliary_encoder_trained': False, 'scientific_training': False, 'scientific_checkpoint_created': False,
         'scientific_checkpoint_loaded': False, 'main_difffas_training': False, 'benchmark_data_access': False,
         'TRAIN_access': False, 'VAL_access': False, 'TEST_access': False, 'synthetic_bank': False,
         'scientific_seed_consumed': False, 'source_patch': 'NONE', 'fidelity': 'CONTROLLED_ADAPTATION'}
SESSION = {'benchmark_manifest_reads': 0, 'benchmark_image_reads': 0, 'TRAIN_reads': 0, 'VAL_reads': 0,
           'TEST_reads': 0, 'firewall_denials': 0, 'backward_calls': 0, 'optimizer_step_calls': 0,
           'torch_save_calls': 0, 'torch_load_calls': 0, 'torch_load_pre_io_interceptions': 2,
           'scientific_checkpoints_created': 0, 'scientific_checkpoint_loads': 0,
           'auxiliary_encoder_training_runs': 0, 'scientific_main_runs': 0, 'scientific_seed_runs_completed': 0}
OVERLAY_LINES = ('classification: DETERMINISTIC_IMPLEMENTATION_CLARIFICATION', 'upstream_author_choice_claimed: false',
                 '  sha256: ' + A7_DOC_SHA, '  sha256: ' + A6_OVERLAY_SHA, '  pinned_commit: ' + PIN,
                 '  source_pin_changed: false', '  official_source_modified: false',
                 '  fidelity_class: CONTROLLED_ADAPTATION', '  deviation: DEV-021', '  new_fidelity_class: false',
                 '  new_deviation: false', '  dtype: float32', '  autocast: false', '  grad_scaler: false',
                 '  matmul_tf32: false', '  cudnn_tf32: false', '  benchmark: false', '  deterministic: true',
                 'use_deterministic_algorithms: NOT_SET',
                 '  preserve_upstream_throwaway_encoder_constructor_rng: true',
                 '  constructor: custom_rn.resnet18(pretrained=False)', '  device: cpu', '  count: 1',
                 '  conditioning_role: false', '  object_returned: false', '  mandatory: true',
                 '  entry: methods/difffas/aux_checkpoint.py::load_frozen_aux_encoder',
                 '  sampler_FAS_sample_py_48: NOT_DECIDED_BY_A7', '  qualification_seed: 60604')
LAUNCH_ONLY = ('PYTHONHASHSEED', 'TMPDIR')
FORBIDDEN_WORDING = ('faithful difffas', 'native difffas', 'official reproduction of', 'auxiliary_encoder_trained:',
                     'auxiliary encoder trained', 'aux_production_runner_qualified', 'real_train_path_qualified',
                     'aux_b256_training_memory_qualified', 'main_difffas_training_graph_qualified',
                     'main_training_graph_qualified', 'scientifically_trained', 'm8_ready',
                     'upstream authors chose fp32', 'new scientific deviation is')


def require(ok, message):
    if not ok:
        raise ValueError('M6D6d: ' + message)


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
    require(p.suffix not in {'.pkl', '.pt', '.pth', '.ckpt', '.parquet'}, 'no weight/array/manifest read')
    return (ROOT / p).read_bytes()


def source_cache():
    root = ROOT / 'third_party/source_cache/difffas'
    require(git('rev-parse', 'HEAD', root=root).decode().strip() == PIN, 'source commit')
    require(git('rev-parse', 'HEAD^{tree}', root=root).decode().strip() == TREE, 'source tree')
    require(git('status', '--porcelain', '--untracked-files=no', root=root).decode().strip() == '',
            'tracked pinned source unmodified')
    for rel, digest in SOURCE_FILES.items():
        require(sha((root / rel).read_bytes()) == digest, rel + ' SHA256')
    return root


def upstream_ast(root, order):
    """Stdlib re-derivation: throwaway body, constructor, and every recorded order step at its exact line."""
    parsed = {rel: ast.parse((root / rel).read_bytes()) for rel in ('FAS_train.py', 'models/unet_autoenc.py',
                                                                    'models/custom_rn.py')}
    cls = next(n for n in parsed['models/unet_autoenc.py'].body if isinstance(n, ast.ClassDef) and
               n.name == 'BeatGANsAutoencModel')
    enc = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'encoder')
    ctor = next(n for n in parsed['models/custom_rn.py'].body if isinstance(n, ast.FunctionDef) and n.name == 'resnet18')
    for step in order:
        hits = [n for n in ast.walk(parsed[step['file']]) if isinstance(n, ast.stmt) and n.lineno == step['line'] and
                ast.unparse(n).splitlines()[0] == step['statement']]
        require(len(hits) == 1, f"order step {step['step']} at {step['file']}:{step['line']}")
    return {'encoder_body': [(ast.unparse(s), s.lineno) for s in enc.body],
            'ctor_defaults': [ast.literal_eval(d) for d in ctor.args.defaults],
            'ctor_return': next(ast.unparse(n.value) for n in ctor.body if isinstance(n, ast.Return))}


def policy_ast():
    tree = ast.parse(read(POLICY))
    consts = {ast.unparse(n.targets[0]): n.value for n in tree.body if isinstance(n, ast.Assign)}
    fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    calls = lambda node: [ast.unparse(c.func) for c in ast.walk(node) if isinstance(c, ast.Call)]
    helper = fns['consume_upstream_encoder_loader_rng']
    ctor = [c for c in ast.walk(helper) if isinstance(c, ast.Call) and ast.unparse(c.func).endswith('resnet18')]
    bare = [n for n in ast.walk(helper) if isinstance(n, ast.Expr) and n.value in ctor]
    assigned = sorted(ast.unparse(t) for n in ast.walk(fns['apply_e07c_precision_policy']) if isinstance(n, ast.Assign)
                      for t in n.targets if ast.unparse(t).startswith('torch.'))
    manual = {'torch.rand', 'torch.randn', 'torch.randint', 'torch.randperm', 'torch.manual_seed', 'torch.set_rng_state',
              'torch.get_rng_state', 'torch.use_deterministic_algorithms', 'torch.autocast', 'torch.amp.GradScaler',
              'torch.load', 'torch.save'}
    return {'overlay_sha': ast.literal_eval(consts['A7_OVERLAY_SHA256']),
            'precision': ast.literal_eval(consts['PRECISION']), 'cudnn': ast.literal_eval(consts['CUDNN']),
            'expected_state': ast.literal_eval(consts['EXPECTED_STATE']),
            'ctor_calls': [ast.unparse(c) for c in ctor], 'ctor_bare_statement': len(bare) == 1,
            'helper_text_clean': not any(t in ast.unparse(helper) for t in ('torchvision', '.cuda', 'torch.load',
                                                                               'torch.save', 'fc =', 'Linear')),
            'assigned_flags': assigned, 'manual_rng_or_forbidden_calls': sorted(manual & set(calls(tree))),
            'main_runner_calls': calls(fns['main_runner_encoder'])}


def check_process(r, lock):
    require(r['status'] == 'PASS' and r['label'] == 'EXECUTION_POLICY_QUALIFICATION_ONLY' and r['milestone'] == 'M6D6d',
            'process pass')
    require(r['qualification_seed'] == QUALIFICATION_SEED and r['experiment_seed'] is None and
            r['auxiliary_encoder_training_seed'] is None and QUALIFICATION_SEED not in (42, 1337, 2026), 'seed')
    require(all(r[k] == v for k, v in SCOPE.items()) and r['counters'] == COUNTERS, 'scope flags / counters')
    s = r['source_before']
    require(s == r['source_after'] and s['worktree_status'] == '', 'source mutation')
    require((s['commit'], s['tree'], s['a6_overlay_sha256']) == (PIN, TREE, A6_OVERLAY_SHA), 'pinned source / A6')
    require(all(s['files_sha256'][k] == v for k, v in SOURCE_FILES.items()), 'process source digests')
    env = r['environment_before']
    require(env == r['environment_after'], 'environment mutation')
    require({k: v for k, v in env.items() if k != 'launch_environment'} ==
            {k: v for k, v in lock['identity'].items() if k != 'launch_environment'}, 'executed identity = lock')
    strip = lambda d: {k: v for k, v in d.items() if k not in LAUNCH_ONLY}
    require(strip(env['launch_environment']) == strip(lock['identity']['launch_environment']) and
            env['launch_environment']['PYTHONHASHSEED'] == str(QUALIFICATION_SEED), 'launch env')
    a7 = r['a7']
    require((a7['overlay_sha256'], a7['document']['sha256'], a7['classification']) ==
            (A7_OVERLAY_SHA, A7_DOC_SHA, 'DETERMINISTIC_IMPLEMENTATION_CLARIFICATION'), 'A7 identity in process')
    p = r['precision']
    require(p['applied_state'] == p['state_after_all_sections'] == p['expected_state'] == EXPECTED_STATE and
            p['library_defaults_before_policy']['cudnn_tf32'] is True and not p['grad_scaler_constructed'] and
            not p['autocast_entered'] and p['use_deterministic_algorithms'] == 'NOT_SET', 'precision state')
    require(r['upstream_precision_requests'] == [], 'pinned source requests no precision feature')
    q = r['rng_equivalence']
    paths = q['paths']
    require(q['initial_cpu_rng_sha256'] == INITIAL_RNG and q['reference_after_cpu_rng_sha256'] == AFTER_RNG and
            q['reference_next_rand_sha256'] == NEXT_RAND and q['bypass_next_rand_sha256'] == BYPASS_RAND, 'RNG digests')
    for name in ('helper', 'direct', 'call_site', 'helper_ctx', 'bypass'):
        p = paths[name]
        require(p['before']['cpu_sha256'] == INITIAL_RNG and p['cuda_memory_allocated_delta_bytes'] == 0 and
                p['throwaway_instances_after'] == 0, name + ': start state / discard')
        require(all(p['before'][k] == p['after'][k] for k in ('cuda_sha256', 'python_random_sha256', 'numpy_sha256')),
                name + ': non-CPU RNG untouched')
        replay = name != 'bypass'
        require(p['after']['cpu_sha256'] == (AFTER_RNG if replay else INITIAL_RNG) and
                p['next_rand']['sha256'] == (NEXT_RAND if replay else BYPASS_RAND), name + ': transition')
    for name in ('helper', 'helper_ctx'):
        require(paths[name]['return_value_plain_data'] and paths[name]['return_value']['object_returned'] is False and
                paths[name]['return_value']['role'] == 'RNG_COMPATIBILITY_ONLY', name + ': plain record')
    ci = q['call_site_interception']
    require(ci['caller'] == {'file': 'unet_autoenc.py', 'function': 'encoder', 'line': 75} and
            not ci['io_performed'] and not ci['deserialization'] and ci['rng_at_torch_load'] == paths['call_site']['after'],
            'call-site interception before I/O')
    d = paths['direct']['object']
    require((d['type'], d['fc'], d['devices']) == ('custom_rn.ResNet', [512, 17], ['cpu']) and
            ci['object']['aggregate_parameter_sha256'] == d['aggregate_parameter_sha256'] and
            ci['object']['type'] == 'models.custom_rn.ResNet', 'throwaway identity')
    f = r['runtime_forward']
    require(f['finite_all'] and f['a6_feature_shapes'] == [[4, 256, 32, 32], [4, 512, 16, 16], [4, 512, 8, 8]] and
            f['outputs']['main_model_output']['shape'] == [4, 6, 256, 256] and
            f['precision_state_inside_forwards'] == ['encoder', 'main_model'] and not f['architecture_changed'] and
            not f['backward'] and not f['optimizer_step'] and not f['checkpoint'], 'runtime forward')
    fw = r['firewall']
    require(not fw['denied'] and not any(fw['attempts'].values()), 'firewall zero denials')
    require(all(a[0] in ('git', 'nvidia-smi') or a == ['/sbin/ldconfig', '-p'] for a in fw['subprocesses']),
            'subprocess allowlist')


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
    require(status == sorted(expected), 'worktree holds exactly the intended M6D6d changes: ' + json.dumps(status))
    require(git('diff', '--cached', '--name-only').decode().strip() == '', 'nothing staged')
    return status


def verify(stage):
    """stage: 'before_ledger' (evidence only), 'before_index' (ledger appended), 'final'."""
    for ref in ('HEAD', 'origin/m6-baselines'):
        require(git('rev-parse', ref).decode().strip() == AUTHORITY, ref + ' authority')
    require(sha(read('docs/spec/GPAT_TransferBench_v1_0_Frozen_Specification_2026.docx')) == SPEC_SHA, 'spec SHA256')
    for path in PRESERVED:
        require(read(path) == git('show', AUTHORITY + ':' + path), 'immutable input ' + path)
    require(sha(read('methods/difffas/runtime_qualification.py')) == M6D6A_HELPERS_SHA and
            sha(read('methods/difffas/aux_checkpoint.py')) == M6D6C_SEAM_SHA, 'M6D6a helpers / M6D6c seam unchanged')
    # A7 authority
    overlay = read(A7_OVERLAY)
    require(sha(overlay) == A7_OVERLAY_SHA and sha(read(A7_DOC)) == A7_DOC_SHA, 'A7 digests')
    lines = set(overlay.decode().splitlines())
    missing = [ln for ln in OVERLAY_LINES if ln not in lines]
    require(not missing, 'A7 overlay decisions: ' + json.dumps(missing))
    pa = policy_ast()
    require(pa['overlay_sha'] == A7_OVERLAY_SHA, 'execution_policy pins the A7 overlay digest')
    require(pa['precision'] == {'dtype': 'float32', 'autocast': False, 'grad_scaler': False, 'matmul_tf32': False,
                                'cudnn_tf32': False} and pa['cudnn'] == {'benchmark': False, 'deterministic': True} and
            pa['expected_state'] == EXPECTED_STATE, 'policy constants')
    require(pa['ctor_calls'] == ["modules['custom_rn'].resnet18(pretrained=False)"] and pa['ctor_bare_statement'] and
            pa['helper_text_clean'], 'helper: one bare pinned constructor, nothing else')
    require(pa['assigned_flags'] == ['torch.backends.cuda.matmul.allow_tf32', 'torch.backends.cudnn.allow_tf32',
                                     'torch.backends.cudnn.benchmark', 'torch.backends.cudnn.deterministic'] and
            pa['manual_rng_or_forbidden_calls'] == [], 'flags set; no manual RNG / autocast / GradScaler / save / load')
    require(pa['main_runner_calls'] == ['consume_upstream_encoder_loader_rng', 'load_frozen_aux_encoder'],
            'main-runner seam: RNG replay then mandatory secure loader')
    # processes
    lock_raw = read('environments/e07c.lock.json')
    require(sha(lock_raw) == LOCK_SHA, 'environment lock digest')
    lock = json.loads(lock_raw)
    raw = [read(p) for p in PROCESSES]
    require(raw[0] == raw[1] and sha(raw[0]) == PROCESS_SHA, 'two processes byte-identical')
    run = json.loads(raw[0])
    check_process(run, lock)
    root = source_cache()
    up = upstream_ast(root, run['upstream_main_order'])
    require(up['encoder_body'][:2] == [('model_autoencoder = resnet18()', 74), ('model_autoencoder = torch.load(path)', 75)]
            and up['ctor_defaults'] == [False, True] and
            up['ctor_return'] == "_resnet('resnet18', BasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)",
            'pinned throwaway + constructor (AST)')
    roles = [s['role'] for s in run['upstream_main_order']]
    require(roles.index('diffusion_construction') < roles.index('throwaway_constructor_rng_consumption') <
            roles.index('secure_frozen_auxiliary_checkpoint_load') < roles.index('encoder_eval') <
            roles.index('training_iterator_cpu_rng'), 'source-derived order')
    for rel, h in run['source_before']['files_sha256'].items():
        require(sha((root / rel).read_bytes()) == h, 'laptop source closure ' + rel)
    # audit record
    audit = json.loads(read(BASE + '.json'))
    require(audit['classification'] == CLASSIFICATION and audit['milestone'] == 'M6D6d' and
            audit['final_status'] == 'PASS' and audit['qualified_statuses'] == QUALIFIED and
            audit['retained_statuses'] == RETAINED and audit['not_qualified'] == NOT_QUALIFIED and
            audit['resolved_open_items'] == RESOLVED and audit['method_status'] == 'IMPLEMENTED_NOT_EXECUTED' and
            audit['fidelity_class'] == 'CONTROLLED_ADAPTATION' and audit['deviation'] == 'DEV-021' and
            audit['new_deviation'] is False, 'status fields')
    require(audit['session_access_audit']['counts'] == SESSION and
            audit['session_access_audit']['scope'] == 'COMPLETE_M6D6D_SESSION_LAPTOP_AND_GPU', 'session access audit')
    require(audit['environment_lock_sha256'] == LOCK_SHA and
            not audit['environment_reverification']['environment_rebuilt_or_mutated'], 'environment re-verification')
    for path, h in audit['artifacts_sha256'].items():
        require(sha(read(path)) == h, 'artifact ' + path)
    require(sorted(audit['artifacts_sha256']) == sorted(set(ARTIFACTS) - {BASE + '.json'}), 'artifact list')
    for path, h in audit['preserved_inputs_sha256'].items():
        require(sha(read(path)) == h, 'preserved input ' + path)
    final_tests = [t for t in audit['tests'] if t['final']]
    require(len(final_tests) == 10 and all(t['failures'] == t['errors'] == 0 for t in final_tests) and
            all(t['skipped'] == 0 for t in final_tests if t['host'] == 'gpu'), 'final test runs')
    require(all(t['audit_denied'] == t['audit_manifest'] == t['audit_image'] == t['audit_data_or_runs'] == 0
                for t in audit['tests']), 'test audit counts')
    log = read(LOG).decode()
    after = [ln for ln in log.splitlines() if ln.startswith('scientific_paths_after:')]
    require(after == ['scientific_paths_after: runs=ABSENT e07c_root=ABSENT aux_seed_42=ABSENT encoder_final=ABSENT'] * 2,
            'no scientific run root or auxiliary checkpoint')
    for token in ('weight_suffix_files_in_m6d6d_after: 0', 'RUNS_ABSENT', 'status_lines 0', 'source_cache_status_lines 0',
                  PROCESS_SHA + '  m6d6d_process_1.json', PROCESS_SHA + '  m6d6d_process_2.json'):
        require(token in log, 'runtime log: ' + token)
    report = read(BASE + '.md').decode().lower()
    for phrase in FORBIDDEN_WORDING:
        require(phrase not in report, 'forbidden wording: ' + phrase)
    for phrase in ('controlled_adaptation', 'implemented_not_executed', 'e07c_production_precision_policy_frozen',
                   'e07c_throwaway_encoder_rng_compatibility_qualified', 'rng_compatibility_only', 'owner-frozen',
                   'not a claim that the upstream authors chose it', 'b256 scientific training memory remains unqualified'):
        require(phrase in report, 'required wording: ' + phrase)
    # ledger
    prefix = git('show', AUTHORITY + ':' + LEDGER)
    current = read(LEDGER)
    require(len(prefix.splitlines()) == LEDGER_PREFIX_ROWS and sha(prefix) == LEDGER_PREFIX_SHA, 'ledger authority')
    require(current.startswith(prefix), f'first {LEDGER_PREFIX_ROWS} ledger rows byte-identical')
    if stage == 'before_ledger':
        require(current == prefix, 'ledger not yet appended')
        worktree(False)
    else:
        require(len(current.splitlines()) == LEDGER_PREFIX_ROWS + 1 and current.endswith(b'\n'), 'exactly one append')
        row = json.loads(current[len(prefix):])
        require((row['classification'], row['milestone'], row['method_id']) == (CLASSIFICATION, 'M6D6d', 'E07c'),
                'ledger row identity')
        require(row['final_status'] == 'PASS' and row['qualified_statuses'] == QUALIFIED and
                row['not_qualified'] == NOT_QUALIFIED and row['method_status'] == 'IMPLEMENTED_NOT_EXECUTED' and
                row['fidelity_class'] == 'CONTROLLED_ADAPTATION', 'ledger status fields')
        require(row['production_precision_policy'] == {'dtype': 'FP32', 'autocast': False, 'grad_scaler': False,
                                                       'matmul_tf32': False, 'cudnn_tf32': False,
                                                       'cudnn_benchmark': False, 'cudnn_deterministic': True} and
                row['preserve_throwaway_encoder_constructor_rng'] is True, 'ledger policy fields')
        require((row['qualification_seed'], row['backward_passes'], row['optimizer_step_calls']) ==
                (QUALIFICATION_SEED, 0, 0) and row['benchmark_data_access'] is False and
                row['scientific_training'] is False and row['auxiliary_encoder_trained'] is False and
                row['experiment_seeds_used'] == [] and row['session_access_counts'] == SESSION, 'ledger scope/access')
        for path, h in row['artifacts_sha256'].items():
            require(sha(read(path)) == h, 'ledger artifact ' + path)
    expected, count = expected_index()
    if stage == 'final':
        require(read(INDEX) == expected, 'CRLF artifact index')
        worktree(True)
    require('torch' not in sys.modules and 'yaml' not in sys.modules, 'static preflight: no torch, no yaml')
    return {'status': 'PASS', 'stage': stage, 'ledger_rows': len(current.splitlines()),
            f'first_{LEDGER_PREFIX_ROWS}_rows_byte_identical': True,
            'artifact_index_rows_before': INDEX_BASELINE_ROWS, 'artifact_index_rows_expected': count,
            'processes_byte_identical': True, 'process_sha256': PROCESS_SHA, 'a7_overlay_sha256': A7_OVERLAY_SHA,
            'session_access_counts': SESSION, 'torch_imported': False}


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
