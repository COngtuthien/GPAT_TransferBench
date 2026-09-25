#!/usr/bin/env python3
"""Verify M6D6c E07c whole-module checkpoint serialization/loader evidence; append-check the ledger; rebuild the index LAST.

STATIC: no Torch import, no CUDA, no model construction, no checkpoint I/O and
no deserialization (the qualification checkpoint no longer exists; nothing here
unpickles anything). Only explicit allowlisted paths are read. Historical index
rows are carried from the authoritative commit without opening their targets
(tools/build_artifact_index.py is NOT used: it hashes every file); only the
explicit new M6D6c artifacts are hashed.
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
AUTHORITY = 'b89b8d0e0ac5f12fe43a7cfb09b02cda255f4661'
SPEC_SHA = 'f7d2371682ad9504f59f0c6984e4b1a741f855f1b1f60838f9d60b160281489e'
PIN = '23f40519ec25a833ebc06842aa6fbab74fad4d15'
TREE = 'd190a5fb4a94cb9e423215f9a24a7863aa17ea65'
CUSTOM_RN_SHA = 'fa788c4d2453b40585b295a8292eaf1afce7a0c622eb8ecc2adec5453a9a64c3'
PRETRAIN_SHA = '3444691f6f2c59b41a07f06db9388b134e849bb889c5f6b03489d47c7e037b16'
CONSUMER_SHA = '127ecd59fbbbd0d191a713aae62b11ea3de7058d42c2889100ff6a44199d2963'
A6_OVERLAY_SHA = 'dd3f29aa8ff96de9c0e2d504e6d07f4788d8fb8d030ffa26ef51443104ca971b'
LOCK_SHA = '0c909de1e3e3e8c129e0d9f4aab6386cee8e4eac0ca45d79423f795cced5f450'
M6D6A_HELPERS_SHA = '1eeb7c037dbb25876eb6257a4eb0dee40bc3a998d5d20c03a2b4b937abe12541'
FROZEN_FORMAT = 'torch.save of the WHOLE nn.Module (matches torch.load(path).cuda())'
QUALIFICATION_SEED = 60603
CHECKPOINT_SHA = 'cf89680a56cdb257e9739fb0ded55522631a9f8291719d972a05c1168460522b'
CHECKPOINT_BYTES = 185141005
CHECKPOINT_PATH = '<runtime_root>/builds/e07c_difffas/m6d6c/checkpoint/e07c_aux_encoder_QUALIFICATION_ONLY.pkl'
INDEX = 'outputs/audit/ARTIFACT_INDEX.csv'
LEDGER = 'outputs/audit/EXECUTION_LEDGER.jsonl'
LEDGER_PREFIX_ROWS = 117
LEDGER_PREFIX_SHA = '41fa3db788872cc49a140726bf2615f594e9d2e30b406330697d127aa82f3d32'
INDEX_BASELINE_ROWS = 664
CLASSIFICATION = 'M6D6C_E07C_AUX_CHECKPOINT_COMPATIBILITY_QUALIFICATION'
BASE = 'outputs/audit/M6D6C_E07C_AUX_CHECKPOINT_COMPATIBILITY'
WRITER = 'outputs/audit/M6D6C_E07C_AUX_CHECKPOINT_WRITER.json'
LOADERS = tuple(f'outputs/audit/M6D6C_E07C_AUX_CHECKPOINT_LOADER_PROCESS_{i}.json' for i in (1, 2))
LOG = 'outputs/audit/M6D6C_E07C_AUX_CHECKPOINT_RUNTIME_LOG.txt'
HARNESS = 'methods/difffas/aux_checkpoint_qualification.py'
SEAM = 'methods/difffas/aux_checkpoint.py'
TESTS = 'tests/test_m6d6c_e07c_aux_checkpoint.py'
PREFLIGHT = 'tools/m6d6c_e07c_aux_checkpoint_preflight.py'
ARTIFACTS = tuple(sorted((WRITER, *LOADERS, LOG, HARNESS, SEAM, TESTS, PREFLIGHT, BASE + '.md', BASE + '.json')))
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
             'methods/common/learned.py', 'methods/common/upstream.py', 'methods/common/config.py',
             'environments/e07c.lock.json', 'environments/e07c.runtime.json',
             'environments/e07c.conda-explicit.txt', 'environments/e07c.pip-freeze.txt',
             'environments/e07c.pip-requirements.txt',
             'outputs/audit/M6D6A_R_E07C_CLEAN_REQUALIFICATION.json',
             'outputs/audit/M6D6B_E07C_AUX_TRAINING_GRAPH_QUALIFICATION.md',
             'outputs/audit/M6D6B_E07C_AUX_TRAINING_GRAPH_QUALIFICATION.json',
             'outputs/audit/M6D6B_E07C_AUX_TRAINING_PROCESS_1.json', 'outputs/audit/M6D6B_E07C_AUX_TRAINING_PROCESS_2.json',
             'outputs/audit/M6D6B_E07C_AUX_TRAINING_RUNTIME_LOG.txt',
             'tests/test_m6d6b_e07c_aux_training_graph.py', 'tests/test_m6d6a_e07c_runtime.py',
             'tests/test_m6c2b3_contract.py', 'tests/test_m6c2b3_encoder.py', 'tests/test_m6c2b3_runtime.py',
             'tools/m6d6b_e07c_aux_training_graph_preflight.py', 'tools/build_artifact_index.py')
QUALIFIED = ['E07c_AUX_CHECKPOINT_WHOLE_MODULE_SERIALIZATION_QUALIFIED', 'E07c_AUX_CHECKPOINT_LOADER_COMPATIBILITY_QUALIFIED',
             'E07c_AUX_CHECKPOINT_SHA_BEFORE_DESERIALIZE_QUALIFIED', 'IMPLEMENTED_NOT_EXECUTED', 'CONTROLLED_ADAPTATION']
RETAINED = ['E07c_EXECUTION_ENVIRONMENT_QUALIFIED', 'E07c_CONDITIONING_ENCODER_RUNTIME_QUALIFIED',
            'E07c_MAIN_ARCHITECTURE_RUNTIME_QUALIFIED', 'E07c_SYNTHETIC_FORWARD_RUNTIME_QUALIFIED',
            'E07c_AUX_ENCODER_TRAINING_GRAPH_QUALIFIED', 'E07c_AUX_ENCODER_OPTIMIZER_STEP_QUALIFIED']
NOT_QUALIFIED = ['REAL_TRAIN_PATH', 'AUXILIARY_ENCODER_200_EPOCH_TRAINING', 'AUX_B256_TRAINING_MEMORY',
                 'AUX_PRODUCTION_RUNNER', 'PRODUCTION_PRECISION_POLICY', 'MAIN_DIFFFAS_TRAINING_GRAPH',
                 'MAIN_CHECKPOINT_RESUME', 'MAIN_RUNNER_ENCODER_LOAD_INTEGRATION', 'SCIENTIFIC_TRAINING', 'M8_BANK']
ELIGIBILITY = {'QUALIFICATION_ONLY': True, 'NOT_ELIGIBLE_FOR_BANK': True, 'NOT_ELIGIBLE_FOR_DOWNSTREAM': True,
               'NOT_ELIGIBLE_FOR_REPORTING': True, 'NOT_A_SCIENTIFIC_CHECKPOINT': True}
SCOPE = {'backward_calls': 0, 'optimizer_step_calls': 0, 'epochs': 0, 'auxiliary_encoder_trained': False,
         'scientific_training': False, 'scientific_checkpoint_created': False, 'scientific_checkpoint_loaded': False,
         'main_difffas_model_constructed': False, 'main_difffas_training': False, 'benchmark_data_access': False,
         'TRAIN_access': False, 'VAL_access': False, 'TEST_access': False, 'synthetic_bank': False,
         'scientific_seed_consumed': False, 'source_patch': 'NONE', 'fidelity': 'CONTROLLED_ADAPTATION', **ELIGIBILITY}
WRITER_COUNTERS = {'optimizer_constructions': 0, 'optimizer_step_calls': 0, 'backward_calls': 0,
                   'autograd_grad_calls': 0, 'torch_save_calls': 1, 'torch_load_calls': 0}
LOADER_COUNTERS = {**WRITER_COUNTERS, 'torch_save_calls': 0, 'torch_load_calls': 5}
SESSION = {'benchmark_manifest_reads': 0, 'benchmark_image_reads': 0, 'TRAIN_reads': 0, 'VAL_reads': 0,
           'TEST_reads': 0, 'firewall_denials': 0, 'backward_calls': 0, 'optimizer_step_calls': 0,
           'qualification_torch_save_calls': 2, 'qualification_torch_load_calls': 10,
           'scientific_checkpoints_created': 0, 'scientific_checkpoint_loads': 0,
           'auxiliary_encoder_training_runs': 0, 'scientific_seed_runs_completed': 0}
WEIGHTS_ONLY = 'WEIGHTS_ONLY_UNPICKLER_REJECTS_CUSTOM_MODULE_GLOBAL'
OUTPUT_SHAPES = {'x32x32': [4, 256, 32, 32], 'x16x16': [4, 512, 16, 16], 'x8x8': [4, 512, 8, 8], 'logits': [4, 7]}
LAUNCH_ONLY = ('PYTHONHASHSEED', 'TMPDIR')
FORBIDDEN_WORDING = ('faithful difffas', 'native difffas', 'official reproduction of', 'auxiliary_encoder_trained:',
                     'auxiliary encoder trained', 'aux_production_runner_qualified', 'real_train_path_qualified',
                     'aux_b256_training_memory_qualified', 'main_difffas_training_graph_qualified',
                     'scientifically_trained', 'm8_ready', 'state_dict substitution adopted')


def require(ok, message):
    if not ok:
        raise ValueError('M6D6c: ' + message)


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
    for rel, digest in (('models/custom_rn.py', CUSTOM_RN_SHA), ('models/pretrain_classifier.py', PRETRAIN_SHA),
                        ('models/unet_autoenc.py', CONSUMER_SHA)):
        require(sha((root / rel).read_bytes()) == digest, rel + ' SHA256')
    return root


def upstream_ast(root):
    """Stdlib re-derivation of the pinned writer/consumer serialization calls (independent of the harness)."""
    writer = ast.parse((root / 'models/pretrain_classifier.py').read_bytes())
    saves = [n for n in ast.walk(writer) if isinstance(n, ast.Call) and ast.unparse(n.func) == 'torch.save']
    consumer = ast.parse((root / 'models/unet_autoenc.py').read_bytes())
    cls = next(n for n in consumer.body if isinstance(n, ast.ClassDef) and n.name == 'BeatGANsAutoencModel')
    enc = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'encoder')
    loads = [n for n in ast.walk(enc) if isinstance(n, ast.Call) and ast.unparse(n.func) == 'torch.load']
    return {'saves': [(ast.unparse(n), n.lineno) for n in saves],
            'loads': [(ast.unparse(n), n.lineno, [k.arg for k in n.keywords]) for n in loads],
            'encoder_body': [ast.unparse(s) for s in enc.body]}


def seam_ast():
    tree = ast.parse(read(SEAM))
    compat = next(n for n in tree.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'LOAD_COMPATIBILITY')
    value = ast.literal_eval(compat.value)
    loads = [ast.unparse(n) for n in ast.walk(tree) if isinstance(n, ast.Call) and ast.unparse(n.func) == 'torch.load']
    saves = [ast.unparse(n) for n in ast.walk(tree) if isinstance(n, ast.Call) and ast.unparse(n.func) == 'torch.save']
    torch_assign = [ast.unparse(t) for n in ast.walk(tree) if isinstance(n, ast.Assign) for t in n.targets
                    if ast.unparse(t).startswith('torch.')]
    names = {n.attr if isinstance(n, ast.Attribute) else n.id for n in ast.walk(tree)
             if isinstance(n, (ast.Attribute, ast.Name))}
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'load_verified_whole_module')
    order = [ast.unparse(s) for s in fn.body]
    frozen = next(n for n in tree.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'FROZEN_FORMAT')
    return {'compat': value, 'loads': loads, 'saves': saves, 'torch_assign': torch_assign,
            'forbidden_names': sorted({'state_dict', 'load_state_dict', 'safetensors', 'jit', 'onnx',
                                       'add_safe_globals', 'safe_globals'} & names),
            'verify_before_import': order.index('raw = read_verified(path, expected_sha256)') < order.index('import torch'),
            'frozen_format': ast.literal_eval(frozen.value)}


def check_common(r, lock):
    require(r['status'] == 'PASS' and r['label'] == 'AUX_CHECKPOINT_SERIALIZATION_QUALIFICATION_ONLY', 'process pass')
    require(r['qualification_seed'] == QUALIFICATION_SEED and r['experiment_seed'] is None and
            r['auxiliary_encoder_training_seed'] is None and QUALIFICATION_SEED not in (42, 1337, 2026), 'seed')
    require(all(r[k] == v for k, v in SCOPE.items()), 'scope flags')
    require(r['checkpoint_path'] == CHECKPOINT_PATH and '/runs/' not in r['checkpoint_path'], 'qualification path')
    require(r['checkpoint_format_authority']['value'] == FROZEN_FORMAT, 'frozen format')
    s = r['source_before']
    require(s == r['source_after'] and s['worktree_status'] == '', 'source mutation')
    require((s['commit'], s['tree'], s['a6_overlay_sha256']) == (PIN, TREE, A6_OVERLAY_SHA), 'pinned source / A6')
    env = r['environment_before']
    require(env == r['environment_after'], 'environment mutation')
    require({k: v for k, v in env.items() if k != 'launch_environment'} ==
            {k: v for k, v in lock['identity'].items() if k != 'launch_environment'}, 'executed identity = lock')
    strip = lambda d: {k: v for k, v in d.items() if k not in LAUNCH_ONLY}
    require(strip(env['launch_environment']) == strip(lock['identity']['launch_environment']) and
            env['launch_environment']['PYTHONHASHSEED'] == str(QUALIFICATION_SEED), 'launch env')
    require('weights_only: bool | None = None' in r['torch_load_signature'] and
            r['torch_load_weights_only_default'] == 'None' and
            set(r['torch_load_env_overrides'].values()) == {None}, 'genuine torch.load signature recorded')
    p = r['precision']
    require(p['label'] == 'ENGINEERING_QUALIFICATION_CONTROLS' and not p['freezes_scientific_precision_policy'] and
            p['production_policy'] == 'PRECISION_POLICY_DECISION_REQUIRED', 'precision not decided')
    fw = r['firewall']
    require(not fw['denied'] and not any(fw['attempts'].values()), 'firewall zero denials')
    require(all(a[0] in ('git', 'nvidia-smi') or a == ['/sbin/ldconfig', '-p'] for a in fw['subprocesses']),
            'subprocess allowlist')
    require(r['warnings'] == [], 'no runtime warnings')


def check_writer(w, lock):
    check_common(w, lock)
    require(w['role'] == 'writer' and w['counters'] == WRITER_COUNTERS, 'writer counters')
    c = w['checkpoint']
    require((c['sha256'], c['size_bytes'], c['serialization_api'], c['format']) ==
            (CHECKPOINT_SHA, CHECKPOINT_BYTES, 'torch.save(model, path)', FROZEN_FORMAT) and
            all(c[k] is True for k in ELIGIBILITY), 'checkpoint record')
    a = c['archive']
    require({'custom_rn.ResNet', 'custom_rn.BasicBlock'} <= set(a['globals']) and
            not [g for g in a['globals'] if g.startswith('torchvision')] and a['pickle_protocol'] == 2,
            'whole-module pickle globals (opcode listing)')
    wr = w['writer']
    require(wr['identity'] == {'block': 'BasicBlock', 'class': 'ResNet', 'fc': {'bias': True, 'in_features': 512,
            'out_features': 7}, 'module': 'custom_rn', 'module_file_sha256': CUSTOM_RN_SHA, 'topology': [3, 4, 6, 3]},
            'writer identity')
    st = wr['state_before_save']
    require(st['parameters'] == 46233707 and st['parameter_tensors'] == 112 and st['buffers'] == 111 and
            st['devices'] == ['cuda:0'] and st['dtypes'] == ['torch.float32'], 'writer state')
    require([e for e in wr['save_trace'] if e[0] == 'torch.save'] ==
            [['torch.save', {'extra_arguments': False, 'object': 'custom_rn.ResNet'}]], 'one plain torch.save')
    require({k: v['shape'] for k, v in w['reference_forward']['outputs'].items()} == OUTPUT_SHAPES, 'reference shapes')
    # torch.save writes through its C++ archive writer (no Python open); the two reads are SHA256 and opcode listing.
    require(w['firewall']['checkpoint_opens'] == {'read': 2, 'write': 0}, 'writer checkpoint opens')


def check_loader(r, w, lock):
    check_common(r, lock)
    require(r['role'] == 'loader' and r['counters'] == LOADER_COUNTERS, 'loader counters')
    require(r['checkpoint_consumed']['sha256'] == CHECKPOINT_SHA and r['firewall']['checkpoint_opens']['write'] == 0,
            'consumed SHA / read-only')
    rej = r['rejection_before_deserialization']
    require(rej['wrong_sha256_one_hex_digit']['events'] == ['open_checkpoint', 'sha256_rejected'] and
            rej['wrong_sha256_one_hex_digit']['torch_load_calls'] == 0 and
            rej['wrong_sha256_one_hex_digit']['find_class_events'] == 0 and
            rej['malformed_sha256']['events'] == [], 'wrong SHA rejected before torch.load')
    p = r['probes']
    require(p['default_no_argument']['call'] == 'torch.load(path)', 'upstream call form probed')
    for name in ('default_no_argument', 'weights_only_true'):
        require(p[name]['outcome'] == 'REJECTED' and p[name]['category'] == WEIGHTS_ONLY and
                p[name]['exception_class'] == '_pickle.UnpicklingError' and p[name]['find_class_globals'] == [],
                name + ' observed rejection')
    f = p['weights_only_false']
    st = w['writer']['state_before_save']
    require(f['outcome'] == 'LOADED' and f['type'] == 'custom_rn.ResNet' and f['is_pinned_custom_rn_ResNet'] and
            f['aggregate_parameter_sha256'] == st['aggregate_parameter_sha256'] and
            f['aggregate_buffer_sha256'] == st['aggregate_buffer_sha256'], 'weights_only=False restores')
    m = p['weights_only_false_without_pinned_module']
    require(m['outcome'] == 'REJECTED' and m['category'] == 'PINNED_CUSTOM_RN_MODULE_NOT_IMPORTABLE', 'module identity')
    require(all(v['sha256_verified_before_call'] for v in p.values()), 'SHA before every probe')
    d = r['compatibility_decision']
    require(d['observed'] == 'DEFAULT_AND_WEIGHTS_ONLY_TRUE_REJECT_WEIGHTS_ONLY_FALSE_RESTORES' and
            d['selected'] == 'EXPLICIT_WEIGHTS_ONLY_FALSE_AFTER_SHA256_VERIFICATION' and
            not d['scientific_adaptation'] and not d['checkpoint_format_changed'] and not d['source_modified'] and
            not d['global_torch_load_patch'], 'decision derived from observation')
    s = r['seam_load']
    require(s['event_order'][:3] == ['open_checkpoint', 'sha256_verified', 'torch.load'] and
            s['event_order'][3].startswith('find_class x') and
            s['torch_load_call'] == {'kwargs': {'weights_only': 'False'}, 'source': 'bytes'}, 'SHA-before-deserialize order')
    require(s['identity'] == w['writer']['identity'] and s['class_is_pinned_module_class'] and s['module_tree_equal'] and
            s['custom_rn_live_module_file'] == 'third_party/source_cache/difffas/models/custom_rn.py' and
            not s['projection_or_adapter_modules_added'], 'loaded identity')
    require((s['parameter_tensors_bitwise_equal'], s['buffer_tensors_bitwise_equal']) == (112, 111) and
            (s['aggregate_parameter_sha256'], s['aggregate_buffer_sha256']) ==
            (st['aggregate_parameter_sha256'], st['aggregate_buffer_sha256']) and
            (s['parameter_devices'], s['buffer_devices']) == (['cuda:0'], ['cuda:0']), 'parameter/buffer round trip')
    require(set(s['find_class_globals']) <= set(w['checkpoint']['archive']['globals']), 'only archive globals')
    fw = r['round_trip_forward']
    require(fw['bitwise_equal'] == {k: True for k in OUTPUT_SHAPES} and fw['tolerance_used'] == 'NONE (bitwise)' and
            all(fw['outputs'][k]['sha256'] == w['reference_forward']['outputs'][k]['sha256'] and
                fw['outputs'][k]['shape'] == OUTPUT_SHAPES[k] for k in OUTPUT_SHAPES), 'bitwise forward round trip')


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
    require(status == sorted(expected), 'worktree holds exactly the intended M6D6c changes: ' + json.dumps(status))
    require(git('diff', '--cached', '--name-only').decode().strip() == '', 'nothing staged')
    return status


def verify(stage):
    """stage: 'before_ledger' (evidence only), 'before_index' (ledger appended), 'final'."""
    for ref in ('HEAD', 'origin/m6-baselines'):
        require(git('rev-parse', ref).decode().strip() == AUTHORITY, ref + ' authority')
    require(sha(read('docs/spec/GPAT_TransferBench_v1_0_Frozen_Specification_2026.docx')) == SPEC_SHA, 'spec SHA256')
    for path in PRESERVED:
        require(read(path) == git('show', AUTHORITY + ':' + path), 'immutable input ' + path)
    require(sha(read('methods/difffas/runtime_qualification.py')) == M6D6A_HELPERS_SHA, 'M6D6a helpers unchanged')
    root = source_cache()
    up = upstream_ast(root)
    require(up['saves'] == [("torch.save(resnet18, './PADISI.pkl')", 45)] and
            up['loads'] == [('torch.load(path)', 75, [])] and
            up['encoder_body'][1:3] == ['model_autoencoder = torch.load(path)',
                                        'model_autoencoder = model_autoencoder.cuda()'], 'pinned writer/consumer (AST)')
    s = seam_ast()
    require(s['compat']['weights_only'] is False and not s['compat']['scientific_adaptation'] and
            not s['compat']['checkpoint_format_changed'] and not s['compat']['global_torch_load_patch'] and
            s['loads'] == ["torch.load(io.BytesIO(raw), weights_only=LOAD_COMPATIBILITY['weights_only'])"] and
            s['saves'] == ['torch.save(model, str(path))'] and s['torch_assign'] == [] and
            s['forbidden_names'] == [] and s['verify_before_import'] and s['frozen_format'] == FROZEN_FORMAT,
            'seam: whole-module format, SHA before import/load, no global patch, no state_dict')
    lock_raw = read('environments/e07c.lock.json')
    require(sha(lock_raw) == LOCK_SHA, 'environment lock digest')
    lock = json.loads(lock_raw)
    w = json.loads(read(WRITER))
    check_writer(w, lock)
    raw_l = [read(p) for p in LOADERS]
    require(raw_l[0] == raw_l[1], 'two loader processes byte-identical')
    for r in raw_l:
        check_loader(json.loads(r), w, lock)
    for rel, h in w['source_before']['files_sha256'].items():
        require(sha((root / rel).read_bytes()) == h, 'laptop source closure ' + rel)
    audit = json.loads(read(BASE + '.json'))
    require(audit['classification'] == CLASSIFICATION and audit['milestone'] == 'M6D6c' and
            audit['final_status'] == 'PASS' and audit['qualified_statuses'] == QUALIFIED and
            audit['retained_statuses'] == RETAINED and audit['not_qualified'] == NOT_QUALIFIED and
            audit['method_status'] == 'IMPLEMENTED_NOT_EXECUTED', 'status fields')
    require(audit['session_access_audit']['counts'] == SESSION and
            audit['session_access_audit']['scope'] == 'COMPLETE_M6D6C_SESSION_LAPTOP_AND_GPU', 'session access audit')
    require(audit['precision_policy_audit']['result'] == 'PRECISION_POLICY_DECISION_REQUIRED' and
            audit['precision_policy_audit']['precision_value_added_to_config'] is False, 'precision audit')
    require(audit['checkpoint']['sha256'] == CHECKPOINT_SHA and audit['checkpoint']['size_bytes'] == CHECKPOINT_BYTES and
            audit['cleanup']['qualification_pkl'] == 'ABSENT' and audit['cleanup']['encoder_final_pkl'] == 'ABSENT',
            'checkpoint record and cleanup')
    require(audit['environment_lock_sha256'] == LOCK_SHA and
            not audit['environment_reverification']['environment_rebuilt_or_mutated'], 'environment re-verification')
    for path, h in audit['artifacts_sha256'].items():
        require(sha(read(path)) == h, 'artifact ' + path)
    require(sorted(audit['artifacts_sha256']) == sorted(set(ARTIFACTS) - {BASE + '.json'}), 'artifact list')
    for path, h in audit['preserved_inputs_sha256'].items():
        require(sha(read(path)) == h, 'preserved input ' + path)
    final_tests = [t for t in audit['tests'] if t['final']]
    require(len(final_tests) == 8 and all(t['failures'] == t['errors'] == 0 for t in final_tests), 'final test runs')
    require(all(t['audit_denied'] == t['audit_manifest'] == t['audit_image'] == t['audit_data_or_runs'] == 0
                for t in audit['tests']), 'test audit counts')
    log = read(LOG).decode()
    after = [ln for ln in log.splitlines() if ln.startswith('scientific_paths_after:')]
    require(after == ['scientific_paths_after: runs=ABSENT e07c_root=ABSENT aux_seed_42=ABSENT encoder_final=ABSENT'] * 4,
            'no scientific run root or auxiliary checkpoint')
    for token in ('QUALIFICATION_PKL_ABSENT', 'ENCODER_FINAL_ABSENT', 'RUNS_ABSENT', 'status_lines 0',
                  'source_cache_status_lines 0'):
        require(token in log, 'cleanup recorded: ' + token)
    report = read(BASE + '.md').decode().lower()
    for phrase in FORBIDDEN_WORDING:
        require(phrase not in report, 'forbidden wording: ' + phrase)
    for phrase in ('controlled_adaptation', 'implemented_not_executed', 'e07c_aux_checkpoint_loader_compatibility_qualified',
                   'precision_policy_decision_required', 'qualification_only', 'not_a_scientific_checkpoint',
                   'b256 scientific training memory remains unqualified'):
        require(phrase in report, 'required wording: ' + phrase)
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
        require((row['classification'], row['milestone'], row['method_id']) == (CLASSIFICATION, 'M6D6c', 'E07c'),
                'ledger row identity')
        require(row['final_status'] == 'PASS' and row['qualified_statuses'] == QUALIFIED and
                row['not_qualified'] == NOT_QUALIFIED and row['method_status'] == 'IMPLEMENTED_NOT_EXECUTED' and
                row['fidelity_class'] == 'CONTROLLED_ADAPTATION', 'ledger status fields')
        require((row['qualification_seed'], row['backward_passes'], row['optimizer_step_calls']) ==
                (QUALIFICATION_SEED, 0, 0) and row['benchmark_data_access'] is False and
                row['scientific_training'] is False and row['auxiliary_encoder_trained'] is False and
                row['scientific_checkpoint_created'] is False and row['experiment_seeds_used'] == [] and
                row['qualification_checkpoint']['label'] == 'QUALIFICATION_ONLY' and
                row['qualification_checkpoint']['sha256'] == CHECKPOINT_SHA and
                row['session_access_counts'] == SESSION, 'ledger scope/access')
        for path, h in row['artifacts_sha256'].items():
            require(sha(read(path)) == h, 'ledger artifact ' + path)
    expected, count = expected_index()
    if stage == 'final':
        require(read(INDEX) == expected, 'CRLF artifact index')
        worktree(True)
    require('torch' not in sys.modules and 'pickle' not in sys.modules, 'static preflight: no torch, no unpickler')
    return {'status': 'PASS', 'stage': stage, 'ledger_rows': len(current.splitlines()),
            f'first_{LEDGER_PREFIX_ROWS}_rows_byte_identical': True,
            'artifact_index_rows_before': INDEX_BASELINE_ROWS, 'artifact_index_rows_expected': count,
            'loader_processes_byte_identical': True, 'checkpoint_sha256': CHECKPOINT_SHA,
            'session_access_counts': SESSION, 'torch_imported': False, 'deserialization_in_preflight': False}


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
