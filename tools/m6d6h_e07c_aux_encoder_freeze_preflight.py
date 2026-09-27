#!/usr/bin/env python3
"""Verify the M6D6h E07c auxiliary-encoder OWNER FREEZE; append-check the ledger; rebuild the index LAST.

STATIC: no Torch, no YAML parser, no pyarrow, no numpy, no PIL, no CUDA, no model, no Parquet decode, no image
read, and never opens or deserializes encoder_final.pkl (or any .pkl/.pt/.pth/.ckpt): the frozen size and SHA256
are validated as recorded evidence. Nothing is trained. Historical files are compared byte-for-byte with the
authority commit through Git. The freeze record is the JSON subset of YAML. Optional --runtime-run-dir (GPU host
only) re-hashes the small runtime JSON/log files and stats (never opens) the checkpoint, read-only.
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
AUTHORITY = 'ee0a9b9166577cac606c142af5bc2e338d75eea1'
M6D6F_COMMIT = '7642b23e60a1d89fcd481b03d4c9cb361f4c6e20'
SPEC_SHA = 'f7d2371682ad9504f59f0c6984e4b1a741f855f1b1f60838f9d60b160281489e'
PIN = '23f40519ec25a833ebc06842aa6fbab74fad4d15'
TREE = 'd190a5fb4a94cb9e423215f9a24a7863aa17ea65'
FROZEN_SHA = '49a24a3a7288782c144eaf4f58e83c76c37c1feeceea6de8a04b96a8fcfe107c'
FROZEN_BYTES = 185136819
MAIN_SEEDS = [42, 1337, 2026]
RUN_ID = '7014cdb366da52e4'
RUN_UUID = '7031fd2f-92d7-465a-94d1-ab99bf8a3960'
RUN_DIR = '/home/student20261/workdir/GPAT_TransferBench_runtime/runs/m6/E07c/aux_encoder/seed_42'
PATH_TEMPLATE = '<runtime_root>/runs/m6/E07c/aux_encoder/seed_42/checkpoints/encoder_final.pkl'
OBSERVED_GPU_PATH = RUN_DIR + '/checkpoints/encoder_final.pkl'
FORMAT = 'torch.save of the WHOLE nn.Module (matches torch.load(path).cuda())'
PENDING = 'SHA256_RECORDED_PENDING_OWNER_FREEZE'
LOADER = 'methods.difffas.aux_checkpoint.load_frozen_aux_encoder'
CLASSIFICATION = 'M6D6H_E07C_AUX_ENCODER_OWNER_FREEZE'
INDEX = 'outputs/audit/ARTIFACT_INDEX.csv'
LEDGER = 'outputs/audit/EXECUTION_LEDGER.jsonl'
LEDGER_PREFIX_ROWS = 122
INDEX_BASELINE_ROWS = 719

OVERLAY = 'configs/amendments/e07c_m6d6h_aux_encoder_freeze.yaml'
RECORD_DOC = 'docs/spec/amendments/GPAT_TransferBench_v1_0_E07c_Aux_Encoder_Freeze_Record_M6D6h.md'
SEAM = 'methods/difffas/aux_checkpoint.py'
B1_TEST = 'tests/test_m6d6g_e07c_aux_scientific_run.py'
B1_OLD_SHA = 'a292d3bd8ad1185f0da8697f7c876372985aae4168521278fb2850d7b77d5bcf'
B1_NEW_SHA = 'f84b36d914ac918d912472c50b5d781a37a96de754ab13982af34619fa7905ca'
B1_REASON = 'PROSPECTIVE_REGRESSION_HARNESS_SCOPE_CORRECTION'
BASE = 'outputs/audit/M6D6H_E07C_AUX_ENCODER_FREEZE'
EV_JSON, EV_MD, LOG = BASE + '.json', BASE + '.md', BASE + '_RUNTIME_LOG.txt'
TESTS, PREFLIGHT = 'tests/test_m6d6h_e07c_aux_encoder_freeze.py', 'tools/m6d6h_e07c_aux_encoder_freeze_preflight.py'
NEW = tuple(sorted((OVERLAY, RECORD_DOC, EV_JSON, EV_MD, LOG, TESTS, PREFLIGHT)))
MODIFIED = tuple(sorted((SEAM, B1_TEST)))
# Bindings the freeze record must carry (and that must equal the files and the authority commit).
BINDINGS = {
    'docs/spec/GPAT_TransferBench_v1_0_Frozen_Specification_2026.docx': SPEC_SHA,
    'configs/methods/e07c_difffas_bin_idfree.yaml': 'dba34a9222b80ed66a73b8662c10cd348ab46e4f00c2e8166d0a4fe6d552bc1c',
    'configs/frozen/difffas_bin_idfree_v1.yaml': 'aa9e984166db3854bba4221098afaef1898474e2e3f1f08a3f80cab0035cf3eb',
    'docs/spec/amendments/GPAT_TransferBench_v1_0_Amendment_A3_Controlled_Reconstruction_E04_E07c.md':
        'b12451537bcc3bc14e96e5bcd2ce390b5fc0c8a4b5c60bff7f40a2333665a67a',
    'docs/spec/amendments/GPAT_TransferBench_v1_0_Amendment_A6_E07c_Feature_Interface_Source_Correction.md':
        '759f72d860ccf49927c214bb4ce84e87bcedd98eb8f06ef5fe7361a758d6bc81',
    'configs/amendments/e07c_a6_feature_interface_source_correction.yaml':
        'dd3f29aa8ff96de9c0e2d504e6d07f4788d8fb8d030ffa26ef51443104ca971b',
    'docs/spec/amendments/GPAT_TransferBench_v1_0_Amendment_A7_E07c_Execution_Policy.md':
        '0a46c3b06e277ad38b17a6e85fe2924441d8d13b1de887bf54903dda9aa292ba',
    'configs/amendments/e07c_a7_execution_policy.yaml': '3e4c758c1421aac6e993724a8a80b99e8b0754e75b983cec5ffca41479f492a3',
    'docs/spec/amendments/GPAT_TransferBench_v1_0_Amendment_A8_E07c_Aux_Resume_Policy.md':
        'deb11b5cd80fa873bde1ef9e87160fed3a8065903cd8c81c9463b087fbf73b2f',
    'configs/amendments/e07c_a8_aux_resume_policy.yaml': 'a098ce509156151ee6ce7ead1e9afa5711264b676a597e2514b1058ed46f471a',
    'environments/e07c.lock.json': '0c909de1e3e3e8c129e0d9f4aab6386cee8e4eac0ca45d79423f795cced5f450',
    'outputs/audit/M6D6G_E07C_AUX_FINAL_CHECKPOINT.json': '5e79da92f165f8af472f8e32f14f52100d0263378e9b461aec3d37976ba73498',
    'outputs/audit/M6D6G_E07C_AUX_SCIENTIFIC_RUN.json': '4dc547fd4e31ab929a9c1c44c33c7781b8cc2da5a0cde8aa736e8539ceca581f'}
# Historical files that M6D6h must keep byte-identical (in addition to every M6D6c..M6D6g evidence file).
PRESERVED = (*BINDINGS,
             'docs/spec/amendments/GPAT_TransferBench_v1_0_Amendment_A1_Fair_IDFree_Main_Track.md',
             'docs/spec/amendments/GPAT_TransferBench_v1_0_Amendment_A2_M6_Baseline_Execution_Contracts.md',
             'docs/spec/amendments/GPAT_TransferBench_v1_0_Amendment_A4_E04_Execution_Gap_Resolution.md',
             'docs/spec/amendments/GPAT_TransferBench_v1_0_Amendment_A5_E05_Blur_Operator_Resolution.md',
             'docs/spec/amendments/GPAT_TransferBench_v1_0_E07c_Aux_Production_Runner_Addendum_M6D6e.md',
             'configs/amendments/e07c_m6d6e_aux_production_runner_contract.yaml',
             'configs/run_logging_v1.yaml', 'methods/difffas/encoder.py', 'methods/difffas/execution_policy.py',
             'methods/difffas/aux_runner.py', 'methods/difffas/aux_runner_io.py', 'methods/difffas/aux_resume.py',
             'methods/difffas/aux_checkpoint_qualification.py', 'tools/run_e07c_aux.py',
             'tools/m6d6g_e07c_aux_scientific_preflight.py')
PRESERVED_TREES = ('configs/frozen', 'configs/methods', 'frozen_config_snapshot', 'manifests', 'environments',
                   'third_party', 'gpatbench')
DATA_INPUTS = ('manifests/split_v1.parquet', 'manifests/artifact_probe_classes_v1.json',
               'manifests/difffas_bin_idfree_train_v1.parquet')
RUNTIME_SMALL = ('resolved_config.yaml', 'run_manifest.json', 'metrics.jsonl', 'checkpoint_index.json',
                 'resume_state_index.json', 'run_summary.json', 'stdout.log', 'stderr.log')
UNCHANGED_SEAM_FUNCTIONS = ('_config', '_external', 'encoder_identity', 'save_whole_module', 'read_verified',
                            'load_verified_whole_module')
TORCH_LOAD = "torch.load(io.BytesIO(raw), weights_only=LOAD_COMPATIBILITY['weights_only'])"
FORBIDDEN_SEAM_NAMES = {'state_dict', 'load_state_dict', 'safetensors', 'jit', 'onnx', 'save_file', 'add_safe_globals',
                        'safe_globals', 'environ', 'getenv', 'argv', 'glob', 'rglob', 'iterdir', 'listdir', 'argparse'}
QUALIFIED = ['E07c_AUXILIARY_ENCODER_SHA_FROZEN_FOR_MAIN', 'AUXILIARY_ENCODER_SHA_FROZEN_FOR_MAIN']
NOT_QUALIFIED = ['MAIN_DIFFFAS_TRAINING_GRAPH', 'MAIN_CHECKPOINT_RESUME', 'MAIN_RUNNER_ENCODER_LOAD_INTEGRATION',
                 'MAIN_PRODUCTION_RUNNER', 'MAIN_DIFFFAS_SCIENTIFIC_TRAINING', 'M8_BANK']
ZERO = {'training_runs': 0, 'scientific_auxiliary_runs': 0, 'main_difffas_scientific_runs': 0, 'optimizer_steps': 0,
        'backward_calls': 0, 'TRAIN_image_reads': 0, 'VAL_image_reads': 0, 'TEST_image_reads': 0,
        'checkpoint_deserializations': 0, 'M8_outputs': 0}


def require(ok, message):
    if not ok:
        raise ValueError('M6D6h: ' + message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def git(*args, root=ROOT):
    return subprocess.check_output(['git', '-C', str(root), *args])


def read(path):
    p = Path(path)
    require(not p.is_absolute() and '..' not in p.parts, 'relative evidence path')
    require(not {'data', 'faces_256', 'runs', 'cache'} & set(p.parts), 'data firewall ' + path)
    require(p.suffix not in {'.pkl', '.pt', '.pth', '.ckpt', '.parquet', '.png', '.jpg'}, 'no weight/manifest/image read')
    return (ROOT / p).read_bytes()


def load(path):
    return json.loads(read(path))


def at_authority(rel):
    return git('show', f'{AUTHORITY}:{rel}')


# ----------------------------------------------------------------- authority / immutability
def authority():
    for ref in ('HEAD', 'origin/m6-baselines'):
        require(git('rev-parse', ref).decode().strip() == AUTHORITY, ref + ' authority')
    require(git('branch', '--show-current').decode().strip() == 'm6-baselines', 'branch')


def historical_evidence():
    """Every committed M6D6c..M6D6g evidence file (outputs/audit) byte-identical to AUTHORITY."""
    paths = [p for p in git('ls-tree', '-r', '--name-only', AUTHORITY, 'outputs/audit').decode().splitlines()
             if Path(p).name.startswith(('M6D6C_', 'M6D6D_', 'M6D6E_', 'M6D6F_', 'M6D6G_'))]
    require(len([p for p in paths if Path(p).name.startswith('M6D6G_')]) == 4, 'four M6D6g evidence files')
    for rel in paths:
        require(read(rel) == at_authority(rel), 'historical evidence byte-identical ' + rel)
    return paths


def frozen_files_unchanged():
    for rel in PRESERVED:
        require(read(rel) == at_authority(rel), 'byte-identical to authority ' + rel)
    for rel, digest in BINDINGS.items():
        require(sha(read(rel)) == digest, 'bound SHA256 ' + rel)
    require(git('diff', '--name-only', AUTHORITY, '--', *PRESERVED_TREES).decode().strip() == '' and
            git('status', '--porcelain', '--untracked-files=all', '--', *PRESERVED_TREES).decode().strip() == '',
            'frozen config/snapshot/manifest/environment/source trees untouched')
    amendments = [p for p in git('ls-tree', '-r', '--name-only', AUTHORITY, 'docs/spec/amendments', 'configs/amendments')
                  .decode().splitlines()]
    for rel in amendments:     # A1..A8 documents/overlays and every earlier addendum/contract
        require(read(rel) == at_authority(rel), 'amendment/addendum byte-identical ' + rel)
    require(not any('Amendment_A9' in p or '_a9_' in p for p in
                    git('ls-files', '--others', '--exclude-standard', 'docs', 'configs').decode().splitlines()), 'no A9')
    for rel in DATA_INPUTS:     # Git blob identity; never opened
        require(git('rev-parse', f'HEAD:{rel}') == git('rev-parse', f'{AUTHORITY}:{rel}'), 'data input blob ' + rel)
    require(git('ls-files', '*.pkl', '*.pt', '*.pth', '*.ckpt', '*.safetensors').decode().strip() == '',
            'no tracked weight file')
    src = ROOT / 'third_party/source_cache/difffas'
    if (src / '.git').exists():
        require(git('rev-parse', 'HEAD', root=src).decode().strip() == PIN, 'pinned source commit')
        require(git('rev-parse', 'HEAD^{tree}', root=src).decode().strip() == TREE, 'pinned source tree')


# ----------------------------------------------------------------- freeze record
def freeze_record():
    raw = read(OVERLAY)
    record = json.loads(raw)
    require(sha(raw) == seam_constants()['FREEZE_RECORD_SHA256'], 'freeze record SHA256 == pinned seam constant')
    require((record['milestone'], record['method_id'], record['role'], record['status'], record['classification'],
             record['record_kind'], record['amendment_created'], record['fidelity_class'], record['deviation'],
             record['new_deviation'], record['new_fidelity_class'], record['new_scientific_rule'],
             record['historical_configs_rewritten'], record['authority_commit']) ==
            ('M6D6h', 'E07c', 'AUXILIARY_CONDITIONING_ENCODER', ['OWNER_FROZEN', 'ADDITIVE', 'NON_DESTRUCTIVE'],
             'DETERMINISTIC_IMPLEMENTATION_CLARIFICATION', 'OWNER_DECISION / ASSET_FREEZE', False,
             'CONTROLLED_ADAPTATION', 'DEV-021', False, False, False, False, AUTHORITY), 'freeze record identity')
    require(record['fidelity'] == {'fidelity_class': 'CONTROLLED_ADAPTATION', 'deviation': 'DEV-021',
                                   'new_deviation': False, 'new_fidelity_class': False}, 'fidelity block')
    require(record['record_document'] == {'path': RECORD_DOC, 'sha256': sha(read(RECORD_DOC))}, 'record document binding')
    require(record['bound_authority_sha256'] == BINDINGS, 'bound authorities (config/A3/A6/A7/A8/lock/M6D6g)')
    require(record['source'] == {'commit': PIN, 'tree': TREE}, 'source pin')
    a = record['frozen_asset']
    require((a['sha256'], a['bytes'], a['format'], a['path_template'], a['observed_gpu_path'], a['weight_bytes_in_git']) ==
            (FROZEN_SHA, FROZEN_BYTES, FORMAT, PATH_TEMPLATE, OBSERVED_GPU_PATH, False), 'frozen asset')
    require(a['observed_gpu_path_role'].startswith('PROVENANCE_ONLY'), 'absolute GPU path is provenance only')
    require(f'external_runtime_path: "{PATH_TEMPLATE}"' in read('configs/methods/e07c_difffas_bin_idfree.yaml').decode(),
            'path template == configured external asset path')
    p = record['producer']
    require((p['milestone'], p['auxiliary_seed'], p['logical_run'], p['run_id'], p['run_uuid'], p['completed_epoch'],
             p['global_step'], p['logical_optimizer_steps'], p['physical_optimizer_steps'], p['superseded_optimizer_steps'],
             p['interruptions'], p['resumes'], p['val_or_test_selection'], p['runtime_git_commit'], p['producer_commit'],
             p['historical_status'], p['historical_status_preserved']) ==
            ('M6D6g', 42, 'ONE_LOGICAL_RUN', RUN_ID, RUN_UUID, 200, 11200, 11200, 11200, 0, 0, 0, False,
             M6D6F_COMMIT, AUTHORITY, PENDING, True), 'producer')
    o = record['owner_freeze']
    require((o['owner_freeze_performed'], o['authoritative_for_main_difffas'], o['frozen_for_main_seeds'],
             o['all_main_seeds_share_one_checkpoint'], o['alternative_checkpoints_authorized'],
             record['owner_freeze_performed'], record['authoritative_for_main_difffas'], record['frozen_for_main_seeds']) ==
            (True, True, MAIN_SEEDS, True, 0, True, True, MAIN_SEEDS), 'owner freeze')
    c = record['consumption']
    require((c['secure_loader'], c['currently_qualified_consumer_scope'], c['M8_integration_qualified'], c['M8_BANK'],
             c['A7_M8_rng_policy_question'], c['recorded_sha256_must_equal_frozen']) ==
            (LOADER, 'MAIN_DIFFFAS', False, 'UNQUALIFIED', 'UNRESOLVED', True), 'consumption scope')
    require(set(record['failure_policy'].values()) == {'STOP_AND_REPORT', 'FORBIDDEN'} and
            record['failure_policy']['automatic_refreeze'] == record['failure_policy']['automatic_retraining'] == 'FORBIDDEN',
            'failure policy')
    require(record['file_protection']['chmod_performed'] is False, 'no chmod')
    f = record['freeze_operation']
    require((f['training_runs'], f['optimizer_steps'], f['backward_calls'], f['checkpoint_deserializations'],
             f['TRAIN_access'], f['VAL_access'], f['TEST_access'], f['M8_outputs'], f['main_difffas_scientific_runs']) ==
            (0, 0, 0, 0, False, False, False, 0, 0), 'freeze operation: no training / data / deserialization')
    doc = read(RECORD_DOC).decode()
    for token in (FROZEN_SHA, '185136819', 'OWNER_FROZEN', 'STOP_AND_REPORT', 'no automatic re-freeze',
                  'M8 remains unqualified', 'no new scientific', '1337', '2026', 'A3 §5.4b', 'not an amendment; no A9'):
        require(token in doc, 'record document token ' + token)
    return record


# ----------------------------------------------------------------- secure seam (stdlib AST only)
def seam_tree(rev=None):
    return ast.parse((at_authority(SEAM) if rev else read(SEAM)).decode())


def seam_constants():
    out = {}
    for n in seam_tree().body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            try:
                out[n.targets[0].id] = ast.literal_eval(n.value)
            except ValueError:
                pass
    return out


def functions(tree):
    return {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}


def seam():
    const = seam_constants()
    require((const['OWNER_FROZEN_SHA256'], const['OWNER_FROZEN_BYTES'], const['FREEZE_RECORD']) ==
            (FROZEN_SHA, FROZEN_BYTES, OVERLAY), 'seam frozen constants')
    now, old = seam_tree(), seam_tree(AUTHORITY)
    fn, fn_old = functions(now), functions(old)
    for name in UNCHANGED_SEAM_FUNCTIONS:     # the M6D6c exact-byte order and the unsafe-load seam are unchanged
        require(ast.unparse(fn[name]) == ast.unparse(fn_old[name]), 'seam function unchanged ' + name)
    require(set(fn) == set(fn_old) | {'load_freeze_record', 'frozen_checkpoint_path'}, 'only two helpers added')
    loads = [ast.unparse(n) for n in ast.walk(now) if isinstance(n, ast.Call) and ast.unparse(n.func) == 'torch.load']
    require(loads == [TORCH_LOAD] and const['LOAD_COMPATIBILITY']['weights_only'] is False,
            'exactly one torch.load(io.BytesIO(raw), weights_only=False)')
    body = [ast.unparse(s) for s in fn['load_verified_whole_module'].body]
    require(body.index('raw = read_verified(path, expected_sha256)') < body.index('import torch'), 'read_verified before torch')
    names = {n.attr if isinstance(n, ast.Attribute) else n.id for n in ast.walk(now) if isinstance(n, (ast.Attribute, ast.Name))}
    require(not FORBIDDEN_SEAM_NAMES & names, 'no state_dict/safetensors/jit/safe_globals/env/CLI/discovery in seam')
    require([ast.unparse(t) for n in ast.walk(now) if isinstance(n, ast.Assign) for t in n.targets
             if ast.unparse(t).startswith('torch.')] == [], 'no global torch.load patch')
    entry = fn['load_frozen_aux_encoder']
    require([a.arg for a in entry.args.args] == ['runtime_root', 'recorded_sha256', 'config'] and
            [ast.unparse(d) for d in entry.args.defaults] == ['None'] and
            [ast.unparse(d) for d in entry.decorator_list] == ['contextmanager'], 'loader signature unchanged')
    steps = [ast.unparse(s) for s in entry.body if not isinstance(s, ast.Expr)]
    order = ["frozen = load_freeze_record(config)['frozen_asset']", "if recorded_sha256 != frozen['sha256']:",
             'path = frozen_checkpoint_path(runtime_root, config)', 'if not path.is_file():',
             "if path.stat().st_size != frozen['bytes']:",
             'asset = verify_future_checkpoint(runtime_root, recorded_sha256, config)',
             "if (Path(asset['path']), asset['size_bytes']) != (path, frozen['bytes']):",
             "with load_verified_whole_module(asset['path'], recorded_sha256, config) as model:"]
    require(len(steps) == len(order) and all(s.startswith(o) for s, o in zip(steps, order)),
            'caller SHA gate -> path -> size gate -> external-asset verification -> exact-byte load')
    text = read(SEAM).decode()
    require('runs/m6' not in text and 'aux_resume' not in text and 'runner_state_epoch' not in text, 'no path literals')
    require("__name__ + '.load_frozen_aux_encoder'" in text and 'methods.difffas.aux_checkpoint' == SEAM[:-3].replace('/', '.'),
            'freeze record names this loader')
    require(read('methods/difffas/execution_policy.py') == at_authority('methods/difffas/execution_policy.py') and
            b'from .aux_checkpoint import load_frozen_aux_encoder' in read('methods/difffas/execution_policy.py'),
            'execution_policy unchanged and still binds the loader')
    return {'unchanged_functions': list(UNCHANGED_SEAM_FUNCTIONS), 'torch_load': loads, 'entry_order': order}


# ----------------------------------------------------------------- B1 (M6D6g test scope correction)
def b1():
    old, new = at_authority(B1_TEST), read(B1_TEST)
    require((sha(old), sha(new)) == (B1_OLD_SHA, B1_NEW_SHA), 'B1 old/new SHA256')
    t_old, t_new = ast.parse(old.decode()), ast.parse(new.decode())
    top_old = [ast.unparse(n) for n in t_old.body]
    top_new = [ast.unparse(n) for n in t_new.body]
    added = [s for s in top_new if s not in top_old]
    require(added[0] == f"M6D6G_COMMIT = '{AUTHORITY}'" and all(s.startswith('class TestM6D6gScientificRun') for s in added[1:])
            and [s for s in top_old if s not in top_new] == [s for s in top_old if s.startswith('class ')],
            'B1 top level: only M6D6G_COMMIT added')
    cls_old = {n.name: ast.unparse(n) for n in t_old.body[-2].body} if isinstance(t_old.body[-2], ast.ClassDef) else None
    cls_new = {n.name: ast.unparse(n) for n in t_new.body[-2].body} if isinstance(t_new.body[-2], ast.ClassDef) else None
    require(cls_old is not None and cls_new is not None and set(cls_old) == set(cls_new), 'B1 same test methods')
    changed = sorted(k for k in cls_old if cls_old[k] != cls_new[k])
    require(changed == ['test_17_scientific_files_byte_identical_to_authority'], 'B1 changes only test_17')
    t17 = cls_new[changed[0]]
    require("git('diff', '--name-only', AUTHORITY, M6D6G_COMMIT, '--', *self.pf.FROZEN_TREES)" in t17 and
            'scientific_files_unchanged' not in t17 and "'status'" not in t17 and "'HEAD'" in t17, 'B1 historical range')
    require(read('tools/m6d6g_e07c_aux_scientific_preflight.py') == at_authority('tools/m6d6g_e07c_aux_scientific_preflight.py'),
            'M6D6g preflight not modified')
    return {'path': B1_TEST, 'old_sha256': B1_OLD_SHA, 'new_sha256': B1_NEW_SHA, 'changed_tests': changed}


# ----------------------------------------------------------------- evidence
def check_evidence():
    ev = load(EV_JSON)
    require((ev['milestone'], ev['classification'], ev['final_status'], ev['method_id'], ev['method_status'],
             ev['fidelity_class'], ev['deviation'], ev['new_deviation'], ev['new_fidelity_class'], ev['amendment_created'],
             ev['authority_commit'], ev['record_kind']) ==
            ('M6D6h', CLASSIFICATION, 'PASS', 'E07c', 'IMPLEMENTED_NOT_EXECUTED', 'CONTROLLED_ADAPTATION', 'DEV-021',
             False, False, False, AUTHORITY, 'OWNER_DECISION / ASSET_FREEZE'), 'evidence identity')
    fr = ev['freeze_record']
    require((fr['path'], fr['sha256'], fr['document_path'], fr['document_sha256'], fr['bound_authority_sha256']) ==
            (OVERLAY, sha(read(OVERLAY)), RECORD_DOC, sha(read(RECORD_DOC)), BINDINGS), 'evidence freeze record')
    fa = ev['frozen_asset']
    require((fa['sha256'], fa['bytes'], fa['path_template'], fa['observed_gpu_path'], fa['format']) ==
            (FROZEN_SHA, FROZEN_BYTES, PATH_TEMPLATE, OBSERVED_GPU_PATH, FORMAT), 'evidence frozen asset')
    pr = ev['producer']
    require((pr['run_id'], pr['run_uuid'], pr['completed_epoch'], pr['global_step'], pr['auxiliary_seed'],
             pr['producer_commit']) == (RUN_ID, RUN_UUID, 200, 11200, 42, AUTHORITY), 'evidence producer')
    h, n = ev['historical_m6d6g'], ev['m6d6h_owner_freeze']
    require((h['final_checkpoint_status'], h['sha256'], h['owner_freeze_performed'], h['authoritative_for_main_difffas'],
             h['evidence_byte_identical_to_authority']) == (PENDING, FROZEN_SHA, False, False, True),
            'historical M6D6g state is distinguished and preserved')
    require(h['evidence_sha256'] == {p: sha(read(p)) for p in h['evidence_sha256']} and len(h['evidence_sha256']) == 4,
            'historical M6D6g evidence SHA256')
    require((n['owner_freeze_performed'], n['authoritative_for_main_difffas'], n['sha256'], n['bytes'],
             n['same_sha256_as_m6d6g'], n['same_bytes_as_m6d6g'], n['frozen_for_main_seeds'],
             n['alternative_checkpoints_authorized']) ==
            (True, True, FROZEN_SHA, FROZEN_BYTES, True, True, MAIN_SEEDS, 0), 'M6D6h owner freeze')
    s = ev['secure_loader']
    require((s['entry'], s['signature'], s['signature_unchanged'], s['freeze_record_sha256_pinned'],
             s['torch_load_expression_count'], s['execution_policy_modified']) ==
            (LOADER, 'load_frozen_aux_encoder(runtime_root, recorded_sha256, config=None)', True,
             sha(read(OVERLAY)), 1, False), 'secure loader binding')
    require(ev['b1_test_correction'] == {'path': B1_TEST, 'reason': B1_REASON, 'scientific_history_changed': False,
                                         'old_sha256': B1_OLD_SHA, 'new_sha256': B1_NEW_SHA,
                                         'changed_tests': ['test_17_scientific_files_byte_identical_to_authority'],
                                         'historical_range': [M6D6F_COMMIT, AUTHORITY],
                                         'm6d6g_evidence_or_ledger_modified': False}, 'B1 record')
    g = ev['gpu_revalidation']
    require((g['size_bytes'], g['sha256'], g['sha256_matches_frozen'], g['size_matches_frozen'], g['read_only'],
             g['aux_training_process_running'], g['main_difffas_process_running'], g['run_summary_completion_status'],
             g['run_summary_final_checkpoint_status'], g['runtime_metadata_unchanged'], g['checkpoint_deserialized'],
             g['chmod_performed']) ==
            (FROZEN_BYTES, FROZEN_SHA, True, True, True, False, False, 'completed', PENDING, True, False, False),
            'GPU read-only revalidation')
    require(ev['counters'] == ZERO, 'no-science counters')
    require((ev['real_checkpoint_deserialized'], ev['training'], ev['TRAIN_access'], ev['VAL_access'], ev['TEST_access'],
             ev['M8_bank'], ev['M8_integration_qualified'], ev['chmod_performed']) ==
            (False, False, False, False, False, False, False, False), 'no deserialization / data / M8 / chmod')
    require(ev['qualified_statuses'] == QUALIFIED and ev['not_qualified'] == NOT_QUALIFIED, 'statuses')
    log = read(LOG).decode()
    for token in (f'{FROZEN_SHA}  {OBSERVED_GPU_PATH}', f'SIZE={FROZEN_BYTES}', 'NO_AUX_TRAINING_PROCESS',
                  'NO_MAIN_DIFFFAS_PROCESS', 'completion_status=completed', f'final_checkpoint_status={PENDING}',
                  'RUNTIME_METADATA_UNCHANGED', 'NO_TORCH_LOAD', 'NO_CHMOD', 'M6D6H_GPU_EXIT=0'):
        require(token in log, 'runtime log token ' + token)
    for name, digest in g['runtime_small_files_sha256'].items():
        require(f'{digest}  {RUN_DIR}/{name}' in log, 'runtime log small-file hash ' + name)
    report = read(EV_MD).decode()
    for token in (FROZEN_SHA, '185136819', PENDING, 'OWNER_FROZEN', *QUALIFIED, *NOT_QUALIFIED, B1_OLD_SHA, B1_NEW_SHA,
                  'CONTROLLED_ADAPTATION', 'DEV-021', 'IMPLEMENTED_NOT_EXECUTED', 'M6D6i'):
        require(token in report, 'report token ' + token)
    return ev


def runtime_rehash(runtime_run_dir):
    """GPU host only: small runtime files vs M6D6g evidence; checkpoint is stat-ed, never opened."""
    run = json.loads(at_authority('outputs/audit/M6D6G_E07C_AUX_SCIENTIFIC_RUN.json'))
    base = Path(runtime_run_dir)
    for name in RUNTIME_SMALL:
        raw = (base / name).read_bytes()
        require({'size_bytes': len(raw), 'sha256': sha(raw)} == run['runtime_files'][name], 'runtime file ' + name)
    require((base / 'checkpoints/encoder_final.pkl').stat().st_size == FROZEN_BYTES, 'runtime checkpoint size')
    return True


# ----------------------------------------------------------------- ledger / index / worktree
def expected_index():
    reader = csv.DictReader(io.StringIO(at_authority(INDEX).decode()))
    baseline = list(reader)
    rows = {r['path']: r for r in baseline}
    require(reader.fieldnames == ['path', 'size_bytes', 'sha256'] and len(rows) == len(baseline) == INDEX_BASELINE_ROWS,
            'baseline index')
    for p in MODIFIED:
        require(p in rows, 'modified file already indexed ' + p)
    for p in NEW:
        require(p not in rows, 'additive artifact ' + p)
    for p in (*MODIFIED, *NEW):
        raw = read(p)
        rows[p] = {'path': p, 'size_bytes': str(len(raw)), 'sha256': sha(raw)}
    out = io.StringIO(newline='')
    writer = csv.DictWriter(out, fieldnames=reader.fieldnames, lineterminator='\r\n')
    writer.writeheader()
    writer.writerows(rows[k] for k in sorted(rows))
    return out.getvalue().encode(), len(rows)


def worktree(ledger_and_index_modified):
    status = sorted(git('status', '--porcelain', '--untracked-files=all').decode().splitlines())
    expected = ['?? ' + p for p in NEW] + [' M ' + p for p in MODIFIED]
    if ledger_and_index_modified:
        expected += [' M ' + INDEX, ' M ' + LEDGER]
    require(status == sorted(expected), 'worktree holds exactly the intended M6D6h changes: ' + json.dumps(status))
    require(git('diff', '--cached', '--name-only').decode().strip() == '', 'nothing staged')


def check_ledger_row(row):
    require((row['milestone'], row['classification'], row['method_id'], row['status'], row['final_status'],
             row['owner_freeze_performed'], row['frozen_checkpoint_sha256'], row['frozen_checkpoint_bytes'],
             row['auxiliary_checkpoint_authoritative_for_main'], row['checkpoint_deserialized'],
             row['scientific_auxiliary_runs_this_milestone'], row['main_difffas_scientific_runs'], row['optimizer_steps'],
             row['backward_calls'], row['TRAIN_access'], row['VAL_access'], row['TEST_access'], row['M8_bank'],
             row['fidelity_class'], row['deviation'], row['new_deviation'], row['authority_commit'], row['commit'],
             row['push']) ==
            ('M6D6h', CLASSIFICATION, 'E07c', 'PASS', 'PASS', True, FROZEN_SHA, FROZEN_BYTES, True, False, 0, 0, 0, 0,
             False, False, False, False, 'CONTROLLED_ADAPTATION', 'DEV-021', False, AUTHORITY, False, False),
            'ledger fields')
    require(row['b1_test_correction'] == load(EV_JSON)['b1_test_correction'], 'ledger B1 provenance')
    require(row['freeze_record_sha256'] == sha(read(OVERLAY)) and row['frozen_for_main_seeds'] == MAIN_SEEDS,
            'ledger freeze record / seeds')
    require(row['qualified_statuses'] == QUALIFIED and row['not_qualified'] == NOT_QUALIFIED and
            row['method_status'] == 'IMPLEMENTED_NOT_EXECUTED', 'ledger statuses')
    require(sorted(row['artifacts_sha256']) == sorted((*NEW, *MODIFIED)), 'ledger artifact list')
    for path, h in row['artifacts_sha256'].items():
        require(sha(read(path)) == h, 'ledger artifact ' + path)


def verify(stage, runtime_run_dir=None):
    """stage: 'before_ledger' (evidence only), 'before_index' (ledger appended), 'final'."""
    authority()
    frozen_files_unchanged()
    hist = historical_evidence()
    record = freeze_record()
    seam()
    b1()
    check_evidence()
    rehashed = runtime_rehash(runtime_run_dir) if runtime_run_dir else None
    prefix = at_authority(LEDGER)
    current = (ROOT / LEDGER).read_bytes()
    require(len(prefix.splitlines()) == LEDGER_PREFIX_ROWS and current.startswith(prefix),
            f'first {LEDGER_PREFIX_ROWS} ledger rows byte-identical')
    if stage == 'before_ledger':
        require(current == prefix, 'ledger not yet appended')
        worktree(False)
    else:
        require(len(current.splitlines()) == LEDGER_PREFIX_ROWS + 1 and current.endswith(b'\n'), 'exactly one append')
        check_ledger_row(json.loads(current[len(prefix):]))
    expected, count = expected_index()
    if stage == 'final':
        require((ROOT / INDEX).read_bytes() == expected, 'CRLF artifact index')
        worktree(True)
    require(not {'torch', 'yaml', 'pyarrow', 'numpy', 'PIL'} & set(sys.modules), 'static preflight: no torch/yaml/pyarrow/numpy/PIL')
    return {'status': 'PASS', 'stage': stage, 'ledger_rows': len(current.splitlines()),
            f'first_{LEDGER_PREFIX_ROWS}_rows_byte_identical': True, 'artifact_index_rows_before': INDEX_BASELINE_ROWS,
            'artifact_index_rows_expected': count, 'artifact_rows_added': len(NEW), 'artifact_rows_rehashed': len(MODIFIED),
            'freeze_record_sha256': sha(read(OVERLAY)), 'frozen_checkpoint_sha256': record['frozen_asset']['sha256'],
            'frozen_checkpoint_bytes': record['frozen_asset']['bytes'], 'historical_evidence_files_checked': len(hist),
            'runtime_rehash': rehashed, 'torch_imported': False, 'checkpoint_bytes_opened': 0, 'training': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before-ledger', action='store_true', help='evidence-only check before the ledger append')
    parser.add_argument('--rebuild-index', action='store_true', help='after the ledger append: write the index LAST')
    parser.add_argument('--runtime-run-dir', help='GPU host only: re-hash small runtime files; stat the checkpoint')
    args = parser.parse_args()
    if args.before_ledger:
        result = verify('before_ledger', args.runtime_run_dir)
    elif args.rebuild_index:
        verify('before_index', args.runtime_run_dir)
        (ROOT / INDEX).write_bytes(expected_index()[0])
        result = verify('final', args.runtime_run_dir)
    else:
        result = verify('final', args.runtime_run_dir)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
