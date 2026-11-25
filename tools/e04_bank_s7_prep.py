#!/usr/bin/env python3
"""E04 S7 readiness, governance DRAFTs and bank reproduction manifest (laptop; deterministic; committed inputs only).

  python3 tools/e04_bank_s7_prep.py build

Writes (nothing is approved, launched or created on the host):
  configs/amendments/learned_bank_s7_launch_authority_E04.DRAFT.yaml   narrowly scoped E04 S7 launch-authority DRAFT
  configs/amendments/learned_bank_storage_location_E04.DRAFT.yaml      additive E04 HDD storage-extension DRAFT
  configs/amendments/learned_bank_production_E04.DRAFT.yaml            production-authority TEMPLATE (needs S7 PASS)
  outputs/audit/E04_BANK_S7_ACCEPTANCE_PLAN.{json,md}                  executable S7 plan, not launched
  outputs/audit/E04_BANK_REPRODUCTION_MANIFEST.json                    training / fixture / future S7 / future production
Loaders in methods.common.learned_bank read only the activated file names (without .DRAFT), so none of these DRAFTs
can unlock S7, storage or production. Source hashes are taken from the adapter commit with `git show` (history-pinned).
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from methods.common import learned_bank as lb  # noqa: E402

ADAPTER_COMMIT = '92ef74c99765109158432fb713a68b3ad238f431'
CONTRACT_COMMIT = '3753ea4'
EVIDENCE_COMMIT = 'e495cc7'
S7_DRAFT = 'configs/amendments/learned_bank_s7_launch_authority_E04.DRAFT.yaml'
STORAGE_DRAFT = 'configs/amendments/learned_bank_storage_location_E04.DRAFT.yaml'
PRODUCTION_DRAFT = 'configs/amendments/learned_bank_production_E04.DRAFT.yaml'
PLAN = 'outputs/audit/E04_BANK_S7_ACCEPTANCE_PLAN'
REPRO = 'outputs/audit/E04_BANK_REPRODUCTION_MANIFEST.json'
PREFLIGHT = 'outputs/audit/E04_BANK_S7_PREP/storage_preflight.json'
FIXTURE_RECORD = 'outputs/audit/E04_BANK_FIXTURE_QUALIFICATION.json'
CONTRACT_EVIDENCE = 'outputs/audit/E04_BANK_CONTRACT_EVIDENCE.json'
TRAINING_REPRO = 'outputs/audit/E04_THREE_SEED_REPRODUCTION_MANIFEST.json'
SOURCES = ('methods/physics_std/bank_adapter.py', 'methods/common/learned_bank.py', 'tools/learned_bank.py',
           'tools/e04_bank_fixture_qualification.py', 'tools/e04_bank_scripts/e04_fixture_qualification.sh',
           'methods/physics_std/network.py', 'methods/physics_std/trace_transfer.py',
           'methods/physics_std/training_graph.py', 'methods/physics_std/production_io.py',
           'methods/physics_std/production_runner.py', 'methods/physics_std/feed_pack.py')
HOST = '/home/student20261/workdir'
RT = HOST + '/GPAT_TransferBench_runtime'
PY = '/home/student20261/miniconda3/envs/gpat-m6-e03/bin/python'
S7_ROOT = HOST + '/m8_learned_banks/s7/<execution_commit>/E04'
ENV_PREFIX = ('env -i HOME=/home/student20261 PATH=/usr/bin:/bin LANG=C.UTF-8 CUBLAS_WORKSPACE_CONFIG=:4096:8 '
              'CUDA_VISIBLE_DEVICES=0 NVIDIA_TF32_OVERRIDE=0 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 '
              'TF_CUDNN_DETERMINISTIC=1 TF_DETERMINISTIC_OPS=1 TF_ENABLE_AUTO_MIXED_PRECISION=0 '
              'TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE=0 TF_ENABLE_CUBLAS_TENSOR_OP_MATH_FP32=0 '
              'TF_ENABLE_CUDNN_RNN_TENSOR_OP_MATH_FP32=0 TF_ENABLE_CUDNN_TENSOR_OP_MATH_FP32=0 TF_ENABLE_ONEDNN_OPTS=0 '
              'TF_XLA_FLAGS=--tf_xla_auto_jit=0 nice -n 10 ' + PY + ' -B')


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sha_at(commit, rel, root=ROOT):
    blob = subprocess.run(['git', '-C', str(root), 'show', '%s:%s' % (commit, rel)], stdout=subprocess.PIPE,
                          check=True).stdout
    return hashlib.sha256(blob).hexdigest()


def dumps(o):
    return json.dumps(o, sort_keys=True, indent=1, ensure_ascii=True) + '\n'


def checkpoints():
    return {str(s): {'run_id': lb.checkpoint_authority('E04', s)['run_id'],
                     'run_root': '<runtime_root>/runs/m6/E04/seed_%d' % s,
                     'files': {lb.CHECKPOINT_FILES['E04'][k]: v for k, v in
                               lb.checkpoint_authority('E04', s)['sha256'].items()},
                     'checkpoint_identity_sha256': lb.checkpoint_identity_sha256('E04', s)} for s in lb.SEEDS}


def build(root=ROOT, out_root=None):
    root = Path(root)
    out = Path(out_root or root)
    contract_sha = sha_file(root / lb.E04_CONTRACT_PATH)
    fam_sha = sha_file(root / lb.E04_FAMILY_CLOSURE_RECORD)
    base_storage_sha = sha_file(root / lb.STORAGE_AUTHORITY_PATH)
    pf = json.loads((root / PREFLIGHT).read_text())
    fixture = json.loads((root / FIXTURE_RECORD).read_text())
    ck = checkpoints()
    idents = {s: v['checkpoint_identity_sha256'] for s, v in ck.items()}
    sources = {rel: sha_at(ADAPTER_COMMIT, rel, root) for rel in SOURCES}
    est = 808567774
    # ------------------------------------------------------------ S7 launch-authority DRAFT
    s7 = {
        'schema_version': lb.E04_S7_AUTHORITY_SCHEMA, 'record': 'E04_S7_LAUNCH_AUTHORITY',
        'status': 'DRAFT_PENDING_OWNER_DECISION', 'approved': False, 'date': '2026-11-25',
        'approval_procedure': 'The owner activates by committing ' + lb.E04_S7_AUTHORITY_PATH + ' with this content, '
                              'status OWNER_APPROVED and approved true, in a commit of its own, after the '
                              'launch_preconditions below hold. This DRAFT is never read by any loader.',
        'scope': 'E04 S7 toy-bank acceptance only: 186 toy pairs x 3 seeds x runs A and B (1116 images) on the GPU. '
                 'Not production, not storage, not retraining, no VAL/TEST, no downstream evaluation.',
        'methods': ['E04'], 'seeds': list(lb.SEEDS), 'contract_path': lb.E04_CONTRACT_PATH,
        'contract_sha256': contract_sha, 'e04_family_closure': lb.E04_FAMILY_CLOSURE_RECORD,
        'e04_family_closure_sha256': fam_sha, 'checkpoint_identities': idents,
        'toy_manifest': {'path': lb.TOY_MANIFEST_PATH, 'sha256': lb.TOY_MANIFEST_SHA256, 'pairs': lb.TOY_N},
        'reproducibility_standard': 'E04-Q5 A_GPU_FP32_BITWISE_REPLAY: byte-identical bank_manifest.jsonl and PNG '
                                    'bytes between runs A and B for every seed; a failure STOPS for owner review '
                                    '(no CPU or tolerance fallback, no graph change, no automatic rerun)',
        'resource_preflight': {'required': True, 'checks': [
            'no tools/run_e04.py, learned-bank or other S7 process (one GPU job at a time); unrelated GPU processes '
            'recorded and never touched',
            'GPU NVIDIA GeForce RTX 3090 GPU-b722cd9d-c9fa-223c-915e-cc09d9b862b7; driver 580.178.04 (D9 policy)',
            'free GPU memory >= 4 GiB (provisional; run A records nvidia-smi peak for the production floor)',
            '>= 2 GiB free on the S7 filesystem (outside the runtime root and outside the HDD bank root)',
            'the three E04 checkpoints re-hash to checkpoint_identities (lb.verify_checkpoint)',
            'tools/learned_bank.py gates --method E04 reports contract ok and s7 ok',
            'clean execution clone at the execution commit; gpat-m6-e03 python present; E04 training checkout '
            '18ba00e untouched']},
        'launch_preconditions': [
            'the E04 independent S7 audit collector/builder (tools/learned_bank_s7_audit.py has no E04 branch at '
            + ADAPTER_COMMIT[:7] + ') implemented and fixture-tested before launch',
            'this record activated by the owner'],
        'not_authorized': ['production banks', 'creating <bank_root>/E04', 'retraining', 'VAL/TEST access',
                           'downstream evaluation', 'DINOv3', 'GPAT']}
    # ------------------------------------------------------------ storage-extension DRAFT
    storage = {
        'schema_version': lb.E04_STORAGE_EXTENSION_SCHEMA, 'record': 'E04_STORAGE_EXTENSION',
        'status': 'DRAFT_PENDING_OWNER_DECISION', 'approved': False, 'date': '2026-11-25', 'methods': ['E04'],
        'approval_procedure': 'The owner activates by committing ' + lb.E04_STORAGE_EXTENSION_PATH + ' with this '
                              'content, status OWNER_APPROVED, approved true and activation_verification.ok true '
                              '(a live lb.storage_preflight on the host at activation), after the E04 S7 record '
                              'passes. The base authority is never edited.',
        'base_storage_authority': {'path': lb.STORAGE_AUTHORITY_PATH, 'sha256': base_storage_sha,
                                   'unchanged': True, 'methods': ['E03', 'E05', 'E06b', 'E06c']},
        'registry_extension': {'bank_root': lb.HDD['bank_root'], 'filesystem_uuid': lb.HDD['filesystem_uuid'],
                               'mount_point': lb.HDD['mount_point'], 'fstype': lb.HDD['fstype'],
                               'layout': lb.E04_BANK_LAYOUT, 'seeds': list(lb.SEEDS)},
        'code_support': {
            'commit': ADAPTER_COMMIT,
            'loader': 'methods.common.learned_bank.load_e04_storage_extension (reads only the activated name)',
            'root_checker': 'registered_bank_methods() returns E03/E05/E06b/E06c unless this extension is activated; '
                            'storage_preflight passes that set to bank_root_contents_ok, so an E04 directory is '
                            'unexpected (refusing every learned bank) until activation',
            'production_path': 'load_storage_authority(method_id="E04") requires the extension; production_bank_dir '
                               'accepts E04 only then; tools/learned_bank.py production creates <bank_root>/E04 '
                               '(mode 0700) at the first authorized E04 launch, never earlier',
            'tests': 'tests/test_e04_bank_adapter.py Governance.test_registry_accepts_e04_only_when_extension_activated'},
        'verified_state': {'utc': '2026-11-25T12:57:14Z', 'record': PREFLIGHT, 'record_sha256': sha_file(root / PREFLIGHT),
                           'ok': pf['ok'], 'checks': pf['checks'], 'unexpected_entries': pf['unexpected_entries'],
                           'filesystem_uuid': pf['filesystem']['filesystem_uuid'], 'mode': pf['mode_octal'],
                           'free_bytes': pf['filesystem']['free_bytes'], 'free_inodes': pf['filesystem']['free_inodes'],
                           'e04_directory_exists': False},
        'estimate': {'per_seed_bytes': est, 'family_bytes': 3 * est,
                     'basis': 'E03 seed-42 PNG total for 8838 synthetic faces (estimate only)'},
        'activation_verification': {'ok': None, 'note': 'set only by the owner activation after a live preflight'},
        'not_changed': ['mount options', 'filesystem permissions', 'existing bank paths', 'existing banks',
                        'the base storage authority']}
    # ------------------------------------------------------------ production-authority TEMPLATE
    production = {
        'schema_version': lb.PRODUCTION_AUTHORITY_SCHEMA, 'record': 'E04_PRODUCTION_AUTHORITY_TEMPLATE',
        'status': 'DRAFT_TEMPLATE_NOT_APPROVABLE_BEFORE_S7', 'approved': False, 'method': 'E04',
        'seeds': list(lb.SEEDS), 'contract_sha256': contract_sha,
        's7_acceptance_record': {'path': 'outputs/audit/LEARNED_BANK_S7_E04.json', 'sha256': None,
                                 'required': 's7_acceptance_passed true, failed_gates [], seeds [42, 1337, 2026]'},
        'storage_extension': {'path': lb.E04_STORAGE_EXTENSION_PATH, 'required_status': 'OWNER_APPROVED'},
        'checkpoint_identities': idents, 'population': {'rows_per_seed': lb.N_SYN, 'family_rows': 3 * lb.N_SYN},
        'one_bank_per_process': 'tools/learned_bank.py production --method E04 --seed S, seeds 42 -> 1337 -> 2026',
        'requirements': ['every S7 toy row equals its production row byte for byte',
                         'independent full-population production audit per bank before the next launch',
                         'family closure only after all three banks pass'],
        'tooling_gap': 'tools/learned_bank_production_authority.py and tools/learned_bank_production_audit.py have no '
                       'E04 branch at ' + ADAPTER_COMMIT[:7] + '; both must be extended before production'}
    # ------------------------------------------------------------ S7 plan
    plan = {
        'record': 'E04_BANK_S7_ACCEPTANCE_PLAN', 'schema': 'e04_bank_s7_plan_v1', 'status': 'PREPARED_NOT_LAUNCHED',
        'date': '2026-11-25', 'adapter_commit': ADAPTER_COMMIT,
        'contract': {'path': lb.E04_CONTRACT_PATH, 'sha256': contract_sha},
        'fixture_qualification': {'path': FIXTURE_RECORD, 'sha256': sha_file(root / FIXTURE_RECORD),
                                  'status': fixture['status']},
        'checkpoints': ck,
        'population': {'toy_manifest': lb.TOY_MANIFEST_PATH, 'toy_manifest_sha256': lb.TOY_MANIFEST_SHA256,
                       'pairs_per_seed': lb.TOY_N, 'seeds': list(lb.SEEDS), 'runs': ['A', 'B'],
                       'expected_images': lb.TOY_N * 3 * 2, 'batch_size': 1,
                       'identity': 'row_id = pair_id; target_live_id / source_spoof_id / dataset from the frozen '
                                   'relation (toy row must equal the relation row at its position)'},
        'environment': {'python': PY, 'environment': 'gpat-m6-e03', 'lock': 'environments/e04_training.lock.json '
                        + lb.ENVIRONMENTS['E04']['lock'][1], 'driver_policy': 'D9 595.84 -> 580.178.04 version only',
                        'gpu': 'NVIDIA GeForce RTX 3090 GPU-b722cd9d-c9fa-223c-915e-cc09d9b862b7',
                        'precision_env': 'methods.physics_std.production_io.PRECISION_ENV',
                        'session_config': 'methods.physics_std.production_runner.session_config',
                        'verified_by': 'bank_adapter.load_real -> production_runner.verify_environment'},
        'workflow': [
            'owner commits ' + lb.E04_S7_AUTHORITY_PATH + ' (from ' + S7_DRAFT + ') after the E04 S7 audit tool exists',
            'git bundle of the execution commit -> ~/workdir/m8_learned_banks/bundles/; fresh clone '
            '~/workdir/m8_learned_banks/clones/s7e04_<execution_commit>; mkdir -p third_party/source_cache and copy '
            'stdn from the GPU checkout (read-only); training checkout 18ba00e never used',
            'resource preflight (S7 authority checks) and atime ledger of all TRAIN/VAL/TEST faces (stat only) -> '
            '~/workdir/m8s7_E04_A_atime_pre.tsv',
            'mkdir -p ' + S7_ROOT + ' (outside the runtime root and the HDD bank root)',
            'tmux m8s7_E04_A (remain-on-exit, pipe-pane log): run A command; exit code -> ~/workdir/m8s7_E04_A.exit; '
            'nvidia-smi --query-compute-apps sampled every 2 s',
            'after A exits 0: tmux m8s7_E04_B: run B command; exit code -> ~/workdir/m8s7_E04_B.exit',
            'compare command; atime ledger post; checkpoint re-hash',
            'independent read-only host collector + laptop builder -> outputs/audit/LEARNED_BANK_S7_E04.json'],
        'commands': {
            'gates': ENV_PREFIX.replace('CUDA_VISIBLE_DEVICES=0', 'CUDA_VISIBLE_DEVICES=') +
                     ' tools/learned_bank.py gates --method E04',
            'run_A': ENV_PREFIX + ' tools/learned_bank.py s7 --method E04 --label A --out-root ' + S7_ROOT +
                     ' --runtime-root ' + RT,
            'run_B': ENV_PREFIX + ' tools/learned_bank.py s7 --method E04 --label B --out-root ' + S7_ROOT +
                     ' --runtime-root ' + RT,
            'compare': ENV_PREFIX.replace('CUDA_VISIBLE_DEVICES=0', 'CUDA_VISIBLE_DEVICES=') +
                       ' tools/learned_bank.py s7-compare --method E04 --out-root ' + S7_ROOT},
        'process_model': 'one OS process per run label holding the three seeds sequentially, a fresh TF graph per '
                         'seed (tf.reset_default_graph), as qualified for E03; no --seed selection (E05-only ruling)',
        'acceptance_gates': [
            'checkpoint SHA256 of all 3 files per seed equal before run A and after run B (lb.verify_checkpoint / '
            'recheck_checkpoint_unchanged)',
            'frozen config, contract, frozen training sources and predecessor conversion lines verified '
            '(bank_adapter.verify_frozen_sources)',
            'environment lock verified on the GPU (production_runner.verify_environment, D9 driver policy)',
            'inference graph structure equal to the fixture-qualified identity (variable_names_sha256, '
            'variable_shapes_sha256, 15 FusedBatchNorm is_training=False, 0 update/gradient/optimizer/random ops)',
            'generator-only restore: 102 restored of 500 checkpoint variables; moving statistics finite',
            'population: 186 rows per seed equal to the toy manifest positions and the frozen relation',
            '186 decodable 256x256 RGB PNGs per seed and run, 0 failures, all raw outputs finite',
            'reader: lb.read_bank ok with expect_mode S7_TOY, expect_rows 186, expect_checkpoint_identity',
            'every PNG re-decoded by the audit: pixel and file SHA256 equal to the manifest; conversion rule string '
            'E04-Q3; raw output finite with recorded min/max per row',
            'bitwise replay: lb.compare_runs(run_A, run_B) bitwise_identical for every seed (E04-Q5)',
            'TRAIN-only allowlist; reader record reads == 2 x 186 O_NOATIME atime-checked reads per seed and run',
            'atime ledger: no VAL/TEST face atime changed; TRAIN atimes unchanged',
            'provenance: authorization S7_TOY bound to the E04 S7 authority, the E04 contract and the family closure',
            'later (production audit): every toy row equals its production row byte for byte'],
        'failure_behavior': 'any failed gate or non-bitwise replay STOPS the E04 family for an owner ruling; evidence '
                            'is preserved; no automatic rerun into an existing root; never a CPU or tolerance fallback',
        'required_owner_authority': [lb.E04_S7_AUTHORITY_PATH],
        'remaining_before_launch': ['E04 branch of the independent S7 audit (tools/learned_bank_s7_audit.py)',
                                    'owner activation of the E04 S7 launch authority'],
        'production_unlock': ['LEARNED_BANK_S7_E04.json PASS for all three seeds', lb.E04_STORAGE_EXTENSION_PATH +
                              ' activated', 'learned_bank_production_E04.yaml activated (template ' + PRODUCTION_DRAFT +
                              ')', 'E04 branches of the production authority/audit tools',
                              'live storage preflight at every launch'],
        'resource_estimate': 'single RTX 3090; batch-1 TF1 generator < 2 GiB expected; CPU warp + scipy offsets '
                             'dominate (estimate 0.3-0.6 s per pair: about 1-2 min per seed per S7 run; 45-90 min per '
                             'production seed); to be measured in run A'}
    # ------------------------------------------------------------ reproduction manifest
    graph = fixture['key_results']['graph']
    repro = {
        'schema': 'e04_bank_reproduction_manifest_v1', 'method': 'E04', 'date': '2026-11-25',
        'reporting_label': 'Physics-STD (controlled geometry/depth reconstruction)', 'fidelity': 'CONTROLLED_ADAPTATION',
        'git': {'start_commit': '6f73eed298ecdd27bf1d7773e42a3b083f33b463', 'contract_commit': CONTRACT_COMMIT,
                'adapter_commit': ADAPTER_COMMIT, 'fixture_evidence_commit': EVIDENCE_COMMIT},
        'owner_ruling': {'path': lb.E04_CONTRACT_PATH, 'sha256': contract_sha, 'decisions': lb.E04_APPROVED_DECISIONS,
                         'draft': {'path': lb.E04_DRAFT_PATH, 'sha256': lb.E04_DRAFT_SHA256}},
        'frozen_config': {'path': lb.METHOD_CONFIGS['E04'][0], 'sha256': lb.METHOD_CONFIGS['E04'][1]},
        'environment_lock': {'path': lb.ENVIRONMENTS['E04']['lock'][0], 'sha256': lb.ENVIRONMENTS['E04']['lock'][1],
                             'interpreter': PY, 'tensorflow': '1.15.5'},
        'checkpoints': ck,
        'source_sha256_at_adapter_commit': sources,
        'inference_graph_identity': {k: graph[k] for k in ('variable_names_sha256', 'variable_shapes_sha256',
                                                           'reachable_op_types_sha256', 'bn_statistics_reads_sha256',
                                                           'global_variables', 'trainable_variables',
                                                           'bn_moving_statistics', 'fused_batch_norm', 'batch_size')},
        'rules': {
            'input_preprocessing': 'D8: canonical RGB uint8 PNG (SHA256-checked, TRAIN-only O_NOATIME) -> '
                                   'float32(uint8)/255 once; no augmentation (D5)',
            'geometry_and_offsets': 'e04_feed_pack_v1 Q140 xy (frozen CPU pre-pass); offsets = '
                                    'dense_offsets(source_xy=spoof Q140, target_xy=live Q140), scipy 1.10.1',
            'normalization': 'E04-Q1 A_MOVING_STATISTICS_PER_PAIR: BN training=False with the restored moving '
                             'statistics, batch 1',
            'synthesis': 'E04-Q2: out = (1-P_w)*(I_live + B~_w + C~_w + T_w) + P_w*I_source_w '
                         '(training_graph.py:32-39 operations and operand order)',
            'conversion': 'E04-Q3: (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8); non-finite = counted failure',
            'encoding': 'E04-Q6: lossless PNG, PIL compress_level 6, optimize False, no metadata',
            'rng': 'none; the experiment seed selects only the checkpoint'},
        'training_reproduction': {'status': 'COMPLETED_AND_CLOSED', 'manifest': TRAINING_REPRO,
                                  'manifest_sha256': sha_file(root / TRAINING_REPRO),
                                  'closure': lb.E04_FAMILY_CLOSURE_RECORD, 'closure_sha256': fam_sha},
        'synthetic_fixture_qualification': {
            'status': 'EXECUTED_PASSED', 'record': FIXTURE_RECORD, 'record_sha256': sha_file(root / FIXTURE_RECORD),
            'host_command': fixture['cpu_fixture_qualification']['host_command'],
            'host_gates': [fixture['cpu_fixture_qualification']['host_gates_passed'],
                           fixture['cpu_fixture_qualification']['host_gates_total']],
            'laptop_command': fixture['cpu_fixture_qualification']['laptop_command'],
            'laptop_tests': fixture['cpu_fixture_qualification']['laptop_tests_ran'],
            'evidence_files_sha256': fixture['cpu_fixture_qualification']['evidence_files'],
            'contract_evidence': {'path': CONTRACT_EVIDENCE, 'sha256': sha_file(root / CONTRACT_EVIDENCE)}},
        'future_s7_execution': {'status': 'PLANNED_NOT_EXECUTED', 'plan': PLAN + '.json', 'authority_draft': S7_DRAFT,
                                'expected_images': lb.TOY_N * 6},
        'future_production_generation': {'status': 'PLANNED_NOT_EXECUTED', 'template': PRODUCTION_DRAFT,
                                         'storage_extension_draft': STORAGE_DRAFT,
                                         'expected_images': {'per_seed': lb.N_SYN, 'family': 3 * lb.N_SYN}},
        'limitations': [
            'bank inference uses stored moving statistics; training-time synthesis used 8-image batch statistics '
            '(controlled adaptation, not official Physics-STD test-time behaviour)',
            'clip-truncate conversion follows the ECCV20-STDN predecessor (reference only where PhySTD is silent)',
            'CPU fixture qualification proves neither GPU bitwise replay nor GPU numerical equality',
            'no real-weight forward has been run; real inputs (feed pack, faces) enter only at S7',
            'S7 audit, production authority and production audit tools have no E04 branch yet']}
    for rel, doc in ((S7_DRAFT, s7), (STORAGE_DRAFT, storage), (PRODUCTION_DRAFT, production), (REPRO, repro),
                     (PLAN + '.json', plan)):
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        (out / rel).write_text(dumps(doc))
    (out / (PLAN + '.md')).write_text(report(plan, s7, storage))
    return {'s7': s7, 'storage': storage, 'production': production, 'plan': plan, 'repro': repro}


def report(plan, s7, storage):
    L = ['# E04 Physics-STD S7 acceptance plan (prepared, NOT launched)', '',
         '**Status:** `%s`. Contract `%s` (`%s`); fixture qualification `%s`.' % (
             plan['status'], plan['contract']['path'], plan['contract']['sha256'][:16],
             plan['fixture_qualification']['status']), '',
         '## Checkpoints', '', '| seed | run_id | identity |', '|---|---|---|']
    L += ['| %s | `%s` | `%s` |' % (s, v['run_id'], v['checkpoint_identity_sha256']) for s, v in
          sorted(plan['checkpoints'].items(), key=lambda x: int(x[0]))]
    p = plan['population']
    L += ['', '## Population', '', '- %d toy pairs (`%s`) x seeds %s x runs %s = **%d images**, batch %d.' % (
        p['pairs_per_seed'], p['toy_manifest_sha256'][:16], p['seeds'], p['runs'], p['expected_images'],
        p['batch_size']), '', '## Workflow', '']
    L += ['%d. %s' % (i, w) for i, w in enumerate(plan['workflow'], 1)]
    L += ['', '## Commands', '']
    L += ['- **%s**: `%s`' % kv for kv in sorted(plan['commands'].items())]
    L += ['', '## Acceptance gates', '']
    L += ['- ' + g for g in plan['acceptance_gates']]
    L += ['', '**Failure:** ' + plan['failure_behavior'], '', '## Remaining before launch', '']
    L += ['- ' + x for x in plan['remaining_before_launch']]
    L += ['', '## Production unlock', '']
    L += ['- ' + x for x in plan['production_unlock']]
    L += ['', '## Governance DRAFTs (not loadable)', '',
          '- `%s` — status `%s`.' % (S7_DRAFT, s7['status']),
          '- `%s` — status `%s`; live HDD state %s: preflight ok %s, unexpected %s, free %.1f GiB.' % (
              STORAGE_DRAFT, storage['status'], storage['verified_state']['utc'], storage['verified_state']['ok'],
              storage['verified_state']['unexpected_entries'], storage['verified_state']['free_bytes'] / 2 ** 30),
          '- `%s` — production template, not approvable before S7.' % PRODUCTION_DRAFT, '',
          '**Resources:** ' + plan['resource_estimate'], '']
    return '\n'.join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['build'])
    ap.parse_args()
    d = build()
    print(d['plan']['status'], d['s7']['status'], d['storage']['status'])


if __name__ == '__main__':
    main()
