"""Writes the per-method owner production authority configs/amendments/learned_bank_production_<M>.yaml
(learned_bank_production_authority_v1) from the committed records, so every bound hash is derived, never typed.

    python3 tools/learned_bank_production_authority.py --method M --source-commit <tooling commit> --date YYYY-MM-DD

Refuses to overwrite an existing authority (an executed bank is never re-authorized) and refuses unless the method's S7
record passed. The commit that adds the file is the execution commit; it may differ from the source commit only in
configs/amendments/learned_bank_production_*.yaml and outputs/audit/** (checked by every production audit).
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from methods.common import learned_bank as lb  # noqa: E402

APPROVAL = ("Owner conditional authorization of 2026-11-25 (session instruction 'GPAT-TransferBench v1.0 - M8 Full "
            "Production Banks for Four S7-Qualified Learned Baselines'): full TRAIN synthetic-bank production for the four "
            "S7-qualified learned methods E06c, E03, E06b and E05, executed sequentially (E06c, E03, E06b, E05; seeds 42, "
            "1337, 2026) with an independent authority, complete generation, independent scientific audit and family "
            "closure per method. No generator retraining, no frozen-rule change, no downstream training, TEST evaluation, "
            "DINOv3, GPAT or E04 bank generation.")
DISPLAY = {'E06c': 'DSDG-BIN-IDFREE', 'E03': 'STDN', 'E06b': 'DSDG-NATIVE', 'E05': 'PCGAN (controlled architecture '
           'resolution)'}


def sha(p):
    return hashlib.sha256((ROOT / p).read_bytes()).hexdigest()


def authority(M, source_commit, date):
    s7rel = 'outputs/audit/LEARNED_BANK_S7_%s.json' % M
    s7 = json.loads((ROOT / s7rel).read_text())
    if not (s7['s7_acceptance_passed'] is True and s7['failed_gates'] == [] and s7['seeds'] == list(lb.SEEDS)):
        raise SystemExit('S7 record of %s did not pass' % M)
    s7auth = json.loads((ROOT / lb.S7_AUTHORITY_PATH).read_text())
    storage = json.loads((ROOT / lb.STORAGE_AUTHORITY_PATH).read_text())
    env = lb.ENVIRONMENTS[M]
    pair = M in lb.PAIR_METHODS
    ck = {}
    for s in lb.SEEDS:
        a = lb.CHECKPOINTS[(M, s)]
        ck[str(s)] = {'run_id': a['run_id'], 'sha256': a['sha256'], 'rule': lb.CHECKPOINT_RULES[M],
                      'checkpoint_identity_sha256': lb.checkpoint_identity_sha256(M, s),
                      'seed_closure_commit': a['seed_closure_commit'], 'family_closure_commit': a['family_closure_commit'],
                      'path': '<runtime_root>/runs/m6/%s/seed_%d/%s' % (M, s, '|'.join(lb.CHECKPOINT_FILES[M].values()))}
    if pair:
        population = {'kind': 'PRODUCTION_FROZEN_TRAIN_POPULATION', 'rows_per_seed': lb.N_SYN,
                      'pairs': {'path': lb.PAIRS_PATH, 'sha256': lb.PAIRS_SHA256},
                      'order': 'every frozen pairs_train_v1 row in canonical 0-based global position order 0..8837',
                      'identity': 'row_id = pair_id; source_spoof_id and target_live_id preserved per row',
                      'faces': 'TRAIN only via TrainFaceReader (split_v1 + M2 allowlist, O_NOATIME, atime stat check, M2 '
                               'SHA256); %d O_NOATIME reads per bank' % (2 * lb.N_SYN)}
        if M == 'E03':
            population['landmarks'] = 'e03_feed_pack_v1 TRAIN side (landmarks_px256 / 256, unflipped); pairs_val never opened'
    else:
        population = {'kind': 'PRODUCTION_FROZEN_TRAIN_POPULATION', 'rows_per_seed': lb.N_SYN,
                      'stream': 'synthetic stream indices 0..8837; 89 official batches of 100, 8900 drawn, the last 62 of '
                                'batch 88 discarded and recorded', 'identity': "synthetic_id '%s_S<seed>_<index:06d>'; "
                                'pair_id, target_live_id and source_spoof_id null' % M,
                      'bank_seeds': {str(s): lb.dsdg_bank_seed(M, s) for s in lb.SEEDS}, 'faces': 'no face is read'}
    rng = {'E05': 'torch.manual_seed(e05_pair_seed(seed, pair_id)) immediately before the single mix forward '
                  'mix = G(E(x_src).spatial, E(x_tgt).global), x_src = source spoof, x_tgt = target live',
           'E03': 'none: Gen(x, training_nn=False) is deterministic at inference (batch 1, no augmentation)',
           'E06b': 'DSDG-Q1 A_OFFICIAL_BATCHES_SEEDED: torch.Generator().manual_seed(bank_seed); per batch noise = '
                   'randn(100,128), noise_s = randn(100,128), z = cat(noise_s, noise, noise)',
           'E06c': 'DSDG-Q1 A_OFFICIAL_BATCHES_SEEDED: torch.Generator().manual_seed(bank_seed); per batch noise = '
                   'randn(100,128), noise_s = randn(100,128), z = cat(noise_s, noise, noise)'}[M]
    doc = {
        'schema_version': 'learned_bank_production_authority_v1',
        'record': 'LEARNED_BANK_PRODUCTION_AUTHORITY_%s' % M,
        'status': 'OWNER_APPROVED', 'approved': True, 'date': date, 'approval': APPROVAL,
        'method': M, 'display_name': DISPLAY[M], 'seeds': list(lb.SEEDS), 'seed_order': list(lb.SEEDS),
        'method_order_in_program': ['E06c', 'E03', 'E06b', 'E05'],
        'contract_path': lb.CONTRACT_PATH, 'contract_sha256': sha(lb.CONTRACT_PATH),
        's7_acceptance_record': {'path': s7rel, 'sha256': sha(s7rel), 'status': s7['status'],
                                 'execution_commit': s7['execution_commit'],
                                 'gates': '%d/%d' % (s7['gates_passed'], s7['gates_total'])},
        's7_launch_authority': {'path': lb.S7_AUTHORITY_PATH, 'sha256': sha(lb.S7_AUTHORITY_PATH)},
        's7_readiness_summary': {'path': 'outputs/audit/M8_S7_READINESS_SUMMARY_V2.json',
                                 'sha256': sha('outputs/audit/M8_S7_READINESS_SUMMARY_V2.json')},
        'e04_records': {'seed_2026_completion': {'path': lb.E04_COMPLETION_RECORD, 'sha256': sha(lb.E04_COMPLETION_RECORD)},
                        'family_closure': {'path': lb.E04_FAMILY_CLOSURE_RECORD,
                                           'sha256': sha(lb.E04_FAMILY_CLOSURE_RECORD)}},
        'frozen_config': {'path': lb.METHOD_CONFIGS[M][0], 'sha256': lb.METHOD_CONFIGS[M][1]},
        'pinned_sources': {'path': lb.SOURCES_PATH, 'sha256': sha(lb.SOURCES_PATH),
                           'family': {'E03': 'STDN', 'E05': 'PCGAN'}.get(M, 'DSDG')},
        'checkpoints': ck,
        'environment': {'name': env['name'], 'python': env['python'],
                        'lock': {'path': env['lock'][0], 'sha256': env['lock'][1]}},
        'gpu': {'model': s7auth['gpu']['model'], 'uuid': s7auth['gpu']['uuid'], 'driver_observed': s7auth['gpu']['driver_observed'],
                'cuda_visible_devices': '0', 'precision': 'C-Q2 A_GPU_FP32_BITWISE_REPLAY settings of the contract'},
        'storage_authority': {'path': lb.STORAGE_AUTHORITY_PATH, 'sha256': sha(lb.STORAGE_AUTHORITY_PATH),
                              'bank_root': storage['bank_root_registry']['bank_root'],
                              'filesystem_uuid': storage['bank_root_registry']['filesystem_uuid'],
                              'mount_point': storage['bank_root_registry']['mount_point'],
                              'layout': storage['bank_root_registry']['layout'],
                              'banks': {str(s): '%s/%s/seed_%d' % (storage['bank_root_registry']['bank_root'], M, s)
                                        for s in lb.SEEDS},
                              'live_preflight': 'methods.common.learned_bank.storage_preflight() all true before every '
                                                'launch (load_production_authority) and in every audit'},
        'population': population, 'rng': rng, 'conversion_rule': lb.CONVERSION_RULES[M],
        'expected_outputs': {'per_seed': lb.N_SYN, 'family': 3 * lb.N_SYN, 'format': '256x256x3 uint8 RGB lossless PNG '
                             '(PIL compress_level 6, optimize False, no metadata); mode 0444; bank and seed dirs 0555',
                             'label': 'spoof', 'failures_allowed': 'only NaN/Inf raw outputs are counted failures; '
                             'production expects 0 (any failure stops the family for owner review)'},
        'code_authority': {
            'source_commit': source_commit,
            'execution_commit': 'the commit that adds this record; it may differ from source_commit only in '
                                'configs/amendments/learned_bank_production_*.yaml and outputs/audit/**',
            'execution_checkout': '~/workdir/m8_learned_banks/clones/prod_%s_<execution_commit> (fresh clean clone from a '
                                  'git bundle; never the GPU training checkout 18ba00e); third_party/source_cache copied '
                                  'from the GPU checkout' % M,
            'cli': 'tools/learned_bank.py production --method %s --seed <S> --runtime-root <runtime_root> (one OS process '
                   'per bank)' % M,
            'launch_scripts': 'tools/learned_bank_production_scripts/m8prod_{launch,wait,post}.sh',
            'audit_tool': 'tools/learned_bank_production_audit.py'},
        'production_audit': {
            'record': 'outputs/audit/LEARNED_BANK_PRODUCTION_%s_SEED<S>.json' % M,
            'evidence': 'outputs/audit/LEARNED_BANK_PRODUCTION/%s_s<S>/{collected.json,bank_manifest.jsonl.gz,'
                        'preflight.json}' % M,
            'family_closure': 'outputs/audit/LEARNED_BANK_PRODUCTION_%s_FAMILY_CLOSURE.json' % M,
            'requires': ['exactly 8838 rows and 8838 outputs, 0 failures, no duplicate identity',
                         'exact population (pairs_train_v1 canonical order or stream 0..8837) re-derived from the '
                         'committed full manifest', 'geometry, dtype, PNG decode, file and pixel SHA256 of every output '
                         'by the library reader AND an independent walk', 'finite raw outputs',
                         'checkpoint identity before and after (sha256 + mtime) for all 12 approved checkpoints',
                         'no optimizer, gradient or save (guards / finalized TF graph)',
                         'TRAIN-only reads, O_NOATIME, zero VAL/TEST/TRAIN face atime or mtime change',
                         'S7-to-production byte equivalence of the 186 toy rows (S7 runs A and B)',
                         'host, GPU, environment, storage UUID, registered path', 'E01/E02 banks unchanged',
                         'every other learned bank unchanged', 'audit committed and pushed before the next launch']},
        'reproducibility': {
            'standard': 'C-Q2 A_GPU_FP32_BITWISE_REPLAY + C-Q3 A_TOY_TWO_RUN_BITWISE (passed S7) and byte equality of '
                        'the 186 S7 toy rows with the production rows',
            'full_bank_replay': 'not required by the contract and not performed; never substituted by a procedural '
                                'or approximate check',
            'procedural_fallback': 'NOT AUTHORIZED'},
        'test_firewall': {'val_test_reads': 0, 'reader': 'TrainFaceReader TRAIN allowlist, O_NOATIME, atime check per read',
                          'atime_ledger': 'stat-only ledger of all canonical faces (TRAIN/VAL/TEST) before and after '
                                          'each bank; no change allowed'},
        'resource_preflight': {'required': True, 'gpu_free_floor_gib': {'E06c': 6, 'E06b': 6, 'E05': 5, 'E03': 4}[M],
                               'ram_floor_gib': 4, 'hdd_free_floor_gib': 20, 'one_learned_bank_process_at_a_time': True,
                               'unrelated_gpu_jobs': 'recorded, never touched'},
        'fail_closed': ['any gate failure stops this family; evidence and partial outputs are preserved untouched',
                        'no automatic rerun and no rerun into an existing bank directory',
                        'HDD disconnect, remount or UUID change: stop immediately, preserve partial evidence',
                        'S7-to-production mismatch: stop the family; nothing is relaxed and no reference is changed',
                        'GPU contention/OOM: never touch other jobs'],
        'authorizes': {'production_bank_generation': True, 'methods': [M], 'generator_retraining': False,
                       'checkpoint_change': False, 'quality_filtering': False, 'val_or_test_access': False,
                       'downstream_training': False, 'test_evaluation': False, 'dinov3': False, 'gpat': False,
                       'e04_bank_generation': False, 'method_status_update': False},
    }
    if M == 'E05':
        doc['e05'] = {
            'o2_ruling': {'path': 'configs/amendments/e05_source_relocation_ruling.yaml',
                          'sha256': sha('configs/amendments/e05_source_relocation_ruling.yaml')},
            'b1_b2_ruling': {'path': 'configs/amendments/e05_s7_b1_b2_ruling.yaml',
                             'sha256': sha('configs/amendments/e05_s7_b1_b2_ruling.yaml')},
            'process_isolation': 'P2_ONE_PROCESS_PER_SEED',
            'process_isolation_note': 'each production bank is its own OS process (tools/learned_bank.py production '
                                      '--seed S); existing_runtime keeps refusing a second load; P1 is not used',
            'runtime_validation': 'O2 source identity, compatibility-patch manifest, generated closure, both CUDA '
                                  'extensions, import resolution and tensor2im at util.util, exactly as in S7 attempt 2',
            'contrast_review': {'path': 'outputs/audit/E05_S7_CONTRAST_DIAGNOSTIC.json',
                                'sha256': sha('outputs/audit/E05_S7_CONTRAST_DIAGNOSTIC.json'),
                                'status': json.loads((ROOT / 'outputs/audit/E05_S7_CONTRAST_DIAGNOSTIC.json').read_text())[
                                    'status'],
                                'disclosure': 'low-contrast outputs; seed 1337 close to collapse onto a near-constant '
                                              'grey image; retained under RAW-MATCHED; no threshold or filter'}}
    return doc


# ------------------------------------------------------------------------------------------------------------- E04
E04_DRAFT = 'configs/amendments/learned_bank_production_E04.DRAFT.yaml'
E04_APPROVAL = ("Owner authorization of 2026-11-25 (session instruction 'GPAT-TransferBench v1.0 - E04 Physics-STD "
                "Final Production Banks, Independent Audit, and Seven-Baseline Bank Readiness'): full E04 TRAIN "
                "synthetic-bank production, seeds 42 -> 1337 -> 2026, 8838 images per seed (26514 in total), only after "
                "all preceding qualification and governance gates pass. No generator retraining, no change of the frozen "
                "inference semantics, no downstream ResNet-18 training, TEST evaluation, DINOv3 or GPAT.")
E04_CONDITIONS = ['E04 S7 180/180 PASS', 'approved E04 scientific generation contract', 'exact final checkpoints',
                  'qualified production tooling', 'approved E04 HDD storage extension', 'frozen TRAIN pair population',
                  'correct GPU and environment', 'complete data firewall', 'independent production audit',
                  'immutable output and provenance policy']
E04_PACK_MANIFEST_SHA256 = 'db1908d56efb23758fee49f88d2158374b85572a213a269419881bd02037520b'


def precision_env_e04():
    """methods/physics_std/production_io.PRECISION_ENV read without importing the module (no numpy/yaml needed)."""
    import ast
    tree = ast.parse((ROOT / 'methods/physics_std/production_io.py').read_text())
    node = next(n for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') == 'PRECISION_ENV')
    return dict(sorted(ast.literal_eval(node.value).items()))


def authority_e04(source_commit, date, root=None):
    """The E04 production authority = the committed DRAFT template completed from the committed records (every hash
    derived here, never typed). Refuses unless the E04 S7 record passed and the E04 storage extension is activated."""
    M = 'E04'
    R = Path(root) if root is not None else ROOT

    def sha(rel):
        return hashlib.sha256((R / rel).read_bytes()).hexdigest()
    draft_raw = (R / E04_DRAFT).read_bytes()
    draft = json.loads(draft_raw)
    s7rel = lb.E04_S7_RECORD_PATH
    s7 = json.loads((R / s7rel).read_text())
    if not (s7['s7_acceptance_passed'] is True and s7['failed_gates'] == [] and s7['seeds'] == list(lb.SEEDS)
            and s7['status'] == lb.E04_S7_RECORD_STATUS and s7['gates_passed'] == s7['gates_total']):
        raise SystemExit('E04 S7 record did not pass')
    ext_p = R / lb.E04_STORAGE_EXTENSION_PATH
    if not ext_p.is_file():
        raise SystemExit('the E04 storage extension is not activated (%s missing)' % lb.E04_STORAGE_EXTENSION_PATH)
    ext = json.loads(ext_p.read_text())
    if not (ext.get('status') == 'OWNER_APPROVED' and ext.get('approved') is True
            and (ext.get('activation_verification') or {}).get('ok') is True):
        raise SystemExit('the E04 storage extension is not an activated owner approval')
    contract_sha = sha(lb.E04_CONTRACT_PATH)
    idents = {str(s): lb.checkpoint_identity_sha256(M, s) for s in lb.SEEDS}
    if draft['contract_sha256'] != contract_sha or draft['checkpoint_identities'] != idents \
            or draft['population'] != {'family_rows': 3 * lb.N_SYN, 'rows_per_seed': lb.N_SYN} \
            or draft['method'] != M or draft['seeds'] != list(lb.SEEDS):
        raise SystemExit('the E04 production DRAFT no longer matches the committed contract/checkpoints')
    s7auth = json.loads((R / lb.E04_S7_AUTHORITY_PATH).read_text())
    ex = s7auth['execution']
    ruling = json.loads((R / lb.E04_INPUT_RULING_PATH).read_text())
    storage = json.loads((R / lb.STORAGE_AUTHORITY_PATH).read_text())
    reg = storage['bank_root_registry']
    env = lb.ENVIRONMENTS[M]
    ck = {}
    for s in lb.SEEDS:
        a = lb.E04_CHECKPOINTS[(M, s)]
        ck[str(s)] = {'run_id': a['run_id'], 'iteration': 150000, 'rule': lb.CHECKPOINT_RULES[M], 'sha256': a['sha256'],
                      'checkpoint_identity_sha256': idents[str(s)], 'seed_closure_commit': a['seed_closure_commit'],
                      'family_closure_commit': a['family_closure_commit'],
                      'path': '<runtime_root>/runs/m6/E04/seed_%d/%s' % (s, '|'.join(lb.CHECKPOINT_FILES[M].values())),
                      'restore': '102 generator variables (72 trainable + 30 BN moving statistics) by their trained '
                                 'names; no discriminator variable, optimizer slot or counter is restored'}
    return {
        'schema_version': lb.PRODUCTION_AUTHORITY_SCHEMA,
        'record': 'LEARNED_BANK_PRODUCTION_AUTHORITY_E04',
        'status': 'OWNER_APPROVED', 'approved': True, 'date': date, 'approval': E04_APPROVAL,
        'conditions': E04_CONDITIONS,
        'derived_from': {'path': E04_DRAFT, 'sha256': sha256_bytes(draft_raw)},
        'method': M, 'display_name': lb.E04_REPORTING_LABEL, 'fidelity': lb.E04_FIDELITY,
        'fidelity_note': 'controlled geometry/depth reconstruction; not an official native Physics-STD reproduction',
        'seeds': list(lb.SEEDS), 'seed_order': list(lb.SEEDS),
        'method_order_in_program': ['E06c', 'E03', 'E06b', 'E05', 'E04'],
        'contract_path': lb.E04_CONTRACT_PATH, 'contract_sha256': contract_sha,
        'decisions': dict(lb.E04_APPROVED_DECISIONS),
        's7_acceptance_record': {'path': s7rel, 'sha256': sha(s7rel), 'status': s7['status'],
                                 'execution_commit': s7['execution_commit'],
                                 'gates': '%d/%d' % (s7['gates_passed'], s7['gates_total'])},
        's7_launch_authority': {'path': lb.E04_S7_AUTHORITY_PATH, 'sha256': sha(lb.E04_S7_AUTHORITY_PATH)},
        'input_allowlist_ruling': {'path': lb.E04_INPUT_RULING_PATH, 'sha256': sha(lb.E04_INPUT_RULING_PATH),
                                   'decision': ruling['decision']},
        'e04_records': {'seed_2026_completion': {'path': lb.E04_COMPLETION_RECORD, 'sha256': sha(lb.E04_COMPLETION_RECORD)},
                        'family_closure': {'path': lb.E04_FAMILY_CLOSURE_RECORD,
                                           'sha256': sha(lb.E04_FAMILY_CLOSURE_RECORD)},
                        'final_checkpoints_csv': {'path': 'outputs/audit/E04_THREE_SEED_FINAL_CHECKPOINTS.csv',
                                                  'sha256': sha('outputs/audit/E04_THREE_SEED_FINAL_CHECKPOINTS.csv')}},
        'frozen_config': {'path': lb.METHOD_CONFIGS[M][0], 'sha256': lb.METHOD_CONFIGS[M][1]},
        'checkpoints': ck,
        'environment': {'name': env['name'], 'python': env['python'], 'tensorflow': ex['tensorflow'],
                        'lock': {'path': env['lock'][0], 'sha256': env['lock'][1]},
                        'precision_env': precision_env_e04(),
                        'session': 'methods/physics_std/production_runner.session_config (E04 training session '
                                   'configuration, verified against environments/e04_training.runtime.json)',
                        'generic_python_forbidden': True},
        'gpu': {'model': ex['gpu_name'], 'uuid': ex['gpu_uuid'], 'driver_observed': ex['driver'],
                'driver_policy': ex['driver_policy'], 'cuda_visible_devices': '0',
                'precision': 'E04-Q5 A_GPU_FP32_BITWISE_REPLAY (the E04 training precision environment)'},
        'storage_authority': {'path': lb.STORAGE_AUTHORITY_PATH, 'sha256': sha(lb.STORAGE_AUTHORITY_PATH),
                              'bank_root': reg['bank_root'], 'filesystem_uuid': reg['filesystem_uuid'],
                              'mount_point': reg['mount_point'], 'layout': lb.E04_BANK_LAYOUT,
                              'banks': {str(s): '%s/E04/seed_%d' % (reg['bank_root'], s) for s in lb.SEEDS},
                              'live_preflight': 'methods.common.learned_bank.storage_preflight() all true before every '
                                                'launch (load_production_authority) and in every audit'},
        'storage_extension': {'path': lb.E04_STORAGE_EXTENSION_PATH, 'sha256': sha(lb.E04_STORAGE_EXTENSION_PATH)},
        'inference': {
            'per_pair': 'one source-spoof and one target-live TRAIN face; batch size 1; one OS process per seed',
            'checkpoint': 'the exact iteration_150000 final checkpoint of the matching seed',
            'generator': 'network.generator with InferenceLayers (tf.layers.batch_normalization training=False, the '
                         'restored moving statistics) [E04-Q1 A_MOVING_STATISTICS_PER_PAIR]',
            'geometry': 'frozen e04_feed_pack_v1 Q140 landmarks; dense_offsets(source_xy=spoof Q140, target_xy=live '
                        'Q140); no landmark re-estimation',
            'warp': 'methods/physics_std/trace_transfer.py warp on CPU (bilinear, border clamp)',
            'synthesis': 'out = (1-P_w)*(I_live + B~_w + C~_w + T_w) + P_w*I_source_w, training_graph.py:32-39 '
                         '[E04-Q2 A_PAPER_SPOOF_SYNTHESIS]; no harder-sample augmentation, no stochastic perturbation',
            'auxiliary_outputs': 'none [E04-Q4 A_NONE]',
            'no_training': 'no optimizer, gradient, UPDATE_OPS, assignment or checkpoint write; finalized TF graph'},
        'population': {'kind': 'PRODUCTION_FROZEN_TRAIN_POPULATION', 'rows_per_seed': lb.N_SYN,
                       'family_rows': 3 * lb.N_SYN, 'pairs': {'path': lb.PAIRS_PATH, 'sha256': lb.PAIRS_SHA256},
                       'feed_pack': {'name': 'e04_feed_pack_v1', 'manifest_sha256': E04_PACK_MANIFEST_SHA256},
                       'order': 'every frozen pairs_train_v1 row in canonical ascending pair_id order, global '
                                'positions 0..8837',
                       'identity': 'row_id = pair_id; target_live_id, source_spoof_id and dataset preserved per row',
                       'faces': 'TRAIN only via TrainFaceReader (feed-pack TRAIN allowlist, O_NOATIME, atime stat '
                                'check, M2 face SHA256); %d O_NOATIME reads per bank' % (2 * lb.N_SYN),
                       'cross_check': 'split_v1 and M2 accounting of every used face and of the whole feed pack by the '
                                      'independent production auditor (owner ruling)',
                       'no_additional_pairs': True, 'pairs_val_opened': False},
        'rng': 'none: E04 bank inference is deterministic (BN inference mode, batch 1, no sampling or augmentation)',
        'conversion_rule': lb.CONVERSION_RULES[M],
        'expected_outputs': {'per_seed': lb.N_SYN, 'family': 3 * lb.N_SYN, 'format': '256x256x3 uint8 RGB lossless PNG '
                             '(PIL compress_level 6, optimize False, no metadata); mode 0444; bank and seed dirs 0555',
                             'label': 'spoof', 'inclusion': 'RAW-MATCHED: every numerically valid result is included',
                             'failures_allowed': 'only NaN/Inf raw outputs are counted failures (the frozen rule); '
                                                 'production expects 0 (any failure stops the family for owner review)'},
        'code_authority': {
            'source_commit': source_commit,
            'execution_commit': 'the commit that adds this record; it may differ from source_commit only in '
                                'configs/amendments/learned_bank_production_E04.yaml and outputs/audit/**',
            'execution_checkout': '~/workdir/m8_learned_banks/clones/prod_E04_<execution_commit> (fresh clean clone from '
                                  'a git bundle; never the E04 training checkout 18ba00e); third_party/source_cache/stdn '
                                  'copied from the GPU checkout',
            'cli': 'tools/learned_bank.py production --method E04 --seed <S> --runtime-root <runtime_root> (one OS '
                   'process per bank)',
            'launch_scripts': 'tools/learned_bank_production_scripts/m8prod_{launch,wait,post}.sh',
            'audit_tool': 'tools/learned_bank_production_audit.py (E04 branch)'},
        'production_audit': {
            'record': 'outputs/audit/LEARNED_BANK_PRODUCTION_E04_SEED<S>.json',
            'evidence': 'outputs/audit/LEARNED_BANK_PRODUCTION/E04_s<S>/{collected.json,bank_manifest.jsonl.gz,'
                        'preflight.json}',
            'family_closure': 'outputs/audit/LEARNED_BANK_PRODUCTION_E04_FAMILY_CLOSURE.json',
            'requires': ['exactly 8838 rows and 8838 outputs, 0 failures, no missing or duplicate pair_id',
                         'rows equal pairs_train_v1 in canonical order (row_index = global position)',
                         'every face TRAIN in split_v1 with its M2 face SHA256; whole-feed-pack cross-check',
                         'BN inference mode, 102 restored generator variables, 0 gradient / optimizer / mutating ops',
                         'clip-truncate conversion; PNG structure, file and pixel SHA256 of every output by the '
                         'library reader AND an independent walk', 'finite raw outputs',
                         'checkpoint identity before and after (sha256 + mtime) for all 15 approved checkpoints',
                         'TRAIN-only reads, O_NOATIME, zero VAL/TEST/TRAIN face atime or mtime change',
                         'S7-to-production byte and pixel equivalence of the 186 toy rows against S7 runs A and B, '
                         'matched by global position and pair_id from the manifests',
                         'host, GPU, environment lock, storage UUID, registered path', 'E01/E02 banks unchanged',
                         'every other learned bank unchanged', 'audit committed and pushed before the next launch']},
        'reproducibility': {
            'standard': 'E04-Q5 A_GPU_FP32_BITWISE_REPLAY (passed S7, bitwise A/B for all seeds) and byte equality of '
                        'the 186 S7 toy rows with the production rows',
            'full_bank_replay': 'not required by the contract and not performed; never substituted by a procedural '
                                'or approximate check', 'procedural_fallback': 'NOT AUTHORIZED'},
        'test_firewall': {'val_test_reads': 0, 'reader': 'TrainFaceReader TRAIN allowlist, O_NOATIME, atime check per read',
                          'atime_ledger': 'stat-only ledger of all canonical faces (TRAIN/VAL/TEST) before and after '
                                          'each bank; no change allowed'},
        'resource_preflight': {'required': True, 'gpu_free_floor_gib': 4, 'ram_floor_gib': 4, 'hdd_free_floor_gib': 20,
                               'one_learned_bank_process_at_a_time': True, 'unrelated_gpu_jobs': 'recorded, never touched',
                               'measured_s7': '582 MiB GPU per process; about 0.65 GiB PNG per seed'},
        'fail_closed': ['any gate failure stops the E04 family; evidence and partial outputs are preserved untouched',
                        'no automatic rerun and no rerun into an existing bank directory',
                        'HDD disconnect, remount or UUID change: stop immediately, preserve partial evidence',
                        'S7-to-production mismatch: stop the family; nothing is relaxed and no reference is changed',
                        'GPU contention/OOM: never touch other jobs'],
        'scientific_disclosures': [
            'CONTROLLED_ADAPTATION: Physics-STD (controlled geometry/depth reconstruction); not an official native '
            'Physics-STD reproduction',
            'bank inference uses the stored BN moving statistics (batch 1); training-time synthesis used 8-image batch '
            'statistics',
            'float->uint8 clip-truncate follows the ECCV20-STDN predecessor (reference only)',
            'S7 observed one seed-42 pair with raw output about [-1.11, 1.68]; the approved conversion clips to [0, 1] '
            'before the uint8 cast; no dynamic rescaling, replacement or quality filter (RAW-MATCHED)'],
        'not_retroactive': 'authorizes only banks launched after this record is committed; no E04 production bank or '
                           'E04 bank directory existed when it was derived',
        'authorizes': {'production_bank_generation': True, 'methods': [M], 'generator_retraining': False,
                       'checkpoint_change': False, 'inference_semantics_change': False, 'quality_filtering': False,
                       'val_or_test_access': False, 'other_method_production': False, 'downstream_training': False,
                       'test_evaluation': False, 'dinov3': False, 'gpat': False, 'method_status_update': False},
    }


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', required=True, choices=('E06c', 'E03', 'E06b', 'E05', 'E04'))
    ap.add_argument('--source-commit', required=True)
    ap.add_argument('--date', required=True)
    a = ap.parse_args()
    out = ROOT / lb.PRODUCTION_AUTHORITY_TEMPLATE.format(method=a.method)
    if out.exists():
        raise SystemExit('refusing to overwrite %s' % out)
    doc = authority_e04(a.source_commit, a.date) if a.method == 'E04' else authority(a.method, a.source_commit, a.date)
    out.write_text(json.dumps(doc, indent=1) + '\n')
    print(out, hashlib.sha256(out.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
