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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', required=True, choices=('E06c', 'E03', 'E06b', 'E05'))
    ap.add_argument('--source-commit', required=True)
    ap.add_argument('--date', required=True)
    a = ap.parse_args()
    out = ROOT / lb.PRODUCTION_AUTHORITY_TEMPLATE.format(method=a.method)
    if out.exists():
        raise SystemExit('refusing to overwrite %s' % out)
    out.write_text(json.dumps(authority(a.method, a.source_commit, a.date), indent=1) + '\n')
    print(out, hashlib.sha256(out.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
