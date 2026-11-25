#!/usr/bin/env python3
"""E04 S7 reproduction manifest: derived deterministically from outputs/audit/LEARNED_BANK_S7_E04.json (its embedded
host collection) and committed records only -> outputs/audit/E04_S7_REPRODUCTION_MANIFEST.json.

One entry per (seed, run) with method, seeds, scientific / execution commits, owner-ruling and frozen-config SHA256,
checkpoint paths and hashes, toy population and lineage identities, environment (Python, TensorFlow, CUDA, cuDNN,
driver, GPU UUID), BN mode, geometry / warp / synthesis / conversion identities, the exact launch command, working
directory, start / end, exit code, output root, manifest / PNG / pixel SHA256s, A/B replay, atime ledger, TEST firewall,
checkpoint integrity and the audit gates. Git holds hashes only: the PNGs and checkpoints live on the GPU host.

    python3 tools/e04_bank_s7_reproduction_manifest.py [--check]
"""
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORD = 'outputs/audit/LEARNED_BANK_S7_E04.json'
OUT = 'outputs/audit/E04_S7_REPRODUCTION_MANIFEST.json'
SCIENTIFIC_TRAINING_AUTHORITY = '18ba00e94813f005876c8d6235b0b848226b2c1d'
CONTRACT_COMMIT = '3753ea4'


def sha(b):
    return hashlib.sha256(b).hexdigest()


def tool():
    spec = importlib.util.spec_from_file_location('s7a', ROOT / 'tools' / 'learned_bank_s7_audit.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def derive():
    m = tool()
    raw = (ROOT / RECORD).read_bytes()
    d = json.loads(raw)
    c, r = d['collected'], d['reproduction']
    F = m.MANIFEST_FIELDS_E04
    gates = {g['gate']: g['pass'] for g in d['gates']}
    runs = {}
    for s in map(str, d['seeds']):
        rows = [dict(zip(F, x)) for x in c['runs']['A'][s]['manifest_rows']]
        for L in ('A', 'B'):
            run, mk, pf = c['runs'][L][s], c['markers'][s][L], c['preflights'][s][L]
            p = run['provenance']
            a = p['adapter']
            runs['%s_%s' % (s, L)] = {
                'method': 'E04', 'seed': int(s), 'run': L,
                'scientific_training_authority': SCIENTIFIC_TRAINING_AUTHORITY,
                'execution_commit': d['execution_commit'], 'git_head_in_provenance': p['git_head'],
                'owner_ruling_sha256': {'contract': d['contract']['sha256'], 'contract_commit': CONTRACT_COMMIT,
                                        'input_allowlist_ruling': d['input_allowlist_ruling']['sha256'],
                                        's7_authority': d['authority']['sha256']},
                'frozen_config': {'path': m.E04_CONFIG[0], 'sha256': a['config_sha256']},
                'checkpoint': {'run_id': a['checkpoint']['run_id'], 'iteration': 150000,
                               'files': {f['path']: f['sha256'] for f in a['checkpoint']['files'].values()},
                               'identity_sha256': a['checkpoint_identity_sha256'],
                               'unchanged_before_and_after': gates['S.seed%s.checkpoint_unchanged_before_after' % s]},
                'population': {'toy_manifest': m.TOY_MANIFEST[0], 'toy_manifest_sha256': m.TOY_MANIFEST[1],
                               'pairs': len(rows), 'pair_ids_sha256': p['population']['row_ids_sha256'],
                               'lineage_sha256': sha(m.canonical([[x['pair_id'], x['row_index'], x['target_live_id'],
                                                                   x['source_spoof_id'], x['dataset'],
                                                                   x['target_face_sha256'], x['source_face_sha256']]
                                                                  for x in rows]).encode()),
                               'allowlist_source': p['population']['allowlist_source'],
                               'feed_pack_manifest_sha256': p['population']['pack_manifest_sha256']},
                'environment': {'lock': r['environment']['lock'], 'python': p['environment']['python_executable'],
                                'python_version': p['environment']['python_version'], 'framework': a['framework'],
                                'numpy': p['environment']['numpy'], 'pillow': p['environment']['pillow'],
                                'cuda_runtime': pf['environment']['probe']['environment_lock']['native_runtime'][
                                    'cudaRuntimeGetVersion'],
                                'cudnn': pf['environment']['probe']['environment_lock']['native_runtime']['cudnnGetVersion'],
                                'driver': a['driver']['observed_driver_version'], 'driver_policy': a['driver']['driver_policy'],
                                'gpu_uuid': a['driver']['gpu_uuid'], 'gpu_name': a['driver']['gpu_model'],
                                'precision_env': r['environment']['precision_env'], 'host': p['environment']['host']},
                'bn_mode': p['adapter_finalize']['bn_mode'] + ' (FusedBatchNorm is_training=False x%d)'
                           % a['graph']['fused_batch_norm'],
                'inference_graph': {k: a['graph'][k] for k in ('variable_names_sha256', 'variable_shapes_sha256',
                                                                'reachable_op_types_sha256', 'bn_statistics_reads_sha256')},
                'geometry': r['geometry'], 'warp': r['warp'], 'synthesis': r['synthesis'],
                'conversion_rule': a['conversion_rule'], 'png': r['png'],
                'launch_command': mk['command'], 'launch_command_sha256': mk['command_sha256'],
                'working_directory': c['repo'], 'start_utc': mk['launch_time'], 'end_utc': mk['end_time'],
                'exit_code': int(mk['exit_raw']), 'output_dir': run['dir'],
                'manifest_sha256': run['manifest']['sha256'],
                'png_file_and_pixel_sha256_list_sha256': sha(m.canonical(run['png_files']).encode()),
                'preflight_sha256': c['preflights_sha256'][s][L],
                'audit_gates': {g: v for g, v in gates.items() if '.seed%s.run%s.' % (s, L) in g}}
        runs['%s_compare' % s] = {
            'replay': c['replay'][s], 'cli_compare_command': c['cli_compare'][s]['command'],
            'cli_compare_rc': c['cli_compare'][s]['rc'].strip(),
            'atime_ledger': {k: v for k, v in c['atime'][s].items() if k != 'changed_paths'},
            'test_firewall': {'TEST_atime_changed': c['atime'][s]['by_split']['TEST']['atime_changed'],
                              'VAL_atime_changed': c['atime'][s]['by_split']['VAL']['atime_changed'],
                              'inputs_all_TRAIN': gates['F.seed%s.no_val_test_inputs' % s]},
            'seed_qualified': d['per_seed'][s]['qualified']}
    return {
        'schema': 'e04_s7_reproduction_manifest_v1', 'record': RECORD, 'record_sha256': sha(raw), 'status': d['status'],
        'gates': '%d/%d' % (d['gates_passed'], d['gates_total']), 'fidelity': d['fidelity'],
        'execution_clone': c['repo'], 's7_root': c['s7_root'], 'runs': runs,
        'reproduce': {
            'script': 'tools/e04_bank_scripts/e04_s7_reproduce.sh <EXECUTION_COMMIT> <SEED> <NEW_ROOT> [--dry-run]',
            'acceptance_scripts': 'tools/e04_bank_scripts/e04_s7_{clone,preflight,launch,post,collect}.sh',
            'audit': 'tools/learned_bank_s7_audit.py build --collected outputs/audit/LEARNED_BANK_S7_E04_EVIDENCE/collected.json',
            'note': 'a reproduction writes a NEW immutable root and compares byte for byte with the accepted run A; '
                    'it is GPU inference and needs an owner instruction'},
        'evidence': {'collected': 'outputs/audit/LEARNED_BANK_S7_E04_EVIDENCE/collected.json',
                     'host_markers': 'outputs/audit/LEARNED_BANK_S7_E04_EVIDENCE/host_markers.tgz',
                     'bank_metadata': 'outputs/audit/LEARNED_BANK_S7_E04_EVIDENCE/bank_metadata.tgz'},
        'disclosures': r['known_disclosures'],
        'not_a_backup': 'git holds SHA256 identities only; the 1116 PNGs and the checkpoints are stored on the GPU host '
                        'and are not backed up by this repository'}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args(argv)
    body = json.dumps(derive(), indent=1, sort_keys=True) + '\n'
    if a.check:
        ok = (ROOT / OUT).read_text() == body
        print('reproduction manifest re-derives' if ok else 'MISMATCH')
        return 0 if ok else 1
    (ROOT / OUT).write_text(body)
    print(OUT, sha(body.encode()))
    return 0


if __name__ == '__main__':
    sys.exit(main())
