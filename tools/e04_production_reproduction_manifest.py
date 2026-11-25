"""E04 Physics-STD production reproduction manifest, derived only from the committed authority, audits and closure.

    python3 tools/e04_production_reproduction_manifest.py   -> outputs/audit/E04_PRODUCTION_REPRODUCTION_MANIFEST.json
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = 'outputs/audit/E04_PRODUCTION_REPRODUCTION_MANIFEST.json'
SEEDS = (42, 1337, 2026)


def sha(rel, root=ROOT):
    return hashlib.sha256((Path(root) / rel).read_bytes()).hexdigest()


def blob_sha(root, commit, rel):
    import subprocess
    blob = subprocess.run(['git', '-C', str(root), 'show', '%s:%s' % (commit, rel)], capture_output=True,
                          check=True).stdout
    return hashlib.sha256(blob).hexdigest()


def build(root=ROOT):
    root = Path(root)
    auth_rel = 'configs/amendments/learned_bank_production_E04.yaml'
    auth = json.loads((root / auth_rel).read_text())
    clo_rel = 'outputs/audit/LEARNED_BANK_PRODUCTION_E04_FAMILY_CLOSURE.json'
    clo = json.loads((root / clo_rel).read_text())
    runs = {}
    for s in SEEDS:
        rel = 'outputs/audit/LEARNED_BANK_PRODUCTION_E04_SEED%d.json' % s
        d = json.loads((root / rel).read_text())
        r = d['reproduction']
        g = {x['gate']: x for x in d['gates']}
        runs[str(s)] = {
            'run_id': d['run_id'], 'audit_record': {'path': rel, 'sha256': sha(rel, root), 'status': d['status'],
                                                    'gates': '%d/%d' % (d['gates_passed'], d['gates_total'])},
            'source_commit': d['source_commit'], 'execution_commit': d['execution_commit'],
            'working_directory': r['working_directory'], 'command': r['command'], 'command_sha256': r['command_sha256'],
            'launch_script': r['launch_script'], 'post_script': r['post_script'], 'audit_commands': r['audit_commands'],
            'window': d['window'], 'exit_status': r['exit_status'], 'output_root': r['output_root'],
            'contract': r['contract'], 'config': r['config'], 'environment': r['environment'], 'gpu': r['gpu'],
            'inputs': r['inputs'], 'geometry': r['geometry'], 'preprocessing': r['preprocessing'],
            'bn_mode': r['bn_mode'], 'rng': r['rng'], 'conversion_rule': r['conversion_rule'],
            'checkpoint': d['checkpoint'], 'checkpoint_integrity_after': g['S.checkpoint_after']['pass'],
            'bank': d['bank'], 'per_image_hashes': 'every output_sha256 and output_pixel_sha256 in %s' %
                                                  d['bank']['manifest_evidence'],
            's7_to_production': g['S.s7_production_equivalence']['detail'],
            'atime_ledger': r['atime_ledger'], 'atime_by_split': g['F.atime_ledger']['detail'],
            'raw_output_observation': d['raw_output_observation'], 'gpu_resources': d['gpu_resources'],
            'evidence': 'outputs/audit/LEARNED_BANK_PRODUCTION/E04_s%d/{collected.json,bank_manifest.jsonl.gz,'
                        'preflight.json,host_markers.tgz}' % s}
    return {
        'schema': 'e04_production_reproduction_manifest_v1', 'method': 'E04', 'display_name': auth['display_name'],
        'fidelity': auth['fidelity'], 'authority': {'path': auth_rel, 'sha256': sha(auth_rel, root)},
        'storage_extension': auth['storage_extension'], 'contract': {'path': auth['contract_path'],
                                                                   'sha256': auth['contract_sha256']},
        's7_acceptance_record': auth['s7_acceptance_record'], 'input_allowlist_ruling': auth['input_allowlist_ruling'],
        'training_closure': auth['e04_records']['family_closure'],
        'family_closure': {'path': clo_rel, 'sha256': sha(clo_rel, root), 'status': clo['status'],
                           'gates': '%d/%d' % (clo['gates_passed'], clo['gates_total']),
                           'total_outputs': clo['total_outputs']},
        'tooling_at_execution_commit': {p: blob_sha(root, clo['execution_commit'], p) for p in ('tools/learned_bank.py', 'methods/common/learned_bank.py',
                                              'methods/physics_std/bank_adapter.py',
                                              'tools/learned_bank_production_audit.py',
                                              'tools/learned_bank_production_authority.py',
                                              'tools/learned_bank_production_scripts/m8prod_launch.sh',
                                              'tools/learned_bank_production_scripts/m8prod_wait.sh',
                                              'tools/learned_bank_production_scripts/m8prod_post.sh')},
        'host_orchestration': {'scripts_copy': '~/workdir/m8_learned_banks/prod_scripts_<execution_commit>/',
                               'chain': '~/workdir/m8prod_E04_s<S>_chain.sh (seeds 1337, 2026): wait -> dead pane -> '
                                        'post -> collect; seed 42 ran the same steps by hand (post ledger 90 s after '
                                        'end, first post attempt refused PANE NOT DEAD)'},
        'scientific_disclosures': auth['scientific_disclosures'],
        'runs': runs,
        'backup': 'the image banks exist only on the approved HDD; git holds manifests, per-image hashes and audits; '
                  'hashes are not a backup',
    }


def main():
    out = ROOT / OUT
    out.write_text(json.dumps(build(), indent=1, sort_keys=True) + '\n')
    print(out, hashlib.sha256(out.read_bytes()).hexdigest())
    return 0


if __name__ == '__main__':
    sys.exit(main())
