"""E04 D12 frozen-authority verification (read-only; no session, no RunContext, no write outside stdout).

    python -B tools/e04_d12_authority_check.py --repo <checkout> [--no-tf]

Verifies with the production code itself: production_io.verify_contract (frozen config + snapshot, A4, M6D3d, D1-D9
ruling YAML and document, environment lock, runtime record, M6D3d contract, frozen values) and identities; the frozen
CPU pre-pass (prepass.load, pinned manifest, every payload and row) and the E04 feed pack (feed_pack.load, pinned
manifest, every payload and row, pack -> pre-pass binding). Without --no-tf additionally (GPU host, E04 precision
environment): verify_environment, production_runner.prepare (unchanged graph) and the qualified M6D3e reference
construction in its own graph. Prints JSON.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True)
    ap.add_argument('--no-tf', action='store_true')
    args = ap.parse_args()
    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo))
    from methods.physics_std import production_io as pio
    out = {'repo': str(repo)}
    cfg = pio.verify_contract(repo)
    out['verify_contract'] = {'ok': True, 'method_id': cfg['method_id'], 'seeds': cfg['seeds']['experiment_seeds'],
                              'total_iterations': cfg['training']['total_iterations']}
    out['identities'] = pio.identities(repo)
    out['identities_sha256'] = hashlib.sha256(json.dumps(out['identities'], sort_keys=True).encode()).hexdigest()
    if args.no_tf:
        print(json.dumps(out, sort_keys=True))
        return 0
    from methods.physics_std import feed_pack, prepass
    from methods.physics_std.production_qualification import RUNTIME, FACES
    m, samples, _ = prepass.load(pio.prepass_root(RUNTIME), pio.PREPASS_MANIFEST_SHA256)
    out['prepass'] = {'ok': True, 'manifest_sha256': pio.PREPASS_MANIFEST_SHA256, 'counts': m['counts'],
                      'samples': len(samples), 'all_train': all(r['split'] == 'TRAIN' for r in samples)}
    pack = feed_pack.load(pio.pack_root(RUNTIME), pio.PACK_MANIFEST_SHA256)
    out['feed_pack'] = {'ok': True, 'manifest_sha256': pack.manifest_sha256, 'counts': pack.manifest['counts'],
                        'binds_prepass': pack.manifest['prepass_manifest_sha256'] == pio.PREPASS_MANIFEST_SHA256}
    errors = pio.launch_environment_errors(os.environ, pio.QUALIFICATION_SEED)
    if errors:
        raise SystemExit('launch environment: %s' % errors)
    import tensorflow as tf
    from methods.physics_std import production_runner as pr
    from methods.physics_std.production_qualification import reference_construction
    ctl, ids, live, driver, env, reasons = pr.prepare(tf, pio.QUALIFICATION, pio.QUALIFICATION_SEED, RUNTIME, FACES)
    out['environment'] = {'driver': driver, 'live': {k: live[k] for k in (
        'python', 'tensorflow', 'native_runtime', 'gpu', 'precision_env', 'session_config')}}
    out['production_construction'] = list(ctl.construction)
    out['structure'] = ctl.structure
    out['prepare_identities_equal'] = ids == out['identities']
    out['reference_construction'] = list(reference_construction(tf))
    out['construction_equals_reference'] = out['reference_construction'] == out['production_construction']
    print(json.dumps(out, sort_keys=True))
    return 0


if __name__ == '__main__':
    sys.exit(main())
