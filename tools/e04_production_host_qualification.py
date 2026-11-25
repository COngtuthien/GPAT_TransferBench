"""Read-only GPU-host qualification of the E04 production auditor's host functions, before any E04 bank exists.

    env -i HOME=$HOME PATH=/usr/bin:/bin LANG=C.UTF-8 CUDA_VISIBLE_DEVICES= ~/miniconda3/envs/gpat-m6-e05/bin/python \
        -I -B - --repo <clone> < tools/e04_production_host_qualification.py > host_qualification.json

Runs, in the auditor interpreter, on the real E04 S7 seed-42 directories (never written):
  - independent_verification and png_verification_e04 on S7 run A against its own manifest (expect 0 problems, one
    canonical PNG structure for all 186 files);
  - s7_equivalence_e04 with S7 run B standing in for a production bank (expect 186 rows matched by global position and
    pair_id, 0 byte and 0 pixel mismatches, 186 pixel-equal arrays; every row is a field mismatch only because run B's
    mode is S7_TOY, which the production gate rejects -- the negative half of the check);
  - the whole-feed-pack split_v1 / M2 cross-check (s7 tool e04_lineage);
  - all 15 checkpoints re-hashed (checkpoints_all_e04).
The E04 production preflight itself is run separately (audit tool `preflight --method E04 --seed 42`); before the
owner activations it must report the authority refused.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--repo', required=True)
args = ap.parse_args()
repo = Path(args.repo)


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, str(repo / rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


A = load('prod_audit', 'tools/learned_bank_production_audit.py')
lb = A.lb_module(repo)
s7 = A.s7_tool(repo)
rec = json.loads((repo / 'outputs/audit/LEARNED_BANK_S7_E04.json').read_text())
dirs = rec['outputs']['42']['run_dirs']
rows_a = [json.loads(x) for x in (Path(dirs['A']) / lb.MANIFEST).read_text().splitlines()]
rows_b = [json.loads(x) for x in (Path(dirs['B']) / lb.MANIFEST).read_text().splitlines()]
iv = A.independent_verification(lb, Path(dirs['A']), rows_a)
pv = A.png_verification_e04(s7, Path(dirs['A']), rows_a)
eq = A.s7_equivalence_e04(lb, s7, repo, 42, Path(dirs['B']), rows_b)
lin = s7.e04_lineage(repo, {})
ck = A.checkpoints_all_e04(lb)
canon = s7.canonical(dict(s7.E04_PNG_STRUCTURE, scanline_filters_ok=True))
expect = {
    'independent_clean': iv['problem_count'] == 0 and iv['files'] == 186,
    'png_structure_canonical': pv['problem_count'] == 0 and pv['structures'] == {canon: 186},
    's7_equivalence_bytes_and_pixels': all(v['rows_matched'] == 186 and v['byte_mismatch_count'] == 0
                                           and v['pixel_mismatch_count'] == 0 and v['pixel_arrays_equal'] == 186
                                           and v['identity_mismatches'] == [] for v in eq['runs'].values()),
    's7_equivalence_rejects_non_production_mode': all(v['field_mismatch_count'] == 186 for v in eq['runs'].values()),
    'feed_pack_crosscheck': lin['feed_pack_crosscheck']['samples_mismatching_count'] == 0
    and lin['feed_pack_crosscheck']['pairs_equal_pairs_train_v1'] is True
    and lin['feed_pack_crosscheck']['samples'] == 12668,
    'checkpoints_15_ok': len(ck) == 15 and all(v['ok'] for v in ck.values()),
}
print(json.dumps({'phase': 'host_qualification', 'utc': A.utc_now(), 'repo': str(repo), 'git': A.git_state(repo),
                  'python': sys.version.split()[0], 'expectations': expect, 'all_expected': all(expect.values()),
                  'independent': {k: iv[k] for k in ('files', 'files_verified', 'problem_count', 'modes')},
                  'png_verification': pv, 's7_equivalence': {L: {k: v[k] for k in (
                      'rows', 'rows_matched', 'byte_mismatch_count', 'pixel_mismatch_count', 'pixel_arrays_equal',
                      'field_mismatch_count', 'manifest_sha256', 'dir')} for L, v in eq['runs'].items()},
                  'feed_pack': lin['feed_pack_crosscheck'],
                  'checkpoints': {k: v.get('identity') for k, v in ck.items()}}, sort_keys=True, default=str))
