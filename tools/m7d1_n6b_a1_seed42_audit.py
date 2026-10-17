"""M7D1-N6B: read-only completion audit of the fresh M7D1-A1 GPAT-B0 / E08 / seed 42 scientific run
(run_id 92e91ddb78f1e242, launch authority 4466e12, curriculum GPAT-TransferBench-v1.0+M7D1-A1).

Layered on tools/m7d1_close_seed42_audit.py (the v1.0 seed42 closure tool), which is reused unchanged for the generic
completion / AMP / firewall / recovery / EMA collection and gates. This tool adds:

  * the full M7D1-A1 schedule audit: every COMPLETE record's lambda_adv / s_hf / lambda_con / lambda_spec / LR and
    epoch/group mapping compared by exact float equality against the canonical production implementation
    methods.gpat.runtime_contract.curriculum_a1 / main_lr (no re-derived float expression);
  * curriculum_id provenance on the manifest, recovery checkpoint and every EMA candidate;
  * per-epoch scientific-health trends and the v1.0 identity-collapse signature scan (non-gating);
  * read-only listings of runs/m7_a1 (only E08/seed_42 expected) and of the immutable historical v1.0 root;
  * the N6B EMA manifest (epoch, path, size, sha256) and evidence files under outputs/audit/.

Phases:
  collect  (GPU host, gpat-m7-gpu env, `python -I - collect --repo <gpu repo> < this file`) -- read-only, prints JSON.
  build    (repository) -- gates + writes outputs/audit/M7D1_A1_E08_SEED42_{COMPLETION.json,EMA_MANIFEST.txt}.

No selection, VAL/TEST, bank, downstream or training code is run.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

MILESTONE = 'M7D1-N6B'
CLASSIFICATION = 'SCIENTIFIC_TRAINING_COMPLETION_AUDIT'
STATUS_OK = 'M7D1_N6B_A1_SEED42_COMPLETED'
STATUS_BLOCKED = 'M7D1_N6B_A1_SEED42_NEEDS_REVIEW'
RUNTIME = '/home/student20261/workdir/GPAT_TransferBench_runtime'
EXPECTED_OVERRIDES = {
    'run_id': '92e91ddb78f1e242',
    'code_commit': '4466e1258c24e4a5d448dd5b30e8d8768dfe7174',
    'run_root': f'{RUNTIME}/runs/m7_a1/E08/seed_42',
}
CURRICULUM_ID = 'GPAT-TransferBench-v1.0+M7D1-A1'
AMENDMENT_ID = 'M7D1-A1'
AMENDMENT_RECORD = 'outputs/audit/M7D1_A1_ADVERSARIAL_CURRICULUM_AMENDMENT.json'
SPEC_PATH = 'docs/spec/GPAT_TransferBench_v1_0_Frozen_Specification_2026.docx'
SPEC_SHA256 = 'f7d2371682ad9504f59f0c6984e4b1a741f855f1b1f60838f9d60b160281489e'
HISTORICAL_ROOT = f'{RUNTIME}/runs/m7/E08/seed_42'
HISTORICAL_LISTING_DIGEST = 'ac64a06351fa8eb45e3efbef8fc64d297d6d3f0d78ee62f8b3cfcd26b6de893a'
HISTORICAL_FILE_COUNT = 59
A1_ROOT = f'{RUNTIME}/runs/m7_a1'
# owner-listed checkpoints of the A1 lambda_adv schedule (production values, written as repr literals)
LAMBDA_ADV_CHECKPOINTS = {5525: 0.0, 5526: 0.0, 5527: 4.528985507246377e-05, 6629: 0.04995471014492754,
                          6630: 0.05, 6631: 0.05, 16575: 0.05, 16576: 0.10, 16577: 0.10, 66300: 0.10}
CURRICULUM_FIELDS = (('lambda_adv', 'lambda_adv'), ('s_hf', 'scale_hf'), ('lambda_con', 'lambda_con'),
                     ('lambda_spec', 'lambda_spec'))
HEALTH_KEYS = ('D_total', 'D_real', 'D_fake', 'gadv', 'G_total', 'budget', 'bg', 'lm', 'low', 'id', 'parse',
               'artcon', 'spec', 'tv')
CLIP = 1.0
# v1.0 seed42 collapse (M7D1-N2 F01): budget pinned at the artifact_min hinge 0.01 (A ~ 0), lm = 0, bg < 1.4e-7
COLLAPSE_SIGNATURE = {'budget_min': 0.0099, 'bg_max': 1e-6, 'lm_max': 1e-6}
OLD_TOOL = 'tools/m7d1_close_seed42_audit.py'
EVIDENCE = {'completion': 'outputs/audit/M7D1_A1_E08_SEED42_COMPLETION.json',
            'ema_manifest': 'outputs/audit/M7D1_A1_E08_SEED42_EMA_MANIFEST.txt',
            'report': 'outputs/audit/M7D1_A1_E08_SEED42_COMPLETION.md',
            'collected': 'outputs/audit/M7D1_A1_E08_SEED42_COLLECTED.json',
            'sha256': 'outputs/audit/M7D1_A1_E08_SEED42_EVIDENCE_SHA256.txt'}


def load_old(repo):
    spec = importlib.util.spec_from_file_location('m7d1_close_v1', Path(repo) / OLD_TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.EXPECTED = dict(mod.EXPECTED, **EXPECTED_OVERRIDES)   # module-local instance; the v1.0 tool file is unchanged
    return mod


def ema_manifest_text(rows):
    """Deterministic N6B EMA manifest sorted by epoch: `epoch size sha256 path` (path relative to the run root)."""
    rows = sorted(rows, key=lambda r: r['epoch'])
    head = '# M7D1-N6B GPAT-B0/E08/seed42 run 92e91ddb78f1e242 EMA candidates: epoch size_bytes sha256 path\n'
    return head + ''.join(f"{r['epoch']:02d} {r['file_size_bytes']} {r['sha256']} "
                          f"checkpoints/ema_candidates/{r['filename']}\n" for r in rows)


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def listing_digest(root):
    """Same convention as earlier sessions: `find . -type f -printf "%P %s %T@\\n" | sort | sha256sum`."""
    lines = []
    for dirpath, _, filenames in os.walk(root):
        for n in filenames:
            p = Path(dirpath) / n
            st = p.stat()
            lines.append(f'{p.relative_to(root)} {st.st_size} {st.st_mtime_ns // 10**9}.{st.st_mtime_ns % 10**9:09d}0')
    return sha256_bytes(''.join(f'{x}\n' for x in sorted(lines)).encode()), len(lines)


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * (len(xs) - 1) + 0.5))] if xs else None


# ============================================================================ collect (GPU host, read-only)
def scan_a1(path, rc):
    mism, checkpoints = {}, {}
    per_epoch = {}
    sig_updates, run_len, longest, longest_end = [], 0, 0, None
    first_complete_utc = last_complete_utc = None
    with open(path, 'rb') as f:
        for raw in f:
            d = json.loads(raw)
            if d.get('record_type') != 'optimizer_group' or d.get('group_status') != 'COMPLETE':
                continue
            u = d['global_update']
            first_complete_utc = first_complete_utc or d.get('utc')
            last_complete_utc = d.get('utc')
            cur = rc.curriculum_a1(u)
            exp = {k: cur[ck] for ck, k in CURRICULUM_FIELDS}
            exp['learning_rate'] = rc.main_lr(u)
            exp['epoch'] = (u - 1) // rc.UPDATES_PER_EPOCH + 1
            exp['group'] = (u - 1) % rc.UPDATES_PER_EPOCH + 1
            for k, v in exp.items():
                got = d.get(k)
                if not (type(got) is type(v) and got == v):
                    mism.setdefault(k, []).append([u, got, v])
            if u in LAMBDA_ADV_CHECKPOINTS:
                checkpoints[str(u)] = {'logged': d['lambda_adv'], 'production': cur['lambda_adv'],
                                       'logged_hex': float.hex(d['lambda_adv']), 'stage': cur['stage']}
            tl = d['train_losses']
            e = per_epoch.setdefault(d['epoch'], {k: [] for k in HEALTH_KEYS + ('G_grad_norm', 'D_grad_norm')})
            for k in HEALTH_KEYS:
                e[k].append(tl[k])
            e['G_grad_norm'].append(d['G_grad_norm'])
            e['D_grad_norm'].append(d['D_grad_norm'])
            sig = (tl['budget'] >= COLLAPSE_SIGNATURE['budget_min'] and tl['bg'] < COLLAPSE_SIGNATURE['bg_max']
                   and tl['lm'] < COLLAPSE_SIGNATURE['lm_max'])
            if sig:
                sig_updates.append(u)
                run_len += 1
                if run_len > longest:
                    longest, longest_end = run_len, u
            else:
                run_len = 0
    trends = {}
    for ep in sorted(per_epoch):
        e = per_epoch[ep]
        row = {f'{k}_median': statistics.median(e[k]) for k in HEALTH_KEYS + ('G_grad_norm', 'D_grad_norm')}
        row.update({'G_grad_norm_min': min(e['G_grad_norm']), 'G_grad_norm_max': max(e['G_grad_norm']),
                    'D_grad_norm_max': max(e['D_grad_norm']),
                    'G_clip_fraction': sum(x > CLIP for x in e['G_grad_norm']) / len(e['G_grad_norm']),
                    'D_clip_fraction': sum(x > CLIP for x in e['D_grad_norm']) / len(e['D_grad_norm']),
                    'budget_hinge_active_fraction': sum(x > 0 for x in e['budget']) / len(e['budget']),
                    'bg_min': min(e['bg']), 'lm_min': min(e['lm']), 'budget_max': max(e['budget']),
                    'updates': len(e['gadv'])})
        trends[str(ep)] = row
    return {
        'curriculum_mismatches': {k: {'count': len(v), 'first': v[:20]} for k, v in mism.items()},
        'curriculum_mismatch_total': sum(len(v) for v in mism.values()),
        'lambda_adv_checkpoints': checkpoints,
        'per_epoch_health': trends,
        'collapse_signature': {'definition': COLLAPSE_SIGNATURE, 'matching_updates': len(sig_updates),
                               'first_matching': sig_updates[:20], 'longest_consecutive_run': longest,
                               'longest_run_end_update': longest_end},
        'first_complete_utc': first_complete_utc, 'last_complete_utc': last_complete_utc,
    }


def collect(run_root, repo):
    sys.path.insert(0, str(repo))
    from methods.gpat import runtime_contract as rc
    old = load_old(repo)
    root = Path(run_root)
    before_tree, before_small = old.tree_snapshot(root), old.small_hashes(root)
    hist_before = listing_digest(HISTORICAL_ROOT)
    base = old.collect(run_root)
    a1 = scan_a1(root / 'metrics.jsonl', rc)
    a1_listing = sorted(str(p.relative_to(A1_ROOT)) for p in Path(A1_ROOT).rglob('*'))
    hist_after = listing_digest(HISTORICAL_ROOT)
    after_tree, after_small = old.tree_snapshot(root), old.small_hashes(root)
    rc_src = Path(repo) / 'methods' / 'gpat' / 'runtime_contract.py'
    return dict(base, a1=a1, a1_root_listing=a1_listing,
                historical_root={'path': HISTORICAL_ROOT, 'listing_digest_before': hist_before[0],
                                 'listing_digest_after': hist_after[0], 'file_count': hist_after[1]},
                runtime_contract_sha256=old.sha256_file(rc_src), production_curriculum_is_a1=rc.curriculum is
                rc.curriculum_a1, production_curriculum_id=rc.CURRICULUM_ID,
                run_root_listing_digest=listing_digest(root)[0],
                run_root_unchanged_whole_audit=before_tree == after_tree and before_small == after_small)


# ============================================================================ build (repository)
def build(c, repo_root, authority):
    old = load_old(repo_root)
    doc, _ = old.build(c, repo_root, authority)
    gates = old.Gates()
    gates.results = list(doc['gates'])
    G = gates
    a1, man, rec = c['a1'], c['run_manifest'], c['recovery']
    repo = Path(repo_root)

    # H. A1 identity / curriculum
    G('H.manifest_curriculum_id', man['provenance'].get('curriculum_id') == CURRICULUM_ID)
    G('H.recovery_curriculum_id', (rec.get('provenance') or {}).get('curriculum_id') == CURRICULUM_ID)
    G('H.candidate_curriculum_id', all(cd['metadata'].get('curriculum_id') == CURRICULUM_ID for cd in c['candidates']))
    G('H.production_curriculum_is_a1', c['production_curriculum_is_a1'] and c['production_curriculum_id'] == CURRICULUM_ID)
    G('H.runtime_contract_matches_repo', c['runtime_contract_sha256']
      == old.sha256_file(repo / 'methods' / 'gpat' / 'runtime_contract.py'))
    G('H.full_schedule_zero_mismatch', a1['curriculum_mismatch_total'] == 0, a1['curriculum_mismatches'])
    cps = a1['lambda_adv_checkpoints']
    G('H.lambda_adv_checkpoints', sorted(int(k) for k in cps) == sorted(LAMBDA_ADV_CHECKPOINTS) and all(
        cps[str(u)]['logged'] == v and cps[str(u)]['production'] == v for u, v in LAMBDA_ADV_CHECKPOINTS.items()))
    G('H.fresh_from_u1', c['metrics']['update_min'] == 1 and a1['first_complete_utc'] >= '2026-10-10T15:26:55Z')
    G('H.spec_sha256', old.sha256_file(repo / SPEC_PATH) == SPEC_SHA256)
    G('H.amendment_record_present', json.loads((repo / AMENDMENT_RECORD).read_text())['amendment_id'] == AMENDMENT_ID)

    # I. roots
    G('I.a1_root_only_e08_seed42', all(p == 'E08' or p.startswith('E08/seed_42') for p in c['a1_root_listing']))
    h = c['historical_root']
    G('I.historical_root_unchanged', h['listing_digest_before'] == h['listing_digest_after']
      == HISTORICAL_LISTING_DIGEST and h['file_count'] == HISTORICAL_FILE_COUNT, h)
    G('I.run_root_unchanged_whole_audit', c['run_root_unchanged_whole_audit'])

    rows = doc['ema_candidates']['candidates']
    manifest_text = ema_manifest_text(rows)
    retries = doc['amp_retry_summary']['events']
    retry_table = [{'global_update': r['global_update_attempted'], 'epoch': r['epoch'], 'group': r['group'],
                    'retries': len(r['retry_numbers']), 'offending_optimizers': r['offending_optimizers'],
                    'accepted_D_scale': r['complete_D_scale_before'], 'accepted_G_scale': r['complete_G_scale_before']}
                   for r in retries]
    owner = {o: sum(1 for r in retries if o in r['offending_optimizers']) for o in ('D_OPT', 'G_OPT')}
    trends = a1['per_epoch_health']
    small = c['small_file_sha256']
    doc.update({
        'milestone': MILESTONE, 'status': STATUS_OK if not G.failed else STATUS_BLOCKED, 'failed_gates': G.failed,
        'gates': G.results, 'launch_authority_commit': EXPECTED_OVERRIDES['code_commit'],
        'curriculum_id': CURRICULUM_ID, 'amendment_id': AMENDMENT_ID, 'amendment_record': AMENDMENT_RECORD,
        'frozen_specification': {'path': SPEC_PATH, 'sha256': SPEC_SHA256},
        'a1_schedule_audit': {'implementation': 'methods.gpat.runtime_contract.curriculum_a1 + main_lr (exact float ==, '
                                                'type-checked)',
                              'runtime_contract_sha256': c['runtime_contract_sha256'],
                              'fields': [k for _, k in CURRICULUM_FIELDS] + ['learning_rate', 'epoch', 'group'],
                              'updates_checked': c['metrics']['record_type_counts']['optimizer_group_complete'],
                              'mismatch_total': a1['curriculum_mismatch_total'],
                              'mismatches': a1['curriculum_mismatches'],
                              'lambda_adv_checkpoints': cps},
        'amp_retry_table': retry_table,
        'amp_retry_ownership_update_counts': owner,
        'scientific_health': {'per_epoch': trends, 'collapse_signature': a1['collapse_signature'],
                              'clip_threshold': CLIP,
                              'grad_norm_semantics': 'clip_grad_norm_ return value: total norm after unscale, before '
                                                     'clipping',
                              'mask_logging': 'mask statistics (M_mean) are not in metrics.jsonl; mask closure is '
                                              'assessed only indirectly via the budget/bg/lm collapse signature'},
        'run_window_utc': [a1['first_complete_utc'], a1['last_complete_utc']],
        'historical_root': h, 'a1_root_listing': c['a1_root_listing'],
        'ema_candidates': dict(doc['ema_candidates'], sha256_manifest_file=EVIDENCE['ema_manifest'],
                               sha256_manifest_digest=sha256_bytes(manifest_text.encode()),
                               manifest_format='epoch size_bytes sha256 path, one header line, sorted by epoch',
                               best_selected=None),
        'run_root_integrity': {'metrics_jsonl_sha256': c['metrics_sha256'],
                               'run_manifest_sha256': small['run_manifest.json'],
                               'run_summary_sha256': small['run_summary.json'],
                               'checkpoint_index_sha256': small['checkpoint_index.json'],
                               'resolved_config_sha256': small['resolved_config.yaml'],
                               'recovery_sha256': rec['sha256'],
                               'ema_manifest_digest': sha256_bytes(manifest_text.encode()),
                               'file_count': len(c['run_root_tree_after']),
                               'listing_digest': c['run_root_listing_digest'],
                               'listing_digest_convention': 'find . -type f -printf "%P %s %T@\\n" | sort | sha256sum',
                               'tree_sha256': sha256_bytes(json.dumps(c['run_root_tree_after']).encode())},
        'confirmations': dict(doc['confirmations'], n6c_started=False, best_checkpoint_selected=False,
                              historical_root_modified=False),
    })
    doc['provenance']['amp_amendment_binding'] = ('via code commit 4466e12 (amendment present since 058e976) and '
                                                  'run_manifest.amp_policy=ATOMIC_AMP_BACKOFF_RETRY')
    doc['provenance']['curriculum_id'] = man['provenance'].get('curriculum_id')
    doc['recovery_checkpoint']['not_selection'] = 'RECOVERY_ONLY; never a scientific selected checkpoint'
    return doc, manifest_text


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('collect')
    a.add_argument('--run-root', default=EXPECTED_OVERRIDES['run_root'])
    a.add_argument('--repo', required=True)
    b = sub.add_parser('build')
    b.add_argument('--collected', required=True)
    b.add_argument('--authority', required=True)
    b.add_argument('--observations', default=None)
    b.add_argument('--repo-root', default=str(Path(globals().get('__file__', 'tools/x')).resolve().parents[1]))
    args = ap.parse_args(argv)
    if args.cmd == 'collect':
        json.dump(collect(args.run_root, args.repo), sys.stdout, sort_keys=True)
        return 0
    c = json.loads(Path(args.collected).read_text())
    doc, text = build(c, args.repo_root, json.loads(Path(args.authority).read_text()))
    if args.observations:
        doc['observations'] = json.loads(Path(args.observations).read_text())
    root = Path(args.repo_root)
    (root / EVIDENCE['ema_manifest']).write_text(text)
    (root / EVIDENCE['completion']).write_text(json.dumps(doc, indent=2, sort_keys=True) + '\n')
    print(doc['status'], doc['failed_gates'])
    return 0 if doc['status'] == STATUS_OK else 1


if __name__ == '__main__':
    sys.exit(main())
