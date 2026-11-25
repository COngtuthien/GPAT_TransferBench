"""M8 learned-bank production program summary (E06c, E03, E06b, E05) and reproduction manifest.

  verify (GPU host, read-only, fed on stdin; library from --repo):
    CUDA_VISIBLE_DEVICES= <gpat-m6-e05 python> -I -B - verify --repo <clone> > verify.json < tools/learned_bank_production_summary.py
      for every committed bank record: the registered bank directory exists, its manifest still hashes to the recorded
      value, it holds exactly 8838 PNGs, its provenance hash is unchanged; the execution clones exist, are clean and at
      their execution commit; the host copies of the launch scripts equal the committed scripts; every marker file named
      by the records exists; live storage preflight and capacity; E01/E02 production manifests unchanged.
  build (laptop): python3 tools/learned_bank_production_summary.py build --verified outputs/audit/LEARNED_BANK_PRODUCTION/final_host_verification.json
      -> outputs/audit/M8_LEARNED_PRODUCTION_SUMMARY.{json,md}, LEARNED_BANK_PRODUCTION_REPRODUCTION_MANIFEST.json, _SHA256.txt
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

METHODS = ('E06c', 'E03', 'E06b', 'E05')
SEEDS = (42, 1337, 2026)
N = 8838
WORKDIR = '/home/student20261/workdir'
BANK_ROOT = '/media/ext_data/student20261_gpat_m8_banks'
VERIFIED = 'outputs/audit/LEARNED_BANK_PRODUCTION/final_host_verification.json'
OUT = 'outputs/audit/M8_LEARNED_PRODUCTION_SUMMARY'
REPRO = 'outputs/audit/LEARNED_BANK_PRODUCTION_REPRODUCTION_MANIFEST.json'
SCRIPTS = ('m8prod_launch.sh', 'm8prod_wait.sh', 'm8prod_post.sh')


def sha_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def records(repo):
    out = {}
    for m in METHODS:
        for s in SEEDS:
            p = Path(repo) / ('outputs/audit/LEARNED_BANK_PRODUCTION_%s_SEED%d.json' % (m, s))
            if p.is_file():
                out[(m, s)] = json.loads(p.read_text())
    return out


def verify(repo):
    sys.path.insert(0, str(repo))
    os.chdir(str(repo))
    from methods.common import learned_bank as lb
    doc = {'phase': 'verify', 'repo': str(repo), 'banks': {}, 'clones': {}, 'markers': {}}
    doc['utc'] = subprocess.run(['date', '-u', '+%FT%TZ'], capture_output=True, text=True).stdout.strip()
    for (m, s), r in sorted(records(repo).items()):
        d = Path(r['bank']['path'])
        b = d / 'bank'
        rec = {'exists': d.is_dir(), 'manifest_sha256': sha_file(d / lb.MANIFEST) if (d / lb.MANIFEST).is_file() else None,
               'provenance_sha256': sha_file(d / lb.PROVENANCE) if (d / lb.PROVENANCE).is_file() else None,
               'png_files': len([x for x in os.listdir(b) if x.endswith('.png')]) if b.is_dir() else None,
               'entries': sorted(os.listdir(d)) if d.is_dir() else None,
               'bytes': sum((b / x).stat().st_size for x in os.listdir(b)) if b.is_dir() else None,
               'dir_mode': oct(d.stat().st_mode & 0o7777) if d.is_dir() else None}
        doc['banks']['%s|%d' % (m, s)] = rec
        clone = Path(r['reproduction']['clone'])
        if str(clone) not in doc['clones']:
            doc['clones'][str(clone)] = {
                'exists': clone.is_dir(),
                'head': subprocess.run(['git', '-C', str(clone), 'rev-parse', 'HEAD'], capture_output=True,
                                       text=True).stdout.strip(),
                'status': subprocess.run(['git', '-C', str(clone), 'status', '--porcelain', '--untracked-files=all'],
                                         capture_output=True, text=True).stdout.strip()}
        base = '%s/m8prod_%s_s%d' % (WORKDIR, m, s)
        names = [base + x for x in ('.command', '.launch_time', '.exit', '.end_time', '.pane.log', '_preflight.json',
                                     '_atime_pre.tsv', '_atime_post.tsv', '_atime_pre.time', '_atime_post.time',
                                     '_collected.json')]
        doc['markers']['%s|%d' % (m, s)] = {os.path.basename(n): (sha_file(n) if os.path.isfile(n) else None) for n in names}
    doc['host_scripts'] = {n: (sha_file('%s/m8_learned_banks/%s' % (WORKDIR, n))
                               if os.path.isfile('%s/m8_learned_banks/%s' % (WORKDIR, n)) else None) for n in SCRIPTS}
    pf = lb.storage_preflight()
    vfs = os.statvfs(BANK_ROOT)
    doc['storage'] = {'ok': pf['ok'], 'checks': pf['checks'], 'filesystem': pf['filesystem'],
                      'unexpected_entries': pf['unexpected_entries'], 'mode': pf.get('mode_octal'),
                      'total_bytes': vfs.f_blocks * vfs.f_frsize, 'free_bytes': vfs.f_bavail * vfs.f_frsize,
                      'used_bytes': (vfs.f_blocks - vfs.f_bfree) * vfs.f_frsize, 'free_inodes': vfs.f_favail,
                      'bank_root_entries': {m: sorted(os.listdir(Path(BANK_ROOT) / m)) for m in sorted(os.listdir(BANK_ROOT))}}
    e = {}
    for m in ('E01', 'E02'):
        clo = json.loads((Path(repo) / ('outputs/audit/%s_PRODUCTION_FAMILY_CLOSURE.json' % m)).read_text())
        for s, v in clo['per_seed'].items():
            p = Path(v['run_dir']) / 'bank_manifest.parquet'
            e['%s|%s' % (m, s)] = [v['bank_manifest_sha256'], sha_file(p) if p.is_file() else None]
    doc['e01_e02'] = e
    print(json.dumps(doc, sort_keys=True))


def build(v, repo):
    repo = Path(repo)
    recs = records(repo)
    clos = {}
    for m in METHODS:
        p = repo / ('outputs/audit/LEARNED_BANK_PRODUCTION_%s_FAMILY_CLOSURE.json' % m)
        clos[m] = (json.loads(p.read_text()), sha_file(p)) if p.is_file() else (None, None)
    checks = {}
    checks['all_banks_present_and_unchanged'] = len(recs) == 12 and all(
        v['banks'].get('%s|%d' % k, {}).get('exists') and v['banks']['%s|%d' % k]['manifest_sha256']
        == r['bank']['manifest_sha256'] and v['banks']['%s|%d' % k]['png_files'] == N
        and v['banks']['%s|%d' % k]['provenance_sha256'] == r['bank']['provenance_sha256'] for k, r in recs.items())
    checks['all_bank_audits_passed'] = all(r['production_audit_passed'] for r in recs.values())
    checks['all_families_closed'] = all(c and c['status'] == 'LEARNED_BANK_PRODUCTION_FAMILY_COMPLETED_AND_AUDITED'
                                        for c, _ in clos.values())
    checks['clones_clean_at_execution_commit'] = all(
        v['clones'][r['reproduction']['clone']]['exists'] and v['clones'][r['reproduction']['clone']]['status'] == ''
        and v['clones'][r['reproduction']['clone']]['head'] == r['execution_commit'] for r in recs.values())
    checks['host_scripts_equal_committed'] = all(
        v['host_scripts'][n] == sha_file(repo / 'tools/learned_bank_production_scripts' / n) for n in SCRIPTS)
    checks['all_markers_present'] = all(all(x is not None for x in mk.values()) for mk in v['markers'].values()) \
        and all(v['markers']['%s|%d' % k]['m8prod_%s_s%d.command' % k] == r['reproduction']['command_sha256']
                for k, r in recs.items())
    checks['storage_live_ok'] = v['storage']['ok'] is True and v['storage']['filesystem']['filesystem_uuid'] == \
        '608647d2-9cb0-45b0-b999-73c080ba909b' and v['storage']['bank_root_entries'] == {
            m: ['seed_1337', 'seed_2026', 'seed_42'] for m in sorted(METHODS)}
    checks['e01_e02_unchanged'] = all(a == b for a, b in v['e01_e02'].values()) and len(v['e01_e02']) == 6
    total = sum(r['bank']['outputs'] for r in recs.values())
    status = ('FOUR_LEARNED_PRODUCTION_FAMILIES_COMPLETED_AND_AUDITED' if all(checks.values()) and total == 12 * N
              else 'LEARNED_PRODUCTION_INCOMPLETE_SEE_CHECKS')
    table = [{'method': m, 'seed': s, 'run_id': r['run_id'], 'outputs': r['bank']['outputs'],
              'manifest_sha256': r['bank']['manifest_sha256'], 'audit': '%d/%d' % (r['gates_passed'], r['gates_total']),
              'status': r['status'], 'path': r['bank']['path'], 'png_bytes': r['bank']['total_png_bytes'],
              'window': [r['window']['launch'], r['window']['end']], 'execution_commit': r['execution_commit'],
              'authority_sha256': r['authority']['sha256'], 'checkpoint_identity_sha256':
              r['checkpoint']['checkpoint_identity_sha256']}
             for m in METHODS for s in SEEDS for r in [recs.get((m, s))] if r]
    doc = {'schema': 'm8_learned_production_summary_v1', 'status': status, 'checks': checks, 'total_outputs': total,
           'expected_outputs': 12 * N, 'banks': table,
           'families': {m: {'closure': 'outputs/audit/LEARNED_BANK_PRODUCTION_%s_FAMILY_CLOSURE.json' % m,
                             'closure_sha256': h, 'status': c and c['status'],
                             'gates': c and '%d/%d' % (c['gates_passed'], c['gates_total']),
                             'authority': c and c['authority'], 'execution_commit': c and c['execution_commit'],
                             'disclosures': c and c['disclosures']} for m, (c, h) in clos.items()},
           'storage': {k: v['storage'][k] for k in ('total_bytes', 'used_bytes', 'free_bytes', 'free_inodes', 'mode')},
           'storage_identity': {'bank_root': BANK_ROOT, 'uuid': v['storage']['filesystem'].get('filesystem_uuid'),
                                'mount_point': v['storage']['filesystem'].get('mount_point'),
                                'fstype': v['storage']['filesystem'].get('fstype'),
                                'source': v['storage']['filesystem'].get('source')},
           'learned_bank_png_bytes': sum(r['bank']['total_png_bytes'] for r in recs.values()),
           'host_verification': {'path': VERIFIED, 'sha256': sha_file(repo / VERIFIED), 'utc': v['utc']},
           'e05_contrast_review': {'path': 'outputs/audit/E05_S7_CONTRAST_DIAGNOSTIC.json',
                                   'sha256': sha_file(repo / 'outputs/audit/E05_S7_CONTRAST_DIAGNOSTIC.json')},
           'not_done': ['E04 Physics-STD bank adapter, S7 qualification and production banks',
                        'downstream ResNet-18 training and evaluation of all seven baselines', 'TEST evaluation',
                        'DINOv3', 'GPAT', 'outputs/audit/method_status.csv unchanged'],
           'backup': 'the 12 image banks exist only on the approved HDD; git holds manifests (gzip), per-image file and '
                     'pixel SHA256 and the audits, not image bytes; no independent binary backup was made or verified'}
    repro = {'schema': 'learned_bank_production_reproduction_manifest_v1',
             'scripts': {n: {'path': 'tools/learned_bank_production_scripts/' + n,
                             'sha256': sha_file(repo / 'tools/learned_bank_production_scripts' / n),
                             'host_copy': '%s/m8_learned_banks/%s' % (WORKDIR, n)} for n in SCRIPTS},
             'tools': {p: sha_file(repo / p) for p in ('tools/learned_bank.py', 'tools/learned_bank_production_audit.py',
                                                        'tools/learned_bank_production_authority.py',
                                                        'tools/learned_bank_production_summary.py',
                                                        'methods/common/learned_bank.py')},
             'procedure': ['git bundle of m6-baselines -> host; clone to clones/prod_<M>_<execution_commit>; mkdir -p '
                           'third_party/source_cache; cp -a facexzoo stdn swapping_autoencoder from the GPU checkout',
                           'per bank: audit preflight (host, stdin) -> ~/workdir/m8prod_<M>_s<S>_preflight.json (a-w)',
                           'm8prod_launch.sh <M> <S> <execution_commit>; m8prod_wait.sh <M> <S> <max_s>; '
                           'm8prod_post.sh <M> <S>',
                           'audit collect (host, stdin) -> ~/workdir/m8prod_<M>_s<S>_collected.json (a-w) and '
                           'outputs/audit/LEARNED_BANK_PRODUCTION/<M>_s<S>/collected.json; bank_manifest.jsonl copied '
                           'and stored as gzip (level 9, mtime 0)',
                           'audit build (laptop) -> LEARNED_BANK_PRODUCTION_<M>_SEED<S>.*; commit and push before the '
                           'next launch; after three seeds: close -> family closure'],
             'banks': {'%s|%d' % (m, s): {'run_id': r['run_id'], 'path': r['bank']['path'],
                                          'execution_commit': r['execution_commit'], 'command': r['reproduction']['command'],
                                          'command_sha256': r['reproduction']['command_sha256'],
                                          'launch': r['reproduction']['launch_script'],
                                          'post': r['reproduction']['post_script'],
                                          'audit_commands': r['reproduction']['audit_commands'],
                                          'evidence': 'outputs/audit/LEARNED_BANK_PRODUCTION/%s_s%d/' % (m, s),
                                          'record': 'outputs/audit/LEARNED_BANK_PRODUCTION_%s_SEED%d.json' % (m, s),
                                          'manifest_sha256': r['bank']['manifest_sha256'],
                                          'output_sha256_list_sha256': r['bank']['output_sha256_list_sha256'],
                                          'pixel_sha256_list_sha256': r['bank']['pixel_sha256_list_sha256']}
                       for (m, s), r in sorted(recs.items())},
             'paths_verified_on_host': {'utc': v['utc'], 'all_present': checks['all_banks_present_and_unchanged']
                                        and checks['all_markers_present'] and checks['clones_clean_at_execution_commit']},
             'backup': doc['backup']}
    return doc, repro


def report(doc):
    rows = '\n'.join('| %s | %d | `%s` | %d | `%s…` | %s | %s |' % (
        b['method'], b['seed'], b['run_id'], b['outputs'], b['manifest_sha256'][:16], b['audit'],
        'PASSED' if b['status'] == 'LEARNED_BANK_PRODUCTION_PASSED' else b['status']) for b in doc['banks'])
    st = doc['storage']
    return '\n'.join([
        '# M8 learned-bank production summary', '',
        '**Status:** `%s`. %d of %d expected outputs.' % (doc['status'], doc['total_outputs'], doc['expected_outputs']), '',
        '| Method | Seed | Run ID | Output count | Manifest SHA256 | Audit | Status |', '|---|---|---|---|---|---|---|',
        rows, '',
        '## Family closures', '',
        '\n'.join('- %s: `%s` (%s), authority `%s` (`%s…`), execution commit `%s`.' % (
            m, f['status'], f['gates'], f['authority']['path'], f['authority']['sha256'][:12], f['execution_commit'][:7])
            for m, f in doc['families'].items() if f['status']), '',
        '## Final host verification (%s)' % doc['host_verification']['utc'], '',
        '\n'.join('- %s: %s' % (k, 'pass' if v else '**FAIL**') for k, v in doc['checks'].items()), '',
        '## Storage', '',
        '- HDD `%s`, UUID `%s`, mount `%s` (%s, %s), root mode %s.' % (
            doc['storage_identity']['bank_root'], doc['storage_identity']['uuid'], doc['storage_identity']['mount_point'],
            doc['storage_identity']['fstype'], doc['storage_identity']['source'], st['mode']),
        '- Learned-bank PNG bytes: %.2f GiB. Filesystem used %.1f GiB, free %.1f GiB of %.1f GiB; %d free inodes.' % (
            doc['learned_bank_png_bytes'] / 2 ** 30, st['used_bytes'] / 2 ** 30, st['free_bytes'] / 2 ** 30,
            st['total_bytes'] / 2 ** 30, st['free_inodes']),
        '- Backup: %s.' % doc['backup'], '',
        '## Not done (out of scope)', '', '\n'.join('- ' + x for x in doc['not_done']), ''])


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('verify')
    a.add_argument('--repo', required=True)
    b = sub.add_parser('build')
    b.add_argument('--verified', required=True)
    b.add_argument('--repo-root', default=str(Path(globals().get('__file__', 'tools/x')).resolve().parents[1]))
    args = ap.parse_args()
    if args.cmd == 'verify':
        return verify(args.repo)
    root = Path(args.repo_root)
    if Path(args.verified).resolve() != (root / VERIFIED).resolve():
        ap.error('build reads the committed %s' % VERIFIED)
    doc, repro = build(json.loads(Path(args.verified).read_text()), root)
    (root / (OUT + '.json')).write_text(json.dumps(doc, indent=1, sort_keys=True) + '\n')
    (root / (OUT + '.md')).write_text(report(doc))
    (root / REPRO).write_text(json.dumps(repro, indent=1, sort_keys=True) + '\n')
    (root / (OUT + '_SHA256.txt')).write_text(''.join('%s  %s\n' % (sha_file(root / p), p)
                                                    for p in (OUT + '.json', OUT + '.md', REPRO, VERIFIED)))
    print(doc['status'], {k: v for k, v in doc['checks'].items() if not v})
    return 0


if __name__ == '__main__':
    sys.exit(main())
