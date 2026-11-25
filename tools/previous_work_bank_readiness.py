"""Seven previous-work synthetic-bank families: read-only integrity and downstream-readiness audit.

E01 FAS-Aug, E02 FreqSub (bank_writer banks in the runtime root), E03 STDN, E04 Physics-STD, E05 PCGAN, E06b DSDG-NATIVE,
E06c DSDG-BIN-IDFREE (learned banks on the approved HDD root): 7 methods x 3 seeds = 21 banks x 8838 images.

  verify (GPU host, read-only, CPU only, fed on stdin; libraries from the clean clone --repo; locked gpat-m5 python):
    env -i HOME=$HOME PATH=/usr/bin:/bin LANG=C.UTF-8 CUDA_VISIBLE_DEVICES= ~/miniconda3/envs/gpat-m5/bin/python -I -B - \
        verify --repo <clone> < tools/previous_work_bank_readiness.py > <host json>
      for every bank: the path resolved through the family's own downstream reader path (E01/E02: the closure run_dir;
      learned: lb.resolve_registered_bank via the activated registry), the manifest re-hashed against the committed
      audit, the family's library reader with full pixel verification (bank_writer.verify_bank_files /
      lb.read_bank), the PNG file count, E02 edit-map count, the manifest schema, the downstream row interface (pair or
      synthetic id, label spoof, output path, file and pixel sha256), live storage preflight and capacity.
  build (laptop): python3 tools/previous_work_bank_readiness.py build
      -> outputs/audit/PREVIOUS_WORK_SEVEN_BASELINE_BANK_READINESS.{json,md} + _SHA256.txt from the committed host
      verification and the committed audits / closures. Nothing downstream (ResNet-18, TEST, DINOv3, GPAT) is run.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

METHODS = ('E01', 'E02', 'E03', 'E04', 'E05', 'E06b', 'E06c')
LEARNED = ('E03', 'E04', 'E05', 'E06b', 'E06c')
SEEDS = (42, 1337, 2026)
N = 8838
DISPLAY = {'E01': 'FAS-Aug', 'E02': 'FreqSub', 'E03': 'STDN', 'E04': 'Physics-STD (controlled geometry/depth '
           'reconstruction)', 'E05': 'PCGAN (controlled architecture resolution)', 'E06b': 'DSDG-NATIVE',
           'E06c': 'DSDG-BIN-IDFREE'}
VERIFIED = 'outputs/audit/PREVIOUS_WORK_BANK_READINESS/host_verification.json'
OUT = 'outputs/audit/PREVIOUS_WORK_SEVEN_BASELINE_BANK_READINESS'
HDD_UUID = '608647d2-9cb0-45b0-b999-73c080ba909b'


def sha_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def closures(repo):
    out = {}
    for m in ('E01', 'E02'):
        p = Path(repo) / ('outputs/audit/%s_PRODUCTION_FAMILY_CLOSURE.json' % m)
        out[m] = (json.loads(p.read_text()), sha_file(p))
    for m in LEARNED:
        p = Path(repo) / ('outputs/audit/LEARNED_BANK_PRODUCTION_%s_FAMILY_CLOSURE.json' % m)
        out[m] = (json.loads(p.read_text()), sha_file(p)) if p.is_file() else (None, None)
    return out


def expected_banks(repo):
    """(method, seed) -> {path, manifest_sha256, audit_record, audit_sha256} from the committed records."""
    out = {}
    cl = closures(repo)
    for m in ('E01', 'E02'):
        for s, v in cl[m][0]['per_seed'].items():
            out[(m, int(s))] = {'path': v['run_dir'], 'manifest_sha256': v['bank_manifest_sha256'],
                                'audit_record': v['audit_record'], 'audit_sha256': v['audit_record_sha256'],
                                'audit_gates': v['audit_gates'], 'output_count': v['output_count']}
    for m in LEARNED:
        for s in SEEDS:
            rel = 'outputs/audit/LEARNED_BANK_PRODUCTION_%s_SEED%d.json' % (m, s)
            p = Path(repo) / rel
            if p.is_file():
                r = json.loads(p.read_text())
                out[(m, s)] = {'path': r['bank']['path'], 'manifest_sha256': r['bank']['manifest_sha256'],
                               'provenance_sha256': r['bank']['provenance_sha256'], 'audit_record': rel,
                               'audit_sha256': sha_file(p), 'audit_gates': '%d/%d' % (r['gates_passed'],
                                                                                      r['gates_total']),
                               'output_count': r['bank']['outputs'], 'audit_passed': r['production_audit_passed'],
                               'run_id': r['run_id']}
    return out


# ---------------------------------------------------------------------------------------------------------- verify
def verify(repo):
    sys.path.insert(0, str(repo))
    os.chdir(str(repo))
    from methods.common import bank_writer as bw
    from methods.common import learned_bank as lb
    doc = {'phase': 'verify', 'repo': str(repo),
           'utc': subprocess.run(['date', '-u', '+%FT%TZ'], capture_output=True, text=True).stdout.strip(),
           'git_head': subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'], capture_output=True,
                                      text=True).stdout.strip(),
           'git_status': subprocess.run(['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=all'],
                                        capture_output=True, text=True).stdout.strip(),
           'python': sys.version.split()[0], 'executable': sys.executable, 'banks': {}}
    for (m, s), e in sorted(expected_banks(repo).items()):
        rec = {'expected_path': e['path']}
        try:
            if m in ('E01', 'E02'):
                d = Path(e['path'])
                rows = bw.read_manifest(d)
                v = bw.verify_bank_files(d, m)
                rec.update(resolved_path=str(d), manifest_sha256=sha_file(d / bw.BANK_MANIFEST),
                           reader={'library': 'methods.common.bank_writer.verify_bank_files', 'ok': v['ok'],
                                   'rows': v['rows'], 'files_checked': v['files_checked'], 'problems': v['problems']},
                           png_files=len([x for x in os.listdir(d / 'bank') if x.endswith('.png')]),
                           edit_maps=len(os.listdir(d / bw.EDIT_MAP_DIR)) if (d / bw.EDIT_MAP_DIR).is_dir() else None,
                           schema_ok=bool(rows) and all(sorted(r) == sorted(bw.MANIFEST_COLUMNS) for r in rows),
                           downstream_interface_ok=bool(rows) and all(
                               r['generation_status'] == 'success' and r['method_id'] == m and r['experiment_seed'] == s
                               and r['pair_id'] and r['synthetic_output_path'] and r['output_sha256']
                               and r['output_pixel_sha256'] for r in rows),
                           successes=sum(r['generation_status'] == 'success' for r in rows))
            else:
                d = lb.resolve_registered_bank(m, s)
                rb = lb.read_bank(d, expect_method=m, expect_seed=s, expect_mode='PRODUCTION', expect_rows=N,
                                  expect_checkpoint_identity=lb.checkpoint_identity_sha256(m, s))
                rows = rb.pop('rows')
                rec.update(resolved_path=str(d), manifest_sha256=sha_file(d / lb.MANIFEST),
                           provenance_sha256=sha_file(d / lb.PROVENANCE),
                           reader={'library': 'methods.common.learned_bank.read_bank', 'ok': rb['ok'],
                                   'rows': len(rows), 'files_checked': rb.get('files_checked'),
                                   'problems': rb['problems'][:20]},
                           png_files=len([x for x in os.listdir(d / lb.BANK_DIR) if x.endswith('.png')]),
                           schema_ok=bool(rows) and all(set(r) == set(lb.MANIFEST_FIELDS) for r in rows),
                           downstream_interface_ok=bool(rows) and all(
                               r['generation_status'] == 'success' and r['label'] == 'spoof' and r['method_id'] == m
                               and r['bank_seed'] == s and r['row_id'] and r['output_path'] and r['output_sha256']
                               and r['output_pixel_sha256'] for r in rows),
                           successes=sum(r['generation_status'] == 'success' for r in rows),
                           dir_mode=oct(d.stat().st_mode & 0o7777))
        except Exception as exc:
            rec['error'] = '%s: %s' % (type(exc).__name__, exc)
        doc['banks']['%s|%d' % (m, s)] = rec
    pf = lb.storage_preflight()
    st = os.statvfs(lb.HDD['bank_root'])
    doc['storage'] = {'ok': pf['ok'], 'checks': pf['checks'], 'unexpected_entries': pf['unexpected_entries'],
                      'filesystem': pf['filesystem'], 'mode': pf.get('mode_octal'),
                      'registered_methods': list(lb.registered_bank_methods()),
                      'bank_root_entries': {m: sorted(os.listdir(Path(lb.HDD['bank_root']) / m))
                                            for m in sorted(os.listdir(lb.HDD['bank_root']))},
                      'free_bytes': st.f_bavail * st.f_frsize, 'free_inodes': st.f_favail,
                      'total_bytes': st.f_blocks * st.f_frsize}
    print(json.dumps(doc, sort_keys=True, default=str))


# ----------------------------------------------------------------------------------------------------------- build
def build(repo):
    repo = Path(repo)
    v = json.loads((repo / VERIFIED).read_text())
    exp = expected_banks(repo)
    cl = closures(repo)
    banks, checks = [], {}
    for m in METHODS:
        for s in SEEDS:
            e, h = exp.get((m, s)), v['banks'].get('%s|%d' % (m, s)) or {}
            ok = bool(e) and 'error' not in h and h.get('resolved_path') == e['path'] \
                and h.get('manifest_sha256') == e['manifest_sha256'] and h['reader']['ok'] is True \
                and h['reader']['rows'] == N and h['reader']['files_checked'] == N and h['png_files'] == N \
                and h['successes'] == N and h['schema_ok'] is True and h['downstream_interface_ok'] is True \
                and e['output_count'] == N and (m != 'E02' or h.get('edit_maps') == N) \
                and (m not in LEARNED or (e['audit_passed'] is True and h['provenance_sha256'] == e['provenance_sha256']))
            banks.append({'method': m, 'display_name': DISPLAY[m], 'seed': s, 'path': (e or {}).get('path'),
                          'output_count': h.get('png_files'), 'manifest_sha256': (e or {}).get('manifest_sha256'),
                          'audit_record': (e or {}).get('audit_record'), 'audit_sha256': (e or {}).get('audit_sha256'),
                          'audit_gates': (e or {}).get('audit_gates'), 'run_id': (e or {}).get('run_id'),
                          'reader': h.get('reader', {}).get('library'), 'edit_maps': h.get('edit_maps'),
                          'verified': ok})
    fam = {}
    for m in METHODS:
        c, csha = cl[m]
        want = 'E%s_PRODUCTION_FAMILY_COMPLETED_AND_AUDITED' % m[1:] if m in ('E01', 'E02') \
            else 'LEARNED_BANK_PRODUCTION_FAMILY_COMPLETED_AND_AUDITED'
        mine = [b for b in banks if b['method'] == m]
        fam[m] = {'display_name': DISPLAY[m], 'closure_sha256': csha, 'closure_status': c and c['status'],
                  'closure_ok': bool(c) and c['status'] == want,
                  'closure': ('outputs/audit/%s_PRODUCTION_FAMILY_CLOSURE.json' % m if m in ('E01', 'E02')
                              else 'outputs/audit/LEARNED_BANK_PRODUCTION_%s_FAMILY_CLOSURE.json' % m),
                  'seeds': [b['seed'] for b in mine], 'banks_complete': sum(b['verified'] for b in mine),
                  'downstream_readable': all(b['verified'] for b in mine) and len(mine) == 3,
                  'disclosures': (c or {}).get('disclosures') or (c or {}).get('fidelity_disclosures')}
    total = sum(b['output_count'] or 0 for b in banks if b['verified'])
    st = v['storage']
    checks['all_21_banks_verified'] = len(banks) == 21 and all(b['verified'] for b in banks)
    checks['all_7_family_closures'] = all(f['closure_ok'] for f in fam.values())
    checks['total_185598_images'] = total == 21 * N
    checks['verify_clone_clean'] = v['git_status'] == ''
    checks['storage_live_ok'] = st['ok'] is True and st['filesystem']['filesystem_uuid'] == HDD_UUID \
        and st['unexpected_entries'] == [] and st['bank_root_entries'] == {m: ['seed_1337', 'seed_2026', 'seed_42']
                                                                            for m in sorted(LEARNED)} \
        and sorted(st['registered_methods']) == sorted(LEARNED)
    status = ('ALL_SEVEN_PREVIOUS_WORK_BANK_FAMILIES_COMPLETED_AND_AUDITED' if all(checks.values())
              else 'SEVEN_BASELINE_BANK_READINESS_INCOMPLETE_SEE_CHECKS')
    doc = {
        'schema': 'previous_work_seven_baseline_bank_readiness_v1', 'status': status, 'checks': checks,
        'total_images': total, 'expected_images': 21 * N,
        'count_note': 'synthetic PNG images only; E02 edit maps, manifests and provenance sidecars are not counted',
        'families': fam, 'banks': banks,
        'host_verification': {'path': VERIFIED, 'sha256': sha_file(repo / VERIFIED), 'utc': v['utc'],
                              'clone_commit': v['git_head'], 'python': v['python']},
        'storage': {'hdd_bank_root': '/media/ext_data/student20261_gpat_m8_banks', 'uuid': HDD_UUID,
                    'free_bytes': st['free_bytes'], 'total_bytes': st['total_bytes'], 'free_inodes': st['free_inodes'],
                    'e01_e02_location': 'runtime root <runtime_root>/runs/m6/E0x/seed_<s> (NVMe system disk)'},
        'known_limitations': [
            'E01: level-zero no-op operator outputs reproduce the target face (disclosed in the E01 closure); duplicate '
            'pixel groups exist by construction',
            'E02: edit maps preserved beside every output (8838 per bank)',
            'E04: CONTROLLED_ADAPTATION (controlled geometry/depth reconstruction; BN moving statistics; clip-truncate); '
            'not an official native Physics-STD reproduction',
            'E05: PCGAN controlled architecture resolution; low-contrast outputs, seed 1337 near-collapse disclosed and '
            'retained under RAW-MATCHED',
            'E06b: DSDG-NATIVE trained on CASIA+MSU TRAIN spoof coverage only (SiW-Mv2 not instantiable); synthetic ids '
            'without pair lineage',
            'E06c: DSDG-BIN-IDFREE adapted binary variant; synthetic ids without pair lineage',
            'banks exist only on the GPU host (E01/E02 runtime root, learned banks on the HDD); git holds manifests, '
            'hashes and audits; no independent binary backup'],
        'downstream_prerequisites_remaining': [
            'frozen downstream evaluator / bank-loader readiness review and owner authorization',
            'scientific ResNet-18 training and validation per method and seed', 'authorized TEST evaluation',
            'mean +/- standard deviation across seeds; ACER, APCER, BPCER, EER, AUC; ROC/DET',
            'final previous-work comparisons'],
        'not_done': ['no downstream ResNet-18 training, TEST evaluation, DINOv3 or GPAT',
                     'outputs/audit/method_status.csv unchanged'],
    }
    return doc


def report(doc):
    rows = '\n'.join('| %s | %s | 3 | %d/3 | %s | %s |' % (
        m, f['display_name'], f['banks_complete'], f['closure_status'], 'yes' if f['downstream_readable'] else 'NO')
        for m, f in doc['families'].items())
    banks = '\n'.join('| %s | %d | %s | `%s…` | %s | %s |' % (
        b['method'], b['seed'], b['output_count'], (b['manifest_sha256'] or '')[:16], b['audit_gates'],
        'verified' if b['verified'] else '**NOT VERIFIED**') for b in doc['banks'])
    return '\n'.join([
        '# Previous-work synthetic banks: seven-baseline readiness', '',
        '**Status:** `%s`. %d of %d images (%s).' % (doc['status'], doc['total_images'], doc['expected_images'],
                                                     doc['count_note']), '',
        '| Method | Name | Seeds | Banks verified | Family closure | Downstream readable |', '|---|---|---|---|---|---|',
        rows, '', '| Method | Seed | Images | Manifest SHA256 | Audit gates | Host verification |',
        '|---|---|---|---|---|---|', banks, '', '## Checks', '',
        '\n'.join('- %s: %s' % (k, 'pass' if v else '**FAIL**') for k, v in doc['checks'].items()), '',
        '## Known limitations', '', '\n'.join('- ' + x for x in doc['known_limitations']), '',
        '## Remaining downstream prerequisites (not executed)', '',
        '\n'.join('- ' + x for x in doc['downstream_prerequisites_remaining']), ''])


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('verify')
    a.add_argument('--repo', required=True)
    b = sub.add_parser('build')
    b.add_argument('--repo-root', default=str(Path(globals().get('__file__', 'tools/x')).resolve().parents[1]))
    args = ap.parse_args(argv)
    if args.cmd == 'verify':
        return verify(args.repo)
    root = Path(args.repo_root)
    doc = build(root)
    (root / (OUT + '.json')).write_text(json.dumps(doc, indent=1, sort_keys=True) + '\n')
    (root / (OUT + '.md')).write_text(report(doc))
    (root / (OUT + '_SHA256.txt')).write_text(''.join('%s  %s\n' % (sha_file(root / p), p)
                                                    for p in (OUT + '.json', OUT + '.md', VERIFIED)))
    print(doc['status'], {k: v for k, v in doc['checks'].items() if not v})
    return 0 if doc['status'].startswith('ALL_SEVEN') else 1


if __name__ == '__main__':
    sys.exit(main())
