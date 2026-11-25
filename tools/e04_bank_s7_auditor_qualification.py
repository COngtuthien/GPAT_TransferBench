#!/usr/bin/env python3
"""E04 S7 auditor qualification record (laptop). Re-derives every gate from the raw evidence in
outputs/audit/E04_BANK_S7_AUDITOR_QUALIFICATION/ and live repository checks, then writes
outputs/audit/E04_BANK_S7_AUDITOR_QUALIFICATION.{json,md} and _SHA256.txt.

Raw evidence:
  laptop/summary.tsv + laptop/<module>.log   targeted regression modules (repo .venv), exit status per module
  host/host_tests_e05.{log,rc}               tests/test_e04_bank_s7_audit.py on the GPU host, collector interpreter
  host/preflight_dry_s42_A.json              E04 preflight in the real E04 environment (no authority yet)
  host/collect_dry.json                      E04 collector against an empty S7 root (real lineage + pack cross-check)
  fullsuite/{base,cand}_{failures.txt,summary.txt}   whole suite at the session start (2caff0d) and at the
                                             auditor commit, both from clean git worktrees with identical conditions
Live checks: the four accepted S7 records (E03, E05, E06b, E06c) rebuild byte-for-byte with the current tool; the
dry collection builds fail-closed (BLOCKED_BY_AUTHORITY, no output gate passes) while the real feed-pack cross-check
and code-change-scope gates pass.
"""
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EV = 'outputs/audit/E04_BANK_S7_AUDITOR_QUALIFICATION'
OUT = 'outputs/audit/E04_BANK_S7_AUDITOR_QUALIFICATION'
REQUIRED_MODULES_MIN = 31
NEW_MODULE = 'test_e04_bank_s7_audit.py'


def sha(b):
    return hashlib.sha256(b).hexdigest()


def audit_tool():
    spec = importlib.util.spec_from_file_location('s7a', ROOT / 'tools' / 'learned_bank_s7_audit.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def gates(auditor_commit, base_commit, fullsuite_commit):
    m = audit_tool()
    ev = ROOT / EV
    out = []

    def gate(name, ok, detail=None):
        out.append({'gate': name, 'pass': bool(ok), **({'detail': detail} if detail is not None else {})})
    rows = [x.split('\t') for x in (ev / 'laptop' / 'summary.tsv').read_text().splitlines() if x.strip()]
    mods = {r[0]: {'rc': int(r[1]), 'summary': r[2].strip()} for r in rows}
    for name, v in mods.items():
        log = (ev / 'laptop' / (name[:-3] + '.log')).read_text()
        v['log_ok'] = log.rstrip().endswith(('OK', ')')) and '\nOK' in log and 'FAILED' not in log
    gate('Q.targeted_regression_modules_pass', len(mods) >= REQUIRED_MODULES_MIN
         and all(v['rc'] == 0 and v['log_ok'] for v in mods.values()),
         {k: v['summary'] for k, v in sorted(mods.items())})
    gate('Q.new_e04_audit_module_ran', NEW_MODULE in mods and mods[NEW_MODULE]['rc'] == 0
         and int(mods[NEW_MODULE]['summary'].split()[1]) >= 39, mods.get(NEW_MODULE))
    host_log = (ev / 'host' / 'host_tests_e05.log').read_text()
    gate('Q.host_collector_interpreter_tests', (ev / 'host' / 'host_tests_e05.rc').read_text().strip() == 'tests rc=0'
         and host_log.rstrip().endswith('OK') and 'Ran 39 tests' in host_log, host_log.strip().splitlines()[-3:])
    pf = json.loads((ev / 'host' / 'preflight_dry_s42_A.json').read_text())
    pr = pf['environment']['probe']
    gate('Q.host_real_e04_environment_probe', pf['environment']['probe_rc'] == 0 and pr['python'] == m.E04_PY_VERSION
         and pr['tensorflow'] == m.E04_TF and pr['precision_errors'] == [] and pr['environment_lock']['ok'] is True
         and pr['environment_lock']['driver']['gpu_uuid'] == m.GPU_UUID
         and pr['frozen_sources']['files'] == m.E04_FROZEN_FILES and pr['bn_only_difference'] is True
         and pr['contract_sha256'] == m.E04_CONTRACT_SHA256
         and all(pr['checkpoint_counters'][str(s)]['global_iteration'] == 150000 for s in m.SEEDS)
         and 'LEARNED_BANK_BLOCKED_BY_S7_LAUNCH_AUTHORITY' in pr['s7_authority']['error']
         and pf['authority']['ok'] is False and all(v['ok'] for v in pf['checkpoints'].values())
         and pf['inputs'] == {'toy_manifest_sha256': m.TOY_MANIFEST[1],
                              'feed_pack_manifest_sha256': m.E04_PACK_MANIFEST_SHA256, 'faces_root_is_dir': True}
         and 'E04' not in pf['hdd_bank_root_entries'],
         {'python': pr['python'], 'tensorflow': pr['tensorflow'],
          'checkpoint_identities': {k: v['identity'] for k, v in pf['checkpoints'].items()},
          'authority_before_activation': pf['authority']['error']})
    c = json.loads((ev / 'host' / 'collect_dry.json').read_text())
    d = m.build_e04(c, ROOT)
    passed = sorted(g['gate'] for g in d['gates'] if g['pass'])
    gate('Q.dry_collection_fail_closed', d['status'] == 'E04_S7_BLOCKED_BY_AUTHORITY' and not d['s7_acceptance_passed']
         and 'F.feed_pack_allowlist_crosscheck' in passed and 'P.e04_code_change_scope' in passed
         and not any(g.startswith(('S.', 'L.', 'R.seed', 'P.seed')) for g in passed)
         and 'P.e04_authority_in_execution_commit' not in passed,
         {'status': d['status'], 'passed': passed, 'gates_total': d['gates_total']})
    gate('Q.real_feed_pack_allowlist_crosscheck', c['lineage']['feed_pack_crosscheck'] == {
        'manifest_sha256': m.E04_PACK_MANIFEST_SHA256, 'pairs': 8838, 'pairs_equal_pairs_train_v1': True,
        'pairs_split': 'TRAIN', 'payload_sha_ok': True, 'samples': 12668, 'samples_mismatching': [],
        'samples_mismatching_count': 0}, c['lineage']['feed_pack_crosscheck'])
    rebuilt = {}
    for meth in ('E03', 'E05', 'E06b', 'E06c'):
        rec = ROOT / 'outputs' / 'audit' / ('LEARNED_BANK_S7_%s.json' % meth)
        doc = m.build(json.loads(rec.read_text())['collected'], ROOT)
        rebuilt[meth] = (json.dumps(doc, indent=1, sort_keys=True) + '\n').encode() == rec.read_bytes() \
            and m.report(doc) == (rec.parent / ('LEARNED_BANK_S7_%s.md' % meth)).read_text()
    gate('Q.existing_s7_records_rebuild_byte_for_byte', all(rebuilt.values()), rebuilt)
    fs = ev / 'fullsuite'
    base = set((fs / 'base_failures.txt').read_text().split('\n')) - {''}
    cand = set((fs / 'cand_failures.txt').read_text().split('\n')) - {''}
    summ = {k: (fs / ('%s_summary.txt' % k)).read_text().strip() for k in ('base', 'cand')}
    gate('Q.full_suite_no_new_failures', cand <= base,
         {'baseline_commit': base_commit, 'candidate_commit': fullsuite_commit, 'baseline_failing': len(base),
          'candidate_failing': len(cand), 'new_failures': sorted(cand - base), 'fixed': sorted(base - cand),
          'summaries': summ})
    return out, {'modules': mods, 'full_suite': {'baseline_failing_ids': sorted(base), 'candidate_failing_ids': sorted(cand)}}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--auditor-commit', required=True)
    ap.add_argument('--base-commit', required=True)
    ap.add_argument('--fullsuite-commit', required=True)
    a = ap.parse_args(argv)
    g, extra = gates(a.auditor_commit, a.base_commit, a.fullsuite_commit)
    failed = [x['gate'] for x in g if not x['pass']]
    raw = sorted(p for p in (ROOT / EV).rglob('*') if p.is_file())
    doc = {'schema': 'e04_bank_s7_auditor_qualification_v1', 'method': 'E04',
           'status': 'E04_S7_AUDITOR_QUALIFIED' if not failed else 'E04_S7_AUDITOR_QUALIFICATION_FAILED',
           'auditor_commit': a.auditor_commit, 'baseline_commit': a.base_commit, 'fullsuite_candidate_commit': a.fullsuite_commit, 'gates': g, 'failed_gates': failed,
           'gates_passed': len(g) - len(failed), 'gates_total': len(g), **extra,
           'raw_evidence_sha256': {str(p.relative_to(ROOT)): sha(p.read_bytes()) for p in raw},
           'scope': 'qualifies the E04 branch of tools/learned_bank_s7_audit.py before the E04 S7 launch authority is '
                    'activated; no GPU inference, no face read, no output was produced'}
    (ROOT / (OUT + '.json')).write_text(json.dumps(doc, indent=1, sort_keys=True) + '\n')
    fsd = next(x for x in g if x['gate'] == 'Q.full_suite_no_new_failures')['detail']
    md = '\n'.join([
        '# E04 S7 independent auditor qualification', '',
        '**Status:** `%s`, %d/%d gates; auditor commit `%s`, baseline `%s`.' % (
            doc['status'], doc['gates_passed'], doc['gates_total'], a.auditor_commit, a.base_commit), '',
        '| gate | result |', '|---|---|'] + ['| `%s` | %s |' % (x['gate'], 'PASS' if x['pass'] else 'FAIL') for x in g] + [
        '', '## Targeted regression modules (repo .venv, one `discover` per module)', '',
        '| module | exit | result |', '|---|---|---|'] + [
        '| `%s` | %d | %s |' % (k, v['rc'], v['summary']) for k, v in sorted(extra['modules'].items())] + [
        '', '## Whole suite (clean worktrees, identical conditions)', '',
        '- Baseline `%s`: %s' % (a.base_commit, fsd['summaries']['base']),
        '- Candidate `%s` (auditor branch; the later auditor commit only tightened one presence gate and is covered by the targeted modules): %s' % (a.fullsuite_commit, fsd['summaries']['cand']),
        '- Failing tests: baseline %d, candidate %d; new failures: %s; fixed: %s.' % (
            fsd['baseline_failing'], fsd['candidate_failing'], fsd['new_failures'] or 'none', fsd['fixed'] or 'none'),
        '- The pre-existing failures are environment- or history-dependent (see `fullsuite/`); their identities are '
        'listed in the JSON record and are unchanged by this work.', '',
        '## Scope', '', doc['scope'] + '.', ''])
    (ROOT / (OUT + '.md')).write_text(md)
    (ROOT / (OUT + '_SHA256.txt')).write_text(''.join('%s  %s\n' % (sha((ROOT / p).read_bytes()), p)
                                                      for p in (OUT + '.json', OUT + '.md')))
    print(doc['status'], doc['gates_passed'], '/', doc['gates_total'], failed)
    return 0 if not failed else 1


if __name__ == '__main__':
    sys.exit(main())
