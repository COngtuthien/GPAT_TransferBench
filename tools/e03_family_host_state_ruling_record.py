"""E03 family host-state ruling record: verifies the owner ruling of 2026-10-29 on H.host_final_state
(configs/amendments/e03_family_host_state_reboot_resolution.yaml) against the committed E03 evidence and a fresh
read-only host recheck, and writes outputs/audit/E03_FAMILY_HOST_STATE_OWNER_RULING.{json,md} plus its SHA-256 list.

Inputs: the committed per-seed and family E03 evidence, and a host recheck document collected with
tools/e03_family_host_recheck.py (embedded in the output so the build re-derives). The closure tool is exercised in
memory only (committed re-derivation plus negative probes of the gate); nothing on the host is written and no E03
evidence file is modified. Stdlib only.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

RULING_FILE = 'configs/amendments/e03_family_host_state_reboot_resolution.yaml'
DOCUMENT = 'docs/spec/amendments/GPAT_TransferBench_v1_0_E03_Family_Host_State_Reboot_Owner_Ruling.md'
CLOSURE_TOOL = 'tools/e03_three_seed_training_closure.py'
CLOSURE = 'outputs/audit/E03_THREE_SEED_TRAINING_CLOSURE.json'
SEEDS = (42, 1337, 2026)
FAMILY_CLOSURE_COMMIT = '374de40c66818bf27cf3bd98cb1871046bd57c2a'
GATE = 'H.host_final_state'
RESULT = 'PASS_BY_APPROVED_POST_COMPLETION_REBOOT_EVIDENCE'
EVIDENCE = {'record': 'outputs/audit/E03_FAMILY_HOST_STATE_OWNER_RULING.json',
            'report': 'outputs/audit/E03_FAMILY_HOST_STATE_OWNER_RULING.md',
            'sha256': 'outputs/audit/E03_FAMILY_HOST_STATE_OWNER_RULING_SHA256.txt'}


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def epoch(iso):
    return datetime.strptime(iso, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc).timestamp()


def closure_tool(repo):
    spec = importlib.util.spec_from_file_location('e03_family', repo / CLOSURE_TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def gate(doc):
    return next(g for g in doc['gates'] if g['gate'] == GATE)


def build(repo_root, fresh):
    repo = Path(repo_root)
    ruling = json.loads((repo / RULING_FILE).read_text())
    closure = json.loads((repo / CLOSURE).read_text())
    seeds = {s: json.loads((repo / f'outputs/audit/E03_SEED{s}_COMPLETION.json').read_text()) for s in SEEDS}
    reb = seeds[2026]['disclosures']['post_completion_reboot']
    last_end = max(epoch(d['end_utc']) for d in seeds.values())
    tool = closure_tool(repo)
    checks = {}

    def chk(name, ok, **detail):
        checks[name] = dict(detail, **{'pass': bool(ok)})

    states = {'committed_recheck': closure['host_recheck']['host_state'], 'fresh_recheck': fresh['host_state']}
    chk('1_runs_completed_before_verified_reboot', all(
        d['collected']['run_summary']['completion_status'] == 'completed' and
        epoch(d['end_utc']) < epoch(reb['run_boot_last_entry_utc']) and epoch(d['end_utc']) < epoch(reb['kernel_btime_utc'])
        for d in seeds.values()), end_utc={str(s): d['end_utc'] for s, d in seeds.items()},
        run_boot_last_entry_utc=reb['run_boot_last_entry_utc'], kernel_btime_utc=reb['kernel_btime_utc'])
    seed_gates = {g['gate']: g for g in closure['gates']}
    chk('2_original_evidence_valid', all(
        d['status'] == f'E03_SEED{s}_COMPLETED_CLEANLY' and (d['gates_passed'], d['gates_total']) == (88, 88) and
        d['collected']['host']['exit_file'] == '0' and seed_gates[f'A.seed{s}.equals_closure_commit']['pass'] and
        seed_gates[f'A.seed{s}.rederives_current_tool']['pass'] and seed_gates[f'B.seed{s}.closed_clean']['pass']
        for s, d in seeds.items()) and all(
        d['collected']['host']['tmux_pane_dead'] is True and d['collected']['host']['tmux_pane_dead_status'] == 0
        for s, d in seeds.items() if s != 2026) and seeds[2026]['gates_passed_by_approved_alternative'] ==
        ['B.process_exited_zero', 'J.single_e03_session'],
        live_pane_evidence_seeds=[42, 1337], reboot_evidence_seed=2026)
    chk('3_no_e03_tmux_session', all('tmux_sessions' in st and not any(x.startswith('e03_stdn_seed')
                                                                       for x in st['tmux_sessions'])
                                     and set(st['e03_tmux_panes'].values()) == {''} for st in states.values()),
        **{k: st.get('tmux_sessions') for k, st in states.items()})
    chk('4_no_e03_runner_process', all(st['method_processes'] == [] for st in states.values()),
        **{k: st['method_processes'] for k, st in states.items()})
    chk('5_exact_verified_boot', all(st.get('boot_id') == reb['current_boot_id'] for st in states.values()),
        verified_boot_id=reb['current_boot_id'], **{k: st.get('boot_id') for k, st in states.items()})
    chk('6_boot_after_every_seed', all(isinstance(st.get('btime'), int) and st['btime'] > last_end
                                       for st in states.values()),
        latest_seed_end_utc=datetime.fromtimestamp(last_end, timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        **{k: st.get('btime') for k, st in states.items()})
    # 7 and the gate's behaviour: committed closure re-derives byte-identically; the fresh recheck passes; probes
    committed_doc, text = tool.build(repo, closure['host_recheck'])
    fresh_doc, _ = tool.build(repo, fresh)
    probes = {}
    for name, change, expect in (
            ('other_gate_fails', lambda h: h['seeds']['42'].update(metrics_jsonl_sha256='0' * 64), False),
            ('different_boot', lambda h: h['host_state'].update(boot_id='0' * 32), False),
            ('boot_before_last_seed_end', lambda h: h['host_state'].update(btime=int(last_end) - 1), False),
            ('e03_tmux_session_present', lambda h: h['host_state'].update(tmux_sessions=['e03_stdn_seed42']), False),
            ('runner_process_present', lambda h: h['host_state'].update(
                method_processes=['1 python tools/run_e03.py --seed 42']), False),
            ('live_panes_dead_status_0', lambda h: h['host_state'].update(
                e03_tmux_panes={f'e03_stdn_seed{s}': '1 0' for s in SEEDS}, boot_id='0' * 32), True),
            ('live_pane_nonzero_status', lambda h: h['host_state'].update(
                e03_tmux_panes={f'e03_stdn_seed{s}': '1 1' for s in SEEDS}), False)):
        h = copy.deepcopy(closure['host_recheck'])
        change(h)
        g = gate(tool.build(repo, h)[0])
        probes[name] = {'expected_pass': expect, 'pass': g['pass'], 'e03_tmux': g['detail']['e03_tmux'],
                        'as_expected': g['pass'] is expect}
    chk('7_all_other_gates_pass', closure['status'] == 'SCIENTIFIC_TRAINING_COMPLETED_3_OF_3_SEEDS' and
        closure['failed_gates'] == [] and json.loads(json.dumps(committed_doc)) == closure and
        text == (repo / 'outputs/audit/E03_THREE_SEED_FINAL_CHECKPOINTS.csv').read_text() and
        probes['other_gate_fails']['e03_tmux'] == 'FAIL_POST_COMPLETION_REBOOT_EVIDENCE',
        committed_gates=f"{closure['gates_passed']}/{closure['gates_total']}")
    chk('gate_probes_fail_closed_and_live_path_unchanged', all(p['as_expected'] for p in probes.values()),
        probes=probes)
    chk('gate_result_committed_and_fresh', gate(closure)['pass'] and gate(closure)['detail']['e03_tmux'] == RESULT and
        gate(fresh_doc)['pass'] and gate(fresh_doc)['detail']['e03_tmux'] == RESULT and
        fresh_doc['failed_gates'] == [], fresh_status=fresh_doc['status'],
        fresh_gates=f"{fresh_doc['gates_passed']}/{fresh_doc['gates_total']}", fresh_utc=fresh['utc'])
    chk('earlier_disclosures_preserved', sorted(closure['per_seed_disclosures']) == ['1337', '2026'] and
        sorted(closure['per_seed_disclosures']['1337']) == ['pane_log_keyboard_echo'] and
        sorted(closure['per_seed_disclosures']['2026']) == ['post_completion_reboot'] and
        [a['id'] for a in closure['seed_tool_amendments']] ==
        ['seed1337_pane_log_keyboard_echo', 'seed2026_post_completion_reboot_evidence'])
    chk('ruling_record_consistent', ruling['gate'] == GATE and ruling['result'] == RESULT and
        len(ruling['conditions']) == 7 and ruling['changes_trained_model_or_procedure'] is False and
        ruling['family_closure_commit'] == FAMILY_CLOSURE_COMMIT and
        ruling['scientific_training_authority'] == closure['scientific_training_authority'] and
        [r['commit'] for r in ruling['related_rulings']] ==
        [closure['per_seed_evidence'][s]['closure_commit'] for s in ('1337', '2026')])
    failed = [k for k, v in checks.items() if not v['pass']]
    doc = {'milestone': 'E03-FAMILY-HOST-STATE-OWNER-RULING', 'ruling': ruling['ruling'], 'date': ruling['date'],
           'status': 'OWNER_RULING_RECORDED_CONDITIONS_VERIFIED' if not failed else 'OWNER_RULING_CONDITIONS_NOT_MET',
           'failed_checks': failed, 'checks': checks, 'gate': GATE,
           'ruling_file': RULING_FILE, 'ruling_file_sha256': sha256_bytes((repo / RULING_FILE).read_bytes()),
           'document': DOCUMENT, 'document_sha256': sha256_bytes((repo / DOCUMENT).read_bytes()),
           'closure_evidence_sha256': sha256_bytes((repo / CLOSURE).read_bytes()),
           'family_closure_commit': ruling['family_closure_commit'],
           'scientific_training_authority': closure['scientific_training_authority'],
           'not_scientific_authority': True, 'changes_trained_model_or_procedure': False,
           'confirmations': {'e03_evidence_modified': False, 'host_written': False, 'training_launched': False,
                             'test_accessed': False},
           'fresh_host_recheck': fresh}
    return doc


def report(doc):
    rows = ''.join('| `%s` | %s |\n' % (k, 'pass' if v['pass'] else 'FAIL') for k, v in doc['checks'].items())
    probes = ''.join('| `%s` | %s | %s | %s |\n' % (k, 'pass' if p['expected_pass'] else 'fail',
                                                    'pass' if p['pass'] else 'fail', p['e03_tmux'])
                     for k, p in doc['checks']['gate_probes_fail_closed_and_live_path_unchanged']['probes'].items())
    fr = doc['checks']['gate_result_committed_and_fresh']
    return '\n'.join([
        '# E03: family host-state owner ruling (`%s`)' % doc['gate'], '',
        '**Status:** `%s`. Ruling `%s` (%s). Ruling text `%s`, machine-readable record `%s`.'
        % (doc['status'], doc['ruling'], doc['date'], doc['document'], doc['ruling_file']), '',
        'This is an audit-tool amendment record, not a scientific authority. It changes neither the trained E03 models '
        'nor the procedure, and no E03 evidence file was modified. The E03 training authority stays `%s`; the family '
        'closure is `%s`.' % (doc['scientific_training_authority'], doc['family_closure_commit']), '',
        'A fresh read-only host recheck at %s re-builds the family closure to `%s`, %s gates.'
        % (fr['fresh_utc'], fr['fresh_status'], fr['fresh_gates']), '',
        '| check | result |', '|---|---|', rows.rstrip('\n'), '',
        'Gate probes, run in memory on the committed host recheck:', '',
        '| probe | expected gate | gate | tmux sub-result |', '|---|---|---|---|', probes.rstrip('\n'), '',
        'The tmux sub-result covers only the pane condition; the gate also requires the hostname, the GPU identity and '
        'no method process, so `runner_process_present` fails the gate with a passing pane condition.', ''])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--host-recheck', required=True)
    ap.add_argument('--repo-root', default=str(Path(__file__).resolve().parents[1]))
    args = ap.parse_args(argv)
    root = Path(args.repo_root)
    doc = build(root, json.loads(Path(args.host_recheck).read_text()))
    (root / EVIDENCE['record']).write_text(json.dumps(doc, indent=2, sort_keys=True) + '\n')
    (root / EVIDENCE['report']).write_text(report(doc))
    (root / EVIDENCE['sha256']).write_text(''.join('%s  %s\n' % (sha256_bytes((root / p).read_bytes()), p) for p in (
        EVIDENCE['record'], EVIDENCE['report'], RULING_FILE, DOCUMENT)))
    print(doc['status'], doc['failed_checks'])
    return 0 if not doc['failed_checks'] else 1


if __name__ == '__main__':
    sys.exit(main())
