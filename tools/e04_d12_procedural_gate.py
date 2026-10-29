"""E04 D12 procedural-reproducibility acceptance gate: OWNER_APPROVED (c60481b), ACTIVE.

Implements the replacement acceptance criteria of configs/amendments/e04_d12_procedural_reproducibility.yaml (the DRAFT
committed at 57bd1f9, criteria unchanged) for the failed bitwise gate K.real_data_gradient_repeatability. Its only
caller is `tools/e04_production_qualification_evidence.py build-d12`, which writes a NEW requalification record; the
original 41/42 record is never rewritten. gate() refuses unless the approved amendment exists with status
OWNER_APPROVED and pins CRITERIA_SHA256; evaluate() is the pure criterion logic. The superseded K gate is reported as
recorded and never counted as passed. No criterion depends on performance.

Inputs (formats of the unchanged D12 tools): replay process directories of tools/e04_d12_production_replay.py
(summary.json, iterations.jsonl), traced partition records of tools/e04_d12_confound_probe.py --factor d10_trace2
(d12_extra.json 'partitions'), and the gate list of the requalification record. Stdlib only.
"""
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import e04_d11_analysis as d11a  # noqa: E402
import e04_d12_analysis as d12a  # noqa: E402

ACTIVE = True                           # owner approval c60481b; activation commit wires build-d12
APPROVED_AMENDMENT = 'configs/amendments/e04_d12_procedural_reproducibility.yaml'
SUPERSEDED_GATE = 'K.real_data_gradient_repeatability'
BOUND = d12a.BOUND                      # 1e-3, fixed by E04_D12_INVESTIGATION_PROTOCOL.json before any D12 evidence
QUALIFICATION_SEED = 60904
QUALIFIED_CONSTRUCTION = d12a.QUALIFIED_CONSTRUCTION
INITIAL_STATE_SHA256 = '951dad42fd2e2c6d09622bbc12dd50fa7094d4c611db45467fe996beeb107df0'
CANONICAL_SET_SHA256 = d12a.D11_ORIGINAL_CANONICAL_SET
RETVAL_CANONICAL_SHA256 = d12a.D11_ORIGINAL_RETVAL
STEPS = ('step1', 'step2', 'step3')
TRAINABLES_PER_STEP = 72
MIN_REPLAY_PROCESSES, MIN_ITERATIONS, MIN_TRACED_PROCESSES = 3, 3, 2
RETAINED_GATES = (
    'A.frozen_contract_and_owner_resolution', 'A.pinned_manifests', 'A.same_production_identity_all_cases',
    'A.qualification_seed_only', 'B.prepass_repeat_byte_identical', 'B.prepass_population',
    'B.prepass_cpu_qualified_settings', 'B.prepass_firewall', 'B.prepass_samples',
    'C.feed_pack_verified_gpu_and_laptop', 'C.feed_pack_binds_prepass', 'D.det_a_rows_and_hard_draws_equal_replay',
    'D.det_b_rows_and_hard_draws_equal_replay', 'D.det_inputs_identical',
    'D.seed42_full_streams_equal_approved_vectors', 'D.seed1337_full_streams_equal_approved_vectors',
    'D.seed2026_full_streams_equal_approved_vectors', 'E.construction_equals_qualified_m6d3e', 'E.graph_inventory',
    'E.initial_state_identical', 'F.det_a_step_transitions_equal_m6d3e', 'F.det_a_pure_fetch_nonmutating_and_equal',
    'F.det_b_step_transitions_equal_m6d3e', 'F.det_b_pure_fetch_nonmutating_and_equal',
    'F.iteration0_losses_identical_across_processes', 'G.losses_finite', 'H.det_a_periodic_checkpoints',
    'H.det_b_periodic_checkpoints', 'H.terminal_probe_selected_semantics', 'H.failure_probe',
    'I.det_a_summary_and_records', 'I.det_b_summary_and_records', 'I.terminal_probe_summary_and_records',
    'I.failure_probe_summary_and_records', 'J.firewall_all_cases', 'J.atime_test_unchanged', 'J.manifests_unmodified',
    'J.atime_test_unchanged_after_diagnostics', 'J.no_scientific_run_root', 'K.environment_lock_except_driver',
    'K.d9_compatibility_reprobe_matches_accepted_variant')

CRITERIA = {
    'R1_graph_identity': 'every replay process builds the qualified construction (61f1ed58..., 11315 ops; serialized '
                         'GraphDef incl. requested devices) and adds no op during the session; every traced process '
                         'has the canonical partition-graph node set 094566e1... and fetched-output canonical hash '
                         'fea3730e...; executed device types of common ops are equal across traced processes',
    'R2_deterministic_inputs_and_initial_state': 'at least 3 replay processes x 3 iterations at qualification seed '
                                                 '60904; identical initial state 951dad42...; per iteration identical '
                                                 'pair rows and input tensor sha256 across processes',
    'R3_gradient_connectivity': 'for every loss and step: gradient presence equals differentiable reachability with '
                                'finite values and matching shapes; each step has a gradient for all 72 of its '
                                'trainable variables',
    'R4_optimizer_application': 'per step exactly one ApplyAdam per present gradient, targeting its variable and '
                                'consuming that gradient; no stateful random op on a step or BN-update path',
    'R5_state_transitions': 'every semantic replay: only the allowed variables change (BN calls: moving statistics of '
                            'that path; apply: the step variables, their Adam slots, that optimizer beta powers, phase '
                            'and, in step 3, global_iteration); Adam within 4 ulp + 1e-5 |delta| of the TF 1.15 '
                            'ApplyAdam reference; beta powers exact; phase 1, 2, 0 and iteration +1 after step 3; '
                            'every restore bit-exact; diagnostics never mutate state',
    'R6_finite': 'state, losses, gradients and every compared forward tensor finite at every checked iteration',
    'R7_forward_integrity': 'at fixed state: no fed input varies, no data-movement op varies with identical inputs in '
                            'the same call, every varying forward tensor has relative L2 difference <= 1e-3',
    'R8_gradient_integrity': 'for every step replay: functional-variable step-gradient relative L2 difference '
                             '(BN-cancelled convolution biases excluded) <= 1e-3',
    'R9_retained_gates': 'the requalification record contains exactly the 41 retained gates of the original '
                         'qualification and every one passes (checkpoint lifecycle H, records I, TEST isolation J, '
                         'data/RNG D, graph E/F, finiteness G, authority A/B/C, runtime K1/K3-compat)',
    'R10_firewall': 'every replay and traced process: no denied access and no runtime write',
}
CRITERIA_SHA256 = hashlib.sha256(json.dumps(CRITERIA, sort_keys=True).encode()).hexdigest()


def jl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def load_replay(d):
    d = Path(d)
    return {'summary': json.loads((d / 'summary.json').read_text()), 'iterations': jl(d / 'iterations.jsonl')}


def functional_gradient_ratio(gradient_repeat):
    canc = d12a.cancelled(gradient_repeat['per_variable_l2_ref'])
    ref = sum(x ** 2 for n, x in gradient_repeat['per_variable_l2_ref'].items() if n not in canc) ** .5
    dif = sum(x['l2_diff'] ** 2 for x in gradient_repeat['differing_variables'] if x['variable'] not in canc) ** .5
    return dif / ref if ref > 0 else None


def evaluate(replays, traces, gates):
    """replays: list of {'summary', 'iterations'}; traces: list of {'partitions', 'first_trace_place'};
    gates: the requalification record's gate list [{'gate', 'pass'}]. Returns criteria, pass and diagnostics."""
    c = {k: True for k in CRITERIA}
    why = {k: [] for k in CRITERIA}

    def fail(k, msg):
        c[k] = False
        why[k].append(msg)

    if len(replays) < MIN_REPLAY_PROCESSES:
        fail('R2_deterministic_inputs_and_initial_state', 'fewer than %d replay processes' % MIN_REPLAY_PROCESSES)
    if len(traces) < MIN_TRACED_PROCESSES:
        fail('R1_graph_identity', 'fewer than %d traced processes' % MIN_TRACED_PROCESSES)
    diag = {'bitwise_gradient_variables_differing': {}, 'cross_process_post_state_distinct': [],
            'cross_process_losses_distinct': [], 'forward_varying_tensors': [], 'functional_gradient_ratio': {}}
    for i, r in enumerate(replays):
        s, st = r['summary'], r['summary']['static']
        if st['construction'] != QUALIFIED_CONSTRUCTION:
            fail('R1_graph_identity', 'process %d construction %s' % (i, st['construction']))
        if s['graph_operations_before_session'] != s['graph_operations_after_session']:
            fail('R1_graph_identity', 'process %d graph changed during session' % i)
        if s['qualification_seed'] != QUALIFICATION_SEED or s['initial_state_sha256'] != INITIAL_STATE_SHA256:
            fail('R2_deterministic_inputs_and_initial_state', 'process %d seed or initial state' % i)
        if len(r['iterations']) < MIN_ITERATIONS or len(r['iterations']) != s['iterations']:
            fail('R2_deterministic_inputs_and_initial_state', 'process %d iterations' % i)
        if s['firewall']['denied'] or s['firewall']['runtime_writes']:
            fail('R10_firewall', 'process %d firewall' % i)
        for name, v in st['gradient_connectivity'].items():
            if not (v['present_equals_reachable'] and v['all_present_finite_shape']):
                fail('R3_gradient_connectivity', 'process %d %s' % (i, name))
        for step in STEPS:
            v = st['gradient_connectivity'].get(step)
            if v is None or v['present'] != TRAINABLES_PER_STEP:
                fail('R3_gradient_connectivity', 'process %d %s present' % (i, step))
        if set(st['optimizer_application']) != {'1', '2', '3'}:
            fail('R4_optimizer_application', 'process %d steps %s' % (i, sorted(st['optimizer_application'])))
        for k, v in st['optimizer_application'].items():
            if not (v['apply_adam_ops'] == v['present_gradients'] and v['targets_equal'] and
                    v['apply_inputs_are_step_gradients']):
                fail('R4_optimizer_application', 'process %d step %s apply' % (i, k))
            if v['random_ops_on_path'] or v['random_ops_on_bn_path']:
                fail('R4_optimizer_application', 'process %d step %s random ops' % (i, k))
        for it in r['iterations']:
            w = 'process %d t%d' % (i, it['iteration'])
            if not it['forward_state_unchanged']:
                fail('R5_state_transitions', w + ' diagnostic mutated state')
            if not it['state_finite'] or not all(x.get('finite', True) for x in it['forward']['varying']):
                fail('R6_finite', w)
            for v in d12a.forward_violations(it['forward']['varying']):
                fail('R7_forward_integrity', '%s %s %s' % (w, v['tensor'], v['why']))
            diag['forward_varying_tensors'].append(len({x['tensor'] for x in it['forward']['varying']
                                                         if not x['gradient']}))
            if set(it['steps']) != set(STEPS):
                fail('R5_state_transitions', w + ' steps %s' % sorted(it['steps']))
            for step, v in it['steps'].items():
                if not (v['production_restore_exact'] and v['semantic_restore_exact']):
                    fail('R5_state_transitions', '%s %s restore' % (w, step))
                if not v['semantic']:
                    fail('R5_state_transitions', '%s %s no semantic replay' % (w, step))
                for x in v['semantic']:
                    for k in ('scope_ok', 'adam_ok', 'beta_powers_ok', 'counters_ok'):
                        if not x[k]:
                            fail('R5_state_transitions', '%s %s %s' % (w, step, k))
                    if not x['finite']:
                        fail('R6_finite', '%s %s semantic' % (w, step))
                ratio = functional_gradient_ratio(v['gradient_repeat'])
                diag['functional_gradient_ratio']['%s %s' % (w, step)] = ratio
                if ratio is None or not ratio <= BOUND:
                    fail('R8_gradient_integrity', '%s %s ratio %s' % (w, step, ratio))
                diag['bitwise_gradient_variables_differing']['%s %s' % (w, step)] = \
                    len(v['gradient_repeat']['differing_variables'])
    if len({json.dumps([it['pair_rows'] for it in r['iterations']]) for r in replays}) > 1 or \
            len({json.dumps([it['input_sha256'] for it in r['iterations']], sort_keys=True) for r in replays}) > 1:
        fail('R2_deterministic_inputs_and_initial_state', 'inputs differ across processes')
    if replays and len({len(r['iterations']) for r in replays}) == 1:
        for t in range(len(replays[0]['iterations'])):
            diag['cross_process_post_state_distinct'].append(len({r['iterations'][t]['state_sha256_after']
                                                                  for r in replays}))
            diag['cross_process_losses_distinct'].append(len({json.dumps([r['iterations'][t]['steps'].get(s, {}).get(
                'production_losses', [None])[0] for s in STEPS]) for r in replays}))
    ref_place = None
    for j, t in enumerate(traces):
        memo, nodes = d11a.canonical(t['partitions'])
        sset = hashlib.sha256('\n'.join(sorted(set(memo.values()))).encode()).hexdigest()
        retval = hashlib.sha256('\n'.join(sorted(memo[n] for n, v in nodes.items() if v['op'] == '_Retval'))
                                .encode()).hexdigest()
        if sset != CANONICAL_SET_SHA256 or retval != RETVAL_CANONICAL_SHA256:
            fail('R1_graph_identity', 'trace %d canonical partitions' % j)
        dt = {n: d11a.dev_type(d) for n, d in (t.get('first_trace_place') or {}).items()}
        if not dt:
            fail('R1_graph_identity', 'trace %d has no executed placement' % j)
        elif ref_place is None:
            ref_place = dt
        elif any(dt[n] != ref_place[n] for n in set(dt) & set(ref_place)):
            fail('R1_graph_identity', 'trace %d executed placement differs' % j)
    names = [g['gate'] for g in gates]
    retained = [g for g in gates if g['gate'] != SUPERSEDED_GATE]
    if sorted(x['gate'] for x in retained) != sorted(RETAINED_GATES) or len(names) != len(set(names)):
        fail('R9_retained_gates', 'gate set differs from the 41 retained gates')
    for g in retained:
        if not g['pass']:
            fail('R9_retained_gates', g['gate'])
    superseded = [g for g in gates if g['gate'] == SUPERSEDED_GATE]
    return {'criteria_sha256': CRITERIA_SHA256, 'criteria': c, 'failures': {k: v[:50] for k, v in why.items() if v},
            'pass': all(c.values()), 'diagnostics_not_gating': diag,
            'superseded_gate': {'gate': SUPERSEDED_GATE,
                                'recorded_pass': superseded[0]['pass'] if superseded else None,
                                'counted_as_passed': False}}


def require_approval(repo):
    p = Path(repo) / APPROVED_AMENDMENT
    if not p.is_file():
        raise SystemExit('E04 D12 gate inactive: %s does not exist (owner approval required)' % APPROVED_AMENDMENT)
    a = json.loads(p.read_text())
    if not (a.get('status') == 'OWNER_APPROVED' and a.get('approved') is True and
            a.get('acceptance_criteria_sha256') == CRITERIA_SHA256):
        raise SystemExit('E04 D12 gate inactive: %s is not an OWNER_APPROVED amendment pinning criteria %s'
                         % (APPROVED_AMENDMENT, CRITERIA_SHA256))
    return a


def gate(repo, replay_dirs, trace_records, gates):
    if not ACTIVE:
        raise SystemExit('E04 D12 gate inactive')
    require_approval(repo)
    return evaluate([load_replay(d) for d in replay_dirs], trace_records, gates)
