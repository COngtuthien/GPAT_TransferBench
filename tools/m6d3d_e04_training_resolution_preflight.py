#!/usr/bin/env python3
"""Static M6D3d owner-contract/audit validation; no graph, TF or data imports.

The YAML overlay uses JSON syntax (a YAML subset) for dependency-free parsing.
--rebuild-index is the only write mode: it rebuilds the CSV from committed
metadata plus five explicit new artifacts, without opening unchanged targets.
--self-test checks rejection of in-memory contract mutations; no tensors run.
"""
import argparse
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = '80dd758475f985377170308351ed2cf99a809d7d'
BASE = 'outputs/audit/M6D3D_E04_TRAINING_GRAPH_RESOLUTION'
DOCUMENT = ('docs/spec/amendments/'
            'GPAT_TransferBench_v1_0_E04_Training_Graph_Owner_Resolution_Addendum_M6D3d.md')
OVERLAY = 'configs/amendments/e04_training_graph_resolution.yaml'
TOOL = 'tools/m6d3d_e04_training_resolution_preflight.py'
ARTIFACTS = (DOCUMENT, OVERLAY, BASE + '.md', BASE + '.json', TOOL)
LEDGER = 'outputs/audit/EXECUTION_LEDGER.jsonl'
INDEX = 'outputs/audit/ARTIFACT_INDEX.csv'
HISTORY = ('outputs/audit/M6D3B_E04_TRAINING_GRAPH_QUALIFICATION.md',
           'outputs/audit/M6D3B_E04_TRAINING_GRAPH_QUALIFICATION.json',
           'tools/m6d3b_e04_training_preflight.py',
           'outputs/audit/M6D3C_E04_TRAINING_GRAPH_QUALIFICATION.md',
           'outputs/audit/M6D3C_E04_TRAINING_GRAPH_QUALIFICATION.json',
           'tools/m6d3c_e04_training_preflight.py',
           'docs/spec/amendments/GPAT_TransferBench_v1_0_E04_Training_Graph_Resolution_Addendum.md')
PREFIX_SHA = '354d907ecfbfebb757682d075e8184b4289b4858d308445d678040a7ab57dbea'
FINAL_STATUS = [
    'E04_GEOMETRY_RUNTIME_QUALIFIED',
    'E04_TRAINING_GRAPH_CONTRACT_RESOLVED',
    'E04_TRAINING_GRAPH_NOT_YET_QUALIFIED',
    'E04_TRAINING_RUNTIME_NOT_YET_QUALIFIED',
    'IMPLEMENTED_NOT_EXECUTED', 'CONTROLLED_ADAPTATION',
]


def require(ok, reason):
    if not ok:
        raise ValueError('M6D3d contract verification failed: ' + reason)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args])


def read(relative):
    p = Path(relative)
    require(not p.is_absolute() and '..' not in p.parts, 'relative evidence path')
    require(p.parts[0] in {'configs', 'docs', 'methods', 'outputs', 'tools',
                           'tests', 'environments', 'frozen_config_snapshot'},
            'evidence root only')
    require(p.suffix.lower() not in {'.parquet', '.jpg', '.jpeg', '.png', '.bmp',
                                    '.webp', '.avi', '.mp4'}, 'no data/image payload')
    return (ROOT / relative).read_bytes()


def validate_contract(c):
    require(c['status'] == 'E04_TRAINING_GRAPH_CONTRACT_RESOLVED', 'resolved status')
    require(c['authority_commit'] == AUTHORITY, 'authority commit')
    require(c['amendment_document'] == DOCUMENT, 'current additive document')
    require(c['fidelity_class'] == 'CONTROLLED_ADAPTATION', 'controlled fidelity')
    e = c['encoder']
    require(e['provenance_class'] == 'PAPER' and e['status'] == 'RESOLVED',
            'encoder is resolved PAPER evidence')
    require(e['features_hwc'] == {'F1': [128, 128, 64], 'F2': [64, 64, 96],
                                 'F3': [32, 32, 128]}, 'paper feature widths')
    require(e['convblocks'] == ['k3c64s2', 'k3c96s2', 'k3c128s2'], 'paper ConvBlocks')
    p = c['binary_only_p0']
    require(p['definition'] == 'zeros_like(P)' and p['applies_to'] == 'every_spoof_sample',
            'zero P0 for every spoof')
    require(p['provenance_class'] == 'CONTROLLED_RECONSTRUCTION' and
            p['provenance'] == 'CONTROLLED_RECONSTRUCTION_BINARY_ONLY_P0',
            'P0 controlled provenance')
    require(p['negative_prior'] == {'contribution': 0.0,
            'evaluation': 'constant_zero_without_evaluating_normalized_quotient',
            'denominator_epsilon': None}, 'zero-prior 0/0 bypass, no invented epsilon')
    require(p['positive_term'] == '||P - (T_A > beta)||^2' and
            p['positive_term_active'] is True and p['inpainting_branch_active'] is True,
            'active positive mask term and inpainting')
    require(p['attack_type_labels_used'] is False and
            p['dataset_specific_spatial_prior'] is False, 'binary-only supervision')
    h = c['step3_targets']
    require(h['sample_class'] == 'spoof' and h['provenance_class'] == 'PAPER',
            'hard sample paper-derived spoof')
    require(h['depth_M0'] == {'value': 0.0, 'dtype': 'float32',
            'shape_per_sample': [32, 32], 'exact_zero': True}, 'exact zero hard depth')
    require(h['L_S'] == {'target': 'warped/synthesized ground-truth trace',
            'target_stop_gradient': True, 'synthetic_input_stop_gradient': True,
            'provenance_class': 'PAPER'}, 'paper trace target and stops')
    require(h['original_live_depth_rule'] == 'unchanged_A3_A4_not_zeroed',
            'do not zero original live targets')
    o = c['optimization']
    require(o['step_order'] == ['step1_generator_eq23', 'step2_discriminator_L_D',
                              'step3_generator_eq24'], 'G D G ordering')
    require(o['all_steps_every_minibatch'] is True and
            o['separate_sequential_apply_operations'] is True, 'three sequential applies')
    require(o['optimizer_state'] == {'generator_instances': 1,
            'generator_shared_between_steps': [1, 3], 'generator_slots_shared': True,
            'generator_beta_power_accumulators_shared': True,
            'discriminator_instances': 1, 'discriminator_state_separate': True,
            'reset_generator_state_between_steps': False}, 'shared G / separate D Adam')
    adam = {'algorithm': 'Adam', 'semantic_reference': 'tf.train.AdamOptimizer',
            'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-8, 'weight_decay': 0.0}
    require(o['adam'] == adam, 'A4 Adam values')
    require(o['generator_learning_rate_multiplier'] == 1.0 and
            o['discriminator_learning_rate_multiplier'] == 0.5, 'half D LR')
    require(o['global_iteration'] == {'unit': 'complete_Phystd_minibatch_iteration',
            'initial_value': 0, 'increments_per_minibatch': 1,
            'increment_after': 'step3_generator_eq24',
            'increment_on_each_apply': False, 'schedule_index': 'global_iteration',
            'same_schedule_index_for_all_three_steps': True}, 'outer iteration ownership')
    require(o['provenance'] == {'three_steps_and_half_discriminator_lr': 'PAPER',
            'adam_semantics': 'FROZEN_BENCHMARK_CONTRACT_A4',
            'shared_generator_state_and_iteration_ownership': 'CONTROLLED_RECONSTRUCTION',
            'execution_resolution_authority': 'M6D3d_OWNER_APPROVED'}, 'optimizer provenance')
    require(c['stdn_reference_policy'] == {'role': 'OFFICIAL_PREDECESSOR_REFERENCE_ONLY',
            'use_only_where_phystd_silent': True, 'inherit_G_D_RATIO': False,
            'skip_any_phystd_step': False}, 'no STDN ratio scheduling')
    f = c['preserved_contract']
    require(f == {'input_resolution': 256, 'batch_size': 8, 'total_iterations': 150000,
            'weight_init': {'distribution': 'Normal', 'mean': 0.0, 'stddev': 0.02},
            'learning_rate': 5e-5, 'lr_schedule': {'type': 'step_divide', 'factor': 10,
                'every_iterations': 45000},
            'alphas': [100, 5, 1, 1e-4, 10, 1], 'beta': 0.1, 'lambda': 1, 'K': 32,
            'eq23': '100*L_depth + 5*L_G + L_P + 1e-4*L_R',
            'eq24': '10*L_S + L_H', 'terminal_global_iteration': 150000,
            'experiment_seeds': [42, 1337, 2026], 'supervision': 'binary_weak_live_spoof',
            'attack_type_labels': 'FORBIDDEN', 'TEST': 'FORBIDDEN'}, 'frozen settings retained')
    require(c['execution_authorized_by_this_milestone'] is False, 'resolution only')
    require(c['final_status'] == FINAL_STATUS, 'final statuses')


def rejection_checks(c):
    """Semantic regressions checked on dictionaries, without running any graph."""
    changes = [
        (('encoder', 'provenance_class'), 'CONTROLLED_RECONSTRUCTION'),
        (('encoder', 'features_hwc', 'F2'), [64, 64, 64]),
        (('binary_only_p0', 'negative_prior', 'evaluation'), 'normalized_quotient'),
        (('binary_only_p0', 'inpainting_branch_active'), False),
        (('binary_only_p0', 'attack_type_labels_used'), True),
        (('step3_targets', 'depth_M0', 'value'), 1.0),
        (('step3_targets', 'L_S', 'target_stop_gradient'), False),
        (('optimization', 'step_order'), ['step1_generator_eq23',
                                        'step3_generator_eq24', 'step2_discriminator_L_D']),
        (('optimization', 'optimizer_state', 'generator_instances'), 2),
        (('optimization', 'global_iteration', 'increments_per_minibatch'), 3),
        (('optimization', 'global_iteration', 'schedule_index'), 'optimizer_apply_count'),
        (('optimization', 'discriminator_learning_rate_multiplier'), 1.0),
        (('optimization', 'adam', 'epsilon'), 1e-7),
        (('stdn_reference_policy', 'inherit_G_D_RATIO'), True),
    ]
    for path, value in changes:
        invalid = copy.deepcopy(c)
        parent = invalid
        for key in path[:-1]:
            parent = parent[key]
        parent[path[-1]] = value
        try:
            validate_contract(invalid)
        except ValueError:
            continue
        raise ValueError('Invalid contract accepted: ' + '.'.join(path))
    return len(changes)


def expected_index():
    reader = csv.DictReader(io.StringIO(git('show', AUTHORITY + ':' + INDEX).decode()))
    require(reader.fieldnames == ['path', 'size_bytes', 'sha256'], 'index schema')
    baseline = list(reader)
    rows = {r['path']: r for r in baseline}
    require(len(rows) == len(baseline), 'index unique paths')
    for relative in ARTIFACTS:
        require(relative not in rows, 'new additive artifact ' + relative)
        raw = read(relative)
        rows[relative] = {'path': relative, 'size_bytes': str(len(raw)), 'sha256': sha(raw)}
    out = io.StringIO(newline='')
    writer = csv.DictWriter(out, fieldnames=reader.fieldnames, lineterminator='\r\n')
    writer.writeheader()
    writer.writerows(rows[key] for key in sorted(rows))
    return out.getvalue().encode(), len(baseline), len(rows)


def verify(check_index=True, self_test=False):
    c = json.loads(read(OVERLAY))
    validate_contract(c)
    audit = json.loads(read(BASE + '.json'))
    require(audit['decision'] == 'OWNER_RESOLUTION_RECORDED' and
            audit['final_status'] == FINAL_STATUS, 'audit resolution status')
    require(audit['execution'] == {'graph_implementation': False, 'graph_execution': False,
            'optimizer_applications': 0, 'tensorflow_imported_or_run': False,
            'environment_created': False, 'benchmark_data_access': False}, 'no execution')
    require(audit['owner_resolutions_complete'] is True, 'owner decisions recorded')
    table = audit['resolved_decision_table']
    prior = json.loads(read(HISTORY[4]))['training_graph_evidence_resolution']['source_resolution_table']
    require([r['component'] for r in table] == [r['component'] for r in prior],
            'complete 29-component decision mapping')
    for r in table:
        require(r['status'] == 'CONTRACT_RESOLVED' and r['resolution'], 'resolved decision')
    require(set(HISTORY) <= set(audit['preserved_inputs_sha256']), 'history coverage')
    for relative, expected in audit['preserved_inputs_sha256'].items():
        raw = read(relative)
        require(sha(raw) == expected and raw == git('show', AUTHORITY + ':' + relative),
                'unchanged prior authority ' + relative)
    for relative, expected in c['immutable_inputs_sha256'].items():
        require(sha(read(relative)) == expected, 'overlay input ' + relative)
    for relative, expected in audit['resolution_artifacts_sha256'].items():
        require(sha(read(relative)) == expected, 'resolution artifact ' + relative)
    prefix = git('show', AUTHORITY + ':' + LEDGER)
    current = read(LEDGER)
    require(len(prefix) == 265194 and len(prefix.splitlines()) == 103 and
            sha(prefix) == PREFIX_SHA, 'committed 103-row prefix')
    require(current.startswith(prefix) and len(current.splitlines()) == 104,
            'exactly one append / unchanged prefix')
    appended = json.loads(current[len(prefix):])
    require(appended['classification'] == 'M6D3D_E04_TRAINING_GRAPH_RESOLUTION' and
            appended['decision'] == audit['decision'], 'M6D3d ledger row')
    require(set(appended['file_sha256']) == set(ARTIFACTS), 'ledger artifact coverage')
    for relative, expected in appended['file_sha256'].items():
        require(sha(read(relative)) == expected, 'ledger digest ' + relative)
    expected, carried, total = expected_index()
    if check_index:
        require(read(INDEX) == expected, 'CRLF index rebuilt from approved paths only')
    negative = rejection_checks(c) if self_test else 0
    return {'static_contract_and_audit': 'PASS', 'resolved_components': len(table),
            'invalid_contracts_rejected': negative, 'ledger_rows': 104,
            'first_103_rows_byte_identical': True,
            'index_rows_carried_without_target_access': carried, 'index_rows': total,
            'index_checked': check_index, 'final_status': FINAL_STATUS,
            'qualification_performed': False, 'optimizer_applications': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--rebuild-index', action='store_true')
    args = parser.parse_args()
    if args.rebuild_index:
        require(git('rev-parse', 'HEAD').decode().strip() == AUTHORITY,
                'index write only at starting authority')
        verify(check_index=False, self_test=args.self_test)
        raw, _, _ = expected_index()
        (ROOT / INDEX).write_bytes(raw)  # LAST repository write.
    print(json.dumps(verify(self_test=args.self_test), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
