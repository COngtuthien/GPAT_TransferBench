#!/usr/bin/env python3
"""Static M6D4c owner-contract validation; never a numerical training runner.

Run with python3 -I -S -B. The overlay uses the JSON subset of YAML so only
the standard library is needed. Formula strings are compared, never evaluated.
Only fixed evidence paths are read; historical index targets are never opened.
--rebuild-index is the sole write operation and must be the final artifact write.
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
AUTHORITY = '5adb76285b9b9c4dc25bb4a33f3be7d8575a8abc'
OVERLAY = 'configs/amendments/e05_training_runner_resolution.yaml'
ADDENDUM = 'docs/spec/amendments/GPAT_TransferBench_v1_0_E05_Training_Runner_Owner_Resolution_Addendum_M6D4c.md'
BASE = 'outputs/audit/M6D4C_E05_TRAINING_RUNNER_RESOLUTION'
SCRIPT = 'tools/m6d4c_e05_training_resolution_preflight.py'
LEDGER = 'outputs/audit/EXECUTION_LEDGER.jsonl'
INDEX = 'outputs/audit/ARTIFACT_INDEX.csv'
ARTIFACTS = (OVERLAY, ADDENDUM, SCRIPT, BASE+'.md', BASE+'.json')
OWNER = 'CONTROLLED_RECONSTRUCTION_OWNER_RESOLUTION'
PREDECESSOR = 'CONTROLLED_RECONSTRUCTION_PREDECESSOR_DERIVED'
PRESERVED = 'M6D4B_RESOLVED_SEMANTICS_PRESERVED'
FINAL_STATUS = [
    'E05_ARCHITECTURE_RUNTIME_QUALIFIED',
    'E05_TRAINING_RUNNER_CONTRACT_RESOLVED',
    'E05_TRAINING_RUNNER_NOT_YET_QUALIFIED',
    'IMPLEMENTED_NOT_EXECUTED', 'CONTROLLED_ADAPTATION',
]
INPUT_HASHES = {
    'configs/methods/e05_pcgan.yaml': '478756e150c427832315800acba954cedf778bac520eee3bde8cabf7359e71de',
    'configs/amendments/e05_a5_blur_operator_resolution.yaml': 'af28a48b43828652a7423235aba6be239325e3fbfeeaf8fded9bbb57c11a7ca1',
    'outputs/audit/M6D4B_E05_TRAINING_RUNNER_QUALIFICATION.json': 'd35247075e38b3a4d1cc2156faa030bb04058bd1d8b906a242d4e7a88b25519f',
    'outputs/audit/M6D4B_E05_TRAINING_RUNNER_QUALIFICATION.md': '37589973b502e71743b665828c88804b43fcd325ca6a8e82e3f04dcf87b81bf6',
    'docs/spec/amendments/GPAT_TransferBench_v1_0_Amendment_A2_M6_Baseline_Execution_Contracts.md': 'b4fa7bfa75e5a977348c468d1bfe3e0004c3920293ecd9178700f9f4869fca8d',
    'docs/spec/amendments/GPAT_TransferBench_v1_0_Amendment_A5_E05_Blur_Operator_Resolution.md': '05d6c6c271bf23aece3b2fcf9cdef5d94bd32517421a0ac83be26c5a9174d4dd',
    'outputs/audit/M6D4A_E05_RUNTIME_ARCHITECTURE_QUALIFICATION.md': 'f57d4e99c42a633b02ff41d4af0b017c0970260242743bc8752798731411bbc6',
    'methods/pcgan/source_traceability.md': '61645843f65a51e7858787cfbc7cf30f1ff75763c8ded6da41d96410d4c5c1bd',
}
HISTORICAL_HASHES = {
    'outputs/audit/M6D4B_E05_SYNTHETIC_PROCESS_1.json': '29cab67725061aa062e9e30f43aeb26dffe09b6d5a42a8ea6438519b9d9eeea3',
    'outputs/audit/M6D4B_E05_SYNTHETIC_PROCESS_2.json': 'ad5ae34fb3b543e3caca2a08ca44247903f05091e02568ee45e234d6fb948c8e',
    'tests/test_m6d4b_e05_stop.py': 'f010154effb4f4c9e376c4120ab0482e8ee4144824e56363e7d6631a83598439',
    'tools/m6d4b_e05_training_preflight.py': '010176aa1f239d17fd0c909784e095ccec7a9cda14b2cb43441bed8fb28fa9b6',
}
PREFIX_SHA = 'e8c46615e36cd087901cb757ee9f2b3a216b7910a842ae50dbf44880f97cc65d'
TOPICS = dict(zip('ABCDEFGHIJ', (
    'L_rec', 'L_recblur', 'L_advrec', 'L_advmix', 'L_pat',
    'discriminator', 'parameter ownership', 'update schedule',
    'StyleGAN noise', 'batch-one explicit pair')))
DISPOSITIONS = [dict(item=k, topic=v, status='CONTRACT_RESOLVED') for k, v in TOPICS.items()]
EXECUTION = dict.fromkeys((
    'runner_implemented', 'runtime_qualification', 'numerical_qualification',
    'optimizer_instantiated', 'model_constructed', 'model_execution',
    'torch_import', 'tensorflow_import', 'cuda_execution', 'environment_mutation',
    'benchmark_training', 'benchmark_data_access', 'benchmark_image_filename_enumeration',
    'benchmark_image_decode', 'manifest_sample_access', 'TRAIN_sample_access',
    'VAL_sample_access', 'TEST_sample_access', 'scientific_checkpoint',
    'synthetic_bank', 'commit', 'push'), False)
EXECUTION['optimizer_applications'] = 0

# Normative static schema. Exact key sets prohibit hidden extra losses/overrides.
# Provenance is field-specific: existing paper distances are not owner inventions.
EXPECTED = {
    'schema_version': 1,
    'milestone': 'M6D4c',
    'status': 'OWNER_APPROVED_ADDITIVE_CONTROLLED_RECONSTRUCTION',
    'authority_commit': AUTHORITY,
    'method_id': 'E05',
    'reporting_label': 'PCGAN (controlled architecture resolution)',
    'fidelity_class': 'CONTROLLED_ADAPTATION',
    'scope': 'CONTRACT_ONLY_NO_RUNNER_IMPLEMENTATION_OR_EXECUTION',
    'immutable_inputs_sha256': INPUT_HASHES,
    'historical_stop_preserved': True,
    'forbidden_decision_attribution': ['PAPER', 'OFFICIAL_PCGAN', 'AUTHOR_SPECIFIED',
                                       'native PCGAN', 'faithful PCGAN', 'official reproduction'],
    'predecessor': {
        'repository': 'taesungp/swapping-autoencoder-pytorch',
        'commit': '6baa180f1184ee79a6b967f9d80ee0e02a979ac7',
        'role': 'nearest_executable_predecessor_not_official_PCGAN',
        'derived_decision_provenance': PREDECESSOR,
    },
    'pair': {
        'batch_size': 1, 'distinct_source_target': True,
        'input_dtype': 'FP32', 'input_shape_each': [1, 3, 256, 256],
        'encode_source': '(z_pat_src,z_con_src)=E(x_src)',
        'encode_target': '(z_pat_tgt,z_con_tgt)=E(x_tgt)',
        'r_src': 'G(z_pat_src,z_con_src)', 'm': 'G(z_pat_src,z_con_tgt)',
        'spatial_shape': [1, 8, 128, 128], 'global_shape': [1, 2048],
        'even_minibatch_swap': False, 'same_pair_within_iteration': True,
    },
    'distance': {
        'formula': 'sqrt(sum_C,H,W((a[n]-b[n])**2))',
        'name': 'unsquared_Euclidean_L2', 'outer_reduction': 'arithmetic_mean_over_n',
        'pixel_count_normalization': False, 'squared': False,
        'provenance': 'PAPER_EXISTING_FROZEN_EVIDENCE',
        'forbidden_substitutions': ['MSE', 'RMS', 'L1', 'whole_batch_norm'],
    },
    'generator': {
        'L_rec': {
            'formula': 'mean_n(norm2(x_src[n]-r_src[n]))',
            'samples': ['x_src'], 'target_reconstruction_included': False,
            'batch_one': 'one_unsquared_Euclidean_norm_over_source_image',
            'estimator_provenance': OWNER,
            'rationale': 'Eq.2 is generic; Eq.4 and L_advrec identify source reconstruction. Source-only is the minimal explicit-pair estimator without an invented target contribution.',
        },
        'L_recblur': {
            'formula': 'mean_n(norm2(B(x_tgt[n])-B(m[n])))',
            'detach': False, 'provenance': PRESERVED,
            'blur': {'operator': 'avg_pool2d', 'kernel_size': 2, 'stride': 2,
                     'padding': 0, 'ceil_mode': False, 'count_include_pad': False,
                     'application': 'independently_to_target_and_mixed',
                     'input_resolution': 256, 'output_resolution': 128,
                     'learnable_parameters': 0, 'fallbacks': [],
                     'provenance': 'BENCHMARK_DEFINED_CONTROLLED_RECONSTRUCTION_A5'},
        },
        'L_advrec': {'formula': 'mean(softplus(-D(r_src)))', 'provenance': PRESERVED},
        'L_advmix': {'formula': 'mean(softplus(-D(m)))', 'provenance': PRESERVED},
        'L_pat': {
            'reference_crops': 'independent_random_crops(x_src)',
            'candidate_crops': 'independent_random_crops(m)',
            'reference_features': 'extract_features(crops_ref,aggregate=True)',
            'candidate_features': 'extract_features(crops_mixed,aggregate=False)',
            'logits': 'discriminate_features(reference_features,mixed_features)',
            'formula': 'mean(softplus(-PatchD(reference_features,mixed_features)))',
            'detach': False, 'provenance': PRESERVED,
        },
        'logits': 'raw', 'mean_axes': 'batch_and_all_scalar_scores',
        'image_logit_shape': [1, 1], 'patch_logit_shape': [8, 1],
        'weights': {'L_rec': 1, 'L_recblur': 1, 'L_advrec': 1, 'L_advmix': 1, 'L_pat': 1},
        'total': 'L_rec + L_recblur + L_advrec + L_advmix + L_pat',
        'alpha': 0.2, 'beta': 1e-6, 'alpha_beta_scope': 'PMN_only_not_five_term_weights',
    },
    'discriminator': {
        'logits': 'raw', 'mean_axes': 'batch_and_all_scalar_scores',
        'image': {
            'd_src': 'D(x_src)', 'd_tgt': 'D(x_tgt)',
            'd_rec': 'D(r_src.detach())', 'd_mix': 'D(m.detach())',
            'real_samples': ['x_src', 'x_tgt'], 'real_sample_weights': [0.5, 0.5],
            'L_D_real': '0.5*(mean(softplus(-d_src))+mean(softplus(-d_tgt)))',
            'L_D_rec': 'mean(softplus(d_rec))', 'L_D_mix': 'mean(softplus(d_mix))',
            'weights': {'L_D_real': 1.0, 'L_D_rec': 0.5, 'L_D_mix': 0.5},
            'L_D_image': '1.0*L_D_real + 0.5*L_D_rec + 0.5*L_D_mix',
            'logistic_provenance': PREDECESSOR, 'weights_provenance': PREDECESSOR,
            'symmetric_real_samples_provenance': OWNER,
            'symmetric_real_samples_disclosure': 'Both explicit real images contribute equally; this is an owner decision, not a PCGAN paper fact.',
        },
        'patch': {
            'crops_ref': 'independent_random_crops(x_src)',
            'crops_pos': 'another_independent_random_crops(x_src)',
            'crops_fake': 'independent_random_crops(m.detach())',
            'all_three_crop_draws_independent': True,
            'ref_feat': 'extract_features(crops_ref,aggregate=True)',
            'pos_feat': 'extract_features(crops_pos,aggregate=False)',
            'fake_feat': 'extract_features(crops_fake,aggregate=False)',
            'p_real': 'discriminate_features(ref_feat,pos_feat)',
            'p_fake': 'discriminate_features(ref_feat,fake_feat)',
            'L_D_patch_real': 'mean(softplus(-p_real))',
            'L_D_patch_fake': 'mean(softplus(p_fake))',
            'weights': {'L_D_patch_real': 1.0, 'L_D_patch_fake': 1.0},
            'L_D_patch': '1.0*L_D_patch_real + 1.0*L_D_patch_fake',
            'interface_provenance': 'A2_PINNED_EXECUTABLE_ARCHITECTURE_M6D4A_QUALIFIED',
            'construction_and_weights_provenance': PREDECESSOR,
        },
        'L_D_total': 'L_D_image + L_D_patch',
        'combination_weights': {'L_D_image': 1, 'L_D_patch': 1},
        'combination_provenance': OWNER,
        'R1': False, 'patch_R1': False, 'lazy_R1_scaling': False,
        'gradient_penalty': False, 'additional_patch_regularizer': False,
    },
    'excluded_extra_losses': ['VGG', 'perceptual', 'GPAT_identity', 'GPAT_landmark'],
    'optimizers': {
        'G': {'parameters': ['Encoder', 'Generator'], 'name': 'Adam',
              'lr': 1e-6, 'betas': [0.9, 0.999], 'weight_decay': 'NOT_INTRODUCED'},
        'D': {'parameters': ['ImageD', 'PatchD'], 'name': 'Adam',
              'lr': 1e-6, 'betas': [0.9, 0.999], 'weight_decay': 'NOT_INTRODUCED'},
        'overlap': False, 'lazy_R1_scaling': False,
        'grouping_provenance': 'M6D4B_SUPPORTED_GROUPING_PRESERVED',
    },
    'iteration': {
        'unit': 'complete_D_then_G_cycle', 'order': ['D', 'G'],
        'applications_per_iteration': {'D': 1, 'G': 1, 'total': 2},
        'order_provenance': PREDECESSOR, 'grouping_provenance': OWNER,
        'D_step': {
            'fake_generation': 'current_E_G_parameters_from_explicit_pair',
            'r_src_D': 'G(z_pat_src,z_con_src)', 'm_D': 'G(z_pat_src,z_con_tgt)',
            'detach_reconstruction': True, 'detach_mixed': True,
            'objective': 'L_D_total', 'optimizer_applications': 1,
            'parameters_updated': ['ImageD', 'PatchD'], 'E_G_update': False,
        },
        'G_step': {
            'after_D_step': True, 'recompute_encoder_and_generator_paths': True,
            'r_src_G': 'G(z_pat_src,z_con_src)', 'm_G': 'G(z_pat_src,z_con_tgt)',
            'reuse_pre_D_fake_tensors': False, 'same_explicit_pair': True,
            'discriminators': 'now_updated_parameters_frozen_for_parameter_updates',
            'gradients_through_discriminators_to_images_and_E_G': True,
            'objective': 'L_rec + L_recblur + L_advrec + L_advmix + L_pat',
            'optimizer_applications': 1, 'parameters_updated': ['Encoder', 'Generator'],
            'D_PatchD_update': False,
        },
        'counter': {
            'name': 'benchmark_iteration', 'initial': 0,
            'both_steps_share_current_t': True, 'increments_per_complete_cycle': 1,
            'increment_when': 'after_G_step_successfully_completes',
            'increment_per_Adam_application': False, 'transition': 't -> t+1',
            'terminal': 4000,
        },
        'uninterrupted_scientific_run': {'complete_cycles': 4000, 'D_applications': 4000,
                                       'G_applications': 4000, 'count_provenance': OWNER},
        'optimizer_state_convention_is_paper_fact': False,
    },
    'stochasticity': {
        'generator_noise_enabled': True, 'noise_draws': 'fresh_standard_normal_per_pinned_architecture',
        'noise_shape': '[B,1,H,W]', 'initial_trainable_noise_strength': 0,
        'fix_noise': False, 'crop_operator': 'pinned_util.apply_random_crop',
        'crops_per_image': 8, 'crop_size': 128,
        'scale': 'independent_x_y_uniform_[1/8,1/4)', 'horizontal_sign': 'random',
        'offset': '(2*U-1)*(1-scale)', 'grid_sample_align_corners': False,
        'grid_sample_mode': 'bilinear', 'grid_sample_padding_mode': 'zeros',
        'reference_aggregation': 'mean_over_eight_crops_then_expand_to_eight',
        'candidate_aggregation': False, 'center_crop_substitution': False,
        'independent_draws_where_specified': True, 'executed_in_M6D4c': False,
    },
    'precision': {'dtype': 'FP32', 'TF32': False, 'AMP': False,
                  'autocast': False, 'cudnn_benchmark': False},
    'checkpoint': {
        'rule': 'BASELINE_FINAL_STATE_V1', 'terminal_benchmark_iteration': 4000,
        'terminal_save_when': 'after_4000_complete_cycles_if_required_later',
        'extra_optimizer_applications': 0, 'selection_uses_VAL': False,
        'selection_uses_TEST': False, 'best_seed': False, 'produced_in_M6D4c': False,
    },
    'data_policy': {'TEST_allowed': False, 'benchmark_data_access_in_M6D4c': False,
                    'experiment_seeds_unchanged': [42, 1337, 2026]},
    'dispositions': DISPOSITIONS,
    'disposition_scope': 'NORMATIVE_CONTRACT_NOT_NUMERICAL_OR_RUNTIME_QUALIFICATION',
    'final_status': FINAL_STATUS,
    'milestone_execution': EXECUTION,
}


def require(ok, reason):
    if not ok:
        raise ValueError('M6D4c: '+reason)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args])


def read(path):
    require(path in set(ARTIFACTS) | set(INPUT_HASHES) | set(HISTORICAL_HASHES) |
            {LEDGER, INDEX}, 'path not in fixed evidence allowlist')
    target = ROOT / path
    require(target.resolve() == target, 'symlink in evidence path')
    return target.read_bytes()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'duplicate key: '+key)
        result[key] = value
    return result


def parse(raw):
    return json.loads(raw, object_pairs_hook=unique_object)


def exact(actual, expected, path='contract'):
    require(type(actual) is type(expected), path+': type mismatch')
    if isinstance(expected, dict):
        require(actual.keys() == expected.keys(), path+': missing/extra fields')
        for key in expected:
            exact(actual[key], expected[key], path+'.'+key)
    elif isinstance(expected, list):
        require(len(actual) == len(expected), path+': list length')
        for i, value in enumerate(expected):
            exact(actual[i], value, path+'['+str(i)+']')
    else:
        require(actual == expected, path+': unauthorized value')


def validate_contract(contract):
    exact(contract, EXPECTED)


def invalid_contract_self_tests():
    """Reject every changed leaf, missing/extra field, list edit and wrong type.

    Includes changed hashes, source/target estimator, formulas, weights, detach,
    recomputation, schedule/counter, TEST and provenance. All mutants stay in RAM.
    """
    rejected = []

    def reject(label, operation):
        try:
            operation()
        except ValueError:
            rejected.append(label)
        else:
            raise ValueError('invalid-contract mutant accepted: '+label)

    def mutate(path, replacement, label):
        candidate = copy.deepcopy(EXPECTED)
        parent = candidate
        for key in path[:-1]:
            parent = parent[key]
        parent[path[-1]] = replacement
        reject(label, lambda: validate_contract(candidate))

    def walk(value, path):
        label = '.'.join(map(str, path))
        if isinstance(value, dict):
            for key, child in value.items():
                removed = dict(value)
                del removed[key]
                if path:
                    mutate(path, removed, label+':missing:'+key)
                else:
                    reject('missing:'+key, lambda r=removed: validate_contract(r))
                walk(child, path+[key])
            extra = dict(value, unauthorized_extra=True)
            if path:
                mutate(path, extra, label+':extra')
            else:
                reject('extra_root', lambda: validate_contract(extra))
        elif isinstance(value, list):
            mutate(path, value+[None], label+':extra_item')
            for i, child in enumerate(value):
                walk(child, path+[i])
        else:
            replacement = (not value if isinstance(value, bool) else
                           value+1 if isinstance(value, (int, float)) else str(value)+'_INVALID')
            mutate(path, replacement, label+':changed')
            mutate(path, None, label+':wrong_type')

    validate_contract(copy.deepcopy(EXPECTED))
    walk(EXPECTED, [])
    reject('duplicate_json_key', lambda: parse('{"R1":false,"R1":true}'))
    reject('boolean_as_unit_weight', lambda: validate_contract({
        **EXPECTED, 'generator': {**EXPECTED['generator'], 'weights': {
            **EXPECTED['generator']['weights'], 'L_rec': True}}}))
    return rejected


def verify_contract():
    for ref in ('HEAD', 'origin/m6-baselines'):
        require(git('rev-parse', ref).decode().strip() == AUTHORITY, ref+' authority')
    require(git('branch', '--show-current').strip() == b'm6-baselines', 'branch')
    require(git('rev-list', '--left-right', '--count', 'HEAD...origin/m6-baselines').strip()
            == b'0\t0', 'divergence')
    changed = set(git('diff', '--name-only', AUTHORITY, '--').decode().splitlines())
    require(changed <= {LEDGER, INDEX}, 'historical/config/implementation tracked mutation')
    for path, digest in {**INPUT_HASHES, **HISTORICAL_HASHES}.items():
        raw = read(path)
        require(sha(raw) == digest and raw == git('show', AUTHORITY+':'+path),
                'immutable input: '+path)
    validate_contract(parse(read(OVERLAY)))
    rejected = invalid_contract_self_tests()
    require(not {'torch', 'tensorflow'} & set(sys.modules), 'ML framework imported')
    return {'status': 'PASS', 'scope': 'STATIC_CONTRACT_ONLY',
            'invalid_contract_rejections': len(rejected),
            'invalid_contract_rejection_cases': rejected,
            'immutable_input_hashes_verified': len(INPUT_HASHES),
            'additional_historical_hashes_verified': len(HISTORICAL_HASHES),
            'training_runner_qualified': False, 'optimizer_applications': 0}


def expected_index():
    reader = csv.DictReader(io.StringIO(git('show', AUTHORITY+':'+INDEX).decode()))
    baseline = list(reader)
    rows = {row['path']: row for row in baseline}
    require(reader.fieldnames == ['path', 'size_bytes', 'sha256'] and
            len(rows) == len(baseline) == 550, 'baseline index')
    # Ledger and index are intentionally absent from the historical index.
    # Carry historical metadata without dereferencing any historical target.
    for path in ARTIFACTS:
        require(path not in rows, 'new artifact must be additive')
        raw = read(path)
        rows[path] = {'path': path, 'size_bytes': str(len(raw)), 'sha256': sha(raw)}
    output = io.StringIO(newline='')
    writer = csv.DictWriter(output, fieldnames=reader.fieldnames, lineterminator='\r\n')
    writer.writeheader()
    writer.writerows(rows[path] for path in sorted(rows))
    return output.getvalue().encode()


def verify(check_index=True):
    result = verify_contract()
    audit = parse(read(BASE+'.json'))
    exact(audit['validation'], result, 'audit.validation')
    exact(audit['final_status'], FINAL_STATUS, 'audit.final_status')
    exact(audit['execution'], EXECUTION, 'audit.execution')
    exact(audit['dispositions'], DISPOSITIONS, 'audit.dispositions')
    exact(audit['immutable_inputs_sha256'], INPUT_HASHES, 'audit.immutable_inputs')
    require(audit['historical_M6D4b_status'] == 'STOP_AND_REPORT_UNCHANGED', 'historical STOP')
    require(set(audit['artifacts_sha256']) == {OVERLAY, ADDENDUM, SCRIPT}, 'audit artifacts')
    for path, digest in audit['artifacts_sha256'].items():
        require(sha(read(path)) == digest, 'audit artifact hash: '+path)
    prefix = git('show', AUTHORITY+':'+LEDGER)
    require(sha(prefix) == PREFIX_SHA and len(prefix.splitlines()) == 107, 'ledger authority')
    current = read(LEDGER)
    require(current.startswith(prefix) and len(current.splitlines()) == 108, 'one ledger append')
    row = parse(current[len(prefix):])
    require(row['classification'] == 'M6D4C_E05_TRAINING_RUNNER_RESOLUTION', 'ledger classification')
    exact(row['final_status'], FINAL_STATUS, 'ledger.final_status')
    exact(row['execution'], EXECUTION, 'ledger.execution')
    require(set(row['artifacts_sha256']) == set(ARTIFACTS), 'ledger artifact coverage')
    for path, digest in row['artifacts_sha256'].items():
        require(sha(read(path)) == digest, 'ledger artifact hash: '+path)
    if check_index:
        require(read(INDEX) == expected_index(), 'exact CRLF artifact index')
    return {**result, 'ledger_rows': 108, 'first_107_rows_byte_identical': True,
            'artifact_index_rows': 555, 'artifact_index_line_endings': 'CRLF'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--contract-only', action='store_true')
    modes.add_argument('--rebuild-index', action='store_true')
    args = parser.parse_args()
    if args.contract_only:
        result = verify_contract()
    else:
        result = verify(check_index=not args.rebuild_index)
        if args.rebuild_index:
            require((ROOT/INDEX).resolve() == ROOT/INDEX, 'index symlink')
            (ROOT/INDEX).write_bytes(expected_index())
            result = verify()
    # Detailed rejection identities are retained in the audit JSON.
    print(json.dumps({k: v for k, v in result.items()
                      if k != 'invalid_contract_rejection_cases'}, indent=2))


if __name__ == '__main__':
    main()
