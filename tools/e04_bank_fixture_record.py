#!/usr/bin/env python3
"""E04 bank-adapter synthetic-fixture qualification record (laptop; deterministic; reads committed evidence only).

  python3 tools/e04_bank_fixture_record.py build

Inputs (committed under outputs/audit/E04_BANK_FIXTURE_QUALIFICATION/):
  host_evidence.json / host_stdout.txt / host_stderr.txt   tools/e04_bank_fixture_qualification.py on the GPU host CPU
                                                            (tools/e04_bank_scripts/e04_fixture_qualification.sh)
  laptop_tests.txt                                          python3 -m unittest -v tests.test_e04_bank_adapter ...
Writes outputs/audit/E04_BANK_FIXTURE_QUALIFICATION.{json,md} and E04_BANK_FIXTURE_QUALIFICATION_SHA256.txt.
CPU fixture qualification and real-device (GPU) qualification are reported separately; the latter is S7 and pending.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIR = 'outputs/audit/E04_BANK_FIXTURE_QUALIFICATION'
OUT = 'outputs/audit/E04_BANK_FIXTURE_QUALIFICATION'
QUAL_COMMIT = '92ef74c99765109158432fb713a68b3ad238f431'
CONTRACT = 'configs/amendments/e04_bank_generation_contract.yaml'
HOST_COMMAND = ('bash ~/workdir/m8_learned_banks/e04_bank_prep/e04_fixture_qualification_%s.sh %s '
                '(copy of tools/e04_bank_scripts/e04_fixture_qualification.sh; setsid nohup, CUDA hidden)'
                % (QUAL_COMMIT, QUAL_COMMIT))
LAPTOP_COMMAND = 'python3 -m unittest -v tests.test_e04_bank_adapter tests.test_e04_bank_readiness'
CATEGORIES = {'A': 'governance and frozen sources', 'C': 'graph integrity and restore', 'D': 'bank infrastructure',
              'E': 'synthesis integrity (numpy references, roles, offsets)', 'F': 'frozen-graph comparison',
              'G': 'BN inference-mode isolation', 'H': 'conversion, non-finite accounting, firewall',
              'I': 'real E04 checkpoints (hash, index, moving statistics; no forward)'}
STATUS_OK = 'E04_BANK_ADAPTER_SYNTHETIC_QUALIFIED_S7_PENDING'


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dumps(o):
    return json.dumps(o, sort_keys=True, indent=1, ensure_ascii=True) + '\n'


def build(root=ROOT, out_root=None):
    root = Path(root)
    out = Path(out_root or root)
    ev = json.loads((root / DIR / 'host_evidence.json').read_text())
    laptop = (root / DIR / 'laptop_tests.txt').read_text()
    m = re.search(r'^Ran (\d+) tests? in [0-9.]+s\n\n(OK|FAILED.*)$', laptop, re.M)
    laptop_ran, laptop_ok = (int(m.group(1)), m.group(2) == 'OK') if m else (0, False)
    gates = []

    def gate(gid, ok, detail):
        gates.append({'id': gid, 'pass': bool(ok), 'detail': detail})
    gate('host_run_at_qualification_commit', ev['git_head'] == QUAL_COMMIT and ev['error'] is None,
         ev['git_head'])
    gate('host_cpu_only_locked_interpreter', ev['environment']['cuda_visible_devices'] == '' and
         ev['environment']['python_executable'].endswith('envs/gpat-m6-e03/bin/python'), ev['environment'])
    gate('host_all_gates_passed', ev['failed_gates'] == [] and ev['gates_passed'] == ev['gates_total'] == 45,
         [ev['gates_passed'], ev['gates_total']])
    gate('host_stdout_matches', json.loads((root / DIR / 'host_stdout.txt').read_text()) ==
         {'gates_passed': ev['gates_passed'], 'gates_total': ev['gates_total'], 'failed': ev['failed_gates']}, None)
    gate('no_gpu_claim', ev['gpu_bitwise_reproducibility_proven'] is False and
         ev['gpu_numerical_equality_proven'] is False, None)
    real = ev['evidence']['real_checkpoints']
    contract = json.loads((root / CONTRACT).read_text())
    gate('real_checkpoints_equal_contract', all(real[s]['files'] == contract['checkpoints']['per_seed'][s]['sha256']
                                                and real[s]['unchanged_after'] for s in ('42', '1337', '2026')), None)
    gate('laptop_tests_passed', laptop_ok and laptop_ran >= 30, laptop_ran)
    failed = [g['id'] for g in gates if not g['pass']]
    by_cat = {}
    for gid, v in sorted(ev['gates'].items()):
        c = gid.split('.')[0] if '.' in gid else 'environment'
        by_cat.setdefault(c, {'passed': 0, 'total': 0})
        by_cat[c]['total'] += 1
        by_cat[c]['passed'] += int(v['pass'])
    bn = ev['evidence']['bn_mode_isolation']
    num = ev['gates']['E.numpy_reference_decomposition_lowpass_warp_synthesis']['detail']
    doc = {
        'record': 'E04_BANK_FIXTURE_QUALIFICATION', 'schema': 'e04_bank_fixture_qualification_record_v1',
        'status': STATUS_OK if not failed else 'E04_BANK_ADAPTER_SYNTHETIC_QUALIFICATION_FAILED',
        'failed_gates': failed, 'gates': gates, 'qualification_commit': QUAL_COMMIT,
        'contract': {'path': CONTRACT, 'sha256': sha_file(root / CONTRACT)},
        'cpu_fixture_qualification': {
            'status': 'PASSED' if not failed else 'FAILED', 'host_gates_passed': ev['gates_passed'],
            'host_gates_total': ev['gates_total'], 'host_failed_gates': ev['failed_gates'],
            'host_gate_categories': {k: dict(v, meaning=CATEGORIES.get(k, 'environment / data firewall'))
                                     for k, v in sorted(by_cat.items())},
            'host_command': HOST_COMMAND, 'host_wall_seconds': ev['wall_seconds'], 'host_started_utc': ev['started_utc'],
            'host_environment': ev['environment'], 'tensorflow': ev['evidence']['tensorflow'],
            'laptop_command': LAPTOP_COMMAND, 'laptop_tests_ran': laptop_ran, 'laptop_tests_ok': laptop_ok,
            'evidence_files': {n: sha_file(root / DIR / n) for n in ('host_evidence.json', 'host_stdout.txt',
                                                                     'host_stderr.txt', 'laptop_tests.txt')},
            'fixture_inputs': 'random-init fixture weights saved from the frozen TrainingGraph (seed 60912, '
                              'randomized generator moving statistics), synthetic noise/gradient PNGs, synthetic Q140; '
                              'zero opens under <runtime_root>/data'},
        'key_results': {
            'frozen_graph_comparison_max_abs': ev['gates']['F.composition_equals_frozen_synthesized_spoof_bitwise'][
                'detail'],
            'bn_mode_isolation': bn,
            'numpy_reference_max_abs': num,
            'conversion_cases': ev['gates']['H.clip_truncate_cases']['detail'],
            'graph': ev['evidence']['graph'],
            'fixture_banks_manifest_sha256': ev['evidence']['banks'],
            'fixture_synthesis_range': ev['evidence']['fixture_synthesis_range']},
        'real_checkpoints': real,
        'real_device_qualification': {
            'status': 'PENDING_E04_S7',
            'not_shown_by_cpu': ['GPU kernel outputs of the real checkpoints', 'GPU bitwise A/B replay (E04-Q5)',
                                 'real TRAIN faces, feed pack and Q140 geometry through load_e04_inputs',
                                 'environment-lock verification (production_runner.verify_environment) on the GPU',
                                 'production throughput and memory']},
        'not_done': ['E04 S7 GPU inference', 'E04 production banks', '<bank_root>/E04', 'downstream evaluation',
                     'TEST access', 'DINOv3', 'GPAT']}
    (out / OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(str(out / OUT) + '.json').write_text(dumps(doc))
    Path(str(out / OUT) + '.md').write_text(report(doc))
    Path(str(out / OUT) + '_SHA256.txt').write_text(''.join('%s  %s\n' % (sha_file(str(out / OUT) + e),
                                                                           Path(OUT + e).name) for e in ('.json', '.md')))
    return doc


def report(doc):
    c = doc['cpu_fixture_qualification']
    k = doc['key_results']
    L = ['# E04 Physics-STD bank adapter: synthetic-fixture qualification', '',
         '**Status:** `%s` (%d/%d record gates).' % (doc['status'], sum(g['pass'] for g in doc['gates']),
                                                     len(doc['gates'])), '',
         'Adapter commit `%s`; contract `%s` (`%s`).' % (doc['qualification_commit'][:12], doc['contract']['path'],
                                                         doc['contract']['sha256'][:16]), '',
         '## CPU fixture qualification: %s' % c['status'], '',
         '- Host CPU (gpat-m6-e03, TF %s, CUDA hidden): **%d/%d** gates, %.1f s, started %s.' % (
             c['tensorflow'], c['host_gates_passed'], c['host_gates_total'], c['host_wall_seconds'],
             c['host_started_utc']),
         '- Laptop unit tests: %d ran, OK %s (`%s`).' % (c['laptop_tests_ran'], c['laptop_tests_ok'],
                                                         c['laptop_command']), '',
         '| category | passed | meaning |', '|---|---|---|']
    L += ['| %s | %d/%d | %s |' % (n, v['passed'], v['total'], v['meaning']) for n, v in
          sorted(c['host_gate_categories'].items())]
    bn = k['bn_mode_isolation']
    L += ['', '## Key results', '',
          '- Frozen-graph comparison: the adapter composition fed the same 8-image batch reproduces '
          '`TrainingGraph.synthesized_spoof` and every warped component **bitwise** (max abs %s).' %
          k['frozen_graph_comparison_max_abs'],
          '- BN mode: with moving statistics set to the image\'s own batch statistics, inference mode equals the '
          'training-mode forward to max abs %.2e (synthesis); with the approved stored statistics the difference is '
          '%.3g. The statistics source is the only difference.' % (bn['equal_statistics_max_abs']['synth'],
                                                                    bn['approved_moving_statistics_max_abs']['synth']),
          '- Independent numpy references (heads, sigmoid P, low32/low128, five warps, synthesis equation): max abs '
          '%.2e.' % max(k['numpy_reference_max_abs'].values()),
          '- Conversion `(np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)`: inputs %s -> %s.' % (
              k['conversion_cases']['inputs'], k['conversion_cases']['got']),
          '- Inference graph: %d variables (%d trainable, %d moving), %d FusedBatchNorm all `is_training=False`, '
          '0 update/gradient/optimizer/random ops, batch 1.' % (
              k['graph']['global_variables'], k['graph']['trainable_variables'], k['graph']['bn_moving_statistics'],
              k['graph']['fused_batch_norm']), '',
          '## Real E04 checkpoints (hash, index, moving statistics; no forward)', '',
          '| seed | run_id | identity | restored / in checkpoint | moving statistics finite | unchanged |',
          '|---|---|---|---|---|---|']
    for s, v in sorted(doc['real_checkpoints'].items(), key=lambda x: int(x[0])):
        L.append('| %s | `%s` | `%s` | %d / %d | %s | %s |' % (
            s, v['run_id'], v['checkpoint_identity_sha256'][:16], v['coverage']['restored_variables'],
            v['coverage']['checkpoint_variables'], v['moving_statistics']['ok'], v['unchanged_after']))
    L += ['', '## Real-device qualification: %s' % doc['real_device_qualification']['status'], '']
    L += ['- not shown by CPU: ' + x for x in doc['real_device_qualification']['not_shown_by_cpu']]
    L += ['', '## Record gates', '', '| gate | pass |', '|---|---|']
    L += ['| `%s` | %s |' % (g['id'], 'PASS' if g['pass'] else '**FAIL**') for g in doc['gates']]
    L += ['', 'Not done: ' + ', '.join(doc['not_done']) + '.', '']
    return '\n'.join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['build'])
    ap.parse_args()
    d = build()
    print(d['status'], d['failed_gates'])


if __name__ == '__main__':
    main()
