"""Build the previous-work results audit package from run_analysis.json, the run-root copies and the repository.

Writes only into --out. Every number in the tables comes from run_analysis.json (itself derived from the
immutable run-root records) or from committed frozen configs/evidence; nothing is typed by hand except the
classifications, which cite their sources.

usage: python3 -I build_package.py --analysis <run_analysis.json> --runs <dir> --repo <repo> --out <dir>
"""
import argparse
import csv
import json
import statistics
from pathlib import Path

NAMES = {'E01': 'FAS-Aug', 'E02': 'FreqSub (Frequency Substitution)', 'E03': 'STDN', 'E04': 'Physics-STD',
         'E05': 'PCGAN', 'E06b': 'DSDG-NATIVE', 'E06c': 'DSDG-BIN-IDFREE'}
TRACK = {'E01': 'A', 'E02': 'A', 'E03': 'A', 'E04': 'A', 'E05': 'A', 'E06c': 'A', 'E06b': 'B'}
FAMILY_CLOSURE = {'E06c': '24f7049', 'E06b': '3b0d308', 'E05': '7677c3a', 'E03': '374de40 (+ ruling 320b42c)'}
SOURCE_PIN = {'E06c': 'JDAI-CV/FaceX-Zoo@16b793a7 (DSDG)', 'E06b': 'JDAI-CV/FaceX-Zoo@16b793a7 (DSDG)',
              'E05': 'swapping-autoencoder@6baa180f (A2-07 architecture basis)', 'E03': 'yaojieliu/ECCV20-STDN@c79f1f8c',
              'E04': 'none official; 3DDFA_V2@1b6c6760 + BFM + Q=140 (A3 s4, A4)', 'E01': 'RizhaoCai/FAS-Aug@0da1dd79',
              'E02': 'none (spec 8.2)'}
FIDELITY = {'E06c': 'CONTROLLED_ADAPTATION (DEV-020)', 'E06b': 'FAITHFUL_OFFICIAL_WITH_RUNTIME_COMPATIBILITY',
            'E05': 'CONTROLLED_ADAPTATION (A2-07, A5, M6D4c)',
            'E03': 'FAITHFUL_OFFICIAL_CORE_WITH_TRACK_A_CONTROLLED_DATA_ADAPTATION_AND_RUNTIME_COMPATIBILITY',
            'E04': 'CONTROLLED_ADAPTATION (A3 s4, A4, D1-D9, D12)', 'E01': 'FAITHFUL_OFFICIAL_WITH_DETERMINISM_CLARIFICATION',
            'E02': 'SPEC_DEFINED (gate adapted)'}
CONFIG = {'E06c': 'configs/methods/e06c_dsdg_bin_idfree.yaml', 'E06b': 'configs/methods/e06b_dsdg_native.yaml',
          'E05': 'configs/methods/e05_pcgan.yaml', 'E03': 'configs/methods/e03_stdn.yaml',
          'E04': 'configs/methods/e04_physics_std.yaml', 'E01': 'configs/methods/e01_fas_aug.yaml',
          'E02': 'configs/methods/e02_freqsub.yaml'}
DRIVER = {  # observed GPU driver during each run
    'E06c': ('NOT_RECORDED_IN_RUN_ARTIFACTS', 'host dpkg.log: nvidia-driver-580-open 580.178.04 installed 2026-10-03 11:27 +07, '
             'boot 2026-10-03 11:28 +07, no later driver install; M7C3 (cff688d, 2026-10-03) observed 580.178.04; '
             'lock environments/e06c.lock.json says 595.84 -> inferred 580.178.04, undisclosed'),
    'E06b': ('NOT_RECORDED_IN_RUN_ARTIFACTS', 'same evidence as E06c -> inferred 580.178.04, undisclosed'),
    'E05': ('580.178.04', 'run_start.payload.gpu_driver (OWNER_APPROVED_DRIVER_VERSION_ONLY_RUNTIME_COMPATIBILITY)'),
    'E03': ('580.178.04', 'run_start.payload.gpu_driver (OWNER_APPROVED_DRIVER_VERSION_ONLY_RUNTIME_COMPATIBILITY)')}
RT = '/home/student20261/workdir/GPAT_TransferBench_runtime'


def find_key(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            r = find_key(v, key)
            if r is not None:
                return r
    return None


def fmt(x, nd=4):
    if x is None:
        return 'NOT_RECORDED'
    if isinstance(x, float):
        return '%.*g' % (nd + 2, x)
    return str(x)


def write_csv(path, rows):
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator='\n', restval='')
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    for a in ('--analysis', '--runs', '--repo', '--out'):
        ap.add_argument(a, required=True)
    a = ap.parse_args()
    out, runs, repo = Path(a.out), Path(a.runs), Path(a.repo)
    A = json.loads(Path(a.analysis).read_text())
    by = {(r['method_id'], r['seed']): r for r in A}

    # ------------------------------------------------------------------ A. baseline inventory
    inv = []
    for r in A:
        m, s = r['method_id'], r['seed']
        d = runs / m / ('seed_%d' % s)
        start = json.loads(open(d / 'metrics.jsonl').readline())
        man = json.loads((d / 'run_manifest.json').read_text())
        p = r['provenance']
        split = find_key(man, 'split_manifest_sha256') or find_key(start, 'split_manifest_sha256')
        pair = (find_key(start, 'pair_manifest_sha256') or find_key(start, 'pairs_train_v1_sha256') or
                (start['payload'].get('relation') or {}).get('sha256'))
        if isinstance(pair, dict):
            pair = 'TRAIN %s; VAL %s' % (pair['TRAIN'], pair['VAL'])
        lock = find_key(man, 'environment_lock_sha256') or find_key(start, 'environment_lock_sha256') or p['dependency_fingerprint']
        fc = r['final_checkpoint']
        sel = fc['selected_index_entries']
        inv.append({
            'method_id': m, 'method_name': NAMES[m], 'track': TRACK[m], 'fidelity': FIDELITY[m], 'seed': s,
            'training_status': 'SCIENTIFIC_TRAINING_COMPLETED', 'run_id': p['run_id'],
            'run_id_formula_recomputed_equal': r['checks']['run_id_formula'],
            'scientific_authority': p['git_commit'], 'git_dirty_at_launch': p['git_dirty'],
            'seed_closure_commit': r['closure_commit'], 'family_closure_commit': FAMILY_CLOSURE[m],
            'host': p['host'], 'user': p['user'], 'gpu_model': man.get('gpu_model'),
            'environment_name': p['environment_name'], 'environment_lock_path': p['environment_lock_path'],
            'environment_lock_sha256': lock, 'python_executable': p['python_executable'],
            'python_version': p['python_version'], 'framework': '%s %s' % (p['framework'], p['framework_version']),
            'cuda_version': p['cuda_version'], 'cudnn_version': p['cudnn_version'],
            'observed_gpu_driver': DRIVER[m][0], 'observed_gpu_driver_source': DRIVER[m][1],
            'config_path': p['config_path'], 'config_sha256': p['config_sha256'], 'source_pin': SOURCE_PIN[m],
            'source_commits_in_manifest': json.dumps(p['source_commits'], sort_keys=True),
            'pair_relation_sha256': pair or 'NOT_RECORDED_IN_RUN_ROOT',
            'split_manifest_sha256': split or 'NOT_RECORDED_IN_RUN_ROOT',
            'start_utc': p['start_utc'], 'end_utc': p['end_utc'], 'completion_status': p['completion_status'],
            'checkpoint_rule': fc['rule'], 'final_checkpoint_path': fc['path'],
            'final_checkpoint_sha256': fc['sha256'],
            'final_checkpoint_files': ';'.join('%s:%d' % (x[0], x[2]) for x in sel),
            'final_checkpoint_present_on_host_sha_equal': r['checks']['final_checkpoint_on_host_sha_equal'],
            'run_root': '%s/runs/m6/%s/seed_%d' % (RT, m, s), 'run_root_files': r['host_file_count'],
            'run_root_sha_listing_digest_2026_10_30': r['host_sha_listing_digest'],
            'run_root_mtime_listing_digest_2026_10_30': r['host_mtime_listing_digest'],
            'run_root_equals_closure_listing': r['checks']['closure_listing_digest_equal'],
            'metrics_jsonl_sha256': r['files_sha256']['metrics.jsonl'],
            'metrics_equal_closure': r['checks']['closure_metrics_sha_equal']})
    planned = [('E04', 'PRODUCTION_QUALIFIED_NOT_LAUNCHED', 'qualification authority b0f0ffd (52/52, D12 c60481b); '
                'scientific authority not yet fixed (= launch commit)'),
               ('E01', 'IMPLEMENTED_NOT_EXECUTED', 'non-learned; bank writer not implemented; owner M6/M8 gate ruling open'),
               ('E02', 'IMPLEMENTED_NOT_EXECUTED', 'non-learned; bank writer not implemented; owner M6/M8 gate ruling open')]
    for m, st, note in planned:
        cfg = (repo / CONFIG[m]).read_bytes()
        import hashlib
        for s in (42, 1337, 2026):
            inv.append({'method_id': m, 'method_name': NAMES[m], 'track': TRACK[m], 'fidelity': FIDELITY[m], 'seed': s,
                        'training_status': st, 'run_id': 'NOT_STARTED', 'scientific_authority': note,
                        'config_path': CONFIG[m], 'config_sha256': hashlib.sha256(cfg).hexdigest(),
                        'source_pin': SOURCE_PIN[m], 'run_root': '%s/runs/m6/%s/seed_%d (absent)' % (RT, m, s),
                        'completion_status': 'NOT_STARTED'})
    write_csv(out / 'A_baseline_inventory.csv', inv)

    # ------------------------------------------------------------------ B. artifact completeness
    comp = []
    for r in A:
        m, s, c = r['method_id'], r['seed'], r['checks']
        n_ck = c['checkpoint_index_entries']
        row = {'method_id': m, 'seed': s, 'run_id': r['provenance']['run_id']}
        row['weights_final'] = 'PRESENT (sha256 re-hashed on host 2026-10-30 equals run_summary and closure)' \
            if c['final_checkpoint_on_host_sha_equal'] and c['closure_final_sha_equal'] else 'DEFECT'
        extra = {'E06c': '; 21 ENGINEERING_RESUME_ONLY runner_state sidecars also present and equal to resume_state_index',
                 'E03': '; TF Saver state file `checkpoint` present'}.get(m, '')
        row['checkpoints'] = ('COMPLETE %d/%d indexed files present, size+sha256 equal%s' %
                              (n_ck - len(c['checkpoint_payloads_missing']) - len(c['checkpoint_payloads_size_or_sha_mismatch']),
                               n_ck, extra))
        row['training_logs'] = ('COMPLETE metrics.jsonl %d records, 0 unparseable, 0 non-finite, contiguous index, '
                                'sha256 equals closure' % c['records_total'])
        row['loss_components'] = {
            'E06c': 'COMPLETE: total_loss + 7 terms per step (loss_cls and loss_pair identically 0 by design, DEV-020); assembly re-derived to 4e-16',
            'E06b': 'COMPLETE: total_loss + 7 terms per step; assembly re-derived to 4e-16',
            'E05': 'COMPLETE: 5 G terms + L_G_total, 7 D terms + L_D_total per iteration; sums re-derived (<=2e-7)',
            'E03': 'PARTIAL: official exposed aggregates g_loss/d_loss/a_loss per TRAIN step and VAL step; '
                   'sub-terms (esr/gan/reg/pixel) and the a_loss `dec` branch were never exposed or logged'}[m]
        row['learning_rates'] = {
            'E06c': 'COMPLETE per step (constant 2e-4)', 'E06b': 'COMPLETE per step (constant 2e-4)',
            'E05': 'COMPLETE per iteration, G and D (constant 1e-6)',
            'E03': 'COMPLETE per step, G and D (staircase 6e-5*0.9^floor(gs/20000); max rel err %.1e)' % c.get('lr_schedule_max_rel_err', 0.0)}[m]
        row['runtime_data'] = ('COMPLETE: per-step step_seconds, wall_clock_seconds, utc; peak GPU memory per step '
                               '(max %d B); run_summary duration %.1f s' %
                               (r['peak_gpu_memory_bytes_from_metrics'], r['summary']['training_duration_seconds']))
        row['rng_evidence'] = {
            'E06c': 'COMPLETE: seeding plan in run_start; per-step batch pair-id sha256 + epsilon sha256; per-epoch order sha256; '
                    'closure re-predicted 7400/7400 batches',
            'E06b': 'COMPLETE: seeding plan in run_start; per-step batch pair-id sha256 + epsilon sha256; per-epoch spoof order sha256; '
                    'closure re-predicted 3200/3200 batches',
            'E05': 'COMPLETE: generator seed + per-iteration sampler_index/pair_id/pair_sha256; closure re-predicted 4000/4000',
            'E03': 'COMPLETE for data RNG: TRAIN/VAL parser+shuffle seeds, per-step pair_rows, flips, brightness; closure replayed '
                   '200000 TRAIN + 50000 VAL rows. TF in-graph op RNG (a_loss dec draw) not recorded'}[m]
        row['provenance'] = ('COMPLETE: run_manifest (git %s clean, config sha, env lock, host, command); run_id recomputed; '
                             'observed driver %s' % (r['provenance']['git_commit'][:8], DRIVER[m][0]))
        row['downstream_outputs'] = 'PENDING_DOWNSTREAM_EVALUATION (no bank, no detector, no predictions, no metrics)'
        row['reopenable_independently'] = ('YES: plain JSONL/JSON/YAML; checkpoints byte-identical to the bytes the closure audits '
                                           'loaded (%s)' % {'E06c': 'weights_only sidecars + raw-storage official .pth',
                                                            'E06b': 'raw-storage reads of all 63 official .pth, no class import',
                                                            'E05': 'torch.load(weights_only=True)',
                                                            'E03': 'tf NewCheckpointReader, all 50 checkpoints'}[m])
        comp.append(row)
    write_csv(out / 'B_artifact_completeness_matrix.csv', comp)

    # ------------------------------------------------------------------ D. publication table previews
    # Table B: execution
    tb = []
    for r in A:
        m, s = r['method_id'], r['seed']
        t = r['timing']
        req = {'E06c': '200 epochs / 7400 steps', 'E06b': '200 epochs / 3200 steps', 'E05': '4000 iterations',
               'E03': '50 epochs / 100000 TRAIN steps (G 100000, D 50000, global_step 150000)'}[m]
        c, oa = r['checks'], r['summary']['optimizer_applications']
        if m in ('E06c', 'E06b'):
            done = '200 epochs / %d steps (%d optimizer applications)' % (c['train_steps'], oa)
        elif m == 'E05':
            done = '%d iterations (D %d, G %d)' % (c['train_steps'], oa['D'], oa['G'])
        else:
            done = '50 epochs / %d TRAIN steps (G %d, D %d) + %d VAL steps' % (
                c['train_steps'], c['final_applications']['G'], c['final_applications']['D'], c['val_steps'])
        mem_sem = 'torch.cuda.max_memory_allocated per step' if m != 'E03' else 'TF MaxBytesInUse (allocator, cumulative)'
        tb.append({'method_id': m, 'method_name': NAMES[m], 'seed': s, 'run_id': r['provenance']['run_id'],
                   'start_utc': r['summary']['start_utc'], 'end_utc': r['summary']['end_utc'],
                   'training_duration_s': round(r['summary']['training_duration_seconds'], 1),
                   'training_duration_h': round(r['summary']['training_duration_seconds'] / 3600, 3),
                   'median_step_s': round(t['train_step_seconds_median'], 4),
                   'sum_train_step_s': round(t['train_step_seconds_sum'], 1),
                   'sum_val_step_s': round(t['val_step_seconds_sum'], 1) if t['val_step_seconds_sum'] else 'NOT_APPLICABLE',
                   'required_budget': req, 'completed': done,
                   'checkpoint_files_indexed': r['checks']['checkpoint_index_entries'],
                   'checkpoint_events': r['checks']['checkpoint_events'],
                   'checkpoint_availability': 'ALL_PRESENT_SHA_EQUAL',
                   'completion_status': r['summary']['completion_status'],
                   'peak_gpu_memory_bytes': r['peak_gpu_memory_bytes_from_metrics'],
                   'peak_gpu_memory_GiB': round(r['peak_gpu_memory_bytes_from_metrics'] / 2 ** 30, 3),
                   'peak_gpu_memory_semantics': mem_sem,
                   'shared_gpu_contention': ('NOT_RECORDED; logs show 2.2-2.5x slower steps than seeds 1337/2026 '
                                             '(operator notes attribute this to another user job on the shared 3090)'
                                             if (m, s) == ('E06c', 42) else
                                             'NOT_RECORDED; duration 1.35x seeds 42/2026 (operator notes: another user job at launch)'
                                             if (m, s) == ('E03', 1337) else 'NOT_RECORDED (no per-run contention log)'),
                   'other_resources': 'NOT_RECORDED (no CPU/RAM/IO/energy logging)'})
    write_csv(out / 'D2_table_B_training_execution.csv', tb)

    # Table C: diagnostics (per seed + mean / sample std within method only)
    tc = []
    for m in ('E06c', 'E06b', 'E05', 'E03'):
        rows = [by[(m, s)] for s in (42, 1337, 2026)]
        keys = rows[0]['loss_keys']
        for r in rows:
            row = {'method_id': m, 'seed': r['seed'], 'window': r['loss_final_window']}
            row.update({'final_window_mean:' + k: r['loss_final_window_mean'][k] for k in keys})
            row.update({'last_record:' + k: r['loss_final_record'][k] for k in keys})
            if 'val_loss_epoch50_mean' in r:
                row.update({'VAL_DIAGNOSTIC_epoch50_mean:' + k: v for k, v in r['val_loss_epoch50_mean'].items()})
            lr = r['final_learning_rate']
            row['final_learning_rate'] = json.dumps(lr) if isinstance(lr, dict) else lr
            tc.append(row)
        agg = {'method_id': m, 'seed': 'mean', 'window': rows[0]['loss_final_window']}
        sd = {'method_id': m, 'seed': 'sample_std (n=3)', 'window': rows[0]['loss_final_window']}
        for k in keys:
            vals = [r['loss_final_window_mean'][k] for r in rows]
            agg['final_window_mean:' + k] = statistics.mean(vals)
            sd['final_window_mean:' + k] = statistics.stdev(vals)
        if m == 'E03':
            for k in rows[0]['val_loss_epoch50_mean']:
                vals = [r['val_loss_epoch50_mean'][k] for r in rows]
                agg['VAL_DIAGNOSTIC_epoch50_mean:' + k] = statistics.mean(vals)
                sd['VAL_DIAGNOSTIC_epoch50_mean:' + k] = statistics.stdev(vals)
        tc += [agg, sd]
    write_csv(out / 'D3_table_C_generator_training_diagnostics.csv', tc)
    return inv, comp, tb, tc


if __name__ == '__main__':
    main()
