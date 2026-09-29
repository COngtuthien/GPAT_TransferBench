"""M6F-D E06b DSDG-NATIVE PRODUCTION_PATH_QUALIFICATION_ONLY on REAL CASIA+MSU TRAIN data; never a training run.

Three cases, each one fresh process from a fresh model state (no state carried between cases):

  loader   the production DataLoader (8 workers, seeded generator) over the FULL real native relation with an
           ID-only item (no image): one pass proves 15 x 240 + 120, every spoof row once, 8 workers, worker seeds and
           that every real live-partner draw equals methods.dsdg.native.simulate_live_draws. Then 16 real pairs
           (4 per dataset x class) are decoded through the production reader + preprocessing (no model).
  b240     one independent qualification case: a deterministic real 240-row TRAIN subset (60 per dataset x class)
           through the production E06bDataset -> production loader -> E06bTrainer.global_step_run: 12 x 20, forward,
           backward, exactly one Adam step (epoch-1 warm-up branch).
  tail120  one independent qualification case: a disjoint deterministic real 120-row subset (30 per dataset x class)
           through the SAME production batch function: 6 x 20, one Adam step (post-warm-up branch).

The 240 and 120 cases are independent qualification cases, NOT consecutive scientific steps. No weight is saved
(torch.save is forbidden), no checkpoint, no bank, no scientific run root. TRAIN only: split_v1 is read with a TRAIN +
CASIA/MSU filter on allowlisted columns; faces only for sample IDs of the native relation; no VAL/TEST/SiW.
"""
import argparse
import json
import os
from pathlib import Path
import random
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from methods.common.config import sha256_file  # noqa: E402
from methods.dsdg import microbatch_qualification as m6d5c  # noqa: E402  (unchanged instrumentation)
from methods.dsdg import native  # noqa: E402
from methods.dsdg import native_runner as nr  # noqa: E402
from methods.dsdg import runner as e06c_runner  # noqa: E402
from methods.dsdg import runner_io as rio  # noqa: E402
from methods.dsdg import runner_qualification as m6d5e  # noqa: E402  (unchanged firewall base)
from methods.dsdg import runtime as m6d5a  # noqa: E402
from methods.dsdg import training_graph as tg  # noqa: E402
from methods.dsdg import training_qualification as m6d5b  # noqa: E402

MILESTONE = 'M6F-D'
LABEL = 'PRODUCTION_PATH_QUALIFICATION_ONLY'
SEED = nr.QUALIFICATION_SEED
SHORT_TMP = Path('/tmp/gpat-m6fd')
CASES = {'b240': (0, 60, 1), 'tail120': (60, 90, 2)}      # per (dataset, class) group slice, epoch branch
IMAGE_CHECK_PER_GROUP = 4
CATEGORIES = ('SPLIT_METADATA_TRAIN_FILTERED', 'TRAIN_FACE', 'NON_TRAIN_FACE', 'OTHER_MANIFEST', 'VAL_METADATA',
              'TEST_METADATA', 'FACES_ENUMERATION_OR_WRITE', 'SCIENTIFIC_RUN_ROOT', 'OTHER_BENCHMARK_DATA')
EXIT = {'PASS': 0, 'STOP_GATE': 3, 'STOP_OOM': 4, 'STOP_RESOURCE_CONTAMINATION': 5}
require = rio.require


class Firewall(m6d5e.Firewall):
    """The unchanged M6D5e firewall with ONE allowed manifest: split_v1 (read with a TRAIN + CASIA/MSU filter)."""

    def __init__(self, runtime_root, faces_root, write_roots, sample_datasets, allow_faces, access_fd, split_path):
        super().__init__(runtime_root, faces_root, write_roots, sample_datasets, allow_faces, access_fd)
        self.manifest = str(split_path)

    def category(self, event, path):
        if path == self.manifest:
            return 'SPLIT_METADATA_TRAIN_FILTERED'
        cat = super().category(event, path)
        return 'OTHER_MANIFEST' if cat == 'SPLIT_METADATA_VAL_TEST' else cat


def access_audit(path, allowed_ids):
    rows = [line.split('\t') for line in Path(path).read_text(encoding='utf-8').splitlines()]
    require(all(len(r) == 5 for r in rows), 'access log format')
    counts = dict.fromkeys(CATEGORIES, 0)
    faces, pids, denied = set(), set(), 0
    for pid, event, cat, verdict, p in rows:
        counts[cat] = counts.get(cat, 0) + 1
        denied += verdict == 'DENIED'
        pids.add(pid)
        if cat == 'TRAIN_FACE':
            faces.add(Path(p).stem)
    return {'log': Path(path).name, 'log_sha256': sha256_file(path), 'events_logged': len(rows),
            'processes_logged': len(pids), 'counts_by_category': counts, 'denied_events': denied,
            'train_faces_opened_distinct': len(faces), 'train_faces_all_in_native_relation': faces <= set(allowed_ids),
            'VAL_accesses': counts['VAL_METADATA'], 'TEST_accesses': counts['TEST_METADATA'],
            'non_train_face_accesses': counts['NON_TRAIN_FACE'], 'other_manifest_accesses': counts['OTHER_MANIFEST'],
            'scientific_run_root_accesses': counts['SCIENTIFIC_RUN_ROOT'],
            'note': 'every benchmark-path Python audit event of the main process and all forked DataLoader workers; '
                    'faces are classified against the CASIA/MSU TRAIN native relation only'}


def subset_relation(relation, rows, lo, hi):
    """Deterministic stratified subset: relation order (bytewise sample_id) slice [lo, hi) of every dataset x class."""
    groups = {}
    for sid, key, cls in relation.spoof:
        groups.setdefault((key[0], cls), []).append(sid)
    require(sorted(groups) == [(d, c) for d in native.DATASETS for c in (0, 1)], 'four dataset x class groups')
    chosen = {sid for g in groups.values() for sid in g[lo:hi]}
    require(len(chosen) == 4 * (hi - lo), 'group sizes sufficient')
    subjects = {key for sid, key, _ in relation.spoof if sid in chosen}
    keep = [r for r in rows if r['sample_id'] in chosen or
            (r['label_binary'] == 0 and (r['dataset'], r['subject_id_global']) in subjects)]
    return native.NativeRelation(keep), sorted(chosen, key=native.byte_order)


def setup(case):
    storage = rio.faces_root_from_exec_config(nr.EXEC_CONFIG)
    runtime_root, faces_root = Path(storage['runtime_root']), Path(storage['faces_256_root'])
    build = nr.run_root(runtime_root, nr.QUALIFICATION, SEED, case)
    require(build.is_dir() and not any(build.iterdir()), 'fresh empty qualification root (pre-created by the launcher)')
    require(Path(os.environ.get('TMPDIR', '')).resolve() == SHORT_TMP, 'TMPDIR = short worker directory')
    adapter = native.DSDGNativeAdapter()
    config = nr.verify_contract(adapter.config)
    ids = nr.identities(config)
    source = nr.verify_source(adapter, ids)
    lightcnn = nr.lightcnn_identity(m6d5a.LIGHTCNN)
    split_path = (nr.ROOT / config['data']['split_manifest']).resolve()
    access_fd = os.open(build / 'M6FD_ACCESS_LOG.tsv', os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    # allowed faces: every CASIA/MSU TRAIN sample of the native relation (known only after the TRAIN-filtered read)
    firewall = Firewall(runtime_root, faces_root, [build, SHORT_TMP], {}, True, access_fd, split_path)
    sys.addaudithook(firewall)
    rows, relation, sample_datasets, population = nr.load_relation(config)   # the production relation reader
    firewall.samples = dict(sample_datasets)
    require(all(r['split'] == 'TRAIN' and r['dataset'] in native.DATASETS for r in rows), 'TRAIN CASIA/MSU rows only')
    result = {'milestone': MILESTONE, 'label': LABEL, 'scientific': 'NON_SCIENTIFIC', 'case': case,
              'qualification_seed': SEED, 'method_id': nr.METHOD_ID, 'identities': {
                  k: v for k, v in ids.items() if k != 'source_files_sha256'},
              'source': {'commit': source['commit'], 'tree': source['tree'],
                         'files_verified': len(source['files_sha256'])}, 'lightcnn': lightcnn,
              'storage': {k: storage[k] for k in ('exec_config', 'exec_config_sha256')},
              'population': population, 'native_rows_train_casia_msu_only': True,
              'siw_rows_present': any(r['dataset'] == 'siwmv2' for r in rows)}
    require(not result['siw_rows_present'], 'no SiW row')
    return dict(result=result, build=build, runtime_root=runtime_root, faces_root=faces_root, config=config,
                adapter=adapter, source=source, rows=rows, relation=relation, sample_datasets=sample_datasets,
                firewall=firewall, access_fd=access_fd)


def torch_setup(ctx, *, cuda):
    import torch
    import torch.nn.functional as F
    precision = e06c_runner.configure_precision(torch)
    seeding = nr.seed_qualification(torch, SEED)
    counters = m6d5c.new_counters()
    m6d5c.instrument(torch, counters)          # forbids torch.save, autograd.grad and activation checkpointing
    env = m6d5a.environment() if cuda else None
    if cuda:
        lock = json.loads((nr.ROOT / nr.LOCK_PATH).read_text())
        diff = {k: {'lock': v, 'runtime': env.get(k)} for k, v in lock['identity'].items()
                if k != 'launch_environment' and env.get(k) != v}
        require(not diff, 'runtime identity equals the DSDG environment lock: ' + json.dumps(diff))
        snap = m6d5b.gpu_snapshot()
        foreign = [p for p in snap['compute_processes'] if int(p['pid']) != os.getpid()]
        require(not foreign and snap['used_mib'] <= 1024, 'clean GPU before execution')
        ctx['result']['gpu_before'] = snap
    ctx['result'].update(precision=precision, seeding=seeding, environment=env)
    return torch, F, counters


def finish(ctx, counters, allowed_ids):
    os.close(ctx['access_fd'])
    result = ctx['result']
    result['access_audit'] = access_audit(ctx['build'] / 'M6FD_ACCESS_LOG.tsv', allowed_ids)
    result['firewall'] = ctx['firewall'].report()
    result['counters'] = counters
    sci = ctx['runtime_root'] / 'runs' / 'm6' / nr.METHOD_ID
    result['scientific_run_root_exists'] = sci.exists()
    result['files_in_qualification_root'] = sorted(p.name for p in ctx['build'].iterdir())
    a = result['access_audit']
    result['firewall_gates'] = {
        'no_denied_event': not result['firewall']['denied'] and a['denied_events'] == 0,
        'no_VAL_TEST_other_manifest': a['VAL_accesses'] == a['TEST_accesses'] == a['other_manifest_accesses'] == 0,
        'faces_only_native_TRAIN': a['non_train_face_accesses'] == 0 and a['train_faces_all_in_native_relation'],
        'no_scientific_run_root': not result['scientific_run_root_exists'] and a['scientific_run_root_accesses'] == 0,
        'no_weight_saved': counters['checkpoint_saves'] == 0 and not any(
            n.endswith(('.pth', '.pt', '.ckpt', '.pkl')) for n in result['files_in_qualification_root'])}
    result.update(scientific_training_completed=False, scientific_seeds_completed=0, scientific_checkpoints_created=0,
                  scientific_bank_created=False,
                  native_manifest_created=(nr.ROOT / 'manifests/dsdg_identity_pairs_v1.parquet').exists(),
                  VAL_access=False, TEST_access=False)
    return result


# ================================================================= loader (+ real image decode check)
class IdOnly:
    """Production relation + the official draw, returning IDs only (no image) to audit the whole real epoch."""

    def __init__(self, relation):
        self.relation = relation

    def __len__(self):
        return len(self.relation)

    def __getitem__(self, index):
        import torch.utils.data as tud
        info = tud.get_worker_info()
        return {'index': index, 'live_id': self.relation.draw_live(index), 'worker': info.id, 'wseed': str(info.seed)}


def loader_case():
    ctx = setup('loader')
    torch, F, counters = torch_setup(ctx, cuda=False)
    rel, rows, config, result = ctx['relation'], ctx['rows'], ctx['config'], ctx['result']
    result['pre_execution_row_assertions'] = len(nr.assert_native_rows(rel, rows, range(len(rel))))
    loader, generator, evidence = nr.build_loader(torch, IdOnly(rel), nr.QUALIFICATION, SEED, config)
    t0 = time.monotonic()
    batches = [dict(indices=b['index'].tolist(), lives=list(b['live_id']), workers=b['worker'].tolist(),
                    wseeds=[int(s) for s in b['wseed']]) for b in loader]
    seconds = time.monotonic() - t0
    gen = torch.Generator().manual_seed(SEED)
    base = int(torch.empty((), dtype=torch.int64).random_(generator=gen).item())
    perm = torch.randperm(len(rel), generator=gen).tolist()
    expected_batches = [perm[i:i + 240] for i in range(0, len(perm), 240)]
    expected_draws = native.simulate_live_draws(rel, expected_batches, [(base + w) % 2 ** 32 for w in range(8)])
    order = [i for b in batches for i in b['indices']]
    result['loader'] = {'evidence': evidence, 'seconds': seconds, 'batch_sizes': [len(b['indices']) for b in batches],
                        'distinct_worker_ids': sorted({w for b in batches for w in b['workers']}),
                        'draw_sequence_sha256': rio.sha(json.dumps([b['lives'] for b in batches]).encode())}
    gates = {'batch_plan_15x240_plus_120': result['loader']['batch_sizes'] == [240] * 15 + [120],
             'every_spoof_row_once': sorted(order) == list(range(len(rel))),
             'workers_8_observed': result['loader']['distinct_worker_ids'] == list(range(8)),
             'batch_b_served_by_worker_b_mod_8': all(w == k % 8 for k, b in enumerate(batches) for w in b['workers']),
             'worker_seed_base_plus_id': all(s == base + w for b in batches for s, w in zip(b['wseeds'], b['workers'])),
             'indices_match_model': [b['indices'] for b in batches] == expected_batches,
             'real_draws_match_pure_python_model': [b['lives'] for b in batches] == expected_draws,
             'same_subject_every_draw': all(l in rel.pools[rel.spoof[i][1]] for b in batches
                                            for i, l in zip(b['indices'], b['lives']))}
    # ---- real image decode: 16 pairs, 4 per dataset x class, production reader + preprocessing
    reader = rio.CanonicalFaceReader(ctx['faces_root'], ctx['sample_datasets'], datasets=native.DATASETS)
    groups = {}
    for i, (sid, key, cls) in enumerate(rel.spoof):
        groups.setdefault((key[0], cls), []).append(i)
    rng = random.Random(SEED)
    pairs = []
    for g in sorted(groups):
        for i in groups[g][:IMAGE_CHECK_PER_GROUP]:
            sid, key, cls = rel.spoof[i]
            live = rel.draw_live(i, rng)
            s, l = native.canonical_chw(reader(sid)), native.canonical_chw(reader(live))
            pairs.append({'spoof_id': sid, 'live_id': live, 'dataset': key[0], 'subject_id_global': key[1],
                          'attack_macro': 'print' if cls == 0 else 'replay', 'class_index': cls,
                          'same_subject': live in rel.pools[key],
                          'tensors': [{'role': role, 'shape': list(a.shape), 'dtype': str(a.dtype),
                                       'min': float(a.min()), 'max': float(a.max()), 'mean': float(a.mean()),
                                       'sha256': rio.sha(a.tobytes())} for role, a in (('spoof', s), ('live', l))]})
    item = nr.E06bDataset(rel, reader)[groups[sorted(groups)[0]][0]]   # production item path (module random)
    result['image_check'] = {'pairs': pairs, 'production_dataset_item': {
        'keys': sorted(item), 'shape_0': list(item['0'].shape), 'dtype_0': str(item['0'].dtype),
        'type': int(item['type']), 'live_same_subject': item['live_id'] in rel.pools[rel.spoof[int(item['index'])][1]]}}
    gates.update({
        'image_pairs_16_both_classes_both_datasets': len(pairs) == 16 and
            {(p['dataset'], p['class_index']) for p in pairs} == {(d, c) for d in native.DATASETS for c in (0, 1)},
        'image_tensor_contract': all(t['shape'] == [3, 256, 256] and t['dtype'] == 'float32' and
                                     0 <= t['min'] < t['max'] <= 1 for p in pairs for t in p['tensors']),
        'image_pairs_same_subject': all(p['same_subject'] for p in pairs)})
    result['gates'] = gates
    allowed = set(ctx['sample_datasets'])
    result = finish(ctx, counters, allowed)
    passed = all(result['gates'].values()) and all(result['firewall_gates'].values())
    result['status'] = 'PASS' if passed else 'STOP_GATE'
    return ctx['build'], result


# ================================================================= b240 / tail120 (real production batch function)
def batch_case(case):
    lo, hi, epoch = CASES[case]
    ctx = setup(case)
    torch, F, counters = torch_setup(ctx, cuda=True)
    rel, rows, config, result = ctx['relation'], ctx['rows'], ctx['config'], ctx['result']
    sub, chosen = subset_relation(rel, rows, lo, hi)
    B = len(sub)
    result['subset'] = {'rows': B, 'group_slice': [lo, hi], 'spoof_ids': chosen, 'summary': sub.summary(),
                        'independent_case': True, 'not_consecutive_scientific_steps': True}
    result['pre_execution_row_assertions'] = nr.assert_native_rows(sub, rows, range(B))
    reader = rio.CanonicalFaceReader(ctx['faces_root'], ctx['sample_datasets'], datasets=native.DATASETS)
    dataset = nr.E06bDataset(sub, reader)
    from methods.common.upstream import upstream_modules
    with upstream_modules(Path(ctx['source']['root']) / config['source']['relevant_path'], m6d5a.UPSTREAM_MODULES,
                          m6d5a.UPSTREAM_ROOTS) as modules:
        models, binding, lightcnn_load = nr.build_models(torch, modules, ctx['source'], config)
        optimizer = tg.build_optimizer(torch, models['netE_nir'], models['netE_vis'], models['netG'],
                                       config['optimizer']['learning_rate'])
        result['optimizer'] = m6d5b.optimizer_evidence(torch, optimizer, models)
        m6d5c.count_steps(optimizer, counters)
        loader, generator, loader_evidence = nr.build_loader(torch, dataset, nr.QUALIFICATION, SEED, config)
        trainer = nr.E06bTrainer(torch, F, modules['misc.util'], models, optimizer, loader, generator, sub,
                                 nr.QUALIFICATION, SEED, dict(nr.LAMBDAS))
        trainer.set_modes()
        before = {n: m6d5b.cpu_copy(models[n]) for n in models}
        t0 = time.monotonic()
        batch = next(iter(loader))
        load_seconds = time.monotonic() - t0
        try:
            r = trainer.global_step_run(batch, epoch, 0)          # THE production batch-processing function
        except e06c_runner.TrainingStop as exc:
            result.update(status='STOP_OOM' if 'OOM' in str(exc) else 'STOP_GATE', stop=str(exc)[:2000])
            return ctx['build'], finish(ctx, counters, set(ctx['sample_datasets']))
        change = {n: m6d5b.parameter_change(before[n], models[n]) for n in models}
        del models, optimizer, trainer, before     # qualification weights discarded (never saved)
    pairs = [{'spoof_id': s, 'live_id': l, 'label': y} for s, l, y in zip(r['spoof_ids'], r['live_ids'], r['labels'])]
    result.update(
        binding=binding, lightcnn_load={k: lightcnn_load[k] for k in ('matched', 'missing', 'shape_mismatches')},
        loader=loader_evidence, batch_load_seconds=load_seconds, epoch_branch=epoch,
        logical_batch_size=r['batch_size'], physical_microbatch=nr.MICROBATCH, chunk_sizes=r['chunk_sizes'],
        chunks=len(r['chunk_sizes']), pairs=pairs, labels_count={'print_0': r['labels'].count(0),
                                                              'replay_1': r['labels'].count(1)},
        input_range=r['input_range'], losses={k: {'weighted': v, 'raw': v / ({'loss_mmd': 50, 'loss_ip': 1000,
                                                'loss_cls': 10, 'loss_ort': 1, 'loss_pair': 5}.get(k, 1))}
                                             for k, v in r['losses'].items()},
        total_loss=r['total_loss'], owned_gradients=r['owned_gradients'], netCls_grad_nonzero=r['netCls_grad_nonzero'],
        optimizer_applications=trainer_applications(counters), backward_calls=counters['backward_calls'],
        step_seconds=r['step_seconds'], peak_allocated_bytes=r['gpu_memory_bytes'],
        peak_reserved_bytes=r['gpu_reserved_bytes'],
        parameter_change={n: {k: v for k, v in c.items() if k != 'per_tensor'} for n, c in change.items()},
        batch_pair_sha256=r['batch_pair_sha256'], epsilon_sha256=r['epsilon_sha256'],
        weights_discarded=True, workers_configured=loader_evidence['num_workers'],
        worker_note='a single logical batch is served by worker 0; 8-worker behaviour is audited in the loader case')
    result['gates'] = {
        'logical_batch_and_chunks': (r['batch_size'], r['chunk_sizes']) == (B, [20] * (B // 20)) and B in (240, 120),
        'both_classes': min(result['labels_count'].values()) > 0,
        'same_subject_pairs': all(p['live_id'] in sub.pools[sub.spoof[i][1]] for i, p in zip(r['indices'], pairs)),
        'losses_finite': rio.finite([float(v) for v in r['losses'].values()] + [float(r['total_loss'])]),
        'loss_cls_positive_active': r['losses']['loss_cls'] > 0 and r['netCls_grad_nonzero'] > 0,
        'loss_pair_positive': r['losses']['loss_pair'] > 0,
        'owned_gradients_55_finite': r['owned_gradients'] == {'tensors': 55, 'non_none': 55, 'finite': 55},
        'one_optimizer_step': counters['optimizer_applications'] == 1 and counters['backward_calls'] == B // 20,
        'owned_parameters_changed': all(change[n]['changed_tensors'] > 0 for n in ('netE_nir', 'netE_vis', 'netG')),
        'netCls_netIP_unchanged': change['netCls']['changed_elements'] == change['netIP']['changed_elements'] == 0,
        'inputs_in_unit_range': 0 <= r['input_range'][0] <= r['input_range'][1] <= 1}
    result = finish(ctx, counters, set(ctx['sample_datasets']))
    passed = all(result['gates'].values()) and all(result['firewall_gates'].values())
    result['status'] = 'PASS' if passed else 'STOP_GATE'
    return ctx['build'], result


def trainer_applications(counters):
    return counters['optimizer_applications']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=('loader', 'b240', 'tail120'), required=True)
    args = parser.parse_args()
    started = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    build, result = loader_case() if args.case == 'loader' else batch_case(args.case)
    result['started_utc'], result['ended_utc'] = started, time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    name = f'M6FD_E06B_{args.case.upper()}.json'
    (build / name).write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + '\n')
    print(json.dumps({'status': result['status'], 'output': str(build / name)}))
    sys.exit(EXIT[result['status']])


if __name__ == '__main__':
    main()
