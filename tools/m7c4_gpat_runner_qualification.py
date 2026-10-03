"""M7C4 GPAT production-runner qualification (gpat-m7-gpu, RTX 3090). QUALIFICATION ONLY — never a scientific run.

Uses the production Trainer in QUALIFICATION mode (seed 70404) under <runtime_root>/qualification/m7/M7C4/. Opens only
the exact TRAIN rows each scenario needs (positions 0..7 / 8832..8837 of the qualification epoch-1 order, the
DEV022_QUALIFICATION_GROUP, warmup batches 1 and 139 of the qualification warmup epoch 1). No full epoch, no scientific
seed, no VAL/TEST, no scientific candidate, no bank. Each scenario runs in a fresh process (`--scenario NAME`); the
orchestrator (`--all`) runs them sequentially and writes one evidence JSON.

Scenarios: orders, b0_regular, b3_regular, b3_tail, dev022_group, warmup_regular, warmup_tail, warmup_handoff,
resume_ref, resume_part1, resume_part2, wresume_ref, wresume_part1, wresume_part2, candidate_writer.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from methods.gpat import runner_io as rio                   # noqa: E402

SEED = rio.QUALIFICATION_SEED
METHOD = {'b0': 'GPAT-B0', 'b3': 'GPAT-B3'}
SCENARIOS = ('orders', 'b0_regular', 'b3_regular', 'b3_tail', 'dev022_group', 'warmup_regular', 'warmup_tail',
             'warmup_handoff', 'resume_ref', 'resume_part1', 'resume_part2', 'wresume_ref', 'wresume_part1',
             'wresume_part2', 'candidate_writer')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


# ----------------------------------------------------------------------------- metadata-only scenario
def scenario_orders(_):
    records = rio.read_relation()
    out = {'relation': rio.RELATION, 'relation_sha256': rio.RELATION_SHA256, 'rows': len(records),
           'columns': list(rio.RELATION_COLUMNS), 'generator_epoch1': {}, 'warmup_epoch1': {}}
    for mode, seed in [(rio.SCIENTIFIC, s) for s in rio.EXPERIMENT_SEEDS] + [(rio.QUALIFICATION, SEED)]:
        g = rio.epoch_permutation('generator', mode, seed, 1)
        w = rio.epoch_permutation('warmup', mode, seed, 1)
        out['generator_epoch1'][f'{mode}_{seed}'] = rio.order_sha256(records, g)
        out['warmup_epoch1'][f'{mode}_{seed}'] = rio.order_sha256(records, w)
    out['layout'] = rio.layout_report(rio.epoch_permutation('generator', rio.QUALIFICATION, SEED, 1))
    imap = rio.identity_map(records)
    out['identity_map'] = {'K': len(imap['classes']), 'mapping_sha256': imap['mapping_sha256'],
                           'json_sha256': imap['json_sha256']}
    out['images_opened'] = 0
    return out


# ----------------------------------------------------------------------------- shared helpers (torch)
def setup(method, label, *, resume=False, load_teachers=True, num_workers=None):
    import torch
    from methods.gpat import runtime_contract as rc
    from methods.gpat import runner as R
    rc.apply_qualification_determinism(SEED, gpu=True)
    storage = rio.faces_root()
    assets = rio.load_assets()
    tr = R.Trainer(mode=rio.QUALIFICATION, method=METHOD[method], seed=SEED, runtime_root=storage['runtime_root'],
                   faces_root=storage['faces_256_root'], assets=assets, label=label, resume=resume,
                   load_teachers=load_teachers, num_workers=num_workers)
    tr.ctx.open()
    torch.cuda.reset_peak_memory_stats()
    return tr


def tensor_digest(t):
    return sha(t.detach().contiguous().cpu().numpy().tobytes())


def state_digest(obj):
    """SHA-256 over a (nested) state dict: tensors by bytes, everything else by canonical JSON."""
    import torch
    h = hashlib.sha256()

    def walk(node, key=''):
        if torch.is_tensor(node):
            h.update(key.encode() + str(node.dtype).encode() + str(tuple(node.shape)).encode())
            h.update(node.detach().contiguous().cpu().numpy().tobytes())
        elif isinstance(node, dict):
            for k in sorted(node, key=str):
                walk(node[k], f'{key}/{k}')
        elif isinstance(node, (list, tuple)):
            for i, v in enumerate(node):
                walk(v, f'{key}[{i}]')
        else:
            h.update(f'{key}={json.dumps(node, default=str)}'.encode())
    walk(obj)
    return h.hexdigest()


def trainer_state(tr):
    import torch
    from methods.gpat import runner_checkpoint as ck
    mods = {k: getattr(tr.core, k) for k in ('e_art', 'g_res', 'discriminator', 'attack_head', 'identity_head')}
    out = {f'module_{k}': state_digest(m.state_dict()) for k, m in mods.items() if m is not None}
    bn = {k: v for k, v in tr.core.e_art.state_dict().items() if 'running' in k or 'num_batches' in k}
    out['e_art_bn_buffers'] = state_digest(bn)
    for name in ('g_opt', 'd_opt', 'w_opt'):
        o = getattr(tr, name, None)
        if o is not None:
            out[name] = state_digest(o.state_dict())
    for name in ('g_scaler', 'd_scaler', 'w_scaler'):
        s = getattr(tr, name, None)
        if s is not None:
            out[name] = state_digest(s.state_dict())
    out['position'] = dict(tr.position)
    out['rng'] = state_digest(ck.rng_state())
    out['ema'] = None if tr.ema is None else {k: state_digest(v.state_dict()) for k, v in tr.ema.items()}
    return out


def grad_inspector(tr, sink):
    """After unscale_: per-module gradient finiteness / presence / norm; teacher params must have no grad."""
    import torch

    def inspect(stage, step):
        mods = ({'discriminator': tr.core.discriminator} if stage == 'D' else tr.core.generator_modules())
        rec = {}
        for name, m in mods.items():
            grads = [p.grad for p in m.parameters()]
            present = [g for g in grads if g is not None]
            rec[name] = {'params': len(grads), 'with_grad': len(present),
                         'finite': all(bool(torch.isfinite(g).all()) for g in present),
                         'norm': float(torch.norm(torch.stack([g.float().norm() for g in present]))) if present else 0.0}
        if stage == 'G':
            rec['teacher_param_grads_none'] = all(p.grad is None for m in tr.teachers.modules().values()
                                                  for p in m.parameters())
            ih = tr.core.identity_head
            if ih is not None:
                rec['identity_head_fc_grad_norm'] = float(ih.fc.weight.grad.norm()) if ih.fc.weight.grad is not None else 0.0
        sink[stage] = rec
    return inspect


def run_groups(tr, n, *, epoch=1, start_group=1):
    from methods.gpat import runner as R
    tr.position.update(stage='generator', epoch=epoch, next_group=start_group,
                       global_update=rio.update_index(epoch, start_group) - 1)
    tr.step.capture = []
    t0 = time.monotonic()
    tr.run_generator_groups(R.StopFlag(), limit=n)
    import torch
    torch.cuda.synchronize()
    return time.monotonic() - t0


def read_metrics(tr, record_type):
    return [json.loads(l) for l in tr.ctx.path('metrics').read_text().splitlines()
            if json.loads(l)['record_type'] == record_type]


def finish(tr, out):
    import torch
    out['peak_memory'] = {'max_memory_allocated_bytes': int(torch.cuda.max_memory_allocated()),
                          'max_memory_reserved_bytes': int(torch.cuda.max_memory_reserved()),
                          'max_memory_reserved_GiB': round(torch.cuda.max_memory_reserved() / 2 ** 30, 3)}
    out['access'] = tr.access.report()
    out['opened_ids'] = sorted({s for _, s in tr.access.opened})
    out['run_dir'] = str(tr.ctx.run_dir)
    out['labels'] = list(rio.QUALIFICATION_LABELS)
    tr.ctx.close('qualification_completed', {'access': out['access']})
    return out


# ----------------------------------------------------------------------------- generator scenarios
def generator_group(method, label, *, epoch=1, group=1):
    import torch
    tr = setup(method, label)
    sink = {}
    tr.step.inspect = grad_inspector(tr, sink)
    plan = tr.generator_plan(epoch)[group - 1]
    before = {k: state_digest(getattr(tr.core, k).state_dict()) for k in ('e_art', 'g_res', 'discriminator')}
    seconds = run_groups(tr, 1, epoch=epoch, start_group=group)
    rec = read_metrics(tr, 'optimizer_group')[-1]
    after = {k: state_digest(getattr(tr.core, k).state_dict()) for k in ('e_art', 'g_res', 'discriminator')}
    out = {'method': METHOD[method], 'epoch': epoch, 'group': group, 'planned_indices': plan,
           'pair_ids': [[tr.records[i]['pair_id'] for i in mb] for mb in plan], 'group_record': rec,
           'capture': tr.step.capture, 'grads_after_unscale': sink, 'wall_seconds': seconds,
           'params_changed': {k: before[k] != after[k] for k in before},
           'expected_lr': __import__('methods.gpat.runtime_contract', fromlist=['x']).main_lr(rec['global_update']),
           'identity_valid_per_microbatch': [[bool(v) for v in
                                              (rio.identity_labels_for([tr.records[i] for i in mb], tr.imap)[1]
                                               if tr.imap else [False] * len(mb))] for mb in plan]}
    return finish(tr, out)


def scenario_b0_regular(_):
    return generator_group('b0', 'b0_regular')


def scenario_b3_regular(_):
    return generator_group('b3', 'b3_regular')


def scenario_b3_tail(_):
    out = generator_group('b3', 'b3_tail', group=rio.GROUPS_PER_EPOCH)
    out['full_batch_reference'] = tail_reference('b3', out['planned_indices'])
    return out


def tail_reference(method, plan):
    """Independent check of the weighting algebra: weighted microbatch means (4/6, 2/6) vs the full-6 batch means.
    Evaluated without a step, E_art BN in eval mode for both (train-mode BN is batch-statistic dependent)."""
    import torch
    from methods.gpat import runtime_contract as rc
    tr = setup(method, 'b3_tail_reference')
    tr.core.eval()
    cur = rc.curriculum(rio.GROUPS_PER_EPOCH)
    batches = [tr.rd.GeneratorPairs(tr.records, tr.gen_loader.dataset.reader,
                                    rio.identity_labels_for(tr.records, tr.imap))[i] for i in sum(plan, [])]
    for b in batches:
        tr.access.check(tr.records[b['index']]['source_spoof_id'], 'source_spoof_id')
        tr.access.check(tr.records[b['index']]['target_live_id'], 'target_live_id')

    def stack(items):
        from torch.utils.data import default_collate
        from methods.gpat import runner as R
        return R.to_device(default_collate(items))

    def comps(mb):
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                out = tr.core(mb['x_source'], mb['x_target'], scale_hf=cur['s_hf'])
            c = tr.step.components(mb, out, cur)
            from methods.gpat import losses
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                c['D'] = losses.l_d(tr.core.discriminator(mb['x_source']), tr.core.discriminator(out.x_hat.detach()))
        return {k: float(v) for k, v in c.items()}
    full = comps(stack(batches))
    a, b = comps(stack(batches[:4])), comps(stack(batches[4:]))
    weighted = {k: (4 / 6) * a[k] + (2 / 6) * b[k] for k in full}
    rel = {k: abs(weighted[k] - full[k]) / max(1e-12, abs(full[k])) for k in full}
    out = {'weights': [4 / 6, 2 / 6], 'full_batch': full, 'weighted_microbatches': weighted, 'relative_difference': rel,
           'max_relative_difference': max(rel.values())}
    return finish(tr, out)


def scenario_dev022_group(_):
    """DEV022_QUALIFICATION_GROUP: deterministic, TRAIN-only, >=1 labelled (CASIA/MSU) and >=1 masked (SiW) row per
    microbatch, taken in qualification epoch-1 order. It does not redefine the scientific epoch order."""
    import torch
    from methods.gpat import losses
    tr = setup('b3', 'dev022_group')
    perm = rio.epoch_permutation('generator', rio.QUALIFICATION, SEED, 1)
    labelled = [i for i in perm if tr.records[i]['dataset'] != 'siwmv2'][:3]
    masked = [i for i in perm if tr.records[i]['dataset'] == 'siwmv2'][:5]
    group = [[labelled[0], masked[0], labelled[1], masked[1]], [masked[2], labelled[2], masked[3], masked[4]]]
    sink = {}
    tr.step.inspect = grad_inspector(tr, sink)
    tr.step.capture = []
    ds = tr.gen_loader.dataset
    from torch.utils.data import default_collate
    from methods.gpat import runner as R
    mbs = []
    for mb in group:
        items = [ds[i] for i in mb]
        for it in items:
            tr.access.check(tr.records[it['index']]['source_spoof_id'], 'source_spoof_id')
            tr.access.check(tr.records[it['index']]['target_live_id'], 'target_live_id')
        mbs.append(R.to_device(default_collate(items)))
    # independent reference of the group identity CE (same pre-update state, no step)
    with torch.no_grad():
        logits = []
        for mb in mbs:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits.append(tr.core(mb['x_source'], mb['x_target'], scale_hf=0.02).identity_logits.float())
    ce_all = torch.cat([torch.nn.functional.cross_entropy(lg[mb['identity_valid']], mb['identity'][mb['identity_valid']],
                                                          reduction='none') for lg, mb in zip(logits, mbs)])
    n_lab = int(sum(int(mb['identity_valid'].sum()) for mb in mbs))
    rec = tr.step(mbs, 1)
    tr.ctx.log('optimizer_group', dict(rec, epoch=0, group=0, note='DEV022_QUALIFICATION_GROUP'))
    shares = [c['components']['idadv_share'] for c in tr.step.capture]
    sample_weighted = sum(w * (float(ce[...].sum()) / max(1, int(mb['identity_valid'].sum())))
                          for w, ce, mb in zip([0.5, 0.5], [ce_all[:int(mbs[0]['identity_valid'].sum())],
                                                             ce_all[int(mbs[0]['identity_valid'].sum()):]], mbs))
    out = {'label': 'DEV022_QUALIFICATION_GROUP', 'group_indices': group,
           'pair_ids': [[tr.records[i]['pair_id'] for i in mb] for mb in group],
           'labelled_per_microbatch': [int(mb['identity_valid'].sum()) for mb in mbs], 'labelled_total': n_lab,
           'group_record_labelled': rec['labelled_identity_count'], 'shares': shares, 'group_L_idadv': sum(shares),
           'reference_sum_ce_over_labelled': float(ce_all.sum()) / n_lab,
           'sample_weighted_microbatch_means': sample_weighted, 'grads_after_unscale': sink, 'group_record': rec}
    out['abs_diff'] = abs(out['group_L_idadv'] - out['reference_sum_ce_over_labelled'])
    out['differs_from_sample_weighted'] = abs(out['group_L_idadv'] - sample_weighted) > 1e-4
    return finish(tr, out)


# ----------------------------------------------------------------------------- warmup scenarios
def warmup_batch(label, batch_no):
    import torch
    from methods.gpat import runner as R
    from methods.gpat import runtime_contract as rc
    tr = setup('b3', label, load_teachers=False)
    tr.position.update(warmup_epoch=1, warmup_next_batch=batch_no, warmup_step=batch_no - 1)
    bn_before = state_digest({k: v for k, v in tr.core.e_art.state_dict().items() if 'running' in k})
    head_before = state_digest(tr.core.attack_head.state_dict())
    sink = {}
    tr.warm.inspect = lambda step: sink.update(
        finite=all(bool(torch.isfinite(p.grad).all()) for p in step.params if p.grad is not None),
        with_grad=sum(p.grad is not None for p in step.params), params=len(step.params))
    t0 = time.monotonic()
    tr.run_warmup_batches(R.StopFlag(), limit=1)
    torch.cuda.synchronize()
    rec = read_metrics(tr, 'warmup_step')[-1]
    plan = tr.warmup_plan(1)[batch_no - 1]
    out = {'batch_no': batch_no, 'planned_size': len(plan), 'record': rec, 'wall_seconds': time.monotonic() - t0,
           'expected_lr': rc.attack_warmup_lr(batch_no), 'e_art_training': tr.core.e_art.training,
           'bn_buffers_changed': bn_before != state_digest({k: v for k, v in tr.core.e_art.state_dict().items()
                                                             if 'running' in k}),
           'attack_head_changed': head_before != state_digest(tr.core.attack_head.state_dict()),
           'adamw_state_created': len(tr.w_opt.state) > 0, 'warmup_scaler_enabled': tr.w_scaler.is_enabled(),
           'teachers_loaded': tr.teachers is not None, 'grads': sink,
           'target_images_opened': sum(role == 'target_live_id' for role, _ in tr.access.opened)}
    return tr, out


def scenario_warmup_regular(_):
    tr, out = warmup_batch('warmup_regular', 1)
    return finish(tr, out)


def scenario_warmup_tail(_):
    tr, out = warmup_batch('warmup_tail', rio.WARMUP_STEPS_PER_EPOCH)
    return finish(tr, out)


def scenario_warmup_handoff(_):
    tr, out = warmup_batch('warmup_handoff', 1)
    carried = {'e_art': state_digest(tr.core.e_art.state_dict()),
               'attack_head': state_digest(tr.core.attack_head.state_dict())}
    w_opt, w_scaler = tr.w_opt, tr.w_scaler
    w_scaler_state = state_digest(w_scaler.state_dict())
    tr.finish_warmup()
    out['handoff'] = {
        'e_art_preserved': state_digest(tr.core.e_art.state_dict()) == carried['e_art'],
        'attack_head_preserved': state_digest(tr.core.attack_head.state_dict()) == carried['attack_head'],
        'warmup_optimizer_dropped': tr.w_opt is None and tr.warm is None,
        'warmup_scaler_dropped': tr.w_scaler is None,
        'g_opt_is_new_object': tr.g_opt is not w_opt and tr.step.g_opt is tr.g_opt,
        'g_opt_state_empty': len(tr.g_opt.state) == 0,
        'g_scaler_fresh': tr.g_scaler is not w_scaler and tr.g_scaler._scale is None and
        state_digest(tr.g_scaler.state_dict()) != w_scaler_state,
        'warmup_scaler_scale_was': float(w_scaler.get_scale()),
        'g_opt_params_cover_heads': len(tr.g_opt.param_groups[0]['params']) == sum(
            len(list(m.parameters())) for m in tr.core.generator_modules().values()),
        'stage': tr.position['stage']}
    return finish(tr, out)


# ----------------------------------------------------------------------------- resume scenarios (fresh processes)
def scenario_resume_ref(_):
    tr = setup('b3', 'resume_ref')
    run_groups(tr, 1)
    s1 = trainer_state(tr)
    cap1 = tr.step.capture
    tr.step.capture = []
    from methods.gpat import runner as R
    tr.run_generator_groups(R.StopFlag(), limit=1)
    out = {'after_group1': s1, 'after_group2': trainer_state(tr), 'group1_capture': cap1,
           'group2_capture': tr.step.capture, 'group2_record': read_metrics(tr, 'optimizer_group')[-1]}
    return finish(tr, out)


def scenario_resume_part1(_):
    tr = setup('b3', 'resume_split')
    run_groups(tr, 1)
    info = tr.save_recovery()
    out = {'after_group1': trainer_state(tr), 'recovery': info, 'group1_capture': tr.step.capture}
    return finish(tr, out)


def scenario_resume_part2(_):
    tr = setup('b3', 'resume_split', resume=True)
    rec = tr.ctx.ckpt_dir / 'recovery' / 'latest.pt'
    p = tr.load_recovery(rec)
    loaded = trainer_state(tr)
    tr.step.capture = []
    from methods.gpat import runner as R
    tr.run_generator_groups(R.StopFlag(), limit=1)
    out = {'loaded_state': loaded, 'after_group2': trainer_state(tr), 'group2_capture': tr.step.capture,
           'group2_record': read_metrics(tr, 'optimizer_group')[-1], 'recovery_kind': p['kind'],
           'recovery_labels': p['labels']}
    return finish(tr, out)


def _wr(label, resume=False):
    return setup('b3', label, resume=resume, load_teachers=False)


def scenario_wresume_ref(_):
    from methods.gpat import runner as R
    tr = _wr('wresume_ref')
    tr.run_warmup_batches(R.StopFlag(), limit=1)
    s1 = trainer_state(tr)
    tr.run_warmup_batches(R.StopFlag(), limit=1)
    out = {'after_batch1': s1, 'after_batch2': trainer_state(tr), 'records': read_metrics(tr, 'warmup_step')}
    return finish(tr, out)


def scenario_wresume_part1(_):
    from methods.gpat import runner as R
    tr = _wr('wresume_split')
    tr.run_warmup_batches(R.StopFlag(), limit=1)
    info = tr.save_warmup_recovery()
    out = {'after_batch1': trainer_state(tr), 'recovery': info}
    return finish(tr, out)


def scenario_wresume_part2(_):
    from methods.gpat import runner as R
    tr = _wr('wresume_split', resume=True)
    tr.load_warmup_recovery(tr.ctx.ckpt_dir / 'recovery' / 'warmup_latest.pt')
    loaded = trainer_state(tr)
    tr.run_warmup_batches(R.StopFlag(), limit=1)
    out = {'loaded_state': loaded, 'after_batch2': trainer_state(tr), 'records': read_metrics(tr, 'warmup_step')}
    return finish(tr, out)


# ----------------------------------------------------------------------------- candidate writer (synthetic state)
def scenario_candidate_writer(_):
    import torch
    from methods.gpat import runner_checkpoint as ck
    from methods.gpat import runtime_contract as rc
    from methods.gpat.config import load_config
    from methods.gpat.ema import ModelEMA
    from methods.gpat.model import GPATCore
    import torchvision
    rc.apply_qualification_determinism(SEED, gpu=True)
    torch.manual_seed(SEED)
    core = GPATCore(load_config('B3'), pretrained_state=torchvision.models.resnet18(weights=None).state_dict(),
                    with_discriminator=True)
    ema = {'e_art': ModelEMA(core.e_art), 'g_res': ModelEMA(core.g_res)}
    storage = rio.faces_root()
    qdir = Path(storage['runtime_root']).joinpath(*rio.QUALIFICATION_PARTS, f'candidate_writer_q{SEED}')
    meta = {'method': 'GPAT-B3', 'seed': SEED, 'runner_mode': rio.QUALIFICATION, 'epoch': 10, 'global_update': 11050,
            'config_sha256': load_config('B3').sha256, 'code_commit': 'QUALIFICATION_SYNTHETIC_STATE',
            'gpu_env_lock_sha256': rio.GPU_LOCK_SHA256, 'source_manifest_sha256': rio.RELATION_SHA256,
            'teacher_sha256': rio.TEACHER_SHA256, 'ema_decay': rc.EMA_DECAY, 'gamma': 0.0,
            'labels': list(rio.QUALIFICATION_LABELS)}
    payload = ck.candidate_payload(e_art_ema=ema['e_art'], g_res_ema=ema['g_res'], metadata=meta)
    entry = ck.write_candidate(qdir, 10, payload)
    loaded = ck.load(entry['path'])
    try:
        ck.write_candidate(qdir, 10, payload)
        immutable = False
    except Exception:
        immutable = True
    leftovers = [p.name for p in qdir.iterdir() if p.name.startswith('.tmp-')]
    out = {'entry': entry, 'keys': sorted(loaded), 'metadata': loaded['metadata'],
           'e_art_keys_equal': sorted(loaded['e_art_ema']['module']) == sorted(core.e_art.state_dict()),
           'g_res_keys_equal': sorted(loaded['g_res_ema']['module']) == sorted(core.g_res.state_dict()),
           'contains_discriminator_or_heads': any(k.startswith(('discriminator', 'attack', 'identity'))
                                                  for k in list(loaded['e_art_ema']['module']) +
                                                  list(loaded['g_res_ema']['module'])),
           'weights_only_load': True, 'immutable_second_write_refused': immutable, 'tmp_leftovers': leftovers,
           'sha256_recomputed': ck.file_sha256(entry['path']) == entry['sha256'], 'scientific_path': False}
    os.unlink(entry['path'])
    out['bytes_deleted_after_hash'] = not Path(entry['path']).exists()
    out['labels'] = list(rio.QUALIFICATION_LABELS)
    return out


# ----------------------------------------------------------------------------- evidence assembly (laptop, no torch)
def _finite_tree(node):
    if isinstance(node, dict):
        return all(_finite_tree(v) for v in node.values())
    if isinstance(node, (list, tuple)):
        return all(_finite_tree(v) for v in node)
    if isinstance(node, float):
        return node == node and abs(node) != float('inf')
    return True


# Tail check tolerance: fp32 kernel noise between batch shapes 4/2 and 6 (cuDNN/cuBLAS pick different algorithms);
# several components are ~1e-7..1e-6 at initialization (x_hat ~ x_t), so a pure relative bound is meaningless there.
TAIL_ABS, TAIL_REL = 5e-6, 1e-4


def assemble(workdir, authority, lock_sha256):
    """Combine the scenario outputs into one evidence record; every gate is recomputed from raw values here."""
    from methods.gpat import runtime_contract as rc
    w = Path(workdir)
    sc = {n: json.loads((w / f'{n}.json').read_text()) for n in SCENARIOS}
    orch = json.loads((w / 'orchestration.json').read_text())
    g = {}
    o = sc['orders']
    g['orders'] = (o['rows'] == 8838 and o['layout']['microbatches'] == 2210 and o['layout']['optimizer_groups'] == 1105
                   and o['layout']['tail_group'] == [4, 2] and o['identity_map']['K'] == 60 and o['images_opened'] == 0)

    def gen_ok(r, method):
        rec, gr = r['group_record'], r['grads_after_unscale']
        ok = (_finite_tree(rec['train_losses']) and rec['learning_rate'] == r['expected_lr'] and
              gr['G']['teacher_param_grads_none'] is True and gr['D']['discriminator']['finite'] and
              gr['D']['discriminator']['with_grad'] > 0 and
              all(v['finite'] for k, v in gr['G'].items() if isinstance(v, dict)) and
              gr['G']['g_res']['with_grad'] > 0 and gr['G']['e_art']['with_grad'] > 0 and
              gr['G']['g_res']['norm'] > 0 and gr['G']['e_art']['norm'] > 0 and
              rec['learning_rate'] == rc.main_lr(rec['global_update']))
        if method == 'b3':
            ok = ok and 'type' in rec['train_losses'] and 'idadv_share' in rec['train_losses'] and \
                gr['G']['attack_head']['with_grad'] > 0 and \
                (rec['labelled_identity_count'] == 0 or gr['G']['identity_head']['with_grad'] > 0)
        return ok
    g['b0_regular'] = gen_ok(sc['b0_regular'], 'b0') and sc['b0_regular']['group_record']['microbatch_sizes'] == [4, 4]
    g['b3_regular'] = gen_ok(sc['b3_regular'], 'b3') and sc['b3_regular']['group_record']['microbatch_sizes'] == [4, 4]
    t = sc['b3_tail']
    g['b3_tail'] = (gen_ok(t, 'b3') and t['group_record']['microbatch_sizes'] == [4, 2] and
                    t['group_record']['sample_weights'] == [4 / 6, 2 / 6] and t['group_record']['global_update'] == 1105
                    and all(abs(t['full_batch_reference']['weighted_microbatches'][k] - v) <= TAIL_ABS + TAIL_REL * abs(v)
                            for k, v in t['full_batch_reference']['full_batch'].items()))
    d = sc['dev022_group']
    g['dev022'] = (d['labelled_total'] == d['group_record_labelled'] == sum(d['labelled_per_microbatch']) and
                   all(0 < n < 4 for n in d['labelled_per_microbatch']) and
                   abs(d['group_L_idadv'] - d['reference_sum_ce_over_labelled']) <= 1e-4 * max(1.0, abs(d['group_L_idadv']))
                   and d['differs_from_sample_weighted'])
    for name, size, step in (('warmup_regular', 64, 1), ('warmup_tail', 6, 139)):
        r = sc[name]
        g[name] = (r['record']['batch_size'] == size == r['planned_size'] and r['record']['warmup_step'] == step and
                   r['record']['learning_rate'] == r['expected_lr'] == rc.attack_warmup_lr(step) and r['e_art_training']
                   and r['bn_buffers_changed'] and r['adamw_state_created'] and r['warmup_scaler_enabled'] and
                   not r['teachers_loaded'] and r['target_images_opened'] == 0 and r['grads']['finite'] and
                   _finite_tree(r['record']))
    g['warmup_tail'] = g['warmup_tail']          # attack head may legitimately be unchanged only if the step was skipped
    h = sc['warmup_handoff']['handoff']
    g['warmup_handoff'] = all(h[k] for k in ('e_art_preserved', 'attack_head_preserved', 'warmup_optimizer_dropped',
                                             'warmup_scaler_dropped', 'g_opt_is_new_object', 'g_opt_state_empty',
                                             'g_scaler_fresh', 'g_opt_params_cover_heads')) and h['stage'] == 'generator'
    ref, p1, p2 = sc['resume_ref'], sc['resume_part1'], sc['resume_part2']
    resume = {'group1_state_equal': ref['after_group1'] == p1['after_group1'],
              'loaded_equals_saved': p2['loaded_state'] == p1['after_group1'],
              'group2_capture_equal': ref['group2_capture'] == p2['group2_capture'],
              'after_group2_equal': ref['after_group2'] == p2['after_group2'],
              'group2_record_equal': {k: v for k, v in ref['group2_record'].items() if k not in ('utc', 'wall_seconds',
                                      'peak_allocated_bytes')} ==
                                     {k: v for k, v in p2['group2_record'].items() if k not in ('utc', 'wall_seconds',
                                      'peak_allocated_bytes')}}
    g['resume'] = all(resume.values())
    wr, w1, w2 = sc['wresume_ref'], sc['wresume_part1'], sc['wresume_part2']
    strip = lambda recs: [{k: v for k, v in r.items() if k not in ('utc', 'wall_seconds', 'peak_allocated_bytes')}
                          for r in recs]                                                       # noqa: E731
    wresume = {'batch1_state_equal': wr['after_batch1'] == w1['after_batch1'],
               'loaded_equals_saved': w2['loaded_state'] == w1['after_batch1'],
               'after_batch2_equal': wr['after_batch2'] == w2['after_batch2'],
               'batch2_record_equal': strip(wr['records'])[1] == strip(w2['records'])[-1]}
    g['warmup_resume'] = all(wresume.values())
    c = sc['candidate_writer']
    g['candidate_writer'] = (c['keys'] == ['e_art_ema', 'g_res_ema', 'kind', 'metadata'] and
                             c['metadata']['selected'] is False and c['e_art_keys_equal'] and c['g_res_keys_equal'] and
                             not c['contains_discriminator_or_heads'] and c['immutable_second_write_refused'] and
                             c['sha256_recomputed'] and not c['tmp_leftovers'] and c['bytes_deleted_after_hash'] and
                             not any('val' in k.lower() for k in c['metadata']))
    firewall = {k: 0 for k in ('train_images', 'non_train_images', 'val_images', 'test_images', 'val_metadata',
                               'test_metadata')}
    opened = set()
    for name, r in sc.items():
        if 'access' in r:
            for k in firewall:
                firewall[k] += r['access'][k]
            opened |= set(r.get('opened_ids', []))
    firewall.update(unique_ids=len(opened), unique_ids_sha256=sha('\n'.join(sorted(opened)).encode()))
    g['firewall'] = all(firewall[k] == 0 for k in ('non_train_images', 'val_images', 'test_images', 'val_metadata',
                                                     'test_metadata'))
    g['all_scenarios_exit_0'] = all(v['exit_code'] == 0 for v in orch.values()) and len(orch) == len(SCENARIOS)
    resources = {n: {'peak_memory': r.get('peak_memory'), 'wall_seconds': r.get('wall_seconds'),
                     'group_wall_seconds': r.get('group_record', {}).get('wall_seconds') if isinstance(r.get(
                         'group_record'), dict) else None} for n, r in sc.items()}
    peak = max(r['peak_memory']['max_memory_reserved_bytes'] for r in sc.values() if r.get('peak_memory'))
    g['no_oom_below_24GiB'] = peak < 24 * 2 ** 30
    return {'milestone': 'M7C4', 'kind': 'gpat_runner_qualification', 'authority_commit': authority,
            'qualification_seed': SEED, 'labels': list(rio.QUALIFICATION_LABELS), 'gpu_env_lock_sha256': lock_sha256,
            'relation': rio.RELATION, 'relation_sha256': rio.RELATION_SHA256, 'teacher_sha256': rio.TEACHER_SHA256,
            'tail_reference_tolerance': {'abs': TAIL_ABS, 'rel': TAIL_REL,
                                         'abs_differences': {k: abs(t['full_batch_reference']['weighted_microbatches'][k] - v)
                                                             for k, v in t['full_batch_reference']['full_batch'].items()}},
            'orchestration': orch, 'gates': g, 'resume_equivalence': resume, 'warmup_resume_equivalence': wresume,
            'firewall': firewall, 'opened_train_ids': sorted(opened), 'resources': resources,
            'peak_reserved_bytes_max': peak, 'scenarios': sc,
            'status': 'PASS' if all(g.values()) else 'FAIL'}


# ----------------------------------------------------------------------------- orchestration
def run_scenario(name, args):
    out = globals()['scenario_' + name](args)
    Path(args.out).write_text(json.dumps(out, indent=1, sort_keys=True, default=str) + '\n')
    print(f'{name}: done')


def orchestrate(args):
    qroot = Path(args.workdir)
    qroot.mkdir(parents=True, exist_ok=True)
    results = {}
    for name in SCENARIOS:
        out = qroot / f'{name}.json'
        t0 = time.monotonic()
        r = subprocess.run([sys.executable, '-B', __file__, '--scenario', name, '--out', str(out)],
                           env=dict(os.environ), capture_output=True, text=True)
        (qroot / f'{name}.log').write_text(r.stdout + r.stderr)
        results[name] = {'exit_code': r.returncode, 'seconds': round(time.monotonic() - t0, 1)}
        if r.returncode != 0:
            print(f'{name}: FAILED (see {name}.log)')
            break
        print(f'{name}: ok ({results[name]["seconds"]} s)')
    (qroot / 'orchestration.json').write_text(json.dumps(results, indent=1) + '\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--scenario', choices=SCENARIOS)
    ap.add_argument('--out')
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--workdir')
    ap.add_argument('--assemble', action='store_true', help='laptop: combine scenario outputs into the evidence JSON')
    ap.add_argument('--authority')
    args = ap.parse_args()
    if args.assemble:
        ev = assemble(args.workdir, args.authority, rio.GPU_LOCK_SHA256)
        Path(args.out).write_text(json.dumps(ev, indent=1, sort_keys=True, default=str) + '\n')
        print(json.dumps({'status': ev['status'], 'gates': ev['gates']}, indent=1))
        return
    if args.all:
        orchestrate(args)
    else:
        run_scenario(args.scenario, args)


if __name__ == '__main__':
    main()
