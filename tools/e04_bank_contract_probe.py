"""E04 Physics-STD bank-contract evidence probe (read-only, CPU-only; no inference, no data, no GPU).

Purpose: give the owner verifiable facts for the E04 bank-generation DRAFT ruling
(configs/amendments/e04_bank_generation_contract.DRAFT.yaml). It never runs a session on images, never reads a face,
never writes into a run root and never restores a checkpoint into a graph.

  host   (GPU host, gpat-m6-e03 Python 3.8 / TF 1.15, CUDA_VISIBLE_DEVICES='' and a fresh clean clone):
         * sha256 + size of the three iteration_150000 files of every seed;
         * the variable inventory of every final checkpoint (tf.train.NewCheckpointReader, CPU) with the values of
           global_iteration / minibatch_phase and summary statistics of every generator BN moving statistic;
         * graph-only construction of the frozen M6D3e TrainingGraph (no Session) and a dependency analysis of the
           'synthesized_spoof' tensor: placeholders, variables, BN mode, AddN ops, T_A membership, CPU pinning;
         * which variables the original-pass and hard-pass BN UPDATE_OPS assign;
         * the E04 feed-pack and geometry pre-pass manifest sha256.
         Prints one JSON document on stdout.
  build  (laptop): verifies the collected JSON against the committed checkpoint inventory and the pinned identities
         and writes outputs/audit/E04_BANK_CONTRACT_EVIDENCE.{json,md} + _SHA256.txt.
"""
import argparse
import collections
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

SEEDS = (42, 1337, 2026)
FINAL = 'iteration_150000'
SUFFIXES = ('.data-00000-of-00001', '.index', '.meta')
OUT = 'outputs/audit/E04_BANK_CONTRACT_EVIDENCE'
COLLECTED = 'outputs/audit/E04_BANK_CONTRACT_EVIDENCE/collected_host.json'
INVENTORY = 'outputs/audit/E04_THREE_SEED_CHECKPOINT_INVENTORY.csv'
FINAL_CSV = 'outputs/audit/E04_THREE_SEED_FINAL_CHECKPOINTS.csv'
CLOSURE = 'outputs/audit/E04_THREE_SEED_TRAINING_CLOSURE.json'


def sha_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(str(path), 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def dumps(obj):
    return json.dumps(obj, sort_keys=True, indent=1, ensure_ascii=True) + '\n'


# ====================================================================== host
def _classify(name):
    if name in ('global_iteration', 'minibatch_phase') or name.endswith('_power'):
        return 'counter'
    if 'Adam' in name:
        return 'optimizer_slot'
    if 'ExponentialMovingAverage' in name:
        return 'ema_shadow'
    if name.startswith('G/'):
        if name.endswith('/moving_mean') or name.endswith('/moving_variance'):
            return 'G_bn_moving'
        return 'G_trainable'
    if name.startswith('D/'):
        return 'D_bn_moving' if name.endswith(('/moving_mean', '/moving_variance')) else 'D_trainable'
    return 'other'


def _variable_of(tensor):
    op, seen = tensor.op, 0
    while op.type in ('Identity', 'ReadVariableOp') and op.inputs and seen < 8:
        op, seen = op.inputs[0].op, seen + 1
    return op.name if op.type in ('VariableV2', 'VarHandleOp', 'Variable') else None


def _ancestors(op):
    seen, stack = {}, [op]
    while stack:
        o = stack.pop()
        if o.name in seen:
            continue
        seen[o.name] = o
        stack.extend(t.op for t in o.inputs)
        stack.extend(o.control_inputs)
    return seen


def host(runtime_root):
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
    require_cpu = os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))
    import numpy as np
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    from methods.physics_std import production_io as pio
    from methods.physics_std.training_graph import TrainingGraph
    doc = {'phase': 'host', 'cpu_only': require_cpu, 'python': sys.version.split()[0], 'executable': sys.executable,
           'tensorflow': tf.__version__, 'numpy': np.__version__, 'repo': str(repo),
           'utc': subprocess.run(['date', '-u', '+%FT%TZ'], stdout=subprocess.PIPE, universal_newlines=True).stdout.strip(),
           'head': subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'], stdout=subprocess.PIPE,
                                  universal_newlines=True).stdout.strip(),
           'clean': subprocess.run(['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=all'],
                                   stdout=subprocess.PIPE, universal_newlines=True).stdout.strip() == ''}
    rt = Path(runtime_root)
    doc['checkpoints'], doc['checkpoint_variables'] = {}, {}
    for seed in SEEDS:
        prefix = rt / 'runs' / 'm6' / 'E04' / ('seed_%d' % seed) / 'checkpoints' / FINAL
        doc['checkpoints'][str(seed)] = {
            'prefix': str(prefix),
            'files': {FINAL + s: {'bytes': os.path.getsize(str(prefix) + s), 'sha256': sha_file(str(prefix) + s)}
                      for s in SUFFIXES}}
        reader = tf.train.NewCheckpointReader(str(prefix))
        shapes, dtypes = reader.get_variable_to_shape_map(), reader.get_variable_to_dtype_map()
        classes = collections.Counter(_classify(n) for n in shapes)
        moving = {}
        for n in sorted(shapes):
            if _classify(n) == 'G_bn_moving':
                v = reader.get_tensor(n)
                moving[n] = {'shape': list(v.shape), 'finite': bool(np.isfinite(v).all()),
                             'min': float(v.min()), 'max': float(v.max()), 'mean': float(v.mean()),
                             'sha256': hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()}
        doc['checkpoint_variables'][str(seed)] = {
            'count': len(shapes), 'classes': dict(sorted(classes.items())),
            'shapes': {n: list(shapes[n]) for n in sorted(shapes)},
            'dtypes': {n: dtypes[n].name for n in sorted(shapes)},
            'global_iteration': int(reader.get_tensor('global_iteration')),
            'minibatch_phase': int(reader.get_tensor('minibatch_phase')),
            'G_bn_moving': moving}
    # ---- graph-only analysis of the frozen M6D3e construction
    graph = tf.Graph()
    with graph.as_default():
        tf.set_random_seed(0)
        m = TrainingGraph(tf, repo)
        target = m.spoof.op
        anc = _ancestors(target)
        ops = list(anc.values())
        placeholders = sorted(o.name for o in ops if o.type == 'Placeholder')
        variables = sorted(o.name for o in ops if o.type in ('VariableV2', 'VarHandleOp'))
        bn = [o for o in ops if o.type.startswith('FusedBatchNorm')]
        addn = sorted(o.name for o in ops if o.type == 'AddN')
        g_vars = sorted(v.op.name for v in tf.global_variables() if v.op.name.startswith('G/'))

        def update_targets(updates):
            ops_ = [getattr(x, 'op', x) for x in updates]        # collection items may be tensors or ops
            return sorted({_variable_of(op.inputs[0]) for op in ops_ if op.inputs}, key=str)
        doc['graph'] = {
            'construction_ops': len(graph.get_operations()),
            'synthesized_spoof_tensor': m.spoof.name,
            'synthesized_spoof_shape': m.spoof.shape.as_list(),
            'ancestor_ops': len(ops),
            'ancestor_placeholders': placeholders,
            'ancestor_variables': variables,
            'ancestor_variables_outside_G': [v for v in variables if not v.startswith('G/')],
            'ancestor_moving_statistics': [v for v in variables if v.endswith(('moving_mean', 'moving_variance'))],
            'ancestor_fused_batch_norm': len(bn),
            'ancestor_fused_batch_norm_is_training': sorted({bool(o.get_attr('is_training')) for o in bn}),
            'ancestor_addn_ops': addn,
            'T_A_is_ancestor': 'T_A' in anc,
            'reconstruction_T_A_op_type': graph.get_operation_by_name('T_A').type,
            'ancestor_op_types': dict(sorted(collections.Counter(o.type for o in ops).items())),
            'ancestor_ops_requesting_cpu': sum(1 for o in ops if '/cpu:0' in o.device.lower() or
                                               '/device:cpu:0' in o.device.lower()),
            'generator_variables': len(g_vars),
            'generator_variable_names': g_vars,
            'bn_update_targets_original': update_targets(m.bn_original),
            'bn_update_targets_hard': update_targets(m.bn_hard),
            'bn_update_targets_D': update_targets(m.bn_D),
            'bn_update_counts': {'original': len(m.bn_original), 'hard': len(m.bn_hard), 'D': len(m.bn_D)},
            'hard_pass_input_placeholders': sorted(o.name for o in _ancestors(m.hard_rgb.op).values()
                                                   if o.type == 'Placeholder')}
        doc['graph']['generator_variable_shapes'] = {
            v.op.name: v.shape.as_list() for v in tf.global_variables() if v.op.name.startswith('G/')}
    # ---- frozen data identities (manifests only; no face is read)
    doc['feed_pack_manifest_sha256'] = sha_file(pio.pack_root(rt) / 'pack_manifest.json')
    pre = sorted(p.name for p in pio.prepass_root(rt).iterdir())
    doc['prepass_entries'] = pre
    doc['prepass_manifest_candidates'] = {n: sha_file(pio.prepass_root(rt) / n) for n in pre if n.endswith('.json')}
    print(json.dumps(doc, sort_keys=True))


# ====================================================================== laptop build
GATES = []


def gate(gid, ok, detail):
    GATES.append({'id': gid, 'pass': bool(ok), 'detail': detail})


def build(repo, collected_path, out_root=None):
    repo = Path(repo)
    sys.path.insert(0, str(repo))
    from methods.physics_std import production_io as pio
    raw = Path(collected_path).read_bytes()
    c = json.loads(raw)
    del GATES[:]
    with open(str(repo / INVENTORY)) as f:
        inv = list(csv.DictReader(f))
    with open(str(repo / FINAL_CSV)) as f:
        final = {int(r['seed']): r for r in csv.DictReader(f)}
    closure = json.loads((repo / CLOSURE).read_text())
    gate('H.cpu_only_clean_clone', c['cpu_only'] and c['clean'] and c['python'].startswith('3.8') and
         c['tensorflow'].startswith('1.15'), {k: c[k] for k in ('cpu_only', 'clean', 'head', 'python', 'tensorflow')})
    gate('C.closure_status', closure['status'] == 'SCIENTIFIC_TRAINING_COMPLETED_3_OF_3_SEEDS' and
         closure.get('failed_gates') == [], closure['status'])
    ck = {}
    for seed in SEEDS:
        want = {r['file'].split('/')[-1]: (int(r['bytes']), r['sha256']) for r in inv
                if int(r['seed']) == seed and int(r['iteration']) == 150000}
        got = {n: (v['bytes'], v['sha256']) for n, v in c['checkpoints'][str(seed)]['files'].items()}
        gate('C.final_checkpoint_bytes_seed%d' % seed, len(want) == 3 and got == want, got)
        f = final[seed]
        gate('C.final_checkpoint_table_seed%d' % seed,
             (f['final_data_sha256'], f['final_index_sha256'], f['final_meta_sha256']) ==
             (got[FINAL + SUFFIXES[0]][1], got[FINAL + SUFFIXES[1]][1], got[FINAL + SUFFIXES[2]][1]) and
             f['final_checkpoint_prefix'] == 'checkpoints/' + FINAL and f['completion_status'] == 'completed',
             f['run_id'])
        v = c['checkpoint_variables'][str(seed)]
        gate('C.final_state_counters_seed%d' % seed, v['global_iteration'] == 150000 and v['minibatch_phase'] == 0,
             [v['global_iteration'], v['minibatch_phase']])
        gate('C.no_ema_shadow_seed%d' % seed, v['classes'].get('ema_shadow', 0) == 0, v['classes'])
        mv = v['G_bn_moving']
        gate('C.generator_bn_moving_statistics_finite_seed%d' % seed,
             len(mv) > 0 and all(x['finite'] for x in mv.values()) and
             all(x['min'] > 0 for n, x in mv.items() if n.endswith('moving_variance')), len(mv))
        ck[str(seed)] = {'run_id': f['run_id'], 'prefix': 'checkpoints/' + FINAL,
                         'files': {n: {'bytes': b, 'sha256': s} for n, (b, s) in sorted(got.items())},
                         'checkpoint_identity_sha256': hashlib.sha256(json.dumps(
                             {n: s for n, (b, s) in sorted(got.items())}, sort_keys=True).encode()).hexdigest(),
                         'variables': v['count'], 'classes': v['classes'],
                         'generator_bn_moving_variables': len(mv),
                         'moving_variance_min': min(x['min'] for n, x in mv.items() if n.endswith('moving_variance')),
                         'moving_mean_abs_max': max(max(abs(x['min']), abs(x['max'])) for n, x in mv.items()
                                                    if n.endswith('moving_mean'))}
    g = c['graph']
    g_shapes = g['generator_variable_shapes']
    for seed in SEEDS:
        shapes = c['checkpoint_variables'][str(seed)]['shapes']
        gate('G.generator_variables_in_checkpoint_seed%d' % seed,
             all(n in shapes and shapes[n] == s for n, s in g_shapes.items()), len(g_shapes))
    gate('G.synthesized_spoof_inputs', g['ancestor_placeholders'] == ['sampling_offsets', 'synthetic_rgb'],
         g['ancestor_placeholders'])
    gate('G.synthesized_spoof_generator_only', g['ancestor_variables_outside_G'] == [] and
         g['ancestor_moving_statistics'] == [], {'outside_G': g['ancestor_variables_outside_G'],
                                                 'moving': g['ancestor_moving_statistics']})
    gate('G.frozen_graph_bn_uses_batch_statistics', g['ancestor_fused_batch_norm'] > 0 and
         g['ancestor_fused_batch_norm_is_training'] == [True], g['ancestor_fused_batch_norm_is_training'])
    gate('G.synthesized_spoof_independent_of_T_A_AddN', g['T_A_is_ancestor'] is False and
         g['reconstruction_T_A_op_type'] == 'AddN', {'T_A_ancestor': g['T_A_is_ancestor'],
                                                     'ancestor_addn_ops': g['ancestor_addn_ops']})
    gate('G.bn_moving_statistics_shared_original_and_hard',
         g['bn_update_targets_original'] == g['bn_update_targets_hard'] and
         all(n.startswith('G/') for n in g['bn_update_targets_original']) and None not in g['bn_update_targets_original'],
         {'original': len(g['bn_update_targets_original']), 'hard': len(g['bn_update_targets_hard'])})
    gate('D.feed_pack_manifest', c['feed_pack_manifest_sha256'] == pio.PACK_MANIFEST_SHA256,
         c['feed_pack_manifest_sha256'])
    gate('D.prepass_manifest', pio.PREPASS_MANIFEST_SHA256 in c['prepass_manifest_candidates'].values(),
         c['prepass_manifest_candidates'])
    failed = [x['id'] for x in GATES if not x['pass']]
    doc = {'record': 'E04_BANK_CONTRACT_EVIDENCE', 'schema': 'e04_bank_contract_evidence_v1',
           'status': 'E04_BANK_CONTRACT_EVIDENCE_COLLECTED' if not failed else 'E04_BANK_CONTRACT_EVIDENCE_FAILED',
           'failed_gates': failed, 'gates': GATES[:], 'collected': {'path': COLLECTED,
                                                                     'sha256': hashlib.sha256(raw).hexdigest()},
           'probe_tool': {'path': 'tools/e04_bank_contract_probe.py'},
           'host': {k: c[k] for k in ('utc', 'head', 'clean', 'cpu_only', 'python', 'tensorflow', 'numpy', 'executable')},
           'scientific_authority': {'training_commit': closure.get('training_authority', final[42]['scientific_authority']),
                                    'closure_record': CLOSURE,
                                    'closure_sha256': hashlib.sha256((repo / CLOSURE).read_bytes()).hexdigest(),
                                    'checkpoint_rule': 'BASELINE_FINAL_STATE_V1 iteration_150000 within seed'},
           'final_checkpoints': ck,
           'graph_facts': {k: g[k] for k in (
               'construction_ops', 'synthesized_spoof_tensor', 'synthesized_spoof_shape', 'ancestor_ops',
               'ancestor_placeholders', 'ancestor_fused_batch_norm', 'ancestor_fused_batch_norm_is_training',
               'ancestor_addn_ops', 'T_A_is_ancestor', 'ancestor_ops_requesting_cpu', 'ancestor_op_types',
               'generator_variables', 'bn_update_counts', 'hard_pass_input_placeholders')},
           'data_identities': {'feed_pack_manifest_sha256': c['feed_pack_manifest_sha256'],
                               'prepass_manifest_sha256': pio.PREPASS_MANIFEST_SHA256},
           'scope': 'read-only CPU evidence: no Session on data, no face read, no checkpoint restored into a graph, '
                    'no GPU, no VAL/TEST, no write outside this record'}
    out = Path(out_root or repo) / OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    Path(str(out) + '.json').write_text(dumps(doc))
    Path(str(out) + '.md').write_text(report(doc))
    Path(str(out) + '_SHA256.txt').write_text(''.join(
        '%s  %s\n' % (sha_file(str(out) + ext), Path(str(out) + ext).name) for ext in ('.json', '.md')))
    return doc


def report(doc):
    L = ['# E04 bank-contract evidence (read-only CPU probe)', '',
         '**Status:** `%s` (%d/%d gates).' % (doc['status'], sum(g['pass'] for g in doc['gates']), len(doc['gates'])),
         '', 'Host clone `%s` (clean %s), Python %s, TensorFlow %s, CPU only %s, %s.' % (
             doc['host']['head'][:12], doc['host']['clean'], doc['host']['python'], doc['host']['tensorflow'],
             doc['host']['cpu_only'], doc['host']['utc']), '', '## Final checkpoints (BASELINE_FINAL_STATE_V1)', '',
         '| seed | run_id | file | bytes | sha256 |', '|---|---|---|---|---|']
    for s, v in sorted(doc['final_checkpoints'].items(), key=lambda x: int(x[0])):
        for n, f in v['files'].items():
            L.append('| %s | `%s` | `%s` | %d | `%s` |' % (s, v['run_id'], n, f['bytes'], f['sha256']))
    L += ['', '| seed | variables | classes | G BN moving stats | min moving_variance | max abs moving_mean |',
          '|---|---|---|---|---|---|']
    for s, v in sorted(doc['final_checkpoints'].items(), key=lambda x: int(x[0])):
        L.append('| %s | %d | %s | %d | %.3g | %.3g |' % (s, v['variables'], ', '.join(
            '%s %d' % kv for kv in sorted(v['classes'].items())), v['generator_bn_moving_variables'],
            v['moving_variance_min'], v['moving_mean_abs_max']))
    g = doc['graph_facts']
    L += ['', '## Frozen M6D3e graph: the `synthesized_spoof` tensor', '',
          '- Tensor `%s`, shape %s; %d ancestor ops; inputs only %s.' % (
              g['synthesized_spoof_tensor'], g['synthesized_spoof_shape'], g['ancestor_ops'],
              ', '.join('`%s`' % p for p in g['ancestor_placeholders'])),
          '- %d FusedBatchNorm ancestors, `is_training` = %s: the frozen graph normalises with **batch statistics** '
          'of the 8-row batch (4 live targets + 4 spoof sources); no moving statistic is read.' % (
              g['ancestor_fused_batch_norm'], g['ancestor_fused_batch_norm_is_training']),
          '- `T_A` (the AddN that D12 found non-bitwise on GPU) is an ancestor: %s. AddN ancestors: %s.' % (
              g['T_A_is_ancestor'], g['ancestor_addn_ops'] or 'none'),
          '- The original-pass and hard-pass BN update ops assign the same %d generator moving statistics '
          '(update counts %s).' % (g['bn_update_counts']['original'], g['bn_update_counts']),
          '- %d ancestor ops request `/cpu:0` (bilinear resize and warp, as qualified).' % g['ancestor_ops_requesting_cpu'],
          '', '## Gates', '', '| gate | pass |', '|---|---|']
    L += ['| `%s` | %s |' % (x['id'], 'PASS' if x['pass'] else '**FAIL**') for x in doc['gates']]
    L += ['', 'Scope: %s.' % doc['scope'], '']
    return '\n'.join(L)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    h = sub.add_parser('host')
    h.add_argument('--runtime-root', required=True)
    b = sub.add_parser('build')
    b.add_argument('--repo', default=str(Path(__file__).resolve().parents[1]))
    b.add_argument('--collected', default=None)
    a = ap.parse_args()
    if a.cmd == 'host':
        host(a.runtime_root)
    else:
        doc = build(a.repo, a.collected or str(Path(a.repo) / COLLECTED))
        print(doc['status'], doc['failed_gates'])


if __name__ == '__main__':
    main()
