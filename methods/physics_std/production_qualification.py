"""E04 Physics-STD production qualification (QUALIFICATION seed 60904 only; never a scientific result).

GPU host, gpat-m6-e03, E04 training precision environment, PYTHONHASHSEED=60904, from the repository root:
  python -u -B -m methods.physics_std.production_qualification --case <case> --output /tmp/gpat_e04_prodq/<case>.json

Cases (one clean process each; the SAME components as tools/run_e04.py -- prepare(), Controller, run_plan(),
write_checkpoint(), E04RunContext -- bound to <rt>/qualification/e04_production/E04/<case>-q60904-<run_id>):
  det_a, det_b    real TRAIN, 20 iterations, checkpoints after iterations 10 and 20 (periodic semantics). At iteration
                  0 every step is observed: per-variable state before/after (changed trainable / non-trainable sets
                  compared with the committed M6D3e step records), and a separate pure loss fetch before each step
                  (must not mutate state and must equal the losses fetched in the apply run). det_a also builds the
                  qualified M6D3e worker construction (seed 314159) in a separate graph and compares the
                  seed-attribute-free construction signature with the production one.
  terminal_probe  2 iterations, then a qualification-only checkpoint written with the iteration_150000 selected
                  semantics (labelled TERMINAL SEMANTICS PROBE; NOT a 150000-iteration checkpoint).
  failure_probe   1 iteration, then an injected failure: the run closes 'failed' with no checkpoint, and a second open
                  of the same root is refused (fresh only).
An audit hook refuses canonical faces outside the feed pack, manifests, runs/ and runtime writes outside the
qualification root.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from methods.physics_std import production_io as pio

require = pio.require
RUNTIME = Path('/home/student20261/workdir/GPAT_TransferBench_runtime')
FACES = RUNTIME / 'data' / 'processed' / 'faces_256'
PLANS = {'det_a': (20, [10, 20], False, None), 'det_b': (20, [10, 20], False, None),
         'terminal_probe': (2, [2], True, None), 'failure_probe': (2, [], False, 1)}
M6D3E_PROCESS = 'outputs/audit/M6D3E_E04_SYNTHETIC_PROCESS_1.json'


class Firewall:
    def __init__(self):
        self.faces, self.denied, self.writes, self.pack_ids, self.qual_root, self.out = {}, [], [], None, None, None

    def __call__(self, event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes)):
            return
        p = Path(os.fsdecode(args[0])).absolute()
        s = str(p)
        flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
        mode = args[1] if isinstance(args[1], str) else ''
        write = bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)) or \
            any(c in mode for c in 'wax+')
        bad = None
        if s.startswith(str(FACES) + '/'):
            if self.pack_ids is None or p.stem not in self.pack_ids:
                bad = 'face outside the feed pack'
            else:
                self.faces[p.stem] = self.faces.get(p.stem, 0) + 1
        elif '/manifests/' in s and 'source_cache' not in s:
            bad = 'manifest read'
        elif s.startswith(str(RUNTIME / 'runs') + '/'):
            bad = 'runs/ access'
        elif write and s.startswith(str(RUNTIME) + '/') and not (
                self.qual_root and s.startswith(str(self.qual_root.parent) + '/')):
            bad = 'runtime write outside the qualification root'
        if write:
            self.writes.append(s)
        if bad:
            self.denied.append([bad, s])
            raise PermissionError('E04 qualification firewall: %s: %s' % (bad, s))


def reference_construction(tf):
    """The qualified M6D3e worker construction (training_runtime.qualify: seed 314159, /gpu:0) in its own graph."""
    from methods.physics_std.production_runner import construction_signature
    from methods.physics_std.training_graph import TrainingGraph
    graph = tf.Graph()
    with graph.as_default(), tf.device('/gpu:0'):
        tf.set_random_seed(314159)
        TrainingGraph(tf, pio.ROOT)
    return construction_signature(graph)


def qualify(case, output):
    os.environ.setdefault('PYTHONHASHSEED', str(pio.QUALIFICATION_SEED))
    errors = pio.launch_environment_errors(os.environ, pio.QUALIFICATION_SEED)
    require(not errors, 'launch environment: ' + ', '.join(errors))
    import tensorflow as tf
    from methods.physics_std import production_runner as pr
    fw = Firewall()
    ctl, ids, live, driver, env, reasons = pr.prepare(tf, pio.QUALIFICATION, pio.QUALIFICATION_SEED, RUNTIME, FACES)
    fw.pack_ids = set(ctl.pack.samples)
    ctx = pr.E04RunContext(mode=pio.QUALIFICATION, seed=pio.QUALIFICATION_SEED, runtime_root=RUNTIME, environment=env,
                           missing_environment_reasons=reasons, identities=ids, case=case,
                           command_line=' '.join([sys.executable] + sys.argv))
    fw.qual_root = ctx.run_dir
    n, ckpts, probe, fail_after = PLANS[case]
    ref = json.loads((pio.ROOT / M6D3E_PROCESS).read_text())
    observed = {'steps': [], 'pure_losses': {}, 'pure_fetch_mutated': []}
    tnames = {v.name for v in ctl.trainables}

    def on_step(t, phase, step, sess, fd):
        if t != 0:
            return
        if phase == 'before':
            before = ctl.fingerprint(sess)
            keys = pio.STEP_LOSSES[step]
            vals = sess.run([ctl.model.losses[k] for k in keys], fd)
            observed['pure_losses']['step%d' % step] = {k: float(v) for k, v in zip(keys, vals)}
            again = ctl.fingerprint(sess)
            if again['sha256'] != before['sha256']:
                observed['pure_fetch_mutated'].append(step)
            observed['_before'] = before
            observed['_counters_before'] = ctl.counters(sess)
        else:
            after = ctl.fingerprint(sess)
            b = observed.pop('_before')['per_variable']
            changed = sorted(k for k in b if b[k] != after['per_variable'][k])
            c0, c1 = observed.pop('_counters_before'), ctl.counters(sess)
            rs = ref['steps'][step - 1]
            observed['steps'].append({
                'step': step, 'changed_trainables': sorted(set(changed) & tnames),
                'changed_nontrainables': sorted(set(changed) - tnames),
                'equals_m6d3e_changed_trainables': sorted(set(changed) & tnames) == rs['changed_trainables'],
                'equals_m6d3e_changed_nontrainables': sorted(set(changed) - tnames) == rs['changed_nontrainables'],
                'iteration_before_after': [c0['iteration'], c1['iteration']], 'phase_after': c1['phase'],
                'expected_phase_after': 0 if step == 3 else step, 'all_finite': after['all_finite']})

    per_iteration = []

    def on_iteration(t, sess, info, losses):
        per_iteration.append({'iteration': t, 'pair_rows': info['pair_rows'], 'pair_passes': info['pair_passes'],
                              'hard_scales': info['hard_scales'], 'hard_keep': info['hard_keep'],
                              'input_sha256': info['input_sha256'], 'losses': losses})

    result = {'case': case, 'qualification_seed': pio.QUALIFICATION_SEED, 'experiment_seed': None,
              'plan': {'iterations': n, 'checkpoints': ckpts, 'terminal_probe': probe, 'fail_after': fail_after},
              'identities': ids, 'driver': driver, 'structure': ctl.structure, 'rng_plan': pio.rng_plan(
                  pio.QUALIFICATION_SEED), 'run_dir': str(ctx.run_dir), 'run_id': ctx.run_id,
              'environment_live': {k: live[k] for k in ('python', 'tensorflow', 'native_runtime', 'gpu',
                                                        'precision_env', 'session_config')}}
    sys.addaudithook(fw)
    t0 = time.monotonic()
    status = None
    with ctx:
        ctx.log_event('e04_run_start', pr.run_start_payload(ctl, ids, driver, live))
        with tf.Session(config=pr.session_config(tf)) as sess:
            sess.run(tf.global_variables_initializer())
            result['initial_state_sha256'] = ctl.fingerprint(sess)['sha256']
            try:
                out = pr.run_plan(ctx, ctl, sess, n, ckpts, probe=probe, fail_after=fail_after,
                                  on_step=on_step if case in ('det_a', 'det_b') else None, on_iteration=on_iteration)
                result['final_state_sha256'] = ctl.fingerprint(sess)['sha256']
                result['run'] = out
                status = 'completed'
            except pio.E04Stop as exc:
                require(fail_after is not None and 'QUALIFICATION_INJECTED_FAILURE' in str(exc), 'unexpected stop: %s' % exc)
                result['injected_failure'] = str(exc)
                status = 'failed'
        ctx.close(completion_status=status, failure_reason=result.get('injected_failure'), summary={
            'qualification_case': case, 'labels': ctx.labels(), 'global_iteration': len(per_iteration),
            'missing_field_reasons': {'final_or_selected_checkpoint_path': 'QUALIFICATION_ONLY'}})
    result['status'] = status
    result['seconds'] = time.monotonic() - t0
    result['per_iteration'] = per_iteration
    result['observed_iteration0'] = observed
    if case == 'failure_probe':
        again = pr.E04RunContext(mode=pio.QUALIFICATION, seed=pio.QUALIFICATION_SEED, runtime_root=RUNTIME,
                                 environment=env, missing_environment_reasons=reasons, identities=ids, case=case)
        try:
            again.open()
            result['reopen'] = 'ACCEPTED'
        except Exception as exc:                      # RunDirectoryError: fresh only
            result['reopen'] = 'REFUSED: %s' % type(exc).__name__
        result['checkpoints_dir_listing'] = sorted(p.name for p in (ctx.run_dir / 'checkpoints').iterdir()) \
            if (ctx.run_dir / 'checkpoints').exists() else []
    if case == 'det_a':
        result['reference_construction'] = reference_construction(tf)
        result['production_construction'] = list(ctl.construction)
    root = ctx.run_dir
    result['run_root_listing'] = sorted([str(p.relative_to(root)), p.stat().st_size, pio.sha256_file(p)]
                                        for p in root.rglob('*') if p.is_file())
    result['run_summary'] = json.loads((root / 'run_summary.json').read_text())
    result['checkpoint_index'] = json.loads((root / 'checkpoint_index.json').read_text())
    result['firewall'] = {'denied': fw.denied, 'faces_read': len(fw.faces), 'face_opens': sum(fw.faces.values()),
                          'runtime_writes_outside_qualification': [w for w in fw.writes if w.startswith(str(RUNTIME))
                                                                   and not w.startswith(str(root.parent))]}
    Path(output).write_text(json.dumps(result, sort_keys=True) + '\n')
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', required=True, choices=sorted(PLANS))
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    require(os.path.abspath(args.output).startswith('/tmp/'), 'qualification output must be under /tmp')
    r = qualify(args.case, args.output)
    print(json.dumps({'case': r['case'], 'status': r['status'], 'seconds': r['seconds'], 'run_dir': r['run_dir'],
                      'denied': r['firewall']['denied']}))
    return 0 if not r['firewall']['denied'] else 1


if __name__ == '__main__':
    sys.exit(main())
