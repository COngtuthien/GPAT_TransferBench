"""E03 STDN production qualification (QUALIFICATION seed 60903 only; never a scientific result).

Run on the GPU host, gpat-m6-e03, M6D2b precision environment, PYTHONHASHSEED=60903, from the repo root:
  python -u -B -m methods.stdn.production_qualification --case <case> --output /tmp/gpat_e03_prodq/<case>.json

Cases (one clean process each; the SAME production components as tools/run_e03.py -- prepare(), Controller,
run_plan(), write_checkpoint(), E03RunContext -- bound to <rt>/qualification/e03_production/E03/<case>-q60903-<id>):
  val_a, val_b    real TRAIN 201 steps (even G+D / odd G steps, figures at TRAIN steps 0 and 200) -> ckpt-1 ->
                  real VAL 51 steps (figures at VAL steps 0 and 50) -> 8 more TRAIN steps (epoch 2). Two identical
                  processes = the determinism baseline.
  no_val          identical, but the VAL block is skipped (control for VAL non-influence).
  terminal_probe  2 TRAIN steps, then the official saver writes a qualification-only ckpt-50 with the selected
                  semantics (labelled TERMINAL SEMANTICS PROBE; it is NOT a 50-epoch STDN checkpoint).
  failure_probe   1 TRAIN step, then an injected failure: the run closes 'failed', writes no checkpoint, and a second
                  open of the same root is refused (fresh only, no automatic resume).
  predict         TF-only index pipelines with the production seeds (no image, graph or GPU): the independent
                  prediction of the TRAIN/VAL shuffle order over two full passes + NumPy RandomState flip draws.
An audit hook records every file open: canonical faces outside the feed pack, manifests, runs/ and any runtime
write outside the qualification root are refused. Session runs are counted per kind through a SessionInterface proxy.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from methods.stdn import production_io as pio

require = pio.require
RUNTIME = Path('/home/student20261/workdir/GPAT_TransferBench_runtime')
FACES = RUNTIME / 'data' / 'processed' / 'faces_256'
PLANS = {'val_a': ((201, 'checkpoint', 51), (8, None, 0)), 'val_b': ((201, 'checkpoint', 51), (8, None, 0)),
         'no_val': ((201, 'checkpoint', 0), (8, None, 0)), 'terminal_probe': ((2, 'probe50', 0),),
         'failure_probe': ((2, None, 0),)}


class Firewall:
    def __init__(self):
        self.faces = {}            # sample_id -> open count
        self.denied, self.writes, self.pack_ids, self.qual_root = [], [], None, None

    def __call__(self, event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes)):
            return
        p = Path(os.fsdecode(args[0])).absolute()
        s = str(p)
        flags = args[2] if isinstance(args[2], int) else 0
        write = flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
        mode = args[1] if isinstance(args[1], str) else ''
        write = write or any(c in mode for c in 'wax+')
        bad = None
        if s.startswith(str(FACES) + '/'):
            sid = p.stem
            if self.pack_ids is None or sid not in self.pack_ids:
                bad = 'face outside the feed pack'
            else:
                self.faces[sid] = self.faces.get(sid, 0) + 1
        elif '/manifests/' in s and 'source_cache' not in s:
            bad = 'manifest read'
        elif s.startswith(str(RUNTIME / 'runs') + '/'):
            bad = 'runs/ access'
        elif write and s.startswith(str(RUNTIME) + '/') and not (self.qual_root and s.startswith(str(self.qual_root.parent) + '/')):
            bad = 'runtime write outside the qualification root'
        if write:
            self.writes.append(s)
        if bad:
            self.denied.append([bad, s])
            raise PermissionError('E03 qualification firewall: %s: %s' % (bad, s))


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def predict(seed, steps):
    """Independent order/flip predictor: only tf.data index pipelines + numpy RandomState (no controller code)."""
    import tensorflow as tf
    tf.set_random_seed(seed)
    plan = pio.rng_plan(seed)
    out = {}
    for split, (path, digest, n) in pio.PAIR_MANIFESTS.items():
        ds = tf.data.Dataset.from_tensor_slices(np.arange(n, dtype=np.int64))
        ds = ds.shuffle(n, seed=plan[split]['shuffle_seed'], reshuffle_each_iteration=True).repeat(-1).batch(2)
        nxt = ds.make_one_shot_iterator().get_next()
        out[split] = {'nxt': nxt, 'n': n}
    res = {}
    with tf.Session(config=tf.ConfigProto(device_count={'GPU': 0})) as sess:
        for split, d in out.items():
            n = d['n']
            batches = (2 * n) // 2 + 1
            seq = np.concatenate([sess.run(d['nxt']) for _ in range(batches)]).tolist()
            p1, p2 = seq[:n], seq[n:2 * n]
            rng = np.random.RandomState(plan[split]['parser_seed'])
            k = 2 * steps[split]
            flips = [[int(rng.rand() > 0.5), int(rng.rand() > 0.5)] for _ in range(k)]
            res[split] = {'order': seq[:k], 'flips': flips, 'pass1_is_permutation': sorted(p1) == list(range(n)),
                          'pass2_is_permutation': sorted(p2) == list(range(n)), 'reshuffled_each_pass': p1 != p2,
                          'pass1_sha256': sha(json.dumps(p1).encode()), 'pass2_sha256': sha(json.dumps(p2).encode()),
                          'shuffle_seed': plan[split]['shuffle_seed'], 'parser_seed': plan[split]['parser_seed']}
    return res


def wait_quiescent(parser, timeout=120.0, settle=1.5):
    """prefetch(1)/map(1) keep parsing in the background after a pull; wait until the parser is idle."""
    last, since, start = parser.calls, time.monotonic(), time.monotonic()
    while time.monotonic() - start < timeout:
        time.sleep(0.1)
        if parser.calls != last:
            last, since = parser.calls, time.monotonic()
        elif time.monotonic() - since >= settle:
            return last
    raise pio.E03Stop('parser did not become idle')


def qualify(case):
    import tensorflow as tf
    from tensorflow.python.client import session as tf_session
    from methods.stdn import production_runner as pr
    from methods.common.runlog import RunDirectoryError
    fw = Firewall()
    sys.addaudithook(fw)
    errors = pio.launch_environment_errors(os.environ, pio.QUALIFICATION_SEED)
    require(not errors, 'qualification launch environment: ' + ', '.join(errors))
    ctl, ids, live, driver, env, reasons = pr.prepare(tf, pio.QUALIFICATION, pio.QUALIFICATION_SEED, RUNTIME, FACES,
                                                      keep_parser_log=True)
    fw.pack_ids = set(ctl.pack.samples)
    ctx = pr.E03RunContext(mode=pio.QUALIFICATION, seed=pio.QUALIFICATION_SEED, runtime_root=RUNTIME, environment=env,
                           missing_environment_reasons=reasons, identities=ids, case=case,
                           command_line=' '.join([sys.executable] + sys.argv))
    fw.qual_root = ctx.run_dir
    require(not ctx.run_dir.exists() and 'runs' not in ctx.run_dir.relative_to(RUNTIME).parts, 'fresh qual root')

    class CountingSession(tf_session.SessionInterface):
        def __init__(self, inner):
            self.inner, self.counts = inner, {}

        @property
        def graph(self):
            return self.inner.graph

        @property
        def sess_str(self):
            return self.inner.sess_str

        def partial_run_setup(self, *a, **k):
            raise RuntimeError('not used')

        def partial_run(self, *a, **k):
            raise RuntimeError('not used')

        def run(self, fetches, *a, **k):
            flat = fetches if isinstance(fetches, (list, tuple)) else [fetches]
            names = {getattr(f, 'name', str(f)) for f in flat}
            kind = ('train_gd' if ctl.d_op.name in names else 'train_g' if ctl.g_op.name in names else
                    'val' if ctl.losses_val[0].name in names else 'other')
            self.counts[kind] = self.counts.get(kind, 0) + 1
            return self.inner.run(fetches, *a, **k)

    result = {'case': case, 'qualification_seed': pio.QUALIFICATION_SEED, 'experiment_seed': None,
              'plan': PLANS[case], 'run_id': ctx.run_id, 'run_dir': str(ctx.run_dir), 'identities': ids,
              'structure': ctl.structure, 'driver': driver, 'seeding': ctl.seeding,
              'live_environment': {k: v for k, v in live.items() if k != 'packages'},
              'live_packages_sha256': sha(json.dumps(live['packages'], sort_keys=True).encode()),
              'pack': {'manifest_sha256': ctl.pack.manifest_sha256, 'counts': ctl.pack.manifest['counts']},
              'points': {}, 'steps': [], 'scientific_execution': False}
    tp, vp = ctl.parsers['TRAIN'], ctl.parsers['VAL']
    gvars = {v.name: v for v in ctl.variables}
    beta = {k: gvars[k] for k in ('beta1_power:0', 'beta1_power_1:0')}

    def classify(name):
        return ('global_step' if name.startswith('global_step') else
                'adam_slot' if '/Adam' in name or 'beta1_power' in name or 'beta2_power' in name else
                'weight_ema' if 'ExponentialMovingAverage' in name else
                'loss_ema' if name.endswith('/avg:0') else 'bn_moving' if '/moving_' in name else
                'trainable_G' if name.startswith('STDN/') else 'trainable_D' if name.startswith('Disc/') else 'other')
    last = {}

    def on_step(kind, e, step, sess, gd):
        if kind == 'train' and e == 0 and step < 4:
            fp_ = ctl.fingerprint(sess)['per_variable']
            prev = last.get('fp')
            changed = {}
            if prev is not None:
                for n, h in fp_.items():
                    if prev[n] != h:
                        changed[classify(n)] = changed.get(classify(n), 0) + 1
            last['fp'] = fp_
            b = sess.run(list(beta.values()))
            result['steps'].append({'epoch': e, 'step': step, 'gd': gd, 'changed_by_class_vs_previous_step': changed,
                                    'beta1_power': dict(zip(beta, [float(x) for x in b])),
                                    'session_counts': dict(scounter.counts)})
        if case == 'failure_probe' and kind == 'train' and step == 1:
            raise pio.E03Stop('INJECTED QUALIFICATION FAILURE (failure_probe)')

    def on_point(point, e, sess):
        if e != 0:
            return
        calls_t = wait_quiescent(tp)
        calls_v = wait_quiescent(vp)
        f = ctl.fingerprint(sess)
        result['points'][point] = {
            'fingerprint_sha256': f['sha256'], 'per_variable': f['per_variable'], 'all_finite': f['all_finite'],
            'counters': ctl.counters(sess), 'train_parser': {'calls': calls_t, 'draws': tp.draws,
                                                             'rng_state_sha256': tp.rng_state_sha256()},
            'val_parser': {'calls': calls_v, 'draws': vp.draws, 'rng_state_sha256': vp.rng_state_sha256()},
            'session_counts': dict(scounter.counts), 'face_reads': dict(ctl.reader.reads)}

    with tf.Session(config=pr.session_config(tf)) as raw_sess:
        scounter = CountingSession(raw_sess)
        failure = None
        try:
            with ctx:
                raw_sess.run(tf.initializers.global_variables())
                result['points']['init'] = {'fingerprint_sha256': ctl.fingerprint(raw_sess)['sha256']}
                out = pr.run_plan(ctx, ctl, scounter, PLANS[case], on_point=on_point, on_step=on_step)
                calls_t = wait_quiescent(tp)
                f = ctl.fingerprint(raw_sess)
                result['final'] = {'fingerprint_sha256': f['sha256'], 'per_variable': f['per_variable'],
                                   'all_finite': f['all_finite'], 'counters': ctl.counters(raw_sess),
                                   'beta1_power': dict(zip(beta, [float(x) for x in raw_sess.run(list(beta.values()))])),
                                   'train_parser': {'calls': calls_t, 'draws': tp.draws,
                                                    'rng_state_sha256': tp.rng_state_sha256()},
                                   'applications': out['applications'], 'global_step': out['global_step'],
                                   'train_steps': out['train_steps'], 'seconds': out['seconds']}
                result['checkpoints'] = out['checkpoints']
                ctx.close(completion_status='interrupted', summary={
                    'missing_field_reasons': {'final_or_selected_checkpoint_path': 'QUALIFICATION_ONLY: no scientific '
                                                                                   'selection exists'}})
        except pio.E03Stop as exc:
            if case != 'failure_probe':
                raise
            failure = str(exc)
    result['session_counts'] = scounter.counts
    result['parser_log'] = {'TRAIN': tp.log, 'VAL': vp.log}
    result['face_reads'] = ctl.reader.reads
    root = ctx.run_dir
    if case == 'failure_probe':
        summ = json.loads((root / 'run_summary.json').read_text())
        refused = None
        again = pr.E03RunContext(mode=pio.QUALIFICATION, seed=pio.QUALIFICATION_SEED, runtime_root=RUNTIME,
                                 environment=env, missing_environment_reasons=reasons, identities=ids, case=case)
        try:
            again.open()
        except RunDirectoryError as exc:
            refused = str(exc)
        result['failure'] = {'injected': failure, 'completion_status': summ['completion_status'],
                             'failure_reason': summ['failure_reason'],
                             'checkpoint_files': sorted(p.name for p in (root / 'checkpoints').iterdir()),
                             'partial_files': sorted(str(p) for p in root.rglob('*.partial')),
                             'second_open_refused': refused}
    files = sorted(p for p in root.rglob('*') if p.is_file())
    result['root_listing'] = [[str(p.relative_to(root)), p.stat().st_size, pio.sha256_file(p)] for p in files]
    result['metrics_records'] = [json.loads(x) for x in (root / 'metrics.jsonl').read_text().splitlines()]
    result['checkpoint_index'] = json.loads((root / 'checkpoint_index.json').read_text())
    result['run_manifest'] = json.loads((root / 'run_manifest.json').read_text())
    result['run_summary'] = json.loads((root / 'run_summary.json').read_text())
    result['access'] = {'face_opens': len(fw.faces), 'face_open_events': sum(fw.faces.values()),
                        'faces_by_split': {s: sum(1 for k in fw.faces if ctl.pack.samples[k]['split'] == s)
                                           for s in ('TRAIN', 'VAL')},
                        'faces_sha256': sha(json.dumps(sorted(fw.faces)).encode()),
                        'denied': fw.denied, 'runtime_writes_outside_root': sorted(
                            {w for w in fw.writes if w.startswith(str(RUNTIME) + '/') and not w.startswith(str(root))
                             and w != str(root.parent / ('.%s.lock' % root.name))})}
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', required=True, choices=sorted(PLANS) + ['predict'])
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    out = Path(args.output)
    require(str(out.resolve()).startswith('/tmp/') and not out.exists(), 'fresh /tmp output')
    if args.case == 'predict':
        result = {'case': 'predict', 'seed': pio.QUALIFICATION_SEED,
                  'prediction': predict(pio.QUALIFICATION_SEED, {'TRAIN': 209, 'VAL': 51})}
    else:
        result = qualify(args.case)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, sort_keys=True))
    print('E03_QUALIFICATION_CASE_DONE', args.case)
    return 0


if __name__ == '__main__':
    sys.exit(main())
