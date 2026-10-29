"""E04 real-data determinism probe (diagnostic only; qualification seed 60904; gpat-m6-e03, E04 precision env).

Builds the production Controller exactly as tools/run_e04.py would (prepare(): contract, D9 environment check, feed
pack, unchanged TrainingGraph) but opens NO RunContext and writes nothing except --output (under /tmp). For the real
TRAIN iteration-0 and iteration-1 batches of the D4/D7 streams it records:
  * per-loss gradients (G: L_depth, L_G, L_P, L_R, L_S, L_H; D: L_D), fetched twice in-process, per-variable sha256;
  * per-variable state sha256 after each of the three steps of iterations 0 and 1 (production order: BN updates, then
    apply);
so several clean processes can locate where cross-process / in-process non-bitwise behaviour originates.

    python -B tools/e04_realdata_determinism_probe.py --repo <checkout> --output /tmp/<dir>/probe_<k>.json
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True


def sha(a):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    if not os.path.abspath(args.output).startswith('/tmp/'):
        raise SystemExit('output must be under /tmp')
    sys.path.insert(0, str(Path(args.repo).resolve()))
    from methods.physics_std import production_io as pio
    errors = pio.launch_environment_errors(os.environ, pio.QUALIFICATION_SEED)
    if errors:
        raise SystemExit('launch environment: %s' % errors)
    import tensorflow as tf
    from methods.physics_std import production_runner as pr
    rt = Path('/home/student20261/workdir/GPAT_TransferBench_runtime')
    ctl, ids, live, driver, env, reasons = pr.prepare(tf, pio.QUALIFICATION, pio.QUALIFICATION_SEED, rt,
                                                      rt / 'data/processed/faces_256')
    m = ctl.model
    out = {'qualification_seed': pio.QUALIFICATION_SEED, 'identities': ids, 'construction': list(ctl.construction),
           'iterations': []}
    grads = {k: [g for g, _ in m.per_loss_gradients[k] if g is not None] for k in m.per_loss_gradients}
    gvars = {k: [v.name for g, v in m.per_loss_gradients[k] if g is not None] for k in m.per_loss_gradients}
    with tf.Session(config=pr.session_config(tf)) as sess:
        sess.run(tf.global_variables_initializer())
        out['initial_state_sha256'] = ctl.fingerprint(sess)['sha256']
        for t in range(2):
            arrays, info = ctl.feed.next()
            fd = ctl.feed_dict(arrays)
            rec = {'iteration': t, 'pair_rows': info['pair_rows'], 'input_sha256': info['input_sha256'], 'steps': []}
            before = ctl.fingerprint(sess)['sha256']
            rec['gradients'] = {}
            for k in sorted(grads):
                a = sess.run(grads[k], fd)
                b = sess.run(grads[k], fd)
                rec['gradients'][k] = {'per_variable_sha256': dict(zip(gvars[k], (sha(x) for x in a))),
                                       'in_process_repeat_differs': [n for n, x, y in zip(gvars[k], a, b)
                                                                     if sha(x) != sha(y)]}
            rec['gradient_fetch_mutated_state'] = ctl.fingerprint(sess)['sha256'] != before
            for step in (1, 2, 3):
                for updates in m.bn_paths[step]:
                    sess.run(updates, fd)
                sess.run(m.apply[step], fd)
                rec['steps'].append(ctl.fingerprint(sess)['per_variable'])
            out['iterations'].append(rec)
    Path(args.output).write_text(json.dumps(out, sort_keys=True) + '\n')
    print(json.dumps({'status': 'PASS', 'iterations': len(out['iterations'])}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
