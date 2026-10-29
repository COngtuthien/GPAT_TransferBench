"""E04 DRAFT D4/D7 RNG specification: reference implementation and test vectors (numpy only; Python 3.8+).

DRAFT -- proposed for owner approval in docs/spec/amendments/GPAT_TransferBench_v1_0_E04_Production_Owner_Resolution_
Addendum_DRAFT.md; not an E04 scientific authority and not used by any runner.

Seed derivation (domain-separated, never equal to the experiment seed's other uses):
    derive(domain, s) = 1 + (u32_be(SHA256("GPAT_TRANSFERBENCH|E04|<domain>|<s>")[0:4]) mod (2^31 - 1))
D4 pair order (domain PAIR_ORDER): one numpy.random.RandomState(derive('PAIR_ORDER', s)) (legacy MT19937, stream
    frozen by NumPy's compatibility policy); pass p = 0, 1, ... is rng.permutation(8838), the indices addressing rows of
    manifests/pairs_train_v1.parquet in stored order (sorted by pair_id). The passes are concatenated; iteration t
    (0-based) consumes stream[4t:4t+4]; slot j is rgb[j] = target_live, rgb[4+j] = source_spoof of that row.
D7 hard samples (domain HARD_SAMPLE): one numpy.random.RandomState(derive('HARD_SAMPLE', s)); before Step 1 of every
    iteration t: u = rng.uniform(0, 1, (4,1,1,3)); k = rng.uniform(0, 1, (4,1,1,1)); hard_scales = float32(u),
    hard_keep = float32(k >= 0.5) -- exactly the qualified M6D3e worker realization, fed unchanged to Steps 1-3.

    python -I -B tools/e04_rng_test_vectors.py            # prints one JSON document
"""
import hashlib
import json
import sys

ROWS, ITERATIONS, PAIRS_PER_ITERATION = 8838, 150000, 4
SEEDS = (42, 1337, 2026)
QUALIFICATION_SEED = 314159
M6D3E_HARD_FIXTURE_SHA256 = ('b735482c8f1510b23b7b9e89487f671c4870d7138600755b9633d652a731fadd',
                             '7472a1a3ade1c12f2ffe2837ae401d69022dc0504bcf8274b4f49b9ad8b248c4')


def derive(domain, seed):
    d = hashlib.sha256(('GPAT_TRANSFERBENCH|E04|%s|%d' % (domain, seed)).encode('utf-8')).digest()
    return 1 + int.from_bytes(d[:4], 'big') % 2147483647


def pair_stream(seed):
    import numpy as np
    rng = np.random.RandomState(derive('PAIR_ORDER', seed))
    need = ITERATIONS * PAIRS_PER_ITERATION
    passes = []
    while sum(len(x) for x in passes) < need:
        passes.append(rng.permutation(ROWS))
    return np.concatenate(passes)[:need].astype('<i4'), passes


def hard_draws(rng):
    import numpy as np
    u = rng.uniform(0, 1, (4, 1, 1, 3))
    k = rng.uniform(0, 1, (4, 1, 1, 1))
    return u.astype(np.float32), (k >= .5).astype(np.float32)


def hard_stream(rng_seed, iterations=ITERATIONS):
    import numpy as np
    rng = np.random.RandomState(rng_seed)
    h, first, keep_sum = hashlib.sha256(), [], 0.
    for t in range(iterations):
        s, k = hard_draws(rng)
        h.update(s.tobytes())
        h.update(k.tobytes())
        keep_sum += float(k.sum())
        if t < 2:
            first.append({'iteration': t, 'hard_scales': [[float(x) for x in row.ravel()] for row in s],
                          'hard_keep': [float(x) for x in k.ravel()]})
    return {'sha256_float32_scales_then_keep_per_iteration': h.hexdigest(), 'first_iterations': first,
            'keep_rate': keep_sum / (4 * iterations), 'uniform_draws_per_iteration': 16}


def vectors():
    import numpy as np
    out = {'numpy': np.__version__, 'python': sys.version.split()[0], 'seeds': {}}
    for s in SEEDS:
        stream, passes = pair_stream(s)
        batches = stream.reshape(ITERATIONS, PAIRS_PER_ITERATION)
        straddle = [t for t in range(ITERATIONS) if (4 * t) // ROWS != (4 * t + 3) // ROWS]
        dup = [t for t in straddle if len(set(batches[t].tolist())) < PAIRS_PER_ITERATION]
        out['seeds'][str(s)] = {
            'pair_seed': derive('PAIR_ORDER', s), 'hard_seed': derive('HARD_SAMPLE', s),
            'pair_stream': {'rows_drawn': int(stream.size), 'full_passes': int(stream.size // ROWS),
                            'tail_rows': int(stream.size % ROWS), 'permutations_generated': len(passes),
                            'every_full_pass_is_a_permutation': all(
                                sorted(stream[p * ROWS:(p + 1) * ROWS].tolist()) == list(range(ROWS))
                                for p in range(stream.size // ROWS)),
                            'sha256_int32_le': hashlib.sha256(stream.tobytes()).hexdigest(),
                            'pass0_head': stream[:8].tolist(), 'pass1_head': stream[ROWS:ROWS + 8].tolist(),
                            'iteration0_rows': batches[0].tolist(), 'iteration149999_rows': batches[-1].tolist(),
                            'pass_straddling_iterations': len(straddle),
                            'straddling_iterations_with_a_repeated_row': dup},
            'hard_stream': hard_stream(derive('HARD_SAMPLE', s))}
    q = np.random.RandomState(QUALIFICATION_SEED)
    s, k = hard_draws(q)
    out['qualification_realization_check'] = {
        'seed': QUALIFICATION_SEED,
        'hard_scales_sha256': hashlib.sha256(s.tobytes()).hexdigest(),
        'hard_keep_sha256': hashlib.sha256(k.tobytes()).hexdigest(),
        'equals_m6d3e_fixture_sha256': [hashlib.sha256(s.tobytes()).hexdigest(),
                                        hashlib.sha256(k.tobytes()).hexdigest()] == list(M6D3E_HARD_FIXTURE_SHA256)}
    out['derived_seeds_distinct'] = len({v for d in out['seeds'].values() for v in (d['pair_seed'], d['hard_seed'])} |
                                        set(SEEDS)) == 3 * len(SEEDS)
    return out


if __name__ == '__main__':
    json.dump(vectors(), sys.stdout, sort_keys=True)
