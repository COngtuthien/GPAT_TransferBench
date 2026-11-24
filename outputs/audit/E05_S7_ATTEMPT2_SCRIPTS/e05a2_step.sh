#!/bin/bash
# laptop: preflight (host, read-only) -> local check -> launch one process.  usage: e05a2_step.sh <SEED> <A|B>
set -u
SEED=$1; L=$2; H=student20261@100.121.84.44
C=7ff4d36f74cd64d6c405944d3b9fa4aee1fae521
S=/tmp/claude-1000/-home-cong-GPAT-TransferBench/3797250c-c2fa-434c-8464-24ed0229a418/scratchpad
PF=/home/student20261/workdir/m8s7_E05_a2_s${SEED}_${L}_preflight.json
cd /home/cong/GPAT_TransferBench
timeout 900 ssh -o ConnectTimeout=20 -o BatchMode=yes $H "test ! -e $PF && cd ~/workdir && CUDA_VISIBLE_DEVICES= /home/student20261/miniconda3/envs/gpat-m6-e05/bin/python -I -B - preflight --method E05 --repo /home/student20261/workdir/m8_learned_banks/clones/s7e05a2_$C --s7-root /home/student20261/workdir/m8_learned_banks/s7/$C --tag E05_a2 --per-seed --seed $SEED --label $L > $PF && chmod a-w $PF && cat $PF" < tools/learned_bank_s7_audit.py > $S/pf_${SEED}_${L}.json || { echo PREFLIGHT_FAILED; exit 3; }
python3 - $S/pf_${SEED}_${L}.json $SEED $L <<'PY' || exit 4
import json, sys
d = json.load(open(sys.argv[1])); s, L = int(sys.argv[2]), sys.argv[3]
g, ps = d['gpu'], d['per_seed']
checks = {'git': d['git']['head'] == '7ff4d36f74cd64d6c405944d3b9fa4aee1fae521' and d['git']['status_porcelain'] == '',
          'no_processes': d['processes'] == [], 'panes_dead': all(v == '1 0' for v in d['m8s7_panes'].values()),
          'gpu': g['uuid'] == 'GPU-b722cd9d-c9fa-223c-915e-cc09d9b862b7' and g['driver'] == '580.178.04',
          'vram': g['free_mib'] >= 5 * 1024, 'ram': d['memory']['mem_available_bytes'] >= 4 << 30,
          'disk': d['disk']['free_bytes'] >= 2 << 30, 'ckpts': len(d['checkpoints']) == 12 and all(v['ok'] for v in d['checkpoints'].values()),
          'authority': d['authority']['ok'], 'env': d['environment']['probe_rc'] == 0,
          'fresh': ps['seed'] == s and ps['label'] == L and ps['run_dir_exists'] is False and
                   ((L == 'A' and not ps['seed_root_exists']) or (L == 'B' and ps['seed_root_exists'] and ps['run_A_completed']))}
print('preflight', s, L, d['utc'], 'free_mib', g['free_mib'], 'ram_gib', round(d['memory']['mem_available_bytes'] / 2**30, 1),
      'unrelated', [(a['used_mib'], a['ps'][:40]) for a in g['compute_apps']], {k: v for k, v in checks.items() if not v} or 'ALL_OK')
sys.exit(0 if all(checks.values()) else 1)
PY
timeout 1800 ssh -o ConnectTimeout=20 -o ServerAliveInterval=30 -o BatchMode=yes $H "bash ~/workdir/m8_learned_banks/e05a2_launch.sh $SEED $L" || { echo LAUNCH_OR_WAIT_FAILED; exit 5; }
