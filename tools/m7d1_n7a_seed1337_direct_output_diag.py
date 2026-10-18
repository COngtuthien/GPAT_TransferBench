"""M7D1-N7A DIAGNOSTIC_ONLY: direct generator-output check of seed1337 vs historical v1.0 seed42 (collapsed) and
A1 seed42 (healthy). CPU fp32, frozen GPATCore forward (4466e12 clone), 32 TRAIN rows (M7D1-N2 subset rule), no
teachers, no discriminator, no grad. Reads checkpoints into memory first (sha of the exact bytes loaded).
usage: python -I - <clone> < this file   (prints JSON)"""
import hashlib, io, json, os, sys
from datetime import datetime, timezone
from pathlib import Path

CLONE = sys.argv[1]
RT = '/home/student20261/workdir/GPAT_TransferBench_runtime'
ROOTS = {'s1337_a1': f'{RT}/runs/m7_a1/E08/seed_1337', 's42_a1': f'{RT}/runs/m7_a1/E08/seed_42',
         's42_v10': f'{RT}/runs/m7/E08/seed_42'}
EPOCHS = (10, 11, 12, 13)
N_PAIRS, SUBSET_SALT = 32, 'M7D1-N2|fixed-train-diagnostic-subset|v1'
S_HF = 0.10      # curriculum stage 2 value in force at epochs 10..13 (u <= 14365 < 16576) for all three runs

opened, writes = [], []
def hook(event, args):
    if event == 'open' and args and isinstance(args[0], (str, bytes, os.PathLike)):
        p = os.fsdecode(args[0]); opened.append(p)
        mode = args[1] if len(args) > 1 else None; flags = args[2] if len(args) > 2 else 0
        if (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT)):
            writes.append([p, mode, flags])
sys.addaudithook(hook)
sys.path.insert(0, CLONE)
import torch
torch.set_num_threads(4); torch.use_deterministic_algorithms(True); torch.set_grad_enabled(False)
from methods.gpat import runner_io as rio, runner_data as rd
from methods.gpat.config import load_config
from methods.gpat.model import GPATCore
from methods.difffas.aux_runner_io import CanonicalFaceReader


def listing(root):
    lines = []
    for dp, _, fns in os.walk(root):
        for n in fns:
            st = (Path(dp) / n).stat()
            lines.append(f'{(Path(dp) / n).relative_to(root)} {st.st_size} {st.st_mtime_ns // 10**9}.{st.st_mtime_ns % 10**9:09d}0')
    return hashlib.sha256(''.join(f'{x}\n' for x in sorted(lines)).encode()).hexdigest(), len(lines)


def load_bytes(p):
    b = Path(p).read_bytes()
    return torch.load(io.BytesIO(b), map_location='cpu', weights_only=True), hashlib.sha256(b).hexdigest(), len(b)


def q(t, ps=(0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0)):
    t = t.flatten().double()
    if t.numel() > 4_000_000:
        t = t[torch.randperm(t.numel(), generator=torch.Generator().manual_seed(0))[:4_000_000]]
    return {f'p{int(p * 100):02d}' if p < 1 else 'max': float(torch.quantile(t, p)) for p in ps}


def summ(vals):
    v = sorted(vals)
    return {'min': v[0], 'median': v[len(v) // 2], 'mean': sum(v) / len(v), 'max': v[-1]}


def u8(x):
    return torch.clamp(torch.round(255.0 * (x + 1.0) / 2.0), 0, 255).to(torch.uint8)


before = {k: listing(v) for k, v in ROOTS.items() if k != 's1337_a1'}
records = rio.read_relation(Path(CLONE) / rio.RELATION)
access = rio.AccessLog(records)
storage = rio.faces_root()
reader = CanonicalFaceReader(storage['faces_256_root'], rio.sample_datasets(records))
ranked = sorted(records, key=lambda r: hashlib.sha256(f"{SUBSET_SALT}|{r['pair_id']}".encode()).hexdigest())
subset = sorted(ranked[:N_PAIRS], key=lambda r: r['index'])
xs, xt = [], []
for r in subset:
    access.check(r['source_spoof_id'], 'source_spoof_id'); access.check(r['target_live_id'], 'target_live_id')
    xs.append(rd.decode(reader(r['source_spoof_id']))); xt.append(rd.decode(reader(r['target_live_id'])))
x_s, x_t = torch.stack(xs), torch.stack(xt)
cfg = load_config(rio.variant_of('GPAT-B0')[0])
assets = rio.load_assets()
core = GPATCore(cfg, weight_path=assets['assets']['e_art_resnet18_imagenet1k_v1']['path'], with_discriminator=False)
core.eval()


def evaluate(st):
    core.g_res.load_state_dict(st['g_res'], strict=True); core.e_art.load_state_dict(st['e_art'], strict=True)
    core.eval()
    acc = {k: [] for k in ('x_hat', 'A', 'M', 'dLL', 'dLH', 'dHL', 'dHH')}
    for i in range(0, N_PAIRS, 4):
        o = core(x_s[i:i + 4], x_t[i:i + 4], scale_hf=S_HF)
        for k, a in (('x_hat', 'x_hat'), ('A', 'A'), ('M', 'M'), ('dLL', 'delta_LL'), ('dLH', 'delta_LH'),
                     ('dHL', 'delta_HL'), ('dHH', 'delta_HH')):
            acc[k].append(getattr(o, a).float())
    o = {k: torch.cat(v) for k, v in acc.items()}
    res = o['x_hat'] - x_t
    ux, ut = u8(o['x_hat']), u8(x_t)
    per = lambda t: t.mean(dim=tuple(range(1, t.dim()))).tolist()
    return {
        'x_hat_minus_target_mean_abs_per_image': summ(per(res.abs())),
        'x_hat_minus_target_mean_abs_u8_levels_per_image': summ(per(res.abs() * 127.5)),
        'x_hat_minus_target_max_abs': float(res.abs().max()),
        'x_hat_minus_target_rel_l2_per_image': summ((res.flatten(1).norm(dim=1) / x_t.flatten(1).norm(dim=1)).tolist()),
        'u8_x_hat_equals_u8_target_count': int(sum(bool(torch.equal(ux[i], ut[i])) for i in range(N_PAIRS))),
        'u8_pixels_differing_fraction_per_image': summ(per((ux != ut).float())),
        'u8_max_level_diff': int((ux.int() - ut.int()).abs().max()),
        'x_hat_minus_source_mean_abs_per_image': summ(per((o['x_hat'] - x_s).abs())),
        'A_mean_per_image': summ(per(o['A'])), 'A_pixel_quantiles': q(o['A']),
        'M_mean_per_image': summ(per(o['M'])), 'M_pixel_quantiles': q(o['M']),
        'M_fraction_gt_0p99': float((o['M'] > 0.99).float().mean()), 'M_fraction_lt_0p01': float((o['M'] < 0.01).float().mean()),
        'delta_abs_mean': {k: float(o[k].abs().mean()) for k in ('dLL', 'dLH', 'dHL', 'dHH')},
        'delta_HF_abs_quantiles': q(torch.cat([o[k].abs().flatten() for k in ('dLH', 'dHL', 'dHH')])),
    }


out = {'collected_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'), 'torch': str(torch.__version__),
       'clone_head': os.popen(f'git -C {CLONE} rev-parse HEAD').read().strip(), 'scale_hf': S_HF,
       'subset': [{'index': r['index'], 'pair_id': r['pair_id'], 'dataset': r['dataset']} for r in subset],
       'subset_rule': f'sha256("{SUBSET_SALT}|"+pair_id) ascending, first {N_PAIRS} of 8838 TRAIN rows (M7D1-N2 rule)',
       'states': {}}
expected_commit = {'s1337_a1': '4466e1258c24e4a5d448dd5b30e8d8768dfe7174', 's42_a1': '4466e1258c24e4a5d448dd5b30e8d8768dfe7174',
                   's42_v10': '058e976e5a538c4d18e1177f0eb27727cfb735cd'}
expected_seed = {'s1337_a1': 1337, 's42_a1': 42, 's42_v10': 42}
for run, root in ROOTS.items():
    for e in EPOCHS:
        pay, sha, size = load_bytes(f'{root}/checkpoints/ema_candidates/ema_epoch_{e:02d}.pt')
        m = pay['metadata']
        assert m['epoch'] == e and m['seed'] == expected_seed[run] and m['code_commit'] == expected_commit[run] and m['selected'] is False
        st = {'g_res': pay['g_res_ema']['module'], 'e_art': pay['e_art_ema']['module']}
        out['states'][f'{run}/ema_epoch_{e:02d}'] = {'sha256': sha, 'size': size, 'global_update': m['global_update'],
                                                    **evaluate(st)}
        del pay, st
# seed1337 live generator from the latest atomic recovery checkpoint (epoch-end boundary)
rp = f"{ROOTS['s1337_a1']}/checkpoints/recovery/latest.pt"
pay, sha, size = load_bytes(rp)
assert pay['kind'] == 'GPAT_RECOVERY_CHECKPOINT' and pay['provenance']['code_commit'] == expected_commit['s1337_a1']
pos = pay['position']
out['states']['s1337_a1/recovery_live'] = {'sha256': sha, 'size': size, 'position': pos, 'labels': pay['labels'],
                                           **evaluate({'g_res': pay['modules']['g_res'], 'e_art': pay['modules']['e_art']})}
out['states']['s1337_a1/recovery_ema'] = {'sha256': sha, 'position': pos,
                                          **evaluate({'g_res': pay['ema']['g_res']['module'], 'e_art': pay['ema']['e_art']['module']})}
del pay
after = {k: listing(v) for k, v in ROOTS.items() if k != 's1337_a1'}
faces_prefix = str(Path(storage['faces_256_root']).resolve())
allowed = {str(reader.path(r[k])) for r in subset for k in ('source_spoof_id', 'target_live_id')}
faces = sorted({p for p in opened if p.startswith(faces_prefix)})
out['firewall'] = {'access_log': access.report(), 'faces_files_opened': len(faces),
                   'faces_opened_all_in_subset': set(faces) <= allowed,
                   'suspicious_val_test_paths': sorted({p for p in opened if any(w in p.lower() for w in ('/val', '/test', 'val_', 'test_')) and not p.startswith(sys.prefix) and '/site-packages/' not in p}),
                   'run_root_files_opened': sorted({p for p in opened if p.startswith(RT + '/runs/')}),
                   'write_opens': writes}
out['other_roots_listing'] = {'before': before, 'after': after, 'unchanged': before == after}
print(json.dumps(out, indent=1, default=str))
