"""M7D1-N6C: VAL-only evaluation of the 51 M7D1-A1 GPAT-B0 / E08 / seed 42 EMA candidates (GPU host).

Run (GPU host, repo cwd, gpat-m7-gpu, the six M7 launch variables):
    python -I -B <this file> {smoke|full|sanity} --repo <gpu repo> [--epoch E] [--resume]
The ranking / selection itself is the separate repository tool tools/m7d1_n6c_a1_seed42_selection.py; this file only
produces raw per-candidate VAL metrics (and, in `sanity`, post-selection diagnostics that never enter selection).

Authorities (nothing re-implemented that already exists):
  candidates   A10 D16 / runner_checkpoint: checkpoints/ema_candidates/ema_epoch_<NN>.pt, epochs 10..60, bytes SHA-256
               checked against the committed N6B EMA manifest (digest 51f9648e...) before torch.load(weights_only=True);
               recovery/latest.pt is never opened.
  VAL relation spec 10.6 manifests/val_pairs_v1.parquet (A10-bound SHA-256 84d12491...), 1905 rows, split == VAL,
               split_manifest_sha256 == fb9aeb36...; read as bytes through a column allowlist (no split_v1, no TEST).
  forward      GPATCore EMA (E_art + G_res) in eval mode (M7C2a N-06), x_hat_float pre-export, no clamp (M7B R-03);
               OWNER N6C decisions: s_hf = 0.15 (spec 9.3 / gpat_b0.yaml delta_scale_hf) for every candidate, and the
               R-06 fp16 autocast for the E_art/G_res forward exactly as in training (fp32 DWT/IDWT/composition/x_hat/A).
  evaluators   runner.Teachers (M7C3-qualified: AdaFace + adaface_input, FaceXFormer + facexformer_input_exact), fp32;
               ArtifactProbe v1 (A10 D13, sha b5ace6c2...) via teacher_preprocess.artifact_probe_input_float, fp32.
  metrics      per VAL pair, then the plain mean over all evaluated pairs (math.fsum):
                 ID     cos(AdaFace(x_t), AdaFace(x_hat))                               (spec 13)
                 Dice   D09: argmax masks, classes 1..10, a class omitted only if absent from both, mean over classes
                 ArtSim cos(ArtifactProbe(x_s), ArtifactProbe(x_hat))                   (A10 D13)
                 NME    mean_k ||lm_hat_k - lm_t_k||_2 / sqrt(w * h), w/h = bbox of x_t's 68 landmarks (A10 D14),
                        FaceXFormer native normalized coordinates (isotropic, so the ratio is coordinate-free)
                 LFErr  M7C2a N-08 rc.lferr_internal(LL_syn_internal, LL_t) (exactly 0 for gamma = 0)
Firewall: a Python audit hook denies (and counts) any canonical-face open whose id is not a VAL relation id and any
manifest open other than val_pairs_v1.parquet; TEST images / metadata are therefore never readable. Determinism:
rc.apply_qualification_determinism (N-09). Outputs go only to the N6C evaluation root; the scientific run root is read.
"""
import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
from datetime import datetime, timezone

MILESTONE = 'M7D1-N6C'
RUNTIME = '/home/student20261/workdir/GPAT_TransferBench_runtime'
RUN_ROOT = f'{RUNTIME}/runs/m7_a1/E08/seed_42'
EVAL_ROOT = f'{RUNTIME}/eval/m7_a1/E08/seed_42/N6C'
RUN_ID = '92e91ddb78f1e242'
LAUNCH_COMMIT = '4466e1258c24e4a5d448dd5b30e8d8768dfe7174'
CURRICULUM_ID = 'GPAT-TransferBench-v1.0+M7D1-A1'
EMA_MANIFEST = 'outputs/audit/M7D1_A1_E08_SEED42_EMA_MANIFEST.txt'
EMA_MANIFEST_DIGEST = '51f9648ebefa0d41f85277ce646280d33c02911b9de04171ddd61b8e9ba79406'
VAL_PAIRS = 'manifests/val_pairs_v1.parquet'
VAL_PAIRS_SHA256 = '84d124919a52cd8d84a89766f464a4dcde1aeaa7791218f6356813a40904f872'
SPLIT_MANIFEST_SHA256 = 'fb9aeb369a124fc96ba855ef2ce269236c4a743fe960e73ab739412c9cb5092d'
VAL_COLUMNS = ('pair_id', 'dataset', 'source_spoof_id', 'target_live_id', 'split', 'split_manifest_sha256')
VAL_ROWS = 1905
PROBE = 'models/artifact_probe/artifact_probe_v1.pt'
PROBE_SHA256 = 'b5ace6c263d8473215ff2ab825b98541bdcf332251512216cbd93546f32e54ff'
EPOCHS = tuple(range(10, 61))
UPDATES_PER_EPOCH = 1105
EVAL_S_HF = 0.15
GENERATOR_PRECISION = 'R06_FP16_AUTOCAST_E_ART_G_RES'
BATCH = 8
SMOKE_EPOCH, SMOKE_PAIRS = 10, 16
DETERMINISM_SEED = 42
METRICS = ('ID', 'Dice', 'ArtSim', 'NME', 'LFErr')
DICE_CLASSES = tuple(range(1, 11))
LABEL = 'VAL_ONLY_N6C_EVALUATION'
SANITY_LABEL = 'POST_SELECTION_SANITY_ONLY'


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':')).encode('utf-8')


def utc():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def require(ok, msg):
    if not ok:
        raise SystemExit(f'{MILESTONE} STOP: {msg}')


def atomic_write(path, data: bytes):
    path = Path(path)
    require(str(path).startswith(EVAL_ROOT + '/'), f'write outside the evaluation root: {path}')
    require(not path.exists(), f'refusing to overwrite {path}')
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix='.tmp-')
    with os.fdopen(fd, 'wb') as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


# ============================================================================ firewall
class Firewall:
    """Audit hook on every Python-level open (builtins.open and os.open)."""

    def __init__(self, repo, faces_root, val_datasets):
        self.faces = str(Path(faces_root).resolve()) + os.sep
        self.manifests = str((Path(repo) / 'manifests').resolve()) + os.sep
        self.val_pairs = str((Path(repo) / VAL_PAIRS).resolve())
        self.run_root = RUN_ROOT + os.sep
        self.val = dict(val_datasets)
        self.counters = {'val_image_opens': 0, 'non_val_image_opens': 0, 'val_metadata_opens': 0,
                         'other_manifest_opens': 0, 'run_root_opens': 0, 'run_root_write_opens': 0,
                         'writes_outside_eval_root': 0}
        self.val_ids_opened = set()
        self.denied = []
        self.writes_outside = []

    def hook(self, event, args):
        if event != 'open' or not args or isinstance(args[0], int):
            return
        try:
            p = os.path.abspath(os.fsdecode(args[0]))
        except Exception:                                   # noqa: BLE001 -- an unparsable path is not ours
            return
        mode = args[1] if len(args) > 1 else None
        flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
        write = bool(mode and any(c in str(mode) for c in 'wax+')) or bool(
            flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC))
        if p.startswith(self.faces):
            rel = p[len(self.faces):].split(os.sep)
            sid = rel[-1][:-4] if len(rel) == 2 and rel[-1].endswith('.png') else None
            if sid is not None and self.val.get(sid) == rel[0] and not write:
                self.counters['val_image_opens'] += 1
                self.val_ids_opened.add(sid)
            else:
                self.counters['non_val_image_opens'] += 1
                self.denied.append(p)
                raise PermissionError(f'{MILESTONE} firewall: non-VAL canonical face {p}')
        elif p.startswith(self.manifests):
            if p == self.val_pairs and not write:
                self.counters['val_metadata_opens'] += 1
            else:
                self.counters['other_manifest_opens'] += 1
                self.denied.append(p)
                raise PermissionError(f'{MILESTONE} firewall: manifest {p} is not the VAL relation')
        elif p.startswith(self.run_root):
            self.counters['run_root_opens'] += 1
            if write:
                self.counters['run_root_write_opens'] += 1
                self.denied.append(p)
                raise PermissionError(f'{MILESTONE} firewall: the scientific run root is immutable ({p})')
        if write and not p.startswith(EVAL_ROOT + os.sep) and not p.startswith('/dev/'):
            self.counters['writes_outside_eval_root'] += 1
            self.writes_outside.append(p)

    def report(self):
        ids = sorted(self.val_ids_opened)
        return {**self.counters, 'unique_val_ids_opened': len(ids),
                'unique_val_ids_sha256': sha256_bytes('\n'.join(ids).encode('utf-8')),
                'denied': list(self.denied), 'writes_outside': sorted(set(self.writes_outside)),
                'test_images_accessed': 0 if not self.denied else None,
                'test_metadata_accessed': 0 if not self.denied else None}


# ============================================================================ inputs
def read_val(repo):
    """Allowlisted, SHA-checked VAL relation rows in manifest order (bytes read through Python, so audited)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    with open(Path(repo) / VAL_PAIRS, 'rb') as fh:
        raw = fh.read()
    require(sha256_bytes(raw) == VAL_PAIRS_SHA256, 'val_pairs_v1 SHA-256')
    table = pq.read_table(pa.BufferReader(raw), columns=list(VAL_COLUMNS))
    rows = table.to_pylist()
    require(len(rows) == VAL_ROWS, f'{VAL_ROWS} VAL rows')
    for i, r in enumerate(rows):
        require(r['split'] == 'VAL' and r['split_manifest_sha256'] == SPLIT_MANIFEST_SHA256, 'VAL-only relation')
        require(r['pair_id'] == f'PVA{i + 1:06d}', 'VAL pair order')
    datasets = {}
    for r in rows:
        for k in ('source_spoof_id', 'target_live_id'):
            require(datasets.setdefault(r[k], r['dataset']) == r['dataset'], 'VAL id bound to two datasets')
    return rows, datasets, raw


def read_ema_manifest(repo):
    raw = (Path(repo) / EMA_MANIFEST).read_bytes()
    require(sha256_bytes(raw) == EMA_MANIFEST_DIGEST, 'N6B EMA manifest digest')
    out = {}
    for line in raw.decode('utf-8').splitlines():
        if line.startswith('#'):
            continue
        e, size, sha, rel = line.split(' ')
        out[int(e)] = {'size': int(size), 'sha256': sha, 'rel': rel}
    require(sorted(out) == list(EPOCHS), '51 manifest rows, epochs 10..60')
    for e, m in out.items():
        require(m['rel'] == f'checkpoints/ema_candidates/ema_epoch_{e:02d}.pt', 'candidate path')
    return out


def load_candidate(torch, manifest, e):
    m = manifest[e]
    path = Path(RUN_ROOT) / m['rel']
    require('recovery' not in m['rel'] and path.name != 'latest.pt', 'recovery is never a candidate')
    with open(path, 'rb') as fh:
        raw = fh.read()
    sha = sha256_bytes(raw)
    require(len(raw) == m['size'] and sha == m['sha256'], f'epoch {e} candidate bytes differ from the N6B manifest')
    payload = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
    meta = payload['metadata']
    require(payload['kind'] == 'GPAT_EMA_CANDIDATE' and meta['epoch'] == e and meta['selected'] is False,
            f'epoch {e} candidate kind/epoch')
    require(meta['global_update'] == UPDATES_PER_EPOCH * e and meta['seed'] == 42 and meta['method'] == 'GPAT-B0',
            f'epoch {e} candidate position')
    require(meta['code_commit'] == LAUNCH_COMMIT and meta['curriculum_id'] == CURRICULUM_ID, 'candidate provenance')
    return payload, {'epoch': e, 'path': str(path), 'rel': m['rel'], 'sha256': sha, 'size_bytes': len(raw),
                     'global_update': meta['global_update']}


def load_probe(torch, repo, device):
    from gpatbench.probe import model as probe_model
    with open(Path(repo) / PROBE, 'rb') as fh:
        raw = fh.read()
    require(sha256_bytes(raw) == PROBE_SHA256, 'ArtifactProbe v1 SHA-256')
    # The M5 payload stores torch.__version__ (TorchVersion, a str subclass). weights_only stays True; only that one
    # data-only global is allowlisted, scoped to this call, after the byte hash has been verified.
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        payload = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
    net = probe_model.build(pretrained=False)
    net.load_state_dict(payload['model_state_dict'], strict=True)
    net.eval().requires_grad_(False).to(device)
    return net, {'path': PROBE, 'sha256': PROBE_SHA256, 'selected_epoch': payload.get('selected_epoch'),
                 'schema_version': payload.get('schema_version')}


# ============================================================================ metrics
def dice_d09(torch, t, h):
    """A10 D09 per sample: argmax masks [N, H, W]; classes 1..10; omitted only if absent from both; class mean."""
    num = torch.zeros(t.shape[0], dtype=torch.float64, device=t.device)
    cnt = torch.zeros(t.shape[0], dtype=torch.float64, device=t.device)
    for c in DICE_CLASSES:
        pt, ph = (t == c), (h == c)
        st, sh = pt.sum((1, 2)).double(), ph.sum((1, 2)).double()
        inter = (pt & ph).sum((1, 2)).double()
        valid = (st + sh) > 0
        num = num + torch.where(valid, 2.0 * inter / torch.where(valid, st + sh, torch.ones_like(st)),
                                torch.zeros_like(st))
        cnt = cnt + valid.double()
    require(bool((cnt > 0).all()), 'a VAL pair has no parser class 1..10 in either mask (D09 undefined)')
    return num / cnt


def nme(torch, lm_t, lm_h):
    """mean landmark L2 / sqrt(w * h) of the target's 68-landmark bounding box (A10 D14), per sample."""
    t, h = lm_t.double(), lm_h.double()
    d = (h - t).norm(dim=2).mean(dim=1)
    w = t[..., 0].amax(1) - t[..., 0].amin(1)
    hh = t[..., 1].amax(1) - t[..., 1].amin(1)
    norm = (w * hh).sqrt()
    require(bool((norm > 0).all()), 'degenerate target landmark bounding box')
    return d / norm


def mean(values):
    return math.fsum(values) / len(values)


# ============================================================================ evaluation
class Evaluator:
    def __init__(self, repo, n_pairs=None):
        import torch
        self.torch = torch
        self.repo = Path(repo)
        sys.path.insert(0, str(self.repo))
        from methods.gpat import runtime_contract as rc
        from methods.gpat import runner_io as rio
        self.rc, self.rio = rc, rio
        storage = rio.faces_root()
        self.faces_root = storage['faces_256_root']
        self.fw = Firewall(self.repo, self.faces_root, {})
        sys.addaudithook(self.fw.hook)                  # installed before the VAL relation is read
        self.rows, self.val_datasets, _ = read_val(self.repo)
        self.fw.val = dict(self.val_datasets)
        self.rows = self.rows[:n_pairs] if n_pairs else self.rows
        self.ema_manifest = read_ema_manifest(self.repo)
        self.determinism = rc.apply_qualification_determinism(DETERMINISM_SEED, gpu=True)
        from methods.gpat import runner
        from methods.gpat.config import load_config
        from methods.gpat.model import GPATCore
        self.assets = rio.load_assets()
        self.teachers = runner.Teachers(self.assets)
        self.probe, self.probe_identity = load_probe(torch, self.repo, 'cuda')
        self.cfg = load_config(rio.variant_of('GPAT-B0')[0])
        require(self.cfg.gamma == 0.0 and self.cfg.delta_scale_hf == EVAL_S_HF, 'B0 gamma 0 / delta_scale_hf 0.15')
        self.core = GPATCore(self.cfg, weight_path=self.assets['assets']['e_art_resnet18_imagenet1k_v1']['path'],
                             with_discriminator=False).to('cuda')
        from methods.difffas.aux_runner_io import CanonicalFaceReader
        from methods.gpat import runner_data as rd
        self.reader = CanonicalFaceReader(self.faces_root, self.val_datasets)
        self.rd = rd
        self.images = {}
        self._load_images()
        self._targets()

    # ---------------------------------------------------------------- data (decoded once, rd.decode contract)
    def _load_images(self):
        import numpy as np
        for r in self.rows:
            for k in ('source_spoof_id', 'target_live_id'):
                sid = r[k]
                if sid not in self.images:
                    self.images[sid] = np.asarray(self.reader(sid), dtype=np.uint8).copy()

    def batch(self, i):
        torch = self.torch
        rows = self.rows[i:i + BATCH]
        xs = torch.stack([self.rd.decode(self.images[r['source_spoof_id']]) for r in rows]).to('cuda')
        xt = torch.stack([self.rd.decode(self.images[r['target_live_id']]) for r in rows]).to('cuda')
        return xs, xt

    def probe_embed(self, x):
        from gpatbench.probe import model as probe_model
        from methods.gpat import teacher_preprocess as tp
        with self.torch.autocast(device_type='cuda', enabled=False):
            return probe_model.embed(self.probe, tp.artifact_probe_input_float(x.float())).float()

    def _targets(self):
        """Candidate-independent VAL target/source quantities, computed once with the same frozen fp32 evaluators."""
        torch = self.torch
        seg, lm, idt, prs = [], [], [], []
        with torch.no_grad():
            for i in range(0, len(self.rows), BATCH):
                xs, xt = self.batch(i)
                s, l = self.teachers.facexformer(xt)
                seg.append(s.argmax(dim=1).to(torch.uint8).cpu())
                lm.append(l.cpu())
                idt.append(self.teachers.adaface(xt).cpu())
                prs.append(self.probe_embed(xs).cpu())
        self.t_seg, self.t_lm, self.t_id, self.s_probe = (torch.cat(seg), torch.cat(lm), torch.cat(idt),
                                                          torch.cat(prs))

    # ---------------------------------------------------------------- one candidate
    def forward(self, payload):
        torch = self.torch
        self.core.e_art.load_state_dict(payload['e_art_ema']['module'], strict=True)
        self.core.g_res.load_state_dict(payload['g_res_ema']['module'], strict=True)
        self.core.eval()
        require(not self.core.training and not self.core.e_art.training, 'EMA evaluation mode is eval (N-06)')

    def evaluate(self, epoch, *, diagnostics=False):
        torch = self.torch
        import torch.nn.functional as F
        payload, ident = load_candidate(torch, self.ema_manifest, epoch)
        self.forward(payload)
        del payload
        per = {k: [] for k in METRICS}
        diag = {k: [] for k in DIAG_KEYS} if diagnostics else None
        t0 = time.monotonic()
        with torch.no_grad():
            for i in range(0, len(self.rows), BATCH):
                xs, xt = self.batch(i)
                n = xs.shape[0]
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    out = self.core(xs, xt, scale_hf=EVAL_S_HF)
                x_hat = out.x_hat
                require(x_hat.dtype == torch.float32, 'x_hat_float is fp32 (R-06)')
                seg_h, lm_h = self.teachers.facexformer(x_hat)
                id_h = self.teachers.adaface(x_hat)
                pr_h = self.probe_embed(x_hat)
                sl = slice(i, i + n)
                vals = {
                    'ID': F.cosine_similarity(self.t_id[sl].cuda(), id_h, dim=1),
                    'Dice': dice_d09(torch, self.t_seg[sl].cuda().long(), seg_h.argmax(dim=1)),
                    'ArtSim': F.cosine_similarity(self.s_probe[sl].cuda(), pr_h, dim=1),
                    'NME': nme(torch, self.t_lm[sl].cuda(), lm_h),
                    'LFErr': self.rc.lferr_internal(out.LL_syn.float(), out.target_bands[0].float())}
                for k, v in vals.items():
                    v = v.double().cpu()
                    require(v.shape == (n,) and bool(torch.isfinite(v).all()), f'epoch {epoch} {k} non-finite')
                    per[k].extend(float(x) for x in v.tolist())
                if diagnostics:
                    collect_diagnostics(torch, diag, out, xt)
        torch.cuda.synchronize()
        res = {'label': LABEL, 'candidate': ident, 'evaluated_pairs': len(self.rows),
               'pair_ids_sha256': sha256_bytes('\n'.join(r['pair_id'] for r in self.rows).encode('utf-8')),
               'metrics': {k: mean(per[k]) for k in METRICS},
               'per_pair_sha256': {k: sha256_bytes(canonical(per[k])) for k in METRICS},
               'status': 'COMPLETE', 'warnings': [], 'wall_seconds': round(time.monotonic() - t0, 3)}
        return res, per, diag

    def config(self):
        torch = self.torch
        return {'eval_s_hf': EVAL_S_HF, 'generator_precision': GENERATOR_PRECISION, 'gamma': self.cfg.gamma,
                'batch': BATCH, 'ema_evaluation_mode': 'eval', 'x_hat': 'float pre-export, no clamp (M7B R-03)',
                'determinism': self.determinism, 'determinism_seed': DETERMINISM_SEED,
                'val_pairs_sha256': VAL_PAIRS_SHA256, 'split_manifest_sha256': SPLIT_MANIFEST_SHA256,
                'evaluated_pairs': len(self.rows), 'ema_manifest_digest': EMA_MANIFEST_DIGEST,
                'teachers': {k: v['sha256'] for k, v in self.assets['assets'].items() if 'sha256' in v},
                'asset_config_sha256': self.assets['asset_config_sha256'], 'artifact_probe': self.probe_identity,
                'aggregation': 'per-pair metric, then math.fsum mean over all evaluated VAL pairs',
                'torch': str(torch.__version__), 'cuda': str(torch.version.cuda),
                'gpu': torch.cuda.get_device_name(0), 'python': platform.python_version()}


DIAG_KEYS = ('mean_abs_xhat_minus_xt', 'max_abs_xhat_minus_xt', 'u8_identical_pixel_fraction', 'u8_identical_image',
             'A_mean', 'M_mean', 'M_min', 'M_max', 'M_frac_gt_0_5', 'hf_residual_mean_abs', 'hf_target_mean_abs',
             'x_hat_out_of_range_fraction')


def collect_diagnostics(torch, diag, out, xt):
    """POST_SELECTION_SANITY_ONLY quantities from the normal forward path (never used by the selection rule)."""
    xh = out.x_hat.float()
    res = (xh - xt).abs()
    u8h = torch.round((xh.double() + 1.0) * 0.5 * 255.0).clamp(0, 255)
    u8t = torch.round((xt.double() + 1.0) * 0.5 * 255.0).clamp(0, 255)
    same = (u8h == u8t).all(dim=1).double()
    lh, hl, hh = out.target_bands[1], out.target_bands[2], out.target_bands[3]
    hf_res = ((out.LH_syn - lh).abs() + (out.HL_syn - hl).abs() + (out.HH_syn - hh).abs()).float()
    hf_t = (lh.abs() + hl.abs() + hh.abs()).float()
    m = out.M.float()
    vals = {'mean_abs_xhat_minus_xt': res.mean((1, 2, 3)), 'max_abs_xhat_minus_xt': res.amax((1, 2, 3)),
            'u8_identical_pixel_fraction': same.mean((1, 2)), 'u8_identical_image': (same.amin((1, 2)) == 1).double(),
            'A_mean': out.A.float().flatten(1).mean(1), 'M_mean': m.flatten(1).mean(1), 'M_min': m.flatten(1).amin(1),
            'M_max': m.flatten(1).amax(1), 'M_frac_gt_0_5': (m > 0.5).double().flatten(1).mean(1),
            'hf_residual_mean_abs': hf_res.flatten(1).mean(1), 'hf_target_mean_abs': hf_t.flatten(1).mean(1),
            'x_hat_out_of_range_fraction': ((xh < -1) | (xh > 1)).double().flatten(1).mean(1)}
    for k, v in vals.items():
        diag[k].extend(float(x) for x in v.double().cpu().tolist())


def summarize(values):
    s = sorted(values)
    n = len(s)
    return {'n': n, 'mean': mean(s), 'min': s[0], 'p05': s[int(0.05 * (n - 1))], 'median': s[(n - 1) // 2],
            'p95': s[int(0.95 * (n - 1))], 'max': s[-1]}


# ============================================================================ phases
def provenance(repo):
    """Repository HEAD / cleanliness are recorded by the launcher (outside the audited process)."""
    return {'tool': Path(__file__).name, 'tool_sha256': sha256_bytes(Path(__file__).read_bytes()),
            'repo': str(repo), 'started_utc': utc(), 'milestone': MILESTONE, 'run_id': RUN_ID,
            'run_root': RUN_ROOT, 'eval_root': EVAL_ROOT, 'launch_commit': LAUNCH_COMMIT,
            'curriculum_id': CURRICULUM_ID}


def write_result(directory, name, obj):
    atomic_write(Path(directory) / name, json.dumps(obj, indent=1, sort_keys=True).encode('utf-8') + b'\n')


def phase_smoke(repo):
    out = Path(EVAL_ROOT) / 'smoke'
    out.mkdir(exist_ok=False)
    prov = provenance(repo)
    ev = Evaluator(repo, n_pairs=SMOKE_PAIRS)
    res, per, _ = ev.evaluate(SMOKE_EPOCH)
    fw = ev.fw.report()
    ok = (fw['non_val_image_opens'] == 0 and fw['other_manifest_opens'] == 0 and not fw['denied'] and
          fw['val_image_opens'] > 0 and fw['val_metadata_opens'] == 1 and fw['run_root_write_opens'] == 0 and
          set(res['metrics']) == set(METRICS) and all(math.isfinite(v) for v in res['metrics'].values()))
    write_result(out, 'smoke.json', {'label': 'SMOKE_NOT_USED_FOR_SELECTION', 'provenance': prov,
                                     'config': ev.config(), 'result': res, 'per_pair': per, 'firewall': fw,
                                     'smoke_pass': ok, 'finished_utc': utc()})
    print(json.dumps({'smoke_pass': ok, 'metrics': res['metrics'], 'firewall': fw,
                      'wall_seconds': res['wall_seconds']}, indent=1))
    require(ok, 'smoke failed')


def phase_full(repo, resume):
    out = Path(EVAL_ROOT) / 'full'
    out.mkdir(exist_ok=resume)
    prov = provenance(repo)
    ev = Evaluator(repo)
    cfg = ev.config()
    cfg_digest = sha256_bytes(canonical({k: v for k, v in cfg.items() if k not in ('gpu',)}))
    done = []
    for e in EPOCHS:
        path = out / f'epoch_{e:02d}.json'
        if path.exists():
            prev = json.loads(path.read_text())
            require(resume and prev['config_digest'] == cfg_digest and prev['tool_sha256'] == prov['tool_sha256'],
                    f'existing result for epoch {e} under a different configuration')
            done.append(e)
            continue
        res, per, _ = ev.evaluate(e)
        write_result(out, path.name, {**res, 'per_pair': per, 'config_digest': cfg_digest,
                                      'tool_sha256': prov['tool_sha256'], 'finished_utc': utc()})
        print(f'{utc()} epoch {e} done {res["wall_seconds"]}s {json.dumps(res["metrics"])}', flush=True)
    fw = ev.fw.report()
    session = f'session_{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")}.json'
    write_result(out, session, {'provenance': prov, 'config': cfg, 'config_digest': cfg_digest, 'firewall': fw,
                                'skipped_existing': done, 'finished_utc': utc()})
    require(not fw['denied'] and fw['non_val_image_opens'] == 0 and fw['other_manifest_opens'] == 0, 'firewall')
    print(json.dumps({'firewall': fw, 'session': session}, indent=1))


def phase_sanity(repo, epoch):
    """Re-evaluate the selected EMA on the full VAL set (evaluator reproducibility vs the full phase) and record the
    POST_SELECTION_SANITY_ONLY collapse diagnostics. Never alters the selection."""
    out = Path(EVAL_ROOT) / 'sanity'
    out.mkdir(exist_ok=False)
    prov = provenance(repo)
    ev = Evaluator(repo)
    cfg = ev.config()
    res, per, diag = ev.evaluate(epoch, diagnostics=True)
    full = json.loads((Path(EVAL_ROOT) / 'full' / f'epoch_{epoch:02d}.json').read_text())
    repro = {k: {'full': full['metrics'][k], 'rerun': res['metrics'][k], 'bitwise_equal_mean':
                 full['metrics'][k] == res['metrics'][k], 'per_pair_bitwise_equal': full['per_pair'][k] == per[k]}
             for k in METRICS}
    fw = ev.fw.report()
    summary = {k: summarize(v) for k, v in diag.items()}
    write_result(out, f'sanity_epoch_{epoch:02d}.json', {
        'label': SANITY_LABEL, 'provenance': prov, 'config': cfg, 'result': res, 'reproducibility': repro,
        'diagnostics_summary': summary, 'diagnostics_per_pair_sha256': {k: sha256_bytes(canonical(v))
                                                                         for k, v in diag.items()},
        'firewall': fw, 'finished_utc': utc()})
    print(json.dumps({'reproducibility': repro, 'diagnostics_summary': summary, 'firewall': fw}, indent=1))
    require(not fw['denied'], 'firewall')


def phase_collect():
    """Read-only: print every full-phase result (without per-pair lists) plus the sessions / smoke / sanity records."""
    root = Path(EVAL_ROOT)
    out = {'eval_root': EVAL_ROOT, 'collected_utc': utc(), 'files': {}, 'full': {}, 'sessions': [], 'smoke': None,
           'sanity': []}
    for p in sorted(root.rglob('*.json')):
        raw = p.read_bytes()
        out['files'][str(p.relative_to(root))] = {'size': len(raw), 'sha256': sha256_bytes(raw)}
        obj = json.loads(raw)
        rel = p.relative_to(root)
        if rel.parts[0] == 'full' and p.name.startswith('epoch_'):
            obj.pop('per_pair')
            out['full'][str(obj['candidate']['epoch'])] = obj
        elif rel.parts[0] == 'full':
            out['sessions'].append(obj)
        elif rel.parts[0] == 'smoke':
            obj.pop('per_pair')
            out['smoke'] = obj
        elif rel.parts[0] == 'sanity':
            out['sanity'].append(obj)
    print(json.dumps(out, sort_keys=True))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('phase', choices=('smoke', 'full', 'sanity', 'collect'))
    ap.add_argument('--repo', default=None)
    ap.add_argument('--epoch', type=int, default=None)
    ap.add_argument('--resume', action='store_true')
    a = ap.parse_args(argv)
    if a.phase == 'collect':
        return phase_collect()
    require(a.repo and Path(a.repo).is_absolute(), '--repo <absolute gpu repo>')
    require(Path(EVAL_ROOT).is_dir(), 'the launcher pre-creates the evaluation root')
    if a.phase == 'smoke':
        return phase_smoke(a.repo)
    if a.phase == 'full':
        return phase_full(a.repo, a.resume)
    require(a.epoch in EPOCHS, '--epoch in 10..60')
    return phase_sanity(a.repo, a.epoch)


if __name__ == '__main__':
    main()
