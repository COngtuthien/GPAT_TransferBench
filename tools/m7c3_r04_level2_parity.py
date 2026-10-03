"""M7C3 R-04 Level-2 teacher parity on real TRAIN faces (gpat-m7-gpu, RTX 3090). Parity measurement only.

The contract is configs/amendments/gpat_m7c2a_implementation_resolution.yaml resolutions.R-04.level2 (frozen at M7C2a,
thresholds deferred to this GPU qualification). This tool:

  1. selects, per dataset (casia_fasd, msu_mfsd, siwmv2), the up-to-64 unique TRAIN live target ids of
     manifests/pairs_train_v1.parquet with the smallest SHA256(UTF-8 id) (teacher_preprocess.level2_subset);
  2. proves every selected id is TRAIN in manifests/split_v1.parquet, refuses VAL/TEST, and opens ONLY those PNGs
     (each verified against the M2 face_png_sha256 before decoding);
  3. per image runs the frozen reference teachers (gpatbench/preprocess/aux_models.py, uint8 PIL/cv2 path), 8 dithered
     re-quantization replicas (dither BEFORE the teacher's uint8 quantization; M7C2a seed rule) and the committed
     differentiable candidate adapters, all fp32 / eval / requires_grad False, batch 1;
  4. derives the noise floors and applies the frozen envelope rules (no threshold is chosen after looking at results);
  5. checks teacher input-gradient structure and the integrated teacher memory on synthetic tensors.

No training, optimizer, checkpoint, bank, VAL or TEST image. Output: one JSON evidence file.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import warnings

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np                                                  # noqa: E402
import torch                                                        # noqa: E402
import torch.nn.functional as F                                     # noqa: E402

from gpatbench.preprocess import aux_models                         # noqa: E402
from methods.gpat import runtime_contract as rc, teacher_preprocess as tp   # noqa: E402

AUTHORITY = '33955b05ff70289c42386c3ac1623bae4a5fae5e'
DEV = 'cuda'
PAIRS = 'manifests/pairs_train_v1.parquet'
SPLIT = 'manifests/split_v1.parquet'
ACCOUNTING = 'manifests/m2_sample_accounting.parquet'
MANIFEST_SHA = {PAIRS: 'a5e4fdaef236f15730c7e3885b537e08e995faffbe654167fc940f44bbc75243',
                SPLIT: 'fb9aeb369a124fc96ba855ef2ce269236c4a743fe960e73ab739412c9cb5092d',
                ACCOUNTING: '65bb83e69ea7cebcb37373971e5e8f5248609b7963fd9d32aa1c0ba4377c0662'}
RUNTIME = Path('/home/student20261/workdir/GPAT_TransferBench_runtime')
FACES = RUNTIME / 'data/processed/faces_256'
TEACHERS = RUNTIME / 'third_party_weights/m7c3_teachers'
ASSETS = {
    'adaface_weight': (TEACHERS / 'adaface/adaface_ir50_webface4m.ckpt', aux_models.ADAFACE_WEIGHT_SHA256),
    'facexformer_weight': (TEACHERS / 'facexformer/ckpts/model.pt', aux_models.FACEXFORMER_WEIGHT_SHA256),
    'f_art_resnet18_imagenet1k_v1': (Path.home() / '.cache/torch/hub/checkpoints/resnet18-f37072fd.pth',
                                     'f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec'),
    'artifact_probe_v1': (ROOT / 'models/artifact_probe/artifact_probe_v1.pt',
                          'b5ace6c263d8473215ff2ab825b98541bdcf332251512216cbd93546f32e54ff')}
FX_PARSE, FX_LM = aux_models.FX_TASKS['parsing'], aux_models.FX_TASKS['landmarks']
REPLICAS = tp.L2_REPLICAS
IMAGENET_MEAN = torch.tensor(tp.IMAGENET_MEAN).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor(tp.IMAGENET_STD).view(1, 3, 1, 1)


def sha_file(p) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode('utf-8')


# ----------------------------------------------------------------------------- inventory + TRAIN-only selection
def inventory() -> dict:
    out = {}
    for name, (path, want) in ASSETS.items():
        got = sha_file(path) if path.is_file() else None
        out[name] = {'path': str(path), 'sha256': got, 'expected': want, 'match': got == want}
    aux_models.verify_code(TEACHERS / 'facexformer/code', aux_models.FACEXFORMER_CODE_SHA256)
    aux_models.verify_code(TEACHERS / 'adaface/code', aux_models.ADAFACE_CODE_SHA256)
    out['facexformer_code'] = {'commit': '10fe8291f8a64e2ca1daf938e3e0007bd860303b',
                               'files_sha256': aux_models.FACEXFORMER_CODE_SHA256, 'match': True}
    out['adaface_code'] = {'commit': aux_models.ADAFACE_SELECTED['code_commit'],
                           'files_sha256': aux_models.ADAFACE_CODE_SHA256, 'match': True,
                           'architecture': 'IR-50', 'input': 'BGR 112x112, ((u8/255) - 0.5) / 0.5'}
    out['all_match'] = all(v['match'] for v in out.values() if isinstance(v, dict))
    return out


def select_train() -> dict:
    import pyarrow.parquet as pq
    for rel, digest in MANIFEST_SHA.items():
        if sha_file(ROOT / rel) != digest:
            raise SystemExit(f'manifest hash mismatch: {rel}')
    pairs = pq.read_table(ROOT / PAIRS, columns=['target_live_id', 'dataset', 'split']).to_pylist()
    split = {r['sample_id']: r for r in pq.read_table(ROOT / SPLIT, columns=['sample_id', 'dataset', 'split',
                                                                                  'label_binary', 'm2_status']).to_pylist()}
    acct = {r['sample_id']: r for r in pq.read_table(ROOT / ACCOUNTING, columns=['sample_id', 'face_png_sha256',
                                                                                     'final_status']).to_pylist()}
    if {r['split'] for r in pairs} != {'TRAIN'}:
        raise SystemExit('pairs_train_v1 must contain TRAIN rows only')
    eligible = {}
    for r in pairs:
        eligible.setdefault(r['dataset'], set()).add(r['target_live_id'])
    held_out = {s for s, r in split.items() if r['split'] in ('VAL', 'TEST')}
    for ds, ids in eligible.items():
        for s in ids:
            row = split[s]
            if row['split'] != 'TRAIN' or row['label_binary'] != 0 or row['dataset'] != ds or row['m2_status'] != 'COMPLETE':
                raise SystemExit(f'ineligible target {s}')
        if ids & held_out:
            raise SystemExit('VAL/TEST id among eligible targets')
    subset = tp.level2_subset({k: sorted(v) for k, v in eligible.items()})
    flat = [s for ds in tp.L2_DATASETS for s in subset[ds]]
    if len(flat) != len(set(flat)) or set(flat) & held_out:
        raise SystemExit('selection must be unique TRAIN ids')
    records = [{'dataset': ds, 'sample_id': s, 'split': split[s]['split'], 'label_binary': split[s]['label_binary'],
                'face_png_sha256': acct[s]['face_png_sha256']} for ds in tp.L2_DATASETS for s in subset[ds]]
    return {'rule': 'per dataset, the up-to-64 unique TRAIN live target ids of pairs_train_v1 with the smallest '
                    'SHA256(UTF-8 sample_id) hex digests (teacher_preprocess.level2_subset)',
            'manifest_sha256': MANIFEST_SHA, 'eligible_counts': {k: len(v) for k, v in sorted(eligible.items())},
            'selected_counts': {ds: len(subset[ds]) for ds in tp.L2_DATASETS},
            'selected_ids': {ds: subset[ds] for ds in tp.L2_DATASETS},
            'selected_list_sha256': hashlib.sha256(canonical({ds: subset[ds] for ds in tp.L2_DATASETS})).hexdigest(),
            'val_test_ids_in_split_manifest': len(held_out), 'val_test_selected': 0, 'records': records}


class TrainFaceReader:
    """The only image reader in M7C3: opens a canonical face only if it is one of the selected TRAIN records."""

    def __init__(self, records):
        self.allowed = {r['sample_id']: r for r in records}
        self.opened = []

    def read(self, sample_id):
        import cv2
        r = self.allowed.get(sample_id)
        if r is None or r['split'] != 'TRAIN':
            raise PermissionError(f'M7C3 refuses to open non-selected / non-TRAIN sample {sample_id}')
        path = FACES / r['dataset'] / f'{sample_id}.png'
        raw = path.read_bytes()
        got = hashlib.sha256(raw).hexdigest()
        if got != r['face_png_sha256']:
            raise SystemExit(f'face PNG hash mismatch for {sample_id}')
        bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        if rgb.shape != (256, 256, 3) or rgb.dtype != np.uint8:
            raise SystemExit(f'unexpected canonical face {sample_id}: {rgb.shape}')
        self.opened.append({'dataset': r['dataset'], 'sample_id': sample_id, 'split': 'TRAIN', 'png_sha256': got})
        return rgb


# ----------------------------------------------------------------------------- teacher calls (fp32, eval, batch 1)
def fx_outputs(model, x_norm):
    out = {}
    with torch.inference_mode():
        for name, task in (('parse', FX_PARSE), ('lm', FX_LM)):
            res = model(x_norm, None, torch.tensor([task], device=DEV))
            if name == 'parse':
                out['parse'] = res[7][0].float()                            # [11, 224, 224]
            else:
                out['lm'] = res[0].view(-1, 68, 2)[0].float()               # native normalized [68, 2]
    return out


def ada_embedding(model, x_bgr):
    with torch.inference_mode():
        feat, _ = model(x_bgr)
    return F.normalize(feat.float(), dim=1)[0]


def d09_macro_dice(a, b):
    """D09: classes 1..10, a class is omitted when absent from both masks; mean over the remaining classes."""
    vals = []
    for c in range(1, 11):
        ma, mb = a == c, b == c
        sa, sb = int(ma.sum()), int(mb.sum())
        if sa == 0 and sb == 0:
            continue
        vals.append(2.0 * int((ma & mb).sum()) / (sa + sb))
    return float(np.mean(vals)) if vals else None


def compare_fx(ref, other):
    lm_d = (other['lm'] - ref['lm']).abs()
    am_r, am_o = ref['parse'].argmax(0), other['parse'].argmax(0)
    return {'lm_mean': float(lm_d.mean()), 'lm_max': float(lm_d.max()),
            'argmax_agreement': float((am_r == am_o).float().mean()),
            'dice': d09_macro_dice(am_r.cpu().numpy(), am_o.cpu().numpy())}


def stats(values):
    v = np.asarray([x for x in values if x is not None], dtype=np.float64)
    return {'n': int(v.size), 'min': float(v.min()), 'p01': float(np.percentile(v, 1)), 'median': float(np.median(v)),
            'mean': float(v.mean()), 'max': float(v.max())}


def level2(selection) -> dict:
    fx = aux_models.FaceXFormerAdapter(TEACHERS / 'facexformer/code', TEACHERS / 'facexformer/ckpts/model.pt', device=DEV)
    ada = aux_models.AdaFaceAdapter(TEACHERS / 'adaface/code', TEACHERS / 'adaface/adaface_ir50_webface4m.ckpt', device=DEV)
    for m in (fx.model, ada.model):
        m.eval().requires_grad_(False)
    reader = TrainFaceReader(selection['records'])
    w_area = torch.as_tensor(tp.area_matrix(tp.CANONICAL, tp.ADAFACE_INPUT), dtype=torch.float32, device=DEV)
    mean, std = IMAGENET_MEAN.to(DEV), IMAGENET_STD.to(DEV)
    rows = []
    t0 = time.time()
    for rec in selection['records']:
        ds, sid = rec['dataset'], rec['sample_id']
        rgb = reader.read(sid)
        # frozen references (uint8 PIL / cv2 paths, unchanged)
        ref_fx_in = fx.preprocess(rgb)
        ref_fx = fx_outputs(fx.model, ref_fx_in)
        ref_ada = ada_embedding(ada.model, ada.preprocess(rgb))
        ref_ada_np = torch.as_tensor(ada(rgb)['embedding'], device=DEV)
        x = torch.as_tensor(rgb, device=DEV).permute(2, 0, 1)[None].float() / 127.5 - 1.0     # GPAT [-1, 1]
        # candidates (committed differentiable adapters, float, no rounding)
        cand_fx_in = tp.facexformer_input(x)
        cand_fx = fx_outputs(fx.model, cand_fx_in)
        cand_ada = ada_embedding(ada.model, tp.adaface_input(x))
        # dithered re-quantization replicas at each teacher's own uint8 point
        fx_unit = tp.facexformer_unit(x)
        ada_unit = torch.einsum('oh,nchw,pw->ncop', w_area, ((x + 1.0) * 0.5).clamp(0, 1), w_area)
        rep_fx, rep_ada = [], []
        for k in range(REPLICAS):
            q = tp.requantization_replica(fx_unit, ds, sid, k).to(DEV)
            rep_fx.append(compare_fx(ref_fx, fx_outputs(fx.model, (q - mean) / std)))
            qa = tp.requantization_replica(ada_unit, ds, sid, k).to(DEV)
            e = ada_embedding(ada.model, ((qa - 0.5) / 0.5).flip(1))
            rep_ada.append(float(torch.dot(ref_ada, e)))
        rows.append({
            'dataset': ds, 'sample_id': sid,
            'reference_adapter_embedding_selfcheck_max_abs': float((ref_ada - ref_ada_np).abs().max()),
            'input_lsb_candidate_vs_reference_fx': float(((cand_fx_in - ref_fx_in) * std).abs().max() * 255),
            'ada_cos_candidate': float(torch.dot(ref_ada, cand_ada)), 'ada_cos_replicas': rep_ada,
            'fx_candidate': compare_fx(ref_fx, cand_fx), 'fx_replicas': rep_fx})
    elapsed = time.time() - t0
    del fx, ada
    torch.cuda.empty_cache()

    def gather(rs, key_fn):
        return [key_fn(r) for r in rs]

    floors = {
        'adaface_cos': min(c for r in rows for c in r['ada_cos_replicas']),
        'landmark_mean': max(x['lm_mean'] for r in rows for x in r['fx_replicas']),
        'landmark_abs': max(x['lm_max'] for r in rows for x in r['fx_replicas']),
        'parsing_argmax': min(x['argmax_agreement'] for r in rows for x in r['fx_replicas']),
        'parsing_dice': min(x['dice'] for r in rows for x in r['fx_replicas'] if x['dice'] is not None)}
    cand = {
        'adaface_cos': min(r['ada_cos_candidate'] for r in rows),
        'landmark_mean': max(r['fx_candidate']['lm_mean'] for r in rows),
        'landmark_abs': max(r['fx_candidate']['lm_max'] for r in rows),
        'parsing_argmax': min(r['fx_candidate']['argmax_agreement'] for r in rows),
        'parsing_dice': min(r['fx_candidate']['dice'] for r in rows if r['fx_candidate']['dice'] is not None)}
    gates = {'adaface': cand['adaface_cos'] >= floors['adaface_cos'],
             'landmarks': cand['landmark_mean'] <= floors['landmark_mean'] and cand['landmark_abs'] <= floors['landmark_abs'],
             'parsing': cand['parsing_argmax'] >= floors['parsing_argmax'] and cand['parsing_dice'] >= floors['parsing_dice']}
    per_dataset = {}
    for ds in tp.L2_DATASETS:
        rs = [r for r in rows if r['dataset'] == ds]
        if not rs:
            continue
        per_dataset[ds] = {
            'n': len(rs),
            'adaface_cos_replicas': stats([c for r in rs for c in r['ada_cos_replicas']]),
            'adaface_cos_candidate': stats(gather(rs, lambda r: r['ada_cos_candidate'])),
            'landmark_mean_replicas': stats([x['lm_mean'] for r in rs for x in r['fx_replicas']]),
            'landmark_mean_candidate': stats(gather(rs, lambda r: r['fx_candidate']['lm_mean'])),
            'landmark_abs_replicas': stats([x['lm_max'] for r in rs for x in r['fx_replicas']]),
            'landmark_abs_candidate': stats(gather(rs, lambda r: r['fx_candidate']['lm_max'])),
            'parsing_argmax_replicas': stats([x['argmax_agreement'] for r in rs for x in r['fx_replicas']]),
            'parsing_argmax_candidate': stats(gather(rs, lambda r: r['fx_candidate']['argmax_agreement'])),
            'parsing_dice_replicas': stats([x['dice'] for r in rs for x in r['fx_replicas']]),
            'parsing_dice_candidate': stats(gather(rs, lambda r: r['fx_candidate']['dice']))}
    aggregate = {
        'adaface_cos_replicas': stats([c for r in rows for c in r['ada_cos_replicas']]),
        'adaface_cos_candidate': stats([r['ada_cos_candidate'] for r in rows]),
        'landmark_mean_candidate': stats([r['fx_candidate']['lm_mean'] for r in rows]),
        'landmark_abs_candidate': stats([r['fx_candidate']['lm_max'] for r in rows]),
        'parsing_argmax_candidate': stats([r['fx_candidate']['argmax_agreement'] for r in rows]),
        'parsing_dice_candidate': stats([r['fx_candidate']['dice'] for r in rows]),
        'input_lsb_candidate_vs_reference_fx': stats([r['input_lsb_candidate_vs_reference_fx'] for r in rows]),
        'reference_adapter_selfcheck_max_abs': max(r['reference_adapter_embedding_selfcheck_max_abs'] for r in rows)}
    return {'rules': {'adaface': 'PASS iff min_i cos(r_i, c_i) >= min_{i,k} cos(r_i, r_ik)',
                      'landmarks': 'PASS iff max_i mean|c-r| <= max_{i,k} mean|r_k-r| and max_i max|c-r| <= max_{i,k} max|r_k-r|',
                      'parsing': 'PASS iff min_i argmax_agree(c, r) >= min_{i,k} argmax_agree(r_k, r) and '
                                 'min_i D09(c, r) >= min_{i,k} D09(r_k, r)'},
            'replicas_per_image': REPLICAS, 'seed_rule': "int(SHA256('M7C2A-R04-L2|' + dataset + '|' + sample_id + '|' + "
                                                         "str(k)).hexdigest()[:16], 16) % 2**63 (CPU generator)",
            'thresholds_noise_floor': floors, 'candidate': cand, 'gates': gates, 'pass': all(gates.values()),
            'per_dataset': per_dataset, 'aggregate': aggregate, 'images': len(rows), 'elapsed_s': elapsed,
            'opened_images': reader.opened, 'per_sample': rows}


# ----------------------------------------------------------------------------- gradient structure + memory
def teacher_gradient_and_memory(fx_adapter=None) -> dict:
    fx_adapter = fx_adapter or tp.facexformer_input
    fx = aux_models.FaceXFormerAdapter(TEACHERS / 'facexformer/code', TEACHERS / 'facexformer/ckpts/model.pt', device=DEV)
    ada = aux_models.AdaFaceAdapter(TEACHERS / 'adaface/code', TEACHERS / 'adaface/adaface_ir50_webface4m.ckpt', device=DEV)
    for m in (fx.model, ada.model):
        m.eval().requires_grad_(False)
    g = torch.Generator().manual_seed(20261004)
    out = {}
    # structure (N=1, non-saturated synthetic input)
    x = ((torch.rand(1, 3, 256, 256, generator=g) * 1.6 - 0.8).to(DEV)).requires_grad_(True)
    with torch.autocast('cuda', enabled=False):
        seg = fx.model(fx_adapter(x), None, torch.tensor([FX_PARSE], device=DEV))[7]
        lm = fx.model(fx_adapter(x), None, torch.tensor([FX_LM], device=DEV))[0]
        (seg.float().square().mean() + lm.float().square().mean()).backward()
    out['facexformer'] = {'x_grad_finite': bool(torch.isfinite(x.grad).all()), 'x_grad_abs_sum': float(x.grad.abs().sum()),
                          'teacher_param_grads_none': all(p.grad is None for p in fx.model.parameters()),
                          'teacher_requires_grad_any': any(p.requires_grad for p in fx.model.parameters())}
    x2 = x.detach().clone().requires_grad_(True)
    with torch.autocast('cuda', enabled=False):
        feat, _ = ada.model(tp.adaface_input(x2))
        F.normalize(feat, dim=1).sum().backward()
    out['adaface'] = {'x_grad_finite': bool(torch.isfinite(x2.grad).all()), 'x_grad_abs_sum': float(x2.grad.abs().sum()),
                      'teacher_param_grads_none': all(p.grad is None for p in ada.model.parameters()),
                      'teacher_requires_grad_any': any(p.requires_grad for p in ada.model.parameters())}
    with torch.no_grad():
        t_ref = fx.model(fx_adapter(x.detach()), None, torch.tensor([FX_PARSE], device=DEV))[7]
    out['x_t_targets_requires_grad'] = t_ref.requires_grad
    out['pass'] = (all(v['x_grad_finite'] and v['x_grad_abs_sum'] > 0 and v['teacher_param_grads_none'] and
                       not v['teacher_requires_grad_any'] for v in (out['facexformer'], out['adaface'])) and
                   not out['x_t_targets_requires_grad'])

    # memory at physical batch 4 with synthetic GPAT x_hat (teachers run sequentially, fp32, outside autocast)
    import torchvision
    from methods.gpat.artifact_encoder import imagenet_state_dict
    from methods.gpat.highpass import highpass
    r18 = torchvision.models.resnet18(weights=None)
    r18.load_state_dict(imagenet_state_dict(ASSETS['f_art_resnet18_imagenet1k_v1'][0]))
    f_art = torch.nn.Sequential(*list(r18.children())[:-1], torch.nn.Flatten()).to(DEV).eval().requires_grad_(False)
    spec = __import__('importlib.util').util.spec_from_file_location('m7c3q', ROOT / 'tools/m7c3_gpat_gpu_qualification.py')
    q = __import__('importlib.util').util.module_from_spec(spec)
    spec.loader.exec_module(q)

    def peak(fn):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        fn()
        torch.cuda.synchronize()
        return {'max_memory_allocated_GiB': round(torch.cuda.max_memory_allocated() / 2 ** 30, 3),
                'max_memory_reserved_GiB': round(torch.cuda.max_memory_reserved() / 2 ** 30, 3)}

    xb = (torch.rand(4, 3, 256, 256, generator=g) * 1.6 - 0.8).to(DEV)

    def fx_isolated():
        xh = xb.clone().requires_grad_(True)
        with torch.autocast('cuda', enabled=False):
            inp = fx_adapter(xh)
            seg = fx.model(inp, None, torch.tensor([FX_PARSE] * 4, device=DEV))[7]
            lmk = fx.model(inp, None, torch.tensor([FX_LM] * 4, device=DEV))[0]
            (seg.square().mean() + lmk.square().mean()).backward()

    def integrated():
        model = q.build('B3')
        model.train()
        xs, xt = q.uniform((4, 3, 256, 256), 1), q.uniform((4, 3, 256, 256), 2)
        with torch.autocast('cuda', dtype=torch.float16):
            o = model(xs, xt, scale_hf=0.15)
        with torch.autocast('cuda', enabled=False):
            with torch.no_grad():
                tr_inp = fx_adapter(xt)
                t_seg = fx.model(tr_inp, None, torch.tensor([FX_PARSE] * 4, device=DEV))[7]
                t_lm = fx.model(tr_inp, None, torch.tensor([FX_LM] * 4, device=DEV))[0]
                t_id = F.normalize(ada.model(tp.adaface_input(xt))[0], dim=1)
                t_art_s, t_art_t = f_art(highpass(xs)), f_art(highpass(xt))
            h_inp = fx_adapter(o.x_hat)
            h_seg = fx.model(h_inp, None, torch.tensor([FX_PARSE] * 4, device=DEV))[7]
            h_lm = fx.model(h_inp, None, torch.tensor([FX_LM] * 4, device=DEV))[0]
            h_id = F.normalize(ada.model(tp.adaface_input(o.x_hat))[0], dim=1)
            h_art = f_art(highpass(o.x_hat))
            from methods.gpat import losses
            loss = (losses.l_parse(t_seg, h_seg) + losses.l_lm(t_lm.view(-1, 68, 2), h_lm.view(-1, 68, 2)) +
                    losses.l_id(t_id, h_id) + losses.l_artcon(h_art, t_art_s, t_art_t))
        loss.backward()
        out['integrated_loss_finite'] = bool(torch.isfinite(loss))
        out['integrated_g_res_grad_finite'] = all(p.grad is None or bool(torch.isfinite(p.grad).all())
                                                  for p in model.g_res.parameters())
        out['integrated_teacher_param_grads_none'] = all(p.grad is None for m in (fx.model, ada.model, f_art)
                                                         for p in m.parameters())

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        out['memory'] = {'D_facexformer_candidate_isolated_N4': peak(fx_isolated),
                         'integrated_B3_plus_teachers_N4': peak(integrated), 'physical_batch': 4, 'oom': False}
    return out


# ----------------------------------------------------------------------------- attempt 2 (exact forward adapter)
def _q(values, pct):
    return float(np.percentile(np.asarray(values, dtype=np.float64), pct))


def level1_exact_gpu() -> dict:
    """The M7C2a 44-image synthetic corpus: exact-forward uint8 vs frozen PIL bytes, and normalized parity, on CUDA."""
    import importlib.util
    from PIL import Image
    spec = importlib.util.spec_from_file_location('m7c2a_q', ROOT / 'tools/m7c2a_gpat_cpu_qualification.py')
    q = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(q)
    fx_pre = aux_models.FaceXFormerAdapter.preprocess
    names, eq, diffs = [], 0, []
    holder = type('H', (), {'device': DEV})()
    for name, rgb in q.corpus().items():
        rgb = np.ascontiguousarray(rgb)
        ref_u8 = np.asarray(Image.fromarray(rgb).resize((tp.FX_INPUT, tp.FX_INPUT), Image.BICUBIC))
        x = torch.as_tensor(rgb, device=DEV).permute(2, 0, 1)[None].float() / 127.5 - 1.0
        u8 = tp.facexformer_uint8_exact(x)[0].permute(1, 2, 0).to(torch.uint8).cpu().numpy()
        eq += bool(np.array_equal(u8, ref_u8))
        names.append(name)
        diffs.append(float((tp.facexformer_input_exact(x) - fx_pre(holder, rgb)).abs().max()))
    return {'images': len(names), 'uint8_bitwise_equal': eq, 'normalized_max_abs': max(diffs),
            'pass': eq == len(names)}


def surrogate_backward() -> dict:
    """New exact-forward adapter backward vs the approved M7C2a clip-emulating VJP, identical upstream gradient."""
    g = torch.Generator().manual_seed(20261005)
    x = (torch.rand(4, 3, 256, 256, generator=g) * 1.6 - 0.8).to(DEV)
    up = torch.randn(4, 3, 224, 224, generator=g).to(DEV)
    a, b = x.clone().requires_grad_(True), x.clone().requires_grad_(True)
    (tp.facexformer_input_exact(a) * up).sum().backward()
    (tp.facexformer_input(b) * up).sum().backward()
    ga, gb = a.grad.double().flatten(), b.grad.double().flatten()
    return {'strategy': 'custom autograd.Function; backward re-evaluates teacher_preprocess.facexformer_input (M7C2a '
                        'clip-emulating surrogate) under enable_grad and returns torch.autograd.grad(VJP); not recursive',
            'max_abs': float((ga - gb).abs().max()), 'mean_abs': float((ga - gb).abs().mean()),
            'cosine': float(torch.dot(ga, gb) / (ga.norm() * gb.norm())), 'bitwise_equal': bool(torch.equal(a.grad, b.grad)),
            'finite': bool(torch.isfinite(a.grad).all()), 'nonzero': float(a.grad.abs().sum()) > 0,
            'digest_forward': hashlib.sha256(tp.facexformer_input_exact(x).cpu().numpy().tobytes()).hexdigest(),
            'digest_grad': hashlib.sha256(a.grad.cpu().numpy().tobytes()).hexdigest()}


def attempt2(attempt1) -> dict:
    """Same 192 TRAIN ids, same teachers, same seeds; FaceXFormer candidate = exact forward; attempt-1 floors verbatim."""
    from PIL import Image
    import torchvision.transforms.functional as TF
    sel = attempt1['selection']
    recomputed = select_train()
    if recomputed['selected_ids'] != sel['selected_ids'] or recomputed['selected_list_sha256'] != sel['selected_list_sha256']:
        raise SystemExit('selection differs from attempt 1: STOP')
    floors = dict(attempt1['level2']['thresholds_noise_floor'])          # verbatim; never recomputed
    seeds = [tp.level2_seed(r['dataset'], r['sample_id'], k) for r in sel['records'] for k in range(REPLICAS)]
    fx = aux_models.FaceXFormerAdapter(TEACHERS / 'facexformer/code', TEACHERS / 'facexformer/ckpts/model.pt', device=DEV)
    ada = aux_models.AdaFaceAdapter(TEACHERS / 'adaface/code', TEACHERS / 'adaface/adaface_ir50_webface4m.ckpt', device=DEV)
    for m in (fx.model, ada.model):
        m.eval().requires_grad_(False)
    reader = TrainFaceReader(sel['records'])
    faces, u8_equal, norm_abs = {}, 0, []
    for rec in sel['records']:                                            # phase A: input parity only
        rgb = reader.read(rec['sample_id'])
        faces[rec['sample_id']] = rgb
        ref_u8 = np.asarray(TF.resize(Image.fromarray(rgb), [tp.FX_INPUT, tp.FX_INPUT],
                                      interpolation=TF.InterpolationMode.BICUBIC))
        x = torch.as_tensor(rgb, device=DEV).permute(2, 0, 1)[None].float() / 127.5 - 1.0
        u8 = tp.facexformer_uint8_exact(x)[0].permute(1, 2, 0).to(torch.uint8).cpu().numpy()
        u8_equal += bool(np.array_equal(u8, ref_u8))
        norm_abs.append((tp.facexformer_input_exact(x) - fx.preprocess(rgb)).abs().flatten().cpu().numpy())
    allabs = np.concatenate(norm_abs)
    input_parity = {'images': len(faces), 'uint8_bitwise_equal': u8_equal, 'normalized_max_abs': float(allabs.max()),
                    'normalized_mean_abs': float(allabs.mean()), 'normalized_p99_abs': float(np.percentile(allabs, 99)),
                    'pass': u8_equal == len(faces)}
    out = {'adapter': tp.FACEXFORMER_ADAPTER_CLASS, 'selection_sha256': sel['selected_list_sha256'],
           'selection_reused_from_attempt_1': True, 'floors_from_attempt_1': floors, 'floors_recomputed': False,
           'replica_seed_list_sha256': hashlib.sha256(canonical(seeds)).hexdigest(), 'replicas_rerun': False,
           'input_parity': input_parity, 'opened_images': reader.opened}
    if not input_parity['pass']:
        out.update({'status': 'BLOCKED_R04_EXACT_FORWARD', 'pass': False})
        return out
    rows, lm_abs, seg_abs = [], [], []
    for rec in sel['records']:                                            # phase B: teacher outputs vs reference
        ds, sid = rec['dataset'], rec['sample_id']
        rgb = faces[sid]
        x = torch.as_tensor(rgb, device=DEV).permute(2, 0, 1)[None].float() / 127.5 - 1.0
        ref_fx = fx_outputs(fx.model, fx.preprocess(rgb))
        with torch.no_grad():
            cand_fx = fx_outputs(fx.model, tp.facexformer_input_exact(x))
        lm_abs.append(float((cand_fx['lm'] - ref_fx['lm']).abs().max()))
        seg_abs.append(float((cand_fx['parse'] - ref_fx['parse']).abs().max()))
        ref_ada = ada_embedding(ada.model, ada.preprocess(rgb))
        cand_ada = ada_embedding(ada.model, tp.adaface_input(x))
        rows.append({'dataset': ds, 'sample_id': sid, 'ada_cos_candidate': float(torch.dot(ref_ada, cand_ada)),
                     'fx_candidate': compare_fx(ref_fx, cand_fx)})
    del fx, ada
    torch.cuda.empty_cache()
    cand = {'adaface_cos': min(r['ada_cos_candidate'] for r in rows),
            'landmark_mean': max(r['fx_candidate']['lm_mean'] for r in rows),
            'landmark_abs': max(r['fx_candidate']['lm_max'] for r in rows),
            'parsing_argmax': min(r['fx_candidate']['argmax_agreement'] for r in rows),
            'parsing_dice': min(r['fx_candidate']['dice'] for r in rows if r['fx_candidate']['dice'] is not None)}
    gates = {'adaface': cand['adaface_cos'] >= floors['adaface_cos'],
             'landmarks': cand['landmark_mean'] <= floors['landmark_mean'] and cand['landmark_abs'] <= floors['landmark_abs'],
             'parsing': cand['parsing_argmax'] >= floors['parsing_argmax'] and cand['parsing_dice'] >= floors['parsing_dice']}
    out.update({'candidate': cand, 'gates': gates, 'pass': all(gates.values()),
                'status': 'PASS' if all(gates.values()) else 'BLOCKED_R04_LEVEL2',
                'direct_teacher_output_parity': {'landmark_max_abs': max(lm_abs), 'parsing_logit_max_abs': max(seg_abs),
                                                 'landmark_bitwise_all': all(v == 0.0 for v in lm_abs),
                                                 'parsing_bitwise_all': all(v == 0.0 for v in seg_abs)},
                'aggregate': {'adaface_cos_candidate': stats([r['ada_cos_candidate'] for r in rows]),
                              'landmark_mean_candidate': stats([r['fx_candidate']['lm_mean'] for r in rows]),
                              'landmark_abs_candidate': stats([r['fx_candidate']['lm_max'] for r in rows]),
                              'parsing_argmax_candidate': stats([r['fx_candidate']['argmax_agreement'] for r in rows]),
                              'parsing_dice_candidate': stats([r['fx_candidate']['dice'] for r in rows])},
                'per_sample': rows})
    return out


def main_attempt2(args):
    det = rc.apply_qualification_determinism(20261003, gpu=True)
    a1 = json.loads(Path(args.attempt1).read_text())
    inv = inventory()
    if not inv['all_match'] or {k: inv[k]['sha256'] for k in ASSETS} != {k: a1['inventory'][k]['sha256'] for k in ASSETS}:
        raise SystemExit('teacher assets differ from attempt 1: STOP')
    res = {'milestone': 'M7C3', 'kind': 'r04_level2_attempt_2', 'authority_commit': AUTHORITY, 'host': platform.node(),
           'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__, 'cuda': torch.version.cuda,
           'cudnn': torch.backends.cudnn.version(), 'executable': sys.executable, 'determinism': det, 'inventory': inv,
           'attempt_1_sha256': hashlib.sha256(Path(args.attempt1).read_bytes()).hexdigest(),
           'level1_exact_gpu': level1_exact_gpu(), 'surrogate_backward': surrogate_backward()}
    res['attempt_2'] = attempt2(a1)
    if res['attempt_2']['pass']:
        res['teacher_gradient'] = teacher_gradient_and_memory(tp.facexformer_input_exact)
    opened = res['attempt_2']['opened_images']
    res['access'] = {'images_opened': len(opened), 'all_train': all(o['split'] == 'TRAIN' for o in opened),
                     'opened_equals_selection': sorted(o['sample_id'] for o in opened) ==
                     sorted(r['sample_id'] for r in a1['selection']['records']), 'val_images_opened': 0,
                     'test_images_opened': 0}
    res['determinism_after'] = {'use_deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
                                'warn_only': torch.is_deterministic_algorithms_warn_only_enabled()}
    ok = (res['level1_exact_gpu']['pass'] and res['attempt_2']['pass'] and res['access']['all_train'] and
          res['access']['opened_equals_selection'] and res['surrogate_backward']['finite'] and
          res['surrogate_backward']['nonzero'] and res.get('teacher_gradient', {}).get('pass', False))
    res['status'] = 'PASS' if ok else res['attempt_2']['status']
    Path(args.out).write_text(json.dumps(res, indent=1, sort_keys=True) + '\n')
    a2 = res['attempt_2']
    print(json.dumps({'status': res['status'], 'level1': res['level1_exact_gpu'], 'input_parity': a2['input_parity'],
                      'surrogate_backward': {k: v for k, v in res['surrogate_backward'].items() if k != 'strategy'},
                      'candidate': a2.get('candidate'), 'floors': a2['floors_from_attempt_1'], 'gates': a2.get('gates'),
                      'direct': a2.get('direct_teacher_output_parity'),
                      'teacher_gradient_pass': res.get('teacher_gradient', {}).get('pass'),
                      'memory': res.get('teacher_gradient', {}).get('memory')}, indent=1))


def main_timing(args):
    """INFORMATIONAL only (not a gate): batch-4 CUDA timing of the exact adapter, the teacher forward with it, and the
    old clip-emulating adapter. Synthetic tensors; median / p90 / max over a fixed number of synchronized iterations."""
    rc.apply_qualification_determinism(20261003, gpu=True)
    fx = aux_models.FaceXFormerAdapter(TEACHERS / 'facexformer/code', TEACHERS / 'facexformer/ckpts/model.pt', device=DEV)
    fx.model.eval().requires_grad_(False)
    x = (torch.rand(4, 3, 256, 256, generator=torch.Generator().manual_seed(20261006)) * 2 - 1).to(DEV)
    tasks = torch.tensor([FX_PARSE] * 4, device=DEV)
    cases = {'A_exact_adapter_only': lambda: tp.facexformer_input_exact(x),
             'B_teacher_forward_with_exact_adapter': lambda: fx.model(tp.facexformer_input_exact(x), None, tasks),
             'C_old_clip_emulating_adapter_only': lambda: tp.facexformer_input(x)}
    out = {'batch': 4, 'gpu': torch.cuda.get_device_name(0), 'warmup': 10, 'iterations': 50, 'informational_only': True}
    with torch.no_grad():
        for name, fn in cases.items():
            for _ in range(10):
                fn()
            times = []
            for _ in range(50):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                fn()
                torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1000.0)
            out[name] = {'median_ms': float(np.median(times)), 'p90_ms': float(np.percentile(times, 90)),
                         'max_ms': float(np.max(times))}
    Path(args.out).write_text(json.dumps(out, indent=1, sort_keys=True) + '\n')
    print(json.dumps(out, indent=1))


def main_backward_digest(args):
    rc.apply_qualification_determinism(20261003, gpu=True)
    sb = surrogate_backward()
    Path(args.out).write_text(json.dumps({'digest_forward': sb['digest_forward'], 'digest_grad': sb['digest_grad']},
                                         sort_keys=True) + '\n')
    print(sb['digest_forward'], sb['digest_grad'])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True)
    ap.add_argument('--attempt2', dest='attempt1', help='attempt-1 evidence JSON: run attempt 2 with its floors')
    ap.add_argument('--backward-digest', action='store_true')
    ap.add_argument('--timing', action='store_true')
    args = ap.parse_args()
    if args.timing:
        return main_timing(args)
    if args.backward_digest:
        return main_backward_digest(args)
    if args.attempt1:
        return main_attempt2(args)
    det = rc.apply_qualification_determinism(20261003, gpu=True)
    inv = inventory()
    if not inv['all_match']:
        Path(args.out).write_text(json.dumps({'status': 'BLOCKED_ASSET', 'inventory': inv}, indent=1) + '\n')
        raise SystemExit('teacher asset mismatch: STOP before Level 2')
    sel = select_train()
    res = {'milestone': 'M7C3', 'kind': 'r04_level2_teacher_parity', 'authority_commit': AUTHORITY,
           'contract': 'configs/amendments/gpat_m7c2a_implementation_resolution.yaml resolutions.R-04.level2',
           'host': platform.node(), 'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
           'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(), 'executable': sys.executable,
           'determinism': det, 'inventory': inv, 'selection': sel}
    res['level2'] = level2(sel)
    res['teacher_gradient'] = teacher_gradient_and_memory()
    opened = res['level2']['opened_images']
    res['access'] = {'images_opened': len(opened), 'all_train': all(o['split'] == 'TRAIN' for o in opened),
                     'opened_equals_selection': sorted(o['sample_id'] for o in opened) ==
                     sorted(r['sample_id'] for r in sel['records']), 'val_images_opened': 0, 'test_images_opened': 0}
    res['determinism_after'] = {'use_deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
                                'warn_only': torch.is_deterministic_algorithms_warn_only_enabled()}
    res['status'] = 'PASS' if (res['level2']['pass'] and res['teacher_gradient']['pass'] and res['access']['all_train']
                                and res['access']['opened_equals_selection']) else 'FAIL'
    Path(args.out).write_text(json.dumps(res, indent=1, sort_keys=True) + '\n')
    print(json.dumps({'status': res['status'], 'gates': res['level2']['gates'], 'floors': res['level2']['thresholds_noise_floor'],
                      'candidate': res['level2']['candidate'], 'selected': sel['selected_counts'],
                      'selection_sha256': sel['selected_list_sha256'], 'gradient_pass': res['teacher_gradient']['pass'],
                      'memory': res['teacher_gradient']['memory']}, indent=1))


if __name__ == '__main__':
    main()
