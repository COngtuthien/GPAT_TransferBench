"""M6D5b E06c exact training graph: pinned train_generator.py:87-182 semantics.

Transcribes the pinned per-iteration body (forward, seven loss terms, warmup
assembly) and the optimizer construction; the caller invokes zero_grad/backward/
step in pinned order (lines 180-182) so each phase can be measured. Pinned
`misc.util` functions (reparameterize, kl_loss, reconstruction_loss, rgb2gray)
are called, never re-implemented. PINNED_STATEMENTS lets a caller prove the
transcribed statements still match the verified source bytes.

No dataset, DataLoader, image reader, manifest, epoch loop or checkpoint writer
exists here. Torch is passed in by the caller; importing this module is static.
"""
from methods.common.learned import PreparationError

TRAIN_FILE = 'addition_module/DSDG/train_generator.py'
PHYSICAL_BATCH = 240
LABEL = 'SYNTHETIC_PHYSICAL_BATCH_240'
WARMUP_BELOW_EPOCH = 2      # train_generator.py:174 `if epoch < 2`
WARMUP_SCALE = 0.01         # train_generator.py:175-176
FROZEN_LAMBDAS = {'lambda_mmd': 50, 'lambda_ip': 1000, 'lambda_type': 10, 'lambda_ort': 1, 'lambda_pair': 0.0}
LOSS_TERMS = ('loss_rec', 'loss_kl', 'loss_mmd', 'loss_ip', 'loss_pair', 'loss_cls', 'loss_ort')
OPTIMIZER_OWNED = ('netE_nir', 'netE_vis', 'netG')
FORBIDDEN_EXECUTION = ('autocast', 'fp16', 'bf16', 'tf32', 'activation_checkpointing',
                       'cpu_offload', 'parameter_offload', 'reduced_resolution', 'microbatch_averaging')

# line -> exact stripped pinned statement; `args.lambda_*` / `args.lr` bind to the frozen config.
PINNED_STATEMENTS = {
    87: 'optimizer = optim.Adam(list(netE_nir.parameters()) + list(netE_vis.parameters()) + list(netG.parameters()),',
    88: 'lr=args.lr)',
    91: 'criterion_type = torch.nn.CrossEntropyLoss().cuda()',
    92: 'criterionL2 = torch.nn.MSELoss().cuda()',
    105: 'netE_nir.train()', 106: 'netE_vis.train()', 107: 'netG.train()', 108: 'netIP.eval()',
    118: 'img = torch.cat((img_nir, img_vis), 1)',
    121: 'mu_nir, logvar_nir, mu_a, logvar_a = netE_nir(img_nir)',
    122: 'mu_vis, logvar_vis = netE_vis(img_vis)',
    125: 'z_cls = reparameterize(mu_a, logvar_a)',
    126: 'z_nir = reparameterize(mu_nir, logvar_nir)',
    127: 'z_vis = reparameterize(mu_vis, logvar_vis)',
    130: 'pre_spoof = netCls(z_cls)',
    133: 'rec = netG(torch.cat((z_cls, z_nir, z_vis), dim=1))',
    136: 'loss_rec = reconstruction_loss(rec, img, True) / 2.0',
    137: 'loss_kl = (kl_loss(mu_nir, logvar_nir).mean() + kl_loss(mu_vis, logvar_vis).mean()',
    138: '+ kl_loss(mu_a, logvar_a).mean()) / 3.0',
    141: 'loss_mmd = args.lambda_mmd * torch.abs(z_nir.mean(dim=0) - z_vis.mean(dim=0)).mean()',
    144: 'loss_cls = args.lambda_type * criterion_type(pre_spoof, label_spoof)',
    147: 'loss_ort = args.lambda_ort * torch.abs((z_cls * z_nir).sum(dim=1).mean())',
    150: 'rec_nir = rec[:, 0:3, :, :]',
    151: 'rec_vis = rec[:, 3:6, :, :]',
    154: "img_nir = F.interpolate(img_nir, size=(128, 128), mode='bilinear')",
    155: "img_vis = F.interpolate(img_vis, size=(128, 128), mode='bilinear')",
    156: "rec_nir = F.interpolate(rec_nir, size=(128, 128), mode='bilinear')",
    157: "rec_vis = F.interpolate(rec_vis, size=(128, 128), mode='bilinear')",
    159: 'nir_fc = netIP(rgb2gray(img_nir))',
    160: 'vis_fc = netIP(rgb2gray(img_vis))',
    161: 'rec_nir_fc = netIP(rgb2gray(rec_nir))',
    162: 'rec_vis_fc = netIP(rgb2gray(rec_vis))',
    164: 'nir_fc = F.normalize(nir_fc, p=2, dim=1)',
    165: 'vis_fc = F.normalize(vis_fc, p=2, dim=1)',
    166: 'rec_nir_fc = F.normalize(rec_nir_fc, p=2, dim=1)',
    167: 'rec_vis_fc = F.normalize(rec_vis_fc, p=2, dim=1)',
    169: 'loss_ip = args.lambda_ip * (',
    170: 'criterionL2(rec_nir_fc, nir_fc.detach()) + criterionL2(rec_vis_fc, vis_fc.detach())) / 2.0',
    171: 'loss_pair = args.lambda_pair * criterionL2(rec_nir_fc, rec_vis_fc)',
    174: 'if epoch < 2:',
    175: 'loss = loss_rec + 0.01 * loss_kl + 0.01 * loss_mmd + 0.01 * loss_ip + 0.01 * loss_pair + \\',
    176: '0.01 * loss_cls + 0.01 * loss_ort',
    178: 'loss = loss_rec + loss_kl + loss_mmd + loss_ip + loss_pair + loss_cls + loss_ort',
    180: 'optimizer.zero_grad()',
    181: 'loss.backward()',
    182: 'optimizer.step()',
}


def verify_pinned_statements(source_text):
    """Return [] when every transcribed statement is verbatim at its pinned line."""
    lines = source_text.splitlines()
    return [{'line': n, 'expected': s, 'found': lines[n - 1].strip() if n <= len(lines) else None}
            for n, s in PINNED_STATEMENTS.items() if n > len(lines) or lines[n - 1].strip() != s]


def frozen_lambdas(config):
    """Bind `args.lambda_*` to the frozen E06c config; any drift is a STOP, not a fix."""
    losses = config['losses']
    lam = {k: losses[k] for k in FROZEN_LAMBDAS}
    if lam != FROZEN_LAMBDAS or float(lam['lambda_pair']) != 0.0:
        raise PreparationError('frozen E06c loss coefficients differ: ' + repr(lam))
    if config['training']['attack_type'] != 1 or config['optimizer']['learning_rate'] != 2e-4:
        raise PreparationError('frozen attack_type=1 / lr=2e-4 required')
    return lam


def execution_guard(adapter, *, physical_batch_size=PHYSICAL_BATCH, gradient_accumulation_steps=1,
                    replica_factor=1, **forbidden):
    """Exact physical batch 240 x 1 x 1 via the unchanged adapter; every workaround is refused."""
    used = sorted(k for k, v in forbidden.items() if v)
    unknown = sorted(set(forbidden) - set(FORBIDDEN_EXECUTION))
    if unknown:
        raise PreparationError('unknown execution option(s): ' + ', '.join(unknown))
    if used:
        raise PreparationError('M6D5b forbids OOM workaround(s): ' + ', '.join(used))
    batch = adapter.validate_batch(physical_batch_size=physical_batch_size,
                                   gradient_accumulation_steps=gradient_accumulation_steps,
                                   replica_factor=replica_factor)
    if (batch['physical_batch_size'], batch['gradient_accumulation_steps'], batch['replica_factor']) \
            != (PHYSICAL_BATCH, 1, 1):
        raise PreparationError('E06c requires physical batch 240, accumulation 1, replica 1')
    return batch


def build_optimizer(torch, netE_nir, netE_vis, netG, lr):
    """train_generator.py:87-88 verbatim: no netCls, no netIP, no extra Adam arguments."""
    return torch.optim.Adam(list(netE_nir.parameters()) + list(netE_vis.parameters()) + list(netG.parameters()),
                            lr=lr)


def criteria(torch):
    """train_generator.py:91-92 (default reductions: mean)."""
    return torch.nn.CrossEntropyLoss().cuda(), torch.nn.MSELoss().cuda()


def synthetic_pair(torch, batch, device):
    """Deterministic analytic in-memory FP32 inputs in [0,1]; every row and channel distinct.

    Row b uses its own spatial frequencies and phase, so per-sample statistics and
    batch means (MMD, orthogonality, reconstruction mean) see 240 different samples.
    No RNG, file, decoder or manifest is involved.
    """
    pi = 3.141592653589793
    t = torch.linspace(0, 1, 256, dtype=torch.float64, device=device)
    yy, xx = torch.meshgrid(t, t, indexing='ij')
    b = torch.arange(batch, dtype=torch.float64, device=device).view(batch, 1, 1, 1)
    c = torch.arange(3, dtype=torch.float64, device=device).view(1, 3, 1, 1)
    fx, fy = 1 + torch.remainder(b, 17), 1 + torch.remainder(b * 7, 13)
    phase = 2 * pi * b / batch + c
    spoof = 0.5 + 0.5 * torch.sin(fx * pi * xx + phase) * torch.cos(fy * pi * yy - 0.7 * c)
    live = 0.5 + 0.5 * torch.cos(fy * pi * (xx + yy) / 2 + 1.3 * phase) * torch.sin(fx * pi * xx * yy + c + 0.5)
    return spoof.to(torch.float32).contiguous(), live.to(torch.float32).contiguous()


def training_forward(torch, F, util, nets, img_nir, img_vis, label_spoof, lam, criterion_type, criterionL2,
                     mark=lambda phase: None):
    """train_generator.py:118-171, statement for statement (nir = spoof, vis = live).

    Returns every loop local the pinned body keeps alive through backward.
    `mark` only names the phase for OOM attribution; it never touches tensors.
    """
    netE_nir, netE_vis, netG, netCls, netIP = nets
    reparameterize, kl_loss, reconstruction_loss, rgb2gray = (
        util.reparameterize, util.kl_loss, util.reconstruction_loss, util.rgb2gray)
    mark('forward_encoders')
    img = torch.cat((img_nir, img_vis), 1)
    mu_nir, logvar_nir, mu_a, logvar_a = netE_nir(img_nir)
    mu_vis, logvar_vis = netE_vis(img_vis)
    z_cls = reparameterize(mu_a, logvar_a)
    z_nir = reparameterize(mu_nir, logvar_nir)
    z_vis = reparameterize(mu_vis, logvar_vis)
    mark('forward_classifier_generator')
    pre_spoof = netCls(z_cls)
    rec = netG(torch.cat((z_cls, z_nir, z_vis), dim=1))
    mark('loss_construction')
    loss_rec = reconstruction_loss(rec, img, True) / 2.0
    loss_kl = (kl_loss(mu_nir, logvar_nir).mean() + kl_loss(mu_vis, logvar_vis).mean()
               + kl_loss(mu_a, logvar_a).mean()) / 3.0
    loss_mmd = lam['lambda_mmd'] * torch.abs(z_nir.mean(dim=0) - z_vis.mean(dim=0)).mean()
    loss_cls = lam['lambda_type'] * criterion_type(pre_spoof, label_spoof)
    loss_ort = lam['lambda_ort'] * torch.abs((z_cls * z_nir).sum(dim=1).mean())
    mark('lightcnn_path')
    rec_nir = rec[:, 0:3, :, :]
    rec_vis = rec[:, 3:6, :, :]
    img_nir = F.interpolate(img_nir, size=(128, 128), mode='bilinear')
    img_vis = F.interpolate(img_vis, size=(128, 128), mode='bilinear')
    rec_nir = F.interpolate(rec_nir, size=(128, 128), mode='bilinear')
    rec_vis = F.interpolate(rec_vis, size=(128, 128), mode='bilinear')
    nir_fc = netIP(rgb2gray(img_nir))
    vis_fc = netIP(rgb2gray(img_vis))
    rec_nir_fc = netIP(rgb2gray(rec_nir))
    rec_vis_fc = netIP(rgb2gray(rec_vis))
    nir_fc = F.normalize(nir_fc, p=2, dim=1)
    vis_fc = F.normalize(vis_fc, p=2, dim=1)
    rec_nir_fc = F.normalize(rec_nir_fc, p=2, dim=1)
    rec_vis_fc = F.normalize(rec_vis_fc, p=2, dim=1)
    mark('loss_construction_ip')
    loss_ip = lam['lambda_ip'] * (
            criterionL2(rec_nir_fc, nir_fc.detach()) + criterionL2(rec_vis_fc, vis_fc.detach())) / 2.0
    loss_pair = lam['lambda_pair'] * criterionL2(rec_nir_fc, rec_vis_fc)
    return dict(img=img, mu_nir=mu_nir, logvar_nir=logvar_nir, mu_a=mu_a, logvar_a=logvar_a,
                mu_vis=mu_vis, logvar_vis=logvar_vis, z_cls=z_cls, z_nir=z_nir, z_vis=z_vis,
                pre_spoof=pre_spoof, rec=rec, img_nir=img_nir, img_vis=img_vis, rec_nir=rec_nir,
                rec_vis=rec_vis, nir_fc=nir_fc, vis_fc=vis_fc, rec_nir_fc=rec_nir_fc, rec_vis_fc=rec_vis_fc,
                loss_rec=loss_rec, loss_kl=loss_kl, loss_mmd=loss_mmd, loss_cls=loss_cls, loss_ort=loss_ort,
                loss_ip=loss_ip, loss_pair=loss_pair)


def total_loss(t, epoch):
    """train_generator.py:174-178; `t` maps loss names to tensors or floats."""
    if epoch < WARMUP_BELOW_EPOCH:
        return t['loss_rec'] + 0.01 * t['loss_kl'] + 0.01 * t['loss_mmd'] + 0.01 * t['loss_ip'] + \
            0.01 * t['loss_pair'] + 0.01 * t['loss_cls'] + 0.01 * t['loss_ort']
    return t['loss_rec'] + t['loss_kl'] + t['loss_mmd'] + t['loss_ip'] + t['loss_pair'] + t['loss_cls'] + t['loss_ort']

