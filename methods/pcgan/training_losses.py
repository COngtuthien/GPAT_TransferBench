"""Numerical M6D4c objectives; owner-resolved CONTROLLED_ADAPTATION.

No data loading, model construction, optimizer, regularizer or checkpoint I/O.
"""
from .blur import blur_inputs

G_TERMS = ('L_rec', 'L_recblur', 'L_advrec', 'L_advmix', 'L_pat')
D_TERMS = ('L_D_real', 'L_D_rec', 'L_D_mix', 'L_D_patch_real', 'L_D_patch_fake')


def norm2_distance(a, b):
    """Unsquared Euclidean image norm, followed by the sample mean; no epsilon."""
    return (a - b).square().sum(dim=(1, 2, 3)).sqrt().mean()


def generator_losses(src, tgt, reconstruction, mixed, image_d, patch_d, crop):
    from torch.nn.functional import softplus
    bt, bm = blur_inputs(tgt, mixed)
    d_rec, d_mix = image_d(reconstruction), image_d(mixed)
    crops_ref, crops_mix = crop(src), crop(mixed)
    ref = patch_d.extract_features(crops_ref, aggregate=True)
    candidate = patch_d.extract_features(crops_mix, aggregate=False)
    p_mix = patch_d.discriminate_features(ref, candidate)
    terms = dict(L_rec=norm2_distance(src, reconstruction),
                 L_recblur=norm2_distance(bt, bm),
                 L_advrec=softplus(-d_rec).mean(),
                 L_advmix=softplus(-d_mix).mean(), L_pat=softplus(-p_mix).mean())
    terms['L_G_total'] = (terms['L_rec'] + terms['L_recblur'] + terms['L_advrec']
                          + terms['L_advmix'] + terms['L_pat'])
    return terms, dict(blur_target=bt, blur_mixed=bm, d_rec=d_rec, d_mix=d_mix,
                       crops_ref=crops_ref, crops_mix=crops_mix, p_mix=p_mix)


def discriminator_losses(src, tgt, reconstruction, mixed, image_d, patch_d, crop):
    from torch.nn.functional import softplus
    rec_detached, mix_detached = reconstruction.detach(), mixed.detach()
    d_src, d_tgt = image_d(src), image_d(tgt)
    d_rec, d_mix = image_d(rec_detached), image_d(mix_detached)
    crops_ref, crops_pos, crops_fake = crop(src), crop(src), crop(mix_detached)
    ref = patch_d.extract_features(crops_ref, aggregate=True)
    pos = patch_d.extract_features(crops_pos, aggregate=False)
    fake = patch_d.extract_features(crops_fake, aggregate=False)
    p_real = patch_d.discriminate_features(ref, pos)
    p_fake = patch_d.discriminate_features(ref, fake)
    terms = dict(L_D_real=0.5 * (softplus(-d_src).mean() + softplus(-d_tgt).mean()),
                 L_D_rec=softplus(d_rec).mean(), L_D_mix=softplus(d_mix).mean(),
                 L_D_patch_real=softplus(-p_real).mean(),
                 L_D_patch_fake=softplus(p_fake).mean())
    terms['L_D_image'] = (1.0 * terms['L_D_real'] + 0.5 * terms['L_D_rec']
                          + 0.5 * terms['L_D_mix'])
    terms['L_D_patch'] = terms['L_D_patch_real'] + terms['L_D_patch_fake']
    terms['L_D_total'] = terms['L_D_image'] + terms['L_D_patch']
    return terms, dict(d_src=d_src, d_tgt=d_tgt, d_rec=d_rec, d_mix=d_mix,
                       crops_ref=crops_ref, crops_pos=crops_pos, crops_fake=crops_fake,
                       p_real=p_real, p_fake=p_fake,
                       rec_detached=rec_detached, mix_detached=mix_detached)
