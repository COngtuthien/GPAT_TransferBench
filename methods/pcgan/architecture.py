"""Auditable mappings to A2's executable architecture; no model construction."""
from methods.common.learned import authoritative, mapping, PreparationError
from .source import symbol_evidence, literal_option


def architecture_mapping(config, source):
    cfg = authoritative(config)
    arch = cfg['architecture_at_256']
    resolution = cfg['training']['input_resolution']
    spatial_size = resolution // (2 ** arch['netE_num_downsampling_sp'])
    shape = [arch['spatial_code_ch'], spatial_size, spatial_size]
    if ' x '.join(map(str, shape)) != arch['z_pat']:
        raise PreparationError('frozen spatial-code geometry mismatch')
    refs = {
        'encoder': ('encoder.py', 'StyleGAN2ResnetEncoder', 'forward'),
        'spatial_code_path': ('encoder.py', 'StyleGAN2ResnetEncoder', '__init__'),
        'generator': ('generator.py', 'StyleGAN2ResnetGenerator', 'forward'),
        'generator_construction': ('generator.py', 'StyleGAN2ResnetGenerator', '__init__'),
        'modulation_demodulation': ('stylegan2_layers.py', 'ModulatedConv2d', 'forward'),
        'modulated_convolution': ('stylegan2_layers.py', 'ModulatedConv2d', '__init__'),
        'styled_conv': ('stylegan2_layers.py', 'StyledConv', '__init__'),
        'rgb_demodulation_exception': ('stylegan2_layers.py', 'ToRGB', '__init__'),
        'discriminator': ('discriminator.py', 'StyleGAN2Discriminator', '__init__'),
        'discriminator_basis': ('stylegan2_layers.py', 'Discriminator', 'forward'),
        'patch_discriminator': ('patch_discriminator.py', 'StyleGAN2PatchDiscriminator', '__init__'),
    }
    evidence = {k: symbol_evidence(source, 'models/networks/' + f, cls, fn)
                for k, (f, cls, fn) in refs.items()}
    defaults = {}
    for filename, names in {
        'options/__init__.py': ['netE', 'netG', 'netD', 'netPatchD', 'num_classes', 'use_antialias'],
        'models/networks/encoder.py': ['netE_scale_capacity', 'netE_num_downsampling_gl', 'netE_nc_steepness'],
        'models/networks/generator.py': ['netG_scale_capacity', 'netG_num_base_resnet_layers', 'netG_use_noise', 'netG_resnet_ch'],
        'models/networks/discriminator.py': ['netD_scale_capacity'],
        'models/networks/patch_discriminator.py': ['netPatchD_scale_capacity', 'netPatchD_max_nc',
                                                  'patch_size', 'max_num_tiles', 'patch_random_transformation'],
        'models/swapping_autoencoder_model.py': ['global_code_ch', 'patch_min_scale', 'patch_max_scale',
                                                'patch_num_crops', 'patch_use_aggregation'],
    }.items():
        for name in names:
            defaults[name] = {'value': literal_option(source, filename, name),
                              'source_file': filename, 'provenance': 'A2_PINNED_EXECUTABLE_ARCHITECTURE'}
    return {'basis': cfg['source']['executable_architecture_basis'],
            'resolution': resolution, 'z_pat_shape': shape,
            'output_shape_chw': [3, resolution, resolution],
            'output_contract': cfg['output'],
            'settings': mapping(cfg, {
                'opt.crop_size': 'training.input_resolution',
                'opt.netE_num_downsampling_sp': 'architecture_at_256.netE_num_downsampling_sp',
                'opt.spatial_code_ch': 'architecture_at_256.spatial_code_ch',
            }, 'models/networks/encoder.py; generator.py; discriminator.py'),
            'inherited_source_defaults': defaults, 'source_evidence': evidence,
            'modulation': 'StyleGAN2-style modulation/demodulation',
            'to_rgb': 'Pinned ToRGB keeps demodulate=False; do not force demodulation on this head',
            'stylegan_v1_adain_substitution': False,
            'source_training_loop_used': False,
            'source_training_loop_note': 'Architecture reuse only. Upstream swap requires even image batches '
                'and its loss/optimizer loop is not the frozen PCGAN objective.',
            'model_constructed': False}
