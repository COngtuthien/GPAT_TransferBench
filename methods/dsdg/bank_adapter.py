"""E06b DSDG-NATIVE / E06c DSDG-BIN-IDFREE inference-only bank adapter (learned_bank_generation_contract.yaml).

Official inference = FaceX-Zoo addition_module/DSDG/generated.py (pinned 16b793a7):
  define_G(hdim, attack_type) -> netG = DataParallel(Decoder_s(hdim=128)) (:40); load_model(netG, pre_model) (:41);
  netG.eval() (:43); per batch (:46): noise = randn-like(100, hdim) identity noise (:47), then noise_s (:48);
  z = cat(noise_s, noise, noise) (:49); fake = netG(z); spoof = fake[:, 0:3] (:54);
  np.transpose((255 * spoof[i]).astype('uint8'), (1, 2, 0)) (:63).
Approved bank changes (DSDG-Q1/Q2/Q4, C-Q1, C-Q2; nothing else):
  the stream is drawn from g = torch.Generator().manual_seed(bank_seed) with torch.randn(100, 128, generator=g) in the
  official order; the first N samples are kept in stream order and the discarded tail is counted; only the spoof half
  is kept; rows are '<method>_S<seed>_<index:06d>' with null pair lineage; PNG instead of JPEG; cudnn.benchmark False
  and deterministic True (the official benchmark=True is overridden as a determinism control only); FP32, no TF32.
No optimizer, gradient or checkpoint write exists here (methods.common.learned_bank.forbid_training_torch).
"""
import hashlib
from pathlib import Path

import numpy as np

from methods.common import learned_bank as lb

SOURCE_REL = 'third_party/source_cache/facexzoo/addition_module/DSDG'
UPSTREAM_MODULES = ('networks', 'misc.util')
UPSTREAM_ROOTS = ('networks', 'misc', 'data', 'train_generator')
NETG_PARAMETERS = (19_887_494, 21)
HDIM = lb.DSDG_HDIM
ATTACK_TYPE = {'E06b': 2, 'E06c': 1}          # frozen training.attack_type of each method config (define_G argument)


def require(ok, message):
    if not ok:
        raise lb.LearnedBankError('DSDG bank: ' + message)


def source_root(repo_root=None):
    return Path(repo_root or lb.ROOT) / SOURCE_REL


def verify_source_files(repo_root=None):
    """SHA256 of every executed/cited pinned DSDG file equals methods/common/learned_bank_sources.json."""
    return lb.verify_pinned_sources('DSDG', repo_root)


def configure_determinism(torch, *, gpu):
    """C-Q2: FP32, TF32 off, cuDNN deterministic, benchmark off, highest matmul precision, no autocast."""
    import os
    if gpu:
        require(os.environ.get('NVIDIA_TF32_OVERRIDE') == '0', 'NVIDIA_TF32_OVERRIDE=0 required')
        require(os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'CUBLAS_WORKSPACE_CONFIG=:4096:8 required')
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision('highest')
    require(not torch.is_autocast_enabled('cpu') and (not gpu or not torch.is_autocast_enabled('cuda')), 'autocast')
    return {'default_dtype': str(torch.get_default_dtype()), 'matmul_tf32': False, 'cudnn_tf32': False,
            'cudnn_benchmark': False, 'cudnn_deterministic': True, 'matmul_precision': 'highest', 'autocast': False,
            'official_cudnn_benchmark_overridden': True}


def official_load_model(torch, util, model, path):
    """Call the unchanged official misc/util.py::load_model. torch >= 2.6 defaults torch.load to weights_only=True,
    which cannot unpickle the official {'epoch', 'model': DataParallel} checkpoint, so only for this call torch.load
    is given weights_only=False explicitly (the file SHA256 was verified first). Fail-closed addition: every netG
    state entry must come from the checkpoint with identical bytes (the official filter could otherwise skip keys)."""
    loaded = {}
    original = torch.load

    def official_torch_load(f, *args, **kwargs):
        kwargs.setdefault('weights_only', False)
        obj = original(f, *args, **kwargs)
        loaded['obj'] = obj
        return obj
    torch.load = official_torch_load
    try:
        util.load_model(model, str(path))
    finally:
        torch.load = original
    ckpt_state = loaded['obj']['model'].state_dict()
    state = model.state_dict()
    require(set(state) <= set(ckpt_state), 'checkpoint lacks netG keys: %s' % sorted(set(state) - set(ckpt_state))[:3])
    for k, v in state.items():
        require(tuple(v.shape) == tuple(ckpt_state[k].shape) and torch.equal(v.cpu(), ckpt_state[k].cpu()),
                'netG entry %s differs from the checkpoint' % k)
    return {'epoch': loaded['obj'].get('epoch'), 'checkpoint_keys': len(ckpt_state), 'netG_keys': len(state),
            'all_netG_entries_from_checkpoint': True, 'torch_load_weights_only': False,
            'official_function': 'misc/util.py::load_model'}


def parameter_report(module):
    params = list(module.parameters())
    return (sum(p.numel() for p in params), len(params))


class DSDGBankAdapter:
    """Yields one GenResult per stream index, in order, from the approved seeded official batch stream."""

    def __init__(self, torch, method_id, seed, netG, device, *, is_fixture, checkpoint_identity_sha256,
                 contract_sha256, config_sha256, identity_extra):
        require(method_id in lb.DSDG_METHODS, 'E06b or E06c only')
        self.torch, self.method_id, self.netG, self.device = torch, method_id, netG, device
        self.bank_seed = seed
        self.generator_seed = seed
        self.is_fixture = is_fixture
        self.checkpoint_identity_sha256 = checkpoint_identity_sha256
        self.contract_sha256, self.config_sha256 = contract_sha256, config_sha256
        self.stream_seed = lb.dsdg_bank_seed(method_id, seed)
        self._identity = dict(identity_extra)
        self.batches_run = self.forward_calls = self.discarded = 0
        self.plan = None
        self.noise_digest = hashlib.sha256()
        require(not netG.training, 'netG must be in eval mode (generated.py:43)')
        require(parameter_report(netG) == NETG_PARAMETERS, 'netG parameter count %r' % (parameter_report(netG),))

    def identity(self):
        return dict(self._identity, method_id=self.method_id, bank_seed=self.bank_seed,
                    generator_seed=self.generator_seed, stream_seed=self.stream_seed, is_fixture=self.is_fixture,
                    checkpoint_identity_sha256=self.checkpoint_identity_sha256, config_sha256=self.config_sha256,
                    contract_sha256=self.contract_sha256, device=str(self.device),
                    netG_parameters=list(parameter_report(self.netG)))

    def latent_batch(self, g):
        """generated.py:47-49 with the approved seeded generator (DSDG-Q1)."""
        torch = self.torch
        noise = torch.randn(100, HDIM, generator=g)
        noise_s = torch.randn(100, HDIM, generator=g)
        noise = torch.cat((noise_s, noise, noise), dim=1)
        return noise

    def generate(self, population, reader=None):
        torch = self.torch
        require(reader is None, 'DSDG reads no face image')
        require(population.method_id == self.method_id and population.seed == self.bank_seed, 'population identity')
        n = len(population.rows)
        self.plan = lb.dsdg_stream_plan(n)
        g = torch.Generator().manual_seed(self.stream_seed)
        index = 0
        for _ in range(self.plan['batches']):
            z = self.latent_batch(g)
            self.noise_digest.update(z.numpy().tobytes())
            z = z.to(self.device)
            with torch.no_grad():
                fake = self.netG(z)
            self.forward_calls += 1
            self.batches_run += 1
            require(tuple(fake.shape) == (100, 6, lb.CANONICAL, lb.CANONICAL) and fake.dtype == torch.float32,
                    'netG output shape/dtype %r %r' % (tuple(fake.shape), fake.dtype))
            spoof = fake[:, 0:3, :, :].data.cpu().numpy()                     # generated.py:54 (spoof half only)
            del fake
            for i in range(spoof.shape[0]):
                if index >= n:
                    self.discarded += 1
                    continue
                row = population.rows[index]
                require(row.index == index, 'stream index order')
                save_img = spoof[i, :, :, :]
                finite = bool(np.isfinite(save_img).all())
                pixels = np.transpose((255 * save_img).astype('uint8'), (1, 2, 0)) if finite else None  # :63
                yield lb.GenResult(row=row, pixels=pixels, raw_finite=finite,
                                   raw_min=float(np.nanmin(save_img)) if finite else None,
                                   raw_max=float(np.nanmax(save_img)) if finite else None,
                                   failure_reason=None if finite else 'NONFINITE_GENERATOR_OUTPUT',
                                   noise_seed=self.stream_seed)
                index += 1
        require(index == n and self.discarded == self.plan['discarded_final_batch_samples'], 'stream accounting')

    def finalize(self):
        return {'stream_plan': self.plan, 'batches_run': self.batches_run, 'forward_calls': self.forward_calls,
                'discarded_final_batch_samples': self.discarded, 'stream_seed': self.stream_seed,
                'latent_stream_sha256': self.noise_digest.hexdigest(),
                'live_half_stored': False, 'retained_half': 'spoof channels 0:3'}


# ------------------------------------------------------------------ constructors
def load_real(torch, method_id, seed, run_root, *, device='cuda', repo_root=None):
    """S7 / production: the approved epoch-200 netG of (method, seed) on the GPU, via the official define_G/load_model.
    Hash-verifies config, pinned source and checkpoint first; returns (adapter, checkpoint record)."""
    from methods.common.upstream import upstream_modules
    _, contract_sha = lb.load_generation_contract()
    cfg = lb.verify_config_identity(method_id)
    src = verify_source_files(repo_root)
    ckpt = lb.verify_checkpoint(method_id, seed, run_root)
    det = configure_determinism(torch, gpu=True)
    require(torch.cuda.is_available(), 'the approved inference device (CUDA) is required')
    with upstream_modules(source_root(repo_root), UPSTREAM_MODULES, UPSTREAM_ROOTS) as modules:
        networks, util = modules['networks'], modules['misc.util']
        netE_nir, netE_vis, netG, netCls = networks.define_G(hdim=HDIM, attack_type=ATTACK_TYPE[method_id])  # :40
        require(type(netG).__name__ == 'DataParallel' and type(netG.module).__name__ == 'Decoder_s', 'netG class')
        load = official_load_model(torch, util, netG, ckpt['files']['pth']['path'])                          # :41
        netG.eval()                                                                                            # :43
    del netE_nir, netE_vis, netCls
    torch.cuda.empty_cache()
    guards = lb.forbid_training_torch(torch)
    adapter = DSDGBankAdapter(torch, method_id, seed, netG, torch.device(device), is_fixture=False,
                              checkpoint_identity_sha256=ckpt['checkpoint_identity_sha256'],
                              contract_sha256=contract_sha, config_sha256=cfg['sha256'],
                              identity_extra={'checkpoint': ckpt, 'source': src, 'determinism': det, 'load': load,
                                              'training_guards': guards, 'framework': 'torch ' + torch.__version__})
    return adapter, ckpt


def write_fixture_checkpoint(torch, method_id, fixture_seed, fixture_dir, *, repo_root=None):
    """CPU qualification only, before any training guard is installed: a random-init Decoder_s from the pinned source
    (fixture weights, never scientific) saved in the official {'epoch', 'model': DataParallel} payload shape."""
    from methods.common.upstream import upstream_modules
    path = Path(fixture_dir) / ('%s_fixture_netG_seed%d.pth' % (method_id, fixture_seed))
    require(not path.exists(), 'fixture checkpoint exists')
    with upstream_modules(source_root(repo_root), UPSTREAM_MODULES, UPSTREAM_ROOTS) as modules:
        torch.manual_seed(fixture_seed)
        writer = torch.nn.DataParallel(modules['networks'].Decoder_s(hdim=HDIM))   # CPU: forward = module
        torch.save({'epoch': 200, 'model': writer}, str(path))
    return {'path': str(path), 'sha256': lb.sha256_file(path), 'size': path.stat().st_size,
            'mtime_ns': path.stat().st_mtime_ns, 'label': 'FIXTURE_RANDOM_INIT_NOT_SCIENTIFIC'}


def build_fixture(torch, method_id, fixture_seed, fixture_ckpt, *, repo_root=None):
    """CPU qualification: load a fixture checkpoint through the unchanged official load_model into a CPU
    DataParallel(Decoder_s) (define_G itself moves modules to CUDA), eval mode, training guards on."""
    from methods.common.upstream import upstream_modules
    _, contract_sha = lb.load_generation_contract()
    cfg = lb.verify_config_identity(method_id)
    src = verify_source_files(repo_root)
    det = configure_determinism(torch, gpu=False)
    require(lb.sha256_file(fixture_ckpt['path']) == fixture_ckpt['sha256'], 'fixture checkpoint changed')
    with upstream_modules(source_root(repo_root), UPSTREAM_MODULES, UPSTREAM_ROOTS) as modules:
        networks, util = modules['networks'], modules['misc.util']
        netG = torch.nn.DataParallel(networks.Decoder_s(hdim=HDIM))
        load = official_load_model(torch, util, netG, fixture_ckpt['path'])
        netG.eval()
    guards = lb.forbid_training_torch(torch)
    adapter = DSDGBankAdapter(torch, method_id, fixture_seed, netG, torch.device('cpu'), is_fixture=True,
                              checkpoint_identity_sha256=lb.sha256_bytes(lb.canonical_json(
                                  {'pth': fixture_ckpt['sha256']}).encode()),
                              contract_sha256=contract_sha, config_sha256=cfg['sha256'],
                              identity_extra={'fixture_checkpoint': fixture_ckpt, 'source': src, 'determinism': det,
                                              'load': load, 'training_guards': guards,
                                              'framework': 'torch ' + torch.__version__})
    return adapter
