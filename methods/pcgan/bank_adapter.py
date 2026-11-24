"""E05 PCGAN inference-only bank adapter (learned_bank_generation_contract.yaml E05-Q1, Q4, C-Q1, C-Q2).

Per frozen pair (canonical global position order), exactly as the approved contract states:
  x_src = canonical_input(face(source_spoof_id)), x_tgt = canonical_input(face(target_live_id))   [(x/255-0.5)/0.5]
  sp_src, gl_src = E(x_src); sp_tgt, gl_tgt = E(x_tgt)               (training_runner.explicit_pair order)
  torch.manual_seed(pair_seed)                                      (E05-Q1 A_PER_PAIR_SEEDED, immediately before)
  mix = G(sp_src, gl_tgt)                                           (the single mix forward; spec 8.5)
  pixels = util.tensor2im(mix[0])                                   (pinned basis writer: clip((x+1)/2*255).astype(uint8))
The generator keeps the pinned StyleGAN2 NoiseInjection (fresh normal_ noise on every forward,
stylegan2_layers.py:342); its noise is fixed only by the per-pair seed. No zero noise, no augmentation, no
reconstruction forward is written. Faces are read through methods.common.learned_bank.TrainFaceReader (TRAIN
allowlist, O_NOATIME, M2 SHA256) and decoded with the E05 training decoder (methods.dsdg.runner_io._cv2_decode).

The real StyleGAN2 Encoder/Generator run only through the qualified CUDA runtime (custom fused_bias_act / upfirdn2d
kernels, methods.pcgan.production_runner.build_models); they cannot execute on CPU. CPU qualification therefore uses
`InterfaceFixture*` modules (not the PCGAN architecture) that draw their noise with the same pinned operation, to
qualify seeding placement, order independence, conversion and the bank path; the real forward is S7 (GPU) work.
"""
import ast
import json
from pathlib import Path

import numpy as np

from methods.common import learned_bank as lb

FORMAT = 'E05_BASELINE_STATE_V1'
ITERATION = 4000
TENSOR2IM_SOURCE = 'util/util.py'


def require(ok, message):
    if not ok:
        raise lb.LearnedBankError('E05 bank: ' + message)


def canonical_input(rgb):
    """methods.pcgan.production_io.canonical_input (the approved E05 input mapping), imported unchanged."""
    from methods.pcgan import production_io as pio
    return pio.canonical_input(rgb)


def decode_face(raw):
    """The E05 training decoder: cv2.imdecode then BGR->RGB (methods.dsdg.runner_io._cv2_decode)."""
    from methods.dsdg import runner_io as rio
    return rio._cv2_decode(raw)


def pinned_tensor2im(repo_root=None):
    """The unchanged pinned basis util.tensor2im, compiled from its exact source segment (the full util package
    imports visualizer/sklearn helpers unrelated to conversion). The generated runtime's util.py differs from the
    pinned file only in a CUDA-version check (environments/e05.compatibility_patch_manifest.json)."""
    import torch
    src = lb.verify_pinned_sources('PCGAN', repo_root)
    text = (Path(src['root']) / TENSOR2IM_SOURCE).read_text()
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == 'tensor2im')
    segment = ast.get_source_segment(text, node)
    namespace = {'np': np, 'torch': torch, 'tile_images': None}
    exec(compile(segment, str(Path(src['root']) / TENSOR2IM_SOURCE), 'exec'), namespace)
    return namespace['tensor2im'], lb.sha256_bytes(segment.encode())


def runtime_tensor2im(modules, generated_root, closure, pinned_segment_sha):
    """Owner ruling B1 (configs/amendments/e05_s7_b1_b2_ruling.yaml): verify the tensor2im the runtime will call at its
    defining module. `util` is a package; tensor2im must be defined in util.util, whose file must be the qualified
    generated-closure util/util.py (sha256 from the compatibility-patch manifest), the function's code must come from
    that file, and its source segment must equal the pinned basis segment. Returns (function, record)."""
    util = modules.get('util')
    fn = getattr(util, 'tensor2im', None)
    require(callable(fn), 'runtime util.tensor2im missing')
    name = getattr(fn, '__module__', None)
    require(name == 'util.util', 'runtime tensor2im is defined in %r, not util.util' % (name,))
    mod = modules.get(name)
    require(mod is not None and getattr(mod, 'tensor2im', None) is fn, 'util.tensor2im is not util.util.tensor2im')
    generated = Path(generated_root).resolve()
    path = Path(mod.__file__).resolve()
    require(path.is_relative_to(generated) and path.relative_to(generated).as_posix() == TENSOR2IM_SOURCE,
            'runtime tensor2im module %s is not the generated-closure %s' % (path, TENSOR2IM_SOURCE))
    require(Path(fn.__code__.co_filename).resolve() == path, 'tensor2im code does not come from %s' % path)
    raw = path.read_bytes()
    digest = lb.sha256_bytes(raw)
    require(closure.get(TENSOR2IM_SOURCE) == digest, 'runtime %s sha256 %s is not the qualified closure file'
            % (TENSOR2IM_SOURCE, digest))
    text = raw.decode('utf-8')
    node = next((n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == 'tensor2im'), None)
    require(node is not None, 'no tensor2im definition in %s' % TENSOR2IM_SOURCE)
    segment = lb.sha256_bytes(ast.get_source_segment(text, node).encode())
    require(segment == pinned_segment_sha, 'runtime tensor2im differs from pinned')
    return fn, {'module': name, 'file': TENSOR2IM_SOURCE, 'file_sha256': digest, 'segment_sha256': segment}


class PCGANBankAdapter:
    def __init__(self, torch, seed, encoder, generator, tensor2im, device, *, is_fixture, checkpoint_identity_sha256,
                 contract_sha256, config_sha256, identity_extra):
        self.torch, self.E, self.G, self.tensor2im, self.device = torch, encoder, generator, tensor2im, device
        self.method_id, self.bank_seed, self.generator_seed = 'E05', seed, seed
        self.is_fixture = is_fixture
        self.checkpoint_identity_sha256 = checkpoint_identity_sha256
        self.contract_sha256, self.config_sha256 = contract_sha256, config_sha256
        self._identity = dict(identity_extra)
        self.encoder_forwards = self.generator_forwards = 0
        require(not encoder.training and not generator.training, 'Encoder and Generator must be in eval mode')

    def identity(self):
        return dict(self._identity, method_id='E05', bank_seed=self.bank_seed, generator_seed=self.generator_seed,
                    is_fixture=self.is_fixture, checkpoint_identity_sha256=self.checkpoint_identity_sha256,
                    config_sha256=self.config_sha256, contract_sha256=self.contract_sha256, device=str(self.device))

    def mix(self, x_src, x_tgt, pair_seed):
        """One frozen pair: encoder on source then target, seed, single mix forward."""
        torch = self.torch
        with torch.no_grad():
            sp_src, gl_src = self.E(x_src)
            sp_tgt, gl_tgt = self.E(x_tgt)
            self.encoder_forwards += 2
            torch.manual_seed(pair_seed)
            out = self.G(sp_src, gl_tgt)
            self.generator_forwards += 1
        require(tuple(out.shape) == (1, 3, lb.CANONICAL, lb.CANONICAL) and out.dtype == torch.float32,
                'mix shape/dtype %r %r' % (tuple(out.shape), out.dtype))
        return out

    def _tensor(self, rgb):
        return self.torch.from_numpy(canonical_input(rgb))[None].to(self.device)

    def generate(self, population, reader):
        torch = self.torch
        require(population.method_id == 'E05' and population.seed == self.bank_seed, 'population identity')
        require(reader is not None, 'E05 needs the TRAIN face reader')
        for row in population.rows:
            raw_src, sha_src = reader.read(row.source_spoof_id)
            raw_tgt, sha_tgt = reader.read(row.target_live_id)
            pair_seed = lb.e05_pair_seed(self.bank_seed, row.pair_id)
            out = self.mix(self._tensor(decode_face(raw_src)), self._tensor(decode_face(raw_tgt)), pair_seed)
            raw = out[0]
            finite = bool(torch.isfinite(raw).all().item())
            pixels = self.tensor2im(raw) if finite else None
            if pixels is not None:
                require(pixels.dtype == np.uint8 and pixels.shape == (lb.CANONICAL, lb.CANONICAL, 3), 'tensor2im')
            yield lb.GenResult(row=row, pixels=pixels, raw_finite=finite,
                               raw_min=float(raw.min().item()) if finite else None,
                               raw_max=float(raw.max().item()) if finite else None,
                               failure_reason=None if finite else 'NONFINITE_GENERATOR_OUTPUT', noise_seed=pair_seed,
                               target_face_sha256=sha_tgt, source_face_sha256=sha_src)

    def finalize(self):
        return {'encoder_forwards': self.encoder_forwards, 'generator_forwards': self.generator_forwards,
                'reconstruction_forwards': 0, 'noise_rule': 'torch.manual_seed(e05_pair_seed) before G(sp_src, gl_tgt)'}


def configure_precision(torch):
    """The qualified E05 FP32 policy (methods.pcgan.production_runner.configure_precision), unchanged."""
    from methods.pcgan.production_runner import configure_precision as qualified
    return qualified(torch)


def load_checkpoint_state(torch, path, map_location):
    state = torch.load(str(path), map_location=map_location, weights_only=True)
    require(state.get('format') == FORMAT and state.get('benchmark_iteration') == ITERATION,
            'checkpoint format/iteration %r/%r' % (state.get('format'), state.get('benchmark_iteration')))
    require(set(state['models']) == {'Encoder', 'Generator', 'ImageD', 'PatchD'}, 'checkpoint components')
    return state


def checkpoint_structure(torch, path):
    """CPU, read-only: weights_only load and a schema/finiteness report of the E/G states (no model, no forward)."""
    state = load_checkpoint_state(torch, path, 'cpu')
    out = {}
    for name in ('Encoder', 'Generator'):
        sd = state['models'][name]
        out[name] = {'tensors': len(sd), 'elements': int(sum(v.numel() for v in sd.values())),
                     'all_finite': bool(all(torch.isfinite(v).all().item() for v in sd.values()
                                            if v.is_floating_point())),
                     'dtypes': sorted({str(v.dtype) for v in sd.values()})}
    return {'format': state['format'], 'benchmark_iteration': state['benchmark_iteration'], 'models': out}


def load_real(torch, seed, run_root, build_root, *, device='cuda'):
    """S7 / production: the qualified CUDA runtime with the approved iteration_4000 Encoder/Generator states."""
    from methods.pcgan.production_runner import build_models
    _, contract_sha = lb.load_generation_contract()
    cfg = lb.verify_config_identity('E05')
    src = lb.verify_pinned_sources('PCGAN')
    ckpt = lb.verify_checkpoint('E05', seed, run_root)
    precision = configure_precision(torch)
    require(torch.cuda.is_available(), 'the approved inference device (CUDA) is required')
    models, _crop, runtime = build_models(build_root)
    import sys
    util = sys.modules['util']
    _, pinned_segment_sha = pinned_tensor2im()
    generated_root = json.loads((lb.ROOT / 'environments/e05.compatibility_patch_manifest.json').read_text())[
        'generated_root']
    _, t2i = runtime_tensor2im(sys.modules, generated_root, runtime['build_closure']['generated_files_sha256'],
                               pinned_segment_sha)
    state = load_checkpoint_state(torch, ckpt['files']['pth']['path'], 'cuda')
    E, G = models['Encoder'], models['Generator']
    E.load_state_dict(state['models']['Encoder'], strict=True)
    G.load_state_dict(state['models']['Generator'], strict=True)
    E.eval()
    G.eval()
    for name in ('ImageD', 'PatchD'):
        models[name] = None
    del state
    torch.cuda.empty_cache()
    guards = lb.forbid_training_torch(torch)
    adapter = PCGANBankAdapter(torch, seed, E, G, util.tensor2im, torch.device(device), is_fixture=False,
                               checkpoint_identity_sha256=ckpt['checkpoint_identity_sha256'],
                               contract_sha256=contract_sha, config_sha256=cfg['sha256'],
                               identity_extra={'checkpoint': ckpt, 'source': src, 'precision': precision,
                                               'runtime_closure': {k: runtime.get(k) for k in ('build_closure',)},
                                               'training_guards': guards, 'tensor2im_segment_sha256':
                                               pinned_segment_sha, 'tensor2im_runtime': t2i,
                                               'framework': 'torch ' + torch.__version__})
    return adapter, ckpt


# ------------------------------------------------------------------ CPU interface fixture (NOT the PCGAN architecture)
def build_fixture(torch, fixture_seed):
    """Interface-equivalent fixture modules: E(x) -> (spatial [1,8,128,128], global [1,2048]); G draws per-pixel noise
    with the pinned NoiseInjection operation (image.new_empty(b, 1, h, w).normal_()). Random-init, CPU, labelled."""
    nn = torch.nn

    class InterfaceFixtureEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.sp = nn.Conv2d(3, 8, 4, 2, 1)
            self.gl = nn.Linear(3 * 16 * 16, 2048)

        def forward(self, x):
            return self.sp(x), self.gl(nn.functional.avg_pool2d(x, 16).flatten(1))

    class InterfaceFixtureGenerator(nn.Module):
        def __init__(self):
            super().__init__()
            self.mod = nn.Linear(2048, 8)
            self.up = nn.ConvTranspose2d(8, 3, 4, 2, 1)
            self.noise_weight = nn.Parameter(torch.tensor(0.5))

        def forward(self, sp, gl):
            h = sp * (1 + self.mod(gl)[:, :, None, None])
            batch, _, height, width = h.shape
            noise = h.new_empty(batch, 1, height, width).normal_()      # stylegan2_layers.py:342 operation
            return 1.5 * self.up(h + self.noise_weight * noise)         # unbounded, like an RGB head

    torch.manual_seed(fixture_seed)
    E, G = InterfaceFixtureEncoder().eval(), InterfaceFixtureGenerator().eval()
    return E, G
