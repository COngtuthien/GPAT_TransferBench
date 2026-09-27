"""M6D6i: E07c MAIN encoder-load integration + bounded training-graph qualification (QUALIFICATION ONLY).

Static tests run without Torch (fake objects stand in for tensors/modules; nothing is trained or loaded).
Evidence tests validate the recorded GPU evidence through the static preflight. History assertions compare the
state at the commit that ADDED this test (candidate: the worktree) with the M6D6h authority, so later milestones
can extend the repository without breaking them (no B1-style HEAD lock).
"""
import ast
import importlib.util
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from methods.common.config import load_method_config  # noqa: E402
from methods.common.learned import PreparationError  # noqa: E402
from methods.difffas import main_graph as mg  # noqa: E402
from methods.difffas import main_graph_qualification as q  # noqa: E402
from methods.difffas import aux_checkpoint as seam  # noqa: E402
from methods.difffas.source import validate_source, tree  # noqa: E402

AUTHORITY = '84023aed3959d7ffda380b7612b860652579b443'
THIS = 'tests/test_m6d6i_e07c_main_graph.py'
HARNESS = ROOT / 'methods/difffas/main_graph_qualification.py'
GRAPH = ROOT / 'methods/difffas/main_graph.py'
EVIDENCE = ROOT / 'outputs/audit/M6D6I_E07C_MAIN_GRAPH_QUALIFICATION.json'


def load_preflight():
    spec = importlib.util.spec_from_file_location('m6d6i_preflight_under_test',
                                                  ROOT / 'tools/m6d6i_e07c_main_graph_preflight.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git(*args):
    return subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True)


def m6d6i_commit():
    return git('log', '--diff-filter=A', '--format=%H', '-1', '--', THIS).stdout.decode().strip() or None


def at_m6d6i(rel):
    """File bytes at the M6D6i state: the commit that added this test, else the (candidate) worktree."""
    commit = m6d6i_commit()
    return git('show', f'{commit}:{rel}').stdout if commit else (ROOT / rel).read_bytes()


def at_authority(rel):
    return git('show', f'{AUTHORITY}:{rel}').stdout


def code_facts(path):
    """Called names and non-docstring string constants of a module (comments and docstrings excluded)."""
    module = ast.parse(Path(path).read_text())
    docstrings = {id(n.body[0].value) for n in ast.walk(module)
                  if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef)) and n.body and
                  isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
    calls = [ast.unparse(n.func) for n in ast.walk(module) if isinstance(n, ast.Call)]
    strings = [n.value for n in ast.walk(module) if isinstance(n, ast.Constant) and isinstance(n.value, str) and
               id(n) not in docstrings]
    return calls, strings


# ------------------------------------------------------------------ fakes (no torch)
class Log(list):
    def add(self, *event):
        self.append(event)


class FakeScalar:
    def __init__(self, log, name, value):
        self.log, self.name, self.value = log, name, value

    def backward(self):
        self.log.add('backward', self.name)

    def detach(self):
        return self

    def item(self):
        return self.value


class FakeVector:
    def __init__(self, log, name, value):
        self.log, self.name, self.value = log, name, value

    def mean(self):
        return FakeScalar(self.log, self.name, self.value)


class FakeTensor:
    def __init__(self, name, n=4):
        self.name, self.shape = name, (n, 3, 256, 256)

    def to(self, device):
        return self

    def cuda(self):
        return self


class FakeTorch:
    def __init__(self, log):
        self.log = log

    def randint(self, low, high, size, device=None):
        self.log.add('randint', low, high, size, device)
        return 'time_t'


class FakeDiffusion:
    def __init__(self, log):
        self.log, self.kwargs = log, None

    def training_losses(self, model, encoder, **kwargs):
        self.log.add('training_losses', model, encoder)
        self.kwargs = kwargs
        return {'loss': FakeVector(self.log, 'loss', 3.0), 'mse': FakeVector(self.log, 'mse', 1.0),
                'vb': FakeVector(self.log, 'vb', 2.0)}


class FakeStep:
    def __init__(self, log, name):
        self.log, self.name = log, name

    def zero_grad(self):
        self.log.add(self.name + '.zero_grad')

    def step(self):
        self.log.add(self.name + '.step')


def fake_conf(warmup=5000):
    class Sched:
        pass

    class Training:
        scheduler = Sched()

    class Diffusion:
        beta_schedule = {'n_timestep': 1000}

    class Conf:
        training = Training()
        diffusion = Diffusion()
    Conf.training.scheduler.warmup = warmup
    return Conf


def fake_args():
    from types import SimpleNamespace
    return SimpleNamespace(batch_size=4, guidance_prob=0.2, means_size=5, var_size=3, max_epochs=400, use_pair=False,
                           device='cuda', pretrain_path=None)


class TestM6D6iStatic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.torch_before = 'torch' in sys.modules
        cls.cfg = load_method_config('E07c')
        cls.source = validate_source(cls.cfg)
        cls.pf = load_preflight()

    # 01-08 main_graph source traceability -----------------------------------------------------
    def test_01_authority_is_ancestor(self):
        self.assertEqual(git('merge-base', '--is-ancestor', AUTHORITY, 'HEAD').returncode, 0)

    def test_02_replicas_statement_equal_to_pinned_source(self):
        ev = mg.verify_source_equivalence(self.source)
        self.assertEqual([r['replica'].split('::')[1] for r in ev['replicas']],
                         ['build_transform', 'build_dataloader', 'build_training_objects', 'epoch_progress',
                          'train_iteration', 'accumulate'])
        self.assertEqual(ev['replicas'][4]['statements_compared'], 23)
        self.assertEqual(ev['train_script_sha256'], 'c5eb42a1193f7708b9e972db0faac602c82ffe2dc10578b6c9013d81e26e4f84')
        self.assertFalse(ev['training_script_imported'])

    def test_03_scheduler_step_before_optimizer_step(self):
        ev = mg.verify_source_equivalence(self.source)
        self.assertEqual([s['line'] for s in ev['pinned_step_order']], [74, 75, 76, 77, 83])
        body = [ast.unparse(s) for s in mg._own_functions()[1]['train_iteration'].body]
        self.assertLess(body.index('scheduler.step()'), body.index('optimizer.step()'))
        self.assertLess(body.index('loss.backward()'), body.index('scheduler.step()'))
        self.assertLess(body.index('optimizer.zero_grad()'), body.index('loss.backward()'))

    def test_04_mutated_replica_is_rejected(self):
        """Swapping scheduler.step/optimizer.step (a 'fix') must fail the pinned comparison."""
        module = tree(self.source, 'FAS_train.py')
        fn = mg._own_functions()[1]['train_iteration']
        stmts = fn.body
        names = [ast.unparse(s) for s in stmts]
        i, j = names.index('scheduler.step()'), names.index('optimizer.step()')
        stmts[i], stmts[j] = stmts[j], stmts[i]
        lines = dict((r[0], r[2]) for r in mg.REPLICAS)['train_iteration']
        pinned = [mg._pinned_statement(module, line) for line in lines]
        self.assertNotEqual(mg._replica_statements(fn, 0)[:len(pinned)], pinned)

    def test_05_accumulate_is_pinned_function(self):
        module = tree(self.source, 'FAS_train.py')
        pinned = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == 'accumulate')
        self.assertEqual(ast.unparse(pinned), ast.unparse(mg._own_functions()[1]['accumulate']))
        self.assertEqual((pinned.lineno, pinned.end_lineno), (21, 26))

    def test_06_train_iteration_exact_call_order_and_arguments(self):
        log = Log()
        diffusion, opt, sched = FakeDiffusion(log), FakeStep(log, 'optimizer'), FakeStep(log, 'scheduler')
        batch = {'content': FakeTensor('content'), 'GT': FakeTensor('GT'), 'style_spoof': FakeTensor('style')}
        calls = []
        original = mg.accumulate
        mg.accumulate = lambda m1, m2, decay=0.9999: (calls.append((m1, m2, decay)), log.add('accumulate'))
        try:
            out = mg.train_iteration(FakeTorch(log), fake_conf(), fake_args(), batch, 0, 'MODEL', 'EMA', 'ENC',
                                     diffusion, FakeTensor('betas'), opt, sched, 'cuda')
        finally:
            mg.accumulate = original
        kinds = [e[0] for e in log]
        self.assertEqual(kinds, ['randint', 'training_losses', 'optimizer.zero_grad', 'backward',
                                 'scheduler.step', 'optimizer.step', 'accumulate'])
        self.assertEqual(log[0][1:], (0, 1000, (4,), 'cuda'))
        self.assertEqual(log[1][1:], ('MODEL', 'ENC'))
        k = diffusion.kwargs
        self.assertIs(k['x_start'], batch['GT'])
        self.assertEqual(k['cond_input'], [batch['content'], batch['style_spoof']])
        self.assertEqual((k['prob'], k['means_size'], k['var_size'], k['use_pair'], k['t']), (0.8, 5, 3, False, 'time_t'))
        self.assertEqual(calls, [('EMA', 'MODEL', 0)])            # iters=1 < warmup 5000 -> decay 0
        self.assertEqual((out['iters'], out['loss_list'], out['loss_mean_list'], out['loss_vb_list']),
                         (1, [3.0], [1.0], [2.0]))
        self.assertEqual(sorted(batch), ['GT', 'content', 'style_spoof'])   # no label key consumed

    def test_07_ema_decay_after_warmup_is_source_value(self):
        log, calls = Log(), []
        original = mg.accumulate
        mg.accumulate = lambda m1, m2, decay=0.9999: calls.append(decay)
        try:
            mg.train_iteration(FakeTorch(log), fake_conf(), fake_args(),
                               {'content': FakeTensor('c'), 'GT': FakeTensor('g'), 'style_spoof': FakeTensor('s')},
                               5000, 'M', 'E', 'X', FakeDiffusion(log), FakeTensor('b'), FakeStep(log, 'o'),
                               FakeStep(log, 's'), 'cuda')
        finally:
            mg.accumulate = original
        self.assertEqual(calls, [0.9999])

    def test_08_accumulate_semantics(self):
        class Data:
            def __init__(self, log, name):
                self.log, self.name = log, name

            def mul_(self, v):
                self.log.add('mul', self.name, v)
                return self

            def add_(self, other, alpha):
                self.log.add('add', self.name, other.name, alpha)
                return self

        class P:
            def __init__(self, log, name):
                self.data = Data(log, name)

        class M:
            def __init__(self, log, tag):
                self.p = [(n, P(log, tag + n)) for n in ('a', 'b')]

            def named_parameters(self):
                return iter(self.p)
        log = Log()
        mg.accumulate(M(log, 'ema.'), M(log, 'model.'), 0)
        self.assertEqual(list(log), [('mul', 'ema.a', 0), ('add', 'ema.a', 'model.a', 1), ('mul', 'ema.b', 0),
                                     ('add', 'ema.b', 'model.b', 1)])

    # 09-12 construction / args -----------------------------------------------------------------
    def test_09_build_training_objects_order_and_fresh_only(self):
        log = Log()

        class Model:
            def __init__(self, tag):
                self.tag = tag

            def to(self, device):
                log.add('to', self.tag, device)
                return self

            def parameters(self):
                return 'PARAMS'
        made = iter(('model', 'ema'))

        class ConfObj:
            def make_model(self):
                tag = next(made)
                log.add('make_model', tag)
                return Model(tag)

        class Maker:
            def __init__(self, name, value):
                self.name, self.value = name, value

            def make(self, *a):
                log.add(self.name, *a)
                return self.value

        class DC:
            class training:
                optimizer = Maker('optimizer.make', 'OPT')
                scheduler = Maker('scheduler.make', 'SCHED')

            class diffusion:
                beta_schedule = Maker('beta.make', 'BETAS')
        gaussian = lambda betas, predict_xstart: (log.add('gaussian', betas, predict_xstart), 'DIFF')[1]  # noqa: E731
        out = mg.build_training_objects(lambda: (log.add('get_model_conf'), ConfObj())[1], DC, gaussian, fake_args())
        self.assertEqual([e[0] for e in log], ['get_model_conf', 'make_model', 'to', 'get_model_conf', 'make_model', 'to',
                                               'optimizer.make', 'scheduler.make', 'beta.make', 'gaussian'])
        self.assertEqual(log[6], ('optimizer.make', 'PARAMS'))
        self.assertEqual(log[7], ('scheduler.make', 'OPT'))
        self.assertEqual(log[-1], ('gaussian', 'BETAS', False))
        self.assertEqual(out[2:], ('OPT', 'SCHED', 'BETAS', 'DIFF'))
        for bad in ({'pretrain_path': 'x.pt'}, {'use_pair': True}):
            args = fake_args()
            vars(args).update(bad)
            with self.assertRaises(PreparationError):
                mg.build_training_objects(None, None, None, args)

    def test_10_dataloader_and_transform_are_source_calls(self):
        seen = {}
        mg.build_dataloader(lambda *a, **kw: seen.update(args=a, kwargs=kw), 'DS', fake_args())
        self.assertEqual(seen, {'args': ('DS',), 'kwargs': {'batch_size': 4, 'shuffle': True}})   # num_workers default 0

        class T:
            def __getattr__(self, name):
                return lambda *a, **kw: (name, a, kw)
        self.assertEqual(mg.build_transform(T()), ('Compose', ([('Resize', ((256, 256),), {}), ('ToTensor', (), {}),
                                                                ('Normalize', ([0.5] * 3, [0.5] * 3), {})],), {}))

    def test_11_source_args_bound_to_frozen_config_and_pinned_defaults(self):
        args, info = mg.source_args(self.source, self.cfg)
        self.assertEqual(vars(args), vars(fake_args()))
        self.assertEqual(info['pinned_defaults']['use_pair'], True)       # A1 official flag overrides the default
        self.assertEqual(info['a1_overrides']['use_pair']['frozen'], False)
        self.assertEqual((args.batch_size, q.BATCH, self.cfg['training']['batch_size']), (4, 4, 4))

    def test_12_main_graph_is_io_free_and_seed_free(self):
        text = GRAPH.read_text()
        top = [n for n in ast.parse(text).body if isinstance(n, (ast.Import, ast.ImportFrom))]
        self.assertFalse({a.name.split('.')[0] for n in top for a in getattr(n, 'names', [])} & {'torch', 'numpy', 'PIL'})
        calls, strings = code_facts(GRAPH)
        for name in ('torch.save', 'torch.load', 'torch.manual_seed', 'random.seed', 'open', 'ddim_steps',
                     'load_state_dict', 'state_dict', 'save_image', 'print'):
            self.assertFalse([c for c in calls if c == name or c.endswith('.' + name) or c.endswith('.mkdir') or
                              c.endswith('.write') or c.endswith('.write_bytes') or c.endswith('.write_text')], name)
        for token in ('manifests/', '.parquet', 'faces_256', 'runs/', 'FAS_sample', 'split_v1'):
            self.assertFalse([v for v in strings if token in v], token)
        self.assertEqual([c for c in calls if c.endswith('read_bytes')], ['Path(__file__).read_bytes'])

    # 13-18 harness scope ---------------------------------------------------------------------
    def test_13_qualification_seed_is_not_scientific(self):
        self.assertEqual(q.SEED, 60607)
        self.assertNotIn(q.SEED, self.cfg['seeds']['experiment_seeds'])
        self.assertNotEqual(q.SEED, self.cfg['conditioning_encoder']['auxiliary_encoder_training_seed'])
        self.assertEqual(q.LAUNCH_ENV['PYTHONHASHSEED'], '60607')
        self.assertEqual(q.EXPERIMENT_SEED_REASON, 'QUALIFICATION_ONLY_NOT_A_SCIENTIFIC_RUN')
        text = HARNESS.read_text()
        for name in ('seed_adapter', 'apply_seed', 'seed_plan', 'validate_mode_seed', 'apply_framework_seed'):
            self.assertNotIn(name, text)

    def test_14_only_main_runner_encoder_seam(self):
        tree_ = ast.parse(HARNESS.read_text())
        calls = [ast.unparse(n.func) for n in ast.walk(tree_) if isinstance(n, ast.Call)]
        self.assertEqual(calls.count('ep.main_runner_encoder'), 1)
        for forbidden in ('torch.load', 'seam.read_verified', 'seam.load_verified_whole_module',
                          'seam.load_frozen_aux_encoder', 'read_verified', 'load_verified_whole_module',
                          'load_frozen_aux_encoder', 'pickle.load', 'pickle.loads', 'ep.consume_upstream_encoder_loader_rng'):
            self.assertNotIn(forbidden, calls, forbidden)
        self.assertIn("orig['load'](f, *args, **kwargs)", HARNESS.read_text())   # pass-through observer only
        call = next(n for n in ast.walk(tree_) if isinstance(n, ast.Call) and ast.unparse(n.func) == 'ep.main_runner_encoder')
        self.assertEqual([ast.unparse(a) for a in call.args], ['runtime_root', 'seam.OWNER_FROZEN_SHA256', 'cfg'])
        with_node = next(n for n in ast.walk(tree_) if isinstance(n, ast.With) and
                         ast.unparse(n.items[0].context_expr).startswith('ep.main_runner_encoder'))
        self.assertEqual(ast.unparse(with_node.body[0]), 'encoder.eval()')

    def test_15_no_save_sampling_visualization_resume_or_data(self):
        text = HARNESS.read_text()
        calls, strings = code_facts(HARNESS)
        for forbidden in ('torch.save', 'ddim_steps', 'dmod.ddim_steps', 'diffusion.p_sample_loop', 'save_image',
                          'load_state_dict', 'torchvision.utils.save_image'):
            self.assertNotIn(forbidden, calls)
        for token in ('manifests/', '.parquet', 'faces_256', 'split_v1', 'difffas_bin_idfree_train', 'FAS_sample',
                      'save_images_every', 'runs/m6'):
            self.assertFalse([v for v in strings if token in v], token)
        self.assertIn("require(args.pretrain_path is None", text)

    def test_16_synthetic_only_dataset(self):
        self.assertEqual((q.N_ITEMS, q.BATCH, q.ROLES), (8, 4, ('content', 'style_spoof', 'GT')))
        src = inspect.getsource(q.SyntheticTriplets) + inspect.getsource(q.analytic_rgb)
        for token in ('open(', 'Image.open', 'listdir', 'random', 'rand', 'Path('):
            self.assertNotIn(token, src, token)

    def test_17_a6_feature_shapes(self):
        a6 = at_authority('configs/amendments/e07c_a6_feature_interface_source_correction.yaml').decode()
        for name, (h, w, c) in (('x32x32', (32, 32, 256)), ('x16x16', (16, 16, 512)), ('x8x8', (8, 8, 512))):
            self.assertIn(f'{name}: [{h}, {w}, {c}]', a6)
            self.assertEqual(q.ENCODER_SHAPES[name], [4, c, h, w])
        self.assertEqual(q.ENCODER_SHAPES['embg'], [4, 7])
        self.assertEqual(q.MAIN_OUTPUT_SHAPE, [4, 6, 256, 256])

    def test_18_counters_and_accounting(self):
        c = q.EXPECTED_COUNTERS
        ones = ('throwaway_constructor', 'sha_verified_event', 'torch_load', 'AdamW_construction',
                'scheduler_construction', 'DataLoader_iterator', 'batch', 'training_losses', 'encoder_forward',
                'zero_grad', 'backward', 'scheduler_step', 'optimizer_step', 'accumulate')
        zeros = ('torch_save', 'sampling', 'TRAIN_reads', 'VAL_reads', 'TEST_reads', 'manifest_opens', 'M8_outputs',
                 'scientific_runs', 'sha_rejected_event', 'checkpoint_write_attempts')
        self.assertTrue(all(c[k] == 1 for k in ones) and all(c[k] == 0 for k in zeros))
        self.assertEqual(c['checkpoint_read_opens'], 2)
        self.assertNotIn('optimizer_steps', c)                      # no ambiguous generic counter
        self.assertEqual(q.QUALIFIED[:4], ['E07c_MAIN_RUNNER_ENCODER_LOAD_INTEGRATION_QUALIFIED',
                                           'MAIN_RUNNER_ENCODER_LOAD_INTEGRATION',
                                           'E07c_MAIN_DIFFFAS_TRAINING_GRAPH_QUALIFIED', 'MAIN_DIFFFAS_TRAINING_GRAPH'])
        self.assertEqual(q.NOT_QUALIFIED, ['MAIN_PRODUCTION_RUNNER', 'MAIN_CHECKPOINT_RESUME',
                                           'MAIN_DIFFFAS_SCIENTIFIC_TRAINING', 'M8_BANK'])
        self.assertIn('BITWISE_DETERMINISTIC_MAIN_TRAINING', q.NOT_CLAIMED)

    # 19-21 firewall ------------------------------------------------------------------------------
    def firewall(self):
        tmp = tempfile.mkdtemp()
        build = Path(tmp) / 'builds/e07c_difffas/m6d6i'
        frozen = seam.frozen_checkpoint_path(q.rq.RUNTIME)
        fw = q.MainFirewall(build, frozen, [])
        return fw, build, str(frozen)

    def test_19_firewall_allows_only_read_of_exact_frozen_path(self):
        fw, build, frozen = self.firewall()
        fw('open', (frozen, 'rb', os.O_RDONLY | os.O_CLOEXEC))
        fw('open', (frozen, 'r', os.O_RDONLY))
        self.assertEqual(len(fw.checkpoint_opens), 2)
        for args in ((frozen, 'wb', os.O_WRONLY | os.O_CREAT | os.O_TRUNC), (frozen, 'r+b', os.O_RDWR),
                     (frozen, None, os.O_RDONLY | os.O_APPEND)):
            with self.assertRaises(RuntimeError):
                fw('open', args)
        self.assertEqual(len(fw.checkpoint_opens), 2)
        self.assertEqual(len(fw.denied), 3)

    def test_20_firewall_denies_data_other_weights_and_writes(self):
        fw, build, frozen = self.firewall()
        rt = str(q.rq.RUNTIME)
        denied = [(rt + '/runs/m6/E07c/aux_encoder/seed_42/checkpoints/other.pkl', 'rb'),
                  (rt + '/runs/m6/E07c/aux_encoder/seed_42/metrics.jsonl', 'rb'),
                  (rt + '/runs/m6/E07c/seed_42/x.pt', 'rb'), (str(ROOT / 'manifests/split_v1.parquet'), 'rb'),
                  (rt + '/data/processed/faces_256/casia_fasd/a.png', 'rb'), ('/tmp/x.ckpt', 'rb'),
                  ('/tmp/frame.jpg', 'rb'), (str(ROOT / 'outputs/x.json'), 'w'), (frozen + '.partial', 'wb')]
        for path, mode in denied:
            with self.assertRaises(RuntimeError, msg=path):
                fw('open', (path, mode, 0))
        self.assertEqual(len(fw.denied), len(denied))
        fw('open', (str(build) + '/process_1.json', 'w', os.O_WRONLY | os.O_CREAT))      # evidence write allowed
        fw('open', (str(ROOT / 'methods/difffas/main_graph.py'), 'rb', os.O_RDONLY))      # ordinary code read
        with self.assertRaises(RuntimeError):
            fw('subprocess.Popen', ('/bin/sh', ['/bin/sh', '-c', 'x']))

    def test_21_firewall_records_security_trace(self):
        fw, build, frozen = self.firewall()
        fw(seam.EVENT_VERIFIED, (frozen, seam.OWNER_FROZEN_SHA256))
        fw('pickle.find_class', ('custom_rn', 'ResNet'))
        fw('open', (frozen, 'rb', os.O_RDONLY))
        self.assertEqual([e['event'] for e in fw.trace], ['sha256_verified', 'find_class', 'checkpoint_open'])
        self.assertEqual(fw.find_class, ['custom_rn.ResNet'])

    # 22-25 history / preflight ------------------------------------------------------------------
    def test_22_protected_files_unchanged_at_m6d6i(self):
        for rel in self.pf.PROTECTED:
            self.assertEqual(at_m6d6i(rel), at_authority(rel), rel)

    def test_23_historical_evidence_and_ledger_prefix(self):
        paths = [p for p in git('ls-tree', '-r', '--name-only', AUTHORITY, 'outputs/audit').stdout.decode().splitlines()
                 if Path(p).name.startswith(tuple(f'M6D6{c}_' for c in 'ABCDEFGH'))]
        self.assertGreaterEqual(len(paths), 40)
        for rel in paths:
            self.assertEqual(at_m6d6i(rel), at_authority(rel), rel)
        self.assertTrue(at_m6d6i('outputs/audit/EXECUTION_LEDGER.jsonl').startswith(
            at_authority('outputs/audit/EXECUTION_LEDGER.jsonl')))

    def test_24_history_checks_are_scoped_not_head_locked(self):
        src = inspect.getsource(at_m6d6i) + inspect.getsource(m6d6i_commit)
        self.assertIn('--diff-filter=A', src)
        own = ast.parse(Path(__file__).read_text())
        git_calls = [[ast.literal_eval(a) for a in n.args if isinstance(a, ast.Constant)] for n in ast.walk(own)
                     if isinstance(n, ast.Call) and ast.unparse(n.func) == 'git']
        self.assertFalse([c for c in git_calls if c[:1] in (['status'], ['diff'], ['rev-parse'])])
        pf = inspect.getsource(self.pf.check_evidence) + inspect.getsource(self.pf.check_process)
        for token in ('git(', 'worktree(', 'tracked_unchanged(', 'authority()', "'HEAD'"):
            self.assertNotIn(token, pf)

    def test_25_preflight_is_static(self):
        tree_ = ast.parse((ROOT / 'tools/m6d6i_e07c_main_graph_preflight.py').read_text())
        imported = {a.name.split('.')[0] for n in ast.walk(tree_) if isinstance(n, (ast.Import, ast.ImportFrom))
                    for a in (n.names if isinstance(n, ast.Import) else [ast.alias(n.module or '')])}
        self.assertFalse({'torch', 'yaml', 'pyarrow', 'numpy', 'PIL'} & imported)
        with self.assertRaises(ValueError):
            self.pf.read('outputs/x/encoder_final.pkl')
        self.assertEqual('torch' in sys.modules, self.torch_before)


@unittest.skipUnless(EVIDENCE.is_file(), 'M6D6i evidence not yet recorded')
class TestM6D6iEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pf = load_preflight()
        cls.ev = json.loads(EVIDENCE.read_bytes())

    def test_evidence_validates(self):
        self.pf.check_evidence(at_m6d6i)

    def test_qualification_vs_scientific_counters(self):
        self.assertEqual(self.ev['qualification_counters'],
                         {'qualification_processes': 2, 'qualification_checkpoint_deserializations': 2,
                          'qualification_backward_calls': 2, 'qualification_optimizer_steps': 2,
                          'qualification_scheduler_steps': 2, 'qualification_ema_updates': 2})
        self.assertEqual(set(self.ev['scientific_counters'].values()), {0})
        self.assertIsNone(self.ev['experiment_seed'])

    def test_frozen_checkpoint_unchanged(self):
        f = self.ev['frozen_checkpoint']
        self.assertEqual(f['before_process_1'], f['after_process_2'])
        self.assertEqual((f['before_process_1']['sha256'], f['before_process_1']['size_bytes']),
                         (seam.OWNER_FROZEN_SHA256, seam.OWNER_FROZEN_BYTES))

    def test_statuses(self):
        self.assertEqual(self.ev['qualified_statuses'], q.QUALIFIED)
        self.assertEqual(self.ev['not_qualified'], q.NOT_QUALIFIED)
        self.assertFalse(self.ev['production_runner_claimed'] or self.ev['bitwise_deterministic_main_training_claimed'])


if __name__ == '__main__':
    unittest.main()
