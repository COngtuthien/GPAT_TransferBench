"""M6D6j: E07c MAIN DiffFAS production runner (QUALIFICATION ONLY; seed 60608; experiment_seed = null).

Static tests run without Torch: fakes stand in for tensors / modules / pyarrow. They exercise the SAME production
objects (MainRunner.iteration / run, CheckpointStore, read_relation, read_train_membership, MainTrainDataset, the CLI
gates) with small non-frozen schedules only where a loop must terminate. History assertions compare the state at the
commit that ADDED this test (candidate: the worktree) with the M6D6iR authority (no B1 HEAD lock).
"""
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from methods.common.config import load_method_config  # noqa: E402
from methods.common.learned import PreparationError  # noqa: E402
from methods.common.runlog import atomic_write_json  # noqa: E402
from methods.difffas import aux_runner_io as aio  # noqa: E402
from methods.difffas import main_checkpoint as mc  # noqa: E402
from methods.difffas import main_graph as mg  # noqa: E402
from methods.difffas import main_runner as mr  # noqa: E402
from methods.difffas import main_runner_io as mio  # noqa: E402
from methods.difffas.source import validate_source, tree  # noqa: E402

AUTHORITY = '39508719a57e5afde0a8c71bf7db0f9f09ec25e0'
THIS = 'tests/test_m6d6j_e07c_main_runner.py'
EVIDENCE = ROOT / 'outputs/audit/M6D6J_E07C_MAIN_RUNNER_QUALIFICATION.json'


def load_module(rel, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_preflight():
    return load_module('tools/m6d6j_e07c_main_runner_preflight.py', 'm6d6j_preflight_under_test')


def git(*args):
    return subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True)


def m6d6j_commit():
    return git('log', '--diff-filter=A', '--format=%H', '-1', '--', THIS).stdout.decode().strip() or None


def at_m6d6j(rel):
    """File bytes at the M6D6j state: the commit that added this test, else the (candidate) worktree."""
    commit = m6d6j_commit()
    return git('show', f'{commit}:{rel}').stdout if commit else (ROOT / rel).read_bytes()


def at_authority(rel):
    return git('show', f'{AUTHORITY}:{rel}').stdout


def code_facts(path):
    module = ast.parse(Path(path).read_text())
    docs = {id(n.body[0].value) for n in ast.walk(module) if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef))
            and n.body and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
    calls = [ast.unparse(n.func) for n in ast.walk(module) if isinstance(n, ast.Call)]
    strings = [n.value for n in ast.walk(module) if isinstance(n, ast.Constant) and isinstance(n.value, str)
               and id(n) not in docs]
    return calls, strings


# ------------------------------------------------------------------ checkpoint-store fixture
def fake_serializer(payload, fh):
    fh.write(json.dumps({k: str(payload[k]) for k in payload}, sort_keys=True).encode() * 64)


def payload(tag='x'):
    return {k: f'{k}-{tag}' for k in mc.PAYLOAD_KEYS}


def new_store(tmp, labels=('QUALIFICATION_ONLY',), **kw):
    run = Path(tmp) / 'run'
    (run / 'checkpoints').mkdir(parents=True)
    atomic_write_json(run / 'checkpoint_index.json', {'checkpoints': []})
    validate = mc.qualification_validator(mio.ITERATIONS_PER_EPOCH, mio.EPOCHS, mio.TERMINAL_STEP)
    return mc.CheckpointStore(run, validate=validate, serializer=fake_serializer, labels=labels, reserve_bytes=0, **kw)


def write(store, step, kind='periodic', tag=None):
    return store.write(global_step=step, epoch=step // mio.ITERATIONS_PER_EPOCH, kind=kind,
                       payload=payload(tag or str(step)), estimated_bytes=1)


# ------------------------------------------------------------------ loop fakes (no torch)
class FakeCuda:
    def reset_peak_memory_stats(self):
        pass

    def max_memory_allocated(self):
        return 123


class FakeTensor:
    def __init__(self, n):
        self.shape = (n, 3, 256, 256)


class FakeLoader:
    """Yields batches of 4 then a tail of 2 per epoch; each fetch reports its items like MainTrainDataset.on_item."""
    def __init__(self, dataset, sizes):
        self.dataset, self.sizes = dataset, sizes

    def __iter__(self):
        for n in self.sizes:
            for i in range(n):
                self.dataset.on_item({'track_pair_id': f'DFA{i:06d}'})
            yield {'GT': FakeTensor(n), 'content': FakeTensor(n), 'style_spoof': FakeTensor(n)}


class RecordingRunner(mr.MainRunner):
    def checkpoint_transition(self, iters, kind):
        self.trace.append(('checkpoint:' + kind, iters))
        if kind == 'terminal':
            self.terminal = {'global_step': iters}
        return {}, None

    def visualize_slot(self, iters):
        self.trace.append(('visualization_ran', iters))
        self.visualizations.append(iters)


def make_runner(sizes, epochs, save_every, vis_every):
    ds = types.SimpleNamespace(on_item=None)
    args = types.SimpleNamespace(print_loss_every_iters=1, save_checkpoints_every_iters=save_every,
                                 save_images_every_iters=vis_every, max_epochs=epochs)
    ctx = types.SimpleNamespace(records=[], log_step=lambda r: ctx.records.append(r))
    opt = types.SimpleNamespace(param_groups=[{'lr': 4e-7}])
    runner = RecordingRunner(torch=types.SimpleNamespace(cuda=FakeCuda()), torchvision=None, ddim_steps=None, model=None,
                             ema=None, encoder=None, optimizer=opt, scheduler=None, betas=None, diffusion=None,
                             conf=None, args=args, loader=None, dataset=ds, ctx=ctx, store=None,
                             schedule=mr.Schedule(len(sizes), epochs), mode=mio.QUALIFICATION, seed=mio.QUALIFICATION_SEED,
                             tqdm=lambda it, desc=None: types.SimpleNamespace(__iter__=None, set_description=lambda s: None,
                                                                              it=it))
    runner.loader = FakeLoader(ds, sizes)
    return runner, ctx


def fake_train_iteration(torch, conf, args, batch, iters, *rest):
    return {'iters': iters + 1, 'loss_list': [1.0], 'loss_mean_list': [0.9], 'loss_vb_list': [0.1]}


class Progress:
    def __init__(self, it):
        self.it = it

    def __iter__(self):
        return iter(self.it)

    def set_description(self, s):
        self.desc = s


class TestM6D6jStatic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.torch_before = 'torch' in sys.modules
        cls.cfg = load_method_config('E07c')
        cls.source = validate_source(cls.cfg)
        cls.pf = load_preflight()

    # ---------------------------------------------------------- authority / seed / contract
    def test_01_authority_is_ancestor(self):
        self.assertEqual(git('merge-base', '--is-ancestor', AUTHORITY, 'HEAD').returncode, 0)
        self.assertEqual(json.loads(at_m6d6j(mio.CONTRACT_PATH))['authority_commit'], AUTHORITY)

    def test_02_qualification_seed_60608_next_and_unused_before(self):
        self.assertEqual(mio.QUALIFICATION_SEED, 60608)
        used = git('grep', '-h', '-o', '-E', r'\b6060[0-9]\b', AUTHORITY, '--', '*.py', '*.yaml', '*.md').stdout.decode()
        self.assertEqual(sorted(set(used.split())), [f'6060{i}' for i in range(1, 8)])
        for bad in (42, 1337, 2026, 60607, 60605):
            with self.assertRaises(PreparationError):
                mio.validate_mode_seed(mio.QUALIFICATION, bad)
        for bad in (60608, 0, 7):
            with self.assertRaises(PreparationError):
                mio.validate_mode_seed(mio.SCIENTIFIC, bad)
        mio.validate_mode_seed(mio.SCIENTIFIC, 1337)

    def test_03_contract_d1_d2_and_bound_inputs(self):
        c = mio.load_contract()
        self.assertEqual(c['owner_decisions']['D1_visualization']['decision'], 'EXECUTE_EXACT_SOURCE')
        self.assertEqual(c['owner_decisions']['D2_terminal_position']['decision'],
                         'AFTER_FINAL_VISUALIZATION_BEFORE_RUN_COMPLETION')
        self.assertEqual((c['fidelity_class'], c['deviation'], c['new_deviation'], c['amendment_created']),
                         ('CONTROLLED_ADAPTATION', 'DEV-021', False, False))
        self.assertIn('configs/amendments/e07c_m6d6ir_checkpoint_retention.yaml', c['bound_inputs_sha256'])
        self.assertEqual(c['statuses']['not_qualified'], ['MAIN_CHECKPOINT_RESUME', 'MAIN_DIFFFAS_SCIENTIFIC_TRAINING',
                                                          'M8_BANK'])
        self.assertTrue(c['full_science_blockers']['FULL_SCIENCE_BLOCKED_PENDING_SMOKE_TEST_DECISION'])

    def test_04_roots_separated(self):
        rt = '/rt'
        self.assertEqual(str(mio.run_root(rt, mio.SCIENTIFIC, 42)), '/rt/runs/m6/E07c/seed_42')
        q = mio.run_root(rt, mio.QUALIFICATION, 60608, 'a' * 16)
        self.assertEqual(str(q), '/rt/qualification/m6d6j/E07c/q60608-' + 'a' * 16)
        with self.assertRaises(PreparationError):
            mio.run_root(rt, mio.SCIENTIFIC, 60608)
        with self.assertRaises(PreparationError):
            mio.run_root(rt, mio.QUALIFICATION, 42, 'a' * 16)
        self.assertNotEqual(mio.qualification_id('c' * 64, 'd' * 40), mr.__dict__.get('x'))

    # ---------------------------------------------------------- data path
    def fake_pq(self, rows, schema=None, split_rows=None):
        class Table:
            def __init__(self, rows, cols):
                self.rows, self.column_names = rows, cols

            def to_pylist(self):
                return [{c: r[c] for c in self.column_names} for r in self.rows]
        pq = types.SimpleNamespace()
        pq.read_schema = lambda p: types.SimpleNamespace(names=schema or list(rows[0]))
        pq.read_table = lambda p, columns, filters=None: Table(split_rows if filters else rows, list(columns))
        return pq

    def relation_rows(self):
        rows, i = [], 0
        for dataset, n in mio.ROWS_BY_DATASET.items():
            for j in range(n):
                gt = f'{dataset}{j:08d}'.ljust(64, '0')[:64]
                rows.append({'track_pair_id': f'DFA{i:06d}', 'dataset': dataset,
                             'gt_spoof_id': hex(abs(hash((dataset, j))))[2:].rjust(64, 'a')[:64],
                             'guide_spoof_id': 'b' * 64, 'content_live_id': 'c' * 64, 'style_id': 'SPOOF_BINARY',
                             'split': 'TRAIN', 'use_pair': False, 'selection_policy': mio.SELECTION_POLICY,
                             'gt_sha256': 'd' * 64, 'guide_sha256': 'e' * 64, 'seed': 1})
                i += 1
        rows.sort(key=lambda r: (r['dataset'], r['gt_spoof_id']))
        for i, r in enumerate(rows):
            r['track_pair_id'] = f'DFA{i:06d}'
        return rows

    def test_05_relation_reader_allowlist_and_train_only(self):
        rows = self.relation_rows()
        with mock.patch.object(mio, 'sha256_file', return_value=mio.RELATION_SHA256):
            records, ev = mio.read_relation('/x.parquet', pq=self.fake_pq(rows))
        self.assertEqual((len(records), ev['rows_by_dataset'], ev['columns_read']),
                         (8838, mio.ROWS_BY_DATASET, list(mio.RELATION_COLUMNS)))
        self.assertEqual(set(records[0]), {'index', 'track_pair_id', 'dataset', 'gt_spoof_id', 'guide_spoof_id',
                                           'content_live_id'})
        for mutate in (lambda r: r.update(split='VAL'), lambda r: r.update(use_pair=True),
                       lambda r: r.update(style_id='print'), lambda r: r.update(selection_policy='random.choice')):
            bad = copy.deepcopy(rows)
            mutate(bad[5])
            with mock.patch.object(mio, 'sha256_file', return_value=mio.RELATION_SHA256), \
                    self.assertRaises(PreparationError):
                mio.read_relation('/x.parquet', pq=self.fake_pq(bad))
        with mock.patch.object(mio, 'sha256_file', return_value=mio.RELATION_SHA256), self.assertRaises(PreparationError):
            mio.read_relation('/x.parquet', pq=self.fake_pq(rows, schema=list(rows[0]) + ['subject_id_global']))
        with tempfile.NamedTemporaryFile(suffix='.parquet') as fh, self.assertRaises(PreparationError):
            fh.write(b'not the frozen relation')
            fh.flush()
            mio.read_relation(fh.name, pq=self.fake_pq(rows))           # SHA256 before parse

    def test_06_train_membership(self):
        recs = [{'dataset': 'msu_mfsd', 'gt_spoof_id': 'a' * 64, 'guide_spoof_id': 'b' * 64, 'content_live_id': 'c' * 64}]
        split = [{'sample_id': s, 'dataset': 'msu_mfsd', 'split': 'TRAIN', 'm2_status': 'COMPLETE'} for s in 'abc']
        split = [dict(r, sample_id=r['sample_id'] * 64) for r in split]
        with mock.patch.object(mio, 'sha256_file', return_value=mio.SPLIT_MANIFEST_SHA256):
            ids, ev = mio.read_train_membership(recs, '/s.parquet', pq=self.fake_pq(split, split_rows=split))
            self.assertEqual(ids, {'a' * 64: 'msu_mfsd', 'b' * 64: 'msu_mfsd', 'c' * 64: 'msu_mfsd'})
            self.assertEqual(ev['columns_read'], ['sample_id', 'dataset', 'split', 'm2_status'])
            for bad in (split[:2], [dict(split[0], dataset='siwmv2'), *split[1:]],
                        [dict(split[0], m2_status='FAILED'), *split[1:]]):
                with self.assertRaises(PreparationError):
                    mio.read_train_membership(recs, '/s.parquet', pq=self.fake_pq(bad, split_rows=bad))

    def test_07_dataset_items_guide_and_reader(self):
        self.assertIs(mio.CanonicalFaceReader, aio.CanonicalFaceReader)
        reads = []
        recs = [{'index': i, 'track_pair_id': f'DFA{i:06d}', 'dataset': 'd', 'gt_spoof_id': f'g{i}',
                 'guide_spoof_id': f'u{i}', 'content_live_id': f'c{i}'} for i in range(mio.TRAIN_ROWS)]
        ds = mio.MainTrainDataset(recs, lambda s: reads.append(s) or s, lambda x: 'T(' + x + ')')
        item = ds[7]
        self.assertEqual(item, {'content': 'T(c7)', 'style_spoof': 'T(u7)', 'GT': 'T(g7)'})
        self.assertEqual(reads, ['c7', 'g7', 'u7'])                        # FAS_dataset.py read order
        self.assertEqual(len(ds), 8838)
        src = Path(ROOT / 'methods/difffas/main_runner_io.py').read_text()
        self.assertNotIn('random.choice', ''.join(code_facts(ROOT / 'methods/difffas/main_runner_io.py')[0]))
        self.assertNotIn('import random', src)

    def test_08_no_identity_or_attack_label_consumption(self):
        self.assertFalse(set(mio.RELATION_COLUMNS) & set(mio.FORBIDDEN_COLUMNS))
        self.assertEqual(set(mio.FORBIDDEN_COLUMNS) & {'subject_id_global', 'attack_raw', 'attack_macro'},
                         {'subject_id_global', 'attack_raw', 'attack_macro'})
        for rel in ('methods/difffas/main_runner.py', 'methods/difffas/main_checkpoint.py'):
            strings = code_facts(ROOT / rel)[1]
            self.assertFalse([s for s in strings if s in ('subject_id_global', 'attack_raw', 'attack_macro')], rel)

    def test_09_loop_constants(self):
        self.assertEqual((mio.BATCH_SIZE, mio.SHUFFLE, mio.DROP_LAST, mio.WORKERS, mio.ITERATIONS_PER_EPOCH,
                          mio.TAIL_BATCH, mio.EPOCHS, mio.TOTAL_ITERATIONS),
                         (4, True, False, 0, 2210, 2, 400, 884000))
        self.assertEqual((len(mio.PERIODIC_STEPS), mio.PERIODIC_STEPS[0], mio.PERIODIC_STEPS[-1], mio.TERMINAL_STEP),
                         (88, 10000, 880000, 884000))
        self.assertTrue(mr.Schedule.frozen().is_frozen())
        self.assertEqual(mr.Schedule.frozen().total, 884000)

    # ---------------------------------------------------------- source traceability
    def test_10_replicas_equal_pinned_source(self):
        ev = mr.verify_production_source(self.source)
        self.assertEqual(ev['iteration_order'], ['train', 'report', 'metrics', 'save', 'visualize', 'terminal'])
        self.assertEqual(ev['pinned_defaults']['sample_algorithm'], 'ddpm')
        self.assertEqual((ev['pinned_defaults']['sample_initial_noise'], ev['pinned_defaults']['save_images_every_iters'],
                          ev['pinned_defaults']['save_checkpoints_every_iters']), (250, 1000, 10000))
        self.assertEqual(ev['train_script_sha256'], 'c5eb42a1193f7708b9e972db0faac602c82ffe2dc10578b6c9013d81e26e4f84')

    def test_11_mutated_visualization_rejected(self):
        module = tree(self.source, 'FAS_train.py')
        train = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == 'train')
        inner = next(n for n in ast.walk(train) if isinstance(n, ast.For) and n.lineno == 41)
        pinned = [ast.unparse(s) for s in {s.lineno: s for s in inner.body}[117].body]
        own = mr._own()[1]['visualize']
        self.assertEqual([ast.unparse(s) for s in mr._body(own)], pinned)
        text = ast.unparse(own).replace('args.sample_initial_noise', '25')
        self.assertNotEqual([ast.unparse(s) for s in mr._body(ast.parse(text).body[0])], pinned)
        text = ast.unparse(own).replace("args.sample_algorithm == 'ddpm'", "args.sample_algorithm == 'ddim'")
        self.assertNotEqual([ast.unparse(s) for s in mr._body(ast.parse(text).body[0])], pinned)

    def test_12_swapped_slot_order_rejected(self):
        cls = copy.deepcopy(mr._own()[2]['MainRunner'])
        fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'iteration')
        ifs = [i for i, s in enumerate(fn.body) if isinstance(s, ast.If)]
        fn.body[ifs[0]], fn.body[ifs[1]] = fn.body[ifs[1]], fn.body[ifs[0]]
        self.assertEqual(mr.iteration_order(cls), ['train', 'report', 'metrics', 'visualize', 'save', 'terminal'])
        cls = copy.deepcopy(mr._own()[2]['MainRunner'])
        fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'iteration')
        ifs = [i for i, s in enumerate(fn.body) if isinstance(s, ast.If)]
        fn.body[ifs[1]], fn.body[ifs[2]] = fn.body[ifs[2]], fn.body[ifs[1]]
        self.assertNotEqual(mr.iteration_order(cls), mr.ITERATION_ORDER)

    # ---------------------------------------------------------- loop order / D1 / D2 (fakes)
    def run_loop(self, sizes, epochs, save_every, vis_every):
        runner, ctx = make_runner(sizes, epochs, save_every, vis_every)
        runner.tqdm = lambda it, desc=None: Progress(it)
        with mock.patch.object(mg, 'train_iteration', fake_train_iteration):
            iters = runner.run()
        return runner, ctx, iters

    def test_13_iteration_order_and_d1_cadence(self):
        runner, ctx, iters = self.run_loop([4, 4, 2], 2, 4, 2)       # total 6
        self.assertEqual(iters, 6)
        tags = [t for t, _ in runner.trace]
        expected = []
        for s in range(1, 7):
            expected += ['train', 'report']
            if s % 4 == 0:
                expected += ['save', 'checkpoint:periodic']
            if s % 2 == 0:
                expected += ['visualize', 'visualization_ran']
            if s == 6:
                expected += ['terminal', 'checkpoint:terminal']
        self.assertEqual(tags, expected)
        self.assertEqual(runner.visualizations, [2, 4, 6])              # never skipped at the source cadence
        self.assertEqual([r['global_step'] for r in ctx.records], list(range(1, 7)))
        self.assertEqual([r['batch_size'] for r in ctx.records], [4, 4, 2, 4, 4, 2])   # tail batch kept

    def test_14_d2_terminal_after_final_visualization(self):
        runner, _, _ = self.run_loop([4, 4, 2], 2, 4, 2)
        last = runner.trace[-6:]
        self.assertEqual([t for t, _ in last], ['train', 'report', 'visualize', 'visualization_ran', 'terminal',
                                                'checkpoint:terminal'])
        self.assertEqual(sum(1 for t, _ in runner.trace if t == 'checkpoint:terminal'), 1)

    def test_15_frozen_d2_arithmetic(self):
        self.assertEqual((mio.TERMINAL_STEP % mio.SAVE_EVERY, mio.TERMINAL_STEP % mio.VISUALIZE_EVERY), (4000, 0))
        self.assertEqual(sum(1 for s in range(1, mio.TOTAL_ITERATIONS + 1) if s % mio.VISUALIZE_EVERY == 0), 884)

    def test_16_per_step_metrics_explicit_nulls(self):
        r = mio.step_record(mode=mio.QUALIFICATION, seed=60608, epoch=1, global_step=1, iteration=0, learning_rate=4e-7,
                            loss=1.0, mse=0.9, vb=0.1, batch_size=4, batch_track_sha256='a' * 64,
                            wall_clock_seconds=1.0, gpu_memory_bytes=5)
        self.assertEqual((r['train_metrics'], r['val_losses'], r['val_metrics'], r['experiment_seed'],
                          r['qualification_seed'], r['record_granularity']), (None, None, None, None, 60608,
                                                                              'OPTIMIZER_STEP'))
        self.assertTrue({'train_metrics', 'val_losses', 'val_metrics', 'experiment_seed'} <= set(r['missing_field_reasons']))
        self.assertFalse([k for k in r if 'test' in k.lower()])
        with self.assertRaises(PreparationError):
            mio.step_record(mode=mio.SCIENTIFIC, seed=42, epoch=1, global_step=1, iteration=0, learning_rate=1.0,
                            loss=float('nan'), mse=0.0, vb=0.0, batch_size=4, batch_track_sha256='a',
                            wall_clock_seconds=0.0, gpu_memory_bytes=0)

    # ---------------------------------------------------------- checkpoint store (M6D6iR)
    def test_17_checkpoint_payload_filename_and_safe_write(self):
        self.assertEqual(mc.checkpoint_name(10000), 'model_010000.pt')
        self.assertEqual(mc.checkpoint_name(884000), 'model_884000.pt')
        conf = types.SimpleNamespace(training=types.SimpleNamespace(ckpt_path='/r/checkpoints'))
        self.assertEqual(mr.checkpoint_path_expr(conf, 880000), '/r/checkpoints/model_880000.pt')
        m = types.SimpleNamespace(state_dict=lambda: 'M')
        p = mr.checkpoint_payload(m, types.SimpleNamespace(state_dict=lambda: 'E'),
                                  types.SimpleNamespace(state_dict=lambda: 'S'), types.SimpleNamespace(state_dict=lambda: 'O'),
                                  conf)
        self.assertEqual(p, {'model': 'M', 'ema': 'E', 'scheduler': 'S', 'optimizer': 'O', 'conf': conf})
        with tempfile.TemporaryDirectory() as tmp:
            store = new_store(tmp)
            e = write(store, 10000)
            self.assertEqual({k: e[k] for k in mc.INDEX_FIELDS if k not in ('sha256', 'file_size_bytes')},
                             {'path': 'checkpoints/model_010000.pt', 'epoch': 4, 'global_step': 10000,
                              'checkpoint_type': 'periodic', 'selected_for_final': False,
                              'selection_reason': mc.PERIODIC_REASON})
            self.assertEqual((e['bytes_present'], e['bytes_pruned'], e['index_verified']), (True, False, True))
            self.assertEqual(e['sha256'], mc.file_sha256(store.run_dir / e['path']))
            self.assertFalse(list(store.ckpt_dir.glob('*.partial')))
            with self.assertRaises(mc.CheckpointStop):
                store.write(global_step=20000, epoch=9, kind='periodic', payload={'model': 1}, estimated_bytes=1)

    def test_18_successor_gated_prune_and_terminal_protection(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = new_store(tmp)
            a = write(store, 10000)
            self.assertIsNone(store.transition(a['path']))
            b = write(store, 20000)
            ev = store.transition(b['path'])
            self.assertEqual((ev['path'], ev['successor_global_step']), (a['path'], 20000))
            ra = store.entry(a['path'])
            self.assertEqual((ra['bytes_present'], ra['bytes_pruned'], ra['prune_reason'], ra['sha256'],
                              ra['successor_checkpoint_sha256']),
                             (False, True, mc.PRUNE_REASON, a['sha256'], b['sha256']))
            self.assertFalse((store.run_dir / a['path']).exists())
            t = write(store, mio.TERMINAL_STEP, 'terminal')
            self.assertEqual((t['checkpoint_type'], t['selected_for_final'], t['selection_reason'], t['epoch'],
                              t['logical_roles']), ('terminal', True, 'BASELINE_FINAL_STATE_V1', 400,
                                                    ['terminal', 'selected', 'authoritative_final', 'officially_required']))
            self.assertEqual(store.transition(t['path'])['path'], b['path'])
            with self.assertRaises(mc.ProtectedCheckpoint):
                store.prune(t['path'], t['path'])
            self.assertTrue((store.run_dir / t['path']).is_file())
            self.assertEqual([e['path'] for e in store.read_index()['checkpoints']], [a['path'], b['path'], t['path']])

    def test_19_fault_injection_keeps_predecessor(self):
        for fault in ('save', 'hash', 'index', 'verify'):
            with tempfile.TemporaryDirectory() as tmp:
                store = new_store(tmp)
                a = write(store, 10000)
                store.faults = {fault: True}
                with self.assertRaises(mc.CheckpointStop):
                    write(store, 20000)
                store.faults = {}
                self.assertTrue((store.run_dir / a['path']).is_file(), fault)
                self.assertTrue(store.entry(a['path'])['bytes_present'], fault)
                self.assertFalse(list(store.ckpt_dir.glob('*.partial')), fault)
        with tempfile.TemporaryDirectory() as tmp:
            store = new_store(tmp)
            a, b = write(store, 10000), write(store, 20000)
            store.faults = {'prune': True}
            with self.assertRaises(mc.CheckpointStop):
                store.transition(b['path'])
            self.assertTrue((store.run_dir / a['path']).is_file() and store.entry(a['path'])['bytes_present'])

    def test_20_no_prune_before_verification_exact_path_and_space(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = new_store(tmp)
            a, b = write(store, 10000), write(store, 20000)
            index = store.read_index()
            index['checkpoints'][1]['index_verified'] = False
            atomic_write_json(store.index_path, index)
            with self.assertRaises(mc.CheckpointStop):
                store.prune(a['path'], b['path'])
            index['checkpoints'][1]['index_verified'] = True
            atomic_write_json(store.index_path, index)
            (store.run_dir / a['path']).write_bytes(b'tampered')
            with self.assertRaises(mc.CheckpointStop):
                store.prune(a['path'], b['path'])           # recorded bytes only
            with self.assertRaises(mc.CheckpointStop):
                store.exact_path('checkpoints/../model_010000.pt')
        with tempfile.TemporaryDirectory() as tmp:
            store = new_store(tmp)
            a = write(store, 10000)
            store.reserve_bytes = 1 << 62
            with self.assertRaises(mc.StorageStop):
                write(store, 20000)
            self.assertTrue((store.run_dir / a['path']).is_file())
        with tempfile.TemporaryDirectory() as tmp:
            store = new_store(tmp)
            a, b, c = write(store, 10000), write(store, 20000), write(store, 30000)
            with self.assertRaises(mc.CheckpointStop):
                store.prune(a['path'], c['path'])           # not the DIRECT predecessor

    def test_21_no_glob_or_background_cleanup(self):
        calls, _ = code_facts(ROOT / 'methods/difffas/main_checkpoint.py')
        for bad in ('glob', 'rglob', 'iterdir', 'listdir', 'scandir', 'rmtree', 'walk', 'Thread', 'remove'):
            self.assertFalse([c for c in calls if c.split('.')[-1] == bad], bad)
        self.assertEqual(sorted(c for c in calls if c.endswith('unlink')), ['os.unlink', 'os.unlink', 'partial.unlink'])
        src = (ROOT / 'methods/difffas/main_checkpoint.py').read_text()
        self.assertIn('partial.unlink()        # the incomplete successor only; never another checkpoint', src)
        with tempfile.TemporaryDirectory() as tmp:
            store = new_store(tmp, labels=())
            a = write(store, 10000)
            with self.assertRaises(mc.CheckpointStop):
                store.remove_qualification_bytes(a['path'])    # cleanup exists only in qualification roots

    def test_22_scientific_validator_and_qualification_validator_agree(self):
        v = mc.qualification_validator(mio.ITERATIONS_PER_EPOCH, mio.EPOCHS, mio.TERMINAL_STEP)
        s = mc.scientific_validator(self.cfg, 42, mio.ITERATIONS_PER_EPOCH)
        for step, kind in ((10000, 'periodic'), (880000, 'periodic'), (884000, 'terminal')):
            meta = dict(path=f'checkpoints/{mc.checkpoint_name(step)}', epoch=step // 2210, global_step=step,
                        file_size_bytes=10, sha256='a' * 64, checkpoint_type=kind, selected_for_final=kind == 'terminal')
            self.assertEqual(v(dict(meta)), s(dict(meta)))
        with self.assertRaises(PreparationError):
            v(dict(path='x', epoch=4, global_step=10000, file_size_bytes=1, sha256='a' * 64, checkpoint_type='terminal',
                   selected_for_final=True))

    # ---------------------------------------------------------- CLI / scope
    def test_23_cli_refuses_overrides_and_non_scientific_seeds(self):
        cli = load_module('tools/run_e07c_main.py', 'run_e07c_main_under_test')
        for flag in ('--disable-visualization', '--visualization-frequency', '--sample-algorithm', '--resume',
                     '--resume-state', '--checkpoint', '--pretrain-path', '--batch-size', '--no-prune', '--test',
                     '--encoder', '--qualification-seed', '--save-every'):
            with self.assertRaises(SystemExit) as cm, mock.patch('sys.stderr', new=open(os.devnull, 'w')):
                cli.parse(['--seed', '42', '--execution-config', mio.EXEC_CONFIG, flag])
            self.assertEqual(cm.exception.code, 2, flag)
        env = {'PYTHONHASHSEED': '60608', 'NVIDIA_TF32_OVERRIDE': '0', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}
        for seed in (60608, 7, 60607):
            with self.assertRaises(SystemExit), mock.patch('sys.stderr', new=open(os.devnull, 'w')):
                cli.preconditions(cli.parse(['--seed', str(seed), '--execution-config', mio.EXEC_CONFIG]),
                                  environ=dict(env, PYTHONHASHSEED=str(seed)), dirty=False, branch='m6-baselines')
        args = cli.parse(['--seed', '42', '--execution-config', mio.EXEC_CONFIG])
        env42 = dict(env, PYTHONHASHSEED='42')
        for kw in ({'dirty': True, 'branch': 'm6-baselines', 'environ': env42},
                   {'dirty': False, 'branch': 'main', 'environ': env42},
                   {'dirty': False, 'branch': 'm6-baselines', 'environ': dict(env42, NVIDIA_TF32_OVERRIDE='1')}):
            with self.assertRaises(SystemExit), mock.patch('sys.stderr', new=open(os.devnull, 'w')):
                cli.preconditions(args, check_interpreter=False, **kw)
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / 'runs/m6/E07c/seed_42').mkdir(parents=True)
            with self.assertRaises(SystemExit), mock.patch('sys.stderr', new=open(os.devnull, 'w')):
                cli.preconditions(args, environ=env42, dirty=False, branch='m6-baselines', check_interpreter=False,
                                  runtime_root=tmp)                # fresh only: existing root refused
            torch_before = 'torch' in sys.modules              # live modules earlier in a suite may import torch
            plan = cli.preconditions(args, environ=env42, dirty=False, branch='m6-baselines', check_interpreter=False,
                                     runtime_root=str(Path(tmp) / 'fresh'))
            self.assertEqual((plan['experiment_seed'], plan['resume'], plan['torch_imported']), (42, None, torch_before))

    def test_24_no_visualization_switch_no_resume_path(self):
        runner_src = (ROOT / 'methods/difffas/main_runner.py').read_text()
        cls = mr._own()[2]['MainRunner']
        slot = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'visualize_slot')
        self.assertFalse([n for n in ast.walk(slot) if isinstance(n, ast.If)])
        self.assertFalse([n for n in ast.walk(slot) if isinstance(n, ast.Call) and 'set_rng_state' in ast.unparse(n)])
        for token in ('set_rng_state', 'fork_rng', 'manual_seed_all(', 'pretrain_path=', 'load_state_dict'):
            self.assertNotIn(token, runner_src.split('def seed_process')[0] + runner_src.split('def configure_precision')[1])
        self.assertIn('if args.pretrain_path is not None:', (ROOT / 'methods/difffas/main_graph.py').read_text())

    def test_25_quiet_tqdm_is_display_only(self):
        src = ast.unparse(next(n for n in ast.parse((ROOT / 'methods/difffas/main_runner.py').read_text()).body
                               if isinstance(n, ast.FunctionDef) and n.name == 'quiet_tqdm'))
        self.assertIn('disable=True', src)

    # ---------------------------------------------------------- history / preflight
    def test_26_protected_files_unchanged_at_m6d6j(self):
        for rel in self.pf.PROTECTED:
            self.assertEqual(at_m6d6j(rel), at_authority(rel), rel)

    def test_27_historical_evidence_and_ledger_prefix(self):
        paths = [p for p in git('ls-tree', '-r', '--name-only', AUTHORITY, 'outputs/audit').stdout.decode().splitlines()
                 if Path(p).name.startswith(tuple(f'M6D6{c}_' for c in 'ABCDEFGHI') + ('M6D6IR_',))]
        self.assertGreaterEqual(len(paths), 47)
        for rel in paths:
            self.assertEqual(at_m6d6j(rel), at_authority(rel), rel)
        self.assertTrue(at_m6d6j('outputs/audit/EXECUTION_LEDGER.jsonl').startswith(
            at_authority('outputs/audit/EXECUTION_LEDGER.jsonl')))

    def test_28_history_future_safe_and_preflight_static(self):
        import inspect
        self.assertIn('--diff-filter=A', inspect.getsource(m6d6j_commit))
        own = ast.parse(Path(__file__).read_text())
        git_calls = [[ast.literal_eval(a) for a in n.args if isinstance(a, ast.Constant)] for n in ast.walk(own)
                     if isinstance(n, ast.Call) and ast.unparse(n.func) == 'git']
        self.assertFalse([c for c in git_calls if c[:1] in (['status'], ['diff'], ['rev-parse'])])
        src = inspect.getsource(self.pf.check_evidence)
        for token in ('git(', 'worktree(', 'authority()', "'HEAD'"):
            self.assertNotIn(token, src)
        tree_ = ast.parse((ROOT / 'tools/m6d6j_e07c_main_runner_preflight.py').read_text())
        imported = {a.name.split('.')[0] for n in ast.walk(tree_) if isinstance(n, (ast.Import, ast.ImportFrom))
                    for a in (n.names if isinstance(n, ast.Import) else [ast.alias(n.module or '')])}
        self.assertFalse({'torch', 'yaml', 'pyarrow', 'numpy', 'PIL'} & imported)
        self.assertEqual('torch' in sys.modules, self.torch_before)


@unittest.skipUnless(EVIDENCE.is_file(), 'M6D6j evidence not yet recorded')
class TestM6D6jEvidence(unittest.TestCase):
    def test_evidence_validates(self):
        load_preflight().check_evidence(at_m6d6j)

    def test_statuses_and_counters(self):
        ev = json.loads(at_m6d6j('outputs/audit/M6D6J_E07C_MAIN_RUNNER_QUALIFICATION.json'))
        self.assertEqual(ev['scientific_counters'], {'scientific_main_runs': 0, 'scientific_optimizer_steps': 0,
                                                     'experiment_seed_runs': 0, 'scientific_checkpoint_writes': 0})
        self.assertIsNone(ev['experiment_seed'])
        self.assertEqual(ev['not_qualified'], ['MAIN_CHECKPOINT_RESUME', 'MAIN_DIFFFAS_SCIENTIFIC_TRAINING', 'M8_BANK'])


if __name__ == '__main__':
    unittest.main()
