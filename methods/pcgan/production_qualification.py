"""E05 real-TRAIN production qualification harness: QUALIFICATION_ONLY, never scientific.

One GPU process with qualification seed 60901 (never an experiment seed), writing only under
<runtime_root>/qualification/e05_production/E05/ and a new /tmp JSON. It executes the SAME production components as
tools/run_e05.py -- pair-manifest reader, canonical face reader, seeded loader, qualified runtime/architecture
builder, TrainingRunner D -> G cycle via E05Trainer/run_iterations, checkpoint writer and E05RunContext lifecycle --
for 3 real TRAIN benchmark iterations (iteration 1 hits the periodic checkpoint cadence), then exercises the
iteration-4000 terminal-selection semantics in a separate, labelled qualification root WITHOUT 4000 optimizer
steps (no scientific completion record is written: the summary is 'interrupted', not 'completed').

A Python audit-hook firewall is active before the first manifest byte is read: it permits the TRAIN pair manifest and
exactly the canonical faces of the independently predicted qualification pairs, and denies split/VAL/TEST metadata,
every other face, faces enumeration, every other benchmark root, runs/ and writes outside the qualification root,
/tmp and /dev (the M6D4d runtime write policy). Every benchmark-path event is appended to an access log.
num_workers = 0, so all reads are in-process.

Launch (clean worktree, gpat-m6-e05, the production launch environment with PYTHONHASHSEED=60901):
  ... python -u -B -m methods.pcgan.production_qualification --output /tmp/gpat_e05_prodq/result.json
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from methods.common.config import sha256_file  # noqa: E402
from methods.pcgan import production_io as pio  # noqa: E402
from methods.pcgan import production_runner as pr  # noqa: E402

SEED = pio.QUALIFICATION_SEED
MODE = pio.QUALIFICATION
CATEGORIES = ('PAIR_MANIFEST', 'SPLIT_METADATA_VAL_TEST', 'VAL_METADATA', 'TEST_METADATA', 'OTHER_MANIFEST',
              'AUTHORIZED_TRAIN_FACE', 'UNAUTHORIZED_FACE', 'FACES_ENUMERATION_OR_WRITE', 'SCIENTIFIC_RUN_ROOT',
              'OTHER_BENCHMARK_DATA')
require = pio.require


class Firewall:
    """Audit hook: the TRAIN pair manifest + the authorized TRAIN faces only; no other data, no scientific root."""
    WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
    EVENTS = ('open', 'os.listdir', 'os.scandir', 'os.mkdir', 'os.remove', 'os.rmdir', 'os.rename', 'shutil.rmtree')

    def __init__(self, runtime_root, faces_root, write_roots, access_fd):
        self.runtime_root, self.faces = Path(runtime_root), Path(faces_root)
        self.write_roots = [str(Path(p)) + '/' for p in write_roots]
        self.authorized = {}                     # sample_id -> dataset, filled from the independent prediction
        self.manifest = str(ROOT / pio.PAIR_MANIFEST)
        self.events = dict.fromkeys(self.EVENTS, 0)
        self.denied, self.face_opens, self.manifest_opens = [], [], 0
        self.access_fd = access_fd

    def category(self, event, path):
        p = Path(path)
        if p.is_relative_to(ROOT / 'manifests'):
            if path == self.manifest:
                return 'PAIR_MANIFEST'
            name = p.name.lower()
            if name.startswith('split'):
                return 'SPLIT_METADATA_VAL_TEST'
            if 'val' in name:
                return 'VAL_METADATA'
            if 'test' in name:
                return 'TEST_METADATA'
            return 'OTHER_MANIFEST'
        if p.is_relative_to(self.faces):
            rel = p.relative_to(self.faces).parts
            if event != 'open':
                return 'FACES_ENUMERATION_OR_WRITE'
            ok = len(rel) == 2 and rel[1].endswith('.png') and self.authorized.get(rel[1][:-4]) == rel[0]
            return 'AUTHORIZED_TRAIN_FACE' if ok else 'UNAUTHORIZED_FACE'
        if p.is_relative_to(self.runtime_root / 'runs'):
            return 'SCIENTIFIC_RUN_ROOT'
        if p.is_relative_to(self.runtime_root / 'data') or any(p.is_relative_to(ROOT / d) for d in ('data', 'cache', 'runs')):
            return 'OTHER_BENCHMARK_DATA'
        return None

    def writable(self, path):
        return (path == '/dev/null' or path.startswith('/dev/shm/') or
                any(path == r[:-1] or path.startswith(r) for r in self.write_roots))

    def reason(self, event, path, args):
        p = Path(path)
        writing = event not in ('open', 'os.listdir', 'os.scandir')
        if event == 'open':
            mode = args[1] if len(args) > 1 and isinstance(args[1], str) else ''
            flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
            writing = bool(flags & self.WRITE_FLAGS) or bool(set(mode) & set('wax+'))
        if p.is_relative_to(self.runtime_root / 'runs'):
            return 'scientific run root'
        if p.is_relative_to(ROOT / 'manifests'):
            if event == 'open' and path == self.manifest and not writing:
                self.manifest_opens += 1
                return None
            return 'manifest other than the TRAIN pair manifest'
        if p.is_relative_to(self.faces):
            if event != 'open':
                return 'faces_256 enumeration/mutation'
            rel = p.relative_to(self.faces).parts
            ok = len(rel) == 2 and rel[1].endswith('.png') and self.authorized.get(rel[1][:-4]) == rel[0]
            if writing or not ok:
                return 'face outside the authorized qualification TRAIN pairs, or a write'
            self.face_opens.append(path)
            return None
        if p.is_relative_to(self.runtime_root / 'data') or any(p.is_relative_to(ROOT / d) for d in ('data', 'cache', 'runs')):
            return 'benchmark data root'
        if p.suffix == '.parquet':
            return 'parquet outside the TRAIN pair manifest'
        if writing and not self.writable(path):
            return 'write outside the qualification root, /tmp and /dev'
        if p.suffix in ('.pth', '.pt', '.tar', '.ckpt', '.pkl') and not self.writable(path):
            return 'weight file outside this qualification root'
        return None

    def __call__(self, event, args):
        if event not in self.events or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        self.events[event] += 1
        paths = [args[0]] + ([args[1]] if event == 'os.rename' and len(args) > 1 else [])
        for raw in paths:
            path = os.path.abspath(os.fsdecode(raw))
            why = self.reason(event, path, args)
            cat = self.category(event, path)
            if cat is not None:
                os.write(self.access_fd, f'{os.getpid()}\t{event}\t{cat}\t{"DENIED" if why else "ALLOWED"}\t'
                                         f'{path}\n'.encode('utf-8'))
            if why:
                self.denied.append({'event': event, 'path': path, 'reason': why})
                raise PermissionError(f'E05 QUALIFICATION FIREWALL ({why}): {path}')

    def report(self):
        return {'denied': self.denied, 'event_counts': self.events, 'pair_manifest_opens': self.manifest_opens,
                'face_opens': len(self.face_opens), 'authorized_faces': len(self.authorized),
                'note': 'num_workers = 0: every face read happens in this audited process. pyarrow reads the pair '
                        'manifest natively after the audited Python SHA256 open of that exact path.'}


def access_audit(path, expected_paths):
    rows = [line.split('\t') for line in Path(path).read_text(encoding='utf-8').splitlines()]
    require(all(len(r) == 5 for r in rows), 'access log format')
    counts = dict.fromkeys(CATEGORIES, 0)
    faces = []
    for _, _, cat, _, p in rows:
        counts[cat] += 1
        if cat == 'AUTHORIZED_TRAIN_FACE':
            faces.append(p)
    return {'log_sha256': sha256_file(path), 'events_logged': len(rows), 'counts_by_category': counts,
            'denied_events': sum(r[3] == 'DENIED' for r in rows), 'processes_logged': len({r[0] for r in rows}),
            'face_paths_opened_in_order': faces, 'face_paths_equal_expected': faces == expected_paths,
            'VAL_image_accesses': counts['UNAUTHORIZED_FACE'], 'TEST_image_accesses': counts['UNAUTHORIZED_FACE'],
            'VAL_metadata_accesses': counts['SPLIT_METADATA_VAL_TEST'] + counts['VAL_METADATA'],
            'TEST_metadata_accesses': counts['SPLIT_METADATA_VAL_TEST'] + counts['TEST_METADATA'],
            'image_access_note': 'every face path is classified against the authorized TRAIN qualification pairs '
                                 'only; VAL/TEST image accesses are bounded above by UNAUTHORIZED_FACE, which must be 0'}


def predict_order(torch, seed, n, k):
    """Independent of DataLoader: per pass the generator yields one int64 base seed, then randperm(n)."""
    g = torch.Generator()
    g.manual_seed(seed)
    torch.empty((), dtype=torch.int64).random_(generator=g)
    return torch.randperm(n, generator=g).tolist()[:k]


def component_hashes(models):
    out = {}
    for name, model in models.items():
        h = hashlib.sha256()
        for _, p in model.named_parameters():
            h.update(p.detach().cpu().contiguous().numpy().tobytes())
        out[name] = h.hexdigest()
    return out


def adam_steps(runner):
    out = {}
    for group, opt in runner.optimizers.items():
        owners = {id(p) for _, p in runner.groups[group]}
        require(all(id(p) in owners for p in opt.state), 'Adam state ownership ' + group)
        out[group] = sorted({int(s['step'].item()) for s in opt.state.values()}) + [len(opt.state)]
    return out


def expect_refusal(fn, label, result):
    try:
        fn()
    except Exception as exc:          # noqa: BLE001 -- recorded refusal, not swallowed
        result['refusals'][label] = f'{type(exc).__name__}: {exc}'
        return
    raise RuntimeError('expected refusal did not happen: ' + label)


def qualify(result, output):
    import torch
    from methods.common.runlog import git_commit, git_dirty
    from methods.pcgan import PCGANAdapter
    from methods.pcgan.training_runner import TrainingRunner
    require(not git_dirty(), 'qualification runs from a clean committed worktree')
    errors = pio.launch_environment_errors(os.environ, SEED)
    require(not errors, 'launch environment: ' + ', '.join(errors))
    storage = pio.faces_root_from_exec_config()
    rt = Path(storage['runtime_root'])
    qroot = rt.joinpath(*pio.QUALIFICATION_PARTS)
    qroot.mkdir(parents=True, exist_ok=True)                      # before the hook (parents lie outside it)
    access_log = qroot / f'access_{time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())}.tsv'
    access_fd = os.open(access_log, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND, 0o644)
    firewall = Firewall(rt, storage['faces_256_root'], [qroot, '/tmp'], access_fd)
    sys.addaudithook(firewall)
    result.update(git_commit=git_commit(), git_dirty=False, storage=storage, qualification_root=str(qroot),
                  access_log=str(access_log))
    config, contracts = pio.verify_contract()
    ids = pio.identities(config, contracts)
    source = pio.verify_source(PCGANAdapter(config))
    result.update(identities=ids, source={'commit': source['commit'], 'tree': source['tree']})
    rows, sample_datasets, manifest = pio.read_pair_manifest()
    result['pair_manifest'] = manifest
    order = predict_order(torch, SEED, len(rows), pio.QUALIFICATION_ITERATIONS)
    authorized = {}
    expected_paths = []
    for i in order:
        r = rows[i]
        require(r['split'] == 'TRAIN', 'predicted pair is TRAIN')
        for key in ('source_spoof_id', 'target_live_id'):
            authorized[r[key]] = r['dataset']
            expected_paths.append(str(Path(storage['faces_256_root']) / r['dataset'] / f'{r[key]}.png'))
    firewall.authorized = authorized
    result['predicted'] = {'sampler_indices': order, 'pairs': [{k: rows[i][k] for k in pio.ALLOWLIST} for i in order],
                           'rule': 'Generator(60901): one int64 draw, then randperm(8838)[:3]'}
    precision = pr.configure_precision(torch)
    seeding = pr.seed_process(torch, MODE, SEED, config)
    env_identity, driver = pr.verify_environment(SEED)
    result.update(precision=precision, seeding=seeding, environment_identity_equals_lock=True,
                  environment=env_identity, gpu_driver=driver)
    models, crop, runtime = pr.build_models(pio.build_root(rt))
    result['runtime'] = runtime
    runner = TrainingRunner(models, crop)
    trainer = pr.E05Trainer(torch, runner, MODE, SEED)
    result['optimizers'] = trainer.membership
    reader, decode = pr.face_reader(storage['faces_256_root'], sample_datasets)
    loader, generator, loader_ev = pr.build_loader(torch, pr.PairDataset(rows, reader, decode), MODE, SEED, config)
    result['loader'] = loader_ev
    env, reasons = pr.environment_record(torch)
    result['refusals'] = {}
    # ------------------------------------------------------------------ case 'train': 3 real TRAIN iterations
    ctx = pr.E05RunContext(mode=MODE, seed=SEED, runtime_root=rt, environment=env, missing_environment_reasons=reasons,
                           identities=ids, case='train', command_line=' '.join([sys.executable] + sys.argv))
    result['train_run'] = {'run_dir': str(ctx.run_dir), 'run_id': ctx.run_id}
    require(not ctx.run_dir.is_relative_to(rt / 'runs') and ctx.run_dir != pio.run_root(rt, pio.SCIENTIFIC, 42),
            'qualification root cannot collide with a scientific root')
    isolation, records, adam = {}, [], []

    def isolation_probe(stage, live, state):
        if not records and stage in ('before_D', 'after_D', 'after_G'):      # iteration 1 only
            isolation[stage] = component_hashes(live.models)

    def on_record(record):
        records.append(record)
        adam.append(adam_steps(runner))
    from methods.dsdg import runner_io as rio
    with ctx:
        streams = rio.tee_run_logs(ctx.run_dir)
        try:
            ctx.log_event('e05_run_start', {'labels': ctx.labels(), 'seeding': seeding, 'precision': precision,
                                            'identities': ids, 'optimizers': trainer.membership, 'loader': loader_ev,
                                            'pair_manifest': manifest, 'execution_choices': pio.resolved_overlays()})
            expect_refusal(lambda: pr.run_iterations(trainer, loader, ctx, iterations=pio.TOTAL_ITERATIONS,
                                                     t_start=0.0), 'qualification_cannot_run_4000', result)
            t0 = time.monotonic()
            written = pr.run_iterations(trainer, loader, ctx, iterations=pio.QUALIFICATION_ITERATIONS, t_start=t0,
                                        extra_observer=isolation_probe, on_record=on_record)
            result['train_seconds'] = time.monotonic() - t0
            expect_refusal(lambda: pr.write_checkpoint(torch, ctx, trainer, pio.TOTAL_ITERATIONS),
                           'terminal_without_4000_iterations', result)
            expect_refusal(lambda: pr.write_checkpoint(torch, ctx, trainer, pio.TOTAL_ITERATIONS,
                                                       terminal_semantics_probe=True), 'probe_in_train_case', result)
            expect_refusal(lambda: pr.write_checkpoint(torch, ctx, trainer, 1), 'rewrite_periodic', result)
            result['train_written'] = written
            result['applications_end'] = trainer.applications()
            ctx.close(completion_status='interrupted', summary={
                'failure_reason': f'QUALIFICATION_STOP_AFTER_{pio.QUALIFICATION_ITERATIONS}_ITERATIONS '
                                  '(planned; not a failure)',
                'training_duration_seconds': result['train_seconds'], 'peak_vram_bytes': trainer.peak_gpu_memory_bytes,
                'benchmark_iterations': runner.benchmark_iteration, 'optimizer_applications': trainer.applications(),
                'labels': ctx.labels(), 'final_or_selected_checkpoint_path': None,
                'final_or_selected_checkpoint_sha256': None, 'missing_field_reasons': {
                    'final_or_selected_checkpoint_path': 'QUALIFICATION_ONLY: no scientific selection exists',
                    'final_or_selected_checkpoint_sha256': 'QUALIFICATION_ONLY: no scientific selection exists',
                    'seed_level_evaluation_metrics': 'QUALIFICATION_ONLY: not evaluated, not reportable'}})
        finally:
            rio.untee(streams)
    result['records'] = records
    result['adam_steps_after_each_iteration'] = adam
    result['isolation_iteration_1'] = isolation
    # ------------------------------------------------------------------ case 'terminal_probe': semantics only
    probe = pr.E05RunContext(mode=MODE, seed=SEED, runtime_root=rt, environment=env,
                             missing_environment_reasons=reasons, identities=ids, case='terminal_probe',
                             command_line=' '.join([sys.executable] + sys.argv))
    result['probe_run'] = {'run_dir': str(probe.run_dir), 'run_id': probe.run_id}
    with probe:
        expect_refusal(lambda: pr.write_checkpoint(torch, probe, trainer, 1), 'periodic_in_probe_case', result)
        apps = trainer.applications()
        result['probe_written'] = pr.write_checkpoint(torch, probe, trainer, pio.TOTAL_ITERATIONS,
                                                      terminal_semantics_probe=True)
        require(trainer.applications() == apps, 'probe performed no optimizer application')
        probe.close(completion_status='interrupted', summary={
            'failure_reason': ('QUALIFICATION_TERMINAL_SEMANTICS_PROBE (planned): the iteration-4000 '
                               f'path/type/selection semantics written from a {runner.benchmark_iteration}-iteration '
                               'qualification state; not a scientific completion'),
            'benchmark_iterations': runner.benchmark_iteration, 'optimizer_applications': trainer.applications(),
            'labels': probe.labels(), 'final_or_selected_checkpoint_path': None,
            'final_or_selected_checkpoint_sha256': None, 'missing_field_reasons': {
                'final_or_selected_checkpoint_path': 'QUALIFICATION_ONLY: no scientific selection exists',
                'final_or_selected_checkpoint_sha256': 'QUALIFICATION_ONLY: no scientific selection exists',
                'training_duration_seconds': 'terminal-semantics probe: no training in this root',
                'peak_vram_bytes': 'terminal-semantics probe: no training in this root',
                'seed_level_evaluation_metrics': 'QUALIFICATION_ONLY: not evaluated, not reportable'}})
    result['peak_cuda_memory_bytes'] = torch.cuda.max_memory_allocated()
    result['firewall'] = firewall.report()
    os.close(access_fd)
    result['access'] = access_audit(access_log, expected_paths)
    result['expected_face_paths'] = expected_paths
    result['run_files'] = {name: run_files(Path(result[name]['run_dir'])) for name in ('train_run', 'probe_run')}
    result['source_after'] = pio.verify_source(PCGANAdapter(config))['commit']
    result['production_files_sha256_after'] = pio.production_file_hashes()
    result['status'] = 'COLLECTED'


def run_files(run_dir):
    return {str(p.relative_to(run_dir)): {'bytes': p.stat().st_size, 'sha256': sha256_file(p)}
            for p in sorted(run_dir.rglob('*')) if p.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(args.output.is_relative_to('/tmp') and not args.output.exists() and args.output.suffix == '.json',
            'new /tmp JSON evidence')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {'mode': MODE, 'qualification_seed': SEED, 'experiment_seed': None, 'labels': list(pio.QUALIFICATION_LABELS),
              'status': 'STARTED', 'utc_start': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    try:
        qualify(result, args.output)
    except BaseException as exc:
        result.update(status='STOP_AND_REPORT', error_type=type(exc).__name__, error=str(exc),
                      traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        result['utc_end'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + '\n')
    print(json.dumps({'status': result['status'], 'output': str(args.output)}), flush=True)
    return 0 if result['status'] == 'COLLECTED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
