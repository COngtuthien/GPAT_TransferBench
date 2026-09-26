"""M6D6e E07c auxiliary production-runner qualification harness: QUALIFICATION_ONLY, never scientific.

Every process uses qualification seed 60605 (never the auxiliary seed 42 or an experiment seed)
and the SAME production engine as tools/run_e07c_aux.py (methods/difffas/aux_runner.py:
population, transform, Dataset, DataLoader, K7 head, SGD, CE, step, epoch loss, whole-module
save). Real TRAIN data only: manifests/split_v1.parquet (TRAIN-filtered, allowlisted columns)
+ manifests/artifact_probe_classes_v1.json + the canonical faces of the 14467 TRAIN rows.
An audit-hook firewall denies every other manifest, every non-TRAIN face, faces_256
enumeration, <runtime_root>/runs, other benchmark data roots and foreign weight files, and logs
every benchmark-path event (DataLoader workers included) to an append-only audit file.

  b256   ONE exact B=256 production step (zero_grad, forward, CE, backward, SGD step) with
         stage-wise CUDA memory; process 1 and a fresh process 2 must agree bitwise.
         CUDA OOM at any stage -> BLOCKED_BY_AUX_B256_MEMORY (no fallback of any kind).
  epoch  ONE complete real-TRAIN epoch (56 x 256, drop_last) in
         <runtime_root>/qualification/m6d6e/E07c_aux/q60605-<run_id>/ with run_logging_v1 files and
         the epoch-boundary whole-module save; the checkpoint is SHA256-reloaded through the M6D6c
         secure loader, compared with the in-memory model, then removed.

Outputs are QUALIFICATION_ONLY / NOT_A_SCIENTIFIC_CHECKPOINT / NOT_ELIGIBLE_FOR_BANK /
NOT_ELIGIBLE_FOR_DOWNSTREAM / NOT_ELIGIBLE_FOR_REPORTING.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import warnings

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from methods.common.config import sha256_file  # noqa: E402
from methods.common.runlog import atomic_write_json  # noqa: E402
from methods.difffas import aux_runner as engine  # noqa: E402
from methods.difffas import aux_runner_io as aio  # noqa: E402
from methods.difffas import runtime_qualification as rq  # noqa: E402  (unchanged M6D6a helpers)

SEED = aio.QUALIFICATION_SEED
MODE = aio.QUALIFICATION
BUILD_PARTS = ('builds', 'e07c_difffas', 'm6d6e')
# DataLoader workers pass tensors through AF_UNIX sockets under TMPDIR (108-byte path limit): short dedicated dir.
SHORT_TMP = Path('/tmp/gpat-m6d6e')
CLEAN_GPU_MAX_USED_MIB = 1024
OUTPUTS = {'b256_1': 'M6D6E_E07C_AUX_B256_PROCESS_1.json', 'b256_2': 'M6D6E_E07C_AUX_B256_PROCESS_2.json',
           'epoch': 'M6D6E_E07C_AUX_ONE_EPOCH.json'}
ACCESS_LOG = 'M6D6E_E07C_AUX_ACCESS_{}.tsv'
PROBE = 'm6d6e_b256_process_1_probe.pt'
CATEGORIES = ('SPLIT_MANIFEST', 'CLASS_MAP', 'OTHER_MANIFEST', 'TRAIN_FACE', 'NON_TRAIN_FACE',
              'FACES_ENUMERATION_OR_WRITE', 'SCIENTIFIC_RUN_ROOT', 'OTHER_BENCHMARK_DATA', 'FOREIGN_WEIGHT')
ALLOWED_CATEGORIES = ('SPLIT_MANIFEST', 'CLASS_MAP', 'TRAIN_FACE')
WEIGHT_SUFFIXES = ('.pkl', '.pt', '.pth', '.ckpt', '.safetensors')
# M6D6a exact-argv exceptions plus the stdlib platform.platform() processor query (platform._Processor ->
# `uname -p`, read-only), issued by the run-manifest environment record (aux_runner.environment_record).
ALLOWED_EXACT_ARGV = tuple(rq.ALLOWED_EXACT_ARGV) + (['uname', '-p'],)
EXIT = {'PASS': 0, 'STOP_TRAINING_GATE': 3, 'STOP_RESOURCE_CONTAMINATION': 4, 'STOP_NONDETERMINISTIC_FIRST_STEP': 5,
        'STOP_EXISTING_QUALIFICATION_ROOT': 7, 'BLOCKED_BY_AUX_B256_MEMORY': 10}
require = aio.require


class Firewall:
    """Audit hook: split manifest + class map (read) and TRAIN canonical faces only; no other benchmark data.

    Every benchmark-path event (allowed or denied) is appended to `access_fd`, an O_APPEND
    descriptor inherited by forked DataLoader workers, so worker face reads are logged too.
    """
    WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
    EVENTS = ('open', 'os.listdir', 'os.scandir', 'os.mkdir', 'os.remove', 'os.rmdir', 'os.rename', 'shutil.rmtree',
              'subprocess.Popen')

    def __init__(self, runtime_root, faces_root, write_roots, weight_roots, allow_faces, access_fd):
        self.runtime_root, self.faces = Path(runtime_root), Path(faces_root)
        self.write_roots = [str(Path(p)) + '/' for p in write_roots]
        self.weight_roots = [str(Path(p)) + '/' for p in weight_roots]
        self.samples, self.allow_faces, self.access_fd = {}, allow_faces, access_fd
        self.split = str(ROOT / aio.SPLIT_MANIFEST)
        self.class_map = str(ROOT / aio.CLASS_MAP)
        self.events = dict.fromkeys(self.EVENTS, 0)
        self.denied, self.manifest_opens, self.face_opens_main_process, self.subprocesses = [], {}, 0, []

    def is_train_face(self, rel):
        return len(rel) == 2 and rel[1].endswith('.png') and self.samples.get(rel[1][:-4]) == rel[0]

    def category(self, event, path):
        p = Path(path)
        if p.is_relative_to(ROOT / 'manifests'):
            return {self.split: 'SPLIT_MANIFEST', self.class_map: 'CLASS_MAP'}.get(path, 'OTHER_MANIFEST')
        if p.is_relative_to(self.faces):
            if event != 'open':
                return 'FACES_ENUMERATION_OR_WRITE'
            return 'TRAIN_FACE' if self.is_train_face(p.relative_to(self.faces).parts) else 'NON_TRAIN_FACE'
        if p.is_relative_to(self.runtime_root / 'runs') or p.is_relative_to(ROOT / 'runs'):
            return 'SCIENTIFIC_RUN_ROOT'
        if p.is_relative_to(self.runtime_root / 'data') or any(p.is_relative_to(ROOT / d) for d in ('data', 'cache')):
            return 'OTHER_BENCHMARK_DATA'
        if p.suffix.lower() in WEIGHT_SUFFIXES and not self.in_roots(path, self.weight_roots):
            return 'FOREIGN_WEIGHT'
        return None

    @staticmethod
    def in_roots(path, roots):
        return any(path == r[:-1] or path.startswith(r) for r in roots)

    def writable(self, path):
        return path == '/dev/null' or path.startswith('/dev/shm/') or self.in_roots(path, self.write_roots)

    def reason(self, event, path, args, cat):
        writing = event not in ('open', 'os.listdir', 'os.scandir')
        if event == 'open':
            mode = args[1] if len(args) > 1 and isinstance(args[1], str) else ''
            flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
            writing = bool(flags & self.WRITE_FLAGS) or bool(set(mode) & set('wax+'))
        if cat in ('SPLIT_MANIFEST', 'CLASS_MAP'):
            if event == 'open' and not writing:
                self.manifest_opens[cat] = self.manifest_opens.get(cat, 0) + 1
                return None
            return 'manifest enumeration/write'
        if cat == 'TRAIN_FACE':
            if writing:
                return 'write into faces_256'
            if not self.allow_faces:
                return 'image bytes are not permitted before the population is bound'
            self.face_opens_main_process += 1
            return None
        if cat is not None:
            return cat.lower().replace('_', ' ')
        if p_suffix(path) == '.parquet':
            return 'parquet outside the frozen split manifest'
        if writing and not self.writable(path):
            return 'write outside the build/qualification/tmp roots'
        return None

    def __call__(self, event, args):
        if event not in self.events:
            return
        if event == 'subprocess.Popen':
            self.events[event] += 1
            argv = [os.fsdecode(a) for a in (args[1] or [])] if len(args) > 1 else []
            self.subprocesses.append(argv[:4])
            if Path(argv[0]).name not in rq.ALLOWED_SUBPROCESSES and argv not in ALLOWED_EXACT_ARGV:
                self.denied.append({'event': event, 'path': ' '.join(argv[:4]), 'reason': 'subprocess'})
                raise PermissionError('E07c M6D6e SUBPROCESS FIREWALL: ' + ' '.join(argv[:2]))
            return
        if not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        self.events[event] += 1
        paths = [args[0]] + ([args[1]] if event == 'os.rename' and len(args) > 1 else [])
        for raw in paths:
            path = os.path.abspath(os.fsdecode(raw))
            cat = self.category(event, path)
            why = self.reason(event, path, args, cat)
            if cat is not None and self.access_fd is not None:
                os.write(self.access_fd, f'{os.getpid()}\t{event}\t{cat}\t{"DENIED" if why else "ALLOWED"}\t'
                                         f'{path}\n'.encode('utf-8'))
            if why:
                self.denied.append({'event': event, 'path': path, 'reason': why})
                raise PermissionError(f'E07c M6D6e FIREWALL ({why}): {path}')

    def report(self):
        return {'denied': self.denied, 'event_counts': self.events, 'manifest_opens_python': self.manifest_opens,
                'face_opens_in_main_process': self.face_opens_main_process, 'subprocesses': self.subprocesses,
                'note': 'DataLoader worker processes inherit this hook (fork); their face opens are in the access '
                        'log. pyarrow opens the split manifest natively after the audited Python SHA256 open; its '
                        'API calls are recorded separately (pyarrow_calls).'}


def p_suffix(path):
    return Path(path).suffix.lower()


class PyarrowGuard:
    """Record every pyarrow.parquet entry point call; only the frozen split manifest may be passed."""
    NAMES = ('read_table', 'read_metadata', 'read_schema', 'ParquetFile', 'ParquetDataset', 'read_pandas')

    def __init__(self, pq, allowed):
        self.calls, self.allowed = [], str(allowed)
        for name in self.NAMES:
            original = getattr(pq, name)
            setattr(pq, name, self.wrap(name, original))

    def wrap(self, name, original):
        def guarded(source, *args, **kwargs):
            path = os.path.abspath(os.fspath(source))
            self.calls.append({'api': 'pyarrow.parquet.' + name, 'path': os.path.relpath(path, ROOT),
                               'columns': list(kwargs['columns']) if kwargs.get('columns') else None,
                               'filters': repr(kwargs.get('filters')) if kwargs.get('filters') else None})
            require(path == self.allowed, 'pyarrow may read only ' + aio.SPLIT_MANIFEST)
            return original(source, *args, **kwargs)
        return guarded


def access_audit(path, population_ids, consumed_ids=None):
    """Summarize the append-only benchmark access log of one process (and its DataLoader workers)."""
    rows = [line.split('\t') for line in Path(path).read_text(encoding='utf-8').splitlines()]
    require(all(len(r) == 5 for r in rows), 'access log format')
    counts = dict.fromkeys(CATEGORIES, 0)
    denied, face_opens, faces, pids = 0, 0, {}, set()
    for pid, event, cat, verdict, p in rows:
        counts[cat] += 1
        denied += verdict == 'DENIED'
        pids.add(pid)
        if cat == 'TRAIN_FACE':
            face_opens += 1
            faces[Path(p).stem] = faces.get(Path(p).stem, 0) + 1
    out = {'log': Path(path).name, 'log_sha256': sha256_file(path), 'events_logged': len(rows),
           'processes_logged': len(pids), 'counts_by_category': counts, 'denied_events': denied,
           'split_manifest_python_opens': counts['SPLIT_MANIFEST'], 'class_map_python_opens': counts['CLASS_MAP'],
           'unauthorized_manifest_accesses': counts['OTHER_MANIFEST'],
           'train_face_opens': face_opens, 'train_faces_distinct': len(faces),
           'train_faces_max_opens_per_face': max(faces.values()) if faces else 0,
           'train_faces_all_in_population': set(faces) <= set(population_ids),
           'non_train_face_accesses': counts['NON_TRAIN_FACE'],
           'VAL_image_reads': counts['NON_TRAIN_FACE'], 'TEST_image_reads': counts['NON_TRAIN_FACE'],
           'raw_or_other_benchmark_data_accesses': counts['OTHER_BENCHMARK_DATA'],
           'faces_enumeration_or_write': counts['FACES_ENUMERATION_OR_WRITE'],
           'scientific_run_root_accesses': counts['SCIENTIFIC_RUN_ROOT'],
           'foreign_weight_accesses': counts['FOREIGN_WEIGHT'],
           'image_access_note': ('Every canonical face path is classified against the 14467-row TRAIN population. '
                                 'VAL/TEST image reads are bounded above by the non-TRAIN face accesses, which must be 0; '
                                 'raw benchmark images live under <runtime_root>/data (OTHER_BENCHMARK_DATA), which must be 0.')}
    if consumed_ids is not None:
        out['train_faces_opened_equal_consumed_set'] = set(faces) == set(consumed_ids)
    return out


def instrument(torch, counters, *, saves_allowed, loads_allowed):
    """Count backward/zero_grad/step/save/load; forbid autograd.grad, checkpointing, autocast, GradScaler, schedulers."""
    tensor_backward, autograd_backward = torch.Tensor.backward, torch.autograd.backward
    zero_grad, sgd_step = torch.optim.SGD.zero_grad, torch.optim.SGD.step
    save, load = torch.save, torch.load

    def forbid(name):
        def forbidden(*a, **k):
            counters[name] += 1
            raise RuntimeError('M6D6e forbids ' + name)
        return forbidden

    def backward(self, *a, **k):
        counters['backward_calls'] += 1
        return tensor_backward(self, *a, **k)

    def autograd(*a, **k):
        counters['autograd_backward_calls'] += 1
        return autograd_backward(*a, **k)

    def zero(self, set_to_none=True):
        counters['zero_grad_calls'] += 1
        counters['zero_grad_set_to_none_values'] = sorted(set(counters['zero_grad_set_to_none_values']) | {set_to_none})
        return zero_grad(self, set_to_none=set_to_none)

    def step(self, *a, **k):
        counters['sgd_step_calls'] += 1
        return sgd_step(self, *a, **k)

    def counted_save(*a, **k):
        counters['torch_save_calls'] += 1
        require(counters['torch_save_calls'] <= saves_allowed, 'unexpected torch.save')
        return save(*a, **k)

    def counted_load(*a, **k):
        counters['torch_load_calls'] += 1
        require(counters['torch_load_calls'] <= loads_allowed, 'unexpected torch.load')
        return load(*a, **k)
    import torch.utils.checkpoint  # noqa: F401
    torch.Tensor.backward, torch.autograd.backward = backward, autograd
    torch.optim.SGD.zero_grad, torch.optim.SGD.step = zero, step
    for cls in {c for c in vars(torch.optim).values() if isinstance(c, type) and
                issubclass(c, torch.optim.Optimizer) and c is not torch.optim.SGD}:
        cls.step = forbid('other_optimizer_step_calls')
    torch.optim.lr_scheduler.LRScheduler.__init__ = forbid('scheduler_constructions')
    torch.nn.utils.clip_grad_norm_ = forbid('grad_clipping_calls')
    torch.autograd.grad = forbid('autograd_grad_calls')
    torch.utils.checkpoint.checkpoint = forbid('activation_checkpoint_calls')
    torch.autocast.__init__ = forbid('autocast_entries')
    torch.amp.GradScaler.__init__ = forbid('grad_scaler_constructions')
    torch.save, torch.load = counted_save, counted_load


def new_counters():
    return {'backward_calls': 0, 'autograd_backward_calls': 0, 'zero_grad_calls': 0, 'zero_grad_set_to_none_values': [],
            'sgd_step_calls': 0, 'other_optimizer_step_calls': 0, 'scheduler_constructions': 0,
            'grad_clipping_calls': 0, 'autograd_grad_calls': 0, 'activation_checkpoint_calls': 0,
            'autocast_entries': 0, 'grad_scaler_constructions': 0, 'torch_save_calls': 0, 'torch_load_calls': 0}


def gpu_snapshot():
    used = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
                                   text=True).strip().split(',')
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'],
                                   text=True).strip()
    return {'used_mib': int(used[0]), 'total_mib': int(used[1]),
            'compute_processes': [{'pid': a.split(',')[0].strip(), 'used_mib': a.split(',')[1].strip()}
                                  for a in apps.splitlines() if a.strip()]}


def cuda_memory(torch):
    torch.cuda.synchronize()
    return {'allocated_bytes': torch.cuda.memory_allocated(), 'max_allocated_bytes': torch.cuda.max_memory_allocated(),
            'reserved_bytes': torch.cuda.memory_reserved(), 'max_reserved_bytes': torch.cuda.max_memory_reserved()}


def state_digest(model):
    params = {n: engine.tsha(p) for n, p in model.named_parameters()}
    buffers = {n: engine.tsha(b) for n, b in model.named_buffers()}
    return {'parameters': params, 'buffers': buffers,
            'parameters_aggregate_sha256': aio.sha(json.dumps(params, sort_keys=True).encode()),
            'buffers_aggregate_sha256': aio.sha(json.dumps(buffers, sort_keys=True).encode())}


def rng_hashes(torch):
    import random
    import numpy as np
    return {'python_sha256': aio.sha(repr(random.getstate()).encode()),
            'numpy_sha256': aio.sha(repr(np.random.get_state()[1].tolist()).encode() + repr(np.random.get_state()[2:]).encode()),
            'torch_cpu_sha256': aio.sha(torch.get_rng_state().numpy().tobytes()),
            'torch_cuda_sha256': aio.sha(torch.cuda.get_rng_state().numpy().tobytes())}


# ================================================================= setup shared by every mode
def setup(build, mode, *, saves_allowed, loads_allowed):
    storage = aio.faces_root_from_exec_config()
    runtime_root = Path(storage['runtime_root'])
    require(build == runtime_root.joinpath(*BUILD_PARTS), 'dedicated build root')
    qroot = runtime_root.joinpath(*aio.QUALIFICATION_PARTS)
    access_log = build / ACCESS_LOG.format(mode)
    access_fd = os.open(access_log, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND, 0o644)   # fresh per process
    # active before any manifest byte is read; faces resolve only once the population is bound
    firewall = Firewall(runtime_root, storage['faces_256_root'], [build, qroot, SHORT_TMP], [build, qroot],
                        allow_faces=False, access_fd=access_fd)
    sys.addaudithook(firewall)
    import pyarrow.parquet as pq
    guard = PyarrowGuard(pq, ROOT / aio.SPLIT_MANIFEST)
    contract = aio.load_contract()                           # every bound input SHA256 (split + class map included)
    ids = aio.identities(contract)
    source, semantics, adapter, contracts = rq.source_identity()
    config = adapter.config
    require((source['commit'], source['tree']) == (ids['source_commit'], ids['source_tree']) and
            source['worktree_status'] == '', 'pinned source identity, clean')
    from methods.difffas import execution_policy as ep
    from methods.difffas.aux_training_qualification import upstream_training_semantics
    policy = ep.load_policy(config)
    require(policy['sha256'] == ids['a7_overlay_sha256'], 'A7 identity')
    upstream = upstream_training_semantics(source)
    require({k: upstream['loader'][k] for k in ('batch_size', 'shuffle', 'num_workers', 'drop_last')} ==
            {'batch_size': 256, 'shuffle': True, 'num_workers': 6, 'drop_last': True} and upstream['epochs'] == 200,
            'pinned pretrain_classifier.py loader/epochs')
    static = rq.contract(adapter, contracts)
    records, datasets, population = aio.read_train_population(pq=pq)
    firewall.samples, firewall.allow_faces = dict(datasets), True
    result = {'milestone': 'M6D6e', 'mode': mode, 'label': 'QUALIFICATION_ONLY', 'qualification_seed': SEED,
              'auxiliary_encoder_training_seed': None, 'experiment_seed': None, 'scientific_seed_consumed': False,
              'seed_note': '60605 seeds this qualification only; auxiliary 42 and main 42/1337/2026 are not used',
              'labels': list(aio.QUALIFICATION_LABELS), 'runner_mode': MODE, 'storage': storage, 'identities': ids,
              'contract_path': aio.CONTRACT_PATH, 'contract': static, 'population': population,
              'pyarrow_calls': guard.calls, 'source_before': source, 'executable_semantics': semantics,
              'upstream_training_semantics': upstream,
              'a7': {'overlay_sha256': policy['sha256'], 'document': policy['policy']['amendment_document']},
              'm6d6e_file_sha256': aio.m6d6e_file_hashes(), 'pid': os.getpid(),
              'launch_environment': {k: os.environ.get(k) for k in (
                  'CUDA_VISIBLE_DEVICES', 'PYTHONHASHSEED', 'PYTHONDONTWRITEBYTECODE', 'PYTHONNOUSERSITE',
                  'NVIDIA_TF32_OVERRIDE', 'CUBLAS_WORKSPACE_CONFIG', 'TMPDIR')}}
    require(result['launch_environment']['PYTHONHASHSEED'] == str(SEED), f'PYTHONHASHSEED={SEED}')
    result['gpu_before'] = gpu_snapshot()
    foreign = [p for p in result['gpu_before']['compute_processes'] if int(p['pid']) != os.getpid()]
    result['resource_clean'] = not foreign and result['gpu_before']['used_mib'] <= CLEAN_GPU_MAX_USED_MIB
    if not result['resource_clean']:
        result.update(status='STOP_RESOURCE_CONTAMINATION', foreign_compute_processes=foreign)
        return result, None
    import inspect
    import torch
    import torchvision.transforms as transforms
    rq.TORCH_LOAD_WEIGHTS_ONLY_DEFAULT = repr(inspect.signature(torch.load).parameters['weights_only'].default)
    counters = new_counters()
    instrument(torch, counters, saves_allowed=saves_allowed, loads_allowed=loads_allowed)
    result['precision'] = engine.configure_precision(torch, config)
    lock = json.loads((ROOT / engine.LOCK_PATH).read_text())
    env = rq.environment()
    diff = {k: {'lock': v, 'runtime': env.get(k)} for k, v in lock['identity'].items()
            if k != 'launch_environment' and env.get(k) != v}
    require(not diff, 'runtime identity equals environment lock: ' + json.dumps(diff))
    strip = lambda d: {k: v for k, v in d.items() if k not in ('PYTHONHASHSEED', 'TMPDIR')}
    require(strip(env['launch_environment']) == strip(lock['identity']['launch_environment']), 'launch env = lock')
    result['environment_before'] = env
    result['cuda_mem_get_info_before'] = dict(zip(('free_bytes', 'total_bytes'), torch.cuda.mem_get_info()))
    ctx = dict(torch=torch, transforms=transforms, config=config, records=records, datasets=datasets, ids=ids,
               firewall=firewall, counters=counters, runtime_root=runtime_root, storage=storage, source=source,
               access_log=access_log, access_fd=access_fd, env=env, guard=guard)
    return result, ctx


def finish(result, ctx, consumed_ids=None):
    torch, fw = ctx['torch'], ctx['firewall']
    result['counters'] = ctx['counters']
    result['source_after'] = rq.source_identity()[0]
    require(result['source_after'] == ctx['source'], 'pinned source unchanged')
    result['environment_after'] = rq.environment()
    require(result['environment_after'] == ctx['env'], 'environment stable')
    from methods.difffas import execution_policy as ep
    result['precision_state_after'] = ep.precision_state()
    require(result['precision_state_after'] == ep.EXPECTED_STATE, 'A7 state intact to the end')
    result['gpu_after'] = gpu_snapshot()
    c = ctx['counters']
    require(all(c[k] == 0 for k in ('other_optimizer_step_calls', 'scheduler_constructions', 'grad_clipping_calls',
                                     'autograd_grad_calls', 'activation_checkpoint_calls', 'autocast_entries',
                                     'grad_scaler_constructions')), 'no forbidden call')
    result['firewall'] = fw.report()
    require(not fw.denied, 'firewall denials')
    require(all(Path(a[0]).name in rq.ALLOWED_SUBPROCESSES or a in ALLOWED_EXACT_ARGV for a in fw.subprocesses),
            'subprocess allowlist')
    require(all(call['path'] == aio.SPLIT_MANIFEST for call in ctx['guard'].calls) and
            [c_['api'] for c_ in ctx['guard'].calls] == ['pyarrow.parquet.read_metadata', 'pyarrow.parquet.read_schema',
                                                         'pyarrow.parquet.read_table'], 'pyarrow calls')
    os.close(ctx['access_fd'])          # workers have exited: every audited event is in the log
    audit = access_audit(ctx['access_log'], ctx['datasets'], consumed_ids)
    result['benchmark_access_audit'] = audit
    require(audit['denied_events'] == 0 and audit['train_faces_all_in_population'] and
            all(v == 0 for k, v in audit['counts_by_category'].items() if k not in ALLOWED_CATEGORIES) and
            audit['VAL_image_reads'] == audit['TEST_image_reads'] == 0 and
            audit['split_manifest_python_opens'] == 2 and audit['class_map_python_opens'] == 2,
            'TRAIN-only benchmark access audit')
    result['access_summary'] = {
        'split_manifest_physical_reads': {'python_sha256_open': audit['split_manifest_python_opens'],
                                          'pyarrow_api_calls': len(ctx['guard'].calls),
                                          'pyarrow_note': 'native opens (footer + allowlisted column chunks of the '
                                                          'single all-split row group); filtered in C++'},
        'split_manifest_python_opens_note': ('SHA256 only: one by aux_runner_io.load_contract (bound input), one by '
                                             'read_train_population before parsing'),
        'class_map_reads': audit['class_map_python_opens'],
        'class_map_reads_note': 'one read by aux_runner_io.load_contract (bound SHA256), one by read_class_map',
        'TRAIN_rows_exposed_to_dataset': len(ctx['records']), 'VAL_rows_exposed_to_dataset': 0,
        'TEST_rows_exposed_to_dataset': 0, 'TRAIN_canonical_image_reads': audit['train_face_opens'],
        'TRAIN_canonical_images_distinct': audit['train_faces_distinct'],
        'VAL_image_reads': 0, 'TEST_image_reads': 0, 'unauthorized_manifest_reads': 0, 'raw_dataset_image_reads': 0,
        'firewall_denials': 0}
    result.update(VAL_access=False, TEST_access=False, authorized_TRAIN_access=True, scientific_training=False,
                  scientific_checkpoint_created=False, scientific_run_root_written=False, synthetic_bank=False,
                  main_difffas_training=False, downstream_evaluation=False, fidelity='CONTROLLED_ADAPTATION',
                  source_patch='NONE')
    return result


# ================================================================= B256 feasibility (+ fresh-process repeatability)
def first_batch_evidence(torch, trainer, batch):
    x, y, idx = batch
    indices = idx.tolist()
    ids = [trainer.records[i]['sample_id'] for i in indices]
    hist = [0] * aio.K
    for v in y.tolist():
        hist[v] += 1
    return {'indices_sha256': aio.sha(json.dumps(indices).encode()), 'sample_ids': ids,
            'sample_ids_sha256': aio.order_sha256(ids), 'class_histogram': hist,
            'class_histogram_named': dict(zip(aio.CLASSES, hist)),
            'input': {'shape': list(x.shape), 'dtype': str(x.dtype), 'min': float(x.min()), 'max': float(x.max()),
                      'mean': float(x.double().mean()), 'sha256': engine.tsha(x),
                      'within_minus1_plus1': bool(x.min() >= -1.0 and x.max() <= 1.0)},
            'target': {'shape': list(y.shape), 'dtype': str(y.dtype), 'min': int(y.min()), 'max': int(y.max()),
                       'sha256': engine.tsha(y)}}


def transform_check(torch, trainer, batch):
    """Row 0 of the loader batch equals the source transform of the canonical face; Resize is an identity copy."""
    import numpy as np
    x, _, idx = batch
    sid = trainer.records[int(idx[0])]['sample_id']
    img = trainer.dataset.reader(sid)
    resize = trainer.dataset.transform.transforms[0]
    resized = resize(img)
    row = trainer.dataset.transform(img)
    manual = (torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float().div(255) - 0.5) / 0.5
    return {'sample_id': sid, 'pil_mode': img.mode, 'pil_size': list(img.size),
            'resize_identity_bytes': resized.tobytes() == img.tobytes() and resized.size == img.size,
            'loader_row_equals_transform': bool(torch.equal(row, x[0])),
            'manual_formula_bitwise': bool(torch.equal(manual, x[0])),
            'rgb_channel_order': 'PIL RGB -> ToTensor channel 0 = R',
            'main_process_face_reads': 1}


def b256_mode(build, process):
    name = f'b256_{process}'
    result, ctx = setup(build, name, saves_allowed=1 if process == 1 else 0, loads_allowed=1 if process == 2 else 0)
    if ctx is None:
        return result
    torch = ctx['torch']
    result['seeding'] = engine.seed_process(torch, MODE, SEED, ctx['config'])
    result['rng_after_seeding'] = rng_hashes(torch)
    memory = {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with engine.production_components(torch, ctx['transforms'], ctx['config'], ctx['records'], ctx['datasets'],
                                          ctx['storage']['faces_256_root'], MODE, SEED) as trainer:
            memory['after_model_construction'] = cuda_memory(torch)
            result['components'] = trainer.evidence
            before = state_digest(trainer.model)
            result['initial_state'] = {k: before[k] for k in ('parameters_aggregate_sha256', 'buffers_aggregate_sha256')}
            result['rng_before_iterator'] = rng_hashes(torch)
            it = iter(trainer.loader)
            batch = next(it)
            result['first_batch'] = first_batch_evidence(torch, trainer, batch)
            captured = {}

            def on_stage(stage):
                memory[stage] = cuda_memory(torch)
                if stage == 'after_backward':
                    captured['gradients'] = {n: p.grad.detach().cpu().clone() for n, p in trainer.named
                                             if p.grad is not None}
            try:
                r = trainer.step(batch, 1, 0, on_stage)
            except engine.AuxMemoryBlocked as oom:
                memory['at_oom'] = cuda_memory(torch)
                del it
                result.update(status='BLOCKED_BY_AUX_B256_MEMORY', oom_stage=oom.stage, oom_message=str(oom),
                              memory=memory, batch_size=aio.BATCH_SIZE, fallback_attempted=False,
                              completed_backward_calls=trainer.backward_calls,
                              completed_optimizer_steps=trainer.optimizer_applications,
                              precision='A7 FP32 (unchanged)', warnings=sorted({f'{w.category.__name__}: {w.message}'
                                                                                for w in caught}))
                return finish(result, ctx)
            memory['peak_during_step'] = {'max_allocated_bytes': torch.cuda.max_memory_allocated(),
                                          'max_reserved_bytes': torch.cuda.max_memory_reserved()}
            result['transform_check'] = transform_check(torch, trainer, batch)
            require(result['transform_check']['resize_identity_bytes'] and
                    result['transform_check']['loader_row_equals_transform'], 'source transform on canonical faces')
            del it                                   # shut the workers down; nothing else is consumed
            after = state_digest(trainer.model)
            grads = captured['gradients']
            grad_sha = {n: engine.tsha(g) for n, g in grads.items()}
            momentum = {n: engine.tsha(trainer.optimizer.state[p]['momentum_buffer']) for n, p in trainer.named
                        if p in trainer.optimizer.state}
            changed = sorted(n for n in before['parameters'] if before['parameters'][n] != after['parameters'][n])
            unchanged = sorted(set(before['parameters']) - set(changed))
            result['step'] = {k: r[k] for k in ('epoch', 'global_step', 'batch_size', 'batch_sample_sha256',
                                                  'class_histogram', 'loss', 'loss_hex', 'learning_rate',
                                                  'gradients', 'step_seconds')}
            result['step'].update(
                gradient_sha256=grad_sha, gradient_aggregate_sha256=aio.sha(json.dumps(grad_sha, sort_keys=True).encode()),
                gradients_all_finite=all(bool(torch.isfinite(g).all()) for g in grads.values()),
                parameters_after_aggregate_sha256=after['parameters_aggregate_sha256'],
                buffers_after_aggregate_sha256=after['buffers_aggregate_sha256'],
                momentum_aggregate_sha256=aio.sha(json.dumps(momentum, sort_keys=True).encode()),
                momentum_buffers=len(momentum), parameter_tensors_changed=len(changed),
                parameter_tensors_unchanged=unchanged,
                buffers_changed=sum(before['buffers'][n] != after['buffers'][n] for n in before['buffers']),
                optimizer_applications=trainer.optimizer_applications, backward_calls=trainer.backward_calls)
            require(unchanged == sorted(engine.DISCONNECTED) and len(changed) == len(before['parameters']) - 2,
                    'every connected parameter updated; norm.weight/norm.bias untouched')
            result['rng_after_step'] = rng_hashes(torch)
            probe = {'parameters_after': {n: p.detach().cpu().clone() for n, p in trainer.named}, 'gradients': grads}
    result['memory'] = memory
    result['warnings'] = sorted({f'{w.category.__name__}: {w.message}' for w in caught})
    if process == 1:
        path = build / PROBE
        require(not path.exists(), 'fresh probe')
        torch.save(dict(probe, labels=list(aio.QUALIFICATION_LABELS), kind='M6D6E_REPEATABILITY_PROBE_NOT_A_CHECKPOINT'),
                   path)
        result['probe'] = {'path': PROBE, 'bytes': path.stat().st_size, 'sha256': sha256_file(path),
                           'removed_by': 'process 2 after comparison'}
    else:
        reference = json.loads((build / OUTPUTS['b256_1']).read_text())
        ref_probe = torch.load(build / PROBE, map_location='cpu', weights_only=True)
        result['difference_vs_process_1'] = {
            'parameters_after': tensor_diff(torch, ref_probe['parameters_after'], probe['parameters_after']),
            'gradients': tensor_diff(torch, ref_probe['gradients'], probe['gradients'])}
        del ref_probe
        result['comparison_vs_process_1'] = compare(reference, json.loads(json.dumps(result, sort_keys=True)))
        os.remove(build / PROBE)
        result['probe_removed'] = not (build / PROBE).exists()
    require(ctx['counters']['backward_calls'] == ctx['counters']['sgd_step_calls'] == ctx['counters']['zero_grad_calls']
            == 1, 'exactly one zero_grad, backward and SGD step')
    finish(result, ctx)
    result['status'] = 'PASS'
    if process == 2 and not result['comparison_vs_process_1']['all_bitwise_equal']:
        result['status'] = 'STOP_NONDETERMINISTIC_FIRST_STEP'
    return result


def tensor_diff(torch, a, b):
    require(list(a) == list(b), 'probe tensor inventory')
    bitwise, max_abs, nd, na = 0, 0.0, 0.0, 0.0
    for k in a:
        x, y = a[k].double(), b[k].double()
        bitwise += int(torch.equal(a[k], b[k]))
        max_abs = max(max_abs, float((x - y).abs().max()))
        nd += float((x - y).pow(2).sum())
        na += float(x.pow(2).sum())
    return {'tensors': len(a), 'bitwise_equal_tensors': bitwise, 'max_abs_diff': max_abs,
            'relative_l2': nd ** 0.5 / max(na ** 0.5, 1e-300)}


BITWISE_FIELDS = (('rng_after_seeding',), ('initial_state',), ('rng_before_iterator',), ('first_batch', 'indices_sha256'),
                  ('first_batch', 'sample_ids_sha256'), ('first_batch', 'class_histogram'), ('first_batch', 'input'),
                  ('first_batch', 'target'), ('step', 'loss_hex'), ('step', 'batch_sample_sha256'),
                  ('step', 'gradient_aggregate_sha256'), ('step', 'parameters_after_aggregate_sha256'),
                  ('step', 'buffers_after_aggregate_sha256'), ('step', 'momentum_aggregate_sha256'),
                  ('rng_after_step',))


def compare(reference, fresh):
    """Torch-free field comparison (also re-derived by the static preflight)."""
    def get(d, path):
        for k in path:
            d = d[k]
        return d
    rows = {'.'.join(p): get(reference, p) == get(fresh, p) for p in BITWISE_FIELDS}
    diff = fresh.get('difference_vs_process_1')
    if diff is not None:
        rows['parameter_tensors_bitwise'] = diff['parameters_after']['bitwise_equal_tensors'] == diff['parameters_after']['tensors']
        rows['gradient_tensors_bitwise'] = diff['gradients']['bitwise_equal_tensors'] == diff['gradients']['tensors']
    return {'fields': rows, 'all_bitwise_equal': all(rows.values()), 'tolerance': 'NONE (bitwise)'}


# ================================================================= one real-TRAIN epoch
def epoch_mode(build):
    result, ctx = setup(build, 'epoch', saves_allowed=1, loads_allowed=1)
    if ctx is None:
        return result
    torch = ctx['torch']
    from methods.difffas.aux_checkpoint import load_verified_whole_module
    env, reasons = engine.environment_record(torch)
    run = engine.E07cAuxRunContext(mode=MODE, seed=SEED, runtime_root=ctx['runtime_root'], environment=env,
                                   missing_environment_reasons=reasons, identities=ctx['ids'],
                                   command_line=' '.join([sys.executable] + sys.argv))
    result['run_dir'], result['run_id'] = str(run.run_dir), run.run_id
    if run.run_dir.exists():
        result['status'] = 'STOP_EXISTING_QUALIFICATION_ROOT'
        return result
    result['seeding'] = engine.seed_process(torch, MODE, SEED, ctx['config'])
    result['rng_after_seeding'] = rng_hashes(torch)
    first = {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with engine.production_components(torch, ctx['transforms'], ctx['config'], ctx['records'], ctx['datasets'],
                                          ctx['storage']['faces_256_root'], MODE, SEED) as trainer:
            result['components'] = trainer.evidence
            result['initial_state'] = {k: v for k, v in state_digest(trainer.model).items() if k.endswith('sha256')}
            result['rng_before_iterator'] = rng_hashes(torch)

            def on_step(r):
                if r['global_step'] == 1:
                    d = state_digest(trainer.model)
                    first.update(loss_hex=r['loss_hex'], batch_sample_sha256=r['batch_sample_sha256'],
                                 class_histogram=r['class_histogram'],
                                 parameters_after_aggregate_sha256=d['parameters_aggregate_sha256'],
                                 buffers_after_aggregate_sha256=d['buffers_aggregate_sha256'])
            run.open()
            streams = aio.tee_run_logs(run.run_dir)
            try:
                run.log_event('e07c_aux_run_start', {'labels': list(aio.QUALIFICATION_LABELS),
                                                     'seeding': result['seeding'], 'precision': result['precision'],
                                                     'population': ctx_population(result), 'components': trainer.evidence})
                torch.cuda.reset_peak_memory_stats()
                t0 = time.monotonic()
                try:
                    summary, losses = trainer.run_epoch(1, run, t0, on_step)
                except engine.TrainingStop as stop:
                    status = ('BLOCKED_BY_AUX_B256_MEMORY' if isinstance(stop, engine.AuxMemoryBlocked)
                              else 'STOP_TRAINING_GATE')
                    result.update(status=status, stop=str(stop), stage=stop.stage, completed_steps=trainer.global_step)
                    run.close(completion_status='failed', failure_reason=str(stop))
                    return result
                result['epoch_seconds'] = time.monotonic() - t0
                result['epoch1'] = summary
                result['epoch1_losses_hex'] = [float(v).hex() for v in losses]
                result['epoch1_losses'] = losses
                result['first_step'] = dict(first)
                result['optimizer_applications'] = trainer.optimizer_applications
                result['backward_calls'] = trainer.backward_calls
                result['peak_step_gpu_memory_allocated_bytes'] = trainer.peak_step_gpu_memory_bytes
                ck = trainer.checkpoint_event(1, run, ctx['config'])
                result['checkpoint_event'] = ck
                final = aio.checkpoint_path(run.run_dir)
                result['checkpoint_path'] = str(final)
                result['checkpoint_outside_runs'] = not final.is_relative_to(ctx['runtime_root'] / 'runs')
                result['partial_absent_after_replace'] = not final.with_name(final.name + '.partial').exists()
                live = state_digest(trainer.model)
                run.close(completion_status='interrupted', summary={
                    'failure_reason': 'QUALIFICATION_STOP_AFTER_EPOCH_1 (planned; not a failure)',
                    'training_duration_seconds': result['epoch_seconds'],
                    'peak_vram_bytes': trainer.peak_step_gpu_memory_bytes,
                    'final_or_selected_checkpoint_path': None, 'final_or_selected_checkpoint_sha256': None,
                    'labels': list(aio.QUALIFICATION_LABELS), 'source_epoch_loss': summary['source_epoch_loss'],
                    'missing_field_reasons': {
                        'final_or_selected_checkpoint_path': 'QUALIFICATION_ONLY: no scientific final state exists',
                        'final_or_selected_checkpoint_sha256': 'QUALIFICATION_ONLY: no scientific final state exists',
                        'seed_level_evaluation_metrics': 'QUALIFICATION_ONLY: not evaluated, not reportable'}})
            finally:
                aio.untee(streams)
        model = trainer.model                          # custom_rn import scope closed; the object is retained
        # M6D6c secure loader: SHA256 of the exact bytes -> pinned custom_rn -> weights_only=False -> identity -> cuda
        with load_verified_whole_module(final, ck['sha256'], ctx['config']) as loaded:
            ls, ms = loaded.state_dict(), model.state_dict()
            same = list(ls) == list(ms) and all(torch.equal(ls[k], ms[k]) for k in ms)
            result['checkpoint_reload'] = {
                'loader': 'methods/difffas/aux_checkpoint.py::load_verified_whole_module', 'sha256_verified_first': True,
                'class': f'{type(loaded).__module__}.{type(loaded).__name__}',
                'fc': [loaded.fc.in_features, loaded.fc.out_features], 'device': str(next(loaded.parameters()).device),
                'state_dict_entries': len(ls), 'state_dict_equal_in_memory': same,
                'in_memory_parameters_aggregate_sha256': live['parameters_aggregate_sha256']}
            del ls, ms
        require(result['checkpoint_reload']['state_dict_equal_in_memory'], 'reloaded whole module equals the in-memory model')
        result['checkpoint_file_before_cleanup'] = {'bytes': final.stat().st_size, 'sha256': sha256_file(final)}
        require(result['checkpoint_file_before_cleanup']['sha256'] == ck['sha256'], 'bytes unchanged before cleanup')
        os.remove(final)
        index = json.loads(run.path('checkpoint_index').read_text())
        index['qualification_bytes_removed_after_evidence'] = {
            'path': 'checkpoints/' + aio.CHECKPOINT_NAME, 'sha256': ck['sha256'], 'bytes': ck['file_size_bytes'],
            'reason': 'QUALIFICATION_ONLY checkpoint removed after size/SHA256/identity/reload evidence was recorded'}
        atomic_write_json(run.path('checkpoint_index'), index)
        result['checkpoint_cleanup'] = {'removed': not final.exists(),
                                        'checkpoints_dir_listing': sorted(p.name for p in final.parent.iterdir()),
                                        'weight_files_in_run_dir': sorted(str(p.relative_to(run.run_dir)) for p in
                                                                          run.run_dir.rglob('*') if p.suffix in WEIGHT_SUFFIXES)}
        require(result['checkpoint_cleanup']['removed'] and not result['checkpoint_cleanup']['weight_files_in_run_dir'],
                'qualification checkpoint removed')
        result['run_files'] = {str(p.relative_to(run.run_dir)): {'bytes': p.stat().st_size, 'sha256': sha256_file(p)}
                               for p in sorted(run.run_dir.rglob('*')) if p.is_file() and not p.name.startswith('.')}
        result['metrics_records'] = [json.loads(x).get('record_type') for x in run.path('metrics').read_text().splitlines()]
        result['warnings'] = sorted({f'{w.category.__name__}: {w.message}' for w in caught})
    for line in run.path('metrics').read_text().splitlines():
        rec = json.loads(line)
        if rec.get('record_type') == 'event' and rec['event'] == 'e07c_aux_epoch_complete':
            result['logged_epoch_summary_equal'] = rec['payload'] == result['epoch1']
    require(result.get('logged_epoch_summary_equal') is True, 'epoch summary logged to metrics.jsonl')
    require(aio.order_sha256(trainer.last_epoch_order) == result['epoch1']['epoch_sample_order_sha256'], 'consumed order')
    finish(result, ctx, trainer.last_epoch_order)
    c = ctx['counters']
    require(trainer.optimizer_applications == c['sgd_step_calls'] == c['backward_calls'] == c['zero_grad_calls'] ==
            aio.STEPS_PER_EPOCH and c['autograd_backward_calls'] == aio.STEPS_PER_EPOCH and
            c['torch_save_calls'] == c['torch_load_calls'] == 1, 'accounting: 56 zero_grad/backward/step, 1 save, 1 load')
    a = result['benchmark_access_audit']
    require(a['train_face_opens'] == aio.CONSUMED_PER_EPOCH and a['train_faces_distinct'] == aio.CONSUMED_PER_EPOCH and
            a['train_faces_opened_equal_consumed_set'], 'TRAIN face reads = exactly the 14336 consumed samples')
    result['status'] = 'PASS'
    return result


def ctx_population(result):
    return {k: result['population'][k] for k in ('train_rows_materialized', 'class_counts', 'population_order_sha256',
                                                 'rows_by_dataset')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('b256', 'epoch'), required=True)
    parser.add_argument('--process', type=int, choices=(1, 2), default=None)
    parser.add_argument('--build-root', type=Path, required=True)
    args = parser.parse_args()
    build = args.build_root.resolve()
    require(not build.is_relative_to(ROOT) and build.parts[-3:] == BUILD_PARTS, 'dedicated build root')
    require((args.mode == 'b256') == (args.process is not None), '--process only with --mode b256')
    tmp = Path(os.environ.get('TMPDIR', '/'))
    require(tmp == SHORT_TMP and tmp.is_dir() and not tmp.is_symlink() and tmp.resolve() == tmp,
            'TMPDIR is the dedicated short directory ' + str(SHORT_TMP))
    key = f'b256_{args.process}' if args.mode == 'b256' else 'epoch'
    out = build / OUTPUTS[key]
    require(not out.exists(), 'fresh output ' + out.name)
    if key == 'b256_2':
        require((build / OUTPUTS['b256_1']).is_file(), 'process 1 evidence first')
        require(json.loads((build / OUTPUTS['b256_1']).read_text())['status'] == 'PASS', 'process 1 passed')
    if key == 'epoch':
        for k in ('b256_1', 'b256_2'):
            require(json.loads((build / OUTPUTS[k]).read_text())['status'] == 'PASS', 'B256 + repeatability passed first')
    started = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    result = b256_mode(build, args.process) if args.mode == 'b256' else epoch_mode(build)
    result['started_utc'], result['ended_utc'] = started, time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'status': result['status'], 'output': str(out)}))
    sys.exit(EXIT[result['status']])


if __name__ == '__main__':
    main()
