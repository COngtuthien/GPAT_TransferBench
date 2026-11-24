"""E05 B1/B2 real-runtime qualification probe for ONE seed per process (load-only; no generation, no face read)."""
import json, os, sys, time, traceback
from pathlib import Path
sys.path.insert(0, os.getcwd())
SEED = int(os.environ['PROBE_SEED'])
OTHER = {42: 1337, 1337: 2026, 2026: 42}[SEED]
RT = '/home/student20261/workdir/GPAT_TransferBench_runtime'
out = {'seed': SEED, 'pid': os.getpid(), 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
import torch
from methods.common import learned_bank as lb
from methods.pcgan import production_io as pio
from methods.pcgan.bank_adapter import load_real
out['torch'] = [torch.__version__, torch.version.cuda, torch.backends.cudnn.version(), torch.cuda.get_device_name(0)]
try:
    a, ck = load_real(torch, SEED, Path(RT) / 'runs/m6/E05' / ('seed_%d' % SEED), pio.build_root(RT))
    i = a.identity()
    bc = i['runtime_closure']['build_closure']
    out['load'] = {'ok': True, 'checkpoint_identity': ck['checkpoint_identity_sha256'],
                   'checkpoint_files': {k: v['sha256'] for k, v in ck['files'].items()},
                   'tensor2im_runtime': i['tensor2im_runtime'], 'tensor2im_segment_sha256': i['tensor2im_segment_sha256'],
                   'source_identity': bc['source_identity'], 'import_resolution': bc['import_resolution'],
                   'extensions': bc['extensions'], 'manifest_sha256': bc['manifest_sha256'],
                   'jit_build_invoked': bc['jit_build_invoked'], 'precision': i['precision'], 'guards': i['training_guards'],
                   'eval_mode': [not a.E.training, not a.G.training], 'device': str(a.device),
                   'params': {'E': sum(p.numel() for p in a.E.parameters()), 'G': sum(p.numel() for p in a.G.parameters())}}
    del a
    torch.cuda.empty_cache()
except Exception as exc:
    out['load'] = {'ok': False, 'error': '%s: %s' % (type(exc).__name__, exc), 'traceback': traceback.format_exc()[-1500:]}
try:
    load_real(torch, OTHER, Path(RT) / 'runs/m6/E05' / ('seed_%d' % OTHER), pio.build_root(RT))
    out['second_seed_in_same_process'] = {'refused': False}
except Exception as exc:
    out['second_seed_in_same_process'] = {'refused': True, 'error': '%s: %s' % (type(exc).__name__, exc)}
try:
    lb.verify_checkpoint('E05', SEED, Path(RT) / 'runs/m6/E05' / ('seed_%d' % OTHER))
    out['other_seed_checkpoint_path'] = {'refused': False}
except Exception as exc:
    out['other_seed_checkpoint_path'] = {'refused': True, 'error': '%s: %s' % (type(exc).__name__, exc)}
out['peak_cuda_memory_allocated_bytes'] = torch.cuda.max_memory_allocated()
print(json.dumps(out, sort_keys=True, default=str))
