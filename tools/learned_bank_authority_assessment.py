#!/usr/bin/env python3
"""M8 learned-method bank authority assessment (E03 STDN, E05 PCGAN, E06b DSDG-NATIVE, E06c DSDG-BIN-IDFREE).

Two phases, following the seed-completion-audit pattern:

  collect  runs on the GPU host, read-only, standard library only, fed through stdin:
             nice -n 19 ionice -c3 python3 -I -B - collect --repo <gpu repo> < tools/learned_bank_authority_assessment.py
           It re-hashes the 12 approved final checkpoints, reads run summaries/manifests and exit markers, takes a
           lightweight E04 seed-2026 snapshot (process, metrics tail, checkpoint listing, exit marker) and the storage
           state. It reads NO face file, writes nothing on the host, never touches the E04 checkout or runtime, and
           never imports torch/TensorFlow.
  build    runs on the laptop. It compares the collected identities with the committed three-seed closure CSVs,
           checks the source citations behind every bank-contract statement against the pinned official sources in
           third_party/source_cache, and writes outputs/audit/LEARNED_BANK_AUTHORITY_ASSESSMENT.{json,md} and
           LEARNED_BANK_AUTHORITY_ASSESSMENT_SHA256.txt.

It authorizes nothing: no inference, no bank, no GPU use. Bank semantics that the frozen authorities leave open are
reported as owner decisions (configs/amendments/learned_bank_generation_contract.DRAFT.yaml); none is chosen here.
"""
import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

METHODS = ('E03', 'E05', 'E06b', 'E06c')
SEEDS = (42, 1337, 2026)
CLOSURE_CSV = {'E03': 'outputs/audit/E03_THREE_SEED_FINAL_CHECKPOINTS.csv',
               'E05': 'outputs/audit/E05_THREE_SEED_FINAL_CHECKPOINTS.csv',
               'E06b': 'outputs/audit/E06B_THREE_SEED_FINAL_CHECKPOINTS.csv',
               'E06c': 'outputs/audit/E06C_THREE_SEED_FINAL_CHECKPOINTS.csv'}
CLOSURE_JSON = {'E03': 'outputs/audit/E03_THREE_SEED_TRAINING_CLOSURE.json',
                'E05': 'outputs/audit/E05_THREE_SEED_TRAINING_CLOSURE.json',
                'E06b': 'outputs/audit/E06B_THREE_SEED_TRAINING_CLOSURE.json',
                'E06c': 'outputs/audit/E06C_THREE_SEED_TRAINING_CLOSURE.json'}
FAMILY_CLOSURE_COMMIT = {'E03': '374de40', 'E05': '7677c3a', 'E06b': '3b0d308', 'E06c': '24f7049'}
METHOD_CONFIG = {'E03': 'configs/methods/e03_stdn.yaml', 'E05': 'configs/methods/e05_pcgan.yaml',
                 'E06b': 'configs/methods/e06b_dsdg_native.yaml', 'E06c': 'configs/methods/e06c_dsdg_bin_idfree.yaml'}
# Exit markers written by each launch wrapper (E06b seed 42 succeeded as attempt 2 = *_r2).
EXIT_MARKER = {('E03', s): f'e03_s{s}.exit' for s in SEEDS}
EXIT_MARKER.update({('E05', s): f'e05_s{s}.exit' for s in SEEDS})
EXIT_MARKER.update({('E06c', s): f'e06c_s{s}.exit' for s in SEEDS})
EXIT_MARKER.update({('E06b', 42): 'e06b_s42_r2.exit', ('E06b', 1337): 'e06b_s1337.exit',
                    ('E06b', 2026): 'e06b_s2026.exit'})
E04_RUN_ID = 'e7100eb9a5ab0537'
E04_LAUNCH_COMMIT = '18ba00e94813f005876c8d6235b0b848226b2c1d'
E04_TOTAL_STEPS = 150000
METHOD_ENVS = ('gpat-m5', 'gpat-m6-e03', 'gpat-m6-e05', 'gpat-m6-e06c')
OUT = 'outputs/audit/LEARNED_BANK_AUTHORITY_ASSESSMENT'
BASE_COMMIT = '35ad1fdb242809aae822a413ba6f13179bb5060e'   # m6-baselines HEAD when this assessment started
N_TRAIN_SPOOF = 8838


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    return {'argv': cmd, 'rc': p.returncode, 'out': p.stdout}


def final_files(method, root):
    ck = root / 'checkpoints'
    if method == 'E03':
        return {k: ck / f'ckpt-50.{k}' for k in ('data-00000-of-00001', 'index', 'meta')}
    if method == 'E05':
        return {'pth': ck / 'iteration_4000.pth'}
    return {'pth': ck / 'netG_model_epoch_200_iter_0.pth'}


# ----------------------------------------------------------------------------------------------- collect (host)
def collect(repo):
    repo = Path(repo).expanduser().resolve()
    work = repo.parent
    rt = work / 'GPAT_TransferBench_runtime'
    out = {'schema': 'learned_bank_authority_collect_v1', 'collected_utc_start': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
           'host': os.uname().nodename, 'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
           'uptime_s': float(Path('/proc/uptime').read_text().split()[0]), 'runtime_root': str(rt)}
    out['gpu_repo'] = {'head': run(['git', '-C', str(repo), 'rev-parse', 'HEAD'])['out'].strip(),
                       'status_porcelain': run(['git', '-C', str(repo), 'status', '--porcelain'])['out']}
    runs = {}
    for m in METHODS:
        for s in SEEDS:
            root = rt / 'runs' / 'm6' / m / f'seed_{s}'
            rec = {'root': str(root), 'exists': root.is_dir(), 'is_symlink': root.is_symlink()}
            if root.is_dir():
                rec['entries'] = sorted(p.name for p in root.iterdir())
                rec['checkpoint_files'] = sorted(p.name for p in (root / 'checkpoints').iterdir())
                for name in ('run_summary.json', 'run_manifest.json'):
                    p = root / name
                    if p.is_file():
                        rec[name + '.sha256'] = sha256_file(p)
                        rec[name] = json.loads(p.read_text())
                rec['final'] = {}
                for key, p in final_files(m, root).items():
                    rec['final'][key] = ({'path': str(p.relative_to(root)), 'size': p.stat().st_size,
                                          'sha256': sha256_file(p), 'mode': oct(p.stat().st_mode & 0o777)}
                                         if p.is_file() else {'path': str(p.relative_to(root)), 'missing': True})
            em = work / EXIT_MARKER[(m, s)]
            rec['exit_marker'] = {'name': em.name, 'content': em.read_text().strip() if em.is_file() else None}
            runs[f'{m}.seed_{s}'] = rec
    out['runs'] = runs
    # E04 seed 2026: lightweight, read-only
    e04 = rt / 'runs' / 'm6' / 'E04' / 'seed_2026'
    snap = {'root': str(e04)}
    metrics = e04 / 'metrics.jsonl'
    if metrics.is_file():
        size = metrics.stat().st_size
        with open(metrics, 'rb') as f:
            f.seek(max(0, size - (1 << 16)))
            tail = f.read().decode('utf-8', 'replace').strip().splitlines()
        last = json.loads(tail[-1])
        snap['metrics'] = {'size': size, 'mtime': metrics.stat().st_mtime, 'last_global_step': last.get('global_step'),
                           'last_wall_clock_seconds': last.get('wall_clock_seconds'), 'last_epoch': last.get('epoch')}
    snap['checkpoints'] = sorted(p.name for p in (e04 / 'checkpoints').iterdir()) if (e04 / 'checkpoints').is_dir() else None
    snap['exit_marker_present'] = (work / 'e04_s2026.exit').exists()
    ps = run(['ps', '-eo', 'pid,etimes,stat,args'])['out'].splitlines()
    snap['processes'] = [l for l in ps if 'tools/run_e04.py' in l and 'python' in l]
    snap['gpu_checkout_head'] = out['gpu_repo']['head']
    out['e04_seed2026'] = snap
    out['gpu'] = run(['nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.used,memory.total,utilization.gpu',
                      '--format=csv,noheader'])['out'].strip()
    out['gpu_apps'] = run(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader'])['out'].strip()
    # storage
    out['storage'] = {
        'lsblk': run(['lsblk', '-J', '-b', '-o', 'NAME,PATH,SIZE,TYPE,FSTYPE,UUID,MOUNTPOINTS,MODEL,TRAN'])['out'],
        'df_T': run(['df', '-B1', '-T', '/', str(rt), '/media/ext_data'])['out'],
        'df_i': run(['df', '-i', '/', '/media/ext_data'])['out'],
        'findmnt_runtime': run(['findmnt', '-J', '-T', str(rt)])['out'],
        'findmnt_ext': run(['findmnt', '-J', '-T', '/media/ext_data'])['out'],
        'ext_stat': run(['stat', '-c', '%U:%G %a %F', '/media/ext_data'])['out'].strip(),
    }
    bank_sizes = {}
    for m in ('E01', 'E02'):
        for s in SEEDS:
            d = rt / 'runs' / 'm6' / m / f'seed_{s}'
            if d.is_dir():
                pngs = [p for p in d.rglob('*.png')]
                bank_sizes[f'{m}.seed_{s}'] = {'du_bytes': int(run(['du', '-sb', str(d)])['out'].split()[0]),
                                               'png_count': len(pngs), 'png_bytes': sum(p.stat().st_size for p in pngs)}
    out['e01_e02_bank_sizes'] = bank_sizes
    envs = {}
    for e in METHOD_ENVS:
        py = Path.home() / 'miniconda3' / 'envs' / e / 'bin' / 'python'
        envs[e] = {'python': str(py), 'exists': py.exists(),
                   'version': run([str(py), '-I', '-c', 'import sys;print(sys.version.split()[0])'])['out'].strip()
                   if py.exists() else None}
    out['envs'] = envs
    out['collected_utc_end'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    json.dump(out, sys.stdout, indent=1, sort_keys=True)
    sys.stdout.write('\n')


# ----------------------------------------------------------------------------------------------- build (laptop)
ROOT = Path(__file__).resolve().parents[1] if '__file__' in globals() else None
SRC = 'third_party/source_cache'
PIN_KEY = {'facexzoo/addition_module/DSDG/': ('dsdg', 'addition_module/DSDG/'), 'stdn/': ('stdn', ''),
           'swapping_autoencoder/': ('swapping_autoencoder', '')}

# Every statement the bank contract rests on, as (id, method(s), source file, line, exact substring).
# build() fails the gate if the pinned source no longer says exactly this at this line.
CITATIONS = (
    ('STDN.gen_call_train', 'E03', 'stdn/train.py', 44, "M, s, b, C, T = Gen(img, training_nn=training_nn, scope='STDN')"),
    ('STDN.recon', 'E03', 'stdn/train.py', 47, 'recon1 = (1-s)*img - b - tf.image.resize_images(C, [imsize, imsize]) - T'),
    ('STDN.trace', 'E03', 'stdn/train.py', 48, 'trace = img - recon1'),
    ('STDN.trace_warp', 'E03', 'stdn/train.py', 49, 'trace_warp = warping(trace[bsize:,...], reg, imsize)'),
    ('STDN.synth', 'E03', 'stdn/train.py', 50, 'synth1 = img[:bsize,...]+ trace_warp'),
    ('STDN.offset_direction', 'E03', 'stdn/model/dataset.py', 142, 'reg_map_sp = generate_offset_map(lm_sp, lm_li)'),
    ('STDN.input_scale', 'E03', 'stdn/model/dataset.py', 144, 'np.array(image_li,np.float32)/255'),
    ('STDN.test_inference_mode', 'E03', 'stdn/test.py', 55, '_step(config, dataset_test, False)'),
    ('STDN.test_batch1', 'E03', 'stdn/test.py', 49, 'config.BATCH_SIZE = 1'),
    ('STDN.test_raw_saver', 'E03', 'stdn/test.py', 58, 'saver = tf.train.Saver(max_to_keep=50,)'),
    ('STDN.weight_ema_maintained', 'E03', 'stdn/model/model.py', 122,
     'variable_averages = tf.train.ExponentialMovingAverage(config.MOVING_AVERAGE_DECAY, global_step)'),
    ('STDN.figure_clip', 'E03', 'stdn/model/utils.py', 24, 'fig = tf.clip_by_value(fig, 0.0, 1.0)'),
    ('STDN.figure_truncate', 'E03', 'stdn/model/utils.py', 37, 'img = tf.cast(column * 255, tf.uint8)'),
    ('PCGAN.basis_fresh_noise', 'E05', 'swapping_autoencoder/models/networks/stylegan2_layers.py', 342,
     'noise = image.new_empty(batch, 1, height, width).normal_()'),
    ('PCGAN.basis_tensor2im_scale', 'E05', 'swapping_autoencoder/util/util.py', 125,
     'image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0'),
    ('PCGAN.basis_tensor2im_clip', 'E05', 'swapping_autoencoder/util/util.py', 128, 'image_numpy = np.clip(image_numpy, 0, 255)'),
    ('PCGAN.basis_tensor2im_cast', 'E05', 'swapping_autoencoder/util/util.py', 131, 'return image_numpy.astype(imtype)'),
    ('DSDG.final_checkpoint', 'E06b,E06c', 'facexzoo/addition_module/DSDG/generated.py', 23,
     "default='./result/model_oulu_p1/netG_model_epoch_200_iter_0.pth'"),
    ('DSDG.cudnn_benchmark', 'E06b,E06c', 'facexzoo/addition_module/DSDG/generated.py', 36, 'cudnn.benchmark = True'),
    ('DSDG.eval', 'E06b,E06c', 'facexzoo/addition_module/DSDG/generated.py', 43, 'netG.eval()'),
    ('DSDG.batches', 'E06b,E06c', 'facexzoo/addition_module/DSDG/generated.py', 46, 'for n in range(200):'),
    ('DSDG.noise_identity_first', 'E06b,E06c', 'facexzoo/addition_module/DSDG/generated.py', 47,
     'noise = torch.zeros(100, args.hdim).normal_(0, 1)'),
    ('DSDG.noise_spoof_second', 'E06b,E06c', 'facexzoo/addition_module/DSDG/generated.py', 48,
     'noise_s = torch.zeros(100, args.hdim).normal_(0, 1)'),
    ('DSDG.latent_layout', 'E06b,E06c', 'facexzoo/addition_module/DSDG/generated.py', 49,
     'noise = torch.cat((noise_s, noise, noise), dim=1)'),
    ('DSDG.spoof_half', 'E06b,E06c', 'facexzoo/addition_module/DSDG/generated.py', 54, 'spoof = fake[:, 0:3, :, :].data.cpu().numpy()'),
    ('DSDG.truncate_no_clip', 'E06b,E06c', 'facexzoo/addition_module/DSDG/generated.py', 63,
     "save_img = np.transpose((255 * save_img).astype('uint8'), (1, 2, 0))"),
    ('DSDG.jpeg', 'E06b,E06c', 'facexzoo/addition_module/DSDG/generated.py', 65, "save_name = str(i) + '_scene.jpg'"),
    ('DSDG.sigmoid_output', 'E06b,E06c', 'facexzoo/addition_module/DSDG/networks/generator.py', 150,
     'img = torch.sigmoid(self.main(x))'),
    ('DSDG.netG_is_decoder_s', 'E06b,E06c', 'facexzoo/addition_module/DSDG/networks/__init__.py', 11,
     'netG = Decoder_s(hdim=hdim)'),
)

# Per-method frozen facts (with their authority) and the bank decisions that no authority fixes.
CONTRACTS = {
    'E03': {
        'name': 'STDN', 'track': 'A', 'fidelity': 'FAITHFUL_OFFICIAL_CORE_WITH_TRACK_A_CONTROLLED_DATA_ADAPTATION_AND_RUNTIME_COMPATIBILITY',
        'environment': 'gpat-m6-e03 (Python 3.8, TF1; environments/e03_stdn.lock.json)',
        'population': {'rows': N_TRAIN_SPOOF, 'unit': 'frozen pairs_train_v1 row', 'authority': 'spec 6, 7; A1 N_syn'},
        'frozen': [
            'checkpoint: checkpoints/ckpt-50 of seed s (OFFICIAL_LATEST_FINAL_CKPT_50; A2-06 clause 1)',
            'bank seed s uses generator seed s (evaluator contract D12 SEED_MATCHED_RAW)',
            'synthesis graph: trace = s*x_sp + b + resize(C) + T from Gen(x_sp); synth = x_li + warping(trace, offset(lm_sp, lm_li)) '
            '(train.py:44-50, dataset.py:142); spec 8.3 "spoof trace from source_spoof_id, geometric correction to target_live_id"',
            'inputs: canonical RGB256 uint8 / 255 (dataset.py:144, production runner); landmarks_px256 via e03_feed_pack_v1, '
            'which covers every pairs_train_v1 row',
            'output: one canonical 256x256 RGB uint8 synthetic spoof per pair (spec 7, 8.3)'],
        'open': ['E03-Q1 Gen inference mode (BN moving statistics vs batch statistics)',
                 'E03-Q2 weight source (raw variables vs weight-EMA shadow)',
                 'E03-Q3 inference-time augmentation (training brightness/flip off vs on)',
                 'E03-Q5 auxiliary outputs', 'Q4 float -> uint8 conversion', 'C-Q1 encoding',
                 'C-Q2 device/reproducibility', 'C-Q3 S7 acceptance', 'C-Q5 GPU timing', 'storage location (storage DRAFT)'],
        'auxiliary_required': 'none by spec (output only the synthetic spoof direction); trace maps would be optional diagnostics (E03-Q5)',
        'stochastic_at_inference': False,
    },
    'E05': {
        'name': 'PCGAN', 'track': 'A', 'fidelity': 'owner-approved controlled implementation resolutions (not native/faithful/official)',
        'environment': 'gpat-m6-e05 (Python 3.11, torch; environments lock in E05 closure)',
        'population': {'rows': N_TRAIN_SPOOF, 'unit': 'frozen pairs_train_v1 row', 'authority': 'spec 6, 7, 8.5; A1 N_syn'},
        'frozen': [
            'checkpoint: checkpoints/iteration_4000.pth of seed s (BASELINE_FINAL_STATE_V1; A2-06)',
            'bank seed s uses generator seed s (D12)',
            'generation path: mix = G(E(x_src).spatial, E(x_tgt).global), x_src = source_spoof (artifact), x_tgt = target_live '
            '(content) (spec 8.5; methods/pcgan/training_runner.py:63 explicit_pair)',
            'inputs: canonical RGB256 uint8 -> float32 (x/255-0.5)/0.5, no resize/crop/flip (E05 closure protocol.input_mapping)',
            'output: full 256x256 RGB synthetic image (spec 8.5)'],
        'open': ['E05-Q1 StyleGAN2 noise at inference (fresh noise is drawn on every forward)',
                 'Q4 float [-1,1] -> uint8 conversion', 'C-Q1 encoding', 'C-Q2 device/reproducibility',
                 'C-Q3 S7 acceptance', 'C-Q5 GPU timing',
                 'storage location (storage DRAFT)'],
        'auxiliary_required': 'none (spec 8.5 evaluates only the Live->Spoof mix)',
        'stochastic_at_inference': True,
    },
    'E06b': {
        'name': 'DSDG-NATIVE', 'track': 'B (secondary; SiW NOT_INSTANTIABLE_MISSING_SUBJECT_ID)',
        'fidelity': 'FAITHFUL_OFFICIAL_WITH_RUNTIME_COMPATIBILITY',
        'environment': 'gpat-m6-e06c (Python 3.11, torch; environments lock 91416a20...)',
        'population': {'rows': 'DEFERRED_TO_M8', 'unit': 'synthetic sample from noise (not a pair method)',
                       'authority': 'configs/methods/e06b_dsdg_native.yaml generation.n_syn_intended = DEFERRED_TO_M8'},
        'frozen': [
            'checkpoint: checkpoints/netG_model_epoch_200_iter_0.pth of seed s (OFFICIAL_GENERATOR_EPOCH_200; generated.py:23)',
            'bank seed s uses generator seed s (D12)',
            'generation graph: netG = Decoder_s(hdim=128); z = cat(noise_s, noise, noise); output = sigmoid RGB 256x256, '
            'channels 0:3 = spoof (generated.py:47-54; generator.py:150)',
            'DSDG does not consume the pair manifest; pairwise metrics N/A (spec 6, 8.6)'],
        'open': ['E06b-Q0 N_syn (8838 pooled vs 3720 native coverage) - deferred to M8 by the frozen config',
                 'DSDG-Q1 latent sampling stream and seed', 'DSDG-Q2 which output half', 'Q4 float -> uint8 conversion',
                 'DSDG-Q4 bank identity schema for non-pair rows', 'C-Q1 encoding (official JPEG vs PNG)',
                 'C-Q2 device/reproducibility (official cudnn.benchmark=True)', 'C-Q3 S7 acceptance', 'C-Q5 GPU timing',
                 'storage location (storage DRAFT)'],
        'auxiliary_required': 'none for the spoof bank; the paired live half is discarded unless DSDG-Q2 says otherwise',
        'stochastic_at_inference': True,
    },
    'E06c': {
        'name': 'DSDG-BIN-IDFREE', 'track': 'A', 'fidelity': 'CONTROLLED_ADAPTATION (DEV-020)',
        'environment': 'gpat-m6-e06c (Python 3.11, torch; environments lock 91416a20...)',
        'population': {'rows': N_TRAIN_SPOOF, 'unit': 'synthetic sample from noise (not a pair method)',
                       'authority': 'configs/methods/e06c_dsdg_bin_idfree.yaml n_syn_intended 8838; A1'},
        'frozen': [
            'checkpoint: checkpoints/netG_model_epoch_200_iter_0.pth of seed s (OFFICIAL_GENERATOR_EPOCH_200; A2-06 clause 1)',
            'bank seed s uses generator seed s (D12)',
            'generation graph identical to E06b (same pinned FaceX-Zoo 16b793a7, same Decoder_s); DEV-020 changes only '
            'training (pair relation, lambda_pair = 0, binary spoof class), not generation',
            'separate scientific identity: E06c config 7176dd4c, launch authority 84809868, own checkpoints and bank'],
        'open': ['DSDG-Q1 latent sampling stream and seed', 'DSDG-Q2 which output half', 'Q4 float -> uint8 conversion',
                 'DSDG-Q4 bank identity schema for non-pair rows', 'C-Q1 encoding', 'C-Q2 device/reproducibility',
                 'C-Q3 S7 acceptance', 'C-Q5 GPU timing',
                 'storage location (storage DRAFT)'],
        'auxiliary_required': 'none for the spoof bank',
        'stochastic_at_inference': True,
    },
}

# Proposed (not authorized) production plan; becomes executable only after the owner rulings, S7 acceptance and a
# per-family production authority record.
PRODUCTION_PROPOSAL = {
    'method_order': ['E06c', 'E05', 'E03', 'E06b'],
    'method_order_reason': 'Track-A methods first (E06c, E05, E03), smallest/fastest inference first; E06b (Track B, '
                           'budget deferred) last',
    'seed_order': [42, 1337, 2026],
    'source_commits': 'one new learned-bank authority commit (adapters + writer + approved contract) for all four methods; '
                      'checkpoints stay those of the training authorities d547fc9 (E03), efdf753 (E05), d9ecbef (E06b), '
                      '84809868 (E06c); the E04 checkout 18ba00e is never used or moved',
    'environments': {'E03': 'gpat-m6-e03', 'E05': 'gpat-m6-e05', 'E06b': 'gpat-m6-e06c', 'E06c': 'gpat-m6-e06c'},
    'gpu_memory_expected': {'E03': 'TF1 graph, batch 1; must set allow_growth (TF1 otherwise reserves the whole GPU); '
                                   'inference footprint to be measured in S7', 'E05': '< training peak 3.08 GB (no D, no grads)',
                            'E06b': 'Decoder_s only, batch 100: expected ~2-4 GB (to be measured in S7)',
                            'E06c': 'as E06b'},
    'runtime_expected': 'minutes per bank on an unshared 3090 (8838 forwards); E03 adds CPU offset-map interpolation '
                        'per pair (matplotlib triangulation)',
    'layout': {'clone': '~/workdir/m8_learned_banks/clones/<authority>', 'evidence': '~/workdir/m8_learned_banks/evidence/<M>_s<S>/',
               'bank_root': 'per the storage ruling (L1)', 'tmux': 'm8bank_<method>_s<seed> (remain-on-exit, pipe-pane log)',
               'exit_marker': '~/workdir/m8bank_<method>_s<seed>.exit'},
    'launch_preconditions': ['learned_bank_generation_contract.yaml OWNER_APPROVED', 'storage ruling approved',
                             'adapters qualified on synthetic fixtures (CPU) and S7 toy acceptance passed per method',
                             'E04 seed 2026 exited and audited (C-Q5 A) or concurrent use approved',
                             'checkpoint SHA256 re-verified at launch', 'free-space floor gate', 'clean isolated clone'],
    'bank_audit': 'per bank: row count, 0 or counted failures, file and pixel hashes, checkpoint hash, toy-prefix byte '
                  'equality with the S7 acceptance outputs, TRAIN-only O_NOATIME reads with VAL/TEST atime unchanged, '
                  'run-root listing digest; then a family closure per method',
}

GIB = 1 << 30
ROOT_FLOOR_BYTES = 24 * GIB          # tools/e01_e02_storage_preflight_record.py FLOOR_BYTES (E01/E02 authority)
RAW_RGB256_BYTES = 256 * 256 * 3


def _rel(p):
    return str(p.relative_to(ROOT))


def _pinned_sha(rel):
    pins = json.loads((ROOT / 'third_party/source_pins.json').read_text())['sources']
    for prefix, (key, sub) in PIN_KEY.items():
        if rel.startswith(prefix):
            cited = pins[key].get('cited_files', {})
            entry = cited.get(sub + rel[len(prefix):])
            return (entry or {}).get('sha256'), pins[key]['pinned_commit']
    return None, None


def check_citations():
    out = []
    for cid, methods, rel, line, text in CITATIONS:
        p = ROOT / SRC / rel
        rec = {'id': cid, 'methods': methods, 'file': f'{SRC}/{rel}', 'line': line, 'text': text}
        if not p.is_file():
            rec.update(ok=False, problem='source file missing')
        else:
            data = p.read_bytes()
            lines = data.decode('utf-8').split('\n')
            rec['sha256'] = hashlib.sha256(data).hexdigest()
            pinned, commit = _pinned_sha(rel)
            rec['pinned_commit'] = commit
            rec['pinned_sha256'] = pinned
            rec['matches_pin'] = None if pinned is None else pinned == rec['sha256']
            rec['ok'] = line <= len(lines) and text in lines[line - 1] and rec['matches_pin'] is not False
        out.append(rec)
    return out


def build(collected_path):
    c = json.loads(Path(collected_path).read_text())
    gates = []

    def gate(gid, ok, detail=None):
        gates.append({'gate': gid, 'pass': bool(ok), **({'detail': detail} if detail is not None else {})})

    anc = run(['git', '-C', str(ROOT), 'merge-base', '--is-ancestor', BASE_COMMIT, 'HEAD'])['rc'] == 0
    gate('A.base_commit_is_ancestor', anc, BASE_COMMIT)
    gate('A.collect_schema', c.get('schema') == 'learned_bank_authority_collect_v1')
    gate('A.host', c.get('host') == 'nd-System-Product-Name', c.get('host'))
    gate('A.gpu_checkout_is_e04_launch_and_clean',
         c['gpu_repo']['head'] == E04_LAUNCH_COMMIT and c['gpu_repo']['status_porcelain'] == '', c['gpu_repo'])

    # ---- frozen population facts the decision package rests on
    e06b_cfg = (ROOT / METHOD_CONFIG['E06b']).read_text()
    e06c_cfg = (ROOT / METHOD_CONFIG['E06c']).read_text()
    gate('A.e06b_n_syn_deferred_to_m8', '  n_syn_intended: DEFERRED_TO_M8\n' in e06b_cfg)
    gate('A.e06c_n_syn_8838', '\nn_syn_intended: 8838\n' in e06c_cfg)
    ev = json.loads((ROOT / 'configs/amendments/m8_downstream_evaluator_contract.yaml').read_text())
    gate('A.d12_seed_matched_raw', ev.get('approved') is True
         and ev['decisions']['D12_synthetic_input'] == 'SEED_MATCHED_RAW'
         and ev['pinned_execution_details']['D12'].startswith('detector seed s uses the bank of generator seed s'))
    smoke = json.loads((ROOT / 'configs/amendments/m8_smoke_test_contract.yaml').read_text())
    gate('A.s7_requires_own_bank_stage_acceptance', smoke.get('approved') is True and
         'Each learned method (E03-E06c) must pass its own bank-stage acceptance before its full production bank'
         in smoke['pinned_execution_details']['S7'] and smoke['authorizes']['production_banks'] is False)
    for p in ('configs/amendments/learned_bank_generation_contract.DRAFT.yaml',
              'configs/amendments/learned_bank_storage_location.DRAFT.yaml'):
        dp = ROOT / p
        draft = json.loads(dp.read_text()) if dp.is_file() else {}
        gate(f'A.draft_not_approved.{Path(p).name}', dp.is_file() and draft.get('approved') is False
             and draft.get('status', '').startswith('DRAFT'))

    # ---- checkpoint authority (12 runs)
    table = []
    for m in METHODS:
        rows = {int(r['seed']): r for r in csv.DictReader((ROOT / CLOSURE_CSV[m]).read_text().splitlines())}
        closure = json.loads((ROOT / CLOSURE_JSON[m]).read_text())
        gate(f'B.{m}.family_closure_status', closure.get('status') == 'SCIENTIFIC_TRAINING_COMPLETED_3_OF_3_SEEDS'
             and closure.get('failed_gates') == [], closure.get('status'))
        cfg_sha = sha256_file(ROOT / METHOD_CONFIG[m])
        gate(f'B.{m}.frozen_config_sha256', cfg_sha == closure['frozen']['config_sha256'], cfg_sha)
        for s in SEEDS:
            r, got = rows[s], c['runs'][f'{m}.seed_{s}']
            if m == 'E03':
                expected = {'data-00000-of-00001': r['final_data_sha256'], 'index': r['final_index_sha256'],
                            'meta': r['final_meta_sha256']}
                ck_path = r['final_checkpoint_prefix']
            else:
                key = {'E05': 'final_checkpoint_sha256', 'E06b': 'final_netG_sha256', 'E06c': 'final_checkpoint_sha256'}[m]
                path_key = {'E05': 'final_checkpoint_path', 'E06b': 'final_netG_path', 'E06c': 'final_checkpoint_path'}[m]
                expected, ck_path = {'pth': r[key]}, r[path_key]
            observed = {k: v.get('sha256') for k, v in got.get('final', {}).items()}
            rid = got.get('run_manifest.json', {}).get('run_id')
            summ = got.get('run_summary.json', {})
            ok_hash = observed == expected
            ok_exit = got['exit_marker']['content'] in ('0', 'E06C_EXIT=0')
            ok_run = rid == r['run_id']
            ok_summary = (summ.get('test_split_accessed') in (False, None)) and bool(summ.get('checkpoint_selection_rule'))
            gate(f'C.{m}.seed_{s}.root_present', got['exists'] and not got['is_symlink'])
            gate(f'C.{m}.seed_{s}.final_checkpoint_sha256', ok_hash, observed)
            gate(f'C.{m}.seed_{s}.run_id', ok_run, rid)
            gate(f'C.{m}.seed_{s}.exit_marker_zero', ok_exit, got['exit_marker'])
            gate(f'C.{m}.seed_{s}.summary_rule_and_test_flag', ok_summary,
                 {'checkpoint_selection_rule': summ.get('checkpoint_selection_rule'),
                  'test_split_accessed': summ.get('test_split_accessed')})
            table.append({'method': m, 'seed': s, 'run_id': r['run_id'], 'checkpoint': ck_path,
                          'sha256': expected if m == 'E03' else expected['pth'],
                          'source_commit': r.get('scientific_authority') or r.get('authority') or r.get('launch_authority'),
                          'config_sha256': closure['frozen']['config_sha256'],
                          'checkpoint_rule': closure['protocol']['checkpoint_rule'],
                          'final_step': r.get('final_global_step') or r.get('global_step') or 4000,
                          'seed_closure_commit': r['seed_closure_commit'], 'family_closure_commit': FAMILY_CLOSURE_COMMIT[m],
                          'test_access': r.get('TEST_access', r.get('test_access')),
                          'verified': all((got['exists'], ok_hash, ok_exit, ok_run, ok_summary))})

    # ---- source citations
    cites = check_citations()
    for rec in cites:
        gate(f'D.citation.{rec["id"]}', rec['ok'], None if rec['ok'] else rec)

    # ---- E04 (read-only snapshot)
    e = c['e04_seed2026']
    gate('E.e04_process_alive', len(e['processes']) == 1, e['processes'])
    gate('E.e04_no_exit_marker', e['exit_marker_present'] is False)

    # ---- storage
    st = c['storage']
    df_lines = [l.split() for l in st['df_T'].strip().splitlines()[1:]]
    df = {l[6]: {'device': l[0], 'fstype': l[1], 'size': int(l[2]), 'used': int(l[3]), 'avail': int(l[4])} for l in df_lines}
    e01 = [v['png_bytes'] / v['png_count'] for k, v in c['e01_e02_bank_sizes'].items() if k.startswith('E01')]
    per_png_e01 = sum(e01) / len(e01)
    n_banks = {'E03': 3, 'E05': 3, 'E06b': 3, 'E06c': 3}
    rows_per_bank = {'E03': N_TRAIN_SPOOF, 'E05': N_TRAIN_SPOOF, 'E06b': N_TRAIN_SPOOF, 'E06c': N_TRAIN_SPOOF}
    total_outputs = sum(n_banks[m] * rows_per_bank[m] for m in METHODS)
    est = {'outputs_total_if_E06b_8838': total_outputs,
           'per_output_bytes_E01_measured_mean': round(per_png_e01),
           'per_output_bytes_worst_case_incompressible_png': RAW_RGB256_BYTES + 1024,
           'manifest_and_audit_overhead_per_bank_bytes': 16 * 1024 * 1024,
           }
    est['typical_total_bytes'] = int(total_outputs * per_png_e01 + 12 * est['manifest_and_audit_overhead_per_bank_bytes'])
    est['worst_total_bytes'] = int(total_outputs * est['per_output_bytes_worst_case_incompressible_png']
                                   + 12 * est['manifest_and_audit_overhead_per_bank_bytes'])
    est['worst_per_family_bytes'] = est['worst_total_bytes'] // 4
    root_avail = df['/']['avail']
    e04_growth = 9 * (21185796 + 17435 + 2827870) + 220 * 1000 * 1000   # 9 more 15k-step checkpoints + metrics tail
    headroom = root_avail - ROOT_FLOOR_BYTES - e04_growth
    storage = {'root': df['/'], 'ext': df.get('/media/ext_data'), 'df_i': st['df_i'], 'ext_stat': st['ext_stat'],
               'root_floor_bytes': ROOT_FLOOR_BYTES, 'e04_future_growth_bytes_estimate': e04_growth,
               'root_headroom_above_floor_bytes': headroom, 'estimate': est,
               'all_four_families_fit_root_worst_case': headroom >= est['worst_total_bytes'],
               'all_four_families_fit_root_typical': headroom >= est['typical_total_bytes'],
               'one_family_fits_root_worst_case': headroom >= est['worst_per_family_bytes'],
               'ext_mounted': '/media/ext_data' in df,
               'ext_in_runtime_containment_contract': False,
               'recommendation': (
                   'Full learned-bank production for all four families is not proven safe on the root SSD above the '
                   'approved 24 GiB floor in the worst (incompressible) case. Either (a) run one family at a time on the '
                   'root SSD with the per-bank 24 GiB floor gate re-checked before every bank, or (b) approve the additive '
                   'storage-location amendment DRAFT (configs/amendments/learned_bank_storage_location.DRAFT.yaml) for a '
                   'dedicated owner-only bank root on /media/ext_data. Nothing is written to the HDD until approved.')}
    gate('F.ext_mounted_observed', storage['ext_mounted'])

    statuses = {m: 'BANK_BLOCKED_BY_SCIENTIFIC_AUTHORITY' for m in METHODS}
    failed = [g['gate'] for g in gates if not g['pass']]
    doc = {
        'schema': 'learned_bank_authority_assessment_v1',
        'classification': 'M8_LEARNED_BANK_AUTHORITY_ASSESSMENT',
        'authorizes': {'inference': False, 'bank_generation': False, 'gpu_use': False, 'hdd_writes': False},
        'base_commit': BASE_COMMIT,
        'collected': {k: c[k] for k in ('collected_utc_start', 'collected_utc_end', 'host', 'boot_id', 'gpu_repo', 'gpu',
                                        'gpu_apps', 'envs')},
        'collected_sha256': sha256_file(collected_path),
        'e04_seed2026': {**e, 'run_id': E04_RUN_ID, 'launch_commit': E04_LAUNCH_COMMIT, 'total_steps': E04_TOTAL_STEPS,
                         'scientific_completion': 'RUNNING_NOT_COMPLETE', 'interference': 'none: stat/read of metrics tail, '
                         'checkpoint listing and process table only'},
        'checkpoint_authority': table,
        'method_contracts': CONTRACTS,
        'citations': cites,
        'storage': storage,
        'decision_package': 'configs/amendments/learned_bank_generation_contract.DRAFT.yaml',
        'storage_draft': 'configs/amendments/learned_bank_storage_location.DRAFT.yaml',
        'bank_writer_note': 'methods/common/bank_writer.py is an E01/E02 production authority file (hash-pinned in '
                            'configs/amendments/e01_e02_production_authority.yaml); learned-method writers must live in a '
                            'separate module and must not modify it.',
        'production_proposal': PRODUCTION_PROPOSAL,
        'statuses': statuses,
        'gates': gates, 'gates_total': len(gates), 'gates_passed': len(gates) - len(failed), 'failed_gates': failed,
        'status': 'LEARNED_BANK_AUTHORITY_ASSESSED_OWNER_DECISIONS_REQUIRED' if not failed else 'NEEDS_REVIEW',
    }
    return doc


def write_outputs(doc, collected):
    js = ROOT / (OUT + '.json')
    if js.exists():
        raise SystemExit(f'refusing to overwrite {js}')
    js.write_text(json.dumps(doc, indent=1, sort_keys=True) + '\n')
    md = ROOT / (OUT + '.md')
    md.write_text(render_md(doc))
    shas = ''.join(f'{sha256_file(p)}  {_rel(p)}\n' for p in (Path(collected).resolve(), js, md))
    (ROOT / (OUT + '_SHA256.txt')).write_text(shas)
    print(json.dumps({'status': doc['status'], 'gates': f"{doc['gates_passed']}/{doc['gates_total']}",
                      'failed': doc['failed_gates']}))


def render_md(d):
    L = ['# M8 learned-method bank authority assessment (E03, E05, E06b, E06c)', '',
         f"Status: **{d['status']}** ({d['gates_passed']}/{d['gates_total']} gates). Base commit "
         f"`{d['base_commit']}`; host collection {d['collected']['collected_utc_start']}.",
         '', 'This record authorizes nothing. No inference, bank, GPU use or HDD write happened.', '',
         '## 1. Checkpoint authority (12 runs)', '',
         '| Method | Seed | Run | Checkpoint | SHA256 | Rule | Final step | Seed closure | Family closure | Verified |',
         '|---|---|---|---|---|---|---|---|---|---|']
    for r in d['checkpoint_authority']:
        sha = r['sha256']['data-00000-of-00001'][:16] + '… (data)' if isinstance(r['sha256'], dict) else r['sha256'][:16] + '…'
        L.append(f"| {r['method']} | {r['seed']} | `{r['run_id']}` | `{r['checkpoint']}` | `{sha}` | {r['checkpoint_rule']} | "
                 f"{r['final_step']} | `{r['seed_closure_commit'][:7]}` | `{r['family_closure_commit']}` | "
                 f"{'yes' if r['verified'] else 'NO'} |")
    L += ['', 'Full hashes (E03: data, index and meta) are in the JSON. Every hash was recomputed on the host from the '
          'run root and equals the committed three-seed closure CSV.', '',
          '## 2. E04 seed 2026 (read-only)', '']
    e = d['e04_seed2026']
    L += [f"- run `{e['run_id']}`, launch commit `{e['launch_commit']}`, GPU checkout `{e['gpu_checkout_head']}`",
          f"- last metrics global_step {e['metrics']['last_global_step']} of {e['total_steps']}; "
          f"latest checkpoint files: {', '.join(x for x in e['checkpoints'] if x.endswith('.index'))}",
          f"- process: {e['processes']}; exit marker present: {e['exit_marker_present']}; status {e['scientific_completion']}",
          f"- {e['interference']}", '', '## 3. Method bank contracts', '']
    for m, c in d['method_contracts'].items():
        L += [f"### {m} {c['name']} (Track {c['track']})", '', f"Fidelity: {c['fidelity']}. Environment: {c['environment']}.",
              f"Population: {c['population']['rows']} ({c['population']['unit']}; {c['population']['authority']}).", '',
              'Frozen:', ''] + [f'- {x}' for x in c['frozen']] + ['', 'Open (owner decisions):', ''] + \
             [f'- {x}' for x in c['open']] + ['', f"Auxiliary outputs: {c['auxiliary_required']}. "
                                              f"Stochastic at inference: {c['stochastic_at_inference']}.", '',
                                              f"Status: `{m}_{d['statuses'][m]}`", '']
    s = d['storage']
    L += ['## 4. Storage', '',
          f"- root `/` {s['root']['device']} {s['root']['fstype']}: {s['root']['avail'] / GIB:.1f} GiB free; floor "
          f"{s['root_floor_bytes'] / GIB:.0f} GiB; E04 growth reserve {s['e04_future_growth_bytes_estimate'] / GIB:.2f} GiB; "
          f"headroom above floor {s['root_headroom_above_floor_bytes'] / GIB:.1f} GiB",
          f"- `/media/ext_data` {s['ext']['device']} {s['ext']['fstype']}: {s['ext']['avail'] / GIB:.0f} GiB free; "
          f"`{s['ext_stat']}`; outside the runtime_root containment contract",
          f"- estimate for 12 banks x 8838 outputs: typical {s['estimate']['typical_total_bytes'] / GIB:.1f} GiB "
          f"(E01 mean {s['estimate']['per_output_bytes_E01_measured_mean']} B/PNG), worst case "
          f"{s['estimate']['worst_total_bytes'] / GIB:.1f} GiB (incompressible PNG); per family worst "
          f"{s['estimate']['worst_per_family_bytes'] / GIB:.1f} GiB",
          f"- all four families fit above the floor: typical {s['all_four_families_fit_root_typical']}, worst "
          f"{s['all_four_families_fit_root_worst_case']}; one family (worst): {s['one_family_fits_root_worst_case']}",
          f"- {s['recommendation']}", '', '## 5. Source citations', '',
          '| Id | File:line | Pin match | OK |', '|---|---|---|---|']
    for c in d['citations']:
        L.append(f"| {c['id']} | `{c['file']}:{c['line']}` | {c.get('matches_pin')} | {c['ok']} |")
    pp = d['production_proposal']
    L += ['', '## 6. Production proposal (not authorized)', '',
          f"- method order {pp['method_order']} ({pp['method_order_reason']}); seed order {pp['seed_order']}",
          f"- sources: {pp['source_commits']}", f"- environments: {pp['environments']}",
          f"- GPU memory: {pp['gpu_memory_expected']}", f"- runtime: {pp['runtime_expected']}",
          f"- layout: {pp['layout']}", '- preconditions: ' + '; '.join(pp['launch_preconditions']),
          f"- bank audit: {pp['bank_audit']}"]
    L += ['', '## 7. Gates', '', f"Failed: {d['failed_gates'] or 'none'}", '']
    return '\n'.join(L) + '\n'


if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'build':
    ap = argparse.ArgumentParser()
    ap.add_argument('phase')
    ap.add_argument('--collected', required=True)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    d = build(a.collected)
    if a.dry_run:
        print(json.dumps({'status': d['status'], 'failed': d['failed_gates'], 'storage': d['storage']['estimate']}, indent=1))
    else:
        write_outputs(d, a.collected)
    sys.exit(0)

if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'collect':
    ap = argparse.ArgumentParser()
    ap.add_argument('phase')
    ap.add_argument('--repo', required=True)
    collect(ap.parse_args().repo)
    sys.exit(0)
