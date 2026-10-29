"""E04 Physics-STD production-readiness audit (CONTROLLED_ADAPTATION; IMPLEMENTED_NOT_EXECUTED).

Two phases; neither trains, fits, renders or opens a benchmark image:

  collect  (GPU host, `python3 -I -B - collect --requal-dir <dir> < this file`) -- prints one JSON document: the
           fixed 3DDFA asset and Sim3DR extension identities, the E03/E04 environment interpreters and key package
           versions, GPU / process / boot state, E04 run-root absence, the GPU checkout state, manifest and E03 feed-pack
           hashes, and the sha256 of every file in a synthetic re-qualification directory.
  build    (repository) -- runs the static stage-A checks (frozen config = snapshot, M6D3d hash-bound contract, A4
           overlay, pinned source, fixed assets, Q140), compares the synthetic re-qualification outputs with the
           committed M6D3a/M6D3e evidence, and writes outputs/audit/E04_PRODUCTION_READINESS_AUDIT.{json,md}, the
           re-qualification evidence copies and a SHA-256 list. The decision register lists every production decision
           the frozen authority leaves open; nothing in it is chosen here.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

RUNTIME = '/home/student20261/workdir/GPAT_TransferBench_runtime'
REPO_GPU = '/home/student20261/workdir/GPAT_TransferBench'
ENVS = {'gpat-m6-e03': ('tensorflow', 'numpy', 'scipy', 'cv2', 'yaml'),
        'gpat-m6-e04-geometry': ('torch', 'torchvision', 'numpy', 'scipy', 'cv2', 'Cython', 'yaml')}
MANIFESTS = ('manifests/split_v1.parquet', 'manifests/pairs_train_v1.parquet', 'manifests/val_pairs_v1.parquet',
             'configs/methods/e04_physics_std.yaml', 'environments/e04_geometry.runtime.json',
             'environments/e04_training.lock.json', 'environments/e04_geometry.lock.json')
EVIDENCE = {'record': 'outputs/audit/E04_PRODUCTION_READINESS_AUDIT.json',
            'report': 'outputs/audit/E04_PRODUCTION_READINESS_AUDIT.md',
            'sha256': 'outputs/audit/E04_PRODUCTION_READINESS_AUDIT_SHA256.txt',
            'requal': 'outputs/audit/E04_READINESS_REQUAL_%s.json'}
REQUAL = {'M6D3A_CPU': 'm6d3a_cpu.json', 'M6D3A_CUDA': 'm6d3a_cuda.json', 'M6D3E_PROCESS': 'm6d3e_process.json'}
STATUS = 'E04_BLOCKED_BY_SOURCE_GAP'
PAIRS = {'TRAIN': 8838, 'VAL': 1905}
TRAIN_ROWS_CONSUMED = 150000 * 4                     # 4 (target_live, source_spoof) rows per batch-8 minibatch


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(str(path), 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


# ============================================================================ collect (GPU host, read-only)
def collect(requal_dir):
    def sh(*a, cwd=None):
        r = subprocess.run(list(a), stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, cwd=cwd)
        return r.stdout.strip()
    rt = Path(RUNTIME)
    weights = rt / 'third_party_weights' / 'tddfa_v2'
    manifest = json.loads((Path(REPO_GPU) / 'environments/e04_geometry.runtime.json').read_text())
    ext = Path(manifest['sim3dr']['extension_path'])
    envs = {}
    for name, mods in ENVS.items():
        py = Path.home() / 'miniconda3' / 'envs' / name / 'bin' / 'python'
        code = ('import importlib, json, sys\nout = {"python": sys.version.split()[0]}\n'
                'for m in %r:\n    try:\n        out[m] = getattr(importlib.import_module(m), "__version__", "present")\n'
                '    except Exception as e:\n        out[m] = "MISSING: " + type(e).__name__\nprint(json.dumps(out))' % (mods,))
        r = subprocess.run([str(py), '-I', '-B', '-c', code], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                           universal_newlines=True, env={'CUDA_VISIBLE_DEVICES': '', 'PATH': '/usr/bin:/bin',
                                                         'TF_CPP_MIN_LOG_LEVEL': '3'})
        envs[name] = {'interpreter': str(py), 'exists': py.is_file(),
                      'interpreter_sha256': sha256_file(py.resolve()) if py.is_file() else None,
                      'modules': json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 else None}
    procs = [x.strip() for x in sh('ps', '-eo', 'pid,args').splitlines()[1:]
             if 'python' in x and any(k in x for k in ('tools/run_e0', 'training_runtime', 'bank', 'gpat'))
             and str(os.getpid()) != x.split()[0]]
    pack = rt / 'data/processed/e03_feed_pack_v1/pack_manifest.json'
    rq = Path(requal_dir)
    return {
        'phase': 'collect', 'host': sh('hostname'),
        'gpu': sh('nvidia-smi', '--query-gpu=name,driver_version,uuid', '--format=csv,noheader'),
        'gpu_compute_apps': sh('nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'),
        'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip().replace('-', ''),
        'method_processes': procs, 'tmux_sessions': sorted(sh('tmux', 'ls', '-F', '#{session_name}').splitlines()),
        'git': {'head': sh('git', 'rev-parse', 'HEAD', cwd=REPO_GPU),
                'branch': sh('git', 'branch', '--show-current', cwd=REPO_GPU),
                'status_porcelain': sh('git', 'status', '--porcelain', cwd=REPO_GPU)},
        'assets': {str(p.relative_to(weights)): {'bytes': p.stat().st_size, 'sha256': sha256_file(p)}
                   for p in sorted(weights.rglob('*')) if p.is_file()},
        'sim3dr_extension': {'path': str(ext), 'exists': ext.is_file(),
                             'sha256': sha256_file(ext) if ext.is_file() else None,
                             'expected_sha256': manifest['sim3dr']['sha256']},
        'environments': envs,
        'files_sha256': {f: sha256_file(Path(REPO_GPU) / f) for f in MANIFESTS},
        'e03_feed_pack_manifest_sha256': sha256_file(pack) if pack.is_file() else None,
        'faces_256_root_exists': (rt / 'data/processed/faces_256').is_dir(),
        'runs_m6': sorted(os.listdir(str(rt / 'runs/m6'))), 'e04_run_root_exists': (rt / 'runs/m6/E04').exists(),
        'disk_free_bytes': shutil.disk_usage(RUNTIME).free,
        'requal_dir': str(rq), 'requal_files_sha256': {p.name: sha256_file(p) for p in sorted(rq.iterdir())
                                                       if p.is_file()},
    }


# ============================================================================ build (repository)
def flat(o, p=''):
    out = {}
    if isinstance(o, dict):
        for k, v in o.items():
            out.update(flat(v, p + '.' + k))
    else:
        out[p] = o
    return out


def requal_comparison(root, requal):
    ref_a = (root / 'outputs/audit/M6D3A_E04_GEOMETRY_RUNTIME_QUALIFICATION.json').read_text()
    ref_e = json.loads((root / 'outputs/audit/M6D3E_E04_SYNTHETIC_PROCESS_1.json').read_text())
    ref_e2 = json.loads((root / 'outputs/audit/M6D3E_E04_SYNTHETIC_PROCESS_2.json').read_text())
    out = {}
    for key in ('M6D3A_CPU', 'M6D3A_CUDA'):
        d = requal[key]
        e = d['end_to_end']
        out[key] = {'regressor_repeat_byte_identical': d['regressor']['repeat_byte_identical'],
                    'end_to_end_repeat_byte_identical': e['repeat_byte_identical'],
                    'depth_sha256': e['depth_sha256'], 'depth_sha256_in_committed_m6d3a': e['depth_sha256'] in ref_a,
                    'mesh_sha256': e['mesh_sha256'], 'face_mask_pixels': e['face_mask_pixels'],
                    'spoof_exact_zeros': d['spoof']['exact_float32_zeros_32x32'],
                    'benchmark_images_decoded': d['benchmark_images_decoded'], 'optimizer_steps': d['optimizer_steps'],
                    'file_access_denied': d['file_access_audit'].get('denied')}
    p = requal['M6D3E_PROCESS']
    fp, fr = flat(p), flat(ref_e)
    common = [k for k in fp if k in fr]
    diff = sorted(k for k in common if fp[k] != fr[k])

    def step_diff(a, b):
        return [sorted(s for s in x['post_state'] if x['post_state'][s] != y['post_state'][s])
                for x, y in zip(a['steps'], b['steps'])]
    new_vs_ref = step_diff(p, ref_e)
    ref_pair = step_diff(ref_e, ref_e2)
    out['M6D3E_PROCESS'] = {
        'status': p['status'], 'optimizer_applications': p['optimizer_applications'],
        'tests': {k: p['tests'][k] for k in ('tests', 'failures', 'errors', 'skipped')},
        'graph_def_sha256': p['graph_def_sha256'], 'graph_def_equals_committed': p['graph_def_sha256'] ==
        ref_e['graph_def_sha256'], 'fields_common': len(common), 'fields_differing': diff,
        'post_state_differences_per_step': [len(x) for x in new_vs_ref],
        'post_state_difference_networks_per_step': [sorted({k.split('/')[1] for k in x}) for x in new_vs_ref],
        'committed_process_pair_differences_per_step': [len(x) for x in ref_pair],
        'step1_new_differences': new_vs_ref[0],
        'firewall_denied': p['firewall']['denied']}
    return out


def decisions():
    return [
        {'id': 'D1', 'topic': '3DDFA crop/ROI policy for canonical 256 faces', 'scientific': True,
         'blocks': ['depth-M0 pre-pass', 'Q140 correspondences', 'stages E-K'],
         'why_open': ('A3 sets faceboxes_used=false ("pre-cropped canonical 256 faces") but freezes no roi_box; '
                      'methods/physics_std/regressor.py states crop/ROI is outside the qualification. The pinned '
                      'TDDFA.__call__ has two official policies, and the canonical frame differs by dataset (SCRFD 1.25x '
                      'crop for MSU/SiW; CASIA pre-cropped 112 upscaled to 256)'),
         'options': {'A': ('official crop_policy="landmark": parse_roi_box_from_landmark on the frozen M2 '
                           'landmarks_px256 (68 iBUG, FaceXFormer; already on the host in e03_feed_pack_v1 for all '
                           'TRAIN/VAL samples); dataset-uniform; same landmark geometry as E03 (spec 8.4 "reuse the '
                           'STDN-compatible geometric alignment")'),
                     'B': ('official crop_policy="box": parse_roi_box_from_bbox with the whole canonical frame '
                           '[0,0,256,256] as the box (literal reading of the faceboxes note; 1.58x enlargement, '
                           'zero-padded crop)'),
                     'C': 'another rule chosen by the owner'},
         'recommendation': 'A'},
        {'id': 'D2', 'topic': 'geometry pre-pass population and outputs', 'scientific': True,
         'blocks': ['depth-M0 pre-pass', 'trace transfer'],
         'why_open': ('A3 4.4 scopes the depth reconstruction to live faces, but A3 4.3 trace transfer needs Q140 '
                      'correspondences of the source spoof faces as well; the pre-pass population is not stated'),
         'options': {'A': ('TRAIN pairs_train_v1 faces only: 3830 live (fit, dense mesh, M0 32x32 + mask, Q140 xy) and '
                           '8838 spoof (fit, Q140 xy; M0 = exact zeros by paper Eq.21, no render); generated once, '
                           'hashed, reused for seeds 42/1337/2026'),
                     'B': 'A plus VAL val_pairs_v1 faces (748 live, 1905 spoof), only if D6 adopts VAL diagnostics'},
         'recommendation': 'A (B only with D6 = B)'},
        {'id': 'D3', 'topic': 'pre-pass device', 'scientific': False,
         'blocks': ['depth-M0 pre-pass determinism record'],
         'why_open': ('M6D3a: within-device repeats are byte-identical and the final uint8 depth is identical across '
                      'devices, but CPU and CUDA fits/meshes (hence Q140 coordinates) are not bitwise identical'),
         'options': {'A': 'CPU (no GPU contention, repeatable)', 'B': 'CUDA (fresh-process repeats byte-identical)'},
         'recommendation': 'A'},
        {'id': 'D4', 'topic': 'TRAIN pair sampling over pairs_train_v1', 'scientific': True,
         'blocks': ['real TRAIN feed', 'stages F, J'],
         'why_open': ('no official code; the qualified graph fixes 4 (target_live, source_spoof) rows per batch 8, '
                      'but no order/shuffle/pass rule exists (precedent: E05 owner-approved seeded shuffle; E03 R1)'),
         'options': {'A': ('seeded per-pass permutation of the 8838 rows (numpy RandomState(experiment_seed)), '
                           'reshuffled every pass, consumed contiguously across pass boundaries, nothing dropped: '
                           '600000 rows = 67 full passes + 7854'),
                     'B': 'another rule chosen by the owner'},
         'recommendation': 'A'},
        {'id': 'D5', 'topic': 'TRAIN augmentation', 'scientific': True, 'blocks': ['real TRAIN feed'],
         'why_open': ('the frozen E04 contract lists none, but M6D3d stdn_reference_policy (STDN where PhySTD is '
                      'silent) could be read as inheriting STDN TRAIN flips and brightness; flips would also have to '
                      'flip M0 and the Q140 correspondences'),
         'options': {'A': 'none', 'B': 'inherit STDN flips + brightness(0.25) with consistent geometry flips'},
         'recommendation': 'A'},
        {'id': 'D6', 'topic': 'VAL diagnostics', 'scientific': False, 'blocks': ['lifecycle definition'],
         'why_open': 'the config allows VAL for diagnostics only; no source defines a VAL block or cadence',
         'options': {'A': 'no VAL block (TRAIN-only run; VAL and TEST untouched)',
                     'B': 'periodic val_pairs_v1 diagnostic block (non-selecting; needs its own firewall/RNG proof)'},
         'recommendation': 'A'},
        {'id': 'D7', 'topic': 'hard-sample RNG stream', 'scientific': True, 'blocks': ['production controller'],
         'why_open': ('M6D3e realizes per-spoof uniform[0,1] x3 and Bernoulli(.5) draws as supplied tensors with a '
                      'qualification seed; the experiment-seed stream is not defined'),
         'options': {'A': ('a dedicated numpy RandomState seeded 1 + u32be(SHA256("GPAT_TRANSFERBENCH|E04|HARD|<seed>")'
                           '[0:4]) mod (2^31-1), independent of the sampling stream; per iteration 4x3 uniforms then '
                           '4 keep draws, as in M6D3e'),
                     'B': 'another rule chosen by the owner'},
         'recommendation': 'A'},
        {'id': 'D8', 'topic': 'graph input encoding', 'scientific': True, 'blocks': ['real TRAIN feed'],
         'why_open': ('no E04 artifact fixes the pixel range; the sum-reduced losses and frozen alphas depend on it. '
                      'The pinned STDN predecessor feeds RGB float32 / 255 (model/dataset.py:144), which M6D3d '
                      'stdn_reference_policy would select; the M6D3e fixtures are in [0,1]'),
         'options': {'A': 'RGB float32 / 255 in [0,1] (STDN reference)', 'B': 'another range'},
         'recommendation': 'A (confirm)'},
        {'id': 'D9', 'topic': 'driver-version tolerance for E04', 'scientific': False,
         'blocks': ['stage B acceptance', 'launch'],
         'why_open': ('OWNER_APPROVED_DRIVER_VERSION_ONLY_RUNTIME_COMPATIBILITY was approved for E05 and reused by E03; '
                      'no record extends it to E04. Both E04 locks record 595.84; the host runs 580.178.04. Synthetic '
                      're-qualification passes on 580.178.04, but D3 BN moving means now also differ after Step 1 '
                      '(4 tensors; committed M6D3e processes were identical after Step 1)'),
         'options': {'A': 'extend the version-only tolerance to gpat-m6-e03 (E04 training) and gpat-m6-e04-geometry',
                     'B': 'require driver 595.84'},
         'recommendation': 'A, with the Step-1 D3 variance disclosed'},
    ]


ENGINEERING = {
    'checkpoint_cadence': ('frozen as implementation_defined_periodic_plus_terminal: the production contract may fix '
                           'a periodic cadence (for example every 15000 iterations) plus iteration_150000, zero extra '
                           'optimizer steps (A2-06)'),
    'metrics_reporting_interval': 'configured_reporting_interval (run_logging_v1); fixed by the production contract',
    'dense_offsets': ('per-pair Delaunay inverse offsets with the pinned SciPy LinearNDInterpolator (scipy 1.10.1 in '
                      'gpat-m6-e03), computed in the feed from the frozen pre-pass Q140 xy; exact M6D3e function'),
    'workers': 'IMPLEMENTATION_ADAPTABLE_NO_SCIENTIFIC_IMPACT',
}

PRODUCTION_FILES = {
    'existing': ['methods/physics_std/adapter.py', 'methods/physics_std/assets.py', 'methods/physics_std/contract.py',
                 'methods/physics_std/geometry.py', 'methods/physics_std/regressor.py',
                 'methods/physics_std/renderer.py', 'methods/physics_std/runtime.py', 'methods/physics_std/source.py',
                 'methods/physics_std/trace_transfer.py', 'methods/physics_std/network.py',
                 'methods/physics_std/losses.py', 'methods/physics_std/training_contract.py',
                 'methods/physics_std/training_graph.py', 'methods/physics_std/training_runtime.py'],
    'missing': {
        'tools/e04_geometry_prepass.py': 'D1, D2, D3',
        'methods/physics_std/feed_pack.py (hash-verified TRAIN pack: pair rows + pre-pass M0/mask/Q140, no TEST)':
            'D1, D2, D4',
        'methods/physics_std/production_io.py (contract/source/env/pack verifiers, resolved config)': 'D4-D9',
        'methods/physics_std/production_runner.py (controller: feed, 3-step loop, checkpoints, lifecycle)':
            'D4, D5, D6, D7, D8',
        'tools/run_e04.py (fail-closed CLI, fresh only)': 'D4-D9',
        'tools/e04_production_qualification_evidence.py + tests': 'all of the above',
        'tools/e04_seed_completion_audit.py': 'after qualification'},
    'reusable_from_e03_without_modifying_e03': ['methods.common RunContext / run_logging_v1 manifest and metrics '
                                                'writers', 'Python 3.8 compatibility shims (pathlib backport, '
                                                'importlib.metadata path)', 'audit-hook firewall pattern',
                                                'checkpoint index / SHA listing conventions',
                                                'stat-only atime TEST firewall snapshots'],
    'not_reusable': ['E03 graph, input semantics (YUV concat), G_D_RATIO schedule, ckpt-50 cadence, VAL block, '
                     'RNG ruling V, feed pack v1 (no E04 geometry)'],
}


def stages(static_ok, requal):
    a, c = requal['M6D3A_CPU'], requal['M6D3A_CUDA']
    e = requal['M6D3E_PROCESS']
    geom = all(x['regressor_repeat_byte_identical'] and x['end_to_end_repeat_byte_identical'] and
               x['depth_sha256_in_committed_m6d3a'] and x['spoof_exact_zeros'] and not x['file_access_denied']
               for x in (a, c)) and a['depth_sha256'] == c['depth_sha256']
    graph = (e['status'] == 'PASS' and e['optimizer_applications'] == 3 and e['graph_def_equals_committed'] and
             e['tests']['failures'] == e['tests']['errors'] == 0 and not e['firewall_denied'])
    return {
        'A_static_source_and_contract': 'PASS' if static_ok else 'FAIL',
        'B_environment_and_dependencies': ('PASS_SYNTHETIC_PENDING_D9_DRIVER_RULING' if geom and graph else 'FAIL'),
        'C_scientific_graph_construction': 'PASS_SYNTHETIC' if graph else 'FAIL',
        'D_synthetic_forward_and_gradient': ('PASS_SYNTHETIC_WITH_DISCLOSED_D3_VARIANCE' if graph else 'FAIL'),
        'E_deterministic_data_path': 'BLOCKED_D4_D5_D7_D8',
        'F_real_train_val_feed': 'BLOCKED_D4_D5_D6_D8',
        'G_depth_m0_integrity': 'BLOCKED_D1_D2_D3',
        'H_checkpoint_and_lifecycle': 'NOT_IMPLEMENTED',
        'I_test_firewall': 'NOT_IMPLEMENTED (pattern reusable from E03)',
        'J_real_data_production_qualification': 'BLOCKED',
        'K_independent_evidence_verification': 'BLOCKED',
    }


def build(root, host, requal_dir):
    root = Path(root)
    sys.path.insert(0, str(root))
    from methods.physics_std.adapter import PhysicsSTDAdapter
    from methods.physics_std.training_contract import load_contract
    static = {}
    cfg_raw = (root / 'configs/methods/e04_physics_std.yaml').read_bytes()
    static['config_equals_snapshot'] = cfg_raw == (root / 'frozen_config_snapshot/configs/methods/'
                                                          'e04_physics_std.yaml').read_bytes()
    static['config_sha256'] = sha256_bytes(cfg_raw)
    contract = load_contract(root)
    static['m6d3d_contract'] = contract['status']
    prepared = PhysicsSTDAdapter().prepare()
    static['source_verification'] = prepared['source']['verification']
    static['assets_laptop'] = {k: v['sha256'] for k, v in prepared['assets'].items()}
    static['q140_sha256'] = prepared['q140']['vertex_indices_sha256']
    plan = PhysicsSTDAdapter().build_training_plan(42)
    static['checkpoint_policy'] = {k: plan['checkpoint'][k] for k in ('rule', 'terminal_state', 'selection_uses_val',
                                                                       'selection_uses_test', 'cadence')}
    cfg = PhysicsSTDAdapter().config
    expected_assets = {a['path']: a['sha256'] for a in cfg['external_assets']['geometry_engine']['assets']}
    static_ok = (static['config_equals_snapshot'] and static['m6d3d_contract'] == 'E04_TRAINING_GRAPH_CONTRACT_RESOLVED'
                 and static['source_verification'] == 'PASS' and static['assets_laptop'] == expected_assets and
                 static['q140_sha256'] == cfg['controlled_reconstruction']['q140_vertex_set']['vertex_indices_sha256'])
    requal = {k: json.loads((Path(requal_dir) / v).read_text()) for k, v in REQUAL.items()}
    cmp_ = requal_comparison(root, requal)
    host_ok = {
        'assets_match_frozen': {k: v['sha256'] for k, v in host['assets'].items()} == expected_assets,
        'sim3dr_extension_matches_manifest': host['sim3dr_extension']['sha256'] ==
        host['sim3dr_extension']['expected_sha256'],
        'requal_files_match_copies': all(host['requal_files_sha256'].get(v) == sha256_file(Path(requal_dir) / v)
                                         for v in REQUAL.values()),
        'gpu_identity': host['gpu'].endswith('GPU-b722cd9d-c9fa-223c-915e-cc09d9b862b7'),
        'no_method_process': host['method_processes'] == [] and host['gpu_compute_apps'] == '',
        'e04_run_root_absent': host['e04_run_root_exists'] is False,
        'gpu_checkout_clean': host['git']['status_porcelain'] == '',
        'manifests_match_laptop': all(host['files_sha256'][f] == sha256_file(root / f) for f in MANIFESTS),
        'environments_present': all(v['exists'] and v['modules'] and
                                    not any(str(x).startswith('MISSING') for x in v['modules'].values())
                                    for v in host['environments'].values())}
    st = stages(static_ok and all(host_ok.values()), cmp_)
    doc = {
        'milestone': 'E04-PRODUCTION-READINESS-AUDIT', 'status': STATUS,
        'also': ['E04_READINESS_AUDIT_COMPLETE', 'IMPLEMENTED_NOT_EXECUTED', 'CONTROLLED_ADAPTATION'],
        'scientific_execution_authorized': False, 'e04_scientific_authority': None,
        'contract': {
            'method': 'E04 Physics-STD (controlled geometry/depth reconstruction)',
            'fidelity_class': cfg['fidelity_class'], 'forbidden_wording': cfg['forbidden_wording'],
            'source': cfg['source'], 'spec_section': '8.4',
            'authority_chain': ['configs/methods/e04_physics_std.yaml (M6B, frozen, sha %s)' % static['config_sha256'],
                                'Amendment A3 section 4 (controlled geometry/depth reconstruction)',
                                'Amendment A4 (uint8 INTER_AREA depth boundary; Adam semantics)',
                                'A2-06 BASELINE_FINAL_STATE_V1',
                                'M6D3d owner resolution (training graph contract)',
                                'M6D3e implementation record (graph realization)'],
            'architecture': ('encoder 3x3 c32 + ConvBlocks 64/96/128; decoder to 13 channels (B/C/T/P/IP); attention '
                             'depth head 32x32 in [0,1]; four discriminators at 32/96/256/256; 1,763,902 trainable '
                             'parameters (M6D3e)'),
            'training': {'iterations': 150000, 'batch': '8 = 4 target_live + 4 source_spoof (qualified graph layout)',
                         'optimizer': 'Adam beta1 0.9 beta2 0.999 eps 1e-8, no weight decay (A4)',
                         'lr': '5e-5 x 10^-floor(t/45000); D at half the G rate (M6D3d)',
                         'steps': 'per minibatch: G Eq.23 -> D L_D -> G Eq.24; iteration +1 after step 3',
                         'losses': 'Eq.23 100 L_depth + 5 L_G + L_P + 1e-4 L_R; Eq.24 10 L_S + L_H; beta 0.1',
                         'init': 'Normal(0, 0.02)'},
            'depth_m0': {'live': ('3DDFA_V2 1b6c6760 dense mesh -> similar_transform -> Sim3DR z-buffer -> per-sample '
                                  'min-max [0,1] -> uint8 -> INTER_AREA 32x32 -> float32/255 -> clip (A3/A4)'),
                         'spoof': 'exact float32 zeros 32x32 (paper Eq.21)', 'q140_sha256': static['q140_sha256'],
                         'fixed_assets': 'generated once, reused unchanged for seeds 42/1337/2026 (A3 4.2b)'},
            'checkpoint': static['checkpoint_policy'],
            'data': {'TRAIN': 'pairs_train_v1 (8838 rows) for training (config data.pair_manifest)',
                     'VAL': 'allowed for diagnostics only, never selects (D6 open)',
                     'TEST': 'forbidden for every purpose; no code path',
                     'train_rows_consumed': TRAIN_ROWS_CONSUMED,
                     'passes': [TRAIN_ROWS_CONSUMED // PAIRS['TRAIN'], TRAIN_ROWS_CONSUMED % PAIRS['TRAIN']]},
            'seeds': cfg['seeds']['experiment_seeds'], 'qualification_seed': 314159,
            'environments': {'training': 'gpat-m6-e03 (owned by E03; read-only reuse; E04 reference lock)',
                             'geometry_prepass': 'gpat-m6-e04-geometry (Python 3.12, torch 2.14 cu130)'}},
        'static_checks': static, 'host_checks': host_ok, 'requalification': cmp_, 'stages': st,
        'decisions_required': decisions(), 'engineering_within_authority': ENGINEERING,
        'production_files': PRODUCTION_FILES,
        'next_milestone': ('owner production resolution for E04 (D1-D9); then the geometry pre-pass, the feed pack, '
                           'the production controller and CLI, and real-TRAIN production qualification'),
        'confirmations': {'benchmark_images_decoded': 0, 'test_accessed': False, 'e04_run_root_created': False,
                          'scientific_seed_used': False, 'environment_modified': False, 'e03_modified': False,
                          'training_launched': False},
        'host_collect': host}
    return doc


def report(doc):
    c, s, rq = doc['contract'], doc['stages'], doc['requalification']
    e = rq['M6D3E_PROCESS']
    lines = [
        '# E04 Physics-STD: production-readiness audit', '',
        '**Status:** `%s` (readiness audit complete; `IMPLEMENTED_NOT_EXECUTED`, `CONTROLLED_ADAPTATION`). E04 '
        'scientific execution is **not** authorized and no E04 authority exists. The machine-readable record is '
        '`E04_PRODUCTION_READINESS_AUDIT.json`.' % doc['status'], '',
        'The training graph and the fixed-geometry runtime re-qualify on synthetic data under the current host driver. '
        'The real-data path cannot be built yet: nine production decisions (D1–D9) are not fixed by any frozen '
        'artifact, and spec §8.4 requires `BLOCKED_BY_SOURCE_GAP` rather than a choice.', '',
        '## Qualification stages', '', '| stage | status |', '|---|---|']
    lines += ['| %s | `%s` |' % (k.replace('_', ' ', 1), v) for k, v in s.items()]
    lines += ['', '## Synthetic re-qualification (driver %s)' % doc['host_collect']['gpu'].split(', ')[1], '',
              '- Geometry (M6D3a worker), CPU and CUDA: repeat byte-identical; final depth `%s…` on both devices, '
              'equal to the committed M6D3a value; spoof exact zeros; no file-access denial.'
              % rq['M6D3A_CPU']['depth_sha256'][:16],
              '- Training graph (M6D3e worker, qualification seed 314159, 3 applies): `%s`, %d/%d tests; GraphDef '
              '`%s…` equal to the committed hash; %d common fields, differing: %s.'
              % (e['status'], e['tests']['tests'] - e['tests']['failures'] - e['tests']['errors'], e['tests']['tests'],
                 e['graph_def_sha256'][:16], e['fields_common'], ', '.join('`%s`' % x for x in e['fields_differing'])),
              '- Post-step state differs from committed process 1 only in D3: %s tensors after steps 1/2/3 (committed '
              'process pair: %s). The 4 Step-1 differences (`%s`) are new relative to M6D3e and are disclosed '
              'under D9.' % ('/'.join(map(str, e['post_state_differences_per_step'])),
                             '/'.join(map(str, e['committed_process_pair_differences_per_step'])),
                             '`, `'.join(e['step1_new_differences'])),
              '', '## Decisions required from the owner', '', '| id | decision | recommendation | blocks |',
              '|---|---|---|---|']
    lines += ['| %s | %s%s | %s | %s |' % (d['id'], d['topic'], '' if d['scientific'] else ' (engineering)',
                                           d['recommendation'], ', '.join(d['blocks']))
              for d in doc['decisions_required']]
    lines += ['', 'Options and reasons for each decision are in the JSON (`decisions_required`).', '',
              '## Production implementation', '', '| missing file | waits on |', '|---|---|']
    lines += ['| `%s` | %s |' % (k, v) for k, v in doc['production_files']['missing'].items()]
    lines += ['', '## Boundary', '',
              'No benchmark image was decoded and no TEST, VAL or TRAIN sample was accessed. No E04 run root was '
              'created, no experiment seed was used, no environment was modified, and no E03 artifact changed.', '',
              '**Next milestone:** %s.' % doc['next_milestone'], '']
    return '\n'.join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd')
    a = sub.add_parser('collect')
    a.add_argument('--requal-dir', required=True)
    b = sub.add_parser('build')
    b.add_argument('--host', required=True)
    b.add_argument('--requal-dir', required=True)
    b.add_argument('--repo-root', default=str(Path(globals().get('__file__', 'tools/x')).resolve().parents[1]))
    args = ap.parse_args(argv)
    if args.cmd == 'collect':
        json.dump(collect(args.requal_dir), sys.stdout, sort_keys=True)
        return 0
    root = Path(args.repo_root)
    doc = build(root, json.loads(Path(args.host).read_text()), args.requal_dir)
    for key, name in REQUAL.items():
        shutil.copyfile(Path(args.requal_dir) / name, root / (EVIDENCE['requal'] % key))
    (root / EVIDENCE['record']).write_text(json.dumps(doc, indent=2, sort_keys=True) + '\n')
    (root / EVIDENCE['report']).write_text(report(doc))
    files = [EVIDENCE['record'], EVIDENCE['report']] + [EVIDENCE['requal'] % k for k in REQUAL]
    (root / EVIDENCE['sha256']).write_text(''.join('%s  %s\n' % (sha256_file(root / f), f) for f in files))
    print(doc['status'], doc['stages'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
