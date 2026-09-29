#!/usr/bin/env python3
"""Verify the M6FA E06b DSDG-NATIVE contract-resolution candidate; ledger/index LAST.

Every contract value is re-derived from authority bytes: the pinned FaceX-Zoo/DSDG source (by SHA256), the frozen
specification, A1 / A9, fair_track_v1, attack_map_v1, the frozen M3/M4 data audits, the E06c config and the
M6D5c/M6D5e execution records. No value is invented and no frozen config is created.

STATIC: stdlib only (no Torch, YAML parser, numpy, PIL or pyarrow); no GPU, no image, no manifest parquet, no model.
check_contract / derive take a reader (no moving-HEAD lock). The pinned source cache is git-ignored and is read from
disk, anchored by SHA256.

  --before-ledger / --append-ledger --tests JSON / --rebuild-index / (default final)
"""
import argparse
import csv
import datetime as dt
import getpass
import hashlib
import io
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import zipfile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = '8e9ccc371f6c53b75ec6b473d2306aefa9aa672b'
BRANCH = 'm6-baselines'
MILESTONE = 'M6FA'
CLASSIFICATION = 'M6FA_E06B_CONTRACT_RESOLUTION'
SPEC = 'docs/spec/GPAT_TransferBench_v1_0_Frozen_Specification_2026.docx'
SPEC_SHA = 'f7d2371682ad9504f59f0c6984e4b1a741f855f1b1f60838f9d60b160281489e'
INDEX = 'outputs/audit/ARTIFACT_INDEX.csv'
LEDGER = 'outputs/audit/EXECUTION_LEDGER.jsonl'
LEDGER_PREFIX_ROWS = 129
LEDGER_PREFIX_SHA = 'ec576cf407455c752efd44a8c4b42238b1b734ac8f1b047e9a579cbda37c6e28'
INDEX_BASELINE_ROWS = 769
CONFIG_STATUS = 'configs/CONFIG_STATUS.md'
CONFIG_STATUS_AUTHORITY_SHA = '1289925d25adc4127fd67437bacb3be51ae4babbc75984af6929a5a0dce73785'
CONTRACT = 'outputs/audit/M6FA_E06B_CONTRACT_RESOLUTION.json'
CONTRACT_SHA = 'fe287ebe1ec9c19e5f63a40f1498e41f5b7044014c2cc7eb6279a32604555058'
REPORT = 'outputs/audit/M6FA_E06B_CONTRACT_RESOLUTION.md'
TESTS = 'tests/test_m6fa_e06b_contract.py'
PREFLIGHT = 'tools/m6fa_e06b_contract_preflight.py'
NEW = tuple(sorted((CONTRACT, REPORT, TESTS, PREFLIGHT)))
FROZEN_TARGET = 'configs/methods/e06b_dsdg_native.yaml'
A1 = 'docs/spec/amendments/GPAT_TransferBench_v1_0_Amendment_A1_Fair_IDFree_Main_Track.md'
A9_RECORD = 'configs/amendments/difffas_a9_resource_constrained_scope_exclusion.yaml'
FAIR_TRACK = 'configs/frozen/fair_track_v1.yaml'
PAIRS = 'configs/frozen/pairs_v1.yaml'
ATTACK_MAP = 'configs/frozen/attack_map_v1.yaml'
TAXONOMY = 'manifests/artifact_probe_classes_v1.json'
M3_MACRO = 'outputs/audit/M3_DISTRIBUTION_ATTACK_MACRO.csv'
M4_COVERAGE = 'outputs/audit/M4_NATIVE_PAIR_COVERAGE.csv'
E06C_CONFIG = 'configs/methods/e06c_dsdg_bin_idfree.yaml'
M6D5C = 'configs/amendments/e06c_m6d5c_memory_execution_resolution.yaml'
M6D5C_DOC = 'docs/spec/amendments/GPAT_TransferBench_v1_0_E06c_Memory_Execution_Resolution_Addendum_M6D5c.md'
M6D5E = 'configs/amendments/e06c_m6d5e_production_runner_contract.yaml'
LIGHTCNN = 'outputs/audit/M6A4_LIGHTCNN_WEIGHT_PROVENANCE.json'
SRC = 'third_party/source_cache/facexzoo/addition_module/DSDG/'
SOURCE_SHA = {
    SRC + 'train_generator.sh': 'bdb08394532fce03314a42f69d17f04fa1f4da8bb1b6743528ce4a21bda827bc',
    SRC + 'train_generator.py': '5b7bf426f812a67aca67fd635acbb249b2d58680f988bb1288b4c7c21ebfb370',
    SRC + 'data/generation_dataset.py': '0ac299c177a2cf9966212aa31709a28a62dab2ecf0ddd60aa1c36e97c4171dc1',
    SRC + 'networks/__init__.py': '989737735aa0e0e0e36e1c8c7b0f56008f017f42626e3ca5b2519c6e845680c8',
    SRC + 'networks/generator.py': 'c8516e2a3a33c609c605378a50b31ed9a77c483f2acbbc85d4846c1b607cd303',
    SRC + 'misc/util.py': '64aa2d0983988a37d160920275a317b886442a601908e5b3669e487cefe941c7'}
GENERATED_PY = SRC + 'generated.py'
TRAIN_SH_BLOB = '7fcb3f8648137ea16f53e290dba2668ec540a35d'
PINNED_COMMIT = '16b793a7564a4b9308cf94e62bdb2ffacb3a725a'
FRAMES_PER_VIDEO = 8
MICROBATCH = 20


def require(ok, message):
    if not ok:
        raise ValueError('M6FA: ' + message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def git(*args, cwd=ROOT):
    return subprocess.check_output(['git', '-C', str(cwd), *args])


def read(path):
    p = Path(path)
    require(not p.is_absolute() and '..' not in p.parts, 'relative path')
    require(path == TAXONOMY or not {'data', 'faces_256', 'runs', 'cache', 'manifests'} & set(p.parts),
            'data firewall ' + path)
    require(p.suffix not in {'.pkl', '.pt', '.pth', '.ckpt', '.parquet', '.png', '.jpg'}, 'no weight/manifest/image')
    return (ROOT / p).read_bytes()


def worktree_reader(rel):
    return read(rel)


def source(rel, expected=None):
    """Pinned DSDG source (git-ignored cache) read from disk and anchored by SHA256."""
    raw = (ROOT / rel).read_bytes()
    require(sha(raw) == (expected or SOURCE_SHA[rel]), 'pinned source bytes ' + rel)
    return raw.decode()


def at_authority(rel):
    return git('show', f'{AUTHORITY}:{rel}')


def spec_paragraphs(raw):
    xml = zipfile.ZipFile(io.BytesIO(raw)).read('word/document.xml').decode('utf-8')
    return [re.sub(r'<[^>]+>', '', p).replace('&gt;', '>').replace('&lt;', '<').replace('&amp;', '&')
            for p in re.findall(r'<w:p[ >].*?</w:p>', xml, flags=re.S)]


def shell_values(text):
    return {i: tuple(line.split('=', 1)) for i, line in enumerate(text.splitlines(), 1)
            if re.fullmatch(r'[a-z_]+=[^ ]+', line)}


# ----------------------------------------------------------------- derivation (reader-scoped)
def derive(reader=worktree_reader):
    d = {}
    # spec
    raw = reader(SPEC)
    require(sha(raw) == SPEC_SHA, 'frozen specification SHA256')
    paras = spec_paragraphs(raw)
    for text in ('Status: FAITHFUL_OFFICIAL for DSDG-NATIVE using FaceX-Zoo addition_module/DSDG.',
                 'Native method: spoof-pattern latent is explicitly supervised by spoof-type cross-entropy. Use '
                 'attack_macro as the spoof-type target because it is the frozen harmonized label available across '
                 'all three datasets.',
                 'Training live/spoof identity pairing uses only TRAIN identities that possess both live and spoof '
                 'samples. Log usable identity coverage by dataset.',
                 'Generate exactly N_syn samples for each seed. DSDG does not consume the common pair manifest; '
                 'target-to-synthetic ID metrics are N/A unless an explicit identity-conditioned pair can be traced by '
                 'the official generator.'):
        require(text in paras, 'spec 8.6 clause: ' + text[:50])
    i = paras.index('E06b')
    require(paras[i:i + 5] == ['E06b', 'DSDG-NATIVE', 'Native', 'Both', 'Original type-supervised setting'],
            'spec 17 E06b row')
    i = paras.index('DSDG-NATIVE', paras.index('5.2 Track B — NATIVE / FULL'))
    require(paras[i + 1].startswith('Spoof-type label for spoof-pattern classifier'), 'spec 5.2 DSDG-NATIVE row')
    # amendments / frozen configs
    a1 = reader(A1).decode()
    require('move to Track B with coverage CASIA + MSU and SiW\n`NOT_INSTANTIABLE_MISSING_SUBJECT_ID`' in a1,
            'A1 Track-B coverage CASIA + MSU, SiW not instantiable')
    fair = reader(FAIR_TRACK).decode()
    for token in ('  methods: [E06b, E07b, E09, E10, E11]\n', '  native_metadata_allowed: true\n',
                  '    casia_fasd: SUPPORTED\n    msu_mfsd: SUPPORTED\n    siwmv2: NOT_INSTANTIABLE_MISSING_SUBJECT_ID\n'):
        require(token in fair, 'fair_track_v1 track_b ' + token.strip())
    pairs = reader(PAIRS).decode()
    require('  Q-28: RESOLVED_FOR_M4_NATIVE_MANIFEST_SCOPE\n' in pairs and
            'complete M6 DSDG and DiffFAS training-adaptation strategy for SiW-Mv2 is a separate later\n  decision'
            in pairs, 'pairs_v1 Q-28 scope; SiW training strategy unsettled')
    a9 = json.loads(reader(A9_RECORD))
    require(a9['E06b']['status'] == 'ACTIVE_M6_WORK' and a9['E06b']['owner_excluded'] is False and
            a9['m6']['M6_CLOSED'] is False, 'A9: E06b ACTIVE M6 WORK, M6 open')
    # data population (frozen M3 video table x 8 frames; frozen M4 identity coverage)
    macro = {(r['dataset'], r['category']): int(r['TRAIN_videos']) for r in csv.DictReader(io.StringIO(reader(M3_MACRO).decode()))}
    per = {}
    for ds in ('casia_fasd', 'msu_mfsd'):
        cats = {c: v for (dd, c), v in macro.items() if dd == ds}
        require(set(cats) == {'live', 'print', 'replay'}, ds + ' TRAIN categories = live/print/replay')
        per[ds] = {'train_live_frames': cats['live'] * FRAMES_PER_VIDEO,
                   'spoof_by_macro': {m: cats[m] * FRAMES_PER_VIDEO for m in ('print', 'replay')}}
        per[ds]['train_spoof_frames'] = sum(per[ds]['spoof_by_macro'].values())
    cov = {(r['method'], r['dataset']): r for r in csv.DictReader(io.StringIO(reader(M4_COVERAGE).decode()))}
    for ds in ('casia_fasd', 'msu_mfsd'):
        r = cov[('DSDG', ds)]
        require(int(r['train_spoof_rows']) == per[ds]['train_spoof_frames'] and
                int(r['train_live_rows']) == per[ds]['train_live_frames'], 'M3 x 8 == M4 coverage rows ' + ds)
        require(r['eligible_identities_live_and_spoof'] == r['total_train_identities'] and r['identity_coverage'] == '1.0',
                'full identity coverage ' + ds)
        per[ds]['subjects'] = int(r['total_train_identities'])
        per[ds]['subjects_with_live_and_spoof'] = int(r['eligible_identities_live_and_spoof'])
        require(per[ds]['train_live_frames'] % per[ds]['subjects'] == 0, 'live frames divide evenly ' + ds)
        per[ds]['live_frames_per_subject'] = per[ds]['train_live_frames'] // per[ds]['subjects']
    siw = cov[('DSDG', 'siwmv2')]
    require((siw['status'], siw['total_train_identities'], siw['native_rows_materialized']) ==
            ('NOT_INSTANTIABLE_MISSING_SUBJECT_ID', '0', '0'), 'SiW not instantiable, 0 identities')
    d['per_dataset'] = per
    # attack map: every CASIA/MSU spoof token maps to print or replay
    amap = reader(ATTACK_MAP).decode()
    block = amap[amap.index('  casia_fasd:'):amap.index('  siwmv2:')]
    macros = set(re.findall(r'attack_macro: (\w+)', block))
    require(macros == {'print', 'replay'}, 'CASIA+MSU attack_macro vocabulary == {print, replay}: ' + str(macros))
    tax = json.loads(reader(TAXONOMY))
    require(tax['class_index']['print'] < tax['class_index']['replay'], 'frozen taxonomy order print < replay')
    d['vocabulary'] = sorted(macros, key=lambda m: tax['class_index'][m])
    # pinned official source
    sh = source(SRC + 'train_generator.sh')
    sv = {k: (line, v) for line, (k, v) in shell_values(sh).items()}
    d['shell'] = {k: (line, v) for k, (line, v) in sv.items()}
    py = source(SRC + 'train_generator.py')
    lines = py.splitlines()
    require(lines[21] == "parser.add_argument('--lr', default=0.0002, type=float)", 'train_generator.py:22 lr default')
    require(lines[40] == "parser.add_argument('--lambda_pair', default=0.5, type=float)", 'train_generator.py:41 default')
    require('--lr' not in sh and '--lambda_pair $lambda_pair' in sh, 'launch passes --lambda_pair and no --lr')
    require(lines[173].strip() == 'if epoch < 2:' and '0.01 * loss_pair' in lines[174], 'warm-up 174-176')
    require(lines[213].strip() == 'if epoch % args.save_epoch == 0 or epoch == 1:', 'save cadence line 214')
    require('optim.Adam(list(netE_nir.parameters()) + list(netE_vis.parameters()) + list(netG.parameters()),'
            in lines[86], 'optimizer scope 87-88 (netCls excluded)')
    ds_src = source(SRC + 'data/generation_dataset.py').splitlines()
    require(ds_src[92].strip() == 'img_name = random.choice(self.make_pair_dict[label][domain_flag])' and
            ds_src[86].strip() == "if domain_flag == '0':" and ds_src[47].strip() == "img_name_live = self.get_pair(label, '1')",
            'online same-subject random.choice relation (generation_dataset.py:48, 87, 93)')
    gen = (ROOT / GENERATED_PY).read_text()
    require("default='./result/model_oulu_p1/netG_model_epoch_200_iter_0.pth'" in gen, 'generated.py epoch-200 netG')
    # E06c config and execution records
    e06c = reader(E06C_CONFIG).decode()
    for token in ('  effective_batch_size: 240\n', '  all_epochs: 200\n', '  hdim: 128\n', '  learning_rate: 2.0e-4\n',
                  '  lambda_mmd: 50\n', '  lambda_ip: 1000\n', '  lambda_type: 10\n', '  lambda_ort: 1\n',
                  '  lambda_pair_official_default: 5\n', '  rule: OFFICIAL_GENERATOR_EPOCH_200\n',
                  '  experiment_seeds: [42, 1337, 2026]\n',
                  '    sha256: d0750746622270548137b092700811b00dc64d67ca8df042c6cf48cb3b1b9964\n'):
        require(token in e06c, 'E06c config ' + token.strip())
    m5c = json.loads(reader(M6D5C))
    require(m5c['classification'] == 'CONTROLLED_EXECUTION_ADAPTATION' and m5c['method_id'] == 'E06c', 'M6D5c record')
    doc = reader(M6D5C_DOC).decode()
    require('| microbatch | 20 (fixed; not tuned on loss or quality; OOM at 20 → STOP_AND_REPORT) |' in doc and
            'per model: 0 BatchNorm modules and 36 InstanceNorm modules' in doc, 'M6D5c microbatch 20; no BatchNorm')
    dl = json.loads(reader(M6D5E))['dataloader']
    require((dl['batch_size'], dl['shuffle'], dl['num_workers'], dl['drop_last'], dl['persistent_workers']) ==
            (240, True, 8, False, False) and 'seed_torch_worker' in dl['worker_init_fn'], 'E06c production loader')
    d['loader'] = {k: dl[k] for k in ('batch_size', 'shuffle', 'num_workers', 'drop_last', 'persistent_workers')}
    require('d0750746622270548137b092700811b00dc64d67ca8df042c6cf48cb3b1b9964' in reader(LIGHTCNN).decode(),
            'LightCNN weight SHA256 recorded in the M6A4 provenance')
    return d


def check_contract(reader=worktree_reader):
    raw = reader(CONTRACT)
    require(sha(raw) == CONTRACT_SHA, 'contract SHA256 == pinned constant')
    c = json.loads(raw)
    d = derive(reader)
    require((c['milestone'], c['method_id'], c['method_name'], c['track'], c['authority_commit'],
             c['frozen_config_created'], c['frozen_config_target_path']) ==
            (MILESTONE, 'E06b', 'DSDG-NATIVE', 'B_NATIVE_FULL_SECONDARY', AUTHORITY, False, FROZEN_TARGET), 'identity')
    f = c['fidelity']
    require((f['target_fidelity'], f['fidelity_assessed'], f['new_deviation'], f['a1_adaptation_applies']) ==
            ('FAITHFUL_OFFICIAL', False, False, False), 'target fidelity FAITHFUL_OFFICIAL, not yet assessed')
    s = c['source']
    require(s['pinned_commit'] == PINNED_COMMIT and s['official_source_modified'] is False and
            s['cited_files_sha256'] == SOURCE_SHA and s['train_generator_sh_pin']['git_blob'] == TRAIN_SH_BLOB, 'source pin')
    cov = c['dataset_coverage']
    require(cov['train_datasets'] == ['casia_fasd', 'msu_mfsd'] and
            cov['non_instantiable']['siwmv2']['status'] == 'NOT_INSTANTIABLE_MISSING_SUBJECT_ID' and
            cov['non_instantiable']['siwmv2']['rows_used'] == 0 and cov['test_usage'] == 'FORBIDDEN', 'dataset scope')
    for ds, v in d['per_dataset'].items():
        require(cov['per_dataset'][ds] == v, 'per-dataset counts ' + ds)
    tot = {'train_spoof_frames': sum(v['train_spoof_frames'] for v in d['per_dataset'].values()),
           'train_live_frames': sum(v['train_live_frames'] for v in d['per_dataset'].values()),
           'subjects': sum(v['subjects'] for v in d['per_dataset'].values()),
           'spoof_by_macro': {m: sum(v['spoof_by_macro'][m] for v in d['per_dataset'].values()) for m in ('print', 'replay')}}
    require(cov['totals'] == tot and cov['spoof_frames_without_same_subject_live'] == 0, 'totals')
    st = c['spoof_type_supervision']
    require((st['target'], st['vocabulary'], st['K'], st['class_index'], st['K_classification'], st['define_G_attack_type'],
             st['loss_cls_active'], st['netCls_in_optimizer'], st['no_unused_logits']) ==
            ('attack_macro', d['vocabulary'], len(d['vocabulary']), {m: i for i, m in enumerate(d['vocabulary'])},
             'DATA_DERIVED_CLASS_CARDINALITY', 2, True, False, True), 'spoof-type supervision K=2')
    rel = c['native_relation']
    require((rel['semantics'], rel['materialized_pair_list'], rel['native_manifest_created'], rel['spoof_index_order'],
             rel['live_pool_order']) == ('SAME_SUBJECT_ONLINE_RANDOM', False, False, 'ascending sample_id (bytewise)',
                                         'ascending sample_id (bytewise) within each subject'), 'native relation')
    sh = d['shell']
    lo = c['losses']
    for k in ('lambda_mmd', 'lambda_ip', 'lambda_pair', 'lambda_type', 'lambda_ort'):
        require(float(sh[k][1]) == float(lo[k]) and c['field_provenance'][k] == f'OFFICIAL_CODE train_generator.sh:{sh[k][0]}',
                'loss coefficient from the official launch ' + k)
    require(lo['lambda_pair'] == 5 and lo['loss_pair_active'] and lo['removed_terms'] == [], 'lambda_pair = 5 active')
    t = c['training']
    for key, sk in (('all_epochs', 'all_epochs'), ('effective_batch_size', 'batch_size'), ('hdim', 'hdim'),
                    ('workers', 'workers'), ('save_epoch', 'save_epoch'), ('test_epoch', 'test_epoch'),
                    ('pre_epoch', 'pre_epoch')):
        require(int(sh[sk][1]) == t[key], 'training value from the official launch ' + key)
    require((t['learning_rate'], t['attack_type'], t['drop_last'], t['shuffle'], t['scheduler']) == (2e-4, 2, False, True, None),
            'lr / attack_type / loader flags')
    rows = tot['train_spoof_frames']
    full, tail = divmod(rows, t['effective_batch_size'])
    ep = c['epoch_plan']
    require((ep['rows'], ep['global_batches'], ep['optimizer_steps_per_epoch'], ep['tail_batch'], ep['total_optimizer_steps'])
            == (rows, [240] * full + [tail], full + 1, tail, (full + 1) * t['all_epochs']), 'epoch plan')
    x = c['execution_compatibility']
    require((x['microbatch'], x['chunks_per_full_batch'], x['tail_120_chunks'], x['effective_batch_unchanged'],
             x['classification'], x['scientific_method_change'], x['bitwise_equivalence_to_physical_240_claimed']) ==
            (MICROBATCH, 240 // MICROBATCH, [MICROBATCH] * (tail // MICROBATCH), True, 'EXECUTION_RUNTIME_COMPATIBILITY',
             False, False) and tail % MICROBATCH == 0, 'microbatch execution mapping')
    rc = c['randomness_contract']
    require({k: rc['loader'][k] for k in d['loader']} == d['loader'] and rc['num_workers_frozen'] == 8 and
            rc['num_workers_affects_results'] is True, 'worker/randomness contract')
    require(c['checkpoint']['rule'] == 'OFFICIAL_GENERATOR_EPOCH_200' and c['checkpoint']['selection_uses_test'] is False,
            'checkpoint rule')
    require(c['seeds']['experiment_seeds'] == [42, 1337, 2026], 'seeds')
    require(c['unresolved_scientific_fields'] == [], 'zero unresolved scientific fields')
    dm = c['deferred_non_m6']
    require(dm['M8_track_b_generation_budget']['blocks_m6'] is False and
            dm['M8_track_b_generation_budget']['status'] == 'GENUINELY_AMBIGUOUS_DEFERRED_TO_M8', 'M8 deferral')
    op = c['decision_operation']
    require((op['training_runs'], op['optimizer_steps'], op['checkpoint_writes'], op['model_executions'], op['GPU_contacted'],
             op['image_reads'], op['TEST_rows_materialized'], op['banks_generated']) == (0, 0, 0, 0, False, 0, 0, 0),
            'no runtime activity')
    require(c['m6'] == {'M6_CLOSED': False, 'M7_started': False}, 'M6 open; M7 not started')
    md = reader(REPORT).decode()
    for token in ('M6FA', 'E06b', 'DSDG-NATIVE', 'K = 2', 'lambda_pair = 5', 'train_generator.sh:17', 'SAME_SUBJECT_ONLINE_RANDOM',
                  'NOT_INSTANTIABLE_MISSING_SUBJECT_ID', '15 × 240 + 120', '6 × 20', 'num_workers = 8', CONTRACT_SHA,
                  'M6_CLOSED = false', 'M7 HAS NOT STARTED', 'Unresolved scientific fields: none'):
        require(token in md, 'report token ' + token)
    return c, d


# ----------------------------------------------------------------- candidate-time checks (run once, here only)
def authority():
    require(git('branch', '--show-current').decode().strip() == BRANCH, 'branch')
    require(git('rev-parse', 'HEAD').decode().strip() == AUTHORITY, 'uncommitted candidate on the M6A9 authority')
    cache = ROOT / 'third_party/source_cache/facexzoo'
    require(git('rev-parse', 'HEAD', cwd=cache).decode().strip() == PINNED_COMMIT, 'source cache at pinned commit')
    require(git('ls-tree', 'HEAD', 'addition_module/DSDG/train_generator.sh', cwd=cache).decode().split()[2] ==
            TRAIN_SH_BLOB == git('hash-object', 'addition_module/DSDG/train_generator.sh', cwd=cache).decode().strip(),
            'train_generator.sh bytes == pinned commit tree blob')


def tracked_unchanged(bookkeeping):
    changed = sorted(git('diff', '--name-only', AUTHORITY).decode().split())
    require(changed == sorted([CONFIG_STATUS, *bookkeeping]),
            'only CONFIG_STATUS + ledger/index may differ from the authority: ' + json.dumps(changed))
    require(git('diff', '--name-only', '--diff-filter=D', AUTHORITY).decode().strip() == '', 'no deletion')
    require(git('diff', '--cached', '--name-only').decode().strip() == '', 'nothing staged')
    require(not (ROOT / FROZEN_TARGET).exists(), 'frozen E06b config NOT created')
    require(not (ROOT / 'manifests/dsdg_identity_pairs_v1.parquet').exists(), 'no native pair manifest')
    require(not (ROOT / 'outputs/audit/method_status.csv').exists(), 'no method_status.csv')
    base, now = at_authority(CONFIG_STATUS), read(CONFIG_STATUS)
    require(sha(base) == CONFIG_STATUS_AUTHORITY_SHA and now.startswith(base) and len(now) > len(base),
            'CONFIG_STATUS append-only')
    added = now[len(base):].decode()
    for token in ('M6FA', 'E06b', 'K = 2', 'lambda_pair = 5', 'M6_CLOSED = false', 'M7 HAS NOT STARTED'):
        require(token in added, 'CONFIG_STATUS M6FA note token ' + token)


def worktree(bookkeeping):
    status = sorted(git('status', '--porcelain', '--untracked-files=all').decode().splitlines())
    require(status == sorted(['?? ' + p for p in NEW] + [' M ' + p for p in (CONFIG_STATUS, *bookkeeping)]),
            'worktree holds exactly the M6FA candidate: ' + json.dumps(status))


def whitespace():
    for rel in (*NEW, CONFIG_STATUS):
        raw = read(rel)
        require(not re.search(rb'[ \t]\r?\n', raw) and b'\r' not in raw, 'LF, no whitespace before line endings ' + rel)


def expected_index():
    reader = csv.DictReader(io.StringIO(at_authority(INDEX).decode()))
    baseline = list(reader)
    rows = {r['path']: r for r in baseline}
    require(reader.fieldnames == ['path', 'size_bytes', 'sha256'] and len(rows) == len(baseline) == INDEX_BASELINE_ROWS,
            'baseline index')
    require(rows[CONFIG_STATUS]['sha256'] == CONFIG_STATUS_AUTHORITY_SHA, 'baseline CONFIG_STATUS row')
    for p in NEW:
        require(p not in rows, 'additive artifact ' + p)
    for p in (*NEW, CONFIG_STATUS):
        raw = read(p)
        rows[p] = {'path': p, 'size_bytes': str(len(raw)), 'sha256': sha(raw)}
    out = io.StringIO(newline='')
    writer = csv.DictWriter(out, fieldnames=reader.fieldnames, lineterminator='\r\n')
    writer.writeheader()
    writer.writerows(rows[k] for k in sorted(rows))
    return out.getvalue().encode(), len(rows)


LEDGER_EXPECTED = {
    'milestone': MILESTONE, 'classification': CLASSIFICATION, 'status': 'PASS', 'final_status': 'PASS',
    'record_kind': 'CONTRACT_RESOLUTION / AUDIT', 'method_id': 'E06b', 'method_name': 'DSDG-NATIVE',
    'track': 'B_NATIVE_FULL_SECONDARY', 'target_fidelity': 'FAITHFUL_OFFICIAL', 'fidelity_assessed': False,
    'train_datasets': ['casia_fasd', 'msu_mfsd'], 'siwmv2': 'NOT_INSTANTIABLE_MISSING_SUBJECT_ID',
    'spoof_vocabulary': ['print', 'replay'], 'K': 2, 'lambda_pair': 5, 'relation': 'SAME_SUBJECT_ONLINE_RANDOM',
    'train_spoof_frames': 3720, 'epoch_plan': '15 x 240 + 120', 'microbatch': MICROBATCH, 'num_workers_frozen': 8,
    'unresolved_scientific_fields': [], 'deferred': ['M8 E06b N_syn', 'M12 T09 interpretation'],
    'frozen_config_created': False, 'amendment_created': False, 'new_deviation': False, 'M6_closed': False,
    'M7_started': False, 'TEST_access': False, 'GPU_contacted': False, 'M8_bank': False, 'training_runs': 0,
    'optimizer_steps': 0, 'checkpoint_writes': 0, 'image_reads': 0, 'commit': False, 'push': False,
    'authority_commit': AUTHORITY, 'git_commit': AUTHORITY, 'contract_sha256': CONTRACT_SHA,
    'committed_prefix_rows': LEDGER_PREFIX_ROWS, 'committed_prefix_sha256': LEDGER_PREFIX_SHA,
    'artifact_index_rows_before': INDEX_BASELINE_ROWS, 'artifact_index_rows_after': INDEX_BASELINE_ROWS + len(NEW),
    'modified_existing_files': [CONFIG_STATUS]}


def check_ledger_row(row, prefix):
    for k, v in LEDGER_EXPECTED.items():
        require(row[k] == v, 'ledger field ' + k)
    require(sha(prefix) == row['committed_prefix_sha256'], 'ledger prefix binding')
    require(sorted(row['artifacts_sha256']) == sorted((*NEW, CONFIG_STATUS)), 'ledger artifact list')
    for path, h in row['artifacts_sha256'].items():
        require(sha(read(path)) == h, 'ledger artifact ' + path)
    require(row['tests'] and all(t['failures'] == 0 and (t['errors'] == 0 or t.get('errors_preexisting_environmental'))
                                 for t in row['tests']), 'ledger tests: no failures; errors only if pre-existing environmental')


def ledger_row(tests):
    return {**LEDGER_EXPECTED,
            'purpose': 'M6F-A: resolve the E06b DSDG-NATIVE contract from authority (no implementation, no training)',
            'notes': ('Authority-derived E06b contract candidate: CASIA+MSU native same-subject scope (SiW '
                      'NOT_INSTANTIABLE_MISSING_SUBJECT_ID); attack_macro K=2 {print, replay} data-derived; '
                      'lambda_pair=5 from train_generator.sh:17; official 200 epochs / batch 240 / lr 2e-4 / epoch-200 '
                      'checkpoint; M6D5c microbatch 20 reused as execution compatibility (tail 120 = 6x20); loader '
                      'workers frozen at 8. Zero unresolved scientific fields; frozen config not yet created. N_syn '
                      'deferred to M8. M6 open; M7 not started.'),
            'command': ('laptop only: .venv/bin/python -B -m unittest tests.test_m6fa_e06b_contract; python3 -B '
                        'tools/m6fa_e06b_contract_preflight.py --before-ledger / --append-ledger / --rebuild-index'),
            'artifacts_sha256': {p: sha(read(p)) for p in sorted((*NEW, CONFIG_STATUS))},
            'tests': tests, 'cwd': str(ROOT), 'host': socket.gethostname(), 'user': getpass.getuser(),
            'git_dirty': True, 'timestamp_utc': dt.datetime.now(dt.timezone.utc).isoformat()}


def verify(stage):
    authority()
    bookkeeping = {'before_ledger': [], 'before_index': [LEDGER], 'final': [INDEX, LEDGER]}[stage]
    tracked_unchanged(bookkeeping)
    whitespace()
    check_contract()
    prefix = at_authority(LEDGER)
    current = (ROOT / LEDGER).read_bytes()
    require(len(prefix.splitlines()) == LEDGER_PREFIX_ROWS and sha(prefix) == LEDGER_PREFIX_SHA and
            current.startswith(prefix), f'first {LEDGER_PREFIX_ROWS} ledger rows byte-identical')
    if stage == 'before_ledger':
        require(current == prefix, 'ledger not yet appended')
    else:
        require(len(current.splitlines()) == LEDGER_PREFIX_ROWS + 1 and current.endswith(b'\n'), 'exactly one append')
        check_ledger_row(json.loads(current[len(prefix):]), prefix)
    expected, count = expected_index()
    if stage == 'final':
        require((ROOT / INDEX).read_bytes() == expected, 'CRLF sorted artifact index')
    worktree(bookkeeping)
    require(not {'torch', 'yaml', 'pyarrow', 'numpy', 'PIL'} & set(sys.modules), 'static preflight')
    return {'status': 'PASS', 'stage': stage, 'ledger_rows': len(current.splitlines()),
            f'first_{LEDGER_PREFIX_ROWS}_rows_byte_identical': True, 'ledger_prefix_sha256': sha(prefix),
            'artifact_index_rows_before': INDEX_BASELINE_ROWS, 'artifact_index_rows_expected': count,
            'artifact_rows_added': len(NEW), 'modified_existing_files': [CONFIG_STATUS, *bookkeeping]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before-ledger', action='store_true')
    parser.add_argument('--append-ledger', action='store_true')
    parser.add_argument('--tests', help='JSON list of test-run summaries (with --append-ledger)')
    parser.add_argument('--rebuild-index', action='store_true')
    args = parser.parse_args()
    if args.before_ledger:
        result = verify('before_ledger')
    elif args.append_ledger:
        verify('before_ledger')
        row = ledger_row(json.loads(args.tests))
        with open(ROOT / LEDGER, 'ab') as fh:
            fh.write((json.dumps(row, sort_keys=True, ensure_ascii=False) + '\n').encode())
            fh.flush()
            os.fsync(fh.fileno())
        result = verify('before_index')
    elif args.rebuild_index:
        verify('before_index')
        (ROOT / INDEX).write_bytes(expected_index()[0])
        result = verify('final')
    else:
        result = verify('final')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
