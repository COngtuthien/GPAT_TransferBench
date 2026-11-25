"""E04 Physics-STD bank readiness record (laptop; deterministic; no host, GPU or data access).

  python3 tools/e04_bank_readiness.py build

Reads only committed files:
  configs/amendments/e04_bank_generation_contract.DRAFT.yaml          the owner-decision DRAFT
  outputs/audit/E04_BANK_CONTRACT_EVIDENCE.json (+ collected_host.json)  read-only CPU probe of checkpoints / graph
  outputs/audit/E04_BANK_CONTRACT_EVIDENCE/existing_bank_integrity_host.json   host re-verification of the 18 banks
  configs/amendments/e05_near_collapse_reporting_decision.yaml        owner reporting decision (section 9)
and writes
  docs/spec/amendments/GPAT_TransferBench_v1_0_E04_Bank_Generation_Contract_DRAFT.md   (rendering of the DRAFT)
  outputs/audit/E04_BANK_READINESS.{json,md} + E04_BANK_READINESS_SHA256.txt
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRAFT = 'configs/amendments/e04_bank_generation_contract.DRAFT.yaml'
DRAFT_DOC = 'docs/spec/amendments/GPAT_TransferBench_v1_0_E04_Bank_Generation_Contract_DRAFT.md'
EVIDENCE = 'outputs/audit/E04_BANK_CONTRACT_EVIDENCE.json'
INTEGRITY = 'outputs/audit/E04_BANK_CONTRACT_EVIDENCE/existing_bank_integrity_host.json'
PRIOR_VERIFY = 'outputs/audit/LEARNED_BANK_PRODUCTION/final_host_verification.json'
E05_DECISION = 'configs/amendments/e05_near_collapse_reporting_decision.yaml'
STORAGE_AUTHORITY = 'configs/amendments/learned_bank_storage_location.yaml'
CONTRACT = 'configs/amendments/learned_bank_generation_contract.yaml'
TOY_MANIFEST = ('outputs/audit/M8_SMOKE_EVIDENCE/run_a/select/toy_manifest.json',
                '50ac8fb8b9fc6938ba630d0853523d358250137c8fbbc62704d2a73830f7f690')
OUT = 'outputs/audit/E04_BANK_READINESS'
START_COMMIT = '7eefb5d8ef04d78371d93e1f87649d44752379d5'
STATUS_BLOCKED = 'E04_BANK_BLOCKED_BY_SCIENTIFIC_AUTHORITY'
FAMILY_CLOSURES = {
    'E01': 'outputs/audit/E01_PRODUCTION_FAMILY_CLOSURE.json', 'E02': 'outputs/audit/E02_PRODUCTION_FAMILY_CLOSURE.json',
    'E03': 'outputs/audit/LEARNED_BANK_PRODUCTION_E03_FAMILY_CLOSURE.json',
    'E05': 'outputs/audit/LEARNED_BANK_PRODUCTION_E05_FAMILY_CLOSURE.json',
    'E06b': 'outputs/audit/LEARNED_BANK_PRODUCTION_E06b_FAMILY_CLOSURE.json',
    'E06c': 'outputs/audit/LEARNED_BANK_PRODUCTION_E06c_FAMILY_CLOSURE.json'}

# The 18 semantics points of the owner instruction, each classified against the authorities (DRAFT question ids).
SEMANTICS = [
    ('generator graph and components', 'FROZEN', 'M6D3e network.generator (B, C, T, P, I_P heads; depth head unused)', 'implementation qualification of the inference construction'),
    ('source-spoof and target-live inputs', 'FROZEN', 'PAPER Eq.8/13-15; training_graph.py:30-39; pairs_train_v1 roles', None),
    ('geometry / landmarks / auxiliary tensors', 'FROZEN', 'D1-D3 pre-pass, D2 offsets dense_offsets(source=spoof, target=live), e04_feed_pack_v1', None),
    ('preprocessing', 'FROZEN', 'D8 RGB uint8/255 once; D5 no augmentation', None),
    ('output synthesis equation', 'PAPER_AND_FROZEN_GRAPH', 'PAPER Eq.8/13-15 realized as synthesized_spoof', 'E04-Q2 confirms plain vs harder-sample synthesis'),
    ('model inference mode', 'OWNER_RULING', 'not frozen; PhySTD silent', 'E04-Q1'),
    ('batch-normalization behaviour', 'OWNER_RULING', 'frozen graph uses batch statistics (training only); predecessor test uses moving statistics', 'E04-Q1'),
    ('checkpoint restoration', 'FROZEN_PLUS_QUALIFICATION', 'BASELINE_FINAL_STATE_V1 iteration_150000; raw variables (no EMA shadows)', 'restore generator variables only'),
    ('randomness and seed derivation', 'OWNER_RULING', 'none under E04-Q2 A; a new RNG domain only under E04-Q2 B', 'E04-Q2'),
    ('output tensor interpretation', 'FROZEN', 'D8 scale: RGB float32 nominally [0, 1], additive traces may leave it', None),
    ('float-to-uint8 conversion', 'OWNER_RULING', 'no PhySTD writer; predecessor clip + truncating cast is reference only', 'E04-Q3'),
    ('PNG encoding', 'OWNER_RULING', 'learned-bank common encoding does not yet cover E04', 'E04-Q6'),
    ('auxiliary outputs', 'OWNER_RULING', 'spec 7 needs only RGB', 'E04-Q4'),
    ('row identities', 'FROZEN', 'spec 6: pair_id, target_live_id, source_spoof_id, dataset', None),
    ('expected population', 'FROZEN', 'spec 6-7: 8838 TRAIN pairs per seed, 26514 per family', None),
    ('determinism and replay', 'OWNER_RULING', 'D12 is a training standard only', 'E04-Q5'),
    ('S7 toy acceptance', 'OWNER_RULING', 'existing S7 authority covers E03/E05/E06b/E06c only', 'E04-Q5, E04-Q6 and a new E04 S7 launch authority'),
    ('scientific fidelity disclosures', 'FROZEN_PLUS_NEW_DISCLOSURE', 'CONTROLLED_ADAPTATION; new: inference statistics and conversion provenance', 'disclosures follow the approved options'),
]


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dumps(obj):
    return json.dumps(obj, sort_keys=True, indent=1, ensure_ascii=True) + '\n'


def load(root, rel):
    return json.loads((Path(root) / rel).read_text())


def render_draft(d):
    L = ['# GPAT-TransferBench v1.0 — E04 Physics-STD bank-generation contract (DRAFT)', '',
         '**Status:** `%s` · approved: `%s` · %s' % (d['status'], str(d['approved']).lower(), d['date']), '',
         'Machine-readable source: `configs/amendments/e04_bank_generation_contract.DRAFT.yaml`. '
         'This page is rendered from it by `tools/e04_bank_readiness.py`; the YAML governs.', '',
         d['purpose'], '', '**Approval:** ' + d['approval_procedure'], '', '## Source situation', '']
    L += ['- **%s:** %s' % (k.replace('_', ' '), v) for k, v in d['source_situation'].items()]
    L += ['', '## Evidence (read-only CPU probe)', '']
    L += ['- **%s:** %s' % (k.replace('_', ' '), v) for k, v in d['evidence_facts'].items()]
    L += ['', '## Owner questions', '']
    for q in d['owner_questions']:
        L += ['### %s%s' % (q['id'], ' (pixel-affecting)' if q['pixel_affecting'] else ''), '', q['question'], '',
              '| option | rule | provenance | expected effect |', '|---|---|---|---|']
        for o in q['options']:
            L.append('| `%s` | %s | %s | %s |' % (o['id'], o['rule'], o['provenance'], o['expected_effect']))
        L += ['', '**Recommended:** `%s`. %s' % (q['recommended'], q['rationale']), '']
    L += ['## Already fixed by existing authority', '', '| item | rule | source |', '|---|---|---|']
    L += ['| %s | %s | %s |' % (x['item'], x['rule'], x['source']) for x in d['frozen_by_existing_authority']]
    L += ['', '## Needs implementation qualification after approval', '']
    L += ['- ' + x for x in d['requires_implementation_qualification_after_approval']]
    L += ['', '**Randomness:** ' + d['randomness'], '', '**Not authorized by this DRAFT:** ' +
          ', '.join(d['not_authorized_by_this_draft']) + '.', '']
    return '\n'.join(L)


def build(root=ROOT, out_root=None):
    root = Path(root)
    out = Path(out_root or root)
    for d in (out / DRAFT_DOC, out / OUT):
        d.parent.mkdir(parents=True, exist_ok=True)
    draft, ev, integ = load(root, DRAFT), load(root, EVIDENCE), load(root, INTEGRITY)
    prior, e05 = load(root, PRIOR_VERIFY), load(root, E05_DECISION)
    storage_auth = load(root, STORAGE_AUTHORITY)
    gates = []

    def gate(gid, ok, detail):
        gates.append({'id': gid, 'pass': bool(ok), 'detail': detail})

    approved_path = root / DRAFT.replace('.DRAFT', '')     # a later approval must bind this unchanged DRAFT
    approval_ok = not approved_path.exists() or \
        (json.loads(approved_path.read_text()).get('approved_draft') or {}).get('sha256') == sha_file(root / DRAFT)
    gate('draft_not_approved', draft['status'] == 'DRAFT_PENDING_OWNER_DECISION' and draft['approved'] is False and
         approval_ok, draft['status'])
    gate('draft_frozen_inputs', all(sha_file(root / k.split(' (')[0]) == v for k, v in
                                    draft['frozen_inputs_sha256'].items()), len(draft['frozen_inputs_sha256']))
    gate('evidence_collected', ev['status'] == 'E04_BANK_CONTRACT_EVIDENCE_COLLECTED' and ev['failed_gates'] == [] and
         sha_file(root / ev['collected']['path']) == ev['collected']['sha256'], ev['status'])
    gate('existing_banks_unchanged_since_final_verification', integ['banks'] == prior['banks'] and
         integ['e01_e02'] == prior['e01_e02'] and integ['markers'] == prior['markers'] and
         integ['host_scripts'] == prior['host_scripts'], integ['utc'])
    gate('existing_bank_counts', all(v['png_files'] == 8838 and v['exists'] for v in integ['banks'].values()) and
         len(integ['banks']) == 12 and len(integ['e01_e02']) == 6 and
         all(a == b for a, b in integ['e01_e02'].values()), {'learned': len(integ['banks']), 'e01_e02': len(integ['e01_e02'])})
    gate('production_clones_clean', all(v['exists'] and v['status'] == '' for v in integ['clones'].values()),
         sorted(v['head'][:7] for v in integ['clones'].values()))
    fam = {}
    for m, rel in FAMILY_CLOSURES.items():
        rec = load(root, rel)
        fam[m] = {'record': rel, 'sha256': sha_file(root / rel), 'status': rec['status'],
                  'outputs': rec.get('total_outputs', rec.get('outputs'))}
    gate('six_family_closures', all(v['status'].endswith('FAMILY_COMPLETED_AND_AUDITED') for v in fam.values()),
         {m: v['status'] for m, v in fam.items()})
    st = integ['storage']
    reg = storage_auth['bank_root_registry']
    e04_registered = 'E04' in reg['methods']
    gate('storage_preflight_live', st['ok'] and all(st['checks'].values()) and
         st['filesystem']['filesystem_uuid'] == reg['filesystem_uuid'], st['checks'])
    gate('storage_e04_not_silently_added', not e04_registered and 'E04' not in st['bank_root_entries'],
         {'registry_methods': reg['methods'], 'bank_root_entries': sorted(st['bank_root_entries'])})
    gate('e05_reporting_decision', e05['status'] == 'OWNER_DECIDED' and e05['changes_any_generation_contract'] is False
         and all(sha_file(root / k) == v for k, v in e05['evidence_sha256'].items()), e05['decision'])
    failed = [g['id'] for g in gates if not g['pass']]
    pending = [q['id'] for q in draft['owner_questions']]
    pixel = [q['id'] for q in draft['owner_questions'] if q['pixel_affecting']]
    ck = ev['final_checkpoints']
    est_bytes = 808567774
    doc = {
        'record': 'E04_BANK_READINESS', 'schema': 'e04_bank_readiness_v1', 'date': '2026-11-25',
        'status': STATUS_BLOCKED if not failed else 'E04_BANK_READINESS_RECORD_FAILED',
        'failed_gates': failed, 'gates': gates, 'start_commit': START_COMMIT,
        'blocking_owner_questions': pending, 'pixel_affecting_open_questions': pixel,
        'draft': {'path': DRAFT, 'sha256': sha_file(root / DRAFT), 'document': DRAFT_DOC,
                  'recommended': {q['id']: q['recommended'] for q in draft['owner_questions']}},
        'training_verification': {
            'closure': ev['scientific_authority'], 'checkpoints': ck,
            'environment': {'name': 'gpat-m6-e03', 'python': '3.8', 'tensorflow': '1.15.5',
                            'lock': 'environments/e04_training.lock.json a9629d46...',
                            'driver_policy': 'D9 driver-version-only 595.84 -> 580.178.04 (conditional, narrow)',
                            'gpu': 'NVIDIA GeForce RTX 3090 GPU-b722cd9d-c9fa-223c-915e-cc09d9b862b7'},
            'implementation_compatibility': 'the frozen M6D3e graph rebuilds on the host from a clean clone; all 246 '
                                            'generator graph variables (102 model + 144 optimizer slots) exist in every '
                                            'final checkpoint with equal shapes; synthesized_spoof needs only generator '
                                            'variables and the rgb/offset inputs'},
        'semantics_classification': [{'point': p, 'class': c, 'basis': b, 'open_item': o} for p, c, b, o in SEMANTICS],
        'implementation': {
            'status': 'NOT_STARTED_NOT_AUTHORIZED',
            'reason': 'E04-Q1, E04-Q2, E04-Q3 and E04-Q5 are pixel-affecting and unresolved; the instruction forbids an '
                      'arbitrary default. No adapter module, writer path or gate for E04 was added.',
            'added_this_session': ['tools/e04_bank_contract_probe.py (read-only evidence)',
                                   'tools/e04_bank_readiness.py (this record)'],
            'learned_bank_refuses_e04': 'methods.common.learned_bank.METHODS excludes E04; tools/learned_bank.py '
                                        '--method choices exclude E04'},
        's7_proposal': {
            'status': 'PROPOSAL_CONDITIONAL_ON_OWNER_RULINGS',
            'checkpoints': {s: {'run_id': v['run_id'], 'checkpoint_identity_sha256': v['checkpoint_identity_sha256'],
                                'files': v['files']} for s, v in ck.items()},
            'toy_population': {'manifest': TOY_MANIFEST[0], 'sha256': TOY_MANIFEST[1], 'pairs': 186,
                               'rule': 'the approved M8 toy pairs (TRAIN only, 0 TEST rows) for every seed; E04 is a '
                                       'pair method, so rows keep pair_id / target_live_id / source_spoof_id'},
            'environment': 'gpat-m6-e03 (E04 training lock, D9 driver policy, PRECISION_ENV, qualified session '
                           'configuration), approved RTX 3090; fresh clean clone from a git bundle at the S7 authority '
                           'commit, never the training checkout 18ba00e',
            'commands_after_implementation': [
                'tools/learned_bank.py s7 --method E04 --label A --seed <s> (one OS process per seed and run; to be '
                'implemented only after approval)',
                'tools/learned_bank.py s7 --method E04 --label B --seed <s>',
                'tools/learned_bank.py s7-compare --method E04 --seed <s>'],
            'runs': 'two independent runs (A then B) per seed, sequential, one GPU job at a time, tmux with exit '
                    'markers, no automatic rerun into an existing root',
            'outputs': {'per_run_per_seed': 186, 'total': 186 * 2 * 3},
            'reproducibility_standard': 'per E04-Q5 (recommended A: bitwise-identical manifests and PNG bytes between '
                                        'runs A and B; failure stops for an owner ruling)',
            'checkpoint_integrity': 'full SHA256 of the 3 files per seed before run A and after run B; no write',
            'atime_protection': 'TRAIN faces read with O_NOATIME after SHA256 check; atime ledger of all TRAIN, VAL and '
                                'TEST faces before and after: zero changes required',
            'firewall': 'feed-pack TRAIN closure only; any VAL/TEST id refused; no attack_macro read',
            'independent_audit': 'per-seed audit on the host by a separate read-only collector and a laptop builder: '
                                 'population equals the toy manifest, 186 decodable 256x256 RGB PNGs, finite raw '
                                 'outputs, approved conversion re-applied, graph construction record, no optimizer '
                                 'state, reader round trip, A/B comparison',
            'gpu_resources': 'single RTX 3090; inference generator at batch 1 needs well under 2 GiB; CPU warp and '
                             'scipy offsets dominate (estimate 0.3-0.6 s per pair, to be measured in S7: about 1-2 min '
                             'per S7 run, about 45-90 min per production seed); must not interfere with the unrelated '
                             'GPU jobs present on the host',
            'required_owner_authority': ['configs/amendments/e04_bank_generation_contract.yaml (approves this DRAFT)',
                                         'configs/amendments/learned_bank_s7_launch_authority_E04.yaml'],
            'production_unlock': ['E04 S7 record PASS for all three seeds',
                                  'configs/amendments/learned_bank_storage_location_E04.yaml (additive registry)',
                                  'configs/amendments/learned_bank_production_E04.yaml binding the S7 record',
                                  'live storage preflight at every launch; production audit includes toy-row byte '
                                  'equality']},
        'storage': {
            'bank_root': reg['bank_root'], 'filesystem': st['filesystem'], 'mode': st['mode'], 'checks': st['checks'],
            'total_bytes': st['total_bytes'], 'used_bytes': st['used_bytes'], 'free_bytes': st['free_bytes'],
            'free_inodes': st['free_inodes'], 'bank_root_entries': sorted(st['bank_root_entries']),
            'verified_utc': integ['utc'],
            'e04_covered_by_existing_authority': e04_registered,
            'e04_finding': 'not covered: the activated storage authority registers E03, E05, E06b, E06c only, and the '
                           'live preflight (bank_root_contents_ok) refuses any other top-level name, so creating '
                           '<bank_root>/E04 now would block every learned bank. An additive E04 storage record plus a '
                           'registry change in methods.common.learned_bank are required before any E04 directory exists.',
            'e04_estimated_bytes': {'per_seed': est_bytes, 'family': 3 * est_bytes,
                                    'basis': 'E03 seed-42 PNG total (8838 face-sized synthetic images); an estimate'}},
        'existing_banks': {'verified_utc': integ['utc'], 'equal_to_final_verification_utc': prior['utc'],
                           'learned_banks': {k: {'png_files': v['png_files'], 'manifest_sha256': v['manifest_sha256']}
                                             for k, v in sorted(integ['banks'].items())},
                           'e01_e02_manifests': {k: v[1] for k, v in sorted(integ['e01_e02'].items())},
                           'family_closures': fam},
        'e05_reporting_decision': {'path': E05_DECISION, 'sha256': sha_file(root / E05_DECISION),
                                   'decision': e05['decision']},
        'not_done': ['E04 adapter', 'E04 S7 GPU inference', 'E04 production', '<bank_root>/E04', 'downstream training',
                     'TEST access', 'DINOv3', 'GPAT']}
    (out / DRAFT_DOC).write_text(render_draft(draft))
    Path(str(out / OUT) + '.json').write_text(dumps(doc))
    Path(str(out / OUT) + '.md').write_text(report(doc))
    Path(str(out / OUT) + '_SHA256.txt').write_text(''.join('%s  %s\n' % (sha_file(str(out / OUT) + e), Path(OUT + e).name)
                                                            for e in ('.json', '.md')))
    return doc


def report(doc):
    L = ['# E04 Physics-STD bank readiness', '', '**Status:** `%s` (%d/%d record gates).' % (
        doc['status'], sum(g['pass'] for g in doc['gates']), len(doc['gates'])), '',
         'The three E04 generators are closed and verified. Bank inference is not yet defined by any authority: '
         '%d owner questions are open, %d of them pixel-affecting (%s). The DRAFT is `%s` (`%s`).' % (
             len(doc['blocking_owner_questions']), len(doc['pixel_affecting_open_questions']),
             ', '.join(doc['pixel_affecting_open_questions']), doc['draft']['path'], doc['draft']['sha256'][:16]),
         '', '## Final checkpoints', '', '| seed | run_id | checkpoint identity | data sha256 |', '|---|---|---|---|']
    for s, v in sorted(doc['training_verification']['checkpoints'].items(), key=lambda x: int(x[0])):
        L.append('| %s | `%s` | `%s` | `%s` |' % (s, v['run_id'], v['checkpoint_identity_sha256'],
                                                v['files']['iteration_150000.data-00000-of-00001']['sha256']))
    L += ['', '## Semantics classification', '', '| point | class | basis | open item |', '|---|---|---|---|']
    L += ['| %s | %s | %s | %s |' % (x['point'], x['class'], x['basis'], x['open_item'] or '—')
          for x in doc['semantics_classification']]
    L += ['', '## Recommended options (DRAFT, not approved)', '']
    L += ['- `%s`: `%s`' % kv for kv in sorted(doc['draft']['recommended'].items())]
    s = doc['storage']
    L += ['', '## Storage', '', '- `%s`, UUID `%s`, %s, mode %s; preflight checks all true: %s.' % (
        s['bank_root'], s['filesystem']['filesystem_uuid'], s['filesystem']['fstype'], s['mode'],
        all(s['checks'].values())),
          '- Free %.1f GiB of %.1f GiB; %d free inodes; entries %s.' % (
              s['free_bytes'] / 2 ** 30, s['total_bytes'] / 2 ** 30, s['free_inodes'], s['bank_root_entries']),
          '- ' + s['e04_finding'], '']
    e = doc['existing_banks']
    L += ['## Existing banks', '', '- Re-verified %s: all 12 learned banks (8838 PNG each) and 6 E01/E02 manifests equal '
          'the final verification of %s; six family closures `..._COMPLETED_AND_AUDITED`.' % (
              e['verified_utc'], e['equal_to_final_verification_utc']), '',
          '## E05', '', '- Owner reporting decision recorded (`%s`): %s' % (
              doc['e05_reporting_decision']['path'], doc['e05_reporting_decision']['decision']), '',
          '## S7 proposal (conditional)', '']
    p = doc['s7_proposal']
    L += ['- %s: %s' % (k.replace('_', ' '), v if isinstance(v, str) else json.dumps(v)) for k, v in p.items()
          if k not in ('checkpoints',)]
    L += ['', '## Gates', '', '| gate | pass |', '|---|---|']
    L += ['| `%s` | %s |' % (g['id'], 'PASS' if g['pass'] else '**FAIL**') for g in doc['gates']]
    L += ['', 'Not done: ' + ', '.join(doc['not_done']) + '.', '']
    return '\n'.join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['build'])
    ap.parse_args()
    d = build()
    print(d['status'], d['failed_gates'])


if __name__ == '__main__':
    main()
