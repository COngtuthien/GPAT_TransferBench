"""E04 Physics-STD production tooling qualification (framework-free; laptop .venv with numpy, Pillow and pyarrow).

Positive and negative cases of the four E04 production components before any real E04 bank exists:
  - the additive HDD storage extension (tools/e04_storage_extension_activation.py; lb registry / preflight);
  - the E04 production authority (tools/learned_bank_production_authority.py authority_e04; strict validation in
    methods.common.learned_bank.load_production_authority);
  - the launcher (tools/learned_bank_production_scripts/m8prod_launch.sh E04 case);
  - the independent production auditor E04 branch and the family closure (tools/learned_bank_production_audit.py).

The auditor's positive case is a synthetic collection for a synthetic execution commit built in a throwaway detached
git worktree (real history; the current tooling overlaid; an activated storage extension derived from a passing
synthetic host verification; the authority derived by the tool and committed on top), so every git, authority, S7
rebuild, pairs_train_v1, split_v1 and M2 check runs for real. The worktree and its commits are discarded afterwards.
"""
import copy
import gzip
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from methods.common import learned_bank as lb  # noqa: E402

N = lb.N_SYN
SCRIPTS = ROOT / 'tools' / 'learned_bank_production_scripts'
DATE = '2026-11-25'
S7_REL = 'outputs/audit/LEARNED_BANK_S7_E04.json'
GITC = ['git', '-c', 'user.name=e04-qualification', '-c', 'user.email=e04-qualification@invalid', '-c',
        'commit.gpgsign=false']
TOOLING = ['methods/common/learned_bank.py', 'tools/learned_bank.py', 'tools/learned_bank_production_audit.py',
           'tools/learned_bank_production_authority.py', 'tools/learned_bank_production_summary.py',
           'tools/learned_bank_s7_audit.py', 'tools/e04_storage_extension_activation.py',
           'tools/learned_bank_production_scripts/m8prod_launch.sh']


def load(name, rel, root=ROOT):
    spec = importlib.util.spec_from_file_location(name, str(Path(root) / rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha(b):
    return hashlib.sha256(b).hexdigest()


TOOL = load('e04prod_audit', 'tools/learned_bank_production_audit.py')
GEN = load('e04prod_auth', 'tools/learned_bank_production_authority.py')
ACT = load('e04prod_act', 'tools/e04_storage_extension_activation.py')
S7T = load('e04prod_s7', 'tools/learned_bank_s7_audit.py')


def host_verification():
    """A passing host verification document (as tools/e04_storage_extension_activation.py host prints it)."""
    draft = json.loads((ROOT / ACT.DRAFT).read_text())
    checks = {k: True for k in draft['verified_state']['checks']}
    fs = {'filesystem_uuid': ACT.UUID, 'fstype': 'ext4', 'mount_point': '/media/ext_data',
          'mount_options': 'rw,nosuid,nodev,relatime', 'super_options': 'rw', 'source': '/dev/sda2'}
    pf = {'bank_root': ACT.BANK_ROOT, 'exists': True, 'mode_octal': '0o700', 'filesystem': fs, 'unexpected_entries': [],
          'checks': checks, 'ok': True}
    tree = []
    for m in ACT.EXISTING:
        tree.append([m, 'dir', 1001, '0o700'])
        tree += [[m + '/' + s, 'dir', 1001, '0o555'] for s in ACT.SEED_DIRS]
    return {'phase': 'host', 'utc': '2026-11-25T15:00:00Z', 'repo': '/x', 'git_head': 'a' * 40, 'git_status': '',
            'registered_methods_in_clone': list(ACT.EXISTING), 'preflight': pf,
            'preflight_with_e04_registered': copy.deepcopy(pf),
            'root': {'path': ACT.BANK_ROOT, 'realpath': ACT.BANK_ROOT, 'owner': 'student20261', 'mode': '0o700',
                     'is_symlink': False, 'mount_point_identity': fs},
            'capacity': {'total_bytes': 4 << 40, 'free_bytes': 3 << 40, 'free_inodes': 243000000,
                         'total_inodes': 244000000},
            'tree': tree, 'e04_exists': False, 'mountinfo_lines': []}


def copy_into(t, rels, src=ROOT):
    for rel in rels:
        Path(t, rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(str(Path(src) / rel), str(Path(t, rel)))


RECORDS = [lb.E04_CONTRACT_PATH, lb.E04_DRAFT_PATH, lb.E04_S7_AUTHORITY_PATH, lb.E04_INPUT_RULING_PATH,
           lb.E04_FAMILY_CLOSURE_RECORD, lb.E04_COMPLETION_RECORD, lb.E04_S7_RECORD_PATH, lb.STORAGE_AUTHORITY_PATH,
           GEN.E04_DRAFT, ACT.DRAFT, 'outputs/audit/E04_THREE_SEED_FINAL_CHECKPOINTS.csv']


def activated_extension_bytes(h=None):
    raw = json.dumps(h or host_verification(), sort_keys=True).encode()
    return (json.dumps(ACT.derive(ROOT, DATE, raw), indent=1, sort_keys=True) + '\n').encode(), raw


# =================================================================================================== storage
class StorageExtension(unittest.TestCase):
    def test_derived_extension_registers_e04_only_when_loaded(self):
        ext, _ = activated_extension_bytes()
        d = json.loads(ext)
        self.assertEqual((d['status'], d['approved'], d['activation_verification']['ok']), ('OWNER_APPROVED', True, True))
        draft = json.loads((ROOT / ACT.DRAFT).read_text())
        for k in ('registry_extension', 'base_storage_authority', 'methods', 'schema_version', 'estimate'):
            self.assertEqual(d[k], draft[k], k)
        with tempfile.TemporaryDirectory() as t:
            copy_into(t, [lb.STORAGE_AUTHORITY_PATH])
            self.assertEqual(lb.registered_bank_methods(t), lb.METHODS)
            Path(t, lb.E04_STORAGE_EXTENSION_PATH).write_bytes(ext)
            self.assertEqual(lb.registered_bank_methods(t), lb.BANK_METHODS)
            st = lb.load_storage_authority(t, method_id='E04')
            self.assertEqual(lb.production_bank_dir('E04', 1337, st), Path(lb.HDD['bank_root']) / 'E04' / 'seed_1337')
            self.assertEqual(lb.load_storage_authority(t)['methods'], list(lb.METHODS))   # never without method_id
            with self.assertRaises(lb.LearnedBankError):
                lb.production_bank_dir('E04', 42, lb.load_storage_authority(t))

    def test_prerequisite_failures_are_refused(self):
        def mutate(fn):
            h = host_verification()
            fn(h)
            with self.assertRaises(SystemExit):
                ACT.derive(ROOT, DATE, json.dumps(h).encode())
        mutate(lambda h: h['preflight']['filesystem'].update(filesystem_uuid='0' * 8))            # wrong HDD UUID
        mutate(lambda h: h['preflight']['filesystem'].update(mount_point='/media/student20261/x'))   # udisks mount
        mutate(lambda h: h['preflight']['filesystem'].update(fstype='ntfs'))
        mutate(lambda h: h.update(e04_exists=True))                                                  # E04 pre-created
        mutate(lambda h: h['preflight'].update(unexpected_entries=['E07']))                          # unexpected content
        mutate(lambda h: h['tree'].append(['stray', 'symlink', 1001, '0o777']))                      # symlink
        mutate(lambda h: h['root'].update(is_symlink=True))
        mutate(lambda h: h['root'].update(mode='0o2700'))                                            # setgid / mode
        mutate(lambda h: h['root'].update(owner='root'))
        mutate(lambda h: h['capacity'].update(free_bytes=1 << 30))                                   # no space
        mutate(lambda h: h['capacity'].update(free_inodes=10))
        mutate(lambda h: h['preflight']['checks'].update(mount_read_write=False))                    # read-only
        mutate(lambda h: h.update(git_status=' M x'))
        mutate(lambda h: h.update(registered_methods_in_clone=list(lb.BANK_METHODS)))
        mutate(lambda h: h['tree'].pop())                                                           # missing seed dir

    def test_tampered_extension_record_refused(self):
        ext, _ = activated_extension_bytes()
        for patch in (lambda d: d['registry_extension'].update(filesystem_uuid='x'),
                      lambda d: d['registry_extension'].update(bank_root='/media/student20261/x/banks'),
                      lambda d: d['registry_extension'].update(layout='<bank_root>/../E04/seed_<seed>/'),
                      lambda d: d.update(methods=['E04', 'E05']), lambda d: d.update(status='DRAFT'),
                      lambda d: d['activation_verification'].update(ok=False),
                      lambda d: d['base_storage_authority'].update(sha256='0' * 64)):
            d = json.loads(ext)
            patch(d)
            with tempfile.TemporaryDirectory() as t:
                copy_into(t, [lb.STORAGE_AUTHORITY_PATH])
                Path(t, lb.E04_STORAGE_EXTENSION_PATH).write_text(json.dumps(d))
                self.assertEqual(lb.registered_bank_methods(t), lb.METHODS)
                with self.assertRaises(lb.LearnedBankBlocked):
                    lb.load_storage_authority(t, method_id='E04')

    def test_symlink_traversal_and_unexpected_contents_refused(self):
        st = {'bank_root': lb.HDD['bank_root'], 'methods': list(lb.BANK_METHODS)}
        for m, s in (('../E04', 42), ('E04/..', 42), ('E04', '42'), ('E04', 43), ('E07', 42), ('', 42)):
            with self.assertRaises(lb.LearnedBankError):
                lb.production_bank_dir(m, s, st)
        with self.assertRaises(lb.LearnedBankError):
            lb.production_bank_dir('E04', 42, dict(st, bank_root='/media/student20261/x/student20261_gpat_m8_banks'))
        with tempfile.TemporaryDirectory() as t:
            uid = os.getuid()
            for m in lb.METHODS:
                for s in lb.SEEDS:
                    os.makedirs(os.path.join(t, m, 'seed_%d' % s))
            self.assertEqual(lb.bank_root_contents_ok(t, uid, lb.METHODS), [])
            os.makedirs(os.path.join(t, 'E04', 'seed_42'))
            self.assertEqual(lb.bank_root_contents_ok(t, uid, lb.METHODS), ['E04'])        # E04 without extension
            self.assertEqual(lb.bank_root_contents_ok(t, uid, lb.BANK_METHODS), [])
            os.symlink(os.path.join(t, 'E03', 'seed_42'), os.path.join(t, 'E04', 'seed_1337'))
            self.assertEqual(lb.bank_root_contents_ok(t, uid, lb.BANK_METHODS), ['E04/seed_1337'])
            os.makedirs(os.path.join(t, 'E07'))
            os.symlink(os.path.join(t, 'E03'), os.path.join(t, 'E08'))
            self.assertEqual(lb.bank_root_contents_ok(t, uid, lb.BANK_METHODS), ['E04/seed_1337', 'E07', 'E08'])
        with tempfile.TemporaryDirectory() as t:
            link = os.path.join(t, 'root_link')
            os.mkdir(os.path.join(t, 'real'), 0o700)
            os.symlink(os.path.join(t, 'real'), link)
            pf = lb.storage_preflight(link, expected={'bank_root': link})
            self.assertFalse(pf['ok'])
            self.assertFalse(pf['checks']['not_symlink'])

    def test_committed_activation_is_derived(self):
        p = ROOT / ACT.ACTIVATED
        if not p.exists():
            self.skipTest('E04 storage extension not activated at this commit')
        raw = (ROOT / ACT.EVIDENCE).read_bytes()
        d = json.loads(p.read_text())
        self.assertEqual(d, ACT.derive(ROOT, d['date'], raw))
        self.assertEqual(p.read_text(), json.dumps(d, indent=1, sort_keys=True) + '\n')
        self.assertEqual(lb.registered_bank_methods(), lb.BANK_METHODS)


# =================================================================================================== authority
class ProductionAuthority(unittest.TestCase):
    def _root(self, auth=None, ext=True, rels=RECORDS):
        t = tempfile.mkdtemp(prefix='e04auth_')
        self.addCleanup(shutil.rmtree, t, True)
        copy_into(t, rels)
        if ext:
            Path(t, lb.E04_STORAGE_EXTENSION_PATH).write_bytes(activated_extension_bytes()[0])
        if auth is not None:
            Path(t, lb.PRODUCTION_AUTHORITY_TEMPLATE.format(method='E04')).write_text(json.dumps(auth, indent=1) + '\n')
        return t

    def auth(self):
        return GEN.authority_e04('0' * 40, DATE, root=self._root())

    def test_valid_authority_accepted(self):
        a = self.auth()
        t = self._root(a)
        for s in lb.SEEDS:
            r = lb.load_production_authority('E04', s, t, live_storage=False).record()
            self.assertEqual((r['mode'], r['methods'], r['seeds']), ('PRODUCTION', ['E04'], [s]))
            self.assertEqual(sorted(r['records']), sorted([
                lb.E04_CONTRACT_PATH, lb.PRODUCTION_AUTHORITY_TEMPLATE.format(method='E04'), lb.E04_S7_RECORD_PATH,
                lb.STORAGE_AUTHORITY_PATH, lb.E04_STORAGE_EXTENSION_PATH, lb.E04_COMPLETION_RECORD,
                lb.E04_S7_AUTHORITY_PATH, lb.E04_INPUT_RULING_PATH, lb.E04_FAMILY_CLOSURE_RECORD]))
            self.assertEqual(r['contract_sha256'], TOOL.E04_CONTRACT_SHA256)
        self.assertEqual(a['fidelity'], 'CONTROLLED_ADAPTATION')
        self.assertEqual(a['display_name'], 'Physics-STD (controlled geometry/depth reconstruction)')
        self.assertEqual(a['population']['rows_per_seed'] * 3, a['population']['family_rows'])
        self.assertEqual(a['environment']['precision_env'], dict(sorted(S7T.E04_PRECISION_ENV)))

    def test_missing_authority_storage_or_s7_refused(self):
        with self.assertRaises(lb.LearnedBankBlocked) as cm:
            lb.load_production_authority('E04', 42, self._root(None), live_storage=False)
        self.assertEqual(cm.exception.status, lb.STATUS_PRODUCTION)
        t = self._root(self.auth(), ext=False)
        with self.assertRaises(lb.LearnedBankBlocked) as cm:
            lb.load_production_authority('E04', 42, t, live_storage=False)
        self.assertEqual(cm.exception.status, lb.STATUS_STORAGE)
        with self.assertRaises(SystemExit):                       # the tool refuses without an activated extension
            GEN.authority_e04('0' * 40, DATE, root=self._root(ext=False))
        draft_only = self._root()
        d = json.loads(Path(draft_only, GEN.E04_DRAFT).read_text())
        Path(draft_only, lb.PRODUCTION_AUTHORITY_TEMPLATE.format(method='E04')).write_text(json.dumps(d))
        with self.assertRaises(lb.LearnedBankBlocked):
            lb.load_production_authority('E04', 42, draft_only, live_storage=False)

    def test_tampered_authority_refused(self):
        a = self.auth()
        patches = {
            'contract hash': lambda x: x.update(contract_sha256='1' * 64),
            's7 record sha': lambda x: x['s7_acceptance_record'].update(sha256='0' * 64),
            's7 record path': lambda x: x['s7_acceptance_record'].update(path='outputs/audit/LEARNED_BANK_S7_E03.json'),
            'status': lambda x: x.update(status='DRAFT'), 'approved': lambda x: x.update(approved=False),
            'method': lambda x: x.update(method='E03'), 'seeds': lambda x: x.update(seeds=[42]),
            'seed order': lambda x: x.update(seeds=[1337, 42, 2026]),
            'checkpoint sha': lambda x: x['checkpoints']['1337']['sha256'].update(index='0' * 64),
            'checkpoint run id': lambda x: x['checkpoints']['2026'].update(run_id='a42fdd2a2accaf0a'),
            'checkpoint identity': lambda x: x['checkpoints']['42'].update(checkpoint_identity_sha256='0' * 64),
            'checkpoint iteration': lambda x: x['checkpoints']['42'].update(iteration=145000),
            'checkpoint seed swap': lambda x: x['checkpoints'].update({'42': x['checkpoints']['1337']}),
            'extra checkpoint': lambda x: x['checkpoints'].update({'7': x['checkpoints']['42']}),
            'config': lambda x: x['frozen_config'].update(sha256='0' * 64),
            'BN decision': lambda x: x['decisions'].update({'E04-Q1_normalization_statistics': 'B_BATCH_STATISTICS'}),
            'conversion decision': lambda x: x['decisions'].update({'E04-Q3_float_to_uint8': 'B_ROUND'}),
            'fidelity': lambda x: x.update(fidelity='OFFICIAL_REPRODUCTION'),
            'label': lambda x: x.update(display_name='Physics-STD'),
            's7 launch authority': lambda x: x['s7_launch_authority'].update(sha256='0' * 64),
            'input ruling': lambda x: x['input_allowlist_ruling'].update(sha256='0' * 64),
            'training closure': lambda x: x['e04_records']['family_closure'].update(sha256='0' * 64),
            'storage extension': lambda x: x['storage_extension'].update(sha256='0' * 64),
            'scope gpat': lambda x: x['authorizes'].update(gpat=True),
            'scope retrain': lambda x: x['authorizes'].update(generator_retraining=True),
            'scope methods': lambda x: x['authorizes'].update(methods=['E04', 'E05']),
            'scope extra key': lambda x: x['authorizes'].update(downstream_evaluation=True),
        }
        for name, patch in patches.items():
            with self.subTest(patch=name):
                bad = copy.deepcopy(a)
                patch(bad)
                t = self._root(bad)
                with self.assertRaises(lb.LearnedBankBlocked) as cm:
                    lb.load_production_authority('E04', 1337, t, live_storage=False)
                self.assertEqual(cm.exception.status, lb.STATUS_PRODUCTION, name)

    def test_wrong_or_altered_records_refused(self):
        a = self.auth()
        for rel, edit in ((lb.E04_S7_RECORD_PATH, lambda d: d.update(status='E04_S7_PARTIALLY_QUALIFIED')),
                          (lb.E04_S7_AUTHORITY_PATH, lambda d: d.update(note='x')),
                          (lb.E04_INPUT_RULING_PATH, lambda d: d.update(note='x')),
                          (lb.E04_CONTRACT_PATH, lambda d: d.update(note='x'))):
            with self.subTest(record=rel):
                t = self._root(a)
                d = json.loads(Path(t, rel).read_text())
                edit(d)
                Path(t, rel).write_text(json.dumps(d))
                with self.assertRaises(lb.LearnedBankBlocked):
                    lb.load_production_authority('E04', 42, t, live_storage=False)
        t = self._root()
        d = json.loads(Path(t, lb.E04_S7_RECORD_PATH).read_text())
        d['s7_acceptance_passed'] = False
        Path(t, lb.E04_S7_RECORD_PATH).write_text(json.dumps(d))
        with self.assertRaises(SystemExit):
            GEN.authority_e04('0' * 40, DATE, root=t)

    def test_original_methods_unaffected(self):
        for M in lb.METHODS:
            p = ROOT / lb.PRODUCTION_AUTHORITY_TEMPLATE.format(method=M)
            rec = json.loads(p.read_text())
            self.assertEqual(rec, GEN.authority(M, rec['code_authority']['source_commit'], rec['date']))
            r = lb.load_production_authority(M, 42, live_storage=False).record()
            self.assertNotIn(lb.E04_S7_AUTHORITY_PATH, r['records'])
            self.assertNotIn(lb.E04_STORAGE_EXTENSION_PATH, r['records'])

    def test_committed_authority_is_derived(self):
        p = ROOT / lb.PRODUCTION_AUTHORITY_TEMPLATE.format(method='E04')
        if not p.exists():
            self.skipTest('E04 production authority not activated at this commit')
        rec = json.loads(p.read_text())
        self.assertEqual(rec, GEN.authority_e04(rec['code_authority']['source_commit'], rec['date']))
        self.assertEqual(p.read_text(), json.dumps(rec, indent=1) + '\n')
        for s in lb.SEEDS:
            lb.load_production_authority('E04', s, live_storage=False)


# =================================================================================================== launcher
class Launcher(unittest.TestCase):
    def test_e04_case_uses_the_e04_interpreter_and_precision_env(self):
        text = (SCRIPTS / 'm8prod_launch.sh').read_text()
        self.assertEqual(subprocess.run(['bash', '-n', str(SCRIPTS / 'm8prod_launch.sh')]).returncode, 0)
        self.assertIn('  E03|E04) PY=%s\n' % lb.ENVIRONMENTS['E04']['python'], text)
        self.assertEqual(lb.ENVIRONMENTS['E04']['python'], TOOL.E04_PYTHON)
        e04 = 'PATH=/usr/bin:/bin LANG=C.UTF-8 ' + ' '.join('%s=%s' % kv for kv in sorted(S7T.E04_PRECISION_ENV))
        self.assertEqual(TOOL.E03_ENV, e04)
        self.assertEqual(text.count('ENVS="%s"' % e04), 1)
        self.assertIn('nvidia-smi --query-compute-apps', text)
        self.assertIn('gpu_samples.tsv; do test ! -e', text)
        self.assertIn('test ! -e $BANK', text)
        self.assertNotIn('kill', text)

    def test_cli_resolves_e04_storage_through_the_extension(self):
        src = (ROOT / 'tools/learned_bank.py').read_text()
        self.assertIn('storage = lb.load_storage_authority(method_id=args.method)', src)
        self.assertNotIn('storage = lb.load_storage_authority()\n', src)


# =================================================================================================== auditor
WORKDIR = TOOL.WORKDIR
BANK = TOOL.BANK_ROOT + '/E04/seed_42'


def git(wt, *a, check=True):
    return subprocess.run(GITC + ['-C', str(wt)] + list(a), capture_output=True, text=True, check=check)


class Worktree:
    """Throwaway detached worktree: real history, current tooling, activated extension (commit A = source commit),
    derived E04 authority (commit B = execution commit)."""

    def __init__(self):
        self.dir = Path(tempfile.mkdtemp(prefix='e04prodq_')) / 'wt'
        arel = lb.PRODUCTION_AUTHORITY_TEMPLATE.format(method='E04')
        added = subprocess.run(['git', '-C', str(ROOT), 'log', '--format=%H', '--diff-filter=A', '--', arel],
                               capture_output=True, text=True).stdout.split()
        base = (added[0] + '^') if added else 'HEAD'
        git(ROOT, 'worktree', 'add', '-q', '--detach', str(self.dir), base)
        for rel in TOOLING:
            shutil.copyfile(str(ROOT / rel), str(self.dir / rel))
        if not (self.dir / ACT.ACTIVATED).exists():
            ext, raw = activated_extension_bytes()
            (self.dir / ACT.EVIDENCE).parent.mkdir(parents=True, exist_ok=True)
            (self.dir / ACT.EVIDENCE).write_bytes(raw)
            (self.dir / ACT.ACTIVATED).write_bytes(ext)
        git(self.dir, 'add', '-A')
        if git(self.dir, 'status', '--porcelain').stdout.strip():
            git(self.dir, 'commit', '-q', '-m', 'qualification: tooling overlay + storage extension')
        self.A = git(self.dir, 'rev-parse', 'HEAD').stdout.strip()
        r = subprocess.run([sys.executable, '-B', 'tools/learned_bank_production_authority.py', '--method', 'E04',
                            '--source-commit', self.A, '--date', DATE], cwd=str(self.dir), capture_output=True,
                           text=True)
        assert r.returncode == 0, r.stderr
        git(self.dir, 'add', arel)
        git(self.dir, 'commit', '-q', '-m', 'qualification: E04 production authority')
        self.B = git(self.dir, 'rev-parse', 'HEAD').stdout.strip()

    def remove(self):
        git(ROOT, 'worktree', 'remove', '--force', str(self.dir), check=False)
        shutil.rmtree(str(self.dir.parent), True)


def synthetic_collected(wt, S=42):
    """A collection of a bank that satisfies every E04 gate for execution commit wt.B (seed 42): the 186 toy rows are
    the real S7 run-A manifest rows (mode PRODUCTION); every other row follows pairs_train_v1 with its M2 face SHA256."""
    repo = wt.dir
    s7rec = json.loads((repo / S7_REL).read_text())
    s7c = s7rec['collected']
    runA = s7c['runs']['A'][str(S)]
    prov7 = runA['provenance']
    toy = [dict(zip(S7T.MANIFEST_FIELDS_E04, r)) for r in runA['manifest_rows']]
    by_pos = {r['row_index']: r for r in toy}
    rel = TOOL.pair_relation(repo)
    ids = {x[1] for x in rel} | {x[2] for x in rel}
    faces = TOOL.face_table(repo, ids)
    tmpl = toy[0]
    rows = []
    for i, (pid, live, spoof, ds) in enumerate(rel):
        if i in by_pos:
            r = dict(by_pos[i], mode='PRODUCTION')
        else:
            r = dict(tmpl, row_index=i, row_id=pid, pair_id=pid, target_live_id=live, source_spoof_id=spoof, dataset=ds,
                     target_face_sha256=faces[live][2], source_face_sha256=faces[spoof][2], mode='PRODUCTION',
                     output_path='bank/%s.png' % pid, output_sha256=sha(b'o' + pid.encode()),
                     output_pixel_sha256=sha(b'p' + pid.encode()), raw_output_min=0.0, raw_output_max=1.0)
        rows.append(r)
    body = ''.join(lb.canonical_json(r) + '\n' for r in rows).encode('ascii')
    msha = sha(body)
    records = lb.load_production_authority('E04', S, repo, live_storage=False).record()['records']
    clone = '%s/m8_learned_banks/clones/prod_E04_%s' % (WORKDIR, wt.B)
    launch, end = '2099-01-01T00:10:00Z', '2099-01-01T00:40:00Z'
    prov = copy.deepcopy(prov7)
    prov.update(status='completed', rows=N, outputs=N, failure_count=0, failures=[], mode='PRODUCTION', method_id='E04',
                bank_seed=S, run_dir=BANK, git_head=wt.B, manifest_sha256=msha, start_utc='2099-01-01T00:10:30Z',
                end_utc='2099-01-01T00:39:00Z', wall_time_seconds=1700.0,
                post_write_verification={'ok': True, 'files_checked': N, 'problems': [], 'rows': N},
                authorization={'mode': 'PRODUCTION', 'methods': ['E04'], 'seeds': [S],
                               'contract_sha256': TOOL.E04_CONTRACT_SHA256, 'label': 'production', 'records': records},
                adapter_finalize=dict(prov7['adapter_finalize'], forwards=N),
                reader=dict(prov7['reader'], reads=2 * N, atime_checked_reads=2 * N),
                population={'kind': 'PRODUCTION_FROZEN_TRAIN_POPULATION', 'rows': N, 'method_id': 'E04', 'seed': S,
                            'allowlist_source': TOOL.E04_ALLOWLIST_SOURCE, 'pairs_val_opened': False,
                            'pack_manifest_sha256': TOOL.E04_PACK_MANIFEST_SHA256,
                            'relation_source': 'e04_feed_pack_v1 pairs_train.json'})
    real = json.loads((ROOT / 'outputs/audit/LEARNED_BANK_PRODUCTION/E05_s2026/collected.json').read_text())
    ck = copy.deepcopy(real['checkpoints_after'])
    pf7 = s7c['preflights'][str(S)]['A']
    for s in lb.SEEDS:
        ck['E04|%d' % s] = copy.deepcopy(pf7['checkpoints'][str(s)])
    registry = copy.deepcopy(real['registry_after'])
    e0102 = TOOL.e01_e02_expected(repo)
    probe = copy.deepcopy(pf7['environment']['probe'])
    probe.pop('s7_authority')
    probe['production_authority'] = {'records': records, 'mode': 'PRODUCTION', 'methods': ['E04'], 'seeds': [S]}
    storage = {'ok': True, 'bank_root': TOOL.BANK_ROOT, 'exists': True, 'mode_octal': '0o700', 'unexpected_entries': [],
               'checks': {}, 'filesystem': {'filesystem_uuid': TOOL.HDD_UUID, 'mount_point': '/media/ext_data',
                                            'fstype': 'ext4'},
               'capacity': {'free_bytes': 3 << 40, 'free_inodes': 243000000, 'total_bytes': 4 << 40}}
    gpu = {'name': TOOL.GPU_NAME, 'uuid': TOOL.GPU_UUID, 'driver': TOOL.DRIVER, 'free_mib': 14000, 'used_mib': 10000,
           'total_mib': 24576, 'compute_apps': [{'pid': 1, 'name': 'python', 'used_mib': 7000, 'ps': 'other user job'}]}
    command = ('#!/bin/bash\n# M8 learned-bank production: method E04 seed %d\ndate -u +%%FT%%TZ > %s/m8prod_E04_s%d.'
               'launch_time\ncd %s || { echo 97 > %s/m8prod_E04_s%d.exit; exit 97; }\n( while [ ! -s x ]; do '
               'nvidia-smi --query-compute-apps=pid ...; done ) &\nenv -i HOME=/home/student20261 %s nice -n 10 %s -B '
               'tools/learned_bank.py production --method E04 --seed %d --runtime-root %s\necho $? > x\nwait\n'
               % (S, WORKDIR, S, clone, WORKDIR, S, TOOL.E03_ENV, TOOL.E04_PYTHON, S, TOOL.RUNTIME))
    eq_runs = {}
    for L in ('A', 'B'):
        eq_runs[L] = {'dir': s7rec['outputs'][str(S)]['run_dirs'][L],
                      'manifest_sha256': s7rec['outputs'][str(S)]['manifest_sha256'][L], 'rows': 186, 'rows_matched': 186,
                      'missing_in_production': [], 'identity_mismatches': [], 'field_mismatches': [],
                      'field_mismatch_count': 0, 'byte_mismatches': [], 'byte_mismatch_count': 0, 'pixel_mismatches': [],
                      'pixel_mismatch_count': 0, 'pixel_arrays_equal': 186, 'row_indices': [r['row_index'] for r in toy],
                      'pair_ids_sha256': sha('\n'.join(r['pair_id'] for r in toy).encode())}
    stats = {'per_image_std': {'min': 10.0, 'p05': 20.0, 'median': 40.0, 'p95': 60.0, 'max': 70.0},
             'per_image_mean': {'min': 50.0, 'median': 110.0, 'max': 200.0}, 'per_image_min_median': 0,
             'per_image_max_median': 255, 'distinct_pixel_sha256': N, 'exact_duplicate_groups': 0,
             'exact_duplicate_rows': 0, 'coarse_8x8_luma_signature_groups': 0, 'coarse_8x8_luma_signature_rows': 0}
    c = {
        'phase': 'collect', 'method': 'E04', 'seed': S, 'utc': '2099-01-01T01:00:00Z', 'repo': clone,
        'git': {'head': wt.B, 'branch': 'm6-baselines', 'status_porcelain': ''}, 'bank_dir': BANK,
        'collector': {}, 'provenance': prov, 'aborted': False,
        'read_bank': {'ok': True, 'problems': [], 'files_checked': N, 'failures': 0, 'provenance_ok': True,
                      'manifest_sha256': msha},
        'listing': {'entries': sorted([lb.BANK_DIR, lb.MANIFEST, lb.MANIFEST_SHA, lb.PROVENANCE]), 'dir_mode': '0o555',
                    'bank_dir_mode': '0o555',
                    'top_files': {lb.MANIFEST: {'size': len(body), 'sha256': msha, 'mode': '0o444'},
                                  lb.MANIFEST_SHA: {'size': 85, 'sha256': 'x', 'mode': '0o444'},
                                  lb.PROVENANCE: {'size': 1, 'sha256': 'y' * 64, 'mode': '0o444'}}},
        'independent': {'files': N, 'files_verified': N, 'problem_count': 0, 'problems': [], 'modes': ['0o444'],
                        'total_png_bytes': 700000000, 'stats': stats},
        'png_verification': {'structures': {S7T.canonical(dict(S7T.E04_PNG_STRUCTURE, scanline_filters_ok=True)): N},
                             'problem_count': 0, 'problems': []},
        'manifest_inspect': {'sha256': msha, 'bytes': len(body), 'lines': N, 'ends_with_newline': True,
                             'canonical_lines': True, 'fields_ok': True, 'sha_file': '%s  bank_manifest.jsonl\n' % msha},
        's7_equivalence': {'s7_record_sha256': sha((repo / S7_REL).read_bytes()),
                           's7_manifest_sha256_recorded': s7rec['outputs'][str(S)]['manifest_sha256'],
                           'run_dirs_recorded': s7rec['outputs'][str(S)]['run_dirs'], 'match_rule': 'x', 'runs': eq_runs},
        'feed_pack': {'pairs_splits': ['TRAIN'], 'feed_pack_crosscheck': copy.deepcopy(
            s7c['lineage']['feed_pack_crosscheck'])},
        'checkpoints_after': copy.deepcopy(ck),
        'markers': {'command': command, 'command_sha256': sha(command.encode()), 'launch_time': launch, 'exit_raw': '0\n',
                    'end_time': end, 'tmux_pane': '1 0',
                    'pane': {'production_result': {'command': 'production', 'method': 'E04', 'seed': S,
                                                   'result': {'status': 'completed', 'rows': N, 'outputs': N,
                                                              'failure_count': 0, 'manifest_sha256': msha}},
                             'fail_hits': {k: 0 for k in TOOL.FAIL_PATTERNS}, 'lines': 3, 'sha256': 'z', 'tail': []},
                    'gpu_samples': {'samples': 340, 'peak_used_mib_by_pid': {'1': 7000, '4242': 600}, 'sha256': 'g'}},
        'preflight': {'phase': 'preflight', 'method': 'E04', 'seed': S, 'utc': '2099-01-01T00:05:00Z', 'repo': clone,
                      'git': {'head': wt.B, 'branch': 'm6-baselines', 'status_porcelain': ''}, 'processes': [],
                      'bank_panes': {'m8prod_E05_s2026': '1 0'}, 'gpu': gpu,
                      'memory': {'mem_available_bytes': 32 << 30, 'floor_bytes': 4 << 30}, 'storage': storage,
                      'bank_dir': BANK, 'bank_dir_exists': False, 'method_dir_exists': False, 'registry': registry,
                      'e01_e02_manifests': e0102, 'checkpoints': copy.deepcopy(ck),
                      'markers_absent': {k: True for k in TOOL.e04_marker_paths(S) if k != 'preflight'},
                      'tmux_session_absent': True,
                      'authority': {'ok': True, 'record': {'mode': 'PRODUCTION', 'methods': ['E04'], 'seeds': [S],
                                                           'records': records}},
                      'inputs': {'toy_manifest_sha256': lb.TOY_MANIFEST_SHA256,
                                 'feed_pack_manifest_sha256': TOOL.E04_PACK_MANIFEST_SHA256, 'faces_root_is_dir': True},
                      'environment': {'name': 'gpat-m6-e03', 'python': TOOL.E04_PYTHON, 'present': True, 'probe_rc': 0,
                                      'probe': probe}},
        'preflight_sha256': 'p' * 64,
        'atime': {'pre_utc': '2099-01-01T00:09:00Z', 'post_utc': '2099-01-01T00:41:00Z', 'pre_sha256': 'a',
                  'post_sha256': 'b', 'files_pre': 20615, 'files_post': 20615, 'same_paths': True,
                  'by_split': {sp: {'files': n, 'atime_changed': 0, 'mtime_changed': 0, 'missing_post': 0}
                               for sp, n in (('TRAIN', 14467), ('VAL', 3121), ('TEST', 3027))}},
        'storage_after': copy.deepcopy(storage),
        'registry_after': dict(registry, **{'E04/seed_%d' % S: {'manifest_sha256': msha, 'provenance': True,
                                                                 'aborted': False, 'mode': '0o555'}}),
        'e01_e02_manifests_after': copy.deepcopy(e0102), 'processes_after': [], 'gpu_after': gpu,
        'bank_unchanged_during_collect': True,
    }
    return c, body


class Auditor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wt = Worktree()
        cls.c, cls.body = synthetic_collected(cls.wt)
        ev = cls.wt.dir / TOOL.EVIDENCE.format(M='E04', S=42)
        ev.mkdir(parents=True, exist_ok=True)
        (ev / 'bank_manifest.jsonl.gz').write_bytes(gzip.compress(cls.body, 9, mtime=0))
        (ev / 'collected.json').write_text(json.dumps(cls.c))
        cls.ok = TOOL.build_e04(copy.deepcopy(cls.c), cls.wt.dir)

    @classmethod
    def tearDownClass(cls):
        cls.wt.remove()

    def build(self, fn=None):
        c = copy.deepcopy(self.c)
        if fn:
            fn(c)
        return TOOL.build_e04(c, self.wt.dir)

    def assertFails(self, gates, fn):
        doc = self.build(fn)
        self.assertFalse(doc['production_audit_passed'])
        for g in gates:
            self.assertIn(g, doc['failed_gates'])
        return doc

    def test_00_synthetic_bank_passes_every_gate(self):
        d = self.ok
        self.assertEqual(d['failed_gates'], [], [g for g in d['gates'] if not g['pass']])
        self.assertEqual((d['status'], d['gates_passed'], d['gates_total']), ('LEARNED_BANK_PRODUCTION_PASSED', 43, 43))
        self.assertEqual(d['fidelity'], 'CONTROLLED_ADAPTATION')
        TOOL.report_e04(d)
        names = [g['gate'] for g in d['gates']]
        for g in ('S.s7_production_equivalence', 'F.atime_ledger', 'F.no_val_test_inputs', 'S.checkpoint_after',
                  'S.no_training', 'S.population', 'P.e01_e02_banks_unchanged'):   # names the family closure reads
            self.assertIn(g, names)

    def test_01_empty_collection_never_passes(self):
        for c in ({'method': 'E04', 'seed': 42, 'git': {'head': self.wt.B}}, {'method': 'E04', 'seed': 42,
                                                                               'git': {'head': 'x'}}):
            d = TOOL.build_e04(c, self.wt.dir)
            self.assertFalse(d['production_audit_passed'])
            self.assertEqual(d['gates_total'], 43)

    def test_wrong_gpu_environment(self):
        self.assertFails(['P.resources'], lambda c: c['preflight']['gpu'].update(uuid='GPU-0000'))
        self.assertFails(['P.environment_lock'], lambda c: c['preflight']['environment']['probe'].update(
            tensorflow='2.11.0'))
        self.assertFails(['P.environment_lock'], lambda c: c['preflight']['environment']['probe'].update(
            precision_errors=['TF_DETERMINISTIC_OPS']))
        self.assertFails(['S.environment'], lambda c: c['provenance']['environment'].update(
            python_executable='/usr/bin/python3'))
        self.assertFails(['R.command'], lambda c: c['markers'].update(command=c['markers']['command'].replace(
            'TF_DETERMINISTIC_OPS=1 ', '')))
        self.assertFails(['P.resources'], lambda c: c['preflight'].update(processes=['9 python tools/learned_bank.py']))
        self.assertFails(['P.resources'], lambda c: c['preflight']['gpu'].update(free_mib=1000))

    def test_wrong_hdd_or_storage(self):
        self.assertFails(['P.storage'], lambda c: c['preflight']['storage']['filesystem'].update(filesystem_uuid='x'))
        self.assertFails(['F.storage_after'], lambda c: c['storage_after']['filesystem'].update(filesystem_uuid='x'))
        self.assertFails(['F.storage_after'], lambda c: c['storage_after'].update(unexpected_entries=['E07']))
        self.assertFails(['P.fresh_bank_dir'], lambda c: c['preflight'].update(bank_dir_exists=True))
        self.assertFails(['S.provenance_completed'], lambda c: c.update(bank_dir='/media/student20261/x/E04/seed_42'))
        self.assertFails(['F.other_banks_unchanged'], lambda c: c['registry_after'].update(
            {'E03/seed_42': dict(c['registry_after']['E03/seed_42'], manifest_sha256='0' * 64)}))

    def test_face_lineage_allowlist_and_split_crosscheck(self):
        body_rows = [json.loads(x) for x in self.body.decode().splitlines()]
        bad = copy.deepcopy(body_rows)
        bad[5]['target_face_sha256'] = '0' * 64                         # modified face hash
        self.assertIn('S.face_lineage_split_v1_m2', self.build_with_rows(bad)['failed_gates'])
        self.assertFails(['S.train_only_reader'], lambda c: c['provenance']['reader'].update(
            splits_allowlisted=['TRAIN', 'VAL']))
        self.assertFails(['S.train_only_reader'], lambda c: c['provenance']['reader'].update(reads=2 * N - 1))
        self.assertFails(['F.feed_pack_allowlist_crosscheck'], lambda c: c['feed_pack']['feed_pack_crosscheck'].update(
            samples_mismatching_count=1))
        self.assertFails(['F.feed_pack_allowlist_crosscheck'], lambda c: c['feed_pack']['feed_pack_crosscheck'].update(
            pairs_equal_pairs_train_v1=False))

    def test_val_test_access(self):
        self.assertFails(['F.no_val_test_inputs', 'S.train_only_reader'],
                         lambda c: c['provenance']['population'].update(pairs_val_opened=True))
        self.assertFails(['F.atime_ledger'], lambda c: c['atime']['by_split']['TEST'].update(atime_changed=1))
        self.assertFails(['F.atime_ledger'], lambda c: c['atime']['by_split']['VAL'].update(mtime_changed=1))
        self.assertFails(['F.atime_ledger'], lambda c: c['atime'].update(post_utc='2099-01-01T00:20:00Z'))

    def build_with_rows(self, rows):
        body = ''.join(lb.canonical_json(r) + '\n' for r in rows).encode('ascii')
        ev = self.wt.dir / TOOL.EVIDENCE.format(M='E04', S=42) / 'bank_manifest.jsonl.gz'
        old = ev.read_bytes()
        try:
            ev.write_bytes(gzip.compress(body, 9, mtime=0))
            return self.build()
        finally:
            ev.write_bytes(old)

    def test_output_population(self):
        rows = [json.loads(x) for x in self.body.decode().splitlines()]
        self.assertIn('S.manifest_committed', self.build_with_rows(rows[:-1])['failed_gates'])    # wrong count
        dup = copy.deepcopy(rows)
        dup[10] = dict(dup[9], row_index=10)                                                  # duplicate pair id
        d = self.build_with_rows(dup)
        self.assertIn('S.population', d['failed_gates'])
        swapped = copy.deepcopy(rows)
        swapped[3]['source_spoof_id'], swapped[4]['source_spoof_id'] = swapped[4]['source_spoof_id'], \
            swapped[3]['source_spoof_id']                                                     # wrong lineage
        self.assertIn('S.population', self.build_with_rows(swapped)['failed_gates'])
        nonfinite = copy.deepcopy(rows)
        nonfinite[7]['raw_output_finite'] = False
        self.assertIn('S.rows_finite_rgb_uint8', self.build_with_rows(nonfinite)['failed_gates'])
        self.assertFails(['S.independent_verification'], lambda c: c['independent'].update(files=N - 1, problem_count=1))
        self.assertFails(['S.png_structure'], lambda c: c['png_verification'].update(problem_count=1))
        self.assertFails(['S.reader_verification'], lambda c: c['read_bank'].update(ok=False, problems=['x']))
        self.assertFails(['S.png_structure'], lambda c: c['png_verification'].update(structures={'x': N}))

    def test_s7_correspondence(self):
        for k, v in (('byte_mismatch_count', 1), ('pixel_mismatch_count', 1), ('field_mismatch_count', 1),
                     ('rows_matched', 185), ('pixel_arrays_equal', 185), ('identity_mismatches', [[27, 'a', 'b']]),
                     ('manifest_sha256', '0' * 64), ('dir', '/tmp/other')):
            with self.subTest(field=k):
                self.assertFails(['S.s7_production_equivalence'], lambda c: c['s7_equivalence']['runs']['B'].update(
                    {k: v}))
        self.assertFails(['S.s7_production_equivalence'], lambda c: c['s7_equivalence']['runs']['A'].update(
            row_indices=list(range(186))))
        self.assertFails(['S.s7_production_equivalence'], lambda c: c['s7_equivalence']['runs'].pop('B'))

    def test_training_and_checkpoint_integrity(self):
        self.assertFails(['S.no_training'], lambda c: c['provenance']['adapter']['graph'].update(gradient_ops=3))
        self.assertFails(['S.no_training'], lambda c: c['provenance']['adapter']['graph'].update(
            mutating_reachable_ops=['AssignVariableOp']))
        self.assertFails(['S.inference_graph'], lambda c: c['provenance']['adapter']['graph'].update(
            fused_batch_norm_is_training=True))
        self.assertFails(['S.stored_bn_statistics'], lambda c: c['provenance']['adapter']['coverage'].update(
            restored_variables=101))
        self.assertFails(['S.checkpoint_after'], lambda c: c['checkpoints_after']['E04|42']['files']['index'].update(
            sha256='0' * 64))
        self.assertFails(['S.checkpoint_after'], lambda c: c['checkpoints_after']['E04|42']['files']['meta'].update(
            mtime_ns=1))
        self.assertFails(['S.checkpoint_before'], lambda c: c['provenance']['adapter']['checkpoint'].update(
            run_id='a42fdd2a2accaf0a'))
        self.assertFails(['S.config_contract_decisions'], lambda c: c['provenance']['adapter'].update(
            conversion_rule='round'))

    def test_partial_or_failed_execution(self):
        self.assertFails(['R.exit_zero'], lambda c: c['markers'].update(exit_raw='1\n'))
        self.assertFails(['R.result_line'], lambda c: c['markers']['pane']['fail_hits'].update(Traceback=1))
        self.assertFails(['S.provenance_completed'], lambda c: c.update(aborted=True))
        self.assertFails(['S.provenance_completed'], lambda c: c['provenance'].update(status='aborted'))
        self.assertFails(['R.no_process_left'], lambda c: c.update(processes_after=['1 python tools/learned_bank.py']))

    def test_execution_identity(self):
        self.assertFails(['A.execution_checkout'], lambda c: c['git'].update(status_porcelain=' M tools/learned_bank.py'))
        self.assertFails(['A.execution_checkout'], lambda c: c['git'].update(head=self.wt.A))
        self.assertFails(['A.execution_checkout'], lambda c: c.update(repo='/home/student20261/workdir/GPAT_TransferBench'))
        self.assertFails(['A.authority_committed_before_launch'], lambda c: c['markers'].update(
            launch_time='2000-01-01T00:00:00Z'))

    def test_historical_authority_and_audit_tampering(self):
        for rel, edit, gate in ((lb.PRODUCTION_AUTHORITY_TEMPLATE.format(method='E04'),
                                 lambda d: d['checkpoints']['42'].update(iteration=1), 'A.authority_derived'),
                                (lb.E04_S7_RECORD_PATH, lambda d: d['gates'][0].update(detail='edited'),
                                 'A.s7_record_passed_and_rebuilds'),
                                (lb.E04_STORAGE_EXTENSION_PATH, lambda d: d['activation_verification'].update(
                                    utc='2026-01-01T00:00:00Z'), 'A.storage_extension_activated')):
            with self.subTest(record=rel):
                p = self.wt.dir / rel
                old = p.read_bytes()
                try:
                    d = json.loads(old)
                    edit(d)
                    p.write_text(json.dumps(d, indent=1) + '\n')
                    d = self.build()
                    self.assertFalse(d['production_audit_passed'])
                    if gate != 'A.storage_extension_activated':   # an edited extension changes its sha: caught by the
                        self.assertIn(gate, d['failed_gates'])  # authority-binding gates instead
                finally:
                    p.write_bytes(old)

    def test_family_closure_needs_three_passed_banks(self):
        out = self.wt.dir / (TOOL.RECORD.format(M='E04', S=42) + '.json')
        try:
            out.write_text(json.dumps(self.ok, indent=1, sort_keys=True) + '\n')
            d = TOOL.build_closure(self.wt.dir, 'E04')
            self.assertEqual(d['status'], 'LEARNED_BANK_PRODUCTION_FAMILY_INCOMPLETE')
            self.assertIn('C.three_banks_passed', d['failed_gates'])
            for s in (1337, 2026):
                bad = json.loads(json.dumps(self.ok))
                bad.update(seed=s, production_audit_passed=False, failed_gates=['S.population'])
                (self.wt.dir / (TOOL.RECORD.format(M='E04', S=s) + '.json')).write_text(json.dumps(bad))
            d = TOOL.build_closure(self.wt.dir, 'E04')
            self.assertNotEqual(d['status'], 'LEARNED_BANK_PRODUCTION_FAMILY_COMPLETED_AND_AUDITED')
        finally:
            for s in lb.SEEDS:
                p = self.wt.dir / (TOOL.RECORD.format(M='E04', S=s) + '.json')
                if p.exists():
                    p.unlink()


class CollectorHelpers(unittest.TestCase):
    """The host-side E04 verification functions on a tiny real PNG bank (missing / altered / extra files)."""

    def setUp(self):
        import numpy as np
        self.t = Path(tempfile.mkdtemp(prefix='e04col_'))
        self.addCleanup(shutil.rmtree, str(self.t), True)
        (self.t / 'bank').mkdir()
        self.rows = []
        rng = np.random.RandomState(0)
        for i in range(3):
            a = rng.randint(0, 256, size=(256, 256, 3)).astype(np.uint8)
            raw = lb.encode_png(a)
            name = 'PTR%06d.png' % i
            (self.t / 'bank' / name).write_bytes(raw)
            os.chmod(str(self.t / 'bank' / name), 0o444)
            self.rows.append({'row_id': 'PTR%06d' % i, 'generation_status': 'success', 'output_path': 'bank/' + name,
                              'output_sha256': sha(raw), 'output_pixel_sha256': S7T.pixel_identity(a)})

    def test_clean_bank(self):
        iv = TOOL.independent_verification(lb, self.t, self.rows)
        pv = TOOL.png_verification_e04(S7T, self.t, self.rows)
        self.assertEqual((iv['problem_count'], pv['problem_count']), (0, 0))
        self.assertEqual(list(pv['structures'].values()), [3])

    def test_altered_missing_and_extra_png_detected(self):
        p = self.t / 'bank' / 'PTR000001.png'
        os.chmod(str(p), 0o644)
        raw = bytearray(p.read_bytes())
        raw[-20] ^= 1
        p.write_bytes(bytes(raw))
        self.assertGreater(TOOL.independent_verification(lb, self.t, self.rows)['problem_count'], 0)
        self.assertEqual(TOOL.png_verification_e04(S7T, self.t, self.rows)['problem_count'], 1)
        p.unlink()
        self.assertIn('missing 1 outputs', ' '.join(TOOL.independent_verification(lb, self.t, self.rows)['problems']))
        (self.t / 'bank' / 'extra.png').write_bytes(b'x')
        self.assertIn('unexpected entry extra.png', TOOL.independent_verification(lb, self.t, self.rows)['problems'])


if __name__ == '__main__':
    unittest.main()
