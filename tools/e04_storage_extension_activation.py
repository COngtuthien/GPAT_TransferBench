"""E04 additive HDD storage extension: live read-only verification on the GPU host, then the activated record
configs/amendments/learned_bank_storage_location_E04.yaml derived from the committed DRAFT (never typed).

  host (GPU host, read-only, fed on stdin; library from the clean clone --repo; gpat-m6-e05 python, no GPU):
    env -i HOME=$HOME PATH=/usr/bin:/bin CUDA_VISIBLE_DEVICES= <gpat-m6-e05 python> -I -B - host --repo <clone> \
        < tools/e04_storage_extension_activation.py > outputs/audit/E04_STORAGE_EXTENSION_ACTIVATION/host_verification.json
      lb.storage_preflight() of the approved bank root (13 checks; the clone has no activated extension, so E04 is
      still unexpected and must be absent), the same preflight with E04 registered (what the activated registry will
      accept), filesystem identity of the root and of /media/ext_data, the bank-root tree (names, owners, modes, symlinks;
      nothing inside the existing banks is read), mount table line, free bytes and inodes. Creates nothing.
  derive (laptop): python3 tools/e04_storage_extension_activation.py derive --date YYYY-MM-DD
      refuses unless every prerequisite holds; writes the activated record (refuses to overwrite) and prints its sha256.

The owner approval (2026-11-25 session instruction) is limited to registering <bank_root>/E04/seed_<seed>/ on the
already approved root. Nothing is mounted, remounted, formatted, partitioned or chmod-ed; /etc/fstab is untouched; the
base storage authority configs/amendments/learned_bank_storage_location.yaml is never edited; no E04 directory is
created by this tool (the first authorized E04 production launch creates <bank_root>/E04 with mode 0700).
"""
import argparse
import hashlib
import json
import os
import stat
import sys
from pathlib import Path

DRAFT = 'configs/amendments/learned_bank_storage_location_E04.DRAFT.yaml'
ACTIVATED = 'configs/amendments/learned_bank_storage_location_E04.yaml'
BASE = 'configs/amendments/learned_bank_storage_location.yaml'
EVIDENCE = 'outputs/audit/E04_STORAGE_EXTENSION_ACTIVATION/host_verification.json'
BANK_ROOT = '/media/ext_data/student20261_gpat_m8_banks'
UUID = '608647d2-9cb0-45b0-b999-73c080ba909b'
EXISTING = ('E03', 'E05', 'E06b', 'E06c')
SEED_DIRS = ['seed_1337', 'seed_2026', 'seed_42']
FREE_FLOOR = 20 << 30
INODE_FLOOR = 100000
APPROVAL = {'owner_authorized': True, 'date': '2026-11-25',
            'scope': 'additive E04 registry entry <bank_root>/E04/seed_<seed>/ on the approved learned-bank HDD root only',
            'instruction': "GPAT-TransferBench v1.0 - E04 Physics-STD Final Production Banks, Independent Audit, and "
                           "Seven-Baseline Bank Readiness, section 7 (activate learned_bank_storage_location_E04.yaml only "
                           "after verifying the DRAFT and all filesystem prerequisites)",
            'not_authorized': ['mount, remount, format or partition', '/etc/fstab changes',
                               'editing the base storage authority', 'moving or modifying E03/E05/E06b/E06c banks',
                               'unrelated permission changes', 'creating E04 directories before the registry and '
                               'storage-verifier tests pass']}


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


# ------------------------------------------------------------------------------------------------------------ host
def tree(root):
    """Names, types, owners and modes of the bank root and its first two levels (method / seed directories); never
    descends into a bank, never follows a symlink."""
    out = []
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        st = os.lstat(p)
        out.append([name, 'symlink' if stat.S_ISLNK(st.st_mode) else 'dir' if stat.S_ISDIR(st.st_mode) else 'file',
                    st.st_uid, oct(stat.S_IMODE(st.st_mode))])
        if stat.S_ISDIR(st.st_mode) and not stat.S_ISLNK(st.st_mode):
            for sub in sorted(os.listdir(p)):
                sst = os.lstat(os.path.join(p, sub))
                out.append([name + '/' + sub, 'symlink' if stat.S_ISLNK(sst.st_mode) else 'dir'
                            if stat.S_ISDIR(sst.st_mode) else 'file', sst.st_uid, oct(stat.S_IMODE(sst.st_mode))])
    return out


def host(repo):
    import pwd
    import subprocess
    sys.path.insert(0, str(repo))
    os.chdir(str(repo))
    from methods.common import learned_bank as lb
    pf = lb.storage_preflight()
    pf_e04 = lb.storage_preflight(allowed_methods=lb.BANK_METHODS)
    vfs = os.statvfs(BANK_ROOT)
    st = os.lstat(BANK_ROOT)
    mounts = [x for x in Path('/proc/self/mountinfo').read_text().splitlines() if ' /media/ext_data ' in x]
    doc = {'phase': 'host', 'utc': subprocess.run(['date', '-u', '+%FT%TZ'], capture_output=True, text=True).stdout.strip(),
           'repo': str(repo), 'git_head': subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'], capture_output=True,
                                                         text=True).stdout.strip(),
           'git_status': subprocess.run(['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=all'],
                                        capture_output=True, text=True).stdout.strip(),
           'registered_methods_in_clone': list(lb.registered_bank_methods()),
           'preflight': pf, 'preflight_with_e04_registered': pf_e04,
           'root': {'path': BANK_ROOT, 'realpath': os.path.realpath(BANK_ROOT), 'owner': pwd.getpwuid(st.st_uid).pw_name,
                    'mode': oct(stat.S_IMODE(st.st_mode)), 'is_symlink': stat.S_ISLNK(st.st_mode),
                    'mount_point_identity': lb.filesystem_identity('/media/ext_data')},
           'capacity': {'total_bytes': vfs.f_blocks * vfs.f_frsize, 'free_bytes': vfs.f_bavail * vfs.f_frsize,
                        'free_inodes': vfs.f_favail, 'total_inodes': vfs.f_files},
           'tree': tree(BANK_ROOT), 'e04_exists': os.path.lexists(os.path.join(BANK_ROOT, 'E04')),
           'mountinfo_lines': mounts}
    print(json.dumps(doc, sort_keys=True, default=str))


# ---------------------------------------------------------------------------------------------------------- derive
def prerequisites(h, draft, base_sha):
    """Every filesystem / registry prerequisite of the activation, from the committed host verification."""
    pf, fs = h['preflight'], h['preflight']['filesystem']
    uid = {x[2] for x in h['tree']}
    return {
        'host_clone_clean': h['git_status'] == '',
        'clone_has_no_extension': h['registered_methods_in_clone'] == list(EXISTING),
        'preflight_all_checks': pf['ok'] is True and all(pf['checks'].values()) and len(pf['checks']) == 13,
        'preflight_with_e04_registered': h['preflight_with_e04_registered']['ok'] is True,
        'filesystem_uuid': fs['filesystem_uuid'] == UUID == draft['registry_extension']['filesystem_uuid'],
        'ext4': fs['fstype'] == 'ext4' == draft['registry_extension']['fstype'],
        'canonical_mount': fs['mount_point'] == '/media/ext_data' == draft['registry_extension']['mount_point'],
        'read_write': pf['checks']['mount_read_write'] is True,
        'owner_and_mode': h['root']['owner'] == 'student20261' and h['root']['mode'] == '0o700',
        'no_symlink': h['root']['is_symlink'] is False and h['root']['realpath'] == BANK_ROOT
        and all(x[1] == 'dir' for x in h['tree']),
        'bank_root': pf['bank_root'] == BANK_ROOT == draft['registry_extension']['bank_root'],
        'existing_method_directories_valid': [x[0] for x in h['tree'] if '/' not in x[0]] == list(EXISTING)
        and all(sorted(x[0].split('/')[1] for x in h['tree'] if x[0].startswith(m + '/')) == SEED_DIRS
                for m in EXISTING) and len(uid) == 1,
        'no_unexpected_contents': pf['unexpected_entries'] == [],
        'e04_absent': h['e04_exists'] is False,
        'free_bytes': h['capacity']['free_bytes'] >= FREE_FLOOR
        and h['capacity']['free_bytes'] >= 3 * draft['estimate']['per_seed_bytes'],
        'free_inodes': h['capacity']['free_inodes'] >= INODE_FLOOR,
        'draft_binds_unchanged_base_authority': draft['base_storage_authority']['sha256'] == base_sha
        and draft['base_storage_authority']['path'] == BASE,
        'draft_layout': draft['registry_extension']['layout'] == '<bank_root>/E04/seed_<seed>/'
        and draft['registry_extension']['seeds'] == [42, 1337, 2026] and draft['methods'] == ['E04'],
    }


def derive(root, date, h_raw):
    root = Path(root)
    draft_raw = (root / DRAFT).read_bytes()
    draft = json.loads(draft_raw)
    base_sha = sha256_bytes((root / BASE).read_bytes())
    h = json.loads(h_raw)
    checks = prerequisites(h, draft, base_sha)
    if not all(checks.values()):
        raise SystemExit('E04 storage extension prerequisites failed: %s' % sorted(k for k, v in checks.items() if not v))
    out = dict(draft)
    out.update({
        'status': 'OWNER_APPROVED', 'approved': True, 'date': date, 'approval': APPROVAL,
        'derived_from': {'path': DRAFT, 'sha256': sha256_bytes(draft_raw)},
        'activation_verification': {
            'ok': True, 'record': EVIDENCE, 'record_sha256': sha256_bytes(h_raw), 'utc': h['utc'],
            'checks': checks, 'preflight_checks': h['preflight']['checks'],
            'filesystem_uuid': h['preflight']['filesystem']['filesystem_uuid'],
            'mount_point': h['preflight']['filesystem']['mount_point'], 'fstype': h['preflight']['filesystem']['fstype'],
            'free_bytes': h['capacity']['free_bytes'], 'free_inodes': h['capacity']['free_inodes'],
            'e04_directory_exists': h['e04_exists'], 'unexpected_entries': h['preflight']['unexpected_entries'],
            'note': 'live read-only verification on the GPU host immediately before activation; nothing was created, '
                    'mounted or chmod-ed'}})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('host')
    a.add_argument('--repo', required=True)
    d = sub.add_parser('derive')
    d.add_argument('--date', required=True)
    d.add_argument('--repo-root', default=str(Path(globals().get('__file__', 'tools/x')).resolve().parents[1]))
    args = ap.parse_args(argv)
    if args.cmd == 'host':
        return host(args.repo)
    root = Path(args.repo_root)
    out = root / ACTIVATED
    if out.exists():
        raise SystemExit('refusing to overwrite %s' % out)
    doc = derive(root, args.date, (root / EVIDENCE).read_bytes())
    out.write_text(json.dumps(doc, indent=1, sort_keys=True) + '\n')
    print(out, sha256_bytes(out.read_bytes()))
    return 0


if __name__ == '__main__':
    sys.exit(main())
