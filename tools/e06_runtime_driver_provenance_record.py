"""E06b/E06c retrospective runtime-driver provenance record.

The DSDG scientific CLIs (tools/run_e06c.py, tools/run_e06b.py) verify the environment-lock hash and source identity
but never compare the live GPU driver with the lock, and no E06 run root or closure records it. This tool establishes
the driver actually loaded during the six completed E06 scientific runs from host history and committed same-boot
observations, and writes outputs/audit/E06_RUNTIME_DRIVER_PROVENANCE.{json,md} plus its SHA-256 list.

collect: run on the GPU host, read-only, stdlib only, nothing written:
    python3 -I -B - collect --runtime-root <rt> < tools/e06_runtime_driver_provenance_record.py > collected.json
build: run on the laptop; embeds the collected document so the record re-derives:
    python3 -I -B tools/e06_runtime_driver_provenance_record.py build --collected collected.json
"""
import argparse
import glob
import gzip
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

E06_RUNS = (('E06c', 42, '27dcaf23aac18d7a'), ('E06c', 1337, '49a179a4dca77f41'), ('E06c', 2026, '86b631403e3c4ee1'),
            ('E06b', 42, 'aa96129289d080ba'), ('E06b', 1337, '9ffda3ea3f6cc767'), ('E06b', 2026, 'ec71778b28479de5'))
AUTHORITY = {'E06c': '84809868acf472af5ba8cf6ab47165ae038e1012', 'E06b': 'd9ecbef7295fc6b41ced6a4966b49810d36a8c87'}
LOCK = ('environments/e06c.lock.json', '91416a20fef6eb4bbe550dc0ccdc703163f51d8df9168c1418f7a2de48e64e95')
LOCK_DRIVER, OBSERVED = '595.84', '580.178.04'
SWITCH_TXN = '2026-10-03  11:27:17'
NVIDIA_RE = re.compile(r'nvidia|libcuda')
# committed same-host observations of the loaded driver: (record, json path to driver string, json path to time)
OBSERVATIONS = (
    [('outputs/audit/E05_SEED%d_COMPLETION.json' % s, ('collected', 'run_start', 'gpu_driver', 'live_driver_version'),
      ('start_utc',)) for s in (42, 1337, 2026)] +
    [('outputs/audit/E03_SEED%d_COMPLETION.json' % s, ('collected', 'run_start', 'gpu_driver', 'observed_driver_version'),
      ('start_utc',)) for s in (42, 1337, 2026)])
M7C3 = ('outputs/audit/M7C3_GPAT_GPU_QUALIFICATION.json', ('environment', 'nvidia_smi'))
M7C3_HOST_FILE = 'qualification/m7c3/final/gpu_qualification.json'
# E06 qualification evidence (all before the driver change; recorded under the locked driver)
QUALIFICATION = ('outputs/audit/M6D5E_E06C_PRODUCTION_RUNNER_QUALIFICATION.json',
                 'outputs/audit/M6FD_E06B_PRODUCTION_QUALIFICATION.json')
DISCLOSURE = 'configs/amendments/e06_runtime_driver_retrospective_disclosure.yaml'
ACCEPTANCE_DRAFT = 'configs/amendments/e06_runtime_driver_acceptance.DRAFT.yaml'
DOCUMENT = 'docs/spec/amendments/GPAT_TransferBench_v1_0_E06_Retrospective_Runtime_Driver_Provenance_Addendum.md'
EVIDENCE = {'record': 'outputs/audit/E06_RUNTIME_DRIVER_PROVENANCE.json',
            'report': 'outputs/audit/E06_RUNTIME_DRIVER_PROVENANCE.md',
            'sha256': 'outputs/audit/E06_RUNTIME_DRIVER_PROVENANCE_SHA256.txt'}
STATUS_OK, STATUS_FAIL = 'E06_HISTORICAL_DRIVER_ESTABLISHED', 'E06_HISTORICAL_DRIVER_NOT_ESTABLISHED'


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        return 'ERROR: %s' % exc


# ================================================================== collect (GPU host, read-only)
def collect(runtime_root):
    rt = Path(runtime_root)
    apt = []
    for f in sorted(glob.glob('/var/log/apt/history.log*'), key=lambda p: (-int(p.split('.log.')[1].split('.')[0])
                                                                           if '.log.' in p else 0)):
        opener = gzip.open if f.endswith('.gz') else open
        with opener(f, 'rt', errors='replace') as fh:
            txn = None
            for line in fh:
                if line.startswith('Start-Date:'):
                    txn = {'file': os.path.basename(f), 'start': line.split(': ', 1)[1].strip(), 'events': []}
                elif line.startswith('Commandline:') and txn is not None:
                    txn['commandline'] = line.split(': ', 1)[1].strip()
                elif line.startswith('End-Date:') and txn is not None:
                    txn['end'] = line.split(': ', 1)[1].strip()
                    if txn['events']:
                        apt.append(txn)
                    txn = None
                else:
                    m = re.match(r'(Install|Upgrade|Remove|Purge): (.*)', line)
                    if m and txn is not None:
                        for name, ver in re.findall(r'([^ ,()]+) \(([^)]*)\)', m.group(2)):
                            if NVIDIA_RE.search(name):
                                txn['events'].append([m.group(1), name, ver])
    runs = []
    for method, seed, run_id in E06_RUNS:
        d = rt / 'runs/m6' / method / ('seed_%d' % seed)
        man = json.loads((d / 'run_manifest.json').read_text())
        first = open(d / 'metrics.jsonl').readline()
        texts = {n: (d / n).read_text() for n in ('run_manifest.json', 'run_summary.json', 'resolved_config.yaml')}
        texts['metrics.jsonl:run_start'] = first
        env = {k: man['environment'].get(k) for k in ('python_version', 'numpy_version', 'cuda_version', 'cudnn_version',
                                                       'pytorch_version', 'python_executable', 'gpu_model')}
        env['precision'] = json.loads(first)['payload'].get('precision')
        runs.append({'method_id': method, 'seed': seed, 'run_id': man['run_id'], 'git_commit': man['git_commit'],
                     'recorded_environment': env,
                     'start_utc': man['start_utc'], 'end_utc': man['end_utc'], 'completion_status': man['completion_status'],
                     'driver_strings_recorded': sorted({k for k, t in texts.items()
                                                        if re.search(r'driver|595\.84|580\.178', t)})})
    exe = '/home/student20261/miniconda3/envs/gpat-m6-e06c/bin/python'
    probe = ('import importlib.metadata as m, json, sysconfig; print(json.dumps({d.metadata["Name"]: d.version for d in '
             'm.distributions(path=sorted({sysconfig.get_path("purelib"), sysconfig.get_path("platlib")}))}, sort_keys=True))')
    live_env = {'executable': exe, 'python_sha256': sha256_bytes(Path(exe).resolve().read_bytes()),
                'packages': json.loads(run([exe, '-I', '-B', '-c', probe]) or '{}')}
    m7 = rt / M7C3_HOST_FILE
    m7doc = json.loads(m7.read_text())
    return {
        'collector': {'utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'), 'hostname': socket.gethostname(),
                      'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                      'uname_r': os.uname().release, 'python': sys.version.split()[0]},
        'live': {'proc_driver_version': Path('/proc/driver/nvidia/version').read_text().splitlines()[0],
                 'nvidia_smi': run(['nvidia-smi', '--query-gpu=name,driver_version,uuid', '--format=csv,noheader']).strip(),
                 'libcuda': sorted(os.path.basename(p) for p in glob.glob('/usr/lib/x86_64-linux-gnu/libcuda.so.*')),
                 'kernel_module_dirs_with_nvidia': sorted(p.split('/')[3] + '/' + p.split('/')[5]
                                                          for p in glob.glob('/lib/modules/*/kernel/nvidia*'))},
        'live_e06_environment': live_env,
        'wtmp_boots': run(['last', '-x', '--time-format', 'iso', 'reboot', 'shutdown']),
        'apt_nvidia_transactions': apt,
        'm7c3_host_file': {'path': M7C3_HOST_FILE, 'mtime_utc': datetime.fromtimestamp(m7.stat().st_mtime, timezone.utc)
                           .strftime('%Y-%m-%dT%H:%M:%SZ'), 'nvidia_smi': m7doc.get('environment', {}).get('nvidia_smi')},
        'e06_runs': runs}


# ================================================================== build (laptop)
def utc(s):
    return datetime.strptime(s, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)


def dig(d, path):
    for k in path:
        d = d[k]
    return d


def boots(wtmp):
    """Kernel boots from `last -x --time-format iso reboot shutdown` (newest first)."""
    out = []
    for line in wtmp.splitlines():
        m = re.match(r'reboot\s+system boot\s+(\S+)\s+(\S+)\s+-?\s*(\S+)?', line)
        if m:
            start = datetime.fromisoformat(m.group(2)).astimezone(timezone.utc)
            out.append({'kernel': m.group(1), 'start_utc': start.strftime('%Y-%m-%dT%H:%M:%SZ')})
    out.sort(key=lambda b: b['start_utc'])
    for i, b in enumerate(out):
        b['next_boot_utc'] = out[i + 1]['start_utc'] if i + 1 < len(out) else None
    return out


def boot_of(bts, t):
    cand = [b for b in bts if utc(b['start_utc']) <= t and (b['next_boot_utc'] is None or t < utc(b['next_boot_utc']))]
    return cand[0] if cand else None


def apt_time(s):
    """apt history local time (+07, the host zone recorded by wtmp ISO offsets) -> UTC."""
    return datetime.strptime(' '.join(s.split()), '%Y-%m-%d %H:%M:%S').replace(
        tzinfo=datetime.fromisoformat('2026-01-01T00:00:00+07:00').tzinfo).astimezone(timezone.utc)


def build(repo_root, col):
    repo = Path(repo_root)
    checks = {}
    lock_raw = (repo / LOCK[0]).read_bytes()
    lock_driver = json.loads(lock_raw)['identity']['gpu_driver']
    checks['lock_records_595_84_unchanged'] = {
        'pass': sha256_bytes(lock_raw) == LOCK[1] and lock_driver.split(', ')[1] == LOCK_DRIVER,
        'lock': LOCK[0], 'lock_sha256': sha256_bytes(lock_raw), 'gpu_driver': lock_driver}

    txns = col['apt_nvidia_transactions']
    sw = [t for t in txns if t['start'] == SWITCH_TXN]
    ev = sw[0]['events'] if len(sw) == 1 else []
    checks['switch_transaction_installs_580_removes_595'] = {
        'pass': len(sw) == 1 and ['Install', 'nvidia-driver-580-open:amd64', '580.178.04-0ubuntu0.24.04.1'] in ev and
        ['Remove', 'nvidia-driver-595-open:amd64', '595.84-0ubuntu0.24.04.1'] in ev and
        ['Remove', 'libnvidia-compute-595:amd64', '595.84-0ubuntu0.24.04.1'] in ev and
        ['Install', 'linux-modules-nvidia-580-open-7.0.0-38-generic:amd64', '7.0.0-38.38~24.04.4, automatic'] in ev,
        'start_local': SWITCH_TXN, 'end_local': sw[0].get('end') if sw else None,
        'end_utc': apt_time(sw[0]['end']).strftime('%Y-%m-%dT%H:%M:%SZ') if sw else None,
        'commandline': sw[0].get('commandline') if sw else None}
    later = [t for t in txns if apt_time(t['start']) > apt_time(SWITCH_TXN)]
    foreign = [(t['start'], e) for t in later for e in t['events'] if not re.search(r'-580|580\.178', e[1] + e[2])
               and re.search(r'nvidia-driver|libnvidia-compute|linux-modules-nvidia|libcuda|nvidia-utils', e[1])]
    removed580 = [(t['start'], e) for t in later for e in t['events'] if e[0] in ('Remove', 'Purge') and '580' in e[1]]
    checks['no_driver_change_after_switch'] = {
        'pass': not foreign and not removed580, 'later_nvidia_transactions': [{'start': t['start'], 'events': t['events']}
                                                                            for t in later],
        'foreign_driver_events': foreign, 'removed_580': removed580}
    mods = sorted({e[1] for t in txns for e in t['events'] if e[1].startswith('linux-modules-nvidia-') and e[0] == 'Install'})
    mod595 = [m for m in mods if '595' in m]
    checks['no_595_kernel_module_ever_installed_for_running_kernel'] = {
        'pass': mod595 != [] and not any('7.0.0-38' in m for m in mod595) and
        any('580-open-7.0.0-38' in m for m in mods),
        'linux_modules_nvidia_installed': mods}

    bts = boots(col['wtmp_boots'])
    end = checks['switch_transaction_installs_580_removes_595']['end_utc']      # no switch: no run can follow it
    switch_end = utc(end) if end else datetime.max.replace(tzinfo=timezone.utc)
    obs = []
    for path, dpath, tpath in OBSERVATIONS:
        d = json.loads((repo / path).read_text())
        obs.append({'source': path + ':' + '/'.join(dpath), 'utc': dig(d, tpath), 'driver': dig(d, dpath)})
    m7 = json.loads((repo / M7C3[0]).read_text())
    obs.append({'source': M7C3[0] + ':' + '/'.join(M7C3[1]) + ' (time: host file mtime)',
                'utc': col['m7c3_host_file']['mtime_utc'], 'driver': dig(m7, M7C3[1]).split(', ')[1]})
    for o in obs:
        b = boot_of(bts, utc(o['utc']))
        o['boot_start_utc'] = b['start_utc'] if b else None
    checks['observations_all_580'] = {'pass': all(o['driver'] == OBSERVED for o in obs) and
                                      col['m7c3_host_file']['nvidia_smi'] == dig(m7, M7C3[1]), 'observations': obs}

    per_run, run_ok = [], True
    for r in col['e06_runs']:
        s, e = utc(r['start_utc']), utc(r['end_utc'])
        b = boot_of(bts, s)
        same = b is not None and boot_of(bts, e) is b
        inboot = [o for o in obs if b and o['boot_start_utc'] == b['start_utc'] and o['driver'] == OBSERVED]
        ok = (same and b['kernel'] == '7.0.0-38-generic' and s > switch_end and inboot != [] and
              r['git_commit'] == AUTHORITY[r['method_id']] and r['completion_status'] == 'completed' and
              (r['method_id'], r['seed'], r['run_id']) in E06_RUNS and r['driver_strings_recorded'] == [])
        run_ok &= ok
        contemporaneous = [o['source'] for o in inboot if s <= utc(o['utc']) <= e]
        per_run.append(dict(r, boot_start_utc=b['start_utc'] if b else None, kernel=b['kernel'] if b else None,
                            run_within_single_boot=same, starts_after_switch=s > switch_end,
                            same_boot_observations=[o['source'] for o in inboot],
                            contemporaneous_observations=contemporaneous,
                            historical_driver=OBSERVED if ok else 'NOT_ESTABLISHED',
                            confidence='HIGH' if ok else 'NOT_ESTABLISHED', pass_=ok))
    checks['every_e06_run_bound_to_580_boot'] = {'pass': run_ok and len(per_run) == 6, 'runs': per_run}
    ident = json.loads(lock_raw)['identity']
    want = {'python_version': ident['python'].split()[0], 'numpy_version': ident['numpy'], 'cuda_version': ident['cuda'],
            'cudnn_version': ident['cudnn'], 'pytorch_version': ident['torch'], 'python_executable': ident['executable'],
            'gpu_model': lock_driver.split(', ')[0], 'precision': ident['precision']}
    def recorded(r, k):              # the run records a superset of the lock precision keys; compare the lock's keys
        v = r['recorded_environment'].get(k)
        return {p: (v or {}).get(p) for p in want[k]} if k == 'precision' else v
    diffs = {'%s/%d' % (r['method_id'], r['seed']): {k: [want[k], recorded(r, k)] for k in want if recorded(r, k) != want[k]}
             for r in col['e06_runs']}
    checks['recorded_runtime_fields_equal_lock'] = {
        'pass': not any(diffs.values()), 'compared_fields': sorted(want), 'differences': diffs,
        'not_recorded_per_run': ['packages', 'torchvision', 'pillow', 'python_sha256', 'gpu_driver']}
    le = col['live_e06_environment']
    pk = {k: [v, le['packages'].get(k)] for k, v in ident['packages'].items() if le['packages'].get(k) != v}
    checks['live_e06_environment_equals_lock_packages'] = {
        'pass': le['python_sha256'] == ident['python_sha256'] and not pk and len(le['packages']) == len(ident['packages']),
        'python_sha256': le['python_sha256'], 'package_count': len(le['packages']), 'package_differences': pk,
        'note': 'live state at collection time; supports, but does not by itself prove, the historical package state'}
    live = col['live']
    checks['live_host_580_only'] = {
        'pass': OBSERVED in live['proc_driver_version'] and OBSERVED in live['nvidia_smi'] and
        live['libcuda'] == ['libcuda.so.1', 'libcuda.so.' + OBSERVED] and col['collector']['uname_r'] == '7.0.0-38-generic',
        **live, 'uname_r': col['collector']['uname_r']}
    qual = {}
    for p in QUALIFICATION:
        t = (repo / p).read_text()
        qual[p] = {'contains_595_84': LOCK_DRIVER in t, 'contains_580_178_04': OBSERVED in t}
    checks['e06_qualification_was_under_595_84'] = {
        'pass': all(v['contains_595_84'] and not v['contains_580_178_04'] for v in qual.values()), 'records': qual}

    failed = sorted(k for k, v in checks.items() if not v['pass'])
    return {'schema_version': 'e06_runtime_driver_provenance_v1', 'status': STATUS_FAIL if failed else STATUS_OK,
            'failed_checks': failed, 'lock_driver': LOCK_DRIVER, 'historical_driver': OBSERVED if not failed else None,
            'disclosure_file': DISCLOSURE, 'acceptance_draft_file': ACCEPTANCE_DRAFT, 'document': DOCUMENT,
            'boots': bts, 'checks': checks, 'collected': col,
            'limitations': [
                'The kernel journal is not readable by the run account, so no NVRM module-load line is available; the '
                'driver is established from package history, kernel identity, library inventory and same-boot '
                'observations, not from a log line written during each E06 run.',
                'No observation was taken inside any E06 run window; the closest same-boot observations are cited per run.',
                'The E06 qualifications (M6D5e, M6FC/FD) ran under 595.84; no E06 requalification was performed under '
                '580.178.04. Bitwise runtime equivalence between the two drivers is neither claimed nor established.'],
            'erratum': ('The results audit 34d1543 (PREVIOUS_WORK_RESULTS_AUDIT) stated "boot 2026-10-03 11:28 +07"; '
                        '11:28:24 is the first user-visible journal entry. wtmp records the kernel 7.0.0-38 boot at '
                        '11:22:33 +07, before the 580 install (11:27:17-11:27:35); no 595 module existed for that kernel, '
                        'so the 580 module was loaded within that boot after the install. The conclusion (580.178.04) '
                        'is unchanged.')}


def report(doc):
    c = doc['checks']
    rows = ''.join('| `%s` | %s |\n' % (k, 'pass' if v['pass'] else 'FAIL') for k, v in c.items())
    runs = ''.join('| %s | %d | `%s` | %s → %s | %s (%s) | %s | %s |\n' % (
        r['method_id'], r['seed'], r['run_id'], r['start_utc'], r['end_utc'], r['boot_start_utc'], r['kernel'],
        len(r['same_boot_observations']), r['historical_driver'])
        for r in c['every_e06_run_bound_to_580_boot']['runs'])
    obs = ''.join('| %s | %s | %s | %s |\n' % (o['utc'], o['boot_start_utc'], o['driver'], o['source'])
                  for o in sorted(c['observations_all_580']['observations'], key=lambda o: o['utc']))
    bts = ''.join('| %s | %s |\n' % (b['start_utc'], b['kernel']) for b in doc['boots'] if b['start_utc'] >= '2026-09-01')
    sw = c['switch_transaction_installs_580_removes_595']
    return '\n'.join([
        '# E06b / E06c retrospective runtime-driver provenance', '',
        '**Status:** `%s`. Lock driver `%s`; historical driver `%s`.' % (doc['status'], doc['lock_driver'],
                                                                         doc['historical_driver']), '',
        'This is an evidence record for the retrospective disclosure in `%s` (addendum `%s`). It modifies no scientific '
        'source, lock, checkpoint or run output. Host data were collected read-only at %s (boot `%s`).'
        % (doc['disclosure_file'], doc['document'], doc['collected']['collector']['utc'],
           doc['collected']['collector']['boot_id']), '',
        '## Checks', '', '| check | result |', '|---|---|', rows.rstrip('\n'), '',
        '## Driver switch', '',
        'apt transaction `%s` (`%s`) ended %s UTC: it installed `nvidia-driver-580-open 580.178.04` and '
        '`linux-modules-nvidia-580-open-7.0.0-38-generic`, and removed `nvidia-driver-595-open 595.84`, '
        '`libnvidia-compute-595 595.84` and the 595 kernel modules for 7.0.0-28/-31. No 595 kernel module was ever '
        'installed for kernel 7.0.0-38. The only later NVIDIA transaction is a same-version 580 module repackage.'
        % (sw['start_local'], sw['commandline'], sw['end_utc']), '',
        '## Kernel boots (wtmp, since 2026-09-01)', '', '| boot start (UTC) | kernel |', '|---|---|', bts.rstrip('\n'), '',
        '## Same-host driver observations', '', '| time (UTC) | boot | driver | source |', '|---|---|---|---|',
        obs.rstrip('\n'), '',
        '## E06 runs', '', '| method | seed | run_id | window (UTC) | boot (kernel) | same-boot observations | driver |',
        '|---|---|---|---|---|---|---|', runs.rstrip('\n'), '',
        'None of the six run roots records a driver string (run_manifest, run_summary, resolved_config, run_start).', '',
        '## Limitations', '', ''.join('- %s\n' % x for x in doc['limitations']).rstrip('\n'), '',
        '## Erratum to 34d1543', '', doc['erratum'], ''])


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('collect')
    c.add_argument('--runtime-root', default='/home/student20261/workdir/GPAT_TransferBench_runtime')
    b = sub.add_parser('build')
    b.add_argument('--collected', required=True)
    b.add_argument('--repo-root', default=str(Path(__file__).resolve().parents[1]) if '__file__' in globals() else '.')
    a = ap.parse_args(argv)
    if a.cmd == 'collect':
        print(json.dumps(collect(a.runtime_root), indent=1, sort_keys=True))
        return 0
    root = Path(a.repo_root)
    doc = build(root, json.loads(Path(a.collected).read_text()))
    (root / EVIDENCE['record']).write_text(json.dumps(doc, indent=1, sort_keys=True) + '\n')
    (root / EVIDENCE['report']).write_text(report(doc))
    (root / EVIDENCE['sha256']).write_text(''.join('%s  %s\n' % (sha256_bytes((root / p).read_bytes()), p) for p in (
        EVIDENCE['record'], EVIDENCE['report'], DISCLOSURE, ACCEPTANCE_DRAFT, DOCUMENT)))
    print(doc['status'], doc['failed_checks'])
    return 0 if not doc['failed_checks'] else 1


if __name__ == '__main__':
    sys.exit(main())
