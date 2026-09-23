#!/usr/bin/env python3
"""Read-only validation of the M6D3b STOP_AND_REPORT audit; never builds a graph."""
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
HEAD = '374bd5dd7b56960e4ec3554dfbbd084061701cfd'
BASE = 'outputs/audit/M6D3B_E04_TRAINING_GRAPH_QUALIFICATION'
LEDGER = 'outputs/audit/EXECUTION_LEDGER.jsonl'
INDEX = 'outputs/audit/ARTIFACT_INDEX.csv'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError('Audit verification failed: ' + message)


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args])


def verify():
    audit = json.loads((ROOT / (BASE + '.json')).read_text())
    require(audit['decision'] == 'STOP_AND_REPORT', 'stop decision')
    require(audit['stop_reason'] == 'BENCHMARK_DATA_DIRECTORY_ENUMERATION', 'incident disclosure')
    require(audit['geometry_runtime'] == 'QUALIFIED', 'prior geometry qualification')
    require(audit['training_graph'] == 'NOT_YET_IMPLEMENTED', 'no graph qualification')
    require(audit['training_runtime'] == 'NOT_YET_QUALIFIED', 'no runtime qualification')
    require(audit['method_status'] == 'IMPLEMENTED_NOT_EXECUTED', 'method status')
    require(audit['fidelity_class'] == 'CONTROLLED_ADAPTATION', 'fidelity')
    require(audit['environment_lock_sha256'] is None, 'no new training lock')
    require(audit['graph_optimizer_execution']['optimizer_steps'] == 0, 'no optimizer execution')
    require(audit['data_access_incident']['benchmark_directory_enumeration'], 'metadata access acknowledged')
    require(not audit['data_access_incident']['benchmark_image_decode'], 'no image decode')
    for relative, expected in audit['preserved_inputs_sha256'].items():
        raw = (ROOT / relative).read_bytes()
        require(sha(raw) == expected and raw == git('show', HEAD + ':' + relative),
                'preserved input ' + relative)
    require(git('rev-parse', 'HEAD').decode().strip() == HEAD, 'HEAD')
    require(git('rev-list', '--left-right', '--count', 'HEAD...origin/m6-baselines').split() == [b'0', b'0'], 'divergence')
    prefix = git('show', HEAD + ':' + LEDGER)
    current = (ROOT / LEDGER).read_bytes()
    require(len(prefix.splitlines()) == 101 and current.startswith(prefix), '101-row prefix')
    require(len(current.splitlines()) == 102, 'exactly one appended row')
    row = json.loads(current[len(prefix):])
    require(row['classification'] == 'M6D3B_E04_TRAINING_GRAPH_QUALIFICATION' and
            row['decision'] == 'STOP_AND_REPORT', 'ledger records a stopped attempt')
    for relative, expected in row['file_sha256'].items():
        require(sha((ROOT / relative).read_bytes()) == expected, 'ledger hash ' + relative)
    index = (ROOT / INDEX).read_bytes()
    require(b'\r\n' in index and b'\n' not in index.replace(b'\r\n', b''), 'CRLF index')
    sys.path.insert(0, str(ROOT))
    from tools.build_artifact_index import iter_files
    rows = {r['path']: r for r in csv.DictReader(io.StringIO(index.decode()))}
    files = dict(iter_files())
    require(set(rows) == set(files), 'index coverage')
    for relative, path in files.items():
        require(rows[relative]['sha256'] == sha(path.read_bytes()) and
                int(rows[relative]['size_bytes']) == path.stat().st_size, 'index hash ' + relative)
    return {'audit_integrity': 'PASS', 'qualification_decision': 'STOP_AND_REPORT',
            'training_graph': 'NOT_YET_IMPLEMENTED', 'training_runtime': 'NOT_YET_QUALIFIED',
            'ledger_rows': 102, 'artifact_index_rows': len(rows), 'artifact_index_crlf': True,
            'scope': 'Bookkeeping validation only; no graph or runtime qualification'}


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2, sort_keys=True))
