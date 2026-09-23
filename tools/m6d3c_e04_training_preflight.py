#!/usr/bin/env python3
"""Validate the M6D3c evidence STOP; cannot construct or qualify a TF graph.

Default is read-only. --rebuild-index refreshes only the four explicitly named
new artifacts, carrying all other committed index rows forward without opening
their targets. In particular, manifests and runtime data are never inspected.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
HEAD = '28f17846a67279d9b196ef0e0792a998104327f1'
BASE = 'outputs/audit/M6D3C_E04_TRAINING_GRAPH_QUALIFICATION'
LEDGER = 'outputs/audit/EXECUTION_LEDGER.jsonl'
INDEX = 'outputs/audit/ARTIFACT_INDEX.csv'
ADDENDUM = ('docs/spec/amendments/'
            'GPAT_TransferBench_v1_0_E04_Training_Graph_Resolution_Addendum.md')
ARTIFACTS = (ADDENDUM, BASE + '.md', BASE + '.json',
             'tools/m6d3c_e04_training_preflight.py')
PREFIX_SHA = '86614988e10545656dc6aeff36817fe239dd9b0e4685823119cc7e8455d4101b'
PROVENANCE = {'PAPER', 'OFFICIAL_PREDECESSOR', 'FROZEN_BENCHMARK_CONTRACT',
              'CONTROLLED_RECONSTRUCTION', 'ENGINEERING_COMPATIBILITY'}


def require(ok, reason):
    if not ok:
        raise RuntimeError('M6D3c audit verification failed: ' + reason)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args])


def read(relative):
    # No path discovery, target stat, symlink resolution or data fallback.
    parts = Path(relative).parts
    require(not Path(relative).is_absolute() and '..' not in parts,
            'relative repository evidence path required')
    require(parts[0] in {'configs', 'docs', 'methods', 'outputs', 'tools',
                         'tests', 'environments', 'frozen_config_snapshot'},
            'non-evidence root')
    require(Path(relative).suffix.lower() not in
            {'.parquet', '.jpg', '.jpeg', '.png', '.bmp', '.webp', '.avi', '.mp4'},
            'data/image extension')
    return (ROOT / relative).read_bytes()


def expected_index():
    # The ordinary builder uses --others and opens manifest payloads; it must
    # not be called for this milestone. Reuse committed opaque metadata.
    baseline = git('show', HEAD + ':' + INDEX)
    reader = csv.DictReader(io.StringIO(baseline.decode()))
    require(reader.fieldnames == ['path', 'size_bytes', 'sha256'], 'index schema')
    rows = list(reader)
    require(len({r['path'] for r in rows}) == len(rows), 'unique index paths')
    by_path = {r['path']: r for r in rows}
    for relative in ARTIFACTS:
        require(relative not in by_path, 'new artifact already in baseline index')
        raw = read(relative)
        by_path[relative] = {'path': relative, 'size_bytes': str(len(raw)),
                             'sha256': sha(raw)}
    out = io.StringIO(newline='')
    writer = csv.DictWriter(out, fieldnames=reader.fieldnames, lineterminator='\r\n')
    writer.writeheader()
    writer.writerows(by_path[k] for k in sorted(by_path))
    return out.getvalue().encode(), len(rows), len(by_path)


def verify(check_index=True):
    require(git('rev-parse', 'HEAD').decode().strip() == HEAD, 'starting HEAD unchanged')
    audit = json.loads(read(BASE + '.json'))
    require(audit['decision'] == 'STOP_AND_REPORT', 'stop decision')
    require(audit['stop_reason'] == 'UNRESOLVED_SCIENTIFIC_GRAPH_CHOICES', 'stop reason')
    require(audit['geometry_runtime'] == 'E04_GEOMETRY_RUNTIME_QUALIFIED', 'geometry retained')
    require(audit['training_graph'] == 'NOT_YET_IMPLEMENTED', 'graph not qualified')
    require(audit['training_runtime'] == 'NOT_YET_QUALIFIED', 'runtime not qualified')
    require(audit['method_status'] == 'IMPLEMENTED_NOT_EXECUTED', 'method status')
    require(audit['fidelity_class'] == 'CONTROLLED_ADAPTATION', 'fidelity')
    execution = audit['graph_optimizer_execution']
    require(execution['optimizer_steps'] == 0 and not execution['graph_constructed'],
            'no graph or optimizer execution')
    resolution = audit['training_graph_evidence_resolution']
    require(not resolution['complete_authoritative_topology_resolved'] and
            not resolution['executable_graph_contract_frozen'], 'no executable freeze')
    require(len(resolution['source_resolution_table']) >= 26, 'component inventory')
    fields = {'component', 'paper_equation_figure_section', 'exact_published_behavior',
              'stdn_predecessor_evidence', 'current_frozen_e04_contract',
              'implementation_resolution', 'provenance_class'}
    for item in resolution['source_resolution_table']:
        require(fields <= set(item), 'evidence table fields')
        require(set(item['provenance_class']) <= PROVENANCE, 'provenance vocabulary')
    for relative, digest in audit['preserved_inputs_sha256'].items():
        raw = read(relative)
        require(sha(raw) == digest and raw == git('show', HEAD + ':' + relative),
                'preserved input ' + relative)
    require(sha(read(ADDENDUM)) == audit['addendum_sha256'], 'addendum identity')
    require(sha(read(ARTIFACTS[-1])) == audit['preflight_tool_sha256'], 'preflight identity')
    prefix = git('show', HEAD + ':' + LEDGER)
    current = read(LEDGER)
    require(len(prefix) == 262935 and len(prefix.splitlines()) == 102 and
            sha(prefix) == PREFIX_SHA, 'committed 102-row prefix')
    require(current.startswith(prefix) and len(current.splitlines()) == 103,
            'exactly one append and byte-identical prefix')
    row = json.loads(current[len(prefix):])
    require(row['classification'] == 'M6D3C_E04_TRAINING_GRAPH_QUALIFICATION' and
            row['decision'] == 'STOP_AND_REPORT', 'ledger classification')
    require(set(row['file_sha256']) == set(ARTIFACTS), 'ledger artifact coverage')
    for relative, digest in row['file_sha256'].items():
        require(sha(read(relative)) == digest, 'ledger artifact hash ' + relative)
    expected, carried, total = expected_index()
    if check_index:
        require(read(INDEX) == expected, 'index refreshed exactly with CRLF')
    return {'audit_integrity': 'PASS', 'qualification_decision': 'STOP_AND_REPORT',
            'graph_qualified': False, 'training_runtime_qualified': False,
            'optimizer_steps': 0, 'ledger_rows': 103,
            'first_102_rows_byte_identical': True,
            'index_rows_carried_without_target_access': carried,
            'index_rows': total, 'index_checked': check_index,
            'scope': 'Audit integrity only; scientific qualification was not executed.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rebuild-index', action='store_true')
    args = parser.parse_args()
    if args.rebuild_index:
        verify(check_index=False)
        expected, _, _ = expected_index()
        # LAST repository write. Only read-only verification follows.
        (ROOT / INDEX).write_bytes(expected)
    print(json.dumps(verify(), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
