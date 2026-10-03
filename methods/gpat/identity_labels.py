"""GPAT identity class map from caller-supplied TRAIN metadata records (M7C2a IDENTITY_CLASS_ORDER, DEV-022).

The caller passes records; this module never reads TRAIN, VAL or TEST itself. Keys (dataset, source_subject) are
deduplicated, sorted lexicographically by their UTF-8 bytes and enumerated 0..K-1 (K must be 60 when strict). SiW-Mv2
rows carry no identity and are masked (label -1, valid False). No pseudo identities.
"""
from __future__ import annotations

import hashlib
import json

from methods.gpat import runtime_contract as rc

MASKED_DATASETS = ('siwmv2',)
IDENTITY_DATASETS = rc.IDENTITY_DATASETS
EXPECTED_CLASSES = rc.IDENTITY_CLASSES


def identity_keys(records) -> list:
    """(dataset, source_subject) of every identity-labelled record; masked datasets are skipped."""
    keys = []
    for r in records:
        dataset, subject = r['dataset'], r.get('source_subject')
        if dataset in MASKED_DATASETS:
            continue
        keys.append((dataset, subject))
    return keys


def build_identity_map(records, source_manifest_sha256: str, *, strict: bool = True) -> dict:
    keys = identity_keys(records)
    expected = EXPECTED_CLASSES if strict else len(set(keys))
    out = rc.identity_class_map(keys, source_manifest_sha256, expected=expected)
    out['index'] = {(c['dataset'], c['source_subject']): c['index'] for c in out['classes']}
    out['json'] = json.dumps({'classes': out['classes'], 'mapping_sha256': out['mapping_sha256'],
                              'source_manifest_sha256': source_manifest_sha256},
                             sort_keys=True, ensure_ascii=False, separators=(',', ':'))
    out['json_sha256'] = hashlib.sha256(out['json'].encode('utf-8')).hexdigest()
    return out


def labels_for(records, identity_map: dict):
    """(labels, valid) lists for one batch of records; masked rows get label -1 and valid False."""
    labels, valid = [], []
    for r in records:
        if r['dataset'] in MASKED_DATASETS:
            labels.append(-1)
            valid.append(False)
            continue
        labels.append(identity_map['index'][(r['dataset'], r['source_subject'])])
        valid.append(True)
    return labels, valid
