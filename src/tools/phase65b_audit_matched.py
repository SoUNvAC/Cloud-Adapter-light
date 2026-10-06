"""Check paired metadata, not a substitute for raster/label inspection or fitting.

JSON input has rgb/six lists of records; each record contains the audited
pairing keys below plus record_id. No test data is accepted.
"""
import argparse
import json
from pathlib import Path
from phase65a_freeze import allowed, digest

KEYS = ('scene', 'split', 'crs', 'transform', 'shape', 'resolution',
        'radiometry', 'footprint_sha256', 'valid_mask_sha256',
        'normalization_fit_scene_sha256')


def audit(data):
    groups = []
    for name in ('rgb', 'six'):
        records = data[name]
        mapped = {r['record_id']: r for r in records}
        if not mapped or len(mapped) != len(records):
            raise ValueError('Empty or duplicate paired records')
        groups.append(mapped)
    if groups[0].keys() != groups[1].keys():
        raise ValueError('RGB/six sample mismatch')
    for key, rgb in groups[0].items():
        six = groups[1][key]
        if rgb['split'] not in ('fit', 'development_val'):
            raise ValueError('Confirmation/test records forbidden for input development')
        for field in KEYS:
            if rgb[field] is None or rgb[field] == '' or rgb[field] != six[field]:
                raise ValueError(f'Mismatch/missing {field}: {key}')
    return {'status': 'metadata_matched_only', 'pairs': len(groups[0]),
            'training_authorized': False,
            'pending': ['pixel_registration', 'nodata_and_label_validity',
                        'train_only_normalization', 'independent_hard_case_review',
                        'tiny_sample_fit', 'matched_source_retention']}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--pairs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = audit(json.loads(allowed(a.pairs).read_text(encoding='utf-8')))
    result['input_sha256'] = digest(a.pairs)
    out = allowed(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
