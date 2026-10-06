"""Provisional source/target product-time-tile audit, never pixel inspection.

This does not certify local PNG mapping or footprint independence. It preserves
all target records; exclusions/split locking require the complete provenance audit.
"""
import argparse
import csv
import json
import re
from pathlib import Path
from phase65a_freeze import allowed, digest


def records(path):
    with allowed(path).open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def key(product):
    match = re.fullmatch(r'S2[AB]_MSIL1C_(\d{8}T\d{6})_N\d{4}_R\d{3}_(T[A-Z0-9]{5})_\d{8}T\d{6}', product)
    if not match:
        raise ValueError(f'Unrecognized product ID: {product}')
    return match.group(1), match.group(2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--catalogue-tags', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    root = allowed(a.source_root)
    manifest = json.loads(allowed(root / 'source_metadata_manifest.json').read_text())
    source = []
    for split, count in (('train', 8490), ('val', 535)):
        path = allowed(root / f'{split}_metadata.csv')
        if digest(path) != manifest['splits'][split]['sha256']:
            raise ValueError('Source metadata hash mismatch')
        rows = records(path)
        if len(rows) != count:
            raise ValueError('Source metadata row count mismatch')
        source.extend(row['s2_id'] for row in rows)
    tags = records(a.catalogue_tags)
    target = [r['scene'] for r in tags if r['shadows_marked'] == '1']
    source_keys = {key(product) for product in source}
    source_tiles = {tile for _, tile in source_keys}
    detail = [{'product': product, 'tile': key(product)[1],
               'exact_product_overlap': product in set(source),
               'tile_acquisition_overlap': key(product) in source_keys,
               'source_tile_overlap_any_date': key(product)[1] in source_tiles}
              for product in target]
    report = dict(status='provisional_ids_only_not_training_ready',
                  source_rows=len(source), source_unique_products=len(set(source)),
                  candidate_products=len(target),
                  overlap_counts={name: sum(row[name] for row in detail) for name in
                                  ('exact_product_overlap', 'tile_acquisition_overlap',
                                   'source_tile_overlap_any_date')}, candidates=detail,
                  source_manifest_sha256=digest(root / 'source_metadata_manifest.json'),
                  target_tags_sha256=digest(a.catalogue_tags),
                  pending=['local_PNG_to_metadata_mapping', 'spatial_footprint_overlap',
                           'historical_scene_lineage', 'independent_group_split'])
    out = allowed(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != 'candidates'}))


if __name__ == '__main__':
    main()
