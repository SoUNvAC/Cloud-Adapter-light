"""Metadata-only related-product groups; no model selection or split locking.

Connect all 513 products by MGRS tile OR same acquisition date+relative orbit.
Shadow-invalid products participate only as grouping connectors, never samples.
Any source tile/source footprint/historical footprint overlap excludes the whole
related group, conservatively. No model scores, difficulty or GT proportions used.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from phase65a_freeze import allowed, digest
from audit_phase65_source_ids import key


def group_products(tags, source_tiles, source_overlap, historical_overlap):
    products = {r['scene']: r for r in tags}
    if len(products) != len(tags):
        raise ValueError('Duplicate products')
    parents = {p: p for p in products}

    def find(p):
        while parents[p] != p:
            parents[p] = parents[parents[p]]
            p = parents[p]
        return p

    def join(a, b):
        parents[find(b)] = find(a)

    owners = {}
    for product in sorted(products):
        timestamp, tile = key(product)
        orbit = product.split('_')[4]
        for criterion in (('tile', tile), ('pass', timestamp[:8], orbit)):
            if criterion in owners:
                join(product, owners[criterion])
            else:
                owners[criterion] = product
    groups = {}
    for product in products:
        groups.setdefault(find(product), []).append(product)
    result = []
    for members in groups.values():
        members = sorted(members)
        valid = [p for p in members if products[p]['shadows_marked'] == '1']
        flags = []
        if any(key(p)[1] in source_tiles for p in members):
            flags.append('source_tile_any_date')
        if set(members) & source_overlap:
            flags.append('source_footprint')
        if set(members) & historical_overlap:
            flags.append('historical_footprint')
        if not valid:
            flags.append('no_shadow_valid_product')
        result.append(dict(group_id=hashlib.sha256('|'.join(members).encode()).hexdigest(),
                           members=members, shadow_valid_products=valid,
                           exclusion_reasons=flags, eligible=not flags))
    return sorted(result, key=lambda row: row['group_id'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tags', type=Path, required=True)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--source-mapping', type=Path, required=True)
    p.add_argument('--source-spatial', type=Path, required=True)
    p.add_argument('--historical-spatial', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    reports = [json.loads(allowed(path).read_text()) for path in
               (a.source_mapping, a.source_spatial, a.historical_spatial)]
    expected = ('verified_rgb_ordinal_mapping', 'source_spatial_audited_split_not_locked',
                'historical_headers_audited_split_not_locked')
    if tuple(r['status'] for r in reports) != expected:
        raise ValueError('Required provenance audits not complete')
    if reports[1]['source_mapping_sha256'] != digest(a.source_mapping):
        raise ValueError('Source mapping dependency SHA mismatch')
    if reports[1]['target_tags_sha256'] != digest(a.tags):
        raise ValueError('Target metadata changed')
    historical_inputs = reports[2]['input_sha256'].values()
    if digest(a.source_spatial) not in historical_inputs:
        raise ValueError('Historical audit spatial dependency SHA mismatch')
    with allowed(a.tags).open(newline='', encoding='utf-8') as stream:
        tags = list(csv.DictReader(stream))
    if len(tags) != 513 or any(r['shadows_marked'] not in ('0', '1') for r in tags):
        raise ValueError('Catalogue metadata contract violation')
    valid = {r['scene'] for r in tags if r['shadows_marked'] == '1'}
    if any({r['product'] for r in report['findings']} != valid for report in reports[1:]):
        raise ValueError('Audit target product coverage mismatch')
    source_tiles = set()
    source_paths = []
    for split in ('train', 'val'):
        path = allowed(a.source_root / f'{split}_metadata.csv')
        source_paths.append(path)
        if digest(path) != reports[0]['splits'][split]['metadata_sha256']:
            raise ValueError('Source metadata differs from verified source mapping')
        with path.open(newline='', encoding='utf-8') as stream:
            source_tiles.update(key(r['s2_id'])[1] for r in csv.DictReader(stream))
    source_overlap = {r['product'] for r in reports[1]['findings'] if r['source_footprint_overlap']}
    historic_overlap = {r['product'] for r in reports[2]['findings'] if r['historical_overlap']}
    groups = group_products(tags, source_tiles, source_overlap, historic_overlap)
    report = dict(status='groups_prepared_not_split_locked', groups=groups,
                  total_groups=len(groups), eligible_groups=sum(g['eligible'] for g in groups),
                  eligible_products=sum(len(g['shadow_valid_products']) for g in groups if g['eligible']),
                  excluded_groups=sum(not g['eligible'] for g in groups),
                  grouping_rule='MGRS tile OR same acquisition date+relative orbit, transitive',
                  training_authorized=False, target_pixels_read=False,
                  input_sha256={str(allowed(path)): digest(path) for path in
                                (a.tags, a.source_mapping, a.source_spatial, a.historical_spatial, *source_paths)},
                  pending=['preregister_group_based_split_and_support', 'complete_acquisition',
                           'array_and_loader_audit'])
    out = allowed(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != 'groups'}))


if __name__ == '__main__':
    main()
