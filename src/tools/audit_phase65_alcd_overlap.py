"""Provisional IDs/tile audit only; no labels, metrics, or image pixels opened."""
import argparse
import csv
import json
import tarfile
from collections import Counter
from pathlib import Path

from download_phase65_alcd import NAME
from download_phase65_sentinel import save
from phase65a_freeze import allowed, digest


def records(path):
    with allowed(path).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def audit(root, intake, source_root, tags):
    report = json.loads(allowed(intake).read_text())
    if report['status'] != 'metadata_intake_not_training_ready':
        raise ValueError('Missing verified intake')
    root, source_root = allowed(root), allowed(source_root)
    manifest = json.loads((source_root / 'source_metadata_manifest.json').read_text())
    source = []
    for split in ('train', 'val'):
        path = source_root / f'{split}_metadata.csv'
        if digest(path) != manifest['splits'][split]['sha256']:
            raise ValueError('Source CSV checksum mismatch')
        source.extend(row['s2_id'] for row in records(path))
    catalogue = [row['scene'] for row in records(tags)]
    source_tiles = {p.split('_')[5].removeprefix('T') for p in source}
    catalogue_tiles = {p.split('_')[5].removeprefix('T') for p in catalogue}
    candidates = []
    for directory, row in report['scenes'].items():
        metadata = row['parameters']
        if metadata is None:
            raise ValueError('Missing acquisition metadata')
        candidates.append(dict(directory=directory, product=metadata['cloudy_product_name'],
                               tile=metadata['tile'], cloudy_date=metadata['cloudy_date'],
                               source_exact_product_overlap=metadata['cloudy_product_name'] in source,
                               source_tile_overlap_any_date=metadata['tile'] in source_tiles,
                               catalogue_exact_product_overlap=metadata['cloudy_product_name'] in catalogue,
                               catalogue_tile_overlap_any_date=metadata['tile'] in catalogue_tiles))
    with tarfile.open(root / NAME, 'r:gz') as archive:
        names = [m.name for m in archive if m.isfile()]
    # Only these two TIFF names exist: classification/confidence are not spectral inputs.
    tiffs = [n for n in names if Path(n).suffix.lower() in ('.tif', '.tiff')]
    raw_candidates = [n for n in names if Path(n).suffix.lower() in ('.jp2', '.npy')]
    raw_candidates += [n for n in tiffs if Path(n).name not in ('classification_map.tif', 'confidence_enhanced.tif')]
    products = {r['product'] for r in candidates}
    tiles = {r['tile'] for r in candidates}
    return dict(status='provisional_overlap_missing_multispectral_inputs',
                archive_files=len(names), archive_suffix_counts=dict(Counter(Path(n).suffix for n in names)),
                scene_directories=len(candidates), unique_cloudy_products=len(products), unique_tiles=len(tiles),
                source_overlap_tiles=sorted(tiles & source_tiles),
                catalogue_overlap_tiles=sorted(tiles & catalogue_tiles),
                overlap_directory_counts={k: sum(r[k] for r in candidates) for k in
                    ('source_exact_product_overlap', 'source_tile_overlap_any_date',
                     'catalogue_exact_product_overlap', 'catalogue_tile_overlap_any_date')},
                raw_multispectral_archive_candidates=raw_candidates, candidates=candidates,
                training_authorized=False, pixels_read=False,
                pending=['raw_image_acquisition', 'canonical_acquisition_ids', 'historical_and_source_footprints',
                         'valid_labels_and_shadow_support', 'new_frozen_research_protocol'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('root', 'intake', 'source-root', 'tags', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    output = allowed(args.output)
    if output.exists():
        raise FileExistsError('Preserve prior audit')
    result = audit(args.root, args.intake, args.source_root, args.tags)
    save(output, result)
    print(json.dumps({k: v for k, v in result.items() if k != 'candidates'}))
