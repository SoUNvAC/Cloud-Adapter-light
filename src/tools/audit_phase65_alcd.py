"""Metadata-only inventory of ALCD masks; no TIFF pixels or author metrics read."""
import argparse
import json
import tarfile
from pathlib import Path, PurePosixPath

from download_phase65_alcd import RECORD, NAME, MD5
from download_phase65_sentinel import save
from phase65a_freeze import allowed, digest


def audit(root):
    root = allowed(root)
    status = json.loads((root / 'download_status.json').read_text())
    if status['record'] != RECORD or status['status'] != 'all_files_verified':
        raise ValueError('Release not verified')
    info = status['verified_files'][NAME]
    if info['md5'] != MD5 or info.get('gzip_crc') != 'passed' or info.get('tar_integrity') != 'passed':
        raise ValueError('Missing official or archive integrity verification')
    if digest(root / NAME) != info['sha256']:
        raise ValueError('Archive changed since verification')
    scenes = {}
    with tarfile.open(root / NAME, 'r:gz') as archive:
        for member in archive:
            parts = PurePosixPath(member.name).parts
            if not member.isfile() or 'Classification' not in parts:
                continue
            position = parts.index('Classification')
            if position == 0:
                raise ValueError('Classification without product directory')
            product = parts[position - 1]
            key = '/'.join(parts[:position])
            row = scenes.setdefault(key, dict(product_directory=product, files=[], parameters=None))
            row['files'].append(parts[-1])
            if parts[-1] == 'used_parameters.json':
                if member.size > 2 * 1024 * 1024:
                    raise ValueError('Unexpectedly large metadata JSON')
                row['parameters'] = json.load(archive.extractfile(member))
    products = {r['product_directory'] for r in scenes.values()}
    return dict(record=RECORD, status='metadata_intake_not_training_ready',
                scene_directories=len(scenes), unique_product_directories=len(products),
                duplicate_directory_products=len(scenes) - len(products), scenes=scenes,
                classes={'0': 'invalid', '1': 'unused_invalid', '2': 'low_cloud',
                         '3': 'high_cloud', '4': 'cloud_shadow', '5': 'land', '6': 'water', '7': 'snow'},
                warning='ALCD random-forest reference masks; confidence is not independent label truth or deployment feature.',
                pending=['raw_multispectral_image_availability', 'source_historical_and_catalogue_overlap',
                         'footprint_grid_and_label_validity', 'new_protocol_amendment_before_split_or_training'],
                pixels_read=False, old_split_preserved=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = allowed(args.output)
    if output.exists():
        raise FileExistsError('Preserve previous audit')
    output.parent.mkdir(parents=True, exist_ok=True)
    report = audit(args.root)
    save(output, report)
    print(json.dumps({k: v for k, v in report.items() if k != 'scenes'}, ensure_ascii=False))
