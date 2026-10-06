"""Verify source train/val PNG ordinals against official raw RGB arrays.

Uses the repository's exact min/max conversion. No annotations or test data are
read. Full source RGB LFS SHA checks precede every-patch comparison. Metadata is
the companion CSV in published row order, not its original `index` column.
"""
import argparse
import csv
import json
import urllib.request
from pathlib import Path
import numpy as np
from PIL import Image
from phase65a_freeze import allowed, digest
from download_phase65_sentinel import save


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--raw-root', type=Path, required=True)
    p.add_argument('--png-root', type=Path, required=True)
    p.add_argument('--metadata-root', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    a = p.parse_args()
    raw, png, metadata_root, out = map(allowed, (a.raw_root, a.png_root, a.metadata_root, a.output_root))
    out.mkdir(parents=True, exist_ok=True)
    final = allowed(out / 'mapping_audit.json')
    if final.exists():
        raise ValueError('Existing audit; inspect before rerun')
    manifest = json.loads(allowed(metadata_root / 'source_metadata_manifest.json').read_text())
    commit = manifest['commit']
    report = dict(status='running', source_commit=commit, splits={},
                  test_pixels_read=False, label_pixels_read=False)
    progress = out / 'mapping_progress.json'
    save(progress, report)
    mapping = []
    try:
        for split, count in (('train', 8490), ('val', 535)):
            metadata_path = allowed(metadata_root / f'{split}_metadata.csv')
            if digest(metadata_path) != manifest['splits'][split]['sha256']:
                raise ValueError('Companion metadata SHA mismatch')
            with metadata_path.open(newline='', encoding='utf-8') as stream:
                rows = list(csv.DictReader(stream))
            if len(rows) != count:
                raise ValueError('Unexpected metadata count')
            actual_names = {path.name for path in allowed(png / 'img_dir' / split).glob('*.png')}
            if actual_names != {f'{i}.png' for i in range(count)}:
                raise ValueError('PNG numbering/count mismatch')
            url = f'https://huggingface.co/api/datasets/csaybar/CloudSEN12-high/tree/{commit}/{split}'
            with urllib.request.urlopen(url, timeout=60) as response:
                inventory = {row['path']: row for row in json.load(response)}
            band_hashes = {}
            arrays = []
            for band in ('B4', 'B3', 'B2'):
                path = allowed(raw / split / f'L1C_{band}.dat')
                entry = inventory[f'{split}/L1C_{band}.dat']
                if path.stat().st_size != count * 512 * 512 * 2 or path.stat().st_size != entry['size']:
                    raise ValueError('Raw RGB size mismatch')
                report.update(current_split=split, operation=f'hash_{band}')
                save(progress, report)
                band_hashes[band] = digest(path)
                if band_hashes[band] != entry['lfs']['oid']:
                    raise ValueError(f'Official RGB LFS SHA mismatch: {split}/{band}')
                arrays.append(np.memmap(path, dtype=np.int16, mode='r', shape=(count, 512, 512)))
            mismatches = []
            for i, row in enumerate(rows):
                image_path = allowed(png / 'img_dir' / split / f'{i}.png')
                rgb = np.stack([array[i] for array in arrays], axis=-1).astype(np.float32)
                rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6)
                expected = (rgb * 255).astype(np.uint8)
                with Image.open(image_path) as image:
                    observed = np.asarray(image)
                matches = observed.shape == expected.shape and np.array_equal(observed, expected)
                if not matches:
                    mismatches.append(i)
                mapping.append(dict(split=split, png_ordinal=i, original_index=row['index'],
                                    product=row['s2_id'], exact_rgb_match=bool(matches),
                                    png_sha256=digest(image_path)))
                if i % 100 == 0:
                    report.update(operation='compare_png', current_split=split, compared=i + 1,
                                  mismatch_count=len(mismatches))
                    save(progress, report)
            report['splits'][split] = dict(rows=count, exact_matches=count - len(mismatches),
                                           mismatch_ordinals=mismatches, raw_band_sha256=band_hashes,
                                           metadata_sha256=digest(metadata_path))
        report.update(status='verified_rgb_ordinal_mapping' if
                      all(not v['mismatch_ordinals'] for v in report['splits'].values())
                      else 'failed_rgb_ordinal_mapping', mappings=mapping,
                      converter_sha256=digest(Path(__file__).resolve().parents[1] /
                                               'tools/convert_datasets/create_l1c_l2a.py'),
                      reader_sha256=digest(Path(__file__).resolve().parents[1] /
                                            'dataset/cloudsen12_high.py'))
        with final.open('x', encoding='utf-8') as stream:
            json.dump(report, stream, indent=2)
        save(progress, {k: v for k, v in report.items() if k != 'mappings'})
        if report['status'] != 'verified_rgb_ordinal_mapping':
            raise ValueError('Source PNG mapping not verified; inspect mismatches')
        print(json.dumps({k: v for k, v in report.items() if k != 'mappings'}))
    except Exception as error:
        report.update(status='failed', error=str(error))
        save(progress, {k: v for k, v in report.items() if k != 'mappings'})
        raise


if __name__ == '__main__':
    main()
