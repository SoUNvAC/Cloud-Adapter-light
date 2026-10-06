"""Metadata-only release audit; no label pixels, imagery or model scores inspected.

Full integrity must already have passed the acquisition script on this machine.
Optional other-machine status checks that both independent downloads are identical.
"""
import argparse
import csv
import json
import stat
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from download_phase65_sentinel import PINNED
from phase65a_freeze import allowed, digest


def validate_status(report):
    if report['record'] != '4172871' or report['status'] != 'all_files_verified':
        raise ValueError('Full dataset download/verification not complete')
    if set(report['verified_files']) != set(PINNED):
        raise ValueError('Missing release files')
    for name, md5 in PINNED.items():
        entry = report['verified_files'][name]
        if entry['md5'] != md5 or (name.endswith('.zip') and entry['zip_crc'] != 'passed'):
            raise ValueError(f'File not certified: {name}')


def archive_names(path, extension):
    names = []
    with zipfile.ZipFile(allowed(path)) as archive:
        for info in archive.infolist():
            name = PurePosixPath(info.filename)
            if name.is_absolute() or '..' in name.parts or '\\' in info.filename:
                raise ValueError('Unsafe ZIP member path')
            if stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError('ZIP contains symlink')
            if not info.is_dir() and name.suffix == extension:
                names.append(name.stem)
    if len(names) != len(set(names)):
        raise ValueError('Duplicate product payloads')
    return set(names)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--other-status', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = allowed(args.root)
    status_path = allowed(root / 'download_status.json')
    status = json.loads(status_path.read_text(encoding='utf-8'))
    validate_status(status)
    for name, info in status['verified_files'].items():
        if allowed(root / name).stat().st_size != info['bytes']:
            raise ValueError(f'Size changed after acquisition: {name}')
    if args.other_status:
        other = json.loads(allowed(args.other_status).read_text(encoding='utf-8'))
        validate_status(other)
        for name in PINNED:
            for key in ('bytes', 'md5', 'sha256', 'zip_crc'):
                if status['verified_files'][name][key] != other['verified_files'][name][key]:
                    raise ValueError(f'Two-machine mismatch: {name}/{key}')
    with allowed(root / 'classification_tags.csv').open(newline='', encoding='utf-8') as stream:
        tags = list(csv.DictReader(stream))
    products = {row['scene'] for row in tags}
    if len(tags) != 513 or len(products) != 513:
        raise ValueError('Unexpected catalogue product count')
    if set(row['shadows_marked'] for row in tags) != {'0', '1'}:
        raise ValueError('Unknown shadow validity flag')
    for name in ('masks.zip', 'subscenes.zip'):
        if archive_names(root / name, '.npy') != products:
            raise ValueError(f'Catalogue/ZIP product mismatch: {name}')
    tiles = {r['scene'].split('_')[5] for r in tags if r['shadows_marked'] == '1'}
    report = dict(status='release_verified_not_training_ready', record='4172871',
                  products=len(products), shadows_marked=dict(Counter(r['shadows_marked'] for r in tags)),
                  shadow_valid_tile_count=len(tiles),
                  annotation_groups=dict(Counter(r['dataset'] for r in tags)),
                  independent_two_machine_match=bool(args.other_status),
                  metadata_sha256=digest(root / 'classification_tags.csv'),
                  acquisition_report_sha256=digest(status_path),
                  pending=['CloudSEN12_source_and_historical_overlap_audit',
                           'scene_group_split_lock', 'array_headers_and_valid_mask_audit',
                           'frozen_loader_and_training_config'])
    out = allowed(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
