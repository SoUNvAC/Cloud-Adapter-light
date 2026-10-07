"""Read-only size/cache/checkpoint audit. Never open label/prediction/metric data."""
import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from phase65a_freeze import allowed


def audit(root):
    root = allowed(root)
    entries, links = [], []
    totals, suffixes = defaultdict(int), defaultdict(int)
    for directory, folders, files in os.walk(root, followlinks=False):
        for name in folders[:]:
            path = Path(directory) / name
            if path.is_symlink():
                links.append(path.relative_to(root).as_posix())
                folders.remove(name)
        for name in files:
            path = Path(directory) / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                links.append(relative)
                continue
            stat = path.stat()
            row = dict(path=relative, bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
            entries.append(row)
            totals[relative.split('/')[0]] += stat.st_size
            suffixes[path.suffix.lower() or '<none>'] += stat.st_size
    # Hash only model checkpoints with equal sizes. Do not inspect predictions,
    # masks, TIFF/NPY arrays, metrics, or sealed-test report contents.
    sizes = defaultdict(list)
    for row in entries:
        if Path(row['path']).suffix == '.pth':
            sizes[row['bytes']].append(row)
    duplicates = []
    for candidates in sizes.values():
        if len(candidates) < 2:
            continue
        hashes = defaultdict(list)
        for row in candidates:
            path = root / row['path']
            h = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    h.update(block)
            if path.stat().st_size != row['bytes'] or path.stat().st_mtime_ns != row['mtime_ns']:
                raise ValueError('File changed during checkpoint hash audit')
            hashes[h.hexdigest()].append(row)
        for sha, rows in hashes.items():
            if len(rows) > 1:
                duplicates.append(dict(sha256=sha, bytes_per_file=rows[0]['bytes'],
                                       redundant_bytes=(len(rows)-1)*rows[0]['bytes'],
                                       paths=[r['path'] for r in rows]))
    temporary = [r for r in entries if '__pycache__' in Path(r['path']).parts or
                 Path(r['path']).suffix.lower() in ('.pyc', '.tmp', '.partial', '.bak') or
                 Path(r['path']).name.endswith('~')]
    caches = [r for r in entries if any('cache' in part.lower() for part in Path(r['path']).parts)]
    return dict(status='read_only_candidates_not_deletion_authorization', root=str(root),
                total_bytes=sum(r['bytes'] for r in entries), file_count=len(entries),
                directories=sorted(totals.items(), key=lambda v: -v[1]),
                suffix_bytes=sorted(suffixes.items(), key=lambda v: -v[1]),
                largest_files=sorted(entries, key=lambda r: -r['bytes'])[:30],
                temporary_candidates=temporary, temporary_bytes=sum(r['bytes'] for r in temporary),
                empty_files=[r['path'] for r in entries if r['bytes']==0],
                cache_path_bytes=sum(r['bytes'] for r in caches),
                cache_directories=sorted({str(Path(r['path']).parent) for r in caches}),
                duplicate_checkpoints=duplicates,
                duplicate_checkpoint_redundant_bytes=sum(d['redundant_bytes'] for d in duplicates),
                symlinks=links, inventory=entries,
                caveat='Cache names alone do not prove rebuildability. Duplicate checkpoints may be referenced by separate runs. Preserve manifests/logs and required source checkpoint. No deletion/move/packing performed.')


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args=p.parse_args()
    out=allowed(args.output)
    if out.exists():
        raise FileExistsError('Preserve previous audit')
    out.parent.mkdir(parents=True, exist_ok=True)
    report=audit(args.root)
    with out.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({k:v for k,v in report.items() if k not in
                      ('inventory','empty_files','cache_directories','temporary_candidates','largest_files')}, ensure_ascii=False))
