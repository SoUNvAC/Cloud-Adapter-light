"""User-authorized remote cleanup after third-party backup. Plan then execute.

Never loads predictions/labels/metrics. Preserves Phase65 lineage and source model.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

REPO = Path('/home/scv/Cloud-Adapter-light')
SOURCE = 'src/work_dirs/phase64_source_parent/seed64/best_mIoU_iter_4000.pth'
SOURCE_SHA = '64dd9a20288c35ab3362b8b1c1dafc216d9c44d60868475177e139d6be859dc9'
ROOTS = ('src/work_dirs', 'src/result_backups')
SUFFIXES = {'.pth', '.npz', '.npy', '.engine', '.onnx', '.zip', '.tar', '.tgz', '.partial', '.pyc'}


def selection(relative):
    path = Path(relative)
    if relative == SOURCE or path.is_relative_to('src/work_dirs/phase65a'):
        return False
    if path.is_relative_to('src/result_backups'):
        return path.suffix.lower() in {'.tar', '.zip', '.tgz', '.partial'}
    return path.is_relative_to('src/work_dirs') and path.suffix.lower() in SUFFIXES


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def safe(relative):
    raw = REPO / relative
    resolved = raw.resolve()
    if raw != resolved or not any(resolved.is_relative_to(REPO / root) for root in ROOTS):
        raise ValueError('Noncanonical/outside/symlink path rejected: ' + relative)
    return resolved


def preflight():
    if Path(__file__).resolve().parents[2] != REPO:
        raise ValueError('Cleanup runs only in specified remote repository')
    if digest(safe(SOURCE)) != SOURCE_SHA:
        raise ValueError('Critical source checkpoint hash mismatch')
    processes = subprocess.check_output(['ps', '-eo', 'pid,args'], text=True)
    for line in processes.splitlines()[1:]:
        fields = line.strip().split(None, 1)
        if len(fields) != 2:
            continue
        command = fields[1]
        executable = command.split()[0].split('/')[-1]
        if (executable.startswith('python') or executable in ('torchrun', 'accelerate')) and re.search(
                r'train\.py|torchrun|backup_phase65\.py\s+pack|receive_phase65_archive\.py', command):
            raise ValueError('Active training/backup process: ' + fields[0])


def inventory():
    rows = []
    for root in ROOTS:
        path = REPO / root
        if not path.exists():
            continue
        for directory, folders, files in os.walk(path, followlinks=False):
            for folder in folders:
                if (Path(directory) / folder).is_symlink():
                    raise ValueError('Symlink directory requires review')
            for name in files:
                candidate = Path(directory) / name
                relative = candidate.relative_to(REPO).as_posix()
                safe(relative)
                stat = candidate.stat()
                rows.append(dict(path=relative, bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                                 delete=selection(relative)))
    return rows


def output_path(path):
    path = path.resolve()
    if not path.is_relative_to(Path('/home/scv/shared/phase65_cleanup')):
        raise ValueError('Cleanup evidence must be outside result tree in shared/phase65_cleanup')
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write(path, payload):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(payload, stream, indent=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    preflight()
    plan = output_path(a.plan)
    if not a.execute:
        rows = inventory()
        report = dict(authorization='User confirmed third-party backup and requested maximal remote cleanup on 2026-10-07.',
                      roots=ROOTS, source_sha256=SOURCE_SHA, inventory=rows,
                      delete_files=sum(r['delete'] for r in rows),
                      delete_bytes=sum(r['bytes'] for r in rows if r['delete']),
                      retain_bytes=sum(r['bytes'] for r in rows if not r['delete']),
                      free_bytes_before=os.statvfs(REPO).f_bavail * os.statvfs(REPO).f_frsize)
        write(plan, report)
        print(json.dumps({k:v for k,v in report.items() if k!='inventory'}))
        return
    report = json.loads(plan.read_text())
    if report['source_sha256'] != SOURCE_SHA or list(report['roots']) != list(ROOTS):
        raise ValueError('Plan scope mismatch')
    current = {r['path']:r for r in inventory()}
    if current != {r['path']:r for r in report['inventory']}:
        raise ValueError('Result tree changed after plan; no deletion')
    deleted = []
    journal = output_path(plan.with_suffix('.deleted.jsonl'))
    with journal.open('x', encoding='utf-8') as stream:
        for row in report['inventory']:
            if not row['delete']:
                continue
            if not selection(row['path']):
                raise ValueError('Invalid deletion policy')
            target = safe(row['path'])
            stat = target.stat()
            if stat.st_size != row['bytes'] or stat.st_mtime_ns != row['mtime_ns']:
                raise ValueError('File changed before deletion')
            target.unlink()
            stream.write(json.dumps(row) + '\n')
            stream.flush()
            deleted.append(row)
    if digest(safe(SOURCE)) != SOURCE_SHA:
        raise ValueError('Retained source verification failed')
    remaining = inventory()
    summary = dict(status='completed', removed_files=len(deleted),
                   removed_bytes=sum(r['bytes'] for r in deleted),
                   retained_bytes=sum(r['bytes'] for r in remaining), source_sha256=SOURCE_SHA,
                   free_bytes_after=os.statvfs(REPO).f_bavail*os.statvfs(REPO).f_frsize,
                   training_authorized=False)
    write(output_path(plan.with_suffix('.completed.json')), summary)
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
