"""Archive ALL work_dirs artifacts without following external symlinks.

Run pack remotely, scp the tar+inventory, then verify locally. No extraction.
Use a new destination; source work_dirs must be quiescent for a consistent backup.
"""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path
from phase65a_freeze import allowed, digest


def pack(root, archive):
    root, archive = allowed(root), allowed(archive)
    if archive.is_relative_to(root):
        raise ValueError('Archive must be outside source tree')
    inventory_path = archive.with_suffix(archive.suffix + '.json')
    if archive.exists() or inventory_path.exists():
        raise ValueError('Backup destination already exists')
    entries = sorted(root.rglob('*'))
    if any(p.is_symlink() for p in entries):
        raise ValueError('Source includes symlinks; review before backing up')
    files = [p for p in entries if p.is_file()]
    if not files:
        raise ValueError('No files to back up')
    inventory = {}
    for p in files:
        allowed(p)
        if p.is_symlink() or any(a.is_symlink() for a in p.parents if a != root.parent):
            raise ValueError(f'Symlink requires explicit inventory review: {p}')
        inventory[p.relative_to(root).as_posix()] = dict(sha256=digest(p), size=p.stat().st_size)
    archive.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation; uncompressed tar avoids spending GPU-host CPU on pth compression.
    with archive.open('xb') as stream, tarfile.open(fileobj=stream, mode='w') as tar:
        for p in files:
            tar.add(p, arcname=p.relative_to(root).as_posix(), recursive=False)
    for p in files:
        if digest(p) != inventory[p.relative_to(root).as_posix()]['sha256']:
            raise ValueError('Source changed during backup; archive is NOT verified')
    if files != sorted(p for p in root.rglob('*') if p.is_file()):
        raise ValueError('Source file list changed during backup; archive is NOT verified')
    payload = dict(files=inventory, archive_sha256=digest(archive), root=str(root))
    with inventory_path.open('x', encoding='utf-8') as stream:
        json.dump(payload, stream, indent=2)
    return {'status': 'packed_requires_local_verification', 'files': len(files)}


def verify(archive):
    archive = allowed(archive)
    manifest = json.loads(allowed(archive.with_suffix(archive.suffix + '.json')).read_text())
    if digest(archive) != manifest['archive_sha256']:
        raise ValueError('Archive SHA mismatch')
    seen = set()
    with tarfile.open(archive) as tar:
        for member in tar:
            if not member.isfile() or member.name in seen or member.name not in manifest['files']:
                raise ValueError('Unexpected or duplicate archive member')
            seen.add(member.name)
            h = hashlib.sha256()
            with tar.extractfile(member) as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    h.update(chunk)
            expected = manifest['files'][member.name]
            if h.hexdigest() != expected['sha256'] or member.size != expected['size']:
                raise ValueError(f'Content mismatch: {member.name}')
    if seen != manifest['files'].keys():
        raise ValueError('Missing backup members')
    return {'status': 'verified', 'files': len(seen)}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('pack', 'verify'))
    p.add_argument('--root', type=Path)
    p.add_argument('--archive', type=Path, required=True)
    args = p.parse_args()
    if args.action == 'pack' and args.root is None:
        p.error('--root required for pack')
    print(json.dumps(pack(args.root, args.archive) if args.action == 'pack' else verify(args.archive)))
