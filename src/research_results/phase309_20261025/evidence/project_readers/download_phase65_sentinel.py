"""Download the immutable Zenodo 4172871 release, resumably, with MD5+SHA/ZIP checks.

No label inspection or extraction is performed. Both machines run this same script.
"""
import argparse
import hashlib
import json
import os
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path
from phase65a_freeze import allowed

RECORD = '4172871'
BASE = 'https://zenodo.org'
PINNED = {
    'classification_tags.csv': '6911e5a8915daf9a98638eb21ba4afd3',
    'README.pdf': '48fb6afa0195a3736d4ce122d007be36',
    'masks.zip': 'c955efe74c52d07f8e8bb02d5143e182',
    'alt_masks.zip': '0140a8b500d85cc8553ec8ba0a304bde',
    'shapefiles.zip': '3ed79b74eb84431e68f764457b1f00ac',
    'thumbnails.zip': 'ac054a7940e0680e768bdd824e0ee8af',
    'subscenes.zip': '0ad1de0ebeaff529782f456cad2e966f',
}


def save(path, payload):
    temporary = allowed(path.with_suffix(path.suffix + '.tmp'))
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(allowed(path))


def request(url, headers=None):
    return urllib.request.urlopen(urllib.request.Request(url, headers={
        'User-Agent': 'Cloud-Adapter-light-phase65/1.0', **(headers or {})}), timeout=90)


def verify(path, expected_size, expected_md5):
    path = allowed(path)
    if path.stat().st_size != expected_size:
        raise ValueError(f'Size mismatch: {path.name}')
    md5, sha = hashlib.md5(), hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            md5.update(block)
            sha.update(block)
    if md5.hexdigest() != expected_md5:
        raise ValueError(f'Official MD5 mismatch: {path.name}; keep file for diagnosis')
    info = {'bytes': expected_size, 'md5': md5.hexdigest(), 'sha256': sha.hexdigest(),
            'zip_crc': 'not_applicable'}
    if path.suffix == '.zip':
        with zipfile.ZipFile(path) as z:
            bad = z.testzip()
            if bad is not None:
                raise ValueError(f'ZIP CRC failed: {path.name}/{bad}')
            info.update(zip_crc='passed', members=len(z.infolist()))
    return info


def transfer(url, path, size, report, progress):
    partial = allowed(path.with_suffix(path.suffix + '.partial'))
    for attempt in range(8):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset == size:
            break
        if offset > size:
            raise ValueError('Partial larger than published size')
        if shutil.disk_usage(path.parent).free < size - offset + 2 * 1024 ** 3:
            raise ValueError('Insufficient space for remaining download + 2GiB reserve')
        try:
            headers = {'Range': f'bytes={offset}-'} if offset else {}
            with request(url, headers) as response:
                if offset and (response.status != 206 or not
                               response.headers.get('Content-Range', '').startswith(f'bytes {offset}-')):
                    raise ValueError('Server did not honor resume range; partial preserved')
                with partial.open('ab' if offset else 'wb') as stream:
                    last_report = 0
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        stream.write(chunk)
                        offset += len(chunk)
                        if offset > size:
                            raise ValueError('Response exceeds published size')
                        if time.monotonic() - last_report > 10:
                            report.update(current_file=path.name, current_bytes=offset,
                                          current_size=size, updated_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
                            save(progress, report)
                            last_report = time.monotonic()
            if offset != size:
                raise OSError(f'Incomplete body: {offset}/{size}')
            break
        except (OSError, TimeoutError) as error:
            report['last_network_error'] = str(error)
            save(progress, report)
            if attempt == 7:
                raise
            time.sleep(min(5 * (attempt + 1), 30))
    # Never promote an unverified payload to its final name.
    result = verify(partial, size, PINNED[path.name])
    # partial suffix is not .zip; perform ZIP CRC explicitly before promotion.
    if path.suffix == '.zip':
        with zipfile.ZipFile(partial) as z:
            bad = z.testzip()
            if bad:
                raise ValueError(f'ZIP CRC failed: {bad}')
            result.update(zip_crc='passed', members=len(z.infolist()))
    partial.rename(path)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--small-only', action='store_true')
    args = parser.parse_args()
    root = allowed(args.root)
    root.mkdir(parents=True, exist_ok=True)
    lock = allowed(root / 'DOWNLOAD_LOCK')
    lock.mkdir()  # exclusive; stale locks require inspection, never silently bypass
    (lock / 'PID').write_text(str(os.getpid()))
    progress = root / 'download_status.json'
    report = dict(record=RECORD, status='running', verified_files={}, started_at_utc=
                  time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    try:
        metadata_path = root / 'record.json'
        if args.verify_only:
            metadata = json.loads(allowed(metadata_path).read_text(encoding='utf-8'))
        else:
            with request(f'{BASE}/api/records/{RECORD}') as response:
                metadata = json.load(response)
            if str(metadata['id']) != RECORD:
                raise ValueError('Unexpected record ID')
            save(metadata_path, metadata)
        files = {f['key']: f for f in metadata['files']}
        if set(files) != set(PINNED):
            raise ValueError('Unexpected release file set')
        for name, expected_md5 in PINNED.items():
            entry = files[name]
            if entry['checksum'] != 'md5:' + expected_md5:
                raise ValueError(f'Release checksum changed: {name}')
        save(progress, report)
        for name in PINNED:
            if args.small_only and name in ('thumbnails.zip', 'subscenes.zip'):
                continue
            entry, path = files[name], allowed(root / name)
            report.update(current_file=name, status='verifying' if path.exists() else 'downloading')
            save(progress, report)
            if path.exists():
                result = verify(path, entry['size'], PINNED[name])
            elif args.verify_only:
                raise FileNotFoundError(f'Missing release file: {name}')
            else:
                result = transfer(f'{BASE}/records/{RECORD}/files/{name}?download=1',
                                  path, entry['size'], report, progress)
            report['verified_files'][name] = result
            save(progress, report)
        report.update(status='small_files_verified' if args.small_only else 'all_files_verified',
                      completed_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        save(progress, report)
        print(json.dumps(report, ensure_ascii=False))
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        save(progress, report)
        raise
    finally:
        (lock / 'PID').unlink()
        lock.rmdir()


if __name__ == '__main__':
    main()
