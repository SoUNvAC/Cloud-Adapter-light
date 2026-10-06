"""Acquire ONLY CloudSEN12-high train/val metadata at an immutable HF commit.

Never downloads source test metadata, images, labels or metrics. Verifies LFS
SHA256 (or Git blob SHA1) from the commit-specific published file inventory.
"""
import argparse
import hashlib
import json
import urllib.request
from pathlib import Path
from phase65a_freeze import allowed

REPO = 'csaybar/CloudSEN12-high'


def load_json(url):
    with urllib.request.urlopen(url, timeout=60) as stream:
        return json.load(stream)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output-root', type=Path, required=True)
    a = p.parse_args()
    root = allowed(a.output_root)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = allowed(root / 'source_metadata_manifest.json')
    if manifest_path.exists():
        raise ValueError('Existing metadata acquisition; verify manifest instead of overwriting')
    sha = load_json(f'https://huggingface.co/api/datasets/{REPO}')['sha']
    records = {}
    for split in ('train', 'val'):
        inventory = load_json(f'https://huggingface.co/api/datasets/{REPO}/tree/{sha}/{split}')
        item = next(row for row in inventory if row['path'] == f'{split}/metadata.csv')
        url = f'https://huggingface.co/datasets/{REPO}/resolve/{sha}/{split}/metadata.csv'
        path = allowed(root / f'{split}_metadata.csv')
        with urllib.request.urlopen(url, timeout=60) as stream:
            content = stream.read()
        if len(content) != item['size']:
            raise ValueError('Metadata size mismatch')
        sha256 = hashlib.sha256(content).hexdigest()
        if item.get('lfs'):
            if sha256 != item['lfs']['oid']:
                raise ValueError('Metadata LFS SHA256 mismatch')
        else:
            git_blob_sha = hashlib.sha1(f'blob {len(content)}\0'.encode() + content).hexdigest()
            if git_blob_sha != item['oid']:
                raise ValueError('Metadata Git blob SHA1 mismatch')
        with path.open('xb') as stream:
            stream.write(content)
        records[split] = dict(url=url, bytes=len(content), sha256=sha256,
                              published_inventory=item)
    manifest = dict(repository=REPO, commit=sha, splits=records,
                    status='published_train_val_metadata_verified', test_accessed=False,
                    local_png_mapping_verified=False)
    with manifest_path.open('x', encoding='utf-8') as stream:
        json.dump(manifest, stream, indent=2)
    print(json.dumps(manifest))


if __name__ == '__main__':
    main()
