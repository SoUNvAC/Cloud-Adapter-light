"""Read-only Phase212 original identity/file inventory; no model loading or forward."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

ROOTS = [Path('/home/scv/Cloud-Adapter-light'), Path('/home/scv/shared')]
EXPECTED = {
    'src/work_dirs/phase64_source_parent/seed64/best_mIoU_iter_4000.pth': '64dd9a20288c35ab3362b8b1c1dafc216d9c44d60868475177e139d6be859dc9',
    'src/work_dirs/phase64_target_parent/l8/shared_parent/seed64/best_mIoU_iter_3500.pth': '2d9dfac9a8182445e52cc284cbd475f3c1ed413873305e2215565e50d3c576fe',
    'src/work_dirs/phase64_target_parent/sparcs/shared_parent/seed64/best_mIoU_iter_3000.pth': '69a49f09718a7e0a132a05ea36fb8d9150f90087233a3b1b3475a86c43b38827',
}


def allowed(p):
    q = p.resolve()
    return any(q == r or r in q.parents for r in ROOTS)


out = {'status': 'original_inventory_only', 'originals': [], 'alternate_named_files': [],
       'archive_paths': [], 'search_truncated': False, 'search_max_directory_depth': 6,
       'model_calls': 0, 'training': 0, 'reference_pixels_read': False}
for rel, expected in EXPECTED.items():
    p = ROOTS[0] / rel
    item = {'path': str(p), 'exists': p.is_file(), 'expected_sha256': expected}
    if p.is_file():
        assert allowed(p), 'Symlink target outside permitted roots'
        h = hashlib.sha256()
        with p.open('rb') as f:
            for b in iter(lambda: f.read(8*1024*1024), b''):
                h.update(b)
        item.update(bytes=p.stat().st_size, sha256=h.hexdigest(), SHA_match=h.hexdigest() == expected)
    out['originals'].append(item)
start = time.monotonic()
seen = 0
for root in ROOTS:
    for dp, ds, fs in os.walk(root, followlinks=False):
        depth = len(Path(dp).relative_to(root).parts)
        ds[:] = [d for d in ds if d not in {'.git', 'node_modules', '__pycache__'}
                 and not (Path(dp)/d).is_symlink() and depth < 6]
        for name in fs:
            seen += 1
            p = Path(dp)/name
            if not allowed(p):
                continue
            if name in {Path(k).name for k in EXPECTED}:
                out['alternate_named_files'].append({'path': str(p), 'bytes': p.stat().st_size})
            if name.endswith(('.tar', '.tar.gz', '.tgz', '.zip', '.7z')) and any(
                    token in str(p).lower() for token in ['backup', 'phase64']):
                out['archive_paths'].append({'path': str(p), 'bytes': p.stat().st_size})
        if seen > 300000 or time.monotonic()-start > 25:
            out['search_truncated'] = True
            break
    if out['search_truncated']:
        break
out['files_listed'] = seen
out['shared_top'] = sorted(p.name for p in ROOTS[1].iterdir())
out['remote_filesystem_free_bytes'] = shutil.disk_usage(ROOTS[0]).free
out['Windows_physical_disk_free_bytes'] = 'unknown; filesystem free cannot establish host free'
print(json.dumps(out))
