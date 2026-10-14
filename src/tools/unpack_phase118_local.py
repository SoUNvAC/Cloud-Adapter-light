"""Validate shared-only delivery; keep point scores outside Git."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

R = Path(__file__).resolve().parents[2]
U = R / 'outputs/phase118/phase118_20261014'
D = R / 'data/pixbox_s2_cmix_5036991/phase118_20261014'
O = R / 'src/research_results/phase118_20261014'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('xb') as f:
        f.write(data)


def main():
    receipt = json.loads((U / 'DELIVERY_RECEIPT.json').read_text())
    for name, key in [('small_delivery.zip', 'small_archive'),
                      ('point_scores_private.zip', 'private_scores_archive')]:
        p = U / name
        assert p.stat().st_size == receipt[key + '_bytes']
        assert sha(p) == receipt[key + '_sha256']
    with zipfile.ZipFile(U / 'small_delivery.zip') as z:
        assert len(z.namelist()) == len(set(z.namelist()))
        manifest = json.loads(z.read('SMALL_DELIVERY_SHA256.json'))
        assert set(z.namelist()) == set(manifest) | {'SMALL_DELIVERY_SHA256.json'}
        for name in z.namelist():
            assert PurePosixPath(name).name == name and name.endswith('.json')
            data = z.read(name)
            if name in manifest:
                assert hashlib.sha256(data).hexdigest() == manifest[name]
            save(U / 'delivery' / name, data)
        seal = json.loads(z.read('SCORES_SEALED.json'))
    expected = {f'{key}/{name}' for key in seal['score_sha256']
                for name in ('UNLABELLED_POINT_SCORES.json', 'CPU_POINT_RECONSTRUCTION.json')}
    assert len(expected) == 64
    with zipfile.ZipFile(U / 'point_scores_private.zip') as z:
        assert len(z.namelist()) == 64 and set(z.namelist()) == expected
        for name in z.namelist():
            path = PurePosixPath(name)
            assert not path.is_absolute() and '..' not in path.parts
            data = z.read(name)
            if path.name == 'UNLABELLED_POINT_SCORES.json':
                assert hashlib.sha256(data).hexdigest() == seal['score_sha256'][str(path.parent)]
            else:
                proof = json.loads(data)
                assert proof['max_confidence_error'] <= 1e-5
                assert proof['ensemble_max_error'] <= 1e-5
                assert proof['coverage_and_point_gradient_exact']
            save(D / name, data)
    # These are metadata/aggregate receipts, never the private point score files.
    for name in manifest:
        data = (U / 'delivery' / name).read_bytes()
        save(U / name, data)
        save(O / 'remote_delivery' / name, data)
    for name in ('SMALL_DELIVERY_SHA256.json', 'DELIVERY_RECEIPT.json'):
        data = (U / 'delivery' / name).read_bytes() if name.startswith('SMALL') else (U / name).read_bytes()
        save(O / 'remote_delivery' / name, data)
    print(json.dumps({'status': 'passed', 'private_files': 64, 'small_files': len(manifest),
                      'archive_sha256_verified': True, 'work_dirs_transferred': False}))


if __name__ == '__main__':
    main()
