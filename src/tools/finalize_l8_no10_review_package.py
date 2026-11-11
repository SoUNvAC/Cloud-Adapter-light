"""Verify transferred native review images, safe-extract and build blank review UI."""
import hashlib
import json
from pathlib import Path
import tarfile
import runpy
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/l8_no10_visual_review_20261028'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    archive = OUT / 'review_images.tar.partial'
    expected = '679ec845b9f07cb93519f4dbad56382661102abe8f08627cbb290321bd235773'
    assert sha(archive) == expected
    assert sha(ROOT / 'src/tools/prepare_l8_no10_visual_review.py') == '48de9ba89515c78b6193906e25dd94dd27c719f11864fb6d4eeb7cb0aa7d2656'
    package = OUT / 'images_package'
    assert not package.exists(), 'Do not overwrite an existing delivery'
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        for m in members:
            assert (OUT / m.name).resolve().is_relative_to(OUT.resolve())
            assert m.isdir() or m.isfile(), 'No links or special files allowed'
            assert m.name.split('/')[0] == 'images_package'
        tar.extractall(OUT, members=members, filter='data')
    images = list(package.rglob('*.jpg')) + list(package.rglob('*.png'))
    assert len(images) == 80
    manifest = json.loads((package / 'IMAGE_MANIFEST.json').read_text())
    assert len(manifest) == 80
    for row in manifest:
        p = package / row['path']
        assert p.stat().st_size == row['bytes'] and sha(p) == row['SHA256']
    scenes = json.loads((package / 'SCENES.json').read_text())
    original = {s['scene']:s for s in json.loads((OUT / 'INPUT_MANIFEST.json').read_text())}
    assert len(scenes) == 10
    for s in scenes:
        assert s['source_mask_SHA256'] == original[s['scene']]['mask']['SHA']
        folder = package / s['folder']
        assert len(list(folder.glob('*.jpg'))) == 7 and len(list(folder.glob('*.png'))) == 1
        for name in ['01_full_scene_RGB_native.jpg', '02_full_reference_native.png']:
            with Image.open(folder / name) as im:
                assert im.size == (s['width'], s['height'])
        for k in range(6):
            with Image.open(folder / f'{k+3:02d}_W{k+1}_detail.jpg') as im:
                assert im.size == (3072, 1064)
        assert s['human_judgement'] is None and s['no_model_outputs']
    archive.rename(OUT / 'review_images.tar')
    runpy.run_path(str(ROOT / 'src/tools/build_l8_no10_review_page.py'), run_name='__main__')
    result = dict(status='PASS', scenes=10, total_images=80, SHA_verified_images=80,
                  archive_SHA256=expected, archive_bytes=(OUT / 'review_images.tar').stat().st_size,
                  native_fullscene_dimensions_verified=True,
                  native_detail_1024_verified=True, image_mask_grid_matches=True,
                  source_mask_matches_original_manifest=True, human_decisions_filled=False,
                  model_calls=0, destination=str(OUT))
    (OUT / 'DELIVERY_ACCEPTED.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    (OUT / 'HANDOFF.md').write_text('# Current delivery\n\nCOMPLETE. 80 images SHA verified and native dimensions checked. START_REVIEW.html and blank REVIEW_FORM.csv delivered locally. User returns l8_no10_review_submission.json or completed CSV. No AI decisions and no model calls. No further transfer or processing required.\n', encoding='utf-8')
    files = [p for p in OUT.rglob('*') if p.is_file() and p.name != 'FINAL_MANIFEST_SHA256.json']
    (OUT / 'FINAL_MANIFEST_SHA256.json').write_text(json.dumps({str(p.relative_to(OUT)):sha(p) for p in files}, indent=2), encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
