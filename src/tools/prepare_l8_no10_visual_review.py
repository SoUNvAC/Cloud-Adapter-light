"""CPU-only display of ten already-used L8 no scenes, exactly 80 image files.

Whole-scene native RGB + reference map + six native1024 detail panels per scene.
Three coverage windows and three image/reference-only cloud-edge-dark windows.
No model inference or automatic semantic judgement.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
import rasterio
from PIL import Image, ImageDraw


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    assert a.out.resolve().is_relative_to(Path('/home/scv/shared'))
    a.out.mkdir(parents=True, exist_ok=False)
    items = json.loads(a.manifest.read_text(encoding='utf-8-sig'))
    assert len(items) == 10
    records = []
    palette = np.zeros((256, 3), np.uint8)
    palette[0] = [35, 35, 35]
    palette[128] = [110, 155, 100]
    palette[192] = [255, 200, 60]
    palette[255] = [245, 245, 245]
    palette[64] = [90, 90, 230]
    for n, item in enumerate(items, 1):
        scene = item['scene']
        mask_path = Path(item['mask']['path'])
        image_path = mask_path.with_name(scene + '.TIF')
        assert mask_path.resolve().is_relative_to(Path('/home/scv/shared'))
        assert sha(mask_path) == item['mask']['SHA']
        folder = a.out / f'{n:02d}_{scene}'
        folder.mkdir()
        with rasterio.open(mask_path) as d:
            lab = d.read(1)
            shape, transform, crs = d.shape, d.transform, d.crs
        assert not np.any(lab == 64)
        assert set(np.unique(lab)) <= {0, 64, 128, 192, 255}
        with rasterio.open(image_path) as d:
            assert d.shape == shape and d.transform == transform and d.crs == crs
            assert d.count == 11 and d.dtypes[0] == 'uint8'
            assert [d.colorinterp[k - 1].name for k in [4, 3, 2]] == ['red', 'green', 'blue']
            rgb = d.read([4, 3, 2]).transpose(1, 2, 0)
        h, w = lab.shape
        valid = lab != 0
        small = cv2.resize(rgb, (max(1, w // 8), max(1, h // 8)), interpolation=cv2.INTER_AREA)
        smask = cv2.resize(lab, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
        lum = small.astype(np.float32).mean(2)
        sv = smask != 0
        cloud = (smask == 192) | (smask == 255)
        near = cv2.dilate(cloud.astype(np.uint8), np.ones((41, 41), np.uint8)) > 0
        dark_cut = float(np.percentile(lum[sv], 35))
        candidate = sv & ~cloud & near & (lum < dark_cut)
        score = cv2.boxFilter(candidate.astype(np.float32), -1, (128, 128))
        support = cv2.boxFilter(sv.astype(np.float32), -1, (128, 128))
        score[support < .6] = -1
        positions = []
        vy, vx = np.where(sv)
        # Fixed spatial coverage, moved only to a nearby valid coarse pixel.
        for fy, fx in [(0.25, 0.25), (0.5, 0.75), (0.75, 0.4)]:
            y = vy.min() + fy * (vy.max() - vy.min())
            x = vx.min() + fx * (vx.max() - vx.min())
            idx = np.argmin((vy - y) ** 2 + (vx - x) ** 2)
            positions.append((int(vy[idx] * 8), int(vx[idx] * 8), 'spatial_coverage'))
        for _ in range(3):
            scratch = score.copy()
            for py, px, _ in positions:
                cy, cx = py // 8, px // 8
                scratch[max(0, cy - 100):cy + 101, max(0, cx - 100):cx + 101] = -2
            if scratch.max() <= 0:
                # No image-only candidate: retain missing evidence honestly.
                for py, px, _ in positions:
                    score[max(0, py // 8 - 100):py // 8 + 101, max(0, px // 8 - 100):px // 8 + 101] = -2
                fallback = support.copy()
                fallback[score == -2] = -2
                yy, xx = np.unravel_index(np.argmax(fallback), fallback.shape)
                kind = 'coverage_fallback_no_positive_candidate'
            else:
                yy, xx = np.unravel_index(np.argmax(scratch), scratch.shape)
                kind = 'image_only_cloud_edge_dark_candidate_not_a_shadow_label'
            positions.append((int(yy * 8), int(xx * 8), kind))
        windows = []
        overview = Image.fromarray(rgb)
        draw = ImageDraw.Draw(overview)
        for k, (cy, cx, kind) in enumerate(positions, 1):
            y = max(0, min(h - 1024, cy - 512))
            x = max(0, min(w - 1024, cx - 512))
            windows.append(dict(id=f'W{k}', row=y, col=x, height=1024, width=1024, selection=kind,
                                inside_evaluation_crops=any(int(p['x']) <= cy < int(p['x']) + 512 and
                                                           int(p['y']) <= cx < int(p['y']) + 512 for p in item['patches'])))
            draw.rectangle((x, y, x + 1023, y + 1023), outline=(255, 65, 65), width=5)
            draw.text((x + 15, y + 15), f'W{k}: row{y} col{x}', fill='red', stroke_width=1)
            crop = rgb[y:y + 1024, x:x + 1024]
            enhanced = np.clip((crop.astype(np.float32) / 255) ** .55 * 255, 0, 255).astype(np.uint8)
            lm = lab[y:y + 1024, x:x + 1024]
            panel = Image.new('RGB', (3072, 1064), '#202020')
            dp = ImageDraw.Draw(panel)
            for j, (arr, title) in enumerate([(crop, 'Native RGB B4/B3/B2'),
                                               (enhanced, 'Same RGB brightened gamma=0.55 (display only)'),
                                               (palette[lm], 'Reference: green clear / yellow thin / white cloud / blue shadow')]):
                panel.paste(Image.fromarray(arr), (1024 * j, 40))
                dp.text((1024 * j + 8, 10), title, fill='white')
            panel.save(folder / f'{k + 2:02d}_W{k}_detail.jpg', quality=95, subsampling=0)
        overview.save(folder / '01_full_scene_RGB_native.jpg', quality=95, subsampling=0)
        Image.fromarray(palette[lab]).save(folder / '02_full_reference_native.png', optimize=True)
        receipt = dict(scene=scene, folder=folder.name, height=h, width=w,
                       source_image=str(image_path), source_image_SHA256=sha(image_path),
                       source_mask=str(mask_path), source_mask_SHA256=item['mask']['SHA'],
                       band_order=[4, 3, 2], source_dtype='uint8_existing_release_not_raw_radiance',
                       transform=list(transform), crs=str(crs),
                       image_label_same_native_grid=True,
                       full_images_downsampled=False, detail_pixels_native=True,
                       reference_counts={str(v):int(c) for v,c in zip(*np.unique(lab, return_counts=True))},
                       windows=windows, human_judgement=None, no_model_outputs=True)
        write(folder / 'RECEIPT.json', receipt)
        records.append(receipt)
        write(a.out / 'RUN_STATUS.json', dict(completed_scenes=n, total_scenes=10, epoch=time.time()))
        print(f'{n}/10 {scene} {w}x{h}', flush=True)
        del rgb, overview, lab, valid
    images = [p for p in a.out.rglob('*') if p.suffix.lower() in {'.jpg', '.png'}]
    assert len(images) == 80
    write(a.out / 'SCENES.json', records)
    write(a.out / 'IMAGE_MANIFEST.json', [dict(path=str(p.relative_to(a.out)), bytes=p.stat().st_size,
                                              SHA256=sha(p)) for p in sorted(images)])
    write(a.out / 'COMPLETE.json', dict(scenes=10, images=80, model_calls=0,
                                       human_labels_filled=False,
                                       selection_cannot_prove_absence=True))


if __name__ == '__main__':
    main()
