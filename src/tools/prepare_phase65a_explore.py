"""Extract/audit only already frozen fit and development products. Never resplit."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np

SPLIT_SHA = 'a431745e7a17bbd4a7b9e12ef3bf76767d4eb827911bd5cb5e632683c2177f7b'
SOURCE_SHA = '64dd9a20288c35ab3362b8b1c1dafc216d9c44d60868475177e139d6be859dc9'


def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def convert(raw, mask):
    if raw.shape != (1022,1022,13) or raw.dtype != np.float32:
        raise ValueError('Image header differs from official release')
    if mask.shape != (1022,1022,3) or mask.dtype != np.bool_:
        raise ValueError('Mask header differs from official release')
    six=raw[...,[3,2,1,7,11,12]]
    image_valid=np.isfinite(six).all(axis=-1) & (six != 0).all(axis=-1)
    label_valid=(mask.sum(axis=-1)==1) & image_valid
    target=np.full(raw.shape[:2],255,dtype=np.uint8)
    target[label_valid]=mask.argmax(axis=-1)[label_valid]
    # Linear TOA -> source 0..255 numeric scale; preserve values above one.
    rgb=raw[...,[3,2,1]].copy()*np.float32(255)
    rgb[~image_valid]=0
    brightness=six[image_valid,:3].mean(axis=-1)
    spectral=dict(brightness_mean=float(brightness.mean()),brightness_std=float(brightness.std()),
                  dark_fraction=float((brightness<.08).mean()),
                  b8_mean=float(six[image_valid,3].mean()),b11_mean=float(six[image_valid,4].mean()),
                  b12_mean=float(six[image_valid,5].mean())) if image_valid.any() else {}
    return rgb,target,image_valid,spectral


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--split',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    allowed=[Path('/home/scv/shared'),Path('/home/scv/Cloud-Adapter-light')]
    if not all(any(x.resolve().is_relative_to(r) for r in allowed) for x in [a.root,a.split,a.source,a.output]):
        raise ValueError('Outside authorized roots')
    assert digest(a.split)==SPLIT_SHA and digest(a.source)==SOURCE_SHA
    d=json.loads(a.split.read_text());assert d['status']=='stopped_insufficient_h1_support'
    status=json.loads((a.root/'download_status.json').read_text());assert status['status']=='all_files_verified'
    # Read raw bytes only for authorized cohorts; no reserved pixel headers or arrays.
    selected=[r for r in d['proposed_rows'] if r['split'] in {'fit','development_val'}]
    assert sum(r['split']=='fit' for r in selected)==153 and sum(r['split']=='development_val' for r in selected)==32
    a.output.mkdir(exist_ok=False);(a.output/'arrays').mkdir()
    rows=[];tags={r['scene']:r for r in csv.DictReader((a.root/'classification_tags.csv').open())}
    with zipfile.ZipFile(a.root/'subscenes.zip') as images,zipfile.ZipFile(a.root/'masks.zip') as masks:
        for group in selected:
            products=group['products'] if group['split']=='fit' else [group['representative']]
            for product in products:
                assert tags[product]['shadows_marked']=='1'
                rawbytes=images.read(f'subscenes/{product}.npy');maskbytes=masks.read(f'masks/{product}.npy')
                raw=np.load(io.BytesIO(rawbytes),allow_pickle=False);mask=np.load(io.BytesIO(maskbytes),allow_pickle=False)
                rgb,target,valid,spectral=convert(raw,mask)
                if not (target!=255).any():raise ValueError('Empty valid product')
                prefix=f'arrays/{product}'
                np.save(a.output/(prefix+'_rgb.npy'),rgb);np.save(a.output/(prefix+'_mask.npy'),target)
                np.save(a.output/(prefix+'_image_valid.npy'),valid)
                rows.append(dict(product=product,group_id=group['group_id'],split=group['split'],
                    is_representative=product==group['representative'],h1_public_support=group['h1_supported'],
                    rgb_path=prefix+'_rgb.npy',mask_path=prefix+'_mask.npy',image_valid_path=prefix+'_image_valid.npy',
                    member_sha256=dict(image=hashlib.sha256(rawbytes).hexdigest(),mask=hashlib.sha256(maskbytes).hexdigest()),
                    prepared_sha256={suffix:digest(a.output/(prefix+suffix)) for suffix in ['_rgb.npy','_mask.npy','_image_valid.npy']},
                    label_counts=np.bincount(target[target!=255],minlength=3).tolist(),valid_pixels=int((target!=255).sum()),
                    invalid_pixels=int((target==255).sum()),image_valid_pixels=int(valid.sum()),observable_spectra=spectral))
                print('prepared',group['split'],product,flush=True)
    report=dict(scope='exploratory_fit_development_only',training_authorization='user_20261008_explicit_exploratory_phase65a',
        original_split_sha256=SPLIT_SHA,source_sha256=SOURCE_SHA,official_readme_sha256=digest(a.root/'README.pdf'),
        class_order=['surface_visible','cloud','shadow'],rgb_policy='RGB indices 3,2,1; float TOA*255; no clipping above one; BGR conversion only at loader',
        validity_policy='one-hot exactly one AND six required bands finite and nonzero; invalid ignored255; observable features use image_valid only',
        reserved_cohorts_pixels_read=False,rows=rows)
    report['target_fit_valid_label_budget']=sum(r['valid_pixels'] for r in rows if r['split']=='fit')
    (a.output/'manifest.json').write_text(json.dumps(report,indent=2))
    print('prepared complete',len(rows),report['target_fit_valid_label_budget'],flush=True)


if __name__=='__main__':main()
