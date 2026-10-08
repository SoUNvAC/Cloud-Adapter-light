"""New ALCD label pixels and downloaded imagery only; no predictions or sealed test.

Native 60m labels stay unchanged. Validity requires every contributing image pixel
to be valid in all six bands. Counts are offline audit data, never deployment inputs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import datetime as dt
import numpy as np
import rasterio
from rasterio.io import MemoryFile
import xml.etree.ElementTree as ET

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--archive',type=Path,required=True); ap.add_argument('--root',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    allowed=Path('/home/scv/shared').resolve()
    for path in (a.archive,a.root,a.output):
        if not path.resolve().is_relative_to(allowed): raise ValueError('Outside authorized shared directory')
    if a.output.exists(): raise FileExistsError('Preserve prior audit; choose a new output')
    manifest=json.loads((a.root/'product_manifest.json').read_text())
    status_path=a.root/'imagery_download_status.json'
    download=json.loads(status_path.read_text()) if status_path.exists() else {'files':{}}
    sha=hashlib.sha256()
    with a.archive.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): sha.update(b)
    if sha.hexdigest()!=manifest['archive_sha256']: raise ValueError('Archive changed')
    report={'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'scope':'new_ALCD_only_no_model_no_training','class_counts_are_offline_audit_not_deployment_features':True,'confidence_is_not_truth_accuracy':True,'scenes':[]}
    with tarfile.open(a.archive,'r:gz') as tar:
        for row in manifest['scenes']:
            member=row['label_members'][0]; raw=tar.extractfile(member).read(); r={'key':row['key'],'label_sha256':hashlib.sha256(raw).hexdigest(),'duplicate_labels_identical':all(hashlib.sha256(tar.extractfile(m).read()).hexdigest()==hashlib.sha256(raw).hexdigest() for m in row['label_members'])}
            xml_objects=[o for o in row.get('objects',[]) if o['name'].endswith('.xml') and '/GRANULE/' not in o['name']]
            r['product_radiometry']=[]
            for obj in xml_objects:
                entry=download['files'].get(obj['name'])
                if entry and entry['status']=='verified':
                    tree=ET.parse(a.root/entry['path'])
                    fields=[{'tag':e.tag.split('}')[-1],'value':e.text,'attributes':e.attrib} for e in tree.iter() if e.tag.split('}')[-1] in ('QUANTIFICATION_VALUE','PROCESSING_BASELINE','RADIO_ADD_OFFSET','SPECIAL_VALUE_TEXT','SPECIAL_VALUE_INDEX','PRODUCT_URI','SENSING_TIME')]
                    r['product_radiometry'].extend(fields)
            with MemoryFile(raw) as mem, mem.open() as lab:
                labels=lab.read(1); vals,counts=np.unique(labels,return_counts=True)
                r.update(shape=list(labels.shape),crs=str(lab.crs),transform=list(lab.transform)[:6],bounds=list(lab.bounds),resolution=list(lab.res),nodata=lab.nodata,label_class_counts={str(int(v)):int(c) for v,c in zip(vals,counts)})
                valid=np.isin(labels,[2,3,4,5,6,7]); r['valid_label_pixels']=int(valid.sum())
                r['unexpected_label_values']=[int(v) for v in vals if v not in range(8)]
                r['bands']={}; grid_valid=True; complete=True
                for obj in row.get('objects',[]):
                    if not obj['name'].endswith('.jp2'): continue
                    band=Path(obj['name']).stem.split('_')[-1]; entry=download['files'].get(obj['name'])
                    if not entry or entry['status']!='verified':
                        r['bands'][band]={'status':'not_yet_verified'}; complete=False; continue
                    path=a.root/entry['path']; im_sha=hashlib.sha256()
                    with path.open('rb') as f:
                        for block in iter(lambda:f.read(4*1024*1024),b''): im_sha.update(block)
                    if im_sha.hexdigest()!=entry['sha256']: raise ValueError('Image changed')
                    with rasterio.open(path) as im:
                        ratio=lab.res[0]/im.res[0]
                        aligned=(im.crs==lab.crs and np.allclose(im.bounds,lab.bounds,atol=1e-6,rtol=0) and ratio in (3,6) and im.width==lab.width*ratio and im.height==lab.height*ratio and im.transform.b==0 and im.transform.d==0)
                        r['bands'][band]={'status':'decoded','shape':[im.height,im.width],'crs':str(im.crs),'resolution':list(im.res),'bounds':list(im.bounds),'nodata':im.nodata,'native_grid_matches_label':bool(aligned),'sha256':entry['sha256']}
                        grid_valid &= bool(aligned)
                        if aligned:
                            # GDAL masks: all fine pixels must be valid, rather than
                            # nearest-neighbor hiding missing observations.
                            native=im.read(1)
                            native_valid=(native!=0)&(im.read_masks(1)>0)
                            # Sentinel-2 L1C special value zero is no observation,
                            # even when the JP2 driver does not expose nodata.
                            factor=int(ratio)
                            mask=native_valid.reshape(lab.height,factor,lab.width,factor).all(axis=(1,3))
                            del native,native_valid
                            valid &= mask
                            r['bands'][band]['valid_native_label_cells']=int(mask.sum())
                complete &= set(r['bands'])=={'B02','B03','B04','B08','B11','B12'}
                r['alignment_status']='all_six_grids_match' if complete and grid_valid else ('grid_mismatch' if not grid_valid else 'pending_verified_bands')
                if complete and grid_valid:
                    r['joint_valid_pixels_60m']=int(valid.sum())
                    r['shadow_pixels_joint_valid_60m']=int(((labels==4)&valid).sum())
                r['pixel_semantic_quality']='not_certified_by_geometric_audit_requires_independent_visual_annotation'
            report['scenes'].append(r)
    report['scenes_audited']=len(report['scenes'])
    report['scenes_all_six_grids_match']=sum(s['alignment_status']=='all_six_grids_match' for s in report['scenes'])
    report['status']='geometric_audit_complete_semantics_unconfirmed' if report['scenes_all_six_grids_match']==37 else 'imagery_incomplete_or_grid_mismatch'
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x',encoding='utf-8') as f: json.dump(report,f,ensure_ascii=False,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k!='scenes'},ensure_ascii=False))

if __name__=='__main__': main()
