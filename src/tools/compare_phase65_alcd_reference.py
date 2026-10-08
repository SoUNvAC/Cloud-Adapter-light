"""Compare unchanged ALCD labels with authors' published GDSD revisions.

This measures annotation disagreement, NOT accuracy or model performance. Only
georeferenced, aligned cells whose nine revised 20m labels agree are compared.
"""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import tarfile
import numpy as np
from rasterio.io import MemoryFile

REF_SHA='441d8f6667c9059a8335aa14d2de0772fe5e0a7672aa1a28563f4742ed06516d'
ALCD_SHA='5912fbcfe9edbc1c2cdffbb7ef0119ffdb3099bc9eec37efe301929008bf966d'

def file_sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--archive',type=Path,required=True); ap.add_argument('--reference',type=Path,required=True); ap.add_argument('--manifest',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    for p in (a.archive,a.reference,a.manifest,a.output):
        if not p.resolve().is_relative_to(Path('/home/scv/shared')): raise ValueError('Outside authorized shared')
    if a.output.exists(): raise FileExistsError('Preserve previous report')
    if file_sha(a.archive)!=ALCD_SHA or file_sha(a.reference)!=REF_SHA: raise ValueError('Archive identity changed')
    scenes=json.loads(a.manifest.read_text())['scenes']
    report={'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'reference_record':'14397827','reference_sha256':REF_SHA,'original_sha256':ALCD_SHA,'interpretation':'annotation_disagreement_not_error_rate_not_model_metrics','classes':['Shadow','Cloud','Surface'],'comparison_rule':'native 60m; all nine refined 20m cells agree; valid originals only; exact CRS/bounds/resolution match required','scenes':[]}
    with tarfile.open(a.archive,'r:gz') as original, tarfile.open(a.reference,'r:gz') as reference:
        for member in reference:
            if not member.isfile() or not member.name.endswith('.tif'): continue
            rows=[r for r in scenes if r['parameters']['cloudy_product_name'] in member.name]
            if len(rows)!=1: raise ValueError('Ambiguous revised product match')
            row=rows[0]; old_bytes=original.extractfile(row['label_members'][0]).read(); new_bytes=reference.extractfile(member).read()
            s={'key':row['key'],'reference_member':member.name,'revised_label_sha256':hashlib.sha256(new_bytes).hexdigest(),'original_label_sha256':hashlib.sha256(old_bytes).hexdigest()}
            with MemoryFile(old_bytes) as om, om.open() as old, MemoryFile(new_bytes) as nm, nm.open() as new:
                aligned=old.crs is not None and new.crs==old.crs and np.allclose(old.bounds,new.bounds,rtol=0,atol=1e-6) and old.res==(60,60) and new.res==(20,20) and new.width==3*old.width and new.height==3*old.height and all(t.b==0 and t.d==0 for t in (old.transform,new.transform))
                s.update(original_crs=str(old.crs),reference_crs=str(new.crs),original_shape=[old.height,old.width],reference_shape=[new.height,new.width],grids_match=bool(aligned))
                if aligned:
                    o=old.read(1); n=new.read(1)
                    if not np.isin(n,[0,64,128,255]).all(): raise ValueError('Unexpected reference labels')
                    old_lut=np.full(256,255,dtype='uint8'); old_lut[[2,3]]=1; old_lut[4]=0; old_lut[[5,6,7]]=2
                    new_lut=np.full(256,255,dtype='uint8'); new_lut[64]=0; new_lut[255]=1; new_lut[128]=2
                    o=old_lut[o]; n=new_lut[n].reshape(old.height,3,old.width,3)
                    lo=n.min(axis=(1,3)); hi=n.max(axis=(1,3)); valid=(lo==hi)&(lo<3)&(o<3)
                    matrix=np.bincount((o[valid]*3+lo[valid]).astype('int64'),minlength=9).reshape(3,3)
                    s.update(compared_stable_cells_60m=int(valid.sum()),mixed_or_fill_or_invalid_cells_60m=int(valid.size-valid.sum()),original_to_revised_counts=matrix.tolist(),disagreeing_stable_cells_60m=int(matrix.sum()-matrix.trace()))
                    # Fixed examples for independent visual review, never model-led.
                    s['review_examples']={}
                    for i,j in ((2,0),(0,2),(0,1),(1,0),(2,1),(1,2)):
                        yy,xx=np.where(valid&(o==i)&(lo==j))
                        if yy.size:
                            indices=np.linspace(0,yy.size-1,min(5,yy.size),dtype=int)
                            s['review_examples'][f'{report["classes"][i]}_to_{report["classes"][j]}']=[{'row_60m':int(yy[k]),'col_60m':int(xx[k]),'x':float(old.xy(int(yy[k]),int(xx[k]))[0]),'y':float(old.xy(int(yy[k]),int(xx[k]))[1])} for k in indices]
                else: s['comparison_status']='not_compared_unverified_grid'
            report['scenes'].append(s)
    report['matched_reference_scenes']=len(report['scenes'])
    report['compared_scenes']=sum(s['grids_match'] for s in report['scenes'])
    report['scenes_with_annotation_disagreement']=sum(s.get('disagreeing_stable_cells_60m',0)>0 for s in report['scenes'])
    report['aggregate_original_to_revised_counts']=np.sum([s['original_to_revised_counts'] for s in report['scenes'] if s['grids_match']],axis=0).tolist() if report['compared_scenes'] else []
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x',encoding='utf-8') as f: json.dump(report,f,ensure_ascii=False,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k!='scenes'},ensure_ascii=False))

if __name__=='__main__': main()
