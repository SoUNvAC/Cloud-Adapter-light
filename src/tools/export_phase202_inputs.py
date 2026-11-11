"""Export existing shared scores and no-label point inputs; no model calls."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
R=Path('/home/scv/Cloud-Adapter-light');S=Path('/home/scv/shared');O=S/'phase202_20261017'
def load(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def save(p,v):
 with p.open('x') as f:json.dump(v,f,allow_nan=False)
def main():
 state=load(O/'RUN_STATUS.json');assert state['status']=='completed'
 cfg=load(O/'EXECUTION_LOCK.json');idx=load(O/'SCORE_INDEX.json');assert idx['status']=='sealed' and sha(O/'SCORE_INDEX.json')==state['score_index_sha256']
 out=O/'local_inputs';out.mkdir(exist_ok=False);scores=out/'scores';scores.mkdir();rows=[]
 # Hard links conserve the counted persistent storage; source remains untouched.
 import os
 for row in idx['rows']:
  source=Path(row['path']);assert sha(source)==row['sha256'];dest=scores/source.name;os.link(source,dest);rows.append(dict(row,local_name='scores/'+dest.name))
 for row in cfg['existing']:
  source=Path(row['cache']);assert sha(source)==row['sha256']
  with np.load(source,allow_pickle=False) as z:a=z['scores']
  dest=scores/(row['group']['product']+'.npy');np.save(dest,a,allow_pickle=False)
  rows.append(dict(product=row['group']['product'],group_id=row['group']['group_id'],split='development',sha256=sha(dest),bytes=dest.stat().st_size,local_name='scores/'+dest.name,original_score_sha256=row['sha256']))
 sys.path.insert(0,str(R/'src/tools'));from phase118_common import coordinates
 import rasterio
 from rasterio.windows import Window
 raw=S/'data/pixbox_s2_cmix_5036991/phase104_20261013'
 fp=raw/'point_inputs_private.json';points=load(fp)['rows'];co={x['point_id']:x for x in coordinates()};assert len(points)==9500
 b8=load(R/'src/research_results/phase112_20261014/DOWNLOAD_STATUS.json')['files'];assets=load(R/'src/research_results/phase111_20261014/ASSET_VERIFICATION.json')['files'];bindings=[]
 for pid in sorted({x['product_id'] for x in points}):
  if pid in b8:path=Path(b8[pid]['path']);expected=b8[pid]['sha256']
  else:
   path=S/'phase111_ocm_runtime/assets/B8A.jp2';expected=next(x['sha256'] for x in assets if x['path']==str(path))
  assert sha(path)==expected
  with rasterio.open(path) as ds:
   assert ds.dtypes==('uint16',) and ds.shape==(5490,5490)
   for p in points:
    if p['product_id']!=pid:continue
    q=co[p['point_id']];v=int(ds.read(1,window=Window(int(q['col20']),int(q['row20']),1,1))[0,0]);p['bands']['B8A']=dict(DN=v,TOA=float(np.float32(v)/np.float32(10000)),valid=v not in (0,65535))
  bindings.append(dict(product_id=pid,path=str(path),sha256=expected))
 save(out/'PIXBOX_INPUTS_PRIVATE.json',dict(rows=points,original_six_band_sha256=sha(fp),B8A_bindings=bindings,reference_accessed=False,source='accepted Phase104 aligned area-mean TOA six bands plus unchanged native20m B8A; not Catalogue cube'))
 save(out/'INPUT_INDEX.json',dict(rows=rows,run_status=state,score_index_sha256=sha(O/'SCORE_INDEX.json'),point_inputs_sha256=sha(out/'PIXBOX_INPUTS_PRIVATE.json'),no_labels_exported=True))
 print(json.dumps(dict(status='exported',scores=len(rows),point_inputs=9500,main_calls=state['main_calls'],construction_calls=state['construction_calls'])))
if __name__=='__main__':main()
