"""Independent post-seal CPU reference reader; no fitting or selector changes."""
from phase319_common import *
from collections import defaultdict
import numpy as np
from PIL import Image
import cv2,rasterio
cv2.setNumThreads(4)
from rasterio.windows import Window
SELECTORS=['original_A_Sh','eligible_A_Sh','G_sun','G_reverse','G_conf','unmatched_observed','unmatched_unknown','supported_context_incomplete']
def main():
 guard();lock=load(ROOT/'SELECTION_LOCK.json');assert lock['status']=='sealed' and not (ROOT/'QUALITY_LOCK.json').exists();meta={(r['pool'],r['name']):r for r in load(ROOT/'GEOMETRY_INPUTS.json')};(ROOT/'reference').mkdir();rows=[];native_rows=[];refs=[];totals=defaultdict(lambda:np.zeros(5,np.int64));full=defaultdict(int);eligible=defaultdict(int)
 for e in lock['rows']:
  guard();pool,n,scene=e['pool'],e['name'],e['scene'];r=meta[pool,n];assert sha(e['file'])==e['sha256'];assert sha(r['mask_path'])==r['mask_sha256']
  if pool=='SPARCS':
   raw=np.asarray(Image.open(r['mask_path']));assert np.isin(raw,np.arange(7)).all();native=cv2.resize(raw,(512,512),interpolation=cv2.INTER_NEAREST);truth=np.array([2,2,0,0,0,1,0],np.uint8)[native];valid=np.ones((512,512),bool)
  else:
   assert sha(r['raw_mask_path'])==r['raw_mask_sha256']
   with rasterio.open(r['raw_mask_path']) as h:native=h.read(1,window=Window(r['col'],r['row'],512,512),boundless=True,fill_value=0)
   valid=native!=0;truth=np.full(native.shape,255,np.uint8);truth[native==128]=0;truth[np.isin(native,[192,255])]=1;truth[native==64]=2
  rp=ROOT/'reference'/(pool+'_'+n+'.npz');np.savez_compressed(rp,truth=truth,native=native,valid=valid);refs.append(dict(pool=pool,name=n,path=str(rp),sha256=sha(rp)))
  append(ROOT/'ACCESS.jsonl',dict(path=r['mask_path'] if pool=='SPARCS' else r['raw_mask_path'],kind='post-all-selection-seal reference quality',selection_lock_sha256=sha(ROOT/'SELECTION_LOCK.json')))
  with np.load(e['file']) as z:masks={k:z[k] for k in z.files}
  masks['eligible_A_Sh']=masks['original_A_Sh'] if r['status']=='qualified' else np.zeros_like(valid)
  for key in [(pool,'POOLED'),(pool,scene)]:full[key]+=int(masks['original_A_Sh'].sum());eligible[key]+=int(masks['eligible_A_Sh'].sum())
  for selector in SELECTORS:
   chosen=masks[selector];support=chosen&valid;cc=np.array([int((support&(truth==c)).sum()) for c in range(3)]+[int(chosen.sum()),int((chosen&~valid).sum())],np.int64)
   rows.append(dict(pool=pool,level='image',unit=n,scene=scene,selector=selector,full_A_Sh=int(masks['original_A_Sh'].sum()),true_Surface=int(cc[0]),true_Cloud=int(cc[1]),true_Shadow=int(cc[2]),selected=int(cc[3]),excluded=int(cc[4])))
   for key in [(pool,'POOLED',selector),(pool,scene,selector)]:totals[key]+=cc
   for v in np.unique(native):native_rows.append(dict(pool=pool,image=n,scene=scene,biome=r['biome'],selector=selector,native_class=int(v),selected=int((chosen&(native==v)).sum()),reference_valid=int((support&(native==v)).sum()),native_encoding='SPARCS0shadow1shadow_water2water3snow4land5cloud6flooded' if pool=='SPARCS' else 'L8raw0fill64shadow128clear192thin255cloud'))
 for (pool,unit,sel),cc in totals.items():rows.append(dict(pool=pool,level='pool' if unit=='POOLED' else 'scene',unit=unit,scene=unit,selector=sel,full_A_Sh=full[pool,unit],true_Surface=int(cc[0]),true_Cloud=int(cc[1]),true_Shadow=int(cc[2]),selected=int(cc[3]),excluded=int(cc[4])))
 bykey={(r['pool'],r['level'],r['unit'],r['selector']):r for r in rows}
 def precision(r):return 100*r['true_Shadow']/(r['selected']-r['excluded']) if r['selected']>r['excluded'] else None
 for r in rows:
  r['reference_valid']=r['selected']-r['excluded'];r['reference_coverage_pct']=100*r['reference_valid']/r['selected'] if r['selected'] else None;r['retained_full_A_Sh_pct']=100*r['selected']/r['full_A_Sh'] if r['full_A_Sh'] else None;r['Shadow_precision_pct']=precision(r)
  for label,sel in [('original','eligible_A_Sh'),('conf','G_conf'),('reverse','G_reverse')]:
   base=precision(bykey[r['pool'],r['level'],r['unit'],sel]);r['gain_'+label+'_pp']=r['Shadow_precision_pct']-base if base is not None and r['Shadow_precision_pct'] is not None else None
 table(ROOT/'PRECISION_COVERAGE.csv',rows);table(ROOT/'NATIVE_STRATA.csv',native_rows)
 observations=[]
 for (pool,level,unit,sel),r in bykey.items():
  if sel!='G_sun':continue
  get=lambda s:bykey[pool,level,unit,s]
  observations.append(dict(pool=pool,level=level,unit=unit,full_A_Sh=r['full_A_Sh'],supported=r['selected'],supported_incomplete=get('supported_context_incomplete')['selected'],unmatched_observed=get('unmatched_observed')['selected'],unmatched_unknown=get('unmatched_unknown')['selected'],supported_truth=[r['true_Surface'],r['true_Cloud'],r['true_Shadow']],observed_truth=[get('unmatched_observed')['true_'+c] for c in ['Surface','Cloud','Shadow']],unknown_truth=[get('unmatched_unknown')['true_'+c] for c in ['Surface','Cloud','Shadow']],reason='Outside frozen image or invalid cloud domain is unobserved, not a negative label; Source missed clouds can also cause unmatched shadows'))
 table(ROOT/'OBSERVABILITY.csv',observations);write(ROOT/'QUALITY_LOCK.json',dict(status='complete',selection_lock_sha256=sha(ROOT/'SELECTION_LOCK.json'),references=refs,table_sha256=sha(ROOT/'PRECISION_COVERAGE.csv'),native_strata_sha256=sha(ROOT/'NATIVE_STRATA.csv'),independent_process=True,resources=guard()));print('Phase319 independent quality complete',flush=True)
if __name__=='__main__':main()
