from pathlib import Path
import json,sys
import numpy as np
from scipy import ndimage as ndi
from PIL import Image
R=Path('/home/scv/shared/phase484_20261108')/sys.argv[1]/'network_run'
def main():
 final=json.loads((R/'FINAL_RESULTS.json').read_text());step=final['choices']['best_shadow']
 roots={'off':R/'ablation_off'/f'step_{step}','484':R/'independent'/f'step_{step}'}
 rows=[r for r in json.loads(Path('/home/scv/shared/phase442_20261103/INPUTS.json').read_text()) if r['domain']=='A'];cms={k:np.zeros((3,3),np.int64) for k in roots};totals={k:{'FN':[0,0,0],'FP':[0,0,0]} for k in roots};per=[]
 for row in rows:
  raw=np.array(Image.open(row['mask']));y=raw.copy()
  for k,v in {0:0,1:1,2:1,3:2}.items():y[raw==k]=v
  valid=y!=255;shadow=(y==2)&valid;inside=ndi.distance_transform_edt(np.pad(shadow,1))[1:-1,1:-1];outside=ndi.distance_transform_edt(~np.pad(shadow,1))[1:-1,1:-1] if shadow.any() else np.full(shadow.shape,np.inf)
  bins=lambda d:[d<=3,(d>3)&(d<=8),d>8]
  entry=dict(index=row['index'],group=row['group'],counts={})
  for name,p in roots.items():
   pred=np.load(p/f"{row['index']:03d}.npz")['pred'];cm=np.bincount(y[valid].astype(np.int64)*3+pred[valid],minlength=9).reshape(3,3);cms[name]+=cm
   fn=shadow&(pred!=2);fp=valid&~shadow&(pred==2);counts={'FN':[int((fn&m).sum()) for m in bins(inside)],'FP':[int((fp&m).sum()) for m in bins(outside)]}
   entry['counts'][name]=counts
   for k in counts:totals[name][k]=[a+b for a,b in zip(totals[name][k],counts[k])]
  per.append(entry)
 assert cms['484'].tolist()==next(v for v in final['results'] if v['step']==step)['confusion']
 off=json.loads((R/'ABLATION_OFF.json').read_text());assert cms['off'].tolist()==off['results'][0]['confusion']
 for name,cm in cms.items():assert sum(totals[name]['FN'])==cm[2,:2].sum() and sum(totals[name]['FP'])==cm[:2,2].sum()
 result=dict(passed=True,images=535,forwards=0,bins=['0to3px','3to8px','beyond8px'],totals=totals,delta_on_minus_off={k:[a-b for a,b in zip(totals['484'][k],totals['off'][k])] for k in ['FN','FP']})
 (R/'EFFECT_BOUNDARY_DIAG.json').write_text(json.dumps(result,indent=2));(R/'EFFECT_BOUNDARY_PER_IMAGE.json').write_text(json.dumps(per));print(json.dumps(result))
if __name__=='__main__':main()

