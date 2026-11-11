"""First step: localize frozen Phase431 shadow errors; zero training/forwards."""
from pathlib import Path
import json,math
import numpy as np
from scipy import ndimage as ndi
from PIL import Image
R=Path('/home/scv/shared/phase445_20261106');PRE=Path('/home/scv/shared/phase444_retrain_20261103/source431/predictions/original_training')
def main():
    R.mkdir(exist_ok=True);rows=[r for r in json.loads(Path('/home/scv/shared/phase442_20261103/INPUTS.json').read_text()) if r['domain']=='A'];cm=np.zeros((3,3),np.int64);per=[]
    for i,row in enumerate(rows):
        raw=np.asarray(Image.open(row['mask']));y=raw.copy()
        for k,v in {0:0,1:1,2:1,3:2}.items():y[raw==k]=v
        with np.load(PRE/f"{row['index']:03d}.npz") as p:pred=p['pred']
        valid=y!=255;c=np.bincount(3*y[valid].astype(np.int64)+pred[valid],minlength=9).reshape(3,3);cm+=c
        shadow=(y==2)&valid;fn=shadow&(pred!=2);fp=valid&(y!=2)&(pred==2)
        # Zero padding ensures the crop border is treated as a geometric edge.
        dist=ndi.distance_transform_edt(np.pad(shadow,1))[1:-1,1:-1]
        lab,n=ndi.label(shadow,structure=np.ones((3,3)));areas=np.bincount(lab.ravel());area=areas[lab]
        radius=ndi.maximum(dist,labels=lab,index=np.arange(n+1)) if n else np.array([0.]);maxradius=radius[lab]
        bins={}
        selections={'edge_0_3px':shadow&(dist<=3),'near_edge_3_8px':shadow&(dist>3)&(dist<=8),'interior_gt8px':shadow&(dist>8),'component_le64px':shadow&(area<=64),'component_65_256px':shadow&(area>64)&(area<=256),'component_257_1024px':shadow&(area>256)&(area<=1024),'component_gt1024px':shadow&(area>1024),'thin_component_maxradius_le3px':shadow&(maxradius<=3),'thicker_component':shadow&(maxradius>3)}
        for name,m in selections.items():bins[name]=dict(shadow=int(m.sum()),missed=int((m&fn).sum()),missed_as_surface=int((m&fn&(pred==0)).sum()),missed_as_cloud=int((m&fn&(pred==1)).sum()))
        outside=ndi.distance_transform_edt(~np.pad(shadow,1))[1:-1,1:-1] if shadow.any() else np.full(shadow.shape,np.inf)
        fpbins={}
        for name,m in [('within3px',outside<=3),('3to8px',(outside>3)&(outside<=8)),('beyond8px',outside>8)]:fpbins[name]=dict(surface=int((fp&m&(y==0)).sum()),cloud=int((fp&m&(y==1)).sum()))
        per.append(dict(index=row['index'],group=row['group'],image=row['image'],mask=row['mask'],confusion=c.tolist(),FN=int(fn.sum()),FP=int(fp.sum()),bins=bins,FP_distance=fpbins))
    reference=json.loads(Path('/home/scv/shared/phase431_20261102/cloudsen_original_recheck_20261106/RESULT.json').read_text());assert cm.tolist()==reference['confusion']
    tp=int(cm[2,2]);fn=int(cm[2,:2].sum());fp=int(cm[:2,2].sum());den=tp+fn+fp
    totals={k:{v:sum(r['bins'][k][v] for r in per) for v in per[0]['bins'][k]} for k in per[0]['bins']}
    for v in totals.values():v.update(miss_rate=100*v['missed']/max(v['shadow'],1),share_of_all_misses=100*v['missed']/fn)
    fptotals={k:{v:sum(r['FP_distance'][k][v] for r in per) for v in ['surface','cloud']} for k in ['within3px','3to8px','beyond8px']}
    result=dict(passed=True,images=535,groups=107,forwards=0,training_updates=0,confusion=cm.tolist(),TP=tp,FN=fn,FP=fp,IoU=100*tp/den,recall=100*tp/(tp+fn),precision=100*tp/(tp+fp),FN_only_recovery_to70=math.ceil(.7*den-tp),FP_only_reduction_to70=math.ceil(den-tp/.7),joint_requirement='recovered_FN +0.7*removed_FP >=0.7*(TP+FN+FP)-TP, assuming no new errors',FN_bins=totals,FP_distance=fptotals,semantic_limit='Thin means geometric width only, not low-opacity shadow; water/dark-ground/dark-cloud require visual or ancillary evidence')
    (R/'SUMMARY.json').write_text(json.dumps(result,indent=2));(R/'PER_IMAGE.json').write_text(json.dumps(per));print(json.dumps(result))
if __name__=='__main__':main()
