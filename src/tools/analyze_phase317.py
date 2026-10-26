"""Three-pool independent CPU statistics; exact tied AP with bounded-memory sorted chunks."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import gc,resource
from collections import defaultdict
import numpy as np
import analyze_phase304 as legacy
from phase317_common import *
def mean(v):
 v=[x for x in v if x is not None];return sum(v)/len(v) if v else None
def exact_ap(y,s):
 n=len(y);P=int(np.count_nonzero(y))
 if not P or P==n:return None
 order=np.argsort(s,kind='stable')[::-1];ss=np.asarray(s)[order];yy=np.asarray(y)[order].astype(np.int8);del order;gc.collect();tp=seen=0;area=0.;pending=None
 def flush(pos,total):
  nonlocal tp,seen,area
  tp+=pos;seen+=total;area+=pos*tp/seen
 for i in range(0,n,1000000):
  sc=ss[i:i+1000000];yb=yy[i:i+1000000];ends=np.r_[np.flatnonzero(sc[:-1]!=sc[1:]),len(sc)-1];nums=np.diff(np.r_[-1,ends]);pos=np.diff(np.r_[0,np.cumsum(yb,dtype=np.int64)[ends]])
  if pending is not None:
   value,pp,nn=pending
   if value==sc[0]:pos[0]+=pp;nums[0]+=nn
   else:flush(pp,nn)
  if len(ends)>1:
   pp=pos[:-1];nn=nums[:-1];area+=float(np.sum(pp*(tp+np.cumsum(pp))/(seen+np.cumsum(nn))));tp+=int(pp.sum());seen+=int(nn.sum())
  pending=(sc[-1],int(pos[-1]),int(nums[-1]))
 flush(pending[1],pending[2]);assert tp==P and seen==n
 return area/P
def metric(mat,AP=None):
 diag=np.diag(mat);union=mat.sum(0)+mat.sum(1)-diag;ious=np.divide(diag,union,out=np.full(3,np.nan),where=union>0);N=int(mat.sum());P=int(mat[2].sum());neg=N-P
 return dict(support=N,positive=P,negative=neg,confusion=mat.tolist(),union_classes=int((union>0).sum()),mIoU_pct=100*float(np.nanmean(ious)) if (union>0).any() else None,Surface_IoU_pct=100*float(ious[0]) if union[0] else None,Cloud_IoU_pct=100*float(ious[1]) if union[1] else None,Shadow_IoU_pct=100*float(ious[2]) if union[2] else None,AP_pct=100*AP if AP is not None else None,recall_pct=100*int(mat[2,2])/P if P else None,FPR_pct=100*int(mat[:2,2].sum())/neg if neg else None)
def cache(pool,arm,name):
 if pool=='CMC' and arm in ['Source','PL']:
  p=Path('/home/scv/shared/phase302_20261020/evaluation')/('Source_'+name+'.npz') if arm=='Source' else OLD/'evaluation'/('LoRA_'+name+'.npz')
  lock=load(OLD/'EVALUATION_LOCK.json');expected=next(r['sha256'] for r in lock['rows'] if r['product']==name and r['arm']==('Source' if arm=='Source' else 'LoRA'))
 else:
  r=next(r for r in load(ROOT/pool/arm/'EVALUATION_LOCK.json')['rows'] if r['name']==name);p=Path(r['file']);expected=r['sha256']
 assert sha(p)==expected;return p,expected
def main():
 assert 'torch' not in sys.modules;resource_check();start=time.monotonic();assert load(ROOT/'ALL_SEALED.json')['status']=='sealed' and not (ROOT/'CPU_ACCEPTANCE.json').exists()
 assert abs(exact_ap(np.array([0,1,1,0],bool),np.array([.9,.8,.8,.1],np.float32))-2/3)<1e-12
 refs=load(ROOT/'REFERENCE_LOCK.json')['rows'];strategies=['Source','PL','CB'];comparisons=[('Source','PL'),('Source','CB'),('PL','CB')];per=[];metrics=[];flows=[];cells=[];damage=[];native_rows=[];allpins={};auditAP=[];pooled_mats={};tensors={};transient=[]
 for pool in ['CMC','SPARCS','L8']:
  rr=[r for r in refs if r['pool']==pool];groups=defaultdict(list)
  for r in rr:groups[r['scene']].append(r)
  totalm={a:np.zeros((3,3),np.int64) for a in strategies};totalt={a+'->'+b:np.zeros((3,3,3),np.int64) for a,b in comparisons}
  for scene,rows in sorted(groups.items()):
   mats={a:np.zeros((3,3),np.int64) for a in strategies};ts={a+'->'+b:np.zeros((3,3,3),np.int64) for a,b in comparisons};ys=[];scores={a:[] for a in strategies};native=defaultdict(lambda:dict(support=0,metric_support=0,preds={a:np.zeros(3,np.int64) for a in strategies}));db=defaultdict(int)
   for r in rows:
    assert sha(r['file'])==r['sha256'];allpins[r['file']]=r['sha256']
    with np.load(r['file']) as z:y=z['truth'];v=z['valid'];raw=z['native']
    yy=y[v];ys.append((yy==2));pred={}
    for a in strategies:
     path,h=cache(pool,a,r['name']);allpins[str(path)]=h
     with np.load(path) as z:p=z['pred'];score=z['shadow'];valid=z['valid'];assert not (v&~valid).any() and score.dtype==np.float32 and np.isfinite(score).all()
     pred[a]=p;mm=np.bincount((3*yy+p[v]).astype(np.int64),minlength=9).reshape(3,3);mats[a]+=mm;scores[a].append(score[v])
    for a,b in comparisons:
     t=np.bincount((9*yy+3*pred[a][v]+pred[b][v]).astype(np.int64),minlength=27).reshape(3,3,3);ts[a+'->'+b]+=t
    D=v&(y==0)&(pred['Source']==0)&(pred['PL']==1);B=v&(y==1)&(pred['Source']==2)&(pred['PL']==1);db['D']+=int(D.sum());db['B']+=int(B.sum())
    for c in range(3):db['D_CB_'+C[c]]+=int((D&(pred['CB']==c)).sum());db['B_CB_'+C[c]]+=int((B&(pred['CB']==c)).sum())
    outside=v&(y==0)&(pred['Source']==0)&~D;db['Doutside_CB_Cloud']+=int((outside&(pred['CB']==1)).sum());db['Doutside_CB_Shadow']+=int((outside&(pred['CB']==2)).sum());db['new_Surface_Cloud_vs_PL']+=int((v&(y==0)&(pred['PL']==0)&(pred['CB']==1)).sum());db['new_Surface_Shadow_vs_PL']+=int((v&(y==0)&(pred['PL']==0)&(pred['CB']==2)).sum())
    for n in np.unique(raw):
     reg=raw==n;key=(r['biome'],int(n));entry=native[key];entry['support']+=int(reg.sum());entry['metric_support']+=int((reg&v).sum())
     for a in strategies:entry['preds'][a]+=np.bincount(pred[a][reg&v],minlength=3)
   yy=np.concatenate(ys)
   for a in strategies:
    ss=np.concatenate(scores[a]);ap=exact_ap(yy,ss);mat=mats[a];m=metric(mat,ap);per.append(dict(pool=pool,strategy=a,scene=scene,images=len(rows),**m));totalm[a]+=mat
    if not any(r['pool']==pool and r['strategy']==a for r in auditAP):
     sub=min(len(yy),65536);v1=exact_ap(yy[:sub],ss[:sub]);v2=legacy.ap(yy[:sub],ss[:sub]);assert (v1 is None and v2 is None) or abs(v1-v2)<1e-12;auditAP.append(dict(pool=pool,scene=scene,strategy=a,sample_pixels=sub,stream_AP=v1,independent_AP=v2))
    del ss
   for pair,t in ts.items():
    a,b=pair.split('->');assert np.array_equal(t.sum(2),mats[a]) and np.array_equal(t.sum(1),mats[b]);prefix=dict(pool=pool,comparison=pair,scene=scene);cells+=legacy.cells(t,prefix);flows+=legacy.events(t,prefix);totalt[pair]+=t
   assert sum(db['D_CB_'+c] for c in C)==db['D'] and sum(db['B_CB_'+c] for c in C)==db['B'];damage.append(dict(pool=pool,scene=scene,**dict(db)))
   for (biome,n),e in native.items():
    names={'CMC':{0:'Surface',1:'Cloud',2:'Shadow'},'SPARCS':{0:'shadow',1:'shadow_over_water',2:'water',3:'snow',4:'land',5:'cloud',6:'flooded'},'L8':{0:'fill',64:'shadow',128:'clear',192:'thin_cloud',255:'cloud'}}
    native_rows.append(dict(pool=pool,scene=scene,biome=biome,native_code=n,native_class=names[pool][n],reference_support=e['support'],metric_support=e['metric_support'],Source_parent_predictions=e['preds']['Source'].tolist(),PL_parent_predictions=e['preds']['PL'].tolist(),CB_parent_predictions=e['preds']['CB'].tolist(),scene_mIoU={a:next(r['mIoU_pct'] for r in reversed(per) if r['pool']==pool and r['scene']==scene and r['strategy']==a) for a in strategies},scene_union={a:metric(mats[a])['union_classes'] for a in strategies},note='L8 biome snow_ice/water is scene-stratum not pixelwise snow/water truth; rawfill excluded; CMC no finer native subclasses'))
   del ys,scores,yy;gc.collect()
  # One strategy at a time: exact pooled AP without retaining three full score pools or loading models.
  N=sum(r['support'] for r in rr)
  for a in strategies:
   sp=ROOT/'tmp'/('AP_'+pool+'_'+a+'.scores');yp=ROOT/'tmp'/('AP_'+pool+'_'+a+'.truth');ss=np.memmap(sp,dtype=np.float32,mode='w+',shape=(N,));yy=np.memmap(yp,dtype=np.bool_,mode='w+',shape=(N,));offset=0
   for r in rr:
    with np.load(r['file']) as z:v=z['valid'];y=z['truth']
    path,h=cache(pool,a,r['name'])
    with np.load(path) as z:sc=z['shadow']
    n=int(v.sum());ss[offset:offset+n]=sc[v];yy[offset:offset+n]=y[v]==2;offset+=n
   assert offset==N;ss.flush();yy.flush();ap=exact_ap(yy,ss);m=metric(totalm[a],ap);pg=[r for r in per if r['pool']==pool and r['strategy']==a]
   metrics.append(dict(pool=pool,strategy=a,group_count=len(pg),images=len(rr),group_mean_mIoU_pct=mean([r['mIoU_pct'] for r in pg]),pooled_mIoU_pct=m['mIoU_pct'],Surface_IoU_pct=m['Surface_IoU_pct'],Cloud_IoU_pct=m['Cloud_IoU_pct'],Shadow_IoU_pct=m['Shadow_IoU_pct'],macro_Shadow_AP_pct=mean([r['AP_pct'] for r in pg]),pooled_Shadow_AP_pct=m['AP_pct'],macro_Shadow_recall_pct=mean([r['recall_pct'] for r in pg]),macro_Shadow_FPR_pct=mean([r['FPR_pct'] for r in pg]),AP_groups=sum(r['AP_pct'] is not None for r in pg),recall_groups=sum(r['recall_pct'] is not None for r in pg),FPR_groups=sum(r['FPR_pct'] is not None for r in pg),support=N,positive=m['positive'],negative=m['negative'],confusion=m['confusion']))
   transient.append(dict(file=sp.name,bytes=sp.stat().st_size,used='exactfloat32 pooledAP scratch, no newmodelcall',removed_after_AP=True));del ss,yy;gc.collect();sp.unlink();yp.unlink()
  for pair,t in totalt.items():
   a,b=pair.split('->');assert np.array_equal(t.sum(2),totalm[a]) and np.array_equal(t.sum(1),totalm[b]);prefix=dict(pool=pool,comparison=pair,scene='POOLED');cells+=legacy.cells(t,prefix);flows+=legacy.events(t,prefix)
  pooled_mats[pool]={a:m.tolist() for a,m in totalm.items()};tensors[pool]={p:t.tolist() for p,t in totalt.items()};print('CPU statistics',pool,flush=True)
 for pool in ['CMC','SPARCS','L8']:
  rr=[r for r in damage if r['pool']==pool];damage.append(dict(pool=pool,scene='POOLED',**{k:sum(r[k] for r in rr) for k in rr[0] if k not in ['pool','scene']}))
 cmc=next(r for r in damage if r['pool']=='CMC' and r['scene']=='POOLED');assert cmc['D']==506798 and cmc['B']==784478
 for r in metrics:
  src=next(t for t in metrics if t['pool']==r['pool'] and t['strategy']=='Source');pl=next(t for t in metrics if t['pool']==r['pool'] and t['strategy']=='PL')
  for k in ['group_mean_mIoU_pct','pooled_mIoU_pct','Shadow_IoU_pct','macro_Shadow_recall_pct','macro_Shadow_FPR_pct']:
   r[k+'_minus_Source_pp']=None if r[k] is None or src[k] is None else r[k]-src[k];r[k+'_minus_PL_pp']=None if r[k] is None or pl[k] is None else r[k]-pl[k]
 support=[]
 for pool in ['CMC','SPARCS','L8']:
  d=inputs(pool);q=load(OLD/'PSEUDO_LOCK.json') if pool=='CMC' else load(ROOT/pool/'PSEUDO_LOCK.json');ep={a:(dict(attempts=1000,updates=955,skipped=45) if pool=='CMC' and a=='PL' else load(ROOT/pool/a/'ENDPOINT_LOCK.json')) for a in ['PL','CB']}
  support.append(dict(pool=pool,fit_images=len(d['fit']),fit_scenes=len({r.get('group_id',r.get('scene')) for r in d['fit']}),val_images=len(d['val']),val_scenes=len({r.get('group_id',r.get('scene')) for r in d['val']}),input=d['input'],accepted=q['accepted'],accepted_classes=q['classes'],class_proportions=[n/q['accepted'] if q['accepted'] else None for n in q['classes']],nonempty_products=q['nonempty_products'],class_present_steps=q['class_present_steps'],PL_attempts=ep['PL']['attempts'],PL_updates=ep['PL']['updates'],PL_skipped=ep['PL']['skipped'],CB_attempts=ep['CB']['attempts'],CB_updates=ep['CB']['updates'],CB_skipped=ep['CB']['skipped'],history='developed/exposed pool; exploratory one seed; SPARCS/L8 display512 differs from originalrandomnative supervisedtraining',L8_shadow_qualification='yesonly' if pool=='L8' else 'NA'))
 for name,rows in [('METRICS.csv',metrics),('PER_SCENE_METRICS.csv',per),('TRANSITIONS27.csv',cells),('ERROR_FLOW.csv',flows),('DAMAGE_BENEFIT.csv',damage),('PER_SCENE_NATIVE.csv',native_rows),('DATA_AND_SUPPORT.csv',support)]:table(ROOT/name,rows)
 write(ROOT/'CPU_READOUT.json',dict(confusions=pooled_mats,tensors=tensors));write(ROOT/'CPU_INPUT_SHA256.json',allpins);write(ROOT/'AP_INDEPENDENT_CHECKS.json',auditAP);write(ROOT/'TRANSIENT_ANALYSIS.json',transient)
 write(ROOT/'CPU_ACCEPTANCE.json',dict(status='passed',no_torch=True,allfive_before_references=True,historical_CMC_D=506798,historical_CMC_B=784478,pooled_AP_exact_float32_no_score_quantization=True,all_transition_margins=True,seconds=time.monotonic()-start,resources=resource_check()))
 print('Phase317 CPU statistics passed')
if __name__=='__main__':main()
