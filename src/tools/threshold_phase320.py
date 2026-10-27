"""One pool/arm at a time: exact float32 ordering, complete ties, oracle points only."""
from phase320_common import *
from phase320_staircase import Staircase,TiedAP
from collections import defaultdict
from fractions import Fraction
import gc
import numpy as np
def binary(tp,fp,P,N):
 return dict(TP=int(tp),FP=int(fp),FN=int(P-tp),TN=int(N-fp),support=int(P+N),positive=int(P),negative=int(N),recall_pct=100*tp/P if P else None,FPR_pct=100*fp/N if N else None,precision_pct=100*tp/(tp+fp) if tp+fp else None,Shadow_IoU_pct=100*tp/(P+fp) if P+fp else None)
def counts(mat):return int(mat[2,2]),int(mat[:2,2].sum()),int(mat[2].sum()),int(mat[:2].sum())
def exact_macro(confusions,index):
 fractions=[]
 for mat in confusions:
  tp,fp,P,N=counts(mat);num,den=(tp,P) if index=='recall' else (fp,N)
  if den:fractions.append(Fraction(num,den))
 return sum(fractions,Fraction())/len(fractions) if fractions else None
def initial(pool,arm,rows,scenes,raws,rawc,legacy,per_scene):
 mats={sc:np.zeros((3,3),np.int64) for sc in scenes};offset=0;schema=[]
 for row in rows:
  guard()
  with open_npz(row['reference']) as z:y=z['truth'];v=z['valid']
  with open_npz(row['cache'][arm]) as z:
   pred=z['pred'];valid=z['valid'];score=z['shadow'];assert score.dtype==np.float32 and np.isfinite(score).all();assert pred.shape==valid.shape==score.shape==v.shape and not (v&~valid).any() and np.isin(pred,[0,1,2]).all();schema.append(dict(pool=pool,arm=arm,name=row['name'],keys=z.files,score_dtype=str(score.dtype),shape=list(score.shape),score_definition='softmax(s)[2] per frozen writer, never q; unused extra keys not required'))
  assert int(v.sum())==row['expected_support'];yy=y[v];ss=score[v];n=len(yy);gid=scenes.index(row['scene']);raws[offset:offset+n]=ss;rawc[offset:offset+n]=2*gid+(yy==2).astype(np.uint8);offset+=n;mats[row['scene']]+=np.bincount((3*yy+pred[v]).astype(np.int64),minlength=9).reshape(3,3)
 assert offset==len(raws);pooled=sum(mats.values());expected=np.array(json.loads(legacy['confusion']),np.int64);assert np.array_equal(pooled,expected)
 for sc,mat in mats.items():assert np.array_equal(mat,np.array(json.loads(per_scene[sc]['confusion']),np.int64))
 tp,fp,P,N=counts(pooled);mr=100*float(exact_macro(mats.values(),'recall'));mf=100*float(exact_macro(mats.values(),'FPR'));iou=100*tp/(P+fp)
 differences=dict(macro_recall_pct=mr-float(legacy['macro_Shadow_recall_pct']),macro_FPR_pct=mf-float(legacy['macro_Shadow_FPR_pct']),Shadow_IoU_pct=iou-float(legacy['Shadow_IoU_pct']));assert all(abs(v)<=1e-6 for v in differences.values());assert len(mats)==int(legacy['group_count']) and P+N==int(legacy['support'])
 result=dict(pool=pool,arm=arm,groups=len(mats),positive_groups=sum(counts(m)[2]>0 for m in mats.values()),negative_groups=sum(counts(m)[3]>0 for m in mats.values()),support=P+N,TP=tp,FP=fp,FN=P-tp,TN=N-fp,macro_recall_pct=mr,macro_FPR_pct=mf,pooled_recall_pct=100*tp/P,pooled_FPR_pct=100*fp/N,Shadow_IoU_pct=iou,original_group_mean_three_class_mIoU_pct=float(legacy['group_mean_mIoU_pct']),original_pooled_three_class_mIoU_pct=float(legacy['pooled_mIoU_pct']),error_vs317_pp=differences,integer_confusions_exact=True,pass_check=True)
 return result,mats,schema
def scratch(pool,arm,N):
 ledger=load(ROOT/'TEMP_FILES.json') if (ROOT/'TEMP_FILES.json').exists() else [];paths=[]
 for suffix,itemsize in [('score.raw',4),('code.raw',1),('score.sorted',4),('code.sorted',1)]:
  p=ROOT/'tmp'/(pool+'_'+arm+'_'+suffix);assert not p.exists();paths.append(p);ledger.append(dict(path=str(p),bytes=N*itemsize,status='registered_before_creation',pool=pool,arm=arm))
 write(ROOT/'TEMP_FILES.json',ledger);return paths
def direct_points(pool,arm,rows,scenes,summary,cb_mats,targets):
 points={};scene_out=[];table_out=[];roundoff=0.;fractions={}
 for target in ['CB_original_FPR','CB_original_recall']:
  for kind,p in summary[target].items():
   if p is not None:points.setdefault(p['threshold'],{sc:np.zeros(4,np.int64) for sc in scenes})
 for row in rows:
  guard()
  with open_npz(row['reference']) as z:y=z['truth'];v=z['valid']
  with open_npz(row['cache'][arm]) as z:score=z['shadow']
  yy=y[v]==2;ss=score[v];P=int(yy.sum());N=len(yy)-P
  for t,per in points.items():
   pred=np.zeros(len(ss),bool) if isinstance(t,str) else ss>=t;tp=int(np.count_nonzero(pred&yy));fp=int(np.count_nonzero(pred&~yy));per[row['scene']]+=np.array([tp,fp,P,N],np.int64)
 targetR=exact_macro(cb_mats,'recall');targetF=exact_macro(cb_mats,'FPR')
 for target in ['CB_original_FPR','CB_original_recall']:
  for kind,p in summary[target].items():
   if p is None:
    table_out.append(dict(pool=pool,arm=arm,target=target,point_kind=kind,threshold='no_neighbor_in_domain',selected_pixels=None,tie_block_pixels=None,TP=None,FP=None,FN=None,TN=None,macro_recall_pct=None,macro_FPR_pct=None,pooled_recall_pct=None,pooled_FPR_pct=None,precision_pct=None,Shadow_IoU_pct=None,three_class_mIoU_pct=None,staircase_index=None,recall_minus_CB_original_pp=None,FPR_minus_CB_original_pp=None,budget_satisfied=None,interpretation='oracle_diagnostic: no staircase point on this side of target metric'));continue
   per=points[p['threshold']];tot=np.sum(list(per.values()),axis=0);tp,fp,P,N=map(int,tot);assert tp==p['TP'] and fp==p['FP'] and tp+fp==p['selected_pixels'];rfracs=[Fraction(int(a),int(c)) for a,b,c,d in per.values() if c];ffracs=[Fraction(int(b),int(d)) for a,b,c,d in per.values() if d];rf=sum(rfracs,Fraction())/len(rfracs);ff=sum(ffracs,Fraction())/len(ffracs);r=100*float(rf);f=100*float(ff);roundoff=max(roundoff,abs(r-p['macro_recall_pct']),abs(f-p['macro_FPR_pct']));assert roundoff<=1e-6
   good=ff<=targetF if target=='CB_original_FPR' else rf>=targetR
   if kind=='selected':assert good,'selected point failed exact rational scene-budget check; preserve, diagnose without tolerance relaxation'
   row=dict(pool=pool,arm=arm,target=target,point_kind=kind,**p);row.update(macro_recall_pct=r,macro_FPR_pct=f,recall_minus_CB_original_pp=r-targets['CB_original_macro_recall_pct'],FPR_minus_CB_original_pp=f-targets['CB_original_macro_FPR_pct'],budget_satisfied=good,interpretation='oracle_diagnostic; validation truth used; binary Shadow threshold; three-class mIoU unavailable');table_out.append(row)
   if kind=='selected':
    fractions[target]=dict(macro_recall_exact=str(rf),macro_FPR_exact=str(ff),CB_macro_recall_exact=str(targetR),CB_macro_FPR_exact=str(targetF),matches_or_exceeds_CB_at_no_greater_FPR=rf>=targetR and ff<=targetF)
    for sc,cc in per.items():
     a,b,c,d=map(int,cc);scene_out.append(dict(pool=pool,arm=arm,target=target,scene=sc,threshold=p['threshold'],**binary(a,b,c,d),three_class_mIoU_pct=None,interpretation='oracle_diagnostic; pure positive/negative scene retained with undefined metric NA'))
 return table_out,scene_out,dict(max_stream_macro_roundoff_pp=roundoff,exact_fraction_selected_budgets=fractions,all_selected_counts_replayed_from_original_cache=True)
def main():
 guard();l=lock();assert not (ROOT/'THRESHOLD_ACCEPTANCE.json').exists();original=[];ranking=[];matches=[];scene_rows=[];states={};schema=[];qa=[];legacy={(r['pool'],r['strategy']):r for r in read(P317/'METRICS.csv')};perold=read(P317/'PER_SCENE_METRICS.csv')
 for pool in POOLS:
  spec=l['threshold'][pool]
  if spec['status']!='ready':states[pool]=dict(status='not_testable',reason='locked old caches/reference identity unavailable');continue
  scenes=sorted({r['scene'] for r in spec['rows']});N=sum(r['expected_support'] for r in spec['rows']);rows=sorted(spec['rows'],key=lambda r:(r['scene'],r['name']));cbmats=[np.array(json.loads(r['confusion']),np.int64) for r in perold if r['pool']==pool and r['strategy']=='CB']
  assert N*18+500000000<16*1024**3
  for arm in ARMS:
   guard();paths=scratch(pool,arm,N);raws=np.memmap(paths[0],dtype=np.float32,mode='w+',shape=N);rawc=np.memmap(paths[1],dtype=np.uint8,mode='w+',shape=N)
   base,mats,schemas=initial(pool,arm,rows,scenes,raws,rawc,legacy[pool,arm],{r['scene']:r for r in perold if r['pool']==pool and r['strategy']==arm});original.append(base);schema+=schemas;table(ROOT/'ORIGINAL_POINT_REPRODUCTION.csv',original);raws.flush();rawc.flush()
   print(pool,arm,'original argmax reproduced; exact global ordering',N,flush=True)
   order=np.argsort(raws,kind='quicksort')[::-1];ss=np.memmap(paths[2],dtype=np.float32,mode='w+',shape=N);cc=np.memmap(paths[3],dtype=np.uint8,mode='w+',shape=N)
   for i in range(0,N,1000000):guard();idx=order[i:i+1000000];ss[i:i+len(idx)]=raws[idx];cc[i:i+len(idx)]=rawc[idx]
   ss.flush();cc.flush();del idx,order;raws._mmap.close();rawc._mmap.close();del raws,rawc;gc.collect()
   ps=[counts(mats[sc])[2] for sc in scenes];ns=[counts(mats[sc])[3] for sc in scenes];st=Staircase(ps,ns,spec['CB_original_macro_FPR_pct'],spec['CB_original_macro_recall_pct']);ap=TiedAP(sum(ps),N);aps=[TiedAP(p,p+n) for p,n in zip(ps,ns)]
   for i in range(0,N,1000000):
    guard();sc=ss[i:i+1000000];code=cc[i:i+1000000];st.push(sc,code);ap.push(sc,(code&1)!=0);gid=code//2
    for g,aa in enumerate(aps):
     if aa.P and aa.P!=aa.N:mask=gid==g;aa.push(sc[mask],(code[mask]&1)!=0)
   summary=st.finish();pooledAP=100*ap.finish();sceneAP=[None if (a:=v.finish()) is None else 100*a for v in aps];macroAP=mean(sceneAP);ex=legacy[pool,arm];ad=dict(macro_AP_pp=macroAP-float(ex['macro_Shadow_AP_pct']),pooled_AP_pp=pooledAP-float(ex['pooled_Shadow_AP_pct']));assert all(abs(v)<=1e-6 for v in ad.values())
   for sc,a in zip(scenes,sceneAP):
    old=next(r for r in perold if r['pool']==pool and r['strategy']==arm and r['scene']==sc);oa=None if old['AP_pct']=='NA' else float(old['AP_pct']);assert (a is None and oa is None) or a is not None and oa is not None and abs(a-oa)<=1e-6
   out,sp,accept=direct_points(pool,arm,rows,scenes,summary,cbmats,spec);matches+=out;scene_rows+=sp;qa.append(dict(pool=pool,arm=arm,**accept,AP_errors_pp=ad,scene_AP_317_exact_tolerance=True,monotonic_TP_FP=True,ties_preserved=True,all_unique_float32_scores=True))
   ranking.append(dict(pool=pool,arm=arm,macro_AP_pct=macroAP,pooled_AP_pct=pooledAP,error_vs_original_AP_pp=ad,exact_unique_score_count=summary['exact_unique_score_count'],original_argmax=base,prescribed_threshold_points=summary,scene_AP_pct=dict(zip(scenes,sceneAP)),scope='oracle_diagnostic, no three-class bias reconstruction'))
   del sc,code,gid;ss._mmap.close();cc._mmap.close();del ss,cc;gc.collect();ledger=load(ROOT/'TEMP_FILES.json')
   for path in paths:
    e=next(e for e in ledger if e['path']==str(path));assert path.parent.resolve()==(ROOT/'tmp').resolve() and path.stat().st_size==e['bytes'];e.update(status='verified_complete_then_removed',sha256=sha(path));path.unlink()
   write(ROOT/'TEMP_FILES.json',ledger);table(ROOT/'THRESHOLD_MATCH.csv',matches);table(ROOT/'THRESHOLD_SCENE.csv',scene_rows);table(ROOT/'RANKING_SUMMARY.csv',ranking);print(pool,arm,'exact staircase, AP, selected budget and replay counts passed',flush=True)
  states[pool]=dict(status='passed',groups=len(scenes),images=len(rows),support=N)
 write(ROOT/'CACHE_SCHEMA.json',schema);write(ROOT/'THRESHOLD_ACCEPTANCE.json',dict(status='complete',pools=states,checks=qa,exact_float32_no_binning_no_sampling=True,three_class_threshold_mIoU='NA',resources=guard()));print('Phase320 three-pool threshold oracle complete',flush=True)
if __name__=='__main__':main()
