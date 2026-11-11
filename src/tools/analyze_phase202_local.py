"""Fixed local-only label joins, four fits, five calibration thresholds, evaluation.

Stages are exclusive, resumable only between stages. Never rerun a successful fit.
"""
import argparse,csv,gc,hashlib,io,json,os,sys,time,warnings,zipfile
from pathlib import Path
R=Path(__file__).resolve().parents[2];O=R/'outputs/phase202/phase202_20261017';I=O/'local_inputs'
sys.path.insert(0,str(O/'software'))
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
P=('E','O1','ON','OX','X3');COL={'O1':[0],'ON':[0,1,2,3],'OX':[0,4,5,6],'X3':[4,5,6]}
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def save(name,v):
 with (O/name).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def csvsave(name,rows):
 with (O/name).open('x',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def config():
 audit=load(O/'SHARED_CACHE_AUDIT.json');members=load(R/'src/research_results/phase122_20261015/MEMBER_INPUT_LOCK.json');assert sha(R/'src/research_results/phase122_20261015/MEMBER_INPUT_LOCK.json')==audit['member_lock_sha256']
 idx=load(I/'INPUT_INDEX.json');assert len(idx['rows'])==185 and idx['run_status']['status']=='completed'
 return audit,members,{x['product']:x for x in idx['rows']}
def cube(g,index,features=True):
 root=R/'data/sentinel2_cloud_mask_catalogue_4172871';pid=g['product'];ri=index[pid];sp=I/ri['local_name'];assert sha(sp)==ri['sha256']
 arrays=[]
 for arch,member,key in [('subscenes.zip','subscenes/','image_member_sha256'),('masks.zip','masks/','mask_member_sha256')]:
  with zipfile.ZipFile(root/arch) as z:b=z.read(member+pid+'.npy')
  assert hashlib.sha256(b).hexdigest()==g[key];arrays.append(np.load(io.BytesIO(b),allow_pickle=False))
 raw,mask=arrays;observed=np.isfinite(raw)&(raw!=0);good=observed&(raw!=np.float32(65535/10000))
 valid=(mask.sum(-1)==1)&observed[...,[3,2,1,7,11,12]].all(-1)&good[...,[3,2,8,1,11,12]].all(-1)
 assert int(valid.sum())==g['common_valid'];y=(mask.argmax(-1)[valid]==2);score=np.load(sp,allow_pickle=False);score=score[valid] if score.ndim==2 else score
 assert len(score)==len(y) and np.isfinite(score).all();flat=np.flatnonzero(valid)
 x=np.column_stack([score,raw[...,3][valid],raw[...,2][valid],raw[...,8][valid],raw[...,1][valid],raw[...,11][valid],raw[...,12][valid]]).astype(np.float32) if features else None
 return x,y,score,flat
def transform(x):
 z=x.astype(np.float64);v=np.clip(z[:,0],1e-6,1-1e-6);z[:,0]=np.log(v)-np.log1p(-v);return z
def score(x,policy,models):
 if policy=='E':return x[:,0].astype(np.float64)
 m=models[policy];out=np.empty(len(x),np.float64)
 for a in range(0,len(x),131072):
  z=transform(x[a:a+131072])[:,COL[policy]];out[a:a+len(z)]=expit(((z-m['mean'])/m['std'])@np.asarray(m['coef'])+m['intercept'])
 return out
def ap(y,s):return float(average_precision_score(y,s)) if y.any() and (~y).any() else None
def independent_ap(y,s):
 if not y.any() or not (~y).any():return None
 order=np.argsort(-s,kind='stable');v=s[order];t=y[order];ends=np.r_[np.flatnonzero(v[:-1]!=v[1:]),len(v)-1];tp=np.cumsum(t,dtype=np.int64)[ends];return float(np.dot(np.diff(np.r_[0,tp]),tp/(ends+1))/tp[-1])
def pix():
 idx=load(I/'INPUT_INDEX.json');assert sha(I/'PIXBOX_INPUTS_PRIVATE.json')==idx['point_inputs_sha256'];inputs=load(I/'PIXBOX_INPUTS_PRIVATE.json')['rows'];refs={r['point_ids'][0]:r for r in load(R/'data/pixbox_s2_cmix_5036991/phase103_20261013/deduplicated_locations.json')};by={};seals=load(R/'src/research_results/phase112_20261014/SCORES_SEALED.json')['score_sha256']
 for pid in sorted({r['product_id'] for r in inputs}):
  path=R/'data/pixbox_s2_cmix_5036991/phase112_20261014/omni'/pid/'UNLABELLED_POINT_SCORES.json';assert sha(path)==seals[pid]
  for q in load(path)['points']:by[q['point_id']]=q
 assert set(by)==set(refs)=={r['point_id'] for r in inputs};groups={};baseline={};loss=[]
 for q in inputs:
  pid=q['point_id'];r=refs[pid];e=by[pid];gid=r['group_id'];assert all(str(r[k])==str(q[k])==str(e[k]) for k in ('product_id','row20','col20'))
  ref=r['reference'];eligible=ref!='ignore';ok=bool(e['valid'] and e.get('coverage',True) and e.get('author_valid',True) and all(q['bands'][b]['valid'] for b in ('B04','B03','B8A','B02','B11','B12')))
  loss.append(dict(point_id=pid,group_id=gid,reference_eligible=eligible,E_valid=bool(e['valid']),feature_valid=ok,common=eligible and ok))
  for coll,keep in ((baseline,eligible and e['valid']),(groups,eligible and ok)):
   v=coll.setdefault(gid,dict(x=[],y=[],products=set()))
   if keep:v['x'].append([e['shadow_confidence']]+[q['bands'][b]['TOA'] for b in ('B04','B03','B8A','B02','B11','B12')]);v['y'].append(ref=='positive_ground_cloud_shadow');v['products'].add(r['product_id'])
 for coll in (groups,baseline):
  for g in coll.values():g['x']=np.asarray(g['x'],np.float64).reshape(-1,7);g['y']=np.asarray(g['y'],bool);g['products']=','.join(sorted(g['products']))
 return groups,baseline,loss
def baseline():
 audit,_,idx=config();vals=[];yy=[];ss=[]
 for rec in audit['existing']:
  _,y,s,_=cube(rec['group'],idx,False);a=ap(y,s);vals.append(a);yy.append(y);ss.append(s)
 cat=dict(macro_AP_pp=float(np.mean([v for v in vals if v is not None]))*100,pooled_AP_pp=ap(np.concatenate(yy),np.concatenate(ss))*100,positive=sum(int(y.sum()) for y in yy),negative=sum(int((~y).sum()) for y in yy));del yy,ss
 _,old,loss=pix();y=np.concatenate([g['y'] for g in old.values()]);s=np.concatenate([g['x'][:,0] for g in old.values()]);pv=[ap(g['y'],g['x'][:,0]) for g in old.values()];pixr=dict(macro_AP_pp=float(np.mean([v for v in pv if v is not None]))*100,pooled_AP_pp=ap(y,s)*100,positive=int(y.sum()),negative=int((~y).sum()))
 for pool,actual in [('Catalogue',cat),('PixBox',pixr)]:
  with (R/'src/research_results/phase119_20261015'/f'{pool}_summary.csv').open(encoding='utf-8-sig') as f:expected=next(x for x in csv.DictReader(f) if x['setting']=='E')
  for k in ('macro_AP_pp','pooled_AP_pp'):assert abs(actual[k]-float(expected[k]))<1e-8,(pool,k,actual[k],expected[k])
 save('BASELINE_REPRODUCED.json',dict(Catalogue=cat,PixBox=pixr,original_workpoint=.5,new_readout_evaluation=False));csvsave('PIXBOX_SUPPORT_PRIVATE.csv',loss)
def fit():
 assert (O/'BASELINE_REPRODUCED.json').is_file() and not (O/'FIT_STARTED.json').exists()
 audit,members,idx=config();rng=np.random.default_rng(202017);xs=[];ys=[];ws=[];rows=[];ids={}
 for g in members['fit']:
  x,y,_,flat=cube(g,idx);choose=rng.choice(len(y),min(8192,len(y)),replace=False);xs.append(transform(x[choose]));ys.append(y[choose]);ws.append(np.full(len(choose),1/len(choose)));ids[g['group_id']]=flat[choose];rows.append(dict(group_id=g['group_id'],product=g['product'],sampled=len(choose),positive=int(y[choose].sum()),row_sha256=hashlib.sha256(flat[choose].astype('<i8').tobytes()).hexdigest()));del x;gc.collect()
 np.savez_compressed(O/'FIT_ROW_IDS_PRIVATE.npz',**ids);x=np.concatenate(xs);y=np.concatenate(ys);w=np.concatenate(ws);w/=w.mean();del xs,ys,ws
 save('FIT_STARTED.json',dict(code_sha256=sha(Path(__file__)),input_index_sha256=sha(I/'INPUT_INDEX.json'),member_lock_sha256=audit['member_lock_sha256'],sample_rows=rows,sample_ids_sha256=sha(O/'FIT_ROW_IDS_PRIVATE.npz'),sampling_seed=202017,max_fits=4,features=COL,kwargs=dict(C=1,penalty='l2',solver='lbfgs',fit_intercept=True,max_iter=2000,tol=1e-8,class_weight=None)))
 models={}
 for policy in P[1:]:
  z=x[:,COL[policy]];mean=np.average(z,axis=0,weights=w);std=np.sqrt(np.average((z-mean)**2,axis=0,weights=w));std[std==0]=1
  with warnings.catch_warnings(record=True) as caught:
   warnings.simplefilter('always');model=LogisticRegression(C=1,penalty='l2',solver='lbfgs',fit_intercept=True,max_iter=2000,tol=1e-8,class_weight=None).fit((z-mean)/std,y,sample_weight=w)
  v=dict(mean=mean.tolist(),std=std.tolist(),coef=model.coef_[0].tolist(),intercept=float(model.intercept_[0]),iterations=model.n_iter_.tolist(),warnings=[str(a.message) for a in caught],converged=bool(model.n_iter_[0]<2000 and not caught),classes=model.classes_.tolist())
  models[policy]=v;save('MODEL_'+policy+'.json',v);assert v['converged'],policy+' did not converge; no retuning'
 save('MODELS_FROZEN.json',dict(models=models,fit_started_sha256=sha(O/'FIT_STARTED.json'),model_sha256={p:sha(O/('MODEL_'+p+'.json')) for p in P[1:]},O1_sign=float(np.sign(models['O1']['coef'][0])),evaluation_not_run=True))
def exact_threshold(neg):
 den=np.array([len(a) for a in neg],np.int64);supported=den>0;neg=[np.sort(a.astype(np.float64)) for a in neg]
 bits=lambda v:int(np.float64(v).view(np.uint64));value=lambda v:np.uint64(v).view(np.float64).item()
 lo=min(bits(a[0]) for a in neg if len(a));hi=max(bits(a[-1]) for a in neg if len(a))
 def counts(b,side):return np.array([len(a)-np.searchsorted(a,b,side=side) for a in neg],np.int64)
 def rate(c):return float(np.mean(c[supported]/den[supported]))
 while lo<hi:
  mid=(lo+hi)//2
  if rate(counts(value(mid),'right'))<=.01:hi=mid
  else:lo=mid+1
 b=value(lo);t=float(np.nextafter(np.float64(b),np.inf));safe=counts(b,'right');unsafe=counts(b,'left')
 assert rate(safe)<=.01<rate(unsafe) and np.nextafter(t,-np.inf)==b
 assert all(int((a>=t).sum())==int(c) for a,c in zip(neg,safe));assert all(int((a>=b).sum())==int(c) for a,c in zip(neg,unsafe))
 return dict(threshold=t,boundary=b,macro_FPR=rate(safe),unsafe_macro_FPR=rate(unsafe),FP=safe.tolist(),negative=den.tolist(),score_arithmetic='float64 exact uint64 ordered lattice; observed tie boundary then nextafter; no float32 readout quantization',independent_direct_counts=True)
def calibration_data(members,idx):
 groups={}
 for g in members['calibration']:
  x,y,_,_=cube(g,idx);groups[g['group_id']]=dict(x=x,y=y,products=g['product'])
 return groups
def calibrate():
 assert not (O/'THRESHOLDS_FROZEN.json').exists();models=load(O/'MODELS_FROZEN.json')['models'];_,members,idx=config();groups=calibration_data(members,idx);result={}
 for p in P:
  neg=[score(g['x'],p,models)[~g['y']] for g in groups.values()];result[p]=exact_threshold(neg);del neg;gc.collect()
 save('THRESHOLDS_FROZEN.json',dict(thresholds=result,model_lock_sha256=sha(O/'MODELS_FROZEN.json'),calibration_groups=list(groups),development_not_evaluated=True,PixBox_not_evaluated=True,reference='calibration references only; no oracle evaluation threshold'))
def interval(v):
 v=np.asarray(v,np.float64)
 if len(v)<2:return dict(mean=float(v.mean()) if len(v) else None,CI95=None,upper95=None,n=len(v))
 rng=np.random.default_rng(202018);draw=v[rng.integers(len(v),size=(10000,len(v)))].mean(1)
 return dict(mean=float(v.mean()),CI95=np.quantile(draw,[.025,.975]).tolist(),upper95=float(np.quantile(draw,.95)),n=len(v))
def evaluate_pool(pool,groups,models,thresholds):
 rows=[];summaries=[];contrasts=[];aps={};truth=np.concatenate([g['y'] for g in groups.values()]);checks=0
 for p in P:
  allscores=[];part=[];clipdelta=[]
  for gid,g in groups.items():
   y=g['y'];s=score(g['x'],p,models);n=int((~y).sum());pos=int(y.sum());pred=s>=thresholds[p]['threshold'];tp=int((pred&y).sum());fp=int((pred&~y).sum());a=ap(y,s);a2=independent_ap(y,s);assert a is None and a2 is None or abs(a-a2)<1e-12;checks+=1
   row=dict(pool=pool,setting=p,group_id=gid,products=g['products'],positive=pos,negative=n,AP=a,TP=tp,FP=fp,recall=tp/pos if pos and n else None,FPR=fp/n if n else None,precision=tp/(tp+fp) if tp+fp else None,threshold=thresholds[p]['threshold'],sparse=bool(pos and (pos<5 or n<20)))
   rows.append(row);part.append(row);allscores.append(s)
   if p=='E' and a is not None:clipdelta.append((ap(y,np.clip(s,1e-6,1-1e-6))-a)*100)
  aps[p]={r['group_id']:r['AP'] for r in part};pooled=np.concatenate(allscores);pred=pooled>=thresholds[p]['threshold'];tp=int((pred&truth).sum());fp=int((pred&~truth).sum());nr=int((~truth).sum());pp=int(truth.sum());fpr=[r['FPR'] for r in part if r['FPR'] is not None];rec=[r['recall'] for r in part if r['recall'] is not None];av=[r['AP'] for r in part if r['AP'] is not None];ci=interval(fpr)
  summary=dict(pool=pool,setting=p,groups=len(groups),AP_groups=len(av),macro_AP_pp=float(np.mean(av))*100 if av else None,pooled_AP_pp=ap(truth,pooled)*100,macro_recall_pp=float(np.mean(rec))*100 if rec else None,macro_FPR_pp=ci['mean']*100,FPR_CI95_pp=json.dumps([x*100 for x in ci['CI95']]) if ci['CI95'] else '',FPR_upper95_pp=ci['upper95']*100 if ci['upper95'] is not None else None,worst_group_FPR_pp=max(fpr)*100,groups_FPR_gt1pct=sum(v>.01 for v in fpr),zero_recall_groups=sum(v==0 for v in rec),no_shadow_group_FPR_pp=json.dumps({r['group_id']:r['FPR']*100 for r in part if r['positive']==0 and r['negative']}),pooled_positive=pp,pooled_negative=nr,pooled_recall=tp/pp,pooled_precision=tp/(tp+fp) if tp+fp else None,pooled_FPR=fp/nr,threshold=thresholds[p]['threshold'],E_clip_AP_delta_pp=json.dumps(clipdelta),budget_point_met=ci['mean']<=.01,budget_upper_supported=ci['upper95'] is not None and ci['upper95']<=.01)
  summaries.append(summary);del pooled,allscores;gc.collect()
 for p in P:
  delta=[(aps['OX'][gid]-a)*100 for gid,a in aps[p].items() if a is not None];ci=interval(delta);contrasts.append(dict(pool=pool,comparison='OX-'+p,**ci,improve=sum(v>1 for v in delta),stable=sum(abs(v)<=1 for v in delta),decline=sum(v< -1 for v in delta),worst=min(delta),primary=p=='ON',bootstrap_seed=202018,resamples=10000))
 for r in rows:r['OX_minus_AP_pp']=(aps['OX'][r['group_id']]-r['AP'])*100 if r['AP'] is not None else None
 csvsave(pool+'_PER_GROUP.csv',rows);csvsave(pool+'_SUMMARY.csv',summaries);save(pool+'_CONTRASTS.json',dict(rows=contrasts,conditional_fixed_models=True,unit='related groups',not_independent_confirmation=True));return checks
def evaluate():
 lock=load(O/'THRESHOLDS_FROZEN.json');assert sha(O/'MODELS_FROZEN.json')==lock['model_lock_sha256'];models=load(O/'MODELS_FROZEN.json')['models'];audit,members,idx=config();checks=0
 cal=calibration_data(members,idx);checks+=evaluate_pool('Calibration',cal,models,lock['thresholds']);del cal;gc.collect();dev={}
 for r in audit['existing']:
  g=r['group'];x,y,_,_=cube(g,idx);dev[g['group_id']]=dict(x=x,y=y,products=g['product'])
 checks+=evaluate_pool('Catalogue',dev,models,lock['thresholds']);del dev;gc.collect();groups,_,_=pix();checks+=evaluate_pool('PixBox',groups,models,lock['thresholds'])
 save('CPU_ACCEPTANCE.json',dict(status='passed',independent_AP_checks=checks,threshold_direct_counts=True,fit_count=4,threshold_count=5,bootstrap=10000,seed=202018,labels_joined_local_only=True,V4_measured=False,E_adaptation_measured=False,PixBox_not_scene_FPR=True))
def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['baseline','fit','calibrate','evaluate']);a=p.parse_args();started=time.time();globals()[a.stage]();print(json.dumps(dict(stage=a.stage,status='completed',seconds=time.time()-started)),flush=True)
if __name__=='__main__':main()
