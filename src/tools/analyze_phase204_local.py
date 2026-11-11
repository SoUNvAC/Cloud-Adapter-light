"""Phase204: frozen cache-only warning screen followed by local reference analysis."""
import collections,csv,gc,hashlib,json,os,sys,time
from pathlib import Path
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import analyze_phase203_local as old
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score,roc_auc_score
R=old.R;P=old.O;O=R/'outputs/phase204/phase204_20261019';T=old.T
SEEDS=(12201,12301);FOCUS={'Catalogue':('32TML','35RPM','52UDG'),'PixBox':('19FDV','33VVE','44SKJ')}
START=time.time();FILES={}
def save(n,v):
 with (O/n).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def table(n,rows):
 assert rows,n
 with (O/n).open('x',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def record(p):
 p=Path(p);h=old.sha(p);FILES[str(p.relative_to(R))]=h;return h
def checked_old(arm,pid,pool):
 if pool=='Catalogue':
  base=R/'data/sentinel2_cloud_mask_catalogue_4172871'/('phase123_20261016' if arm=='C123' else 'phase122_20261015');stage='R_Catalogue_reuse' if arm=='R' else 'C_Catalogue';p=base/stage/'scores/Catalogue'/(pid+'.npz')
 else:
  base=R/'data/pixbox_s2_cmix_5036991'/('phase119_20261015' if arm=='R' else ('phase123_20261016' if arm=='C123' else 'phase122_20261015'));stage='R' if arm=='R' else 'C_PixBox';p=base/stage/'scores/PixBox'/pid/'UNLABELLED_POINT_SCORES.json'
 record(p)
 return old.cat_old(arm,pid) if pool=='Catalogue' else old.pix_old(arm,pid)
def label(g):return ','.join(sorted(set(v.split('_T')[-1][:5] for v in g.split(','))))
def signals(pool,gid,products,scores):
 a=scores[0].astype(np.float64);ra=rankdata(a,method='average');out=[]
 for seed,b in zip(SEEDS,scores[1:]):
  b=b.astype(np.float64);rb=rankdata(b,method='average');constant0=bool(a.min()==a.max());constant1=bool(b.min()==b.max());rank=None if constant0 or constant1 else float(1-np.corrcoef(ra,rb)[0,1])
  def h(x):x=np.clip(x,1e-6,1-1e-6);return -x*np.log(x)-(1-x)*np.log1p(-x)
  out.append(dict(pool=pool,group_id=gid,tile=label(products),products=products,seed=seed,n=len(a),constant0=constant0,constant1=constant1,tied0=len(a)-len(np.unique(a)),tied1=len(b)-len(np.unique(b)),RANK=rank,CHANGE=float(np.mean(np.abs(b-a))),ENTROPY=float(np.mean(h(b))-np.mean(h(a)))))
 return out
def first_stage(plan):
 out=[]
 for row in plan:
  if row['pool']!='Catalogue':continue
  pid=row['product_id'];v=[checked_old(a,pid,'Catalogue') for a in ('R','C122','C123')];assert all(np.array_equal(a['flat_indices'],v[0]['flat_indices']) for a in v)
  common=np.logical_and.reduce([a['valid']&np.isfinite(a['scores']) for a in v]);out+=signals('Catalogue',row['group']['group_id'],pid,[a['scores'][common] for a in v])
 points=[{} for _ in range(3)]
 for row in plan:
  if row['pool']!='PixBox':continue
  for arm,d in zip(('R','C122','C123'),points):
   for q in checked_old(arm,row['product_id'],'PixBox'):assert q['point_id'] not in d;d[q['point_id']]=q
 groups=collections.defaultdict(lambda:[[],[],[],set()]);assert all(len(d)==9500 for d in points)
 for pid,q in points[0].items():
  v=[d[pid] for d in points];assert all(all(str(x[k])==str(q[k]) for k in ('product_id','group_id','row20','col20')) for x in v)
  if not all(x['valid'] and x.get('coverage',True) and x.get('author_valid',True) and np.isfinite(x['shadow_confidence']) for x in v):continue
  g=groups[q['group_id']];g[3].add(q['product_id'])
  for i,x in enumerate(v):g[i].append(x['shadow_confidence'])
 for gid,g in groups.items():out+=signals('PixBox',gid,','.join(sorted(g[3])),[np.asarray(x,np.float32) for x in g[:3]])
 table('UNLABELLED_SIGNALS.csv',out);rankings=[]
 for pool,n,k in [('Catalogue',32,7),('PixBox',10,2)]:
  for seed in SEEDS:
   subset=[r for r in out if r['pool']==pool and r['seed']==seed];assert len(subset)==n
   for signal in ('RANK','CHANGE','ENTROPY'):
    ordered=sorted(subset,key=lambda r:(r[signal] is None,-r[signal] if r[signal] is not None else 0,hashlib.sha256(('PHASE204_TIE_V1|'+r['group_id']).encode()).hexdigest()))
    for i,r in enumerate(ordered):rankings.append(dict(pool=pool,seed=seed,signal=signal,group_id=r['group_id'],tile=r['tile'],value=r[signal],rank=i+1 if r[signal] is not None else None,selected=i<k and r[signal] is not None,quota=k,total_groups=n))
 table('ALL_RANKINGS.csv',rankings)
 save('PRE_REFERENCE_SEAL.json',dict(epoch=time.time(),signals_sha256=old.sha(O/'UNLABELLED_SIGNALS.csv'),rankings_sha256=old.sha(O/'ALL_RANKINGS.csv'),reference_join_started=False,historical_exposure_not_blinded=True,position_selection='Catalogue inherited reference-valid grid; PixBox all common image-valid points including ignore',files=FILES.copy()))
 return out,rankings
def metrics(y,s):
 if not y.any() or not (~y).any():return None,None
 return float(average_precision_score(y,s)),float(roc_auc_score(y,s))
def second_stage(plan):
 flows=[];strata=[];subtasks=[];truth=[];positions=[];focusrows=[];support=[]
 for pool,fn in [('Catalogue',old.cat_groups),('PixBox',old.pix_groups)]:
  rows=[r for r in plan if r['pool']==pool];groups,sup=fn(rows);support+=sup
  for gid,g in groups.items():
   tile=label(g['products']);y=g['truth'];p0=g['scores']['shared_R_00'];pred0=p0.astype(np.float64)>=T;target=y==2
   if pool=='Catalogue':
    row=next(r for r in rows if r['group']['group_id']==gid);a=old.cat_old('R',row['product_id']);vv=[old.cat_old(x,row['product_id'])['valid'] for x in ('R','C122','C123')];coords=a['flat_indices'][np.logical_and.reduce(vv)]
   else:
    ref=old.load(R/'data/pixbox_s2_cmix_5036991/phase103_20261013/deduplicated_locations.json');rr={q['point_ids'][0]:q for q in ref};pts={};other=[{} for _ in range(3)]
    for row in rows:
     if row['product_id'] not in g['products'].split(','):continue
     for arm,d in zip(('R','C122','C123'),other):
      for q in old.pix_old(arm,row['product_id']):d[q['point_id']]=q
    for pid,q in other[0].items():
     if q['group_id']==gid and rr[pid]['reference']!='ignore' and all(d[pid]['valid'] and d[pid].get('coverage',True) and d[pid].get('author_valid',True) for d in other):pts[pid]=q
    coords=list(pts.values());assert len(coords)==len(y)
   for seed in SEEDS:
    keys={'00':'shared_R_00',**{s:f'{seed}_{s}' for s in ('10','01','11')}};a0=old.ap(target,p0);s11=g['scores'][keys['11']];ap11=old.ap(target,s11);neg=int((~target).sum());fpr0=float(np.mean(pred0[~target])) if neg else None;fpr11=float(np.mean((s11.astype(np.float64)>=T)[~target])) if neg else None
    truth.append(dict(pool=pool,group_id=gid,tile=tile,seed=seed,positive=int(target.sum()),negative=neg,AP00=a0,AP11=ap11,delta_AP_pp=(ap11-a0)*100 if a0 is not None else None,damaged=(ap11-a0)<-.01 if a0 is not None else None,FPR00=fpr0,FPR11=fpr11,delta_FPR=fpr11-fpr0 if neg else None))
    baseaucs={}
    for state,key in keys.items():
     s=g['scores'][key];pred=s.astype(np.float64)>=T;localflows=[]
     for name,code in [('Surface',0),('Cloud',1),('Shadow',2)]:
      mask=y==code;correct0=pred0==target;correct=pred==target;v=dict(pool=pool,group_id=gid,tile=tile,seed=seed,state=state,reference=name,n=int(mask.sum()),correct_to_wrong=int((mask&correct0&~correct).sum()),wrong_to_correct=int((mask&~correct0&correct).sum()),both_correct=int((mask&correct0&correct).sum()),both_wrong=int((mask&~correct0&~correct).sum()));flows.append(v);localflows.append(v)
      if tile in FOCUS[pool]:
       arr=p0[mask];edges=np.unique(np.quantile(arr,np.arange(0,1.01,.1))) if len(arr) else []
       bins=np.searchsorted(edges[1:-1],arr,side='left') if len(edges)>1 else np.zeros(len(arr),int)
       for b in np.unique(bins):sel=bins==b;delta=s[mask][sel].astype(np.float64)-arr[sel];strata.append(dict(pool=pool,group_id=gid,tile=tile,seed=seed,state=state,reference=name,bin=int(b),count=int(sel.sum()),score00_min=float(arr[sel].min()),score00_max=float(arr[sel].max()),delta_median=float(np.median(delta))))
     if tile in FOCUS[pool]:
      av={}
      for name,c in [('All',None),('Surface',0),('Cloud',1)]:
       mask=np.ones(len(y),bool) if c is None else ((y==2)|(y==c));ap,auc=metrics(target[mask],s[mask]);av[name]=(ap,auc);subtasks.append(dict(pool=pool,group_id=gid,tile=tile,seed=seed,state=state,negative_class=name,positive=int(target[mask].sum()),negative=int((~target[mask]).sum()),AP=ap,AUC=auc))
      if state=='00':baseaucs=av
      sh=localflows[2];sf=localflows[0];cl=localflows[1];focusrows.append(dict(pool=pool,tile=tile,group_id=gid,seed=seed,state=state,AP00=a0,AP=av['All'][0],Surface_AUC_delta=(av['Surface'][1]-baseaucs['Surface'][1]) if av['Surface'][1] is not None else None,Cloud_AUC_delta=(av['Cloud'][1]-baseaucs['Cloud'][1]) if av['Cloud'][1] is not None else None,new_FN=sh['correct_to_wrong'],rescued_TP=sh['wrong_to_correct'],new_FP=sf['correct_to_wrong']+cl['correct_to_wrong'],removed_FP=sf['wrong_to_correct']+cl['wrong_to_correct']))
    if tile in FOCUS[pool]:
     pred=s11.astype(np.float64)>=T
     for code,name in [(2,'Shadow'),(0,'Surface'),(1,'Cloud')]:
      mask=(y==code)&((pred0&~pred) if code==2 else (~pred0&pred));ix=np.flatnonzero(mask);den=int((y==code).sum());chosen=ix[np.unique(np.linspace(0,len(ix)-1,min(12,len(ix))).round().astype(int))] if len(ix) else []
      positions.append(dict(pool=pool,tile=tile,group_id=gid,seed=seed,reference=name,error='new_FN' if code==2 else 'new_FP',count=len(ix),denominator=den,fraction=len(ix)/den if den else None,representatives=[dict(flat_index=int(coords[i]),row20=int(coords[i]//1022),col20=int(coords[i]%1022)) if pool=='Catalogue' else {k:coords[i][k] for k in ('point_id','product_id','row20','col20')} for i in chosen]))
      if pool=='Catalogue':
       # Preserve every new-error original index; deterministic ascending-grid quantiles select examples.
       namefile=f'{seed}_{tile}_{name}_new_errors.npy';np.save(O/namefile,np.asarray(coords)[ix],allow_pickle=False)
   gc.collect()
  del groups;gc.collect()
 table('ALL_GROUP_TRUTH.csv',truth);table('ALL_ERROR_FLOWS.csv',flows);table('REFERENCE_SCORE_STRATA.csv',strata);table('FOCUS_SUBTASKS.csv',subtasks);table('TABLE2_FOCUS.csv',focusrows);save('FOCUS_POSITIONS.json',dict(selection='ascending original flat index / original point order, equally spaced ordinal quantiles; at most12 per reference class and seed',rows=positions));save('COMMON_SUPPORT.json',dict(rows=support));return truth
def warning(out,ranks,truth):
 rows=[];consistency=[]
 for pool,N,k in [('Catalogue',32,7),('PixBox',10,2)]:
  for signal in ('RANK','CHANGE','ENTROPY'):
   for seed in SEEDS:
    rank=[r for r in ranks if r['pool']==pool and r['seed']==seed and r['signal']==signal];tt={r['group_id']:r for r in truth if r['pool']==pool and r['seed']==seed};selected=[r for r in rank if r['selected']];known=[r for r in rank if tt[r['group_id']]['damaged'] is not None and r['value'] is not None];M=sum(r['damaged'] is True for r in tt.values());hit=sum(tt[r['group_id']]['damaged'] is True for r in selected);den=sum(max(0,r['delta_FPR'] or 0) for r in tt.values());num=sum(max(0,tt[r['group_id']]['delta_FPR'] or 0) for r in selected);y=np.array([tt[r['group_id']]['damaged'] for r in known],bool);s=np.array([r['value'] for r in known]);ap,auc=metrics(y,s)
    rows.append(dict(pool=pool,seed=seed,signal=signal,reviewed=len(selected),total=N,AP_damage_total=M,detected=hit,missed=M-hit,selected_known_not_damaged=sum(tt[r['group_id']]['damaged'] is False for r in selected),selected_AP_unknown=sum(tt[r['group_id']]['damaged'] is None for r in selected),random_expected=k*M/N,NA_signal=sum(r['value'] is None for r in rank),conditional_AP_groups=len(known),conditional_damage_base_rate=float(y.mean()) if len(y) else None,conditional_AUROC=auc,conditional_AUPRC=ap,positive_delta_FPR_coverage=num/den if den else None,detected_fixed=','.join(r['tile'] for r in selected if r['tile'] in ('32TML','40XDR','19FDV','54TWN')),missed_tiles=','.join(r['tile'] for r in tt.values() if r['damaged'] is True and not any(x['group_id']==r['group_id'] for x in selected))))
   aa={r['group_id']:r for r in ranks if r['pool']==pool and r['seed']==SEEDS[0] and r['signal']==signal};bb={r['group_id']:r for r in ranks if r['pool']==pool and r['seed']==SEEDS[1] and r['signal']==signal};ids=[g for g in aa if aa[g]['rank'] is not None and bb[g]['rank'] is not None];consistency.append(dict(pool=pool,signal=signal,groups=len(ids),rank_correlation=float(np.corrcoef([aa[g]['rank'] for g in ids],[bb[g]['rank'] for g in ids])[0,1]),top_overlap=sum(aa[g]['selected'] and bb[g]['selected'] for g in ids)))
 table('TABLE1_WARNING.csv',rows);table('SEED_RANK_CONSISTENCY.csv',consistency)
def main():
 O.mkdir(parents=True,exist_ok=True);assert not (O/'INPUT_CODE_LOCK.json').exists()
 record(Path(__file__));record(R/'src/tools/analyze_phase203_local.py');record(R/'src/research_plans/PHASE204_DAMAGE_LOCALIZATION_AND_UNLABELLED_WARNING_20261019.md');record(P/'PROVENANCE.json');record(P/'INPUT_DELIVERY_LOCK.json')
 manifest=old.load(P/'INPUT_DELIVERY_LOCK.json')['manifest']['files']
 for n,h in manifest.items():assert record(P/'delivery'/n)==h,n
 plan=old.load(P/'delivery/EVALUATION_INPUT_PLAN.json')['rows'];save('INPUT_CODE_LOCK.json',dict(files=FILES.copy(),forward=0,fit=0,download=0,threshold=T,location_rule='ascending original flat index / point order, evenly spaced ordinal quantiles per changed reference class, max12',decile_rule='00 within-reference quantiles; duplicate edges merged; equal scores stay together'))
 out,ranks=first_stage(plan);print('signals sealed; joining local references',flush=True);truth=second_stage(plan);warning(out,ranks,truth)
 save('COMPUTE_RECEIPT.json',dict(seconds=time.time()-START,neural_calls=0,fits=0,downloads=0,bootstrap=0,new_bytes=sum(p.stat().st_size for p in O.rglob('*') if p.is_file()),input_files=FILES));print('analysis complete',flush=True)
if __name__=='__main__':main()
