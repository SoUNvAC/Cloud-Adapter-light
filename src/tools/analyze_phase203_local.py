"""Local references, seven frozen functions; fixed AP decomposition, zero fits."""
import collections,csv,gc,hashlib,io,json,os,sys,time,zipfile
from pathlib import Path
R=Path(__file__).resolve().parents[2];O=R/'outputs/phase203/phase203_20261018';D=O/'delivery';T=.7693686485290528
sys.path.insert(0,str(R/'outputs/phase202/phase202_20261017/software'))
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import numpy as np
from sklearn.metrics import average_precision_score
OLD=R/'outputs/phase122/phase122_20261015';PRE=R/'outputs/phase123/phase123_20261016'
STATES=('shared_R_00','12201_11','12301_11','12201_10','12201_01','12301_10','12301_01')
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for a in iter(lambda:f.read(4*1024**2),b''):h.update(a)
 return h.hexdigest()
def save(name,v):
 with (O/name).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def csvsave(name,rows):
 with (O/name).open('x',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def read(p):
 with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def unpack():
 receipt=load(O/'DELIVERY_RECEIPT.json');p=O/'phase203_delivery.zip';assert p.stat().st_size==receipt['bytes'] and sha(p)==receipt['sha256'];assert not D.exists();D.mkdir()
 with zipfile.ZipFile(p) as z:
  manifest=json.loads(z.read('DELIVERY_MANIFEST.json'));assert len(z.namelist())==len(manifest['files'])+1
  for n,h in manifest['files'].items():
   assert not n.startswith('/') and ':' not in n and '\\' not in n and '..' not in Path(n).parts;b=z.read(n);assert hashlib.sha256(b).hexdigest()==h;dest=D/n;dest.parent.mkdir(parents=True,exist_ok=True)
   with dest.open('xb') as f:f.write(b)
 save('INPUT_DELIVERY_LOCK.json',dict(archive_sha256=receipt['sha256'],manifest=manifest,reference_join_not_yet_performed=True,code_sha256=sha(Path(__file__))))
 return manifest
def cat_old(arm,pid):
 base=R/'data/sentinel2_cloud_mask_catalogue_4172871'/('phase123_20261016' if arm=='C123' else 'phase122_20261015');stage='R_Catalogue_reuse' if arm=='R' else 'C_Catalogue';meta=PRE if arm=='C123' else OLD;p=base/stage/'scores/Catalogue'/(pid+'.npz');assert sha(p)==load(meta/'delivery'/stage/'SCORES_SEALED.json')['scores_sha256']['Catalogue/'+pid]
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in ('scores','valid','flat_indices')}
def cat_groups(rows):
 groups={};supports=[]
 for row in rows:
  pid=row['product_id'];old={a:cat_old(a,pid) for a in ('R','C122','C123')};a=old['R'];common=np.logical_and.reduce([v['valid'] for v in old.values()]);assert all(np.array_equal(v['flat_indices'],a['flat_indices']) for v in old.values());scores={STATES[0]:a['scores'][common],'12201_11':old['C122']['scores'][common],'12301_11':old['C123']['scores'][common]}
  for state in STATES[3:]:
   p=D/state/'R_Catalogue/scores/Catalogue'/(pid+'.npz')
   with np.load(p,allow_pickle=False) as z:assert np.array_equal(z['flat_indices'],a['flat_indices']) and np.array_equal(z['valid'],a['valid']) and np.isfinite(z['scores']).all();scores[state]=z['scores'][common].copy()
  g=row['group'];indices=a['flat_indices'];path=R/'data/sentinel2_cloud_mask_catalogue_4172871/masks.zip'
  with zipfile.ZipFile(path) as z:b=z.read('masks/'+pid+'.npy')
  assert hashlib.sha256(b).hexdigest()==g['mask_member_sha256'];m=np.load(io.BytesIO(b),allow_pickle=False);assert (m.reshape(-1,3)[indices].sum(1)==1).all();y=m.argmax(-1).reshape(-1)[indices][common].astype(np.uint8)
  groups[g['group_id']]=dict(products=pid,truth=y,scores=scores);supports.append(dict(pool='Catalogue',group_id=g['group_id'],original=len(common),common=int(common.sum()),loss=int((~common).sum()),new_states_drop_no_points=True))
 return groups,supports
def pix_old(arm,pid):
 if arm=='R':base=R/'data/pixbox_s2_cmix_5036991/phase119_20261015';meta=R/'outputs/phase119/phase119_20261015';stage='R'
 else:base=R/'data/pixbox_s2_cmix_5036991'/('phase123_20261016' if arm=='C123' else 'phase122_20261015');meta=PRE if arm=='C123' else OLD;stage='C_PixBox'
 p=base/stage/'scores/PixBox'/pid/'UNLABELLED_POINT_SCORES.json';assert sha(p)==load(meta/'delivery'/stage/'SCORES_SEALED.json')['scores_sha256']['PixBox/'+pid];return load(p)['points']
def pix_groups(rows):
 scores={s:{} for s in STATES}
 for row in rows:
  pid=row['product_id']
  for arm,state in [('R','shared_R_00'),('C122','12201_11'),('C123','12301_11')]:
   for q in pix_old(arm,pid):assert q['point_id'] not in scores[state];scores[state][q['point_id']]=q
  for state in STATES[3:]:
   for q in load(D/state/'R_PixBox/scores/PixBox'/pid/'UNLABELLED_POINT_SCORES.json')['points']:scores[state][q['point_id']]=q
 assert all(len(v)==9500 for v in scores.values());refpath=R/'data/pixbox_s2_cmix_5036991/phase103_20261013/deduplicated_locations.json';assert sha(refpath)=='b1448f3fbd05c9e77af7ed0ea50dbde4be7480a35a4c532a2fec45789af25f96';refs={q['point_ids'][0]:q for q in load(refpath)};groups=collections.defaultdict(lambda:dict(products=set(),truth=[],scores={s:[] for s in STATES}));loss=collections.Counter();mapping={'negative_surface':0,'negative_cloud':1,'positive_ground_cloud_shadow':2}
 for pid,q in scores['shared_R_00'].items():
  rr=[scores[s][pid] for s in ('shared_R_00','12201_11','12301_11')];common=all(v['valid'] and v.get('coverage',True) and v.get('author_valid',True) for v in rr)
  for s in STATES[3:]:assert scores[s][pid]['valid']==q['valid'];assert all(str(scores[s][pid][k])==str(q[k]) for k in ('product_id','group_id','row20','col20'))
  r=refs[pid];assert all(str(q[k])==str(r[k]) for k in ('product_id','group_id','row20','col20'));g=groups[q['group_id']];g['products'].add(q['product_id']);loss['original']+=1;loss['input_common']+=int(common)
  if not common:loss['input_loss']+=1;continue
  if r['reference']=='ignore':loss['ignore']+=1;continue
  g['truth'].append(mapping[r['reference']])
  for s in STATES:g['scores'][s].append(scores[s][pid]['shadow_confidence'])
 for g in groups.values():g['products']=','.join(sorted(g['products']));g['truth']=np.array(g['truth'],np.uint8);g['scores']={s:np.array(v,np.float32) for s,v in g['scores'].items()}
 assert sum(int((g['truth']==2).sum()) for g in groups.values())==379;assert sum(int((g['truth']!=2).sum()) for g in groups.values())==8190
 return dict(groups),[dict(pool='PixBox',**loss,negative_denominator=8190,positive=379,no_Phase202_extra_mask=True)]
def ap(y,s):return float(average_precision_score(y,s)) if y.any() and (~y).any() else None
def independent_ap(y,s):
 if not y.any() or not (~y).any():return None
 order=np.argsort(-s,kind='stable');v=s[order];t=y[order];ends=np.r_[np.flatnonzero(v[:-1]!=v[1:]),len(v)-1];tp=np.cumsum(t,dtype=np.int64)[ends];return float(np.dot(np.diff(np.r_[0,tp]),tp/(ends+1))/tp[-1])
def interval(delta):
 v=np.array(delta,np.float64);idx=np.random.default_rng(203018).integers(len(v),size=(10000,len(v)));draw=v[idx].mean(1);ind=np.array([sum(float(v[i]) for i in row)/len(v) for row in idx]);assert np.max(np.abs(draw-ind))<1e-10;return dict(mean_pp=float(v.mean()),CI95_pp=np.quantile(draw,[.025,.975]).tolist(),groups=len(v),improved_gt1pp=int((v>1).sum()),stable_within1pp=int((np.abs(v)<=1).sum()),declined_lt_minus1pp=int((v< -1).sum()),worst_pp=float(v.min()),bootstrap_seed=203018,bootstrap_draws=10000)
def evaluate(pool,groups):
 per=[];summ=[];mechanism=[];intervals=[];checks=0;apby={};yall=np.concatenate([g['truth']==2 for g in groups.values()])
 for seed in (12201,12301):
  keys={'00':'shared_R_00','10':f'{seed}_10','01':f'{seed}_01','11':f'{seed}_11'}
  for state,key in keys.items():
   part=[];allscore=[];apby[seed,state]={}
   for gid,g in groups.items():
    y=g['truth']==2;s=g['scores'][key];a=ap(y,s);a2=independent_ap(y,s);assert a is None and a2 is None or abs(a-a2)<1e-12;checks+=1;apby[seed,state][gid]=a;pred=s.astype(np.float64)>=T;pos=int(y.sum());neg=int((~y).sum());tp=int((pred&y).sum());fp=int((pred&~y).sum())
    row=dict(pool=pool,group_id=gid,products=g['products'],seed=seed,state=state,positive=pos,negative=neg,AP=a,TP=tp,FP=fp,Shadow_to_nonShadow=pos-tp,nonShadow_to_Shadow=fp,recall=tp/pos if pos and neg else None,FPR=fp/neg if neg else None,positive_score_median=float(np.median(s[y])) if pos else None,negative_score_median=float(np.median(s[~y])) if neg else None,threshold=T)
    for label,code in [('Surface',0),('Cloud',1)]:v=g['truth']==code;row[label+'_negative']=int(v.sum());row[label+'_FP']=int((pred&v).sum())
    per.append(row);part.append(row);allscore.append(s)
   aval=[r['AP'] for r in part if r['AP'] is not None];rr=[r['recall'] for r in part if r['recall'] is not None];ff=[r['FPR'] for r in part if r['FPR'] is not None];pooled=np.concatenate(allscore);pa=ap(yall,pooled);ipa=independent_ap(yall,pooled);assert abs(pa-ipa)<1e-12;checks+=1
   summ.append(dict(pool=pool,seed=seed,state=state,groups=len(groups),AP_groups=len(aval),macro_AP_pp=float(np.mean(aval))*100,pooled_AP_pp=pa*100,fixed_threshold_macro_recall_pp=float(np.mean(rr))*100,macro_FPR_pp=float(np.mean(ff))*100,worst_group_FPR_pp=max(ff)*100,groups_FPR_gt1pct=sum(v>.01 for v in ff),zero_recall_groups=sum(v==0 for v in rr),threshold=T))
   del pooled,allscore;gc.collect()
  deltas={k:[] for k in ('11-00','10-11','01-00','10-00','I','W','B')}
  for gid,g in groups.items():
   a={s:apby[seed,s][gid] for s in keys};pos=int((g['truth']==2).sum());out=dict(pool=pool,group_id=gid,products=g['products'],positive=pos,seed=seed,**{'AP'+s+'_pp':v*100 if v is not None else None for s,v in a.items()})
   v={k:None for k in deltas}
   if all(z is not None for z in a.values()):
    v={'11-00':(a['11']-a['00'])*100,'10-11':(a['10']-a['11'])*100,'01-00':(a['01']-a['00'])*100,'10-00':(a['10']-a['00'])*100,'I':(a['11']-a['10']-a['01']+a['00'])*100,'W':((a['10']-a['00'])+(a['11']-a['01']))*50,'B':((a['01']-a['00'])+(a['11']-a['10']))*50};assert abs(v['W']+v['B']-v['11-00'])<1e-10;checks+=1
    for k,z in v.items():deltas[k].append(z)
   out.update(v);mechanism.append(out)
  for k,v in deltas.items():intervals.append(dict(pool=pool,seed=seed,comparison=k,**interval(v),conditional_fixed_functions=True,not_training_causal_share=True))
 # Historical AP and support reproduction, before interpreting new contrasts.
 historical=read(R/'src/research_results/phase123_20261016'/(pool+'_per_group.csv'));arm={'00':'R','11':'C122'}
 for row in per:
  if row['state'] not in ('00','11'):continue
  a='R' if row['state']=='00' else ('C122' if row['seed']==12201 else 'C123');old=next(v for v in historical if v['group_id']==row['group_id'] and v['setting'].startswith(a+'_'));assert int(old['positive'])==row['positive'] and int(old['negative'])==row['negative'];assert old['AP']=='' and row['AP'] is None or abs(float(old['AP'])-row['AP'])<1e-12;checks+=1
 csvsave(pool+'_PER_GROUP.csv',per);csvsave(pool+'_SUMMARY.csv',summ);csvsave(pool+'_MECHANISM.csv',mechanism);save(pool+'_INTERVALS.json',dict(rows=intervals));return checks
def main():
 started=time.time();manifest=unpack();assert manifest['neural_calls']==4865;plan=load(D/'EVALUATION_INPUT_PLAN.json')['rows'];checks=0;support=[]
 for pool,fn in [('Catalogue',cat_groups),('PixBox',pix_groups)]:
  groups,sup=fn([r for r in plan if r['pool']==pool]);support.extend(sup);checks+=evaluate(pool,groups);del groups;gc.collect()
 save('COMMON_SUPPORT_ACCEPTANCE.json',dict(rows=support,no_new_mask_or_drops=True,no_calibration_inputs_read=True,reference_join_local_only=True));save('LOCAL_CPU_ACCEPTANCE.json',dict(status='passed',independent_checks=checks,bootstrap_seed=203018,draws=10000,no_seed_group_inflation=True,fits=0,neural_calls=0,threshold=T,elapsed_seconds=time.time()-started,code_sha256=sha(Path(__file__))))
 print(json.dumps(dict(status='local_evaluation_complete',independent_checks=checks,seconds=time.time()-started)),flush=True)
if __name__=='__main__':main()
