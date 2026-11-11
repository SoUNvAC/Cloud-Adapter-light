"""Reference-local Phase205 six-policy AP/official-decision evaluation, no fits."""
import collections,csv,gc,hashlib,io,json,os,sys,time,zipfile
from pathlib import Path
R=Path(__file__).resolve().parents[2];O=R/'outputs/phase206/phase206_20261019';D=O/'delivery'
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
sys.path.insert(0,str(R/'outputs/phase202/phase202_20261017/software'))
import numpy as np
from sklearn.metrics import average_precision_score,confusion_matrix
from analyze_phase203_local import independent_ap
OLD=R/'outputs/phase205/phase205_20261019/delivery';OLDNAMES=('R4','C4-12201','C4-12301','E4','E4-12201','E4-12301');NAMES=OLDNAMES+tuple(f'{f}_{s}_{t}' for f in ('M','E') for s in (12201,12301) for t in ('10','01'));CHECKS=0
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def save(n,v):
 with (O/n).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def table(n,rows):
 with (O/n).open('x',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def unpack():
 receipt=load(O/'DELIVERY_RECEIPT.json');archive=O/'phase206_delivery.zip';assert archive.stat().st_size==receipt['bytes'] and sha(archive)==receipt['sha256']
 if D.exists():
  lock=load(O/'INPUT_DELIVERY_LOCK.json');manifest=lock['manifest'];assert lock['archive_sha256']==receipt['sha256']
  for n,h in manifest['files'].items():assert sha(D/n)==h
  return manifest
 D.mkdir()
 with zipfile.ZipFile(archive) as z:
  manifest=json.loads(z.read('DELIVERY_MANIFEST.json'));assert len(z.namelist())==len(manifest['files'])+1
  for n,h in manifest['files'].items():
   assert not n.startswith('/') and ':' not in n and '\\' not in n and '..' not in Path(n).parts;b=z.read(n);assert hashlib.sha256(b).hexdigest()==h;npath=D/n;npath.parent.mkdir(parents=True,exist_ok=True)
   with npath.open('xb') as f:f.write(b)
 save('INPUT_DELIVERY_LOCK.json',dict(archive_sha256=receipt['sha256'],manifest=manifest,reference_join_not_yet_performed=True,code_sha256=sha(Path(__file__))))
 return manifest
def cat(rows,sealed,expected):
 groups={};sup=[]
 for row in rows:
  pid=row['product_id'];scores={};pred={};ids=None;valid=None
  for name in NAMES:
   p=(OLD if name in OLDNAMES else D)/'infer'/name/'scores/Catalogue'/(pid+'.npz');assert sha(p)==sealed[name]['Catalogue/'+pid]
   with np.load(p,allow_pickle=False) as z:
    assert set(z.files)=={'scores','valid','flat_indices','raw_class','parent_class'}
    if ids is None:ids=z['flat_indices'].copy();valid=z['valid'].copy()
    else:assert np.array_equal(ids,z['flat_indices']) and np.array_equal(valid,z['valid'])
    assert np.isfinite(z['scores']).all() and np.array_equal(z['parent_class'],np.array([0,1,1,2],np.uint8)[z['raw_class']]);scores[name]=z['scores'][valid].copy();pred[name]=z['parent_class'][valid].copy()
  ex=expected[pid];assert hashlib.sha256(ids.tobytes()).hexdigest()==ex['flat_indices_sha256'];assert hashlib.sha256(valid.tobytes()).hexdigest()==ex['common_valid_sha256'];assert int(valid.sum())==ex['count']
  with zipfile.ZipFile(R/'data/sentinel2_cloud_mask_catalogue_4172871/masks.zip') as z:b=z.read('masks/'+pid+'.npy')
  assert hashlib.sha256(b).hexdigest()==row['group']['mask_member_sha256'];mask=np.load(io.BytesIO(b),allow_pickle=False);assert (mask.reshape(-1,3)[ids].sum(1)==1).all();y=mask.argmax(-1).reshape(-1)[ids][valid].astype(np.uint8);gid=row['group']['group_id'];assert gid not in groups;groups[gid]=dict(products=pid,y=y,scores=scores,pred=pred);sup.append(dict(pool='Catalogue',product_id=pid,positions=len(valid),common_valid=int(valid.sum()),loss=int((~valid).sum()),all_fourteen_identical=True,reference_member_sha256=row['group']['mask_member_sha256']))
 return groups,sup
def pix(rows,sealed,expected):
 maps={name:{} for name in NAMES};support=[]
 for row in rows:
  pid=row['product_id'];lists={}
  for name in NAMES:
   p=(OLD if name in OLDNAMES else D)/'infer'/name/'scores/PixBox'/pid/'UNLABELLED_POINT_SCORES.json';assert sha(p)==sealed[name]['PixBox/'+pid];points=load(p)['points'];lists[name]=points
   for q in points:assert q['point_id'] not in maps[name];maps[name][q['point_id']]=q
  for name in NAMES:assert [q['point_id'] for q in lists[name] if q['valid']]==expected[pid]['common_image_valid_point_ids']
  support.append(dict(pool='PixBox',product_id=pid,positions=len(lists['R4']),common_valid=sum(q['valid'] for q in lists['R4']),loss=sum(not q['valid'] for q in lists['R4']),all_fourteen_identical=True,reference_member_sha256=None))
 assert all(len(d)==9500 for d in maps.values());p=R/'data/pixbox_s2_cmix_5036991/phase103_20261013/deduplicated_locations.json';assert sha(p)=='b1448f3fbd05c9e77af7ed0ea50dbde4be7480a35a4c532a2fec45789af25f96';refs={q['point_ids'][0]:q for q in load(p)};code={'negative_surface':0,'negative_cloud':1,'positive_ground_cloud_shadow':2};groups=collections.defaultdict(lambda:dict(products=set(),y=[],scores={n:[] for n in NAMES},pred={n:[] for n in NAMES}));ignored=0
 for pid,q in maps['R4'].items():
  rr=refs[pid];v=[maps[n][pid] for n in NAMES];assert all(all(str(t[k])==str(q[k])==str(rr[k]) for k in ('product_id','group_id','row20','col20')) and t['valid']==q['valid'] for t in v)
  if not q['valid']:continue
  assert all(np.isfinite(t['shadow_confidence']) and t['parent_class']==[0,1,1,2][t['raw_class']] for t in v)
  if rr['reference']=='ignore':ignored+=1;continue
  g=groups[q['group_id']];g['products'].add(q['product_id']);g['y'].append(code[rr['reference']])
  for n in NAMES:g['scores'][n].append(maps[n][pid]['shadow_confidence']);g['pred'][n].append(maps[n][pid]['parent_class'])
 for g in groups.values():g['products']=','.join(sorted(g['products']));g['y']=np.asarray(g['y'],np.uint8);g['scores']={n:np.asarray(v,np.float32) for n,v in g['scores'].items()};g['pred']={n:np.asarray(v,np.uint8) for n,v in g['pred'].items()}
 assert len(groups)==10 and ignored==931 and sum(int((g['y']==2).sum()) for g in groups.values())==379 and sum(int((g['y']!=2).sum()) for g in groups.values())==8190
 return dict(groups),support
def ap(y,s):
 global CHECKS
 a=float(average_precision_score(y,s)) if y.any() and (~y).any() else None;b=independent_ap(y,s);assert a is None and b is None or abs(a-b)<1e-12;CHECKS+=1;return a
def metric(y,p,s):
 global CHECKS
 mat=np.bincount((3*y+p).astype(np.int64),minlength=9).reshape(3,3);second=confusion_matrix(y,p,labels=[0,1,2]);assert np.array_equal(mat,second);CHECKS+=1
 shadow=y==2;decision=p==2;pos=int(shadow.sum());neg=int((~shadow).sum());tp=int((shadow&decision).sum());fp=int((~shadow&decision).sum());assert tp==mat[2,2] and fp==mat[0,2]+mat[1,2];CHECKS+=2
 return dict(positive=pos,negative=neg,AP=ap(shadow,s),TP=tp,FP=fp,FN=pos-tp,TN=neg-fp,recall=tp/pos if pos else None,negative_shadow_fraction=fp/neg if neg else None,precision=tp/(tp+fp) if tp+fp else None,Shadow_to_Surface=int(mat[2,0]),Shadow_to_Cloud=int(mat[2,1]),confusion=mat.tolist()),mat

def ci(values):
 a=np.asarray(values,np.float64);ix=np.random.default_rng(206019).integers(len(a),size=(10000,len(a)));m=a[ix].mean(1)
 return dict(mean_delta_AP_pp=float(a.mean()),CI95_pp=np.quantile(m,[.025,.975]).tolist(),n_groups=len(a),bootstrap_draws=10000,bootstrap_seed=206019,improved=int((a>1).sum()),stable=int((abs(a)<=1).sum()),damaged=int((a< -1).sum()),worst_delta_AP_pp=float(a.min()))
def names(seed,family):
 return {'00':'R4' if family=='member' else 'E4','11':('C4-' if family=='member' else 'E4-')+str(seed),'10':('M_' if family=='member' else 'E_')+str(seed)+'_10','01':('M_' if family=='member' else 'E_')+str(seed)+'_01'}
def analyze(pool,groups):
 per=[];summary=[];lookup={};summ={}
 for name in NAMES:
  parts=[];matrix=np.zeros((3,3),np.int64)
  for gid,g in groups.items():
   met,mat=metric(g['y'],g['pred'][name],g['scores'][name]);matrix+=mat;rr=dict(pool=pool,strategy=name,group_id=gid,tile=','.join(sorted({p.split('_T')[-1][:5] for p in g['products'].split(',')})),products=g['products'],**met);parts.append(rr);per.append(rr);lookup[name,gid]=rr
  y=np.concatenate([g['y'] for g in groups.values()]);score=np.concatenate([g['scores'][name] for g in groups.values()]);union=matrix.sum(0)+matrix.sum(1)-np.diag(matrix);iou=[float(matrix[i,i]/union[i]) if union[i] else None for i in range(3)];pos=int((y==2).sum());neg=int((y!=2).sum())
  def mean(k):
   vv=[r[k] for r in parts if r[k] is not None]
   return float(np.mean(vv))*100 if vv else None
  rr=dict(pool=pool,strategy=name,groups=len(groups),AP_groups=sum(r['AP'] is not None for r in parts),macro_AP_pp=mean('AP'),pooled_AP_pp=100*ap(y==2,score),macro_recall_pp=mean('recall'),macro_negative_shadow_fraction_pp=mean('negative_shadow_fraction'),max_group_negative_shadow_fraction_pp=100*max(r['negative_shadow_fraction'] for r in parts if r['negative_shadow_fraction'] is not None),macro_precision_pp=mean('precision'),pooled_recall_pp=100*float(matrix[2,2]/pos),pooled_negative_shadow_fraction_pp=100*float((matrix[0,2]+matrix[1,2])/neg),pooled_precision_pp=100*float(matrix[2,2]/matrix[:,2].sum()) if matrix[:,2].sum() else None,Catalogue_pooled_mIoU_pp=float(np.mean([v for v in iou if v is not None]))*100 if pool=='Catalogue' else None,Catalogue_Surface_IoU_pp=iou[0]*100 if pool=='Catalogue' and iou[0] is not None else None,Catalogue_Cloud_IoU_pp=iou[1]*100 if pool=='Catalogue' and iou[1] is not None else None,Catalogue_Shadow_IoU_pp=iou[2]*100 if pool=='Catalogue' and iou[2] is not None else None,positive=pos,negative=neg,zero_recall_AP_groups=sum(r['positive']>0 and r['recall']==0 for r in parts),confusion=matrix.tolist());summary.append(rr);summ[name]=rr;del y,score;gc.collect()
 # Original 00/11 metrics must reproduce 205 exactly before any interpretation.
 old_per={(r['strategy'],r['group_id']):r for r in csv.DictReader((OLD.parent/(pool+'_PER_GROUP.csv')).open(encoding='utf-8-sig'))}
 for name in OLDNAMES:
  for gid in groups:
   a=lookup[name,gid];b=old_per[name,gid]
   for k in ('positive','negative','TP','FP','FN','TN','Shadow_to_Surface','Shadow_to_Cloud'):assert a[k]==int(b[k])
   assert a['AP'] is None and b['AP']=='' or abs(a['AP']-float(b['AP']))<1e-12
 table(pool+'_PER_GROUP.csv',per);save(pool+'_ALL_METRICS.json',summary)
 table1=[];table2=[];intervals=[];table3=[];repeated=[]
 for family in ('member','ensemble'):
  for seed in (12201,12301):
   ns=names(seed,family)
   for state,n in ns.items():table1.append(dict(pool=pool,seed=seed,family=family,state=state,**{k:v for k,v in summ[n].items() if k not in ('pool','confusion')}))
   vals={k:[] for k in ('10-11','01-00','10-00','I')}
   for gid in groups:
    rr={t:lookup[n,gid] for t,n in ns.items()};b=rr['00'];r=dict(pool=pool,group_id=gid,tile=b['tile'],products=b['products'],positive=b['positive'],negative=b['negative'],seed=seed,family=family)
    for t in ('00','10','01','11'):r['AP'+t+'_pp']=100*rr[t]['AP'] if rr[t]['AP'] is not None else None
    if b['AP'] is not None:
     a={t:r['AP'+t+'_pp'] for t in rr};d={'10-11':a['10']-a['11'],'01-00':a['01']-a['00'],'10-00':a['10']-a['00'],'I':a['11']-a['10']-a['01']+a['00']}
    else:d={k:None for k in vals}
    for k,v in d.items():r[k+'_pp']=v;vals[k]+=([v] if v is not None else [])
    for contrast,t,u in [('10-11','10','11'),('01-00','01','00'),('10-00','10','00')]:
     for met in ('recall','negative_shadow_fraction'):
      r[contrast+'_'+met+'_pp']=(rr[t][met]-rr[u][met])*100 if rr[t][met] is not None and rr[u][met] is not None else None
    table2.append(r)
   for k,v in vals.items():intervals.append(dict(pool=pool,family=family,seed=seed,contrast=k,**ci(v)))
  for gid in groups:
   b=lookup[names(12201,family)['00'],gid]
   for contrast,t,u in [('10-11','10','11'),('01-00','01','00'),('10-00','10','00'),('11-00','11','00')]:
    dv=[]
    for seed in (12201,12301):ns=names(seed,family);a,c=lookup[ns[t],gid],lookup[ns[u],gid];dv.append((a['AP']-c['AP'])*100 if a['AP'] is not None else None)
    repeated.append(dict(pool=pool,family=family,group_id=gid,tile=b['tile'],positive=b['positive'],contrast=contrast,seed12201_delta_AP_pp=dv[0],seed12301_delta_AP_pp=dv[1],both_improved=all(v>1 for v in dv) if dv[0] is not None else None,both_damaged=all(v< -1 for v in dv) if dv[0] is not None else None,both_stable=all(abs(v)<=1 for v in dv) if dv[0] is not None else None))
 for state in ('10','11'):
  both=0;versus=[]
  for gid in groups:
   b=lookup['E4',gid];dv=[(lookup[names(seed,'member')[state],gid]['AP']-b['AP'])*100 if b['AP'] is not None else None for seed in (12201,12301)];both+=int(dv[0] is not None and all(v< -1 for v in dv));versus.append(dict(pool=pool,group_id=gid,tile=b['tile'],positive=b['positive'],state=state,seed12201_delta_AP_pp=dv[0],seed12301_delta_AP_pp=dv[1],both_damaged=dv[0] is not None and all(v< -1 for v in dv)))
  table(pool+'_MEMBER_'+state+'_VS_E4_GROUPS.csv',versus)
  for seed in (12201,12301):
   a=summ[names(seed,'member')[state]];b=summ['E4'];row=dict(pool=pool,seed=seed,member_state=state,both_seed_damaged_groups=both,AP_groups=a['AP_groups'])
   for key in ('macro_AP_pp','pooled_AP_pp','macro_recall_pp','macro_negative_shadow_fraction_pp'):row['delta_'+key]=a[key]-b[key]
   v=[r['seed'+str(seed)+'_delta_AP_pp'] for r in versus if r['seed'+str(seed)+'_delta_AP_pp'] is not None];intervals.append(dict(pool=pool,family='member_vs_E4',seed=seed,contrast=state+'-E4',**ci(v)));table3.append(row)
 table(pool+'_TABLE1.csv',table1);table(pool+'_TABLE2.csv',table2);table(pool+'_TABLE3.csv',table3);table(pool+'_REPEATED_CHANGES.csv',repeated);save(pool+'_INTERVALS.json',dict(rows=intervals))

def main():
 start=time.time();manifest=unpack();assert manifest['neural_calls']<=8000;sealed=load(OLD/'infer/SCORES_SEALED.json')['scores'];new=load(D/'infer/SCORES_SEALED.json')['scores'];assert len(new)==8;sealed.update(new);assert set(sealed)==set(NAMES);plan=load(D/'EVALUATION_INPUT_PLAN.json')['rows'];expected={r['product_id']:r for r in load(D/'EXPECTED_SUPPORT.json')['rows']};support=[]
 for pool,fn in [('Catalogue',cat),('PixBox',pix)]:
  if pool=='Catalogue' and (O/'CPU_RECOVERY_LOCK.json').exists():
   recovery=load(O/'CPU_RECOVERY_LOCK.json')
   for n,h in recovery['preserved_Catalogue_outputs'].items():assert sha(O/n)==h
   support+=recovery['Catalogue_support'];print('Catalogue complete first-pass outputs reused unchanged',flush=True);continue
  groups,sup=fn([r for r in plan if r['pool']==pool],sealed,expected);support+=sup;assert len(groups)==(32 if pool=='Catalogue' else 10)
  assert sum(int((g['y']==2).sum()) for g in groups.values())==(655961 if pool=='Catalogue' else 379)
  assert sum(int((g['y']!=2).sum()) for g in groups.values())==(32767527 if pool=='Catalogue' else 8190)
  analyze(pool,groups);del groups;gc.collect();print(pool,'complete',flush=True)
 save('COMMON_SUPPORT_ACCEPTANCE.json',dict(rows=support,all14_identical=True,reference_local_only=True,PixBox_ignored=931,NA_no_shadow_retained_for_negative_statistics=True))
 save('CPU_ACCEPTANCE.json',dict(status='passed',Catalogue_checks_reused_from_first_pass=True,undefined_precision_rule='NA if no predicted shadow; never0 orNaN',independent_AP_confusion_checks_current_pass=CHECKS,seconds=time.time()-start,neural_calls=0,fits=0,bootstrap_seed=206019,bootstrap_draws=10000,code_sha256=sha(Path(__file__)),formal_independent_confirmation=False,original_205_reproduced=True))
if __name__=='__main__':main()
