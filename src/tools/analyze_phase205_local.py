"""Reference-local Phase205 six-policy AP/official-decision evaluation, no fits."""
import collections,csv,gc,hashlib,io,json,os,sys,time,zipfile
from pathlib import Path
R=Path(__file__).resolve().parents[2];O=R/'outputs/phase205/phase205_20261019';D=O/'delivery'
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
sys.path.insert(0,str(R/'outputs/phase202/phase202_20261017/software'))
import numpy as np
from sklearn.metrics import average_precision_score,confusion_matrix
from analyze_phase203_local import independent_ap
NAMES=('R4','C4-12201','C4-12301','E4','E4-12201','E4-12301');CHECKS=0
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
 receipt=load(O/'DELIVERY_RECEIPT.json');archive=O/'phase205_delivery.zip';assert archive.stat().st_size==receipt['bytes'] and sha(archive)==receipt['sha256'];assert not D.exists();D.mkdir()
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
   p=D/'infer'/name/'scores/Catalogue'/(pid+'.npz');assert sha(p)==sealed[name]['Catalogue/'+pid]
   with np.load(p,allow_pickle=False) as z:
    assert set(z.files)=={'scores','valid','flat_indices','raw_class','parent_class'}
    if ids is None:ids=z['flat_indices'].copy();valid=z['valid'].copy()
    else:assert np.array_equal(ids,z['flat_indices']) and np.array_equal(valid,z['valid'])
    assert np.isfinite(z['scores']).all() and np.array_equal(z['parent_class'],np.array([0,1,1,2],np.uint8)[z['raw_class']]);scores[name]=z['scores'][valid].copy();pred[name]=z['parent_class'][valid].copy()
  ex=expected[pid];assert hashlib.sha256(ids.tobytes()).hexdigest()==ex['flat_indices_sha256'];assert hashlib.sha256(valid.tobytes()).hexdigest()==ex['common_valid_sha256'];assert int(valid.sum())==ex['count']
  with zipfile.ZipFile(R/'data/sentinel2_cloud_mask_catalogue_4172871/masks.zip') as z:b=z.read('masks/'+pid+'.npy')
  assert hashlib.sha256(b).hexdigest()==row['group']['mask_member_sha256'];mask=np.load(io.BytesIO(b),allow_pickle=False);assert (mask.reshape(-1,3)[ids].sum(1)==1).all();y=mask.argmax(-1).reshape(-1)[ids][valid].astype(np.uint8);gid=row['group']['group_id'];assert gid not in groups;groups[gid]=dict(products=pid,y=y,scores=scores,pred=pred);sup.append(dict(pool='Catalogue',product_id=pid,positions=len(valid),common_valid=int(valid.sum()),loss=int((~valid).sum()),six_strategies_identical=True,reference_member_sha256=row['group']['mask_member_sha256']))
 return groups,sup
def pix(rows,sealed,expected):
 maps={name:{} for name in NAMES};support=[]
 for row in rows:
  pid=row['product_id'];lists={}
  for name in NAMES:
   p=D/'infer'/name/'scores/PixBox'/pid/'UNLABELLED_POINT_SCORES.json';assert sha(p)==sealed[name]['PixBox/'+pid];points=load(p)['points'];lists[name]=points
   for q in points:assert q['point_id'] not in maps[name];maps[name][q['point_id']]=q
  for name in NAMES:assert [q['point_id'] for q in lists[name] if q['valid']]==expected[pid]['common_image_valid_point_ids']
  support.append(dict(pool='PixBox',product_id=pid,positions=len(lists['R4']),common_valid=sum(q['valid'] for q in lists['R4']),loss=sum(not q['valid'] for q in lists['R4']),six_strategies_identical=True,reference_member_sha256=None))
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
def intervals(v):
 global CHECKS
 a=np.asarray(v,np.float64);idx=np.random.default_rng(205019).integers(len(a),size=(10000,len(a)));m=a[idx].mean(1);ind=np.asarray([sum(float(a[i]) for i in rr)/len(rr) for rr in idx]);assert np.max(abs(m-ind))<1e-10;CHECKS+=10000
 return dict(mean_delta_AP_pp=float(a.mean()),CI95_pp=np.quantile(m,[.025,.975]).tolist(),AP_groups=len(a),bootstrap_draws=10000,bootstrap_seed=205019,improved_gt1pp=int((a>1).sum()),stable_within1pp=int((abs(a)<=1).sum()),damaged_lt_minus1pp=int((a< -1).sum()),worst_delta_AP_pp=float(a.min()))
def evaluate(pool,groups):
 per=[];summary=[];base={};matrices={}
 for name in NAMES:
  mat=np.zeros((3,3),np.int64);parts=[];ss=[]
  for gid,g in groups.items():
   met,m=metric(g['y'],g['pred'][name],g['scores'][name]);mat+=m;row=dict(pool=pool,strategy=name,group_id=gid,tile=','.join(sorted(set(p.split('_T')[-1][:5] for p in g['products'].split(',')))),products=g['products'],**met);per.append(row);parts.append(row);base[name,gid]=row;ss.append(g['scores'][name])
  y=np.concatenate([g['y'] for g in groups.values()]);score=np.concatenate(ss);pa=ap(y==2,score);pos=int((y==2).sum());neg=int((y!=2).sum());union=mat.sum(0)+mat.sum(1)-np.diag(mat);iou=[float(mat[i,i]/union[i]) if union[i] else None for i in range(3)];matrices[name]=mat.tolist()
  summary.append(dict(pool=pool,strategy=name,groups=len(groups),AP_groups=sum(r['AP'] is not None for r in parts),macro_AP_pp=100*float(np.mean([r['AP'] for r in parts if r['AP'] is not None])),pooled_AP_pp=pa*100,official_decision_macro_recall_pp=100*float(np.mean([r['recall'] for r in parts if r['recall'] is not None])),macro_negative_shadow_fraction_pp=100*float(np.mean([r['negative_shadow_fraction'] for r in parts if r['negative_shadow_fraction'] is not None])),pooled_recall_pp=float(mat[2,2]/pos)*100 if pos else None,pooled_negative_shadow_fraction_pp=float((mat[0,2]+mat[1,2])/neg)*100 if neg else None,pooled_precision_pp=float(mat[2,2]/mat[:,2].sum())*100 if mat[:,2].sum() else None,Catalogue_pooled_mIoU_pp=float(np.mean([v for v in iou if v is not None]))*100 if pool=='Catalogue' else None,Catalogue_Surface_IoU_pp=iou[0]*100 if pool=='Catalogue' and iou[0] is not None else None,Catalogue_Cloud_IoU_pp=iou[1]*100 if pool=='Catalogue' and iou[1] is not None else None,Catalogue_Shadow_IoU_pp=iou[2]*100 if pool=='Catalogue' and iou[2] is not None else None,positive=pos,negative=neg,zero_recall_AP_groups=sum(r['positive']>0 and r['recall']==0 for r in parts),no_reference_shadow_groups=sum(r['positive']==0 for r in parts),decision='raw4 argmax then parent merge; no threshold'))
  del score,ss,y;gc.collect()
 comparisons=[];bootstrap=[]
 for family,b,c1,c2 in [('member','R4','C4-12201','C4-12301'),('ensemble','E4','E4-12201','E4-12301')]:
  deltas={12201:[],12301:[]}
  for gid,g in groups.items():
   a=base[b,gid];r1=base[c1,gid];r2=base[c2,gid];d1=(r1['AP']-a['AP'])*100 if a['AP'] is not None else None;d2=(r2['AP']-a['AP'])*100 if a['AP'] is not None else None
   if d1 is not None:deltas[12201].append(d1);deltas[12301].append(d2)
   comparisons.append(dict(pool=pool,family=family,group_id=gid,tile=a['tile'],positive=a['positive'],negative=a['negative'],baseline_AP=a['AP'],seed12201_delta_AP_pp=d1,seed12301_delta_AP_pp=d2,both_damage=(d1< -1 and d2< -1) if d1 is not None else None,seed12201_delta_recall_pp=(r1['recall']-a['recall'])*100 if a['recall'] is not None else None,seed12301_delta_recall_pp=(r2['recall']-a['recall'])*100 if a['recall'] is not None else None,seed12201_delta_negative_shadow_pp=(r1['negative_shadow_fraction']-a['negative_shadow_fraction'])*100 if a['negative_shadow_fraction'] is not None else None,seed12301_delta_negative_shadow_pp=(r2['negative_shadow_fraction']-a['negative_shadow_fraction'])*100 if a['negative_shadow_fraction'] is not None else None))
  for seed,v in deltas.items():bootstrap.append(dict(pool=pool,family=family,seed=seed,**intervals(v)))
 table(pool+'_PER_GROUP.csv',per);table(pool+'_SUMMARY.csv',summary);table(pool+'_REPEATED_DAMAGE.csv',comparisons);save(pool+'_INTERVALS.json',dict(rows=bootstrap));save(pool+'_CONFUSION.json',dict(matrices=matrices,rows='reference Surface Cloud Shadow',cols='official prediction Surface Cloud Shadow',Catalogue_IoU_rule='sum confusion across valid pixels; union-zero classes excluded from mean; no absent class filled0',PixBox='reference point confusion only, not full-scene mIoU/FPR'))
def main():
 start=time.time();manifest=unpack();assert manifest['neural_calls']==6600;sealed=load(D/'infer/SCORES_SEALED.json')['scores'];plan=load(D/'EVALUATION_INPUT_PLAN.json')['rows'];expected={r['product_id']:r for r in load(D/'EXPECTED_SUPPORT.json')['rows']};support=[]
 for pool,fn in [('Catalogue',cat),('PixBox',pix)]:
  groups,sup=fn([r for r in plan if r['pool']==pool],sealed,expected);assert len(groups)==(32 if pool=='Catalogue' else 10);support+=sup;evaluate(pool,groups);del groups;gc.collect();print(pool,'evaluated',flush=True)
 save('COMMON_SUPPORT_ACCEPTANCE.json',dict(rows=support,PixBox_reference_positive=379,PixBox_reference_negative=8190,PixBox_ignore=931,no_Phase202_extra_mask=True,reference_only_local=True,all_six_same_support=True))
 save('CPU_ACCEPTANCE.json',dict(status='passed',independent_checks=CHECKS,seconds=time.time()-start,neural_calls=0,fits=0,bootstrap_seed=205019,draws=10000,code_sha256=sha(Path(__file__)),no_threshold_search=True,official_fourclass_argmax_then_parent=True));print(json.dumps(dict(status='complete',checks=CHECKS,seconds=time.time()-start)),flush=True)
if __name__=='__main__':main()
