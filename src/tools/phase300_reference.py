"""Phase300 fixed-support readers and independently checked AP/confusion.
Transplanted unchanged reader/metric functions from Phase208 and independent AP
from Phase203; source hashes recorded below. No dependency on untracked scripts.
"""
import collections,csv,gc,hashlib,io,json,os,sys,zipfile
from pathlib import Path
R=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(R/'outputs/phase202/phase202_20261017/software'))
import numpy as np
from sklearn.metrics import average_precision_score,confusion_matrix
O=R/'outputs/phase300/phase300_20261020'
D=O/'remote_backup/phase300_20261020'
OLD=R/'outputs/phase205/phase205_20261019/delivery'
OLDNAMES=('R4','E4');NAMES=('R4','E4','B','M');CHECKS=0

ORIGINAL_SOURCE_SHA256={'analyze_phase208_local.py': '89ac0e6b76ade694f44959bbdd1d5bdfb18d65e6dacff9d5239fdefe8f8caa9a', 'analyze_phase203_local.py': '39e1fc15f8fe46c788c1e5d1d52701a5cf327c01d756639c4b0f1d92c0217344'}

def independent_ap(y,s):
 if not y.any() or not (~y).any():return None
 order=np.argsort(-s,kind='stable');v=s[order];t=y[order];ends=np.r_[np.flatnonzero(v[:-1]!=v[1:]),len(v)-1];tp=np.cumsum(t,dtype=np.int64)[ends];return float(np.dot(np.diff(np.r_[0,tp]),tp/(ends+1))/tp[-1])

def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()

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
  assert hashlib.sha256(b).hexdigest()==row['group']['mask_member_sha256'];mask=np.load(io.BytesIO(b),allow_pickle=False);assert (mask.reshape(-1,3)[ids].sum(1)==1).all();y=mask.argmax(-1).reshape(-1)[ids][valid].astype(np.uint8);gid=row['group']['group_id'];assert gid not in groups;groups[gid]=dict(products=pid,y=y,scores=scores,pred=pred);sup.append(dict(pool='Catalogue',product_id=pid,positions=len(valid),common_valid=int(valid.sum()),loss=int((~valid).sum()),all_four_identical=True,reference_member_sha256=row['group']['mask_member_sha256']))
 return groups,sup

def pix(rows,sealed,expected):
 maps={name:{} for name in NAMES};support=[]
 for row in rows:
  pid=row['product_id'];lists={}
  for name in NAMES:
   p=(OLD if name in OLDNAMES else D)/'infer'/name/'scores/PixBox'/pid/'UNLABELLED_POINT_SCORES.json';assert sha(p)==sealed[name]['PixBox/'+pid];points=load(p)['points'];lists[name]=points
   for q in points:assert q['point_id'] not in maps[name];maps[name][q['point_id']]=q
  for name in NAMES:assert [q['point_id'] for q in lists[name] if q['valid']]==expected[pid]['common_image_valid_point_ids']
  support.append(dict(pool='PixBox',product_id=pid,positions=len(lists['R4']),common_valid=sum(q['valid'] for q in lists['R4']),loss=sum(not q['valid'] for q in lists['R4']),all_four_identical=True,reference_member_sha256=None))
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
