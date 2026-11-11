"""Catalogue-only reconstruction, fixed five rows and paired pooled mIoU."""
import ast,csv,gc,hashlib,io,json,os,sys,time,zipfile
from pathlib import Path
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import numpy as np
R=Path(__file__).resolve().parents[2];O=R/'outputs/phase210/phase210_20261020';P=R/'outputs/phase208/phase208_20261020';D=P/'delivery';OLD=R/'outputs/phase205/phase205_20261019/delivery';Q=R/'outputs/phase209/phase209_20261020';NAMES=['R4','R4_calibrated','M','B','E4']
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def save(n,d):
 with (O/n).open('x',encoding='utf8') as f:json.dump(d,f,indent=2,allow_nan=False)
def csvsave(n,rows):
 with (O/n).open('x',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def iou(m):
 m=np.asarray(m,np.int64);u=m.sum(-2)+m.sum(-1)-np.diagonal(m,axis1=-2,axis2=-1);out=np.full(u.shape,np.nan);np.divide(np.diagonal(m,axis1=-2,axis2=-1),u,out=out,where=u>0);return out,np.nanmean(out,axis=-1)
def assemble(old,shadow,positions,conditional):
 missing=np.flatnonzero((old==2)&~shadow);assert np.array_equal(positions,missing) and len(conditional)==len(missing) and np.isin(conditional,[0,1]).all();new=np.where(shadow,2,old).astype(np.uint8);new[positions]=conditional;assert np.array_equal(new==2,shadow);assert np.array_equal(new[(old!=2)&~shadow],old[(old!=2)&~shadow]);return new
def test():
 old=np.array([2,1,0,2],np.uint8);shadow=np.array([False,True,False,True]);assert assemble(old,shadow,np.array([0]),np.array([1],np.uint8)).tolist()==[1,2,0,2]
 logits=np.array([[2,1],[1.9,1],[1.9,1],[3,4]],np.float32);raw=np.argmax(logits[:3],axis=0);assert raw.tolist()==[0,0] and np.array([0,1,1],np.uint8)[raw].tolist()==[0,0]
 x=np.array([.5],np.float32).astype(np.float64);assert not (x>=np.nextafter(.5,np.inf)).any() and (x>=.5).all()
 vals,mi=iou(np.diag([10,0,0]));assert mi==1 and np.isnan(vals[1:]).all();vals,mi=iou(np.array([[9,0,1],[0,0,0],[0,0,0]]));assert vals[0]==.9 and vals[2]==0 and mi==.45
 print('conditional raw argmax, addedShadow, float64 boundary and union-zero tests passed')
def product(q,ref=True):
 pid=q['product_id'];ex={r['product_id']:r for r in load(D/'EXPECTED_SUPPORT.json')['rows'] if r['pool']=='Catalogue'}[pid];seals=load(OLD/'infer/SCORES_SEALED.json')['scores'];seals.update(load(D/'infer/SCORES_SEALED.json')['scores']);preds={};flat=None;valid=None;score=None
 for name in ['R4','E4','M','B']:
  p=(OLD if name in ['R4','E4'] else D)/'infer'/name/'scores/Catalogue'/(pid+'.npz');assert sha(p)==seals[name]['Catalogue/'+pid]
  with np.load(p,allow_pickle=False) as z:
   if flat is None:flat=z['flat_indices'];valid=z['valid'];score=z['scores'][valid].astype(np.float64)
   else:assert np.array_equal(flat,z['flat_indices']) and np.array_equal(valid,z['valid'])
   assert np.array_equal(z['parent_class'],np.array([0,1,1,2],np.uint8)[z['raw_class']]);preds[name]=z['parent_class'][valid].copy()
 assert hashlib.sha256(flat.tobytes()).hexdigest()==ex['flat_indices_sha256'] and hashlib.sha256(valid.tobytes()).hexdigest()==ex['common_valid_sha256'] and valid.sum()==ex['count'];assert sha(Q/'THRESHOLD_FROZEN.json')==load(O/'MISSING_PLAN.json')['threshold_source_sha256']
 t=load(O/'THRESHOLD_REFERENCE.json')['threshold'];shadow=score>=np.float64(t);assert hashlib.sha256(shadow.tobytes()).hexdigest()==q['new_shadow_mask_sha256']
 with np.load(O/q['missing_file'],allow_pickle=False) as z:missing=z['common_positions'];assert np.array_equal(flat[valid][missing],z['flat_indices'])
 if len(missing):
  seal=load(O/'recovery/RECOVERY_SEALED.json');rec=next(r for r in seal['rows'] if r['product_id']==pid);p=O/'recovery'/rec['file'];assert sha(p)==rec['sha256'] and rec['default_raw4_mismatch']==0 and rec['max_shadow_score_error']<=1e-5
  with np.load(p,allow_pickle=False) as z:
   assert np.array_equal(z['common_positions'],missing) and np.array_equal(z['flat_indices'],flat[valid][missing]);assert np.array_equal(z['conditional_parent'],np.array([0,1,1],np.uint8)[z['conditional_raw']]);assert np.array_equal(z['cached_shadow_scores'].astype(np.float64),score[missing]);conditional=z['conditional_parent'].copy()
 else:conditional=np.empty(0,np.uint8)
 preds['R4_calibrated']=assemble(preds['R4'],shadow,missing,conditional)
 if not ref:return preds,flat[valid],shadow
 with zipfile.ZipFile(R/'data/sentinel2_cloud_mask_catalogue_4172871/masks.zip') as z:b=z.read('masks/'+pid+'.npy')
 assert hashlib.sha256(b).hexdigest()==q['group']['mask_member_sha256'];mask=np.load(io.BytesIO(b),allow_pickle=False).reshape(-1,3)[flat[valid]];assert (mask.sum(1)==1).all();y=np.argmax(mask,axis=1).astype(np.uint8)
 return preds,y,shadow
def main():
 start=time.time();plan=load(O/'MISSING_PLAN.json');tlock=sha(O/'THRESHOLD_REFERENCE.json');manifest=load(O/'recovery/DELIVERY_MANIFEST.json')
 for n,h in manifest['files'].items():assert sha(O/'recovery'/n)==h
 seal=load(O/'recovery/RECOVERY_SEALED.json');assert seal['parameters_buffers_unchanged'] and seal['all_eval'] and len(seal['rows'])==24
 prior={(q['strategy'],q['group_id']):q for q in csv.DictReader((P/'Catalogue_PER_GROUP.csv').open(encoding='utf-8-sig'))};b209={q['group_id']:q for q in csv.DictReader((Q/'B_GROUPS.csv').open(encoding='utf-8-sig')) if q['pool']=='Catalogue' and q['strategy']=='R4_calibrated'};per=[];flows=[];masks=[];predhash=[]
 for q in plan['rows']:
  preds,y,shadow=product(q);gid=q['group']['group_id'];old=preds['R4'];new=preds['R4_calibrated'];assert int(((old==2)&~shadow).sum())==q['missing_positions'] and int(((old!=2)&shadow).sum())==q['added_shadow_positions'];classes=['Surface','Cloud','Shadow'];flow=np.bincount((y.astype(np.int64)*9+old*3+new),minlength=27).reshape(3,3,3)
  for i in range(3):
   for j in range(3):
    for k in range(3):flows.append(dict(group_id=gid,product=q['product_id'],reference=classes[i],old_prediction=classes[j],new_prediction=classes[k],count=int(flow[i,j,k])))
  for name in NAMES:
   pr=preds[name];mat=np.bincount(y*3+pr,minlength=9).reshape(3,3);vals,mi=iou(mat);pos=int((y==2).sum());neg=int((y!=2).sum());tp=int(mat[2,2]);fp=int(mat[:2,2].sum());row=dict(strategy=name,group_id=gid,tile=q['product_id'].split('_T')[-1][:5],product=q['product_id'],count=len(y),positive=pos,negative=neg,TP=tp,FP=fp,FN=pos-tp,TN=neg-fp,recall=tp/pos if pos else None,negative_shadow_fraction=fp/neg,Surface_IoU=vals[0] if np.isfinite(vals[0]) else None,Cloud_IoU=vals[1] if np.isfinite(vals[1]) else None,Shadow_IoU=vals[2] if np.isfinite(vals[2]) else None,mIoU=float(mi),confusion=mat.tolist());per.append(row)
   if name!='R4_calibrated':assert mat.tolist()==ast.literal_eval(prior[name,gid]['confusion'])
   else:
    for k in ['TP','FP','FN','TN','positive','negative']:assert row[k]==float(b209[gid][k])
   predhash.append(dict(strategy=name,group_id=gid,prediction_sha256=hashlib.sha256(pr.tobytes()).hexdigest()))
  masks.append(dict(group_id=gid,product=q['product_id'],new_shadow_mask_sha256=hashlib.sha256(shadow.tobytes()).hexdigest(),removed_shadow=q['missing_positions'],added_shadow=q['added_shadow_positions'],common_support=len(y),all_positions_resolved=True));del preds,y,shadow;gc.collect()
 summary=[];matrices={};groupmis={}
 for name in NAMES:
  rr=[q for q in per if q['strategy']==name];mm=np.array([q['confusion'] for q in rr],np.int64);matrices[name]=mm;groupmis[name]=np.array([q['mIoU'] for q in rr]);m=mm.sum(0);vals,mi=iou(m);summary.append(dict(strategy=name,groups=32,macro_mIoU_pp=float(groupmis[name].mean())*100,pooled_mIoU_pp=float(mi)*100,pooled_Surface_IoU_pp=float(vals[0])*100,pooled_Cloud_IoU_pp=float(vals[1])*100,pooled_Shadow_IoU_pp=float(vals[2])*100,macro_recall_pp=float(np.mean([q['recall'] for q in rr if q['recall'] is not None]))*100,macro_negative_pp=float(np.mean([q['negative_shadow_fraction'] for q in rr]))*100,confusion=m.tolist()))
 oldsummary=load(P/'Catalogue_ALL_METRICS.json')
 for q in summary:
  if q['strategy']!='R4_calibrated':assert abs(q['pooled_mIoU_pp']-next(s for s in oldsummary if s['strategy']==q['strategy'])['Catalogue_pooled_mIoU_pp'])<1e-10
  else:
   old209=next(s for s in load(Q/'TABLE_B.json') if s['pool']=='Catalogue' and s['strategy']=='R4_calibrated');assert abs(q['pooled_Shadow_IoU_pp']-old209['Shadow_IoU_pp'])<1e-10 and abs(q['macro_recall_pp']-old209['macro_recall_pp'])<1e-10 and abs(q['macro_negative_pp']-old209['macro_negative_pp'])<1e-10
 samples=np.random.default_rng(210020).integers(32,size=(10000,32));boot={n:iou(matrices[n][samples].sum(1))[1]*100 for n in NAMES};table2=[];diff=[]
 for right in ['M','B','R4']:
  v=(groupmis['R4_calibrated']-groupmis[right])*100;pooled=next(q['pooled_mIoU_pp'] for q in summary if q['strategy']=='R4_calibrated')-next(q['pooled_mIoU_pp'] for q in summary if q['strategy']==right);worst=int(v.argmin());q=plan['rows'][worst];table2.append(dict(contrast='R4_calibrated-'+right,pooled_delta_mIoU_pp=pooled,paired_group_CI95_pp=np.quantile(boot['R4_calibrated']-boot[right],[.025,.975]).tolist(),macro_delta_mIoU_pp=float(v.mean()),worst_group=q['group']['group_id'],worst_product=q['product_id'],worst_tile=q['product_id'].split('_T')[-1][:5],worst_delta_mIoU_pp=float(v[worst]),groups=32,bootstrap_draws=10000,bootstrap_seed=210020,threshold_fixed=True))
  for i,q in enumerate(plan['rows']):diff.append(dict(contrast='R4_calibrated-'+right,group_id=q['group']['group_id'],product=q['product_id'],delta_mIoU_pp=float(v[i])))
 csvsave('PER_GROUP.csv',per);csvsave('ERROR_FLOWS.csv',flows);csvsave('GROUP_DIFFERENCES.csv',diff);csvsave('TABLE1.csv',summary);save('TABLE1.json',summary);csvsave('TABLE2.csv',table2);save('TABLE2.json',table2);save('RECONSTRUCTION_ACCEPTANCE.json',dict(status='passed',rows=masks,prediction_hashes=predhash,all32_default_four_confusions_reproduced=True,all209_binary_counts_reproduced=True,all_missing_resolved=True,all_changed_positions_traceable=True,threshold_source_unchanged=sha(O/'THRESHOLD_REFERENCE.json')==tlock,absent_class_rule='union==0 excluded; reference-absent but predicted-present union>0 counts IoU0',pooled_iou_from_summed_confusions=True,bootstrap_pooled_recomputed_each_draw=True,AP_original_scores_only=True,no_calibration_or_PixBox_inputs_read=True,seconds=time.time()-start,code_sha256=sha(Path(__file__))))
 print(json.dumps(dict(status='analyzed',seconds=time.time()-start,table1=summary,table2=table2)),flush=True)
if __name__=='__main__':test() if '--test' in sys.argv else main()
