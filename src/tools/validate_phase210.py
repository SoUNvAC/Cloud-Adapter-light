"""Independent Catalogue reconstruction and paired pooled-confusion bootstrap."""
import ast,csv,hashlib,io,json,sys,time,zipfile
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[2];O=R/'outputs/phase210/phase210_20261020';P=R/'outputs/phase208/phase208_20261020';D=P/'delivery';OLD=R/'outputs/phase205/phase205_20261019/delivery';Q=R/'outputs/phase209/phase209_20261020'
sys.path.insert(0,str(R/'outputs/phase202/phase202_20261017/software'))
from sklearn.metrics import confusion_matrix
CHECKS=0
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def check(v):
 global CHECKS
 assert v
 CHECKS+=1
def rows(n):
 with (O/n).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def scalar_iou(m):
 out=[]
 for i in range(3):
  union=int(m[i,:].sum()+m[:,i].sum()-m[i,i]);out.append(float(m[i,i]/union) if union else None)
 return out,sum(x for x in out if x is not None)/sum(x is not None for x in out)
def pooled_values(m):
 diag=np.stack([m[:,i,i] for i in range(3)],axis=1);union=m.sum(1)+m.sum(2)-diag;value=np.zeros(union.shape,np.float64);np.divide(diag,union,out=value,where=union>0);return value.sum(1)/(union>0).sum(1)*100
def main():
 start=time.time();plan=load(O/'MISSING_PLAN.json');seal=load(O/'recovery/RECOVERY_SEALED.json');check(sha(Q/'THRESHOLD_FROZEN.json')==plan['threshold_source_sha256']==sha(O/'THRESHOLD_REFERENCE.json'));t=load(O/'THRESHOLD_REFERENCE.json')['threshold'];check(t==0.5895397663116456);lookup={(r['strategy'],r['group_id']):r for r in rows('PER_GROUP.csv')};flowlookup={(r['group_id'],r['reference'],r['old_prediction'],r['new_prediction']):int(r['count']) for r in rows('ERROR_FLOWS.csv')};names=['R4','R4_calibrated','M','B','E4'];matrices={n:[] for n in names};mis={n:[] for n in names};r209={r['group_id']:r for r in csv.DictReader((Q/'B_GROUPS.csv').open(encoding='utf-8-sig')) if r['pool']=='Catalogue' and r['strategy']=='R4_calibrated'};oldper={(r['strategy'],r['group_id']):r for r in csv.DictReader((P/'Catalogue_PER_GROUP.csv').open(encoding='utf-8-sig'))};classes=['Surface','Cloud','Shadow'];needed=set()
 seals=load(OLD/'infer/SCORES_SEALED.json')['scores'];seals.update(load(D/'infer/SCORES_SEALED.json')['scores'])
 with zipfile.ZipFile(R/'data/sentinel2_cloud_mask_catalogue_4172871/masks.zip') as zmask:
  for q in plan['rows']:
   pid=q['product_id'];gid=q['group']['group_id'];pred={};s=None;ids=None;valid=None
   for n in ['R4','M','B','E4']:
    f=(OLD if n in ['R4','E4'] else D)/'infer'/n/'scores/Catalogue'/(pid+'.npz');check(sha(f)==seals[n]['Catalogue/'+pid])
    with np.load(f,allow_pickle=False) as z:
     if n=='R4':ids=z['flat_indices'];valid=z['valid'];s=z['scores'][valid].astype(np.float64)
     else:check(np.array_equal(ids,z['flat_indices']) and np.array_equal(valid,z['valid']))
     pred[n]=z['parent_class'][valid].copy()
   old=pred['R4'];shadow=s>=np.float64(t);mask=(old==2)&~shadow;missing=np.flatnonzero(mask);new=np.full(len(old),-1,np.int16);new[shadow]=2;unchanged=(old!=2)&~shadow;new[unchanged]=old[unchanged]
   if mask.any():
    needed.add(pid);rec=next(r for r in seal['rows'] if r['product_id']==pid);f=O/'recovery'/rec['file'];check(sha(f)==rec['sha256'] and rec['default_raw4_mismatch']==0 and rec['max_shadow_score_error']<=1e-5)
    with np.load(f,allow_pickle=False) as z:
     check(np.array_equal(z['common_positions'],missing) and np.array_equal(z['flat_indices'],ids[valid][missing]));raw=z['conditional_raw'];check(np.isin(raw,[0,1,2]).all());parent=(raw!=0).astype(np.int16);check(np.array_equal(parent,z['conditional_parent']));new[missing]=parent
   check((new>=0).all() and np.array_equal(new==2,shadow) and np.array_equal(new[unchanged],old[unchanged]));check(int(mask.sum())==q['missing_positions'] and int(((old!=2)&shadow).sum())==q['added_shadow_positions']);pred['R4_calibrated']=new
   b=zmask.read('masks/'+pid+'.npy');check(hashlib.sha256(b).hexdigest()==q['group']['mask_member_sha256']);label=np.load(io.BytesIO(b),allow_pickle=False).reshape(-1,3)[ids[valid]];check((label.sum(1)==1).all());y=label.argmax(1)
   for n in names:
    mat=confusion_matrix(y,pred[n],labels=[0,1,2]);check(mat.tolist()==ast.literal_eval(lookup[n,gid]['confusion']));vals,mi=scalar_iou(mat);check(abs(mi-float(lookup[n,gid]['mIoU']))<1e-12);matrices[n].append(mat);mis[n].append(mi)
    for i,k in enumerate(['Surface_IoU','Cloud_IoU','Shadow_IoU']):check(lookup[n,gid][k]=='' if vals[i] is None else abs(vals[i]-float(lookup[n,gid][k]))<1e-12)
    if n!='R4_calibrated':check(mat.tolist()==ast.literal_eval(oldper[n,gid]['confusion']))
    else:
     check(int(mat[2,2])==int(float(r209[gid]['TP'])) and int(mat[:2,2].sum())==int(float(r209[gid]['FP'])));check(int(mat[2,:].sum()-mat[2,2])==int(float(r209[gid]['FN'])) and int(mat[:2,:2].sum())==int(float(r209[gid]['TN'])))
   for i in range(3):
    actual=confusion_matrix(old[y==i],new[y==i],labels=[0,1,2])
    for j in range(3):
     for k in range(3):check(int(actual[j,k])==flowlookup[gid,classes[i],classes[j],classes[k]])
 check(needed=={r['product_id'] for r in seal['rows']} and len(needed)==24 and sum(q['missing_positions'] for q in plan['rows'])==172518)
 # Different bootstrap implementation: frequency weights and einsum, no per-group IoU substitution.
 indices=np.random.default_rng(210020).integers(32,size=(10000,32));weights=np.zeros((10000,32),np.int64)
 for i in range(32):np.add.at(weights,(np.arange(10000),indices[:,i]),1)
 boot={};totals={}
 for n in names:
  mm=np.asarray(matrices[n],np.int64);tot=mm.sum(0);totals[n]=scalar_iou(tot)[1]*100;boot[n]=pooled_values(np.einsum('dg,gij->dij',weights,mm));row=next(r for r in load(O/'TABLE1.json') if r['strategy']==n);check(abs(totals[n]-row['pooled_mIoU_pp'])<1e-10 and abs(np.mean(mis[n])*100-row['macro_mIoU_pp'])<1e-10)
 for q in load(O/'TABLE2.json'):
  right=q['contrast'].removeprefix('R4_calibrated-');check(abs(totals['R4_calibrated']-totals[right]-q['pooled_delta_mIoU_pp'])<1e-10);check(np.max(np.abs(np.quantile(boot['R4_calibrated']-boot[right],[.025,.975])-q['paired_group_CI95_pp']))<1e-10);v=100*(np.asarray(mis['R4_calibrated'])-np.asarray(mis[right]));check(abs(v.mean()-q['macro_delta_mIoU_pp'])<1e-10 and abs(v.min()-q['worst_delta_mIoU_pp'])<1e-10 and plan['rows'][int(v.argmin())]['group']['group_id']==q['worst_group'])
 events=[json.loads(s) for s in (O/'recovery/CALL_LEDGER.jsonl').read_text().splitlines()];check([e['call'] for e in events]==list(range(1,len(events)+1)));check(sum(e['kind']=='R4_conditional_recovery' for e in events)==96 and len(events)==seal['total_calls']<=512);status=load(O/'recovery/RUN_STATUS.json');check(status['seconds']<=3600 and status['cpu_peak_bytes']<=16*1024**3 and status['gpu_peak_bytes']<=20*1024**3 and status['minimum_host_free_bytes']>=4000000000);check(seal['before']==seal['after'] and seal['parameters_buffers_unchanged'] and seal['all_eval']);check(seal['model_sha256']=='7f6e4202e17ee73efa4aba7abb5c34f4f90a9f7eb42480820714994dff2db660')
 out=dict(status='passed',checks=CHECKS,seconds=time.time()-start,code_sha256=sha(Path(__file__)),independent_sklearn_confusions=True,independent_conditional_reconstruction=True,independent_frequency_weighted_pooled_bootstrap=True,all32_groups=True,all209_binary_counts_exact=True,raw4_reproduction_from_sealed_remote_receipts=True,no_calibration_or_PixBox_inputs_read=True,neural_calls_added=0,missing_positions_resolved=172518)
 with (O/'INDEPENDENT_ACCEPTANCE.json').open('x') as f:json.dump(out,f,indent=2)
 print(json.dumps(out),flush=True)
if __name__=='__main__':main()
