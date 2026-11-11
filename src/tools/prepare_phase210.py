"""Image/label-free missing-conditional-class plan from sealed score caches."""
import csv,hashlib,json,shutil,zipfile
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[2];O=R/'outputs/phase210/phase210_20261020';P=R/'outputs/phase208/phase208_20261020/delivery';OLD=R/'outputs/phase205/phase205_20261019/delivery'
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def save(n,d):
 with (O/n).open('x',encoding='utf8') as f:json.dump(d,f,indent=2,allow_nan=False)
def main():
 O.mkdir(exist_ok=False,parents=True);(O/'missing_positions').mkdir();tpath=R/'outputs/phase209/phase209_20261020/THRESHOLD_FROZEN.json';t=load(tpath)['threshold'];assert t==0.5895397663116456
 shutil.copyfile(tpath,O/'THRESHOLD_REFERENCE.json');shutil.copyfile(P/'V4_WEIGHT_LOCK.json',O/'V4_WEIGHT_LOCK.json')
 plan=[q for q in load(P/'EVALUATION_INPUT_PLAN.json')['rows'] if q['pool']=='Catalogue'];expected={q['product_id']:q for q in load(P/'EXPECTED_SUPPORT.json')['rows'] if q['pool']=='Catalogue'};seals=load(OLD/'infer/SCORES_SEALED.json')['scores']['R4'];rows=[];cacheaudit=[]
 for q in plan:
  pid=q['product_id'];p=OLD/'infer/R4/scores/Catalogue'/(pid+'.npz');assert sha(p)==seals['Catalogue/'+pid]
  with np.load(p,allow_pickle=False) as z:
   assert set(z.files)=={'scores','valid','flat_indices','raw_class','parent_class'};flat=z['flat_indices'];valid=z['valid'];slots=np.flatnonzero(valid);commonflat=flat[valid];old=z['parent_class'][valid];score=z['scores'][valid].astype(np.float64);raw=z['raw_class'][valid]
   ex=expected[pid];assert hashlib.sha256(flat.tobytes()).hexdigest()==ex['flat_indices_sha256'] and hashlib.sha256(valid.tobytes()).hexdigest()==ex['common_valid_sha256'] and len(slots)==ex['count'];assert np.array_equal(old,np.array([0,1,1,2],np.uint8)[raw])
   new=score>=np.float64(t);missing=np.flatnonzero((old==2)&~new);added=(old!=2)&new;target=O/'missing_positions'/(pid+'.npz');np.savez_compressed(target,common_positions=missing,original_slots=slots[missing],flat_indices=commonflat[missing])
   row=dict(**q,cache_relative='infer/R4/scores/Catalogue/'+pid+'.npz',cache_sha256=sha(p),cache_remote_path='/home/scv/shared/phase205_20261019/infer/R4/scores/Catalogue/'+pid+'.npz',support_count=len(slots),support_flat_indices_sha256=hashlib.sha256(commonflat.tobytes()).hexdigest(),support_valid_sha256=ex['common_valid_sha256'],new_shadow_mask_sha256=hashlib.sha256(new.tobytes()).hexdigest(),missing_positions=len(missing),added_shadow_positions=int(added.sum()),unchanged_nonshadow_positions=int(((old!=2)&~new).sum()),missing_file='missing_positions/'+pid+'.npz',missing_file_sha256=sha(target),required_new_calls=len(q['actual_patch_coordinates']) if len(missing) else 0)
   rows.append(row);cacheaudit.append(dict(product_id=pid,cache_sha256=sha(p),keys=z.files,full_raw_logits_present=False,nonshadow_conditional_argmax_present=False))
 assert len(rows)==32 and sum(q['required_new_calls'] for q in rows)==96 and sum(q['missing_positions'] for q in rows)==172518
 save('MISSING_PLAN.json',dict(rows=rows,threshold=t,threshold_source_sha256=sha(tpath),main_calls=sum(q['required_new_calls'] for q in rows),constructor_allowance=20,planned_upper_calls=sum(q['required_new_calls'] for q in rows)+20,hard_cap=512,GPU_seconds_cap=3600,CPU_bytes_cap=16*1024**3,GPU_bytes_cap=20*1024**3,new_persistent_bytes_cap=2000000000,host_free_bytes_floor=4000000000,reference_semantics_read=False,no_PixBox=True,no_calibration_inputs=True,all_missing_locked_before_forward=True))
 save('CACHE_AUDIT.json',dict(rows=cacheaudit,eligible_reusable_conditional_cache=None,source_scope='205 original R4 and208 controlled delivery manifests/source: full raw logits were temporary only; 209 is calibration-only scalar scores, not eligible evaluation raw information',no_other_model_substitution=True))
 with (O/'MISSING_PRODUCTS.csv').open('x',encoding='utf-8-sig',newline='') as f:
  keys=['product_id','cache_sha256','support_count','missing_positions','added_shadow_positions','unchanged_nonshadow_positions','required_new_calls','missing_file_sha256'];w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:q[k] for k in keys} for q in rows)
 sources={n:sha(R/'src/tools'/n) for n in ['phase119_common.py','phase118_common.py','phase200_common.py','install_phase111_runtime.py']}
 files={p.relative_to(O).as_posix():sha(p) for p in O.rglob('*') if p.is_file()}
 save('EXECUTION_LOCK.json',dict(files=files,sources=sources,code_sha256=sha(R/'src/tools/execute_phase210.py'),common_helper_sha256=sha(R/'src/tools/execute_phase209.py'),protocol_sha256=sha(R/'src/research_plans/PHASE210_R4_FIXED_THRESHOLD_MIOU_20261020.md'),old208_delivery_manifest_sha256=sha(R/'outputs/phase208/phase208_20261020/DELIVERY_COMPLETE.json'),old209_delivery_manifest_sha256=sha(R/'outputs/phase209/phase209_20261020/DELIVERY_COMPLETE.json'),phase208_score_seal_sha256=sha(P/'infer/SCORES_SEALED.json'),phase205_score_seal_sha256=sha(OLD/'infer/SCORES_SEALED.json')))
 with zipfile.ZipFile(O/'phase210_bootstrap.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
  for p in O.rglob('*'):
   if p.is_file() and p.name!='phase210_bootstrap.zip':z.write(p,p.relative_to(O).as_posix())
  for n in ['execute_phase210.py','execute_phase209.py']:z.write(R/'src/tools'/n,'code/'+n)
 print(json.dumps(dict(products=32,rerun_products=24,missing_positions=172518,main_calls=96,planned_upper=116,cap=512,bootstrap_sha256=sha(O/'phase210_bootstrap.zip'))))
if __name__=='__main__':main()
