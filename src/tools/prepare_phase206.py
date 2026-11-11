"""Phase206 baseline reference recheck and immutable input lock; no neural calls."""
import csv,gc,json,hashlib,shutil,sys,time
from pathlib import Path
R=Path(__file__).resolve().parents[2];O=R/'outputs/phase206/phase206_20261019';OLD=R/'outputs/phase205/phase205_20261019'
sys.path.insert(0,str(R/'src/tools'))
import analyze_phase205_local as a
import numpy as np

def sha(p):return a.sha(p)
def save(n,v):
 with (O/n).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,allow_nan=False)
def main():
 start=time.time();sealed=a.load(OLD/'delivery/infer/SCORES_SEALED.json')['scores'];plan=a.load(OLD/'EVALUATION_INPUT_PLAN.json');expected={r['product_id']:r for r in a.load(OLD/'EXPECTED_SUPPORT.json')['rows']};checks=0
 for pool,fn in [('Catalogue',a.cat),('PixBox',a.pix)]:
  groups,sup=fn([r for r in plan['rows'] if r['pool']==pool],sealed,expected);old=list(csv.DictReader((OLD/(pool+'_PER_GROUP.csv')).open(encoding='utf-8-sig')));lookup={(r['strategy'],r['group_id']):r for r in old};summ=list(csv.DictReader((OLD/(pool+'_SUMMARY.csv')).open(encoding='utf-8-sig')))
  for name in a.NAMES:
   matrices=[];scores=[];ys=[];aps=[]
   for gid,g in groups.items():
    m,mat=a.metric(g['y'],g['pred'][name],g['scores'][name]);prior=lookup[name,gid]
    for k in ('positive','negative','TP','FP','FN','TN','Shadow_to_Surface','Shadow_to_Cloud'):assert m[k]==int(prior[k]);checks+=1
    assert (m['AP'] is None and prior['AP']=='') or abs(m['AP']-float(prior['AP']))<1e-12
    if m['AP'] is not None:aps.append(m['AP'])
    matrices.append(mat);scores.append(g['scores'][name]);ys.append(g['y']);checks+=1
   prior=next(r for r in summ if r['strategy']==name);pooled=a.ap(np.concatenate(ys)==2,np.concatenate(scores));assert abs(pooled*100-float(prior['pooled_AP_pp']))<1e-10 and abs(np.mean(aps)*100-float(prior['macro_AP_pp']))<1e-10
   assert np.array_equal(np.sum(matrices,0),a.load(OLD/(pool+'_CONFUSION.json'))['matrices'][name]);checks+=3
  del groups;gc.collect();print(pool,'baseline rechecked',flush=True)
 files=['EVALUATION_INPUT_PLAN.json','EXPECTED_SUPPORT.json','V4_WEIGHT_LOCK.json']
 for n in files:shutil.copyfile(OLD/n,O/n)
 shutil.copyfile(OLD/'delivery/gate/SMP_WHITELIST.json',O/'SMP_WHITELIST.json')
 save('BASELINE_ACCEPTANCE.json',dict(status='passed',checks=checks,all_6_policies_all_42_groups_recalculated=True,AP_ties_independent=True,original_confusion_exact=True,seconds=time.time()-start,neural_calls=0,fits=0,original_complete_sha256=sha(OLD/'DELIVERY_COMPLETE.json'),source_files={str(p.relative_to(R).as_posix()):sha(p) for p in [OLD/'delivery/infer/SCORES_SEALED.json',OLD/'Catalogue_PER_GROUP.csv',OLD/'PixBox_PER_GROUP.csv',OLD/'Catalogue_SUMMARY.csv',OLD/'PixBox_SUMMARY.csv']}))
 inherited=a.load(OLD/'INPUT_LOCK.json')['source_sha256'];protocol=R/'src/research_plans/PHASE206_V4_PARAMETER_BN_STATE_CHECK_20261019.md'
 save('INPUT_LOCK.json',dict(protocol_sha256=sha(protocol),files={n:sha(O/n) for n in files+['SMP_WHITELIST.json','BASELINE_ACCEPTANCE.json']},source_sha256=inherited,planned_main_calls=5730,approved_total_cap=8000,GPU_stage_seconds_cap=14400,CPU_bytes_cap=16*1024**3,GPU_bytes_cap=20*1024**3,new_persistent_bytes_cap=6000000000,host_free_bytes_floor=4000000000,no_fit_or_calibration_inputs=True))
 save('CODE_LOCK.json',dict(files={'execute_phase206.py':sha(R/'src/tools/execute_phase206.py')}))
 print('Phase206 input and original-output checks passed',flush=True)
if __name__=='__main__':main()
