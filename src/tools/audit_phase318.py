"""Independent CPU64 weights, selector/curve/provenance and tensor-margin audit."""
from collections import defaultdict
import numpy as np
from phase318_common import *
from phase318_selectors import selection,masses,product
def main():
 resource_check();assert 'torch' not in sys.modules and load(ROOT/'CPU_ACCEPTANCE.json')['status']=='passed';assert not (ROOT/'INDEPENDENT_ACCEPTANCE.json').exists()
 info=load(ROOT/'SELECTION_LOCK.json');assert sha(ROOT/'SELECTION_AND_WEIGHT.csv')==info['table_sha'];records=[];initial=None;calls=[json.loads(x) for x in (ROOT/'CALLS.jsonl').read_text().splitlines()]
 for pool in ['CMC','SPARCS','L8']:
  locked={int(r['attempt']):r for r in read(ROOT/'SELECTION_AND_WEIGHT.csv') if r['pool']==pool};actual={}
  for st in schedule(pool):
   p,g,c=selection(pool,st);fm=masses(p,g);cm=masses(p,c);assert fm==cm;row=locked[st['attempt']];assert ah(p)==row['pseudo_sha'] and ah(g)==row['Flip_sha'] and ah(c)==row['Conf_sha'];actual[st['attempt']]=(p.shape,fm,ah(p),ah(g),ah(c))
  for arm in ['Flip-CB','Conf-CB']:
   d=ROOT/pool/arm
   if (d/'EQUIVALENT_ARM.json').exists():
    e=load(d/'EQUIVALENT_ARM.json');target=Path(e['target']);assert (target/'ENDPOINT_LOCK.json').exists()
    assert info['pools'][pool]['identical_all_steps']
    if target.is_relative_to(PRIOR):assert info['pools'][pool]['full_all_steps'] and max(load(ROOT/pool/'Flip-CB/OLD_CB_EQUIVALENCE.json')[k] for k in ['loss_error','gradient_error','update_error'])<=1e-6
    assert not any(x['pool']==pool and x['arm']==arm and x['kind']=='train' for x in calls);records.append(dict(pool=pool,arm=arm,mode='strict_equivalent_reference',target=str(target),updates=0,skipped=0,attempts=0));continue
   ep=load(d/'ENDPOINT_LOCK.json');gate=load(d/'MODEL_ACCEPTANCE.json');curve=[json.loads(x) for x in (d/'TRAIN_CURVE.jsonl').read_text().splitlines()];assert len(curve)==1000 and sum(x['updated'] for x in curve)==info['pools'][pool]['updates']==ep['updates'];assert ep['skipped']==1000-ep['updates'] and ep['base_unchanged']
   for f,k in [('initial_lora.pth','initial_sha'),('final_lora.pth','final_sha'),('TRAIN_CURVE.jsonl','curve_sha')]:assert sha(d/f)==ep[k]
   assert gate['parameters']==294912 and gate['zero_lora_error']<=1e-6 and gate['only_lora_grad'] and gate['initial_optimizer_grad_RNG_restored'] and gate['fresh_initial_equals304']
   if initial is None:initial=gate['initial_tensor_SHA']
   else:assert initial==gate['initial_tensor_SHA']
   assert gate['initial_tensor_SHA']==load(PRIOR/'CMC/CB/MODEL_ACCEPTANCE.json')['initial_tensor_SHA']
   for row in curve:
    _,m,pm,fg,cg=actual[row['attempt']];assert row['mask_sha']==pm and row['selected_sha']==(fg if arm=='Flip-CB' else cg)
    for k in ['counts','N','K','b','k','class_mass','denominator']:assert row[k]==m[k]
    assert row['updated']==bool(m['N'])
    if row['updated']:
     log=row['logged'];assert np.allclose(log['actual_class_mass'],m['class_mass'],rtol=1e-6,atol=1e-6);assert abs(log['actual_denominator']-m['denominator'])<=1e-6*m['denominator'];expected=sum(log['weighted_CE_sum'])/log['actual_denominator'];assert abs(row['loss']-expected)<=1e-6*max(1,abs(expected));assert np.isfinite(row['grad_norm'])
   relevant=[x for x in calls if x['pool']==pool and x['arm']==arm];assert sum(x['kind']=='train' for x in relevant)==ep['updates'];assert sum(x['kind']=='probe' for x in relevant)==(16 if pool=='CMC' else 2)
   records.append(dict(pool=pool,arm=arm,mode='trained_once',target=str(d),updates=ep['updates'],skipped=ep['skipped'],attempts=1000))
  product.cache_clear()
 strategies=['Source','PL','CB','Flip-CB','Conf-CB'];matrices={(r['pool'],r['strategy'],r['scene']):np.array(json.loads(r['confusion']),np.int64) for r in read(ROOT/'PER_SCENE_METRICS.csv')};by=defaultdict(list)
 for r in read(ROOT/'TRANSITIONS27.csv'):by[r['pool'],r['comparison'],r['scene']].append(r)
 for (pool,pair,scene),rr in by.items():
  assert len(rr)==27;t=np.zeros((3,3,3),np.int64)
  for r in rr:t[C.index(r['truth']),C.index(r['Source_pred']),C.index(r['student_pred'])]=int(r['count'])
  a,b=pair.split('->')
  def matrix(arm):return sum((v for (p,s,g),v in matrices.items() if p==pool and s==arm),np.zeros((3,3),np.int64)) if scene=='POOLED' else matrices[pool,arm,scene]
  assert np.array_equal(t.sum(2),matrix(a)) and np.array_equal(t.sum(1),matrix(b))
 native=defaultdict(lambda:np.zeros((3,3),np.int64));maps={'CMC':{0:0,1:1,2:2},'SPARCS':{0:2,1:2,2:0,3:0,4:0,5:1,6:0},'L8':{64:2,128:0,192:1,255:1}}
 for r in read(ROOT/'PER_SCENE_NATIVE.csv'):
  parent=maps[r['pool']].get(int(r['native_code']))
  for arm in strategies:
   v=np.array(json.loads(r[arm.replace('-','_')+'_parent_predictions']),np.int64);assert int(v.sum())==int(r['metric_support'])
   if parent is None:assert not v.any()
   else:native[r['pool'],arm,r['scene']][parent]+=v
 for key,m in matrices.items():assert np.array_equal(native[key],m),key
 for r in read(ROOT/'DAMAGE_BENEFIT.csv'):
  for arm in ['CB','Flip-CB','Conf-CB']:assert sum(int(r['D_'+arm+'_'+c]) for c in C)==int(r['D']) and sum(int(r['B_'+arm+'_'+c]) for c in C)==int(r['B'])
 for file in [ROOT/'CPU_INPUT_SHA256.json',ROOT/'QUALITY_INPUT_SHA256.json',PRIOR/'REFERENCE_INPUT_SHA256.json']:
  for n,h in load(file).items():assert sha(n)==h,n
 for n,h in load(ROOT/'EXECUTION_LOCK.json')['inputs'].items():assert sha(n)==h,n
 assert len(calls)<=11000 and load(ROOT/'ALL_SEALED.json')['gpu_seconds']<=43200 and budget_bytes()<=12000000000
 assert sha(ROOT/'DECISION_WINDOW.json')==sha(PRIOR/'DECISION_WINDOW.json')
 table(ROOT/'TRAINING_IDENTITY.csv',records);write(ROOT/'INDEPENDENT_ACCEPTANCE.json',dict(status='passed',same_Q90_masks=True,paired_class_masses_float64_rtol=1e-6,all_new_train_curves=True,all_transition_margins=True,all_native_margins=True,original317_metrics_D_B_and_CMC_error_reproduced=True,fit_L8_reference_gap=load(ROOT/'QUALITY_ACCEPTANCE.json')['L8_fit_gap_images'],calls=len(calls),no_neural_calls=True,resources=resource_check()))
 print('Phase318 independent audit passed')
if __name__=='__main__':main()
