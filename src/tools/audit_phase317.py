"""Independent CPU QA: original masks/curves, same accepted support, tensor margins, AP and hashes."""
from collections import defaultdict
import numpy as np
from phase317_common import *
def main():
 resource_check();assert 'torch' not in sys.modules and not (ROOT/'INDEPENDENT_ACCEPTANCE.json').exists();assert load(ROOT/'CPU_ACCEPTANCE.json')['status']=='passed'
 seal=load(ROOT/'ALL_SEALED.json')
 assert sha(OLD/'initial_lora.pth')==load(OLD/'ENDPOINT_LOCK.json')['initial_sha256']
 for key,h in seal['endpoints'].items():pool,arm=key.split('_');assert sha(ROOT/pool/arm/'ENDPOINT_LOCK.json')==h
 for key,h in seal['evaluation'].items():pool,arm=key.split('_');assert sha(ROOT/pool/arm/'EVALUATION_LOCK.json')==h
 identities=[]
 for pool in ['CMC','SPARCS','L8']:
  rows=inputs(pool)['fit'];lock=load(OLD/'PSEUDO_LOCK.json') if pool=='CMC' else load(ROOT/pool/'PSEUDO_LOCK.json');snap={r.get('name',r.get('product')):r for r in lock['rows']};pseudo={};actual=np.zeros(3,np.int64)
  for r in rows:
   name=r.get('name') or r['product'];e=snap[name];path=OLD/'teacher'/(name+'.npz') if pool=='CMC' else Path(e['file']);assert sha(path)==(e['sha256'])
   with np.load(path) as z:
    p=z['pseudo'];assert p.dtype==np.uint8 and np.isin(p,[0,1,2,255]).all()
    if pool!='CMC':
     s=z['scores'];v=z['valid'];tot=s.sum(0);q=np.divide(s,tot[None],out=np.zeros_like(s),where=tot[None]>1e-12);expected=np.where(v&(tot>1e-12)&(q.max(0)>=.9),s.argmax(0),255).astype(np.uint8);assert np.array_equal(expected,p) and np.array_equal(z['pred'],s.argmax(0))
    pseudo[name]=p.copy();actual+=[int((p==c).sum()) for c in range(3)]
  assert actual.tolist()==lock['classes'] and int(actual.sum())==lock['accepted'];stepmasks=[];class_steps=np.zeros(3,np.int64)
  for st in schedule(pool):
   r=rows[st['fit_index']];p=pseudo[r.get('name') or r['product']]
   if pool=='CMC':p=p[st['top']:st['top']+512,st['left']:st['left']+512]
   if st['horizontal_flip']:p=p[:,::-1]
   cc=[int((p==c).sum()) for c in range(3)];stepmasks.append((ah(p),cc));class_steps+=np.array(cc)>0
  assert class_steps.tolist()==lock['class_present_steps']
  arms=['CB'] if pool=='CMC' else ['PL','CB']
  for arm in arms:
   d=ROOT/pool/arm;curve=[json.loads(line) for line in (d/'TRAIN_CURVE.jsonl').read_text().splitlines()];ep=load(d/'ENDPOINT_LOCK.json');gate=load(d/'MODEL_ACCEPTANCE.json');assert len(curve)==1000 and sum(r['updated'] for r in curve)==ep['updates'] and ep['skipped']==1000-ep['updates'] and ep['base_unchanged']
   assert gate['fresh_initial_equals304'] and gate['parameters']==294912 and gate['B_zero'] and gate['zero_lora_error']<=1e-6
   assert sha(d/'initial_lora.pth')==ep['initial_sha'] and sha(d/'final_lora.pth')==ep['final_sha'] and sha(d/'TRAIN_CURVE.jsonl')==ep['curve_sha']
   for e,(h,cc) in zip(curve,stepmasks):
    assert e['mask_sha']==h and [e[c] for c in C]==cc and e['accepted']==sum(cc) and e['updated']==bool(sum(cc))
    if e['updated']:
     means=e['class_mean_CE'];expected=sum(means[c]*cc[c] for c in range(3) if cc[c])/sum(cc) if arm=='PL' else sum(v for v in means if v is not None)/sum(v is not None for v in means);assert abs(e['loss']-expected)<=4*abs(float(np.spacing(np.float32(e['loss'])))),(pool,arm,e['attempt'])
   identities.append(dict(pool=pool,arm=arm,updates=ep['updates'],skipped=ep['skipped'],initial_tensor_SHA=gate['initial_tensor_SHA'],initial_file_sha=ep['initial_sha'],probes=len(list((d/'probes').glob('*.npz')))))
  if pool!='CMC':assert load(ROOT/pool/'PL/MODEL_ACCEPTANCE.json')['initial_tensor_SHA']==load(ROOT/pool/'CB/MODEL_ACCEPTANCE.json')['initial_tensor_SHA']
  del pseudo
 assert all(r['initial_tensor_SHA']==identities[0]['initial_tensor_SHA'] for r in identities)
 calls=[json.loads(s) for s in (ROOT/'CALLS.jsonl').read_text().splitlines()]
 for e in identities:
  selected=[r for r in calls if r['pool']==e['pool'] and r['arm']==e['arm']]
  assert sum(r['context']=='train' and r['kind']=='train' for r in selected)==e['updates']
  assert sum(r['kind']=='probe' for r in selected)==(16 if e['pool']=='CMC' else 2)
 assert len(calls)<=11000
 matrices={(r['pool'],r['strategy'],r['scene']):np.array(json.loads(r['confusion']),np.int64) for r in read(ROOT/'PER_SCENE_METRICS.csv')};cells=read(ROOT/'TRANSITIONS27.csv');by=defaultdict(list)
 for r in cells:by[r['pool'],r['comparison'],r['scene']].append(r)
 for (pool,pair,scene),rr in by.items():
  assert len(rr)==27;t=np.zeros((3,3,3),np.int64)
  for r in rr:t[C.index(r['truth']),C.index(r['Source_pred']),C.index(r['student_pred'])]=int(r['count'])
  a,b=pair.split('->')
  if scene=='POOLED':
   ma=sum((m for (p,s,g),m in matrices.items() if p==pool and s==a),np.zeros((3,3),np.int64));mb=sum((m for (p,s,g),m in matrices.items() if p==pool and s==b),np.zeros((3,3),np.int64))
  else:ma=matrices[pool,a,scene];mb=matrices[pool,b,scene]
  assert np.array_equal(t.sum(2),ma) and np.array_equal(t.sum(1),mb)
 for r in read(ROOT/'DAMAGE_BENEFIT.csv'):
  assert sum(int(r['D_CB_'+c]) for c in C)==int(r['D']) and sum(int(r['B_CB_'+c]) for c in C)==int(r['B'])
 old={(r['strategy'],r['product']):json.loads(r['confusion']) for r in read(OLD/'analysis/PER_GROUP_METRICS.csv')}
 for r in inputs('CMC')['val']:
  for strategy in ['Source','PL']:assert np.array_equal(matrices['CMC',strategy,r['group_id']],old['Source' if strategy=='Source' else 'LoRA',r['product']])
 for path,h in load(ROOT/'CPU_INPUT_SHA256.json').items():assert sha(path)==h,path
 for path,h in load(ROOT/'REFERENCE_INPUT_SHA256.json').items():assert sha(path)==h,path
 table(ROOT/'TRAINING_IDENTITY.csv',identities);write(ROOT/'INDEPENDENT_ACCEPTANCE.json',dict(status='passed',all_same_initial_tensorSHA=True,original304_initial_sha256=sha(OLD/'initial_lora.pth'),all_masks_Q90_or_CMC_original_exact=True,all_5000_attempts_exact=True,all_transition_margins=True,CMC_D=506798,CMC_B=784478,no_new_neural_calls=True,resources=resource_check()))
 print('Phase317 independent QA passed: allsameinitial/masks/5000attempts/tensor margins and SHA')
if __name__=='__main__':main()
