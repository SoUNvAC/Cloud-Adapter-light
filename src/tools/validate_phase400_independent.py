"""Independent CPU-only Phase400 acceptance. No model construction or forward."""
from pathlib import Path
import json,hashlib,time,os
import numpy as np
from sklearn.metrics import average_precision_score
R=Path('/home/scv/shared/phase400_20261031')
def js(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 start=time.monotonic();out=R/'acceptance_independent';assert not out.exists();out.mkdir();checks=[];aprows=[];source_rows=[];traces_receipts=[];lock=js(R/'INPUT_LOCK.json');targets={v['product']:v for v in lock['target_rows'] if v['split']=='development_val'};src=lock['source_val'];assert len(targets)==32 and len(src)==535
 manifest=js(R/'delivery/SHA256SUMS.json')
 for name,h in manifest.items():assert sha(R/'delivery'/name)==h
 assert sha('/home/scv/Cloud-Adapter-light/src/work_dirs/phase64_source_parent/seed64/best_mIoU_iter_4000.pth')==lock['Source_SHA'];checks.append('Source and formal delivery SHA unchanged')
 paths={('S',None):R/'evaluations/Source/METRICS.json'}
 ends=js(R/'SELECTION_LOCK.json')['endpoints']
 for arm in ('MH','M','H'):
  curves={i:js(R/'evaluations'/arm/('step_'+str(i))/'METRICS.json') for i in range(500,4001,500)}
  selected=max(curves,key=lambda i:(curves[i]['mIoU'],-i));assert next(e['step'] for e in ends if e['arm']==arm and e['role']=='best')==selected
  for i,v in curves.items():
   c=np.sum([g['confusion'] for g in v['per_group']],axis=0);d=c.diagonal();u=c.sum(0)+c.sum(1)-d;assert abs(np.mean(d/u)*100-v['mIoU'])<1e-10
  checks.append(arm+' all8 validation pooled mIoU and earliest exact-tie selection')
 for e in ends:paths[e['arm'],e['step']]=R/'evaluations'/e['arm']/('step_'+str(e['step']))/'METRICS.json'
 for (arm,step),path in paths.items():
  v=js(path);pool_s=[];pool_y=[];per=[];assert len(v['per_group'])==32
  for g in v['per_group']:
   expected=targets[g['product']];cm=np.array(g['confusion'],dtype=np.int64);assert cm.sum(1).tolist()==expected['label_counts'];assert sha(R/g['score_file'])==g['score_SHA']
   with np.load(R/g['score_file'],allow_pickle=False) as z:truth=z['truth'];score=z['shadow'];valid=truth!=255;score=score[valid].astype(np.float64);labels=truth[valid]==2
   assert np.isfinite(score).all() and np.all((score>=0)&(score<=1));assert np.bincount(truth[valid],minlength=3).tolist()==expected['label_counts'];P=int(labels.sum());N=int((~labels).sum());a=float(average_precision_score(labels,score))*100 if P and N else None;per.append({'group_id':g['group_id'],'product':g['product'],'AP':a,'positive':P,'negative':N});pool_s.append(score);pool_y.append(labels)
  a=float(average_precision_score(np.concatenate(pool_y),np.concatenate(pool_s)))*100;macro=float(np.mean([x['AP'] for x in per if x['AP'] is not None]));aprows.append({'arm':arm,'step':step,'macro_AP':macro,'pooled_AP':a,'per_group':per});del pool_s,pool_y;checks.append(str((arm,step))+' independent sklearn exact-ties group and pooled AP')
 for p in sorted((R/'source_evaluations').glob('*/METRICS.json')):
  v=js(p);records=v['per_input'];assert len(records)==535;cm=np.zeros((3,3),np.int64)
  for i,g in enumerate(records):
   assert g['name']==src[i]['name'];c=np.asarray(g['confusion'],np.int64);assert c.sum(1).tolist()==src[i]['parent_counts'];cm+=c;source_rows.append({'evaluation':v['tag'],**g})
  assert cm.tolist()==v['confusion'];d=cm.diagonal();u=cm.sum(0)+cm.sum(1)-d;assert abs(float(np.mean(d/u)*100)-v['mIoU'])<1e-10
  if v['source_branch']=='off':assert v['branch_off_max_abs']<=1e-5 and v['branch_off_prediction_disagreement_images']==0
  checks.append(v['tag']+' all535 parent support/per-image aggregation/off parity')
 for kind,n in (('short',120),('formal',4000)):
  values={a:[json.loads(x) for x in (R/'work_dirs'/kind/a/'STEP_TRACE.jsonl').read_text().splitlines()] for a in ('MH','M','H')}
  for a,rows in values.items():
   assert len(rows)==n
   for i,v in enumerate(rows):
    assert v['step']==i+1 and v['draw']==i
    for k in ('draw','product','image_SHA','label_SHA','augmentation_seed'):assert v[k]==values['MH'][i][k]
    assert v['LR'][0]==values['MH'][i]['LR'][0]
   folder=R/'work_dirs'/kind/a;freeze=[js(p) for p in folder.glob('FREEZE_*.json')];assert all(v['passed'] for v in freeze) and len({v['Source_state_SHA'] for v in freeze})==1
   scope=js(folder/'ACTIVE_SCOPE.json');assert scope['count']=={'MH':380577,'M':354433,'H':26144}[a]
   traces_receipts.append({'kind':kind,'arm':a,'steps':n,'STEP_TRACE_SHA':sha(folder/'STEP_TRACE.jsonl'),'ACTIVE_SCOPE_SHA':sha(folder/'ACTIVE_SCOPE.json'),'frozen_checks':len(freeze),'Source_frozen_state_SHA':freeze[0]['Source_state_SHA'],'active_count':scope['count'],'MsRE_initial_SHA':scope['MsRE_initial_SHA'],'head_initial_SHA':scope['head_initial_SHA'],'mean_train_step_seconds':float(np.mean([v['seconds'] for v in rows])),'SAVE_TIMING':js(folder/'SAVE_TIMING.json')})
  assert len({v['Source_frozen_state_SHA'] for v in traces_receipts if v['kind']==kind})==1
  checks.append(kind+' complete paired input/GT/augmentation/LR, frozen buffers and trainable count')
 for e in ends:assert sha(e['checkpoint'])==e['checkpoint_SHA']
 costs=[js(p) for p in (R/'costs').glob('*.json')];ledger=[json.loads(x) for x in (R/'CALL_LEDGER.jsonl').read_text().splitlines()];assert all(c['status']=='passed' for c in costs);assert len(ledger)==sum(c['calls'] for c in costs);assert sum(c['updates'] for c in costs)==12363;assert sum(c['seconds'] for c in costs)<43200
 checks.append('all calls/12363 updates/GPU cumulative ledger; six endpoint file SHA')
 receipt={'passed':True,'checks':checks,'seconds':time.monotonic()-start,'model_calls':0,'training_updates':0,'forward_ledger_total':len(ledger),'updates_all_stages':12363,'validator_SHA':sha(Path(__file__)),'independent_AP':'sklearn.average_precision_score on saved FP32 shadow scores promoted float64; exact ties grouped; no softmax','AP':aprows,'train_receipts':traces_receipts,'source_per_image':source_rows,'formal_updates':12000,'short_updates':360,'gate_updates':3}
 (out/'CPU_ACCEPTANCE.json').write_text(json.dumps(receipt,allow_nan=False));return receipt
if __name__=='__main__':main()
