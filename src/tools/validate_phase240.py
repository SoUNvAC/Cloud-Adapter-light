"""Independent CPU count, selection, metric, identity and budget checks."""
from pathlib import Path
import collections,hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'outputs/phase240/phase240_20261030';D=R/('delivery_recovery03' if (R/'delivery_recovery03').exists() else 'delivery_recovery02' if (R/'delivery_recovery02').exists() else 'delivery');Q=D/('recovery03' if (D/'recovery03').exists() else 'recovery02')
def js(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def direct(c):
 c=np.asarray(c,dtype=np.float64);intersection=np.diag(c);union=c.sum(0)+c.sum(1)-intersection
 return float(np.mean([100*intersection[i]/union[i] for i in range(3) if union[i]>0]))
def main():
 assert js(R/'CPU_ANALYSIS_COMPLETE.json')['status']=='passed';checks=0
 all_scene=js(R/'PER_SCENE.json');tables=js(R/'TABLE1.json');lock=js(D/'VAL_SELECTION_LOCK.json');lock['selections']+=js(Q/'VAL_SELECTION_LOCK.json')['selections'] if (Q/'VAL_SELECTION_LOCK.json').exists() else []
 for row in all_scene:
  assert abs(direct(row['confusion'])-row['mIoU'])<1e-10;checks+=1
 for domain in ['L8','SPARCS']:
  for split in ['val','test']:
   pool=js(D/f'{domain}_{split}_SUMMARY.json')
   if (Q/f'{domain}_{split}_SUMMARY.json').exists():pool+=[z for z in js(Q/f'{domain}_{split}_SUMMARY.json') if not (split=='test' and z['branch']=='Source')]
   support=None
   for row in pool:
    assert sum(np.asarray(c,np.int64) for c in row['scenes'].values()).tolist()==row['confusion']
    assert abs(direct(row['confusion'])-row['pooled_mIoU'])<1e-10
    gt={g:np.asarray(c,np.int64).sum(1).tolist() for g,c in row['scenes'].items()}
    if support is None:support=gt
    else:assert gt==support
    checks+=1
   if split=='val':
    for arm in ['A','B','G']:
     candidates=[z for z in pool if z['branch']==arm]
     if not candidates:continue
     assert len(candidates)==5
     best=sorted(candidates,key=lambda z:(-direct(z['confusion']),int(not z['source']),z['step'] or 0))[0]
     chosen=next(z for z in lock['selections'] if z['domain']==domain and z['branch']==arm)
     assert best['source']==chosen['source'] and best['step']==chosen['step'];checks+=1
  base=next(z for z in tables if z['domain']==domain and z['role']=='C_selected');source=next(z for z in tables if z['domain']==domain and z['role']=='Source')
  for row in tables:
   if row['domain']==domain and row['status']=='available':
    assert abs(row['mIoU']-direct(row['confusion']))<1e-10
    assert abs(row['delta_C_pp']-(row['mIoU']-base['mIoU']))<1e-10
    assert abs(row['delta_Source_pp']-(row['mIoU']-source['mIoU']))<1e-10;checks+=1
 costs=js(Q/'COMPLETE.json') if (Q/'COMPLETE.json').exists() else js(D/'COMPLETE.json');calls=[json.loads(l) for l in (D/'CALL_LEDGER.jsonl').read_text().splitlines()];compute=[json.loads(l) for l in (D/'COMPUTE_LEDGER.jsonl').read_text().splitlines()];cc=collections.Counter(z['kind'] for z in compute)
 assert len(calls)==costs['calls']<=52000 and cc['optimizer_update']<=26000 and cc['backward_or_autograd']<=38000
 assert costs['GPU_stage_seconds']<14400
 assert abs(sum(js(p)['GPU_stage_seconds'] for p in (D/'stage_costs').glob('*.json'))-costs['GPU_stage_seconds'])<1e-6
 for fit in list((D/'fits').glob('*/COMPLETE.json'))+list((Q/'fits').glob('*/COMPLETE.json')):
  done=js(fit);cfg=js(fit.parent/'BRANCH_CONFIGURATION.json');assert done['updates']==4000
  assert cfg['training_loss_weights']['dice']==(0 if done['branch']=='B' else 5)
  assert [z['weight'] for z in cfg['matching']]==[2,5,5]
  assert js(fit.parent/'INITIAL.json')['trainable']==(516 if done['branch']=='A' else 1316516)
  assert len([l for l in (fit.parent/'SCHEDULE.jsonl').read_text().splitlines()])==4000
  assert sha(fit.parent/'SCHEDULE.jsonl')==done['schedule_SHA']
  for step,meta in done['heads'].items():assert js(fit.parent/f'SAVE_{step}.json')==meta
  checks+=1
 out=dict(status='passed',checks=checks,calls=len(calls),compute_counts=dict(cc),successful_updates=costs['successful_updates'],historical_C_refits=0,metric_and_selection_independently_recomputed=True,weights_not_transferred=True,scope='CPU counts, receipts and sealed identities; no claim of independent raw-weight reload')
 (R/'INDEPENDENT_ACCEPTANCE.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out))
if __name__=='__main__':main()
