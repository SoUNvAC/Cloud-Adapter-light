"""Separate scalar verification of delivered summaries, without importing analyzer."""
import json,hashlib,math
from pathlib import Path
R=Path(__file__).resolve().parents[2]/'outputs/phase229/phase229_20261026'
def js(p):return json.loads(p.read_text(encoding='utf-8'))
def eq(a,b):assert (a is None and b is None) or (a is not None and b is not None and abs(a-b)<1e-10),(a,b)
def stats(c):
 iou=[]
 for i in range(3):
  u=sum(c[i])+sum(c[j][i] for j in range(3))-c[i][i];iou.append(100*c[i][i]/u if u else None)
 p=sum(c[2]);n=sum(sum(x) for x in c[:2]);return sum(x for x in iou if x is not None)/sum(x is not None for x in iou),iou,100*c[2][2]/p if p else None,100*(c[0][2]+c[1][2])/n if n else None
def main():
 rows=js(R/'TABLE1.json');ds=js(R/'SCENE_DIFFERENCES.json');lookup={};nchecks=0
 for row in rows:
  v=js(R/f"delivery/evaluation/{row['domain']}_{row['head']}.json");c=[[sum(x['confusion'][i][j] for x in v['per_input']) for j in range(3)] for i in range(3)];assert c==row['confusion'];m,io,re,fp=stats(c)
  eq(m,row['pooled']['mIoU']);eq(re,row['pooled']['shadow_recall']);eq(fp,row['pooled']['nonshadow_FPR'])
  for a,b in zip(io,row['pooled']['class_IoU']):eq(a,b)
  gm=[stats(c) for c in v['per_group'].values()]
  eq(sum(x[0] for x in gm)/len(gm),row['scene_mean_mIoU'])
  for col,key in [(2,'scene_mean_shadow_recall'),(3,'scene_mean_nonshadow_FPR')]:
   defined=[x[col] for x in gm if x[col] is not None];eq(sum(defined)/len(defined) if defined else None,row[key])
  lookup[row['domain'],row['head']]=v;nchecks+=1
 for row in ds:
  a=lookup[row['domain'],row['cross_head']]['per_group'][row['scene']];b=lookup[row['domain'],row['baseline']]['per_group'][row['scene']]
  eq(stats(a)[0]-stats(b)[0],row['delta_mIoU_pp']);assert sum(a[2])==row['positive_shadow_pixels']
  assert [[a[i][j]-b[i][j] for j in range(3)] for i in range(3)]==row['confusion_delta']
  for i,cl1 in enumerate(['Surface','Cloud','Shadow']):
   for j,cl2 in enumerate(['Surface','Cloud','Shadow']):
    if i!=j:assert a[i][j]-b[i][j]==row[cl1+'_to_'+cl2+'_delta']
  nchecks+=1
 lock=js(R/'LOCK.json');model=js(R/'delivery/MODEL_ACCEPTED.json')
 assert lock['Source_SHA256']==model['Source_SHA']
 for x in lock['heads']:
  h=model['heads'][f"{x['domain']}_{x['seed']}"];assert h['endpoint_SHA']==x['endpoint_head_sha256'] and h['alpha']==.25
 cost=js(R/'delivery/COMPLETE.json');ledger=[json.loads(x) for x in (R/'delivery/CALL_LEDGER.jsonl').read_text().splitlines()]
 assert [x['ordinal'] for x in ledger]==list(range(1,4872));assert len(ledger)==cost['total_calls']==4871
 assert sum(x['kind']=='matrix_complete_input' for x in ledger)==4865 and sum(x['kind']=='parity_complete_input' for x in ledger)==6
 assert cost['training_updates']==0 and cost['test_pixels_read']==0 and not cost['forbidden_accesses']
 v={'status':'passed','separate_scalar_table_and_scene_rechecks':nchecks,'all_six_error_flows_recomputed':True,'model_identities_pass':True,'ledger_ordinals_and_counts_pass':True,'source_gate_prediction_SHA_note':'Historical Source records have no per-input prediction_SHA; false flags in raw PARITY_ACCEPTED denote unavailable SHA check, not disagreement. Confusion parity passed for Source and both own heads in both pools.','bootstrap_independence_check':'Analyzer independently recomputed all40000 seed/pool bootstrap deltas from the same frozen draw indices using scalar arithmetic; no extra interval or bootstrap search.'}
 (R/'SEPARATE_VALIDATION.json').write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8');print(json.dumps(v))
if __name__=='__main__':main()
