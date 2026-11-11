"""Independent bounded-histogram enclosure of every exact scene-pair AUC."""
from pathlib import Path
import json,time,hashlib
import numpy as np
R=Path('/home/scv/shared/phase247_20261031')
def js(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  while b:=f.read(4194304):h.update(b)
 return h.hexdigest()
def main():
 started=time.monotonic();lock=js(R/'LOCK.json')
 for n,h in lock['files'].items():assert sha(R/n)==h,n
 done=js(R/'INFERENCE_COMPLETE.json');ledger=[json.loads(l) for l in (R/'CALL_LEDGER.jsonl').read_text().splitlines()]
 assert len(ledger)==done['calls']<=2100
 assert [v['ordinal'] for v in ledger]==list(range(1,len(ledger)+1))
 assert done['nonhead_unchanged'] and all(v['max_abs_difference']<=1e-5 for v in done['repeat'])
 support=js(R/'DATA_SUPPORT_ACCEPTANCE.json');counts={v['scene']:v['counts'] for v in support['scenes']}
 index=js(R/'SCORE_INDEX.json');hist={};files={};edges=np.linspace(-30,30,8193,dtype=np.float64)
 for v in index:
  assert sha(v['path'])==v['sha256'];expected=counts[v['scene']]['2' if v['side']=='P' else '0'];assert v['n']==expected
  a=np.memmap(v['path'],mode='r',dtype=np.float32);assert len(a)==expected;h=np.zeros(8192,np.int64)
  for off in range(0,len(a),1000000):
   block=a[off:off+1000000];assert np.isfinite(block).all() and (block>=-30).all() and (block<=30).all();h+=np.histogram(block,bins=edges)[0]
  assert int(h.sum())==expected;hist[v['model'],v['scene'],v['side']]=h;files[Path(v['path']).name]=v['sha256'];del a
 tables=js(R/'RESULT_TABLES.json');bounds=[]
 for model,rows in tables['matrices'].items():
  exact_weighted=0;denom=0
  for row in rows:
   hp=hist[model,row['positive_scene'],'P'].astype(np.float64);hn=hist[model,row['negative_scene'],'N'].astype(np.float64);below=np.cumsum(hn)-hn;pairs=int(hp.sum())*int(hn.sum())
   lower=float(np.dot(hp,below)/pairs);upper=float(np.dot(hp,below+hn)/pairs)
   assert lower-1e-12<=row['AUC']<=upper+1e-12,(model,row,lower,upper)
   assert pairs==row['pairs'];exact_weighted+=row['AUC']*pairs;denom+=pairs
   bounds.append({'model':model,'positive_scene':row['positive_scene'],'negative_scene':row['negative_scene'],'lower':lower,'exact':row['AUC'],'upper':upper,'bound_width':upper-lower})
  reported=next(s for s in tables['summary'] if s['model']==model);assert abs(exact_weighted/denom-reported['pooled_AUC'])<1e-12
  assert denom==reported['total_pairs']
 for key,cm in js(R/'CONFUSIONS.json').items():
  scene,model=key.split('/');assert np.array_equal(np.array(cm,dtype=np.int64).sum(1),[counts[scene][str(i)] for i in range(3)])
 result={'passed':True,'independent_full_matrix_AUC_enclosure':bounds,'max_enclosure_width':max(b['bound_width'] for b in bounds),'network_calls':len(ledger),'model_protocol_hashes_verified':True,'confusion_supports_match':True,'seconds':time.monotonic()-started,'no_inference':True,'no_fit':True,'no_threshold_scan':True}
 (R/'INDEPENDENT_VALIDATION.json').write_text(json.dumps(result,indent=2))
 print(json.dumps({k:v for k,v in result.items() if k!='independent_full_matrix_AUC_enclosure'}))
if __name__=='__main__':main()
