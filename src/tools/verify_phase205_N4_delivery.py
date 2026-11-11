"""Verify retained original N4 scores and reconcile all7746 attempted calls."""
import hashlib,json,sys,zipfile
from pathlib import Path
R=Path(__file__).resolve().parents[2];O=R/'outputs/phase205/phase205_20261019';T=O/'N4_transfer';D=O/'N4_delivery'
sys.path.insert(0,str(R/'outputs/phase202/phase202_20261017/software'))
import numpy as np
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def main():
 receipt=load(T/'DELIVERY_RECEIPT.json');a=T/'N4_scores_delivery.zip';assert a.stat().st_size==receipt['bytes'] and sha(a)==receipt['sha256'];assert not D.exists();D.mkdir();checks=0
 with zipfile.ZipFile(a) as z:
  m=json.loads(z.read('DELIVERY_MANIFEST.json'));assert len(z.namelist())==len(m['files'])+1
  for n,h in m['files'].items():
   assert not n.startswith('/') and ':' not in n and '\\' not in n and '..' not in Path(n).parts;b=z.read(n);assert hashlib.sha256(b).hexdigest()==h;dest=D/n;dest.parent.mkdir(parents=True,exist_ok=True)
   with dest.open('xb') as f:f.write(b)
   checks+=1
 seal=load(D/'SCORES_SEALED.json');assert seal['cumulative_calls']==receipt['cumulative_calls']==7746 and seal['all_extra_calls']==1146 and seal['unchanged_original_N4'];expected={r['product_id']:r for r in load(O/'EXPECTED_SUPPORT.json')['rows']};assert len(seal['scores'])==48
 for key,h in seal['scores'].items():
  pool,pid=key.split('/',1);p=D/'scores'/pool/(pid+'.npz') if pool=='Catalogue' else D/'scores'/pool/pid/'UNLABELLED_POINT_SCORES.json';assert sha(p)==h
  if pool=='Catalogue':
   with np.load(p,allow_pickle=False) as z:assert hashlib.sha256(z['flat_indices'].tobytes()).hexdigest()==expected[pid]['flat_indices_sha256'];assert hashlib.sha256(z['valid'].tobytes()).hexdigest()==expected[pid]['common_valid_sha256'];assert np.isfinite(z['scores']).all() and np.array_equal(z['parent_class'],np.array([0,1,1,2],np.uint8)[z['raw_class']])
  else:
   pts=load(p)['points'];assert [q['point_id'] for q in pts if q['valid']]==expected[pid]['common_image_valid_point_ids'];assert all(not q['valid'] or np.isfinite(q['shadow_confidence']) and q['parent_class']==[0,1,1,2][q['raw_class']] for q in pts)
  checks+=4
 lines=(T/'CALL_LEDGER_FINAL.jsonl').read_bytes().splitlines(keepends=True);events=[json.loads(x) for x in lines];assert len(events)==7746 and [x['global_call'] for x in events]==list(range(1,7747));assert b''.join(lines[:6600])==(O/'delivery/CALL_LEDGER.jsonl').read_bytes();assert all(x['stage']=='N4_score_recovery' and x['kind']=='inference' for x in events[6600:]);assert sum(x['kind']=='training' for x in events)==2000
 assert max(x['epoch'] for x in events if x['kind']=='training')<1792425600
 result=dict(status='passed',checks=checks+6,all_actual_calls=7746,main_calls=6600,N4_repeated_inference_calls=1146,training_calls=2000,no_new_fit_or_comparison=True,N4_support_unchanged=True,main_ledger_prefix_unchanged=True,training_before_20261020_Beijing=True,combined_stage_seconds=receipt['cumulative_stage_seconds'],archive_sha256=receipt['sha256'],final_ledger_sha256=sha(T/'CALL_LEDGER_FINAL.jsonl'))
 with (O/'N4_RETENTION_ACCEPTANCE.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(result))
if __name__=='__main__':main()
