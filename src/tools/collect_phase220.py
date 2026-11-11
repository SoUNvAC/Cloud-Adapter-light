"""Small immutable shared delivery only; zero new model calls or reference reads."""
from pathlib import Path
import json,hashlib,tarfile,collections
R=Path('/home/scv/shared/phase220_20261024')
def js(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 assert (R/'EXIT_CODE').read_text().strip()=='0';done=js(R/'COMPLETE.json');assert done['units']==4800 and done['GPU_seconds']<3600
 count=collections.Counter();expected=1
 for mode in ['val','replay']:
  rows=[json.loads(v) for v in (R/mode/'CALL_LEDGER.jsonl').read_text().splitlines()]
  assert [v['ordinal'] for v in rows]==list(range(expected,expected+len(rows)));expected+=len(rows);count.update(v['kind'] for v in rows);access=js(R/mode/'ACCESS_AUDIT.json');assert not access['forbidden_attempts'] and all(access[n]==0 for n in ['source_train_label_reads','official_source_test_reads','target_inference','training_steps','backwards','fits']);assert js(R/mode/'COMPLETE.json')['all_frozen_tensors_checked_each_input']
 assert expected==4801 and sum(count.values())==4800
 for n,h in js(R/'CODE_LOCK.json').items():assert sha(R/'code'/n)==h
 lock=js(R/'MODEL_LOCK.json');assert sha(lock['Source']['path'])==lock['Source']['SHA'] and all(sha(v['path'])==v['SHA'] for v in lock['students'].values())
 archive=R.parent/'phase220_delivery.tar.gz';assert not archive.exists();partial=archive.with_name(archive.name+'.partial');assert not partial.exists()
 (R/'REMOTE_COLLECTION_RECEIPT.json').write_text(json.dumps({'status':'passed','calls':dict(count),'units':4800,'zero_training_backwards_fits':True,'no_target_inference':True,'Source_and6_head_files_unchanged':True,'new_model_calls':0},indent=2))
 names=[p for p in R.rglob('*') if p.is_file() and p.suffix not in ['.pth','.npy'] and not p.name.endswith('.tmp') and p.name!='DELIVERY_MANIFEST.json'];m={str(p.relative_to(R)):{'bytes':p.stat().st_size,'SHA':sha(p)} for p in names};(R/'DELIVERY_MANIFEST.json').write_text(json.dumps(m,indent=2))
 with tarfile.open(partial,'w:gz') as t:
  for p in names+[R/'DELIVERY_MANIFEST.json']:t.add(p,arcname=str(p.relative_to(R)),recursive=False)
 partial.replace(archive);print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'SHA':sha(archive),'remote_scope_bytes':sum(p.stat().st_size for p in R.rglob('*') if p.is_file())}))
if __name__=='__main__':main()
