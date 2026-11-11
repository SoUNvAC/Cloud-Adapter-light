"""CPU-only record inventory and small shared report package; no weights/arrays/images."""
from pathlib import Path
import hashlib,json,collections,tarfile,resource
R=Path('/home/scv/shared/phase218_20261023')
def js(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 assert (R/'EXIT_CODE').read_text().strip()=='0';c=js(R/'COMPLETE.json');assert c['total_units']==4758 and c['GPU_seconds']<3600
 ledger=[json.loads(x) for x in (R/'CALL_LEDGER.jsonl').read_text().splitlines()];assert [x['ordinal'] for x in ledger]==list(range(1,4759)) and dict(collections.Counter(x['kind'] for x in ledger))==c['counts']
 rows=[json.loads(x) for x in (R/'PER_IMAGE.jsonl').read_text().splitlines()];assert len(rows)==3164 and len({(r['pool'],r['name'],r['seed'],r['arm']) for r in rows})==3164
 for s in [21701,21702]:
  for a in ['C','D']:
   for pool,n,roi in [('Replay256',256,249),('Source_val535',535,107)]:
    rr=[r for r in rows if r['seed']==s and r['arm']==a and r['pool']==pool];assert len(rr)==n and len({r['ROI'] for r in rr})==roi
    assert all(('reference_classes' in r)==(pool=='Source_val535') for r in rr)
 freeze=js(R/'FROZEN_FINAL.json');assert freeze['all_tensors_exact'] and freeze['initial_digests']==freeze['final_digests'] and freeze['optimizer_updates']==0
 access=js(R/'ACCESS_AUDIT.json');assert not access['forbidden_attempts'] and access['source_train_reference_reads']==access['source_official_test_reads']==access['target_evaluation_reads']==0
 for r in js(R/'REPRODUCTION_ACCEPTED.json').values():assert r['prediction_SHA']==r['expected_SHA'] and r['all535_input_confusions_exact'] and r['all107_ROI_confusions_exact']
 (R/'REMOTE_CPU_ACCEPTANCE.json').write_text(json.dumps({'status':'passed','units':4758,'records':3164,'Replay256_249_ROI':True,'Sourceval535_107_ROI':True,'parameters_buffers_frozen':True,'old217_exact_reproduction':True,'source_train_labels_absent':True,'new_model_calls':0,'CPU_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024},indent=2))
 files=[p for p in R.rglob('*') if p.is_file() and p.name!='DELIVERY_MANIFEST.json' and not p.name.endswith('.tmp')];assert not any(p.suffix in ['.pth','.npy','.npz','.png','.jpg','.tif'] for p in files)
 manifest={str(p.relative_to(R)):{'bytes':p.stat().st_size,'SHA':sha(p)} for p in files};(R/'DELIVERY_MANIFEST.json').write_text(json.dumps(manifest,indent=2));archive=R.parent/'phase218_delivery.tar.gz';assert not archive.exists()
 with tarfile.open(archive,'w:gz') as t:
  for p in files+[R/'DELIVERY_MANIFEST.json']:t.add(p,arcname=str(p.relative_to(R)),recursive=False)
 print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'SHA':sha(archive),'remote_scope_bytes':sum(p.stat().st_size for p in R.rglob('*') if p.is_file())}))
if __name__=='__main__':main()
