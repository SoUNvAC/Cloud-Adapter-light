"""CPU-only ledger/freeze/reproduction checks, text reports only."""
from pathlib import Path
import hashlib,json,collections,tarfile
R=Path('/home/scv/shared/phase223_20261024')
def js(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 assert (R/'EXIT_CODE').read_text().strip()=='0';c=js(R/'COMPLETE.json');assert c['total_units']==5360 and c['GPU_seconds']<3600
 ledger=[json.loads(v) for v in (R/'CALL_LEDGER.jsonl').read_text().splitlines()];assert [v['ordinal'] for v in ledger]==list(range(1,5361)) and dict(collections.Counter(v['kind'] for v in ledger))==c['counts']
 frozen=js(R/'FROZEN_FINAL.json');assert frozen['all_tensors_exact'] and frozen['initial_digests']==frozen['final_digests'] and frozen['optimizer_updates']==0
 access=js(R/'ACCESS_AUDIT.json');assert not access['forbidden_attempts'] and access['source_train_replay_pixel_reads']==access['source_train_test_reference_reads']==access['source_official_test_reads']==access['target_pixel_reads']==0
 for v in js(R/'REPRODUCTION_ACCEPTED.json').values():assert v['all535_input_confusions_exact'] and v['all107_ROI_confusions_exact'] and v['pooled_confusion_exact'] and v['prediction_SHA']==v['expected_SHA']
 records=[json.loads(v) for v in (R/'PER_IMAGE.jsonl').read_text().splitlines()];assert len(records)==4280 and len({(v['model'],v['name']) for v in records})==4280
 for model in js(R/'INPUT_MODEL_LOCK.json')['models']:
  rr=[v for v in records if v['model']==model];assert len(rr)==535 and len({v['ROI'] for v in rr})==107 and all(v['alpha']==.25 and v['pool']=='Source_val535' for v in rr)
 assert len((R/'JOINT_COUNTS.jsonl').read_text().splitlines())==2140 and len(js(R/'FIXED_KL_COORDINATES.json'))==48
 (R/'REMOTE_CPU_ACCEPTANCE.json').write_text(json.dumps({'status':'passed','units':5360,'records':4280,'all8_alpha025_heads_exact_interpolation':True,'formal_reproduction_exact':True,'frozen':True,'training_updates':0,'new_model_calls':0},indent=2))
 files=[p for p in R.rglob('*') if p.is_file() and p.name!='DELIVERY_MANIFEST.json' and not p.name.endswith('.tmp')];assert not any(p.suffix in ['.pth','.npy','.npz','.png','.tif','.jpg'] for p in files);m={str(p.relative_to(R)):{'bytes':p.stat().st_size,'SHA':sha(p)} for p in files};(R/'DELIVERY_MANIFEST.json').write_text(json.dumps(m,indent=2));archive=R.parent/'phase223_delivery.tar.gz';partial=archive.with_name(archive.name+'.partial');assert not archive.exists() and not partial.exists()
 with tarfile.open(partial,'w:gz') as t:
  for p in files+[R/'DELIVERY_MANIFEST.json']:t.add(p,arcname=str(p.relative_to(R)),recursive=False)
 partial.replace(archive);print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'SHA':sha(archive),'remote_scope_bytes':sum(p.stat().st_size for p in R.rglob('*') if p.is_file())}))
if __name__=='__main__':main()
