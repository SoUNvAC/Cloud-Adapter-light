"""Verify completed Phase205 outputs and deliver scores/receipts only, no weights."""
import hashlib,json,os,resource,time,zipfile
from pathlib import Path
ROOT=Path('/home/scv/shared/phase205_20261019')
def load(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def save(p,v):
 with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def main():
 start=time.time();assert (ROOT/'EXIT_CODE').read_text().strip()=='0';status=load(ROOT/'CONTROLLER_STATUS.json');assert status['total_calls']==6600 and status['stage_seconds_total']<14400
 events=[json.loads(s) for s in (ROOT/'CALL_LEDGER.jsonl').read_text().splitlines()];assert len(events)==status['total_calls'] and [s['global_call'] for s in events]==list(range(1,len(events)+1))
 import numpy as np
 from safetensors.numpy import load_file
 white=load(ROOT/'gate/SMP_WHITELIST.json');keys=set(white['decoder_state']);weight=next(w for w in load(ROOT/'V4_WEIGHT_LOCK.json')['weights'] if 'regnet' in w['filename']);wp=ROOT.parent/'phase200_20261017/assets'/weight['filename'];assert sha(wp)==weight['sha256'];original=load_file(str(wp));audit=[]
 for seed in (12201,12301):
  d=ROOT/f'train_{seed}';assert (d/'EXIT_CODE').read_text().strip()=='0';receipt=load(d/'TRAINING_RECEIPT.json');assert receipt['steps']==1000 and receipt['seed']==seed and receipt['encoder_unchanged'] and receipt['schedule_sha256']==sha(ROOT/f'{seed}_PAIRED_CROP_SCHEDULE.json')
  initial=load_file(str(d/'decoder_initial.safetensors'));final=load_file(str(d/'decoder_final.safetensors'));assert set(initial)==set(final)==keys
  assert all(np.array_equal(initial[k],original[k]) for k in keys);assert sha(d/'decoder_final.safetensors')==receipt['final_sha256']==load(ROOT/'TRAINING_SEAL.json')[str(seed)]
  assert all(final[k].shape==initial[k].shape and final[k].dtype==initial[k].dtype and np.isfinite(final[k]).all() for k in keys)
  changed=[k for k in keys if not np.array_equal(initial[k],final[k])];assert changed;assert not any(k.startswith('encoder.') for k in keys)
  audit.append(dict(seed=seed,initial_matches_original_R4=True,only_decoder_head_saved=True,initial_sha256=sha(d/'decoder_initial.safetensors'),final_sha256=sha(d/'decoder_final.safetensors'),changed_keys=sorted(changed),encoder_unchanged_execution_assertion=receipt['encoder_unchanged']))
 save(ROOT/'ENDPOINT_CPU_STATE_ACCEPTANCE.json',dict(status='passed',rows=audit,neural_calls=0,fit=0))
 sealed=load(ROOT/'infer/SCORES_SEALED.json');assert sealed['before_local_reference_join'] and sealed['all_model_parameters_buffers_unchanged'];paths=[]
 for strategy,index in sealed['scores'].items():
  assert len(index)==48
  for key,h in index.items():
   pool,pid=key.split('/',1);p=ROOT/'infer'/strategy/'scores'/pool/(pid+'.npz') if pool=='Catalogue' else ROOT/'infer'/strategy/'scores'/pool/pid/'UNLABELLED_POINT_SCORES.json';assert sha(p)==h;paths.append(p)
 assert len(paths)==288
 # Small metadata, failed attempts, code and bookkeeping. Explicitly exclude all weights.
 paths+=list(ROOT.glob('*.json'))+list(ROOT.glob('*.csv'))+[ROOT/'CALL_LEDGER.jsonl',ROOT/'EXIT_CODE']
 for directory in ('gate','gate_failure_01','train_12201','train_12301','infer','code','code_versions/initial'):
  p=ROOT/directory
  if p.exists():paths+=[x for x in p.iterdir() if x.is_file() and x.suffix in ('.json','.py')]
 paths=sorted(set(paths));assert all('.safetensors'!=p.suffix and 'work_dirs' not in p.parts for p in paths)
 manifest=dict(files={p.relative_to(ROOT).as_posix():sha(p) for p in paths},total_file_bytes=sum(p.stat().st_size for p in paths),neural_calls=len(events),GPU_stage_seconds=status['stage_seconds_total'],source_code_sha256=sha(Path(__file__)),no_weights=True,no_work_dirs=True,no_semantic_reference_uploaded=True)
 save(ROOT/'DELIVERY_MANIFEST.json',manifest);archive=ROOT/'phase205_delivery.zip'
 with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_STORED,allowZip64=True) as z:
  z.write(ROOT/'DELIVERY_MANIFEST.json','DELIVERY_MANIFEST.json')
  for p in paths:z.write(p,p.relative_to(ROOT).as_posix())
 receipt=dict(bytes=archive.stat().st_size,sha256=sha(archive),files=len(paths)+1,manifest_sha256=sha(ROOT/'DELIVERY_MANIFEST.json'),remote_new_bytes_after_archive=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()),CPU_seconds=time.time()-start,CPU_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,extra_neural_calls=0)
 assert receipt['remote_new_bytes_after_archive']+2*receipt['bytes']<6000000000;save(ROOT/'DELIVERY_RECEIPT.json',receipt);print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
