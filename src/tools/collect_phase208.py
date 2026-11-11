"""Exclusive controlled Phase208 score/text archive; no original/new weights or images."""
import hashlib,json,zipfile,resource,time
from pathlib import Path
R=Path('/home/scv/shared/phase208_20261020')
def load(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def save(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False)
def main():
 start=time.time();assert (R/'EXIT_CODE').read_text().strip()=='0';status=load(R/'CONTROLLER_STATUS.json');events=[json.loads(v) for v in (R/'CALL_LEDGER.jsonl').read_text().splitlines()];assert len(events)==status['total_calls']<=5000 and status['stage_seconds_total']<=14400;assert [q['global_call'] for q in events]==list(range(1,len(events)+1))
 for arm in ('M','B'):
  q=load(R/f'train_{arm}/TRAINING_RECEIPT.json');assert q['steps']==q['optimizer_steps']==1000 and q['original_parameters_buffers_unchanged'] and q['original_all_eval'];assert sha(R/f'train_{arm}/module_final.safetensors')==q['final_sha256']==load(R/'TRAINING_SEAL.json')[arm]
 seal=load(R/'infer/SCORES_SEALED.json');assert seal['all_model_parameters_buffers_unchanged'] and set(seal['scores'])=={'M','B'};paths=[]
 for name,index in seal['scores'].items():
  assert len(index)==48
  for k,h in index.items():
   pool,pid=k.split('/',1);p=R/'infer'/name/'scores'/pool/((pid+'.npz') if pool=='Catalogue' else pid+'/UNLABELLED_POINT_SCORES.json');assert sha(p)==h;paths.append(p)
 assert load(R/'benchmark/BENCHMARK_ACCEPTANCE.json')['status']=='passed'
 paths+=list(R.glob('*.json'))+[R/'CALL_LEDGER.jsonl',R/'EXIT_CODE']
 for stage in ('gate','train_M','train_B','infer','benchmark','code'):
  paths += [p for p in (R/stage).iterdir() if p.is_file() and p.suffix in ('.json','.py')]
 # Any failed engineering material stays archived with its exact budget, never overwritten.
 for directory in R.glob('*failure*'):
  if directory.is_dir():paths += [p for p in directory.rglob('*') if p.is_file() and p.suffix in ('.json','.py','.log','.jsonl')]
 paths=sorted(set(paths));assert all(p.suffix not in ('.safetensors','.npy') and 'work_dirs' not in p.parts for p in paths)
 manifest=dict(files={p.relative_to(R).as_posix():sha(p) for p in paths},network_calls=len(events),GPU_stage_seconds=status['stage_seconds_total'],no_weights_or_images=True,no_reference_uploaded=True,no_work_dirs=True);save(R/'DELIVERY_MANIFEST.json',manifest)
 archive=R/'phase208_delivery.zip'
 with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_STORED,allowZip64=True) as z:
  z.write(R/'DELIVERY_MANIFEST.json','DELIVERY_MANIFEST.json')
  for p in paths:z.write(p,p.relative_to(R).as_posix())
 size=sum(p.stat().st_size for p in R.rglob('*') if p.is_file());assert size+2*archive.stat().st_size<6000000000
 save(R/'DELIVERY_RECEIPT.json',dict(bytes=archive.stat().st_size,sha256=sha(archive),files=len(paths)+1,remote_new_bytes=size,combined_upper_before_local_reports=size+2*archive.stat().st_size,CPU_seconds=time.time()-start,CPU_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,extra_neural_calls=0));print(json.dumps(load(R/'DELIVERY_RECEIPT.json')))
if __name__=='__main__':main()
