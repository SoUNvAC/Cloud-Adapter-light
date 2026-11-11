"""Phase206 exclusive score/receipt archive; no weights, work_dirs or reference."""
import hashlib,json,time,zipfile,resource
from pathlib import Path
R=Path('/home/scv/shared/phase206_20261019')
def load(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def save(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False)
def main():
 start=time.time();assert (R/'EXIT_CODE').read_text().strip()=='0';s=load(R/'CONTROLLER_STATUS.json');events=[json.loads(v) for v in (R/'CALL_LEDGER.jsonl').read_text().splitlines()];assert len(events)==s['total_calls']<=8000 and [q['global_call'] for q in events]==list(range(1,len(events)+1));assert s['stage_seconds_total']<=14400
 assert not any(q['kind']=='training' for q in events);assert load(R/'RECONSTRUCTION_ACCEPTANCE.json')['status']=='passed'
 paths=[]
 for stage in ('reconstruction','infer'):
  seal=load(R/stage/'SCORES_SEALED.json');assert seal['all_model_parameters_buffers_unchanged']
  for name,index in seal['scores'].items():
   assert len(index)==(2 if stage=='reconstruction' else 48)
   for k,h in index.items():
    pool,pid=k.split('/',1);p=R/stage/name/'scores'/pool/((pid+'.npz') if pool=='Catalogue' else pid+'/UNLABELLED_POINT_SCORES.json');assert sha(p)==h;paths.append(p)
 paths+=list(R.glob('*.json'))+[R/'CALL_LEDGER.jsonl',R/'EXIT_CODE']
 for stage in ('reconstruction','infer','code'):paths+=[p for p in (R/stage).iterdir() if p.is_file() and p.suffix in ('.json','.py')]
 paths=sorted(set(paths));assert all(p.suffix!='.safetensors' and 'work_dirs' not in p.parts for p in paths)
 manifest=dict(files={p.relative_to(R).as_posix():sha(p) for p in paths},neural_calls=len(events),GPU_stage_seconds=s['stage_seconds_total'],weights=False,semantic_reference=False,work_dirs=False)
 save(R/'DELIVERY_MANIFEST.json',manifest);archive=R/'phase206_delivery.zip'
 with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_STORED,allowZip64=True) as z:
  z.write(R/'DELIVERY_MANIFEST.json','DELIVERY_MANIFEST.json')
  for p in paths:z.write(p,p.relative_to(R).as_posix())
 size=sum(p.stat().st_size for p in R.rglob('*') if p.is_file());assert size+2*archive.stat().st_size<6000000000
 save(R/'DELIVERY_RECEIPT.json',dict(bytes=archive.stat().st_size,sha256=sha(archive),files=len(paths)+1,remote_new_bytes=size,combined_upper_bound_before_local_reports=size+2*archive.stat().st_size,CPU_seconds=time.time()-start,peak_CPU_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,extra_calls=0))
 print(json.dumps(load(R/'DELIVERY_RECEIPT.json')))
if __name__=='__main__':main()
