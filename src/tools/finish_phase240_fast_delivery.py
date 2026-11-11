"""One hidden receiver: wait for G exit on one SSH connection, then CPU delivery.

The waiting process consumes no model calls or agent polling turns. It never
launches a training job and preserves any failed stage for the hourly agent.
"""
from pathlib import Path
import base64,hashlib,importlib.util,json,os,subprocess,sys,time,traceback,zipfile
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'outputs/phase240/phase240_20261030'
STATUS=R/'LOCAL_FAST_DELIVERY_STATUS.json'

def js(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def state(stage,**kwargs):
 p=STATUS.with_suffix('.tmp');p.write_text(json.dumps(dict(PID=os.getpid(),stage=stage,epoch=time.time(),**kwargs),indent=2),encoding='utf-8');p.replace(STATUS)

def pack():
 D=R/'delivery_recovery03';Q=D/'recovery03';p=R/'PHASE240_G_RECOVERY_DELIVERY_20261030.zip';assert not p.exists()
 names=['RESULT.md','TABLE1.csv','TABLE2.csv','TABLE3_ALL_SCENES.csv','VAL_ALL_CANDIDATES.csv','PER_SCENE.csv','L8_TEST_STRATA.csv','TRAIN_BUDGET.csv','SCENE_EXPOSURE.csv','HEAD_RESUME_IDENTITIES.csv','GRADIENT_DIAGNOSTIC_SUMMARY.csv','A_MASK_INVARIANCE.csv','DIAGNOSTICS.json','COST.json','INDEPENDENT_ACCEPTANCE.json']
 entries={n:R/n for n in names}
 for f in Q.rglob('*'):
  if f.is_file() and f.suffix not in ['.pth','.npz','.npy','.zip']:entries['G_recovery03/'+f.relative_to(Q).as_posix()]=f
 for sub in ['recovery01','recovery02','stage_costs']:
  for f in (D/sub).rglob('*'):
   if f.is_file() and f.suffix not in ['.pth','.npz','.npy','.zip']:entries['preserved_history/'+sub+'/'+f.relative_to(D/sub).as_posix()]=f
 for sub in ['L8_G','SPARCS_G']:
  for f in (D/'fits'/sub).rglob('*'):
   if f.is_file() and f.suffix not in ['.pth','.npz','.npy','.zip']:entries['preserved_history/initial_failed_fits/'+sub+'/'+f.name]=f
 for n in ['CALL_LEDGER.jsonl','COMPUTE_LEDGER.jsonl','BUNDLE_SHA.json','VALID_MASK_REUSE.json']:
  entries['global_ledger/'+n]=D/n
 manifest={n:dict(SHA256=sha(f),bytes=f.stat().st_size) for n,f in entries.items()}
 with zipfile.ZipFile(p,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for n,f in entries.items():z.write(f,n)
  z.writestr('PACKAGE_SHA256.json',json.dumps(manifest,indent=2))
  z.writestr('README.txt','Phase240 final tables and G engineering-recovery receipts/code/identity/ledger package. No pth or large arrays are included. Actual G heads and optimizer/RNG resumes remain in authorized shared/recovery03/fits. All prior engineering failures and costs remain in preserved_history. Read RESULT.md: data results and interpretation are separate.\n')
 with zipfile.ZipFile(p) as z:
  assert z.testzip() is None
  for n,v in manifest.items():assert hashlib.sha256(z.read(n)).hexdigest()==v['SHA256']
 receipt=dict(status='passed',file=p.name,SHA256=sha(p),bytes=p.stat().st_size,entries=len(manifest),CRC_and_each_entry_SHA_verified=True,weights_not_transferred=True)
 (R/'G_RECOVERY_PACKAGE_ACCEPTED.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
 return receipt

def main():
 assert not STATUS.exists(),'Receiver already dispatched; inspect PID rather than duplicate'
 state('waiting_for_G_exit',neural_calls=0)
 spec=importlib.util.spec_from_file_location('collector',ROOT/'src/tools/collect_phase240.py');collector=importlib.util.module_from_spec(spec);spec.loader.exec_module(collector)
 wait='''from pathlib import Path
import time,os,json
q=Path('/home/scv/shared/phase240_20261030/recovery03');deadline=time.monotonic()+7200
while not (q/'EXIT_CODE').exists():
 if time.monotonic()>deadline:raise TimeoutError('G completion wait exceeded2h; no training restarted')
 pid=json.loads((q/'LAUNCH.json').read_text())['PID'];p=Path('/proc')/str(pid)
 if not p.exists() or (p/'stat').read_text().split()[2]=='Z':raise RuntimeError('G controller exited without EXIT_CODE; inspect preserved error')
 time.sleep(60)
'''
 remote=wait+collector.REMOTE
 cmd='/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python -B -c "import base64;exec(base64.b64decode(\''+base64.b64encode(remote.encode()).decode()+'\'))"'
 response=None
 for attempt in range(4):
  proc=subprocess.run(['ssh','-o','ConnectTimeout=10','-o','ServerAliveInterval=5','-o','ServerAliveCountMax=2','gzs',cmd],capture_output=True,timeout=7500)
  (R/f'fast_receiver_ssh_attempt{attempt}.error.log').write_bytes(proc.stderr)
  if proc.returncode==0:response=proc.stdout;break
  # Retry connection problems, never a remote code/budget assertion.
  if b'Traceback' in proc.stderr:raise RuntimeError(proc.stderr.decode(errors='replace'))
  if attempt<3:time.sleep(5)
 assert response is not None,'SSH receiver connection failed; no jobs restarted'
 state('receiving_and_SHA_verifying',compressed_bytes=len(response))
 import gzip
 result=json.loads(gzip.decompress(response));D=R/'delivery_recovery03';D.mkdir(exist_ok=True);manifest={}
 assert result['status']['recovery03/EXIT_CODE'].strip() in ['0','2']
 for n,x in result.pop('files').items():
  pp=D/n;assert pp.resolve().is_relative_to(D.resolve());b=base64.b64decode(x['data']);assert len(b)==x['bytes'] and hashlib.sha256(b).hexdigest()==x['SHA'];pp.parent.mkdir(parents=True,exist_ok=True)
  if pp.exists():assert pp.read_bytes()==b,'Existing snapshot preserved'
  else:pp.write_bytes(b)
  manifest[n]={'SHA':x['SHA'],'bytes':len(b)}
 result['manifest']=manifest;result['compressed_bytes']=len(response);(R/'TRANSFER_ACCEPTED.json').write_text(json.dumps(result,indent=2),encoding='utf-8');(R/'REMOTE_STATUS.json').write_text(json.dumps(result['status'],indent=2),encoding='utf-8')
 for n,h in js(R/'LOCAL_ANALYSIS_CODE_SEAL.json').items():assert sha(ROOT/'src/tools'/n)==h,n
 for name,done in [('analyze_phase240.py','CPU_ANALYSIS_COMPLETE.json'),('validate_phase240.py','INDEPENDENT_ACCEPTANCE.json'),('finalize_phase240.py','DELIVERY_COMPLETE.json')]:
  state('CPU_'+name)
  assert not (R/done).exists(),'Do not overwrite completed CPU stage'
  with (R/(name+'.console.log')).open('x') as out,(R/(name+'.error.log')).open('x') as err:
   p=subprocess.run([sys.executable,'-X','utf8','-B',str(ROOT/'src/tools'/name)],cwd=ROOT,stdout=out,stderr=err,timeout=10800)
  if p.returncode:raise RuntimeError(name+' failed; see preserved CPU error log; no model work repeated')
 state('packaging_G_recovery')
 receipt=pack()
 (R/'HANDOFF.md').write_text('# Current delivery — COMPLETE\n\nFinal RESULT.md, three tables, full scene CSVs and PHASE240_G_RECOVERY_DELIVERY_20261030.zip are local. Independent CPU acceptance, ZIP CRC and each packaged SHA passed. See G_RECOVERY_PACKAGE_ACCEPTED.json and COST.json for actual completion/partial status. No further neural calls or transfer required. Notify user with links and delete phase240-hourly-delivery automation. No new phase.\n',encoding='utf-8')
 state('complete',package=receipt)
 complete=js(R/'DELIVERY_COMPLETE.json');complete['G_recovery_package']=receipt;(R/'DELIVERY_COMPLETE.json').write_text(json.dumps(complete,indent=2),encoding='utf-8')
 excluded={'DELIVERY_MANIFEST.json','DELIVERY_COMPLETE.json','LOCAL_FAST_RECEIVER.console.log','LOCAL_FAST_RECEIVER.error.log'}
 mf={p.relative_to(R).as_posix():sha(p) for p in R.rglob('*') if p.is_file() and p.name not in excluded and 'cpu_smoke_partial_v01' not in p.parts}
 (R/'DELIVERY_MANIFEST.json').write_text(json.dumps(mf,indent=2),encoding='utf-8')
 print(json.dumps(receipt),flush=True)

if __name__=='__main__':
 try:main()
 except BaseException:
  state('failed',error=traceback.format_exc(),no_model_calls_added_by_receiver=True);raise
