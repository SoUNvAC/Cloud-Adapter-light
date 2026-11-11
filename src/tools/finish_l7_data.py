"""Integrated continuation of fixed L7 preparation; no new experiment arms."""
from pathlib import Path
import json,subprocess,base64,sys,time

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/phase246/phase246_20261031/integrated_finish'
REMOTE='/home/scv/shared/phase246_20261031/integrated_finish_2015'

def main():
 mode=sys.argv[1] if len(sys.argv)>1 else 'status'
 init="from pathlib import Path\nimport json,os,subprocess,base64,hashlib\nroot=Path("+repr(REMOTE)+")\nout={}\n"
 program=init
 if mode=='start':
  assets=json.loads((OUT/'SELECTED_2015_ASSETS.json').read_text());sel=[x for x in assets if x['name'] in ['p26_r46.tar.gz','p35_r42.tar.gz','p39_r25.tar.gz']]
  urls={x['name']:x['url'] for x in sel};expected={x['name']:x['size'] for x in sel}
  code=(ROOT/'src/tools/phase246_remote_v2.py').read_text()
  code=code.replace("ROOT=Path('/home/scv/shared/phase246_20261031')","ROOT=Path("+repr(REMOTE)+")")
  for name,old in [('p26_r46.tar.gz',307791735),('p35_r42.tar.gz',255052281),('p39_r25.tar.gz',293324706)]:code=code.replace(str(old),str(expected[name]))
  code=code.replace("url='https://landsat.usgs.gov/cloud-validation/cca_irish_2015/'+filename","url="+repr(urls)+"[filename]")
  compile(code,'intake_2015.py','exec');(OUT/'intake_2015.py').write_text(code,encoding='utf-8')
  program+='''
root.mkdir(parents=True,exist_ok=True)
payload=base64.b64decode(PAYLOAD);script=root/'intake_2015.py'
if script.exists() and script.read_bytes()!=payload:raise RuntimeError('different existing script; no overwrite')
script.write_bytes(payload)
if not (root/'TASK_PID').exists():
 with (root/'task.log').open('ab') as log:p=subprocess.Popen(['/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python','-B','-u',str(script)],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True,env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})
 (root/'TASK_PID').write_text(str(p.pid));out['started_pid']=p.pid
'''.replace('PAYLOAD',repr(base64.b64encode(code.encode()).decode()))
 elif mode=='transfer_prepare':
  program+='''
import signal,time
pid=int((root/'TASK_PID').read_text());proc=Path('/proc')/str(pid)
if proc.exists() and str(root/'intake_2015.py').encode() in (proc/'cmdline').read_bytes():os.kill(pid,signal.SIGTERM)
receipt={'stopped_own_download_pid':pid,'partial_files_preserved':{p.name:p.stat().st_size for p in root.glob('*.part')},'reason':'local verified faster route; same fixed files only'}
(root/'TRANSFER_RECOVERY.json').write_text(json.dumps(receipt));out['transfer_recovery']=receipt
'''
 elif mode=='finalize_transferred':
  assets=json.loads((OUT/'SELECTED_2015_ASSETS.json').read_text());expected={x['name']:x['size'] for x in assets if x['name'] in ['p26_r46.tar.gz','p39_r25.tar.gz']}
  program+='''
for filename,size in EXPECTED.items():
 incoming=root/(filename+'.incoming');target=root/filename
 if target.exists():
  if target.stat().st_size!=size:raise RuntimeError('existing transferred size mismatch')
 elif incoming.exists() and incoming.stat().st_size==size:incoming.rename(target)
 else:raise RuntimeError('incoming transfer incomplete')
# Reuse the existing intake code only for archive validation; completed files are not downloaded.
with (root/'validation.log').open('ab') as log:
 p=subprocess.Popen(['/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python','-B','-u',str(root/'intake_2015.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True,env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})
(root/'TASK_PID').write_text(str(p.pid));out['validation_pid']=p.pid
'''.replace('EXPECTED',repr(expected))
 elif mode=='analyze':
  code=(ROOT/'src/tools/finish_l7_compare_remote.py').read_bytes()
  program+='''
script=root/'finish_l7_compare_remote.py';script.write_bytes(base64.b64decode(PAYLOAD))
if not (root/'ANALYZE_PID').exists():
 with (root/'analyze.log').open('ab') as log:p=subprocess.Popen(['/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python','-B','-u',str(script)],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True,env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','GDAL_CACHEMAX':'64'})
 (root/'ANALYZE_PID').write_text(str(p.pid));out['analysis_started_pid']=p.pid
'''.replace('PAYLOAD',repr(base64.b64encode(code).decode()))
 program+='''
if root.exists():
 out['scene_status']={p.name:json.loads(p.read_text()) for p in root.glob('STATUS*.json')}
 out['archives']={p.name:p.stat().st_size for p in root.glob('*.gz*') if p.is_file()}
 out['archive_receipts']={p.name:json.loads(p.read_text()) for p in root.glob('*_ARCHIVE.json')}
 for name in ['ARCHIVE_SUMMARY.json','FINAL_DATA_READINESS.json','ANALYSIS_STATUS.json']:
  if (root/name).exists():out[name]=json.loads((root/name).read_text())
 for name in ['task.log','analyze.log']:
  if (root/name).exists():out[name]=(root/name).read_text(errors='replace')[-1800:]
print(json.dumps(out))
'''
 attempts=[]
 for i in range(4):
  try:
   r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','ConnectionAttempts=1','-o','ServerAliveInterval=5','-o','ServerAliveCountMax=2','gzs','/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python -B -'],input=program,capture_output=True,text=True,encoding='utf-8',timeout=70)
   attempts.append({'attempt':i+1,'exit':r.returncode,'stderr':r.stderr[-1000:]})
   if r.returncode==0:
    v=json.loads(r.stdout);(OUT/'LATEST_REMOTE.json').write_text(json.dumps(v,indent=2),encoding='utf-8')
    for name,obj in v.get('archive_receipts',{}).items():(OUT/name).write_text(json.dumps(obj,indent=2),encoding='utf-8')
    for name in ['ARCHIVE_SUMMARY.json','FINAL_DATA_READINESS.json','ANALYSIS_STATUS.json']:
     if name in v:(OUT/name).write_text(json.dumps(v[name],indent=2),encoding='utf-8')
    brief={k:x for k,x in v.items() if k not in ['archive_receipts','FINAL_DATA_READINESS.json']}
    if 'FINAL_DATA_READINESS.json' in v:brief['final_summary']={k:x for k,x in v['FINAL_DATA_READINESS.json'].items() if k!='results'}
    print(json.dumps(brief,indent=2));break
  except Exception as e:attempts.append({'attempt':i+1,'error':str(e)[:600]})
  if i<3:time.sleep(3)
 (OUT/('SSH_'+mode+'_'+str(time.time_ns())+'.json')).write_text(json.dumps(attempts,indent=2),encoding='utf-8')

if __name__=='__main__':main()
