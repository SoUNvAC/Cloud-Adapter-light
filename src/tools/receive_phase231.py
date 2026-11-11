"""Verified per-file streaming, resumable without duplicate durable TAR archives."""
from pathlib import Path
import base64,hashlib,json,os,subprocess,time,traceback
R=Path(__file__).resolve().parents[2]/'outputs/phase231/phase231_20261026';PY='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
def js(p):return json.loads(p.read_text(encoding='utf-8'))
def save(n,v):(R/n).write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
REMOTE='''from pathlib import Path
import json,sys,hashlib
r=Path('/home/scv/shared/phase231_20261026');assert (r/'EXIT_CODE').read_text().strip()=='0';m=json.loads((r/'DELIVERY_MANIFEST.json').read_text());request=json.load(sys.stdin)
for name in request:
 assert name=='DELIVERY_MANIFEST.json' or name in m
 p=r/name;assert p.resolve().is_relative_to(r) and p.is_file()
 h={'name':name,'bytes':p.stat().st_size,'SHA':hashlib.sha256(p.read_bytes()).hexdigest()};sys.stdout.buffer.write((json.dumps(h)+'\\n').encode());sys.stdout.buffer.flush()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(65536),b''):sys.stdout.buffer.write(b)
 sys.stdout.buffer.flush()
sys.stdout.buffer.write(b'{"end":true}\\n');sys.stdout.buffer.flush()
'''
def main():
 assert not (R/'RECEIVER_PID').exists();(R/'RECEIVER_PID').write_text(str(os.getpid()));D=R/'delivery';D.mkdir(exist_ok=True);m=None;attempts=[];remote=js(R/'REMOTE_DELIVERY_READY.json');expected_manifest=remote['manifest_SHA']
 for attempt in range(21,25):
  if attempt>21:time.sleep(5)
  if (D/'DELIVERY_MANIFEST.json').exists():assert sha(D/'DELIVERY_MANIFEST.json')==expected_manifest;m=js(D/'DELIVERY_MANIFEST.json')
  names=['DELIVERY_MANIFEST.json'] if m is None else [n for n,v in m.items() if not (D/n).is_file() or (D/n).stat().st_size!=v['bytes'] or sha(D/n)!=v['SHA']]
  if not names and m is not None:break
  assert sum(p.stat().st_size for p in R.rglob('*') if p.is_file())+remote['bytes']+10_000_000<512*1024**2
  cmd='/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python -B -c "import base64;exec(base64.b64decode(\''+base64.b64encode(REMOTE.encode()).decode()+'\'))"'
  with (R/f'transfer_attempt{attempt}.error.log').open('x') as err:
   proc=subprocess.Popen(['ssh','-o','ConnectTimeout=10','-o','ServerAliveInterval=5','-o','ServerAliveCountMax=2','gzs',cmd],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err);save('RECEIVER_STATUS.json',{'stage':'stream','PID':os.getpid(),'transfer_PID':proc.pid,'attempt':attempt,'remaining_files':len(names)})
   completed=0
   try:
    proc.stdin.write(json.dumps(names).encode());proc.stdin.close()
    while True:
     line=proc.stdout.readline(16384);assert line,'SSH EOF before end record';head=json.loads(line)
     if head.get('end'):break
     name=head['name'];assert name in names and not Path(name).is_absolute() and '..' not in Path(name).parts;p=D/name;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+f'.partial_attempt{attempt}');assert not tmp.exists();remaining=head['bytes'];h=hashlib.sha256()
     with tmp.open('xb') as f:
      while remaining:
       b=proc.stdout.read(min(65536,remaining));assert b,'Partial file stream';f.write(b);h.update(b);remaining-=len(b)
     assert h.hexdigest()==head['SHA'];assert name=='DELIVERY_MANIFEST.json' or (head['SHA']==m[name]['SHA'] and head['bytes']==m[name]['bytes'])
     assert not p.exists(),'Existing corrupt output preserved; do not overwrite';tmp.rename(p);completed+=1
     if completed%20==0:save('RECEIVER_STATUS.json',{'stage':'stream','PID':os.getpid(),'attempt':attempt,'completed_in_stream':completed,'stream_files':len(names),'current':name})
    rc=proc.wait();assert rc==0
   except BaseException:
    if proc.poll() is None:proc.kill()
    proc.wait();attempts.append({'attempt':attempt,'completed_files':completed,'failure':traceback.format_exc()});save('TRANSFER_ATTEMPTS.json',attempts);continue
  attempts.append({'attempt':attempt,'completed_files':completed,'returncode':rc});save('TRANSFER_ATTEMPTS.json',attempts)
  # Manifest handshake uses one of the at-most-four bounded streams.
  if m is None:
   m=js(D/'DELIVERY_MANIFEST.json');assert sha(D/'DELIVERY_MANIFEST.json')==expected_manifest
   # Request all remaining cache files in a separate bounded stream below.
   save('RECEIVER_MANIFEST_ACCEPTED.json',{'manifest_SHA':expected_manifest,'files':len(m)})
 if m is None or not all((D/n).is_file() and (D/n).stat().st_size==v['bytes'] and sha(D/n)==v['SHA'] for n,v in m.items()):raise RuntimeError('Transfer incomplete after bounded streams; do not restart model or overwrite completed files')
 assert sha(D/'BIAS_FROZEN.json')==js(D/'BIAS_SEAL.json')['BIAS_FROZEN_SHA'];save('TRANSFER_ACCEPTED.json',{'status':'passed','files':len(m),'bytes':sum(v['bytes'] for v in m.values()),'all_lengths_SHA_pass':True,'no_TAR_or_duplicate_score_archive':True});(R/'RECEIVER_EXIT_CODE').write_text('0\n')
 for n,receipt in [('validate_phase231_bias.py','BIAS_ACCEPTED_LOCAL.json'),('analyze_phase231.py','CPU_ANALYSIS_COMPLETE.json'),('validate_phase231.py','INDEPENDENT_ACCEPTANCE.json')]:
  if (R/receipt).exists() and js(R/receipt)['status']=='passed':continue
  save('RECEIVER_STATUS.json',{'stage':n,'PID':os.getpid()})
  with (R/(n+'.console.log')).open('x') as out,(R/(n+'.error.log')).open('x') as err:rc=subprocess.run([PY,'-X','utf8','-B',str(Path(__file__).parent/n)],stdout=out,stderr=err,timeout=3600).returncode
  assert rc==0,(n,rc)
 save('RECEIVER_STATUS.json',{'stage':'CPU_validation_complete','PID':os.getpid()});(R/'LOCAL_EXIT_CODE').write_text('0\n')
if __name__=='__main__':
 try:main()
 except BaseException:(R/'RECEIVER_FAILURE.txt').write_text(traceback.format_exc());(R/'LOCAL_EXIT_CODE').write_text('1\n');raise
