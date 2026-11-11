"""Stream existing shared score cache without a remote persistent archive."""
import hashlib,json,os,subprocess,tarfile,time,traceback
from pathlib import Path,PurePosixPath
R=Path(__file__).resolve().parents[2]/'outputs/phase230/phase230_20261026'
PY='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
def save(n,v):(R/n).write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def main():
 assert not (R/'RECEIVER_PID').exists();(R/'RECEIVER_PID').write_text(str(os.getpid()));save('RECEIVER_STATUS.json',{'stage':'stream_transfer','PID':os.getpid(),'epoch':time.time()})
 attempts=[]
 for attempt in range(2,5):
  time.sleep(5);p=R/f'delivery_attempt{attempt}.tar.partial';assert not p.exists()
  assert sum(q.stat().st_size for q in R.rglob('*') if q.is_file())+160_200_000+170_000_000+160_200_000<512*1024**2,'Insufficient combined output allowance for preserved partial and another complete transfer'
  with p.open('xb') as out,(R/f'transfer_attempt{attempt}.error.log').open('x') as err:
   proc=subprocess.Popen(['ssh','-o','ConnectTimeout=10','-o','ServerAliveInterval=5','-o','ServerAliveCountMax=2','gzs','tar -C /home/scv/shared/phase230_20261026 -cf - .'],stdout=out,stderr=err);save('TRANSFER_PID.json',{'PID':proc.pid});rc=proc.wait()
  attempts.append({'attempt':attempt,'returncode':rc,'partial_bytes':p.stat().st_size});save('TRANSFER_ATTEMPTS.json',attempts)
  if rc==0:break
 assert rc==0,rc;assert p.stat().st_size<170_000_000
 save('RECEIVER_STATUS.json',{'stage':'extract_and_verify','PID':os.getpid(),'archive_bytes':p.stat().st_size});d=R/'delivery';assert not d.exists();d.mkdir()
 with tarfile.open(p,'r:') as z:
  for m in z.getmembers():
   n=PurePosixPath(m.name);assert not n.is_absolute() and '..' not in n.parts and (m.isfile() or m.isdir())
  z.extractall(d,filter='data')
 manifest=d/'DELIVERY_MANIFEST.json';assert sha(manifest)=='8ec4a64d7e26935fd48d17d4ae53ed031e31fb5d273e21b6a1a7fd17c6871d47'
 files=json.loads(manifest.read_text())
 for n,v in files.items():q=d/n;assert q.stat().st_size==v['bytes'] and sha(q)==v['SHA'],n
 assert len(files)==86
 actual=sum(q.stat().st_size for q in R.rglob('*') if q.is_file())+sum(v['bytes'] for v in files.values())+manifest.stat().st_size
 assert actual<512*1024**2
 save('TRANSFER_ACCEPTED.json',{'status':'passed','files':len(files),'all_file_SHA_and_length_passed':True,'archive_SHA':sha(p),'archive_bytes':p.stat().st_size,'remote_payload_bytes':sum(v['bytes'] for v in files.values()),'combined_new_bytes_so_far':actual,'no_remote_durable_archive':True})
 save('RECEIVER_STATUS.json',{'stage':'local_analysis','PID':os.getpid()});(R/'RECEIVER_EXIT_CODE').write_text('0\n')
 for name in ['analyze_phase230.py','validate_phase230.py']:
  with (R/(name+'.console.log')).open('x') as out,(R/(name+'.error.log')).open('x') as err:rc=subprocess.run([PY,'-X','utf8','-B',str(Path(__file__).parent/name)],stdout=out,stderr=err).returncode
  assert rc==0,(name,rc)
 save('RECEIVER_STATUS.json',{'stage':'CPU_analysis_and_validation_complete','PID':os.getpid()});(R/'LOCAL_EXIT_CODE').write_text('0\n')
if __name__=='__main__':
 try:main()
 except BaseException:
  (R/'RECEIVER_FAILURE.txt').write_text(traceback.format_exc());save('RECEIVER_STATUS.json',{'stage':'failed','PID':os.getpid()});(R/'LOCAL_EXIT_CODE').write_text('1\n');raise
