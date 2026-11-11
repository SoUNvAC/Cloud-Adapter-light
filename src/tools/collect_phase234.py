"""One compact status request; completed counts gzip transferred, no pixel arrays."""
from pathlib import Path
import base64,gzip,hashlib,json,subprocess,time
R=Path(__file__).resolve().parents[2]/'outputs/phase234/phase234_20261027'
REMOTE='''from pathlib import Path
import base64,gzip,hashlib,json,sys,os
r=Path('/home/scv/shared/phase234_20261027');out={'status':{n:(r/n).read_text() for n in ['RUN_STATUS.json','EXIT_CODE','FAILURE.txt','RESOURCE_FAILURE.txt'] if (r/n).exists()}}
if (r/'EXIT_CODE').exists():
 files={}
 for p in r.rglob('*'):
  if p.is_file() and 'expected' not in p.parts and p.name not in ['bootstrap.zip']:
   assert p.stat().st_size<100_000_000;b=p.read_bytes();files[p.relative_to(r).as_posix()]={'SHA':hashlib.sha256(b).hexdigest(),'data':base64.b64encode(b).decode(),'bytes':len(b)}
 out.update(files=files,remote_persistent_bytes=sum(p.stat().st_size for p in r.rglob('*') if p.is_file()),remote_filesystem_available_bytes=os.statvfs(r).f_bavail*os.statvfs(r).f_frsize)
sys.stdout.buffer.write(gzip.compress(json.dumps(out).encode(),compresslevel=5))
'''
def main():
 cmd="/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python -B -c \"import base64;exec(base64.b64decode('"+base64.b64encode(REMOTE.encode()).decode()+"'))\""
 for a in range(4):
  p=subprocess.run(['ssh','-o','ConnectTimeout=10','-o','ServerAliveInterval=5','-o','ServerAliveCountMax=2','gzs',cmd],capture_output=True,timeout=180)
  if p.returncode==0:break
  if a<3:time.sleep(5)
 assert p.returncode==0,p.stderr.decode();v=json.loads(gzip.decompress(p.stdout));(R/'REMOTE_STATUS.json').write_text(json.dumps(v['status'],indent=2));print(json.dumps(v['status']))
 if 'files' not in v:return
 d=R/'delivery';d.mkdir(exist_ok=True);manifest={}
 for n,x in v.pop('files').items():
  pp=d/n;assert pp.resolve().is_relative_to(d.resolve());b=base64.b64decode(x['data']);assert len(b)==x['bytes'] and hashlib.sha256(b).hexdigest()==x['SHA'];pp.parent.mkdir(parents=True,exist_ok=True)
  if pp.exists():assert pp.read_bytes()==b,'Existing outputs preserved'
  else:pp.write_bytes(b)
  manifest[n]={'SHA':x['SHA'],'bytes':len(b)}
 v['manifest']=manifest;v['compressed_transport_bytes']=len(p.stdout);(R/'TRANSFER_ACCEPTED.json').write_text(json.dumps(v,indent=2));print('all returned report SHA and lengths verified')
if __name__=='__main__':main()
