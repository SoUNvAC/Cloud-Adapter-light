"""Retrieve only SHA-verified small reports; remote d3 statistics stay in shared."""
from pathlib import Path
import base64,hashlib,json,subprocess,time
R=Path(__file__).resolve().parents[2]/'outputs/phase233/phase233_20261027'
REMOTE='''from pathlib import Path
import base64,hashlib,json,os
r=Path('/home/scv/shared/phase233_20261027')
out={'status':{n:(r/n).read_text() for n in ['RUN_STATUS.json','EXIT_CODE','FAILURE.txt','RESOURCE_FAILURE.txt'] if (r/n).exists()}}
if (r/'EXIT_CODE').exists() and (r/'EXIT_CODE').read_text().strip()=='0':
 files={}
 for p in r.rglob('*'):
  if p.is_file() and 'd_cache' not in p.parts:
   assert p.stat().st_size<4000000;b=p.read_bytes();files[p.relative_to(r).as_posix()]={'SHA':hashlib.sha256(b).hexdigest(),'data':base64.b64encode(b).decode(),'bytes':len(b)}
 out.update(files=files,remote_persistent_bytes=sum(p.stat().st_size for p in r.rglob('*') if p.is_file()),evaluation_files=len(list((r/'evaluation').glob('*.json'))),remote_host_filesystem_available_bytes=os.statvfs(r).f_bavail*os.statvfs(r).f_frsize)
print(json.dumps(out))
'''
def main():
 cmd="/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python -B -c \"import base64;exec(base64.b64decode('"+base64.b64encode(REMOTE.encode()).decode()+"'))\""
 for a in range(4):
  p=subprocess.run(['ssh','-o','ConnectTimeout=10','-o','ServerAliveInterval=5','-o','ServerAliveCountMax=2','gzs',cmd],capture_output=True,timeout=90)
  if p.returncode==0:break
  if a<3:time.sleep(5)
 assert p.returncode==0,p.stderr.decode();v=json.loads(p.stdout);(R/'REMOTE_STATUS.json').write_text(json.dumps(v['status'],indent=2));print(json.dumps(v['status']))
 if 'files' not in v:return
 d=R/'delivery';d.mkdir(exist_ok=True);manifest={}
 for n,entry in v['files'].items():
  pp=d/n;assert pp.resolve().is_relative_to(d.resolve());b=base64.b64decode(entry['data']);assert hashlib.sha256(b).hexdigest()==entry['SHA'] and len(b)==entry['bytes'];pp.parent.mkdir(parents=True,exist_ok=True)
  if pp.exists():assert pp.read_bytes()==b,'Preserve old different output'
  else:pp.write_bytes(b)
  manifest[n]={'SHA':entry['SHA'],'bytes':len(b)}
 v.pop('files');v['manifest']=manifest;(R/'TRANSFER_ACCEPTED.json').write_text(json.dumps(v,indent=2));print('small reports verified; counts only; no logits or reference cache transferred')
if __name__=='__main__':main()
