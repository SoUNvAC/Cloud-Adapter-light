"""Small metadata-only full backup; no archive, pixels, inference or statistics locally."""
import sys,json,hashlib,base64,subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[2];O=Path('/home/scv/shared/phase321_20261028');L=R/'outputs/phase321/phase321_20261028'
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
 if sys.argv[1]=='stream':
  assert sys.platform=='linux';files={}
  for p in sorted(O.rglob('*')):
   if p.is_file():
    b=p.read_bytes();files[p.relative_to(O).as_posix()]={'sha256':sha(b),'bytes':len(b),'base64':base64.b64encode(b).decode()}
  assert sum(x['bytes'] for x in files.values())<100000000
  print(json.dumps({'files':files}));return
 if sys.argv[1]=='receive':
  assert not (L/'remote_backup').exists();L.mkdir(parents=True,exist_ok=True)
  cmd=['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','CheckHostIP=no','-o','UpdateHostKeys=no','-o','ConnectTimeout=30','gzs','source /home/scv/miniconda3/etc/profile.d/conda.sh && conda activate cloud-lite-pt210 && cd /home/scv/Cloud-Adapter-light && python -B src/tools/deliver_phase321.py stream']
  p=subprocess.run(cmd,capture_output=True);(L/'TRANSFER_STDERR.log').write_bytes(p.stderr)
  assert p.returncode==0;d=json.loads(p.stdout);out=L/'remote_backup';out.mkdir();receipt={}
  for n,e in d['files'].items():
   p=(out/n).resolve();assert p.is_relative_to(out.resolve());b=base64.b64decode(e['base64']);assert len(b)==e['bytes'] and sha(b)==e['sha256'];p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b);receipt[n]={'bytes':len(b),'sha256':sha(p.read_bytes())}
  (L/'BACKUP_RECEIPT.json').write_text(json.dumps({'status':'completed','files':receipt},indent=2)+'\n');print(json.dumps({'status':'completed','files':len(receipt),'bytes':sum(e['bytes'] for e in receipt.values())}));return
 raise ValueError('stream or receive')
if __name__=='__main__':main()
