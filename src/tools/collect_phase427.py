"""Compressed transfer of text evidence; score/feature arrays remain remote."""
from pathlib import Path
import json,hashlib,zlib,base64
from collect_phase241 import ssh
R=Path(__file__).resolve().parents[2]/'outputs/phase427/phase427_20261101'
def main():
 status=json.loads(ssh("""from pathlib import Path
import json
r=Path('/home/scv/shared/phase427_20261101');v={}
for n in ['FAILURE.json','RUN_STATUS.json','ACCEPTANCE.json','COMPLETE.json']:
 if (r/n).exists():v[n]=json.loads((r/n).read_text())
if (r/'EXIT_CODE').exists():v['exit_code']=(r/'EXIT_CODE').read_text()
print(json.dumps(v))
"""))
 (R/'LAST_STATUS.json').write_text(json.dumps(status,indent=2));print(json.dumps(status),flush=True)
 if 'exit_code' not in status:return
 raw=ssh("""from pathlib import Path
import json,hashlib,zlib,base64
r=Path('/home/scv/shared/phase427_20261101');out={}
for p in r.rglob('*'):
 if p.is_file() and p.suffix in ['.json','.jsonl','.csv','.py','.md'] and not any(x in p.parts for x in ['configs','scratch']):
  b=p.read_bytes();out[p.relative_to(r).as_posix()]={'text':b.decode(),'SHA':hashlib.sha256(b).hexdigest()}
print(base64.b64encode(zlib.compress(json.dumps(out).encode())).decode())
""")
 files=json.loads(zlib.decompress(base64.b64decode(raw)));manifest={}
 for n,v in files.items():
  p=R/n;assert p.resolve().is_relative_to(R.resolve());p.parent.mkdir(parents=True,exist_ok=True);b=v['text'].encode();assert hashlib.sha256(b).hexdigest()==v['SHA'];p.write_bytes(b);manifest[n]=v['SHA']
 (R/'TRANSFER_MANIFEST.json').write_text(json.dumps(manifest,indent=2));print('transferred',len(manifest),'files')
if __name__=='__main__':main()
