"""Read one status snapshot; transfer only explicitly named small reports."""
import json,os
from pathlib import Path
from collect_phase241 import ssh
L=Path(__file__).resolve().parents[2]/'outputs/phase403/phase403_20261031'
R='/home/scv/shared/phase403_20261031'
def main():
    script=f"""from pathlib import Path
import json,os
r=Path('{R}');v={{}}
for n in ['LAUNCH.json','STATUS.json','RUN_INFO.json','ENDPOINT_GATE.json','FAILURE.json']:
 if (r/n).exists():v[n]=json.loads((r/n).read_text())
if (r/'EXIT_CODE').exists():v['exit_code']=(r/'EXIT_CODE').read_text()
if 'LAUNCH.json' in v:
 try:os.kill(v['LAUNCH.json']['PID'],0);v['alive']=True
 except ProcessLookupError:v['alive']=False
print(json.dumps(v))
"""
    data=ssh(script);(L/'LAST_STATUS.json').write_bytes(data);v=json.loads(data);print(json.dumps(v))
    if v.get('exit_code')!='0':return
    if (L/'TRANSFER_ACCEPTED.json').exists():return
    names=['summary.csv','per_group.csv','source_per_image.csv','RUN_INFO.json','INPUT_EXPECTED.json','ENDPOINT_GATE.json','CALL_LEDGER.jsonl']
    script=f"""from pathlib import Path
import json,hashlib
r=Path('{R}');out={{}}
for n in {names!r}:
 b=(r/n).read_bytes();assert len(b)<10*1024**2;out[n]={{'text':b.decode(),'SHA':hashlib.sha256(b).hexdigest()}}
print(json.dumps(out))
"""
    import hashlib
    files=json.loads(ssh(script))
    for n,item in files.items():
        b=item['text'].encode();assert hashlib.sha256(b).hexdigest()==item['SHA'];(L/n).write_bytes(b)
    (L/'TRANSFER_ACCEPTED.json').write_text(json.dumps({n:v['SHA'] for n,v in files.items()},indent=2))
if __name__=='__main__':main()
