from pathlib import Path
import subprocess,json,hashlib
from collect_phase241 import ssh
root=Path(__file__).resolve().parent
for phase in [420]:
 remote=f'/home/scv/shared/phase{phase}_20261101'
 files=[root/'report_phase420.py',root/'watch_report_phase420.py']
 subprocess.run(['scp','-o','ConnectTimeout=10',*[str(p) for p in files],'gzs:'+remote+'/'],check=True,timeout=120)
 hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
 script=f"""from pathlib import Path
import subprocess,json,hashlib,os
r=Path('{remote}');hashes={hashes!r}
for n,h in hashes.items():assert hashlib.sha256((r/n).read_bytes()).hexdigest()==h
(r/'REPORT_CODE_LOCK.json').write_text(json.dumps(hashes))
with (r/'report_watcher.log').open('x') as f:
 p=subprocess.Popen(['/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python','-B','-u',str(r/'watch_report_phase420.py'),'{phase}'],cwd=r,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({{'phase':{phase},'PID':p.pid}}))
"""
 print(ssh(script).decode())
