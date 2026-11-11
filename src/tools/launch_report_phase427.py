from pathlib import Path
import subprocess,json,hashlib
from collect_phase241 import ssh
T=Path(__file__).parent
p=T/'watch_report_phase427.py'
p.write_text((T/'watch_report_phase425.py').read_text().replace('report_phase425.py','report_phase427.py'),encoding='utf-8')
paths=[p,T/'report_phase427.py']
subprocess.run(['scp','-o','ConnectTimeout=10',*[str(x) for x in paths],'gzs:/home/scv/shared/phase427_20261101/'],check=True,timeout=120)
print(ssh("""from pathlib import Path
import subprocess,hashlib,json
r=Path('/home/scv/shared/phase427_20261101')
files={n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in ['watch_report_phase427.py','report_phase427.py']}
(r/'REPORT_CODE_LOCK.json').write_text(json.dumps(files))
with (r/'report_watcher.log').open('x') as f:
 p=subprocess.Popen(['/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python','-B',str(r/'watch_report_phase427.py'),'427'],cwd=r,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(p.pid)
""").decode())
