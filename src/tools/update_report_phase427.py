from pathlib import Path
import subprocess,time
from collect_phase241 import ssh
p=Path(__file__).parent/'report_phase427.py'
for i in range(4):
 r=subprocess.run(['scp','-o','ConnectTimeout=10',str(p),'gzs:/home/scv/shared/phase427_20261101/report_phase427.py'],timeout=120)
 if r.returncode==0:break
 if i==3:raise RuntimeError('upload failed')
 time.sleep(5)
print(ssh("""from pathlib import Path
import json,hashlib
r=Path('/home/scv/shared/phase427_20261101');assert not (r/'report.log').exists()
lock={n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in ['watch_report_phase427.py','report_phase427.py']}
(r/'REPORT_CODE_LOCK.json').write_text(json.dumps(lock))
print('report audit updated before execution')
""").decode())
