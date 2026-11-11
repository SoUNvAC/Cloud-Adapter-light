from collect_phase241 import ssh
print(ssh('''from pathlib import Path
import subprocess,os,json
r=Path('/home/scv/shared/phase446_20261106')
code="import time,subprocess; from pathlib import Path; r=Path('/home/scv/shared/phase446_20261106'); " + "\\nwhile not (r/'network_run/COMPLETE.json').exists():\\n if (r/'network_run/FAILURE.json').exists(): raise SystemExit(1)\\n time.sleep(10)\\nsubprocess.run(['/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python','-u',str(r/'verify_shadow446.py')],check=True)"
with (r/'verify.log').open('w') as log:
 p=subprocess.Popen(['/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python','-u','-c',code],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=r,env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4'))
print(json.dumps(dict(verification_watcher_pid=p.pid)))
''').decode())
