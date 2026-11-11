from collect_phase241 import ssh
print(ssh('''from pathlib import Path
import subprocess,os,json
r=Path('/home/scv/shared/phase446_20261106')
assert (r/'shadow_branch446.py').exists() and (r/'train_shadow446.py').exists()
assert not (r/'network_run').exists()
with (r/'network_run.log').open('w') as log:
 p=subprocess.Popen(['/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python','-u',str(r/'train_shadow446.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=r,env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4'))
(r/'NETWORK_LAUNCH.json').write_text(json.dumps(dict(pid=p.pid,steps=6000)))
print(json.dumps(dict(pid=p.pid)))
''').decode())
