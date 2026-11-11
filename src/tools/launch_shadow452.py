from collect_phase241 import ssh
import sys
arm=sys.argv[1];assert arm in ['A','B']
print(ssh('''from pathlib import Path
import subprocess,os,json
arm='ARM_PLACEHOLDER';r=Path('/home/scv/shared/phase452_20261106');d=r/arm
assert not (d/'network_run').exists()
code="import subprocess; from pathlib import Path; r=Path('/home/scv/shared/phase452_20261106'); exe='/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python'; subprocess.run([exe,'-u',str(r/'train_shadow452.py'),'"+arm+"'],check=True); subprocess.run([exe,'-u',str(r/'verify_shadow452.py'),'"+arm+"'],check=True)"
with (d/'pipeline.log').open('w') as log:
 p=subprocess.Popen(['/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python','-u','-c',code],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=r,env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4'))
(d/'LAUNCH.json').write_text(json.dumps(dict(pid=p.pid,steps=6000)));print(json.dumps(dict(arm=arm,pid=p.pid)))
'''.replace('ARM_PLACEHOLDER',arm)).decode())
