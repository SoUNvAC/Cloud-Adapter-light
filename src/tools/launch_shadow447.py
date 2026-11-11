from collect_phase241 import ssh
print(ssh('''from pathlib import Path
import subprocess,os,json
r=Path('/home/scv/shared/phase447_20261106'); jobs=[]
for arm in ['a','b']:
 d=r/arm.upper();suffix='447'+arm
 assert not (d/'network_run').exists()
 code="import subprocess; from pathlib import Path; r=Path("+repr(str(d))+"); exe='/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python'; subprocess.run([exe,'-u',str(r/'train_shadow"+suffix+".py')],check=True); subprocess.run([exe,'-u',str(r/'verify_shadow"+suffix+".py')],check=True)"
 with (d/'pipeline.log').open('w') as log:
  p=subprocess.Popen(['/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python','-u','-c',code],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=d,env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4'))
 jobs.append(dict(arm=arm,pid=p.pid,steps=6000))
(r/'LAUNCH.json').write_text(json.dumps(jobs));print(json.dumps(jobs))
''').decode())
