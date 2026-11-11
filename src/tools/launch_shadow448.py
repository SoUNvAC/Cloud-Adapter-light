from collect_phase241 import ssh
print(ssh('''from pathlib import Path
import subprocess,os,json,sys
r=Path('/home/scv/shared/phase448_20261106');sys.path.insert(0,str(r))
import torch
from shadow_branch448 import shift,LocalOcclusionBranch
x=torch.arange(25).view(1,1,5,5).float()
assert torch.equal(shift(x,0,1)[...,:,:-1],x[...,1:])
assert shift(x,0,1)[...,:,-1].count_nonzero()==0
assert torch.equal(shift(x,-1,0)[...,1:,:],x[...,:-1,:])
b=LocalOcclusionBranch();assert sum(p.numel() for p in b.parameters())==37002
(r/'UNIT_CHECK.json').write_text(json.dumps(dict(passed=True,shift_direction=True,outside_zero=True,parameters=37002)))
assert not (r/'network_run').exists()
code="import subprocess; from pathlib import Path; r=Path('/home/scv/shared/phase448_20261106'); exe='/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python'; subprocess.run([exe,'-u',str(r/'train_shadow448.py')],check=True); subprocess.run([exe,'-u',str(r/'verify_shadow448.py')],check=True)"
with (r/'pipeline.log').open('w') as log:
 p=subprocess.Popen(['/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python','-u','-c',code],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=r,env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4'))
(r/'LAUNCH.json').write_text(json.dumps(dict(pid=p.pid,steps=6000)));print(json.dumps(dict(pid=p.pid,unit_check=True)))
''').decode())
