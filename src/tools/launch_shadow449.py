from collect_phase241 import ssh
print(ssh('''from pathlib import Path
import subprocess,os,json,sys
r=Path('/home/scv/shared/phase449_20261106');sys.path[:0]=[str(r),'/home/scv/shared/phase445_20261106']
import torch
from shadow_branch449 import LocalOcclusionBranch,boundary_features
from shadow_branch445 import LocalOcclusionBranch as Old
x=torch.full((1,3,16,16),.4);prob=torch.full_like(x,1/3);f=boundary_features(x,prob)
assert f.shape==(1,55,16,16) and torch.isfinite(f).all() and f[:,:54].abs().max()<1e-5
rgb=torch.ones_like(x)*.7;rgb[:,:,:,8:]=.3;prob=torch.zeros_like(x);prob[:,0,:,:8]=.9;prob[:,2,:,:8]=.1;prob[:,0,:,8:]=.1;prob[:,2,:,8:]=.9
f=boundary_features(rgb,prob);assert f[:,0:3,:,7:9].mean()<0
# Shared parameter initialization and RNG state must match the baseline.
torch.manual_seed(123);old=Old();rng=torch.get_rng_state();torch.manual_seed(123);b=LocalOcclusionBranch();assert torch.equal(rng,torch.get_rng_state())
assert all(torch.equal(v,b.state_dict()[k]) for k,v in old.state_dict().items())
assert sum(p.numel() for p in b.parameters())==38075
(r/'UNIT_CHECK.json').write_text(json.dumps(dict(passed=True,uniform_difference_zero=True,known_shadow_transition_sign=True,common_initial_parameters_equal=True,RNG_equal=True,parameters=38075)))
assert not (r/'network_run').exists()
code="import subprocess; from pathlib import Path; r=Path('/home/scv/shared/phase449_20261106'); exe='/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python'; subprocess.run([exe,'-u',str(r/'train_shadow449.py')],check=True); subprocess.run([exe,'-u',str(r/'verify_shadow449.py')],check=True)"
with (r/'pipeline.log').open('w') as log:
 p=subprocess.Popen(['/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python','-u','-c',code],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=r,env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4'))
(r/'LAUNCH.json').write_text(json.dumps(dict(pid=p.pid,steps=6000)));print(json.dumps(dict(pid=p.pid,unit_check=True)))
''').decode())
