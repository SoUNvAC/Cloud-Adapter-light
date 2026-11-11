from collect_phase241 import ssh
print(ssh('''from pathlib import Path
import subprocess,os,json,sys
r=Path('/home/scv/shared/phase451_20261106');sys.path[:0]=[str(r),'/home/scv/shared/phase450_20261106']
import torch
from shadow_branch451 import LocalOcclusionBranch,remote_reference
from shadow_branch450 import LocalOcclusionBranch as Old
z=torch.ones(1,24,32,32);rgb=torch.full((1,3,32,32),.4);prob=torch.full((1,3,32,32),1/3)
a,b,m=remote_reference(z,rgb,prob);assert torch.allclose(a,z,atol=1e-5) and torch.allclose(b,rgb,atol=1e-5) and torch.isfinite(m).all()
rgb[:,:,12:20,12:20]=.1;_,ref,_=remote_reference(z,rgb,prob);assert ref[0,:,16,16].mean()>.35
z=z.requires_grad_();a,b,m=remote_reference(z,rgb,prob);(a.square().mean()+b.square().mean()).backward();assert z.grad is not None and torch.isfinite(z.grad).all()
torch.manual_seed(123);old=Old();rng=torch.get_rng_state();torch.manual_seed(123);new=LocalOcclusionBranch();assert torch.equal(rng,torch.get_rng_state());assert all(torch.equal(v,new.state_dict()[k]) for k,v in old.state_dict().items());assert sum(p.numel() for p in new.parameters())==40653
(r/'UNIT_CHECK.json').write_text(json.dumps(dict(passed=True,uniform_preserved=True,remote_reference_reaches_outside_dark_region=True,finite_gradients=True,common_initial_parameters_equal=True,RNG_equal=True,parameters=40653)))
assert not (r/'network_run').exists()
code="import subprocess; from pathlib import Path; r=Path('/home/scv/shared/phase451_20261106'); exe='/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python'; subprocess.run([exe,'-u',str(r/'train_shadow451.py')],check=True); subprocess.run([exe,'-u',str(r/'verify_shadow451.py')],check=True)"
with (r/'pipeline.log').open('w') as log:
 p=subprocess.Popen(['/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python','-u','-c',code],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=r,env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4'))
(r/'LAUNCH.json').write_text(json.dumps(dict(pid=p.pid,steps=6000)));print(json.dumps(dict(pid=p.pid,unit_check=True)))
''').decode())
