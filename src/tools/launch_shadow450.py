from collect_phase241 import ssh
print(ssh('''from pathlib import Path
import subprocess,os,json,sys
r=Path('/home/scv/shared/phase450_20261106');sys.path[:0]=[str(r),'/home/scv/shared/phase449_20261106']
import torch
from shadow_branch450 import LocalOcclusionBranch,boundary_features,box
from shadow_branch449 import LocalOcclusionBranch as Old
rgb=torch.ones((1,3,24,24))*.7;rgb[:,:,:,12:]=.3;prob=torch.full_like(rgb,1/3)
luminance=(rgb*torch.tensor([.2126,.7152,.0722])[None,:,None,None]).sum(1,keepdim=True)
f=boundary_features(rgb,prob,guide=-box((luminance+.02).log(),5));assert torch.isfinite(f).all() and f[:,0:3,:,11:13].mean()<0
oldf=boundary_features(rgb,prob);assert oldf[:,:54].abs().max()<1e-4
torch.manual_seed(123);old=Old();rng=torch.get_rng_state();torch.manual_seed(123);b=LocalOcclusionBranch();assert torch.equal(rng,torch.get_rng_state());assert all(torch.equal(v,b.state_dict()[k]) for k,v in old.state_dict().items());assert sum(p.numel() for p in b.parameters())==39148
(r/'UNIT_CHECK.json').write_text(json.dumps(dict(passed=True,image_edge_without_teacher_edge=True,finite=True,common_initial_parameters_equal=True,RNG_equal=True,parameters=39148)))
assert not (r/'network_run').exists()
code="import subprocess; from pathlib import Path; r=Path('/home/scv/shared/phase450_20261106'); exe='/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python'; subprocess.run([exe,'-u',str(r/'train_shadow450.py')],check=True); subprocess.run([exe,'-u',str(r/'verify_shadow450.py')],check=True)"
with (r/'pipeline.log').open('w') as log:
 p=subprocess.Popen(['/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python','-u','-c',code],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=r,env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4'))
(r/'LAUNCH.json').write_text(json.dumps(dict(pid=p.pid,steps=6000)));print(json.dumps(dict(pid=p.pid,unit_check=True)))
''').decode())
