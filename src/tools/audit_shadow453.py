from pathlib import Path
import json,hashlib,torch
from augment_shadow453 import paired_shadow
root=Path('/home/scv/shared/phase453_20261106')
torch.set_num_threads(2)
rgb=torch.full((4,3,512,512),128.,device='cuda')
y=torch.zeros((4,512,512),dtype=torch.long,device='cuda')
y[0,:100]=1;y[1,:100]=2;y[2,:100]=255;y[3]=1
a,b,s=paired_shadow(rgb,y,1);a2,b2,_=paired_shadow(rgb,y,1)
assert torch.equal(a,a2) and torch.equal(b,b2)
assert torch.equal(a.permute(0,2,3,1)[y!=0],rgb.permute(0,2,3,1)[y!=0])
assert torch.equal(b[y!=0],y[y!=0])
assert ((b!=y)&(b==2)).any()
assert torch.equal(a[3],rgb[3]) and torch.equal(b[3],y[3])
assert a.min()>0 and a.max()<255
result=dict(passed=True,checks=['deterministic','only_surface_modified','existing_cloud_shadow_ignore_preserved','new_shadow_core_present','no_surface_no_change','no_clipping'],synthetic_example=s,code_sha={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*.py')})
(root/'A/network_run/AUGMENTATION_AUDIT.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
