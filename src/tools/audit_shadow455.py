from pathlib import Path
import json,hashlib,sys,torch
sys.path.insert(0,'/home/scv/shared/phase450_20261106')
from shadow_branch450 import LocalOcclusionBranch as Old
from shadow_branch455 import LocalOcclusionBranch as New
torch.set_num_threads(4);torch.manual_seed(45501)
root=Path('/home/scv/shared/phase455_20261106')
state=torch.load('/home/scv/shared/phase450_20261106/network_run/branch_6000.pth',map_location='cpu')['state_dict']
old=Old().cuda().eval();old.load_state_dict(state,strict=True)
new=New().cuda().eval();loaded=new.load_state_dict(state,strict=False)
assert loaded.missing_keys and all(n.startswith('continuity.') for n in loaded.missing_keys) and not loaded.unexpected_keys
rgb=torch.rand(1,3,512,512,device='cuda');detail=torch.randn(1,64,128,128,device='cuda')
features=[torch.randn(1,384,s,s,device='cuda') for s in [128,64,32,16]];base=torch.randn(1,3,512,512,device='cuda').log_softmax(1)
with torch.no_grad():
 a=old(detail,features,rgb,base)[0];b=new(detail,features,rgb,base)[0]
assert torch.equal(a,b)
optimizer=torch.optim.AdamW(new.continuity.parameters(),lr=2e-4)
before={n:p.detach().clone() for n,p in new.continuity.named_parameters()}
for _ in range(3):
 optimizer.zero_grad();output,_,_=new(detail,features,rgb,base)
 loss=output[:,2].square().mean();assert torch.isfinite(loss);loss.backward()
 assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in new.continuity.parameters())
 optimizer.step()
changed={n:not torch.equal(p,before[n]) for n,p in new.continuity.named_parameters()};assert all(changed.values()),changed
result=dict(passed=True,zero_output_exact=True,new_parameters=sum(p.numel() for p in new.continuity.parameters()),branch_parameters=sum(p.numel() for p in new.parameters()),new_parameter_updates=changed,code_sha={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*.py')})
(root/'ARCHITECTURE_AUDIT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
