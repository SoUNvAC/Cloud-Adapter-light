from pathlib import Path
import json,hashlib,sys,torch
sys.path.insert(0,'/home/scv/shared/phase450_20261106')
from shadow_branch450 import LocalOcclusionBranch as Old
from shadow_branch458 import LocalOcclusionBranch as New
torch.set_num_threads(4);torch.manual_seed(45501)
root=Path('/home/scv/shared/phase458_20261106')
state=torch.load('/home/scv/shared/phase450_20261106/network_run/branch_6000.pth',map_location='cpu')['state_dict']
old=Old().cuda().eval();old.load_state_dict(state,strict=True)
new=New().cuda().eval();loaded=new.load_state_dict(state,strict=False)
assert loaded.missing_keys and all(n.startswith('region.') for n in loaded.missing_keys) and not loaded.unexpected_keys
rgb=torch.rand(1,3,512,512,device='cuda');detail=torch.randn(1,64,128,128,device='cuda')
features=[torch.randn(1,384,s,s,device='cuda') for s in [128,64,32,16]];base=torch.randn(1,3,512,512,device='cuda').log_softmax(1)
with torch.no_grad():
 a=old(detail,features,rgb,base)[0];b=new(detail,features,rgb,base)[0]
assert torch.equal(a,b)
optimizer=torch.optim.AdamW(new.region.parameters(),lr=2e-4)
before={n:p.detach().clone() for n,p in new.region.named_parameters()}
signals={n:0. for n,p in new.region.named_parameters()}
y=torch.zeros((1,512,512),dtype=torch.long,device='cuda');y[:,:,256:]=2;y[:,:8]=255
for _ in range(3):
 optimizer.zero_grad();output,_,_=new(detail,features,rgb,base)
 aux,_=new.region.auxiliary_loss(y);loss=output[:,2].square().mean()+aux;assert torch.isfinite(loss);loss.backward()
 for n,p in new.region.named_parameters():signals[n]+=float(p.grad.abs().sum()) if p.grad is not None else 0.
 assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in new.region.parameters())
 optimizer.step()
assert all(v>0 for v in signals.values()),signals
ignored,_=new.region.auxiliary_loss(torch.full_like(y,255));assert ignored==0
changed={n:not torch.equal(p,before[n]) for n,p in new.region.named_parameters()};assert all(changed.values()),changed
result=dict(passed=True,zero_output_exact=True,new_parameters=sum(p.numel() for p in new.region.parameters()),branch_parameters=sum(p.numel() for p in new.parameters()),new_parameter_updates=changed,nonzero_gradient_sums=signals,ignore_mask_passed=True,code_sha={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*.py')})
(root/'ARCHITECTURE_AUDIT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
