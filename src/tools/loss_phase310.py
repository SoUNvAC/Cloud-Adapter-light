"""T=1 forward KL on P; separate accepted CE mean; no q-based distillation."""
import math
import torch
from torch.nn import functional as F
def losses(s,teacher,pseudo,P):
    A=pseudo!=255
    pl=F.cross_entropy(s,pseudo,ignore_index=255) if A.any() else s.sum()*0
    lt=F.log_softmax(teacher.detach(),dim=1);ls=F.log_softmax(s,dim=1)
    sp=(lt.exp()*(lt-ls)).sum(1)[P].mean() if P.any() else s.sum()*0
    return pl,sp
def tests():
    p=torch.tensor([.6,.3,.1],dtype=torch.float64);q=torch.tensor([.2,.5,.3],dtype=torch.float64)
    s=q.log().reshape(1,3,1,1).requires_grad_();t=p.log().reshape(1,3,1,1)
    ignore=torch.full((1,1,1),255,dtype=torch.long);mask=torch.ones((1,1,1),dtype=torch.bool)
    pl,sp=losses(s,t,ignore,mask);manual=sum(float(a)*math.log(float(a/b)) for a,b in zip(p,q))
    assert abs(float(sp)-manual)<1e-12 and abs(float(sp)-sum(float(b)*math.log(float(b/a)) for a,b in zip(p,q)))>1e-3
    assert float(pl)==0 and torch.isfinite(sp)
    empty=losses(s,t,ignore,~mask);assert all(float(x)==0 for x in empty);sum(empty).backward();assert s.grad is not None
    return dict(forward_KL_manual_error=abs(float(sp)-manual),empty_P_differentiable_zero=True,empty_A_differentiable_zero=True,no_reversed_KL=True,temperature=1.,lambda_SP=1.)
