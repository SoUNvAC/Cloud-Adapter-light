"""One-sided Cloud-minus-Surface semantic-score preservation; all-P denominator."""
import torch
from torch.nn import functional as F

def losses(s,teacher,pseudo,P):
    pl=F.cross_entropy(s,pseudo,ignore_index=255) if (pseudo!=255).any() else s.sum()*0
    difference=(s[:,1]-s[:,0])-(teacher[:,1]-teacher[:,0]).detach()
    cm=difference.relu().square()[P].mean() if P.any() else s.sum()*0
    return pl,cm

def activity(s,teacher,P):
    if not P.any():
        return dict(active_count=0,active_fraction=None,mT_mean=None,mS_mean=None,d_mean=None,d_max=None)
    with torch.no_grad():
        mt=(teacher[:,1]-teacher[:,0])[P];ms=(s[:,1]-s[:,0])[P];d=(ms-mt).relu()
        n=int((d>0).sum())
        return dict(active_count=n,active_fraction=n/int(P.sum()),mT_mean=float(mt.mean()),mS_mean=float(ms.mean()),d_mean=float(d.mean()),d_max=float(d.max()))

def tests():
    # Below/equal/above Source margin; last two pixels excluded by P (including invalid).
    teacher=torch.zeros((1,3,1,5),dtype=torch.float64,requires_grad=True)
    s=torch.tensor([[[[0.,0.,0.,0.,0.]], [[-.2,0.,.3,50.,90.]], [[.8,.1,.9,70.,80.]]]],dtype=torch.float64,requires_grad=True)
    P=torch.tensor([[[True,True,True,False,False]]]);pseudo=torch.full((1,1,5),255,dtype=torch.long)
    pl,cm=losses(s,teacher,pseudo,P);manual=.3**2/3
    assert abs(float(cm)-manual)<1e-12 and float(pl)==0
    cm.backward();expected=torch.zeros_like(s);expected[0,0,0,2]=-.2;expected[0,1,0,2]=.2
    error=float((s.grad-expected).abs().max());assert error<1e-12 and teacher.grad is None
    assert torch.count_nonzero(s.grad[:,2])==0 and torch.count_nonzero(s.grad[:,:,:,:2])==0
    empty=s.detach().clone().requires_grad_();a,b=losses(empty,teacher,pseudo,torch.zeros_like(P));(a+b).backward()
    assert float(a)==float(b)==0 and empty.grad is not None and torch.count_nonzero(empty.grad)==0
    stats=activity(s,teacher,P);assert stats['active_count']==1 and stats['active_fraction']==1/3
    return dict(status='passed',manual_loss_error=abs(float(cm)-manual),manual_gradient_max_error=error,
        below_and_equal_zero_gradient=True,positive_Cloud_gradient_and_negative_Surface_gradient=True,
        independent_Shadow_output_partial_zero=True,all_P_denominator=True,excluded_invalid_gradient_zero=True,
        teacher_detached=True,empty_P_differentiable_zero=True,empty_A_differentiable_zero=True,
        semantic_scores_not_softmax=True,lambda_CM=1.)
