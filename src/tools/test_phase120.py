"""Synthetic-only acceptance, no Omni model construction or forward."""
import torch
from phase120_common import objectives

def main():
    torch.set_num_threads(2)
    s=torch.tensor([[[[1.,2.]],[[.2,-.4]],[[.6,.9]],[[-1.,.3]]]],requires_grad=True)
    t=torch.tensor([[[[.3,-2.]],[[1.3,.4]],[[.7,1.9]],[[.2,-.8]]]],requires_grad=True)
    y=torch.tensor([[[2,255]]]);ce,kl=objectives(s,y,t)
    p=(t.detach()/2).softmax(1);q=(s/2).softmax(1)
    manual=4*(p*(p.log()-q.log())).sum(1)[y!=255].mean()
    reverse=4*(q*(q.log()-p.log())).sum(1)[y!=255].mean()
    assert abs(float(kl-manual))<1e-6 and abs(float(kl-reverse))>1e-5
    parent=torch.stack((s[:,0],torch.logsumexp(s[:,1:3],1),s[:,3]),1)
    assert torch.allclose(ce,-parent.log_softmax(1)[0,2,0,0])
    same=objectives(s,y,s.detach())[1];assert abs(float(same))<1e-6
    (ce+kl).backward();assert t.grad is None and torch.isfinite(s.grad).all()
    assert torch.count_nonzero(s.grad[:,:,:,1])==0 and torch.count_nonzero(s.grad[:,:,:,0])>0
    print({'KL_direction':'teacher||student','T_squared':4,'ignored_pixel_gradient':0,'teacher_gradient':None,
        'CE_parent_mapping':True,'same_logits_KL':float(same),'Omni_model_calls':0})
if __name__=='__main__':main()
