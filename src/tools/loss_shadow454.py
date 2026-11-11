"""Within-image shadow/nonshadow confusion ranking on real training labels."""
import torch
from torch.nn import functional as F

def confusion_ranking(logits,y,k=1024,margin=1.):
    terms=[];counts=[]
    for scores,truth in zip(logits,y):
        for c in (0,1):
            # Pairwise score difference preserves the actual shadow-vs-class boundary.
            difference=scores[2]-scores[c]
            positive=difference[truth==2];negative=difference[truth==c]
            if not positive.numel() or not negative.numel():continue
            size=min(k,positive.numel(),negative.numel())
            hardest_positive=positive.topk(size,largest=False).values
            hardest_negative=negative.topk(size,largest=True).values
            terms.append(F.softplus(margin+hardest_negative-hardest_positive).mean())
            counts.append(dict(negative_class=c,pairs=size))
    loss=torch.stack(terms).mean() if terms else logits.sum()*0
    return loss,dict(loss=float(loss.detach()),groups=counts,margin=margin,maximum_pairs=k)

if __name__=='__main__':
    x=torch.zeros(1,3,1,3,requires_grad=True);y=torch.tensor([[[0,2,255]]])
    loss,_=confusion_ranking(x,y);loss.backward()
    assert x.grad[0,2,0,0]>0 and x.grad[0,2,0,1]<0
    assert (x.grad[:,:,:,2]==0).all()
    empty,_=confusion_ranking(x,torch.full_like(y,255));assert empty==0
    # Swapping the shadow/nonshadow ordering must increase the penalty.
    good=torch.tensor([[[[2.,0.]],[[0.,0.]],[[0.,2.]]]])
    bad=good.flip(-1);labels=torch.tensor([[[0,2]]])
    assert confusion_ranking(good,labels)[0]<confusion_ranking(bad,labels)[0]
    print('RANKING_GRADIENT_AND_MASK_CHECK_PASSED')
