"""Pure image-defined disjoint A/P masks; never reference labels or scene metadata."""
import numpy as np
def partitions(s,valid):
    assert s.dtype==np.float32 and s.shape==(3,1022,1022) and np.isfinite(s).all() and (s>=0).all()
    total=s.sum(0);q=np.divide(s,total[None],out=np.zeros_like(s),where=total[None]>1e-12)
    pred=s.argmax(0).astype(np.uint8);maximum=q.max(0)
    A=valid&(total>1e-12)&(maximum>=.90)
    P=valid&(total>1e-12)&(pred==0)&(maximum<.90)
    assert not (A&P).any() and not ((A|P)&~valid).any()
    pseudo=np.where(A,pred,255).astype(np.uint8)
    return A,P,pseudo
