"""Label-free selection; original-image row-major ties survive crop and training flip."""
from functools import lru_cache
import numpy as np
from phase318_common import *
def crop(a,pool,step):
 if pool=='CMC':a=a[step['top']:step['top']+512,step['left']:step['left']+512]
 return a[:,::-1].copy() if step['horizontal_flip'] else a.copy()
@lru_cache(maxsize=4)
def product(pool,index):
 row=inputs(pool)['fit'][index];n=name(row)
 path=OLD/'teacher'/(n+'.npz') if pool=='CMC' else PRIOR/pool/'teacher'/(n+'.npz')
 with np.load(path,allow_pickle=False) as z:
  p=z['pseudo'].copy();valid=z['valid'].copy()
  if pool=='CMC':q=np.where(p==2,z['maximum'],0).astype(np.float32)
  else:
   s=z['scores'];total=s.sum(0);q=np.divide(s[2],total,out=np.zeros_like(total),where=total>1e-12)
 with np.load(ROOT/pool/'flip_teacher'/(n+'.npz'),allow_pickle=False) as z:
  assert np.array_equal(z['valid'],valid);g=(p==2)&z['valid']&(z['pred']==2)&(z['q_shadow']>=np.float32(.9))
 ids=np.arange(p.size,dtype=np.int64).reshape(p.shape)
 return p,q,g,ids
def selection(pool,step):
 p,q,g,ids=product(pool,step['fit_index']);p,q,g,ids=[crop(a,pool,step) for a in [p,q,g,ids]]
 k=int(g.sum());sh=p==2;chosen=np.zeros(p.shape,bool);ix=np.flatnonzero(sh)
 if k:
  order=np.lexsort((ids.ravel()[ix],-q.ravel()[ix]));chosen.ravel()[ix[order[:k]]]=True
 assert int(chosen.sum())==k and not (g&~sh).any() and not (chosen&~sh).any()
 return p,g,chosen
def masses(p,selected):
 cc=np.array([(p==c).sum() for c in range(3)],np.int64);N=int(cc.sum());K=int((cc>0).sum());b=np.divide(N,K*cc,out=np.zeros(3,np.float64),where=cc>0) if N else np.zeros(3,np.float64)
 k=int(selected.sum());mass=cc*b;mass[2]=k*b[2]+int(cc[2])-k
 return dict(counts=cc.tolist(),N=N,K=K,b=b.tolist(),k=k,class_mass=mass.tolist(),denominator=float(mass.sum()))
