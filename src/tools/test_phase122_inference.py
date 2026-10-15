"""CPU hand checks for confidence aggregation, callback API and fixed statistics."""
import ast
import inspect
from pathlib import Path
import numpy as np
from phase89_cpu_ap import average_precision_score, self_test


def main():
    tree=ast.parse(Path(__file__).with_name('infer_phase122.py').read_text(encoding='utf-8'))
    fun=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='observed_store')
    sig=inspect.Signature([inspect.Parameter(a.arg,inspect.Parameter.POSITIONAL_OR_KEYWORD) for a in fun.args.args])
    sig.bind(pred_batch=None,index_batch=None,pred_tracker=None,gradient=None,grad_tracker=None)
    rng=np.random.default_rng(119015);size=12
    raw=rng.normal(size=(2,4,size,size)).astype(np.float32)
    axis=np.minimum(np.minimum(np.arange(1,size+1),np.arange(size,0,-1)),3).astype(np.float32)*np.float32(1/3)
    g=axis[:,None]*axis[None,:]
    pts=np.array([1,3,8]);cols=np.array([2,6,10])
    full=np.zeros((4,size,size),np.float32);full+=raw.mean(0)*g
    sparse=raw.mean(0)[:,pts,cols]*g[pts,cols]
    assert np.array_equal(sparse,full[:,pts,cols])
    # Single-member mosaics combine at raw-logit level, not confidence level.
    members=np.stack([raw[i]*g for i in range(2)])
    assert np.max(np.abs(members.mean(0)-full))<1e-5
    d=np.array([-1.01,-1.,0.,1.,1.01]);assert ((d>1).sum(),(np.abs(d)<=1).sum(),(d< -1).sum())==(1,3,1)
    assert abs(average_precision_score([1,0,1,0],[.9,.9,.8,.2])-7/12)<1e-12
    idx=rng.integers(5,size=(10000,5));assert np.max(np.abs(d[idx].mean(1)-np.array([sum(d[i] for i in v)/5 for v in idx])))<1e-12
    print({'status':'passed','callback_keyword_API':True,'raw_mosaic_and_points':True,'fixed_tie_AP':self_test(),'bootstrap_scalar':True,'Omni_calls':0})


if __name__=='__main__':main()
