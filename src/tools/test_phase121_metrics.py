"""Independent synthetic exact-boundary, unequal-group and paired-seed checks."""
import numpy as np
from phase121_metrics import oracle,evaluate,POLICIES
def main():
    rng=np.random.default_rng(121015)
    for case in range(100):
        groups={}
        for i in range(3):
            n=int(rng.integers(2,60));y=rng.integers(0,2,n).astype(bool)
            if i==0:y[0]=False
            s=(rng.integers(1,20,n)/20).astype(np.float32)
            groups[str(i)]={'truth':y,'X':s}
        actual,detail=oracle(groups,'X','synthetic')
        neg=np.concatenate([g['X'][~g['truth']] for g in groups.values()]).astype(np.float64)
        candidates=np.sort(np.unique(np.r_[neg,np.nextafter(neg,np.inf)]))
        eligible=[]
        for t in candidates:
            rr=[np.count_nonzero((g['X'].astype(np.float64)>=t)&~g['truth'])/(~g['truth']).sum()
                for g in groups.values() if (~g['truth']).any()]
            if sum(rr)/len(rr)<=.01:eligible.append(t)
        assert actual['threshold']==min(eligible)
        assert actual['macro_negative_FP_pp']<=1
    # 32 equal-weight groups allow 32 negatives in one group, unlike a tail constraint.
    gs={str(i):dict(truth=np.zeros(100,bool),X=np.linspace(.2,.9,100,dtype=np.float32) if i==0 else np.full(100,.01,np.float32)) for i in range(32)}
    o,d=oracle(gs,'X','synthetic');assert d[0]['FP']==32 and o['max_group_negative_FP_pp']==32
    # Explicit no-positive group retains an equal vote in the constraint.
    g={'a':dict(truth=np.array([1,0],bool),X=np.array([.6,.5],np.float32)),
       'b':dict(truth=np.zeros(100,bool),X=np.zeros(100,np.float32))}
    o,d=oracle(g,'X','synthetic');assert o['negative_groups']==2 and d[0]['TP']==1
    # No-negative support cannot invent a threshold.
    assert oracle({'a':dict(truth=np.ones(2,bool),X=np.ones(2,np.float32))},'X','test')[0]['threshold'] is None
    gs={}
    for i in range(3):
        y=np.array([1,0,1,0],bool);gs[str(i)]={'truth':y,'products':str(i)}
        for p in POLICIES:gs[str(i)][p]=rng.random(4).astype(np.float32)
    per,summaries,cmp=evaluate(gs,'test');assert len(summaries)==7 and len(cmp)==9
    for name in ('C-R','D-C','D-R'):
        a,b=name.split('-');expected=[]
        for row in per:
            expected.append(sum(row[f'{a}{seed}_minus_{b if b=="R" else b+str(seed)}_AP_pp'] for seed in (12001,12002))/2)
        assert abs(next(c for c in cmp if c['comparison']=='two_seed_'+name)['mean_delta_AP_pp']-np.mean(expected))<1e-10
    print(dict(status='passed',exhaustive_oracle_cases=100,exact_ties=True,float64_boundary=True,
               all_negative_groups=True,macro_not_pooled=True,within_group_seed_average=True,model_calls=0))
if __name__=='__main__':main()
