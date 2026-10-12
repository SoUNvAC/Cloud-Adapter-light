"""Exact negative-boundary oracle with mean and worst-group constraints."""
import numpy as np
from phase91_math import scan

def rates(scores,truths,t):
    den=[int(np.count_nonzero(y!=2)) for y in truths]
    fp=[int(np.count_nonzero(s[y!=2]>=t)) for s,y in zip(scores,truths)]
    rr=[f/n for f,n in zip(fp,den) if n]
    return dict(denominators=den,false_positives=fp,macro_fpr=float(np.mean(rr)) if rr else None,max_group_fpr=max(rr) if rr else None)

def oracles(scores,truths):
    assert all(s.dtype==np.float64 and np.isfinite(s).all() for s in scores)
    # Prior exact integer-refined mean scan identifies the first unsafe negative
    # tie. Use its upward float64 boundary rather than an observed full score.
    mean=scan(scores,truths,[.01])[0]
    if mean['threshold'] is None:
        return [dict(kind=k,threshold=None,status='no_negative_support') for k in ['O_mean','O_tail']]
    boundary=mean['next_lower_observed_threshold'];assert boundary is not None
    tail_boundaries=[]
    for s,y in zip(scores,truths):
        neg=s[y!=2];n=len(neg)
        if not n:continue
        allowed=int(np.floor(.1*n))
        while allowed/n>.1:allowed-=1
        while (allowed+1)/n<=.1:allowed+=1
        # Descending index allowed is the first negative that would exceed 10%.
        tail_boundaries.append(float(np.partition(neg,n-allowed-1)[n-allowed-1]))
    out=[]
    for kind,b in [('O_mean',boundary),('O_tail',max(boundary,max(tail_boundaries)))]:
        t=float(np.nextafter(np.float64(b),np.inf));at=rates(scores,truths,t);lower=rates(scores,truths,b)
        assert at['macro_fpr']<=.01 and (kind=='O_mean' or at['max_group_fpr']<=.1)
        violations=[]
        if lower['macro_fpr']>.01:violations.append('mean_FPR>1%')
        if kind=='O_tail' and lower['max_group_fpr']>.1:violations.append('max_group_FPR>10%')
        assert violations
        assert np.nextafter(t,-np.inf)==b
        out.append(dict(kind=kind,status='oracle_non_deployable',threshold=t,unsafe_negative_boundary=b,
            lower_adjacent_threshold=b,lower_adjacent=lower,violations=violations,
            boundary_tie_counts=[int(np.count_nonzero(s[y!=2]==b)) for s,y in zip(scores,truths)],
            boundary_rule='nextafter(first unsafe negative tie,+inf); exact score>=threshold',**at))
    assert out[1]['threshold']>=out[0]['threshold']
    return out

def self_test():
    rng=np.random.default_rng(99012)
    for i in range(100):
        sizes=rng.integers(1,35,size=3)
        ss=[rng.integers(0,12,size=n).astype(np.float64)/12 for n in sizes]
        yy=[rng.integers(0,3,size=n) for n in sizes];yy[0][0]=0
        # An almost tied free positive must not be lost by float32 quantization.
        if i==0:ss[0][0]=.5;ss[0][-1]=.5+1e-10;yy[0][-1]=2
        candidates=sorted(set(v for s,y in zip(ss,yy) for x in s[y!=2] for v in [float(x),float(np.nextafter(x,np.inf))]))
        for p in oracles(ss,yy):
            feasible=[]
            for t in candidates:
                r=rates(ss,yy,t)
                if r['macro_fpr']<=.01 and (p['kind']=='O_mean' or r['max_group_fpr']<=.1):feasible.append(t)
            assert p['threshold']==min(feasible)
    s=np.array([.5,.5+1e-10],np.float64);y=np.array([0,2])
    assert (s>=oracles([s],[y])[0]['threshold']).tolist()==[False,True]
    assert all(p['threshold'] is None for p in oracles([s],[np.array([2,2])]))
    ss=[np.linspace(.2,.9,100,dtype=np.float64)]+[np.full(100,.01,np.float64) for _ in range(31)]
    yy=[np.zeros(100,int) for _ in ss]
    points=oracles(ss,yy)
    assert points[0]['false_positives'][0]==32 and points[1]['false_positives'][0]==10
    assert points[0]['max_group_fpr']==.32 and points[1]['max_group_fpr']==.1
    assert points[1]['threshold']>points[0]['threshold']
    return dict(exhaustive_negative_and_nextafter_cases=100,free_positive_gap=True,no_negative_support=True,atomic_ties=True,reject_all_supported=True,tail_binding_32_group_case=True)

if __name__=='__main__':print(self_test())
