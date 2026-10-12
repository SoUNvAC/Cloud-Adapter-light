"""Exact-ties oracle thresholds for an equal-group FPR constraint, not deployment."""
import numpy as np


def macro_counts(counts, denominators):
    valid=np.asarray(denominators)>0
    return float(np.mean(np.asarray(counts)[valid]/np.asarray(denominators)[valid])) if valid.any() else None


def scan(scores, truths, budgets):
    """Sort negatives once; refine by exact integer counts, include free positives.

    Lowest effective threshold means lowest observed full score that is feasible;
    when no observed score is feasible use nextafter(max full score,+inf).
    Float64 probability ties are atomic; AP elsewhere retains Phase90 float32.
    """
    assert len(scores)==len(truths) and all(len(s)==len(y) for s,y in zip(scores,truths))
    assert all(np.isfinite(s).all() for s in scores)
    denominators=np.array([np.count_nonzero(y!=2) for y in truths],np.int64)
    supported=denominators>0;m=int(supported.sum())
    if not m:
        return [dict(budget=b,threshold=None,status='no_negative_support',denominators=denominators.tolist()) for b in budgets]
    parts=[np.asarray(s,dtype=np.float64)[np.asarray(y)!=2] for s,y in zip(scores,truths)]
    negatives=np.concatenate(parts)
    gids=np.concatenate([np.full(len(s),i,np.uint16) for i,s in enumerate(parts)])
    order=np.argsort(-negatives,kind='stable');values=negatives[order];groups=gids[order]
    ends=np.flatnonzero(np.r_[values[1:]!=values[:-1],True])
    weights=1./denominators[groups]/m
    approximate=np.cumsum(weights,dtype=np.float64)[ends]
    del weights,order,gids,negatives,parts
    def counts(i):
        return np.bincount(groups[:ends[i]+1],minlength=len(scores)) if i>=0 else np.zeros(len(scores),np.int64)
    results=[]
    for budget in budgets:
        assert 0<=budget<1
        index=int(np.searchsorted(approximate,budget,side='right'))-1
        fp=counts(index);rate=macro_counts(fp,denominators)
        # Floating cumulative weights only locate a candidate; feasibility is
        # determined with exact per-group integer counts, not cumulative weights.
        while index>=0 and rate>budget:
            index-=1;fp=counts(index);rate=macro_counts(fp,denominators)
        while index+1<len(ends):
            following=counts(index+1)
            if macro_counts(following,denominators)>budget:break
            index+=1;fp=following;rate=macro_counts(fp,denominators)
        assert rate<=budget
        if index+1<len(ends):
            boundary=float(values[ends[index+1]])
            options=[float(np.min(s[s>boundary])) for s in scores if np.any(s>boundary)]
            threshold=min(options) if options else float(np.nextafter(max(float(np.max(s)) for s in scores if len(s)),np.inf))
            nextfp=counts(index+1);next_rate=macro_counts(nextfp,denominators)
            assert next_rate>budget
        else:
            boundary=None;threshold=min(float(np.min(s)) for s in scores if len(s));nextfp=None;next_rate=None
        actual=np.array([np.count_nonzero((s>=threshold)&(y!=2)) for s,y in zip(scores,truths)])
        assert np.array_equal(actual,fp)
        results.append(dict(status='oracle_only',budget=float(budget),threshold=threshold,
            macro_fpr=rate,denominators=denominators.tolist(),false_positives=fp.tolist(),
            next_lower_observed_threshold=boundary,next_batch_macro_fpr=next_rate,
            next_batch_false_positives=None if nextfp is None else nextfp.tolist(),
            next_entire_tie_batch_violates=next_rate>budget if next_rate is not None else None,
            score_precision='float64 original60m probability; >=; atomic exact ties'))
    return results


def cpu_validate(point,scores,truths):
    """Direct group counting, no sort/weighted cumulative scan implementation."""
    if point['threshold'] is None:
        assert all(not np.count_nonzero(y!=2) for y in truths);return
    rates=[];next_rates=[]
    for i,(s,y) in enumerate(zip(scores,truths)):
        negative=np.asarray(y)!=2;n=int(negative.sum())
        assert n==point['denominators'][i]
        fp=int(np.sum((s>=point['threshold'])&negative));assert fp==point['false_positives'][i]
        if n:rates.append(fp/n)
        t=point['next_lower_observed_threshold']
        if t is not None:
            following=int(np.sum((s>=t)&negative));assert following==point['next_batch_false_positives'][i]
            if n:next_rates.append(following/n)
    actual=sum(rates)/len(rates)
    assert actual<=point['budget'] and abs(actual-point['macro_fpr'])<1e-15
    if next_rates:
        assert sum(next_rates)/len(next_rates)>point['budget']
        assert abs(sum(next_rates)/len(next_rates)-point['next_batch_macro_fpr'])<1e-15


def self_test():
    # Macro constraints differ from pooled constraints under unequal group sizes.
    cases=[([np.array([.8,.8,.6,.4])],[np.array([0,0,2,2])],[0.,.5]),
           ([np.array([.9,.8,.7,.6]),np.array([.95])],[np.array([0,0,0,2]),np.array([0])],[.25,.5]),
           ([np.array([.9,.8]),np.array([.7,.6])],[np.array([2,2]),np.array([0,0])],[0.,.5]),
           ([np.array([.5,.5])],[np.array([0,2])],[0.,.5]),
           ([np.array([.9,.8,.7])],[np.array([0,2,2])],[0.]),
           ([np.array([.8,.7]),np.array([])],[np.array([0,0]),np.array([],int)],[0.,.5])]
    checked=0
    for scores,truths,budgets in cases:
        for point in scan(scores,truths,budgets):
            cpu_validate(point,scores,truths)
            thresholds=sorted(set(np.concatenate(scores).tolist()))
            feasible=[]
            for t in thresholds:
                rates=[np.count_nonzero((s>=t)&(y!=2))/np.count_nonzero(y!=2) for s,y in zip(scores,truths) if np.count_nonzero(y!=2)]
                if sum(rates)/len(rates)<=point['budget']:feasible.append(t)
            expected=min(feasible) if feasible else float(np.nextafter(max(thresholds),np.inf))
            assert point['threshold']==expected
            checked+=1
    empty=scan([np.array([.1,.9])],[np.array([2,2])],[.01])[0]
    assert empty['threshold'] is None and empty['status']=='no_negative_support'
    cpu_validate(empty,[np.array([.1,.9])],[np.array([2,2])])
    # A free positive below the lowest accepted negative must still be admitted.
    free=scan([np.array([.9,.7,.1])],[np.array([0,2,0])],[.5])[0]
    assert free['threshold']==.7
    rng=np.random.default_rng(91012)
    for _ in range(100):
        sizes=rng.integers(1,17,size=3)
        ss=[rng.integers(0,7,size=k).astype(np.float64)/7 for k in sizes]
        yy=[rng.integers(0,3,size=k) for k in sizes];yy[0][0]=0
        budget=float(rng.choice([0.,.01,.1,.25,.5]))
        point=scan(ss,yy,[budget])[0];cpu_validate(point,ss,yy)
        feasible=[]
        for t in sorted(set(np.concatenate(ss).tolist())):
            rates=[np.count_nonzero((s>=t)&(y!=2))/np.count_nonzero(y!=2) for s,y in zip(ss,yy) if np.count_nonzero(y!=2)]
            if sum(rates)/len(rates)<=budget:feasible.append(t)
        expected=min(feasible) if feasible else float(np.nextafter(max(np.concatenate(ss)),np.inf))
        assert point['threshold']==expected
    return dict(exhaustive_observed_threshold_cases=checked,no_negative_support=True,
                atomic_ties=True,all_nonshadow_feasible=True,unequal_group_denominators=True,
                single_class_and_empty_group=True,free_positive_threshold=True,random_exhaustive_cases=100)
