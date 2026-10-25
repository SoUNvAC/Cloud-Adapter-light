"""Unflipped, tie-aware ROC AUC verified by an independent average-rank implementation."""
import numpy as np
from scipy.stats import rankdata
CHECKS=0

def auc(y,s):
    global CHECKS
    assert y.dtype==bool and s.ndim==1 and len(y)==len(s) and np.isfinite(s).all()
    positive=int(y.sum());negative=len(y)-positive
    if not positive or not negative:return None
    order=np.argsort(s,kind='stable');v=s[order];yy=y[order]
    starts=np.r_[0,np.flatnonzero(v[1:]!=v[:-1])+1]
    sizes=np.diff(np.r_[starts,len(v)]);pos=np.add.reduceat(yy.astype(np.int64),starts);neg=sizes-pos
    less_negative=np.cumsum(neg,dtype=np.int64)-neg
    value=float(np.dot(pos.astype(np.float64),less_negative+.5*neg)/(positive*negative))
    ranks=rankdata(s,method='average')
    independent=float((ranks[y].sum()-positive*(positive+1)/2)/(positive*negative))
    assert 0<=value<=1 and abs(value-independent)<1e-12
    CHECKS+=1
    return value

def numerical_tests():
    assert auc(np.array([True,True]),np.array([1.,2.])) is None
    assert auc(np.array([False,False]),np.array([1.,2.])) is None
    assert auc(np.array([],bool),np.array([],np.float32)) is None
    assert auc(np.array([False,True]),np.array([2.,2.]))==.5
    assert auc(np.array([False,True]),np.array([0.,1.]))==1
    assert auc(np.array([False,True]),np.array([1.,0.]))==0
    assert auc(np.array([False,True,False,True]),np.array([0.,0.,1.,1.]))==.5
    return dict(single_class_NA=True,empty_NA=True,all_ties_half=True,positive_and_reverse_fixed=True,tie_groups_rank_sum_equal=True)
