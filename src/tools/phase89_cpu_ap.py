"""Independent non-interpolated AP by score bins; no sklearn dependency."""
import numpy as np

def average_precision_score(y_true,y_score):
    y=np.asarray(y_true,dtype=bool).ravel();s=np.asarray(y_score).ravel()
    assert y.shape==s.shape and len(y) and np.isfinite(s).all() and y.any()
    # Ascending bins, then reverse cumulative counts at each distinct threshold.
    # Unlike production curve(), no descending observation sort or tie-end indices.
    values,bins=np.unique(s,return_inverse=True)
    positives=np.bincount(bins[y],minlength=len(values)).astype(np.float64)
    totals=np.bincount(bins,minlength=len(values)).astype(np.float64)
    tp=np.cumsum(positives[::-1])[::-1];seen=np.cumsum(totals[::-1])[::-1]
    return float(np.dot(positives/y.sum(),tp/seen))

def self_test():
    assert abs(average_precision_score([1,0,1,0],[.9,.9,.8,.2])-7/12)<1e-14
    assert abs(average_precision_score([1,0,0],[.5,.5,.5])-1/3)<1e-14
    assert average_precision_score([1,1,0],[.9,.8,.1])==1
    rng=np.random.default_rng(89011)
    for size in [7,31,120]:
        for _ in range(20):
            s=rng.integers(0,9,size=size);y=rng.random(size)<.4;y[0]=True
            # Slow threshold oracle: each positive receives precision at its score.
            expected=sum(float(y[s>=v].sum())/int((s>=v).sum()) for v in s[y])/int(y.sum())
            assert abs(average_precision_score(y,s)-expected)<1e-14
    return dict(hand_calculated_tie_cases=3,brute_force_threshold_cases=60,sklearn_not_installed_not_claimed=True)

if __name__=='__main__':print(self_test())
