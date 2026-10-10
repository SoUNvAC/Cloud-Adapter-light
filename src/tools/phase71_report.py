"""Fixed descriptive related-group intervals, conditional on fitted models."""
import numpy as np


def interval(values):
    a=np.asarray(values,float)
    if not len(a) or not np.isfinite(a).all():raise ValueError('Missing/nonfinite group support')
    rng=np.random.default_rng(71010);means=a[rng.integers(0,len(a),(10000,len(a)))].mean(1)
    return dict(mean=float(a.mean()),n_groups=len(a),ci95=np.quantile(means,[.025,.975]).tolist(),
        upper95=float(np.quantile(means,.95)),replicates=10000,seed=71010,independent_confirmation=False,conditional_on_fixed_model=True)
