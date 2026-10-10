"""Frozen-model inference with related groups, never independent pixels."""
import numpy as np

BOOTSTRAPS = 10000
SEED = 66010


def interval(values):
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError('Expected finite group values')
    result = dict(n_groups=len(x), mean=float(x.mean()) if len(x) else None,
                  ci95=None, upper95_one_sided=None, seed=SEED,
                  resamples=BOOTSTRAPS, method='equal-group percentile bootstrap',
                  conditional_on_frozen_models=True)
    if len(x) < 2:
        return result
    rng = np.random.default_rng(SEED)
    means = x[rng.integers(0, len(x), (BOOTSTRAPS, len(x)))].mean(axis=1)
    result.update(ci95=np.quantile(means, [.025, .975]).tolist(),
                  upper95_one_sided=float(np.quantile(means, .95)))
    return result


def primary(pairs, minimum_support=2):
    # Pair first, then resample related groups; never bootstrap pixels or models separately.
    if len({r['group_id'] for r in pairs}) != len(pairs):
        raise ValueError('Duplicate statistical group')
    delta = np.asarray([r['delta_ap'] * 100 for r in pairs])
    stats = interval(delta)
    stats.update(unit='AP percentage points', improved=int((delta > 1).sum()),
                 stable=int((np.abs(delta) <= 1).sum()), declined=int((delta < -1).sum()),
                 worst_delta_ap_pp=float(delta.min()) if len(delta) else None,
                 minimum_support=minimum_support)
    ci = stats['ci95']
    stats['decision'] = ('insufficient_support' if len(delta) < minimum_support or ci is None
                         else 'positive_increment_independently_supported' if ci[0] > 0
                         else 'increment_not_independently_confirmed')
    return stats


def fpr_constraint(values):
    stats = interval(values)
    mean, upper = stats['mean'], stats['upper95_one_sided']
    stats.update(budget=.01, groups_above_budget=int(np.sum(np.asarray(values) > .01)),
                 point_estimate_meets_budget=None if mean is None else mean <= .01,
                 upper_bound_meets_budget=None if upper is None else upper <= .01)
    return stats
