"""Fixed Phase93 feature and independent numeric contracts; no estimator fits."""
import numpy as np

SEED = 93012
COLUMNS = {'G6': [5, 6, 7, 2, 3, 4], 'G7': [5, 6, 7, 2, 3, 4, 1]}
PARAMS = dict(loss='log_loss', learning_rate=.05, max_iter=200,
              max_leaf_nodes=15, max_depth=4, min_samples_leaf=20,
              l2_regularization=1., max_bins=255, max_features=1.,
              categorical_features=None, monotonic_cst=None, interaction_cst=None,
              class_weight=None, early_stopping=False, warm_start=False,
              random_state=SEED)


def selected(x, name):
    return np.ascontiguousarray(np.asarray(x)[:, COLUMNS[name]], dtype=np.float64)


def equal_weights(sizes):
    assert sizes and all(n > 0 for n in sizes)
    n = sum(sizes)
    w = np.concatenate([np.full(k, n / (len(sizes) * k)) for k in sizes])
    assert abs(w.sum() - n) <= n * 1e-12
    return w


def independent_probability(model, x):
    """Walk finite numeric trees without sklearn's prediction implementation.

    Used on deterministic probes only, not presented as a second full inference.
    Histogram boosting leaf values already include the learning rate.
    """
    x = np.asarray(x, np.float64)
    assert np.isfinite(x).all()
    raw = np.full(len(x), float(model._baseline_prediction.ravel()[0]))
    for predictors in model._predictors:
        assert len(predictors) == 1
        nodes = predictors[0].nodes
        index = np.zeros(len(x), np.int64)
        while True:
            active = np.flatnonzero(nodes['is_leaf'][index] == 0)
            if not len(active):
                break
            n = nodes[index[active]]
            assert not np.any(n['is_categorical'])
            go_left = x[active, n['feature_idx']] <= n['num_threshold']
            index[active] = np.where(go_left, n['left'], n['right'])
        raw += nodes['value'][index]
    return np.exp(-np.logaddexp(0., -raw)).astype(np.float32)


def probes(x, maximum=257):
    assert len(x)
    return np.asarray(x)[np.linspace(0, len(x)-1, min(maximum, len(x)), dtype=int)]
