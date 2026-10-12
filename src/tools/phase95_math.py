"""Phase95 fixed same-area support and typed conservative threshold contracts."""
import numpy as np


def support(truth):
    y = np.asarray(truth)
    assert y.shape == (1022, 1022) and set(np.unique(y)) <= {0, 1, 2, 255}
    blocks = y[:1020, :1020].reshape(340, 3, 340, 3).transpose(0, 2, 1, 3)
    coarse = blocks[:, :, 0, 0]
    keep = (coarse != 255) & (blocks == coarse[:, :, None, None]).all(axis=(2, 3))
    independent = np.ones((340, 340), bool)
    for dy in range(3):
        for dx in range(3):
            independent &= (y[dy:1020:3, dx:1020:3] == coarse) & (coarse != 255)
    assert np.array_equal(keep, independent)
    receipts = []
    for c in range(3):
        full = int((y == c).sum()); cropped = int((y[:1020, :1020] == c).sum())
        retained = int((keep & (coarse == c)).sum())
        receipts.append(dict(class_id=c, full_valid_pixels=full, cropped_pixels=cropped,
                             retained_blocks=retained, retained_pixels=9*retained,
                             retention_of_full=9*retained/full if full else None,
                             retention_of_crop=9*retained/cropped if cropped else None))
    return keep, coarse[keep], receipts


def same_scores(scores, truth, keep, probability):
    full = np.full(truth.shape, np.nan, dtype=np.asarray(scores).dtype)
    full[truth != 255] = scores
    a = full[:1020, :1020]
    fine = a.reshape(340, 3, 340, 3).transpose(0, 2, 1, 3)[keep].reshape(-1)
    prob = probability(full).astype(np.float64)[:1020, :1020]
    mean = prob.reshape(340, 3, 340, 3).mean(axis=(1, 3), dtype=np.float64)[keep]
    independent = sum(prob[dy::3, dx::3] for dy in range(3) for dx in range(3))[keep]/9
    assert np.isfinite(fine).all() and np.isfinite(mean).all()
    err = float(np.max(np.abs(mean-independent))) if len(mean) else 0.
    assert err <= 2e-15 and len(fine) == 9*len(mean)
    return fine, mean, independent, err


def threshold(negatives, dtype):
    """Lowest feasible observed negative tie, exact group counts, explicit dtype."""
    dtype = np.dtype(dtype)
    assert dtype in (np.dtype('float32'), np.dtype('float64'))
    arrays = [np.asarray(x, dtype=dtype) for x in negatives if len(x)]
    assert arrays and all(np.isfinite(x).all() for x in arrays)
    lengths = np.array([len(x) for x in arrays], np.int64)
    values = np.concatenate(arrays)
    gids = np.repeat(np.arange(len(arrays), dtype=np.int32), lengths)
    order = np.argsort(-values, kind='stable'); values = values[order]; gids = gids[order]
    ends = np.r_[np.flatnonzero(values[:-1] != values[1:]), len(values)-1]
    weights = 1/(len(arrays)*lengths)
    cumulative = np.cumsum(weights[gids], dtype=np.float64)
    k = int(np.searchsorted(cumulative[ends], .01, side='right'))-1
    def exact(j):
        if j < 0: return 0.
        counts = np.bincount(gids[:int(ends[j])+1], minlength=len(arrays))
        return float(np.mean(counts/lengths))
    while k >= 0 and exact(k) > .01: k -= 1
    while k+1 < len(ends) and exact(k+1) <= .01: k += 1
    selected = values[ends[k]] if k >= 0 else np.nextafter(values[0], dtype.type(np.inf))
    following = values[ends[k+1]] if k+1 < len(ends) else None
    counts = [int(np.count_nonzero(x >= selected)) for x in arrays]
    rate = float(np.mean(np.array(counts)/lengths))
    assert rate == exact(k) and rate <= .01
    next_rate = exact(k+1) if following is not None else None
    assert next_rate is None or next_rate > .01
    return dict(threshold=float(selected), threshold_hex=float(selected).hex(), dtype=dtype.name,
                negative_groups=len(arrays), group_weight=1/len(arrays), macro_fpr=rate,
                next_observed_threshold=float(following) if following is not None else None,
                next_macro_fpr=next_rate, false_positive=counts,
                rule='lowest feasible observed negative tie; >=; exact group counts; no free-positive threshold')


def self_test():
    rng = np.random.default_rng(95012)
    for dtype in [np.float32, np.float64]:
        for _ in range(25):
            arrays = [rng.integers(0, 12, size=int(rng.integers(40, 150))).astype(dtype)/17 for j in range(4)]
            t = threshold(arrays, dtype)
            candidates = np.unique(np.concatenate(arrays))
            eligible = [float(v) for v in candidates if np.mean([np.mean(a >= v) for a in arrays]) <= .01]
            expected = min(eligible) if eligible else float(np.nextafter(candidates[-1], dtype(np.inf)))
            assert t['threshold'] == expected
    a = np.full(100, .5, np.float64); a[-1] = np.nextafter(.5, 1.)
    t = threshold([a], np.float64)
    assert t['threshold'] == a[-1] and np.count_nonzero(a >= t['threshold']) == 1
    assert float(np.float32(t['threshold'])) != t['threshold']
    y = np.zeros((1022, 1022), np.uint8); y[0, 0] = 2; y[3, 3] = 255; y[6:9, 6:9] = 2
    k, coarse, counts = support(y)
    assert not k[0, 0] and not k[1, 1] and k[2, 2] and len(coarse) == 340*340-2
    s = rng.random(int((y != 255).sum())).astype(np.float32)
    fine, mean, independent, err = same_scores(s, y, k, lambda x: x)
    assert len(fine) == 9*len(mean) and sum(r['retained_pixels'] for r in counts) == len(fine)
    return dict(random_bruteforce_cases=50, float64_adjacent_tie_test=True, crop_mixed_invalid_and_area_test=True)


if __name__ == '__main__': print(self_test())
