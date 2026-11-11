"""Fixed loss grouping and simultaneous two-group PCGrad. No extra weights."""
import math


def loss_groups(terms):
    def reduce(value):
        if isinstance(value, (list, tuple)):
            return sum(v.mean() for v in value)
        return value.mean()
    cls, spatial = [], []
    for name, value in terms.items():
        if name.endswith('loss_cls'):
            cls.append(reduce(value))
        elif name.endswith(('loss_mask', 'loss_dice')):
            spatial.append(reduce(value))
        else:
            raise AssertionError('Unspecified loss key: ' + name)
    assert cls and spatial
    return sum(cls), sum(spatial)


def dense_grads(grads, params):
    import torch
    return [torch.zeros_like(p) if g is None else g for g, p in zip(grads, params)]


def project(gc, gm):
    import torch
    # Accumulation in float64 stabilizes dot/norm tests without changing tensor dtype.
    dot = sum((c.double() * m.double()).sum() for c, m in zip(gc, gm))
    a = sum(c.double().square().sum() for c in gc)
    b = sum(m.double().square().sum() for m in gm)
    d, aa, bb = float(dot), float(a), float(b)
    conflict = d < 0 and aa > 1e-12 and bb > 1e-12
    if conflict:
        # Both projections use original gradients, never the already projected one.
        updated = [(c - (d / bb) * m) + (m - (d / aa) * c) for c, m in zip(gc, gm)]
    else:
        updated = [c + m for c, m in zip(gc, gm)]
    result = dict(dot=d, cls_squared_norm=aa, spatial_squared_norm=bb,
                  cos=d / math.sqrt(aa * bb) if aa > 0 and bb > 0 else None,
                  branch='conflict' if conflict else 'zero_norm' if min(aa, bb) <= 1e-12 else 'non_conflict',
                  norm_before=math.sqrt(max(0., aa + bb + 2 * d)),
                  norm_after=math.sqrt(sum(float(g.double().square().sum()) for g in updated)))
    assert all(torch.isfinite(g).all() for g in updated)
    return updated, result


def array_gate():
    import torch
    results = []
    for c, m, expected in [([1., 0.], [-1., 1.], 'conflict'),
                            ([1., 2.], [2., 3.], 'non_conflict'),
                            ([0., 0.], [1., 1.], 'zero_norm')]:
        c, m = torch.tensor(c), torch.tensor(m)
        got, stats = project([c], [m])
        d, a, b = float(c @ m), float(c @ c), float(m @ m)
        want = (c - d / b * m) + (m - d / a * c) if expected == 'conflict' else c + m
        assert torch.equal(got[0], want) and stats['branch'] == expected
        results.append(stats)
    return results


def paired_gradients(loss, terms, params, budget, parity=False):
    import torch
    lc, lm = loss_groups(terms)
    assert torch.allclose(lc + lm, loss, atol=1e-5, rtol=1e-6)
    budget.compute('backward_or_autograd')
    raw_c = torch.autograd.grad(lc, params, retain_graph=True, allow_unused=True)
    budget.compute('backward_or_autograd')
    raw_m = torch.autograd.grad(lm, params, retain_graph=parity, allow_unused=True)
    gc, gm = dense_grads(raw_c, params), dense_grads(raw_m, params)
    if parity:
        budget.compute('backward_or_autograd')
        baseline = dense_grads(torch.autograd.grad(loss, params, allow_unused=True), params)
        sums = [c + m for c, m in zip(gc, gm)]
        checks = [torch.allclose(a, b, atol=1e-5, rtol=1e-4) for a, b in zip(sums, baseline)]
        assert all(checks), 'Grouped gradients do not reproduce ordinary loss gradient'
        parity_receipt = dict(passed=True, atol=1e-5, rtol=1e-4,
                              max_abs=max(float((a-b).abs().max()) for a,b in zip(sums, baseline)),
                              cls_None=sum(g is None for g in raw_c),
                              spatial_None=sum(g is None for g in raw_m),
                              both_None=sum(c is None and m is None for c,m in zip(raw_c, raw_m)),
                              gate_updates=0)
    else:
        parity_receipt = None
    grads, stats = project(gc, gm)
    return grads, stats, parity_receipt
