"""Training-only, scale-normalized H residual energy; no extra forward."""
import torch


def install(model):
    terms = []
    handles = []

    def capture(module, args, output):
        if torch.is_grad_enabled():
            reference = args[0].detach().float().square().mean()
            terms.append(output.float().square().mean() / (reference + 1e-6))

    for head in model.backbone.target_head_delta:
        handles.append(head.register_forward_hook(capture))
    original = model.loss

    def loss(*args, **kwargs):
        terms.clear()
        result = original(*args, **kwargs)
        assert len(terms) == 4, len(terms)
        penalty = torch.stack(terms).mean()
        assert penalty.requires_grad and torch.isfinite(penalty).item()
        assert penalty.item() >= 0
        result['loss_residual_energy'] = penalty
        terms.clear()
        return result

    model.loss = loss
    model._phase406_handles = handles


def self_test():
    from types import SimpleNamespace
    heads = torch.nn.ModuleList([torch.nn.Linear(2, 2, bias=False) for _ in range(4)])
    for head in heads:
        torch.nn.init.eye_(head.weight)
    feature = torch.ones(3, 2, requires_grad=True)
    model = SimpleNamespace(backbone=SimpleNamespace(target_head_delta=heads))
    def original():
        for head in heads:
            head(feature)
        return {'loss_task': feature.sum() * 0}
    model.loss = original
    install(model)
    value = model.loss()['loss_residual_energy']
    assert abs(value.item() - 1) < 2e-6
    value.backward()
    assert all(h.weight.grad is not None and h.weight.grad.abs().sum() > 0 for h in heads)
    with torch.no_grad():
        for head in heads:
            head.weight.zero_()
    assert model.loss()['loss_residual_energy'].item() == 0
    return {'passed': True, 'nonzero_penalty_gradient_all_four_heads': True, 'zero_residual_zero_penalty': True}
