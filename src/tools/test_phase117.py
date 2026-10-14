"""Synthetic CPU operator checks only, never load pretrained models or data."""
import io
import numpy as np
import torch
from run_phase117 import arrays, parent_logits


def main():
    raw = np.ones((1022, 1022, 13), np.float32)
    for i in range(13):
        raw[..., i] *= np.float32(.1 * (i + 1))
    mask = np.zeros((1022, 1022, 3), np.bool_)
    mask[..., 0] = True
    raw[0, 0, 8] = 0
    raw[0, 1, 8] = np.float32(65535 / 10000)
    raw[0, 2, 8] = np.nan
    raw[0, 3, 7] = 0
    mask[0, 4, 1] = True
    def zip_mock(a):
        buffer = io.BytesIO()
        np.save(buffer, a, allow_pickle=False)
        class Mock:
            def read(self, name):
                return buffer.getvalue()
        return Mock()
    six, truth, binding = arrays(np, zip_mock(raw), zip_mock(mask),
                                {'representative': 'synthetic', 'group_id': 'toy', 'split': 'fit'})
    assert binding['support_loss'] == 3
    assert (truth[0, :5] == 255).all()
    assert np.allclose(six[:, 1, 1], [.4, .3, .9, .2, 1.2, 1.3])
    torch.manual_seed(11701)
    patch = torch.nn.Conv2d(6, 3, 1)
    torch.nn.init.zeros_(patch.weight)
    torch.nn.init.zeros_(patch.bias)
    z, q = torch.randn(1, 3, 8, 8), torch.randn(1, 3, 8, 8)
    u = z + .25 * torch.tanh(patch(torch.cat((z, q), 1)))
    assert torch.equal(u, z) and sum(p.numel() for p in patch.parameters()) == 21
    logits = torch.cat((u, u[:, :1] * .5), 1)
    merged = parent_logits(logits, torch).softmax(1)
    original = logits.softmax(1)
    assert torch.allclose(merged[:, 1], original[:, 1] + original[:, 2], atol=1e-6)
    torch.nn.functional.cross_entropy(parent_logits(logits, torch), torch.zeros((1, 8, 8), dtype=torch.long)).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in patch.parameters())
    assert sum(float(p.grad.abs().sum()) for p in patch.parameters()) > 0
    print('Synthetic CPU tests passed: band identity, support, zero patch,21 parameters, parent logsumexp, gradient. Omni calls0.')


if __name__ == '__main__':
    main()
