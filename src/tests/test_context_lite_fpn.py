import unittest

import torch

from cloud_adapter.models.decode_heads.lite_fpn_pixel_decoder import (
    ContextLiteFPNPixelDecoder,
    LiteFPNPixelDecoder,
)


class ContextLiteFPNTest(unittest.TestCase):
    def make_decoder(self, context=False):
        cls = ContextLiteFPNPixelDecoder if context else LiteFPNPixelDecoder
        return cls(
            in_channels=[24, 32, 96, 320],
            strides=[4, 8, 16, 32],
            feat_channels=128,
            out_channels=128,
            num_outs=3,
            norm_cfg=dict(type="GN", num_groups=32),
        )

    def test_shapes_identity_gate_gradients_and_overhead(self):
        decoder = self.make_decoder(context=True)
        decoder.init_weights()
        self.assertEqual(torch.count_nonzero(decoder.context_projection.weight), 0)
        self.assertEqual(torch.count_nonzero(decoder.context_projection.bias), 0)
        features = [
            torch.randn(2, channels, size, size, requires_grad=True)
            for channels, size in zip((24, 32, 96, 320), (32, 16, 8, 4))
        ]
        projected = decoder.context_projection(
            torch.nn.functional.adaptive_avg_pool2d(features[-1], 1)
        )
        self.assertTrue(
            torch.equal(2.0 * projected.sigmoid(), torch.ones_like(projected))
        )
        mask, pyramid = decoder(features)
        self.assertEqual(mask.shape, (2, 128, 32, 32))
        self.assertEqual(
            [tuple(item.shape[-2:]) for item in pyramid],
            [(4, 4), (8, 8), (16, 16)],
        )
        (mask.mean() + sum(item.mean() for item in pyramid)).backward()
        self.assertIsNotNone(decoder.context_projection.weight.grad)
        base_parameters = sum(p.numel() for p in self.make_decoder().parameters())
        context_parameters = sum(p.numel() for p in decoder.parameters())
        self.assertLess(context_parameters - base_parameters, 42000)


if __name__ == "__main__":
    unittest.main()
