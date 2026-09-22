import unittest

import torch

from cloud_adapter.models.decode_heads.lite_fpn_pixel_decoder import (
    DetailLiteFPNPixelDecoder,
    LiteFPNPixelDecoder,
)


class DetailLiteFPNTest(unittest.TestCase):
    def make_decoder(self, detail=False):
        cls = DetailLiteFPNPixelDecoder if detail else LiteFPNPixelDecoder
        kwargs = dict(
            in_channels=[24, 32, 96, 320],
            strides=[4, 8, 16, 32],
            feat_channels=128,
            out_channels=128,
            num_outs=3,
            norm_cfg=dict(type="GN", num_groups=32),
        )
        if detail:
            kwargs.update(detail_channels=16, detail_gate_init=0.1)
        return cls(**kwargs)

    def test_shapes_gradients_and_small_overhead(self):
        decoder = self.make_decoder(detail=True)
        decoder.init_weights()
        features = [
            torch.randn(2, channels, size, size, requires_grad=True)
            for channels, size in zip((24, 32, 96, 320), (32, 16, 8, 4))
        ]
        mask, pyramid = decoder(features)
        self.assertEqual(mask.shape, (2, 128, 32, 32))
        self.assertEqual(
            [tuple(item.shape[-2:]) for item in pyramid],
            [(4, 4), (8, 8), (16, 16)],
        )
        mask.mean().backward()
        self.assertIsNotNone(decoder.detail_gate.grad)
        base_parameters = sum(p.numel() for p in self.make_decoder().parameters())
        detail_parameters = sum(p.numel() for p in decoder.parameters())
        self.assertLess(detail_parameters - base_parameters, 5000)

    def test_gate_validation(self):
        with self.assertRaises(ValueError):
            DetailLiteFPNPixelDecoder(
                in_channels=[24, 32],
                strides=[4, 8],
                feat_channels=128,
                out_channels=128,
                num_outs=2,
                detail_channels=0,
            )


if __name__ == "__main__":
    unittest.main()
