import argparse
import os
from pathlib import Path
import sys

# xFormers kernels are useful for eager CUDA inference but are not portable
# ONNX operators. This must be set before importing the project model modules.
os.environ.setdefault("XFORMERS_DISABLED", "1")

import torch
import torch.nn.functional as F
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint
from mmseg.registry import MODELS
from torch import Tensor, nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cloud_adapter.models  # noqa: E402,F401


class StaticMask2FormerLogits(nn.Module):
    """Static-shape deployment graph with RGB normalization included."""

    def __init__(self, segmentor: nn.Module, precision: str):
        super().__init__()
        self.segmentor = segmentor
        self.compute_dtype = (
            torch.float16 if precision == "fp16" else torch.float32
        )
        self.register_buffer(
            "input_mean",
            torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "input_std",
            torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1),
            persistent=False,
        )

    def normalize(self, rgb_images: Tensor) -> Tensor:
        normalized = (rgb_images.float() - self.input_mean) / self.input_std
        return normalized.to(dtype=self.compute_dtype)

    def forward(self, rgb_images: Tensor) -> Tensor:
        inputs = self.normalize(rgb_images)
        features = self.segmentor.extract_feat(inputs)
        all_cls_scores, all_mask_preds = self.segmentor.decode_head(features, None)
        class_probabilities = F.softmax(all_cls_scores[-1], dim=-1)[..., :-1]
        mask_probabilities = F.interpolate(
            all_mask_preds[-1],
            size=rgb_images.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).sigmoid()
        return torch.einsum(
            "bqc,bqhw->bchw", class_probabilities, mask_probabilities
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export the standard-op Phase 12 student to static ONNX"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def check_onnx_dependency():
    try:
        import onnx
    except ImportError as error:
        raise RuntimeError(
            "ONNX is required. Install the project-pinned version with: "
            "python -m pip install onnx==1.15.0"
        ) from error
    return onnx


def build_wrapper(args):
    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get("default_scope", "mmseg"))
    model = MODELS.build(cfg.model)
    model.init_weights()
    load_checkpoint(model, args.checkpoint, map_location="cpu", strict=False)
    model = model.cuda().eval()
    if args.precision == "fp16":
        model = model.half()
    return StaticMask2FormerLogits(model, args.precision).cuda().eval(), cfg


def verify_pytorch_path(wrapper, cfg, rgb_images):
    normalized = wrapper.normalize(rgb_images)
    with torch.inference_mode():
        wrapped_logits = wrapper(rgb_images)
        features = wrapper.segmentor.extract_feat(normalized)
        reference_logits = wrapper.segmentor.decode_head.predict(
            features,
            [
                dict(
                    ori_shape=(rgb_images.shape[-2], rgb_images.shape[-1]),
                    img_shape=(rgb_images.shape[-2], rgb_images.shape[-1]),
                    pad_shape=(rgb_images.shape[-2], rgb_images.shape[-1]),
                    padding_size=[0, 0, 0, 0],
                    flip=False,
                )
            ],
            cfg.model.test_cfg,
        )
    difference = (wrapped_logits.float() - reference_logits.float()).abs()
    max_abs = difference.max().item()
    mean_abs = difference.mean().item()
    if max_abs > 5e-3:
        raise RuntimeError(
            "Deployment wrapper differs from MMSeg prediction: "
            f"max_abs={max_abs:.6g}, mean_abs={mean_abs:.6g}"
        )
    print(
        "PyTorch wrapper parity: "
        f"max_abs={max_abs:.6g}, mean_abs={mean_abs:.6g}"
    )


def main():
    args = parse_args()
    if args.input_size <= 0:
        raise ValueError("input-size must be positive")
    if args.opset < 17:
        raise ValueError("Mask2Former export requires opset 17 or newer")

    output_path = Path(args.output)
    if output_path.exists() and not args.force:
        raise FileExistsError(f"Output already exists: {output_path} (use --force)")
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    onnx = check_onnx_dependency()

    wrapper, cfg = build_wrapper(args)
    size = args.input_size
    # Engine contract: BCHW RGB float32 values in the inclusive range 0..255.
    example = torch.rand(1, 3, size, size, device="cuda") * 255.0
    verify_pytorch_path(wrapper, cfg, example)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            example,
            str(output_path),
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=["rgb_images"],
            output_names=["seg_logits"],
            dynamic_axes=None,
        )

    exported = onnx.load(str(output_path), load_external_data=False)
    onnx.checker.check_model(exported)
    custom_domains = sorted(
        {
            node.domain
            for node in exported.graph.node
            if node.domain not in ("", "ai.onnx")
        }
    )
    if custom_domains:
        raise RuntimeError(
            "Export contains non-standard ONNX operator domains: "
            + ", ".join(custom_domains)
        )
    input_info = exported.graph.input[0]
    output_info = exported.graph.output[0]
    print(
        f"ONNX checker: passed (opset {args.opset}, "
        f"{len(exported.graph.node)} standard-domain nodes)"
    )
    print(f"Input: {input_info.name} = float32[1,3,{size},{size}] RGB 0..255")
    print(f"Output: {output_info.name} = [1,4,{size},{size}] semantic logits")
    print(f"ONNX size: {output_path.stat().st_size / 2**20:.2f} MiB")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    os.chdir(REPO_ROOT)
    main()
