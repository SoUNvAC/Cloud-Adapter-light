"""Expand the two RGB input stems into an explicit six-channel source init."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


PATCH_KEY = "backbone.patch_embed.proj.weight"
STEM_PREFIX = "backbone.cloud_adapter.cnn.proj_1x1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expand_mixing_weight(weight: torch.Tensor) -> torch.Tensor:
    """Preserve expected activation scale while adding three mean-RGB filters."""

    if weight.ndim != 4 or weight.shape[1] != 3:
        raise ValueError(f"Expected [out,3,k,k], got {tuple(weight.shape)}")
    mean = weight.mean(dim=1, keepdim=True)
    return torch.cat((weight * 0.5, mean.repeat(1, 3, 1, 1) * 0.5), dim=1)


def expand_depthwise_weight(weight: torch.Tensor) -> torch.Tensor:
    if weight.ndim != 4 or tuple(weight.shape[1:]) != (1, 1, 1) or weight.shape[0] != 3:
        raise ValueError(f"Expected [3,1,1,1], got {tuple(weight.shape)}")
    return torch.cat((weight, weight.mean(dim=0, keepdim=True).repeat(3, 1, 1, 1)))


def expand_channel_vector(value: torch.Tensor, *, variance: bool = False) -> torch.Tensor:
    if value.ndim != 1 or value.shape[0] != 3:
        raise ValueError(f"Expected three-channel vector, got {tuple(value.shape)}")
    fill = value.mean().repeat(3)
    if variance:
        fill = value.mean().clamp_min(torch.finfo(value.dtype).eps).repeat(3)
    return torch.cat((value, fill))


def convert(state: dict[str, torch.Tensor]) -> dict[str, tuple[int, ...]]:
    state[PATCH_KEY] = expand_mixing_weight(state[PATCH_KEY])
    depthwise = f"{STEM_PREFIX}.conv_dw.weight"
    state[depthwise] = expand_depthwise_weight(state[depthwise])
    for suffix in ("weight", "bias", "running_mean", "running_var"):
        key = f"{STEM_PREFIX}.bn1.{suffix}"
        state[key] = expand_channel_vector(state[key], variance=suffix == "running_var")
    pointwise = f"{STEM_PREFIX}.conv_pw.weight"
    state[pointwise] = expand_mixing_weight(state[pointwise])
    return {
        key: tuple(state[key].shape)
        for key in (PATCH_KEY, depthwise, pointwise)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()
    source = Path(args.input)
    output = Path(args.output)
    summary_path = Path(args.summary)
    checkpoint = torch.load(source, map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint)
    shapes = convert(state)
    checkpoint.setdefault("meta", {})["phase64b_six_channel_initialization"] = {
        "band_order": ["red", "green", "blue", "nir", "swir1", "swir2"],
        "rule": "RGB weights times 0.5; three added channels use mean-RGB weights times 0.5",
        "source_checkpoint_sha256": sha256(source),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    summary = {
        "input": str(source.resolve()),
        "input_sha256": sha256(source),
        "output": str(output.resolve()),
        "output_sha256": sha256(output),
        "expanded_shapes": {key: list(value) for key, value in shapes.items()},
        "initialization": checkpoint["meta"]["phase64b_six_channel_initialization"],
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
