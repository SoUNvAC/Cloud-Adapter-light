import argparse
from collections import OrderedDict
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create compact FP16 checkpoints and fake-quantized W8 checkpoints "
            "for deployment accuracy screening."
        )
    )
    parser.add_argument("--input", required=True, help="Source MMEngine checkpoint")
    parser.add_argument("--output", required=True, help="Output inference checkpoint")
    parser.add_argument(
        "--mode",
        choices=("fp16", "w8-sim"),
        required=True,
        help="FP16 storage or per-output-channel symmetric W8 simulation",
    )
    parser.add_argument(
        "--scope",
        choices=("all", "vfm", "adapter", "decoder", "adapter-decoder"),
        default="all",
        help="Eligible module group for W8 simulation",
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite an existing output file"
    )
    return parser.parse_args()


def in_scope(name: str, scope: str) -> bool:
    is_adapter = name.startswith("backbone.cloud_adapter.")
    is_vfm = name.startswith("backbone.") and not is_adapter
    is_decoder = name.startswith("decode_head.")
    if scope == "all":
        return True
    if scope == "vfm":
        return is_vfm
    if scope == "adapter":
        return is_adapter
    if scope == "decoder":
        return is_decoder
    return is_adapter or is_decoder


def is_quantizable_weight(name: str, tensor: torch.Tensor, scope: str) -> bool:
    return (
        name.endswith(".weight")
        and tensor.is_floating_point()
        and tensor.ndim >= 2
        and in_scope(name, scope)
    )


def fake_quantize_w8_per_output_channel(tensor: torch.Tensor):
    value = tensor.detach().float()
    reduce_dims = tuple(range(1, value.ndim))
    max_abs = value.abs().amax(dim=reduce_dims, keepdim=True)
    scale = (max_abs / 127.0).clamp_min(torch.finfo(torch.float32).eps)
    quantized = torch.round(value / scale).clamp_(-127, 127)
    restored = quantized * scale
    return restored, value


def checkpoint_state(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must be a mapping")
    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, dict):
        raise TypeError("Checkpoint state_dict must be a mapping")
    return state


def main():
    args = parse_args()
    source_path = Path(args.input)
    output_path = Path(args.output)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if output_path.exists() and not args.force:
        raise FileExistsError(f"Output already exists: {output_path} (use --force)")

    checkpoint = torch.load(source_path, map_location="cpu")
    source_state = checkpoint_state(checkpoint)
    output_state = OrderedDict()
    quantized_tensors = 0
    quantized_parameters = 0
    floating_parameters = 0
    error_sq_sum = 0.0
    reference_sq_sum = 0.0
    max_abs_error = 0.0

    for name, value in source_state.items():
        if name.startswith("teacher."):
            continue
        if not torch.is_tensor(value):
            output_state[name] = value
            continue
        value = value.detach().cpu()
        if not value.is_floating_point():
            output_state[name] = value
            continue

        floating_parameters += value.numel()
        if args.mode == "w8-sim" and is_quantizable_weight(
            name, value, args.scope
        ):
            restored, reference = fake_quantize_w8_per_output_channel(value)
            error = restored - reference
            error_sq_sum += error.square().sum().item()
            reference_sq_sum += reference.square().sum().item()
            max_abs_error = max(max_abs_error, error.abs().max().item())
            output_state[name] = restored.half()
            quantized_tensors += 1
            quantized_parameters += value.numel()
        else:
            output_state[name] = value.half()

    metadata = {}
    if args.mode == "w8-sim" and quantized_tensors == 0:
        raise RuntimeError(
            f"No quantizable weights matched scope {args.scope!r}; "
            "check the checkpoint key prefixes"
        )
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("meta"), dict):
        metadata.update(checkpoint["meta"])
    metadata["deployment_export"] = dict(
        source=str(source_path),
        mode=args.mode,
        scope=args.scope if args.mode == "w8-sim" else None,
        note=(
            "W8 simulation stores dequantized FP16 weights for accuracy screening; "
            "it is not an INT8 runtime engine."
            if args.mode == "w8-sim"
            else "Inference-only FP16-weight checkpoint."
        ),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(meta=metadata, state_dict=output_state), output_path)

    source_mib = source_path.stat().st_size / 2**20
    output_mib = output_path.stat().st_size / 2**20
    print(f"Mode: {args.mode}")
    if args.mode == "w8-sim":
        relative_rmse = (
            (error_sq_sum / reference_sq_sum) ** 0.5
            if reference_sq_sum > 0
            else 0.0
        )
        print(f"Scope: {args.scope}")
        print(f"Quantized tensors: {quantized_tensors}")
        print(f"Quantized parameters: {quantized_parameters / 1e6:.3f} M")
        print(f"Quantized share of floating parameters: "
              f"{100.0 * quantized_parameters / floating_parameters:.2f}%")
        print(f"Weight relative RMSE: {relative_rmse:.6f}")
        print(f"Maximum absolute weight error: {max_abs_error:.6f}")
    print(f"Source checkpoint: {source_mib:.2f} MiB")
    print(f"Output checkpoint: {output_mib:.2f} MiB")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
