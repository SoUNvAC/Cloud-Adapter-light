import argparse
import copy
from pathlib import Path
import re

import torch


FC1_PATTERN = re.compile(r"^(.*backbone\.blocks\.(\d+)\.mlp\.)fc1\.weight$")


def select_channels(fc1_weight, fc2_weight, target_hidden):
    if fc1_weight.ndim != 2 or fc2_weight.ndim != 2:
        raise ValueError("DINO MLP weights must be matrices")
    if fc1_weight.shape[0] != fc2_weight.shape[1]:
        raise ValueError("fc1/fc2 hidden dimensions do not match")
    if not 0 < target_hidden <= fc1_weight.shape[0]:
        raise ValueError("invalid target hidden dimension")
    score = torch.linalg.vector_norm(fc1_weight.float(), dim=1)
    score *= torch.linalg.vector_norm(fc2_weight.float(), dim=0)
    indices = torch.topk(score, target_hidden, largest=True, sorted=False).indices
    return torch.sort(indices).values


def prune_state_dict(state_dict, target_ratio):
    result = copy.copy(state_dict)
    converted = []
    for key, fc1_weight in state_dict.items():
        match = FC1_PATTERN.match(key)
        if not match:
            continue
        prefix, block_index = match.groups()
        fc1_bias_key = prefix + "fc1.bias"
        fc2_weight_key = prefix + "fc2.weight"
        fc2_bias_key = prefix + "fc2.bias"
        fc2_weight = state_dict[fc2_weight_key]
        target_hidden = int(fc1_weight.shape[1] * target_ratio)
        indices = select_channels(fc1_weight, fc2_weight, target_hidden)
        result[key] = fc1_weight.index_select(0, indices).clone()
        if fc1_bias_key in state_dict:
            result[fc1_bias_key] = state_dict[fc1_bias_key].index_select(0, indices).clone()
        result[fc2_weight_key] = fc2_weight.index_select(1, indices).clone()
        if fc2_bias_key in state_dict:
            result[fc2_bias_key] = state_dict[fc2_bias_key].clone()
        converted.append(int(block_index))
    if sorted(converted) != list(range(12)):
        raise RuntimeError(f"Expected DINO blocks 0..11, converted {converted}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument("--target-ratio", type=float, required=True)
    args = parser.parse_args()
    if args.target_ratio not in {4.0, 3.0, 2.5, 2.0}:
        raise SystemExit("target ratio is outside the preregistered screen")

    source = torch.load(args.source, map_location="cpu")
    state_dict = source.get("state_dict", source)
    pruned = prune_state_dict(state_dict, args.target_ratio)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"meta": source.get("meta", {}), "state_dict": pruned}, output)
    print(f"Saved ratio {args.target_ratio:g} checkpoint to {output}")


if __name__ == "__main__":
    main()
