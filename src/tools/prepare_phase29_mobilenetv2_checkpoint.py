import argparse
from pathlib import Path

import torch


def convert_state_dict(state_dict):
    converted = {}
    for key, value in state_dict.items():
        if key.startswith("backbone."):
            converted[key.removeprefix("backbone.")] = value
        elif not key.startswith("head."):
            raise ValueError(f"Unexpected MobileNetV2 checkpoint key: {key}")
    if not converted or "conv1.conv.weight" not in converted:
        raise RuntimeError("MobileNetV2 backbone weights were not found")
    return converted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("output")
    args = parser.parse_args()

    source = torch.load(args.source, map_location="cpu")
    state_dict = source.get("state_dict", source)
    converted = convert_state_dict(state_dict)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": converted}, output)
    print(f"Saved {len(converted)} MobileNetV2 backbone tensors to {output}")


if __name__ == "__main__":
    main()
