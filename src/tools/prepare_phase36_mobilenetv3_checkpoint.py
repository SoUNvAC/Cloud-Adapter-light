import argparse
import hashlib
from pathlib import Path

import torch
from torchvision.models import MobileNet_V3_Large_Weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2
    source = weights.get_state_dict(progress=True, check_hash=True)
    features = {
        key: value.cpu()
        for key, value in source.items()
        if key.startswith("features.")
    }
    if not features or "features.0.0.weight" not in features:
        raise RuntimeError("Unexpected torchvision MobileNetV3 checkpoint layout")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": features,
            "source": str(weights),
            "num_feature_tensors": len(features),
        },
        output,
    )
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(f"OUTPUT={output.resolve()}")
    print(f"SHA256={digest}")
    print(f"FEATURE_TENSORS={len(features)}")


if __name__ == "__main__":
    main()
