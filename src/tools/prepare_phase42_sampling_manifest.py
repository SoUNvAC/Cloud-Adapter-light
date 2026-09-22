import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-samples", type=int, default=8490)
    parser.add_argument("--rich-fraction", type=float, default=0.25)
    args = parser.parse_args()
    if not 0.0 < args.rich_fraction <= 1.0:
        raise ValueError("rich-fraction must be in (0, 1]")

    ann_dir = Path(args.ann_dir)
    paths = sorted(ann_dir.glob("*.png"), key=lambda path: path.name)
    if len(paths) != args.expected_samples:
        raise RuntimeError(
            f"Expected {args.expected_samples} annotations, found {len(paths)}"
        )
    rows = []
    digest = hashlib.sha256()
    for index, path in enumerate(paths):
        data = path.read_bytes()
        digest.update(path.name.encode("utf-8") + b"\0" + data)
        with Image.open(path) as image:
            target = np.asarray(image)
        valid = (target >= 0) & (target < 4)
        weak = valid & ((target == 2) | (target == 3))
        rows.append(
            {
                "index": index,
                "path": path.name,
                "weak_prevalence": float(weak.sum() / valid.sum()),
            }
        )
    expected_names = {f"{index}.png" for index in range(len(paths))}
    if expected_names != {row["path"] for row in rows}:
        raise RuntimeError("Annotation filenames are not the expected 0..N-1 set")

    ranked = sorted(rows, key=lambda row: (-row["weak_prevalence"], row["path"]))
    rich_count = math.ceil(len(rows) * args.rich_fraction)
    rich_rows = ranked[:rich_count]
    result = {
        "phase": 42,
        "split": "train",
        "sample_count": len(rows),
        "rich_fraction": args.rich_fraction,
        "rich_count": rich_count,
        "rich_probability": 0.5,
        "weak_classes": [2, 3],
        "annotation_sha256": digest.hexdigest(),
        "minimum_rich_prevalence": rich_rows[-1]["weak_prevalence"],
        "ordered_paths": [row["path"] for row in rows],
        "rich_indices": [row["index"] for row in rich_rows],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in {"rich_indices", "ordered_paths"}}, indent=2))


if __name__ == "__main__":
    main()
