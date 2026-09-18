import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


SPLITS = ("train", "val", "test")
EXPECTED_COUNTS = {"train": 8490, "val": 535, "test": 975}
CLASS_NAMES = ("clear", "thick cloud", "thin cloud", "cloud shadow")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit the immutable CloudSEN12 High L1C train/val/test protocol."
    )
    parser.add_argument("--data-root", default="data/cloudsen12_high_l1c")
    parser.add_argument(
        "--output",
        default="work_dirs/phase21_protocol/cloudsen12_high_l1c_manifest.json",
    )
    parser.add_argument(
        "--max-class-tv",
        type=float,
        default=0.10,
        help="Maximum total-variation distance from train class proportions.",
    )
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_label_png(path):
    with Image.open(path) as image:
        labels = np.asarray(image, dtype=np.uint8)
    if labels.ndim != 2:
        raise ValueError(f"Expected a single-channel label PNG: {path}")
    height, width = labels.shape
    return width, height, labels


def normalized(histogram):
    total = sum(histogram)
    return [count / total for count in histogram]


def total_variation(left, right):
    return 0.5 * sum(abs(a - b) for a, b in zip(left, right))


def audit_split(data_root, split):
    image_dir = data_root / "img_dir" / split
    ann_dir = data_root / "ann_dir" / split
    if not image_dir.is_dir() or not ann_dir.is_dir():
        raise FileNotFoundError(f"Missing {split} image/annotation directory")

    images = {path.stem: path for path in image_dir.glob("*.png")}
    annotations = {path.stem: path for path in ann_dir.glob("*.png")}
    missing_annotations = sorted(set(images) - set(annotations))
    missing_images = sorted(set(annotations) - set(images))
    if missing_annotations or missing_images:
        raise RuntimeError(
            f"Unpaired {split} data: missing_annotations={missing_annotations[:5]}, "
            f"missing_images={missing_images[:5]}"
        )

    records = []
    class_histogram = [0] * len(CLASS_NAMES)
    invalid_values = set()
    image_hashes = set()
    pair_hashes = set()
    sort_key = lambda value: (0, int(value)) if value.isdigit() else (1, value)
    for stem in sorted(images, key=sort_key):
        image_path = images[stem]
        ann_path = annotations[stem]
        image_hash = sha256(image_path)
        ann_hash = sha256(ann_path)
        pair_hash = hashlib.sha256(f"{image_hash}:{ann_hash}".encode()).hexdigest()
        width, height, labels = read_label_png(ann_path)
        unique_values = np.unique(labels)
        invalid_values.update(
            int(value) for value in unique_values if value >= len(CLASS_NAMES)
        )
        sample_histogram = np.bincount(
            labels.reshape(-1), minlength=len(CLASS_NAMES)
        )[: len(CLASS_NAMES)].astype(np.int64).tolist()
        class_histogram = [
            total + sample
            for total, sample in zip(class_histogram, sample_histogram)
        ]
        records.append(
            {
                "id": stem,
                "image": image_path.relative_to(data_root).as_posix(),
                "annotation": ann_path.relative_to(data_root).as_posix(),
                "width": width,
                "height": height,
                "image_sha256": image_hash,
                "annotation_sha256": ann_hash,
                "pair_sha256": pair_hash,
                "class_histogram": sample_histogram,
            }
        )
        image_hashes.add(image_hash)
        pair_hashes.add(pair_hash)

    return {
        "count": len(records),
        "class_histogram": class_histogram,
        "class_proportions": normalized(class_histogram),
        "invalid_label_values": sorted(invalid_values),
        "records": records,
        "_image_hashes": image_hashes,
        "_pair_hashes": pair_hashes,
    }


def main():
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    output = Path(args.output).resolve()
    audited = {split: audit_split(data_root, split) for split in SPLITS}

    gates = {}
    for split in SPLITS:
        gates[f"{split}_count"] = audited[split]["count"] == EXPECTED_COUNTS[split]
        gates[f"{split}_valid_labels"] = not audited[split]["invalid_label_values"]

    overlaps = {}
    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1 :]:
            key = f"{left}_{right}"
            image_overlap = audited[left]["_image_hashes"] & audited[right]["_image_hashes"]
            pair_overlap = audited[left]["_pair_hashes"] & audited[right]["_pair_hashes"]
            overlaps[key] = {
                "exact_image_overlap": len(image_overlap),
                "exact_pair_overlap": len(pair_overlap),
            }
            gates[f"{key}_no_exact_image_overlap"] = not image_overlap
            gates[f"{key}_no_exact_pair_overlap"] = not pair_overlap

    train_proportions = audited["train"]["class_proportions"]
    class_tv = {
        split: total_variation(train_proportions, audited[split]["class_proportions"])
        for split in ("val", "test")
    }
    gates["val_class_tv"] = class_tv["val"] <= args.max_class_tv
    gates["test_class_tv"] = class_tv["test"] <= args.max_class_tv

    for split in SPLITS:
        audited[split].pop("_image_hashes")
        audited[split].pop("_pair_hashes")

    passed = all(gates.values())
    manifest = {
        "protocol": "CloudSEN12 High official train/val/test partitions",
        "data_root": data_root.as_posix(),
        "classes": CLASS_NAMES,
        "expected_counts": EXPECTED_COUNTS,
        "max_class_tv": args.max_class_tv,
        "class_total_variation_from_train": class_tv,
        "overlaps": overlaps,
        "gates": gates,
        "passed": passed,
        "splits": audited,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    summary = {
        "passed": passed,
        "counts": {split: audited[split]["count"] for split in SPLITS},
        "class_proportions": {
            split: audited[split]["class_proportions"] for split in SPLITS
        },
        "class_total_variation_from_train": class_tv,
        "overlaps": overlaps,
        "failed_gates": [name for name, value in gates.items() if not value],
        "manifest": output.as_posix(),
    }
    print(json.dumps(summary, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
