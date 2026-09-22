import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("XFORMERS_DISABLED", "1")

import numpy as np
from PIL import Image
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from export_phase12_onnx import build_wrapper  # noqa: E402
from validate_phase12_onnxruntime import load_rgb_image  # noqa: E402


def checkpoint_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="work_dirs/phase45_protocol_audit/scene_disjoint_manifest.csv")
    parser.add_argument("--output-root", default="work_dirs/phase45_teacher_student")
    parser.add_argument("--config", default="configs/protocol/phase22_clean_v8_l1c.py")
    parser.add_argument("--checkpoint", default="work_dirs/phase22_clean_v8/seed42/best_mIoU_iter_40000.pth")
    parser.add_argument("--confidence", type=float, default=0.90)
    parser.add_argument("--min-image-coverage", type=float, default=0.05)
    parser.add_argument("--expected-images", type=int, default=6502)
    args = parser.parse_args()
    output_root = Path(args.output_root)
    pseudo_dir = output_root / "pseudo_masks"
    pseudo_dir.mkdir(parents=True, exist_ok=True)
    with Path(args.manifest).open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["new_split"] == "target_train"]
    rows.sort(key=lambda row: row["name"])
    if len(rows) != args.expected_images:
        raise RuntimeError(f"Expected {args.expected_images} target-train images, found {len(rows)}")
    wrapper_args = argparse.Namespace(
        config=args.config, checkpoint=args.checkpoint, precision="fp16",
        active_block_indices=None,
    )
    wrapper, _ = build_wrapper(wrapper_args)
    accepted_by_class = np.zeros(4, dtype=np.int64)
    predicted_by_class = np.zeros(4, dtype=np.int64)
    accepted_total = 0
    pixel_total = 0
    kept = []
    for index, row in enumerate(rows, 1):
        rgb = torch.from_numpy(load_rgb_image(Path(row["image_path"]), 512)).cuda()
        with torch.inference_mode():
            probabilities = torch.softmax(wrapper(rgb).float(), dim=1)
            confidence, prediction = probabilities.max(dim=1)
        confidence = confidence[0].cpu().numpy()
        prediction = prediction[0].cpu().numpy().astype(np.uint8)
        accepted = confidence >= args.confidence
        coverage = float(accepted.mean())
        pixel_total += prediction.size
        accepted_total += int(accepted.sum())
        predicted_by_class += np.bincount(prediction.ravel(), minlength=4)
        for class_index in range(4):
            accepted_by_class[class_index] += int(np.count_nonzero(accepted & (prediction == class_index)))
        pseudo = prediction.copy()
        pseudo[~accepted] = 255
        if coverage >= args.min_image_coverage:
            pseudo_path = pseudo_dir / row["name"]
            Image.fromarray(pseudo).save(pseudo_path)
            item = dict(row)
            item["pseudo_mask_path"] = pseudo_path.as_posix()
            item["pseudo_coverage"] = f"{coverage:.10f}"
            kept.append(item)
        if index == 1 or index % 100 == 0 or index == len(rows):
            print(f"pseudo labels: {index}/{len(rows)}, kept={len(kept)}", flush=True)
    fields = list(rows[0]) + ["pseudo_mask_path", "pseudo_coverage"]
    with (output_root / "pseudo_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(kept)
    gates = {
        "expected_target_train_images": len(rows) == args.expected_images,
        "at_least_1000_training_images": len(kept) >= 1000,
        "all_classes_have_accepted_pixels": bool(np.all(accepted_by_class > 0)),
        "target_labels_not_read": True,
        "target_test_sealed": True,
    }
    summary = {
        "phase": "45B-teacher-pseudo",
        "teacher_config": args.config,
        "teacher_checkpoint": args.checkpoint,
        "teacher_checkpoint_sha256": checkpoint_sha256(Path(args.checkpoint)),
        "confidence_threshold": args.confidence,
        "minimum_image_coverage": args.min_image_coverage,
        "target_train_images": len(rows),
        "kept_training_images": len(kept),
        "accepted_pixel_fraction": accepted_total / pixel_total,
        "predicted_pixels_by_source_class": predicted_by_class.tolist(),
        "accepted_pixels_by_source_class": accepted_by_class.tolist(),
        "target_labels_read": False,
        "target_test_evaluated": False,
        "gates": gates,
        "passed": all(gates.values()),
    }
    (output_root / "pseudo_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
