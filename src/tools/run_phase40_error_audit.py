import argparse
import csv
import json
import math
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
from validate_phase12_onnxruntime import list_images, load_rgb_image  # noqa: E402


CLASS_NAMES = ("clear", "thick cloud", "thin cloud", "cloud shadow")
WEAK = (2, 3)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--ann-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-images", type=int, default=535)
    parser.add_argument("--expected-miou", type=float, default=67.35)
    parser.add_argument("--max-miou-delta", type=float, default=0.05)
    parser.add_argument("--boundary-gap", type=float, default=8.0)
    parser.add_argument("--prevalence-gap", type=float, default=8.0)
    return parser.parse_args()


def annotation(path):
    with Image.open(path) as image:
        value = np.asarray(image)
    if value.ndim != 2:
        raise ValueError(f"Expected single-channel annotation: {path}")
    return value.astype(np.int64, copy=False)


def update(confusion, prediction, target, mask=None):
    valid = (target >= 0) & (target < len(CLASS_NAMES))
    if mask is not None:
        valid &= mask
    encoded = len(CLASS_NAMES) * target[valid] + prediction[valid]
    confusion += np.bincount(encoded, minlength=16).reshape(4, 4)
    return int(valid.sum())


def boundaries(target, valid):
    result = np.zeros_like(valid)
    horizontal = valid[:, 1:] & valid[:, :-1] & (target[:, 1:] != target[:, :-1])
    result[:, 1:] |= horizontal
    result[:, :-1] |= horizontal
    vertical = valid[1:, :] & valid[:-1, :] & (target[1:, :] != target[:-1, :])
    result[1:, :] |= vertical
    result[:-1, :] |= vertical
    return result & valid


def metrics(confusion):
    confusion = confusion.astype(np.float64)
    true_positive = np.diag(confusion)
    ground_truth = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    union = ground_truth + predicted - true_positive
    iou = np.divide(true_positive, union, out=np.full(4, np.nan), where=union > 0)
    recall = np.divide(true_positive, ground_truth, out=np.full(4, np.nan),
                       where=ground_truth > 0)
    return {
        "mIoU": 100.0 * float(np.nanmean(iou)),
        "class_iou": [100.0 * float(value) for value in iou],
        "class_recall": [100.0 * float(value) for value in recall],
        "weak_mIoU": 100.0 * float(np.nanmean(iou[list(WEAK)])),
        "weak_recall": 100.0 * float(np.nanmean(recall[list(WEAK)])),
    }


def finite(value):
    if isinstance(value, list):
        return all(math.isfinite(item) for item in value)
    return math.isfinite(value)


def main():
    args = parse_args()
    image_dir, ann_dir = Path(args.image_dir), Path(args.ann_dir)
    paths = list_images(image_dir, 2**31 - 1)
    if len(paths) != args.expected_images:
        raise RuntimeError(f"Expected {args.expected_images} images, found {len(paths)}")
    if len({path.relative_to(image_dir).as_posix() for path in paths}) != len(paths):
        raise RuntimeError("Duplicate validation image paths")

    wrapper_args = argparse.Namespace(config=args.config,
                                      checkpoint=args.checkpoint,
                                      precision="fp16",
                                      active_block_indices=None)
    wrapper, _ = build_wrapper(wrapper_args)
    overall = np.zeros((4, 4), dtype=np.int64)
    boundary_confusion = np.zeros((4, 4), dtype=np.int64)
    interior_confusion = np.zeros((4, 4), dtype=np.int64)
    records = []

    for index, image_path in enumerate(paths, 1):
        relative = image_path.relative_to(image_dir)
        target = annotation(ann_dir / relative)
        rgb = torch.from_numpy(load_rgb_image(image_path, 512)).cuda()
        with torch.inference_mode():
            prediction = wrapper(rgb).argmax(dim=1)[0].cpu().numpy().astype(np.int64)
        if prediction.shape != target.shape:
            raise ValueError(f"Shape mismatch for {relative}")
        valid = (target >= 0) & (target < 4)
        edge = boundaries(target, valid)
        weak_pixels = valid & np.isin(target, WEAK)
        item_confusion = np.zeros((4, 4), dtype=np.int64)
        update(overall, prediction, target)
        update(item_confusion, prediction, target)
        update(boundary_confusion, prediction, target, edge)
        update(interior_confusion, prediction, target, valid & ~edge)
        records.append({
            "path": relative.as_posix(),
            "weak_prevalence": float(weak_pixels.sum() / valid.sum()),
            "mIoU": metrics(item_confusion)["mIoU"],
            "confusion": item_confusion,
        })
        if index == 1 or index % 50 == 0 or index == len(paths):
            print(f"Evaluated {index}/{len(paths)}: {relative}", flush=True)

    ordered = sorted(records, key=lambda row: (row["weak_prevalence"], row["path"]))
    groups = np.array_split(np.arange(len(ordered)), 4)
    quartiles = []
    for number, indices in enumerate(groups, 1):
        confusion = sum((ordered[int(i)]["confusion"] for i in indices),
                        np.zeros((4, 4), dtype=np.int64))
        row_metrics = metrics(confusion)
        quartiles.append({
            "quartile": number,
            "images": len(indices),
            "min_weak_prevalence": ordered[int(indices[0])]["weak_prevalence"],
            "max_weak_prevalence": ordered[int(indices[-1])]["weak_prevalence"],
            "weak_mIoU": row_metrics["weak_mIoU"],
            "class_iou": row_metrics["class_iou"],
        })

    overall_metrics = metrics(overall)
    boundary_metrics = metrics(boundary_confusion)
    interior_metrics = metrics(interior_confusion)
    boundary_recall_gap = interior_metrics["weak_recall"] - boundary_metrics["weak_recall"]
    prevalence_miou_gap = quartiles[-1]["weak_mIoU"] - quartiles[0]["weak_mIoU"]
    if boundary_recall_gap >= args.boundary_gap:
        decision = "high_resolution_boundary_preserving_architecture"
    elif prevalence_miou_gap >= args.prevalence_gap:
        decision = "rarity_aware_sampling"
    else:
        decision = "context_modeling"

    strata_finite = all(
        finite(row["weak_mIoU"]) and finite(row["class_iou"])
        for row in quartiles
    ) and all(finite(row[key]) for row in (boundary_metrics, interior_metrics)
              for key in ("weak_mIoU", "weak_recall"))
    gates = {
        "expected_unique_pairs": len(paths) == args.expected_images,
        "aggregate_mIoU_reproduced": abs(overall_metrics["mIoU"] - args.expected_miou)
                                      <= args.max_miou_delta,
        "boundary_and_interior_nonempty": int(boundary_confusion.sum()) > 0
                                          and int(interior_confusion.sum()) > 0,
        "strata_finite": strata_finite,
        "internal_test_sealed": True,
    }
    summary = {
        "phase": 40,
        "audit_only": True,
        "selection_split": "val",
        "internal_test_evaluated": False,
        "test_evaluated": False,
        "checkpoint": args.checkpoint,
        "evaluated_images": len(paths),
        "thresholds": {"expected_mIoU": args.expected_miou,
                       "max_mIoU_delta": args.max_miou_delta,
                       "boundary_recall_gap": args.boundary_gap,
                       "prevalence_mIoU_gap": args.prevalence_gap},
        "overall": overall_metrics,
        "boundary": boundary_metrics,
        "interior": interior_metrics,
        "boundary_recall_gap": boundary_recall_gap,
        "prevalence_quartiles": quartiles,
        "prevalence_mIoU_gap": prevalence_miou_gap,
        "selected_direction": decision,
        "gates": gates,
        "passed": all(gates.values()),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "per_image.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("path", "weak_prevalence", "mIoU"))
        writer.writeheader()
        writer.writerows({key: row[key] for key in writer.fieldnames} for row in records)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = (
        "Phase 40 frozen validation error audit\n"
        "========================================\n\n"
        f"Images: {len(paths)}\n"
        f"Aggregate mIoU: {overall_metrics['mIoU']:.4f}\n"
        f"Weak mIoU: {overall_metrics['weak_mIoU']:.4f}\n"
        f"Boundary weak recall: {boundary_metrics['weak_recall']:.4f}\n"
        f"Interior weak recall: {interior_metrics['weak_recall']:.4f}\n"
        f"Interior-boundary recall gap: {boundary_recall_gap:.4f}\n"
        f"Q4-Q1 prevalence weak-mIoU gap: {prevalence_miou_gap:.4f}\n"
        f"Selected direction: {decision}\n"
        f"Integrity passed: {summary['passed']}\n"
        "CloudSEN internal test remains sealed.\n"
    )
    (output_dir / "PHASE40_REPORT.txt").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
