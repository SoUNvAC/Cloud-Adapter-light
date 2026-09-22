import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("XFORMERS_DISABLED", "1")

import numpy as np
from PIL import Image
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from export_phase12_onnx import build_wrapper  # noqa: E402
from validate_phase12_onnxruntime import load_rgb_image  # noqa: E402


CLASS_NAMES = ("clear", "cloud_shadow", "thin_cloud", "cloud")
SOURCE_TO_TARGET = np.asarray([0, 3, 2, 1], dtype=np.int64)
WEAK = (1, 2)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def binary_boundary(binary):
    result = np.zeros(binary.shape, dtype=bool)
    horizontal = binary[:, 1:] != binary[:, :-1]
    result[:, 1:] |= horizontal
    result[:, :-1] |= horizontal
    vertical = binary[1:, :] != binary[:-1, :]
    result[1:, :] |= vertical
    result[:-1, :] |= vertical
    return result


def dilate_one(mask):
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    height, width = mask.shape
    result = np.zeros_like(mask)
    for dy in range(3):
        for dx in range(3):
            result |= padded[dy:dy + height, dx:dx + width]
    return result


def confusion_metrics(confusion):
    values = confusion.astype(np.float64)
    true_positive = np.diag(values)
    ground_truth = values.sum(axis=1)
    predicted = values.sum(axis=0)
    union = ground_truth + predicted - true_positive
    iou = np.divide(true_positive, union, out=np.full(4, np.nan), where=union > 0)
    recall = np.divide(
        true_positive, ground_truth, out=np.full(4, np.nan), where=ground_truth > 0
    )
    return {
        "mIoU": 100.0 * float(np.nanmean(iou)),
        "weak_mIoU": 100.0 * float(np.nanmean(iou[list(WEAK)])),
        "per_class_iou": {CLASS_NAMES[i]: 100.0 * float(iou[i]) for i in range(4)},
        "per_class_recall": {
            CLASS_NAMES[i]: 100.0 * float(recall[i]) for i in range(4)
        },
    }


def boundary_metrics(counts):
    rows = {}
    values = []
    for class_index, class_name in enumerate(CLASS_NAMES):
        row = counts[class_index]
        precision = row["matched_pred"] / row["pred"] if row["pred"] else math.nan
        recall = row["matched_gt"] / row["gt"] if row["gt"] else math.nan
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else math.nan
        rows[class_name] = {
            "precision": 100.0 * precision,
            "recall": 100.0 * recall,
            "f1": 100.0 * f1,
            **row,
        }
        values.append(f1)
    return {
        "tolerance_pixels": 1,
        "macro_f1": 100.0 * float(np.nanmean(values)),
        "per_class": rows,
    }


def load_rows(manifest, split):
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["new_split"] == split]
    rows.sort(key=lambda row: row["name"])
    return rows


def evaluate(name, config, checkpoint, rows, output_root):
    wrapper_args = argparse.Namespace(
        config=str(config), checkpoint=str(checkpoint), precision="fp16",
        active_block_indices=None,
    )
    wrapper, _ = build_wrapper(wrapper_args)
    confusion = np.zeros((4, 4), dtype=np.int64)
    boundary = [dict(pred=0, gt=0, matched_pred=0, matched_gt=0) for _ in range(4)]
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for index, row in enumerate(rows, 1):
        image_path = Path(row["image_path"])
        mask_path = Path(row["mask_path"])
        with Image.open(mask_path) as image:
            target = np.asarray(image).astype(np.int64, copy=False)
        rgb = torch.from_numpy(load_rgb_image(image_path, 512)).cuda()
        with torch.inference_mode():
            source_prediction = wrapper(rgb).argmax(dim=1)[0].cpu().numpy()
        prediction = SOURCE_TO_TARGET[source_prediction]
        valid = (target >= 0) & (target < 4)
        encoded = 4 * target[valid] + prediction[valid]
        confusion += np.bincount(encoded, minlength=16).reshape(4, 4)
        for class_index in range(4):
            predicted_edge = binary_boundary(prediction == class_index)
            target_edge = binary_boundary(target == class_index)
            boundary[class_index]["pred"] += int(predicted_edge.sum())
            boundary[class_index]["gt"] += int(target_edge.sum())
            boundary[class_index]["matched_pred"] += int(
                np.count_nonzero(predicted_edge & dilate_one(target_edge))
            )
            boundary[class_index]["matched_gt"] += int(
                np.count_nonzero(target_edge & dilate_one(predicted_edge))
            )
        if index == 1 or index % 100 == 0 or index == len(rows):
            print(f"{name}: evaluated {index}/{len(rows)}", flush=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    metrics = confusion_metrics(confusion)
    result = {
        "name": name,
        "config": str(config),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "evaluated_images": len(rows),
        "label_remap_source_to_l8": SOURCE_TO_TARGET.tolist(),
        "metrics": metrics,
        "boundary": boundary_metrics(boundary),
        "confusion": confusion.tolist(),
        "elapsed_seconds": elapsed,
        "images_per_second": len(rows) / elapsed,
        "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / (1024 ** 2),
    }
    output = output_root / name
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    del wrapper
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="work_dirs/phase45_protocol_audit/scene_disjoint_manifest.csv",
    )
    parser.add_argument("--audit-summary", default="work_dirs/phase45_protocol_audit/summary.json")
    parser.add_argument("--output-root", default="work_dirs/phase45_source_only")
    parser.add_argument("--expected-images", type=int, default=1905)
    parser.add_argument(
        "--expected-manifest-sha256",
        default="527fd1ecc81e0e088729626d6d2d1b2dc23fdd2bf901a33537c91822c8dcd533",
    )
    args = parser.parse_args()
    manifest = Path(args.manifest)
    audit = json.loads(Path(args.audit_summary).read_text(encoding="utf-8"))
    if audit.get("passed") is not True:
        raise RuntimeError("Phase 45A protocol audit did not pass")
    if audit.get("manifest_sha256") != args.expected_manifest_sha256:
        raise RuntimeError("Phase 45A manifest hash changed")
    rows = load_rows(manifest, "target_val")
    if len(rows) != args.expected_images:
        raise RuntimeError(f"Expected {args.expected_images} target-val images, found {len(rows)}")

    models = (
        (
            "v8",
            Path("configs/protocol/phase22_clean_v8_l1c.py"),
            Path("work_dirs/phase22_clean_v8/seed42/best_mIoU_iter_40000.pth"),
        ),
        (
            "phase37_compact",
            Path("configs/protocol/phase37_mobilenetv2_litefpn_weakclass_l1c.py"),
            Path("work_dirs/phase37_weakclass_supervision/seed42/best_mIoU_iter_38000.pth"),
        ),
    )
    output_root = Path(args.output_root)
    results = [evaluate(name, config, checkpoint, rows, output_root)
               for name, config, checkpoint in models]
    finite_values = [
        value
        for row in results
        for value in (row["metrics"]["mIoU"], row["metrics"]["weak_mIoU"],
                      row["boundary"]["macro_f1"])
    ]
    summary = {
        "phase": "45B-source-only",
        "selection_split": "scene_disjoint_target_val",
        "target_test_evaluated": False,
        "cloudsen_internal_test_evaluated": False,
        "manifest_sha256": audit["manifest_sha256"],
        "models": {row["name"]: row for row in results},
        "gates": {
            "phase45a_passed": True,
            "expected_target_val_images": all(row["evaluated_images"] == args.expected_images
                                               for row in results),
            "metrics_finite": all(math.isfinite(value) for value in finite_values),
            "target_test_sealed": True,
            "cloudsen_internal_test_sealed": True,
        },
    }
    summary["passed"] = all(summary["gates"].values())
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
