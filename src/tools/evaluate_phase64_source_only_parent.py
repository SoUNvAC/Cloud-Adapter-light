#!/usr/bin/env python3
"""Evaluate the frozen four-class source model in the Phase 64 parent space."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
import sys

os.environ.setdefault("XFORMERS_DISABLED", "1")

import cv2
import numpy as np
import torch
from PIL import Image


TOOLS_ROOT = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_ROOT.parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from export_phase12_onnx import build_wrapper  # noqa: E402
from validate_phase12_onnxruntime import load_rgb_image  # noqa: E402


PARENT_CLASSES = ("surface_visible", "cloud", "shadow")
SOURCE_TO_PARENT = np.asarray([0, 1, 1, 2], dtype=np.int64)
L8_TO_PARENT = np.asarray([0, 2, 1, 1], dtype=np.int64)
SPARCS_TO_PARENT = np.asarray([2, 2, 0, 0, 0, 1, 0], dtype=np.int64)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def paired_directory_records(root: Path, split: str) -> list[dict[str, str]]:
    image_dir = root / "img_dir" / split
    mask_dir = root / "ann_dir" / split
    images = {path.name: path for path in image_dir.iterdir() if path.is_file()}
    masks = {path.name: path for path in mask_dir.iterdir() if path.is_file()}
    if set(images) != set(masks):
        raise RuntimeError(f"Unpaired source files in {split}")
    return [
        {
            "name": name,
            "scene": name,
            "image_path": str(images[name]),
            "mask_path": str(masks[name]),
        }
        for name in sorted(images)
    ]


def manifest_records(path: Path, split: str) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["new_split"] == split]
    rows.sort(key=lambda row: (row["scene"], row["name"]))
    if not rows:
        raise RuntimeError(f"No {split!r} records in {path}")
    return rows


def l8_shadow_yes_records(
    manifest: Path, metadata: Path, split: str
) -> list[dict[str, str]]:
    with metadata.open(newline="", encoding="utf-8") as handle:
        status = {row["scene"]: row["usgs_shadows"].lower() for row in csv.DictReader(handle)}
    rows = manifest_records(manifest, split)
    missing = sorted({row["scene"] for row in rows if row["scene"] not in status})
    if missing:
        raise RuntimeError(f"Missing Shadows? metadata for scenes: {missing}")
    return [row for row in rows if status[row["scene"]] == "yes"]


def metrics_from_confusion(confusion: np.ndarray) -> dict[str, object]:
    values = confusion.astype(np.float64)
    tp = np.diag(values)
    union = values.sum(axis=0) + values.sum(axis=1) - tp
    iou = np.divide(tp, union, out=np.full(3, np.nan), where=union > 0)
    return {
        "mIoU": 100.0 * float(np.nanmean(iou)),
        "per_class_iou": {
            PARENT_CLASSES[index]: 100.0 * float(iou[index]) for index in range(3)
        },
        "confusion": confusion.tolist(),
        "valid_pixels": int(confusion.sum()),
    }


def bootstrap_scene_confusions(
    scene_confusions: dict[str, np.ndarray], draws: int, seed: int
) -> dict[str, object]:
    scenes = sorted(scene_confusions)
    if not scenes or draws <= 0:
        raise ValueError("bootstrap requires scenes and positive draws")
    rng = np.random.default_rng(seed)
    miou = []
    per_class = [[] for _ in range(3)]
    for _ in range(draws):
        sampled = rng.choice(scenes, size=len(scenes), replace=True)
        confusion = sum((scene_confusions[scene] for scene in sampled), np.zeros((3, 3), dtype=np.int64))
        metric = metrics_from_confusion(confusion)
        miou.append(float(metric["mIoU"]))
        for index, name in enumerate(PARENT_CLASSES):
            per_class[index].append(float(metric["per_class_iou"][name]))

    def interval(values: list[float]) -> dict[str, float]:
        finite = np.asarray([value for value in values if math.isfinite(value)])
        return {
            "median": float(np.median(finite)),
            "ci95_low": float(np.quantile(finite, 0.025)),
            "ci95_high": float(np.quantile(finite, 0.975)),
        }

    return {
        "unit": "scene",
        "scene_count": len(scenes),
        "draws": draws,
        "seed": seed,
        "mIoU": interval(miou),
        "per_class_iou": {
            name: interval(per_class[index]) for index, name in enumerate(PARENT_CLASSES)
        },
    }


def load_parent_target(mask_path: Path, mapping: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    # Pillow preserves class indices in palette-mode PNGs.  OpenCV expands
    # those masks to RGB/BGR palette colours and therefore destroys the
    # semantic IDs used by mmseg's LoadAnnotations transform.
    with Image.open(mask_path) as image:
        target = np.asarray(image)
    if target.ndim == 3:
        if target.shape[2] != 1:
            raise RuntimeError(f"Mask is not single-channel: {mask_path} shape={target.shape}")
        target = target[:, :, 0]
    if target.shape != size:
        target = cv2.resize(target, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
    target = target.astype(np.int64, copy=False)
    valid = (target >= 0) & (target < len(mapping))
    result = np.full(target.shape, 255, dtype=np.int64)
    result[valid] = mapping[target[valid]]
    return result


def evaluate_domain(
    wrapper,
    name: str,
    rows: list[dict[str, str]],
    target_mapping: np.ndarray,
    *,
    input_size: int,
    bootstrap_draws: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    confusion = np.zeros((3, 3), dtype=np.int64)
    per_scene: dict[str, np.ndarray] = defaultdict(lambda: np.zeros((3, 3), dtype=np.int64))
    score_sum_error_max = 0.0
    with torch.inference_mode():
        for index, row in enumerate(rows, start=1):
            rgb = torch.from_numpy(load_rgb_image(Path(row["image_path"]), input_size)).cuda()
            fine_scores = wrapper(rgb).float().clamp_min(0)
            fine_mass = fine_scores.sum(dim=1, keepdim=True).clamp_min(1e-12)
            fine_probs = fine_scores / fine_mass
            parent_probs = torch.stack(
                (
                    fine_probs[:, 0],
                    fine_probs[:, 1] + fine_probs[:, 2],
                    fine_probs[:, 3],
                ),
                dim=1,
            )
            score_sum_error_max = max(
                score_sum_error_max,
                float((parent_probs.sum(dim=1) - 1.0).abs().amax().item()),
            )
            prediction = parent_probs.argmax(dim=1)[0].cpu().numpy()
            target = load_parent_target(
                Path(row["mask_path"]), target_mapping, prediction.shape
            )
            valid = target != 255
            encoded = 3 * target[valid] + prediction[valid]
            item_confusion = np.bincount(encoded, minlength=9).reshape(3, 3)
            confusion += item_confusion
            per_scene[row["scene"]] += item_confusion
            if index == 1 or index % 100 == 0 or index == len(rows):
                print(f"{name}: {index}/{len(rows)}", flush=True)
    result = {
        "domain": name,
        "images": len(rows),
        "scenes": len(per_scene),
        "target_test_evaluated": False,
        "parent_probability_sum_error_max": score_sum_error_max,
        "metrics": metrics_from_confusion(confusion),
        "scene_bootstrap": bootstrap_scene_confusions(
            dict(per_scene), bootstrap_draws, bootstrap_seed
        ),
    }
    finite = [
        result["metrics"]["mIoU"],
        *result["metrics"]["per_class_iou"].values(),
    ]
    result["gates"] = {
        "records_present": bool(rows),
        "metrics_finite": all(math.isfinite(float(value)) for value in finite),
        "parent_probabilities_normalized": score_sum_error_max <= 1e-5,
        "target_test_sealed": True,
    }
    result["passed"] = all(result["gates"].values())
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/protocol/phase22_clean_v8_l1c.py")
    parser.add_argument(
        "--checkpoint",
        default="work_dirs/phase22_clean_v8/seed42/best_mIoU_iter_40000.pth",
    )
    parser.add_argument("--cloudsen-root", default="data/cloudsen12_high_l1c")
    parser.add_argument(
        "--l8-manifest", default="work_dirs/phase45_protocol_audit/scene_disjoint_manifest.csv"
    )
    parser.add_argument(
        "--l8-shadow-metadata",
        default="research_plans/protocol_data/l8_biome_usgs_shadow_status.csv",
    )
    parser.add_argument(
        "--sparcs-manifest",
        default=(
            "work_dirs/phase64_sparcs_protocol_verified_v2/"
            "scene_disjoint_manifest.csv"
        ),
    )
    parser.add_argument("--output", default="work_dirs/phase64_source_only_parent/summary.json")
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=64)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    wrapper_args = argparse.Namespace(
        config=args.config,
        checkpoint=str(checkpoint),
        precision="fp16",
        active_block_indices=None,
    )
    wrapper, _ = build_wrapper(wrapper_args)
    domains = (
        (
            "cloudsen12_source_val",
            paired_directory_records(Path(args.cloudsen_root), "val"),
            SOURCE_TO_PARENT,
        ),
        (
            "l8_biome_target_val_shadows_yes",
            l8_shadow_yes_records(
                Path(args.l8_manifest), Path(args.l8_shadow_metadata), "target_val"
            ),
            L8_TO_PARENT,
        ),
        (
            "sparcs_target_val",
            manifest_records(Path(args.sparcs_manifest), "target_val"),
            SPARCS_TO_PARENT,
        ),
    )
    results = {
        name: evaluate_domain(
            wrapper,
            name,
            rows,
            mapping,
            input_size=args.input_size,
            bootstrap_draws=args.bootstrap_draws,
            bootstrap_seed=args.bootstrap_seed,
        )
        for name, rows, mapping in domains
    }
    summary = {
        "phase": "64-day2-source-only-parent",
        "config": args.config,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "input": "RGB",
        "parent_classes": PARENT_CLASSES,
        "prediction_mapping": "probability aggregation: clear / (thick+thin) / shadow",
        "domains": results,
        "passed": all(row["passed"] for row in results.values()),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
