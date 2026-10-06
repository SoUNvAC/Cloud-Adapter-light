#!/usr/bin/env python3
"""Describe frozen L8 scene-wise RGB MsRE changes without excluding scenes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from evaluate_phase64_source_only_parent import metrics_from_confusion


SEEDS = (64, 65, 66)


def load_domain(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))["domains"]["l8"]


def scene_metrics(confusion: np.ndarray) -> dict[str, float]:
    return metrics_from_confusion(confusion)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path("work_dirs/phase64_rgb_recheck")
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("work_dirs/phase45_protocol_audit/scene_disjoint_manifest.csv"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("work_dirs/phase64_rgb_recheck/l8_scene_analysis.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("work_dirs/phase64_rgb_recheck/l8_scene_analysis.csv"),
    )
    args = parser.parse_args()

    parent = json.loads(
        (args.root / "three_parent_target_baseline.json").read_text(encoding="utf-8")
    )["domains"]["l8"]
    baseline = {
        scene: np.asarray(value, dtype=np.int64)
        for scene, value in parent["per_scene_confusion"].items()
    }
    adapted_paths = {
        64: args.root / "paired_scene.json",
        65: args.root / "paired_scene_l8_seed65.json",
        66: args.root / "paired_scene_l8_seed66.json",
    }
    adapted = {
        seed: {
            scene: np.asarray(value, dtype=np.int64)
            for scene, value in load_domain(path)["msre"][
                "per_scene_confusion"
            ].items()
        }
        for seed, path in adapted_paths.items()
    }
    if any(set(rows) != set(baseline) for rows in adapted.values()):
        raise RuntimeError("Seed and source-parent scene sets differ")

    with args.manifest.open(newline="", encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle))
    metadata = {}
    for scene in baseline:
        rows = [
            row
            for row in manifest
            if row["new_split"] == "target_val" and row["scene"] == scene
        ]
        if not rows:
            raise RuntimeError(f"Missing manifest rows for {scene}")
        metadata[scene] = {"biome": rows[0]["biome"], "patches": len(rows)}

    records = []
    for scene in sorted(baseline):
        source_confusion = baseline[scene]
        source_metric = scene_metrics(source_confusion)
        gt = source_confusion.sum(axis=1)
        valid = int(gt.sum())
        shadow_pixels = int(gt[2])
        if shadow_pixels <= 0:
            raise RuntimeError(f"No shadow pixels in audited scene {scene}")
        per_seed = []
        for seed in SEEDS:
            adapted_confusion = adapted[seed][scene]
            if not np.array_equal(adapted_confusion.sum(axis=1), gt):
                raise RuntimeError(f"GT marginals changed for {scene} seed {seed}")
            metric = scene_metrics(adapted_confusion)
            per_seed.append(
                {
                    "seed": seed,
                    "delta_mIoU": float(metric["mIoU"] - source_metric["mIoU"]),
                    "delta_shadow_iou": float(
                        metric["per_class_iou"]["shadow"]
                        - source_metric["per_class_iou"]["shadow"]
                    ),
                    "delta_shadow_to_surface_pp": float(
                        100.0
                        * (adapted_confusion[2, 0] - source_confusion[2, 0])
                        / shadow_pixels
                    ),
                    "delta_shadow_to_cloud_pp": float(
                        100.0
                        * (adapted_confusion[2, 1] - source_confusion[2, 1])
                        / shadow_pixels
                    ),
                }
            )
        record = {
            "scene": scene,
            **metadata[scene],
            "valid_pixels": valid,
            "gt_surface_pct": 100.0 * float(gt[0]) / valid,
            "gt_cloud_pct": 100.0 * float(gt[1]) / valid,
            "gt_shadow_pct": 100.0 * float(gt[2]) / valid,
            "source_parent_mIoU": source_metric["mIoU"],
            "source_parent_shadow_iou": source_metric["per_class_iou"]["shadow"],
            "source_shadow_to_surface_pct": (
                100.0 * float(source_confusion[2, 0]) / shadow_pixels
            ),
            "source_shadow_to_cloud_pct": (
                100.0 * float(source_confusion[2, 1]) / shadow_pixels
            ),
            "source_shadow_correct_pct": (
                100.0 * float(source_confusion[2, 2]) / shadow_pixels
            ),
            "mean_delta_mIoU": float(
                np.mean([row["delta_mIoU"] for row in per_seed])
            ),
            "sd_delta_mIoU": float(
                np.std([row["delta_mIoU"] for row in per_seed], ddof=1)
            ),
            "mean_delta_shadow_iou": float(
                np.mean([row["delta_shadow_iou"] for row in per_seed])
            ),
            "sd_delta_shadow_iou": float(
                np.std([row["delta_shadow_iou"] for row in per_seed], ddof=1)
            ),
            "mean_delta_shadow_to_surface_pp": float(
                np.mean([row["delta_shadow_to_surface_pp"] for row in per_seed])
            ),
            "mean_delta_shadow_to_cloud_pp": float(
                np.mean([row["delta_shadow_to_cloud_pp"] for row in per_seed])
            ),
            "mean_delta_shadow_correct_pp": float(
                -np.mean(
                    [
                        row["delta_shadow_to_surface_pp"]
                        + row["delta_shadow_to_cloud_pp"]
                        for row in per_seed
                    ]
                )
            ),
            "per_seed": per_seed,
        }
        records.append(record)

    result = {
        "phase": "64-rgb-l8-scene-description",
        "comparison": "three-parent source checkpoint vs MsRE from same checkpoint",
        "seeds": list(SEEDS),
        "scene_count": len(records),
        "target_test_read": False,
        "confusion_change_unit": (
            "percentage points of true-shadow pixels; negative means fewer errors"
        ),
        "records": records,
        "interpretation_guard": (
            "Hypothesis generation only. No scene is excluded and no post-hoc group "
            "is used as an estimand."
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    fields = [key for key in records[0] if key != "per_seed"]
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in records)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
