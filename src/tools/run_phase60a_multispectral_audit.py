"""Phase 60A: fixed-readout multispectral information-sufficiency audit.

This deliberately reuses the frozen Phase 54 aligned spectral cache.  It never
reads the raw archive, target-test, or CloudSEN internal-test data.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = Path(__file__).resolve().parent
for item in (REPO_ROOT, TOOLS_ROOT):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from audit_phase56_readout import apply_ridge, binary_probe, fit_ridge  # noqa: E402
from phase56_protocol import (  # noqa: E402
    SOURCE_CLASS_NAMES,
    deterministic_samples,
    load_source_order_mask,
    read_shadow_yes_rows,
    sha256,
)
from run_phase55_mechanism_audit import binary_auc  # noqa: E402


CONDITIONS = {
    "S0_rgb": ("red", "green", "blue"),
    "S1_rgb_nir": ("red", "green", "blue", "nir"),
    "S2_rgb_nir_swir1": ("red", "green", "blue", "nir", "swir1"),
    "S3_rgb_nir_swir1_swir2": (
        "red", "green", "blue", "nir", "swir1", "swir2",
    ),
    "S4_common6_indices": (
        "red", "green", "blue", "nir", "swir1", "swir2",
        "ndvi", "ndmi", "ndsi", "nbr",
    ),
}
INDEX_FORMULAE = {
    "ndvi": "(nir-red)/(nir+red)",
    "ndmi": "(nir-swir1)/(nir+swir1)",
    "ndsi": "(green-swir1)/(green+swir1)",
    "nbr": "(nir-swir2)/(nir+swir2)",
}


def cache_path(cache_root: Path, split: str, row: dict) -> Path:
    return cache_root / split / f"{Path(row['name']).stem}.npz"


def resize_rgb(path: str, shape=(32, 32)) -> np.ndarray:
    with Image.open(path) as image:
        rgb = image.convert("RGB").resize((shape[1], shape[0]), Image.Resampling.BOX)
        return np.asarray(rgb, dtype=np.float32).transpose(2, 0, 1) / 255.0


def safe_index(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    denominator = left + right
    return np.divide(
        left - right, denominator,
        out=np.zeros_like(left, dtype=np.float32),
        where=np.abs(denominator) > 1e-6,
    )


def feature_bank(row: dict, cache: Path) -> dict[str, np.ndarray]:
    with np.load(cache) as item:
        auxiliary = item["auxiliary"][:3].astype(np.float32)
    rgb = resize_rgb(row["image_path"], auxiliary.shape[1:])
    red, green, blue = rgb
    nir, swir1, swir2 = auxiliary
    return {
        "red": red, "green": green, "blue": blue,
        "nir": nir, "swir1": swir1, "swir2": swir2,
        "ndvi": safe_index(nir, red),
        "ndmi": safe_index(nir, swir1),
        "ndsi": safe_index(green, swir1),
        "nbr": safe_index(nir, swir2),
    }


def stack_condition(bank: dict[str, np.ndarray], condition: str) -> np.ndarray:
    return np.stack([bank[name] for name in CONDITIONS[condition]], axis=0)


def sample_map(feature: np.ndarray, coordinates: np.ndarray, original_shape) -> np.ndarray:
    height, width = feature.shape[1:]
    yy = np.minimum(coordinates[:, 0] * height // original_shape[0], height - 1)
    xx = np.minimum(coordinates[:, 1] * width // original_shape[1], width - 1)
    return feature[:, yy, xx].T


def confusion_metrics(confusion: np.ndarray) -> dict:
    confusion = np.asarray(confusion, dtype=np.int64)
    true_positive = np.diag(confusion).astype(np.float64)
    ground_truth, predicted = confusion.sum(1), confusion.sum(0)
    union = ground_truth + predicted - true_positive
    iou = np.divide(true_positive, union, out=np.zeros(4), where=union > 0)
    class_iou = dict(zip(SOURCE_CLASS_NAMES, (100.0 * iou).tolist()))
    thin, shadow = class_iou["thin_cloud"], class_iou["cloud_shadow"]
    return {
        "mIoU": 100.0 * float(iou.mean()),
        "class_iou": class_iou,
        "thin_shadow_harmonic_mean": 2.0 * thin * shadow / max(thin + shadow, 1e-12),
        "confusion": confusion.tolist(),
    }


def bootstrap_scene(scene_confusions: dict[str, np.ndarray], seed=60, draws=2000) -> dict:
    names = sorted(scene_confusions)
    generator = np.random.default_rng(seed)
    values = defaultdict(list)
    for _ in range(draws):
        sampled = generator.choice(names, size=len(names), replace=True)
        metric = confusion_metrics(sum((scene_confusions[name] for name in sampled), np.zeros((4, 4), np.int64)))
        values["mIoU"].append(metric["mIoU"])
        values["thin_cloud_iou"].append(metric["class_iou"]["thin_cloud"])
        values["cloud_shadow_iou"].append(metric["class_iou"]["cloud_shadow"])
        values["weak_harmonic_mean"].append(metric["thin_shadow_harmonic_mean"])
    return {
        key: {
            "median": float(np.median(value)),
            "ci95": np.quantile(value, [0.025, 0.975]).tolist(),
        }
        for key, value in values.items()
    }


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="work_dirs/phase45_protocol_audit/scene_disjoint_manifest.csv")
    parser.add_argument("--metadata", default="research_plans/protocol_data/l8_biome_usgs_shadow_status.csv")
    parser.add_argument("--selection", default="work_dirs/phase50_active_1pct/selection.json")
    parser.add_argument("--cache-root", default="work_dirs/phase54_information_audit/cache")
    parser.add_argument("--output", default="work_dirs/phase60/phase60a_summary.json")
    parser.add_argument("--cap-per-class", type=int, default=32)
    parser.add_argument("--seed", type=int, default=60)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--ridge", type=float, default=0.01)
    args = parser.parse_args()

    train, validation, _ = read_shadow_yes_rows(args.manifest, args.metadata, args.selection)
    cache_root = Path(args.cache_root)
    absent = [
        str(cache_path(cache_root, split, row))
        for split, rows in (("train", train), ("val", validation))
        for row in rows if not cache_path(cache_root, split, row).is_file()
    ]
    if absent:
        raise FileNotFoundError(f"Missing {len(absent)} frozen Phase 54 caches; first={absent[0]}")

    train_x = {name: [] for name in CONDITIONS}
    train_y = []
    sample_ids = []
    for row in train:
        mask = load_source_order_mask(row)
        coordinates, labels = deterministic_samples(mask, row["name"], args.cap_per_class, args.seed)
        bank = feature_bank(row, cache_path(cache_root, "train", row))
        for condition in CONDITIONS:
            train_x[condition].append(sample_map(stack_condition(bank, condition), coordinates, mask.shape))
        train_y.append(labels)
        sample_ids.extend(f"{row['name']}:{int(y)}:{int(x)}" for y, x in coordinates)
    labels = np.concatenate(train_y)
    models = {
        condition: fit_ridge(np.concatenate(parts), labels, True, args.ridge)
        for condition, parts in train_x.items()
    }
    binary_models = {
        condition: binary_probe(np.concatenate(train_x[condition]), labels, 3, 2, args.ridge)
        for condition in CONDITIONS
    }

    results = {}
    for condition in CONDITIONS:
        scene_confusions, biome_confusions = {}, {}
        sampled_binary_labels, sampled_binary_scores = [], []
        sampled_confusion = np.zeros((4, 4), dtype=np.int64)
        for index, row in enumerate(validation, 1):
            mask = load_source_order_mask(row)
            bank = feature_bank(row, cache_path(cache_root, "val", row))
            feature = stack_condition(bank, condition)
            height, width = feature.shape[1:]
            flat_scores = apply_ridge(models[condition], feature.reshape(feature.shape[0], -1).T)
            low_prediction = flat_scores.argmax(1).reshape(height, width).astype(np.uint8)
            prediction = np.asarray(
                Image.fromarray(low_prediction).resize(
                    (mask.shape[1], mask.shape[0]), Image.Resampling.NEAREST
                )
            )
            confusion = np.bincount(4 * mask.reshape(-1) + prediction.reshape(-1), minlength=16).reshape(4, 4)
            scene_confusions.setdefault(row["scene"], np.zeros((4, 4), np.int64))
            scene_confusions[row["scene"]] += confusion
            biome_confusions.setdefault(row["biome"], np.zeros((4, 4), np.int64))
            biome_confusions[row["biome"]] += confusion

            coordinates, sampled_labels = deterministic_samples(
                mask, row["name"], args.cap_per_class, args.seed
            )
            sampled_features = sample_map(feature, coordinates, mask.shape)
            sampled_scores = apply_ridge(models[condition], sampled_features)
            sampled_prediction = sampled_scores.argmax(1)
            sampled_confusion += np.bincount(
                4 * sampled_labels + sampled_prediction, minlength=16
            ).reshape(4, 4)
            keep = (sampled_labels == 2) | (sampled_labels == 3)
            if keep.any():
                binary = binary_models[condition]
                standardized = (sampled_features[keep].astype(np.float64) - binary[0]) / binary[1]
                sampled_binary_scores.append(standardized @ binary[2][:-1] + binary[2][-1])
                sampled_binary_labels.append((sampled_labels[keep] == 3).astype(np.uint8))
            if index % 200 == 0 or index == len(validation):
                print(f"{condition}: {index}/{len(validation)}", flush=True)

        aggregate = sum(scene_confusions.values(), np.zeros((4, 4), np.int64))
        results[condition] = {
            "channels": list(CONDITIONS[condition]),
            "dense_32x32_readout_nearest_upsample": confusion_metrics(aggregate),
            "sampled_same_pixels": confusion_metrics(sampled_confusion),
            "shadow_vs_thin_AUROC_sampled": binary_auc(
                np.concatenate(sampled_binary_labels), np.concatenate(sampled_binary_scores)
            ),
            "scene_bootstrap": bootstrap_scene(scene_confusions, args.seed, args.bootstrap_draws),
            "by_biome": {
                biome: confusion_metrics(value) for biome, value in sorted(biome_confusions.items())
            },
        }

    baseline = results["S0_rgb"]["dense_32x32_readout_nearest_upsample"]
    common_gate_rows = {}
    for condition in ("S1_rgb_nir", "S2_rgb_nir_swir1", "S3_rgb_nir_swir1_swir2", "S4_common6_indices"):
        item = results[condition]["dense_32x32_readout_nearest_upsample"]
        common_gate_rows[condition] = {
            "thin_at_least_32_31": item["class_iou"]["thin_cloud"] >= 32.31,
            "shadow_at_least_20": item["class_iou"]["cloud_shadow"] >= 20.0,
            "mIoU_gain_vs_S0_at_least_2": item["mIoU"] >= baseline["mIoU"] + 2.0,
            "mIoU_gain_vs_S0": item["mIoU"] - baseline["mIoU"],
        }
        common_gate_rows[condition]["passed"] = all(
            value for key, value in common_gate_rows[condition].items() if key != "mIoU_gain_vs_S0"
        )

    summary = {
        "phase": "60A",
        "objective": "multispectral information sufficiency with a frozen balanced linear ridge readout",
        "protocol": {
            "target_train": "frozen Phase 50 selection intersected with USGS Shadows?=yes",
            "train_images": len(train),
            "validation": "target_val, USGS Shadows?=yes only",
            "validation_images": len(validation),
            "validation_scenes": len({row["scene"] for row in validation}),
            "pixel_sampling": f"deterministic maximum {args.cap_per_class} pixels/image/class, seed={args.seed}",
            "classifier": f"same class-balanced multiclass ridge readout, ridge={args.ridge}",
            "dense_rasterization": "predict on aligned 32x32 spectral cells, nearest-neighbor upsample to 512x512",
            "fixed_indices": INDEX_FORMULAE,
            "sample_identity_shared_across_S0_S4": True,
            "sample_id_sha256": __import__("hashlib").sha256("\n".join(sample_ids).encode()).hexdigest(),
            "manifest_sha256": sha256(args.manifest),
            "metadata_sha256": sha256(args.metadata),
            "selection_sha256": sha256(args.selection),
            "target_test_evaluated": False,
            "cloudsen_internal_test_evaluated": False,
        },
        "conditions": results,
        "S5_all_landsat_bands": {
            "status": "blocked_by_authorized_scope",
            "reason": "The authorized repository contains aligned RGB/NIR/SWIR1/SWIR2 cache only; the 11-band raw archive is outside /home/scv/Cloud-Adapter-light.",
            "fabricated_result": False,
        },
        "continue_gate_by_common_condition": common_gate_rows,
        "passed": any(row["passed"] for row in common_gate_rows.values()),
    }
    summary["decision"] = "continue_multispectral_common_band_line" if summary["passed"] else "stop_multispectral_line"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(output), "passed": summary["passed"], "gates": common_gate_rows}, indent=2))


if __name__ == "__main__":
    main()
