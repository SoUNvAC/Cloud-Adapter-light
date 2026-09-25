"""Evaluate and gate the three-group, three-seed Phase 62 experiment."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluate_phase46_factorized import source_rows  # noqa: E402
from evaluate_phase52a_fix import load_shadow_labelled_rows  # noqa: E402
from export_phase12_onnx import build_wrapper  # noqa: E402
from run_phase45_source_only import confusion_metrics, sha256  # noqa: E402
from validate_phase12_onnxruntime import load_rgb_image  # noqa: E402


GROUPS = {
    "complete12": "configs/protocol/phase62_complete12_l1c.py",
    "ignore65": "configs/protocol/phase62_ignore65_l1c.py",
    "partial65": "configs/protocol/phase62_partial65_l1c.py",
}
SEEDS = (62, 63, 64)
IDENTITY = np.arange(4, dtype=np.int64)
SOURCE_TO_TARGET = np.asarray([0, 3, 2, 1], dtype=np.int64)


def harmonic(thin, shadow):
    return 2.0 * thin * shadow / max(thin + shadow, 1e-12)


def add_derived(metrics):
    classes = metrics["per_class_iou"]
    metrics["thin_shadow_harmonic_mean"] = harmonic(
        classes["thin_cloud"], classes["cloud_shadow"]
    )
    metrics["non_clear_mIoU"] = float(np.mean([
        classes["cloud_shadow"], classes["thin_cloud"], classes["cloud"]
    ]))
    return metrics


def checkpoint_for(root, group, seed):
    paths = sorted((root / group / f"seed{seed}").glob("best_mIoU_iter_*.pth"))
    if len(paths) != 1:
        raise RuntimeError(f"Expected one best checkpoint for {group}/seed{seed}, found {paths}")
    return paths[0]


def evaluate_rows(
    wrapper, rows, label_map, per_scene=False, derive_target_metrics=True,
    name="evaluation",
):
    confusion = np.zeros((4, 4), dtype=np.int64)
    scene_confusions = {}
    started = time.perf_counter()
    for index, row in enumerate(rows, 1):
        with Image.open(row["mask_path"]) as image:
            target = np.asarray(image).astype(np.int64, copy=False)
        inputs = torch.from_numpy(load_rgb_image(Path(row["image_path"]), 512)).cuda()
        with torch.inference_mode():
            source_prediction = wrapper(inputs).argmax(dim=1)[0].cpu().numpy()
        prediction = label_map[source_prediction]
        valid = (target >= 0) & (target < 4)
        encoded = 4 * target[valid] + prediction[valid]
        item = np.bincount(encoded, minlength=16).reshape(4, 4)
        confusion += item
        if per_scene:
            scene_confusions.setdefault(row["scene"], np.zeros((4, 4), dtype=np.int64))
            scene_confusions[row["scene"]] += item
        if index == 1 or index % 200 == 0 or index == len(rows):
            print(f"{name}: {index}/{len(rows)}", flush=True)
    metrics = confusion_metrics(confusion)
    if derive_target_metrics:
        metrics = add_derived(metrics)
    result = {
        "evaluated_images": len(rows),
        "metrics": metrics,
        "confusion": confusion.tolist(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    if per_scene:
        result["scene_metrics"] = {
            scene: {
                "confusion": values.tolist(),
                "metrics": add_derived(confusion_metrics(values)),
            }
            for scene, values in sorted(scene_confusions.items())
        }
    return result


def evaluate_run(config, checkpoint, target_rows, source_validation, group, seed):
    wrapper_args = argparse.Namespace(
        config=str(config), checkpoint=str(checkpoint), precision="fp16",
        active_block_indices=None,
    )
    wrapper, _ = build_wrapper(wrapper_args)
    target = evaluate_rows(
        wrapper, target_rows, SOURCE_TO_TARGET, per_scene=True,
        name=f"{group}/seed{seed}/target",
    )
    wrapper.segmentor.backbone.set_target_enabled(False)
    source = evaluate_rows(
        wrapper, source_validation, IDENTITY, per_scene=False,
        derive_target_metrics=False,
        name=f"{group}/seed{seed}/source-disabled-target",
    )
    result = {
        "group": group,
        "seed": seed,
        "config": str(config),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "target": target,
        "source_disabled_target": source,
    }
    del wrapper
    torch.cuda.empty_cache()
    return result


def aggregate(rows):
    keys = (
        "mIoU", "weak_mIoU", "thin_shadow_harmonic_mean", "non_clear_mIoU"
    )
    result = {}
    for key in keys:
        values = np.asarray([row["target"]["metrics"][key] for row in rows])
        result[key] = {"mean": float(values.mean()), "std": float(values.std(ddof=1)), "values": values.tolist()}
    for class_name in ("clear", "cloud", "thin_cloud", "cloud_shadow"):
        values = np.asarray([
            row["target"]["metrics"]["per_class_iou"][class_name] for row in rows
        ])
        result[f"iou_{class_name}"] = {
            "mean": float(values.mean()), "std": float(values.std(ddof=1)), "values": values.tolist()
        }
    source_values = np.asarray([row["source_disabled_target"]["metrics"]["mIoU"] for row in rows])
    result["source_mIoU_disabled_target"] = {
        "mean": float(source_values.mean()), "std": float(source_values.std(ddof=1)),
        "values": source_values.tolist(), "minimum": float(source_values.min()),
    }
    return result


def paired_seed_delta(partial, baseline, metric, class_name=None):
    def value(row):
        if class_name is not None:
            return row["target"]["metrics"]["per_class_iou"][class_name]
        return row["target"]["metrics"][metric]
    differences = np.asarray([value(left) - value(right) for left, right in zip(partial, baseline)])
    return {"values": differences.tolist(), "mean": float(differences.mean())}


def hierarchical_scene_bootstrap(partial, baseline, draws=10000, seed=62):
    scenes = sorted(partial[0]["target"]["scene_metrics"])
    partial_values = np.asarray([
        [row["target"]["scene_metrics"][scene]["metrics"]["thin_shadow_harmonic_mean"] for scene in scenes]
        for row in partial
    ])
    baseline_values = np.asarray([
        [row["target"]["scene_metrics"][scene]["metrics"]["thin_shadow_harmonic_mean"] for scene in scenes]
        for row in baseline
    ])
    if not np.isfinite(partial_values).all() or not np.isfinite(baseline_values).all():
        raise RuntimeError("Non-finite scene-level weak harmonic metric")
    differences = partial_values - baseline_values
    generator = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        seed_indices = generator.integers(0, len(SEEDS), size=len(SEEDS))
        scene_indices = generator.integers(0, len(scenes), size=len(scenes))
        samples[index] = differences[np.ix_(seed_indices, scene_indices)].mean()
    scene_means = differences.mean(axis=0)
    return {
        "method": "paired hierarchical bootstrap over seeds and scenes",
        "draws": draws,
        "seed": seed,
        "scenes": scenes,
        "paired_difference_mean": float(differences.mean()),
        "paired_difference_95ci": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
        "probability_difference_positive": float(np.mean(samples > 0)),
        "per_scene_mean_difference": {scene: float(value) for scene, value in zip(scenes, scene_means)},
        "positive_scene_directions": int(np.count_nonzero(scene_means > 0)),
        "total_scenes": len(scenes),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="work_dirs/phase62_partial_labels")
    parser.add_argument("--manifest", default="work_dirs/phase45_protocol_audit/scene_disjoint_manifest.csv")
    parser.add_argument("--metadata", default="research_plans/protocol_data/l8_biome_usgs_shadow_status.csv")
    parser.add_argument("--data-root", default=os.environ.get("CLOUD_ADAPTER_DATA_ROOT", "data/cloudsen12_high_l1c"))
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    args = parser.parse_args()

    root = Path(args.root)
    preflight = json.loads((root / "preflight.json").read_text(encoding="utf-8"))
    if preflight.get("passed") is not True:
        raise RuntimeError("Phase 62 preflight did not pass")
    target_rows = load_shadow_labelled_rows(args.manifest, args.metadata)
    source_validation = source_rows(Path(args.data_root))
    results = {}
    for group, config in GROUPS.items():
        group_rows = []
        for seed in SEEDS:
            output = root / group / f"seed{seed}" / "evaluation.json"
            checkpoint = checkpoint_for(root, group, seed)
            row = evaluate_run(Path(config), checkpoint, target_rows, source_validation, group, seed)
            output.write_text(json.dumps(row, indent=2), encoding="utf-8")
            group_rows.append(row)
        results[group] = group_rows

    aggregates = {group: aggregate(rows) for group, rows in results.items()}
    partial, baseline = results["partial65"], results["ignore65"]
    deltas = {
        "thin_shadow_harmonic_mean": paired_seed_delta(partial, baseline, "thin_shadow_harmonic_mean"),
        "thin_iou": paired_seed_delta(partial, baseline, None, "thin_cloud"),
        "shadow_iou": paired_seed_delta(partial, baseline, None, "cloud_shadow"),
        "clear_iou": paired_seed_delta(partial, baseline, None, "clear"),
        "target_mIoU": paired_seed_delta(partial, baseline, "mIoU"),
        "non_clear_mIoU": paired_seed_delta(partial, baseline, "non_clear_mIoU"),
    }
    bootstrap = hierarchical_scene_bootstrap(
        partial, baseline, draws=args.bootstrap_draws, seed=62
    )
    gates = {
        "partial_improves_mean_thin_shadow_harmonic": deltas["thin_shadow_harmonic_mean"]["mean"] > 0,
        "no_thin_shadow_tradeoff_mean_thin_nonnegative": deltas["thin_iou"]["mean"] >= 0,
        "no_thin_shadow_tradeoff_mean_shadow_nonnegative": deltas["shadow_iou"]["mean"] >= 0,
        "scene_bootstrap_95ci_direction_positive": bootstrap["paired_difference_95ci"][0] > 0,
        "at_least_six_of_eight_scenes_improve": bootstrap["positive_scene_directions"] >= 6,
        "source_forgetting_within_existing_0_05_gate_all_partial_seeds": (
            aggregates["partial65"]["source_mIoU_disabled_target"]["minimum"] >= 73.93
        ),
        "non_clear_mean_improves": deltas["non_clear_mIoU"]["mean"] > 0,
        "not_clear_only": (
            deltas["thin_shadow_harmonic_mean"]["mean"] > 0
            and deltas["non_clear_mIoU"]["mean"] > 0
        ),
        "expected_three_seeds_per_group": all(len(rows) == 3 for rows in results.values()),
        "expected_target_images": all(
            row["target"]["evaluated_images"] == 963 for rows in results.values() for row in rows
        ),
        "expected_source_images": all(
            row["source_disabled_target"]["evaluated_images"] == 535 for rows in results.values() for row in rows
        ),
        "target_test_sealed": True,
        "cloudsen_internal_test_sealed": True,
    }
    passed = all(gates.values())
    summary = {
        "phase": 62,
        "protocol": {
            "groups": list(GROUPS), "seeds": list(SEEDS), "iterations": 4000,
            "partial_set": ["clear", "cloud_shadow"],
            "partial_loss": "-log(p_clear + p_shadow)",
            "validation": "target_val Shadows?=yes only",
            "source_forgetting_threshold_mIoU": 73.93,
        },
        "preflight": preflight,
        "runs": results,
        "aggregates": aggregates,
        "partial65_minus_ignore65": deltas,
        "scene_bootstrap": bootstrap,
        "gates": gates,
        "passed": passed,
        "decision": (
            "partial_label_baseline_supported_consider_label_transfer_matrix"
            if passed else "stop_complex_label_transfer_matrix"
        ),
        "target_test_evaluated": False,
        "cloudsen_internal_test_evaluated": False,
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "aggregates": aggregates,
        "partial65_minus_ignore65": deltas,
        "scene_bootstrap": bootstrap,
        "gates": gates,
        "passed": passed,
        "decision": summary["decision"],
    }, indent=2))


if __name__ == "__main__":
    main()
