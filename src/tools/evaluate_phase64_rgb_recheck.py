#!/usr/bin/env python3
"""Fair inference-only branch switching and paired-scene audit for Phase 64 RGB."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
import sys

os.environ.setdefault("XFORMERS_DISABLED", "1")

import numpy as np
import torch


TOOLS_ROOT = Path(__file__).resolve().parent
SRC_ROOT = TOOLS_ROOT.parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from evaluate_phase64_source_only_parent import (  # noqa: E402
    L8_TO_PARENT,
    SOURCE_TO_PARENT,
    SPARCS_TO_PARENT,
    l8_shadow_yes_records,
    load_parent_target,
    manifest_records,
    metrics_from_confusion,
    paired_directory_records,
    sha256_file,
)
from export_phase12_onnx import build_wrapper  # noqa: E402
from validate_phase12_onnxruntime import load_rgb_image  # noqa: E402


PARENT_CLASSES = ("surface_visible", "cloud", "shadow")
TARGETS = ("l8", "sparcs")
METHODS = ("msre", "lora")


def wrapper_args(config: str, checkpoint: Path) -> argparse.Namespace:
    return argparse.Namespace(
        config=config,
        checkpoint=str(checkpoint),
        precision="fp16",
        active_block_indices=None,
    )


def build(config: str, checkpoint: Path):
    wrapper, _ = build_wrapper(wrapper_args(config, checkpoint))
    return wrapper


def set_branch(wrapper, method: str, enabled: bool) -> None:
    backbone = wrapper.segmentor.backbone
    setter = {
        "msre": "set_target_enabled",
        "lora": "set_lora_enabled",
    }[method]
    if not hasattr(backbone, setter):
        raise RuntimeError(f"{type(backbone).__name__} has no {setter}")
    getattr(backbone, setter)(enabled)


def release(wrapper) -> None:
    wrapper.cpu()
    del wrapper
    gc.collect()
    torch.cuda.empty_cache()


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def normalized_parent_scores(wrapper, rgb: torch.Tensor, source_fine: bool) -> torch.Tensor:
    scores = wrapper(rgb).float().clamp_min(0)
    probabilities = scores / scores.sum(dim=1, keepdim=True).clamp_min(1e-12)
    if source_fine:
        if probabilities.shape[1] != 4:
            raise RuntimeError(f"Expected four source classes, got {probabilities.shape[1]}")
        probabilities = torch.stack(
            (
                probabilities[:, 0],
                probabilities[:, 1] + probabilities[:, 2],
                probabilities[:, 3],
            ),
            dim=1,
        )
    elif probabilities.shape[1] != 3:
        raise RuntimeError(f"Expected three parent classes, got {probabilities.shape[1]}")
    return probabilities


def evaluate_rows(
    wrapper,
    rows: list[dict[str, str]],
    mapping: np.ndarray,
    *,
    source_fine: bool,
    label: str,
    input_size: int = 512,
) -> dict[str, object]:
    total = np.zeros((3, 3), dtype=np.int64)
    per_scene: dict[str, np.ndarray] = defaultdict(
        lambda: np.zeros((3, 3), dtype=np.int64)
    )
    digest = hashlib.sha256()
    with torch.inference_mode():
        for index, row in enumerate(rows, start=1):
            rgb = torch.from_numpy(
                load_rgb_image(Path(row["image_path"]), input_size)
            ).cuda()
            probabilities = normalized_parent_scores(wrapper, rgb, source_fine)
            prediction = probabilities.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
            digest.update(prediction.tobytes(order="C"))
            target = load_parent_target(
                Path(row["mask_path"]), mapping, prediction.shape
            )
            valid = target != 255
            item = np.bincount(
                3 * target[valid] + prediction[valid], minlength=9
            ).reshape(3, 3)
            total += item
            per_scene[row["scene"]] += item
            if index == 1 or index % 100 == 0 or index == len(rows):
                print(f"{label}: {index}/{len(rows)}", flush=True)
    return {
        "metrics": metrics_from_confusion(total),
        "prediction_sha256": digest.hexdigest(),
        "per_scene_confusion": {
            scene: confusion.tolist() for scene, confusion in sorted(per_scene.items())
        },
        "images": len(rows),
        "scenes": len(per_scene),
    }


def compare_runtime_state(source_wrapper, adapted_wrapper) -> dict[str, object]:
    source_model = source_wrapper.segmentor
    adapted_model = adapted_wrapper.segmentor
    source_state = source_model.state_dict()
    adapted_state = adapted_model.state_dict()
    source_parameters = {name for name, _ in source_model.named_parameters()}
    source_buffers = {name for name, _ in source_model.named_buffers()}

    def compare(names: set[str]) -> dict[str, object]:
        missing = sorted(name for name in names if name not in adapted_state)
        mismatched = []
        max_abs = 0.0
        for name in sorted(names - set(missing)):
            left = source_state[name]
            right = adapted_state[name]
            if left.shape != right.shape or left.dtype != right.dtype:
                mismatched.append(name)
                continue
            if not torch.equal(left, right):
                mismatched.append(name)
                if left.is_floating_point():
                    max_abs = max(
                        max_abs,
                        float((left.float() - right.float()).abs().max().item()),
                    )
        return {
            "source_tensor_count": len(names),
            "missing_count": len(missing),
            "missing": missing,
            "mismatch_count": len(mismatched),
            "mismatched": mismatched,
            "max_abs_difference": max_abs,
            "all_exact": not missing and not mismatched,
        }

    return {
        "parameters": compare(source_parameters),
        "buffers": compare(source_buffers),
        "adapted_extra_state_keys": sorted(set(adapted_state) - set(source_state)),
    }


def metric_delta(adapted: dict[str, object], baseline: dict[str, object]) -> dict[str, float]:
    return {
        "mIoU": float(adapted["mIoU"]) - float(baseline["mIoU"]),
        "shadow_iou": float(adapted["per_class_iou"]["shadow"])
        - float(baseline["per_class_iou"]["shadow"]),
    }


def interval(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "ci95_low": float(np.quantile(array, 0.025)),
        "ci95_high": float(np.quantile(array, 0.975)),
        "probability_positive": float(np.mean(array > 0)),
    }


def paired_statistics(
    baseline: dict[str, object],
    adapted: dict[str, object],
    *,
    draws: int,
    seed: int,
) -> dict[str, object]:
    base_scene = {
        name: np.asarray(value, dtype=np.int64)
        for name, value in baseline["per_scene_confusion"].items()
    }
    adapted_scene = {
        name: np.asarray(value, dtype=np.int64)
        for name, value in adapted["per_scene_confusion"].items()
    }
    if set(base_scene) != set(adapted_scene):
        raise RuntimeError("Paired evaluations do not contain the same scenes")
    scenes = sorted(base_scene)
    if len(scenes) < 2:
        raise RuntimeError("Paired scene statistics need at least two scenes")

    per_scene = []
    for scene in scenes:
        base_metric = metrics_from_confusion(base_scene[scene])
        adapted_metric = metrics_from_confusion(adapted_scene[scene])
        per_scene.append(
            {
                "scene": scene,
                "source_only": base_metric,
                "msre": adapted_metric,
                "delta": metric_delta(adapted_metric, base_metric),
            }
        )

    rng = np.random.default_rng(seed)
    bootstrap_miou = []
    bootstrap_shadow = []
    for _ in range(draws):
        sampled = rng.choice(scenes, size=len(scenes), replace=True)
        base_confusion = sum(
            (base_scene[scene] for scene in sampled),
            np.zeros((3, 3), dtype=np.int64),
        )
        adapted_confusion = sum(
            (adapted_scene[scene] for scene in sampled),
            np.zeros((3, 3), dtype=np.int64),
        )
        delta = metric_delta(
            metrics_from_confusion(adapted_confusion),
            metrics_from_confusion(base_confusion),
        )
        bootstrap_miou.append(delta["mIoU"])
        bootstrap_shadow.append(delta["shadow_iou"])

    leave_one_out = []
    zero = np.zeros((3, 3), dtype=np.int64)
    for omitted in scenes:
        kept = [scene for scene in scenes if scene != omitted]
        base_confusion = sum((base_scene[scene] for scene in kept), zero.copy())
        adapted_confusion = sum((adapted_scene[scene] for scene in kept), zero.copy())
        leave_one_out.append(
            {
                "omitted_scene": omitted,
                "delta": metric_delta(
                    metrics_from_confusion(adapted_confusion),
                    metrics_from_confusion(base_confusion),
                ),
            }
        )

    full_delta = metric_delta(adapted["metrics"], baseline["metrics"])
    loo_miou = [row["delta"]["mIoU"] for row in leave_one_out]
    loo_shadow = [row["delta"]["shadow_iou"] for row in leave_one_out]
    return {
        "unit": "scene",
        "scene_count": len(scenes),
        "bootstrap_draws": draws,
        "bootstrap_seed": seed,
        "full_delta": full_delta,
        "bootstrap_delta_mIoU": interval(bootstrap_miou),
        "bootstrap_delta_shadow_iou": interval(bootstrap_shadow),
        "per_scene": per_scene,
        "leave_one_scene_out": leave_one_out,
        "loo_summary": {
            "delta_mIoU_min": float(min(loo_miou)),
            "delta_mIoU_max": float(max(loo_miou)),
            "delta_shadow_iou_min": float(min(loo_shadow)),
            "delta_shadow_iou_max": float(max(loo_shadow)),
            "all_delta_mIoU_positive": all(value > 0 for value in loo_miou),
            "all_delta_shadow_iou_positive": all(value > 0 for value in loo_shadow),
        },
    }


def set_target_environment(target: str, method: str, source_parent: Path) -> None:
    os.environ["PHASE64_TARGET"] = target
    os.environ["PHASE64_METHOD"] = method
    os.environ["PHASE64_PARENT_CHECKPOINT"] = str(source_parent)


def source_switch(args) -> dict[str, object]:
    source_rows = paired_directory_records(args.cloudsen_root, "val")
    source_parent_wrapper = build(args.source_parent_config, args.source_parent)
    parent_baseline = evaluate_rows(
        source_parent_wrapper,
        source_rows,
        SOURCE_TO_PARENT,
        source_fine=False,
        label="source-parent",
    )

    legacy_wrapper = build(args.legacy_source_config, args.legacy_source)
    legacy_baseline = evaluate_rows(
        legacy_wrapper,
        source_rows,
        SOURCE_TO_PARENT,
        source_fine=True,
        label="legacy-four-class-source-only",
    )
    release(legacy_wrapper)

    conditions: dict[str, dict[str, object]] = {}
    for target in TARGETS:
        conditions[target] = {}
        for method in METHODS:
            checkpoint = (
                args.experiment_root
                / "phase64_target_parent"
                / target
                / method
                / "seed64"
                / args.checkpoint_names[target][method]
            )
            set_target_environment(target, method, args.source_parent)
            wrapper = build(args.target_config, checkpoint)
            state_audit = compare_runtime_state(source_parent_wrapper, wrapper)
            set_branch(wrapper, method, True)
            enabled = evaluate_rows(
                wrapper,
                source_rows,
                SOURCE_TO_PARENT,
                source_fine=False,
                label=f"{target}-{method}-enabled",
            )
            set_branch(wrapper, method, False)
            switched = evaluate_rows(
                wrapper,
                source_rows,
                SOURCE_TO_PARENT,
                source_fine=False,
                label=f"{target}-{method}-known-domain-switch",
            )
            conditions[target][method] = {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "shared_state_audit": state_audit,
                "adaptation_branch_enabled": enabled,
                "known_domain_switch_off": switched,
                "enabled_delta_vs_source_parent": metric_delta(
                    enabled["metrics"], parent_baseline["metrics"]
                ),
                "switch_delta_vs_source_parent": metric_delta(
                    switched["metrics"], parent_baseline["metrics"]
                ),
                "switch_prediction_exact": (
                    switched["prediction_sha256"]
                    == parent_baseline["prediction_sha256"]
                ),
            }
            release(wrapper)

    release(source_parent_wrapper)
    return {
        "task": "phase64-rgb-source-branch-switch",
        "input": "RGB",
        "evaluation_contract": {
            "split": "CloudSEN val",
            "images": len(source_rows),
            "input_size": 512,
            "precision": "fp16",
            "prediction_space": "three parents",
            "target_test_read": False,
        },
        "legacy_four_class_source_only": legacy_baseline,
        "source_parent_checkpoint": {
            "path": str(args.source_parent),
            "sha256": sha256_file(args.source_parent),
            "evaluation": parent_baseline,
        },
        "conditions": conditions,
        "explanation_of_previous_0_04": (
            "The previous comparison mixed a legacy four-class checkpoint with "
            "probability aggregation and a separately trained three-parent source "
            "checkpoint. This audit uses the latter as the exact switch-off reference."
        ),
    }


def target_rows(args, target: str) -> tuple[list[dict[str, str]], np.ndarray]:
    if target == "l8":
        return (
            l8_shadow_yes_records(
                args.l8_manifest, args.l8_shadow_metadata, "target_val"
            ),
            L8_TO_PARENT,
        )
    return manifest_records(args.sparcs_manifest, "target_val"), SPARCS_TO_PARENT


def paired_scene(args) -> dict[str, object]:
    legacy_wrapper = build(args.legacy_source_config, args.legacy_source)
    domains: dict[str, object] = {}
    selected_targets = (args.paired_target,) if args.paired_target else TARGETS
    for target in selected_targets:
        rows, mapping = target_rows(args, target)
        baseline = evaluate_rows(
            legacy_wrapper,
            rows,
            mapping,
            source_fine=True,
            label=f"{target}-source-only",
        )
        if args.paired_checkpoint:
            checkpoint = args.paired_checkpoint
        else:
            checkpoint = (
                args.experiment_root
                / "phase64_target_parent"
                / target
                / "msre"
                / "seed64"
                / args.checkpoint_names[target]["msre"]
            )
        set_target_environment(target, "msre", args.source_parent)
        wrapper = build(args.target_config, checkpoint)
        set_branch(wrapper, "msre", True)
        adapted = evaluate_rows(
            wrapper,
            rows,
            mapping,
            source_fine=False,
            label=f"{target}-msre",
        )
        statistics = paired_statistics(
            baseline, adapted, draws=args.bootstrap_draws, seed=args.bootstrap_seed
        )
        domains[target] = {
            "images": len(rows),
            "source_only": baseline,
            "msre": adapted,
            "paired_scene_statistics": statistics,
            "msre_checkpoint": str(checkpoint),
            "msre_checkpoint_sha256": sha256_file(checkpoint),
        }
        release(wrapper)
    release(legacy_wrapper)
    return {
        "task": "phase64-rgb-paired-scene-bootstrap",
        "input": "RGB",
        "seed": args.paired_seed,
        "target_test_read": False,
        "domains": domains,
    }


def checkpoint_names() -> dict[str, dict[str, str]]:
    return {
        "l8": {
            "msre": "best_mIoU_iter_500.pth",
            "lora": "best_mIoU_iter_3500.pth",
        },
        "sparcs": {
            "msre": "best_mIoU_iter_2500.pth",
            "lora": "best_mIoU_iter_3000.pth",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=("source-switch", "paired-scene"))
    parser.add_argument(
        "--legacy-source-config",
        default="configs/protocol/phase22_clean_v8_l1c.py",
    )
    parser.add_argument(
        "--legacy-source",
        type=Path,
        default=Path("work_dirs/phase22_clean_v8/seed42/best_mIoU_iter_40000.pth"),
    )
    parser.add_argument(
        "--source-parent-config",
        default="configs/protocol/phase64_source_parent_rgb.py",
    )
    parser.add_argument(
        "--source-parent",
        type=Path,
        default=Path("work_dirs/phase64_source_parent/seed64/best_mIoU_iter_4000.pth"),
    )
    parser.add_argument(
        "--target-config",
        default="configs/protocol/phase64_target_parent_rgb.py",
    )
    parser.add_argument("--experiment-root", type=Path, default=Path("work_dirs"))
    parser.add_argument("--cloudsen-root", type=Path, default=Path("data/cloudsen12_high_l1c"))
    parser.add_argument(
        "--l8-manifest",
        type=Path,
        default=Path("work_dirs/phase45_protocol_audit/scene_disjoint_manifest.csv"),
    )
    parser.add_argument(
        "--l8-shadow-metadata",
        type=Path,
        default=Path("research_plans/protocol_data/l8_biome_usgs_shadow_status.csv"),
    )
    parser.add_argument(
        "--sparcs-manifest",
        type=Path,
        default=Path("work_dirs/phase64_sparcs_protocol_verified_v2/scene_disjoint_manifest.csv"),
    )
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=6401)
    parser.add_argument("--paired-target", choices=TARGETS)
    parser.add_argument("--paired-seed", type=int, default=64)
    parser.add_argument("--paired-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.checkpoint_names = checkpoint_names()
    if args.bootstrap_draws < 1000:
        raise ValueError("At least 1000 bootstrap draws are required")
    if bool(args.paired_target) != bool(args.paired_checkpoint):
        raise ValueError(
            "--paired-target and --paired-checkpoint must be provided together"
        )

    result = source_switch(args) if args.task == "source-switch" else paired_scene(args)
    result = json_safe(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
