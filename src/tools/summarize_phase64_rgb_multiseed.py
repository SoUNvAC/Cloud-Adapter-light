#!/usr/bin/env python3
"""Aggregate paired RGB MsRE scene results over seeds 64/65/66."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_phase64_source_only_parent import metrics_from_confusion


TARGETS = ("l8", "sparcs")
SEEDS = (64, 65, 66)


def delta(adapted: np.ndarray, baseline: np.ndarray) -> tuple[float, float]:
    a = metrics_from_confusion(adapted)
    b = metrics_from_confusion(baseline)
    return (
        float(a["mIoU"]) - float(b["mIoU"]),
        float(a["per_class_iou"]["shadow"])
        - float(b["per_class_iou"]["shadow"]),
    )


def interval(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "ci95_low": float(np.quantile(array, 0.025)),
        "ci95_high": float(np.quantile(array, 0.975)),
        "probability_positive": float(np.mean(array > 0)),
    }


def load_domain(path: Path, target: str) -> dict:
    row = json.loads(path.read_text(encoding="utf-8"))
    if target not in row["domains"]:
        raise RuntimeError(f"{path} does not contain {target}")
    return row["domains"][target]


def aggregate(target: str, rows: dict[int, dict], draws: int, seed: int) -> dict:
    scenes = sorted(rows[64]["source_only"]["per_scene_confusion"])
    baseline = {
        scene: np.asarray(
            rows[64]["source_only"]["per_scene_confusion"][scene], dtype=np.int64
        )
        for scene in scenes
    }
    adapted = {}
    per_seed = []
    for repeat_seed in SEEDS:
        row = rows[repeat_seed]
        if sorted(row["source_only"]["per_scene_confusion"]) != scenes:
            raise RuntimeError(f"scene mismatch for {target} seed {repeat_seed}")
        adapted[repeat_seed] = {
            scene: np.asarray(
                row["msre"]["per_scene_confusion"][scene], dtype=np.int64
            )
            for scene in scenes
        }
        stats = row["paired_scene_statistics"]
        per_seed.append(
            {
                "seed": repeat_seed,
                "source_only": row["source_only"]["metrics"],
                "msre": row["msre"]["metrics"],
                "full_delta": stats["full_delta"],
                "scene_bootstrap_delta_mIoU": stats["bootstrap_delta_mIoU"],
                "scene_bootstrap_delta_shadow_iou": stats[
                    "bootstrap_delta_shadow_iou"
                ],
                "loo_summary": stats["loo_summary"],
                "checkpoint_sha256": row["msre_checkpoint_sha256"],
            }
        )

    rng = np.random.default_rng(seed)
    miou_draws = []
    shadow_draws = []
    for _ in range(draws):
        sampled_seeds = rng.choice(SEEDS, size=len(SEEDS), replace=True)
        sampled_scenes = rng.choice(scenes, size=len(scenes), replace=True)
        base_confusion = sum(
            (baseline[scene] for scene in sampled_scenes),
            np.zeros((3, 3), dtype=np.int64),
        )
        seed_deltas = []
        for repeat_seed in sampled_seeds:
            adapted_confusion = sum(
                (adapted[int(repeat_seed)][scene] for scene in sampled_scenes),
                np.zeros((3, 3), dtype=np.int64),
            )
            seed_deltas.append(delta(adapted_confusion, base_confusion))
        miou_draws.append(float(np.mean([value[0] for value in seed_deltas])))
        shadow_draws.append(float(np.mean([value[1] for value in seed_deltas])))

    leave_one_scene_out = []
    for omitted in scenes:
        kept = [scene for scene in scenes if scene != omitted]
        base_confusion = sum(
            (baseline[scene] for scene in kept),
            np.zeros((3, 3), dtype=np.int64),
        )
        values = []
        for repeat_seed in SEEDS:
            adapted_confusion = sum(
                (adapted[repeat_seed][scene] for scene in kept),
                np.zeros((3, 3), dtype=np.int64),
            )
            values.append(delta(adapted_confusion, base_confusion))
        leave_one_scene_out.append(
            {
                "omitted_scene": omitted,
                "mean_delta_mIoU": float(np.mean([value[0] for value in values])),
                "mean_delta_shadow_iou": float(
                    np.mean([value[1] for value in values])
                ),
            }
        )

    seed_miou = [row["full_delta"]["mIoU"] for row in per_seed]
    seed_shadow = [row["full_delta"]["shadow_iou"] for row in per_seed]
    loo_miou = [row["mean_delta_mIoU"] for row in leave_one_scene_out]
    loo_shadow = [row["mean_delta_shadow_iou"] for row in leave_one_scene_out]
    return {
        "target": target,
        "seeds": list(SEEDS),
        "scene_count": len(scenes),
        "per_seed": per_seed,
        "across_seed_point_summary": {
            "delta_mIoU_mean": float(np.mean(seed_miou)),
            "delta_mIoU_sample_std": float(np.std(seed_miou, ddof=1)),
            "delta_mIoU_min": float(min(seed_miou)),
            "delta_mIoU_max": float(max(seed_miou)),
            "delta_shadow_iou_mean": float(np.mean(seed_shadow)),
            "delta_shadow_iou_sample_std": float(np.std(seed_shadow, ddof=1)),
            "delta_shadow_iou_min": float(min(seed_shadow)),
            "delta_shadow_iou_max": float(max(seed_shadow)),
        },
        "hierarchical_seed_scene_bootstrap": {
            "draws": draws,
            "seed": seed,
            "delta_mIoU": interval(miou_draws),
            "delta_shadow_iou": interval(shadow_draws),
        },
        "leave_one_scene_out_across_seed_mean": leave_one_scene_out,
        "loo_summary": {
            "delta_mIoU_min": float(min(loo_miou)),
            "delta_mIoU_max": float(max(loo_miou)),
            "delta_shadow_iou_min": float(min(loo_shadow)),
            "delta_shadow_iou_max": float(max(loo_shadow)),
            "all_delta_mIoU_positive": all(value > 0 for value in loo_miou),
            "all_delta_shadow_iou_positive": all(value > 0 for value in loo_shadow),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path("work_dirs/phase64_rgb_recheck")
    )
    parser.add_argument("--draws", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=6466)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("work_dirs/phase64_rgb_recheck/multiseed_summary.json"),
    )
    args = parser.parse_args()
    if args.draws < 1000:
        raise ValueError("At least 1000 hierarchical bootstrap draws are required")

    rows: dict[str, dict[int, dict]] = {target: {} for target in TARGETS}
    seed64 = args.root / "paired_scene.json"
    for target in TARGETS:
        rows[target][64] = load_domain(seed64, target)
        for repeat_seed in (65, 66):
            rows[target][repeat_seed] = load_domain(
                args.root / f"paired_scene_{target}_seed{repeat_seed}.json", target
            )
    summary = {
        "phase": "64-rgb-msre-multiseed-recheck",
        "input": "RGB",
        "source_checkpoint_shared_across_target_seeds": True,
        "target_test_read": False,
        "domains": {
            target: aggregate(target, rows[target], args.draws, args.bootstrap_seed)
            for target in TARGETS
        },
        "limitations": [
            "three target-adaptation seeds share one source checkpoint",
            "CUDA warn-only deterministic mode is not bitwise deterministic",
            "SPARCS validation contains only ten scenes",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
