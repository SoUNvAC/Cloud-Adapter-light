#!/usr/bin/env python3
"""Summarize the frozen Phase 64 quick-screen matrix and apply its stop gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
TARGETS = ("l8", "sparcs")
METHODS = ("shared_parent", "full", "lora", "msre")
PARENT_CLASSES = ("surface_visible", "cloud", "shadow")
TRAIN_ITER = re.compile(r"Iter\(train\)\s+\[\s*(\d+)/")
METRIC_VALUE = re.compile(
    r"([A-Za-z]+(?:/[a-z_]+)?):\s+"
    r"(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
)
LOSS_VALUE = re.compile(
    r"(?:^|\s)loss:\s+(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metric_values(line: str) -> dict[str, float]:
    return {name: float(value) for name, value in METRIC_VALUE.findall(line)}


def parse_training_log(path: Path) -> dict[str, object]:
    current_iter = None
    validations: list[dict[str, object]] = []
    losses: list[float] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        iteration = TRAIN_ITER.search(line)
        if iteration:
            current_iter = int(iteration.group(1))
        losses.extend(float(value) for value in LOSS_VALUE.findall(line))
        if "Iter(val)" not in line or "mIoU:" not in line:
            continue
        values = metric_values(line)
        if current_iter is None or not all(
            key in values
            for key in (
                "mIoU",
                "IoU/surface_visible",
                "IoU/cloud",
                "IoU/shadow",
            )
        ):
            raise RuntimeError(f"Incomplete validation metric line in {path}: {line}")
        validations.append(
            {
                "iteration": current_iter,
                "mIoU": values["mIoU"],
                "per_class_iou": {
                    name: values[f"IoU/{name}"] for name in PARENT_CLASSES
                },
            }
        )
    if not validations:
        raise RuntimeError(f"No complete validation records in {path}")
    # CheckpointHook uses a strict improvement rule, so retain the earliest
    # iteration when rounded logged mIoU values tie.
    best = max(validations, key=lambda row: (float(row["mIoU"]), -int(row["iteration"])))
    return {
        "best": best,
        "validation_count": len(validations),
        "loss_observations": len(losses),
        "losses_finite": bool(losses) and all(math.isfinite(value) for value in losses),
    }


def parse_evaluation_log(path: Path) -> dict[str, object]:
    records = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "Iter(test)" not in line or "mIoU:" not in line:
            continue
        values = metric_values(line)
        if all(
            key in values
            for key in (
                "mIoU",
                "IoU/surface_visible",
                "IoU/cloud",
                "IoU/shadow",
            )
        ):
            records.append(
                {
                    "mIoU": values["mIoU"],
                    "per_class_iou": {
                        name: values[f"IoU/{name}"] for name in PARENT_CLASSES
                    },
                }
            )
    if len(records) != 1:
        raise RuntimeError(f"Expected one complete test record in {path}, got {len(records)}")
    return records[0]


def read_exit_code(path: Path) -> int | None:
    if not path.is_file():
        return None
    return int(path.read_text(encoding="utf-8").strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-only",
        type=Path,
        default=SRC_ROOT / "work_dirs/phase64_source_only_parent/summary.json",
    )
    parser.add_argument(
        "--model-audit",
        type=Path,
        default=SRC_ROOT / "work_dirs/phase64_config_audit/model_summary.json",
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=SRC_ROOT / "work_dirs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SRC_ROOT / "work_dirs/phase64_final_summary/summary.json",
    )
    args = parser.parse_args()

    source_only = json.loads(args.source_only.read_text(encoding="utf-8"))
    model_audit = json.loads(args.model_audit.read_text(encoding="utf-8"))
    trainable = {
        row["method"]: int(row["trainable_parameters"])
        for row in model_audit["models"]
    }
    source_baseline = float(
        source_only["domains"]["cloudsen12_source_val"]["metrics"]["mIoU"]
    )
    target_baselines = {
        "l8": source_only["domains"]["l8_biome_target_val_shadows_yes"]["metrics"],
        "sparcs": source_only["domains"]["sparcs_target_val"]["metrics"],
    }

    rows: dict[str, dict[str, object]] = {}
    complete = True
    for target in TARGETS:
        rows[target] = {}
        for method in METHODS:
            train_dir = (
                args.experiment_root
                / "phase64_target_parent"
                / target
                / method
                / "seed64"
            )
            retention_dir = (
                args.experiment_root
                / "phase64_source_retention"
                / target
                / method
                / "seed64"
            )
            artifacts = {
                "train_complete": (train_dir / "TRAIN_COMPLETE").is_file(),
                "train_exit_code": read_exit_code(train_dir / "EXIT_CODE"),
                "retention_complete": (retention_dir / "EVALUATION_COMPLETE").is_file(),
                "retention_exit_code": read_exit_code(retention_dir / "EXIT_CODE"),
            }
            row_complete = (
                artifacts["train_complete"]
                and artifacts["train_exit_code"] == 0
                and artifacts["retention_complete"]
                and artifacts["retention_exit_code"] == 0
            )
            complete = complete and row_complete
            row: dict[str, object] = {
                "target": target,
                "method": method,
                "seed": 64,
                "trainable_parameters": trainable[method],
                "artifacts": artifacts,
                "complete": row_complete,
            }
            if row_complete:
                training = parse_training_log(train_dir / "phase64_console.log")
                best = training["best"]
                checkpoint = train_dir / f"best_mIoU_iter_{best['iteration']}.pth"
                if not checkpoint.is_file():
                    raise RuntimeError(f"Best checkpoint missing: {checkpoint}")
                retention = parse_evaluation_log(
                    retention_dir / "evaluation_console.log"
                )
                baseline = target_baselines[target]
                row.update(
                    {
                        "target_best": best,
                        "target_delta_mIoU": float(best["mIoU"])
                        - float(baseline["mIoU"]),
                        "target_shadow_delta": float(
                            best["per_class_iou"]["shadow"]
                        )
                        - float(baseline["per_class_iou"]["shadow"]),
                        "source_retention": retention,
                        "source_delta_mIoU": float(retention["mIoU"])
                        - source_baseline,
                        "checkpoint": checkpoint.relative_to(SRC_ROOT).as_posix(),
                        "checkpoint_sha256": sha256_file(checkpoint),
                        "validation_count": training["validation_count"],
                        "loss_observations": training["loss_observations"],
                        "losses_finite": training["losses_finite"],
                    }
                )
            rows[target][method] = row

    gates: dict[str, bool] = {"matrix_complete": complete}
    if complete:
        primary = [rows[target]["msre"] for target in TARGETS]
        gates.update(
            {
                "msre_both_targets_improve": all(
                    float(row["target_delta_mIoU"]) > 0 for row in primary
                ),
                "msre_average_target_gain_ge_2": sum(
                    float(row["target_delta_mIoU"]) for row in primary
                )
                / len(primary)
                >= 2.0,
                "msre_source_drop_le_1_each_target": all(
                    float(row["source_delta_mIoU"]) >= -1.0 for row in primary
                ),
                "msre_trainable_le_500k": trainable["msre"] <= 500_000,
                # Predefined operational meaning of "no catastrophic collapse":
                # retain at least 10 Shadow IoU and lose no more than 10 points
                # from the frozen source-only baseline on either target.
                "msre_shadow_not_catastrophic": all(
                    float(row["target_best"]["per_class_iou"]["shadow"]) >= 10.0
                    and float(row["target_shadow_delta"]) >= -10.0
                    for row in primary
                ),
                "all_logged_losses_finite": all(
                    bool(rows[target][method]["losses_finite"])
                    for target in TARGETS
                    for method in METHODS
                ),
                "target_test_sealed": True,
            }
        )

    passed = complete and all(gates.values())
    summary = {
        "phase": "64-day3-stop-gate",
        "input": "RGB",
        "seed": 64,
        "source_only_checkpoint_sha256": source_only["checkpoint_sha256"],
        "source_only": {
            "source_mIoU": source_baseline,
            "target_metrics": target_baselines,
        },
        "methods": rows,
        "primary_method": "msre",
        "gates": gates,
        "passed": passed,
        "decision": (
            "continue_to_phase64b_common_six_band"
            if passed
            else ("wait_for_matrix" if not complete else "stop_current_data_backbone_combination")
        ),
        "limitations": [
            "single-seed quick screen",
            "point-estimate stop gate; adapted-model scene bootstrap not run",
            "CUDA warn-only deterministic mode is not bitwise deterministic",
        ],
    }
    output = args.output.resolve()
    output.relative_to(REPO_ROOT.resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not complete:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
