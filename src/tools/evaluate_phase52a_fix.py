import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np

from cloud_adapter.datasets.manifest_seg import load_usgs_shadow_status
from evaluate_phase46_factorized import source_rows
from evaluate_phase52_sparse_msre import disable_target
from run_phase45_source_only import evaluate


IDENTITY = np.arange(4, dtype=np.int64)
SOURCE_TO_TARGET = np.asarray([0, 3, 2, 1], dtype=np.int64)


def load_shadow_labelled_rows(manifest, metadata_path):
    metadata = load_usgs_shadow_status(metadata_path)
    with Path(manifest).open(newline="", encoding="utf-8") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["new_split"] == "target_val"
            and metadata[row["scene"]] == "yes"
        ]
    rows.sort(key=lambda row: row["name"])
    if len(rows) != 963 or len({row["scene"] for row in rows}) != 8:
        raise RuntimeError(
            f"Expected 963 patches from 8 Shadows?=yes validation scenes, "
            f"found {len(rows)} from {len({row['scene'] for row in rows})}"
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/protocol/phase52a_fix_shadow_status_1pct_l1c.py",
    )
    parser.add_argument(
        "--baseline-config", default="configs/protocol/phase22_clean_v8_l1c.py"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--baseline-checkpoint",
        default="work_dirs/phase22_clean_v8/seed42/best_mIoU_iter_40000.pth",
    )
    parser.add_argument(
        "--manifest",
        default="work_dirs/phase45_protocol_audit/scene_disjoint_manifest.csv",
    )
    parser.add_argument(
        "--metadata",
        default="research_plans/protocol_data/l8_biome_usgs_shadow_status.csv",
    )
    parser.add_argument(
        "--data-root",
        default=os.environ.get("CLOUD_ADAPTER_DATA_ROOT", "data/cloudsen12_high_l1c"),
    )
    parser.add_argument(
        "--preflight", default="work_dirs/phase52a_fix_shadow_status_1pct/preflight.json"
    )
    parser.add_argument(
        "--baseline-benchmark",
        default="work_dirs/phase52a_fix_shadow_status_1pct/baseline_benchmark.json",
    )
    parser.add_argument(
        "--candidate-benchmark",
        default="work_dirs/phase52a_fix_shadow_status_1pct/candidate_benchmark.json",
    )
    parser.add_argument(
        "--output-root", default="work_dirs/phase52a_fix_shadow_status_1pct/eval"
    )
    args = parser.parse_args()

    preflight = json.loads(Path(args.preflight).read_text(encoding="utf-8"))
    baseline_benchmark = json.loads(Path(args.baseline_benchmark).read_text(encoding="utf-8"))
    candidate_benchmark = json.loads(Path(args.candidate_benchmark).read_text(encoding="utf-8"))
    output_root = Path(args.output_root)
    target_rows = load_shadow_labelled_rows(args.manifest, args.metadata)
    source = evaluate(
        "source_val_disabled_target",
        Path(args.config),
        Path(args.checkpoint),
        source_rows(Path(args.data_root)),
        output_root,
        label_map=IDENTITY,
        wrapper_transform=disable_target,
    )
    baseline = evaluate(
        "target_val_usgs_shadow_yes_source_only",
        Path(args.baseline_config),
        Path(args.baseline_checkpoint),
        target_rows,
        output_root,
        label_map=SOURCE_TO_TARGET,
    )
    target = evaluate(
        "target_val_usgs_shadow_yes_candidate",
        Path(args.config),
        Path(args.checkpoint),
        target_rows,
        output_root,
        label_map=SOURCE_TO_TARGET,
    )
    target_gain = target["metrics"]["mIoU"] - baseline["metrics"]["mIoU"]
    weak_gain = target["metrics"]["weak_mIoU"] - baseline["metrics"]["weak_mIoU"]
    boundary_gain = target["boundary"]["macro_f1"] - baseline["boundary"]["macro_f1"]
    shadow_gain = (
        target["metrics"]["per_class_iou"]["cloud_shadow"]
        - baseline["metrics"]["per_class_iou"]["cloud_shadow"]
    )
    latency_overhead = (
        candidate_benchmark["latency_mean_ms"]
        / baseline_benchmark["latency_mean_ms"]
        - 1.0
    )
    values = (
        source["metrics"]["mIoU"],
        baseline["metrics"]["mIoU"],
        target["metrics"]["mIoU"],
        target["metrics"]["weak_mIoU"],
        target["boundary"]["macro_f1"],
        target_gain,
        weak_gain,
        boundary_gain,
        shadow_gain,
        latency_overhead,
    )
    gates = {
        "preflight_passed": preflight.get("passed") is True,
        "source_forgetting_at_most_0_05": source["metrics"]["mIoU"] >= 73.93,
        "corrected_target_gain_at_least_3_miou": target_gain >= 3.0,
        "corrected_weak_gain_at_least_4_iou": weak_gain >= 4.0,
        "corrected_boundary_gain_at_least_3_f1": boundary_gain >= 3.0,
        "corrected_shadow_iou_not_below_source_only": shadow_gain >= 0.0,
        "target_params_exactly_380577": preflight["target_trainable_parameters"] == 380577,
        "latency_overhead_at_most_20pct": latency_overhead <= 0.20,
        "expected_source_images": source["evaluated_images"] == 535,
        "expected_corrected_target_images": target["evaluated_images"] == 963,
        "metrics_finite": all(math.isfinite(value) for value in values),
        "target_test_sealed": True,
        "cloudsen_internal_test_sealed": True,
    }
    passed = all(gates.values())
    summary = {
        "phase": "52A-fix",
        "method": "phase52a_with_official_usgs_shadow-availability_protocol",
        "protocol": {
            "training_selection": "same frozen 65 Phase 50 patches",
            "shadows_no_training_rule": "target class 0 -> ignore; thin/thick retained",
            "validation_rule": "only USGS Shadows?=yes target-val scenes",
            "iterations": 4000,
        },
        "source": source,
        "corrected_source_only_baseline": baseline,
        "target": target,
        "gains": {
            "target_mIoU": target_gain,
            "target_weak_mIoU": weak_gain,
            "target_boundary_f1": boundary_gain,
            "target_shadow_iou": shadow_gain,
        },
        "target_trainable_parameters": preflight["target_trainable_parameters"],
        "benchmarks": {
            "baseline": baseline_benchmark,
            "candidate": candidate_benchmark,
            "latency_overhead_fraction": latency_overhead,
        },
        "target_test_evaluated": False,
        "cloudsen_internal_test_evaluated": False,
        "gates": gates,
        "passed": passed,
        "decision": (
            "phase52a_fix_passed" if passed else "stop_phase52a_fix_shadow_status_route"
        ),
    }
    root = output_root.parent
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
