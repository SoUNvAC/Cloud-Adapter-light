import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

from evaluate_phase46_factorized import source_rows
from run_phase45_source_only import evaluate, load_rows

IDENTITY = np.arange(4, dtype=np.int64)
SOURCE_TO_TARGET = np.asarray([0, 3, 2, 1], dtype=np.int64)


def disable_target(wrapper):
    wrapper.segmentor.backbone.set_target_enabled(False)
    return wrapper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/protocol/phase52_sparse_msre_1pct_l1c.py")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default="work_dirs/phase45_protocol_audit/scene_disjoint_manifest.csv")
    parser.add_argument("--data-root", default=os.environ.get("CLOUD_ADAPTER_DATA_ROOT", "data/cloudsen12_high_l1c"))
    parser.add_argument("--preflight", default="work_dirs/phase52_sparse_msre_1pct/preflight.json")
    parser.add_argument("--baseline-benchmark", default="work_dirs/phase52_sparse_msre_1pct/baseline_benchmark.json")
    parser.add_argument("--candidate-benchmark", default="work_dirs/phase52_sparse_msre_1pct/candidate_benchmark.json")
    parser.add_argument("--output-root", default="work_dirs/phase52_sparse_msre_1pct/eval")
    args = parser.parse_args()

    preflight = json.loads(Path(args.preflight).read_text(encoding="utf-8"))
    baseline_benchmark = json.loads(Path(args.baseline_benchmark).read_text(encoding="utf-8"))
    candidate_benchmark = json.loads(Path(args.candidate_benchmark).read_text(encoding="utf-8"))
    output_root = Path(args.output_root)
    source = evaluate(
        "source_val_disabled_target", Path(args.config), Path(args.checkpoint),
        source_rows(Path(args.data_root)), output_root, label_map=IDENTITY,
        wrapper_transform=disable_target,
    )
    target = evaluate(
        "target_val", Path(args.config), Path(args.checkpoint),
        load_rows(Path(args.manifest), "target_val"), output_root,
        label_map=SOURCE_TO_TARGET,
    )
    target_gain = target["metrics"]["mIoU"] - 43.1241
    weak_gain = target["metrics"]["weak_mIoU"] - 15.3059
    boundary_gain = target["boundary"]["macro_f1"] - 13.4763
    latency_overhead = (
        candidate_benchmark["latency_mean_ms"] / baseline_benchmark["latency_mean_ms"] - 1.0
    )
    values = (source["metrics"]["mIoU"], target["metrics"]["mIoU"],
              target["metrics"]["weak_mIoU"], target["boundary"]["macro_f1"],
              latency_overhead)
    gates = {
        "preflight_passed": preflight.get("passed") is True,
        "source_forgetting_at_most_0_05": source["metrics"]["mIoU"] >= 73.93,
        "target_gain_at_least_3_miou": target_gain >= 3.0,
        "weak_gain_at_least_4_iou": weak_gain >= 4.0,
        "boundary_gain_at_least_3_f1": boundary_gain >= 3.0,
        "target_params_at_most_500k": preflight["target_trainable_parameters"] <= 500000,
        "latency_overhead_at_most_20pct": latency_overhead <= 0.20,
        "expected_source_images": source["evaluated_images"] == 535,
        "expected_target_images": target["evaluated_images"] == 1905,
        "metrics_finite": all(math.isfinite(value) for value in values),
        "target_test_sealed": True,
        "cloudsen_internal_test_sealed": True,
    }
    passed = all(gates.values())
    summary = {
        "phase": "52A",
        "method": "frozen-source_sparse-vanilla-msre16_target-head-delta",
        "source": source,
        "target": target,
        "gains": {"target_mIoU": target_gain, "target_weak_mIoU": weak_gain,
                  "target_boundary_f1": boundary_gain},
        "target_trainable_parameters": preflight["target_trainable_parameters"],
        "benchmarks": {"baseline": baseline_benchmark, "candidate": candidate_benchmark,
                       "latency_overhead_fraction": latency_overhead},
        "target_test_evaluated": False,
        "cloudsen_internal_test_evaluated": False,
        "gates": gates,
        "passed": passed,
        "decision": "proceed_to_phase52b_ablation" if passed else "stop_phase52_msre_route",
    }
    root = output_root.parent
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
