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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/protocol/phase50_active_1pct_v8_l1c.py")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default="work_dirs/phase45_protocol_audit/scene_disjoint_manifest.csv")
    parser.add_argument("--selection", default="work_dirs/phase50_active_1pct/selection.json")
    parser.add_argument("--data-root", default=os.environ.get("CLOUD_ADAPTER_DATA_ROOT", "data/cloudsen12_high_l1c"))
    parser.add_argument("--output-root", default="work_dirs/phase50_active_1pct/eval")
    args = parser.parse_args()
    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    output_root = Path(args.output_root)
    source = evaluate(
        "source_val", Path(args.config), Path(args.checkpoint),
        source_rows(Path(args.data_root)), output_root, label_map=IDENTITY,
    )
    target = evaluate(
        "target_val", Path(args.config), Path(args.checkpoint),
        load_rows(Path(args.manifest), "target_val"), output_root,
        label_map=SOURCE_TO_TARGET,
    )
    target_gain = target["metrics"]["mIoU"] - 43.1241
    weak_gain = target["metrics"]["weak_mIoU"] - 15.3059
    gates = {
        "exactly_65_selected_patches": selection["selected_images"] == 65,
        "annotation_fraction_at_most_1pct": selection["selection_fraction"] <= 0.01,
        "selection_did_not_read_labels": selection["target_labels_read_during_selection"] is False,
        "source_forgetting_at_most_1_miou": source["metrics"]["mIoU"] >= 72.98,
        "target_gain_at_least_5_miou": target_gain >= 5.0,
        "expected_source_images": source["evaluated_images"] == 535,
        "expected_target_images": target["evaluated_images"] == 1905,
        "metrics_finite": all(math.isfinite(value) for value in (source["metrics"]["mIoU"], target["metrics"]["mIoU"], target["metrics"]["weak_mIoU"], target["boundary"]["macro_f1"])),
        "target_test_evaluated": False,
        "cloudsen_internal_test_evaluated": False,
    }
    passed = all(gates.values())
    summary = {
        "phase": 50,
        "method": selection["method"],
        "annotation_budget": {"selected": 65, "available": 6502, "fraction": selection["selection_fraction"]},
        "source_baseline_mIoU": 73.98,
        "target_baseline_mIoU": 43.1241,
        "target_baseline_weak_mIoU": 15.3059,
        "source": source,
        "target": target,
        "gains": {"target_mIoU": target_gain, "target_weak_mIoU": weak_gain, "target_boundary_f1": target["boundary"]["macro_f1"] - 13.4763},
        "gates": gates,
        "passed": passed,
        "decision": "proceed_to_fewshot_active_da_validation" if passed else "stop_adaptation_route_begin_trustworthy_segmentation_audit",
    }
    root = output_root.parent
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
