import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

from evaluate_phase46_factorized import source_rows
from evaluate_phase52_sparse_msre import disable_target
from run_phase45_source_only import evaluate, load_rows

IDENTITY = np.arange(4, dtype=np.int64)
SOURCE_TO_TARGET = np.asarray([0, 3, 2, 1], dtype=np.int64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/protocol/phase53_shadow_aux_msre_1pct_l1c.py")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default="work_dirs/phase45_protocol_audit/scene_disjoint_manifest.csv")
    parser.add_argument("--data-root", default=os.environ.get("CLOUD_ADAPTER_DATA_ROOT", "data/cloudsen12_high_l1c"))
    parser.add_argument("--preflight", default="work_dirs/phase53_shadow_aux_msre_1pct/preflight.json")
    parser.add_argument("--output-root", default="work_dirs/phase53_shadow_aux_msre_1pct/eval")
    args = parser.parse_args()

    preflight = json.loads(Path(args.preflight).read_text(encoding="utf-8"))
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
    metrics = target["metrics"]
    boundary = target["boundary"]["per_class"]
    finite = (
        source["metrics"]["mIoU"], metrics["mIoU"], metrics["weak_mIoU"],
        metrics["per_class_iou"]["cloud_shadow"],
        metrics["per_class_iou"]["thin_cloud"], boundary["cloud_shadow"]["f1"],
    )
    gates = {
        "preflight_passed": preflight.get("passed") is True,
        "exactly_380577_target_params": preflight["target_trainable_parameters"] == 380577,
        "source_forgetting_at_most_0_05": source["metrics"]["mIoU"] >= 73.93,
        "target_miou_at_least_46_1241": metrics["mIoU"] >= 46.1241,
        "weak_miou_at_least_19_3059": metrics["weak_mIoU"] >= 19.3059,
        "shadow_iou_at_least_16_4714": metrics["per_class_iou"]["cloud_shadow"] >= 16.4714,
        "shadow_boundary_f1_at_least_10": boundary["cloud_shadow"]["f1"] >= 10.0,
        "thin_iou_at_least_27_6299": metrics["per_class_iou"]["thin_cloud"] >= 27.6299,
        "expected_source_images": source["evaluated_images"] == 535,
        "expected_target_images": target["evaluated_images"] == 1905,
        "metrics_finite": all(math.isfinite(value) for value in finite),
        "target_test_sealed": True,
        "cloudsen_internal_test_sealed": True,
    }
    passed = all(gates.values())
    summary = {
        "phase": 53,
        "method": "phase52_msre_plus_parameter_free_balanced_shadow_bce",
        "single_changed_mechanism": "balanced_cloud_shadow_binary_auxiliary_loss",
        "shadow_aux_weight": 0.5,
        "source": source,
        "target": target,
        "deltas_vs_source_only": {
            "mIoU": metrics["mIoU"] - 43.124083667668856,
            "weak_mIoU": metrics["weak_mIoU"] - 15.305855005022394,
            "shadow_iou": metrics["per_class_iou"]["cloud_shadow"] - 16.471411684230425,
            "thin_iou": metrics["per_class_iou"]["thin_cloud"] - 14.140298325814369,
            "shadow_boundary_f1": boundary["cloud_shadow"]["f1"] - 17.098992344345206,
            "thin_boundary_f1": boundary["thin_cloud"]["f1"] - 2.883804866927965,
        },
        "target_trainable_parameters": preflight["target_trainable_parameters"],
        "inference_graph_unchanged_from_phase52": True,
        "target_test_evaluated": False,
        "cloudsen_internal_test_evaluated": False,
        "gates": gates,
        "passed": passed,
        "decision": "proceed_to_independent_validation" if passed else "stop_shadow_aux_mechanism",
    }
    root = output_root.parent
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
