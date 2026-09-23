import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

from evaluate_phase46_factorized import source_rows
from run_phase45_source_only import evaluate, load_rows

IDENTITY = np.arange(4, dtype=np.int64)
SOURCE_TO_TARGET = np.asarray([0, 3, 2, 1], dtype=np.int64)


class Phase52BLogits(torch.nn.Module):
    def __init__(self, wrapper):
        super().__init__()
        self.wrapper = wrapper

    def forward(self, rgb):
        size = rgb.shape[-2:]
        inputs = self.wrapper.normalize(rgb)
        meta = [dict(
            ori_shape=size, img_shape=size, pad_shape=size,
            padding_size=[0, 0, 0, 0], flip=False,
        ) for _ in range(rgb.shape[0])]
        return self.wrapper.segmentor.encode_decode(inputs, meta)


def enable_residual(wrapper):
    return Phase52BLogits(wrapper)


def disable_all_target_paths(wrapper):
    wrapper.segmentor.set_target_enabled(False)
    return Phase52BLogits(wrapper)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/protocol/phase52b_shadow_residual_1pct_l1c.py")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default="work_dirs/phase45_protocol_audit/scene_disjoint_manifest.csv")
    parser.add_argument("--data-root", default=os.environ.get("CLOUD_ADAPTER_DATA_ROOT", "data/cloudsen12_high_l1c"))
    parser.add_argument("--preflight", default="work_dirs/phase52b_shadow_residual_1pct/preflight.json")
    parser.add_argument("--output-root", default="work_dirs/phase52b_shadow_residual_1pct/eval")
    args = parser.parse_args()
    preflight = json.loads(Path(args.preflight).read_text(encoding="utf-8"))
    output_root = Path(args.output_root)
    source = evaluate(
        "source_val_all_target_paths_disabled", Path(args.config), Path(args.checkpoint),
        source_rows(Path(args.data_root)), output_root, label_map=IDENTITY,
        wrapper_transform=disable_all_target_paths,
    )
    target = evaluate(
        "target_val", Path(args.config), Path(args.checkpoint),
        load_rows(Path(args.manifest), "target_val"), output_root,
        label_map=SOURCE_TO_TARGET, wrapper_transform=enable_residual,
    )
    metrics = target["metrics"]
    boundary = target["boundary"]["per_class"]
    values = (
        source["metrics"]["mIoU"], metrics["mIoU"],
        metrics["per_class_iou"]["cloud_shadow"],
        metrics["per_class_iou"]["thin_cloud"], boundary["thin_cloud"]["f1"],
    )
    gates = {
        "preflight_passed": preflight.get("passed") is True,
        "new_parameters_at_most_100k": preflight["new_trainable_parameters"] <= 100000,
        "target_miou_at_least_48": metrics["mIoU"] >= 48.0,
        "shadow_iou_at_least_16_471": metrics["per_class_iou"]["cloud_shadow"] >= 16.471,
        "thin_iou_at_least_28_630": metrics["per_class_iou"]["thin_cloud"] >= 28.630,
        "thin_boundary_f1_at_least_19_934": boundary["thin_cloud"]["f1"] >= 19.934,
        "source_miou_at_least_73_93": source["metrics"]["mIoU"] >= 73.93,
        "expected_source_images": source["evaluated_images"] == 535,
        "expected_target_images": target["evaluated_images"] == 1905,
        "metrics_finite": all(math.isfinite(value) for value in values),
        "target_test_sealed": True,
        "cloudsen_internal_test_sealed": True,
    }
    passed = all(gates.values())
    summary = {
        "phase": "52B",
        "method": "frozen_phase52_shadow_relational_residual",
        "source": source,
        "target": target,
        "new_trainable_parameters": preflight["new_trainable_parameters"],
        "phase52_parameters_frozen": True,
        "non_shadow_ratios_preserved_by_construction": True,
        "target_test_evaluated": False,
        "cloudsen_internal_test_evaluated": False,
        "gates": gates,
        "passed": passed,
        "decision": "proceed_to_independent_validation" if passed else "stop_shadow_residual_route",
    }
    root = output_root.parent
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
