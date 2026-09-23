import argparse
import json
import math
import os
from pathlib import Path
import sys

os.environ.setdefault("XFORMERS_DISABLED", "1")

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluate_phase46_factorized import FactorizedInferenceWrapper, source_rows
from export_phase12_onnx import build_wrapper
from run_phase45_source_only import evaluate, load_rows
from validate_phase12_onnxruntime import load_rgb_image


def factor_occupancy(config, checkpoint, rows):
    args = argparse.Namespace(config=str(config), checkpoint=str(checkpoint),
                              precision="fp16", active_block_indices=None)
    deployment, _ = build_wrapper(args)
    model = deployment.segmentor
    total = torch.zeros(3, device="cuda", dtype=torch.float64)
    pixels = 0
    for index, row in enumerate(rows, 1):
        rgb = torch.from_numpy(load_rgb_image(Path(row["image_path"]), 512)).cuda()
        normalized = deployment.normalize(rgb)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            logits, features = model._base_outputs(normalized, [dict(
                ori_shape=(512, 512), img_shape=(512, 512), pad_shape=(512, 512),
                padding_size=[0, 0, 0, 0], flip=False)])
            base = model._base_factor_logits(logits)
            base = torch.nn.functional.interpolate(base, size=features.shape[-2:],
                                                    mode="bilinear", align_corners=False)
            probabilities = (base + model.factor_residual(features)).sigmoid()
        total += probabilities.double().sum((0, 2, 3))
        pixels += probabilities.shape[0] * probabilities.shape[2] * probabilities.shape[3]
        if index == 1 or index % 250 == 0 or index == len(rows):
            print(f"occupancy {checkpoint}: {index}/{len(rows)}", flush=True)
    return (total / pixels).cpu().tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/protocol/phase47_pixel_factorized_v8_l1c.py")
    parser.add_argument("--source-checkpoint", default="work_dirs/phase47_pixel_factorized_v8/seed42/best_mIoU_iter_1000.pth")
    parser.add_argument("--adapted-checkpoint", default="work_dirs/phase49_target_factor_adapter/final.pth")
    parser.add_argument("--manifest", default="work_dirs/phase45_protocol_audit/scene_disjoint_manifest.csv")
    parser.add_argument("--source-root", default=os.environ.get("CLOUD_ADAPTER_DATA_ROOT", "data/cloudsen12_high_l1c"))
    parser.add_argument("--output-root", default="work_dirs/phase49_target_factor_adapter/eval")
    args = parser.parse_args()
    config, checkpoint = Path(args.config), Path(args.adapted_checkpoint)
    target_train = load_rows(Path(args.manifest), "target_train")
    output_root = Path(args.output_root)
    source = evaluate("source_val", config, checkpoint, source_rows(Path(args.source_root)),
                      output_root, label_map=np.arange(4),
                      wrapper_transform=FactorizedInferenceWrapper)
    target = evaluate("target_val", config, checkpoint,
                      load_rows(Path(args.manifest), "target_val"), output_root,
                      label_map=np.asarray([0, 3, 2, 1]),
                      wrapper_transform=FactorizedInferenceWrapper)
    before = factor_occupancy(config, Path(args.source_checkpoint), target_train)
    after = factor_occupancy(config, checkpoint, target_train)
    drift = [abs(a - b) for a, b in zip(after, before)]
    gates = {
        "source_mIoU_at_least_72_98": source["metrics"]["mIoU"] >= 72.98,
        "target_mIoU_at_least_44_1241": target["metrics"]["mIoU"] >= 44.1241,
        "target_weak_mIoU_at_least_17_3059": target["metrics"]["weak_mIoU"] >= 17.3059,
        "target_boundary_f1_at_least_14_4763": target["boundary"]["macro_f1"] >= 14.4763,
        "factor_occupancy_drift_at_most_0_05": max(drift) <= 0.05,
        "target_train_labels_read": False, "target_test_evaluated": False,
        "cloudsen_internal_test_evaluated": False,
        "metrics_finite": all(math.isfinite(v) for v in (
            source["metrics"]["mIoU"], target["metrics"]["mIoU"],
            target["metrics"]["weak_mIoU"], target["boundary"]["macro_f1"])),
    }
    summary = {"phase": 49, "method": "source_anchored_target_factor_adapter",
               "source": source, "target": target,
               "factor_occupancy_before": before, "factor_occupancy_after": after,
               "factor_occupancy_absolute_drift": drift, "gates": gates,
               "passed": all(gates.values())}
    summary["decision"] = "proceed_to_multiseed" if summary["passed"] else "stop_adjacent_da_variants"
    root = output_root.parent
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
