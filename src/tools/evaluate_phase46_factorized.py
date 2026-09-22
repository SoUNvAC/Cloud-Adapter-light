import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

from run_phase45_source_only import evaluate, load_rows


IDENTITY = np.arange(4, dtype=np.int64)
SOURCE_TO_TARGET = np.asarray([0, 3, 2, 1], dtype=np.int64)


def source_rows(data_root):
    image_root = data_root / "img_dir" / "val"
    mask_root = data_root / "ann_dir" / "val"
    rows = []
    for image_path in sorted(image_root.glob("*.png")):
        mask_path = mask_root / image_path.name
        if not mask_path.is_file():
            raise FileNotFoundError(mask_path)
        rows.append(
            {
                "name": image_path.stem,
                "image_path": str(image_path),
                "mask_path": str(mask_path),
            }
        )
    if len(rows) != 535:
        raise RuntimeError(f"Expected 535 source-val images, found {len(rows)}")
    return rows


def exact_reconstruction_error():
    generator = torch.Generator().manual_seed(46)
    logits = torch.randn(2, 4, 17, 19, generator=generator, dtype=torch.float64)
    clear, thick, thin, shadow = logits.unbind(dim=1)
    shadow_factor = shadow - torch.logsumexp(logits[:, :3], dim=1)
    cloud_factor = torch.logsumexp(torch.stack((thick, thin), dim=1), dim=1) - clear
    thick_factor = thick - thin
    p_shadow = shadow_factor.sigmoid()
    p_cloud = cloud_factor.sigmoid()
    p_thick = thick_factor.sigmoid()
    reconstructed = torch.stack(
        (
            (1 - p_shadow) * (1 - p_cloud),
            (1 - p_shadow) * p_cloud * p_thick,
            (1 - p_shadow) * p_cloud * (1 - p_thick),
            p_shadow,
        ),
        dim=1,
    )
    reference = logits.softmax(dim=1)
    return {
        "max_probability_error": float((reconstructed - reference).abs().max()),
        "max_probability_sum_error": float(
            (reconstructed.sum(dim=1) - 1.0).abs().max()
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--config", default="configs/protocol/phase46_factorized_v8_l1c.py"
    )
    parser.add_argument(
        "--manifest",
        default="work_dirs/phase45_protocol_audit/scene_disjoint_manifest.csv",
    )
    parser.add_argument(
        "--data-root",
        default=os.environ.get("CLOUD_ADAPTER_DATA_ROOT", "data/cloudsen12_high_l1c"),
    )
    parser.add_argument("--output-root", default="work_dirs/phase46_factorized_v8/eval")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    config = Path(args.config)
    output_root = Path(args.output_root)
    source = evaluate(
        "source_val", config, checkpoint, source_rows(Path(args.data_root)),
        output_root, label_map=IDENTITY,
    )
    target = evaluate(
        "target_val", config, checkpoint,
        load_rows(Path(args.manifest), "target_val"), output_root,
        label_map=SOURCE_TO_TARGET,
    )
    equivalence = exact_reconstruction_error()
    gates = {
        "source_val_miou_at_least_72_98": source["metrics"]["mIoU"] >= 72.98,
        "target_val_miou_at_least_43_1241": target["metrics"]["mIoU"] >= 43.1241,
        "target_weak_miou_at_least_16_3059": target["metrics"]["weak_mIoU"] >= 16.3059,
        "expected_source_images": source["evaluated_images"] == 535,
        "expected_target_images": target["evaluated_images"] == 1905,
        "zero_residual_exact": equivalence["max_probability_error"] < 1e-12,
        "probability_sum_exact": equivalence["max_probability_sum_error"] < 1e-12,
        "metrics_finite": all(
            math.isfinite(value)
            for value in (
                source["metrics"]["mIoU"], target["metrics"]["mIoU"],
                target["metrics"]["weak_mIoU"], target["boundary"]["macro_f1"],
            )
        ),
        "target_train_labels_read": False,
        "target_test_evaluated": False,
        "cloudsen_internal_test_evaluated": False,
    }
    summary = {
        "phase": 46,
        "method": "frozen_v8_factorized_logit_residual",
        "trainable_parameters": 15,
        "source_baseline_mIoU": 73.98,
        "target_baseline_mIoU": 43.1241,
        "target_baseline_weak_mIoU": 15.3059,
        "source": source,
        "target": target,
        "zero_residual_equivalence": equivalence,
        "gates": gates,
        "passed": all(gates.values()),
    }
    summary["decision"] = (
        "proceed_to_class_conditional_prototypes"
        if summary["passed"]
        else "move_factor_heads_to_pixel_features"
    )
    root = output_root.parent
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
