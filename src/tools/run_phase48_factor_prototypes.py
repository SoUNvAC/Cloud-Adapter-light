import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("XFORMERS_DISABLED", "1")

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from export_phase12_onnx import build_wrapper
from run_phase45_source_only import (
    boundary_metrics,
    binary_boundary,
    confusion_metrics,
    dilate_one,
    load_rows,
)
from validate_phase12_onnxruntime import load_rgb_image


FACTOR_NAMES = ("shadow", "cloud_presence", "thick_given_cloud")
SOURCE_TO_TARGET = np.asarray([0, 3, 2, 1], dtype=np.int64)


def metas(batch, height=512, width=512):
    return [
        dict(
            ori_shape=(height, width), img_shape=(height, width),
            pad_shape=(height, width), padding_size=[0, 0, 0, 0], flip=False,
        )
        for _ in range(batch)
    ]


def load_model(config, checkpoint):
    args = argparse.Namespace(
        config=str(config), checkpoint=str(checkpoint), precision="fp16",
        active_block_indices=None,
    )
    deployment, _ = build_wrapper(args)
    return deployment, deployment.segmentor


def extract(deployment, model, rgb):
    normalized = deployment.normalize(rgb)
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.float16
    ):
        logits, features = model._base_outputs(normalized, metas(rgb.shape[0]))
        base = model._base_factor_logits(logits)
        base = F.interpolate(
            base, size=features.shape[-2:], mode="bilinear", align_corners=False
        )
        factors = base + model.factor_residual(features)
    return factors.float(), F.normalize(features.float(), dim=1)


def empty_accumulator(channels):
    return {
        name: {
            "negative": {"sum": torch.zeros(channels, device="cuda"), "count": 0},
            "positive": {"sum": torch.zeros(channels, device="cuda"), "count": 0},
        }
        for name in FACTOR_NAMES
    }


def accumulate(accumulator, name, state, features, mask):
    count = int(mask.sum().item())
    if count == 0:
        return
    row = accumulator[name][state]
    row["sum"] += (features * mask.unsqueeze(1)).sum(dim=(0, 2, 3))
    row["count"] += count


def source_prototypes(deployment, model, data_root):
    image_root = data_root / "img_dir" / "train"
    mask_root = data_root / "ann_dir" / "train"
    images = sorted(image_root.glob("*.png"))
    if len(images) != 8490:
        raise RuntimeError(f"Expected 8490 source images, found {len(images)}")
    accumulator = empty_accumulator(128)
    for index, image_path in enumerate(images, 1):
        mask_path = mask_root / image_path.name
        if not mask_path.is_file():
            raise FileNotFoundError(mask_path)
        rgb = torch.from_numpy(load_rgb_image(image_path, 512)).cuda()
        _, features = extract(deployment, model, rgb)
        with Image.open(mask_path) as image:
            labels = torch.from_numpy(np.asarray(image).copy()).cuda().long()
        labels = F.interpolate(
            labels[None, None].float(), size=features.shape[-2:], mode="nearest"
        )[:, 0].long()
        accumulate(accumulator, "shadow", "positive", features, labels == 3)
        accumulate(accumulator, "shadow", "negative", features, labels != 3)
        accumulate(
            accumulator, "cloud_presence", "positive", features,
            (labels == 1) | (labels == 2),
        )
        accumulate(accumulator, "cloud_presence", "negative", features, labels == 0)
        accumulate(accumulator, "thick_given_cloud", "positive", features, labels == 1)
        accumulate(accumulator, "thick_given_cloud", "negative", features, labels == 2)
        if index == 1 or index % 250 == 0 or index == len(images):
            print(f"source prototypes: {index}/{len(images)}", flush=True)
    return accumulator


def target_prototypes(deployment, model, rows):
    accumulator = empty_accumulator(128)
    for index, row in enumerate(rows, 1):
        rgb = torch.from_numpy(load_rgb_image(Path(row["image_path"]), 512)).cuda()
        factors_a, features_a = extract(deployment, model, rgb)
        factors_b, features_b = extract(deployment, model, rgb.flip(-1))
        factors_b = factors_b.flip(-1)
        features_b = features_b.flip(-1)
        probabilities_a = factors_a.sigmoid()
        probabilities_b = factors_b.sigmoid()
        probabilities = 0.5 * (probabilities_a + probabilities_b)
        features = F.normalize(0.5 * (features_a + features_b), dim=1)
        agreement = (probabilities_a >= 0.5) == (probabilities_b >= 0.5)
        for factor_index, name in enumerate(FACTOR_NAMES):
            values = probabilities[:, factor_index].flatten()
            low = torch.quantile(values, 0.10)
            high = torch.quantile(values, 0.90)
            consistent = agreement[:, factor_index]
            accumulate(
                accumulator, name, "positive", features,
                consistent & (probabilities[:, factor_index] >= high),
            )
            accumulate(
                accumulator, name, "negative", features,
                consistent & (probabilities[:, factor_index] <= low),
            )
        if index == 1 or index % 200 == 0 or index == len(rows):
            print(f"target prototypes: {index}/{len(rows)}", flush=True)
    return accumulator


def finalize(source, target, source_weight=0.75):
    result = {}
    counts = {"source": {}, "target": {}}
    for name in FACTOR_NAMES:
        result[name] = {}
        counts["source"][name] = {}
        counts["target"][name] = {}
        for state in ("negative", "positive"):
            source_row = source[name][state]
            target_row = target[name][state]
            if source_row["count"] <= 0 or target_row["count"] <= 0:
                raise RuntimeError(f"Empty prototype: {name}/{state}")
            source_proto = F.normalize(
                source_row["sum"] / source_row["count"], dim=0
            )
            target_proto = F.normalize(
                target_row["sum"] / target_row["count"], dim=0
            )
            result[name][state] = F.normalize(
                source_weight * source_proto + (1.0 - source_weight) * target_proto,
                dim=0,
            )
            counts["source"][name][state] = source_row["count"]
            counts["target"][name][state] = target_row["count"]
    return result, counts


def adapted_prediction(deployment, model, rgb, prototypes, temperature, weight):
    factors, features = extract(deployment, model, rgb)
    prototype_logits = []
    for name in FACTOR_NAMES:
        positive = prototypes[name]["positive"].view(1, -1, 1, 1)
        negative = prototypes[name]["negative"].view(1, -1, 1, 1)
        prototype_logits.append(
            ((features * positive).sum(1) - (features * negative).sum(1))
            / temperature
        )
    prototype_logits = torch.stack(prototype_logits, dim=1)
    adapted = factors + weight * prototype_logits
    adapted = F.interpolate(
        adapted, size=rgb.shape[-2:], mode="bilinear", align_corners=False
    )
    return model._reconstruct(adapted).argmax(dim=1)[0].cpu().numpy()


def evaluate_target(deployment, model, rows, prototypes, temperature, weight):
    confusion = np.zeros((4, 4), dtype=np.int64)
    boundary = [dict(pred=0, gt=0, matched_pred=0, matched_gt=0) for _ in range(4)]
    started = time.perf_counter()
    for index, row in enumerate(rows, 1):
        rgb = torch.from_numpy(load_rgb_image(Path(row["image_path"]), 512)).cuda()
        source_prediction = adapted_prediction(
            deployment, model, rgb, prototypes, temperature, weight
        )
        prediction = SOURCE_TO_TARGET[source_prediction]
        with Image.open(row["mask_path"]) as image:
            target = np.asarray(image).astype(np.int64, copy=False)
        valid = (target >= 0) & (target < 4)
        confusion += np.bincount(
            4 * target[valid] + prediction[valid], minlength=16
        ).reshape(4, 4)
        for class_index in range(4):
            predicted_edge = binary_boundary(prediction == class_index)
            target_edge = binary_boundary(target == class_index)
            boundary[class_index]["pred"] += int(predicted_edge.sum())
            boundary[class_index]["gt"] += int(target_edge.sum())
            boundary[class_index]["matched_pred"] += int(
                np.count_nonzero(predicted_edge & dilate_one(target_edge))
            )
            boundary[class_index]["matched_gt"] += int(
                np.count_nonzero(target_edge & dilate_one(predicted_edge))
            )
        if index == 1 or index % 100 == 0 or index == len(rows):
            print(f"target-val adapted: {index}/{len(rows)}", flush=True)
    return {
        "evaluated_images": len(rows),
        "metrics": confusion_metrics(confusion),
        "boundary": boundary_metrics(boundary),
        "confusion": confusion.tolist(),
        "elapsed_seconds": time.perf_counter() - started,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/protocol/phase47_pixel_factorized_v8_l1c.py"
    )
    parser.add_argument(
        "--checkpoint",
        default="work_dirs/phase47_pixel_factorized_v8/seed42/best_mIoU_iter_1000.pth",
    )
    parser.add_argument(
        "--manifest",
        default="work_dirs/phase45_protocol_audit/scene_disjoint_manifest.csv",
    )
    parser.add_argument(
        "--source-root",
        default=os.environ.get("CLOUD_ADAPTER_DATA_ROOT", "data/cloudsen12_high_l1c"),
    )
    parser.add_argument("--output", default="work_dirs/phase48_factor_prototypes/summary.json")
    args = parser.parse_args()
    target_train = load_rows(Path(args.manifest), "target_train")
    target_val = load_rows(Path(args.manifest), "target_val")
    if len(target_train) != 6502 or len(target_val) != 1905:
        raise RuntimeError("Unexpected Phase 45 manifest counts")
    deployment, model = load_model(Path(args.config), Path(args.checkpoint))
    source = source_prototypes(deployment, model, Path(args.source_root))
    target = target_prototypes(deployment, model, target_train)
    prototypes, counts = finalize(source, target, source_weight=0.75)
    result = evaluate_target(
        deployment, model, target_val, prototypes, temperature=0.1, weight=0.5
    )
    metrics = result["metrics"]
    boundary_f1 = result["boundary"]["macro_f1"]
    gates = {
        "target_mIoU_at_least_44_1241": metrics["mIoU"] >= 44.1241,
        "target_weak_mIoU_at_least_17_3059": metrics["weak_mIoU"] >= 17.3059,
        "target_boundary_f1_at_least_14_4763": boundary_f1 >= 14.4763,
        "prototype_counts_nonzero": all(
            count > 0 for domain in counts.values()
            for factor in domain.values() for count in factor.values()
        ),
        "expected_target_train_images": len(target_train) == 6502,
        "expected_target_val_images": result["evaluated_images"] == 1905,
        "target_train_labels_read": False,
        "target_test_evaluated": False,
        "cloudsen_internal_test_evaluated": False,
        "metrics_finite": all(
            math.isfinite(value)
            for value in (metrics["mIoU"], metrics["weak_mIoU"], boundary_f1)
        ),
    }
    summary = {
        "phase": 48,
        "method": "dual_view_factor_conditional_prototypes",
        "source_checkpoint": args.checkpoint,
        "source_val_mIoU_unchanged": 73.21792948901869,
        "source_forgetting_mIoU": 0.0,
        "target_train_images": len(target_train),
        "target_train_labels_read": False,
        "selection_quantiles": [0.10, 0.90],
        "source_prototype_weight": 0.75,
        "target_prototype_weight": 0.25,
        "prototype_temperature": 0.1,
        "prototype_logit_weight": 0.5,
        "prototype_counts": counts,
        "target": result,
        "gates": gates,
        "passed": all(gates.values()),
    }
    summary["decision"] = (
        "proceed_to_learned_class_conditional_adapter"
        if summary["passed"]
        else "close_fixed_prototype_inference"
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
