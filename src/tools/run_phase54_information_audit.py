import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("XFORMERS_DISABLED", "1")

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cloud_adapter.models.phase54_shadow_probe import (  # noqa: E402
    CONDITIONS,
    FixedShadowProbe,
    apply_shadow_residual,
    neighbouring_cloud_probability,
)
from cloud_adapter.models.segmentors.shadow_aux_encoder_decoder import (  # noqa: E402
    balanced_shadow_bce,
)
from cloud_adapter.models.segmentors.shadow_residual_encoder_decoder import (  # noqa: E402
    shadow_boundary_band,
    shadow_tversky,
)
from export_phase12_onnx import build_wrapper  # noqa: E402
from phase54_raw_data import (  # noqa: E402
    load_manifest,
    read_multispectral_patch,
    reconstruct_converter_rgb,
)
from validate_phase12_onnxruntime import load_rgb_image  # noqa: E402


TARGET_TO_SOURCE = np.asarray([0, 3, 2, 1], dtype=np.int64)
SOURCE_TO_TARGET = np.asarray([0, 3, 2, 1], dtype=np.int64)
TARGET_NAMES = ("clear", "cloud_shadow", "thin_cloud", "thick_cloud")
SEEDS = (42, 123, 3407)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_train_rows(manifest, selection):
    selected = {
        row["name"] for row in json.loads(Path(selection).read_text(encoding="utf-8"))["selected"]
    }
    rows = [
        row for row in load_manifest(manifest, "target_train")
        if row["name"] in selected
    ]
    rows.sort(key=lambda row: row["name"])
    if len(rows) != 65 or len(selected) != 65:
        raise RuntimeError(f"Expected the frozen 65 patches, found {len(rows)}")
    return rows


def cache_path(cache_root, split, row):
    return Path(cache_root) / split / f"{Path(row['name']).stem}.npz"


def normalized_probabilities(scores):
    scores = scores.float().clamp_min(1e-8)
    return scores / scores.sum(dim=1, keepdim=True).clamp_min(1e-8)


def create_cache(args):
    if not (Path(args.raw_root) / "DOWNLOAD_COMPLETE").is_file():
        raise RuntimeError("Raw Landsat download/extraction is incomplete")
    train_rows = selected_train_rows(args.manifest, args.selection)
    val_rows = load_manifest(args.manifest, "target_val")
    if len(val_rows) != 1905:
        raise RuntimeError(f"Expected 1,905 target-val patches, found {len(val_rows)}")
    wrapper_args = argparse.Namespace(
        config=args.config, checkpoint=args.checkpoint, precision="fp16",
        active_block_indices=None,
    )
    wrapper, _ = build_wrapper(wrapper_args)
    maximum_alignment_error = 0.0
    records = []
    for split, rows in (("train", train_rows), ("val", val_rows)):
        (Path(args.cache_root) / split).mkdir(parents=True, exist_ok=True)
        for index, row in enumerate(rows, 1):
            output = cache_path(args.cache_root, split, row)
            if output.is_file():
                records.append((split, row["name"], str(output)))
                continue
            raw, solar = read_multispectral_patch(args.raw_root, row)
            expected_rgb = np.asarray(Image.open(row["image_path"]), dtype=np.int16)
            rebuilt_rgb = reconstruct_converter_rgb(raw).astype(np.int16)
            alignment_error = float(np.abs(rebuilt_rgb - expected_rgb).mean())
            maximum_alignment_error = max(maximum_alignment_error, alignment_error)
            if alignment_error > 1.0:
                raise RuntimeError(f"Raw/RGB alignment failed for {row['name']}: {alignment_error}")
            rgb = torch.from_numpy(load_rgb_image(Path(row["image_path"]), 512)).cuda()
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=True
            ):
                normalized = wrapper.normalize(rgb)
                features = wrapper.segmentor.extract_feat(normalized)
                feature = features[args.feature_index].float()
                class_scores, mask_scores = wrapper.segmentor.decode_head(features, None)
                class_probabilities = F.softmax(class_scores[-1], dim=-1)[..., :-1]
                mask_probabilities = F.interpolate(
                    mask_scores[-1], size=rgb.shape[-2:], mode="bilinear",
                    align_corners=False,
                ).sigmoid()
                scores = torch.einsum(
                    "bqc,bqhw->bchw", class_probabilities, mask_probabilities
                )
                probabilities = normalized_probabilities(scores)
            feature_size = feature.shape[-2:]
            spectral = torch.from_numpy(raw[..., 3:6].transpose(2, 0, 1)).cuda()[None]
            spectral = F.interpolate(spectral, size=feature_size, mode="area")
            solar_tensor = torch.from_numpy(solar).cuda().view(1, 4, 1, 1)
            solar_tensor = solar_tensor.expand(1, 4, *feature_size)
            cloud = probabilities[:, 1:2] + probabilities[:, 2:3]
            cloud = F.interpolate(cloud, size=feature_size, mode="area")
            cloud_context = neighbouring_cloud_probability(cloud)
            auxiliary = torch.cat((spectral, solar_tensor, cloud_context), dim=1)
            with Image.open(row["mask_path"]) as image:
                target_label = np.asarray(image, dtype=np.uint8).copy()
            source_label = TARGET_TO_SOURCE[target_label]
            np.savez(
                output,
                feature=feature[0].half().cpu().numpy(),
                auxiliary=auxiliary[0].half().cpu().numpy(),
                probabilities=probabilities[0].half().cpu().numpy(),
                source_label=source_label,
            )
            records.append((split, row["name"], str(output)))
            if index == 1 or index % 100 == 0 or index == len(rows):
                print(f"cache {split}: {index}/{len(rows)}", flush=True)
    manifest_hash = sha256(args.manifest)
    cache_summary = {
        "phase": "54A-cache",
        "config": args.config,
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": sha256(args.checkpoint),
        "manifest_sha256": manifest_hash,
        "feature_index": args.feature_index,
        "train_images": len(train_rows),
        "val_images": len(val_rows),
        "maximum_rgb_alignment_mae_0_255": maximum_alignment_error,
        "records": len(records),
        "target_test_read": False,
        "cloudsen_internal_test_read": False,
    }
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "cache_summary.json").write_text(
        json.dumps(cache_summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(cache_summary, indent=2))


def load_cached(path, device="cuda"):
    with np.load(path) as item:
        return (
            torch.from_numpy(item["feature"].copy()).to(device=device, dtype=torch.float32),
            torch.from_numpy(item["auxiliary"].copy()).to(device=device, dtype=torch.float32),
            torch.from_numpy(item["probabilities"].copy()).to(device=device, dtype=torch.float32),
            torch.from_numpy(item["source_label"].copy()).to(device=device, dtype=torch.long),
        )


def confusion_metrics(confusion):
    values = confusion.astype(np.float64)
    true_positive = np.diag(values)
    truth = values.sum(axis=1)
    predicted = values.sum(axis=0)
    union = truth + predicted - true_positive
    iou = np.divide(true_positive, union, out=np.full(4, np.nan), where=union > 0)
    return {
        "mIoU": 100 * float(np.nanmean(iou)),
        "per_class_iou": {
            TARGET_NAMES[index]: 100 * float(iou[index]) for index in range(4)
        },
    }


def evaluate(model, condition, rows, cache_root):
    model.eval()
    confusion = np.zeros((4, 4), dtype=np.int64)
    with torch.inference_mode():
        for row in rows:
            feature, auxiliary, probabilities, labels = load_cached(
                cache_path(cache_root, "val", row)
            )
            residual = model(feature[None], auxiliary[None], condition)
            residual = F.interpolate(
                residual, size=probabilities.shape[-2:], mode="bilinear", align_corners=False
            )
            adapted = apply_shadow_residual(probabilities[None], residual)[0]
            prediction = adapted.argmax(dim=0).cpu().numpy()
            target = SOURCE_TO_TARGET[labels.cpu().numpy()]
            predicted_target = SOURCE_TO_TARGET[prediction]
            encoded = 4 * target.reshape(-1) + predicted_target.reshape(-1)
            confusion += np.bincount(encoded, minlength=16).reshape(4, 4)
    return {**confusion_metrics(confusion), "confusion": confusion.tolist()}


def train_one(args, condition, seed, train_rows, val_rows):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = FixedShadowProbe().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    positive = []
    all_items = []
    for row in train_rows:
        item = (*load_cached(cache_path(args.cache_root, "train", row)), row["name"])
        all_items.append(item)
        if bool((item[3] == 3).any()):
            positive.append(item)
    if len(positive) != 8:
        raise RuntimeError(f"Expected eight shadow-positive train patches, found {len(positive)}")
    generator = random.Random(seed)
    best = None
    history = []
    run_root = Path(args.output_root) / condition / f"seed{seed}"
    run_root.mkdir(parents=True, exist_ok=True)
    for iteration in range(1, args.max_iters + 1):
        model.train()
        batch = [generator.choice(positive)] + [generator.choice(all_items) for _ in range(3)]
        feature = torch.stack([item[0] for item in batch])
        auxiliary = torch.stack([item[1] for item in batch])
        probabilities = torch.stack([item[2] for item in batch])
        labels = torch.stack([item[3] for item in batch])
        flips = [generator.random() < 0.5 for _ in batch]
        for batch_index, flip in enumerate(flips):
            if flip:
                feature[batch_index] = feature[batch_index].flip(-1)
                auxiliary[batch_index] = auxiliary[batch_index].flip(-1)
                probabilities[batch_index] = probabilities[batch_index].flip(-1)
                labels[batch_index] = labels[batch_index].flip(-1)
        progress = min(1.0, iteration / args.max_iters)
        learning_rate = 1e-3 * (1.0 - progress) ** 0.9
        if iteration <= 100:
            learning_rate *= 0.1 + 0.9 * iteration / 100.0
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        residual = model(feature, auxiliary, condition)
        residual = F.interpolate(
            residual, size=probabilities.shape[-2:], mode="bilinear", align_corners=False
        ).squeeze(1)
        base_shadow = probabilities[:, 3].clamp(1e-6, 1 - 1e-6)
        shadow_logit = torch.logit(base_shadow) + residual
        target = labels == 3
        valid = labels != 255
        loss_bce = balanced_shadow_bce(shadow_logit, labels, 255, 3)
        loss_tversky = shadow_tversky(shadow_logit.sigmoid(), target, valid)
        boundary = shadow_boundary_band(labels, 3, radius=1) & valid
        loss_boundary = (
            F.binary_cross_entropy_with_logits(
                shadow_logit[boundary], target[boundary].to(shadow_logit.dtype)
            ) if boundary.any() else shadow_logit.sum() * 0
        )
        loss = loss_bce + loss_tversky + 0.5 * loss_boundary
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if iteration % args.val_interval == 0:
            metrics = evaluate(model, condition, val_rows, args.cache_root)
            row = {"iteration": iteration, "loss": float(loss.detach()), **metrics}
            history.append(row)
            key = (
                metrics["per_class_iou"]["cloud_shadow"],
                metrics["per_class_iou"]["thin_cloud"], metrics["mIoU"],
            )
            if best is None or key > best[0]:
                best = (key, row)
                torch.save(model.state_dict(), run_root / "best.pth")
            print(
                f"{condition} seed={seed} iter={iteration}: "
                f"shadow={key[0]:.4f} thin={key[1]:.4f} mIoU={key[2]:.4f}",
                flush=True,
            )
    result = {
        "condition": condition,
        "seed": seed,
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "train_images": len(train_rows),
        "shadow_positive_train_images": len(positive),
        "val_images": len(val_rows),
        "max_iters": args.max_iters,
        "best": best[1],
        "history": history,
    }
    (run_root / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def aggregate(args, results):
    grouped = {condition: [] for condition in CONDITIONS}
    for result in results:
        grouped[result["condition"]].append(result)
    means = {}
    for condition, rows in grouped.items():
        means[condition] = {
            name: float(np.mean([row["best"]["per_class_iou"][name] for row in rows]))
            for name in ("cloud_shadow", "thin_cloud")
        }
        means[condition]["mIoU"] = float(np.mean([row["best"]["mIoU"] for row in rows]))
    rgb = grouped["rgb"]
    comparisons = {}
    eligible = []
    for condition in CONDITIONS[1:]:
        rows = grouped[condition]
        per_seed_shadow_gain = [
            row["best"]["per_class_iou"]["cloud_shadow"]
            - rgb[index]["best"]["per_class_iou"]["cloud_shadow"]
            for index, row in enumerate(rows)
        ]
        mean_shadow_gain = means[condition]["cloud_shadow"] - means["rgb"]["cloud_shadow"]
        mean_thin_delta = means[condition]["thin_cloud"] - means["rgb"]["thin_cloud"]
        passed = mean_shadow_gain >= 5.0 and min(per_seed_shadow_gain) >= 2.0 and mean_thin_delta >= -1.0
        comparisons[condition] = {
            "mean_shadow_iou_gain": mean_shadow_gain,
            "per_seed_shadow_iou_gain": per_seed_shadow_gain,
            "mean_thin_iou_delta": mean_thin_delta,
            "passed": passed,
        }
        # The fifth condition may improve through spatial cloud context alone;
        # it is diagnostic but cannot satisfy the spectral/geometry gate.
        if passed and condition in CONDITIONS[1:4]:
            eligible.append(condition)
    parameter_counts = {row["trainable_parameters"] for row in results}
    gates = {
        "all_conditions_three_seeds": all(len(rows) == 3 for rows in grouped.values()),
        "identical_parameter_count": len(parameter_counts) == 1,
        "parameter_count_is_12929": parameter_counts == {12929},
        "same_65_train_and_1905_val": all(
            row["train_images"] == 65 and row["val_images"] == 1905 for row in results
        ),
        "finite_metrics": all(
            math.isfinite(row["best"]["mIoU"]) and all(
                math.isfinite(value) for value in row["best"]["per_class_iou"].values()
            ) for row in results
        ),
        "material_information_gain": bool(eligible),
        "target_test_sealed": True,
        "cloudsen_internal_test_sealed": True,
    }
    passed = all(gates.values())
    summary = {
        "phase": "54A",
        "diagnostic_supervised_oracle": True,
        "conditions": list(CONDITIONS),
        "seeds": list(SEEDS),
        "means": means,
        "comparisons_to_rgb": comparisons,
        "eligible_conditions": eligible,
        "gates": gates,
        "passed": passed,
        "decision": "information_increment_material" if passed else "information_increment_not_material",
        "target_test_evaluated": False,
        "cloudsen_internal_test_evaluated": False,
    }
    (Path(args.output_root) / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return summary


def run_training(args):
    cache_summary = json.loads(
        (Path(args.output_root) / "cache_summary.json").read_text(encoding="utf-8")
    )
    if cache_summary["train_images"] != 65 or cache_summary["val_images"] != 1905:
        raise RuntimeError("Invalid Phase 54 cache")
    train_rows = selected_train_rows(args.manifest, args.selection)
    val_rows = load_manifest(args.manifest, "target_val")
    results = []
    for condition in CONDITIONS:
        for seed in SEEDS:
            summary_path = Path(args.output_root) / condition / f"seed{seed}" / "summary.json"
            if summary_path.is_file():
                results.append(json.loads(summary_path.read_text(encoding="utf-8")))
            else:
                results.append(train_one(args, condition, seed, train_rows, val_rows))
    return aggregate(args, results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("cache", "train", "all"), default="all")
    parser.add_argument("--raw-root", default="/home/scv/shared/data/l8_biome_raw")
    parser.add_argument("--manifest", default="work_dirs/phase45_protocol_audit/scene_disjoint_manifest.csv")
    parser.add_argument("--selection", default="work_dirs/phase50_active_1pct/selection.json")
    parser.add_argument("--config", default="configs/protocol/phase22_clean_v8_l1c.py")
    parser.add_argument("--checkpoint", default="work_dirs/phase22_clean_v8/seed42/best_mIoU_iter_40000.pth")
    parser.add_argument("--cache-root", default="work_dirs/phase54_information_audit/cache")
    parser.add_argument("--output-root", default="work_dirs/phase54_information_audit")
    parser.add_argument("--feature-index", type=int, default=2)
    parser.add_argument("--max-iters", type=int, default=10000)
    parser.add_argument("--val-interval", type=int, default=1000)
    args = parser.parse_args()
    if args.stage in ("cache", "all"):
        create_cache(args)
    if args.stage in ("train", "all"):
        run_training(args)


if __name__ == "__main__":
    main()
