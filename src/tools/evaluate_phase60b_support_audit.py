"""Fit one frozen shallow readout per Phase 60B method/budget and evaluate it."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = Path(__file__).resolve().parent
for item in (REPO_ROOT, TOOLS_ROOT):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from audit_phase56_readout import apply_ridge, fit_ridge, metrics  # noqa: E402
from evaluate_phase56_readout_dense import feature_map  # noqa: E402
from phase56_protocol import (  # noqa: E402
    SOURCE_CLASS_NAMES, load_source_order_mask, read_shadow_yes_rows, sha256,
)
from run_phase55_mechanism_audit import backbone_layers_and_outputs, build, normalized_input  # noqa: E402


def load_array(root, name):
    return np.load(Path(root) / f"{name}.npy", mmap_mode="r")


def confusion_metrics(confusion):
    confusion = np.asarray(confusion, dtype=np.int64)
    true_positive = np.diag(confusion).astype(np.float64)
    ground_truth, predicted = confusion.sum(1), confusion.sum(0)
    union = ground_truth + predicted - true_positive
    iou = np.divide(true_positive, union, out=np.zeros(4), where=union > 0)
    class_iou = dict(zip(SOURCE_CLASS_NAMES, (100.0 * iou).tolist()))
    thin, shadow = class_iou["thin_cloud"], class_iou["cloud_shadow"]
    return {
        "mIoU": 100.0 * float(iou.mean()),
        "class_iou": class_iou,
        "thin_shadow_harmonic_mean": 2.0 * thin * shadow / max(thin + shadow, 1e-12),
        "confusion": confusion.tolist(),
    }


def effective_parameters(model):
    scale = model["scale"].clip(min=1e-12)
    kernel = model["coefficients"][:-1] / scale[:, None]
    bias = model["coefficients"][-1] - (model["mean"] / scale) @ model["coefficients"][:-1]
    return kernel.astype(np.float32), bias.astype(np.float32)


def read_manifest(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def selection_support(names, row_by_name):
    image_presence = np.zeros((len(names), 4), dtype=bool)
    pixels = np.zeros(4, dtype=np.int64)
    scenes, biomes = set(), set()
    for index, name in enumerate(names):
        row = row_by_name[name]
        counts = np.bincount(load_source_order_mask(row).reshape(-1), minlength=4)
        image_presence[index] = counts > 0
        pixels += counts
        scenes.add(row["scene"])
        biomes.add(row["biome"])
    return {
        "images": len(names), "scenes": len(scenes), "biomes": len(biomes),
        "images_with_thin": int(image_presence[:, 2].sum()),
        "images_with_shadow": int(image_presence[:, 3].sum()),
        "images_with_both_thin_and_shadow": int((image_presence[:, 2] & image_presence[:, 3]).sum()),
        "pixel_counts_source_order": dict(zip(SOURCE_CLASS_NAMES, map(int, pixels))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", default="work_dirs/phase60/phase60b_selections.json")
    parser.add_argument("--train-cache", default="work_dirs/phase60/phase60b_feature_cache/adapted/target_train_union")
    parser.add_argument("--validation-cache", default="work_dirs/phase56_readout/cache/adapted/target_val")
    parser.add_argument("--manifest", default="work_dirs/phase45_protocol_audit/scene_disjoint_manifest.csv")
    parser.add_argument("--metadata", default="research_plans/protocol_data/l8_biome_usgs_shadow_status.csv")
    parser.add_argument("--phase50-selection", default="work_dirs/phase50_active_1pct/selection.json")
    parser.add_argument("--config", default="configs/protocol/phase52a_fix_shadow_status_1pct_l1c.py")
    parser.add_argument("--checkpoint", default="work_dirs/phase52a_fix_shadow_status_1pct/seed52/best_mIoU_iter_1000.pth")
    parser.add_argument("--output", default="work_dirs/phase60/phase60b_summary.json")
    parser.add_argument("--ridge", type=float, default=0.01)
    parser.add_argument("--dense-chunk-readouts", type=int, default=8)
    args = parser.parse_args()

    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    train_meta = json.loads((Path(args.train_cache) / "metadata.json").read_text(encoding="utf-8"))
    val_meta = json.loads((Path(args.validation_cache) / "metadata.json").read_text(encoding="utf-8"))
    if train_meta["sampling_seed"] != val_meta["sampling_seed"] or train_meta["cap_per_image_class"] != val_meta["cap_per_image_class"]:
        raise RuntimeError("Training and validation feature caches do not share the same pixel-sampling protocol")
    if val_meta["images"] != 963:
        raise RuntimeError("Expected the frozen 963-image Shadows?=yes validation cache")

    union_names = train_meta["image_ids"]
    union_index = {name: index for index, name in enumerate(union_names)}
    x_train = load_array(args.train_cache, "pixel_decoder_mask")
    y_train = load_array(args.train_cache, "label")
    image_index = load_array(args.train_cache, "image_index")
    x_val = load_array(args.validation_cache, "pixel_decoder_mask")
    y_val = load_array(args.validation_cache, "label")
    all_rows = [row for row in read_manifest(args.manifest) if row["new_split"] == "target_train"]
    row_by_name = {row["name"]: row for row in all_rows}

    requested = []
    for method, descriptor in selection["selections"].items():
        for budget, names in descriptor.get("budgets", {}).items():
            requested.append((f"{method}@{budget}", method, int(budget), names, descriptor["oracle_upper_bound_only"]))
        if method == "phase50_exact_nominal65":
            requested.append((method, method, 65, descriptor["names"], False))

    fitted, entries = {}, {}
    for key, method, budget, names, oracle in requested:
        chosen_indices = {union_index[name] for name in names}
        keep = np.fromiter((int(value) in chosen_indices for value in image_index), dtype=bool, count=len(image_index))
        class_counts = np.bincount(np.asarray(y_train[keep], dtype=np.int64), minlength=4)
        entry = {
            "method": method, "requested_budget": budget,
            "oracle_upper_bound_only": oracle,
            "support": selection_support(names, row_by_name),
            "sampled_class_counts": dict(zip(SOURCE_CLASS_NAMES, map(int, class_counts))),
        }
        if np.any(class_counts == 0):
            entry.update({
                "status": "not_fittable_missing_sampled_class",
                "missing_classes": [SOURCE_CLASS_NAMES[i] for i in np.flatnonzero(class_counts == 0)],
            })
            entries[key] = entry
            continue
        model = fit_ridge(x_train[keep], y_train[keep], True, args.ridge)
        fitted[key] = model
        entry["status"] = "fit"
        entry["sampled_validation"] = metrics(apply_ridge(model, x_val), y_val)
        entries[key] = entry
        print(f"fit {key}: {len(names)} images, {int(keep.sum())} pixels", flush=True)

    _, validation, _ = read_shadow_yes_rows(args.manifest, args.metadata, args.phase50_selection)
    model = build(args.config, args.checkpoint).eval().cuda()
    keys = list(fitted)
    confusions = {key: np.zeros((4, 4), dtype=np.int64) for key in keys}
    effective = {key: effective_parameters(fitted[key]) for key in keys}
    captures = {"pixel": None}

    def hook(_module, _inputs, output):
        captures["pixel"] = output

    handle = model.decode_head.pixel_decoder.register_forward_hook(hook)
    for image_number, row in enumerate(validation, 1):
        mask = load_source_order_mask(row)
        inputs = normalized_input(model, row)
        with torch.inference_mode():
            layers, features, _ = backbone_layers_and_outputs(model, inputs)
            model.decode_head(features, None)
            pixel = feature_map("pixel_decoder_mask", layers, captures["pixel"]).float()
            target = torch.from_numpy(mask.astype(np.int64)).cuda()
            for start in range(0, len(keys), args.dense_chunk_readouts):
                chunk = keys[start:start + args.dense_chunk_readouts]
                kernels = torch.from_numpy(np.stack([effective[key][0] for key in chunk])).cuda()
                biases = torch.from_numpy(np.stack([effective[key][1] for key in chunk])).cuda()
                logits = torch.einsum("bchw,rck->brkhw", pixel, kernels)[0] + biases[:, :, None, None]
                logits = F.interpolate(logits, size=mask.shape, mode="bilinear", align_corners=False)
                predictions = logits.argmax(dim=1)
                for offset, key in enumerate(chunk):
                    encoded = 4 * target + predictions[offset]
                    confusions[key] += torch.bincount(encoded.reshape(-1), minlength=16).reshape(4, 4).cpu().numpy()
        if image_number == 1 or image_number % 100 == 0 or image_number == len(validation):
            print(f"dense support validation: {image_number}/{len(validation)}", flush=True)
        del inputs, layers, features, pixel
        torch.cuda.empty_cache()
    handle.remove()

    for key in keys:
        entries[key]["dense_validation"] = confusion_metrics(confusions[key])
    oracle65 = entries.get("oracle_thin_shadow_relation@65", {})
    class65 = entries.get("oracle_class_balanced@65", {})
    exact65 = entries.get("phase50_exact_nominal65", {})
    oracle65_pass = any(
        item.get("dense_validation", {}).get("class_iou", {}).get("thin_cloud", -1) >= 32.0
        and item.get("dense_validation", {}).get("class_iou", {}).get("cloud_shadow", -1) >= 20.0
        for item in (oracle65, class65)
    )
    exact65_pass = (
        exact65.get("dense_validation", {}).get("class_iou", {}).get("thin_cloud", -1) >= 32.0
        and exact65.get("dense_validation", {}).get("class_iou", {}).get("cloud_shadow", -1) >= 20.0
    )
    first_oracle_success = None
    for budget in (16, 32, 65, 130, 325):
        candidates = [entries.get(f"oracle_class_balanced@{budget}", {}), entries.get(f"oracle_thin_shadow_relation@{budget}", {})]
        if any(
            item.get("dense_validation", {}).get("class_iou", {}).get("thin_cloud", -1) >= 32.0
            and item.get("dense_validation", {}).get("class_iou", {}).get("cloud_shadow", -1) >= 20.0
            for item in candidates
        ):
            first_oracle_success = budget
            break
    if oracle65_pass and not exact65_pass:
        interpretation = "sampling_bottleneck_switch_to_active_domain_adaptation"
    elif first_oracle_success in (130, 325):
        interpretation = "one_percent_claim_rejected_more_labels_required"
    elif first_oracle_success is None:
        interpretation = "fundamental_data_label_or_sensor_observability_problem"
    else:
        interpretation = "current_selection_not_proven_to_be_the_only_bottleneck"

    summary = {
        "phase": "60B",
        "protocol": {
            "selection_sha256": sha256(args.selection),
            "checkpoint_sha256": sha256(args.checkpoint),
            "representation": "frozen Phase52 pixel_decoder_mask",
            "classifier": f"same class-balanced multiclass ridge readout, ridge={args.ridge}",
            "sampling_seed": train_meta["sampling_seed"],
            "cap_per_image_class": train_meta["cap_per_image_class"],
            "validation_images": len(validation),
            "validation_filter": "USGS Shadows?=yes only",
            "oracle_allowed_in_final_method": False,
            "target_test_evaluated": False,
            "cloudsen_internal_test_evaluated": False,
        },
        "results": entries,
        "key_judgement": {
            "phase50_nominal65_uses_only_12_legal_images": True,
            "oracle65_reaches_thin32_shadow20": oracle65_pass,
            "current_phase50_reaches_thin32_shadow20": exact65_pass,
            "first_oracle_budget_reaching_thresholds": first_oracle_success,
            "interpretation": interpretation,
        },
    }
    output = Path(args.output)
    output.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(output), **summary["key_judgement"]}, indent=2))


if __name__ == "__main__":
    main()
