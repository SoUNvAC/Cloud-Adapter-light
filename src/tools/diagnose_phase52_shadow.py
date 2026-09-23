import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from export_phase12_onnx import build_wrapper  # noqa: E402
from validate_phase12_onnxruntime import load_rgb_image  # noqa: E402


TARGET_NAMES = ("clear", "cloud_shadow", "thin_cloud", "thick_cloud")


def read_rows(manifest, selection):
    selected = {row["name"] for row in selection["selected"]}
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if row["new_split"] == "target_train" and row["name"] in selected
        ]
    rows.sort(key=lambda row: row["name"])
    if len(rows) != 65 or len(selected) != 65:
        raise RuntimeError(f"Expected 65 selected rows, found {len(rows)}/{len(selected)}")
    return rows


def occupancy_and_shadow_flow(summary):
    confusion = np.asarray(summary["target"]["confusion"], dtype=np.int64)
    shadow_row = confusion[1]
    return {
        "predicted_shadow_pixels": int(confusion[:, 1].sum()),
        "predicted_shadow_fraction": float(confusion[:, 1].sum() / confusion.sum()),
        "true_shadow_pixels": int(shadow_row.sum()),
        "true_shadow_prediction_counts": {
            name: int(shadow_row[index]) for index, name in enumerate(TARGET_NAMES)
        },
        "true_shadow_prediction_fractions": {
            name: float(shadow_row[index] / shadow_row.sum())
            for index, name in enumerate(TARGET_NAMES)
        },
    }


def normalized(vector):
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def cosine_distance(left, right):
    return float(1.0 - np.dot(normalized(left), normalized(right)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/protocol/phase52_sparse_msre_1pct_l1c.py")
    parser.add_argument("--checkpoint", default="work_dirs/phase52_sparse_msre_1pct/seed52/best_mIoU_iter_1000.pth")
    parser.add_argument("--manifest", default="work_dirs/phase45_protocol_audit/scene_disjoint_manifest.csv")
    parser.add_argument("--selection", default="work_dirs/phase50_active_1pct/selection.json")
    parser.add_argument("--source-summary", default="work_dirs/phase45_source_only/summary.json")
    parser.add_argument("--phase52-summary", default="work_dirs/phase52_sparse_msre_1pct/summary.json")
    parser.add_argument("--feature-index", type=int, default=2)
    parser.add_argument("--output", default="work_dirs/phase52_shadow_diagnosis/summary.json")
    args = parser.parse_args()

    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    rows = read_rows(Path(args.manifest), selection)
    source = json.loads(Path(args.source_summary).read_text(encoding="utf-8"))["models"]["v8"]
    phase52 = json.loads(Path(args.phase52_summary).read_text(encoding="utf-8"))

    selected_counts = np.zeros(4, dtype=np.int64)
    shadow_images = 0
    shadow_biomes = Counter()
    for row in rows:
        with Image.open(row["mask_path"]) as image:
            labels = np.asarray(image, dtype=np.int64)
        selected_counts += np.bincount(labels.reshape(-1), minlength=4)[:4]
        if np.any(labels == 1):
            shadow_images += 1
            shadow_biomes[row["biome"]] += 1

    wrapper_args = argparse.Namespace(
        config=args.config, checkpoint=args.checkpoint, precision="fp16",
        active_block_indices=None,
    )
    wrapper, _ = build_wrapper(wrapper_args)
    all_features, all_labels, all_luminance, all_biomes = [], [], [], []
    for index, row in enumerate(rows, 1):
        rgb_np = load_rgb_image(Path(row["image_path"]), 512)
        rgb = torch.from_numpy(rgb_np).cuda()
        with Image.open(row["mask_path"]) as image:
            labels_np = np.asarray(image, dtype=np.int64).copy()
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=True
        ):
            features = wrapper.segmentor.extract_feat(wrapper.normalize(rgb))[
                args.feature_index
            ].float()
        size = features.shape[-2:]
        labels = F.interpolate(
            torch.from_numpy(labels_np).cuda()[None, None].float(),
            size=size, mode="nearest",
        )[0, 0].long()
        rgb_small = F.interpolate(rgb.float(), size=size, mode="area")[0]
        luminance = 0.2126 * rgb_small[0] + 0.7152 * rgb_small[1] + 0.0722 * rgb_small[2]
        all_features.append(features[0].permute(1, 2, 0).cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        all_luminance.append(luminance.cpu().numpy())
        all_biomes.append(np.full(size, row["biome"], dtype=object))
        if index == 1 or index % 10 == 0 or index == len(rows):
            print(f"feature diagnosis: {index}/{len(rows)}", flush=True)

    features = np.concatenate([value.reshape(-1, value.shape[-1]) for value in all_features])
    labels = np.concatenate([value.reshape(-1) for value in all_labels])
    luminance = np.concatenate([value.reshape(-1) for value in all_luminance])
    biomes = np.concatenate([value.reshape(-1) for value in all_biomes])
    prototypes = {
        TARGET_NAMES[index]: features[labels == index].mean(axis=0)
        for index in range(4)
    }
    clear_luminance = luminance[labels == 0]
    dark_threshold = float(np.quantile(clear_luminance, 0.25))
    dark_clear = (labels == 0) & (luminance <= dark_threshold)
    prototypes["dark_clear_q25"] = features[dark_clear].mean(axis=0)
    shadow = prototypes["cloud_shadow"]
    distances = {
        name: cosine_distance(shadow, prototype)
        for name, prototype in prototypes.items() if name != "cloud_shadow"
    }
    biome_distances = {}
    for biome in sorted(set(biomes.tolist())):
        biome_shadow = (labels == 1) & (biomes == biome)
        biome_dark = dark_clear & (biomes == biome)
        if biome_shadow.any() and biome_dark.any():
            biome_distances[biome] = {
                "shadow_pixels_at_feature_scale": int(biome_shadow.sum()),
                "dark_clear_pixels_at_feature_scale": int(biome_dark.sum()),
                "shadow_to_dark_clear_cosine_distance": cosine_distance(
                    features[biome_shadow].mean(axis=0),
                    features[biome_dark].mean(axis=0),
                ),
            }

    source_iou = source["metrics"]["per_class_iou"]
    phase52_iou = phase52["target"]["metrics"]["per_class_iou"]
    source_boundary = source["boundary"]["per_class"]
    phase52_boundary = phase52["target"]["boundary"]["per_class"]
    result = {
        "phase": "52-shadow-diagnosis",
        "checkpoint": args.checkpoint,
        "feature_output_index": args.feature_index,
        "class_iou_delta": {
            name: phase52_iou[name] - source_iou[name] for name in source_iou
        },
        "weak_boundary_f1": {
            name: {
                "source_only": source_boundary[name]["f1"],
                "phase52": phase52_boundary[name]["f1"],
                "delta": phase52_boundary[name]["f1"] - source_boundary[name]["f1"],
            }
            for name in ("thin_cloud", "cloud_shadow")
        },
        "source_val_mIoU_disabled_target": phase52["source"]["metrics"]["mIoU"],
        "source_only_shadow_flow": occupancy_and_shadow_flow({"target": source}),
        "phase52_shadow_flow": occupancy_and_shadow_flow(phase52),
        "selected_1pct_labels": {
            "images": len(rows),
            "pixel_counts": {TARGET_NAMES[i]: int(selected_counts[i]) for i in range(4)},
            "pixel_fractions": {
                TARGET_NAMES[i]: float(selected_counts[i] / selected_counts.sum())
                for i in range(4)
            },
            "images_containing_shadow": shadow_images,
            "shadow_image_fraction": shadow_images / len(rows),
            "shadow_biome_image_counts": dict(sorted(shadow_biomes.items())),
            "shadow_biome_coverage": len(shadow_biomes),
        },
        "feature_prototypes": {
            "definition": "Phase52 output index 2; dark proxy is lowest-luminance quartile of true-clear feature cells in selected 1%",
            "dark_clear_luminance_threshold_0_255": dark_threshold,
            "feature_cell_counts": {
                **{TARGET_NAMES[i]: int(np.count_nonzero(labels == i)) for i in range(4)},
                "dark_clear_q25": int(dark_clear.sum()),
            },
            "shadow_cosine_distances": distances,
            "per_biome_shadow_to_dark_clear": biome_distances,
            "water_or_mountain_labels_available": False,
        },
        "target_test_evaluated": False,
        "cloudsen_internal_test_evaluated": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
