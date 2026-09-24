"""Prepare a blinded Phase 61D review packet; never synthesizes human labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = Path(__file__).resolve().parent
for item in (REPO_ROOT, TOOLS_ROOT):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from audit_phase56_readout import apply_ridge, fit_ridge, softmax  # noqa: E402
from evaluate_phase56_readout_dense import feature_map, readout_logits  # noqa: E402
from phase56_protocol import deterministic_samples, load_source_order_mask, read_shadow_yes_rows, sha256  # noqa: E402
from run_phase55_mechanism_audit import backbone_layers_and_outputs, build, normalized_input  # noqa: E402
from run_phase60a_multispectral_audit import cache_path, feature_bank, sample_map, stack_condition  # noqa: E402


LABELS = (
    "definite clear", "definite thin", "definite thick", "definite cloud shadow",
    "terrain/water shadow", "ambiguous haze/cirrus", "uncertain boundary",
)


def stable_value(text):
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


def boundary_mask(mask):
    result = np.zeros(mask.shape, dtype=bool)
    result[1:, :] |= mask[1:, :] != mask[:-1, :]
    result[:-1, :] |= mask[:-1, :] != mask[1:, :]
    result[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    result[:, :-1] |= mask[:, :-1] != mask[:, 1:]
    # A small tolerance band distinguishes boundary from class interior.
    tensor = torch.from_numpy(result.astype(np.float32))[None, None]
    return F.max_pool2d(tensor, 7, stride=1, padding=3)[0, 0].bool().numpy()


def train_s4(train, cache_root, cap=32, seed=60, ridge=0.01):
    features, labels = [], []
    for row in train:
        mask = load_source_order_mask(row)
        coordinates, sampled = deterministic_samples(mask, row["name"], cap, seed)
        bank = feature_bank(row, cache_path(cache_root, "train", row))
        features.append(sample_map(stack_condition(bank, "S4_common6_indices"), coordinates, mask.shape))
        labels.append(sampled)
    return fit_ridge(np.concatenate(features), np.concatenate(labels), True, ridge)


def candidates_from_pixels(row, source_name, coordinates, confidence, boundary, maximum=4):
    if len(coordinates) == 0:
        return []
    order = sorted(
        range(len(coordinates)),
        key=lambda index: stable_value(f"{row['name']}:{source_name}:{coordinates[index, 0]}:{coordinates[index, 1]}"),
    )
    result = []
    region_counts = Counter()
    for index in order:
        y, x = map(int, coordinates[index])
        region = "boundary" if boundary[y, x] else "interior"
        if region_counts[region] >= maximum:
            continue
        result.append({
            "source_stratum": source_name, "name": row["name"], "scene": row["scene"],
            "biome": row["biome"], "image_path": row["image_path"], "mask_path": row["mask_path"],
            "center_y": y, "center_x": x, "region": region,
            "s4_confidence": float(confidence[y, x]),
        })
        region_counts[region] += 1
    return result


def choose_balanced(records, budget):
    if len(records) < budget:
        raise RuntimeError(f"Only {len(records)} blind-review candidates for budget {budget}")
    threshold = float(np.median([row["s4_confidence"] for row in records]))
    groups = defaultdict(list)
    for row in records:
        row["confidence_stratum"] = "high" if row["s4_confidence"] >= threshold else "low"
        key = (row["source_stratum"], row["biome"], row["confidence_stratum"], row["region"])
        groups[key].append(row)
    for key in groups:
        groups[key].sort(key=lambda row: stable_value(f"61D:{row['name']}:{row['center_y']}:{row['center_x']}"))
    selected, image_counts = [], Counter()
    keys = sorted(groups)
    while len(selected) < budget:
        progress = False
        for key in keys:
            while groups[key] and image_counts[groups[key][0]["name"]] >= 2:
                groups[key].pop(0)
            if groups[key] and len(selected) < budget:
                row = groups[key].pop(0)
                selected.append(row)
                image_counts[row["name"]] += 1
                progress = True
        if not progress:
            break
    if len(selected) < budget:
        remainder = sorted(
            [row for values in groups.values() for row in values],
            key=lambda row: stable_value(f"61D-fill:{row['name']}:{row['center_y']}:{row['center_x']}"),
        )
        selected.extend(remainder[:budget - len(selected)])
    if len(selected) != budget:
        raise RuntimeError(f"Could select only {len(selected)}/{budget} review units")
    return selected, threshold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/protocol/phase52a_fix_shadow_status_1pct_l1c.py")
    parser.add_argument("--checkpoint", default="work_dirs/phase52a_fix_shadow_status_1pct/seed52/best_mIoU_iter_1000.pth")
    parser.add_argument("--manifest", default="work_dirs/phase45_protocol_audit/scene_disjoint_manifest.csv")
    parser.add_argument("--metadata", default="research_plans/protocol_data/l8_biome_usgs_shadow_status.csv")
    parser.add_argument("--selection", default="work_dirs/phase50_active_1pct/selection.json")
    parser.add_argument("--phase54-cache", default="work_dirs/phase54_information_audit/cache")
    parser.add_argument("--phase56-summary", default="work_dirs/phase56_readout/readout_summary.json")
    parser.add_argument("--output-root", default="work_dirs/phase61/blind_review")
    parser.add_argument("--budget", type=int, default=200)
    parser.add_argument("--crop-size", type=int, default=128)
    args = parser.parse_args()

    train, validation, _ = read_shadow_yes_rows(args.manifest, args.metadata, args.selection)
    s4_model = train_s4(train, Path(args.phase54_cache))
    phase56 = json.loads(Path(args.phase56_summary).read_text(encoding="utf-8"))
    chosen = phase56["best_parameter_limited_readout"]
    artifact = np.load(chosen["artifact"])
    model = build(args.config, args.checkpoint).eval().cuda()
    captures = {"pixel": None}

    def hook(_module, _inputs, output):
        captures["pixel"] = output

    handle = model.decode_head.pixel_decoder.register_forward_hook(hook)
    candidate_records = []
    for index, row in enumerate(validation, 1):
        mask = load_source_order_mask(row)
        inputs = normalized_input(model, row)
        with torch.inference_mode():
            layers, features, _ = backbone_layers_and_outputs(model, inputs)
            class_scores, mask_scores = model.decode_head(features, None)
            class_probability = F.softmax(class_scores[-1].float(), dim=-1)[..., :-1]
            mask_probability = mask_scores[-1].float().sigmoid()
            phase52_logits = torch.einsum("bqc,bqhw->bchw", class_probability, mask_probability)
            phase52_logits = F.interpolate(phase52_logits, size=mask.shape, mode="bilinear", align_corners=False)
            phase52_prediction = phase52_logits.argmax(1)[0].cpu().numpy()
            selected_feature = feature_map(chosen["representation"], layers, captures["pixel"])
            phase56_logits = readout_logits(selected_feature, artifact, chosen["readout"])
            phase56_logits = F.interpolate(phase56_logits, size=mask.shape, mode="bilinear", align_corners=False)
            phase56_prediction = phase56_logits.argmax(1)[0].cpu().numpy()

        bank = feature_bank(row, cache_path(Path(args.phase54_cache), "val", row))
        s4_feature = stack_condition(bank, "S4_common6_indices")
        s4_scores = apply_ridge(s4_model, s4_feature.reshape(s4_feature.shape[0], -1).T)
        s4_probability = softmax(s4_scores).max(1).reshape(s4_feature.shape[1:])
        s4_confidence = np.asarray(
            Image.fromarray(s4_probability.astype(np.float32), mode="F").resize(
                (mask.shape[1], mask.shape[0]), Image.Resampling.BILINEAR
            )
        )
        boundary = boundary_mask(mask)
        phase52_coords = np.argwhere((phase52_prediction == 2) & (mask == 3))
        phase56_coords = np.argwhere((phase56_prediction == 3) & (mask == 2))
        candidate_records.extend(candidates_from_pixels(
            row, "phase52_pred_thin_gt_shadow", phase52_coords, s4_confidence, boundary
        ))
        candidate_records.extend(candidates_from_pixels(
            row, "phase56_pred_shadow_gt_thin", phase56_coords, s4_confidence, boundary
        ))
        if index == 1 or index % 100 == 0 or index == len(validation):
            print(f"61D candidate inference: {index}/{len(validation)} candidates={len(candidate_records)}", flush=True)
        del inputs, layers, features, class_scores, mask_scores
        torch.cuda.empty_cache()
    handle.remove()

    selected, confidence_threshold = choose_balanced(candidate_records, args.budget)
    output_root = Path(args.output_root)
    image_root = output_root / "images"
    image_root.mkdir(parents=True, exist_ok=True)
    sealed = []
    for index, row in enumerate(selected, 1):
        tile_id = f"P61D-{index:04d}-{stable_value(row['name'] + str(row['center_y']) + str(row['center_x'])):016x}"[-26:]
        with Image.open(row["image_path"]) as image:
            image = image.convert("RGB")
            half = args.crop_size // 2
            left = min(max(row["center_x"] - half, 0), image.width - args.crop_size)
            top = min(max(row["center_y"] - half, 0), image.height - args.crop_size)
            crop = image.crop((left, top, left + args.crop_size, top + args.crop_size))
            crop.save(image_root / f"{tile_id}.png")
        gt_source = int(load_source_order_mask(row)[row["center_y"], row["center_x"]])
        sealed.append({
            "tile_id": tile_id, **{key: value for key, value in row.items() if key not in ("image_path", "mask_path")},
            "original_source_order_label": gt_source,
            "original_label_name": ("definite clear", "definite thick", "definite thin", "definite cloud shadow")[gt_source],
            "crop_left": left, "crop_top": top, "crop_size": args.crop_size,
        })

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "sealed_manifest.json").write_text(json.dumps({
        "phase": "61D", "blinding": "Reviewer files and image names contain no GT/model/source stratum.",
        "budget": len(sealed), "labels": list(LABELS), "confidence_median_threshold": confidence_threshold,
        "checkpoint_sha256": sha256(args.checkpoint), "phase56_artifact_sha256": sha256(chosen["artifact"]),
        "records": sealed,
    }, indent=2), encoding="utf-8")
    for reviewer in ("reviewer_A", "reviewer_B"):
        with (output_root / f"{reviewer}.csv").open("w", newline="", encoding="utf-8") as handle_csv:
            writer = csv.DictWriter(handle_csv, fieldnames=("tile_id", "label", "notes"))
            writer.writeheader()
            for row in sealed:
                writer.writerow({"tile_id": row["tile_id"], "label": "", "notes": ""})
    (output_root / "INSTRUCTIONS.md").write_text(
        "# Phase 61D blind review\n\n"
        "Review images independently. Do not open `sealed_manifest.json` until both CSV files are final. "
        "For each tile, enter exactly one label in the assigned reviewer CSV:\n\n"
        + "\n".join(f"- `{label}`" for label in LABELS)
        + "\n\nDo not discuss labels between reviewers before locking both files.\n",
        encoding="utf-8",
    )
    strata = Counter((row["source_stratum"], row["biome"], row["confidence_stratum"], row["region"]) for row in sealed)
    summary = {
        "phase": "61D-packet", "status": "awaiting_two_independent_human_reviews",
        "review_units": len(sealed), "candidate_units": len(candidate_records),
        "strata": {"|".join(key): value for key, value in sorted(strata.items())},
        "reviewer_files_blank": True, "human_agreement_metrics_available": False,
        "reason": "Cohen/Fleiss kappa and consensus validity require two independent completed reviewer files.",
        "sealed_manifest_sha256": sha256(output_root / "sealed_manifest.json"),
        "target_test_evaluated": False, "cloudsen_internal_test_evaluated": False,
    }
    (output_root / "packet_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
