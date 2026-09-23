import argparse
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path
import sys

os.environ.setdefault("XFORMERS_DISABLED", "1")

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from export_phase12_onnx import build_wrapper  # noqa: E402
from run_phase45_source_only import load_rows, sha256  # noqa: E402
from validate_phase12_onnxruntime import load_rgb_image  # noqa: E402


def stable_unit(name):
    value = hashlib.sha256(f"phase50:{name}".encode("utf-8")).digest()[:8]
    return int.from_bytes(value, "big") / float(2**64 - 1)


def largest_remainder_quotas(rows, budget):
    counts = Counter(row["biome"] for row in rows)
    raw = {key: budget * value / len(rows) for key, value in counts.items()}
    quotas = {key: int(math.floor(value)) for key, value in raw.items()}
    remainder = budget - sum(quotas.values())
    order = sorted(raw, key=lambda key: (-(raw[key] - quotas[key]), key))
    for key in order[:remainder]:
        quotas[key] += 1
    return quotas


def normalized_probabilities(scores):
    scores = scores.float().clamp_min(1e-8)
    return scores / scores.sum(dim=1, keepdim=True).clamp_min(1e-8)


def score_rows(wrapper, rows):
    records = []
    for index, row in enumerate(rows, 1):
        rgb_np = load_rgb_image(Path(row["image_path"]), 512)
        rgb = torch.from_numpy(rgb_np).cuda()
        unit = stable_unit(row["name"])
        brightness = 0.9 + 0.2 * unit
        contrast = 0.9 + 0.2 * (1.0 - unit)
        mean = rgb.mean(dim=(2, 3), keepdim=True)
        strong = ((rgb - mean) * contrast + mean) * brightness
        strong = strong.clamp(0.0, 255.0).flip(-1)
        with torch.inference_mode():
            original = normalized_probabilities(wrapper(rgb))
            augmented = normalized_probabilities(wrapper(strong)).flip(-1)
        average = 0.5 * (original + augmented)
        entropy_map = -(average * average.clamp_min(1e-8).log()).sum(dim=1) / math.log(4)
        inconsistency = (original - augmented).abs().mean(dim=1)
        weak = torch.maximum(average[:, 2], average[:, 3]).flatten()
        top_count = max(1, weak.numel() // 10)
        weak_top = weak.topk(top_count).values.mean().item()
        prediction_histogram = average.mean(dim=(0, 2, 3)).cpu().numpy()
        rgb_unit = rgb_np[0] / 255.0
        feature = np.concatenate(
            [rgb_unit.mean(axis=(1, 2)), rgb_unit.std(axis=(1, 2)), prediction_histogram]
        )
        records.append(
            {
                **row,
                "weak_top10": weak_top,
                "inconsistency": inconsistency.mean().item(),
                "entropy": entropy_map.mean().item(),
                "feature": feature.tolist(),
            }
        )
        if index == 1 or index % 100 == 0 or index == len(rows):
            print(f"Phase 50 scoring: {index}/{len(rows)}", flush=True)
    return records


def zscore(values):
    values = np.asarray(values, dtype=np.float64)
    return (values - values.mean()) / max(values.std(), 1e-12)


def select_diverse(records, budget):
    weak = zscore([row["weak_top10"] for row in records])
    inconsistency = zscore([row["inconsistency"] for row in records])
    entropy = zscore([row["entropy"] for row in records])
    features = np.asarray([row["feature"] for row in records], dtype=np.float64)
    features = (features - features.mean(axis=0)) / np.maximum(features.std(axis=0), 1e-12)
    utility = 0.4 * weak + 0.35 * inconsistency + 0.25 * entropy
    for index, row in enumerate(records):
        row["utility"] = float(utility[index])
        row["feature_index"] = index
    quotas = largest_remainder_quotas(records, budget)
    selected = []
    scene_counts = Counter()
    for biome, quota in sorted(quotas.items()):
        candidates = [row for row in records if row["biome"] == biome]
        candidates.sort(key=lambda row: (-row["utility"], row["name"]))
        pool = candidates[: max(quota * 20, quota)]
        chosen = []
        while len(chosen) < quota:
            eligible = [row for row in pool if row not in chosen and scene_counts[row["scene"]] < 2]
            if not eligible:
                eligible = [row for row in candidates if row not in chosen and scene_counts[row["scene"]] < 2]
            if not eligible:
                raise RuntimeError(f"Could not satisfy Phase 50 quota for {biome}")
            if not chosen:
                winner = eligible[0]
            else:
                chosen_features = features[[row["feature_index"] for row in chosen]]
                scored = []
                for row in eligible:
                    distance = np.linalg.norm(
                        chosen_features - features[row["feature_index"]], axis=1
                    ).min()
                    scored.append((row["utility"] + 0.5 * distance, row["name"], row))
                winner = max(scored, key=lambda value: (value[0], value[1]))[2]
            chosen.append(winner)
            scene_counts[winner["scene"]] += 1
        selected.extend(chosen)
    selected.sort(key=lambda row: row["name"])
    return selected, quotas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/protocol/phase22_clean_v8_l1c.py")
    parser.add_argument("--checkpoint", default="work_dirs/phase22_clean_v8/seed42/best_mIoU_iter_40000.pth")
    parser.add_argument("--manifest", default="work_dirs/phase45_protocol_audit/scene_disjoint_manifest.csv")
    parser.add_argument("--phase49", default="work_dirs/phase49_target_factor_adapter/summary.json")
    parser.add_argument("--output", default="work_dirs/phase50_active_1pct/selection.json")
    parser.add_argument("--budget", type=int, default=65)
    args = parser.parse_args()
    phase49 = json.loads(Path(args.phase49).read_text(encoding="utf-8"))
    if phase49.get("decision") != "stop_adjacent_da_variants":
        raise RuntimeError("Phase 49 terminal DA stop is required")
    rows = load_rows(Path(args.manifest), "target_train")
    if len(rows) != 6502 or args.budget != 65:
        raise RuntimeError("Frozen Phase 50 budget requires 65 of 6502 target-train patches")
    wrapper_args = argparse.Namespace(
        config=args.config, checkpoint=args.checkpoint, precision="fp16",
        active_block_indices=None,
    )
    wrapper, _ = build_wrapper(wrapper_args)
    scored = score_rows(wrapper, rows)
    selected, quotas = select_diverse(scored, args.budget)
    result = {
        "phase": 50,
        "method": "weak_inconsistency_entropy_biome_diverse_active_selection",
        "target_train_images": len(rows),
        "selected_images": len(selected),
        "selection_fraction": len(selected) / len(rows),
        "budget_definition": "fully_annotated_512x512_target_train_patches",
        "weights": {"weak_top10": 0.4, "flip_photometric_inconsistency": 0.35, "entropy": 0.25, "diversity": 0.5},
        "constraints": {"biome_quotas": quotas, "max_patches_per_scene": 2, "candidate_pool_multiplier": 20},
        "target_labels_read_during_selection": False,
        "checkpoint_sha256": sha256(Path(args.checkpoint)),
        "selected": [
            {key: row[key] for key in ("name", "biome", "scene", "utility", "weak_top10", "inconsistency", "entropy")}
            for row in selected
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "selected"}, indent=2))


if __name__ == "__main__":
    main()
