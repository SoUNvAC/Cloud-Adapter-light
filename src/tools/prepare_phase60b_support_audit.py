"""Prepare Phase 60B selections and frozen adapted-feature cache.

Oracle selectors intentionally read target-train masks and are tagged as upper
bounds.  Non-oracle selectors never inspect those masks.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = Path(__file__).resolve().parent
for item in (REPO_ROOT, TOOLS_ROOT):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from cache_phase56_readout_features import cache_model_split  # noqa: E402
from phase56_protocol import load_source_order_mask, sha256  # noqa: E402
from select_phase50_active_1pct import (  # noqa: E402
    largest_remainder_quotas,
    score_rows,
    stable_unit,
    zscore,
)
from export_phase12_onnx import build_wrapper  # noqa: E402


BUDGETS = (16, 32, 65, 130, 325)


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def eligible_rows(manifest, metadata):
    status = {
        row["scene"]: row["usgs_shadows"].strip().lower()
        for row in read_rows(metadata)
    }
    rows = [
        row for row in read_rows(manifest)
        if row["new_split"] == "target_train" and status.get(row["scene"]) == "yes"
    ]
    rows.sort(key=lambda row: row["name"])
    if len(rows) != 1560 or len({row["scene"] for row in rows}) != 18:
        raise RuntimeError(f"Expected 1,560 patches/18 Shadows?=yes scenes, got {len(rows)}")
    return rows


def normalized_features(records):
    values = np.asarray([row["feature"] for row in records], dtype=np.float64)
    return (values - values.mean(0)) / np.maximum(values.std(0), 1e-12)


def farthest_order(features, priority=None):
    count = len(features)
    center = features.mean(0)
    first = int(np.argmax(np.linalg.norm(features - center, axis=1)))
    selected = [first]
    minimum = np.linalg.norm(features - features[first], axis=1)
    while len(selected) < count:
        score = minimum.copy()
        if priority is not None:
            distance_z = zscore(score)
            score = np.asarray(priority) + 0.5 * distance_z
        score[selected] = -np.inf
        winner = int(np.argmax(score))
        selected.append(winner)
        minimum = np.minimum(minimum, np.linalg.norm(features - features[winner], axis=1))
    return selected


def phase50_rebased(records, features, budget):
    weak = zscore([row["weak_top10"] for row in records])
    inconsistency = zscore([row["inconsistency"] for row in records])
    entropy = zscore([row["entropy"] for row in records])
    utility = 0.4 * weak + 0.35 * inconsistency + 0.25 * entropy
    quotas = largest_remainder_quotas(records, budget)
    scene_cap = int(math.ceil(budget / len({row["scene"] for row in records}))) + 1
    selected, scene_counts = [], Counter()
    for biome, quota in sorted(quotas.items()):
        indices = [index for index, row in enumerate(records) if row["biome"] == biome]
        candidates = sorted(indices, key=lambda i: (-utility[i], records[i]["name"]))
        pool = candidates[:max(quota * 20, quota)]
        chosen = []
        while len(chosen) < quota:
            eligible = [i for i in pool if i not in chosen and scene_counts[records[i]["scene"]] < scene_cap]
            if not eligible:
                eligible = [i for i in candidates if i not in chosen]
            if not eligible:
                raise RuntimeError(f"Cannot fill rebased Phase 50 quota for {biome}")
            if not chosen:
                winner = eligible[0]
            else:
                distance = np.asarray([
                    np.linalg.norm(features[chosen] - features[i], axis=1).min()
                    for i in eligible
                ])
                scores = utility[eligible] + 0.5 * distance
                winner = eligible[int(np.argmax(scores))]
            chosen.append(winner)
            scene_counts[records[winner]["scene"]] += 1
        selected.extend(chosen)
    return sorted(selected, key=lambda i: records[i]["name"]), quotas, scene_cap


def label_statistics(rows):
    stats = []
    for index, row in enumerate(rows, 1):
        counts = np.bincount(load_source_order_mask(row).reshape(-1), minlength=4).astype(np.float64)
        fractions = counts / counts.sum()
        thin, shadow = fractions[2], fractions[3]
        relation = 2.0 * thin * shadow / max(thin + shadow, 1e-12)
        stats.append({"counts": counts, "fractions": fractions, "relation": relation})
        if index % 250 == 0 or index == len(rows):
            print(f"oracle label audit: {index}/{len(rows)}", flush=True)
    return stats


def oracle_class_balanced_order(stats, records):
    fractions = np.stack([item["fractions"] for item in stats])
    global_frequency = fractions.mean(0).clip(min=1e-9)
    rarity = 1.0 / global_frequency
    selected, accumulated = [], np.zeros(4, dtype=np.float64)
    remaining = set(range(len(records)))
    while remaining:
        candidates = np.asarray(sorted(remaining))
        need = rarity / np.sqrt(accumulated + 1e-6)
        score = fractions[candidates] @ need
        winner = int(candidates[int(np.argmax(score))])
        selected.append(winner)
        accumulated += fractions[winner]
        remaining.remove(winner)
    return selected


def oracle_relation_order(stats, records, features):
    relation = np.asarray([item["relation"] for item in stats])
    fractions = np.stack([item["fractions"] for item in stats])
    relation_z = zscore(relation)
    selected, remaining = [], set(range(len(records)))
    covered_scenes, covered_biomes = set(), set()
    minimum = np.full(len(records), np.inf)
    while remaining:
        candidates = np.asarray(sorted(remaining))
        if selected:
            diversity = zscore(minimum[candidates])
        else:
            diversity = np.zeros(len(candidates))
        both = ((fractions[candidates, 2] > 0) & (fractions[candidates, 3] > 0)).astype(float)
        novelty = np.asarray([
            (records[i]["scene"] not in covered_scenes) + 0.5 * (records[i]["biome"] not in covered_biomes)
            for i in candidates
        ])
        score = 4.0 * both + relation_z[candidates] + 0.25 * diversity + 0.25 * novelty
        winner = int(candidates[int(np.argmax(score))])
        selected.append(winner)
        remaining.remove(winner)
        covered_scenes.add(records[winner]["scene"])
        covered_biomes.add(records[winner]["biome"])
        minimum = np.minimum(minimum, np.linalg.norm(features - features[winner], axis=1))
    return selected


def summarize_names(records, indices):
    return [records[index]["name"] for index in indices]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="work_dirs/phase45_protocol_audit/scene_disjoint_manifest.csv")
    parser.add_argument("--metadata", default="research_plans/protocol_data/l8_biome_usgs_shadow_status.csv")
    parser.add_argument("--phase50-selection", default="work_dirs/phase50_active_1pct/selection.json")
    parser.add_argument("--source-config", default="configs/protocol/phase22_clean_v8_l1c.py")
    parser.add_argument("--source-checkpoint", default="work_dirs/phase22_clean_v8/seed42/best_mIoU_iter_40000.pth")
    parser.add_argument("--adapted-config", default="configs/protocol/phase52a_fix_shadow_status_1pct_l1c.py")
    parser.add_argument("--adapted-checkpoint", default="work_dirs/phase52a_fix_shadow_status_1pct/seed52/best_mIoU_iter_1000.pth")
    parser.add_argument("--output-root", default="work_dirs/phase60")
    parser.add_argument("--cap-per-class", type=int, default=32)
    # Seed 56 deliberately matches the frozen Phase 56 validation cache, so the
    # support audit changes only selected images rather than sampled pixels.
    parser.add_argument("--seed", type=int, default=56)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rows = eligible_rows(args.manifest, args.metadata)
    score_cache = output_root / "phase60b_pool_scores.json"
    if score_cache.is_file():
        scored = json.loads(score_cache.read_text(encoding="utf-8"))["records"]
        if [row["name"] for row in scored] != [row["name"] for row in rows]:
            raise RuntimeError("Existing pool-score cache does not match the frozen eligible pool")
        for scored_row, row in zip(scored, rows):
            scored_row.update(row)
    else:
        wrapper_args = argparse.Namespace(
            config=args.source_config, checkpoint=args.source_checkpoint,
            precision="fp16", active_block_indices=None,
        )
        wrapper, _ = build_wrapper(wrapper_args)
        scored = score_rows(wrapper, rows)
        score_cache.write_text(json.dumps({
            "phase": "60B-pool-scores", "eligible_images": len(rows),
            "source_checkpoint_sha256": sha256(args.source_checkpoint),
            "target_labels_read": False,
            "records": [{
                key: row[key] for key in (
                    "name", "biome", "scene", "weak_top10", "inconsistency", "entropy", "feature"
                )
            } for row in scored],
        }, indent=2), encoding="utf-8")
        del wrapper
        torch.cuda.empty_cache()

    features = normalized_features(scored)
    random_order = sorted(range(len(rows)), key=lambda i: (stable_unit(rows[i]["name"]), rows[i]["name"]))
    entropy_order = sorted(range(len(rows)), key=lambda i: (-scored[i]["entropy"], rows[i]["name"]))
    core_order = farthest_order(features)
    hybrid_order = farthest_order(features, zscore([row["entropy"] for row in scored]))
    oracle_stats = label_statistics(rows)
    class_order = oracle_class_balanced_order(oracle_stats, rows)
    relation_order = oracle_relation_order(oracle_stats, rows, features)

    methods = {
        "random": {"oracle": False, "order": random_order},
        "entropy": {"oracle": False, "order": entropy_order},
        "diversity_core_set": {"oracle": False, "order": core_order},
        "uncertainty_plus_diversity": {"oracle": False, "order": hybrid_order},
        "oracle_class_balanced": {"oracle": True, "order": class_order},
        "oracle_thin_shadow_relation": {"oracle": True, "order": relation_order},
    }
    selections = {}
    for method, descriptor in methods.items():
        selections[method] = {
            "oracle_upper_bound_only": descriptor["oracle"],
            "budgets": {str(budget): summarize_names(rows, descriptor["order"][:budget]) for budget in BUDGETS},
        }
    selections["phase50_policy_rebased"] = {
        "oracle_upper_bound_only": False,
        "warning": "Phase 50 max-2-per-scene is infeasible above 36 in the 18-scene legal pool; scene cap is rebased by budget.",
        "budgets": {}, "details": {},
    }
    for budget in BUDGETS:
        indices, quotas, scene_cap = phase50_rebased(scored, features, budget)
        selections["phase50_policy_rebased"]["budgets"][str(budget)] = summarize_names(rows, indices)
        selections["phase50_policy_rebased"]["details"][str(budget)] = {
            "biome_quotas": quotas, "scene_cap": scene_cap,
        }

    frozen_phase50 = json.loads(Path(args.phase50_selection).read_text(encoding="utf-8"))
    eligible_names = {row["name"] for row in rows}
    exact_names = sorted(row["name"] for row in frozen_phase50["selected"] if row["name"] in eligible_names)
    if len(exact_names) != 12:
        raise RuntimeError(f"Expected 12 legal images in frozen Phase 50 selection, got {len(exact_names)}")
    selections["phase50_exact_nominal65"] = {
        "oracle_upper_bound_only": False,
        "nominal_budget": 65,
        "usable_images": 12,
        "names": exact_names,
        "warning": "53/65 images are excluded because their scenes have no verified shadow annotation.",
    }

    all_names = set(exact_names)
    for descriptor in selections.values():
        for names in descriptor.get("budgets", {}).values():
            all_names.update(names)
    union_rows = [row for row in rows if row["name"] in all_names]
    document = {
        "phase": "60B-selection",
        "eligible_pool": {
            "images": len(rows), "scenes": len({row["scene"] for row in rows}),
            "filter": "target_train and USGS Shadows?=yes only",
            "requested_budgets": list(BUDGETS),
            "fractions_of_eligible_pool": {str(b): b / len(rows) for b in BUDGETS},
            "original_6502_denominator_fractions": {str(b): b / 6502 for b in BUDGETS},
        },
        "selection_protocol": {
            "non_oracle_labels_read": False,
            "oracle_labels_read": True,
            "oracle_allowed_in_final_method": False,
            "manifest_sha256": sha256(args.manifest),
            "metadata_sha256": sha256(args.metadata),
            "source_checkpoint_sha256": sha256(args.source_checkpoint),
            "sampling_seed": args.seed,
        },
        "selections": selections,
        "union_images": len(union_rows),
    }
    selection_path = output_root / "phase60b_selections.json"
    selection_path.write_text(json.dumps(document, indent=2), encoding="utf-8")

    feature_root = output_root / "phase60b_feature_cache"
    metadata_path = feature_root / "adapted" / "target_train_union" / "metadata.json"
    if metadata_path.is_file():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing.get("image_ids") != [row["name"] for row in union_rows]:
            raise RuntimeError("Existing Phase 60B feature cache has a different union")
        cache_meta = existing
    else:
        cache_meta = cache_model_split(
            "adapted", args.adapted_config, args.adapted_checkpoint,
            union_rows, "target_train_union", feature_root,
            args.cap_per_class, args.seed,
        )
    document["feature_cache"] = cache_meta
    selection_path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    print(json.dumps({
        "selection": str(selection_path), "selection_sha256": sha256(selection_path),
        "eligible": len(rows), "union": len(union_rows), "feature_cache": str(metadata_path),
    }, indent=2))


if __name__ == "__main__":
    main()
