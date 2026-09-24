"""Phase 56C label support/co-occurrence audit without using unlabelled-shadow scenes."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from phase56_protocol import load_source_order_mask, read_shadow_yes_rows, sha256  # noqa: E402


def category(mask):
    thin, shadow = bool(np.any(mask == 2)), bool(np.any(mask == 3))
    if thin and shadow:
        return "thin_and_shadow"
    if thin:
        return "thin_only"
    if shadow:
        return "shadow_only"
    return "neither"


def summarize(rows):
    result = {
        key: {"patches": 0, "pixels": 0, "thin_pixels": 0, "shadow_pixels": 0}
        for key in ("shadow_only", "thin_only", "thin_and_shadow", "neither")
    }
    class_pixels = np.zeros(4, dtype=np.int64)
    per_scene = {}
    for row in rows:
        mask = load_source_order_mask(row)
        key = category(mask)
        counts = np.bincount(mask.reshape(-1), minlength=4)
        class_pixels += counts
        result[key]["patches"] += 1
        result[key]["pixels"] += int(mask.size)
        result[key]["thin_pixels"] += int(counts[2])
        result[key]["shadow_pixels"] += int(counts[3])
        scene = per_scene.setdefault(row["scene"], {"patches": 0, "thin_and_shadow": 0})
        scene["patches"] += 1
        scene["thin_and_shadow"] += key == "thin_and_shadow"
    return {
        "cooccurrence": result,
        "class_pixels_source_order": class_pixels.tolist(),
        "scenes": per_scene,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="work_dirs/phase45_protocol_audit/scene_disjoint_manifest.csv")
    parser.add_argument("--metadata", default="research_plans/protocol_data/l8_biome_usgs_shadow_status.csv")
    parser.add_argument("--selection", default="work_dirs/phase50_active_1pct/selection.json")
    parser.add_argument("--output", default="work_dirs/phase56_readout/label_support_summary.json")
    args = parser.parse_args()
    train, validation, status = read_shadow_yes_rows(args.manifest, args.metadata, args.selection)

    selected = json.loads(Path(args.selection).read_text(encoding="utf-8"))["selected"]
    selected_names = {row["name"] for row in selected}
    with Path(args.manifest).open(newline="", encoding="utf-8") as handle:
        all_selected = [
            row for row in csv.DictReader(handle)
            if row["new_split"] == "target_train" and row["name"] in selected_names
        ]
    excluded = [row for row in all_selected if status[row["scene"]] == "no"]
    train_summary = summarize(train)
    cooccurrence_patches = train_summary["cooccurrence"]["thin_and_shadow"]["patches"]
    summary = {
        "phase": "56C",
        "protocol": {
            "selected_total": len(all_selected),
            "selected_shadow_yes_used": len(train),
            "selected_shadow_no_excluded": len(excluded),
            "validation_shadow_yes_used": len(validation),
            "shadow_no_pixel_labels_read": False,
            "manifest_sha256": sha256(args.manifest),
            "metadata_sha256": sha256(args.metadata),
            "selection_sha256": sha256(args.selection),
        },
        "target_train_shadow_yes": train_summary,
        "target_val_shadow_yes": summarize(validation),
        "gradient_observation_upper_bound": {
            "eligible_patches": len(train),
            "thin_and_shadow_patches": cooccurrence_patches,
            "note": "This is label support, not an observed optimizer-batch frequency.",
        },
        "claims": {
            "cooccurrence_generalization_allowed": cooccurrence_patches > 5,
            "reason": (
                "More than five independent co-occurrence patches are available."
                if cooccurrence_patches > 5 else
                "At most five co-occurrence patches; do not state a population-level gradient rule."
            ),
        },
        "target_test_evaluated": False,
        "cloudsen_internal_test_evaluated": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
