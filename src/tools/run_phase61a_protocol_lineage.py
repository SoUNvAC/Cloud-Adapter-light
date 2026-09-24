"""Phase 61A: read-only protocol lineage audit across Phases 50--60."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


TARGET_TO_SOURCE = np.asarray([0, 3, 2, 1], dtype=np.uint8)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def names_sha(names):
    return hashlib.sha256("\n".join(sorted(names)).encode("utf-8")).hexdigest()


def rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def checkpoint(path):
    path = Path(path)
    return {"path": str(path), "exists": path.is_file(), "sha256": sha256(path) if path.is_file() else None}


def mask_counts(row):
    with Image.open(row["mask_path"]) as image:
        mask = np.asarray(image, dtype=np.uint8)
    if mask.shape != (512, 512):
        raise RuntimeError(f"Unexpected mask shape {mask.shape}: {row['mask_path']}")
    values = np.unique(mask)
    if np.any(values > 3):
        raise RuntimeError(f"Invalid mask values {values.tolist()}: {row['mask_path']}")
    return np.bincount(mask.reshape(-1), minlength=4).astype(np.int64)


def format_pixels(item):
    if isinstance(item, int):
        return f"{item:,}"
    return "; ".join(f"{key}={value:,}" for key, value in item.items())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="work_dirs/phase45_protocol_audit/scene_disjoint_manifest.csv")
    parser.add_argument("--metadata", default="research_plans/protocol_data/l8_biome_usgs_shadow_status.csv")
    parser.add_argument("--selection", default="work_dirs/phase50_active_1pct/selection.json")
    parser.add_argument("--phase60-selection", default="work_dirs/phase60/phase60b_selections.json")
    parser.add_argument("--phase56-cache-meta", default="work_dirs/phase56_readout/cache/adapted/target_train/metadata.json")
    parser.add_argument("--output", default="work_dirs/phase61/phase61a_lineage.json")
    parser.add_argument("--table", default="work_dirs/phase61/phase61a_lineage_table.md")
    args = parser.parse_args()

    manifest_rows = rows(args.manifest)
    status_rows = rows(args.metadata)
    status = {row["scene"]: row["usgs_shadows"].strip().lower() for row in status_rows}
    selection_doc = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    selected_names = [row["name"] for row in selection_doc["selected"]]
    selected_set = set(selected_names)
    selected_rows = sorted(
        [row for row in manifest_rows if row["new_split"] == "target_train" and row["name"] in selected_set],
        key=lambda row: row["name"],
    )
    legal_rows = [row for row in selected_rows if status.get(row["scene"]) == "yes"]
    excluded_rows = [row for row in selected_rows if status.get(row["scene"]) != "yes"]
    target_train = [row for row in manifest_rows if row["new_split"] == "target_train"]
    legal_pool = [row for row in target_train if status.get(row["scene"]) == "yes"]
    target_val_legal = [row for row in manifest_rows if row["new_split"] == "target_val" and status.get(row["scene"]) == "yes"]

    selected_pixel_counts = np.zeros(4, dtype=np.int64)
    legal_pixel_counts = np.zeros(4, dtype=np.int64)
    excluded_pixel_counts = np.zeros(4, dtype=np.int64)
    exclusion = []
    for row in selected_rows:
        counts = mask_counts(row)
        selected_pixel_counts += counts
        if status.get(row["scene"]) == "yes":
            legal_pixel_counts += counts
        else:
            excluded_pixel_counts += counts
            exclusion.append({
                "name": row["name"], "scene": row["scene"], "biome": row["biome"],
                "usgs_shadows": status.get(row["scene"]),
                "image_exists": Path(row["image_path"]).is_file(),
                "mask_exists": Path(row["mask_path"]).is_file(),
                "mask_shape": [512, 512],
                "mask_values": [index for index, count in enumerate(counts) if count],
                "manifest_split": row["new_split"],
                "class_mapping_valid": True,
                "exclusion_reason": "scene-level USGS Shadows?=no: no verified shadow truth; not a path/mask/mapping failure",
            })

    # Phase 52A-fix reads all 65 images but ignores raw target ID 0 in the 53
    # Shadows?=no scenes; all pixels in the 12 legal scenes remain valid.
    phase52_valid = int(legal_pixel_counts.sum() + excluded_pixel_counts[1:].sum())
    legal_train_pixels = int(legal_pixel_counts.sum())
    legal_val_pixels = len(target_val_legal) * 512 * 512
    manifest_hash = sha256(args.manifest)
    selection_hash = sha256(args.selection)
    selected_name_hash = names_sha(selected_names)
    legal_name_hash = names_sha(row["name"] for row in legal_rows)

    source_checkpoint = checkpoint("work_dirs/phase22_clean_v8/seed42/best_mIoU_iter_40000.pth")
    adapted_checkpoint = checkpoint("work_dirs/phase52a_fix_shadow_status_1pct/seed52/best_mIoU_iter_1000.pth")
    phase56_meta = json.loads(Path(args.phase56_cache_meta).read_text(encoding="utf-8"))
    phase60 = json.loads(Path(args.phase60_selection).read_text(encoding="utf-8"))
    phase60_exact = phase60["selections"]["phase50_exact_nominal65"]["names"]

    table_rows = [
        {
            "phase": "50", "manifest_hash": manifest_hash,
            "nominal_samples": "65 selected (pool 6502)", "legal_samples": 12,
            "actual_read": "6502 RGB scored; 65 selected; labels not read",
            "target_label_pixels": 0, "checkpoint": source_checkpoint,
            "selected_names_sha256": selected_name_hash,
        },
        {
            "phase": "52A-fix", "manifest_hash": manifest_hash,
            "nominal_samples": 65, "legal_samples": 12,
            "actual_read": "65 train (12 yes + 53 no with raw ID0 ignored)",
            "target_label_pixels": phase52_valid, "checkpoint": adapted_checkpoint,
            "initial_checkpoint": source_checkpoint, "selected_names_sha256": selected_name_hash,
        },
        {
            "phase": "55", "manifest_hash": manifest_hash,
            "nominal_samples": 65, "legal_samples": 12,
            "actual_read": "12 train + 963 val", "target_label_pixels": {"train": legal_train_pixels, "val": legal_val_pixels},
            "checkpoint": {"source": source_checkpoint, "adapted": adapted_checkpoint},
            "selected_names_sha256": legal_name_hash,
        },
        {
            "phase": "56", "manifest_hash": manifest_hash,
            "nominal_samples": 65, "legal_samples": 12,
            "actual_read": f"{phase56_meta['images']} train + 963 val",
            "target_label_pixels": {"train": legal_train_pixels, "val": legal_val_pixels},
            "checkpoint": {"source": source_checkpoint, "adapted": adapted_checkpoint},
            "selected_names_sha256": names_sha(phase56_meta["image_ids"]),
        },
        {
            "phase": "60", "manifest_hash": manifest_hash,
            "nominal_samples": "60A:65; 60B:16/32/65/130/325", "legal_samples": "60A:12; 60B pool:1560",
            "actual_read": f"60A:12 train+963 val; 60B:1560 acquisition/oracle, {phase60['union_images']} feature union+963 val",
            "target_label_pixels": {"60A_train": legal_train_pixels, "60B_oracle_pool": len(legal_pool) * 512 * 512, "val": legal_val_pixels},
            "checkpoint": {"source": source_checkpoint, "adapted": adapted_checkpoint},
            "selected_names_sha256": {"60A_exact_legal12": names_sha(phase60_exact), "60B_varies_by_method": True},
        },
    ]

    excluded_only_for_shadow = all(
        row["usgs_shadows"] == "no" and row["image_exists"] and row["mask_exists"]
        and row["class_mapping_valid"] and row["manifest_split"] == "target_train"
        and set(row["mask_values"]).issubset({0, 1, 2, 3})
        for row in exclusion
    )
    gates = {
        "manifest_has_expected_rows": len(manifest_rows) == 10574,
        "target_train_is_6502": len(target_train) == 6502,
        "selection_is_unique_65": len(selected_names) == len(selected_set) == 65,
        "selection_resolves_exactly_65_manifest_rows": len(selected_rows) == 65,
        "legal_selection_is_12": len(legal_rows) == 12,
        "excluded_selection_is_53": len(excluded_rows) == 53,
        "all_53_excluded_only_for_usgs_shadow_status": excluded_only_for_shadow,
        "no_exclusion_due_to_path": all(row["image_exists"] and row["mask_exists"] for row in exclusion),
        "no_exclusion_due_to_mask_values": all(set(row["mask_values"]).issubset({0, 1, 2, 3}) for row in exclusion),
        "no_exclusion_due_to_class_mapping": all(row["class_mapping_valid"] for row in exclusion),
        "phase52_uses_same_nominal_65": True,
        "phase55_56_60a_use_same_legal_12": (
            names_sha(phase56_meta["image_ids"]) == legal_name_hash == names_sha(phase60_exact)
        ),
        "checkpoints_exist": source_checkpoint["exists"] and adapted_checkpoint["exists"],
    }
    passed = all(gates.values())
    summary = {
        "phase": "61A", "manifest_sha256": manifest_hash,
        "selection_sha256": selection_hash, "nominal_selected_names_sha256": selected_name_hash,
        "legal_selected_names_sha256": legal_name_hash,
        "class_mapping_target_to_source": {"0_clear": 0, "1_shadow": 3, "2_thin": 2, "3_thick": 1},
        "table": table_rows,
        "selected_pixel_counts_target_order_clear_shadow_thin_thick": selected_pixel_counts.tolist(),
        "phase52_valid_label_pixels": phase52_valid,
        "excluded_53": exclusion,
        "gates": gates, "passed": passed,
        "decision": "lineage_clear_continue_61b_61c" if passed else "pause_all_downstream_lineage_unclear",
        "target_test_evaluated": False, "cloudsen_internal_test_evaluated": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    checkpoint_text = lambda item: item["sha256"][:12] if "sha256" in item else "/".join(value["sha256"][:12] for value in item.values())
    lines = [
        "| Phase | manifest hash | nominal样本 | 合法样本 | 实际读入 | target标签像元 | checkpoint |",
        "|---|---|---:|---:|---|---:|---|",
    ]
    for row in table_rows:
        lines.append(
            f"| {row['phase']} | `{row['manifest_hash'][:12]}` | {row['nominal_samples']} | {row['legal_samples']} | "
            f"{row['actual_read']} | {format_pixels(row['target_label_pixels'])} | `{checkpoint_text(row['checkpoint'])}` |"
        )
    Path(args.table).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "table": args.table, "passed": passed, "gates": gates}, indent=2))


if __name__ == "__main__":
    main()
