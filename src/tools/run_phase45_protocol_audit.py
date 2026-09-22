import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


PATCH_RE = re.compile(
    r"^(?P<biome>.+)_(?P<scene>[^_]+)_patch_(?P<x>\d+)_(?P<y>\d+)\.[^.]+$"
)
CLASS_NAMES = ("clear", "cloud_shadow", "thin_cloud", "cloud")
SPLIT_NAMES = ("target_train", "target_val", "target_test")


def sha256_text(lines):
    digest = hashlib.sha256()
    for line in lines:
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def stable_scene_order(scene):
    return hashlib.sha256(f"phase45-v1:{scene}".encode("utf-8")).hexdigest()


def split_scenes(records):
    by_biome = defaultdict(set)
    for row in records:
        by_biome[row["biome"]].add(row["scene"])
    assignment = {}
    allocation = {}
    for biome, scene_set in sorted(by_biome.items()):
        scenes = sorted(scene_set, key=stable_scene_order)
        count = len(scenes)
        if count < 3:
            raise ValueError(f"Biome {biome!r} has fewer than three scenes")
        val_count = max(1, round(count * 0.2))
        test_count = max(1, round(count * 0.2))
        if val_count + test_count >= count:
            val_count = test_count = 1
        train_count = count - val_count - test_count
        groups = {
            "target_train": scenes[:train_count],
            "target_val": scenes[train_count:train_count + val_count],
            "target_test": scenes[train_count + val_count:],
        }
        allocation[biome] = {key: len(value) for key, value in groups.items()}
        for split, values in groups.items():
            for scene in values:
                if scene in assignment:
                    raise ValueError(f"Scene {scene!r} belongs to multiple biomes")
                assignment[scene] = split
    return assignment, allocation


def boundary_mask(mask):
    boundary = np.zeros(mask.shape, dtype=bool)
    boundary[1:, :] |= mask[1:, :] != mask[:-1, :]
    boundary[:-1, :] |= mask[:-1, :] != mask[1:, :]
    boundary[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    boundary[:, :-1] |= mask[:, :-1] != mask[:, 1:]
    return boundary


def git_value(*args):
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/l8_biome")
    parser.add_argument("--output-root", default="work_dirs/phase45_protocol_audit")
    args = parser.parse_args()
    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    records = []
    unmatched = []
    missing_pairs = []
    for old_split in ("train", "test"):
        image_dir = data_root / "img_dir" / old_split
        mask_dir = data_root / "ann_dir" / old_split
        image_paths = {path.name: path for path in image_dir.glob("*") if path.is_file()}
        mask_paths = {path.name: path for path in mask_dir.glob("*") if path.is_file()}
        for name in sorted(set(image_paths) ^ set(mask_paths)):
            missing_pairs.append({"old_split": old_split, "name": name})
        for name in sorted(set(image_paths) & set(mask_paths)):
            match = PATCH_RE.match(name)
            if not match:
                unmatched.append({"old_split": old_split, "name": name})
                continue
            parsed = match.groupdict()
            records.append({
                "name": name,
                "old_split": old_split,
                "biome": parsed["biome"],
                "scene": parsed["scene"],
                "x": int(parsed["x"]),
                "y": int(parsed["y"]),
                "image_path": image_paths[name].as_posix(),
                "mask_path": mask_paths[name].as_posix(),
            })

    name_counts = Counter(row["name"] for row in records)
    patch_counts = Counter((row["scene"], row["x"], row["y"]) for row in records)
    duplicate_names = sorted(name for name, count in name_counts.items() if count > 1)
    duplicate_patch_keys = sorted(
        f"{scene}:{x}:{y}" for (scene, x, y), count in patch_counts.items() if count > 1
    )
    old_scene_sets = {
        split: {row["scene"] for row in records if row["old_split"] == split}
        for split in ("train", "test")
    }
    historical_overlap = sorted(old_scene_sets["train"] & old_scene_sets["test"])

    assignment, allocation = split_scenes(records)
    for row in records:
        row["new_split"] = assignment[row["scene"]]

    pixel_counts = {split: np.zeros(4, dtype=np.int64) for split in SPLIT_NAMES}
    boundary_counts = {split: np.zeros(4, dtype=np.int64) for split in SPLIT_NAMES}
    invalid_label_counts = Counter()
    unreadable_masks = []
    shapes = Counter()
    for index, row in enumerate(records, start=1):
        try:
            mask = np.asarray(Image.open(row["mask_path"]))
        except Exception as exc:
            unreadable_masks.append({"name": row["name"], "error": repr(exc)})
            continue
        if mask.ndim != 2:
            unreadable_masks.append({"name": row["name"], "error": f"shape={mask.shape}"})
            continue
        shapes[f"{mask.shape[0]}x{mask.shape[1]}"] += 1
        labels, counts = np.unique(mask, return_counts=True)
        for label, count in zip(labels.tolist(), counts.tolist()):
            if 0 <= int(label) < 4:
                pixel_counts[row["new_split"]][int(label)] += int(count)
            else:
                invalid_label_counts[int(label)] += int(count)
        edge = boundary_mask(mask)
        for label in range(4):
            boundary_counts[row["new_split"]][label] += int(np.count_nonzero(edge & (mask == label)))
        if index % 1000 == 0:
            print(f"audited_masks={index}/{len(records)}", flush=True)

    scene_sets = {
        split: {row["scene"] for row in records if row["new_split"] == split}
        for split in SPLIT_NAMES
    }
    split_scene_disjoint = all(
        not (scene_sets[left] & scene_sets[right])
        for index, left in enumerate(SPLIT_NAMES)
        for right in SPLIT_NAMES[index + 1:]
    )
    split_rows = {split: [row for row in records if row["new_split"] == split]
                  for split in SPLIT_NAMES}
    all_biomes = {row["biome"] for row in records}
    biome_coverage = {
        split: sorted({row["biome"] for row in rows})
        for split, rows in split_rows.items()
    }

    manifest_fields = (
        "new_split", "old_split", "biome", "scene", "x", "y",
        "name", "image_path", "mask_path",
    )
    manifest_path = output_root / "scene_disjoint_manifest.csv"
    ordered_records = sorted(
        records, key=lambda row: (SPLIT_NAMES.index(row["new_split"]), row["scene"], row["x"], row["y"])
    )
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in manifest_fields} for row in ordered_records)
    for split, rows in split_rows.items():
        lines = [row["name"] for row in sorted(rows, key=lambda item: item["name"])]
        (output_root / f"{split}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    stats = {}
    for split in SPLIT_NAMES:
        total = int(pixel_counts[split].sum())
        stats[split] = {
            "patches": len(split_rows[split]),
            "scenes": len(scene_sets[split]),
            "biomes": biome_coverage[split],
            "pixels": {CLASS_NAMES[i]: int(pixel_counts[split][i]) for i in range(4)},
            "pixel_fraction": {
                CLASS_NAMES[i]: float(pixel_counts[split][i] / total) if total else math.nan
                for i in range(4)
            },
            "boundary_pixels": {
                CLASS_NAMES[i]: int(boundary_counts[split][i]) for i in range(4)
            },
            "interior_pixels": {
                CLASS_NAMES[i]: int(pixel_counts[split][i] - boundary_counts[split][i])
                for i in range(4)
            },
        }

    gates = {
        "records_present": bool(records),
        "all_pairs_complete": not missing_pairs,
        "all_names_parseable": not unmatched,
        "no_duplicate_names": not duplicate_names,
        "no_duplicate_patch_keys": not duplicate_patch_keys,
        "all_masks_readable_2d": not unreadable_masks,
        "labels_subset_0_3": not invalid_label_counts,
        "replacement_scene_disjoint": split_scene_disjoint,
        "all_classes_in_each_split": all(np.all(pixel_counts[split] > 0) for split in SPLIT_NAMES),
        "all_biomes_in_each_split": all(set(values) == all_biomes for values in biome_coverage.values()),
    }
    tracked = git_value("status", "--short", "--untracked-files=no")
    result = {
        "phase": "45A",
        "audit_only": True,
        "data_root": str(data_root.resolve()),
        "git_commit": git_value("rev-parse", "HEAD"),
        "tracked_worktree_clean": tracked == "",
        "historical": {
            "patch_counts": {split: sum(row["old_split"] == split for row in records)
                             for split in ("train", "test")},
            "scene_counts": {split: len(values) for split, values in old_scene_sets.items()},
            "scene_overlap_count": len(historical_overlap),
            "scene_disjoint": not historical_overlap,
        },
        "replacement_allocation_by_biome": allocation,
        "replacement": stats,
        "mask_shapes": dict(sorted(shapes.items())),
        "manifest_sha256": sha256_text(
            ",".join(str(row[key]) for key in manifest_fields) for row in ordered_records
        ),
        "diagnostics": {
            "missing_pairs": missing_pairs[:100],
            "unmatched_names": unmatched[:100],
            "duplicate_names": duplicate_names[:100],
            "duplicate_patch_keys": duplicate_patch_keys[:100],
            "unreadable_masks": unreadable_masks[:100],
            "invalid_label_counts": dict(invalid_label_counts),
        },
        "gates": gates,
        "passed": all(gates.values()),
    }
    result["decision"] = "proceed_to_phase45b" if result["passed"] else "stop_and_repair_protocol"
    (output_root / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    report = [
        "Phase 45A cross-domain protocol audit",
        "=====================================",
        "",
        f"Audit passed: {result['passed']}",
        f"Historical patches: {result['historical']['patch_counts']}",
        f"Historical scenes: {result['historical']['scene_counts']}",
        f"Historical train/test scene overlap: {len(historical_overlap)}",
        f"Replacement patches: { {key: value['patches'] for key, value in stats.items()} }",
        f"Replacement scenes: { {key: value['scenes'] for key, value in stats.items()} }",
        f"Mask shapes: {dict(shapes)}",
        f"Manifest SHA-256: {result['manifest_sha256']}",
        f"Decision: {result['decision']}",
        "Shared dataset files were read only and were not modified.",
    ]
    (output_root / "PHASE45A_REPORT.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

