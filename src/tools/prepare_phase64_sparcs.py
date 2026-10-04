#!/usr/bin/env python3
"""Safely extract, audit, and split the official L8 SPARCS release.

The USGS archive is a validation collection, not an official train/val/test
benchmark.  This tool therefore creates a deterministic project split grouped
by source Landsat product ID.  It never rewrites image or mask pixels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from PIL import Image


SCENE_RE = re.compile(r"(LC8\d{13}[A-Z]{3}\d{2})")
SPLITS = ("target_train", "target_val", "target_test")
NATIVE_CLASS_NAMES = (
    "shadow",
    "shadow_over_water",
    "water",
    "snow",
    "land",
    "cloud",
    "flooded",
)
PARENT_BY_NATIVE = np.asarray([2, 2, 0, 0, 0, 1, 0], dtype=np.uint8)
PARENT_CLASS_NAMES = ("surface_visible", "cloud", "shadow")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_rows(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(",".join(str(row[field]) for field in fields).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def validate_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    files = []
    for member in archive.infolist():
        path = PurePosixPath(member.filename.replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Unsafe archive member: {member.filename}")
        if not member.is_dir():
            files.append(member)
    if not files:
        raise ValueError("SPARCS archive contains no files")
    return files


def safe_extract(archive_path: Path, data_root: Path) -> None:
    data_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        validate_members(archive)
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f"CRC failure in archive member: {bad}")
        archive.extractall(data_root)


def sample_key(path: Path, suffix: str) -> str:
    name = path.name
    if not name.lower().endswith(suffix.lower()):
        raise ValueError(f"{path} does not end with {suffix}")
    return name[: -len(suffix)]


def source_scene(key: str) -> str:
    match = SCENE_RE.search(key.upper())
    if not match:
        raise ValueError(f"Cannot recover Landsat source scene from {key!r}")
    return match.group(1)


def stable_scene_order(scene: str) -> str:
    return hashlib.sha256(f"phase64-sparcs-v1:{scene}".encode("utf-8")).hexdigest()


def assign_splits(scenes: set[str]) -> dict[str, str]:
    ordered = sorted(scenes, key=stable_scene_order)
    if len(ordered) < 10:
        raise ValueError("SPARCS needs at least ten independent source scenes")
    val_count = max(1, round(len(ordered) * 0.125))
    test_count = max(1, round(len(ordered) * 0.125))
    train_count = len(ordered) - val_count - test_count
    if train_count <= 0:
        raise ValueError("No source scenes remain for target_train")
    groups = {
        "target_train": ordered[:train_count],
        "target_val": ordered[train_count : train_count + val_count],
        "target_test": ordered[train_count + val_count :],
    }
    return {scene: split for split, values in groups.items() for scene in values}


def discover_pairs(data_root: Path) -> list[dict[str, Any]]:
    photos = {
        sample_key(path, "_photo.png"): path
        for path in data_root.rglob("*_photo.png")
        if path.is_file()
    }
    masks = {
        sample_key(path, "_mask.png"): path
        for path in data_root.rglob("*_mask.png")
        if path.is_file()
    }
    if set(photos) != set(masks):
        missing_photos = sorted(set(masks) - set(photos))
        missing_masks = sorted(set(photos) - set(masks))
        raise ValueError(
            f"Unpaired SPARCS files: missing_photos={missing_photos[:5]}, "
            f"missing_masks={missing_masks[:5]}"
        )

    all_tiffs = [path for path in data_root.rglob("*.tif") if path.is_file()]
    tiffs_by_key: dict[str, list[Path]] = defaultdict(list)
    for path in all_tiffs:
        lower = path.stem.lower()
        for key in photos:
            if lower.startswith(key.lower()):
                tiffs_by_key[key].append(path)
                break

    rows = []
    for key in sorted(photos):
        candidates = sorted(tiffs_by_key.get(key, []))
        non_qa = [path for path in candidates if "qa" not in path.name.lower()]
        multispectral = non_qa[0] if len(non_qa) == 1 else None
        rows.append(
            {
                "name": key,
                "scene": source_scene(key),
                "image_path": photos[key].resolve().as_posix(),
                "mask_path": masks[key].resolve().as_posix(),
                "multispectral_path": (
                    multispectral.resolve().as_posix() if multispectral else ""
                ),
                "tiff_candidates": len(candidates),
                "non_qa_tiff_candidates": len(non_qa),
            }
        )
    return rows


def audit_dataset(
    archive_path: Path,
    data_root: Path,
    *,
    expected_samples: int = 80,
    expected_size: int = 1000,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with zipfile.ZipFile(archive_path) as archive:
        members = validate_members(archive)
        bad_member = archive.testzip()
    rows = discover_pairs(data_root)
    assignment = assign_splits({row["scene"] for row in rows})
    native_counts = {split: np.zeros(7, dtype=np.int64) for split in SPLITS}
    parent_counts = {split: np.zeros(3, dtype=np.int64) for split in SPLITS}
    shapes = Counter()
    invalid = Counter()
    unreadable = []
    for row in rows:
        row["new_split"] = assignment[row["scene"]]
        try:
            with Image.open(row["mask_path"]) as image:
                mask = np.asarray(image)
        except Exception as exc:  # pragma: no cover - diagnostics for real files
            unreadable.append({"name": row["name"], "error": repr(exc)})
            continue
        if mask.ndim == 3 and mask.shape[2] == 1:
            mask = mask[:, :, 0]
        if mask.ndim != 2:
            unreadable.append({"name": row["name"], "error": f"shape={mask.shape}"})
            continue
        shapes[f"{mask.shape[0]}x{mask.shape[1]}"] += 1
        labels, counts = np.unique(mask, return_counts=True)
        for label, count in zip(labels.tolist(), counts.tolist()):
            label = int(label)
            count = int(count)
            if 0 <= label <= 6:
                native_counts[row["new_split"]][label] += count
                parent_counts[row["new_split"]][int(PARENT_BY_NATIVE[label])] += count
            else:
                invalid[label] += count

    scene_sets = {
        split: {row["scene"] for row in rows if row["new_split"] == split}
        for split in SPLITS
    }
    split_disjoint = all(
        not (scene_sets[left] & scene_sets[right])
        for index, left in enumerate(SPLITS)
        for right in SPLITS[index + 1 :]
    )
    gates = {
        "archive_crc_valid": bad_member is None,
        "exact_sample_count": len(rows) == expected_samples,
        "all_masks_readable_2d": not unreadable,
        "all_masks_expected_size": shapes == {f"{expected_size}x{expected_size}": len(rows)},
        "labels_subset_0_6": not invalid,
        "source_scene_recovered": all(row["scene"] for row in rows),
        "scene_disjoint_split": split_disjoint,
        "all_parents_in_each_split": all(np.all(parent_counts[split] > 0) for split in SPLITS),
    }
    fields = (
        "new_split",
        "scene",
        "name",
        "image_path",
        "mask_path",
        "multispectral_path",
        "tiff_candidates",
        "non_qa_tiff_candidates",
    )
    ordered = sorted(rows, key=lambda row: (SPLITS.index(row["new_split"]), row["scene"], row["name"]))
    summary = {
        "phase": "64-sparcs-protocol",
        "source": "USGS L8 SPARCS validation data",
        "archive_path": archive_path.resolve().as_posix(),
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": sha256_file(archive_path),
        "archive_file_members": len(members),
        "samples": len(rows),
        "source_scenes": len({row["scene"] for row in rows}),
        "split_samples": {split: sum(row["new_split"] == split for row in rows) for split in SPLITS},
        "split_scenes": {split: len(scene_sets[split]) for split in SPLITS},
        "mask_shapes": dict(sorted(shapes.items())),
        "native_pixel_counts": {
            split: {NATIVE_CLASS_NAMES[i]: int(values[i]) for i in range(7)}
            for split, values in native_counts.items()
        },
        "parent_pixel_counts": {
            split: {PARENT_CLASS_NAMES[i]: int(values[i]) for i in range(3)}
            for split, values in parent_counts.items()
        },
        "multispectral_pairing": {
            "exactly_one_non_qa_tiff": sum(row["non_qa_tiff_candidates"] == 1 for row in rows),
            "ambiguous_or_missing": [
                row["name"] for row in rows if row["non_qa_tiff_candidates"] != 1
            ],
        },
        "manifest_sha256": sha256_rows(ordered, fields),
        "diagnostics": {
            "unreadable": unreadable,
            "invalid_label_counts": dict(invalid),
        },
        "gates": gates,
        "passed": all(gates.values()),
    }
    summary["decision"] = (
        "proceed_to_phase64_sparcs_loader"
        if summary["passed"]
        else "stop_and_repair_sparcs_protocol"
    )
    return ordered, summary


def write_outputs(rows: list[dict[str, Any]], summary: dict[str, Any], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    fields = (
        "new_split",
        "scene",
        "name",
        "image_path",
        "mask_path",
        "multispectral_path",
        "tiff_candidates",
        "non_qa_tiff_candidates",
    )
    with (output_root / "scene_disjoint_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)
    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=Path("../data/sparcs/l8cloudmasks.zip"))
    parser.add_argument("--data-root", type=Path, default=Path("../data/sparcs/extracted"))
    parser.add_argument("--output-root", type=Path, default=Path("work_dirs/phase64_sparcs_protocol"))
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--expected-samples", type=int, default=80)
    parser.add_argument("--expected-size", type=int, default=1000)
    args = parser.parse_args()
    if args.extract:
        safe_extract(args.archive, args.data_root)
    rows, summary = audit_dataset(
        args.archive,
        args.data_root,
        expected_samples=args.expected_samples,
        expected_size=args.expected_size,
    )
    write_outputs(rows, summary, args.output_root)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
