"""Shared, deterministic protocol helpers for Phase 56.

The Landsat-8 Biome ``Shadows?=no`` scenes do not contain a verified shadow
annotation.  They are therefore excluded from every Phase 56 fit/evaluation
set rather than being interpreted as shadow-negative examples.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

SOURCE_CLASS_NAMES = ("clear", "thick_cloud", "thin_cloud", "cloud_shadow")
TARGET_TO_SOURCE = np.asarray([0, 3, 2, 1], dtype=np.int64)
INJECTION_INDICES = (2, 5, 8, 11)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_shadow_yes_rows(manifest, metadata, selection):
    with Path(metadata).open(newline="", encoding="utf-8") as handle:
        status_rows = list(csv.DictReader(handle))
    status = {row["scene"]: row["usgs_shadows"].strip().lower() for row in status_rows}
    if len(status) != 96 or set(status.values()) != {"yes", "no"}:
        raise RuntimeError("Unexpected Landsat-8 Biome Shadows? metadata snapshot")

    selected_doc = json.loads(Path(selection).read_text(encoding="utf-8"))
    selected = {row["name"] for row in selected_doc["selected"]}
    if len(selected) != 65:
        raise RuntimeError(f"Expected the frozen 65-image selection, found {len(selected)}")

    with Path(manifest).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    train = [
        row for row in rows
        if row["new_split"] == "target_train"
        and row["name"] in selected
        and status.get(row["scene"]) == "yes"
    ]
    validation = [
        row for row in rows
        if row["new_split"] == "target_val" and status.get(row["scene"]) == "yes"
    ]
    train.sort(key=lambda row: row["name"])
    validation.sort(key=lambda row: row["name"])
    if len(train) != 12:
        raise RuntimeError(f"Expected 12 selected Shadows?=yes patches, found {len(train)}")
    if len(validation) != 963 or len({row["scene"] for row in validation}) != 8:
        raise RuntimeError("Expected 963 validation patches from 8 Shadows?=yes scenes")
    return train, validation, status


def load_source_order_mask(row) -> np.ndarray:
    with Image.open(row["mask_path"]) as image:
        target = np.asarray(image, dtype=np.uint8).copy()
    if target.ndim != 2 or np.any(target > 3):
        raise RuntimeError(f"Invalid four-class mask: {row['mask_path']}")
    return TARGET_TO_SOURCE[target]


def deterministic_samples(mask, image_id, cap_per_class=32, seed=56):
    """Return original-resolution coordinates with a fixed per-class cap."""
    coordinates, labels = [], []
    for class_index in range(4):
        candidates = np.argwhere(mask == class_index)
        token = f"{seed}:{image_id}:{class_index}".encode("utf-8")
        class_seed = int.from_bytes(hashlib.sha256(token).digest()[:8], "little")
        generator = np.random.default_rng(class_seed)
        if len(candidates) > cap_per_class:
            candidates = candidates[
                generator.choice(len(candidates), cap_per_class, replace=False)
            ]
        coordinates.append(candidates.astype(np.int32, copy=False))
        labels.append(np.full(len(candidates), class_index, dtype=np.uint8))
    if not coordinates:
        return np.empty((0, 2), np.int32), np.empty(0, np.uint8)
    return np.concatenate(coordinates), np.concatenate(labels)


def sample_spatial(feature, coordinates, original_shape):
    """Nearest-cell deterministic sampling from BCHW or BNC tensors."""
    array = feature.detach()
    if array.ndim == 3:
        side = int(round(array.shape[1] ** 0.5))
        if side * side != array.shape[1]:
            raise RuntimeError(f"Token count {array.shape[1]} is not square")
        array = array[0].reshape(side, side, array.shape[2])
    elif array.ndim == 4:
        array = array[0].permute(1, 2, 0)
    else:
        raise RuntimeError(f"Expected BNC/BCHW feature, got {tuple(array.shape)}")
    height, width = array.shape[:2]
    source_h, source_w = original_shape
    yy = np.minimum(coordinates[:, 0] * height // source_h, height - 1)
    xx = np.minimum(coordinates[:, 1] * width // source_w, width - 1)
    return array[yy, xx]
