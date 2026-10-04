#!/usr/bin/env python3
"""Instantiate all Phase 64 datasets and verify paths, counts, and mappings."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

from mmengine.config import Config
from mmseg.registry import DATASETS
from mmseg.utils import register_all_modules


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
SHARED_ROOT = Path("/home/scv/shared")
sys.path.insert(0, str(SRC_ROOT))
import cloud_adapter  # noqa: E402,F401  register project datasets


SOURCE_CONFIG = SRC_ROOT / "configs/protocol/phase64_source_parent_rgb.py"
TARGET_CONFIG = SRC_ROOT / "configs/protocol/phase64_target_parent_rgb.py"
PARENT_MAPS = {
    "cloudsen_train": {0: 0, 1: 1, 2: 1, 3: 2},
    "cloudsen_val": {0: 0, 1: 1, 2: 1, 3: 2},
    "cloudsen_test": {0: 0, 1: 1, 2: 1, 3: 2},
    "l8_train": {0: 0, 1: 2, 2: 1, 3: 1},
    "l8_val": {0: 0, 1: 2, 2: 1, 3: 1},
    "sparcs_train": {0: 2, 1: 2, 2: 0, 3: 0, 4: 0, 5: 1, 6: 0},
    "sparcs_val": {0: 2, 1: 2, 2: 0, 3: 0, 4: 0, 5: 1, 6: 0},
}
EXPECTED_COUNTS = {
    "cloudsen_train": 8490,
    "cloudsen_val": 535,
    "cloudsen_test": 975,
    "l8_val": 963,
    "sparcs_train": 60,
    "sparcs_val": 10,
}


@contextmanager
def scoped_environment(values: dict[str, str]):
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def within_authorized_roots(path: Path) -> bool:
    resolved = path.resolve()
    for root in (REPO_ROOT.resolve(), SHARED_ROOT.resolve()):
        try:
            resolved.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def summarize_dataset(name: str, dataset_config) -> dict:
    dataset = DATASETS.build(dataset_config)
    missing_images = []
    missing_masks = []
    escaped_paths = []
    maps = set()
    for index in range(len(dataset)):
        row = dataset.get_data_info(index)
        image_path = Path(row["img_path"])
        mask_path = Path(row["seg_map_path"])
        if not image_path.is_file():
            missing_images.append(str(image_path))
        if not mask_path.is_file():
            missing_masks.append(str(mask_path))
        for value in (image_path, mask_path):
            if not within_authorized_roots(value):
                escaped_paths.append(str(value.resolve()))
        maps.add(tuple(sorted((int(k), int(v)) for k, v in row["label_map"].items())))

    observed_maps = [dict(values) for values in sorted(maps)]
    expected_count = EXPECTED_COUNTS.get(name)
    gates = {
        "nonempty": len(dataset) > 0,
        "expected_count": expected_count is None or len(dataset) == expected_count,
        "all_images_exist": not missing_images,
        "all_masks_exist": not missing_masks,
        "all_paths_authorized": not escaped_paths,
        "single_frozen_parent_map": observed_maps == [PARENT_MAPS[name]],
        "three_parent_metainfo": tuple(dataset.metainfo["classes"])
        == ("surface_visible", "cloud", "shadow"),
    }
    return {
        "name": name,
        "samples": len(dataset),
        "expected_samples": expected_count,
        "observed_label_maps": observed_maps,
        "missing_images": missing_images[:10],
        "missing_masks": missing_masks[:10],
        "escaped_paths": escaped_paths[:10],
        "gates": gates,
        "passed": all(gates.values()),
    }


def audit() -> dict:
    register_all_modules(init_default_scope=True)
    os.chdir(SRC_ROOT)
    source = Config.fromfile(SOURCE_CONFIG)
    datasets = [
        ("cloudsen_train", source.train_dataloader.dataset),
        ("cloudsen_val", source.val_dataloader.dataset),
        ("cloudsen_test", source.test_dataloader.dataset),
    ]
    for target in ("l8", "sparcs"):
        with scoped_environment(
            {
                "PHASE64_TARGET": target,
                "PHASE64_METHOD": "shared_parent",
                "PHASE64_PARENT_CHECKPOINT": "work_dirs/frozen-parent.pth",
            }
        ):
            config = Config.fromfile(TARGET_CONFIG)
        datasets.extend(
            [
                (f"{target}_train", config.train_dataloader.dataset),
                (f"{target}_val", config.val_dataloader.dataset),
            ]
        )
    rows = [summarize_dataset(name, config) for name, config in datasets]
    summary = {
        "phase": "64-day2-data-readiness",
        "authorized_roots": [
            REPO_ROOT.resolve().as_posix(),
            SHARED_ROOT.resolve().as_posix(),
        ],
        "datasets": rows,
    }
    summary["passed"] = all(row["passed"] for row in rows)
    summary["decision"] = (
        "proceed_to_source_parent_training"
        if summary["passed"]
        else "stop_and_repair_data_protocol"
    )
    canonical = json.dumps(summary, ensure_ascii=False, sort_keys=True).encode("utf-8")
    summary["audit_sha256"] = hashlib.sha256(canonical).hexdigest()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = audit()
    rendered = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output = args.output.resolve()
        try:
            output.relative_to(REPO_ROOT.resolve())
        except ValueError as exc:
            raise ValueError("Output must remain inside the repository") from exc
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
