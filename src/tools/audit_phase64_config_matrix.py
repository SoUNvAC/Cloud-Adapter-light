#!/usr/bin/env python3
"""Parse and verify the frozen Phase 64 Day-2 configuration matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path

from mmengine.config import Config


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
SOURCE_CONFIG = SRC_ROOT / "configs/protocol/phase64_source_parent_rgb.py"
TARGET_CONFIG = SRC_ROOT / "configs/protocol/phase64_target_parent_rgb.py"
SOURCE_RETENTION_CONFIG = (
    SRC_ROOT / "configs/protocol/phase64_source_retention_rgb.py"
)
TARGETS = ("l8", "sparcs")
METHODS = ("shared_parent", "full", "lora", "msre")
EXPECTED_MODEL_TYPES = {
    "shared_parent": ("FrozenBackboneEncoderDecoder", None),
    "full": (
        "EncoderDecoder",
        "Phase64FullFineTuneCloudAdapterDinoVisionTransformer",
    ),
    "lora": (
        "FrozenHeadEncoderDecoder",
        "Phase64LoRACloudAdapterDinoVisionTransformer",
    ),
    "msre": (
        "FrozenHeadEncoderDecoder",
        "TargetMsRECloudAdapterDinoVisionTransformer",
    ),
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def audit() -> dict:
    source = Config.fromfile(SOURCE_CONFIG)
    source_gates = {
        "three_parent_classes": source.model.decode_head.num_classes == 3,
        "source_parent_dataset": (
            source.train_dataloader.dataset.type
            == "Phase64CloudSENParentDataset"
        ),
        "source_backbone_frozen": source.model.type == "FrozenBackboneEncoderDecoder",
        "source_seed_64": source.randomness.seed == 64,
        "source_4000_steps": source.train_cfg.max_iters == 4000,
    }

    rows = []
    for target in TARGETS:
        for method in METHODS:
            with scoped_environment(
                {
                    "PHASE64_TARGET": target,
                    "PHASE64_METHOD": method,
                    "PHASE64_PARENT_CHECKPOINT": "work_dirs/frozen-parent.pth",
                }
            ):
                config = Config.fromfile(TARGET_CONFIG)
            expected_model, expected_backbone = EXPECTED_MODEL_TYPES[method]
            actual_backbone = config.model.backbone.get("type")
            expected_dataset = (
                "Phase64L8ParentDataset"
                if target == "l8"
                else "Phase64SparcsManifestDataset"
            )
            gates = {
                "model_type": config.model.type == expected_model,
                "backbone_type": (
                    expected_backbone is None or actual_backbone == expected_backbone
                ),
                "train_parent_dataset": (
                    config.train_dataloader.dataset.type == expected_dataset
                ),
                "val_parent_dataset": (
                    config.val_dataloader.dataset.type == expected_dataset
                ),
                "same_batch_size": (
                    config.train_dataloader.batch_size
                    == config.val_dataloader.batch_size
                    == 1
                ),
                "seed_64": config.randomness.seed == 64,
                "steps_4000": config.train_cfg.max_iters == 4000,
                "checkpoint_required": (
                    config.load_from == "work_dirs/frozen-parent.pth"
                ),
                "isolated_work_dir": (
                    config.work_dir
                    == f"./work_dirs/phase64_target_parent/{target}/{method}/seed64"
                ),
            }
            if target == "l8":
                gates["complete_shadow_only"] = (
                    config.train_dataloader.dataset.usgs_shadows_filter == "yes"
                    and config.val_dataloader.dataset.usgs_shadows_filter == "yes"
                )
            else:
                gates["scene_split_train_val"] = (
                    config.train_dataloader.dataset.split == "target_train"
                    and config.val_dataloader.dataset.split == "target_val"
                )
            rows.append(
                {
                    "target": target,
                    "method": method,
                    "model_type": config.model.type,
                    "backbone_type": actual_backbone,
                    "gates": gates,
                    "passed": all(gates.values()),
                }
            )

    retention_rows = []
    for target in TARGETS:
        for method in METHODS:
            with scoped_environment(
                {
                    "PHASE64_TARGET": target,
                    "PHASE64_METHOD": method,
                    "PHASE64_PARENT_CHECKPOINT": "work_dirs/frozen-parent.pth",
                }
            ):
                config = Config.fromfile(SOURCE_RETENTION_CONFIG)
            gates = {
                "source_parent_dataset": (
                    config.test_dataloader.dataset.type
                    == "Phase64CloudSENParentDataset"
                ),
                "source_val_split": config.test_dataloader.dataset.data_prefix
                == dict(img_path="img_dir/val", seg_map_path="ann_dir/val"),
                "msre_target_path_gated": (
                    method != "msre"
                    or config.model.backbone.target_enabled is False
                ),
            }
            retention_rows.append(
                {
                    "target": target,
                    "method": method,
                    "gates": gates,
                    "passed": all(gates.values()),
                }
            )

    summary = {
        "phase": "64-day2-config-matrix",
        "source_config_sha256": sha256_file(SOURCE_CONFIG),
        "target_config_sha256": sha256_file(TARGET_CONFIG),
        "source_retention_config_sha256": sha256_file(
            SOURCE_RETENTION_CONFIG
        ),
        "source_gates": source_gates,
        "matrix": rows,
        "source_retention_matrix": retention_rows,
    }
    summary["passed"] = all(source_gates.values()) and all(
        row["passed"] for row in rows
    ) and all(row["passed"] for row in retention_rows)
    summary["decision"] = (
        "proceed_to_model_and_data_preflight"
        if summary["passed"]
        else "stop_and_repair_config_matrix"
    )
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
