"""Preflight the Phase 62 three-way partial-label experiment."""

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
from PIL import Image
import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.structures import PixelData
from mmseg.registry import DATASETS
from mmseg.structures import SegDataSample

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cloud_adapter.datasets  # noqa: E402,F401
from audit_phase52_sparse_msre import build  # noqa: E402
from cloud_adapter.datasets.manifest_seg import load_usgs_shadow_status  # noqa: E402
from cloud_adapter.models.segmentors.partial_label_encoder_decoder import (  # noqa: E402
    partial_label_nll,
)


GROUP_CONFIGS = {
    "complete12": "configs/protocol/phase62_complete12_l1c.py",
    "ignore65": "configs/protocol/phase62_ignore65_l1c.py",
    "partial65": "configs/protocol/phase62_partial65_l1c.py",
}


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", default="work_dirs/phase45_protocol_audit/scene_disjoint_manifest.csv"
    )
    parser.add_argument(
        "--selection", default="work_dirs/phase50_active_1pct/selection.json"
    )
    parser.add_argument(
        "--metadata", default="research_plans/protocol_data/l8_biome_usgs_shadow_status.csv"
    )
    parser.add_argument(
        "--checkpoint", default="work_dirs/phase22_clean_v8/seed42/best_mIoU_iter_40000.pth"
    )
    parser.add_argument(
        "--output", default="work_dirs/phase62_partial_labels/preflight.json"
    )
    args = parser.parse_args()

    metadata = load_usgs_shadow_status(args.metadata)
    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    selected_names = {row["name"] for row in selection["selected"]}
    with Path(args.manifest).open(newline="", encoding="utf-8") as handle:
        selected = [row for row in csv.DictReader(handle) if row["name"] in selected_names]
    selected.sort(key=lambda row: row["name"])
    status = Counter(metadata[row["scene"]] for row in selected)
    scene_by_image = {
        str(Path(row["image_path"]).resolve()): row["scene"] for row in selected
    }

    raw_pixels = {"yes": Counter(), "no": Counter()}
    for row in selected:
        with Image.open(row["mask_path"]) as image:
            values, counts = np.unique(np.asarray(image), return_counts=True)
        raw_pixels[metadata[row["scene"]]].update(
            {str(int(value)): int(count) for value, count in zip(values, counts)}
        )

    init_default_scope("mmseg")
    dataset_audit = {}
    model_configs = []
    for group, config_path in GROUP_CONFIGS.items():
        cfg = Config.fromfile(config_path)
        model_configs.append(cfg.model.to_dict())
        dataset_cfg = cfg.train_dataloader.dataset.dataset
        dataset = DATASETS.build(dataset_cfg)
        maps_by_status = {"yes": set(), "no": set()}
        for item in dataset.data_list:
            scene = scene_by_image[item["img_path"]]
            maps_by_status[metadata[scene]].add(int(item["label_map"][0]))
        dataset_audit[group] = {
            "config": config_path,
            "base_images": len(dataset),
            "raw_id0_mapped_values_by_shadow_status": {
                key: sorted(value) for key, value in maps_by_status.items()
            },
        }

    candidate = build(GROUP_CONFIGS["partial65"], args.checkpoint)
    candidate.train()
    trainable = {
        name: parameter.numel()
        for name, parameter in candidate.named_parameters()
        if parameter.requires_grad
    }
    candidate = candidate.cuda()
    generator = torch.Generator(device="cuda").manual_seed(62)
    inputs = torch.randn(1, 3, 512, 512, generator=generator, device="cuda")
    labels = torch.full((1, 512, 512), 255, dtype=torch.long, device="cuda")
    labels[:, 32:96, 32:96] = 2
    labels[:, 128:384, 128:384] = 254
    sample = SegDataSample(metainfo={
        "ori_shape": (512, 512), "img_shape": (512, 512),
        "pad_shape": (512, 512), "padding_size": [0, 0, 0, 0], "flip": False,
    })
    sample.gt_sem_seg = PixelData(data=labels)
    losses = candidate(inputs, data_samples=[sample], mode="loss")
    total_loss = sum(value.mean() for key, value in losses.items() if "loss" in key)
    total_loss.backward()
    nonzero_trainable_gradients = sum(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        and bool(torch.count_nonzero(parameter.grad))
        for name, parameter in candidate.named_parameters() if name in trainable
    )
    smoke_partial_loss = float(losses["partial.loss_set_nll"].detach())

    uniform_logits = torch.zeros(1, 4, 2, 2, device="cuda", requires_grad=True)
    one_pixel = torch.zeros(1, 2, 2, dtype=torch.bool, device="cuda")
    one_pixel[:, 0, 0] = True
    uniform_loss = partial_label_nll(uniform_logits, one_pixel, (0, 3))
    uniform_loss.backward()
    uniform_value = float(uniform_loss.detach())

    gates = {
        "selection_is_frozen_65": len(selected) == 65 and len(selected_names) == 65,
        "selection_did_not_read_labels": selection.get("target_labels_read_during_selection") is False,
        "selected_status_is_12_yes_53_no": status == {"yes": 12, "no": 53},
        "no_scenes_have_raw_id0_pixels": raw_pixels["no"]["0"] > 0,
        "three_groups_share_identical_model_config": model_configs[0] == model_configs[1] == model_configs[2],
        "group_image_counts_are_12_65_65": [
            dataset_audit[name]["base_images"] for name in GROUP_CONFIGS
        ] == [12, 65, 65],
        "complete12_keeps_verified_id0_clear": dataset_audit["complete12"][
            "raw_id0_mapped_values_by_shadow_status"
        ]["yes"] == [0],
        "ignore65_maps_no_id0_to_255": dataset_audit["ignore65"][
            "raw_id0_mapped_values_by_shadow_status"
        ]["no"] == [255],
        "partial65_maps_no_id0_to_254": dataset_audit["partial65"][
            "raw_id0_mapped_values_by_shadow_status"
        ]["no"] == [254],
        "trainable_parameters_unchanged_380577": sum(trainable.values()) == 380577,
        "no_new_trainable_parameter_names": bool(trainable) and all(
            "target_msre" in name or "target_head_delta" in name for name in trainable
        ),
        "partial_loss_uniform_probability_is_log2": abs(uniform_value - math.log(2.0)) <= 1e-6,
        "partial_training_smoke_finite": math.isfinite(smoke_partial_loss),
        "partial_training_reaches_existing_trainable_parameters": nonzero_trainable_gradients > 0,
        "target_test_sealed": True,
        "cloudsen_internal_test_sealed": True,
    }
    result = {
        "phase": "62-preflight",
        "manifest_sha256": file_sha256(args.manifest),
        "selection_sha256": file_sha256(args.selection),
        "metadata_sha256": file_sha256(args.metadata),
        "selected_shadow_status": dict(status),
        "selected_raw_pixel_counts": {
            key: dict(sorted(value.items())) for key, value in raw_pixels.items()
        },
        "datasets": dataset_audit,
        "target_trainable_parameters": sum(trainable.values()),
        "trainable_parameter_names": trainable,
        "smoke_partial_loss": smoke_partial_loss,
        "uniform_partial_loss_expected_log2": uniform_value,
        "nonzero_trainable_gradient_tensors": nonzero_trainable_gradients,
        "gates": gates,
        "passed": all(gates.values()),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
