"""Cache matched source/Phase52 features for the Phase 56 readout audit."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("XFORMERS_DISABLED", "1")

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from phase56_protocol import (  # noqa: E402
    INJECTION_INDICES,
    SOURCE_CLASS_NAMES,
    deterministic_samples,
    load_source_order_mask,
    read_shadow_yes_rows,
    sample_spatial,
    sha256,
)
from run_phase55_mechanism_audit import (  # noqa: E402
    backbone_layers_and_outputs,
    build,
    normalized_input,
)


def _save_arrays(root, arrays):
    root.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, values in arrays.items():
        value = np.concatenate(values, axis=0) if isinstance(values, list) else values
        path = root / f"{name}.npy"
        np.save(path, value, allow_pickle=False)
        files[name] = {
            "path": str(path),
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": sha256(path),
        }
    return files


def _query_attribution(class_scores, mask_scores, coordinates, original_shape):
    class_probability = F.softmax(class_scores[-1].float(), dim=-1)[0, :, :-1]
    sampled_masks = sample_spatial(mask_scores[-1].sigmoid(), coordinates, original_shape)
    # sample_spatial treats BQHW as BCHW, hence [samples, queries].
    contribution = sampled_masks.float()[:, :, None] * class_probability[None, :, :]
    semantic = contribution.sum(dim=1).clamp_min(1e-12)
    logits = semantic.log()
    predicted = semantic.argmax(dim=1)
    localized_query = sampled_masks.argmax(dim=1)
    return {
        "final_logits": logits.cpu().numpy().astype(np.float32),
        "predicted_class": predicted.cpu().numpy().astype(np.uint8),
        "localization_query": localized_query.cpu().numpy().astype(np.uint16),
        "localization_query_class": class_probability[localized_query].argmax(dim=1)
        .cpu().numpy().astype(np.uint8),
        "localization_strength": sampled_masks.max(dim=1).values.cpu().numpy().astype(np.float16),
    }, sampled_masks, class_probability


def cache_model_split(model_name, config, checkpoint, rows, split, output_root, cap, seed):
    model = build(config, checkpoint).eval().cuda()
    arrays = {
        "label": [], "row": [], "column": [], "image_index": [], "scene_index": [],
        "final_logits": [], "predicted_class": [], "localization_query": [],
        "localization_query_class": [], "localization_strength": [],
    }
    for index in INJECTION_INDICES:
        arrays[f"block_{index}"] = []
    for key in ("pixel_decoder_mask", "pixel_decoder_memory_0",
                "pixel_decoder_memory_1", "pixel_decoder_memory_2"):
        arrays[key] = []
    image_ids, scene_ids, class_counts = [], [], np.zeros(4, dtype=np.int64)

    for image_index, row in enumerate(rows):
        mask = load_source_order_mask(row)
        coordinates, labels = deterministic_samples(mask, row["name"], cap, seed)
        if len(labels) == 0:
            continue
        captures = {"pixel": None}

        def pixel_hook(_module, _inputs, output):
            captures["pixel"] = output

        handle = model.decode_head.pixel_decoder.register_forward_hook(pixel_hook)
        inputs = normalized_input(model, row)
        with torch.inference_mode():
            layers, features, _ = backbone_layers_and_outputs(model, inputs)
            class_scores, mask_scores = model.decode_head(features, None)
        handle.remove()
        if captures["pixel"] is None:
            raise RuntimeError("Mask2Former pixel decoder hook did not run")
        mask_feature, memories = captures["pixel"]
        if len(memories) != 3:
            raise RuntimeError(f"Expected three pixel-decoder memories, got {len(memories)}")

        for block_index in INJECTION_INDICES:
            sampled = sample_spatial(layers[block_index], coordinates, mask.shape)
            arrays[f"block_{block_index}"].append(
                sampled.float().cpu().numpy().astype(np.float16)
            )
        arrays["pixel_decoder_mask"].append(
            sample_spatial(mask_feature, coordinates, mask.shape)
            .float().cpu().numpy().astype(np.float16)
        )
        for memory_index, memory in enumerate(memories):
            arrays[f"pixel_decoder_memory_{memory_index}"].append(
                sample_spatial(memory, coordinates, mask.shape)
                .float().cpu().numpy().astype(np.float16)
            )
        attribution, _, _ = _query_attribution(
            class_scores, mask_scores, coordinates, mask.shape
        )
        for key, value in attribution.items():
            arrays[key].append(value)
        arrays["label"].append(labels)
        arrays["row"].append(coordinates[:, 0].astype(np.uint16))
        arrays["column"].append(coordinates[:, 1].astype(np.uint16))
        arrays["image_index"].append(np.full(len(labels), image_index, np.uint16))
        scene_index = scene_ids.index(row["scene"]) if row["scene"] in scene_ids else len(scene_ids)
        if row["scene"] not in scene_ids:
            scene_ids.append(row["scene"])
        arrays["scene_index"].append(np.full(len(labels), scene_index, np.uint8))
        image_ids.append(row["name"])
        class_counts += np.bincount(labels, minlength=4)
        if image_index == 0 or (image_index + 1) % 50 == 0 or image_index + 1 == len(rows):
            print(f"{model_name}/{split}: {image_index + 1}/{len(rows)}", flush=True)
        del inputs, layers, features, class_scores, mask_scores, captures
        torch.cuda.empty_cache()

    cache_root = output_root / model_name / split
    files = _save_arrays(cache_root, arrays)
    metadata_path = cache_root / "metadata.json"
    metadata = {
        "model": model_name,
        "split": split,
        "config": config,
        "checkpoint": checkpoint,
        "checkpoint_sha256": sha256(checkpoint),
        "shadow_filter": "USGS Shadows?=yes only",
        "image_ids": image_ids,
        "scene_ids": scene_ids,
        "images": len(rows),
        "samples": int(class_counts.sum()),
        "class_counts": dict(zip(SOURCE_CLASS_NAMES, map(int, class_counts))),
        "cap_per_image_class": cap,
        "sampling_seed": seed,
        "files": files,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    del model
    torch.cuda.empty_cache()
    return {**metadata, "metadata_path": str(metadata_path), "metadata_sha256": sha256(metadata_path)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-config", default="configs/protocol/phase22_clean_v8_l1c.py")
    parser.add_argument("--adapted-config", default="configs/protocol/phase52a_fix_shadow_status_1pct_l1c.py")
    parser.add_argument("--source-checkpoint", default="work_dirs/phase22_clean_v8/seed42/best_mIoU_iter_40000.pth")
    parser.add_argument("--adapted-checkpoint", default="work_dirs/phase52a_fix_shadow_status_1pct/seed52/best_mIoU_iter_1000.pth")
    parser.add_argument("--manifest", default="work_dirs/phase45_protocol_audit/scene_disjoint_manifest.csv")
    parser.add_argument("--metadata", default="research_plans/protocol_data/l8_biome_usgs_shadow_status.csv")
    parser.add_argument("--selection", default="work_dirs/phase50_active_1pct/selection.json")
    parser.add_argument("--output-root", default="work_dirs/phase56_readout/cache")
    parser.add_argument("--cap-per-image-class", type=int, default=32)
    parser.add_argument("--seed", type=int, default=56)
    args = parser.parse_args()

    train, validation, _ = read_shadow_yes_rows(args.manifest, args.metadata, args.selection)
    output_root = Path(args.output_root)
    entries = []
    for model_name, config, checkpoint in (
        ("source", args.source_config, args.source_checkpoint),
        ("adapted", args.adapted_config, args.adapted_checkpoint),
    ):
        for split, rows in (("target_train", train), ("target_val", validation)):
            entries.append(cache_model_split(
                model_name, config, checkpoint, rows, split, output_root,
                args.cap_per_image_class, args.seed,
            ))
    manifest = {
        "phase": "56A",
        "protocol": {
            "manifest": args.manifest,
            "manifest_sha256": sha256(args.manifest),
            "metadata": args.metadata,
            "metadata_sha256": sha256(args.metadata),
            "selection": args.selection,
            "selection_sha256": sha256(args.selection),
            "training_images": len(train),
            "validation_images": len(validation),
            "target_test_evaluated": False,
            "cloudsen_internal_test_evaluated": False,
        },
        "caches": entries,
    }
    output = output_root.parent / "feature_cache_manifest.json"
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(output), "sha256": sha256(output)}, indent=2))


if __name__ == "__main__":
    main()
