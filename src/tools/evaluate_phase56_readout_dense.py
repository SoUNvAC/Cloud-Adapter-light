"""Full-pixel validation for the readout selected only on target-train CV."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
    SOURCE_CLASS_NAMES, load_source_order_mask, read_shadow_yes_rows, sha256,
)
from run_phase55_mechanism_audit import (  # noqa: E402
    backbone_layers_and_outputs, build, normalized_input,
)


def feature_map(representation, layers, pixel_output):
    if representation.startswith("block_"):
        index = int(representation.split("_")[1])
        value = layers[index]
        side = int(round(value.shape[1] ** 0.5))
        return value.permute(0, 2, 1).reshape(value.shape[0], value.shape[2], side, side)
    mask_feature, memories = pixel_output
    if representation == "pixel_decoder_mask":
        return mask_feature
    if representation.startswith("pixel_decoder_memory_"):
        return memories[int(representation.rsplit("_", 1)[1])]
    raise ValueError(representation)


def readout_logits(feature, artifact, readout):
    mean = torch.from_numpy(artifact["mean"]).to(feature.device)[None, :, None, None]
    scale = torch.from_numpy(artifact["scale"]).to(feature.device)[None, :, None, None]
    normalized = (feature.float() - mean) / scale
    if readout == "cosine":
        normalized = F.normalize(normalized, dim=1)
        centroids = torch.from_numpy(artifact["centroids"]).to(feature.device)
        return torch.einsum("bchw,kc->bkhw", normalized, centroids)
    coefficients = torch.from_numpy(artifact["coefficients"]).to(feature.device)
    return (
        torch.einsum("bchw,ck->bkhw", normalized, coefficients[:-1])
        + coefficients[-1][None, :, None, None]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--readout-summary", default="work_dirs/phase56_readout/readout_summary.json")
    parser.add_argument("--config", default="configs/protocol/phase52a_fix_shadow_status_1pct_l1c.py")
    parser.add_argument("--checkpoint", default="work_dirs/phase52a_fix_shadow_status_1pct/seed52/best_mIoU_iter_1000.pth")
    parser.add_argument("--manifest", default="work_dirs/phase45_protocol_audit/scene_disjoint_manifest.csv")
    parser.add_argument("--metadata", default="research_plans/protocol_data/l8_biome_usgs_shadow_status.csv")
    parser.add_argument("--selection", default="work_dirs/phase50_active_1pct/selection.json")
    parser.add_argument("--output", default="work_dirs/phase56_readout/dense_readout_summary.json")
    args = parser.parse_args()

    audit = json.loads(Path(args.readout_summary).read_text(encoding="utf-8"))
    chosen = audit["best_parameter_limited_readout"]
    artifact = np.load(chosen["artifact"])
    _, validation, _ = read_shadow_yes_rows(args.manifest, args.metadata, args.selection)
    model = build(args.config, args.checkpoint).eval().cuda()
    confusion = np.zeros((4, 4), dtype=np.int64)
    ece_count = np.zeros(15, dtype=np.int64)
    ece_confidence = np.zeros(15, dtype=np.float64)
    ece_correct = np.zeros(15, dtype=np.int64)
    nll_sum, pixel_count = 0.0, 0

    for index, row in enumerate(validation, 1):
        captures = {"pixel": None}

        def hook(_module, _inputs, output):
            captures["pixel"] = output

        handle = model.decode_head.pixel_decoder.register_forward_hook(hook)
        inputs = normalized_input(model, row)
        with torch.inference_mode():
            layers, features, _ = backbone_layers_and_outputs(model, inputs)
            # Run the decoder once so every representation has the same graph/input.
            model.decode_head(features, None)
            selected_feature = feature_map(chosen["representation"], layers, captures["pixel"])
            logits = readout_logits(selected_feature, artifact, chosen["readout"])
            mask = load_source_order_mask(row)
            logits = F.interpolate(logits, size=mask.shape, mode="bilinear", align_corners=False)
            probability = logits.softmax(dim=1)[0]
            prediction = probability.argmax(dim=0).cpu().numpy()
            target = torch.from_numpy(mask.astype(np.int64)).to(probability.device)
            confidence, predicted_torch = probability.max(dim=0)
            true_probability = probability.gather(0, target.unsqueeze(0)).squeeze(0)
        handle.remove()
        encoded = 4 * mask.reshape(-1) + prediction.reshape(-1)
        confusion += np.bincount(encoded, minlength=16).reshape(4, 4)
        confidence_np = confidence.cpu().numpy().reshape(-1)
        correct_np = (predicted_torch == target).cpu().numpy().reshape(-1)
        bins = np.minimum((confidence_np * 15).astype(np.int64), 14)
        ece_count += np.bincount(bins, minlength=15)
        ece_confidence += np.bincount(bins, weights=confidence_np, minlength=15)
        ece_correct += np.bincount(bins, weights=correct_np.astype(np.int64), minlength=15).astype(np.int64)
        nll_sum += float(-true_probability.clamp_min(1e-12).log().sum().cpu())
        pixel_count += mask.size
        if index == 1 or index % 100 == 0 or index == len(validation):
            print(f"dense readout validation: {index}/{len(validation)}", flush=True)
        del inputs, layers, features, logits, probability, captures
        torch.cuda.empty_cache()

    true_positive = np.diag(confusion).astype(np.float64)
    ground_truth, predicted = confusion.sum(1), confusion.sum(0)
    union = ground_truth + predicted - true_positive
    iou = np.divide(true_positive, union, out=np.zeros(4), where=union > 0)
    recall = np.divide(true_positive, ground_truth, out=np.zeros(4), where=ground_truth > 0)
    nonzero = ece_count > 0
    ece = np.sum(
        ece_count[nonzero] / pixel_count
        * np.abs(ece_correct[nonzero] / ece_count[nonzero]
                 - ece_confidence[nonzero] / ece_count[nonzero])
    )
    dense_metrics = {
        "mIoU": 100.0 * float(iou.mean()),
        "class_iou": dict(zip(SOURCE_CLASS_NAMES, (100.0 * iou).tolist())),
        "class_recall": dict(zip(SOURCE_CLASS_NAMES, (100.0 * recall).tolist())),
        "ECE": float(ece), "NLL": nll_sum / pixel_count,
        "confusion": confusion.tolist(), "evaluated_images": len(validation),
        "evaluated_pixels": pixel_count,
    }
    phase52_miou = audit["phase52_target_mIoU_anchor"]
    gates = {
        "shadow_iou_at_least_13_86": dense_metrics["class_iou"]["cloud_shadow"] >= 13.86,
        "thin_iou_at_least_32_31": dense_metrics["class_iou"]["thin_cloud"] >= 32.31,
        "target_miou_gain_at_least_2": dense_metrics["mIoU"] >= phase52_miou + 2.0,
        "selection_used_target_train_only": chosen["selection_cv"]["selection_split"] == "target_train_scene_fold_only",
        "evaluated_expected_963_shadow_yes_images": len(validation) == 963,
        "target_test_sealed": True,
        "cloudsen_internal_test_sealed": True,
    }
    passed = all(gates.values())
    summary = {
        "phase": "56B-dense-validation", "selected_readout": chosen,
        "checkpoint": args.checkpoint, "checkpoint_sha256": sha256(args.checkpoint),
        "metrics": dense_metrics, "sampled_diagnostics": chosen["sampled_target_val_metrics"],
        "gates": gates, "passed": passed,
        "decision": "continue_to_phase57" if passed else "stop_readout_and_ontology_mainline",
        "target_test_evaluated": False, "cloudsen_internal_test_evaluated": False,
    }
    output = Path(args.output)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
