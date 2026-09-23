import argparse
import json
import math
import os
from pathlib import Path
import sys

os.environ.setdefault("XFORMERS_DISABLED", "1")

import numpy as np
from PIL import Image
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluate_phase46_factorized import source_rows  # noqa: E402
from export_phase12_onnx import build_wrapper  # noqa: E402
from run_phase45_source_only import binary_boundary, load_rows, sha256  # noqa: E402
from validate_phase12_onnxruntime import load_rgb_image  # noqa: E402


def error_auroc(total, errors):
    correct = total - errors
    error_count = errors.sum()
    correct_count = correct.sum()
    if error_count == 0 or correct_count == 0:
        return math.nan
    # Lower confidence means higher error score. For each error bin, count
    # correct pixels at strictly higher confidence plus half of ties.
    correct_above = np.cumsum(correct[::-1])[::-1] - correct
    numerator = np.sum(errors * (correct_above + 0.5 * correct))
    return float(numerator / (error_count * correct_count))


def take_fraction(histogram, values, fraction, ascending):
    target = histogram.sum() * fraction
    used = 0.0
    value = 0.0
    indices = range(len(histogram)) if ascending else range(len(histogram) - 1, -1, -1)
    for index in indices:
        available = float(histogram[index])
        amount = min(available, target - used)
        if available > 0:
            value += float(values[index]) * amount / available
        used += amount
        if used >= target - 1e-9:
            break
    return value, used


def summarize(state):
    total = state["total"]
    errors = state["errors"]
    pixels = int(total.sum())
    error_count = int(errors.sum())
    base_risk = error_count / pixels
    uncertain_errors, uncertain_pixels = take_fraction(total, errors, 0.20, True)
    accepted_errors, accepted_pixels = take_fraction(total, errors, 0.80, False)
    risk80 = accepted_errors / accepted_pixels
    coverages = []
    risks = []
    cumulative_total = 0.0
    cumulative_errors = 0.0
    for index in range(len(total) - 1, -1, -1):
        cumulative_total += total[index]
        cumulative_errors += errors[index]
        if cumulative_total:
            coverages.append(cumulative_total / pixels)
            risks.append(cumulative_errors / cumulative_total)
    aurc = float(np.trapz(np.asarray(risks), np.asarray(coverages)))
    ece = 0.0
    calibration_bins = []
    fine_per_calibration = len(total) // 20
    for start in range(0, len(total), fine_per_calibration):
        stop = min(start + fine_per_calibration, len(total))
        count = int(total[start:stop].sum())
        if not count:
            continue
        bin_errors = int(errors[start:stop].sum())
        confidence = float(state["confidence_sum"][start:stop].sum() / count)
        accuracy = 1.0 - bin_errors / count
        ece += count / pixels * abs(accuracy - confidence)
        calibration_bins.append(
            {"lower": start / len(total), "upper": stop / len(total), "count": count,
             "confidence": confidence, "accuracy": accuracy}
        )
    return {
        "pixels": pixels,
        "accuracy": 1.0 - base_risk,
        "error_risk": base_risk,
        "ece_20": ece,
        "nll": state["nll_sum"] / pixels,
        "brier": state["brier_sum"] / pixels,
        "aurc": aurc,
        "error_detection_auroc": error_auroc(total, errors),
        "weak_error_detection_auroc": error_auroc(state["weak_total"], state["weak_errors"]),
        "boundary_error_detection_auroc": error_auroc(state["boundary_total"], state["boundary_errors"]),
        "lowest_confidence_20pct_error_recall": uncertain_errors / error_count,
        "lowest_confidence_20pct_error_lift": (uncertain_errors / uncertain_pixels) / base_risk,
        "risk_at_80pct_coverage": risk80,
        "risk_reduction_at_80pct_coverage": (base_risk - risk80) / base_risk,
        "calibration_bins": calibration_bins,
    }


def evaluate_uncertainty(name, wrapper, rows, channel_order, weak_indices, bins=1000):
    state = {
        "total": np.zeros(bins, dtype=np.int64),
        "errors": np.zeros(bins, dtype=np.int64),
        "confidence_sum": np.zeros(bins, dtype=np.float64),
        "weak_total": np.zeros(bins, dtype=np.int64),
        "weak_errors": np.zeros(bins, dtype=np.int64),
        "boundary_total": np.zeros(bins, dtype=np.int64),
        "boundary_errors": np.zeros(bins, dtype=np.int64),
        "nll_sum": 0.0,
        "brier_sum": 0.0,
    }
    channel_order = torch.as_tensor(channel_order, device="cuda")
    for row_index, row in enumerate(rows, 1):
        rgb = torch.from_numpy(load_rgb_image(Path(row["image_path"]), 512)).cuda()
        with Image.open(row["mask_path"]) as image:
            target_np = np.asarray(image).astype(np.int64, copy=False)
        target = torch.from_numpy(target_np.copy()).cuda()
        with torch.inference_mode():
            scores = wrapper(rgb).float().clamp_min(1e-8)[:, channel_order]
            probabilities = scores / scores.sum(dim=1, keepdim=True).clamp_min(1e-8)
        probabilities = probabilities[0]
        confidence, prediction = probabilities.max(dim=0)
        valid = (target >= 0) & (target < 4)
        confidence = confidence[valid].clamp(0.0, 1.0)
        prediction = prediction[valid]
        labels = target[valid]
        errors = prediction.ne(labels)
        indices = torch.clamp((confidence * bins).long(), max=bins - 1)
        total = torch.bincount(indices, minlength=bins).cpu().numpy()
        error_hist = torch.bincount(indices[errors], minlength=bins).cpu().numpy()
        confidence_sum = torch.bincount(indices, weights=confidence.double(), minlength=bins).cpu().numpy()
        state["total"] += total
        state["errors"] += error_hist
        state["confidence_sum"] += confidence_sum
        weak = torch.zeros_like(labels, dtype=torch.bool)
        for weak_index in weak_indices:
            weak |= labels.eq(weak_index)
        state["weak_total"] += torch.bincount(indices[weak], minlength=bins).cpu().numpy()
        state["weak_errors"] += torch.bincount(indices[weak & errors], minlength=bins).cpu().numpy()
        boundary_np = binary_boundary(target_np)
        boundary = torch.from_numpy(boundary_np).cuda()[valid]
        state["boundary_total"] += torch.bincount(indices[boundary], minlength=bins).cpu().numpy()
        state["boundary_errors"] += torch.bincount(indices[boundary & errors], minlength=bins).cpu().numpy()
        selected_probabilities = probabilities.permute(1, 2, 0)[valid]
        true_probability = selected_probabilities.gather(1, labels[:, None]).squeeze(1).clamp_min(1e-8)
        state["nll_sum"] += float((-true_probability.log()).sum().item())
        one_hot = torch.nn.functional.one_hot(labels, num_classes=4).float()
        state["brier_sum"] += float(((selected_probabilities - one_hot) ** 2).sum().item())
        if row_index == 1 or row_index % 100 == 0 or row_index == len(rows):
            print(f"{name}: audited {row_index}/{len(rows)}", flush=True)
    result = summarize(state)
    result["images"] = len(rows)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/protocol/phase22_clean_v8_l1c.py")
    parser.add_argument("--checkpoint", default="work_dirs/phase22_clean_v8/seed42/best_mIoU_iter_40000.pth")
    parser.add_argument("--manifest", default="work_dirs/phase45_protocol_audit/scene_disjoint_manifest.csv")
    parser.add_argument("--data-root", default=os.environ.get("CLOUD_ADAPTER_DATA_ROOT", "data/cloudsen12_high_l1c"))
    parser.add_argument("--phase50", default="work_dirs/phase50_active_1pct/summary.json")
    parser.add_argument("--output", default="work_dirs/phase51_trust_audit/summary.json")
    args = parser.parse_args()
    phase50 = json.loads(Path(args.phase50).read_text(encoding="utf-8"))
    if phase50.get("decision") != "stop_adaptation_route_begin_trustworthy_segmentation_audit":
        raise RuntimeError("Phase 50 adaptation stop is required")
    wrapper_args = argparse.Namespace(config=args.config, checkpoint=args.checkpoint, precision="fp16", active_block_indices=None)
    wrapper, _ = build_wrapper(wrapper_args)
    source = evaluate_uncertainty("source_val", wrapper, source_rows(Path(args.data_root)), [0, 1, 2, 3], (2, 3))
    target = evaluate_uncertainty("target_val", wrapper, load_rows(Path(args.manifest), "target_val"), [0, 3, 2, 1], (1, 2))
    gates = {
        "target_error_auroc_at_least_0_70": bool(target["error_detection_auroc"] >= 0.70),
        "target_weak_error_auroc_at_least_0_65": bool(target["weak_error_detection_auroc"] >= 0.65),
        "target_boundary_error_auroc_at_least_0_65": bool(target["boundary_error_detection_auroc"] >= 0.65),
        "bottom20_error_recall_at_least_0_40": bool(target["lowest_confidence_20pct_error_recall"] >= 0.40),
        "risk_reduction_at_80_coverage_at_least_0_30": bool(target["risk_reduction_at_80pct_coverage"] >= 0.30),
        "expected_source_images": source["images"] == 535,
        "expected_target_images": target["images"] == 1905,
        "metrics_finite": all(math.isfinite(value) for row in (source, target) for key, value in row.items() if key not in ("calibration_bins",) and isinstance(value, float)),
        "target_test_sealed": True,
        "cloudsen_internal_test_sealed": True,
    }
    passed = all(gates.values())
    result = {
        "phase": 51,
        "method": "frozen_source_only_confidence_risk_audit",
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": sha256(Path(args.checkpoint)),
        "confidence": "normalized_mask2former_semantic_scores_max_probability",
        "source": source,
        "target": target,
        "gates": gates,
        "passed": passed,
        "decision": "proceed_to_trustworthy_segmentation" if passed else "end_tgrs_main_method_route",
        "target_train_evaluated": False,
        "target_test_evaluated": False,
        "cloudsen_internal_test_evaluated": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
