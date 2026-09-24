"""Fit/evaluate deterministic parameter-limited readouts on Phase 56 caches."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from phase56_protocol import SOURCE_CLASS_NAMES, sha256  # noqa: E402
from run_phase55_mechanism_audit import binary_auc  # noqa: E402

REPRESENTATIONS = (
    "block_2", "block_5", "block_8", "block_11", "pixel_decoder_mask",
    "pixel_decoder_memory_0", "pixel_decoder_memory_1", "pixel_decoder_memory_2",
)


def load_array(root, model, split, name):
    return np.load(Path(root) / model / split / f"{name}.npy", mmap_mode="r")


def softmax(scores):
    shifted = scores - scores.max(axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=1, keepdims=True)


def fit_ridge(features, labels, balanced, ridge=0.01):
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-6] = 1.0
    design = np.concatenate(((x - mean) / scale, np.ones((len(x), 1))), axis=1)
    if balanced:
        counts = np.bincount(y, minlength=4).astype(np.float64)
        if np.any(counts == 0):
            raise RuntimeError(f"Training cache lacks a class: {counts.tolist()}")
        weights = 1.0 / (4.0 * counts[y])
    else:
        weights = np.full(len(y), 1.0 / len(y))
    target = np.eye(4, dtype=np.float64)[y]
    gram = design.T @ (weights[:, None] * design)
    penalty = np.eye(design.shape[1], dtype=np.float64) * ridge
    penalty[-1, -1] = 0.0
    coefficients = np.linalg.solve(gram + penalty, design.T @ (weights[:, None] * target))
    return {"mean": mean, "scale": scale, "coefficients": coefficients}


def apply_ridge(model, features):
    x = (np.asarray(features, dtype=np.float64) - model["mean"]) / model["scale"]
    return x @ model["coefficients"][:-1] + model["coefficients"][-1]


def fit_cosine(features, labels):
    x = np.asarray(features, dtype=np.float64)
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-6] = 1.0
    x = (x - mean) / scale
    centroids = np.stack([x[labels == index].mean(axis=0) for index in range(4)])
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True).clip(min=1e-12)
    return {"mean": mean, "scale": scale, "centroids": centroids}


def apply_cosine(model, features):
    x = (np.asarray(features, dtype=np.float64) - model["mean"]) / model["scale"]
    x /= np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-12)
    return x @ model["centroids"].T


def fit_named_readout(name, features, labels, ridge):
    if name == "linear":
        return fit_ridge(features, labels, False, ridge)
    if name == "balanced":
        return fit_ridge(features, labels, True, ridge)
    if name == "cosine":
        return fit_cosine(features, labels)
    raise ValueError(name)


def apply_named_readout(name, model, features):
    return apply_cosine(model, features) if name == "cosine" else apply_ridge(model, features)


def scene_fold_cv(features, labels, scene_indices, readout_name, ridge, folds=3):
    unique_scenes = np.unique(scene_indices)
    if len(unique_scenes) < folds:
        raise RuntimeError(f"Need at least {folds} target-train scenes, found {len(unique_scenes)}")
    scene_to_fold = {int(scene): index % folds for index, scene in enumerate(unique_scenes)}
    fold_id = np.asarray([scene_to_fold[int(scene)] for scene in scene_indices])
    rows = []
    for fold in range(folds):
        train, validation = fold_id != fold, fold_id == fold
        model = fit_named_readout(readout_name, features[train], labels[train], ridge)
        rows.append(metrics(
            apply_named_readout(readout_name, model, features[validation]),
            labels[validation],
        ))
    return {
        "folds": folds,
        "selection_split": "target_train_scene_fold_only",
        "fold_mIoU": [row["mIoU"] for row in rows],
        "mean_mIoU": float(np.mean([row["mIoU"] for row in rows])),
        "fold_shadow_iou": [row["class_iou"]["cloud_shadow"] for row in rows],
        "fold_thin_iou": [row["class_iou"]["thin_cloud"] for row in rows],
    }


def calibration_error(probabilities, labels, bins=15):
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    correct = prediction == labels
    result = 0.0
    for lower in np.linspace(0.0, 1.0, bins, endpoint=False):
        upper = lower + 1.0 / bins
        mask = (confidence >= lower) & (confidence < upper if upper < 1 else confidence <= upper)
        if mask.any():
            result += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(result)


def metrics(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = softmax(scores)
    prediction = probabilities.argmax(axis=1)
    confusion = np.bincount(4 * labels + prediction, minlength=16).reshape(4, 4)
    true_positive = np.diag(confusion).astype(np.float64)
    ground_truth = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    union = ground_truth + predicted - true_positive
    iou = np.divide(true_positive, union, out=np.zeros(4), where=union > 0)
    recall = np.divide(true_positive, ground_truth, out=np.zeros(4), where=ground_truth > 0)
    auc = [binary_auc(labels == index, probabilities[:, index]) for index in range(4)]
    shadow, thin, clear = scores[:, 3], scores[:, 2], scores[:, 0]
    result = {
        "mIoU": 100.0 * float(iou.mean()),
        "class_iou": dict(zip(SOURCE_CLASS_NAMES, (100.0 * iou).tolist())),
        "class_recall": dict(zip(SOURCE_CLASS_NAMES, (100.0 * recall).tolist())),
        "class_auroc": dict(zip(SOURCE_CLASS_NAMES, auc)),
        "ECE": calibration_error(probabilities, labels),
        "NLL": float(-np.log(probabilities[np.arange(len(labels)), labels].clip(min=1e-12)).mean()),
        "confusion": confusion.tolist(),
        "margin_quantiles": {
            "shadow_minus_thin_on_gt_shadow": np.quantile(
                (shadow - thin)[labels == 3], [0.05, 0.25, 0.5, 0.75, 0.95]
            ).tolist(),
            "shadow_minus_clear_on_gt_shadow": np.quantile(
                (shadow - clear)[labels == 3], [0.05, 0.25, 0.5, 0.75, 0.95]
            ).tolist(),
        },
    }
    result["weak_arithmetic_mean"] = 0.5 * (
        result["class_iou"]["thin_cloud"] + result["class_iou"]["cloud_shadow"]
    )
    thin_iou, shadow_iou = result["class_iou"]["thin_cloud"], result["class_iou"]["cloud_shadow"]
    result["weak_harmonic_mean"] = 2 * thin_iou * shadow_iou / max(thin_iou + shadow_iou, 1e-12)
    return result


def binary_probe(features, labels, positive, negative, ridge=0.01):
    keep = (labels == positive) | (labels == negative)
    x, y = np.asarray(features[keep], np.float64), (labels[keep] == positive).astype(np.int64)
    mean, scale = x.mean(0), x.std(0)
    scale[scale < 1e-6] = 1.0
    design = np.c_[(x - mean) / scale, np.ones(len(x))]
    counts = np.bincount(y, minlength=2).astype(np.float64)
    weights = 1.0 / (2.0 * counts[y])
    target = np.where(y == 1, 1.0, -1.0)
    gram = design.T @ (weights[:, None] * design)
    penalty = np.eye(design.shape[1]) * ridge
    penalty[-1, -1] = 0
    coefficient = np.linalg.solve(gram + penalty, design.T @ (weights * target))
    return mean, scale, coefficient


def apply_binary(model, features):
    mean, scale, coefficient = model
    x = (np.asarray(features, np.float64) - mean) / scale
    return x @ coefficient[:-1] + coefficient[-1]


def state_delta(source_path, adapted_path):
    def state(path):
        value = torch.load(path, map_location="cpu")
        return value.get("state_dict", value)
    source, adapted = state(source_path), state(adapted_path)
    changed, unchanged, added, removed = [], [], [], []
    for name in sorted(set(source) | set(adapted)):
        if name not in source:
            added.append(name)
        elif name not in adapted:
            removed.append(name)
        else:
            left, right = source[name], adapted[name]
            if left.shape != right.shape or not torch.equal(left, right):
                changed.append(name)
            else:
                unchanged.append(name)
    return {
        "source_checkpoint_sha256": sha256(source_path),
        "adapted_checkpoint_sha256": sha256(adapted_path),
        "changed": changed, "added": added, "removed": removed,
        "changed_count": len(changed), "unchanged_count": len(unchanged),
        "changed_groups": {
            "backbone": sum("backbone" in name for name in changed),
            "decode_head": sum("decode_head" in name for name in changed),
            "target_msre": sum("target_msre" in name for name in changed + added),
            "target_head_delta": sum("target_head_delta" in name for name in changed + added),
        },
    }


def query_attribution(cache_root, model):
    labels = load_array(cache_root, model, "target_val", "label")
    localized_class = load_array(cache_root, model, "target_val", "localization_query_class")
    prediction = load_array(cache_root, model, "target_val", "predicted_class")
    localization_correct = localized_class == labels
    final_correct = prediction == labels
    rows = {}
    for index, name in enumerate(SOURCE_CLASS_NAMES):
        keep = labels == index
        rows[name] = {
            "samples": int(keep.sum()),
            "top_localization_query_class_accuracy": float(localization_correct[keep].mean()),
            "final_pixel_accuracy": float(final_correct[keep].mean()),
            "localized_but_class_assignment_wrong_fraction": float(
                (localization_correct[keep] & ~final_correct[keep]).mean()
            ),
        }
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", default="work_dirs/phase56_readout/cache")
    parser.add_argument("--cache-manifest", default="work_dirs/phase56_readout/feature_cache_manifest.json")
    parser.add_argument("--source-checkpoint", default="work_dirs/phase22_clean_v8/seed42/best_mIoU_iter_40000.pth")
    parser.add_argument("--adapted-checkpoint", default="work_dirs/phase52a_fix_shadow_status_1pct/seed52/best_mIoU_iter_1000.pth")
    parser.add_argument("--phase52-summary", default="work_dirs/phase52a_fix_shadow_status_1pct/summary.json")
    parser.add_argument("--output", default="work_dirs/phase56_readout/readout_summary.json")
    parser.add_argument("--ridge", type=float, default=0.01)
    args = parser.parse_args()

    cache_manifest = json.loads(Path(args.cache_manifest).read_text(encoding="utf-8"))
    phase52 = json.loads(Path(args.phase52_summary).read_text(encoding="utf-8"))
    phase52_miou = float(phase52["target"]["metrics"]["mIoU"])
    results, fitted = {}, {}
    for representation in REPRESENTATIONS:
        results[representation], fitted[representation] = {}, {}
        for train_model in ("source", "adapted"):
            x_train = load_array(args.cache_root, train_model, "target_train", representation)
            y_train = load_array(args.cache_root, train_model, "target_train", "label")
            fitted[representation][train_model] = {
                "linear": fit_ridge(x_train, y_train, False, args.ridge),
                "balanced": fit_ridge(x_train, y_train, True, args.ridge),
                "cosine": fit_cosine(x_train, y_train),
            }
            binary_models = {
                "shadow_vs_thin": binary_probe(x_train, y_train, 3, 2, args.ridge),
                "shadow_vs_clear": binary_probe(x_train, y_train, 3, 0, args.ridge),
            }
            for val_model in ("source", "adapted"):
                x_val = load_array(args.cache_root, val_model, "target_val", representation)
                y_val = load_array(args.cache_root, val_model, "target_val", "label")
                key = f"train_{train_model}__eval_{val_model}"
                entry = {}
                for readout_name, readout in fitted[representation][train_model].items():
                    scores = apply_named_readout(readout_name, readout, x_val)
                    entry[readout_name] = metrics(scores, y_val)
                entry["binary_probes"] = {}
                for probe_name, probe in binary_models.items():
                    positive, negative = ((3, 2) if probe_name.endswith("thin") else (3, 0))
                    keep = (y_val == positive) | (y_val == negative)
                    scores = apply_binary(probe, x_val[keep])
                    labels = (y_val[keep] == positive).astype(np.uint8)
                    entry["binary_probes"][probe_name] = {
                        "AUROC": binary_auc(labels, scores),
                        "samples": int(keep.sum()),
                    }
                results[representation][key] = entry
                print(f"audited {representation}: {key}", flush=True)

    original = {}
    for model in ("source", "adapted"):
        logits = load_array(args.cache_root, model, "target_val", "final_logits")
        labels = load_array(args.cache_root, model, "target_val", "label")
        original[model] = metrics(logits, labels)

    cv_selection = []
    adapted_scene = load_array(args.cache_root, "adapted", "target_train", "scene_index")
    adapted_labels = load_array(args.cache_root, "adapted", "target_train", "label")
    for representation in REPRESENTATIONS:
        adapted_features = load_array(args.cache_root, "adapted", "target_train", representation)
        for readout_name in ("linear", "balanced", "cosine"):
            cv = scene_fold_cv(
                adapted_features, adapted_labels, adapted_scene, readout_name, args.ridge
            )
            cv_selection.append((cv["mean_mIoU"], representation, readout_name, cv))
    best_cv = max(cv_selection, key=lambda row: row[0])
    best_representation, best_readout = best_cv[1], best_cv[2]
    best_metrics = results[best_representation]["train_adapted__eval_adapted"][best_readout]
    chosen = fitted[best_representation]["adapted"][best_readout]
    artifact_path = Path(args.output).parent / "selected_readout.npz"
    artifact = {
        "mean": chosen["mean"].astype(np.float32),
        "scale": chosen["scale"].astype(np.float32),
    }
    if best_readout == "cosine":
        artifact["centroids"] = chosen["centroids"].astype(np.float32)
    else:
        artifact["coefficients"] = chosen["coefficients"].astype(np.float32)
    np.savez(artifact_path, **artifact)
    selection_rows = {
        f"{representation}/{readout}": cv
        for _, representation, readout, cv in cv_selection
    }
    summary = {
        "phase": "56B",
        "cache_manifest": args.cache_manifest,
        "cache_manifest_sha256": sha256(args.cache_manifest),
        "state_dict_delta": state_delta(args.source_checkpoint, args.adapted_checkpoint),
        "original_cached_logits": original,
        "readouts": results,
        "query_attribution": {
            model: query_attribution(args.cache_root, model) for model in ("source", "adapted")
        },
        "best_parameter_limited_readout": {
            "representation": best_representation,
            "combination": "train_adapted__eval_adapted",
            "readout": best_readout,
            "selection": "highest mean mIoU in 3-fold scene-level CV using target-train only",
            "selection_cv": best_cv[3],
            "artifact": str(artifact_path),
            "artifact_sha256": sha256(artifact_path),
            "trainable_parameters": int(
                chosen["centroids"].size if best_readout == "cosine"
                else chosen["coefficients"].size
            ),
            "sampled_target_val_metrics": best_metrics,
        },
        "target_train_scene_cv_selection": selection_rows,
        "phase52_target_mIoU_anchor": phase52_miou,
        "dense_evaluation_pending": True,
        "target_test_evaluated": False,
        "cloudsen_internal_test_evaluated": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "dense_evaluation_pending": True,
                      "best": summary["best_parameter_limited_readout"]}, indent=2))


if __name__ == "__main__":
    main()
