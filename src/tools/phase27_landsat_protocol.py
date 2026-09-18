import math

import numpy as np


TRUTH_FILL = 0
TRUTH_CLEAR = 128
TRUTH_THIN = 192
TRUTH_OPAQUE = 255
ALLOWED_TRUTH_VALUES = {TRUTH_FILL, TRUTH_CLEAR, TRUTH_THIN, TRUTH_OPAQUE}


def landsat_dn_to_rgb_scale(digital_number, multiplier, additive, sun_elevation):
    digital_number = np.asarray(digital_number, dtype=np.float32)
    if not (0.0 < sun_elevation <= 90.0):
        raise ValueError("sun_elevation must be in (0, 90]")
    reflectance = (multiplier * digital_number + additive) / math.sin(
        math.radians(sun_elevation)
    )
    reflectance[digital_number == 0] = 0.0
    return np.clip(reflectance, 0.0, 1.0) * 255.0


def map_truth_binary(truth):
    """Return cloud target and valid mask for the USGS C2 manual truth."""
    truth = np.asarray(truth)
    valid = truth != TRUTH_FILL
    cloud = np.isin(truth, (TRUTH_THIN, TRUTH_OPAQUE))
    return cloud.astype(np.uint8), valid


def map_cloud_adapter_binary(prediction):
    """CloudSEN12 classes: clear=0, thick=1, thin=2, shadow=3."""
    prediction = np.asarray(prediction)
    if prediction.size and (prediction.min() < 0 or prediction.max() > 3):
        raise ValueError("Cloud-Adapter prediction contains a class outside 0..3")
    return np.isin(prediction, (1, 2)).astype(np.uint8)


def map_cloud_adapter_three_class(prediction):
    """Harmonize clear/shadow to clear, thick to opaque, and thin to thin."""
    prediction = np.asarray(prediction)
    if prediction.size and (prediction.min() < 0 or prediction.max() > 3):
        raise ValueError("Cloud-Adapter prediction contains a class outside 0..3")
    mapped = np.zeros(prediction.shape, dtype=np.uint8)
    mapped[prediction == 2] = 1
    mapped[prediction == 1] = 2
    return mapped


def map_truth_three_class(truth):
    truth = np.asarray(truth)
    mapped = np.zeros(truth.shape, dtype=np.uint8)
    mapped[truth == TRUTH_THIN] = 1
    mapped[truth == TRUTH_OPAQUE] = 2
    return mapped


def decode_cfmask_cloud(qa_pixel):
    """C2 L8/9 QA_PIXEL bits 1=dilated, 2=cirrus, 3=cloud."""
    qa_pixel = np.asarray(qa_pixel, dtype=np.uint16)
    flags = (1 << 1) | (1 << 2) | (1 << 3)
    return ((qa_pixel & flags) != 0).astype(np.uint8)


def update_binary_confusion(confusion, prediction, target, valid):
    prediction = np.asarray(prediction, dtype=np.uint8)
    target = np.asarray(target, dtype=np.uint8)
    valid = np.asarray(valid, dtype=bool)
    if prediction.shape != target.shape or target.shape != valid.shape:
        raise ValueError("prediction, target, and valid mask shapes must match")
    encoded = 2 * target[valid] + prediction[valid]
    confusion += np.bincount(encoded, minlength=4).reshape(2, 2)


def update_multiclass_confusion(confusion, prediction, target, valid, classes=3):
    prediction = np.asarray(prediction, dtype=np.uint8)
    target = np.asarray(target, dtype=np.uint8)
    valid = np.asarray(valid, dtype=bool)
    if prediction.shape != target.shape or target.shape != valid.shape:
        raise ValueError("prediction, target, and valid mask shapes must match")
    encoded = classes * target[valid] + prediction[valid]
    confusion += np.bincount(encoded, minlength=classes**2).reshape(classes, classes)


def safe_ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else float("nan")


def binary_metrics(confusion):
    confusion = np.asarray(confusion, dtype=np.int64)
    if confusion.shape != (2, 2):
        raise ValueError("binary confusion must have shape (2, 2)")
    tn, fp = confusion[0]
    fn, tp = confusion[1]
    iou = safe_ratio(tp, tp + fp + fn)
    precision = safe_ratio(tp, tp + fp)
    recall = safe_ratio(tp, tp + fn)
    specificity = safe_ratio(tn, tn + fp)
    f1 = safe_ratio(2 * tp, 2 * tp + fp + fn)
    accuracy = safe_ratio(tp + tn, confusion.sum())
    balanced_accuracy = (
        0.5 * (recall + specificity)
        if math.isfinite(recall) and math.isfinite(specificity)
        else float("nan")
    )
    return {
        "cloud_iou": 100.0 * iou,
        "cloud_f1": 100.0 * f1,
        "cloud_precision": 100.0 * precision,
        "cloud_recall": 100.0 * recall,
        "specificity": 100.0 * specificity,
        "balanced_accuracy": 100.0 * balanced_accuracy,
        "accuracy": 100.0 * accuracy,
    }


def scene_cloud_iou(confusion):
    confusion = np.asarray(confusion, dtype=np.int64)
    if confusion.shape != (2, 2):
        raise ValueError("binary confusion must have shape (2, 2)")
    _, fp = confusion[0]
    fn, tp = confusion[1]
    union = tp + fp + fn
    return 100.0 if union == 0 else 100.0 * float(tp / union)


def multiclass_metrics(confusion, names=("clear", "thin_cloud", "opaque_cloud")):
    confusion = np.asarray(confusion, dtype=np.int64)
    if confusion.shape != (len(names), len(names)):
        raise ValueError("multiclass confusion shape does not match class names")
    true_positive = np.diag(confusion).astype(np.float64)
    target = confusion.sum(axis=1).astype(np.float64)
    predicted = confusion.sum(axis=0).astype(np.float64)
    union = target + predicted - true_positive
    iou = np.divide(
        true_positive,
        union,
        out=np.full_like(true_positive, np.nan),
        where=union > 0,
    )
    recall = np.divide(
        true_positive,
        target,
        out=np.full_like(true_positive, np.nan),
        where=target > 0,
    )
    return {
        "mIoU": 100.0 * float(np.nanmean(iou)),
        "class_iou": {name: 100.0 * float(iou[index]) for index, name in enumerate(names)},
        "class_recall": {
            name: 100.0 * float(recall[index]) for index, name in enumerate(names)
        },
    }


def percentile_interval(values, confidence=0.95):
    alpha = (1.0 - confidence) / 2.0
    return [
        float(np.quantile(values, alpha)),
        float(np.quantile(values, 1.0 - alpha)),
    ]


def paired_scene_bootstrap(model_values, baseline_values, replicates=10000, seed=20260918):
    model_values = np.asarray(model_values, dtype=np.float64)
    baseline_values = np.asarray(baseline_values, dtype=np.float64)
    if model_values.ndim != 1 or baseline_values.shape != model_values.shape:
        raise ValueError("paired bootstrap inputs must be equal-length vectors")
    if len(model_values) < 2 or replicates <= 0:
        raise ValueError("paired bootstrap needs at least two scenes and positive replicates")
    if not np.isfinite(model_values).all() or not np.isfinite(baseline_values).all():
        raise ValueError("paired bootstrap inputs must be finite")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(model_values), size=(replicates, len(model_values)))
    model_means = model_values[indices].mean(axis=1)
    baseline_means = baseline_values[indices].mean(axis=1)
    differences = model_means - baseline_means
    return {
        "replicates": replicates,
        "seed": seed,
        "model_scene_macro_mean": float(model_values.mean()),
        "model_scene_macro_95ci": percentile_interval(model_means),
        "cfmask_scene_macro_mean": float(baseline_values.mean()),
        "cfmask_scene_macro_95ci": percentile_interval(baseline_means),
        "paired_difference_mean": float((model_values - baseline_values).mean()),
        "paired_difference_95ci": percentile_interval(differences),
    }
