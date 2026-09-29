"""Export frozen-model unreliability evidence for the Phase 63 point audit.

This is an inference-only program.  The 210 human decisions are interpreted as
point-level set labels at ``center_y, center_x``; the 128 px crop is context,
not dense ground truth.  No risk model is fitted here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


CORE4 = ("clear", "thick_cloud", "thin_cloud", "cloud_shadow")
CORE4_SET = frozenset(CORE4)
CLASS_INDEX = {name: index for index, name in enumerate(CORE4)}
COARSE_MAP = {
    "clear": "clear",
    "thick_cloud": "cloud",
    "thin_cloud": "cloud",
    "cloud_shadow": "shadow",
}
BINARY_MAP = {
    "clear": "non_cloud",
    "thick_cloud": "cloud",
    "thin_cloud": "cloud",
    "cloud_shadow": "non_cloud",
}
EXPECTED_FINAL_LABEL_COUNTS = {
    "clear": 39,
    "thin_cloud": 42,
    "thick_cloud": 10,
    "cloud_shadow": 71,
    "haze_cirrus": 19,
    "terrain_water_shadow": 12,
    "unobservable_nodata": 12,
    "boundary_mixed": 4,
    "uncertain": 1,
}
EXPECTED_FILES = {
    "sealed": "7c9468778ccb44127610918ae537a64d7e1e0af95469c403a0c6c1453d587c77",
    "reviewer_a": "9c00b9349862d7604884507955c64ddffe5eacab81983000016f2827eaf2fcda",
    "reviewer_b": "a4166d07bcb03237d956d4c661f88a0483b60d2d0b63de5aba2e6fcb43bca525",
    "adjudication": "90fa08860ea01025b5c750db8b4506f65338e6c3f039017ba8e9b43e9231dce4",
    "protocol_manifest": "885c2d7f61cae23409a7b1ccbe65739c302fc1e59b089aab0cc05badf674872b",
}
SOURCE_CHECKPOINT_FINGERPRINT = "d33a81337544e294a42007c6b353ee13c901ebb1e1f5261fd1d95a79e3888012"
ADAPTED_CHECKPOINT_FINGERPRINT = "9655ee83c63840ecb31351efef0582676529c318fd0d1eca5fd0929db702eb2e"
NON_CORE_LABELS = frozenset({
    "terrain_water_shadow", "haze_cirrus", "boundary_mixed",
    "unobservable_nodata", "uncertain",
})


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_hash(path: str | Path, expected: str, name: str) -> str:
    actual = sha256(path)
    if actual != expected.lower():
        raise RuntimeError(f"Frozen {name} hash mismatch: expected={expected}, actual={actual}")
    return actual


def verify_fingerprint(actual: str, expected: str, name: str) -> None:
    """Verify a full SHA256 or an explicitly abbreviated prefix...suffix."""
    actual, expected = actual.lower(), expected.lower()
    if len(actual) != 64 or any(character not in "0123456789abcdef" for character in actual):
        raise RuntimeError(f"Invalid actual SHA256 for {name}: {actual}")
    if "..." in expected:
        prefix, suffix = expected.split("...", 1)
        matched = bool(prefix and suffix and actual.startswith(prefix) and actual.endswith(suffix))
    else:
        matched = actual == expected
    if not matched:
        raise RuntimeError(f"Frozen {name} fingerprint mismatch: expected={expected}, actual={actual}")


def parse_label_set(value: str) -> frozenset[str]:
    labels = frozenset(part.strip() for part in value.strip().split("|") if part.strip())
    allowed = CORE4_SET | NON_CORE_LABELS
    if not labels or not labels.issubset(allowed) or len(labels) > 2:
        raise RuntimeError(f"Invalid frozen set label: {value!r}")
    return labels


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def read_label_file(path: str | Path, column: str) -> dict[str, frozenset[str]]:
    rows = read_csv(path)
    if not rows or not {"tile_id", column}.issubset(rows[0]):
        raise RuntimeError(f"Missing tile_id/{column} in {path}")
    identifiers = [row["tile_id"].strip() for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError(f"Duplicate tile IDs in {path}")
    return {row["tile_id"].strip(): parse_label_set(row[column]) for row in rows}


def assemble_frozen_labels(
    records: list[dict], reviewer_a_path: str | Path,
    reviewer_b_path: str | Path, adjudication_path: str | Path,
) -> tuple[dict[str, frozenset[str]], dict]:
    expected = {row["tile_id"] for row in records}
    review_a = read_label_file(reviewer_a_path, "label_set")
    review_b = read_label_file(reviewer_b_path, "label_set")
    if set(review_a) != expected or set(review_b) != expected:
        raise RuntimeError("Reviewer IDs do not exactly match the frozen 210 point units")
    disagreements = {tile_id for tile_id in expected if review_a[tile_id] != review_b[tile_id]}
    adjudication_rows = read_csv(adjudication_path)
    required = {
        "tile_id", "reviewer_A_label_set", "reviewer_B_label_set",
        "final_label_set", "rationale",
    }
    if not adjudication_rows or not required.issubset(adjudication_rows[0]):
        raise RuntimeError("Invalid frozen adjudication columns")
    adjudication_ids = [row["tile_id"].strip() for row in adjudication_rows]
    if set(adjudication_ids) != disagreements or len(adjudication_ids) != len(set(adjudication_ids)):
        raise RuntimeError("Adjudication must contain each and only A/B disagreement once")
    final = {}
    missing_rationales = []
    for row in adjudication_rows:
        tile_id = row["tile_id"].strip()
        if parse_label_set(row["reviewer_A_label_set"]) != review_a[tile_id]:
            raise RuntimeError(f"Reviewer A label changed in adjudication: {tile_id}")
        if parse_label_set(row["reviewer_B_label_set"]) != review_b[tile_id]:
            raise RuntimeError(f"Reviewer B label changed in adjudication: {tile_id}")
        final[tile_id] = parse_label_set(row["final_label_set"])
        if not row["rationale"].strip():
            missing_rationales.append(tile_id)
    for tile_id in expected - disagreements:
        final[tile_id] = review_a[tile_id]
    if len(final) != 210 or len(disagreements) != 79 or len(missing_rationales) != 79:
        raise RuntimeError(
            "Frozen Phase61D2 counts changed: "
            f"final={len(final)}, disagreements={len(disagreements)}, "
            f"missing_rationales={len(missing_rationales)}"
        )
    final_counts = Counter(next(iter(labels)) for labels in final.values() if len(labels) == 1)
    if any(len(labels) != 1 for labels in final.values()) or dict(final_counts) != EXPECTED_FINAL_LABEL_COUNTS:
        raise RuntimeError(
            "Frozen final-label composition changed; refusing implicit remapping: "
            f"{dict(sorted(final_counts.items()))}"
        )
    return final, {
        "units": len(final), "disagreements": len(disagreements),
        "missing_rationales": len(missing_rationales),
        "missing_rationale_tile_ids": sorted(missing_rationales),
        "rationales_imputed": False,
    }


def validate_scene_split(records: list[dict], split_path: str | Path) -> dict[str, str]:
    """Consume the Phase63A JSON lock, with CSV support for isolated tests."""
    split_path = Path(split_path)
    if split_path.suffix.lower() == ".json":
        document = json.loads(split_path.read_text(encoding="utf-8"))
        if document.get("phase") != "63A-scene-freeze" or document.get("status") != "locked_scene_disjoint":
            raise RuntimeError("Phase63A did not produce a valid scene-disjoint split lock")
        if document.get("scene_disjoint_audit", {}).get("passed") is not True:
            raise RuntimeError("Phase63A split lock does not certify scene disjointness")
        if document.get("review_integrity", {}).get("missing_rationales_preserved_without_imputation") is not True:
            raise RuntimeError("Phase63A lock does not preserve the 79 missing rationales")
        lock_rows = document.get("records")
        if not isinstance(lock_rows, list):
            raise RuntimeError("Phase63A split lock has no records")
        rows = []
        expected_records = {row["tile_id"]: row for row in records}
        for row in lock_rows:
            if row.get("tile_id") not in expected_records:
                raise RuntimeError(f"Unknown tile in Phase63A split lock: {row.get('tile_id')}")
            sealed_row = expected_records[row["tile_id"]]
            if row.get("scene") != sealed_row["scene"] or row.get("biome") != sealed_row["biome"]:
                raise RuntimeError(f"Phase63A altered scene/biome for {row['tile_id']}")
            split = {
                "risk_score_development_only": "development",
                "locked_final_validation_only": "confirmation",
            }.get(row.get("phase63_role"))
            if split is None:
                raise RuntimeError(f"Unknown Phase63A role for {row['tile_id']}: {row.get('phase63_role')}")
            rows.append({"tile_id": row["tile_id"], "split": split})
    else:
        rows = read_csv(split_path)
        if not rows or not {"tile_id", "split"}.issubset(rows[0]):
            raise RuntimeError("Phase63 split must have tile_id,split columns")
    mapping = {}
    for row in rows:
        tile_id, split = row["tile_id"].strip(), row["split"].strip().lower()
        if tile_id in mapping or split not in {"development", "confirmation"}:
            raise RuntimeError(f"Invalid or duplicate Phase63 split row: {row}")
        mapping[tile_id] = split
    expected = {row["tile_id"] for row in records}
    if set(mapping) != expected:
        raise RuntimeError("Phase63 split IDs do not exactly match all 210 frozen units")
    counts = Counter(mapping.values())
    if counts != Counter({"development": 160, "confirmation": 50}):
        raise RuntimeError(f"Expected frozen 160/50 split, found {dict(counts)}")
    scenes = defaultdict(set)
    for row in records:
        scenes[mapping[row["tile_id"]]].add(row["scene"])
    overlap = scenes["development"] & scenes["confirmation"]
    if overlap:
        raise RuntimeError(f"Scene leakage across Phase63 split: {sorted(overlap)}")
    return mapping


def normalize_semantic_scores(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim < 1 or values.shape[0] != 4 or not np.isfinite(values).all():
        raise RuntimeError(f"Expected finite four-class scores, got {values.shape}")
    if np.any(values < 0):
        shifted = values - values.max(axis=0, keepdims=True)
        values = np.exp(shifted)
    denominator = values.sum(axis=0, keepdims=True)
    if np.any(denominator <= 0):
        raise RuntimeError("Model scores cannot be normalized to probabilities")
    probabilities = values / denominator
    if not np.isfinite(probabilities).all():
        raise RuntimeError("Non-finite model probabilities")
    return probabilities


def normalized_entropy(probabilities: np.ndarray) -> float:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    return float(-(probabilities * np.log(probabilities.clip(1e-12))).sum() / math.log(4.0))


def jensen_shannon(probabilities: list[np.ndarray]) -> float:
    stack = np.asarray(probabilities, dtype=np.float64)
    mean = stack.mean(axis=0)
    divergences = (stack * (np.log(stack.clip(1e-12)) - np.log(mean.clip(1e-12)))).sum(axis=1)
    return float(divergences.mean() / math.log(2.0))


def boundary_distance(prediction: np.ndarray, y: int, x: int) -> float:
    prediction = np.asarray(prediction)
    boundary = np.zeros(prediction.shape, dtype=bool)
    horizontal = prediction[:, 1:] != prediction[:, :-1]
    vertical = prediction[1:, :] != prediction[:-1, :]
    boundary[:, 1:] |= horizontal
    boundary[:, :-1] |= horizontal
    boundary[1:, :] |= vertical
    boundary[:-1, :] |= vertical
    coordinates = np.argwhere(boundary)
    if not len(coordinates):
        return float(math.hypot(*prediction.shape))
    difference = coordinates - np.asarray([y, x])
    return float(np.sqrt(np.min(np.sum(difference * difference, axis=1))))


def local_observability(
    rgb: np.ndarray, y: int, x: int, radius: int,
    spectral: np.ndarray | None = None,
) -> dict[str, float | int]:
    height, width = rgb.shape[:2]
    y0, y1 = max(0, y - radius), min(height, y + radius + 1)
    x0, x1 = max(0, x - radius), min(width, x + radius + 1)
    patch = np.asarray(rgb[y0:y1, x0:x1], dtype=np.float64)
    if patch.size == 0 or not np.isfinite(patch).all():
        raise RuntimeError("Invalid RGB observability window")
    scaled = patch / 255.0
    gray = scaled.mean(axis=2)
    gradients = []
    if gray.shape[0] > 1:
        gradients.append(np.abs(np.diff(gray, axis=0)).reshape(-1))
    if gray.shape[1] > 1:
        gradients.append(np.abs(np.diff(gray, axis=1)).reshape(-1))
    gradient = np.concatenate(gradients) if gradients else np.zeros(1)
    rgb_nodata_fraction = float(np.all(patch == 0, axis=2).mean())
    result = {
        "rgb_nodata_fraction": rgb_nodata_fraction,
        "brightness_mean": float(gray.mean()),
        "brightness_std": float(gray.std()),
        "texture_gradient_mean": float(gradient.mean()),
        "texture_gradient_std": float(gradient.std()),
        "spectral_cache_available": int(spectral is not None),
        "spectral_available_channels": 3,
        "spectral_missing_fraction": 0.5 + 0.5 * rgb_nodata_fraction,
    }
    if spectral is not None:
        spectral = np.asarray(spectral, dtype=np.float64)
        if spectral.ndim != 3 or spectral.shape[0] < 3:
            raise RuntimeError("Spectral cache must be CHW with at least NIR/SWIR1/SWIR2")
        sy0, sy1 = y0 * spectral.shape[1] // height, max(y0 * spectral.shape[1] // height + 1, math.ceil(y1 * spectral.shape[1] / height))
        sx0, sx1 = x0 * spectral.shape[2] // width, max(x0 * spectral.shape[2] // width + 1, math.ceil(x1 * spectral.shape[2] / width))
        local = spectral[:3, sy0:sy1, sx0:sx1]
        finite = np.isfinite(local)
        result.update({
            "spectral_available_channels": 6,
            "spectral_missing_fraction": float(
                0.5 * rgb_nodata_fraction + 0.5 * (~finite | (local == 0)).mean()
            ),
            "nir_mean": float(np.nanmean(np.where(finite, local[0], np.nan))),
            "swir1_mean": float(np.nanmean(np.where(finite, local[1], np.nan))),
            "swir2_mean": float(np.nanmean(np.where(finite, local[2], np.nan))),
        })
        if not all(math.isfinite(result[key]) for key in ("nir_mean", "swir1_mean", "swir2_mean")):
            raise RuntimeError("Spectral cache window contains no finite observation")
    else:
        result.update({"nir_mean": "", "swir1_mean": "", "swir2_mean": ""})
    return result


def point_evidence(
    adapted_views: list[np.ndarray], source_probabilities: np.ndarray,
    y: int, x: int, independent_cloud_probability: float | None,
) -> dict:
    if not adapted_views:
        raise RuntimeError("At least one adapted prediction view is required")
    points = [view[:, y, x] for view in adapted_views]
    adapted, source = points[0], source_probabilities[:, y, x]
    sorted_probability = np.sort(adapted)
    thin, shadow = adapted[CLASS_INDEX["thin_cloud"]], adapted[CLASS_INDEX["cloud_shadow"]]
    top1 = int(np.argmax(adapted))
    view_top1 = np.asarray([np.argmax(item) for item in points])
    result = {
        "adapted_prediction": CORE4[top1],
        "adapted_probability_clear": float(adapted[0]),
        "adapted_probability_thick_cloud": float(adapted[1]),
        "adapted_probability_thin_cloud": float(thin),
        "adapted_probability_cloud_shadow": float(shadow),
        "softmax_entropy": normalized_entropy(adapted),
        "top1_top2_margin": float(sorted_probability[-1] - sorted_probability[-2]),
        "top1_probability": float(sorted_probability[-1]),
        "margin_uncertainty": float(1.0 - sorted_probability[-1] + sorted_probability[-2]),
        "tta_scale_js_divergence": jensen_shannon(points),
        "tta_top1_disagreement": float(1.0 - np.bincount(view_top1, minlength=4).max() / len(points)),
        "source_adapted_js_divergence": jensen_shannon([source, adapted]),
        "source_adapted_top1_disagreement": int(np.argmax(source) != top1),
        "thin_shadow_log_probability_ratio": float(math.log(max(thin, 1e-12) / max(shadow, 1e-12))),
        "thin_shadow_ambiguity": float(math.exp(-abs(math.log(max(thin, 1e-12) / max(shadow, 1e-12))))),
    }
    coarse3 = np.asarray([0, 1, 1, 2], dtype=np.int8)
    coarse2 = np.asarray([0, 1, 1, 0], dtype=np.int8)
    result.update({
        "tta_coarse3_top1_disagreement": float(
            1.0 - np.bincount(coarse3[view_top1], minlength=3).max() / len(points)
        ),
        "tta_binary_top1_disagreement": float(
            1.0 - np.bincount(coarse2[view_top1], minlength=2).max() / len(points)
        ),
        "source_adapted_coarse3_disagreement": int(coarse3[np.argmax(source)] != coarse3[top1]),
        "source_adapted_binary_disagreement": int(coarse2[np.argmax(source)] != coarse2[top1]),
    })
    fine_cloud = float(adapted[CLASS_INDEX["thick_cloud"]] + thin)
    if independent_cloud_probability is None:
        result.update({
            "hierarchy_conflict": "",
            "hierarchy_feature_status": "unavailable_structurally_identical_without_independent_coarse_head",
            "fine_cloud_probability": fine_cloud,
            "independent_cloud_probability": "",
        })
    else:
        if not 0.0 <= independent_cloud_probability <= 1.0:
            raise RuntimeError("Independent coarse cloud probability must be in [0,1]")
        result.update({
            "hierarchy_conflict": abs(independent_cloud_probability - fine_cloud),
            "hierarchy_feature_status": "available_independent_coarse_probability",
            "fine_cloud_probability": fine_cloud,
            "independent_cloud_probability": independent_cloud_probability,
        })
    return result


def label_outcomes(labels: frozenset[str], predicted: str) -> dict:
    fine_evaluable = labels.issubset(CORE4_SET)
    fine_correct = bool(fine_evaluable and predicted in labels)
    coarse_truths = {COARSE_MAP[label] for label in labels if label in COARSE_MAP}
    coarse_evaluable = bool(fine_evaluable and len(coarse_truths) == 1)
    coarse_prediction = COARSE_MAP[predicted]
    binary_truths = {BINARY_MAP[label] for label in labels if label in BINARY_MAP}
    binary_evaluable = bool(fine_evaluable and len(binary_truths) == 1)
    binary_prediction = BINARY_MAP[predicted]
    return {
        "fine_evaluable_core4": int(fine_evaluable),
        "fine_correct": int(fine_correct),
        "fine_failure": int(not fine_correct),
        "coarse_evaluable": int(coarse_evaluable),
        "coarse_truth": next(iter(coarse_truths)) if coarse_evaluable else "",
        "coarse_prediction": coarse_prediction,
        "coarse_correct": int(coarse_evaluable and coarse_prediction in coarse_truths),
        "binary_evaluable": int(binary_evaluable),
        "binary_truth": next(iter(binary_truths)) if binary_evaluable else "",
        "binary_prediction": binary_prediction,
        "binary_correct": int(binary_evaluable and binary_prediction in binary_truths),
        "thin_membership": int("thin_cloud" in labels),
        "shadow_membership": int("cloud_shadow" in labels),
    }


def read_independent_coarse(path: str | Path | None, expected_ids: set[str]) -> dict[str, float]:
    if path is None:
        return {}
    rows = read_csv(path)
    if not rows or not {"tile_id", "p_cloud"}.issubset(rows[0]):
        raise RuntimeError("Independent coarse file must have tile_id,p_cloud")
    result = {row["tile_id"].strip(): float(row["p_cloud"]) for row in rows}
    if set(result) != expected_ids or not all(math.isfinite(value) and 0 <= value <= 1 for value in result.values()):
        raise RuntimeError("Independent coarse probabilities must cover all 210 units with finite [0,1] values")
    return result


def load_rgb(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if rgb.shape != (512, 512, 3):
        raise RuntimeError(f"Phase63 requires exact 512x512 RGB input, got {rgb.shape}: {path}")
    return rgb


def predict_views(
    wrapper, rgb: np.ndarray, scales: tuple[int, ...], include_horizontal_flip: bool = True,
) -> list[np.ndarray]:
    import torch
    import torch.nn.functional as functional

    tensor = torch.from_numpy(rgb.transpose(2, 0, 1).copy())[None].cuda().float()
    views = []
    with torch.inference_mode():
        base = wrapper(tensor)
        views.append(normalize_semantic_scores(base[0].float().cpu().numpy()))
        if include_horizontal_flip:
            flipped = wrapper(tensor.flip(-1)).flip(-1)
            views.append(normalize_semantic_scores(flipped[0].float().cpu().numpy()))
        for size in scales:
            if size == 512:
                continue
            scaled = functional.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False)
            output = wrapper(scaled)
            output = functional.interpolate(output.float(), size=(512, 512), mode="bilinear", align_corners=False)
            views.append(normalize_semantic_scores(output[0].cpu().numpy()))
    return views


def build_wrapper(config: str, checkpoint: str):
    os.environ.setdefault("XFORMERS_DISABLED", "1")
    tools_root = Path(__file__).resolve().parent
    repo_root = tools_root.parent
    for item in (repo_root, tools_root):
        if str(item) not in sys.path:
            sys.path.insert(0, str(item))
    from export_phase12_onnx import build_wrapper as project_build_wrapper

    namespace = argparse.Namespace(
        config=config, checkpoint=checkpoint, precision="fp16", active_block_indices=None,
    )
    wrapper, _ = project_build_wrapper(namespace)
    return wrapper


def spectral_for_image(root: str | Path | None, name: str) -> np.ndarray | None:
    if root is None:
        return None
    path = Path(root) / f"{Path(name).stem}.npz"
    if not path.is_file():
        return None
    with np.load(path) as item:
        if "auxiliary" not in item:
            raise RuntimeError(f"Missing auxiliary in spectral cache: {path}")
        return item["auxiliary"][:3].astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sealed", default="work_dirs/phase61d2_calibrated_review/sealed_manifest.json")
    parser.add_argument("--reviewer-a", default="work_dirs/phase61d2_calibrated_review/phase61d2_reviewer_packet/reviewer_A_main.csv")
    parser.add_argument("--reviewer-b", default="work_dirs/phase61d2_calibrated_review/reviewer_B_main_canonical.csv")
    parser.add_argument("--adjudication", default="work_dirs/phase61d2_calibrated_review/completed_reviews/adjudicator_completed_raw.csv")
    parser.add_argument("--split-lock", required=True, help="B-group Phase63A JSON split lock")
    parser.add_argument("--protocol-manifest", default="work_dirs/phase45_protocol_audit/scene_disjoint_manifest.csv")
    parser.add_argument("--source-config", default="configs/protocol/phase22_clean_v8_l1c.py")
    parser.add_argument("--adapted-config", default="configs/protocol/phase52a_fix_shadow_status_1pct_l1c.py")
    parser.add_argument("--source-checkpoint", default="work_dirs/phase22_clean_v8/seed42/best_mIoU_iter_40000.pth")
    parser.add_argument("--adapted-checkpoint", default="work_dirs/phase52a_fix_shadow_status_1pct/seed52/best_mIoU_iter_1000.pth")
    parser.add_argument("--expected-source-sha256", default=SOURCE_CHECKPOINT_FINGERPRINT)
    parser.add_argument("--expected-adapted-sha256", default=ADAPTED_CHECKPOINT_FINGERPRINT)
    parser.add_argument("--spectral-cache-root", default="work_dirs/phase54_information_audit/cache/val")
    parser.add_argument("--independent-coarse-csv")
    parser.add_argument(
        "--output",
        default="work_dirs/phase63_selective_multigranularity/unreliability_evidence.csv",
    )
    parser.add_argument(
        "--metadata-output",
        default="work_dirs/phase63_selective_multigranularity/unreliability_evidence.meta.json",
    )
    parser.add_argument("--local-radius", type=int, default=8)
    parser.add_argument("--scales", type=int, nargs="+", default=(448, 512, 560))
    args = parser.parse_args()

    paths = {
        "sealed": Path(args.sealed), "reviewer_a": Path(args.reviewer_a),
        "reviewer_b": Path(args.reviewer_b), "adjudication": Path(args.adjudication),
        "protocol_manifest": Path(args.protocol_manifest),
    }
    frozen_hashes = {name: verify_hash(path, EXPECTED_FILES[name], name) for name, path in paths.items()}
    sealed = json.loads(paths["sealed"].read_text(encoding="utf-8"))
    records = [row for row in sealed["records"] if row["cohort"] != "calibration"]
    if len(records) != 210 or sealed.get("calibration_units_excluded_from_statistics") != 40:
        raise RuntimeError("Expected exactly 210 main units and 40 excluded calibration units")
    if sealed.get("protocol_manifest_sha256") != frozen_hashes["protocol_manifest"]:
        raise RuntimeError("Sealed document references a different protocol manifest")
    split = validate_scene_split(records, args.split_lock)
    final_labels, label_diagnostic = assemble_frozen_labels(
        records, paths["reviewer_a"], paths["reviewer_b"], paths["adjudication"],
    )
    manifest_rows = {row["name"]: row for row in read_csv(paths["protocol_manifest"])}
    if any(row["name"] not in manifest_rows for row in records):
        raise RuntimeError("A frozen review unit is absent from the protocol manifest")
    independent_coarse = read_independent_coarse(args.independent_coarse_csv, set(final_labels))

    source_sha, adapted_sha = sha256(args.source_checkpoint), sha256(args.adapted_checkpoint)
    verify_fingerprint(source_sha, args.expected_source_sha256, "Phase22 seed42 iter40000")
    verify_fingerprint(adapted_sha, args.expected_adapted_sha256, "Phase52A-fix seed52 iter1000")

    by_image = defaultdict(list)
    for record in records:
        by_image[record["name"]].append(record)
    evidence = {}
    adapted_wrapper = build_wrapper(args.adapted_config, args.adapted_checkpoint)
    for image_index, (name, image_records) in enumerate(sorted(by_image.items()), 1):
        rgb = load_rgb(manifest_rows[name]["image_path"])
        views = predict_views(adapted_wrapper, rgb, tuple(args.scales))
        spectral = spectral_for_image(args.spectral_cache_root, name)
        prediction = views[0].argmax(axis=0)
        for record in image_records:
            y, x = int(record["center_y"]), int(record["center_x"])
            if not (0 <= y < 512 and 0 <= x < 512):
                raise RuntimeError(f"Center outside image: {record['tile_id']}")
            row = point_evidence(views, views[0], y, x, independent_coarse.get(record["tile_id"]))
            row["boundary_distance_pixels"] = boundary_distance(prediction, y, x)
            row["boundary_proximity"] = 1.0 / (1.0 + row["boundary_distance_pixels"])
            row.update(local_observability(rgb, y, x, args.local_radius, spectral))
            evidence[record["tile_id"]] = row
        print(f"adapted evidence: {image_index}/{len(by_image)} images", flush=True)
    del adapted_wrapper
    try:
        import torch
        torch.cuda.empty_cache()
    except ImportError:
        pass

    source_wrapper = build_wrapper(args.source_config, args.source_checkpoint)
    for image_index, (name, image_records) in enumerate(sorted(by_image.items()), 1):
        rgb = load_rgb(manifest_rows[name]["image_path"])
        source = predict_views(source_wrapper, rgb, (512,), include_horizontal_flip=False)[0]
        for record in image_records:
            tile_id = record["tile_id"]
            y, x = int(record["center_y"]), int(record["center_x"])
            adapted_point = np.asarray([
                evidence[tile_id][f"adapted_probability_{name}"] for name in CORE4
            ])
            source_point = source[:, y, x]
            evidence[tile_id]["source_adapted_js_divergence"] = jensen_shannon([source_point, adapted_point])
            evidence[tile_id]["source_adapted_top1_disagreement"] = int(
                np.argmax(source_point) != np.argmax(adapted_point)
            )
            coarse3 = np.asarray([0, 1, 1, 2], dtype=np.int8)
            coarse2 = np.asarray([0, 1, 1, 0], dtype=np.int8)
            evidence[tile_id]["source_adapted_coarse3_disagreement"] = int(
                coarse3[np.argmax(source_point)] != coarse3[np.argmax(adapted_point)]
            )
            evidence[tile_id]["source_adapted_binary_disagreement"] = int(
                coarse2[np.argmax(source_point)] != coarse2[np.argmax(adapted_point)]
            )
        print(f"source evidence: {image_index}/{len(by_image)} images", flush=True)

    output_rows = []
    for record in records:
        tile_id = record["tile_id"]
        labels = final_labels[tile_id]
        row = {
            "tile_id": tile_id, "split": split[tile_id], "scene": record["scene"],
            "biome": record["biome"], "name": record["name"],
            "center_y": int(record["center_y"]), "center_x": int(record["center_x"]),
            "context_crop_size": int(record["crop_size"]),
            "final_label_set": "|".join(sorted(labels)),
            **evidence[tile_id],
        }
        row.update(label_outcomes(labels, row["adapted_prediction"]))
        output_rows.append(row)
    if len(output_rows) != 210 or any(not math.isfinite(float(row["fine_failure"])) for row in output_rows):
        raise RuntimeError("Invalid Phase63 output rows")

    output = Path(args.output)
    metadata_output = Path(args.metadata_output)
    if output.exists() or metadata_output.exists():
        raise RuntimeError("Refusing to overwrite an existing Phase63 evidence export")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    label_counts = Counter(label for labels in final_labels.values() for label in labels)
    metadata = {
        "phase": "63B-evidence-export", "status": "complete", "units": 210,
        "unit_of_analysis": "point-level center_y/center_x with 128px human context",
        "forbidden_interpretation": "These results are not pixel-dense labels or dense mIoU.",
        "calibration_units_excluded": 40,
        "split": {"development": 160, "confirmation_locked": 50, "scene_overlap": 0},
        "frozen_hashes": frozen_hashes,
        "split_lock": {"path": str(args.split_lock), "sha256": sha256(args.split_lock)},
        "checkpoints": {
            "source": {"identity": "Phase22 seed42 iter40000", "sha256": source_sha},
            "adapted": {"identity": "Phase52A-fix seed52 iter1000", "sha256": adapted_sha},
        },
        "label_diagnostic": label_diagnostic,
        "label_membership_counts": dict(sorted(label_counts.items())),
        "model_output_class_order": list(CORE4),
        "fine_failure_definition": "non-core4/set-ineligible final label OR adapted fine top1 not in allowed core4 set",
        "coarse_mapping": COARSE_MAP,
        "binary_mapping": BINARY_MAP,
        "non_core_coarse_policy": "fail-closed as coarse_evaluable=0; never mapped to clear",
        "thin_shadow_coverage_denominator": "explicit final-label membership only",
        "region_aggregation": {
            "model_correctness": "center point only",
            "uncertainty": "center point only",
            "observability": f"local {(2 * args.local_radius + 1)}x{(2 * args.local_radius + 1)} neighborhood",
            "boundary_distance": "center-to-nearest adapted top1 boundary in the 512x512 tile",
        },
        "probability_construction": "normalize nonnegative Mask2Former semantic scores across four classes",
        "tta": {"horizontal_flip": True, "scales": list(args.scales)},
        "hierarchy_feature": (
            "independent coarse p_cloud supplied" if independent_coarse
            else "unavailable/structurally_identical: P(thin)+P(thick) is not an independent coarse head"
        ),
        "spectral_cache_root": args.spectral_cache_root,
        "output": {"path": str(output), "sha256": sha256(output)},
        "trained_new_network": False,
    }
    metadata_output.write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(output), "metadata": str(metadata_output), "units": 210}, indent=2))


if __name__ == "__main__":
    main()
