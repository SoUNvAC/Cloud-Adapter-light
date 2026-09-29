"""Fit a shallow Phase 63 risk calibrator and evaluate locked confirmation data.

Only the 160-unit development split is available to fitting.  The 50-unit
confirmation split is passed to prediction after the fitted state is frozen.
The positive AUROC class is ``fine_failure``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


BASE_FEATURES = (
    "softmax_entropy",
    "margin_uncertainty",
    "top1_probability",
    "tta_scale_js_divergence",
    "tta_top1_disagreement",
    "tta_coarse3_top1_disagreement",
    "tta_binary_top1_disagreement",
    "source_adapted_js_divergence",
    "source_adapted_top1_disagreement",
    "source_adapted_coarse3_disagreement",
    "source_adapted_binary_disagreement",
    "boundary_proximity",
    "thin_shadow_ambiguity",
    "rgb_nodata_fraction",
    "brightness_mean",
    "brightness_std",
    "texture_gradient_mean",
    "texture_gradient_std",
    "spectral_cache_available",
    "spectral_available_channels",
    "spectral_missing_fraction",
)
OPTIONAL_ALL_FINITE_FEATURES = ("hierarchy_conflict", "nir_mean", "swir1_mean", "swir2_mean")
REQUIRED_COLUMNS = {
    "tile_id", "split", "scene", "biome", "final_label_set",
    "fine_failure", "fine_evaluable_core4", "coarse_evaluable", "coarse_correct",
    "binary_evaluable", "binary_correct",
    "thin_membership", "shadow_membership", "hierarchy_feature_status",
    *BASE_FEATURES,
}
ALLOWED_MODELS = ("logistic", "gam", "isotonic_entropy")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not REQUIRED_COLUMNS.issubset(rows[0]):
        raise RuntimeError(f"Evidence is empty or missing columns: {sorted(REQUIRED_COLUMNS - set(rows[0] if rows else []))}")
    return rows


def finite_float(row: dict[str, str], key: str) -> float:
    try:
        value = float(row[key])
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"Missing/non-numeric {key} for {row.get('tile_id')}") from error
    if not math.isfinite(value):
        raise RuntimeError(f"Non-finite {key} for {row.get('tile_id')}")
    return value


def validate_evidence(rows: list[dict[str, str]], metadata: dict, evidence_path: str | Path) -> None:
    identifiers = [row["tile_id"] for row in rows]
    if len(rows) != 210 or len(set(identifiers)) != 210:
        raise RuntimeError("Phase63 evidence must contain exactly 210 unique point units")
    counts = Counter(row["split"] for row in rows)
    if counts != Counter({"development": 160, "confirmation": 50}):
        raise RuntimeError(f"Expected development/confirmation=160/50, found {dict(counts)}")
    scenes = defaultdict(set)
    for row in rows:
        scenes[row["split"]].add(row["scene"])
        for key in BASE_FEATURES:
            finite_float(row, key)
        for key in (
            "fine_failure", "fine_evaluable_core4", "coarse_evaluable", "coarse_correct",
            "binary_evaluable", "binary_correct", "thin_membership", "shadow_membership",
        ):
            value = finite_float(row, key)
            if value not in (0.0, 1.0):
                raise RuntimeError(f"Expected binary {key} for {row['tile_id']}")
    overlap = scenes["development"] & scenes["confirmation"]
    if overlap:
        raise RuntimeError(f"Scene leakage in exported evidence: {sorted(overlap)}")
    if metadata.get("units") != 210 or metadata.get("trained_new_network") is not False:
        raise RuntimeError("Evidence metadata does not certify the frozen 210-unit inference-only protocol")
    if metadata.get("output", {}).get("sha256") != sha256(evidence_path):
        raise RuntimeError("Evidence CSV hash does not match its metadata")
    diagnostic = metadata.get("label_diagnostic", {})
    if diagnostic.get("missing_rationales") != 79 or diagnostic.get("rationales_imputed") is not False:
        raise RuntimeError("The 79 missing adjudication rationales were changed or imputed")


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=np.int8).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(labels) != len(scores) or not np.isfinite(scores).all():
        raise RuntimeError("Invalid labels/scores for AUROC")
    positive = labels == 1
    negative = labels == 0
    n_positive, n_negative = int(positive.sum()), int(negative.sum())
    if not n_positive or not n_negative:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    begin = 0
    while begin < len(scores):
        end = begin + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[begin]:
            end += 1
        ranks[order[begin:end]] = 0.5 * (begin + 1 + end)
        begin = end
    return float((ranks[positive].sum() - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative))


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    output = np.empty_like(values)
    positive = values >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def fit_logistic(design: np.ndarray, labels: np.ndarray, l2: float = 1.0) -> dict:
    x = np.asarray(design, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if x.ndim != 2 or len(x) != len(y) or set(np.unique(y)) != {0.0, 1.0}:
        raise RuntimeError("Development fit needs a finite matrix and both error classes")
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    normalized = (x - mean) / scale
    matrix = np.column_stack((np.ones(len(x)), normalized))
    class_count = np.bincount(y.astype(np.int64), minlength=2).astype(np.float64)
    sample_weight = np.where(y == 1, 0.5 / class_count[1], 0.5 / class_count[0]) * len(y)
    coefficient = np.zeros(matrix.shape[1], dtype=np.float64)
    penalty = np.eye(matrix.shape[1], dtype=np.float64) * l2
    penalty[0, 0] = 0.0
    for _ in range(100):
        probability = sigmoid(matrix @ coefficient)
        gradient = matrix.T @ (sample_weight * (probability - y)) + penalty @ coefficient
        curvature = sample_weight * probability * (1.0 - probability)
        hessian = matrix.T @ (curvature[:, None] * matrix) + penalty
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        coefficient -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    if not np.isfinite(coefficient).all():
        raise RuntimeError("Logistic risk fit did not converge to finite coefficients")
    return {"mean": mean, "scale": scale, "coefficient": coefficient, "l2": l2}


def predict_logistic(model: dict, design: np.ndarray) -> np.ndarray:
    normalized = (np.asarray(design, dtype=np.float64) - model["mean"]) / model["scale"]
    return sigmoid(model["coefficient"][0] + normalized @ model["coefficient"][1:])


def gam_basis_fit(x: np.ndarray) -> tuple[np.ndarray, dict]:
    """Fixed-knot additive piecewise-linear basis; no interactions or tuning."""
    x = np.asarray(x, dtype=np.float64)
    knots = np.quantile(x, (0.25, 0.50, 0.75), axis=0)
    parts = []
    for index in range(x.shape[1]):
        parts.append(x[:, index:index + 1])
        for knot in knots[:, index]:
            parts.append(np.maximum(0.0, x[:, index:index + 1] - knot))
    return np.concatenate(parts, axis=1), {"knots": knots}


def gam_basis_apply(x: np.ndarray, state: dict) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    knots = state["knots"]
    parts = []
    for index in range(x.shape[1]):
        parts.append(x[:, index:index + 1])
        for knot in knots[:, index]:
            parts.append(np.maximum(0.0, x[:, index:index + 1] - knot))
    return np.concatenate(parts, axis=1)


def fit_isotonic(scores: np.ndarray, labels: np.ndarray) -> dict:
    """Pool-adjacent-violators isotonic calibration fitted on development only."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if set(np.unique(labels)) != {0.0, 1.0}:
        raise RuntimeError("Isotonic development fit requires both error classes")
    order = np.argsort(scores, kind="mergesort")
    sorted_score, sorted_label = scores[order], labels[order]
    blocks = []
    begin = 0
    while begin < len(scores):
        end = begin + 1
        while end < len(scores) and sorted_score[end] == sorted_score[begin]:
            end += 1
        blocks.append([float(sorted_score[end - 1]), float(sorted_label[begin:end].sum()), end - begin])
        begin = end
    index = 0
    while index < len(blocks) - 1:
        if blocks[index][1] / blocks[index][2] <= blocks[index + 1][1] / blocks[index + 1][2]:
            index += 1
            continue
        blocks[index][0] = blocks[index + 1][0]
        blocks[index][1] += blocks[index + 1][1]
        blocks[index][2] += blocks[index + 1][2]
        blocks.pop(index + 1)
        index = max(index - 1, 0)
    return {
        "upper": np.asarray([block[0] for block in blocks]),
        "probability": np.asarray([block[1] / block[2] for block in blocks]),
    }


def predict_isotonic(model: dict, scores: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(model["upper"], np.asarray(scores), side="left")
    indices = np.clip(indices, 0, len(model["probability"]) - 1)
    return model["probability"][indices]


def feature_schema(development: list[dict[str, str]], all_rows: list[dict[str, str]]) -> dict:
    continuous = list(BASE_FEATURES)
    optional_status = {}
    for key in OPTIONAL_ALL_FINITE_FEATURES:
        available = all(row.get(key, "").strip() != "" for row in all_rows)
        partial = any(row.get(key, "").strip() != "" for row in all_rows)
        if partial and not available and key == "hierarchy_conflict":
            raise RuntimeError("Hierarchy conflict must cover all units or remain unavailable")
        if available:
            for row in all_rows:
                finite_float(row, key)
            continuous.append(key)
        optional_status[key] = (
            "used" if available else "unavailable_partial_not_imputed" if partial else "unavailable"
        )
    hierarchy_status = {row["hierarchy_feature_status"] for row in all_rows}
    if "hierarchy_conflict" in continuous:
        if hierarchy_status != {"available_independent_coarse_probability"}:
            raise RuntimeError("Hierarchy conflict is not backed by an independent coarse probability")
    elif hierarchy_status != {"unavailable_structurally_identical_without_independent_coarse_head"}:
        raise RuntimeError("Unexpected or mixed hierarchy feature provenance")
    biomes = sorted({row["biome"] for row in development})
    return {"continuous": continuous, "development_biomes": biomes, "optional_status": optional_status}


def design_matrix(rows: list[dict[str, str]], schema: dict) -> np.ndarray:
    continuous = np.asarray([
        [finite_float(row, key) for key in schema["continuous"]] for row in rows
    ], dtype=np.float64)
    categorical = np.zeros((len(rows), len(schema["development_biomes"]) + 1), dtype=np.float64)
    lookup = {name: index for index, name in enumerate(schema["development_biomes"])}
    for row_index, row in enumerate(rows):
        categorical[row_index, lookup.get(row["biome"], len(lookup))] = 1.0
    return np.concatenate((continuous, categorical), axis=1)


def fit_risk_model(kind: str, development: list[dict[str, str]], schema: dict) -> dict:
    labels = np.asarray([int(row["fine_failure"]) for row in development], dtype=np.int8)
    if kind == "isotonic_entropy":
        model = fit_isotonic(
            np.asarray([finite_float(row, "softmax_entropy") for row in development]), labels,
        )
        return {"kind": kind, "isotonic": model, "fit_units": len(development)}
    matrix = design_matrix(development, schema)
    if kind == "gam":
        matrix, basis = gam_basis_fit(matrix)
        return {
            "kind": kind, "basis": basis, "logistic": fit_logistic(matrix, labels),
            "fit_units": len(development),
        }
    if kind == "logistic":
        return {"kind": kind, "logistic": fit_logistic(matrix, labels), "fit_units": len(development)}
    raise RuntimeError(f"Deep or unsupported risk model forbidden: {kind}")


def predict_risk(model: dict, rows: list[dict[str, str]], schema: dict) -> np.ndarray:
    if model["kind"] == "isotonic_entropy":
        scores = np.asarray([finite_float(row, "softmax_entropy") for row in rows])
        return predict_isotonic(model["isotonic"], scores)
    matrix = design_matrix(rows, schema)
    if model["kind"] == "gam":
        matrix = gam_basis_apply(matrix, model["basis"])
    return predict_logistic(model["logistic"], matrix)


def selection_metrics(rows: list[dict[str, str]], risk: np.ndarray, coverage: float = 0.60) -> dict:
    if not 0 < coverage <= 1 or len(rows) != len(risk):
        raise RuntimeError("Invalid selective coverage request")
    order = np.argsort(risk, kind="mergesort")
    selected_n = max(1, int(math.ceil(coverage * len(rows))))
    selected = order[:selected_n]
    failure = np.asarray([int(row["fine_failure"]) for row in rows])
    thin = np.asarray([int(row["thin_membership"]) for row in rows], dtype=bool)
    shadow = np.asarray([int(row["shadow_membership"]) for row in rows], dtype=bool)
    coarse_evaluable = np.asarray([int(row["coarse_evaluable"]) for row in rows], dtype=bool)
    coarse_correct = np.asarray([int(row["coarse_correct"]) for row in rows], dtype=bool)
    binary_evaluable = np.asarray([int(row["binary_evaluable"]) for row in rows], dtype=bool)
    binary_correct = np.asarray([int(row["binary_correct"]) for row in rows], dtype=bool)
    selected_mask = np.zeros(len(rows), dtype=bool)
    selected_mask[selected] = True
    baseline_error = float(failure.mean())
    selected_error = float(failure[selected].mean())
    relative = None if baseline_error == 0 else float((baseline_error - selected_error) / baseline_error)
    baseline_coarse = None if not coarse_evaluable.any() else float(coarse_correct[coarse_evaluable].mean())
    selected_coarse_mask = selected_mask & coarse_evaluable
    selected_coarse = None if not selected_coarse_mask.any() else float(coarse_correct[selected_coarse_mask].mean())
    baseline_binary = None if not binary_evaluable.any() else float(binary_correct[binary_evaluable].mean())
    selected_binary_mask = selected_mask & binary_evaluable
    selected_binary = None if not selected_binary_mask.any() else float(binary_correct[selected_binary_mask].mean())
    return {
        "requested_coverage": coverage,
        "selected_units": selected_n,
        "actual_coverage": selected_n / len(rows),
        "risk_threshold_inclusive": float(risk[order[selected_n - 1]]),
        "fine_error_rate_all_210_style_units": baseline_error,
        "fine_error_rate_selected": selected_error,
        "fine_error_relative_reduction": relative,
        "thin_membership_total": int(thin.sum()),
        "thin_membership_selected": int((thin & selected_mask).sum()),
        "thin_coverage": None if not thin.any() else float((thin & selected_mask).sum() / thin.sum()),
        "shadow_membership_total": int(shadow.sum()),
        "shadow_membership_selected": int((shadow & selected_mask).sum()),
        "shadow_coverage": None if not shadow.any() else float((shadow & selected_mask).sum() / shadow.sum()),
        "coarse_accuracy_all_evaluable": baseline_coarse,
        "coarse_accuracy_selected_evaluable_diagnostic": selected_coarse,
        "coarse_accuracy_routed_all_evaluable": baseline_coarse,
        "coarse_routed_change_percentage_points": 0.0 if baseline_coarse is not None else None,
        "coarse_route_note": "Rejection changes granularity, not the frozen adapted coarse prediction; routed-all coarse accuracy is structurally unchanged.",
        "binary_accuracy_all_evaluable": baseline_binary,
        "binary_accuracy_selected_evaluable_diagnostic": selected_binary,
        "binary_accuracy_routed_all_evaluable": baseline_binary,
        "binary_routed_change_percentage_points": 0.0 if baseline_binary is not None else None,
    }


def risk_coverage_curve(rows: list[dict[str, str]], risk: np.ndarray) -> list[dict]:
    return [selection_metrics(rows, risk, coverage / 100.0) for coverage in range(5, 101, 5)]


def subset_report(rows: list[dict[str, str]], risk: np.ndarray) -> dict:
    labels = np.asarray([int(row["fine_failure"]) for row in rows])
    return {
        "units": len(rows),
        "scenes": len({row["scene"] for row in rows}),
        "fine_failures": int(labels.sum()),
        "error_detection_auroc": binary_auc(labels, risk),
        "at_60pct_fine_coverage": selection_metrics(rows, risk, 0.60),
    }


def scene_bootstrap(rows: list[dict[str, str]], risk: np.ndarray, draws: int, seed: int) -> dict:
    scenes = sorted({row["scene"] for row in rows})
    if len(scenes) < 2 or draws < 100:
        raise RuntimeError("Scene bootstrap needs at least two scenes and 100 draws")
    indices = {scene: np.asarray([i for i, row in enumerate(rows) if row["scene"] == scene]) for scene in scenes}
    generator = np.random.default_rng(seed)
    values = defaultdict(list)
    for _ in range(draws):
        sampled_scenes = generator.choice(scenes, size=len(scenes), replace=True)
        sampled = np.concatenate([indices[scene] for scene in sampled_scenes])
        sampled_rows = [rows[index] for index in sampled]
        sampled_risk = risk[sampled]
        auc = binary_auc(
            np.asarray([int(row["fine_failure"]) for row in sampled_rows]), sampled_risk,
        )
        metrics = selection_metrics(sampled_rows, sampled_risk, 0.60)
        if auc is not None:
            values["auroc"].append(auc)
        for source, target in (
            ("fine_error_relative_reduction", "relative_error_reduction"),
            ("thin_coverage", "thin_coverage"),
            ("shadow_coverage", "shadow_coverage"),
        ):
            if metrics[source] is not None:
                values[target].append(metrics[source])
    result = {
        "method": "nonparametric resampling of whole confirmation scenes",
        "draws_requested": draws, "seed": seed, "scene_count": len(scenes),
    }
    for key in ("auroc", "relative_error_reduction", "thin_coverage", "shadow_coverage"):
        data = np.asarray(values[key], dtype=np.float64)
        result[key] = {
            "valid_draws": len(data),
            "ci95": None if not len(data) else [float(value) for value in np.quantile(data, (0.025, 0.975))],
        }
    return result


def serializable_model(model: dict) -> dict:
    def convert(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        return value
    return convert(model)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evidence",
        default="work_dirs/phase63_selective_multigranularity/unreliability_evidence.csv",
    )
    parser.add_argument(
        "--evidence-metadata",
        default="work_dirs/phase63_selective_multigranularity/unreliability_evidence.meta.json",
    )
    parser.add_argument("--risk-model", choices=ALLOWED_MODELS, default="logistic")
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=63)
    parser.add_argument(
        "--output",
        default="work_dirs/phase63_selective_multigranularity/selective_risk_summary.json",
    )
    args = parser.parse_args()

    rows = read_rows(args.evidence)
    metadata = json.loads(Path(args.evidence_metadata).read_text(encoding="utf-8"))
    validate_evidence(rows, metadata, args.evidence)
    development = [row for row in rows if row["split"] == "development"]
    confirmation = [row for row in rows if row["split"] == "confirmation"]
    schema = feature_schema(development, rows)

    # The locked confirmation objects are not passed to fit_risk_model.
    fitted = fit_risk_model(args.risk_model, development, schema)
    development_risk = predict_risk(fitted, development, schema)
    confirmation_risk = predict_risk(fitted, confirmation, schema)
    if not np.isfinite(development_risk).all() or not np.isfinite(confirmation_risk).all():
        raise RuntimeError("Risk model emitted non-finite values")

    development_report = subset_report(development, development_risk)
    confirmation_report = subset_report(confirmation, confirmation_risk)
    bootstrap = scene_bootstrap(
        confirmation, confirmation_risk, args.bootstrap_draws, args.bootstrap_seed,
    )
    snow_indices = [index for index, row in enumerate(confirmation) if row["biome"] == "snow_ice"]
    snow_rows = [confirmation[index] for index in snow_indices]
    snow_report = (
        {"units": 0, "status": "absent_but_not_excluded"}
        if not snow_rows else subset_report(snow_rows, confirmation_risk[snow_indices])
    )
    confirm_at_60 = confirmation_report["at_60pct_fine_coverage"]
    dev_at_60 = development_report["at_60pct_fine_coverage"]
    auc_ci = bootstrap["auroc"]["ci95"]
    valid_auc_draw_fraction = bootstrap["auroc"]["valid_draws"] / args.bootstrap_draws
    gates = {
        "confirmation_error_detection_auroc_at_least_0_70": (
            confirmation_report["error_detection_auroc"] is not None
            and confirmation_report["error_detection_auroc"] >= 0.70
        ),
        "confirmation_scene_bootstrap_auroc_ci_lower_above_0_50": (
            auc_ci is not None and valid_auc_draw_fraction >= 0.95 and auc_ci[0] > 0.50
        ),
        "fine_coverage_at_least_0_60": confirm_at_60["actual_coverage"] >= 0.60,
        "fine_error_relative_reduction_at_least_0_25": (
            confirm_at_60["fine_error_relative_reduction"] is not None
            and confirm_at_60["fine_error_relative_reduction"] >= 0.25
        ),
        "thin_coverage_at_least_0_30": (
            confirm_at_60["thin_coverage"] is not None and confirm_at_60["thin_coverage"] >= 0.30
        ),
        "shadow_coverage_at_least_0_30": (
            confirm_at_60["shadow_coverage"] is not None and confirm_at_60["shadow_coverage"] >= 0.30
        ),
        "coarse_performance_drop_at_most_1pp": (
            confirm_at_60["coarse_routed_change_percentage_points"] is not None
            and confirm_at_60["coarse_routed_change_percentage_points"] >= -1.0
        ),
        "development_confirmation_direction_consistent": (
            development_report["error_detection_auroc"] is not None
            and confirmation_report["error_detection_auroc"] is not None
            and development_report["error_detection_auroc"] > 0.50
            and confirmation_report["error_detection_auroc"] > 0.50
            and dev_at_60["fine_error_relative_reduction"] is not None
            and confirm_at_60["fine_error_relative_reduction"] is not None
            and dev_at_60["fine_error_relative_reduction"] > 0
            and confirm_at_60["fine_error_relative_reduction"] > 0
        ),
        "snow_ice_reported_without_exclusion": "units" in snow_report,
        "confirmation_not_used_for_fit_or_tuning": fitted["fit_units"] == 160,
        "all_210_units_in_fine_failure_target": len(rows) == 210,
    }
    passed = all(gates.values())
    output_document = {
        "phase": "63C-selective-risk", "status": "complete",
        "decision": "continue_selective_multigranularity_route" if passed else "stop_selective_multigranularity_route",
        "passed": passed,
        "protocol": {
            "unit": "point-level center judgment; 128px crop is context only",
            "not_a_dense_metric": True,
            "positive_error_class": "fine_failure",
            "fine_failure": "non-core4/set-ineligible adjudication OR adapted fine top1 outside allowed core4 set",
            "risk_model": args.risk_model,
            "allowed_model_family": list(ALLOWED_MODELS),
            "fit_split": "development only (160)",
            "locked_evaluation_split": "confirmation only (50)",
            "hyperparameter_search_on_confirmation": False,
            "selection": "accept lowest predicted-risk units",
            "primary_coverage": 0.60,
            "thin_shadow_coverage_denominator": "explicit adjudicated membership",
            "coarse_mapping": metadata.get("coarse_mapping"),
            "binary_mapping": metadata.get("binary_mapping"),
            "non_core_coarse_policy": "not evaluable; never coerced to clear",
            "missing_adjudication_rationales": 79,
            "rationales_imputed": False,
        },
        "inputs": {
            "evidence": {"path": str(args.evidence), "sha256": sha256(args.evidence)},
            "metadata": {"path": str(args.evidence_metadata), "sha256": sha256(args.evidence_metadata)},
        },
        "feature_schema": schema,
        "fitted_risk_model": serializable_model(fitted),
        "development": development_report,
        "confirmation_locked": confirmation_report,
        "confirmation_scene_bootstrap": bootstrap,
        "confirmation_risk_coverage_curve": risk_coverage_curve(confirmation, confirmation_risk),
        "snow_ice_confirmation": snow_report,
        "gates": gates,
        "trained_new_network": False,
        "scientific_warning": "Point-level human judgments are uncertain audit labels, not an absolute pixel-dense gold standard.",
    }
    output = Path(args.output)
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite existing Phase63 evaluation: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(output_document, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(output), "passed": passed, "decision": output_document["decision"], "gates": gates}, indent=2))


if __name__ == "__main__":
    main()
