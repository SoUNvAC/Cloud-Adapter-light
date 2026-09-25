"""Score calibrated Phase 61D2 reviews and enforce the publication gates.

Calibration units are structurally excluded.  The third reviewer is accepted
only for exact A/B disagreements; neither raw review is overwritten.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from prepare_phase61d2_calibrated_review import ATOMIC_LABELS, SEMANTIC_LABELS


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_label_set(value):
    raw = value.strip()
    if not raw:
        raise ValueError("blank label_set; use unobservable_nodata or uncertain instead")
    labels = tuple(sorted({part.strip() for part in raw.split("|") if part.strip()}))
    invalid = sorted(set(labels) - set(ATOMIC_LABELS))
    if invalid:
        raise ValueError(f"unknown labels: {invalid}")
    if len(labels) > 2:
        raise ValueError("set-valued labels are limited to the two smallest plausible semantic labels")
    if len(labels) > 1 and not set(labels).issubset(SEMANTIC_LABELS):
        raise ValueError("boundary_mixed, unobservable_nodata, and uncertain must be singleton labels")
    return frozenset(labels)


def canonical(labels):
    return "|".join(sorted(labels))


def read_review(path, expected_ids, label_column="label_set"):
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {"tile_id", label_column}
    if not rows or not required.issubset(rows[0]):
        raise RuntimeError(f"Missing columns {sorted(required)} in {path}")
    ids = [row["tile_id"].strip() for row in rows]
    duplicates = sorted(key for key, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise RuntimeError(f"Duplicate tile IDs in {path}: {duplicates}")
    if set(ids) != set(expected_ids):
        raise RuntimeError(
            f"Review IDs do not match sealed main cohort in {path}; "
            f"missing={sorted(set(expected_ids) - set(ids))}; extra={sorted(set(ids) - set(expected_ids))}"
        )
    result = {}
    for row in rows:
        tile_id = row["tile_id"].strip()
        try:
            result[tile_id] = parse_label_set(row[label_column])
        except ValueError as error:
            raise RuntimeError(f"Invalid {label_column} for {tile_id} in {path}: {error}") from error
    return result, {"path": str(path), "sha256": sha256(path), "units": len(result)}


def cohen_kappa(left, right):
    categories = sorted(set(left) | set(right))
    count = len(left)
    observed = sum(a == b for a, b in zip(left, right)) / count
    left_counts, right_counts = Counter(left), Counter(right)
    expected = sum(left_counts[item] * right_counts[item] for item in categories) / (count * count)
    if expected >= 1.0:
        return 1.0 if observed == 1.0 else None
    return (observed - expected) / (1.0 - expected)


def gwet_ac1(left, right):
    """Multicategory AC1 using the mean category marginals of two raters."""
    categories = sorted(set(left) | set(right))
    count = len(left)
    observed = sum(a == b for a, b in zip(left, right)) / count
    if len(categories) == 1:
        return 1.0
    left_counts, right_counts = Counter(left), Counter(right)
    marginals = np.asarray([
        (left_counts[item] + right_counts[item]) / (2.0 * count) for item in categories
    ])
    expected = float(np.sum(marginals * (1.0 - marginals)) / (len(categories) - 1))
    return (observed - expected) / max(1.0 - expected, 1e-12)


def membership_iou(left, right, labels):
    result = {}
    for label in labels:
        left_mask = np.asarray([label in value for value in left], dtype=bool)
        right_mask = np.asarray([label in value for value in right], dtype=bool)
        intersection = int(np.count_nonzero(left_mask & right_mask))
        union = int(np.count_nonzero(left_mask | right_mask))
        result[label] = {
            "iou": None if union == 0 else intersection / union,
            "iou_percent": None if union == 0 else 100.0 * intersection / union,
            "intersection": intersection,
            "union": union,
            "left_positive": int(left_mask.sum()),
            "right_positive": int(right_mask.sum()),
        }
    return result


def original_vs_final_iou(records, final, labels):
    result = {}
    for label in labels:
        original_mask = np.asarray([row["original_label_name"] == label for row in records])
        final_mask = np.asarray([label in final[row["tile_id"]] for row in records])
        intersection = int(np.count_nonzero(original_mask & final_mask))
        union = int(np.count_nonzero(original_mask | final_mask))
        result[label] = {
            "iou": None if union == 0 else intersection / union,
            "iou_percent": None if union == 0 else 100.0 * intersection / union,
            "intersection": intersection,
            "union": union,
            "original_positive": int(original_mask.sum()),
            "adjudicated_positive": int(final_mask.sum()),
        }
    return result


def disagreement_type(left, right):
    union = left | right
    if "boundary_mixed" in union:
        return "boundary_involved"
    if {"thin_cloud", "thick_cloud"}.issubset(union):
        return "thin_vs_thick"
    if {"clear", "cloud_shadow"}.issubset(union):
        return "clear_vs_cloud_shadow"
    if {"cloud_shadow", "terrain_water_shadow"}.issubset(union):
        return "cloud_shadow_vs_terrain_water_shadow"
    if {"thin_cloud", "cloud_shadow"}.issubset(union):
        return "thin_vs_cloud_shadow"
    if "unobservable_nodata" in union:
        return "unobservable_involved"
    if "uncertain" in union:
        return "uncertain_involved"
    return "other"


def read_calibration_lock(path, sealed):
    path = Path(path)
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("status") != "locked_before_independent_review":
        raise RuntimeError("Calibration lock is absent or not final")
    if document.get("manual_sha256") != sealed["manual_sha256"]:
        raise RuntimeError("Calibration lock references a different operation manual")
    if document.get("calibration_units") != sealed["calibration_units_excluded_from_statistics"]:
        raise RuntimeError("Calibration lock unit count does not match sealed packet")
    return {"path": str(path), "sha256": sha256(path), **document}


def write_adjudication_template(path, disagreements, review_a, review_b):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "tile_id", "reviewer_A_label_set", "reviewer_B_label_set",
            "final_label_set", "rationale",
        ))
        writer.writeheader()
        for tile_id in disagreements:
            writer.writerow({
                "tile_id": tile_id,
                "reviewer_A_label_set": canonical(review_a[tile_id]),
                "reviewer_B_label_set": canonical(review_b[tile_id]),
                "final_label_set": "",
                "rationale": "",
            })


def read_adjudication(path, disagreement_ids, review_a, review_b):
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "tile_id", "reviewer_A_label_set", "reviewer_B_label_set",
        "final_label_set", "rationale",
    }
    if not rows or not required.issubset(rows[0]):
        raise RuntimeError(f"Invalid adjudication columns in {path}")
    ids = [row["tile_id"].strip() for row in rows]
    if set(ids) != set(disagreement_ids) or len(ids) != len(set(ids)):
        raise RuntimeError("Adjudicator file must contain each and only A/B disagreement exactly once")
    final = {}
    for row in rows:
        tile_id = row["tile_id"].strip()
        if row["reviewer_A_label_set"].strip() != canonical(review_a[tile_id]):
            raise RuntimeError(f"Reviewer A value was altered in adjudication row {tile_id}")
        if row["reviewer_B_label_set"].strip() != canonical(review_b[tile_id]):
            raise RuntimeError(f"Reviewer B value was altered in adjudication row {tile_id}")
        if not row["rationale"].strip():
            raise RuntimeError(f"Missing adjudication rationale for {tile_id}")
        try:
            final[tile_id] = parse_label_set(row["final_label_set"])
        except ValueError as error:
            raise RuntimeError(f"Invalid adjudication for {tile_id}: {error}") from error
    return final, {"path": str(path), "sha256": sha256(path), "units": len(rows)}


def subset_summary(records, review_a, review_b):
    ids = [row["tile_id"] for row in records]
    left = [canonical(review_a[tile_id]) for tile_id in ids]
    right = [canonical(review_b[tile_id]) for tile_id in ids]
    return {
        "units": len(ids),
        "exact_agreement": float(np.mean([a == b for a, b in zip(left, right)])),
        "compatible_set_agreement": float(np.mean([
            bool(review_a[tile_id] & review_b[tile_id]) for tile_id in ids
        ])),
        "cohen_kappa_exact_set_categories": cohen_kappa(left, right),
        "gwet_ac1_exact_set_categories": gwet_ac1(left, right),
        "reviewer_membership_iou": membership_iou(
            [review_a[tile_id] for tile_id in ids],
            [review_b[tile_id] for tile_id in ids],
            ("thin_cloud", "cloud_shadow"),
        ),
    }


def gate_decision(agreement, original_iou):
    kappa = agreement["cohen_kappa_exact_set_categories"]
    ac1 = agreement["gwet_ac1_exact_set_categories"]
    thin_reviewer = agreement["reviewer_membership_iou"]["thin_cloud"]["iou"]
    shadow_reviewer = agreement["reviewer_membership_iou"]["cloud_shadow"]["iou"]
    thin_original = original_iou["thin_cloud"]["iou"]
    shadow_original = original_iou["cloud_shadow"]["iou"]
    gates = {
        "four_class_miou_may_be_primary": bool(kappa is not None and ac1 is not None and kappa >= 0.40 and ac1 >= 0.40),
        "terminate_four_class_hard_label_route": bool(
            thin_reviewer is None or shadow_reviewer is None or thin_reviewer < 0.30 or shadow_reviewer < 0.30
        ),
        "supports_annotation_process_shift": bool(
            thin_reviewer is not None and shadow_reviewer is not None
            and thin_original is not None and shadow_original is not None
            and min(thin_reviewer, shadow_reviewer) > 0.60
            and min(thin_original, shadow_original) < 0.40
        ),
        "return_to_model_problem": bool(
            thin_reviewer is not None and shadow_reviewer is not None
            and thin_original is not None and shadow_original is not None
            and min(thin_reviewer, shadow_reviewer) >= 0.60
            and min(thin_original, shadow_original) >= 0.60
        ),
    }
    if gates["terminate_four_class_hard_label_route"]:
        conclusion = "terminate_four_class_hard_label_route"
    elif not gates["four_class_miou_may_be_primary"]:
        conclusion = "four_class_miou_not_primary"
    elif gates["supports_annotation_process_shift"]:
        conclusion = "supports_annotation_process_shift"
    elif gates["return_to_model_problem"]:
        conclusion = "return_to_model_problem"
    else:
        conclusion = "inconclusive_mixed_label_and_model_evidence"
    return {**gates, "conclusion": conclusion}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sealed", default="work_dirs/phase61d2_calibrated_review/sealed_manifest.json")
    parser.add_argument("--calibration-lock", required=True)
    parser.add_argument("--reviewer-a", required=True)
    parser.add_argument("--reviewer-b", required=True)
    parser.add_argument("--adjudication")
    parser.add_argument("--adjudication-template", default="work_dirs/phase61d2_calibrated_review/adjudicator_disagreements.csv")
    parser.add_argument("--output", default="work_dirs/phase61d2_calibrated_review/review_metrics.json")
    args = parser.parse_args()

    sealed_path = Path(args.sealed)
    sealed = json.loads(sealed_path.read_text(encoding="utf-8"))
    calibration_ids = {row["tile_id"] for row in sealed["records"] if row["cohort"] == "calibration"}
    main_records = [row for row in sealed["records"] if row["cohort"] != "calibration"]
    main_ids = [row["tile_id"] for row in main_records]
    if len(calibration_ids) != sealed["calibration_units_excluded_from_statistics"]:
        raise RuntimeError("Calibration cohort count mismatch")
    if len(main_ids) != sealed["main_units"] or calibration_ids & set(main_ids):
        raise RuntimeError("Invalid calibration/main cohort separation")

    calibration_lock = read_calibration_lock(args.calibration_lock, sealed)
    review_a, diagnostic_a = read_review(args.reviewer_a, main_ids)
    review_b, diagnostic_b = read_review(args.reviewer_b, main_ids)
    disagreements = [tile_id for tile_id in main_ids if review_a[tile_id] != review_b[tile_id]]
    if not args.adjudication:
        write_adjudication_template(args.adjudication_template, disagreements, review_a, review_b)

    agreement = subset_summary(main_records, review_a, review_b)
    by_cohort = {
        cohort: subset_summary([row for row in main_records if row["cohort"] == cohort], review_a, review_b)
        for cohort in ("independent", "confirmation")
    }
    by_biome = {
        biome: subset_summary([row for row in main_records if row["biome"] == biome], review_a, review_b)
        for biome in sorted({row["biome"] for row in main_records})
    }
    disagreement_breakdown = Counter(
        disagreement_type(review_a[tile_id], review_b[tile_id]) for tile_id in disagreements
    )
    region_breakdown = Counter(
        next(row["region"] for row in main_records if row["tile_id"] == tile_id)
        for tile_id in disagreements
    )
    summary = {
        "phase": "61D2",
        "status": "awaiting_third_expert_adjudication" if disagreements else "complete_no_disagreements",
        "sealed_manifest": {"path": str(sealed_path), "sha256": sha256(sealed_path)},
        "calibration": calibration_lock,
        "calibration_units_excluded_from_all_reported_statistics": len(calibration_ids),
        "main_units": len(main_records),
        "reviewer_files": {"A": diagnostic_a, "B": diagnostic_b},
        "agreement": agreement,
        "by_cohort": by_cohort,
        "by_biome": by_biome,
        "disagreements": {
            "units": len(disagreements),
            "types": dict(sorted(disagreement_breakdown.items())),
            "regions": dict(sorted(region_breakdown.items())),
            "adjudication_template": None if args.adjudication else str(args.adjudication_template),
        },
        "human_result_warning": "No original-vs-final or hard-gate conclusion exists until disagreement-only adjudication is complete.",
    }

    if args.adjudication:
        adjudicated, adjudication_diagnostic = read_adjudication(
            args.adjudication, disagreements, review_a, review_b
        )
        final = {
            tile_id: review_a[tile_id] if review_a[tile_id] == review_b[tile_id] else adjudicated[tile_id]
            for tile_id in main_ids
        }
        original_iou = original_vs_final_iou(
            main_records, final, ("clear", "thick_cloud", "thin_cloud", "cloud_shadow")
        )
        final_by_cohort = {
            cohort: original_vs_final_iou(
                [row for row in main_records if row["cohort"] == cohort],
                final, ("thin_cloud", "cloud_shadow"),
            )
            for cohort in ("independent", "confirmation")
        }
        final_by_biome = {
            biome: original_vs_final_iou(
                [row for row in main_records if row["biome"] == biome],
                final, ("thin_cloud", "cloud_shadow"),
            )
            for biome in sorted({row["biome"] for row in main_records})
        }
        summary.update({
            "status": "complete_adjudicated",
            "adjudication": adjudication_diagnostic,
            "final_label_provenance": {
                "exact_A_B_agreement_units": len(main_ids) - len(disagreements),
                "third_expert_disagreement_only_units": len(disagreements),
                "raw_A_B_reviews_preserved": True,
            },
            "original_vs_adjudicated_membership_iou": original_iou,
            "original_vs_adjudicated_by_cohort": final_by_cohort,
            "original_vs_adjudicated_by_biome": final_by_biome,
            "hard_gates": gate_decision(agreement, original_iou),
            "human_result_warning": None,
        })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
