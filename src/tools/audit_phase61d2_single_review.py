"""Audit one completed Phase 61D2 main review without inventing agreement.

This is a deliberately downgraded endpoint for the case where a second
independent reviewer is unavailable.  It reports coverage and descriptive
comparisons, but never emits kappa, AC1, reviewer IoU, consensus, adjudication,
or a four-class hard-gate conclusion.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from score_phase61d2_calibrated_review import canonical, parse_label_set


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_single_review(path, expected_ids):
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {"tile_id", "label_set", "notes"}
    if not rows or not required.issubset(rows[0]):
        raise RuntimeError(f"Missing required columns in {path}: {sorted(required)}")
    ids = [row["tile_id"].strip() for row in rows]
    duplicates = sorted(tile_id for tile_id, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise RuntimeError(f"Duplicate review IDs: {duplicates}")
    if set(ids) != set(expected_ids):
        raise RuntimeError(
            f"Review IDs do not match sealed main cohort; "
            f"missing={sorted(set(expected_ids) - set(ids))}; "
            f"extra={sorted(set(ids) - set(expected_ids))}"
        )
    parsed = {}
    note_count = 0
    for row in rows:
        tile_id = row["tile_id"].strip()
        try:
            parsed[tile_id] = parse_label_set(row["label_set"])
        except ValueError as error:
            raise RuntimeError(f"Invalid label_set for {tile_id}: {error}") from error
        note_count += bool(row["notes"].strip())
    return parsed, {
        "path": str(path), "sha256": sha256(path), "units": len(rows),
        "nonblank_notes": int(note_count),
    }


def original_membership_iou(records, review, labels=("thin_cloud", "cloud_shadow")):
    result = {}
    for label in labels:
        original = [row["original_label_name"] == label for row in records]
        reviewed = [label in review[row["tile_id"]] for row in records]
        intersection = sum(left and right for left, right in zip(original, reviewed))
        union = sum(left or right for left, right in zip(original, reviewed))
        result[label] = {
            "iou": None if union == 0 else intersection / union,
            "iou_percent": None if union == 0 else 100.0 * intersection / union,
            "intersection": intersection,
            "union": union,
            "original_positive": sum(original),
            "single_reviewer_positive": sum(reviewed),
        }
    return result


def descriptive_subset(records, review):
    values = [review[row["tile_id"]] for row in records]
    explicit_nonhard = [
        len(value) > 1
        or bool(value & {"boundary_mixed", "unobservable_nodata", "uncertain"})
        for value in values
    ]
    return {
        "units": len(records),
        "label_counts": dict(sorted(Counter(canonical(value) for value in values).items())),
        "set_valued_units": sum(len(value) > 1 for value in values),
        "boundary_mixed_units": sum(value == {"boundary_mixed"} for value in values),
        "uncertain_units": sum(value == {"uncertain"} for value in values),
        "unobservable_nodata_units": sum(value == {"unobservable_nodata"} for value in values),
        "explicit_ambiguity_or_unassessable_units": sum(explicit_nonhard),
        "explicit_ambiguity_or_unassessable_fraction": (
            sum(explicit_nonhard) / len(records) if records else None
        ),
        "original_vs_single_reviewer_membership_iou_exploratory": original_membership_iou(
            records, review
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sealed", default="work_dirs/phase61d2_calibrated_review/sealed_manifest.json")
    parser.add_argument("--calibration-lock", default="work_dirs/phase61d2_calibrated_review/calibration_lock.json")
    parser.add_argument("--review", required=True)
    parser.add_argument("--reviewer-id", default="reviewer_A")
    parser.add_argument("--reviewer-caveat", required=True)
    parser.add_argument("--output", default="work_dirs/phase61d2_calibrated_review/single_reviewer_audit.json")
    parser.add_argument("--packet-summary", default="work_dirs/phase61d2_calibrated_review/packet_summary.json")
    args = parser.parse_args()

    sealed_path = Path(args.sealed)
    lock_path = Path(args.calibration_lock)
    output = Path(args.output)
    packet_path = Path(args.packet_summary)
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite existing single-review audit: {output}")
    sealed = json.loads(sealed_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "locked_before_independent_review":
        raise RuntimeError("Calibration was not locked before the main review")
    calibration = [row for row in sealed["records"] if row["cohort"] == "calibration"]
    records = [row for row in sealed["records"] if row["cohort"] != "calibration"]
    if len(calibration) != 40 or len(records) != 210:
        raise RuntimeError("Unexpected Phase 61D2 calibration/main cohort sizes")
    review, review_diagnostic = read_single_review(
        args.review, [row["tile_id"] for row in records]
    )
    overall = descriptive_subset(records, review)
    by_cohort = {
        cohort: descriptive_subset([row for row in records if row["cohort"] == cohort], review)
        for cohort in ("independent", "confirmation")
    }
    by_biome = {
        biome: descriptive_subset([row for row in records if row["biome"] == biome], review)
        for biome in sorted({row["biome"] for row in records})
    }
    summary = {
        "phase": "61D2-single-review-audit",
        "status": "single_review_complete_insufficient_for_interrater_inference",
        "evidence_level": "exploratory_single_reviewer_only",
        "reviewer_id": args.reviewer_id,
        "reviewer_caveat_verbatim": args.reviewer_caveat,
        "sealed_manifest": {"path": str(sealed_path), "sha256": sha256(sealed_path)},
        "calibration_lock": {"path": str(lock_path), "sha256": sha256(lock_path)},
        "review_file": review_diagnostic,
        "calibration_units_excluded_from_all_statistics": len(calibration),
        "overall": overall,
        "by_cohort": by_cohort,
        "by_biome": by_biome,
        "unavailable_endpoints": {
            "cohen_kappa": "requires at least two independent reviewers",
            "gwet_ac1": "requires at least two independent reviewers",
            "reviewer_membership_iou": "requires at least two independent reviewers",
            "adjudicated_consensus": "requires independent disagreement and a third expert",
            "annotation_process_shift_gate": "not evaluable from one reviewer",
            "return_to_model_problem_gate": "not evaluable from one reviewer",
        },
        "publication_decision": {
            "four_class_miou_may_be_primary_from_61d2": False,
            "reason": "Independent reviewer reproducibility was not measured.",
            "allowed_use": (
                "Descriptive, hypothesis-generating evidence with the reviewer caveat; "
                "not a gold standard, error-rate estimate, or proof of annotation-policy shift."
            ),
        },
    }
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    if packet.get("status") != "calibration_locked_awaiting_independent_reviews":
        raise RuntimeError(f"Unexpected packet status: {packet.get('status')}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    packet.update({
        "status": "single_review_complete_insufficient_for_interrater_inference",
        "single_reviewer_complete": True,
        "independent_double_review_complete": False,
        "main_reviews_complete": False,
        "agreement_metrics_available": False,
        "formal_human_results_available": False,
        "human_results_available": False,
        "single_reviewer_audit": {"path": str(output), "sha256": sha256(output)},
    })
    packet_path.write_text(json.dumps(packet, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
