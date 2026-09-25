"""Score two or more completed Phase 61D blind-review CSV files.

The default is deliberately strict: every sealed unit must have a valid label.
An explicit blank policy may instead mark blanks as incomplete or as reviewer
abstentions caused by solid-color imagery with no assessable information.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from prepare_phase61d_blind_review import LABELS


ORIGINAL_LABELS = LABELS[:4]
LABEL_ALIASES = {
    # One observed spreadsheet truncation; this is a spelling repair, not a
    # semantic imputation. All other unknown values remain fatal.
    "ambiguous haze/cirru": "ambiguous haze/cirrus",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_review(path, expected_ids, blank_policy="error"):
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {"tile_id", "label"}
    if not rows or not required.issubset(rows[0]):
        raise RuntimeError(f"Missing required CSV columns in {path}: {sorted(required)}")
    ids = [row["tile_id"].strip() for row in rows]
    duplicates = sorted(tile_id for tile_id, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise RuntimeError(f"Duplicate reviewer IDs in {path}: {duplicates}")
    if set(ids) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(ids))
        extra = sorted(set(ids) - set(expected_ids))
        raise RuntimeError(f"Reviewer IDs do not match sealed manifest: {path}; missing={missing}; extra={extra}")

    values, aliases = {}, Counter()
    for row in rows:
        value = row["label"].strip()
        if value in LABEL_ALIASES:
            aliases[value] += 1
            value = LABEL_ALIASES[value]
        values[row["tile_id"].strip()] = value or None
    invalid = sorted({value for value in values.values() if value is not None and value not in LABELS})
    if invalid:
        raise RuntimeError(f"Invalid labels in {path}: {invalid}")
    blank = [tile_id for tile_id in expected_ids if values[tile_id] is None]
    if blank and blank_policy == "error":
        raise RuntimeError(f"Blank labels in {path}: {blank}")
    diagnostics = {
        "path": str(path),
        "sha256": sha256(path),
        "sealed_units": len(expected_ids),
        "labeled_units": len(expected_ids) - len(blank),
        "blank_units": len(blank),
        "blank_tile_ids": blank,
        "blank_interpretation": (
            "unassessable_solid_image" if blank_policy == "unassessable_solid_image"
            else "incomplete" if blank_policy == "incomplete" else None
        ),
        "canonicalized_labels": dict(sorted(aliases.items())),
    }
    return values, diagnostics


def cohen(left, right):
    count = len(left)
    if count == 0:
        raise RuntimeError("No complete cases for Cohen kappa")
    observed = sum(a == b for a, b in zip(left, right)) / count
    left_counts, right_counts = Counter(left), Counter(right)
    expected = sum(left_counts[label] * right_counts[label] for label in LABELS) / (count * count)
    return (observed - expected) / max(1.0 - expected, 1e-12), observed


def fleiss(matrix):
    matrix = np.asarray(matrix, dtype=np.int64)
    if matrix.shape[0] == 0:
        raise RuntimeError("No complete cases for Fleiss kappa")
    raters = matrix.sum(1)
    if not np.all(raters == raters[0]) or raters[0] < 2:
        raise RuntimeError("Fleiss kappa requires the same two or more raters per unit")
    agreement = ((matrix * matrix).sum(1) - raters) / (raters * (raters - 1))
    proportions = matrix.sum(0) / matrix.sum()
    expected = float((proportions * proportions).sum())
    return float((agreement.mean() - expected) / max(1.0 - expected, 1e-12))


def class_iou(left, right, labels):
    result = {}
    left = np.asarray(left)
    right = np.asarray(right)
    for label in labels:
        left_mask = left == label
        right_mask = right == label
        intersection = int(np.count_nonzero(left_mask & right_mask))
        union = int(np.count_nonzero(left_mask | right_mask))
        result[label] = {
            "iou_percent": 100.0 * intersection / union if union else None,
            "intersection": intersection,
            "union": union,
            "left_count": int(left_mask.sum()),
            "right_count": int(right_mask.sum()),
        }
    return result


def mean_or_none(values):
    return float(np.mean(values)) if values else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sealed", default="work_dirs/phase61/blind_review/sealed_manifest.json")
    parser.add_argument("--reviews", nargs="+", required=True)
    parser.add_argument("--output", default="work_dirs/phase61/blind_review/review_metrics.json")
    blank_group = parser.add_mutually_exclusive_group()
    blank_group.add_argument(
        "--allow-incomplete", action="store_true",
        help="Score only units completed by every reviewer and mark output preliminary.",
    )
    blank_group.add_argument(
        "--blank-means-unassessable-solid-image", action="store_true",
        help="Treat blank cells as documented reviewer abstentions for solid-color, non-assessable imagery.",
    )
    args = parser.parse_args()
    if len(args.reviews) < 2:
        raise RuntimeError("Phase 61D requires at least two independent reviewer files")

    sealed_path = Path(args.sealed)
    sealed = json.loads(sealed_path.read_text(encoding="utf-8"))
    records = sealed["records"]
    ids = [row["tile_id"] for row in records]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate tile IDs in sealed manifest")
    record_by_id = {row["tile_id"]: row for row in records}

    blank_policy = (
        "unassessable_solid_image" if args.blank_means_unassessable_solid_image
        else "incomplete" if args.allow_incomplete else "error"
    )
    parsed = [read_review(path, ids, blank_policy) for path in args.reviews]
    reviews = [item[0] for item in parsed]
    diagnostics = [item[1] for item in parsed]
    complete_ids = [tile_id for tile_id in ids if all(review[tile_id] is not None for review in reviews)]
    if not complete_ids:
        raise RuntimeError("No units have labels from every reviewer")
    complete_id_set = set(complete_ids)
    excluded_ids = [tile_id for tile_id in ids if tile_id not in complete_id_set]
    blank_any = len(complete_ids) != len(ids)

    pairwise = {}
    for left in range(len(reviews)):
        for right in range(left + 1, len(reviews)):
            pair_ids = [tile_id for tile_id in ids if reviews[left][tile_id] and reviews[right][tile_id]]
            left_labels = [reviews[left][tile_id] for tile_id in pair_ids]
            right_labels = [reviews[right][tile_id] for tile_id in pair_ids]
            kappa, agreement = cohen(left_labels, right_labels)
            pairwise[f"{left}-{right}"] = {
                "reviewer_files": [Path(args.reviews[left]).name, Path(args.reviews[right]).name],
                "complete_case_units": len(pair_ids),
                "cohen_kappa": kappa,
                "exact_agreement": agreement,
                "class_iou_between_reviewers": class_iou(left_labels, right_labels, LABELS),
            }

    count_matrix = np.asarray([
        [sum(review[tile_id] == label for review in reviews) for label in LABELS]
        for tile_id in complete_ids
    ])
    consensus = [
        LABELS[int(np.argmax(row))] if np.count_nonzero(row == row.max()) == 1 else "no_consensus"
        for row in count_matrix
    ]
    original = [record_by_id[tile_id]["original_label_name"] for tile_id in complete_ids]
    consensus_mask = np.asarray([value != "no_consensus" for value in consensus])

    reviewer_vs_original = {}
    for index, review in enumerate(reviews):
        reviewer_ids = [tile_id for tile_id in ids if review[tile_id] is not None]
        reviewer_vs_original[str(index)] = {
            "reviewer_file": Path(args.reviews[index]).name,
            "units": len(reviewer_ids),
            "exact_agreement": mean_or_none([
                review[tile_id] == record_by_id[tile_id]["original_label_name"] for tile_id in reviewer_ids
            ]),
            "class_iou": class_iou(
                [review[tile_id] for tile_id in reviewer_ids],
                [record_by_id[tile_id]["original_label_name"] for tile_id in reviewer_ids],
                ORIGINAL_LABELS,
            ),
        }

    original_thin_shadow_ids = [
        tile_id for tile_id in complete_ids
        if record_by_id[tile_id]["original_label_name"] in ("definite thin", "definite cloud shadow")
    ]
    thin_shadow_calls = [
        tile_id for tile_id in complete_ids
        if all(review[tile_id] in ("definite thin", "definite cloud shadow") for review in reviews)
    ]

    biome = {}
    complete_index = {tile_id: index for index, tile_id in enumerate(complete_ids)}
    for name in sorted({row["biome"] for row in records}):
        biome_ids = [tile_id for tile_id in complete_ids if record_by_id[tile_id]["biome"] == name]
        biome_consensus = [consensus[complete_index[tile_id]] for tile_id in biome_ids]
        biome_original = [record_by_id[tile_id]["original_label_name"] for tile_id in biome_ids]
        agreed = [value != "no_consensus" for value in biome_consensus]
        biome[name] = {
            "sealed_units": sum(row["biome"] == name for row in records),
            "complete_case_units": len(biome_ids),
            "reviewer_exact_agreement": mean_or_none([
                len({review[tile_id] for review in reviews}) == 1 for tile_id in biome_ids
            ]),
            "consensus_rate": mean_or_none(agreed),
            "original_consensus_agreement": mean_or_none([
                predicted == truth for predicted, truth in zip(biome_consensus, biome_original)
                if predicted != "no_consensus"
            ]),
        }

    consensus_valid = np.asarray(consensus)[consensus_mask]
    original_valid = np.asarray(original)[consensus_mask]
    if blank_any and blank_policy == "unassessable_solid_image":
        status = "complete_with_unassessable_units"
        scope_note = (
            "Metrics exclude reviewer abstentions explicitly documented as solid-color imagery "
            "with no assessable information."
        )
    elif blank_any:
        status = "incomplete_preliminary"
        scope_note = "Complete-case metrics only; do not use as the final Phase 61D conclusion."
    else:
        status = "complete"
        scope_note = None
    summary = {
        "phase": "61D-scored",
        "status": status,
        "human_review_complete": status != "incomplete_preliminary",
        "blank_policy": blank_policy,
        "scope_note": scope_note,
        "sealed_manifest": {"path": str(sealed_path), "sha256": sha256(sealed_path)},
        "reviewers": len(reviews),
        "sealed_units": len(ids),
        "jointly_evaluable_units": len(complete_ids),
        "complete_case_units": len(complete_ids),
        "excluded_from_pairwise_units": len(ids) - len(complete_ids),
        "excluded_tile_ids": excluded_ids,
        "excluded_unit_reason": "unassessable_solid_image" if blank_policy == "unassessable_solid_image" else "incomplete",
        "reviewer_files": diagnostics,
        "pairwise": pairwise,
        "fleiss_kappa_complete_cases": fleiss(count_matrix),
        "thin_shadow_agreement": {
            "original_thin_or_shadow_complete_cases": len(original_thin_shadow_ids),
            "exact_reviewer_agreement_on_original_thin_or_shadow": mean_or_none([
                len({review[tile_id] for review in reviews}) == 1 for tile_id in original_thin_shadow_ids
            ]),
            "both_reviewers_called_thin_or_shadow_units": len(thin_shadow_calls),
            "thin_vs_shadow_agreement_when_both_used_these_labels": mean_or_none([
                len({review[tile_id] for review in reviews}) == 1 for tile_id in thin_shadow_calls
            ]),
        },
        "consensus": {
            "method": "unique plurality; with two reviewers this requires exact agreement",
            "units": int(consensus_mask.sum()),
            "rate_among_complete_cases": float(consensus_mask.mean()),
            "original_consensus_agreement": mean_or_none(list(consensus_valid == original_valid)),
            "class_iou_original_vs_consensus": class_iou(consensus_valid, original_valid, ORIGINAL_LABELS),
        },
        "reviewer_vs_original": reviewer_vs_original,
        "by_biome": biome,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
