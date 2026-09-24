"""Score two or more completed Phase 61D blind-review CSV files."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from prepare_phase61d_blind_review import LABELS


def read_review(path, expected_ids):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    values = {row["tile_id"]: row["label"].strip() for row in rows}
    if set(values) != set(expected_ids):
        raise RuntimeError(f"Reviewer IDs do not match sealed manifest: {path}")
    invalid = {value for value in values.values() if value not in LABELS}
    if invalid:
        raise RuntimeError(f"Blank/invalid labels in {path}: {sorted(invalid)}")
    return values


def cohen(left, right):
    count = len(left)
    observed = sum(a == b for a, b in zip(left, right)) / count
    left_counts, right_counts = Counter(left), Counter(right)
    expected = sum(left_counts[label] * right_counts[label] for label in LABELS) / (count * count)
    return (observed - expected) / max(1.0 - expected, 1e-12), observed


def fleiss(matrix):
    matrix = np.asarray(matrix, dtype=np.int64)
    raters = matrix.sum(1)[0]
    agreement = ((matrix * matrix).sum(1) - raters) / (raters * (raters - 1))
    proportions = matrix.sum(0) / matrix.sum()
    expected = float((proportions * proportions).sum())
    return float((agreement.mean() - expected) / max(1.0 - expected, 1e-12))


def class_iou(consensus, original):
    result = {}
    for label in LABELS[:4]:
        predicted = np.asarray([value == label for value in consensus])
        truth = np.asarray([value == label for value in original])
        intersection = int(np.count_nonzero(predicted & truth))
        union = int(np.count_nonzero(predicted | truth))
        result[label] = 100.0 * intersection / max(union, 1)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sealed", default="work_dirs/phase61/blind_review/sealed_manifest.json")
    parser.add_argument("--reviews", nargs="+", required=True)
    parser.add_argument("--output", default="work_dirs/phase61/blind_review/review_metrics.json")
    args = parser.parse_args()
    sealed = json.loads(Path(args.sealed).read_text(encoding="utf-8"))
    records = sealed["records"]
    ids = [row["tile_id"] for row in records]
    reviews = [read_review(path, ids) for path in args.reviews]
    labels_by_reviewer = [[review[tile_id] for tile_id in ids] for review in reviews]
    pairwise = {}
    for left in range(len(reviews)):
        for right in range(left + 1, len(reviews)):
            kappa, agreement = cohen(labels_by_reviewer[left], labels_by_reviewer[right])
            pairwise[f"{left}-{right}"] = {"cohen_kappa": kappa, "agreement": agreement}
    count_matrix = np.asarray([
        [sum(review[tile_id] == label for review in reviews) for label in LABELS]
        for tile_id in ids
    ])
    consensus = [LABELS[int(np.argmax(row))] if np.count_nonzero(row == row.max()) == 1 else "no_consensus" for row in count_matrix]
    original = [row["original_label_name"] for row in records]
    valid = np.asarray([value != "no_consensus" for value in consensus])
    thin_shadow_indices = [index for index, value in enumerate(original) if value in ("definite thin", "definite cloud shadow")]
    thin_shadow_agreement = float(np.mean([
        len({review[ids[index]] for review in reviews}) == 1 for index in thin_shadow_indices
    ]))
    biome = {}
    for name in sorted({row["biome"] for row in records}):
        indices = [index for index, row in enumerate(records) if row["biome"] == name]
        biome[name] = {
            "units": len(indices),
            "unanimous_fraction": float(np.mean([len({review[ids[index]] for review in reviews}) == 1 for index in indices])),
            "original_consensus_agreement": float(np.mean([
                consensus[index] == original[index] for index in indices if consensus[index] != "no_consensus"
            ])) if any(consensus[index] != "no_consensus" for index in indices) else None,
        }
    summary = {
        "phase": "61D-scored", "reviewers": len(reviews), "units": len(ids),
        "pairwise": pairwise, "fleiss_kappa": fleiss(count_matrix),
        "thin_shadow_unanimous_agreement": thin_shadow_agreement,
        "consensus_rate": float(valid.mean()),
        "original_consensus_agreement": float(np.mean(np.asarray(consensus)[valid] == np.asarray(original)[valid])),
        "class_iou_original_vs_consensus": class_iou(np.asarray(consensus)[valid], np.asarray(original)[valid]),
        "by_biome": biome,
    }
    Path(args.output).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
