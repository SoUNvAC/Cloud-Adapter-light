"""Phase 61C: collapse identical four-class predictions into coarser taxonomies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


TAXONOMIES = {
    "four_class": {
        "names": ("clear", "thick", "thin", "shadow"),
        "source_map": np.asarray([0, 1, 2, 3]),
    },
    "three_class": {
        "names": ("clear", "cloud", "shadow"),
        "source_map": np.asarray([0, 1, 1, 2]),
    },
    "two_class_cloud_noncloud": {
        "names": ("non_cloud", "cloud"),
        # Cloud shadow is non-cloud in the literal cloud/non-cloud taxonomy.
        "source_map": np.asarray([0, 1, 1, 0]),
    },
    "two_class_clear_contaminated": {
        "names": ("clear", "contaminated"),
        "source_map": np.asarray([0, 1, 1, 1]),
    },
}


def reorder_confusion(confusion, current_order):
    current_order = list(current_order)
    desired = ["clear", "thick", "thin", "shadow"]
    indices = [current_order.index(name) for name in desired]
    value = np.asarray(confusion, dtype=np.int64)
    return value[np.ix_(indices, indices)]


def collapse(confusion, mapping, names):
    count = len(names)
    result = np.zeros((count, count), dtype=np.int64)
    for truth in range(4):
        for prediction in range(4):
            result[mapping[truth], mapping[prediction]] += confusion[truth, prediction]
    true_positive = np.diag(result).astype(np.float64)
    ground_truth, predicted = result.sum(1), result.sum(0)
    union = ground_truth + predicted - true_positive
    iou = np.divide(true_positive, union, out=np.zeros(count), where=union > 0)
    recall = np.divide(true_positive, ground_truth, out=np.zeros(count), where=ground_truth > 0)
    return {
        "mIoU": 100.0 * float(iou.mean()),
        "class_iou": dict(zip(names, (100.0 * iou).tolist())),
        "class_recall": dict(zip(names, (100.0 * recall).tolist())),
        "confusion": result.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase52", default="work_dirs/phase52a_fix_shadow_status_1pct/summary.json")
    parser.add_argument("--phase56", default="work_dirs/phase56_readout/dense_readout_summary.json")
    parser.add_argument("--phase60", default="work_dirs/phase60/phase60b_summary.json")
    parser.add_argument("--output", default="work_dirs/phase61/phase61c_granularity.json")
    args = parser.parse_args()

    p52 = json.loads(Path(args.phase52).read_text(encoding="utf-8"))
    p56 = json.loads(Path(args.phase56).read_text(encoding="utf-8"))
    p60 = json.loads(Path(args.phase60).read_text(encoding="utf-8"))
    predictions = {
        "phase52a_source_only": reorder_confusion(
            p52["corrected_source_only_baseline"]["confusion"],
            ("clear", "shadow", "thin", "thick"),
        ),
        "phase52a_adapted": reorder_confusion(
            p52["target"]["confusion"],
            ("clear", "shadow", "thin", "thick"),
        ),
        "phase56_selected_readout": reorder_confusion(
            p56["metrics"]["confusion"],
            ("clear", "thick", "thin", "shadow"),
        ),
        "phase60_random65_readout": reorder_confusion(
            p60["results"]["random@65"]["dense_validation"]["confusion"],
            ("clear", "thick", "thin", "shadow"),
        ),
        "phase60_oracle_relation325_upper_bound": reorder_confusion(
            p60["results"]["oracle_thin_shadow_relation@325"]["dense_validation"]["confusion"],
            ("clear", "thick", "thin", "shadow"),
        ),
    }
    results = {}
    for prediction_name, confusion in predictions.items():
        results[prediction_name] = {
            name: collapse(confusion, descriptor["source_map"], descriptor["names"])
            for name, descriptor in TAXONOMIES.items()
        }
        four = results[prediction_name]["four_class"]["mIoU"]
        results[prediction_name]["recovery_vs_four_class"] = {
            name: item["mIoU"] - four
            for name, item in results[prediction_name].items()
            if isinstance(item, dict) and "mIoU" in item and name != "four_class"
        }

    summary = {
        "phase": "61C",
        "principle": "Each row reuses one fixed 4x4 confusion matrix; only ground-truth and prediction IDs are collapsed.",
        "taxonomies": {
            name: {"classes": list(item["names"]), "source_order_map": item["source_map"].tolist()}
            for name, item in TAXONOMIES.items()
        },
        "results": results,
        "interpretation_rule": (
            "Large recovery for three/two classes with persistent four-class failure supports a semantic-boundary mismatch; "
            "it does not by itself prove dataset label incompatibility without Phase 61D independent review."
        ),
        "target_test_evaluated": False,
        "cloudsen_internal_test_evaluated": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        name: {
            taxonomy: round(metric["mIoU"], 4)
            for taxonomy, metric in row.items() if isinstance(metric, dict) and "mIoU" in metric
        }
        for name, row in results.items()
    }, indent=2))


if __name__ == "__main__":
    main()
