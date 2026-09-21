import argparse
import json
import math
from pathlib import Path
import statistics

from mmseg_log_metrics import parse_last_evaluation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="work_dirs/phase28_l8_external")
    parser.add_argument("--min-compact-miou", type=float, default=35.0)
    parser.add_argument("--max-mean-drop", type=float, default=6.5)
    parser.add_argument("--max-compact-std", type=float, default=1.0)
    parser.add_argument(
        "--output", default="work_dirs/phase28_l8_external/summary.json"
    )
    args = parser.parse_args()

    root = Path(args.root)
    rows = {}
    for model in ("v8", "resnet18"):
        runs = []
        for seed in (42, 123, 3407):
            evaluation = parse_last_evaluation(root / model / f"seed{seed}.log")
            runs.append(
                {
                    "seed": seed,
                    "metrics": evaluation["aggregate"],
                    "per_class": evaluation["per_class"],
                }
            )
        values = [float(run["metrics"]["mIoU"]) for run in runs]
        rows[model] = {
            "runs": runs,
            "mean_mIoU": statistics.fmean(values),
            "std_mIoU": statistics.stdev(values),
        }

    compact_values = [
        float(run["metrics"]["mIoU"]) for run in rows["resnet18"]["runs"]
    ]
    mean_drop = rows["v8"]["mean_mIoU"] - rows["resnet18"]["mean_mIoU"]
    all_metrics = [
        float(run["metrics"]["mIoU"])
        for model in rows.values()
        for run in model["runs"]
    ]
    gates = {
        "all_compact_mIoU": all(
            value >= args.min_compact_miou for value in compact_values
        ),
        "mean_drop_from_v8": mean_drop <= args.max_mean_drop,
        "compact_std_mIoU": rows["resnet18"]["std_mIoU"] <= args.max_compact_std,
        "finite_metrics": all(math.isfinite(value) for value in all_metrics),
        "internal_test_sealed": True,
    }
    result = {
        "phase": 28,
        "evaluation_dataset": "Landsat-8 Biome test",
        "selection_split": "external_test_zero_shot",
        "external_test_evaluated": True,
        "internal_test_evaluated": False,
        "test_evaluated": False,
        "label_remap": [0, 3, 2, 1],
        "thresholds": {
            "min_compact_mIoU": args.min_compact_miou,
            "max_mean_drop": args.max_mean_drop,
            "max_compact_std": args.max_compact_std,
        },
        "models": rows,
        "mean_compact_drop_from_v8": mean_drop,
        "gates": gates,
        "passed": all(gates.values()),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
