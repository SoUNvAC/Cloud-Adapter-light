import argparse
import json
import math
from pathlib import Path
import statistics

from mmseg_log_metrics import parse_last_evaluation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase31-summary", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--min-miou", type=float, default=68.0)
    parser.add_argument("--min-weak-miou", type=float, default=51.0)
    parser.add_argument("--min-speedup", type=float, default=2.0)
    parser.add_argument("--max-parameters-m", type=float, default=3.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    phase31 = json.loads(Path(args.phase31_summary).read_text(encoding="utf-8"))
    root = Path(args.root)
    evaluation = parse_last_evaluation(root / "seed42" / "val_eval.log")
    metrics = evaluation["aggregate"]
    per_class = evaluation["per_class"]
    weak_miou = statistics.fmean(
        [per_class["thin cloud"]["IoU"], per_class["cloud shadow"]["IoU"]]
    )
    benchmark = json.loads((root / "seed42" / "benchmark.json").read_text())
    baseline = phase31["baseline_benchmark"]
    speedup = float(baseline["latency_mean_ms"]) / float(benchmark["latency_mean_ms"])
    parameters_m = float(benchmark["parameters_m"])
    finite = all(
        math.isfinite(float(value))
        for value in [*metrics.values(), weak_miou, speedup, parameters_m]
    )
    gates = {
        "validation_mIoU": float(metrics["mIoU"]) >= args.min_miou,
        "weak_class_mIoU": weak_miou >= args.min_weak_miou,
        "native_fp16_speedup": speedup >= args.min_speedup,
        "parameters_m": parameters_m <= args.max_parameters_m,
        "finite_metrics": finite,
        "internal_test_sealed": True,
    }
    result = {
        "phase": 37,
        "selection_split": "val",
        "internal_test_evaluated": False,
        "test_evaluated": False,
        "seed": 42,
        "class_weights": [1.0, 1.0, 1.5, 1.25, 0.1],
        "thresholds": {
            "min_mIoU": args.min_miou,
            "min_weak_mIoU": args.min_weak_miou,
            "min_speedup": args.min_speedup,
            "max_parameters_m": args.max_parameters_m,
        },
        "validation": metrics,
        "per_class": per_class,
        "weak_class_mIoU": weak_miou,
        "baseline_benchmark": baseline,
        "candidate_benchmark": benchmark,
        "speedup": speedup,
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
