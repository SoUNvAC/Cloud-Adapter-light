import argparse
import json
import math
from pathlib import Path

from mmseg_log_metrics import parse_last_evaluation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase26-summary", required=True)
    parser.add_argument("--root", default="work_dirs/phase31_mobilenetv2_litefpn")
    parser.add_argument("--min-miou", type=float, default=65.5)
    parser.add_argument("--min-speedup", type=float, default=2.0)
    parser.add_argument("--max-parameters-m", type=float, default=3.0)
    parser.add_argument(
        "--output",
        default="work_dirs/phase31_mobilenetv2_litefpn/summary.json",
    )
    args = parser.parse_args()

    phase26 = json.loads(Path(args.phase26_summary).read_text(encoding="utf-8"))
    if phase26.get("phase") != 26 or phase26.get("passed") is not True:
        raise RuntimeError("Missing passed Phase 26 benchmark anchor")
    root = Path(args.root)
    evaluation = parse_last_evaluation(root / "seed42" / "val_eval.log")
    benchmark = json.loads((root / "seed42" / "benchmark.json").read_text())
    baseline = phase26["baseline_benchmark"]
    miou = float(evaluation["aggregate"]["mIoU"])
    latency = float(benchmark["latency_mean_ms"])
    speedup = float(baseline["latency_mean_ms"]) / latency
    parameters_m = float(benchmark["parameters_m"])
    finite = all(
        math.isfinite(value) for value in (miou, latency, speedup, parameters_m)
    ) and latency > 0
    gates = {
        "validation_mIoU": miou >= args.min_miou,
        "native_fp16_speedup": speedup >= args.min_speedup,
        "parameters_m": parameters_m <= args.max_parameters_m,
        "finite_metrics": finite,
        "internal_test_sealed": True,
    }
    result = {
        "phase": 31,
        "selection_split": "val",
        "internal_test_evaluated": False,
        "test_evaluated": False,
        "seed": 42,
        "thresholds": {
            "min_mIoU": args.min_miou,
            "min_speedup": args.min_speedup,
            "max_parameters_m": args.max_parameters_m,
        },
        "validation": evaluation["aggregate"],
        "per_class": evaluation["per_class"],
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
