import argparse
import json
import math
from pathlib import Path

from mmseg_log_metrics import parse_last_evaluation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase22-summary", required=True)
    parser.add_argument("--root", default="work_dirs/phase26_resnet18_pilot")
    parser.add_argument("--min-miou", type=float, default=68.0)
    parser.add_argument("--min-speedup", type=float, default=1.25)
    parser.add_argument("--max-parameters-m", type=float, default=18.0)
    parser.add_argument(
        "--output", default="work_dirs/phase26_resnet18_pilot/summary.json"
    )
    args = parser.parse_args()

    phase22 = json.loads(Path(args.phase22_summary).read_text(encoding="utf-8"))
    if phase22.get("phase") != 22 or phase22.get("passed") is not True:
        raise RuntimeError("Phase 22 V8 anchor has not passed")
    if phase22.get("test_evaluated") is not False:
        raise RuntimeError("Phase 22 did not keep test sealed")

    root = Path(args.root)
    evaluation = parse_last_evaluation(root / "seed42" / "val_eval.log")
    baseline = json.loads((root / "baseline_benchmark.json").read_text())
    candidate = json.loads((root / "seed42" / "benchmark.json").read_text())
    baseline_latency = float(baseline["latency_mean_ms"])
    candidate_latency = float(candidate["latency_mean_ms"])
    candidate_miou = float(evaluation["aggregate"]["mIoU"])
    parameter_count = float(candidate["parameters_m"])
    speedup = baseline_latency / candidate_latency
    finite_latencies = all(
        math.isfinite(value) and value > 0
        for value in (baseline_latency, candidate_latency)
    )
    gates = {
        "validation_mIoU": candidate_miou >= args.min_miou,
        "native_fp16_speedup": speedup >= args.min_speedup,
        "parameters_m": parameter_count <= args.max_parameters_m,
        "finite_positive_latencies": finite_latencies,
    }
    result = {
        "phase": 26,
        "selection_split": "val",
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
        "candidate_benchmark": candidate,
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
