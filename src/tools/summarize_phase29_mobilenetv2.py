import argparse
import json
import math
from pathlib import Path

from mmseg_log_metrics import parse_last_evaluation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase26-summary", required=True)
    parser.add_argument("--phase28-summary", required=True)
    parser.add_argument("--root", default="work_dirs/phase29_mobilenetv2_pilot")
    parser.add_argument("--min-val-miou", type=float, default=66.0)
    parser.add_argument("--min-external-miou", type=float, default=35.0)
    parser.add_argument("--max-external-drop", type=float, default=6.5)
    parser.add_argument("--min-speedup", type=float, default=2.0)
    parser.add_argument("--max-parameters-m", type=float, default=7.5)
    parser.add_argument(
        "--output", default="work_dirs/phase29_mobilenetv2_pilot/summary.json"
    )
    args = parser.parse_args()

    phase26 = json.loads(Path(args.phase26_summary).read_text(encoding="utf-8"))
    phase28 = json.loads(Path(args.phase28_summary).read_text(encoding="utf-8"))
    if phase26.get("phase") != 26 or phase26.get("passed") is not True:
        raise RuntimeError("Missing passed Phase 26 benchmark anchor")
    if phase28.get("phase") != 28 or phase28.get("internal_test_evaluated") is not False:
        raise RuntimeError("Invalid Phase 28 external anchor")

    root = Path(args.root)
    validation = parse_last_evaluation(root / "seed42" / "val_eval.log")
    external = parse_last_evaluation(root / "seed42" / "l8_eval.log")
    benchmark = json.loads((root / "seed42" / "benchmark.json").read_text())
    baseline = phase26["baseline_benchmark"]
    baseline_latency = float(baseline["latency_mean_ms"])
    latency = float(benchmark["latency_mean_ms"])
    val_miou = float(validation["aggregate"]["mIoU"])
    external_miou = float(external["aggregate"]["mIoU"])
    v8_external_mean = float(phase28["models"]["v8"]["mean_mIoU"])
    external_drop = v8_external_mean - external_miou
    speedup = baseline_latency / latency
    parameters_m = float(benchmark["parameters_m"])
    finite = all(
        math.isfinite(value)
        for value in (val_miou, external_miou, latency, speedup, parameters_m)
    ) and latency > 0
    gates = {
        "validation_mIoU": val_miou >= args.min_val_miou,
        "external_mIoU": external_miou >= args.min_external_miou,
        "external_drop_from_v8": external_drop <= args.max_external_drop,
        "native_fp16_speedup": speedup >= args.min_speedup,
        "parameters_m": parameters_m <= args.max_parameters_m,
        "finite_metrics": finite,
        "internal_test_sealed": True,
    }
    result = {
        "phase": 29,
        "selection_split": "val_and_external_zero_shot",
        "external_test_evaluated": True,
        "internal_test_evaluated": False,
        "test_evaluated": False,
        "seed": 42,
        "thresholds": {
            "min_val_mIoU": args.min_val_miou,
            "min_external_mIoU": args.min_external_miou,
            "max_external_drop": args.max_external_drop,
            "min_speedup": args.min_speedup,
            "max_parameters_m": args.max_parameters_m,
        },
        "validation": validation["aggregate"],
        "validation_per_class": validation["per_class"],
        "external": external["aggregate"],
        "external_per_class": external["per_class"],
        "v8_external_mean_mIoU": v8_external_mean,
        "external_drop_from_v8": external_drop,
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
