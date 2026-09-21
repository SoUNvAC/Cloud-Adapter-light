import argparse
import json
import math
import re
from pathlib import Path

from mmseg_log_metrics import parse_last_evaluation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--phase31-summary", required=True)
    parser.add_argument("--phase28-summary", required=True)
    parser.add_argument("--min-miou", type=float, default=35.0)
    parser.add_argument("--max-drop", type=float, default=6.5)
    parser.add_argument("--expected-images", type=int, default=2643)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    phase31 = json.loads(Path(args.phase31_summary).read_text(encoding="utf-8"))
    phase28 = json.loads(Path(args.phase28_summary).read_text(encoding="utf-8"))
    if phase31.get("phase") != 31 or phase31.get("passed") is not True:
        raise SystemExit("Phase 32 requires passed Phase 31")
    if phase31.get("internal_test_evaluated") is not False:
        raise SystemExit("Phase 31 internal test seal is invalid")
    if phase28.get("phase") != 28:
        raise SystemExit("Phase 28 baseline summary is invalid")

    evaluation = parse_last_evaluation(Path(args.log))
    metrics = evaluation["aggregate"]
    miou = float(metrics["mIoU"])
    v8_mean = float(phase28["models"]["v8"]["mean_mIoU"])
    drop = v8_mean - miou
    text = Path(args.log).read_text(encoding="utf-8", errors="replace")
    progress = re.findall(r"Iter\(test\)\s*\[\s*(\d+)\s*/\s*(\d+)\]", text)
    observed_images = int(progress[-1][1]) if progress else None
    finite = all(math.isfinite(float(value)) for value in metrics.values())
    gates = {
        "external_mIoU": miou >= args.min_miou,
        "drop_from_v8_mean": drop <= args.max_drop,
        "finite_metrics": finite,
        "expected_images": observed_images == args.expected_images,
        "internal_test_sealed": True,
    }
    result = {
        "phase": 32,
        "evaluation_dataset": "Landsat-8 Biome test",
        "selection_split": "external_test_zero_shot",
        "external_test_evaluated": True,
        "internal_test_evaluated": False,
        "test_evaluated": False,
        "label_remap": [0, 3, 2, 1],
        "checkpoint_source": "phase31_seed42_best_val",
        "thresholds": {
            "min_mIoU": args.min_miou,
            "max_drop_from_v8_mean": args.max_drop,
            "expected_images": args.expected_images,
        },
        "metrics": metrics,
        "per_class": evaluation["per_class"],
        "observed_images": observed_images,
        "v8_external_mean_mIoU": v8_mean,
        "drop_from_v8_mean": drop,
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
