import argparse
import json
import math
from pathlib import Path
import statistics

from mmseg_log_metrics import parse_last_evaluation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase31-summary", required=True)
    parser.add_argument("--phase31-root", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--min-run-miou", type=float, default=64.5)
    parser.add_argument("--min-mean-miou", type=float, default=65.3)
    parser.add_argument("--max-std-miou", type=float, default=0.75)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    phase31 = json.loads(Path(args.phase31_summary).read_text(encoding="utf-8"))
    if phase31.get("phase") != 31 or phase31.get("passed") is not True:
        raise SystemExit("Phase 33 requires passed Phase 31")
    if phase31.get("internal_test_evaluated") is not False:
        raise SystemExit("CloudSEN internal test is not sealed")

    runs = []
    for seed in (42, 123, 3407):
        run_dir = (
            Path(args.phase31_root) / "seed42"
            if seed == 42
            else Path(args.root) / f"seed{seed}"
        )
        evaluation = parse_last_evaluation(run_dir / "val_eval.log")
        checkpoints = sorted(run_dir.glob("best_mIoU_iter_*.pth"))
        if len(checkpoints) != 1:
            raise SystemExit(f"Expected one best checkpoint in {run_dir}")
        runs.append({
            "seed": seed,
            "checkpoint": checkpoints[0].as_posix(),
            "validation": evaluation["aggregate"],
            "per_class": evaluation["per_class"],
        })

    mious = [float(row["validation"]["mIoU"]) for row in runs]
    mean_miou = statistics.fmean(mious)
    std_miou = statistics.stdev(mious)
    speedup = float(phase31["speedup"])
    parameters_m = float(phase31["candidate_benchmark"]["parameters_m"])
    finite = all(
        math.isfinite(float(value))
        for row in runs for value in row["validation"].values()
    )
    gates = {
        "all_runs_mIoU": all(value >= args.min_run_miou for value in mious),
        "mean_mIoU": mean_miou >= args.min_mean_miou,
        "std_mIoU": std_miou <= args.max_std_miou,
        "frozen_speedup": speedup >= 2.0,
        "frozen_parameters_m": parameters_m <= 3.0,
        "finite_metrics": finite,
        "internal_test_sealed": True,
    }
    result = {
        "phase": 33,
        "selection_split": "val",
        "internal_test_evaluated": False,
        "test_evaluated": False,
        "thresholds": {
            "min_run_mIoU": args.min_run_miou,
            "min_mean_mIoU": args.min_mean_miou,
            "max_std_mIoU": args.max_std_miou,
            "min_speedup": 2.0,
            "max_parameters_m": 3.0,
        },
        "runs": runs,
        "mean_mIoU": mean_miou,
        "std_mIoU": std_miou,
        "frozen_speedup": speedup,
        "frozen_parameters_m": parameters_m,
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
