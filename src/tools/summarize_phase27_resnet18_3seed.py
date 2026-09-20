import argparse
import json
from pathlib import Path
import statistics

from mmseg_log_metrics import parse_last_evaluation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase26-summary", required=True)
    parser.add_argument("--phase26-root", required=True)
    parser.add_argument("--root", default="work_dirs/phase27_resnet18_3seed")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 3407])
    parser.add_argument("--min-run-miou", type=float, default=67.0)
    parser.add_argument("--min-mean-miou", type=float, default=67.8)
    parser.add_argument("--max-std-miou", type=float, default=0.60)
    parser.add_argument(
        "--output", default="work_dirs/phase27_resnet18_3seed/summary.json"
    )
    args = parser.parse_args()

    phase26 = json.loads(Path(args.phase26_summary).read_text(encoding="utf-8"))
    if phase26.get("phase") != 26 or phase26.get("passed") is not True:
        raise RuntimeError("Phase 26 pilot has not passed")
    if phase26.get("test_evaluated") is not False:
        raise RuntimeError("Phase 26 did not keep test sealed")

    runs = []
    for seed in args.seeds:
        run_dir = (
            Path(args.phase26_root) / "seed42"
            if seed == 42
            else Path(args.root) / f"seed{seed}"
        )
        evaluation = parse_last_evaluation(run_dir / "val_eval.log")
        checkpoints = sorted(run_dir.glob("best_mIoU_iter_*.pth"))
        if len(checkpoints) != 1:
            raise RuntimeError(f"Expected one best checkpoint in {run_dir}")
        runs.append(
            {
                "seed": seed,
                "checkpoint": checkpoints[0].as_posix(),
                "validation": evaluation["aggregate"],
                "per_class": evaluation["per_class"],
            }
        )

    mious = [float(run["validation"]["mIoU"]) for run in runs]
    mean_miou = statistics.fmean(mious)
    std_miou = statistics.stdev(mious)
    speedup = float(phase26["speedup"])
    parameters_m = float(phase26["candidate_benchmark"]["parameters_m"])
    gates = {
        "all_runs_mIoU": all(value >= args.min_run_miou for value in mious),
        "mean_mIoU": mean_miou >= args.min_mean_miou,
        "std_mIoU": std_miou <= args.max_std_miou,
        "frozen_speedup": speedup >= 1.25,
        "frozen_parameters_m": parameters_m <= 18.0,
    }
    result = {
        "phase": 27,
        "selection_split": "val",
        "test_evaluated": False,
        "thresholds": {
            "min_run_mIoU": args.min_run_miou,
            "min_mean_mIoU": args.min_mean_miou,
            "max_std_mIoU": args.max_std_miou,
            "min_speedup": 1.25,
            "max_parameters_m": 18.0,
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
