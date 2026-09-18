import argparse
import json
from pathlib import Path
import statistics

from mmseg_log_metrics import parse_last_evaluation


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize and gate the three-seed Phase 22 validation runs."
    )
    parser.add_argument("--root", default="work_dirs/phase22_clean_v8")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 3407])
    parser.add_argument("--min-run-miou", type=float, default=69.0)
    parser.add_argument("--min-mean-miou", type=float, default=69.5)
    parser.add_argument("--max-std-miou", type=float, default=0.50)
    parser.add_argument(
        "--output",
        default="work_dirs/phase22_clean_v8/summary.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.root)
    runs = []
    for seed in args.seeds:
        run_dir = root / f"seed{seed}"
        log_path = run_dir / "val_eval.log"
        if not log_path.is_file():
            raise FileNotFoundError(f"Missing validation log: {log_path}")
        checkpoints = sorted(run_dir.glob("best_mIoU_iter_*.pth"))
        if len(checkpoints) != 1:
            raise RuntimeError(
                f"Expected exactly one best checkpoint in {run_dir}, got {checkpoints}"
            )
        evaluation = parse_last_evaluation(log_path)
        runs.append(
            {
                "seed": seed,
                "checkpoint": checkpoints[0].as_posix(),
                "validation": evaluation["aggregate"],
                "per_class": evaluation["per_class"],
            }
        )

    mious = [run["validation"]["mIoU"] for run in runs]
    mean_miou = statistics.fmean(mious)
    std_miou = statistics.stdev(mious)
    gates = {
        "all_runs_miou": all(value >= args.min_run_miou for value in mious),
        "mean_miou": mean_miou >= args.min_mean_miou,
        "std_miou": std_miou <= args.max_std_miou,
    }
    result = {
        "phase": 22,
        "selection_split": "val",
        "test_evaluated": False,
        "thresholds": {
            "min_run_miou": args.min_run_miou,
            "min_mean_miou": args.min_mean_miou,
            "max_std_miou": args.max_std_miou,
        },
        "runs": runs,
        "mean_mIoU": mean_miou,
        "std_mIoU": std_miou,
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
