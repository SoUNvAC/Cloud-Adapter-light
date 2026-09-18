import argparse
import json
from pathlib import Path
import statistics

from mmseg_log_metrics import parse_last_evaluation


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize and gate paired clean V8-to-V12 validation runs."
    )
    parser.add_argument("--phase22-summary", required=True)
    parser.add_argument("--root", default="work_dirs/phase23_clean_v12")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 3407])
    parser.add_argument("--min-run-miou", type=float, default=66.5)
    parser.add_argument("--min-mean-miou", type=float, default=67.0)
    parser.add_argument("--max-std-miou", type=float, default=0.60)
    parser.add_argument("--max-mean-paired-drop", type=float, default=3.0)
    parser.add_argument(
        "--output",
        default="work_dirs/phase23_clean_v12/summary.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    phase22_path = Path(args.phase22_summary)
    phase22 = json.loads(phase22_path.read_text(encoding="utf-8"))
    if not phase22.get("passed") or phase22.get("test_evaluated") is not False:
        raise RuntimeError(
            "Phase 22 must pass on validation without evaluating test before Phase 23"
        )
    v8_by_seed = {run["seed"]: run for run in phase22["runs"]}
    if set(args.seeds) != set(v8_by_seed):
        raise RuntimeError(
            f"Seed mismatch: requested={args.seeds}, phase22={sorted(v8_by_seed)}"
        )

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
        v12_metrics = evaluation["aggregate"]
        v8_miou = float(v8_by_seed[seed]["validation"]["mIoU"])
        runs.append(
            {
                "seed": seed,
                "source_v8_checkpoint": v8_by_seed[seed]["checkpoint"],
                "checkpoint": checkpoints[0].as_posix(),
                "validation": v12_metrics,
                "per_class": evaluation["per_class"],
                "paired_mIoU_drop_from_v8": v8_miou - v12_metrics["mIoU"],
            }
        )

    mious = [run["validation"]["mIoU"] for run in runs]
    drops = [run["paired_mIoU_drop_from_v8"] for run in runs]
    mean_miou = statistics.fmean(mious)
    std_miou = statistics.stdev(mious)
    mean_drop = statistics.fmean(drops)
    gates = {
        "phase22_passed": True,
        "all_runs_miou": all(value >= args.min_run_miou for value in mious),
        "mean_miou": mean_miou >= args.min_mean_miou,
        "std_miou": std_miou <= args.max_std_miou,
        "mean_paired_drop": mean_drop <= args.max_mean_paired_drop,
    }
    result = {
        "phase": 23,
        "selection_split": "val",
        "test_evaluated": False,
        "phase22_summary": phase22_path.as_posix(),
        "thresholds": {
            "min_run_miou": args.min_run_miou,
            "min_mean_miou": args.min_mean_miou,
            "max_std_miou": args.max_std_miou,
            "max_mean_paired_drop": args.max_mean_paired_drop,
        },
        "runs": runs,
        "mean_mIoU": mean_miou,
        "std_mIoU": std_miou,
        "mean_paired_mIoU_drop_from_v8": mean_drop,
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
