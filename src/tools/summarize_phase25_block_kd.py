import argparse
import json
from pathlib import Path
import statistics

from mmseg_log_metrics import parse_last_evaluation


WEAK_CLASSES = ("thin cloud", "cloud shadow")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Gate paired three-seed weak-class/boundary KD results."
    )
    parser.add_argument("--phase23-summary", required=True)
    parser.add_argument("--phase24-summary", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 3407])
    parser.add_argument("--max-run-paired-drop", type=float, default=1.0)
    parser.add_argument("--max-mean-paired-drop", type=float, default=0.50)
    parser.add_argument("--max-std-miou", type=float, default=0.60)
    parser.add_argument("--max-mean-weak-drop", type=float, default=0.50)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def weak_miou(run):
    per_class = run.get("per_class", {})
    missing = [name for name in WEAK_CLASSES if name not in per_class]
    if missing:
        raise RuntimeError(f"Missing weak-class IoU values: {missing}")
    return statistics.fmean(per_class[name]["IoU"] for name in WEAK_CLASSES)


def main():
    args = parse_args()
    phase23 = json.loads(Path(args.phase23_summary).read_text(encoding="utf-8"))
    phase24 = json.loads(Path(args.phase24_summary).read_text(encoding="utf-8"))
    if not phase23.get("passed") or phase23.get("test_evaluated") is not False:
        raise RuntimeError("Phase 23 has not passed with test sealed")
    if not phase24.get("passed") or phase24.get("test_evaluated") is not False:
        raise RuntimeError("Phase 24 has not passed with test sealed")
    candidate_row = next(
        (row for row in phase24["candidates"] if row["name"] == args.candidate),
        None,
    )
    if candidate_row is None or not candidate_row.get("qualifies"):
        raise RuntimeError(f"Candidate {args.candidate!r} did not qualify in Phase 24")

    baseline_by_seed = {int(run["seed"]): run for run in phase23["runs"]}
    if set(args.seeds) != set(baseline_by_seed):
        raise RuntimeError("Phase 23 seeds do not match Phase 25")

    runs = []
    root = Path(args.root)
    for seed in args.seeds:
        run_dir = root / f"seed{seed}"
        checkpoints = sorted(run_dir.glob("best_mIoU_iter_*.pth"))
        if len(checkpoints) != 1:
            raise RuntimeError(
                f"Expected exactly one best checkpoint in {run_dir}, got {checkpoints}"
            )
        evaluation = parse_last_evaluation(run_dir / "val_eval.log")
        current = {
            "seed": seed,
            "checkpoint": checkpoints[0].as_posix(),
            "validation": evaluation["aggregate"],
            "per_class": evaluation["per_class"],
        }
        baseline = baseline_by_seed[seed]
        current["paired_mIoU_drop_from_v12"] = (
            baseline["validation"]["mIoU"] - current["validation"]["mIoU"]
        )
        current["weak_mIoU"] = weak_miou(current)
        current["paired_weak_mIoU_drop_from_v12"] = (
            weak_miou(baseline) - current["weak_mIoU"]
        )
        runs.append(current)

    mious = [run["validation"]["mIoU"] for run in runs]
    drops = [run["paired_mIoU_drop_from_v12"] for run in runs]
    weak_drops = [run["paired_weak_mIoU_drop_from_v12"] for run in runs]
    mean_drop = statistics.fmean(drops)
    mean_weak_drop = statistics.fmean(weak_drops)
    std_miou = statistics.stdev(mious)
    gates = {
        "candidate_qualified_in_phase24": True,
        "all_run_paired_drops": all(
            drop <= args.max_run_paired_drop for drop in drops
        ),
        "mean_paired_drop": mean_drop <= args.max_mean_paired_drop,
        "std_miou": std_miou <= args.max_std_miou,
        "mean_weak_drop": mean_weak_drop <= args.max_mean_weak_drop,
    }
    result = {
        "phase": 25,
        "candidate": args.candidate,
        "active_block_indices": candidate_row["benchmark"]["active_block_indices"],
        "phase24_native_fp16_speedup": candidate_row["speedup"],
        "selection_split": "val",
        "test_evaluated": False,
        "thresholds": {
            "max_run_paired_drop": args.max_run_paired_drop,
            "max_mean_paired_drop": args.max_mean_paired_drop,
            "max_std_miou": args.max_std_miou,
            "max_mean_weak_drop": args.max_mean_weak_drop,
        },
        "runs": runs,
        "mean_mIoU": statistics.fmean(mious),
        "std_mIoU": std_miou,
        "mean_paired_mIoU_drop_from_v12": mean_drop,
        "mean_paired_weak_mIoU_drop_from_v12": mean_weak_drop,
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
