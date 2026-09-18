import argparse
import json
from pathlib import Path
import statistics

from mmseg_log_metrics import parse_last_evaluation


METHODS = ("frozen", "rein", "cloud_adapter")
WEAK_CLASSES = ("thin cloud", "cloud shadow")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Gate the paired three-seed Phase 29 fair PEFT baselines."
    )
    parser.add_argument("--audit", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 3407])
    parser.add_argument("--min-run-miou", type=float, default=50.0)
    parser.add_argument("--max-std-miou", type=float, default=0.75)
    parser.add_argument("--min-cloud-gain-vs-frozen", type=float, default=1.0)
    parser.add_argument("--min-cloud-weak-gain-vs-frozen", type=float, default=1.0)
    parser.add_argument("--min-cloud-difference-vs-rein", type=float, default=-0.50)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def weak_miou(run):
    return statistics.fmean(run["per_class"][name]["IoU"] for name in WEAK_CLASSES)


def main():
    args = parse_args()
    audit = json.loads(Path(args.audit).read_text(encoding="utf-8"))
    if not audit.get("passed") or audit.get("test_evaluated") is not False:
        raise RuntimeError("Phase 29 parameter/schedule audit did not pass")
    root = Path(args.root)
    methods = {}
    for method in METHODS:
        runs = []
        for seed in args.seeds:
            run_dir = root / method / f"seed{seed}"
            checkpoints = sorted(run_dir.glob("best_mIoU_iter_*.pth"))
            if len(checkpoints) != 1:
                raise RuntimeError(
                    f"Expected one best checkpoint in {run_dir}, got {checkpoints}"
                )
            evaluation = parse_last_evaluation(run_dir / "val_eval.log")
            run = {
                "seed": seed,
                "checkpoint": checkpoints[0].as_posix(),
                "validation": evaluation["aggregate"],
                "per_class": evaluation["per_class"],
            }
            run["weak_mIoU"] = weak_miou(run)
            runs.append(run)
        mious = [run["validation"]["mIoU"] for run in runs]
        methods[method] = {
            "runs": runs,
            "mean_mIoU": statistics.fmean(mious),
            "std_mIoU": statistics.stdev(mious),
            "mean_weak_mIoU": statistics.fmean(run["weak_mIoU"] for run in runs),
        }

    paired_cloud_frozen = []
    paired_cloud_rein = []
    paired_weak_frozen = []
    for index, seed in enumerate(args.seeds):
        cloud = methods["cloud_adapter"]["runs"][index]
        frozen = methods["frozen"]["runs"][index]
        rein = methods["rein"]["runs"][index]
        if any(run["seed"] != seed for run in (cloud, frozen, rein)):
            raise RuntimeError("Method runs are not paired by seed")
        paired_cloud_frozen.append(
            cloud["validation"]["mIoU"] - frozen["validation"]["mIoU"]
        )
        paired_cloud_rein.append(
            cloud["validation"]["mIoU"] - rein["validation"]["mIoU"]
        )
        paired_weak_frozen.append(cloud["weak_mIoU"] - frozen["weak_mIoU"])

    mean_gain_frozen = statistics.fmean(paired_cloud_frozen)
    mean_difference_rein = statistics.fmean(paired_cloud_rein)
    mean_weak_gain_frozen = statistics.fmean(paired_weak_frozen)
    gates = {
        "audit_passed": True,
        "all_runs_sane": all(
            run["validation"]["mIoU"] >= args.min_run_miou
            for method in methods.values()
            for run in method["runs"]
        ),
        "all_method_std": all(
            method["std_mIoU"] <= args.max_std_miou for method in methods.values()
        ),
        "cloud_gain_vs_frozen": mean_gain_frozen
        >= args.min_cloud_gain_vs_frozen,
        "cloud_weak_gain_vs_frozen": mean_weak_gain_frozen
        >= args.min_cloud_weak_gain_vs_frozen,
        "cloud_noninferior_to_rein": mean_difference_rein
        >= args.min_cloud_difference_vs_rein,
    }
    result = {
        "phase": 29,
        "selection_split": "val",
        "test_evaluated": False,
        "seeds": args.seeds,
        "thresholds": {
            "min_run_mIoU": args.min_run_miou,
            "max_std_mIoU": args.max_std_miou,
            "min_cloud_gain_vs_frozen": args.min_cloud_gain_vs_frozen,
            "min_cloud_weak_gain_vs_frozen": args.min_cloud_weak_gain_vs_frozen,
            "min_cloud_difference_vs_rein": args.min_cloud_difference_vs_rein,
        },
        "parameter_audit": audit,
        "methods": methods,
        "paired_cloud_minus_frozen": paired_cloud_frozen,
        "paired_cloud_minus_rein": paired_cloud_rein,
        "paired_cloud_weak_minus_frozen": paired_weak_frozen,
        "mean_cloud_gain_vs_frozen": mean_gain_frozen,
        "mean_cloud_difference_vs_rein": mean_difference_rein,
        "mean_cloud_weak_gain_vs_frozen": mean_weak_gain_frozen,
        "gates": gates,
        "passed": all(gates.values()),
        "stop_loss": {
            "on_failure": "do not claim a Cloud-Adapter-specific advantage",
            "next_direction": "replace the adaptation module with the strongest fair baseline before continuing structured compression",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
