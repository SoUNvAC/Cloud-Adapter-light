import argparse
import json
from pathlib import Path

from mmseg_log_metrics import parse_last_evaluation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="work_dirs/phase23_repair_lrsweep")
    parser.add_argument("--multipliers", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--baseline-8k-miou", type=float, default=67.58)
    parser.add_argument("--min-improvement", type=float, default=0.25)
    parser.add_argument("--output", default="work_dirs/phase23_repair_lrsweep/summary.json")
    args = parser.parse_args()

    rows = []
    for mult in args.multipliers:
        run_dir = Path(args.root) / f"mult{mult}_seed42"
        checkpoints = sorted(run_dir.glob("best_mIoU_iter_*.pth"))
        marker = run_dir / "VAL_EVAL_COMPLETE"
        if len(checkpoints) != 1 or not marker.is_file():
            raise RuntimeError(f"Incomplete multiplier {mult} sweep")
        evaluation = parse_last_evaluation(run_dir / "val_eval.log")
        miou = float(evaluation["aggregate"]["mIoU"])
        rows.append(
            {
                "fpn_lr_multiplier": mult,
                "checkpoint": checkpoints[0].as_posix(),
                "validation": evaluation["aggregate"],
                "per_class": evaluation["per_class"],
                "improvement_over_trial_a_5x_at_8k": miou - args.baseline_8k_miou,
                "qualified": miou >= args.baseline_8k_miou + args.min_improvement,
            }
        )

    qualified = [row for row in rows if row["qualified"]]
    selected = sorted(
        qualified,
        key=lambda row: (-row["validation"]["mIoU"], row["fpn_lr_multiplier"]),
    )[0] if qualified else None
    result = {
        "phase": 23,
        "repair": "A",
        "selection_split": "val",
        "test_evaluated": False,
        "baseline_5x_best_8k_mIoU": args.baseline_8k_miou,
        "min_improvement": args.min_improvement,
        "qualification_mIoU": args.baseline_8k_miou + args.min_improvement,
        "candidates": rows,
        "selected_multiplier": None if selected is None else selected["fpn_lr_multiplier"],
        "passed": selected is not None,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if selected is None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
