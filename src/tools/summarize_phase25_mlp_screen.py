import argparse
import json
import math
from pathlib import Path

from mmseg_log_metrics import parse_last_evaluation


CANDIDATES = ("ratio4", "ratio3", "ratio2p5", "ratio2")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase22-summary", required=True)
    parser.add_argument("--root", default="work_dirs/phase25_mlp_screen")
    parser.add_argument("--min-speedup", type=float, default=1.08)
    parser.add_argument("--max-val-miou-drop", type=float, default=5.0)
    parser.add_argument("--output", default="work_dirs/phase25_mlp_screen/summary.json")
    args = parser.parse_args()

    phase22 = json.loads(Path(args.phase22_summary).read_text(encoding="utf-8"))
    if phase22.get("phase") != 22 or phase22.get("passed") is not True:
        raise RuntimeError("Phase 22 V8 anchor has not passed")
    if phase22.get("test_evaluated") is not False:
        raise RuntimeError("Phase 22 did not keep test sealed")

    rows = []
    for name in CANDIDATES:
        candidate = Path(args.root) / name
        evaluation = parse_last_evaluation(candidate / "val_eval.log")
        benchmark = json.loads((candidate / "benchmark.json").read_text(encoding="utf-8"))
        rows.append(
            {
                "name": name,
                "validation": evaluation["aggregate"],
                "per_class": evaluation["per_class"],
                "benchmark": benchmark,
            }
        )

    baseline = rows[0]
    baseline_miou = float(baseline["validation"]["mIoU"])
    baseline_latency = float(baseline["benchmark"]["latency_mean_ms"])
    selected = []
    for row in rows:
        latency = float(row["benchmark"]["latency_mean_ms"])
        row["val_mIoU_drop"] = baseline_miou - float(row["validation"]["mIoU"])
        row["speedup"] = baseline_latency / latency
        row["qualifies"] = (
            row["name"] != "ratio4"
            and math.isfinite(latency)
            and latency > 0
            and row["speedup"] >= args.min_speedup
            and row["val_mIoU_drop"] <= args.max_val_miou_drop
        )
        if row["qualifies"]:
            selected.append(row)

    selected.sort(
        key=lambda row: (
            row["benchmark"]["latency_mean_ms"],
            -row["validation"]["mIoU"],
        )
    )
    phase22_seed42 = next(
        float(run["validation"]["mIoU"])
        for run in phase22["runs"]
        if int(run["seed"]) == 42
    )
    gates = {
        "baseline_matches_phase22_seed42": abs(baseline_miou - phase22_seed42) <= 0.05,
        "finite_positive_latencies": all(
            math.isfinite(float(row["benchmark"]["latency_mean_ms"]))
            and float(row["benchmark"]["latency_mean_ms"]) > 0
            for row in rows
        ),
        "has_qualified_candidate": bool(selected),
    }
    result = {
        "phase": 25,
        "selection_split": "val",
        "test_evaluated": False,
        "screen_type": "path-strength structured DINO MLP channel pruning",
        "thresholds": {
            "min_speedup": args.min_speedup,
            "max_val_mIoU_drop": args.max_val_miou_drop,
        },
        "candidates": rows,
        "selected_candidate": None if not selected else selected[0]["name"],
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
