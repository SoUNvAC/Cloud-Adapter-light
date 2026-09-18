import argparse
import json
from pathlib import Path
import re


CANDIDATES = ("baseline12", "blocks10", "blocks8", "blocks6", "blocks4")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Gate the Phase 24 zero-shot static block-pruning screen."
    )
    parser.add_argument("--phase23-summary", required=True)
    parser.add_argument("--root", default="work_dirs/phase24_block_screen")
    parser.add_argument("--min-speedup", type=float, default=1.15)
    parser.add_argument("--max-val-miou-drop", type=float, default=8.0)
    parser.add_argument(
        "--output", default="work_dirs/phase24_block_screen/summary.json"
    )
    return parser.parse_args()


def parse_last_miou(path):
    pattern = re.compile(r"\bmIoU:\s*(-?(?:\d+(?:\.\d*)?|\.\d+))")
    result = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            result = float(match.group(1))
    if result is None:
        raise RuntimeError(f"No mIoU found in {path}")
    return result


def main():
    args = parse_args()
    phase23_path = Path(args.phase23_summary)
    phase23 = json.loads(phase23_path.read_text(encoding="utf-8"))
    if not phase23.get("passed") or phase23.get("test_evaluated") is not False:
        raise RuntimeError(
            "Phase 23 must pass on validation with test sealed before screening"
        )

    root = Path(args.root)
    rows = []
    for name in CANDIDATES:
        candidate_dir = root / name
        benchmark = json.loads(
            (candidate_dir / "benchmark.json").read_text(encoding="utf-8")
        )
        rows.append(
            {
                "name": name,
                "val_mIoU": parse_last_miou(candidate_dir / "val_eval.log"),
                "benchmark": benchmark,
            }
        )

    baseline = rows[0]
    baseline_miou = baseline["val_mIoU"]
    baseline_latency = baseline["benchmark"]["latency_mean_ms"]
    selected = []
    for row in rows:
        row["val_mIoU_drop"] = baseline_miou - row["val_mIoU"]
        row["speedup"] = baseline_latency / row["benchmark"]["latency_mean_ms"]
        row["qualifies"] = (
            row["name"] != "baseline12"
            and row["speedup"] >= args.min_speedup
            and row["val_mIoU_drop"] <= args.max_val_miou_drop
        )
        if row["qualifies"]:
            selected.append(row["name"])

    gates = {
        "baseline_matches_phase23_seed42": abs(
            baseline_miou
            - next(
                run["validation"]["mIoU"]
                for run in phase23["runs"]
                if int(run["seed"]) == 42
            )
        )
        <= 0.05,
        "finite_positive_latencies": all(
            row["benchmark"]["latency_mean_ms"] > 0 for row in rows
        ),
        "has_qualified_candidate": bool(selected),
    }
    result = {
        "phase": 24,
        "selection_split": "val",
        "test_evaluated": False,
        "screen_type": "zero-shot static block pruning, seed 42",
        "thresholds": {
            "min_speedup": args.min_speedup,
            "max_val_mIoU_drop": args.max_val_miou_drop,
        },
        "candidates": rows,
        "selected_candidates": selected,
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
