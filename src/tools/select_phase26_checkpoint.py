import argparse
import json
from pathlib import Path
import statistics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Freeze the representative Phase 25 checkpoint for deployment."
    )
    parser.add_argument("--phase25-root", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def find_passed_summaries(root: Path):
    summaries = []
    for path in sorted(root.glob("*/summary.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("passed") and result.get("test_evaluated") is False:
            summaries.append((path, result))
    return summaries


def select_median_run(runs):
    if len(runs) != 3:
        raise RuntimeError(f"Expected exactly three Phase 25 runs, found {len(runs)}")
    ordered = sorted(
        runs,
        key=lambda run: (float(run["validation"]["mIoU"]), int(run["seed"])),
    )
    return ordered[1]


def main():
    args = parse_args()
    summaries = find_passed_summaries(Path(args.phase25_root))
    if len(summaries) != 1:
        raise RuntimeError(
            "Expected exactly one passed Phase 25 candidate before Phase 26; "
            f"found {[str(path) for path, _ in summaries]}"
        )
    summary_path, summary = summaries[0]
    selected = select_median_run(summary["runs"])
    mious = [float(run["validation"]["mIoU"]) for run in summary["runs"]]
    result = {
        "phase": 26,
        "selection_split": "val",
        "test_evaluated": False,
        "selection_rule": "median validation mIoU; seed breaks exact ties",
        "phase25_summary": summary_path.as_posix(),
        "candidate": summary["candidate"],
        "active_block_indices": summary["active_block_indices"],
        "seed": int(selected["seed"]),
        "checkpoint": selected["checkpoint"],
        "reference_validation_mIoU": float(selected["validation"]["mIoU"]),
        "phase25_mean_validation_mIoU": statistics.fmean(mious),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
