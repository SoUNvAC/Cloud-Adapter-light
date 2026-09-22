import argparse
import csv
import json
import math
from pathlib import Path

from run_phase45_source_only import evaluate


def manifest_rows(path, split):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["new_split"] == split]
    return sorted(rows, key=lambda row: row["name"])


def one_checkpoint(path):
    matches = sorted(path.glob("best_mIoU_iter_*.pth"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one best checkpoint in {path}, found {len(matches)}")
    return matches[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="work_dirs/phase45_protocol_audit/scene_disjoint_manifest.csv")
    parser.add_argument("--source-summary", default="work_dirs/phase45_source_only/summary.json")
    parser.add_argument("--output-root", default="work_dirs/phase45_oracle")
    parser.add_argument("--expected-images", type=int, default=1905)
    args = parser.parse_args()
    rows = manifest_rows(Path(args.manifest), "target_val")
    if len(rows) != args.expected_images:
        raise RuntimeError(f"Expected {args.expected_images} target-val images, found {len(rows)}")
    source = json.loads(Path(args.source_summary).read_text(encoding="utf-8"))
    if source.get("passed") is not True or source.get("target_test_evaluated") is not False:
        raise RuntimeError("Valid sealed Phase 45 source-only evidence is required")
    output_root = Path(args.output_root)
    specifications = (
        ("v8", Path("configs/protocol/phase45_oracle_v8_l8.py"),
         one_checkpoint(Path("work_dirs/phase45_oracle_v8/seed42"))),
        ("phase37_compact", Path("configs/protocol/phase45_oracle_compact_l8.py"),
         one_checkpoint(Path("work_dirs/phase45_oracle_compact/seed42"))),
    )
    oracle = [evaluate(name, config, checkpoint, rows, output_root, range(4))
              for name, config, checkpoint in specifications]
    gaps = {}
    for row in oracle:
        baseline = source["models"][row["name"]]
        gaps[row["name"]] = {
            "mIoU": row["metrics"]["mIoU"] - baseline["metrics"]["mIoU"],
            "weak_mIoU": row["metrics"]["weak_mIoU"] - baseline["metrics"]["weak_mIoU"],
            "boundary_macro_f1": row["boundary"]["macro_f1"] - baseline["boundary"]["macro_f1"],
        }
    primary = gaps["v8"]
    if primary["mIoU"] >= 8.0:
        decision = "proceed_to_factorized_method"
    elif primary["mIoU"] >= 5.0 and primary["weak_mIoU"] >= 10.0:
        decision = "proceed_to_factorized_method_gray_zone"
    else:
        decision = "stop_cross_domain_method_direction"
    finite = [value for values in gaps.values() for value in values.values()]
    summary = {
        "phase": "45B-oracle",
        "primary_model": "v8",
        "selection_split": "scene_disjoint_target_val",
        "target_test_evaluated": False,
        "cloudsen_internal_test_evaluated": False,
        "models": {row["name"]: row for row in oracle},
        "source_to_oracle_gaps": gaps,
        "thresholds": {"stop_below_mIoU_gap": 5.0, "proceed_mIoU_gap": 8.0,
                       "gray_zone_min_weak_gap": 10.0},
        "decision": decision,
        "gates": {
            "expected_target_val_images": all(row["evaluated_images"] == args.expected_images for row in oracle),
            "metrics_finite": all(math.isfinite(value) for value in finite),
            "target_test_sealed": True,
            "cloudsen_internal_test_sealed": True,
        },
    }
    summary["passed"] = all(summary["gates"].values())
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
