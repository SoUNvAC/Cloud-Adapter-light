import argparse
import json
import math
from pathlib import Path

from run_phase45_source_only import SOURCE_TO_TARGET, evaluate, load_rows


def source_rows(image_dir, mask_dir):
    image_dir, mask_dir = Path(image_dir), Path(mask_dir)
    rows = []
    for image_path in sorted(image_dir.glob("*.png")):
        mask_path = mask_dir / image_path.name
        if not mask_path.is_file():
            raise FileNotFoundError(mask_path)
        rows.append({"name": image_path.name, "image_path": str(image_path), "mask_path": str(mask_path)})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="work_dirs/phase45_protocol_audit/scene_disjoint_manifest.csv")
    parser.add_argument("--checkpoint", default="work_dirs/phase45_teacher_student/student/iter_10000.pth")
    parser.add_argument("--output-root", default="work_dirs/phase45_teacher_student/eval")
    parser.add_argument("--source-only-summary", default="work_dirs/phase45_source_only/summary.json")
    parser.add_argument("--phase22-summary", default="work_dirs/phase22_clean_v8/summary.json")
    args = parser.parse_args()
    config = Path("configs/protocol/phase45_teacher_student_v8_l8.py")
    checkpoint = Path(args.checkpoint)
    output_root = Path(args.output_root)
    target = evaluate(
        "target_val", config, checkpoint,
        load_rows(Path(args.manifest), "target_val"), output_root, SOURCE_TO_TARGET,
    )
    source = evaluate(
        "source_val", config, checkpoint,
        source_rows("data/cloudsen12_high_l1c/img_dir/val", "data/cloudsen12_high_l1c/ann_dir/val"),
        output_root, range(4),
    )
    source_only = json.loads(Path(args.source_only_summary).read_text(encoding="utf-8"))
    phase22 = json.loads(Path(args.phase22_summary).read_text(encoding="utf-8"))
    source_reference = next(row["validation"]["mIoU"] for row in phase22["runs"] if row["seed"] == 42)
    target_gain = target["metrics"]["mIoU"] - source_only["models"]["v8"]["metrics"]["mIoU"]
    source_forgetting = source_reference - source["metrics"]["mIoU"]
    gates = {
        "target_gain_at_least_1": target_gain >= 1.0,
        "source_forgetting_at_most_1": source_forgetting <= 1.0,
        "metrics_finite": all(math.isfinite(value) for value in (target_gain, source_forgetting)),
        "target_test_sealed": True,
        "cloudsen_internal_test_sealed": True,
    }
    summary = {
        "phase": "45B-teacher-student",
        "target_test_evaluated": False,
        "cloudsen_internal_test_evaluated": False,
        "checkpoint_selection": "fixed_iter_10000_without_target_labels",
        "target": target,
        "source": source,
        "source_only_target_mIoU": source_only["models"]["v8"]["metrics"]["mIoU"],
        "source_reference_mIoU": source_reference,
        "target_gain_mIoU": target_gain,
        "source_forgetting_mIoU": source_forgetting,
        "thresholds": {"min_target_gain_mIoU": 1.0, "max_source_forgetting_mIoU": 1.0},
        "gates": gates,
        "passed": all(gates.values()),
        "decision": "retain_teacher_student_baseline" if all(gates.values()) else "close_fixed_teacher_pseudolabel_direction",
    }
    Path("work_dirs/phase45_teacher_student/summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
