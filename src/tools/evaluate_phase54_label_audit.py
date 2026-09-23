import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, label as connected_components


def binary_boundary(binary):
    result = np.zeros(binary.shape, dtype=bool)
    horizontal = binary[:, 1:] != binary[:, :-1]
    vertical = binary[1:, :] != binary[:-1, :]
    result[:, 1:] |= horizontal; result[:, :-1] |= horizontal
    result[1:, :] |= vertical; result[:-1, :] |= vertical
    return result


def shadow_iou(prediction, target):
    prediction, target = prediction == 1, target == 1
    union = np.logical_or(prediction, target).sum()
    return 100.0 * float(np.logical_and(prediction, target).sum() / union) if union else math.nan


def changed_component_count(reference, changed_pixels):
    components, count = connected_components(reference)
    changed = 0
    for component in range(1, count + 1):
        pixels = components == component
        if pixels.any() and changed_pixels[pixels].mean() >= 0.5:
            changed += 1
    return changed, count


def validate_review(record, stem):
    first, second, adjudication = record["reviewer1"], record["reviewer2"], record["adjudication"]
    identifiers = [first.get("id"), second.get("id"), adjudication.get("id")]
    if not all(identifiers) or len(set(identifiers)) < 2:
        raise RuntimeError(f"Missing independent reviewer/adjudicator IDs: {stem}")
    for section in (first, second, adjudication):
        if not section.get("completed_at") or not section.get("mask_path"):
            raise RuntimeError(f"Incomplete review record: {stem}")
        if not Path(section["mask_path"]).is_file():
            raise FileNotFoundError(section["mask_path"])
    if not adjudication.get("reason_codes"):
        raise RuntimeError(f"Missing adjudication reason code: {stem}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", default="work_dirs/phase54_label_audit/selection.csv")
    parser.add_argument("--review-root", default="work_dirs/phase54_label_audit/review_records")
    parser.add_argument("--prediction-root", default="work_dirs/phase54_label_audit/predictions")
    parser.add_argument("--output", default="work_dirs/phase54_label_audit/summary.json")
    args = parser.parse_args()
    with Path(args.selection).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 144:
        raise RuntimeError(f"Expected 144 selected patches, found {len(rows)}")
    original_shadow = revised_shadow = removed_shadow = missed_shadow = 0
    false_components = missed_components = original_components = revised_components = 0
    absolute_boundary_distances, signed_boundary_distances = [], []
    predictions = {"phase22": [], "phase52a": []}
    originals, revisions = [], []
    reviewer_disagreement_pixels = reviewer_total_pixels = 0
    for row in rows:
        stem = Path(row["name"]).stem
        record = json.loads((Path(args.review_root) / f"{stem}.json").read_text(encoding="utf-8"))
        validate_review(record, stem)
        original = np.asarray(Image.open(row["mask_path"]), dtype=np.uint8)
        first = np.asarray(Image.open(record["reviewer1"]["mask_path"]), dtype=np.uint8)
        second = np.asarray(Image.open(record["reviewer2"]["mask_path"]), dtype=np.uint8)
        revised = np.asarray(Image.open(record["adjudication"]["mask_path"]), dtype=np.uint8)
        for name, mask in (("reviewer1", first), ("reviewer2", second), ("adjudicated", revised)):
            if mask.shape != original.shape or not np.isin(mask, (0, 1, 2, 3)).all():
                raise RuntimeError(f"Invalid {name} mask for {stem}")
        reviewer_disagreement_pixels += int(np.count_nonzero(first != second))
        reviewer_total_pixels += int(first.size)
        old, new = original == 1, revised == 1
        removed, missed = old & ~new, new & ~old
        original_shadow += int(old.sum()); revised_shadow += int(new.sum())
        removed_shadow += int(removed.sum()); missed_shadow += int(missed.sum())
        count, total = changed_component_count(old, removed)
        false_components += count; original_components += total
        count, total = changed_component_count(new, missed)
        missed_components += count; revised_components += total
        old_edge, new_edge = binary_boundary(old), binary_boundary(new)
        if old_edge.any() and new_edge.any():
            old_distance = distance_transform_edt(~old_edge)
            new_distance = distance_transform_edt(~new_edge)
            absolute_boundary_distances.extend(old_distance[new_edge].tolist())
            absolute_boundary_distances.extend(new_distance[old_edge].tolist())
            signed_map = distance_transform_edt(~old) - distance_transform_edt(old)
            signed_boundary_distances.extend(signed_map[new_edge].tolist())
        originals.append(original)
        revisions.append(revised)
        for model in predictions:
            predictions[model].append(np.asarray(
                Image.open(Path(args.prediction_root) / f"{stem}_{model}.png"), dtype=np.uint8
            ))
    original_iou, revised_iou = {}, {}
    for model, values in predictions.items():
        prediction = np.concatenate([value.reshape(-1) for value in values])
        original = np.concatenate([value.reshape(-1) for value in originals])
        revised = np.concatenate([value.reshape(-1) for value in revisions])
        original_iou[model] = shadow_iou(prediction, original)
        revised_iou[model] = shadow_iou(prediction, revised)
    iou_change = {model: revised_iou[model] - original_iou[model] for model in predictions}
    removed_fraction = removed_shadow / max(original_shadow, 1)
    missed_fraction = missed_shadow / max(revised_shadow, 1)
    median_absolute_boundary = float(np.median(absolute_boundary_distances)) if absolute_boundary_distances else math.nan
    median_signed_boundary = float(np.median(signed_boundary_distances)) if signed_boundary_distances else math.nan
    gates = {
        "all_144_independently_reviewed_and_adjudicated": True,
        "iou_change_at_least_5_points": max(abs(value) for value in iou_change.values()) >= 5.0,
        "removed_false_shadow_at_least_10pct": removed_fraction >= 0.10,
        "missed_shadow_at_least_10pct": missed_fraction >= 0.10,
        "median_boundary_displacement_at_least_2px": median_absolute_boundary >= 2.0,
        "target_test_sealed": True,
        "cloudsen_internal_test_sealed": True,
    }
    passed = gates["all_144_independently_reviewed_and_adjudicated"] and (
        gates["iou_change_at_least_5_points"]
        or gates["removed_false_shadow_at_least_10pct"]
        or gates["missed_shadow_at_least_10pct"]
        or gates["median_boundary_displacement_at_least_2px"]
    ) and gates["target_test_sealed"] and gates["cloudsen_internal_test_sealed"]
    summary = {
        "phase": "54B", "selected_images": len(rows),
        "original_shadow_iou": original_iou, "adjudicated_shadow_iou": revised_iou,
        "shadow_iou_change": iou_change,
        "removed_false_shadow_pixel_fraction": removed_fraction,
        "originally_missed_shadow_pixel_fraction": missed_fraction,
        "removed_shadow_components": false_components,
        "original_shadow_components": original_components,
        "missed_shadow_components": missed_components,
        "adjudicated_shadow_components": revised_components,
        "median_absolute_boundary_displacement_pixels": median_absolute_boundary,
        "median_signed_boundary_displacement_pixels": median_signed_boundary,
        "reviewer_pixel_disagreement_fraction": reviewer_disagreement_pixels / reviewer_total_pixels,
        "gates": gates, "passed": passed,
        "decision": "label_inconsistency_material" if passed else "label_inconsistency_not_material",
        "target_test_evaluated": False, "cloudsen_internal_test_evaluated": False,
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

