"""Score the exact-pixel Phase 63 annotation UI pilot."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from score_phase61d2_calibrated_review import cohen_kappa, gwet_ac1


IDENTIFIABILITY = {
    "fine_classifiable", "mixed_boundary", "insufficient_evidence", "unobservable_nodata"
}
SEMANTIC = {
    "clear", "thin_cloud", "thick_cloud", "haze_cirrus",
    "cloud_shadow", "terrain_water_shadow",
}


def read_review(path: str | Path, expected_ids: list[str]) -> dict[str, dict]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "tile_id", "target_locatable", "identifiability",
        "semantic_label", "elapsed_seconds", "notes",
    }
    if not rows or not required.issubset(rows[0]):
        raise RuntimeError(f"Invalid pilot columns in {path}")
    ids = [row["tile_id"].strip() for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(expected_ids):
        raise RuntimeError(f"Pilot IDs do not match sealed manifest in {path}")
    result = {}
    for row in rows:
        tile_id = row["tile_id"].strip()
        locatable = row["target_locatable"].strip().lower()
        identifiability = row["identifiability"].strip()
        semantic = row["semantic_label"].strip()
        if locatable not in {"yes", "no"}:
            raise RuntimeError(f"Invalid target_locatable for {tile_id}")
        if locatable == "no":
            if identifiability or semantic:
                raise RuntimeError(f"Unlocatable item must not contain judgments: {tile_id}")
        else:
            if identifiability not in IDENTIFIABILITY:
                raise RuntimeError(f"Invalid identifiability for {tile_id}")
            if identifiability == "fine_classifiable":
                if semantic not in SEMANTIC:
                    raise RuntimeError(f"Fine-classifiable item needs one semantic label: {tile_id}")
            elif semantic:
                raise RuntimeError(f"Non-fine item must leave semantic_label blank: {tile_id}")
        try:
            elapsed = float(row["elapsed_seconds"])
        except ValueError as error:
            raise RuntimeError(f"Invalid elapsed_seconds for {tile_id}") from error
        if not 0 < elapsed <= 3600:
            raise RuntimeError(f"elapsed_seconds outside (0, 3600] for {tile_id}")
        result[tile_id] = {
            "locatable": locatable == "yes",
            "identifiability": identifiability,
            "semantic": semantic,
            "elapsed_seconds": elapsed,
        }
    return result


def score(review_a: dict[str, dict], review_b: dict[str, dict], ordered_ids: list[str]) -> dict:
    count = len(ordered_ids)
    locatable_a = sum(review_a[item]["locatable"] for item in ordered_ids)
    locatable_b = sum(review_b[item]["locatable"] for item in ordered_ids)
    both_locatable = [
        item for item in ordered_ids if review_a[item]["locatable"] and review_b[item]["locatable"]
    ]
    ident_a = [review_a[item]["identifiability"] for item in both_locatable]
    ident_b = [review_b[item]["identifiability"] for item in both_locatable]
    ident_exact = sum(a == b for a, b in zip(ident_a, ident_b)) / len(both_locatable)
    both_fine = [
        item for item in both_locatable
        if review_a[item]["identifiability"] == "fine_classifiable"
        and review_b[item]["identifiability"] == "fine_classifiable"
    ]
    semantic_a = [review_a[item]["semantic"] for item in both_fine]
    semantic_b = [review_b[item]["semantic"] for item in both_fine]
    semantic_exact = (
        None if not both_fine
        else sum(a == b for a, b in zip(semantic_a, semantic_b)) / len(both_fine)
    )
    gates = {
        "reviewer_a_locatable_at_least_95pct": locatable_a / count >= 0.95,
        "reviewer_b_locatable_at_least_95pct": locatable_b / count >= 0.95,
        "identifiability_exact_at_least_70pct": ident_exact >= 0.70,
        "identifiability_ac1_at_least_0_40": gwet_ac1(ident_a, ident_b) >= 0.40,
        "both_fine_units_at_least_6": len(both_fine) >= 6,
        "semantic_exact_at_least_50pct": semantic_exact is not None and semantic_exact >= 0.50,
    }
    return {
        "units": count,
        "locatability": {
            "reviewer_A": {"count": locatable_a, "rate": locatable_a / count},
            "reviewer_B": {"count": locatable_b, "rate": locatable_b / count},
            "both_count": len(both_locatable),
        },
        "identifiability": {
            "units": len(both_locatable),
            "exact_agreement": ident_exact,
            "cohen_kappa": cohen_kappa(ident_a, ident_b),
            "gwet_ac1": gwet_ac1(ident_a, ident_b),
            "reviewer_A_distribution": dict(Counter(ident_a)),
            "reviewer_B_distribution": dict(Counter(ident_b)),
        },
        "semantic_when_both_fine": {
            "units": len(both_fine),
            "exact_agreement": semantic_exact,
            "cohen_kappa": None if not both_fine else cohen_kappa(semantic_a, semantic_b),
            "gwet_ac1": None if not both_fine else gwet_ac1(semantic_a, semantic_b),
            "reviewer_A_distribution": dict(Counter(semantic_a)),
            "reviewer_B_distribution": dict(Counter(semantic_b)),
        },
        "elapsed_seconds": {
            "reviewer_A_mean": sum(review_a[item]["elapsed_seconds"] for item in ordered_ids) / count,
            "reviewer_B_mean": sum(review_b[item]["elapsed_seconds"] for item in ordered_ids) / count,
        },
        "gates": gates,
        "ui_pilot_pass": all(gates.values()),
        "causal_limitation": (
            "New items remove memory leakage but do not isolate UI causality from item difficulty."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sealed", required=True)
    parser.add_argument("--review-a", required=True)
    parser.add_argument("--review-b", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    sealed = json.loads(Path(args.sealed).read_text(encoding="utf-8"))
    if sealed.get("phase") != "63-exact-pixel-annotation-ui-pilot":
        raise RuntimeError("Wrong sealed manifest for UI pilot")
    ids = [row["tile_id"] for row in sealed["records"]]
    review_a = read_review(args.review_a, ids)
    review_b = read_review(args.review_b, ids)
    result = score(review_a, review_b, ids)
    output = Path(args.output)
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite pilot result: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
