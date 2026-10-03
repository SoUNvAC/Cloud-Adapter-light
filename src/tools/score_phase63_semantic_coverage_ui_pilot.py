"""Score Phase63 semantic-coverage UI reviews after their hashes are frozen."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from phase56_protocol import sha256
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


def freeze_reviews(sealed_path: str | Path, review_a_path: str | Path, review_b_path: str | Path) -> dict:
    sealed = json.loads(Path(sealed_path).read_text(encoding="utf-8"))
    if sealed.get("phase") != "63-semantic-coverage-exact-pixel-ui-pilot":
        raise RuntimeError("Wrong sealed manifest for semantic-coverage UI pilot")
    ids = [row["tile_id"] for row in sealed["records"]]
    read_review(review_a_path, ids)
    read_review(review_b_path, ids)
    return {
        "status": "raw_reviews_frozen_before_nominal_unseal",
        "sealed_manifest_sha256": sha256(sealed_path),
        "reviewer_A_sha256": sha256(review_a_path),
        "reviewer_B_sha256": sha256(review_b_path),
        "units": len(ids),
    }


def validate_lock(lock: dict, sealed_path, review_a_path, review_b_path) -> None:
    expected = freeze_reviews(sealed_path, review_a_path, review_b_path)
    for key, value in expected.items():
        if lock.get(key) != value:
            raise RuntimeError(f"Frozen review lock mismatch for {key}")


def raw_nominal_counts(records: list[dict], review: dict[str, dict]) -> dict:
    result = {}
    for stratum in sorted({row["sampling_stratum"] for row in records}):
        ids = [row["tile_id"] for row in records if row["sampling_stratum"] == stratum]
        result[stratum] = {
            "units": len(ids),
            "identifiability": dict(Counter(review[item]["identifiability"] for item in ids)),
            "semantic_when_fine": dict(
                Counter(review[item]["semantic"] for item in ids if review[item]["semantic"])
            ),
        }
    return result


def score(sealed: dict, review_a: dict[str, dict], review_b: dict[str, dict]) -> dict:
    records = sealed["records"]
    ids = [row["tile_id"] for row in records]
    locatable_a = sum(review_a[item]["locatable"] for item in ids)
    locatable_b = sum(review_b[item]["locatable"] for item in ids)
    both_locatable = [
        item for item in ids if review_a[item]["locatable"] and review_b[item]["locatable"]
    ]
    ident_a = [review_a[item]["identifiability"] for item in both_locatable]
    ident_b = [review_b[item]["identifiability"] for item in both_locatable]
    ident_exact = sum(a == b for a, b in zip(ident_a, ident_b)) / len(both_locatable)
    interior_ids = {
        row["tile_id"] for row in records if row["sampling_kind"] == "interior"
    }
    both_fine_interior = [
        item for item in ids
        if item in interior_ids
        and review_a[item]["locatable"] and review_b[item]["locatable"]
        and review_a[item]["identifiability"] == "fine_classifiable"
        and review_b[item]["identifiability"] == "fine_classifiable"
    ]
    semantic_a = [review_a[item]["semantic"] for item in both_fine_interior]
    semantic_b = [review_b[item]["semantic"] for item in both_fine_interior]
    semantic_exact = (
        None if not both_fine_interior
        else sum(a == b for a, b in zip(semantic_a, semantic_b)) / len(both_fine_interior)
    )
    artifact = sealed.get("artifact_validation", {})
    artifact_ok = all(
        artifact.get(key) is True
        for key in (
            "coordinates_valid", "pages_valid", "review_templates_blank",
            "reviewer_packet_excludes_nominal_fields",
        )
    )
    gates = {
        "reviewer_A_locatable_at_least_19_of_20": locatable_a >= 19,
        "reviewer_B_locatable_at_least_19_of_20": locatable_b >= 19,
        "zero_coordinate_page_or_submission_format_errors": artifact_ok,
        "identifiability_exact_at_least_70pct": ident_exact >= 0.70,
        "both_fine_interior_at_least_10": len(both_fine_interior) >= 10,
        "semantic_exact_on_both_fine_interior_at_least_60pct": (
            semantic_exact is not None and semantic_exact >= 0.60
        ),
    }
    nominal_by_id = {row["tile_id"]: row["nominal_label_name"] for row in records}
    concordance = {}
    for reviewer_name, review in (("reviewer_A", review_a), ("reviewer_B", review_b)):
        interior = [item for item in ids if item in interior_ids and review[item]["semantic"]]
        concordance[reviewer_name] = {
            "name": "reviewer–nominal-label concordance",
            "fine_interior_units": len(interior),
            "matching_count": sum(review[item]["semantic"] == nominal_by_id[item] for item in interior),
            "pair_counts": dict(
                Counter(f"nominal={nominal_by_id[item]}|reviewer={review[item]['semantic']}" for item in interior)
            ),
            "warning": "Nominal labels are sampling strata, not reference truth; this is not accuracy.",
        }
    return {
        "phase": sealed["phase"],
        "units": len(ids),
        "locatability": {
            "reviewer_A_count": locatable_a,
            "reviewer_B_count": locatable_b,
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
        "semantic_when_both_fine_interior": {
            "units": len(both_fine_interior),
            "exact_agreement": semantic_exact,
            "cohen_kappa": None if not semantic_a else cohen_kappa(semantic_a, semantic_b),
            "gwet_ac1": None if not semantic_a else gwet_ac1(semantic_a, semantic_b),
            "reviewer_A_distribution": dict(Counter(semantic_a)),
            "reviewer_B_distribution": dict(Counter(semantic_b)),
        },
        "nominal_stratum_raw_counts": {
            "reviewer_A": raw_nominal_counts(records, review_a),
            "reviewer_B": raw_nominal_counts(records, review_b),
        },
        "reviewer_nominal_label_concordance": concordance,
        "elapsed_seconds_mean": {
            "reviewer_A": sum(review_a[item]["elapsed_seconds"] for item in ids) / len(ids),
            "reviewer_B": sum(review_b[item]["elapsed_seconds"] for item in ids) / len(ids),
        },
        "gates": gates,
        "ui_pilot_pass": all(gates.values()),
        "interpretation_limit": (
            "This 20-item pilot tests only UI and annotation workflow. It does not estimate per-class "
            "accuracy, label reproducibility, or model performance."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sealed", required=True)
    parser.add_argument("--review-a", required=True)
    parser.add_argument("--review-b", required=True)
    parser.add_argument("--lock", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    lock = json.loads(Path(args.lock).read_text(encoding="utf-8"))
    validate_lock(lock, args.sealed, args.review_a, args.review_b)
    sealed = json.loads(Path(args.sealed).read_text(encoding="utf-8"))
    ids = [row["tile_id"] for row in sealed["records"]]
    review_a = read_review(args.review_a, ids)
    review_b = read_review(args.review_b, ids)
    result = score(sealed, review_a, review_b)
    output = Path(args.output)
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite pilot result: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
