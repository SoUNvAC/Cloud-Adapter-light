from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


SRC_ROOT = Path(__file__).resolve().parents[1]
PHASE61D2_ROOT = SRC_ROOT / "work_dirs" / "phase61d2_calibrated_review"
DEFAULT_SEALED_MANIFEST = PHASE61D2_ROOT / "sealed_manifest.json"
DEFAULT_CALIBRATION_LOCK = PHASE61D2_ROOT / "calibration_lock.json"
DEFAULT_REVIEW_METRICS = PHASE61D2_ROOT / "review_metrics.json"
DEFAULT_OUTPUT = (
    SRC_ROOT
    / "work_dirs"
    / "phase63_selective_multigranularity"
    / "phase63_split_lock.json"
)
DEFAULT_AUDIT_OUTPUT = DEFAULT_OUTPUT.with_name("phase63_split_audit.json")

# This is the real Phase 61D2 sealed manifest synchronized from the remote run.
# Phase 63A must never fall back to the older phase60D manifest.
EXPECTED_SEALED_MANIFEST_SHA256 = (
    "7c9468778ccb44127610918ae537a64d7e1e0af95469c403a0c6c1453d587c77"
)

EXPECTED_MAIN_UNITS = 210
EXPECTED_DEVELOPMENT_UNITS = 160
EXPECTED_CONFIRMATION_UNITS = 50
EXPECTED_CALIBRATION_UNITS = 40
EXPECTED_MISSING_RATIONALE_UNITS = 79
MAIN_COHORTS = ("independent", "confirmation")


class SplitAuditError(RuntimeError):
    """Base class for fail-closed Phase 63A audit errors."""


class InputValidationError(SplitAuditError):
    """An input artifact is absent or violates the frozen Phase 61D2 contract."""


class SceneLeakageError(SplitAuditError):
    """At least one scene occurs in both development and confirmation cohorts."""

    def __init__(self, report: Mapping[str, Any]):
        self.report = dict(report)
        super().__init__(
            f"scene-level leakage across {self.report['conflict_scene_count']} scene(s)"
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise InputValidationError(f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InputValidationError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise InputValidationError(f"{label} must be a JSON object: {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InputValidationError(message)


def _require_int(value: Any, expected: int, field: str) -> None:
    _require(
        type(value) is int and value == expected,
        f"{field} must equal {expected}, got {value!r}",
    )


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(SRC_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _validate_sealed_manifest(
    sealed: Mapping[str, Any], actual_sha256: str, expected_sha256: str
) -> list[dict[str, Any]]:
    _require(
        actual_sha256 == expected_sha256,
        "sealed_manifest SHA256 mismatch: "
        f"expected {expected_sha256}, got {actual_sha256}; refusing fallback or mutation",
    )
    _require(sealed.get("phase") == "61D2", "sealed_manifest phase must be '61D2'")
    _require_int(sealed.get("main_units"), EXPECTED_MAIN_UNITS, "sealed_manifest.main_units")
    _require_int(
        sealed.get("independent_units"),
        EXPECTED_DEVELOPMENT_UNITS,
        "sealed_manifest.independent_units",
    )
    _require_int(
        sealed.get("confirmation_units"),
        EXPECTED_CONFIRMATION_UNITS,
        "sealed_manifest.confirmation_units",
    )
    _require_int(
        sealed.get("calibration_units_excluded_from_statistics"),
        EXPECTED_CALIBRATION_UNITS,
        "sealed_manifest.calibration_units_excluded_from_statistics",
    )

    records = sealed.get("records")
    _require(isinstance(records, list), "sealed_manifest.records must be a list")
    _require(
        len(records) == EXPECTED_MAIN_UNITS + EXPECTED_CALIBRATION_UNITS,
        "sealed_manifest.records must contain exactly 40 calibration + 210 main units, "
        f"got {len(records)}",
    )
    cohorts = Counter()
    all_tile_ids: list[str] = []
    main_records: list[dict[str, Any]] = []
    for index, value in enumerate(records):
        _require(isinstance(value, dict), f"sealed_manifest.records[{index}] must be an object")
        row = dict(value)
        for field in ("tile_id", "scene", "biome", "cohort"):
            _require(
                isinstance(row.get(field), str) and bool(row[field].strip()),
                f"sealed_manifest.records[{index}].{field} must be a non-empty string",
            )
        tile_id = row["tile_id"].strip()
        cohort = row["cohort"].strip()
        all_tile_ids.append(tile_id)
        cohorts[cohort] += 1
        if cohort in MAIN_COHORTS:
            main_records.append(row)
        else:
            _require(
                cohort == "calibration",
                f"unexpected sealed cohort {cohort!r} for tile {tile_id}",
            )

    duplicates = sorted(tile_id for tile_id, count in Counter(all_tile_ids).items() if count > 1)
    _require(not duplicates, f"duplicate sealed tile IDs: {duplicates}")
    _require(
        cohorts == Counter(
            {
                "calibration": EXPECTED_CALIBRATION_UNITS,
                "independent": EXPECTED_DEVELOPMENT_UNITS,
                "confirmation": EXPECTED_CONFIRMATION_UNITS,
            }
        ),
        f"sealed cohort counts mismatch: {dict(sorted(cohorts.items()))}",
    )
    return main_records


def _validate_calibration_lock(
    calibration: Mapping[str, Any], sealed: Mapping[str, Any]
) -> None:
    _require(
        calibration.get("phase") == "61D2-calibration-lock",
        "calibration_lock phase must be '61D2-calibration-lock'",
    )
    _require(
        calibration.get("status") == "locked_before_independent_review",
        "calibration_lock status must be 'locked_before_independent_review'",
    )
    _require_int(
        calibration.get("calibration_units"),
        EXPECTED_CALIBRATION_UNITS,
        "calibration_lock.calibration_units",
    )
    _require(
        calibration.get("calibration_excluded_from_final_statistics") is True,
        "calibration units must remain excluded from final statistics",
    )
    _require(
        calibration.get("manual_sha256") == sealed.get("manual_sha256"),
        "calibration_lock.manual_sha256 does not match sealed_manifest.manual_sha256",
    )


def _validate_review_metrics(
    metrics: Mapping[str, Any],
    sealed_sha256: str,
    calibration_sha256: str,
    main_tile_ids: set[str],
) -> list[str]:
    _require(metrics.get("phase") == "61D2", "review_metrics phase must be '61D2'")
    _require(
        metrics.get("status") == "complete_adjudicated",
        "review_metrics status must be 'complete_adjudicated'",
    )
    _require_int(metrics.get("main_units"), EXPECTED_MAIN_UNITS, "review_metrics.main_units")
    _require_int(
        metrics.get("calibration_units_excluded_from_all_reported_statistics"),
        EXPECTED_CALIBRATION_UNITS,
        "review_metrics.calibration_units_excluded_from_all_reported_statistics",
    )
    sealed_reference = metrics.get("sealed_manifest")
    _require(isinstance(sealed_reference, dict), "review_metrics.sealed_manifest must be an object")
    _require(
        sealed_reference.get("sha256") == sealed_sha256,
        "review_metrics sealed_manifest SHA256 does not match the supplied sealed manifest",
    )
    calibration_reference = metrics.get("calibration")
    _require(isinstance(calibration_reference, dict), "review_metrics.calibration must be an object")
    _require(
        calibration_reference.get("sha256") == calibration_sha256,
        "review_metrics calibration SHA256 does not match the supplied calibration lock",
    )

    adjudication = metrics.get("adjudication")
    _require(isinstance(adjudication, dict), "review_metrics.adjudication must be an object")
    _require_int(
        adjudication.get("units"),
        EXPECTED_MISSING_RATIONALE_UNITS,
        "review_metrics.adjudication.units",
    )
    _require(
        adjudication.get("rationale_required") is False,
        "the explicit Phase61D2 missing-rationale downgrade must remain recorded",
    )
    _require_int(
        adjudication.get("rationale_complete_units"),
        0,
        "review_metrics.adjudication.rationale_complete_units",
    )
    _require_int(
        adjudication.get("missing_rationale_units"),
        EXPECTED_MISSING_RATIONALE_UNITS,
        "review_metrics.adjudication.missing_rationale_units",
    )
    missing_ids = adjudication.get("missing_rationale_tile_ids")
    _require(
        isinstance(missing_ids, list)
        and len(missing_ids) == EXPECTED_MISSING_RATIONALE_UNITS
        and all(isinstance(tile_id, str) and tile_id for tile_id in missing_ids),
        "review_metrics must list exactly 79 non-empty missing-rationale tile IDs",
    )
    _require(
        len(set(missing_ids)) == EXPECTED_MISSING_RATIONALE_UNITS,
        "review_metrics missing-rationale tile IDs must be unique",
    )
    unknown_missing = sorted(set(missing_ids) - main_tile_ids)
    _require(
        not unknown_missing,
        f"missing-rationale IDs absent from the frozen 210 main units: {unknown_missing}",
    )

    disagreements = metrics.get("disagreements")
    _require(isinstance(disagreements, dict), "review_metrics.disagreements must be an object")
    _require_int(
        disagreements.get("units"),
        EXPECTED_MISSING_RATIONALE_UNITS,
        "review_metrics.disagreements.units",
    )
    provenance = metrics.get("final_label_provenance")
    _require(isinstance(provenance, dict), "review_metrics.final_label_provenance must be an object")
    _require_int(
        provenance.get("exact_A_B_agreement_units"),
        EXPECTED_MAIN_UNITS - EXPECTED_MISSING_RATIONALE_UNITS,
        "review_metrics.final_label_provenance.exact_A_B_agreement_units",
    )
    _require_int(
        provenance.get("third_expert_disagreement_only_units"),
        EXPECTED_MISSING_RATIONALE_UNITS,
        "review_metrics.final_label_provenance.third_expert_disagreement_only_units",
    )
    _require(
        provenance.get("raw_A_B_reviews_preserved") is True,
        "review_metrics must state that raw A/B reviews were preserved",
    )
    return sorted(missing_ids)


def scene_leakage_report(main_records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_scene: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: {"independent": [], "confirmation": []}
    )
    for row in main_records:
        by_scene[str(row["scene"])][str(row["cohort"])].append(str(row["tile_id"]))

    conflicts = []
    for scene in sorted(by_scene):
        cohorts = by_scene[scene]
        if cohorts["independent"] and cohorts["confirmation"]:
            development_ids = sorted(cohorts["independent"])
            confirmation_ids = sorted(cohorts["confirmation"])
            conflicts.append(
                {
                    "scene": scene,
                    "development_count": len(development_ids),
                    "confirmation_count": len(confirmation_ids),
                    "development_tile_ids": development_ids,
                    "confirmation_tile_ids": confirmation_ids,
                }
            )
    return {
        "passed": not conflicts,
        "conflict_scene_count": len(conflicts),
        "conflict_scenes": conflicts,
    }


def _build_lock(
    main_records: Sequence[Mapping[str, Any]],
    missing_rationale_ids: Sequence[str],
    input_artifacts: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    ordered = sorted(main_records, key=lambda row: str(row["tile_id"]))
    frozen_records = []
    for row in ordered:
        frozen = {
            "tile_id": str(row["tile_id"]),
            "scene": str(row["scene"]),
            "biome": str(row["biome"]),
            "cohort": str(row["cohort"]),
            "phase63_role": (
                "risk_score_development_only"
                if row["cohort"] == "independent"
                else "locked_final_validation_only"
            ),
        }
        frozen["record_sha256"] = canonical_sha256(frozen)
        frozen_records.append(frozen)

    by_cohort = {}
    for cohort in MAIN_COHORTS:
        rows = [row for row in frozen_records if row["cohort"] == cohort]
        tile_ids = sorted(row["tile_id"] for row in rows)
        scenes = sorted({row["scene"] for row in rows})
        by_cohort[cohort] = {
            "units": len(rows),
            "scenes": len(scenes),
            "tile_ids_sha256": canonical_sha256(tile_ids),
            "scene_ids_sha256": canonical_sha256(scenes),
            "biome_counts": dict(sorted(Counter(row["biome"] for row in rows).items())),
        }

    all_tile_ids = [row["tile_id"] for row in frozen_records]
    all_scenes = sorted({row["scene"] for row in frozen_records})
    snow_records = [row for row in frozen_records if row["biome"] == "snow_ice"]
    return {
        "phase": "63A-scene-freeze",
        "status": "locked_scene_disjoint",
        "source_phase": "61D2",
        "input_artifacts": dict(input_artifacts),
        "frozen_main_units": len(frozen_records),
        "development_units": sum(row["cohort"] == "independent" for row in frozen_records),
        "confirmation_units": sum(row["cohort"] == "confirmation" for row in frozen_records),
        "calibration_units_excluded": EXPECTED_CALIBRATION_UNITS,
        "scene_disjoint_audit": {
            "passed": True,
            "development_scenes": by_cohort["independent"]["scenes"],
            "confirmation_scenes": by_cohort["confirmation"]["scenes"],
            "overlap_scene_count": 0,
            "overlap_scenes": [],
        },
        "usage_policy": {
            "independent": "risk_score_development_only",
            "confirmation": "locked_final_validation_only",
            "confirmation_must_not_influence_feature_calibration_or_thresholds": True,
            "scene_grouping_required": True,
            "snow_ice_must_be_reported_and_must_not_be_removed": True,
        },
        "review_integrity": {
            "adjudicated_main_units": EXPECTED_MAIN_UNITS,
            "missing_rationale_units": len(missing_rationale_ids),
            "missing_rationale_tile_ids_sha256": canonical_sha256(
                sorted(missing_rationale_ids)
            ),
            "missing_rationales_preserved_without_imputation": True,
        },
        "counts": {
            "all_scenes": len(all_scenes),
            "all_biomes": dict(
                sorted(Counter(row["biome"] for row in frozen_records).items())
            ),
            "snow_ice_units": len(snow_records),
            "snow_ice_by_cohort": dict(
                sorted(Counter(row["cohort"] for row in snow_records).items())
            ),
            "by_cohort": by_cohort,
        },
        "hashes": {
            "all_tile_ids_sha256": canonical_sha256(all_tile_ids),
            "all_scene_ids_sha256": canonical_sha256(all_scenes),
            "frozen_records_sha256": canonical_sha256(frozen_records),
        },
        "records": frozen_records,
    }


def audit_and_build_lock(
    sealed_path: Path,
    calibration_path: Path,
    metrics_path: Path,
    *,
    expected_sealed_sha256: str = EXPECTED_SEALED_MANIFEST_SHA256,
) -> dict[str, Any]:
    sealed_path = Path(sealed_path)
    calibration_path = Path(calibration_path)
    metrics_path = Path(metrics_path)
    sealed = _load_json(sealed_path, "Phase61D2 sealed_manifest")
    calibration = _load_json(calibration_path, "Phase61D2 calibration_lock")
    metrics = _load_json(metrics_path, "Phase61D2 review_metrics")
    sealed_sha256 = sha256_file(sealed_path)
    calibration_sha256 = sha256_file(calibration_path)
    metrics_sha256 = sha256_file(metrics_path)

    main_records = _validate_sealed_manifest(
        sealed, sealed_sha256, expected_sealed_sha256
    )
    _validate_calibration_lock(calibration, sealed)
    main_tile_ids = {str(row["tile_id"]) for row in main_records}
    missing_rationale_ids = _validate_review_metrics(
        metrics, sealed_sha256, calibration_sha256, main_tile_ids
    )

    snow_ice_units = sum(row["biome"] == "snow_ice" for row in main_records)
    _require(snow_ice_units > 0, "snow_ice is absent; Phase63A forbids deleting it")

    leakage = scene_leakage_report(main_records)
    if not leakage["passed"]:
        raise SceneLeakageError(
            {
                "phase": "63A-scene-freeze",
                "status": "failed_closed_scene_leakage",
                "lock_written": False,
                "frozen_main_units_audited": len(main_records),
                "development_units": sum(
                    row["cohort"] == "independent" for row in main_records
                ),
                "confirmation_units": sum(
                    row["cohort"] == "confirmation" for row in main_records
                ),
                "snow_ice_units_preserved_in_audit": snow_ice_units,
                "sealed_manifest_sha256": sealed_sha256,
                **leakage,
            }
        )

    inputs = {
        "sealed_manifest": {
            "path": _display_path(sealed_path),
            "sha256": sealed_sha256,
        },
        "calibration_lock": {
            "path": _display_path(calibration_path),
            "sha256": calibration_sha256,
        },
        "review_metrics": {
            "path": _display_path(metrics_path),
            "sha256": metrics_sha256,
        },
    }
    return _build_lock(main_records, missing_rationale_ids, inputs)


def write_lock_atomic(path: Path, lock: Mapping[str, Any]) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(lock, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary_name = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
    return hashlib.sha256(payload).hexdigest()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail-closed Phase63A scene-level split freezer for the true Phase61D2 artifacts."
    )
    parser.add_argument("--sealed", type=Path, default=DEFAULT_SEALED_MANIFEST)
    parser.add_argument("--calibration-lock", type=Path, default=DEFAULT_CALIBRATION_LOCK)
    parser.add_argument("--review-metrics", type=Path, default=DEFAULT_REVIEW_METRICS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=DEFAULT_AUDIT_OUTPUT,
        help="Always preserve a leakage report here; this is not a valid split lock.",
    )
    parser.add_argument(
        "--expected-sealed-sha256",
        default=EXPECTED_SEALED_MANIFEST_SHA256,
        help="Pinned true Phase61D2 sealed manifest SHA256; mismatch fails closed.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        lock = audit_and_build_lock(
            args.sealed,
            args.calibration_lock,
            args.review_metrics,
            expected_sealed_sha256=args.expected_sealed_sha256,
        )
    except SceneLeakageError as exc:
        report = {
            **exc.report,
            "output_path": _display_path(args.output),
            "audit_output_path": _display_path(args.audit_output),
        }
        report_sha256 = write_lock_atomic(args.audit_output, report)
        report["audit_output_sha256"] = report_sha256
        print(json.dumps(report, indent=2, ensure_ascii=False), file=sys.stderr)
        return 2
    except SplitAuditError as exc:
        print(
            json.dumps(
                {
                    "phase": "63A-scene-freeze",
                    "status": "failed_closed_input_validation",
                    "lock_written": False,
                    "output_path": _display_path(args.output),
                    "error": str(exc),
                },
                indent=2,
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 2

    output_sha256 = write_lock_atomic(args.output, lock)
    print(
        json.dumps(
            {
                "phase": "63A-scene-freeze",
                "status": "locked_scene_disjoint",
                "lock_written": True,
                "output_path": _display_path(args.output),
                "output_sha256": output_sha256,
                "frozen_main_units": lock["frozen_main_units"],
                "development_units": lock["development_units"],
                "confirmation_units": lock["confirmation_units"],
                "snow_ice_units": lock["counts"]["snow_ice_units"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
