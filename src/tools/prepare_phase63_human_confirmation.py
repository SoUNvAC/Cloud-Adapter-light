"""Build the blind, scene-independent Phase 63 human confirmation packet.

The selection is pixel-label- and model-blind, while deliberately using the
pre-registered ``Shadows?`` and biome metadata as sampling strata. It uses
eight target-val scenes whose USGS metadata says ``Shadows?=no`` plus three
snow/ice target-train scenes that received zero Phase 50 selected patches.
Target-test remains sealed. Existing Phase 61D2 calibration panels are reused;
no prior human label enters the reviewer-visible packet.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from prepare_phase61d2_calibrated_review import (
    ATOMIC_LABELS,
    assign_ids,
    read_all_bands,
    render_dossier,
    stable_value,
)
from phase56_protocol import sha256


EXPECTED_MANIFEST_SHA256 = "885c2d7f61cae23409a7b1ccbe65739c302fc1e59b089aab0cc05badf674872b"
EXPECTED_SELECTION_SHA256 = "564ad4e2c27d2948c725cf4c5c482159888c2324fe0ab4bdeee25fe774cebc55"
EXPECTED_D2_SEALED_SHA256 = "7c9468778ccb44127610918ae537a64d7e1e0af95469c403a0c6c1453d587c77"
EXPECTED_MANUAL_SHA256 = "cfa8ec5b65474dc4e6d4402cc9db0838db237afd715f081ff88f41cdb068d172"
VAL_NO_PER_SCENE = 4
SNOW_UNSELECTED_PER_SCENE = 6
EXPECTED_VAL_NO_SCENES = 8
EXPECTED_SNOW_UNSELECTED_SCENES = 3
EXPECTED_CONFIRMATION_UNITS = 50


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def verify_hash(path: str | Path, expected: str, label: str) -> str:
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"Frozen {label} hash mismatch: expected={expected}, actual={actual}")
    return actual


def deterministic_center(name: str, margin: int = 96) -> tuple[int, int]:
    if not 0 <= margin < 256:
        raise ValueError("margin must keep a non-empty 512px center region")
    span = 512 - 2 * margin
    x = margin + stable_value(f"phase63-confirm-x:{name}") % span
    y = margin + stable_value(f"phase63-confirm-y:{name}") % span
    return int(y), int(x)


def choose_scene_rows(rows: list[dict[str, str]], count: int, salt: str) -> list[dict[str, str]]:
    if len({row["name"] for row in rows}) != len(rows):
        raise RuntimeError(f"Duplicate patch names in scene pool: {salt}")
    ordered = sorted(rows, key=lambda row: stable_value(f"{salt}:{row['name']}"))
    if len(ordered) < count:
        raise RuntimeError(f"Scene {salt} has only {len(ordered)}/{count} candidate patches")
    selected = []
    for source in ordered[:count]:
        row = dict(source)
        center_y, center_x = deterministic_center(row["name"])
        row.update(
            {
                "center_y": center_y,
                "center_x": center_x,
                "crop_size": 192,
                "cohort": "confirmation",
                "selection_uses_pixel_labels": False,
                "selection_uses_model_predictions": False,
                "selection_uses_protocol_metadata_strata": True,
            }
        )
        selected.append(row)
    return selected


def select_confirmation_records(
    manifest_rows: list[dict[str, str]],
    shadow_status: dict[str, str],
    selected_training_scenes: set[str],
    development_scenes: set[str],
) -> tuple[list[dict[str, str]], dict]:
    by_scene: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in manifest_rows:
        by_scene[row["scene"]].append(row)

    val_no_scenes = sorted(
        scene
        for scene, rows in by_scene.items()
        if rows[0]["new_split"] == "target_val"
        and shadow_status.get(scene) == "no"
        and scene not in development_scenes
    )
    snow_unselected_scenes = sorted(
        scene
        for scene, rows in by_scene.items()
        if rows[0]["new_split"] == "target_train"
        and rows[0]["biome"] == "snow_ice"
        and scene not in selected_training_scenes
        and scene not in development_scenes
    )
    if len(val_no_scenes) != EXPECTED_VAL_NO_SCENES:
        raise RuntimeError(f"Expected 8 unseen target-val Shadows?=no scenes, found {val_no_scenes}")
    if len(snow_unselected_scenes) != EXPECTED_SNOW_UNSELECTED_SCENES:
        raise RuntimeError(
            "Expected 3 zero-gradient-exposure target-train snow scenes, found "
            f"{snow_unselected_scenes}"
        )
    if set(val_no_scenes) & set(snow_unselected_scenes):
        raise RuntimeError("Confirmation scene pools overlap")

    selected: list[dict[str, str]] = []
    for scene in val_no_scenes:
        scene_rows = [row for row in by_scene[scene] if row["new_split"] == "target_val"]
        picked = choose_scene_rows(scene_rows, VAL_NO_PER_SCENE, f"val-no:{scene}")
        for row in picked:
            row["confirmation_source"] = "target_val_shadows_no"
        selected.extend(picked)
    for scene in snow_unselected_scenes:
        scene_rows = [row for row in by_scene[scene] if row["new_split"] == "target_train"]
        picked = choose_scene_rows(
            scene_rows, SNOW_UNSELECTED_PER_SCENE, f"snow-unselected:{scene}"
        )
        for row in picked:
            row["confirmation_source"] = "target_train_zero_selected_scene_snow_stress"
        selected.extend(picked)

    if len(selected) != EXPECTED_CONFIRMATION_UNITS:
        raise RuntimeError(f"Expected 50 confirmation units, found {len(selected)}")
    if len({row["name"] for row in selected}) != len(selected):
        raise RuntimeError("Confirmation patch names are not unique")
    selected_scenes = {row["scene"] for row in selected}
    if selected_scenes & development_scenes:
        raise RuntimeError("New confirmation scenes overlap the 160-unit development scenes")
    if selected_scenes & selected_training_scenes:
        raise RuntimeError("New confirmation scenes overlap Phase50 selected training scenes")
    if any(row["new_split"] == "target_test" for row in selected):
        raise RuntimeError("Target-test was selected despite the sealed-test prohibition")

    audit = {
        "units": len(selected),
        "scenes": len(selected_scenes),
        "scene_ids": sorted(selected_scenes),
        "development_scene_overlap": [],
        "phase50_selected_scene_overlap": [],
        "target_test_units": 0,
        "source_counts": dict(sorted(Counter(row["confirmation_source"] for row in selected).items())),
        "split_counts": dict(sorted(Counter(row["new_split"] for row in selected).items())),
        "biome_counts": dict(sorted(Counter(row["biome"] for row in selected).items())),
        "scene_counts": dict(sorted(Counter(row["scene"] for row in selected).items())),
        "selection_uses_pixel_labels": False,
        "selection_uses_model_predictions": False,
        "selection_uses_protocol_metadata_strata": True,
    }
    return selected, audit


def write_blank_review(path: Path, records: list[dict], label_column: str = "label_set") -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("tile_id", label_column, "notes"))
        writer.writeheader()
        for row in records:
            writer.writerow({"tile_id": row["tile_id"], label_column: "", "notes": ""})


def write_hash_manifest(packet_root: Path) -> None:
    files = sorted(path for path in packet_root.rglob("*") if path.is_file())
    lines = [f"{sha256(path)}  {path.relative_to(packet_root).as_posix()}" for path in files]
    (packet_root / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def zip_packet(packet_root: Path, output_zip: Path) -> None:
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(item for item in packet_root.rglob("*") if item.is_file()):
            archive.write(path, path.relative_to(packet_root.parent).as_posix())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="work_dirs/phase45_protocol_audit/scene_disjoint_manifest.csv")
    parser.add_argument("--metadata", default="research_plans/protocol_data/l8_biome_usgs_shadow_status.csv")
    parser.add_argument("--selection", default="work_dirs/phase50_active_1pct/selection.json")
    parser.add_argument("--d2-sealed", default="work_dirs/phase61d2_calibrated_review/sealed_manifest.json")
    parser.add_argument("--d2-panels", default="work_dirs/phase61d2_calibrated_review/reviewer_packet/panels")
    parser.add_argument("--manual", default="research_plans/PHASE61D2_REVIEW_MANUAL.md")
    parser.add_argument("--raw-root", required=True)
    parser.add_argument(
        "--output-root",
        default="work_dirs/phase63_selective_multigranularity/human_confirmation_review",
    )
    args = parser.parse_args()

    paths = {
        "manifest": Path(args.manifest),
        "selection": Path(args.selection),
        "d2_sealed": Path(args.d2_sealed),
        "manual": Path(args.manual),
    }
    hashes = {
        "manifest": verify_hash(paths["manifest"], EXPECTED_MANIFEST_SHA256, "manifest"),
        "selection": verify_hash(paths["selection"], EXPECTED_SELECTION_SHA256, "selection"),
        "d2_sealed": verify_hash(paths["d2_sealed"], EXPECTED_D2_SEALED_SHA256, "D2 sealed"),
        "manual": verify_hash(paths["manual"], EXPECTED_MANUAL_SHA256, "manual"),
        "metadata": sha256(args.metadata),
    }
    output_root = Path(args.output_root)
    reviewer_root = output_root / "phase63_confirmation_reviewer_packet"
    output_zip = output_root / "phase63_confirmation_reviewer_packet.zip"
    protected = (reviewer_root, output_zip, output_root / "sealed_manifest.json")
    if any(path.exists() for path in protected):
        raise RuntimeError("Refusing to overwrite an existing Phase63 human confirmation packet")

    manifest_rows = read_csv(paths["manifest"])
    status_rows = read_csv(args.metadata)
    shadow_status = {row["scene"]: row["usgs_shadows"].strip().lower() for row in status_rows}
    selection = json.loads(paths["selection"].read_text(encoding="utf-8"))
    selected_training_scenes = {row["scene"] for row in selection["selected"]}
    d2 = json.loads(paths["d2_sealed"].read_text(encoding="utf-8"))
    calibration = [dict(row) for row in d2["records"] if row["cohort"] == "calibration"]
    development = [dict(row) for row in d2["records"] if row["cohort"] == "independent"]
    if len(calibration) != 40 or len(development) != 160:
        raise RuntimeError("Frozen Phase61D2 calibration/development counts changed")
    development_scenes = {row["scene"] for row in development}
    confirmation, selection_audit = select_confirmation_records(
        manifest_rows, shadow_status, selected_training_scenes, development_scenes
    )
    confirmation = assign_ids(
        confirmation, "P63-CONF", "phase63-scene-independent-confirmation-order"
    )

    panel_root = reviewer_root / "panels"
    panel_root.mkdir(parents=True, exist_ok=False)
    d2_panel_root = Path(args.d2_panels)
    for row in calibration:
        source = d2_panel_root / f"{row['tile_id']}.png"
        if not source.is_file():
            raise RuntimeError(f"Missing frozen calibration panel: {source}")
        shutil.copy2(source, panel_root / source.name)

    sealed_confirmation = []
    for index, row in enumerate(confirmation, 1):
        values, mtl, raster_metadata = read_all_bands(args.raw_root, row)
        if not raster_metadata["north_up"]:
            raise RuntimeError(f"North-up check failed: {row['scene']}")
        panel_path = panel_root / f"{row['tile_id']}.png"
        render_dossier(
            row["tile_id"], row, values, mtl, raster_metadata, panel_path,
            phase_title="Phase 63 confirmation",
        )
        sealed_confirmation.append(
            {
                **row,
                "sun_azimuth_degrees": float(mtl["SUN_AZIMUTH"]),
                "sun_elevation_degrees": float(mtl["SUN_ELEVATION"]),
                "date_acquired": mtl.get("DATE_ACQUIRED"),
                **raster_metadata,
            }
        )
        if index == 1 or index % 10 == 0 or index == len(confirmation):
            print(f"Phase63 confirmation panel rendering: {index}/{len(confirmation)}", flush=True)

    shutil.copy2(paths["manual"], reviewer_root / "PHASE61D2_REVIEW_MANUAL.md")
    (reviewer_root / "README.md").write_text(
        "# Phase 63 independent human confirmation\n\n"
        "Two reviewers must use the included Phase61D2 manual. Complete the 40 calibration "
        "items first, discuss calibration, and record consensus before either reviewer starts "
        "the 50 confirmation items. The two confirmation CSVs must then be completed "
        "independently. Reviewers are blind to original labels, model predictions, risk scores, "
        "selection source, and method identity. Use `unobservable_nodata`, `uncertain`, or a "
        "minimal two-label set when required; do not force a four-class answer. Preserve both "
        "raw reviewer files. A third expert will later see only genuine A/B disagreements.\n",
        encoding="utf-8",
    )
    for reviewer in ("reviewer_A", "reviewer_B"):
        write_blank_review(reviewer_root / f"{reviewer}_calibration.csv", calibration)
        write_blank_review(reviewer_root / f"{reviewer}_confirmation.csv", confirmation)
    with (reviewer_root / "calibration_consensus.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("tile_id", "consensus_label_set", "rule_or_counterexample"),
        )
        writer.writeheader()
        for row in calibration:
            writer.writerow(
                {"tile_id": row["tile_id"], "consensus_label_set": "", "rule_or_counterexample": ""}
            )

    images = sorted(panel_root.glob("*.png"))
    if len(images) != 90:
        raise RuntimeError(f"Expected 90 reviewer-visible panels, found {len(images)}")
    for image_path in images:
        with Image.open(image_path) as image:
            if image.size != (1040, 1210) or image.mode != "RGB":
                raise RuntimeError(f"Invalid panel geometry: {image_path}")
    write_hash_manifest(reviewer_root)
    zip_packet(reviewer_root, output_zip)

    sealed = {
        "phase": "63-independent-human-confirmation",
        "status": "awaiting_new_reviewer_calibration",
        "selection_policy": (
            "pixel-label/model-blind deterministic patch and center sampling with pre-registered "
            "Shadows?/biome strata; 8 target-val Shadows?=no scenes plus 3 snow/ice target-train "
            "scenes with zero Phase50 selected patches"
        ),
        "calibration_units_excluded": 40,
        "development_units_frozen_elsewhere": 160,
        "confirmation_units": 50,
        "confirmation_scenes": 11,
        "target_test_read": False,
        "inputs": hashes,
        "selection_audit": selection_audit,
        "development_scene_ids": sorted(development_scenes),
        "phase50_selected_scene_ids": sorted(selected_training_scenes),
        "atomic_labels": list(ATOMIC_LABELS),
        "calibration_records": calibration,
        "confirmation_records": sealed_confirmation,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    sealed_path = output_root / "sealed_manifest.json"
    sealed_path.write_text(json.dumps(sealed, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {
        "phase": "63-independent-human-confirmation-packet",
        "status": "awaiting_new_reviewer_calibration",
        "reviewer_visible_panels": 90,
        "calibration_units": 40,
        "confirmation_units": 50,
        "confirmation_scenes": 11,
        "selection_audit": selection_audit,
        "reviewer_packet_zip": {"path": str(output_zip), "sha256": sha256(output_zip)},
        "sealed_manifest": {"path": str(sealed_path), "sha256": sha256(sealed_path)},
        "manual_sha256": hashes["manual"],
        "reviewer_files_blank": True,
        "target_test_read": False,
    }
    summary_path = output_root / "packet_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
