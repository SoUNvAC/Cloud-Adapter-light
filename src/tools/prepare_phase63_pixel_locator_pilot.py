"""Build the Phase 63 exact-pixel annotation UI pilot.

The pilot is deliberately excluded from model and confirmation statistics.  It
uses 16 previously unreviewed target-train patches (two from one scene in each
biome) and never reads pixel labels, model predictions, risk scores, or the
sealed target-test split.  The reviewer page separates a context locator from
an exact 21x21 nearest-neighbour pixel view.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from phase56_protocol import sha256
from prepare_phase61d2_calibrated_review import (
    BAND_NAMES,
    COMPOSITES,
    assign_ids,
    crop_bounds,
    draw_solar_geometry,
    mark_target,
    read_all_bands,
    render_band,
    stable_value,
)


EXPECTED_MANIFEST_SHA256 = "885c2d7f61cae23409a7b1ccbe65739c302fc1e59b089aab0cc05badf674872b"
EXPECTED_SELECTION_SHA256 = "564ad4e2c27d2948c725cf4c5c482159888c2324fe0ab4bdeee25fe774cebc55"
BIOMES = (
    "barren", "forest", "grass_crops", "shrubland",
    "snow_ice", "urban", "water", "wetlands",
)
PILOT_SCENES = 8
PATCHES_PER_SCENE = 2
PILOT_UNITS = PILOT_SCENES * PATCHES_PER_SCENE
LOCAL_WINDOW = 21
LOCAL_HALF = LOCAL_WINDOW // 2


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def verify_hash(path: str | Path, expected: str, label: str) -> str:
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"Frozen {label} hash mismatch: expected={expected}, actual={actual}")
    return actual


def deterministic_center(name: str, margin: int = 96) -> tuple[int, int]:
    span = 512 - 2 * margin
    if span <= 0 or margin < LOCAL_HALF:
        raise ValueError("margin does not leave a valid exact-pixel window")
    x = margin + stable_value(f"phase63-ui-pilot-x:{name}") % span
    y = margin + stable_value(f"phase63-ui-pilot-y:{name}") % span
    return int(y), int(x)


def select_pilot_records(
    manifest_rows: list[dict[str, str]],
    phase50_selected: list[dict],
    previously_reviewed_scenes: set[str],
) -> tuple[list[dict], dict]:
    selected_names = {row["name"] for row in phase50_selected}
    trained_scenes = {row["scene"] for row in phase50_selected}
    by_biome_scene: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for source in manifest_rows:
        if source["new_split"] != "target_train":
            continue
        if source["scene"] not in trained_scenes:
            continue
        if source["scene"] in previously_reviewed_scenes:
            continue
        if source["name"] in selected_names:
            continue
        by_biome_scene[source["biome"]][source["scene"]].append(source)

    selected: list[dict] = []
    selected_scene_ids = []
    for biome in BIOMES:
        scene_candidates = by_biome_scene.get(biome, {})
        eligible = [scene for scene, rows in scene_candidates.items() if len(rows) >= PATCHES_PER_SCENE]
        if not eligible:
            raise RuntimeError(f"No eligible previously unreviewed target-train scene for biome={biome}")
        scene = min(eligible, key=lambda value: stable_value(f"phase63-ui-pilot-scene:{biome}:{value}"))
        selected_scene_ids.append(scene)
        ordered = sorted(
            scene_candidates[scene],
            key=lambda row: stable_value(f"phase63-ui-pilot-patch:{scene}:{row['name']}"),
        )
        for source in ordered[:PATCHES_PER_SCENE]:
            row = dict(source)
            center_y, center_x = deterministic_center(row["name"])
            row.update(
                {
                    "center_y": center_y,
                    "center_x": center_x,
                    "pilot_role": "ui_locator_only_excluded_from_all_final_statistics",
                    "selection_uses_pixel_labels": False,
                    "selection_uses_model_predictions": False,
                }
            )
            selected.append(row)

    if len(selected) != PILOT_UNITS or len(set(selected_scene_ids)) != PILOT_SCENES:
        raise RuntimeError("Pilot must contain exactly 16 units from 8 distinct scenes")
    if len({row["name"] for row in selected}) != PILOT_UNITS:
        raise RuntimeError("Pilot patch names are not unique")
    if {row["scene"] for row in selected} & previously_reviewed_scenes:
        raise RuntimeError("Pilot scenes overlap a previously reviewed scene")
    if any(row["new_split"] == "target_test" for row in selected):
        raise RuntimeError("Target-test entered the UI pilot")

    audit = {
        "units": PILOT_UNITS,
        "scenes": PILOT_SCENES,
        "scene_ids": sorted(selected_scene_ids),
        "biome_counts": dict(sorted(Counter(row["biome"] for row in selected).items())),
        "split_counts": dict(sorted(Counter(row["new_split"] for row in selected).items())),
        "previously_reviewed_scene_overlap": [],
        "phase50_selected_patch_overlap": [],
        "target_test_units": 0,
        "selection_uses_pixel_labels": False,
        "selection_uses_model_predictions": False,
        "model_training_exposure_note": (
            "Scenes had at least one Phase50 selected patch; pilot patches themselves were not selected. "
            "This is acceptable only because the pilot evaluates annotation UI, not model performance."
        ),
    }
    return selected, audit


def paste_context(canvas: Image.Image, image: Image.Image, title: str, x: int, y: int) -> None:
    resized = image.resize((230, 230), Image.Resampling.NEAREST)
    canvas.paste(resized, (x, y + 18))
    ImageDraw.Draw(canvas).text((x, y), title, fill="white", font=ImageFont.load_default())


def mark_exact_pixel(image: Image.Image, scale: int) -> Image.Image:
    """Mark exactly one center source pixel while leaving its interior visible."""
    if image.size != (LOCAL_WINDOW * scale, LOCAL_WINDOW * scale):
        raise ValueError("exact-pixel marker requires an integer-scaled 21x21 image")
    result = image.copy()
    draw = ImageDraw.Draw(result)
    x0 = LOCAL_HALF * scale
    y0 = LOCAL_HALF * scale
    x1 = x0 + scale - 1
    y1 = y0 + scale - 1
    mid_x = (x0 + x1) // 2
    mid_y = (y0 + y1) // 2
    gap, arm = 5, max(10, 2 * scale)
    segments = (
        ((mid_x, max(0, y0 - gap - arm)), (mid_x, y0 - gap)),
        ((mid_x, y1 + gap), (mid_x, min(result.height - 1, y1 + gap + arm))),
        ((max(0, x0 - gap - arm), mid_y), (x0 - gap, mid_y)),
        ((x1 + gap, mid_y), (min(result.width - 1, x1 + gap + arm), mid_y)),
    )
    draw.rectangle((x0 - 3, y0 - 3, x1 + 3, y1 + 3), outline="black", width=2)
    draw.rectangle((x0 - 1, y0 - 1, x1 + 1, y1 + 1), outline="yellow", width=1)
    for width, color in ((5, "black"), (2, "yellow")):
        for start, end in segments:
            draw.line((start, end), fill=color, width=width)
    return result


def local_view(values, bands, row, scale: int) -> Image.Image:
    center_x, center_y = int(row["center_x"]), int(row["center_y"])
    bounds = (
        center_x - LOCAL_HALF,
        center_y - LOCAL_HALF,
        center_x + LOCAL_HALF + 1,
        center_y + LOCAL_HALF + 1,
    )
    image, _ = render_band(values, bands, bounds)
    image = image.resize((LOCAL_WINDOW * scale, LOCAL_WINDOW * scale), Image.Resampling.NEAREST)
    return mark_exact_pixel(image, scale)


def render_pixel_dossier(tile_id, row, values, mtl, raster_metadata, output_path) -> None:
    context_bounds = crop_bounds(row["center_x"], row["center_y"], 192)
    left, top, _, _ = context_bounds
    canvas = Image.new("RGB", (1040, 1210), "#17191c")
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 10), f"Phase 63 UI pilot | {tile_id}", fill="white")
    draw.text(
        (20, 28),
        f"Target = exactly one center pixel at (x={int(row['center_x'])}, y={int(row['center_y'])}) in the 512px patch.",
        fill="#ffd54a",
    )
    metadata = (
        f"scene={row['scene']}  biome={row['biome']}  north_up={raster_metadata['north_up']}",
        f"date={mtl.get('DATE_ACQUIRED', 'NA')}  time={mtl.get('SCENE_CENTER_TIME', 'NA')}",
        f"sun azimuth={float(mtl['SUN_AZIMUTH']):.3f} deg  elevation={float(mtl['SUN_ELEVATION']):.3f} deg",
        "Top row: 192x192 context only. Do not vote by majority area inside the yellow context locator.",
        "Second/third rows: 21x21 nearest-neighbour views. Hollow square encloses exactly one source pixel.",
        "No original label, prediction, error stratum, risk score, or method identity is shown.",
    )
    for index, line in enumerate(metadata):
        draw.text((20, 48 + 16 * index), line, fill="#d9dde3")

    for index, (title, bands) in enumerate(COMPOSITES):
        context, _ = render_band(values, bands, context_bounds)
        context = mark_target(context, row["center_x"], row["center_y"], left, top)
        paste_context(canvas, context, f"CONTEXT {title}", 20 + index * 250, 150)

    for index, (title, bands) in enumerate(COMPOSITES):
        image = local_view(values, bands, row, scale=11)
        x = 20 + index * 250
        canvas.paste(image, (x, 428))
        draw.text((x, 410), f"PIXEL VIEW {title}", fill="white")

    for index, title in enumerate(BAND_NAMES):
        image = local_view(values, (index + 1,), row, scale=7)
        x = 20 + (index % 6) * 165
        y = 690 + (index // 6) * 170
        canvas.paste(image, (x, y + 18))
        draw.text((x, y), f"PIXEL VIEW {title}", fill="white")

    draw_solar_geometry(canvas, 810, 1040, float(mtl["SUN_AZIMUTH"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, optimize=True)


def write_review_csv(path: Path, records: list[dict]) -> None:
    fields = (
        "tile_id", "target_locatable", "identifiability",
        "semantic_label", "elapsed_seconds", "notes",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow({field: row["tile_id"] if field == "tile_id" else "" for field in fields})


def write_option_tables(packet_root: Path) -> None:
    with (packet_root / "identifiability_options.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("value", "meaning"))
        writer.writerows(
            (
                ("fine_classifiable", "中心像元具备细粒度判读条件；必须继续填写semantic_label"),
                ("mixed_boundary", "中心像元位于混合/配准/类别边界；semantic_label留空"),
                ("insufficient_evidence", "位置清楚但影像证据不足；semantic_label留空"),
                ("unobservable_nodata", "nodata、纯色、填充值或无有效信息；semantic_label留空"),
            )
        )
    with (packet_root / "semantic_options.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("family", "value", "meaning"))
        writer.writerows(
            (
                ("surface", "clear", "无可辨识云、雾霾或阴影污染"),
                ("atmospheric", "thin_cloud", "半透明云，地表纹理仍可见"),
                ("atmospheric", "thick_cloud", "明显不透明或地表被遮蔽的云体"),
                ("atmospheric", "haze_cirrus", "弥散雾霾或高层卷云"),
                ("shadow", "cloud_shadow", "有可信云—影几何关联"),
                ("shadow", "terrain_water_shadow", "更符合地形、水体或局部照明阴影"),
            )
        )


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
    parser.add_argument("--selection", default="work_dirs/phase50_active_1pct/selection.json")
    parser.add_argument(
        "--phase63-sealed",
        default="work_dirs/phase63_selective_multigranularity/human_confirmation_review/sealed_manifest.json",
    )
    parser.add_argument("--manual", default="research_plans/PHASE63_PIXEL_UI_PILOT_MANUAL.md")
    parser.add_argument("--raw-root", required=True)
    parser.add_argument(
        "--output-root",
        default="work_dirs/phase63_selective_multigranularity/pixel_locator_ui_pilot",
    )
    args = parser.parse_args()

    manifest_hash = verify_hash(args.manifest, EXPECTED_MANIFEST_SHA256, "manifest")
    selection_hash = verify_hash(args.selection, EXPECTED_SELECTION_SHA256, "Phase50 selection")
    manifest_rows = read_csv(args.manifest)
    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    phase63 = json.loads(Path(args.phase63_sealed).read_text(encoding="utf-8"))
    previously_reviewed_scenes = set(phase63["selection_audit"]["scene_ids"])
    previously_reviewed_scenes.update(phase63["development_scene_ids"])
    previously_reviewed_scenes.update(row["scene"] for row in phase63["calibration_records"])
    records, audit = select_pilot_records(
        manifest_rows, selection["selected"], previously_reviewed_scenes
    )
    records = assign_ids(records, "P63-UI", "phase63-exact-pixel-ui-pilot-order")

    output_root = Path(args.output_root)
    packet_root = output_root / "phase63_pixel_locator_ui_pilot_packet"
    output_zip = output_root / "phase63_pixel_locator_ui_pilot_packet.zip"
    protected = (packet_root, output_zip, output_root / "sealed_manifest.json")
    if any(path.exists() for path in protected):
        raise RuntimeError("Refusing to overwrite an existing Phase63 UI pilot packet")
    panel_root = packet_root / "panels"
    panel_root.mkdir(parents=True, exist_ok=False)

    sealed_records = []
    for index, row in enumerate(records, 1):
        values, mtl, raster_metadata = read_all_bands(args.raw_root, row)
        if not raster_metadata["north_up"]:
            raise RuntimeError(f"North-up check failed: {row['scene']}")
        render_pixel_dossier(
            row["tile_id"], row, values, mtl, raster_metadata,
            panel_root / f"{row['tile_id']}.png",
        )
        sealed_records.append(
            {
                **row,
                "sun_azimuth_degrees": float(mtl["SUN_AZIMUTH"]),
                "sun_elevation_degrees": float(mtl["SUN_ELEVATION"]),
                "date_acquired": mtl.get("DATE_ACQUIRED"),
                **raster_metadata,
            }
        )
        print(f"Phase63 exact-pixel UI pilot rendering: {index}/{len(records)}", flush=True)

    shutil.copy2(args.manual, packet_root / "PHASE63_PIXEL_UI_PILOT_MANUAL.md")
    write_review_csv(packet_root / "reviewer_A_ui_pilot.csv", records)
    write_review_csv(packet_root / "reviewer_B_ui_pilot.csv", records)
    write_option_tables(packet_root)
    (packet_root / "README.md").write_text(
        "# Phase 63 exact-pixel annotation UI pilot\n\n"
        "This 16-item pilot is excluded from all final model and confirmation statistics. "
        "Each reviewer must work independently and view each item once. The top row is context only; "
        "the hollow square in the nearest-neighbour pixel views encloses exactly one source pixel. "
        "Fill target_locatable first, then identifiability, and fill semantic_label only when "
        "identifiability=fine_classifiable. Record elapsed_seconds per item. Do not discuss pilot "
        "items until both raw CSV files are frozen.\n",
        encoding="utf-8",
    )
    images = sorted(panel_root.glob("*.png"))
    if len(images) != PILOT_UNITS:
        raise RuntimeError(f"Expected {PILOT_UNITS} pilot panels, found {len(images)}")
    for path in images:
        with Image.open(path) as image:
            if image.size != (1040, 1210) or image.mode != "RGB":
                raise RuntimeError(f"Invalid pilot panel geometry: {path}")
    write_hash_manifest(packet_root)
    zip_packet(packet_root, output_zip)

    output_root.mkdir(parents=True, exist_ok=True)
    sealed = {
        "phase": "63-exact-pixel-annotation-ui-pilot",
        "status": "awaiting_independent_review",
        "excluded_from_final_model_and_confirmation_statistics": True,
        "target_test_read": False,
        "inputs": {
            "manifest_sha256": manifest_hash,
            "phase50_selection_sha256": selection_hash,
            "phase63_sealed_sha256": sha256(args.phase63_sealed),
            "manual_sha256": sha256(args.manual),
        },
        "selection_audit": audit,
        "records": sealed_records,
    }
    sealed_path = output_root / "sealed_manifest.json"
    sealed_path.write_text(json.dumps(sealed, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {
        "phase": sealed["phase"],
        "status": sealed["status"],
        "pilot_units": PILOT_UNITS,
        "pilot_scenes": PILOT_SCENES,
        "selection_audit": audit,
        "reviewer_packet_zip": {"path": str(output_zip), "sha256": sha256(output_zip)},
        "sealed_manifest": {"path": str(sealed_path), "sha256": sha256(sealed_path)},
        "target_test_read": False,
    }
    summary_path = output_root / "packet_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
