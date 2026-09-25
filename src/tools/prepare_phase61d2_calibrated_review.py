"""Build the calibrated, multispectral Phase 61D2 blind-review packet.

This program selects review units and renders evidence.  It never creates or
infers a human review label.  Ground truth, model-error strata, and cohort
membership are stored only in the sealed manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = Path(__file__).resolve().parent
for item in (REPO_ROOT, TOOLS_ROOT):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from phase54_raw_data import parse_mtl, scene_directory  # noqa: E402
from phase56_protocol import read_shadow_yes_rows, sha256  # noqa: E402


ATOMIC_LABELS = (
    "clear",
    "thin_cloud",
    "thick_cloud",
    "cloud_shadow",
    "terrain_water_shadow",
    "haze_cirrus",
    "boundary_mixed",
    "unobservable_nodata",
    "uncertain",
)
SEMANTIC_LABELS = frozenset(ATOMIC_LABELS[:6])
ORIGINAL_NAMES = ("clear", "thick_cloud", "thin_cloud", "cloud_shadow")
LEGACY_LABEL_MAP = {
    "definite clear": "clear",
    "definite thick": "thick_cloud",
    "definite thin": "thin_cloud",
    "definite cloud shadow": "cloud_shadow",
}
BAND_NAMES = (
    "B1 coastal", "B2 blue", "B3 green", "B4 red", "B5 NIR",
    "B6 SWIR1", "B7 SWIR2", "B8 pan", "B9 cirrus", "B10 TIR1", "B11 TIR2",
)
COMPOSITES = (
    ("True color 4-3-2", (4, 3, 2)),
    ("Color infrared 5-4-3", (5, 4, 3)),
    ("SWIR cloud/shadow 7-6-4", (7, 6, 4)),
    ("Cirrus/coastal/blue 9-1-2", (9, 1, 2)),
)


def stable_value(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def cache_path(cache_root, split, row):
    return Path(cache_root) / split / f"{Path(row['name']).stem}.npz"


def boundary_mask(mask):
    edges = np.zeros(mask.shape, dtype=bool)
    edges[1:, :] |= mask[1:, :] != mask[:-1, :]
    edges[:-1, :] |= mask[:-1, :] != mask[1:, :]
    edges[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    edges[:, :-1] |= mask[:, :-1] != mask[:, 1:]
    padded = np.pad(edges, 3, mode="constant", constant_values=False)
    dilated = np.zeros_like(edges)
    for offset_y in range(7):
        for offset_x in range(7):
            dilated |= padded[offset_y:offset_y + mask.shape[0], offset_x:offset_x + mask.shape[1]]
    return dilated


def stratified_pick(records, budget, key_fields, salt, max_per_image=None):
    """Deterministic round-robin allocation across all represented strata."""
    groups = defaultdict(list)
    for row in records:
        groups[tuple(row.get(field, "") for field in key_fields)].append(row)
    for key, values in groups.items():
        values.sort(key=lambda row: stable_value(
            f"{salt}:{key}:{row['name']}:{row['center_y']}:{row['center_x']}"
        ))
    selected, image_counts = [], Counter()
    keys = sorted(groups)
    while len(selected) < budget:
        progress = False
        for key in keys:
            values = groups[key]
            while values and max_per_image is not None and image_counts[values[0]["name"]] >= max_per_image:
                values.pop(0)
            if values and len(selected) < budget:
                row = values.pop(0)
                selected.append(row)
                image_counts[row["name"]] += 1
                progress = True
        if not progress:
            break
    if len(selected) != budget:
        raise RuntimeError(f"Could select only {len(selected)}/{budget} units for {salt}")
    return selected


def split_legacy_records(records, calibration_budget=40):
    if len(records) != 200:
        raise RuntimeError(f"Expected the frozen 200-unit Phase 61D packet, found {len(records)}")
    normalized = []
    for source in records:
        row = dict(source)
        row["original_label_name"] = LEGACY_LABEL_MAP.get(
            row["original_label_name"], row["original_label_name"]
        )
        normalized.append(row)
    calibration = stratified_pick(
        normalized, calibration_budget,
        ("source_stratum", "biome", "confidence_stratum", "region"),
        "phase61d2-calibration", max_per_image=2,
    )
    calibration_keys = {
        (row["name"], int(row["center_y"]), int(row["center_x"])) for row in calibration
    }
    independent = [
        dict(row) for row in normalized
        if (row["name"], int(row["center_y"]), int(row["center_x"])) not in calibration_keys
    ]
    if len(independent) != 160:
        raise RuntimeError(f"Expected 160 independent legacy units, found {len(independent)}")
    return calibration, independent


def confirmation_candidates(validation, used_names, cache_root):
    records = []
    for row in validation:
        if row["name"] in used_names:
            continue
        path = cache_path(Path(cache_root), "val", row)
        with np.load(path) as item:
            mask = item["source_label"].astype(np.uint8)
        boundary = boundary_mask(mask)
        for label, label_name in ((2, "thin_cloud"), (3, "cloud_shadow")):
            for region, region_mask in (("boundary", boundary), ("interior", ~boundary)):
                coordinates = np.argwhere((mask == label) & region_mask)
                if len(coordinates) == 0:
                    continue
                order = sorted(
                    range(len(coordinates)),
                    key=lambda index: stable_value(
                        f"61D2-confirm:{row['name']}:{coordinates[index, 0]}:{coordinates[index, 1]}"
                    ),
                )
                y, x = map(int, coordinates[order[0]])
                records.append({
                    "source_stratum": "confirmation_not_model_conditioned",
                    "name": row["name"], "scene": row["scene"], "biome": row["biome"],
                    "center_y": y, "center_x": x, "region": region,
                    "original_source_order_label": label,
                    "original_label_name": label_name,
                })
    return records


def select_confirmation(validation, used_names, cache_root, budget=50):
    if budget % 2:
        raise RuntimeError("Confirmation budget must be even for Thin/Shadow balance")
    candidates = confirmation_candidates(validation, used_names, cache_root)
    thin_pool = [row for row in candidates if row["original_label_name"] == "thin_cloud"]
    thin = stratified_pick(
        thin_pool, budget // 2, ("scene", "biome", "region"),
        "phase61d2-confirm-thin_cloud", max_per_image=1,
    )
    thin_names = {row["name"] for row in thin}
    shadow_pool = [
        row for row in candidates
        if row["original_label_name"] == "cloud_shadow" and row["name"] not in thin_names
    ]
    shadow = stratified_pick(
        shadow_pool, budget // 2, ("scene", "biome", "region"),
        "phase61d2-confirm-cloud_shadow", max_per_image=1,
    )
    selected = thin + shadow
    if len({row["name"] for row in selected}) != budget:
        raise RuntimeError("Confirmation set must use one previously unseen image per review unit")
    return selected, len(candidates)


def read_all_bands(raw_root, row):
    try:
        import rasterio
        from rasterio.windows import Window
        from rasterio.warp import transform as warp_transform
    except ImportError as error:
        raise RuntimeError("Phase 61D2 requires rasterio==1.3.11") from error
    directory = scene_directory(raw_root, row)
    raster_path = directory / f"{row['scene']}.TIF"
    mtl_path = directory / f"{row['scene']}_MTL.txt"
    column, line = int(row["y"]), int(row["x"])
    with rasterio.open(raster_path) as dataset:
        if dataset.count != 11:
            raise RuntimeError(f"Expected 11 bands, found {dataset.count}: {raster_path}")
        values = dataset.read(
            tuple(range(1, 12)), window=Window(column, line, 512, 512),
            boundless=True, fill_value=0,
        ).astype(np.float32)
        center_column = column + int(row["center_x"])
        center_line = line + int(row["center_y"])
        projected_x, projected_y = dataset.xy(center_line, center_column)
        latitude = longitude = None
        if dataset.crs:
            longitude_values, latitude_values = warp_transform(
                dataset.crs, "EPSG:4326", [projected_x], [projected_y]
            )
            longitude, latitude = longitude_values[0], latitude_values[0]
        north_up = bool(dataset.transform.e < 0 and abs(dataset.transform.b) < 1e-9 and abs(dataset.transform.d) < 1e-9)
        raster_metadata = {
            "raster_crs": str(dataset.crs), "north_up": north_up,
            "target_projected_x": float(projected_x), "target_projected_y": float(projected_y),
            "target_longitude": None if longitude is None else float(longitude),
            "target_latitude": None if latitude is None else float(latitude),
        }
    return values, parse_mtl(mtl_path), raster_metadata


def crop_bounds(center_x, center_y, size=192, width=512, height=512):
    half = size // 2
    left = min(max(int(center_x) - half, 0), width - size)
    top = min(max(int(center_y) - half, 0), height - size)
    return left, top, left + size, top + size


def stretch_band(full_band, crop_band):
    valid = full_band[np.isfinite(full_band) & (full_band != 0)]
    if valid.size < 16:
        return np.zeros(crop_band.shape, dtype=np.uint8), (None, None)
    low, high = np.percentile(valid, (2, 98))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros(crop_band.shape, dtype=np.uint8), (float(low), float(high))
    output = np.clip((crop_band - low) / (high - low), 0, 1)
    return np.round(output * 255).astype(np.uint8), (float(low), float(high))


def render_band(values, bands, bounds):
    left, top, right, bottom = bounds
    channels, stretches = [], []
    for band in bands:
        channel, limits = stretch_band(values[band - 1], values[band - 1, top:bottom, left:right])
        channels.append(channel)
        stretches.append(limits)
    if len(channels) == 1:
        image = np.repeat(channels[0][..., None], 3, axis=2)
    else:
        image = np.stack(channels, axis=2)
    return Image.fromarray(image, mode="RGB"), stretches


def mark_target(image, target_x, target_y, left, top):
    result = image.copy()
    draw = ImageDraw.Draw(result)
    x, y = int(target_x - left), int(target_y - top)
    radius, arm = 10, 6
    segments = (
        ((x - radius, y - radius), (x - radius + arm, y - radius)),
        ((x - radius, y - radius), (x - radius, y - radius + arm)),
        ((x + radius, y - radius), (x + radius - arm, y - radius)),
        ((x + radius, y - radius), (x + radius, y - radius + arm)),
        ((x - radius, y + radius), (x - radius + arm, y + radius)),
        ((x - radius, y + radius), (x - radius, y + radius - arm)),
        ((x + radius, y + radius), (x + radius - arm, y + radius)),
        ((x + radius, y + radius), (x + radius, y + radius - arm)),
    )
    for width, color in ((5, "black"), (2, "yellow")):
        for start, end in segments:
            draw.line((start, end), fill=color, width=width)
    return result


def paste_panel(canvas, image, title, x, y, size=210):
    thumb = image.resize((size, size), Image.Resampling.NEAREST)
    canvas.paste(thumb, (x, y + 20))
    ImageDraw.Draw(canvas).text((x, y), title, fill="white", font=ImageFont.load_default())


def draw_solar_geometry(canvas, x, y, azimuth):
    draw = ImageDraw.Draw(canvas)
    center = (x + 70, y + 70)
    radius = 48
    angle = math.radians(azimuth)
    dx, dy = radius * math.sin(angle), -radius * math.cos(angle)
    draw.ellipse((center[0] - 55, center[1] - 55, center[0] + 55, center[1] + 55), outline="white", width=2)
    draw.text((center[0] - 4, center[1] - 67), "N", fill="white")
    draw.line((center[0], center[1], center[0] + dx, center[1] + dy), fill="#ffd54a", width=5)
    draw.line((center[0], center[1], center[0] - dx, center[1] - dy), fill="#68b5ff", width=5)
    draw.text((x, y + 132), "yellow: bearing to sun", fill="#ffd54a")
    draw.text((x, y + 148), "blue: expected shadow direction", fill="#68b5ff")


def render_dossier(tile_id, row, values, mtl, raster_metadata, output_path, crop_size=192):
    bounds = crop_bounds(row["center_x"], row["center_y"], crop_size)
    left, top, _, _ = bounds
    canvas = Image.new("RGB", (1040, 1210), "#17191c")
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 12), f"Phase 61D2 | {tile_id} | classify the bracketed pixel", fill="white")
    latitude = raster_metadata["target_latitude"]
    longitude = raster_metadata["target_longitude"]
    location = "lat=NA  lon=NA" if latitude is None else f"lat={latitude:.6f}  lon={longitude:.6f}"
    metadata_lines = [
        f"scene={row['scene']}  biome={row['biome']}  north_up={raster_metadata['north_up']}",
        f"date={mtl.get('DATE_ACQUIRED', 'NA')}  time={mtl.get('SCENE_CENTER_TIME', 'NA')}",
        f"sun azimuth={float(mtl['SUN_AZIMUTH']):.3f} deg  elevation={float(mtl['SUN_ELEVATION']):.3f} deg",
        location,
        "Display: per-band 2-98% stretch over the full 512px tile; values are relative, not calibrated temperature.",
        "No original label, prediction, error stratum, or method identity is shown.",
    ]
    for index, line in enumerate(metadata_lines):
        draw.text((20, 34 + 16 * index), line, fill="#d9dde3")

    for index, (title, bands) in enumerate(COMPOSITES):
        image, _ = render_band(values, bands, bounds)
        image = mark_target(image, row["center_x"], row["center_y"], left, top)
        paste_panel(canvas, image, title, 20 + (index % 4) * 250, 145, 230)

    for index, title in enumerate(BAND_NAMES):
        image, _ = render_band(values, (index + 1,), bounds)
        image = mark_target(image, row["center_x"], row["center_y"], left, top)
        paste_panel(canvas, image, title, 20 + (index % 4) * 250, 415 + (index // 4) * 245, 220)
    draw_solar_geometry(canvas, 790, 930, float(mtl["SUN_AZIMUTH"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, optimize=True)


def assign_ids(records, prefix, salt):
    ordered = sorted(records, key=lambda row: stable_value(
        f"{salt}:{row['name']}:{row['center_y']}:{row['center_x']}"
    ))
    for index, row in enumerate(ordered, 1):
        token = stable_value(f"{salt}:id:{row['name']}:{row['center_y']}:{row['center_x']}")
        row["tile_id"] = f"{prefix}-{index:04d}-{token:016x}"
    return ordered


def write_blank_review(path, records):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("tile_id", "label_set", "notes"))
        writer.writeheader()
        for row in records:
            writer.writerow({"tile_id": row["tile_id"], "label_set": "", "notes": ""})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-sealed", default="work_dirs/phase61/blind_review/sealed_manifest.json")
    parser.add_argument("--manifest", default="work_dirs/phase45_protocol_audit/scene_disjoint_manifest.csv")
    parser.add_argument("--metadata", default="research_plans/protocol_data/l8_biome_usgs_shadow_status.csv")
    parser.add_argument("--selection", default="work_dirs/phase50_active_1pct/selection.json")
    parser.add_argument("--phase54-cache", default="work_dirs/phase54_information_audit/cache")
    parser.add_argument("--raw-root", required=True, help="Directory containing l8biome/<biome>/<scene>.")
    parser.add_argument("--manual", default="research_plans/PHASE61D2_REVIEW_MANUAL.md")
    parser.add_argument("--output-root", default="work_dirs/phase61d2_calibrated_review")
    parser.add_argument("--calibration-budget", type=int, default=40)
    parser.add_argument("--confirmation-budget", type=int, default=50)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    protected_outputs = (
        output_root / "sealed_manifest.json",
        output_root / "reviewer_packet/reviewer_A_calibration.csv",
        output_root / "reviewer_packet/reviewer_B_calibration.csv",
        output_root / "reviewer_packet/reviewer_A_main.csv",
        output_root / "reviewer_packet/reviewer_B_main.csv",
    )
    existing = [str(path) for path in protected_outputs if path.exists()]
    if existing:
        raise RuntimeError(
            "Refusing to overwrite an existing Phase 61D2 packet or reviewer files: "
            + ", ".join(existing)
        )

    legacy_path = Path(args.legacy_sealed)
    legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
    calibration, independent = split_legacy_records(legacy["records"], args.calibration_budget)
    _train, validation, _status = read_shadow_yes_rows(args.manifest, args.metadata, args.selection)
    used_names = {row["name"] for row in legacy["records"]}
    confirmation, confirmation_candidates_n = select_confirmation(
        validation, used_names, args.phase54_cache, args.confirmation_budget
    )

    manifest_rows = {row["name"]: row for row in read_csv(args.manifest)}
    for cohort, records in (("calibration", calibration), ("independent", independent), ("confirmation", confirmation)):
        for row in records:
            if row["name"] not in manifest_rows:
                raise RuntimeError(f"Review image absent from protocol manifest: {row['name']}")
            row.update({key: manifest_rows[row["name"]][key] for key in ("x", "y")})
            row["cohort"] = cohort

    calibration = assign_ids(calibration, "P61D2-CAL", "phase61d2-calibration-order")
    main_records = assign_ids(independent + confirmation, "P61D2-EVAL", "phase61d2-main-order")
    panel_root = output_root / "reviewer_packet" / "panels"
    sealed_records = []
    for index, row in enumerate(calibration + main_records, 1):
        values, mtl, raster_metadata = read_all_bands(args.raw_root, row)
        if not raster_metadata["north_up"]:
            raise RuntimeError(f"Solar-direction diagram requires north-up raster: {row['scene']}")
        render_dossier(
            row["tile_id"], row, values, mtl, raster_metadata,
            panel_root / f"{row['tile_id']}.png",
        )
        sealed_records.append({
            **row,
            "sun_azimuth_degrees": float(mtl["SUN_AZIMUTH"]),
            "sun_elevation_degrees": float(mtl["SUN_ELEVATION"]),
            "date_acquired": mtl.get("DATE_ACQUIRED"),
            **raster_metadata,
        })
        if index == 1 or index % 25 == 0 or index == len(calibration) + len(main_records):
            print(f"61D2 panel rendering: {index}/{len(calibration) + len(main_records)}", flush=True)

    manual_path = Path(args.manual)
    reviewer_root = output_root / "reviewer_packet"
    reviewer_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(manual_path, reviewer_root / "PHASE61D2_REVIEW_MANUAL.md")
    (reviewer_root / "README.md").write_text(
        "# Phase 61D2 reviewer packet\n\n"
        "Read `PHASE61D2_REVIEW_MANUAL.md` first. Complete the 40-unit calibration files, "
        "hold the joint calibration discussion, and lock the protocol before either reviewer "
        "starts the 210-unit main file. Reviewers must work independently on the main file. "
        "Do not request or open `sealed_manifest.json`. Blank cells are forbidden; use "
        "`unobservable_nodata` when every displayed band lacks usable information.\n",
        encoding="utf-8",
    )
    for reviewer in ("reviewer_A", "reviewer_B"):
        write_blank_review(reviewer_root / f"{reviewer}_calibration.csv", calibration)
        write_blank_review(reviewer_root / f"{reviewer}_main.csv", main_records)
    with (reviewer_root / "calibration_consensus.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("tile_id", "agreed_label_set", "rule_or_counterexample"))
        writer.writeheader()
        for row in calibration:
            writer.writerow({"tile_id": row["tile_id"], "agreed_label_set": "", "rule_or_counterexample": ""})

    sealed_path = output_root / "sealed_manifest.json"
    sealed_document = {
        "phase": "61D2",
        "status": "awaiting_calibration_and_human_review",
        "blinding": "Reviewers see all spectral evidence and metadata, but no original label, model output, method, error stratum, or cohort identity.",
        "calibration_units_excluded_from_statistics": len(calibration),
        "main_units": len(main_records),
        "independent_units": len(independent),
        "confirmation_units": len(confirmation),
        "confirmation_candidate_units": confirmation_candidates_n,
        "confirmation_selection": "never-reviewed images; balanced original Thin/Shadow; no model error or confidence conditioning",
        "atomic_labels": list(ATOMIC_LABELS),
        "set_label_policy": "one atomic label, or a pipe-separated set of two semantic labels",
        "legacy_sealed_sha256": sha256(legacy_path),
        "protocol_manifest_sha256": sha256(args.manifest),
        "manual_sha256": sha256(manual_path),
        "records": sealed_records,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    sealed_path.write_text(json.dumps(sealed_document, indent=2), encoding="utf-8")
    packet_summary = {
        "phase": "61D2-packet",
        "status": "awaiting_calibration_lock",
        "human_results_available": False,
        "calibration_units_excluded": len(calibration),
        "main_units": len(main_records),
        "independent_units": len(independent),
        "confirmation_units": len(confirmation),
        "all_11_bands_rendered": True,
        "recommended_composites": [title for title, _ in COMPOSITES],
        "solar_metadata_rendered": True,
        "reviewer_files_blank": True,
        "sealed_manifest_sha256": sha256(sealed_path),
        "manual_sha256": sha256(manual_path),
        "cohort_counts": dict(Counter(row["cohort"] for row in sealed_records)),
        "biome_counts_main": dict(Counter(row["biome"] for row in main_records)),
    }
    (output_root / "packet_summary.json").write_text(json.dumps(packet_summary, indent=2), encoding="utf-8")
    print(json.dumps(packet_summary, indent=2))


if __name__ == "__main__":
    main()
