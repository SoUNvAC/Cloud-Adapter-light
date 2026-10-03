"""Build the Phase 63 semantic-coverage exact-pixel UI pilot.

Original labels are used only inside this sealed sampler to guarantee semantic
coverage.  They are never displayed to reviewers and are not reference truth.
The sampler fails closed when any quota, boundary definition, scene rule, or
patch rule cannot be met.
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

import numpy as np
from PIL import Image

try:
    from scipy.ndimage import distance_transform_edt as scipy_distance_transform_edt
except ImportError:  # Local lightweight test environments use the exact bounded fallback.
    scipy_distance_transform_edt = None

from phase56_protocol import sha256
from prepare_phase61d2_calibrated_review import assign_ids, read_all_bands, stable_value
from prepare_phase63_pixel_locator_pilot import (
    render_pixel_dossier,
    write_hash_manifest,
    write_option_tables,
    write_review_csv,
    zip_packet,
)


EXPECTED_MANIFEST_SHA256 = "885c2d7f61cae23409a7b1ccbe65739c302fc1e59b089aab0cc05badf674872b"
EXPECTED_METADATA_CANONICAL_SHA256 = "31370f85fbac477b6f5fccf5a046191f44ccdb45acd0a12d65270f5811d0c6b7"
LABELS = {0: "clear", 1: "thick_cloud", 2: "thin_cloud", 3: "cloud_shadow"}
INTERIOR_QUOTAS = {
    "interior_clear": 4,
    "interior_thin_cloud": 4,
    "interior_thick_cloud": 4,
    "interior_cloud_shadow": 4,
}
BOUNDARY_QUOTAS = {
    "boundary_thin_cloud__thick_cloud": 2,
    "boundary_cloud__cloud_shadow": 1,
    "boundary_clear__cloud_shadow": 1,
}
QUOTAS = {**INTERIOR_QUOTAS, **BOUNDARY_QUOTAS}
BOUNDARY_PAIRS = {
    "boundary_thin_cloud__thick_cloud": ((2,), (1,)),
    "boundary_cloud__cloud_shadow": ((1, 2), (3,)),
    "boundary_clear__cloud_shadow": ((0,), (3,)),
}
PILOT_UNITS = 20
MIN_INTERIOR_DISTANCE = 4.0
PREFERRED_INTERIOR_DISTANCE = 5.0
MAX_PAIR_BOUNDARY_DISTANCE = 1.0
MIN_RGB_VALID_CONTEXT_FRACTION = 0.50
MARGIN = 96


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def verify_hash(path: str | Path, expected: str, label: str) -> str:
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"Frozen {label} hash mismatch: expected={expected}, actual={actual}")
    return actual


def canonical_csv_sha256(path: str | Path) -> str:
    """Hash CSV records independent of UTF-8 BOM and platform line endings."""
    rows = read_csv(path)
    payload = json.dumps(
        sorted(rows, key=lambda row: tuple(sorted(row.items()))),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def shift(array: np.ndarray, dy: int, dx: int, fill: int = -999) -> np.ndarray:
    result = np.full(array.shape, fill, dtype=array.dtype)
    src_y0, src_y1 = max(0, -dy), min(array.shape[0], array.shape[0] - dy)
    src_x0, src_x1 = max(0, -dx), min(array.shape[1], array.shape[1] - dx)
    dst_y0, dst_y1 = max(0, dy), min(array.shape[0], array.shape[0] + dy)
    dst_x0, dst_x1 = max(0, dx), min(array.shape[1], array.shape[1] + dx)
    result[dst_y0:dst_y1, dst_x0:dst_x1] = array[src_y0:src_y1, src_x0:src_x1]
    return result


def any_label_boundary(mask: np.ndarray) -> np.ndarray:
    valid = np.isin(mask, tuple(LABELS))
    boundary = np.zeros(mask.shape, dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            other = shift(mask, dy, dx)
            boundary |= valid & np.isin(other, tuple(LABELS)) & (mask != other)
    return boundary


def pair_boundary_seed(mask: np.ndarray, first: tuple[int, ...], second: tuple[int, ...]) -> np.ndarray:
    seed = np.zeros(mask.shape, dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            other = shift(mask, dy, dx)
            seed |= (np.isin(mask, first) & np.isin(other, second)) | (
                np.isin(mask, second) & np.isin(other, first)
            )
    return seed


def within_euclidean_distance(seed: np.ndarray, radius: float, inclusive: bool) -> np.ndarray:
    """Exact bounded Euclidean dilation without an optional SciPy dependency."""
    result = np.zeros(seed.shape, dtype=bool)
    source = seed.astype(np.int8, copy=False)
    limit = int(np.ceil(radius))
    for dy in range(-limit, limit + 1):
        for dx in range(-limit, limit + 1):
            distance = float(np.hypot(dy, dx))
            if (distance <= radius) if inclusive else (distance < radius):
                result |= shift(source, dy, dx, fill=0).astype(bool)
    return result


def exact_distance_to_seed(seed: np.ndarray, y: int, x: int) -> float | None:
    coordinates = np.argwhere(seed)
    if len(coordinates) == 0:
        return None
    squared = np.square(coordinates[:, 0] - y) + np.square(coordinates[:, 1] - x)
    return float(np.sqrt(np.min(squared)))


def deterministic_coordinate(coordinates: np.ndarray, salt: str) -> tuple[int, int]:
    if len(coordinates) == 0:
        raise RuntimeError(f"No coordinates for {salt}")
    index = stable_value(salt) % len(coordinates)
    y, x = coordinates[index]
    return int(y), int(x)


def extract_patch_candidates(
    row: dict,
    mask: np.ndarray,
    mask_path: Path,
    mask_sha256: str | None = None,
    observability_mask: np.ndarray | None = None,
    rgb_valid_context_fraction: np.ndarray | None = None,
) -> list[dict]:
    if mask.shape != (512, 512):
        raise RuntimeError(f"Expected 512x512 source-label window: {mask_path}")
    values = set(np.unique(mask).tolist())
    if not values.issubset(set(LABELS) | {255}):
        raise RuntimeError(f"Illegal source-label values {sorted(values)}: {mask_path}")
    allowed = np.zeros(mask.shape, dtype=bool)
    allowed[MARGIN:512 - MARGIN, MARGIN:512 - MARGIN] = True
    if observability_mask is not None:
        if observability_mask.shape != mask.shape:
            raise RuntimeError("Observability mask shape mismatch")
        allowed &= observability_mask
    boundary = any_label_boundary(mask)
    if scipy_distance_transform_edt is None:
        within_3 = within_euclidean_distance(boundary, 3.0, inclusive=True)
        within_lt_5 = within_euclidean_distance(boundary, 5.0, inclusive=False)
        boundary_distance = None
    else:
        boundary_distance = scipy_distance_transform_edt(~boundary) if np.any(boundary) else None
        within_3 = np.zeros(mask.shape, dtype=bool) if boundary_distance is None else boundary_distance <= 3.0
        within_lt_5 = np.zeros(mask.shape, dtype=bool) if boundary_distance is None else boundary_distance < 5.0
    candidates: list[dict] = []
    mask_digest = mask_sha256 or sha256(mask_path)

    for label_id, label_name in LABELS.items():
        stratum = f"interior_{label_name}"
        valid = allowed & (mask == label_id) & ~within_3
        preferred = valid & ~within_lt_5
        pool = np.argwhere(preferred if np.any(preferred) else valid)
        if len(pool) == 0:
            continue
        y, x = deterministic_coordinate(pool, f"phase63-semantic-ui:{stratum}:{row['name']}")
        distance = (
            None if boundary_distance is None else float(boundary_distance[y, x])
        )
        if scipy_distance_transform_edt is None:
            distance = exact_distance_to_seed(boundary, y, x)
        if distance is not None and distance <= 3.0:
            raise RuntimeError("Internal interior-distance computation failure")
        candidates.append(
            {
                **row,
                "center_y": y,
                "center_x": x,
                "sampling_kind": "interior",
                "sampling_stratum": stratum,
                "nominal_label_id": label_id,
                "nominal_label_name": label_name,
                "distance_to_any_label_boundary_px": distance,
                "pair_boundary_distance_px": None,
                "boundary_side_labels": [],
                "interior_preferred_ge_5px": bool(distance is None or distance >= 5.0),
                "nominal_mask_path": str(mask_path),
                "nominal_mask_sha256": mask_digest,
                "rgb_valid_context_fraction": (
                    None if rgb_valid_context_fraction is None
                    else float(rgb_valid_context_fraction[y, x])
                ),
            }
        )

    for stratum, (first, second) in BOUNDARY_PAIRS.items():
        seed = pair_boundary_seed(mask, first, second)
        if not np.any(seed):
            continue
        # Choose an actual interface-side pixel (distance 0), a stricter subset
        # of the pre-registered <=1 px band.
        valid = allowed & seed
        pool = np.argwhere(valid)
        if len(pool) == 0:
            continue
        y, x = deterministic_coordinate(pool, f"phase63-semantic-ui:{stratum}:{row['name']}")
        pair_distance = exact_distance_to_seed(seed, y, x)
        any_distance = exact_distance_to_seed(boundary, y, x)
        if pair_distance is None or pair_distance > MAX_PAIR_BOUNDARY_DISTANCE:
            raise RuntimeError("Internal pair-boundary distance computation failure")
        neighbor_values = {int(mask[y, x])}
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                yy, xx = y + dy, x + dx
                if 0 <= yy < 512 and 0 <= xx < 512:
                    value = int(mask[yy, xx])
                    if value in set(first) | set(second):
                        neighbor_values.add(value)
        candidates.append(
            {
                **row,
                "center_y": y,
                "center_x": x,
                "sampling_kind": "boundary",
                "sampling_stratum": stratum,
                "nominal_label_id": int(mask[y, x]),
                "nominal_label_name": LABELS[int(mask[y, x])],
                "distance_to_any_label_boundary_px": any_distance,
                "pair_boundary_distance_px": pair_distance,
                "boundary_side_labels": sorted(LABELS[value] for value in neighbor_values),
                "interior_preferred_ge_5px": None,
                "nominal_mask_path": str(mask_path),
                "nominal_mask_sha256": mask_digest,
                "rgb_valid_context_fraction": (
                    None if rgb_valid_context_fraction is None
                    else float(rgb_valid_context_fraction[y, x])
                ),
            }
        )
    return candidates


def _candidate_order(candidate: dict, stratum: str) -> int:
    return stable_value(
        f"phase63-semantic-ui-assignment:{stratum}:{candidate['scene']}:{candidate['name']}:"
        f"{candidate['center_y']}:{candidate['center_x']}"
    )


def solve_assignment(candidates: list[dict]) -> tuple[list[dict], dict]:
    by_stratum: dict[str, list[dict]] = defaultdict(list)
    for candidate in candidates:
        by_stratum[candidate["sampling_stratum"]].append(candidate)
    missing = {
        stratum: {"needed": quota, "candidate_patches": len(by_stratum.get(stratum, []))}
        for stratum, quota in QUOTAS.items()
        if len(by_stratum.get(stratum, [])) < quota
    }
    if missing:
        raise RuntimeError(f"Insufficient pre-registered stratum candidates: {missing}")
    for stratum in by_stratum:
        by_stratum[stratum].sort(key=lambda item: _candidate_order(item, stratum))

    available_scenes = len({candidate["scene"] for candidate in candidates})
    if available_scenes < PILOT_UNITS // 2:
        raise RuntimeError(
            "No assignment can satisfy 20 units with <=2 points/scene: "
            f"only {available_scenes} candidate scenes"
        )

    slots = []
    for stratum, quota in sorted(QUOTAS.items(), key=lambda item: len(by_stratum[item[0]])):
        slots.extend([stratum] * quota)
    node_limit = 1_000_000
    scenes_by_stratum = {
        stratum: sorted(
            {item["scene"] for item in items},
            key=lambda scene: stable_value(f"phase63-semantic-ui-scene:{stratum}:{scene}"),
        )
        for stratum, items in by_stratum.items()
    }

    def attempt(max_per_scene: int, minimum_distinct_scenes: int):
        nodes = 0
        selected_scene_slots: list[tuple[str, str]] = []
        scene_counts: Counter = Counter()
        interior_scenes: dict[str, set[str]] = defaultdict(set)

        def assign_unique_patches() -> list[dict] | None:
            ordered_slots = sorted(
                enumerate(selected_scene_slots),
                key=lambda item: len(
                    [
                        row for row in by_stratum[item[1][0]]
                        if row["scene"] == item[1][1]
                    ]
                ),
            )
            assigned: dict[int, dict] = {}
            used_patches: set[str] = set()

            def match(index: int) -> bool:
                if index == len(ordered_slots):
                    return True
                original_index, (stratum, scene) = ordered_slots[index]
                for candidate in by_stratum[stratum]:
                    if candidate["scene"] != scene or candidate["name"] in used_patches:
                        continue
                    assigned[original_index] = candidate
                    used_patches.add(candidate["name"])
                    if match(index + 1):
                        return True
                    used_patches.remove(candidate["name"])
                    del assigned[original_index]
                return False

            if not match(0):
                return None
            return [assigned[index] for index in range(len(selected_scene_slots))]

        def recurse(index: int) -> list[dict] | None:
            nonlocal nodes
            nodes += 1
            if nodes > node_limit:
                raise RuntimeError("Assignment search exceeded deterministic node limit")
            remaining = len(slots) - index
            if len(scene_counts) + remaining < minimum_distinct_scenes:
                return None
            remaining_strata = slots[index:]
            possible_new_scenes = {
                scene
                for stratum in remaining_strata
                for scene in scenes_by_stratum[stratum]
                if scene not in scene_counts
            }
            if len(scene_counts) + min(remaining, len(possible_new_scenes)) < minimum_distinct_scenes:
                return None
            for interior_stratum in INTERIOR_QUOTAS:
                still_available = remaining_strata.count(interior_stratum)
                possible_class_scenes = {
                    scene
                    for scene in scenes_by_stratum[interior_stratum]
                    if scene not in interior_scenes[interior_stratum]
                }
                if len(interior_scenes[interior_stratum]) + min(
                    still_available, len(possible_class_scenes)
                ) < 3:
                    return None
            if index == len(slots):
                if len(scene_counts) < minimum_distinct_scenes:
                    return None
                if not all(len(interior_scenes[stratum]) >= 3 for stratum in INTERIOR_QUOTAS):
                    return None
                return assign_unique_patches()
            stratum = slots[index]
            ordered = sorted(
                scenes_by_stratum[stratum],
                key=lambda scene: (
                    scene_counts[scene] > 0,
                    scene_counts[scene],
                    stable_value(f"phase63-semantic-ui-scene:{stratum}:{scene}"),
                ),
            )
            for scene in ordered:
                if scene_counts[scene] >= max_per_scene:
                    continue
                selected_scene_slots.append((stratum, scene))
                scene_counts[scene] += 1
                if stratum in INTERIOR_QUOTAS:
                    interior_scenes[stratum].add(scene)
                result = recurse(index + 1)
                if result is not None:
                    return result
                if stratum in INTERIOR_QUOTAS:
                    if not any(
                        selected_stratum == stratum and selected_scene == scene
                        for selected_stratum, selected_scene in selected_scene_slots[:-1]
                    ):
                        interior_scenes[stratum].discard(scene)
                scene_counts[scene] -= 1
                if scene_counts[scene] == 0:
                    del scene_counts[scene]
                selected_scene_slots.pop()
            return None

        return recurse(0), nodes

    solution, nodes = (None, 0)
    if available_scenes >= PILOT_UNITS:
        solution, nodes = attempt(max_per_scene=1, minimum_distinct_scenes=PILOT_UNITS)
    max_per_scene = 1
    minimum_distinct = PILOT_UNITS
    if solution is None:
        max_per_scene = 2
        for minimum_distinct in range(
            min(PILOT_UNITS - 1, available_scenes), PILOT_UNITS // 2 - 1, -1
        ):
            solution, nodes = attempt(max_per_scene=2, minimum_distinct_scenes=minimum_distinct)
            if solution is not None:
                break
    if solution is None:
        raise RuntimeError("No assignment satisfies quotas, scene<=2, patch<=1, and class scene coverage")

    scene_counts = Counter(row["scene"] for row in solution)
    patch_counts = Counter(row["name"] for row in solution)
    stratum_counts = Counter(row["sampling_stratum"] for row in solution)
    if dict(stratum_counts) != QUOTAS:
        raise RuntimeError(f"Internal quota mismatch: {dict(stratum_counts)}")
    if max(scene_counts.values()) > 2 or max(patch_counts.values()) > 1:
        raise RuntimeError("Internal scene/patch constraint failure")
    class_scene_counts = {
        stratum: len({row["scene"] for row in solution if row["sampling_stratum"] == stratum})
        for stratum in INTERIOR_QUOTAS
    }
    if min(class_scene_counts.values()) < 3:
        raise RuntimeError(f"Interior class scene coverage failure: {class_scene_counts}")
    audit = {
        "units": len(solution),
        "available_candidate_scenes": available_scenes,
        "distinct_scenes": len(scene_counts),
        "max_points_per_scene": max(scene_counts.values()),
        "max_points_per_patch": max(patch_counts.values()),
        "stratum_counts": dict(sorted(stratum_counts.items())),
        "interior_class_scene_counts": dict(sorted(class_scene_counts.items())),
        "scene_counts": dict(sorted(scene_counts.items())),
        "assignment_rule": (
            "try 20 distinct scenes first; if infeasible, maximize distinct scenes with <=2 points/scene"
        ),
        "minimum_distinct_scenes_solved": minimum_distinct,
        "search_nodes_final_attempt": nodes,
    }
    return solution, audit


def read_raw_source_mask_window(dataset, row: dict) -> np.ndarray:
    from rasterio.windows import Window

    column, line = int(row["y"]), int(row["x"])
    raw = dataset.read(
        1, window=Window(column, line, 512, 512), boundless=True, fill_value=128
    ).astype(np.uint8)
    return map_raw_fixedmask(raw)


def map_raw_fixedmask(raw: np.ndarray) -> np.ndarray:
    """Map the documented L8 Biome fixedmask bytes into source class order."""
    legal = {0, 64, 128, 192, 255}
    values = set(np.unique(raw).tolist())
    if not values.issubset(legal):
        raise RuntimeError(f"Illegal raw fixedmask values: {sorted(values)}")
    source = np.full(raw.shape, 255, dtype=np.int16)
    source[raw == 128] = 0     # clear
    source[raw == 255] = 1     # thick cloud
    source[raw == 192] = 2     # thin cloud
    source[raw == 64] = 3      # cloud shadow
    # Raw 0 is Fill and remains invalid=255; it is never sampled as clear.
    return source


def read_rgb_observability_window(dataset, row: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return target-valid/context-valid masks without using semantic labels."""
    from rasterio.windows import Window

    if dataset.count != 11:
        raise RuntimeError(f"Expected 11 image bands, found {dataset.count}: {dataset.name}")
    column, line = int(row["y"]), int(row["x"])
    rgb = dataset.read(
        (2, 3, 4), window=Window(column, line, 512, 512), boundless=True, fill_value=0
    )
    valid = np.all(np.isfinite(rgb) & (rgb != 0), axis=0)
    integral = np.pad(valid.astype(np.int32), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    size = 2 * MARGIN
    counts = (
        integral[size:, size:]
        - integral[:-size, size:]
        - integral[size:, :-size]
        + integral[:-size, :-size]
    )
    context_fraction = np.zeros(valid.shape, dtype=np.float32)
    context_fraction[MARGIN:512 - MARGIN, MARGIN:512 - MARGIN] = (
        counts[: 512 - 2 * MARGIN, : 512 - 2 * MARGIN] / float(size * size)
    )
    observable = valid & (context_fraction >= MIN_RGB_VALID_CONTEXT_FRACTION)
    return observable, context_fraction


def load_candidates(
    manifest_rows: list[dict[str, str]],
    shadow_status: dict[str, str],
    raw_root: Path,
    excluded_scenes: set[str],
    excluded_patches: set[str],
) -> tuple[list[dict], dict]:
    try:
        import rasterio
    except ImportError as error:
        raise RuntimeError("Raw fixedmask sampling requires rasterio") from error
    all_candidates: list[dict] = []
    mask_files_read = []
    skipped = Counter()
    by_scene: dict[str, list[dict]] = defaultdict(list)
    for row in manifest_rows:
        if row["new_split"] != "target_train" or shadow_status.get(row["scene"]) != "yes":
            continue
        if row["scene"] in excluded_scenes:
            skipped["previously_reviewed_scene"] += 1
            continue
        if row["name"] in excluded_patches:
            skipped["previously_reviewed_patch"] += 1
            continue
        by_scene[row["scene"]].append(row)
    for scene in sorted(by_scene):
        rows = by_scene[scene]
        biome = rows[0]["biome"]
        fixedmask_path = raw_root / "l8biome" / biome / scene / f"{scene}_fixedmask.TIF"
        image_path = raw_root / "l8biome" / biome / scene / f"{scene}.TIF"
        if not fixedmask_path.is_file():
            raise RuntimeError(f"Missing authorized raw fixedmask: {fixedmask_path}")
        if not image_path.is_file():
            raise RuntimeError(f"Missing authorized raw image: {image_path}")
        mask_digest = sha256(fixedmask_path)
        with rasterio.open(fixedmask_path) as dataset, rasterio.open(image_path) as image_dataset:
            if (dataset.width, dataset.height) != (image_dataset.width, image_dataset.height):
                raise RuntimeError(f"Image/mask geometry mismatch: {scene}")
            for row in sorted(rows, key=lambda item: item["name"]):
                mask = read_raw_source_mask_window(dataset, row)
                observable, context_fraction = read_rgb_observability_window(image_dataset, row)
                if not np.any(observable):
                    skipped["unobservable_rgb_context"] += 1
                    continue
                all_candidates.extend(
                    extract_patch_candidates(
                        row,
                        mask,
                        fixedmask_path,
                        mask_digest,
                        observable,
                        context_fraction,
                    )
                )
        mask_files_read.append(str(fixedmask_path))
    return all_candidates, {
        "raw_fixedmask_scenes_read": len(mask_files_read),
        "raw_root": str(raw_root),
        "raw_value_to_source_order": {
            "0": "invalid_fill", "64": "cloud_shadow", "128": "clear",
            "192": "thin_cloud", "255": "thick_cloud",
        },
        "shadows_status_filter": "yes",
        "model_arrays_read": [],
        "observability_rule": (
            "target B2/B3/B4 all finite and nonzero; centered 192x192 RGB-valid fraction >=0.50"
        ),
        "skipped": dict(sorted(skipped.items())),
        "candidate_counts": dict(sorted(Counter(c["sampling_stratum"] for c in all_candidates).items())),
        "candidate_scene_counts": {
            stratum: len({c["scene"] for c in all_candidates if c["sampling_stratum"] == stratum})
            for stratum in QUOTAS
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="work_dirs/phase45_protocol_audit/scene_disjoint_manifest.csv")
    parser.add_argument("--metadata", default="research_plans/protocol_data/l8_biome_usgs_shadow_status.csv")
    parser.add_argument(
        "--phase63-sealed",
        default="work_dirs/phase63_selective_multigranularity/human_confirmation_review/sealed_manifest.json",
    )
    parser.add_argument(
        "--old-ui-sealed",
        default="work_dirs/phase63_selective_multigranularity/pixel_locator_ui_pilot/sealed_manifest.json",
    )
    parser.add_argument(
        "--manual", default="research_plans/PHASE63_SEMANTIC_COVERAGE_UI_PILOT_MANUAL.md"
    )
    parser.add_argument("--raw-root", required=True)
    parser.add_argument(
        "--output-root",
        default="work_dirs/phase63_selective_multigranularity/semantic_coverage_ui_pilot",
    )
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument(
        "--candidate-audit-only",
        action="store_true",
        help="Report sealed candidate availability before attempting the constrained assignment.",
    )
    args = parser.parse_args()

    manifest_hash = verify_hash(args.manifest, EXPECTED_MANIFEST_SHA256, "manifest")
    metadata_canonical_hash = canonical_csv_sha256(args.metadata)
    if metadata_canonical_hash != EXPECTED_METADATA_CANONICAL_SHA256:
        raise RuntimeError(
            "Frozen Shadows metadata canonical hash mismatch: "
            f"expected={EXPECTED_METADATA_CANONICAL_SHA256}, actual={metadata_canonical_hash}"
        )
    metadata_hash = sha256(args.metadata)
    manifest_rows = read_csv(args.manifest)
    metadata_rows = read_csv(args.metadata)
    shadow_status = {row["scene"]: row["usgs_shadows"].strip().lower() for row in metadata_rows}
    if len(shadow_status) != 96 or set(shadow_status.values()) != {"yes", "no"}:
        raise RuntimeError("Unexpected frozen Shadows? metadata")
    phase63 = json.loads(Path(args.phase63_sealed).read_text(encoding="utf-8"))
    old_ui = json.loads(Path(args.old_ui_sealed).read_text(encoding="utf-8"))
    excluded_scenes = set(phase63["selection_audit"]["scene_ids"])
    excluded_scenes.update(phase63["development_scene_ids"])
    excluded_scenes.update(row["scene"] for row in phase63["calibration_records"])
    excluded_scenes.update(row["scene"] for row in old_ui["records"])
    excluded_patches = {row["name"] for row in phase63["confirmation_records"]}
    excluded_patches.update(row["name"] for row in phase63["calibration_records"])
    excluded_patches.update(row["name"] for row in old_ui["records"])

    candidates, candidate_audit = load_candidates(
        manifest_rows,
        shadow_status,
        Path(args.raw_root),
        excluded_scenes,
        excluded_patches,
    )
    if args.candidate_audit_only:
        print(json.dumps({"status": "candidate_audit_only", **candidate_audit}, indent=2))
        return
    selected, assignment_audit = solve_assignment(candidates)
    selected = assign_ids(selected, "P63-SCUI", "phase63-semantic-coverage-ui-pilot-order")
    audit = {
        **candidate_audit,
        **assignment_audit,
        "previously_reviewed_scene_overlap": sorted(
            {row["scene"] for row in selected} & excluded_scenes
        ),
        "previously_reviewed_patch_overlap": sorted(
            {row["name"] for row in selected} & excluded_patches
        ),
        "target_test_units": sum(row["new_split"] == "target_test" for row in selected),
        "selection_uses_original_labels_as_sealed_strata": True,
        "nominal_labels_are_reference_truth": False,
        "selection_uses_model_predictions": False,
    }
    if (
        audit["previously_reviewed_scene_overlap"]
        or audit["previously_reviewed_patch_overlap"]
        or audit["target_test_units"]
    ):
        raise RuntimeError(f"Leakage audit failed: {audit}")
    if args.audit_only:
        print(json.dumps({"status": "audit_only_pass", "selection_audit": audit}, indent=2))
        return

    output_root = Path(args.output_root)
    packet_root = output_root / "phase63_semantic_coverage_ui_pilot_packet"
    output_zip = output_root / "phase63_semantic_coverage_ui_pilot_packet.zip"
    protected = (packet_root, output_zip, output_root / "sealed_manifest.json")
    if any(path.exists() for path in protected):
        raise RuntimeError("Refusing to overwrite an existing semantic-coverage UI pilot")
    panel_root = packet_root / "panels"
    panel_root.mkdir(parents=True, exist_ok=False)

    sealed_records = []
    for index, row in enumerate(selected, 1):
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
        print(f"Phase63 semantic-coverage UI pilot rendering: {index}/{len(selected)}", flush=True)

    shutil.copy2(args.manual, packet_root / "PHASE63_SEMANTIC_COVERAGE_UI_PILOT_MANUAL.md")
    write_review_csv(packet_root / "reviewer_A_ui_pilot.csv", selected)
    write_review_csv(packet_root / "reviewer_B_ui_pilot.csv", selected)
    write_option_tables(packet_root)
    (packet_root / "README.md").write_text(
        "# Phase 63 semantic-coverage exact-pixel UI pilot\n\n"
        "Original labels were used solely as sealed sampling strata to ensure semantic coverage. "
        "They are neither displayed to reviewers nor treated as reference truth. There is no intended "
        "answer: judge only the single pixel inside the hollow square. Reviewers must work independently, "
        "view each item once, record elapsed_seconds, and freeze both raw CSVs before any nominal sampling "
        "field is unsealed. This 20-item pilot is excluded from final model and confirmation statistics.\n",
        encoding="utf-8",
    )
    images = sorted(panel_root.glob("*.png"))
    if len(images) != PILOT_UNITS:
        raise RuntimeError(f"Expected {PILOT_UNITS} pilot panels, found {len(images)}")
    for path in images:
        with Image.open(path) as image:
            if image.size != (1040, 1210) or image.mode != "RGB":
                raise RuntimeError(f"Invalid panel geometry: {path}")
    write_hash_manifest(packet_root)
    zip_packet(packet_root, output_zip)

    output_root.mkdir(parents=True, exist_ok=True)
    sealed = {
        "phase": "63-semantic-coverage-exact-pixel-ui-pilot",
        "status": "awaiting_independent_review",
        "excluded_from_final_model_and_confirmation_statistics": True,
        "sampling_statement": (
            "Original labels were used solely as sealed sampling strata to ensure semantic coverage. "
            "They were neither displayed to reviewers nor treated as reference truth."
        ),
        "target_test_read": False,
        "artifact_validation": {
            "coordinates_valid": True,
            "pages_valid": True,
            "review_templates_blank": True,
            "reviewer_packet_excludes_nominal_fields": True,
        },
        "inputs": {
            "manifest_sha256": manifest_hash,
            "shadow_metadata_sha256": metadata_hash,
            "shadow_metadata_canonical_sha256": metadata_canonical_hash,
            "phase63_sealed_sha256": sha256(args.phase63_sealed),
            "old_ui_sealed_sha256": sha256(args.old_ui_sealed),
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
