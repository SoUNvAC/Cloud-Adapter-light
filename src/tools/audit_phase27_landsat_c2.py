import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from phase27_landsat_protocol import ALLOWED_TRUTH_VALUES


REQUIRED_RASTERS = ("b2", "b3", "b4", "truth", "qa_pixel")
REQUIRED_METADATA_KEYS = (
    "REFLECTANCE_MULT_BAND_2",
    "REFLECTANCE_ADD_BAND_2",
    "REFLECTANCE_MULT_BAND_3",
    "REFLECTANCE_ADD_BAND_3",
    "REFLECTANCE_MULT_BAND_4",
    "REFLECTANCE_ADD_BAND_4",
    "SUN_ELEVATION",
)
OFFICIAL_ITEM_ID = "61015b2fd34ef8d7055d6395"
OFFICIAL_DOI = "10.5066/P9FI4A0Y"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit the official USGS Landsat 8 C2 cloud-truth dataset."
    )
    parser.add_argument("--root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-scenes", type=int, default=48)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def sha256_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_mtl(path: Path):
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", maxsplit=1)
        values[key.strip()] = value.strip().strip('"')
    missing = [key for key in REQUIRED_METADATA_KEYS if key not in values]
    if missing:
        raise RuntimeError(f"{path} is missing MTL fields: {missing}")
    parsed = {key: float(values[key]) for key in REQUIRED_METADATA_KEYS}
    if not (0.0 < parsed["SUN_ELEVATION"] <= 90.0):
        raise RuntimeError(f"Invalid SUN_ELEVATION in {path}")
    return parsed


def resolve_file(root: Path, relative, label):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} must be a safe path relative to dataset root")
    resolved_root = root.resolve()
    path = (root / relative).resolve()
    if path != resolved_root and resolved_root not in path.parents:
        raise ValueError(f"{label} escapes dataset root")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def raster_signature(dataset):
    return {
        "width": dataset.width,
        "height": dataset.height,
        "count": dataset.count,
        "crs": dataset.crs.to_string() if dataset.crs else None,
        "transform": list(dataset.transform)[:6],
    }


def truth_histogram(dataset):
    histogram = {value: 0 for value in ALLOWED_TRUTH_VALUES}
    unexpected = set()
    for _, window in dataset.block_windows(1):
        values, counts = np.unique(dataset.read(1, window=window), return_counts=True)
        for value, count in zip(values.tolist(), counts.tolist()):
            value = int(value)
            if value in histogram:
                histogram[value] += int(count)
            else:
                unexpected.add(value)
    return histogram, sorted(unexpected)


def main():
    args = parse_args()
    if args.expected_scenes <= 0:
        raise ValueError("expected-scenes must be positive")
    try:
        import rasterio
    except ImportError as error:
        raise RuntimeError("Install rasterio==1.3.10 for Phase 27") from error

    root = Path(args.root)
    manifest_path = Path(args.manifest)
    if not root.is_dir():
        raise FileNotFoundError(root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scenes = manifest.get("scenes", [])
    if len(scenes) != args.expected_scenes:
        raise RuntimeError(
            f"Expected {args.expected_scenes} manifest scenes, found {len(scenes)}"
        )
    scene_ids = [str(scene.get("scene_id", "")) for scene in scenes]
    if any(not value for value in scene_ids) or len(set(scene_ids)) != len(scene_ids):
        raise RuntimeError("Manifest scene_id values must be nonempty and unique")

    audited = []
    aggregate_truth = {value: 0 for value in ALLOWED_TRUTH_VALUES}
    for index, scene in enumerate(scenes, start=1):
        scene_id = str(scene["scene_id"])
        missing_fields = [key for key in (*REQUIRED_RASTERS, "mtl") if key not in scene]
        if missing_fields:
            raise RuntimeError(f"Scene {scene_id} is missing fields: {missing_fields}")
        paths = {
            key: resolve_file(root, scene[key], f"{scene_id}.{key}")
            for key in REQUIRED_RASTERS
        }
        paths["mtl"] = resolve_file(root, scene["mtl"], f"{scene_id}.mtl")
        signatures = {}
        with rasterio.open(paths["truth"]) as truth_dataset:
            reference_signature = raster_signature(truth_dataset)
            if truth_dataset.count != 1:
                raise RuntimeError(f"Scene {scene_id} truth must have one band")
            histogram, unexpected = truth_histogram(truth_dataset)
        if unexpected:
            raise RuntimeError(
                f"Scene {scene_id} has unexpected truth labels: {unexpected}"
            )
        if sum(histogram[value] for value in (128, 192, 255)) == 0:
            raise RuntimeError(f"Scene {scene_id} contains no scored truth pixels")
        for key in REQUIRED_RASTERS:
            with rasterio.open(paths[key]) as dataset:
                signature = raster_signature(dataset)
                signatures[key] = signature
                if dataset.count != 1:
                    raise RuntimeError(f"Scene {scene_id} {key} must have one band")
                if signature != reference_signature:
                    raise RuntimeError(
                        f"Scene {scene_id} {key} is not pixel-aligned with truth"
                    )
        metadata = parse_mtl(paths["mtl"])
        hashes = {key: sha256_file(path) for key, path in paths.items()}
        for value, count in histogram.items():
            aggregate_truth[value] += count
        audited.append(
            {
                "scene_id": scene_id,
                "paths": {key: Path(scene[key]).as_posix() for key in paths},
                "sha256": hashes,
                "raster": reference_signature,
                "truth_histogram": {str(key): value for key, value in histogram.items()},
                "toa_reflectance_metadata": metadata,
            }
        )
        print(f"Audited {index}/{len(scenes)}: {scene_id}")

    content_gates = {
        "official_sciencebase_item": manifest.get("sciencebase_item_id")
        == OFFICIAL_ITEM_ID,
        "official_doi": OFFICIAL_DOI.lower()
        in str(manifest.get("source", "")).lower(),
        "cc0_license": str(manifest.get("license", "")).upper()
        in ("CC0", "CC0-1.0", "CC0 1.0"),
        "scene_count": len(scenes) == args.expected_scenes,
        "clear_pixels_present": aggregate_truth[128] > 0,
        "thin_cloud_pixels_present": aggregate_truth[192] > 0,
        "opaque_cloud_pixels_present": aggregate_truth[255] > 0,
        "all_rasters_pixel_aligned": True,
        "all_truth_labels_valid": True,
        "all_files_sha256_hashed": True,
    }
    result = {
        "phase": 27,
        "dataset": manifest.get("dataset"),
        "source": manifest.get("source"),
        "sciencebase_item_id": manifest.get("sciencebase_item_id"),
        "license": manifest.get("license"),
        "expected_scenes": args.expected_scenes,
        "scene_count": len(scenes),
        "truth_histogram": {
            str(key): value for key, value in aggregate_truth.items()
        },
        "manifest_sha256": sha256_file(manifest_path),
        "scenes": audited,
        "gates": content_gates,
        "passed": all(content_gates.values()),
        "test_evaluated": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("scene_count", "truth_histogram", "gates", "passed")}, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
