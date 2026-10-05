"""Freeze Phase 64B six-band lineage, radiometry, geometry, and source stats."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


BANDS = ("red", "green", "blue", "nir", "swir1", "swir2")
S2_FILES = ("B4", "B3", "B2", "B8", "B11", "B12")
S2_COUNTS = {"train": 8490, "val": 535}


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(path: Path, allowed=("target_train", "target_val")):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["new_split"] in allowed]
    if not rows or any(row["new_split"] not in allowed for row in rows):
        raise RuntimeError(f"Development-only manifest audit failed: {path}")
    return rows


def cloudsen_files(root: Path):
    records = []
    for split, count in S2_COUNTS.items():
        expected_bytes = count * 512 * 512 * np.dtype(np.int16).itemsize
        for band in S2_FILES:
            path = root / split / f"L1C_{band}.dat"
            if not path.is_file() or path.stat().st_size != expected_bytes:
                raise RuntimeError(
                    f"CloudSEN file missing or wrong size: {path}; expected {expected_bytes}"
                )
            records.append(
                {"split": split, "band": band, "bytes": path.stat().st_size}
            )
    if (root / "test").exists():
        raise RuntimeError("CloudSEN test must not be downloaded or read in Phase64B")
    return records


def source_stats(root: Path, chunk_size: int):
    count = S2_COUNTS["train"]
    arrays = [
        np.memmap(
            root / "train" / f"L1C_{band}.dat",
            dtype=np.int16,
            mode="r",
            shape=(count, 512, 512),
        )
        for band in S2_FILES
    ]
    total = np.zeros(6, dtype=np.float64)
    total_sq = np.zeros(6, dtype=np.float64)
    pixels = 0
    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        raw = np.stack([array[start:stop] for array in arrays], axis=-1)
        valid10 = np.any(raw != 0, axis=-1)
        image = np.clip(raw.astype(np.float32) * 1.0e-4, 0.0, 1.5)
        batch = stop - start
        image20 = image.reshape(batch, 256, 2, 256, 2, 6).mean(axis=(2, 4))
        valid20 = valid10.reshape(batch, 256, 2, 256, 2).mean(axis=(2, 4)) >= 0.999
        selected = image20[valid20]
        total += selected.sum(axis=0, dtype=np.float64)
        total_sq += np.square(selected, dtype=np.float64).sum(axis=0)
        pixels += selected.shape[0]
        print(f"source_stats {stop}/{count}", flush=True)
    mean = total / pixels
    variance = np.maximum(total_sq / pixels - mean * mean, 0.0)
    return {
        "bands": list(BANDS),
        "valid_20m_pixels": int(pixels),
        "mean": mean.tolist(),
        "std": np.sqrt(variance).tolist(),
        "source": "CloudSEN official train split only",
        "applied_to": "CloudSEN val, L8 train/val, SPARCS train/val",
    }


def l8_audit(manifest: Path, raw_root: Path):
    import rasterio

    rows = read_manifest(manifest)
    seen = {}
    for row in rows:
        path = raw_root / "l8biome" / row["biome"] / row["scene"] / f"{row['scene']}.TIF"
        seen.setdefault(row["scene"], path)
    metadata = []
    for scene, path in sorted(seen.items()):
        with rasterio.open(path) as dataset:
            record = {
                "scene": scene,
                "count": dataset.count,
                "dtypes": sorted(set(dataset.dtypes)),
                "resolution": list(dataset.res),
                "crs": str(dataset.crs),
            }
        if record["count"] != 11 or record["dtypes"] != ["uint8"]:
            raise RuntimeError(f"Unexpected L8 product: {path}: {record}")
        if any(abs(value - 30.0) > 1e-6 for value in record["resolution"]):
            raise RuntimeError(f"L8 is not on its native 30 m grid: {path}")
        metadata.append(record)
    return {"patches": len(rows), "scenes": len(metadata), "scene_metadata": metadata}


def sparcs_audit(manifest: Path):
    import rasterio

    rows = read_manifest(manifest)
    metadata = []
    for row in rows:
        path = Path(row["multispectral_path"])
        mtl = path.with_name(f"{row['scene']}_mtl.txt")
        with rasterio.open(path) as dataset:
            record = {
                "name": row["name"],
                "scene": row["scene"],
                "count": dataset.count,
                "dtypes": sorted(set(dataset.dtypes)),
                "shape": list(dataset.shape),
                "resolution": list(dataset.res),
                "mtl_exists": mtl.is_file(),
            }
        if (
            record["count"] != 10
            or record["dtypes"] != ["uint16"]
            or record["shape"] != [1000, 1000]
            or any(abs(value - 30.0) > 1e-6 for value in record["resolution"])
            or not record["mtl_exists"]
        ):
            raise RuntimeError(f"Unexpected SPARCS pair: {path}: {record}")
        metadata.append(record)
    return {
        "samples": len(rows),
        "scenes": len({row["scene"] for row in rows}),
        "pair_metadata": metadata,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cloudsen-root", required=True)
    parser.add_argument("--l8-manifest", required=True)
    parser.add_argument("--l8-raw-root", required=True)
    parser.add_argument("--sparcs-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--chunk-size", type=int, default=8)
    args = parser.parse_args()
    cloudsen_root = Path(args.cloudsen_root)
    l8_manifest = Path(args.l8_manifest)
    sparcs_manifest = Path(args.sparcs_manifest)
    summary = {
        "phase": "64B-0",
        "target_test_reads": 0,
        "band_order": list(BANDS),
        "sensor_bands": {
            "sentinel2": ["B4", "B3", "B2", "B8", "B11", "B12"],
            "landsat8": ["B4", "B3", "B2", "B5", "B6", "B7"],
        },
        "radiometry": {
            "cloudsen": "L1C int16 DN * 0.0001, finite clip [0,1.5]",
            "l8_biome": "packaged uint8 / 255; not recoverable physical reflectance",
            "sparcs": "MTL multiplicative/additive coefficients then divide by sin(SUN_ELEVATION), clip [0,1.5]",
        },
        "nodata_saturation": {
            "nodata": "all-six raw zero is invalid and excluded from loss/evaluation",
            "saturation": "retained then clipped to reflectance/encoding range [0,1.5]",
        },
        "spatial": {
            "sentinel2": "legacy 10 m common grid -> 20 m by 2x2 area averaging; labels nearest-neighbor",
            "l8_biome": "native 30 m, no resampling",
            "sparcs": "native 30 m, no resampling; sliding inference",
            "physical_scale_equivalence_claimed": False,
        },
        "normalization": "statistics from CloudSEN official train split only",
        "cloudsen_files": cloudsen_files(cloudsen_root),
        "source_train_stats": source_stats(cloudsen_root, args.chunk_size),
        "l8": l8_audit(l8_manifest, Path(args.l8_raw_root)),
        "sparcs": sparcs_audit(sparcs_manifest),
        "lineage": {
            "l8_manifest_sha256": hash_file(l8_manifest),
            "sparcs_manifest_sha256": hash_file(sparcs_manifest),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("phase", "target_test_reads", "source_train_stats", "lineage")}, indent=2))


if __name__ == "__main__":
    main()
