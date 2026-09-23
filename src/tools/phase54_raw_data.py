import argparse
import csv
import math
import re
from pathlib import Path

import numpy as np
from PIL import Image


MTL_PATTERN = re.compile(r"^\s*([A-Z0-9_]+)\s*=\s*([^\s]+)\s*$")


def scene_directory(raw_root, row):
    return Path(raw_root) / "l8biome" / row["biome"] / row["scene"]


def parse_mtl(path):
    values = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        match = MTL_PATTERN.match(line)
        if match:
            values[match.group(1)] = match.group(2).strip('"')
    return values


def solar_encoding(mtl):
    elevation = math.radians(float(mtl["SUN_ELEVATION"]))
    azimuth = math.radians(float(mtl["SUN_AZIMUTH"]))
    return np.asarray(
        [math.sin(elevation), math.cos(elevation),
         math.sin(azimuth), math.cos(azimuth)],
        dtype=np.float32,
    )


def read_multispectral_patch(raw_root, row, swap_xy=True):
    try:
        import rasterio
        from rasterio.windows import Window
    except ImportError as error:
        raise RuntimeError("Phase 54 requires rasterio==1.3.11") from error
    directory = scene_directory(raw_root, row)
    raster_path = directory / f"{row['scene']}.TIF"
    mtl_path = directory / f"{row['scene']}_MTL.txt"
    x, y = int(row["x"]), int(row["y"])
    column, line = (y, x) if swap_xy else (x, y)
    with rasterio.open(raster_path) as dataset:
        # Landsat B4/B3/B2 are read only for alignment verification. The audit
        # auxiliaries are B5 NIR and B6/B7 SWIR1/SWIR2.
        values = dataset.read(
            (4, 3, 2, 5, 6, 7),
            window=Window(column, line, 512, 512),
            boundless=True,
            fill_value=0,
        ).transpose(1, 2, 0)
    if values.shape != (512, 512, 6):
        raise RuntimeError(f"Unexpected patch shape {values.shape}: {raster_path}")
    # TorchGeo's frozen L8Biome artifact is an 11-band uint8 product. Applying
    # the original DN MTL coefficients to this derived product would be wrong;
    # optical channels therefore retain its documented 0..255 encoding.
    return values.astype(np.float32) / 255.0, solar_encoding(parse_mtl(mtl_path))


def reconstruct_converter_rgb(multispectral):
    rgb = multispectral[..., :3]
    low, high = float(rgb.min()), float(rgb.max())
    return ((rgb - low) / (high - low + 1e-6) * 255).astype(np.uint8)


def load_manifest(path, split=None):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if split is not None:
        rows = [row for row in rows if row["new_split"] == split]
    return rows


def verify_alignment(args):
    rows = [row for row in load_manifest(args.manifest) if row["biome"] == args.biome]
    rows = rows[: args.samples]
    if not rows:
        raise RuntimeError(f"No {args.biome} rows in manifest")
    errors = {"x_is_column": [], "x_is_line": []}
    for row in rows:
        expected = np.asarray(Image.open(row["image_path"]), dtype=np.int16)
        for name, swap in (("x_is_column", False), ("x_is_line", True)):
            raw, _solar = read_multispectral_patch(args.raw_root, row, swap_xy=swap)
            rebuilt = reconstruct_converter_rgb(raw).astype(np.int16)
            errors[name].append(float(np.abs(rebuilt - expected).mean()))
    result = {key: float(np.mean(value)) for key, value in errors.items()}
    print(result)
    winner = min(result, key=result.get)
    if result[winner] > args.max_mean_abs_error:
        raise RuntimeError(f"Neither coordinate convention aligns: {result}")
    print(f"alignment={winner}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", required=True)
    parser.add_argument(
        "--manifest",
        default="work_dirs/phase45_protocol_audit/scene_disjoint_manifest.csv",
    )
    parser.add_argument("--biome", default="barren")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--max-mean-abs-error", type=float, default=1.0)
    verify_alignment(parser.parse_args())


if __name__ == "__main__":
    main()
