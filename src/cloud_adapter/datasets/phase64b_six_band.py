"""Audited common-six-band datasets and transforms for Phase 64B.

The only model input is ordered as R/G/B/NIR/SWIR1/SWIR2.  Sentinel-2 and
Landsat retain sensor-specific radiometric conversions and native physical
pixel sizes; no claim of equal ground sampling distance is implied.
"""

from __future__ import annotations

import csv
import math
import re
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from mmcv.transforms import BaseTransform
from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS, TRANSFORMS

from .phase64_parent import (
    CLOUDSEN_TO_PARENT,
    PARENT_CLASSES,
    PARENT_PALETTE,
    Phase64L8ParentDataset,
)
from .sparcs_manifest import Phase64SparcsManifestDataset


COMMON_BAND_ORDER = ("red", "green", "blue", "nir", "swir1", "swir2")
S2_FILES = ("B4", "B3", "B2", "B8", "B11", "B12")
LANDSAT_BAND_INDEXES = (4, 3, 2, 5, 6, 7)
S2_SPLIT_SIZES = {"train": 8490, "val": 535}
_MTL_PATTERN = re.compile(r"^\s*([A-Z0-9_]+)\s*=\s*([^\s]+)\s*$")


def _assert_development_split(split: str) -> None:
    if split not in {"train", "val", "target_train", "target_val"}:
        raise ValueError(
            f"Phase64B refuses non-development split {split!r}; target-test is sealed"
        )


def _manifest_rows(path: Path, split: str) -> list[dict[str, str]]:
    _assert_development_split(split)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["new_split"] == split]
    if not rows:
        raise RuntimeError(f"No {split!r} records in {path}")
    return rows


@lru_cache(maxsize=32)
def _memmap(path: str, count: int) -> np.memmap:
    return np.memmap(
        path,
        dtype=np.int16,
        mode="r",
        shape=(count, 512, 512),
    )


@lru_cache(maxsize=256)
def _read_mtl(path: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        match = _MTL_PATTERN.match(line)
        if match:
            values[match.group(1)] = match.group(2).strip('"')
    return values


def _landsat_toa_reflectance(
    values: np.ndarray, mtl_path: str
) -> np.ndarray:
    """Convert Landsat Level-1 DN to sun-angle-corrected TOA reflectance."""

    metadata = _read_mtl(mtl_path)
    sine_elevation = math.sin(math.radians(float(metadata["SUN_ELEVATION"])))
    if sine_elevation <= 0:
        raise ValueError(f"Invalid SUN_ELEVATION in {mtl_path}")
    converted = np.empty(values.shape, dtype=np.float32)
    for channel, band in enumerate(LANDSAT_BAND_INDEXES):
        multiplier = float(metadata[f"REFLECTANCE_MULT_BAND_{band}"])
        offset = float(metadata[f"REFLECTANCE_ADD_BAND_{band}"])
        converted[..., channel] = (
            values[..., channel].astype(np.float32) * multiplier + offset
        ) / sine_elevation
    return converted


@DATASETS.register_module()
class Phase64BCloudSENSixBandParentDataset(BaseSegDataset):
    """CloudSEN L1C memmaps paired with the frozen official train/val labels."""

    METAINFO = dict(classes=PARENT_CLASSES, palette=PARENT_PALETTE)

    def __init__(
        self,
        memmap_root: str,
        label_root: str,
        split: str,
        **kwargs,
    ) -> None:
        _assert_development_split(split)
        if split not in S2_SPLIT_SIZES:
            raise ValueError("CloudSEN Phase64B permits only official train or val")
        self.memmap_root = Path(memmap_root)
        self.label_root = Path(label_root)
        self.phase64b_split = split
        super().__init__(
            img_suffix=".dat",
            seg_map_suffix=".png",
            reduce_zero_label=False,
            data_root="",
            data_prefix=dict(img_path="", seg_map_path=""),
            **kwargs,
        )

    def load_data_list(self):
        count = S2_SPLIT_SIZES[self.phase64b_split]
        split_root = self.memmap_root / self.phase64b_split
        band_paths = [split_root / f"L1C_{band}.dat" for band in S2_FILES]
        missing = [str(path) for path in band_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing CloudSEN six-band memmaps: " + ", ".join(missing))
        labels = self.label_root / "ann_dir" / self.phase64b_split
        data_list = []
        for index in range(count):
            label_path = labels / f"{index}.png"
            if not label_path.is_file():
                raise FileNotFoundError(label_path)
            data_list.append(
                dict(
                    img_path=str(split_root / f"sample_{index}.six-band"),
                    seg_map_path=str(label_path),
                    phase64b_source="cloudsen_l1c",
                    phase64b_band_paths=[str(path) for path in band_paths],
                    phase64b_sample_index=index,
                    phase64b_sample_count=count,
                    label_map=dict(CLOUDSEN_TO_PARENT),
                    reduce_zero_label=False,
                    seg_fields=[],
                )
            )
        return data_list


@DATASETS.register_module()
class Phase64BL8SixBandParentDataset(Phase64L8ParentDataset):
    """Read the audited L8 Biome patches from the 30 m 11-band product."""

    def __init__(self, raw_root: str, **kwargs) -> None:
        self.phase64b_raw_root = Path(raw_root)
        super().__init__(**kwargs)

    def load_data_list(self):
        items = super().load_data_list()
        rows = _manifest_rows(self.manifest_path, self.manifest_split)
        by_name = {row["name"]: row for row in rows}
        for item in items:
            name = Path(item["img_path"]).name
            row = by_name.get(name)
            if row is None:
                raise RuntimeError(f"L8 manifest metadata missing for {name}")
            scene_root = self.phase64b_raw_root / "l8biome" / row["biome"] / row["scene"]
            raster_path = scene_root / f"{row['scene']}.TIF"
            item.update(
                img_path=str(raster_path),
                phase64b_source="l8_biome_uint8",
                phase64b_window=(int(row["y"]), int(row["x"]), 512, 512),
                phase64b_scene=row["scene"],
                phase64b_biome=row["biome"],
            )
        return items


@DATASETS.register_module()
class Phase64BSparcsSixBandParentDataset(Phase64SparcsManifestDataset):
    """Read each paired SPARCS Level-1 multispectral TIFF at native 30 m."""

    def load_data_list(self):
        items = super().load_data_list()
        rows = _manifest_rows(self.manifest_path, self.manifest_split)
        by_photo = {str(Path(row["image_path"]).resolve()): row for row in rows}
        for item in items:
            row = by_photo.get(str(Path(item["img_path"]).resolve()))
            if row is None:
                raise RuntimeError(f"SPARCS manifest metadata missing for {item['img_path']}")
            raster_path = Path(row["multispectral_path"])
            mtl_path = raster_path.with_name(f"{row['scene']}_mtl.txt")
            item.update(
                img_path=str(raster_path),
                phase64b_source="sparcs_level1_dn",
                phase64b_mtl_path=str(mtl_path),
                phase64b_scene=row["scene"],
            )
        return items


@TRANSFORMS.register_module()
class LoadPhase64BSixBandImage(BaseTransform):
    """Load exactly R/G/B/NIR/SWIR1/SWIR2 with frozen radiometry."""

    def transform(self, results: dict) -> dict:
        source = results["phase64b_source"]
        if source == "cloudsen_l1c":
            count = int(results["phase64b_sample_count"])
            index = int(results["phase64b_sample_index"])
            values = np.stack(
                [_memmap(path, count)[index] for path in results["phase64b_band_paths"]],
                axis=-1,
            )
            valid = np.any(values != 0, axis=-1)
            image = values.astype(np.float32) * 1.0e-4
            radiometry = "sentinel2_l1c_dn_times_1e-4"
        else:
            try:
                import rasterio
                from rasterio.windows import Window
            except ImportError as error:  # pragma: no cover - environment guard
                raise RuntimeError("Phase64B requires rasterio") from error
            with rasterio.open(results["img_path"]) as dataset:
                if source == "l8_biome_uint8":
                    if dataset.count != 11 or any(dtype != "uint8" for dtype in dataset.dtypes):
                        raise RuntimeError(
                            f"Expected 11-band uint8 L8 Biome product: {results['img_path']}"
                        )
                    column, row, width, height = results["phase64b_window"]
                    raw = dataset.read(
                        LANDSAT_BAND_INDEXES,
                        window=Window(column, row, width, height),
                        boundless=True,
                        fill_value=0,
                    ).transpose(1, 2, 0)
                    image = raw.astype(np.float32) / 255.0
                    radiometry = "l8_biome_irreversible_uint8_div_255"
                elif source == "sparcs_level1_dn":
                    if dataset.count != 10 or any(dtype != "uint16" for dtype in dataset.dtypes):
                        raise RuntimeError(
                            f"Expected 10-band uint16 SPARCS product: {results['img_path']}"
                        )
                    raw = dataset.read(LANDSAT_BAND_INDEXES).transpose(1, 2, 0)
                    image = _landsat_toa_reflectance(raw, results["phase64b_mtl_path"])
                    radiometry = "landsat_level1_mtl_toa_sun_angle_corrected"
                else:
                    raise ValueError(f"Unknown Phase64B source {source!r}")
            valid = np.any(raw != 0, axis=-1)

        image = np.nan_to_num(image, nan=0.0, posinf=1.5, neginf=0.0)
        image = np.clip(image, 0.0, 1.5).astype(np.float32, copy=False)
        results["img"] = image
        results["img_shape"] = image.shape[:2]
        results["ori_shape"] = image.shape[:2]
        results["phase64b_valid_mask"] = valid.astype(np.uint8)
        results["phase64b_band_order"] = COMMON_BAND_ORDER
        results["phase64b_radiometry"] = radiometry
        return results


@TRANSFORMS.register_module()
class FinalizePhase64BSpatialProtocol(BaseTransform):
    """Apply the frozen 20 m S2 grid and mask invalid observations."""

    def transform(self, results: dict) -> dict:
        source = results["phase64b_source"]
        valid = results.pop("phase64b_valid_mask").astype(bool)
        if source == "cloudsen_l1c":
            # The legacy CloudSEN memmaps are delivered on a common 10 m grid.
            # Area averaging produces the preregistered common 20 m input grid.
            results["img"] = cv2.resize(
                results["img"], (256, 256), interpolation=cv2.INTER_AREA
            )
            valid_fraction = cv2.resize(
                valid.astype(np.float32), (256, 256), interpolation=cv2.INTER_AREA
            )
            valid = valid_fraction >= 0.999
            if "gt_seg_map" in results:
                results["gt_seg_map"] = cv2.resize(
                    results["gt_seg_map"], (256, 256), interpolation=cv2.INTER_NEAREST
                )
        if "gt_seg_map" in results:
            if results["gt_seg_map"].shape != valid.shape:
                raise RuntimeError(
                    f"Image/label validity mismatch: {valid.shape} vs "
                    f"{results['gt_seg_map'].shape}"
                )
            results["gt_seg_map"][~valid] = 255
        results["img_shape"] = results["img"].shape[:2]
        results["ori_shape"] = results["img"].shape[:2]
        return results

