"""Manifest-backed L8 SPARCS datasets for the frozen Phase 64 protocol."""

from __future__ import annotations

import csv
from pathlib import Path

from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS


SPARCS_NATIVE_CLASSES = (
    "shadow",
    "shadow over water",
    "water",
    "snow",
    "land",
    "cloud",
    "flooded",
)
SPARCS_PARENT_CLASSES = ("surface_visible", "cloud", "shadow")
SPARCS_NATIVE_TO_PARENT = {0: 2, 1: 2, 2: 0, 3: 0, 4: 0, 5: 1, 6: 0}


def sparcs_label_map(granularity: str) -> dict[int, int]:
    if granularity == "native":
        return {index: index for index in range(7)}
    if granularity == "parent":
        return dict(SPARCS_NATIVE_TO_PARENT)
    raise ValueError("granularity must be 'native' or 'parent'")


@DATASETS.register_module()
class Phase64SparcsManifestDataset(BaseSegDataset):
    """Read RGB previews and native masks from the audited SPARCS manifest.

    The RGB preview is used only by the Phase 64 minimal ontology screen.  The
    manifest retains the original multiband TIFF path separately for later
    common-band experiments; this class does not pretend the preview is a
    multispectral observation.
    """

    METAINFO = dict(
        classes=SPARCS_PARENT_CLASSES,
        palette=[[80, 130, 70], [245, 245, 245], [70, 70, 70]],
    )

    def __init__(self, manifest_path: str, split: str, granularity: str = "parent", **kwargs):
        self.manifest_path = Path(manifest_path)
        self.manifest_split = split
        self.granularity = granularity
        if granularity == "native":
            self.METAINFO = dict(
                classes=SPARCS_NATIVE_CLASSES,
                palette=[
                    [70, 70, 70],
                    [45, 65, 90],
                    [30, 110, 200],
                    [210, 235, 255],
                    [80, 130, 70],
                    [245, 245, 245],
                    [70, 150, 155],
                ],
            )
        elif granularity != "parent":
            raise ValueError("granularity must be 'native' or 'parent'")
        super().__init__(
            img_suffix=".png",
            seg_map_suffix=".png",
            reduce_zero_label=False,
            data_prefix=dict(img_path="", seg_map_path=""),
            **kwargs,
        )

    def load_data_list(self):
        with self.manifest_path.open(newline="", encoding="utf-8") as handle:
            rows = [
                row for row in csv.DictReader(handle)
                if row["new_split"] == self.manifest_split
            ]
        rows.sort(key=lambda row: (row["scene"], row["name"]))
        if not rows:
            raise RuntimeError(
                f"No SPARCS records for {self.manifest_split!r} in {self.manifest_path}"
            )
        label_map = sparcs_label_map(self.granularity)
        return [
            dict(
                img_path=str(Path(row["image_path"]).resolve()),
                seg_map_path=str(Path(row["mask_path"]).resolve()),
                label_map=label_map,
                reduce_zero_label=False,
                seg_fields=[],
            )
            for row in rows
        ]
