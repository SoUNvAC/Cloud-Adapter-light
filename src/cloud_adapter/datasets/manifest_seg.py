import csv
from pathlib import Path

from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS


@DATASETS.register_module()
class Phase45L8ManifestDataset(BaseSegDataset):
    """Read the frozen Phase 45 scene-level Landsat manifest."""

    METAINFO = dict(
        classes=("clear", "cloud shadow", "thin cloud", "cloud"),
        palette=[[0, 0, 0], [85, 85, 85], [170, 170, 170], [255, 255, 255]],
    )

    def __init__(self, manifest_path, split, mask_column="mask_path", **kwargs):
        self.manifest_path = Path(manifest_path)
        self.manifest_split = split
        self.mask_column = mask_column
        super().__init__(
            img_suffix=".png",
            seg_map_suffix=".png",
            reduce_zero_label=False,
            data_prefix=dict(img_path="", seg_map_path=""),
            **kwargs,
        )

    def load_data_list(self):
        rows = []
        with self.manifest_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["new_split"] == self.manifest_split:
                    rows.append(row)
        rows.sort(key=lambda row: row["name"])
        if not rows:
            raise RuntimeError(
                f"No records for {self.manifest_split!r} in {self.manifest_path}"
            )
        return [
            dict(
                img_path=str(Path(row["image_path"]).resolve()),
                seg_map_path=str(Path(row[self.mask_column]).resolve()),
                label_map=self.label_map,
                reduce_zero_label=self.reduce_zero_label,
                seg_fields=[],
            )
            for row in rows
        ]
