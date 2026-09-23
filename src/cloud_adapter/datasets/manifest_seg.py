import csv
import json
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

    def __init__(
        self,
        manifest_path,
        split,
        mask_column="mask_path",
        selection_path=None,
        target_to_source=False,
        **kwargs,
    ):
        self.manifest_path = Path(manifest_path)
        self.manifest_split = split
        self.mask_column = mask_column
        self.selection_path = Path(selection_path) if selection_path else None
        self.target_to_source = bool(target_to_source)
        if self.target_to_source:
            # BaseSegDataset validates configured class names against METAINFO
            # before load_data_list can attach the pixel-value label map.
            self.METAINFO = dict(
                classes=("clear", "thick cloud", "thin cloud", "cloud shadow"),
                palette=[[0, 0, 0], [255, 255, 255], [170, 170, 170], [85, 85, 85]],
            )
        super().__init__(
            img_suffix=".png",
            seg_map_suffix=".png",
            reduce_zero_label=False,
            data_prefix=dict(img_path="", seg_map_path=""),
            **kwargs,
        )

    def load_data_list(self):
        selected_names = None
        if self.selection_path is not None:
            selection = json.loads(self.selection_path.read_text(encoding="utf-8"))
            selected_names = {row["name"] for row in selection["selected"]}
            if not selected_names:
                raise RuntimeError(f"Empty selection in {self.selection_path}")
        rows = []
        with self.manifest_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if (
                    row["new_split"] == self.manifest_split
                    and (selected_names is None or row["name"] in selected_names)
                ):
                    rows.append(row)
        rows.sort(key=lambda row: row["name"])
        if not rows:
            raise RuntimeError(
                f"No records for {self.manifest_split!r} in {self.manifest_path}"
            )
        data_list = []
        for row in rows:
            item = dict(
                img_path=str(Path(row["image_path"]).resolve()),
                seg_map_path=str(Path(row[self.mask_column]).resolve()),
                label_map=self.label_map,
                reduce_zero_label=self.reduce_zero_label,
                seg_fields=[],
            )
            if self.target_to_source:
                # Landsat order: clear, shadow, thin, thick. CloudSEN/model
                # order: clear, thick, thin, shadow.
                item["label_map"] = {0: 0, 1: 3, 2: 2, 3: 1}
            data_list.append(item)
        if selected_names is not None and len(data_list) != len(selected_names):
            raise RuntimeError(
                f"Selection has {len(selected_names)} names but dataset found "
                f"{len(data_list)} records"
            )
        return data_list
