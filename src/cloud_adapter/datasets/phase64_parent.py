"""Three-parent dataset views frozen by the Phase 64 ontology protocol."""

from __future__ import annotations

from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS

from .manifest_seg import Phase45L8ManifestDataset


PARENT_CLASSES = ("surface_visible", "cloud", "shadow")
PARENT_PALETTE = [[80, 130, 70], [245, 245, 245], [70, 70, 70]]
CLOUDSEN_TO_PARENT = {0: 0, 1: 1, 2: 1, 3: 2}
L8_TO_PARENT = {0: 0, 1: 2, 2: 1, 3: 1}


@DATASETS.register_module()
class Phase64CloudSENParentDataset(BaseSegDataset):
    """Map official CloudSEN High labels to surface/cloud/shadow."""

    METAINFO = dict(classes=PARENT_CLASSES, palette=PARENT_PALETTE)

    def __init__(
        self,
        img_suffix=".png",
        seg_map_suffix=".png",
        reduce_zero_label=False,
        **kwargs,
    ) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs,
        )

    def load_data_list(self):
        rows = super().load_data_list()
        for row in rows:
            row["label_map"] = dict(CLOUDSEN_TO_PARENT)
        return rows


@DATASETS.register_module()
class Phase64L8ParentDataset(Phase45L8ManifestDataset):
    """Map audited L8 masks to parents, requiring complete shadow scenes."""

    METAINFO = dict(classes=PARENT_CLASSES, palette=PARENT_PALETTE)

    def __init__(self, **kwargs) -> None:
        if kwargs.get("usgs_shadows_filter", "yes") != "yes":
            raise ValueError("Phase64L8ParentDataset requires usgs_shadows_filter='yes'")
        kwargs["usgs_shadows_filter"] = "yes"
        super().__init__(target_to_source=False, **kwargs)

    def load_data_list(self):
        rows = super().load_data_list()
        for row in rows:
            row["label_map"] = dict(L8_TO_PARENT)
        return rows
