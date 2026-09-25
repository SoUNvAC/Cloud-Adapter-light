import csv
import json
from pathlib import Path

from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS


def load_usgs_shadow_status(path):
    if path is None:
        return None
    metadata_path = Path(path)
    with metadata_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"scene", "usgs_shadows"}
    if not rows or not required.issubset(rows[0]):
        raise RuntimeError(f"Invalid USGS shadow metadata: {metadata_path}")
    result = {}
    for row in rows:
        scene, value = row["scene"], row["usgs_shadows"].lower()
        if value not in ("yes", "no") or scene in result:
            raise RuntimeError(f"Invalid USGS Shadows? row: {row}")
        result[scene] = value
    if len(result) != 96 or sum(value == "yes" for value in result.values()) != 32:
        raise RuntimeError(
            f"Expected 96 scenes and 32 Shadows?=yes rows, got {len(result)} and "
            f"{sum(value == 'yes' for value in result.values())}"
        )
    return result


def l8_target_to_source_label_map(
    shadow_status=None,
    mask_unlabelled_clear=False,
    partial_unlabelled_clear=False,
    partial_label_index=254,
):
    label_map = {0: 0, 1: 3, 2: 2, 3: 1}
    if mask_unlabelled_clear and partial_unlabelled_clear:
        raise ValueError("Cannot both ignore and partially supervise raw target ID 0")
    if mask_unlabelled_clear and shadow_status == "no":
        label_map[0] = 255
    if partial_unlabelled_clear and shadow_status == "no":
        if int(partial_label_index) not in range(4, 255):
            raise ValueError("partial_label_index must be an unused uint8 value in [4, 254]")
        label_map[0] = int(partial_label_index)
    return label_map


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
        usgs_shadow_metadata_path=None,
        usgs_shadows_filter=None,
        mask_unlabelled_shadow_clear=False,
        partial_unlabelled_shadow_clear=False,
        partial_label_index=254,
        **kwargs,
    ):
        self.manifest_path = Path(manifest_path)
        self.manifest_split = split
        self.mask_column = mask_column
        self.selection_path = Path(selection_path) if selection_path else None
        self.target_to_source = bool(target_to_source)
        self.usgs_shadow_status = load_usgs_shadow_status(usgs_shadow_metadata_path)
        if usgs_shadows_filter not in (None, "yes", "no"):
            raise ValueError("usgs_shadows_filter must be None, 'yes', or 'no'")
        self.usgs_shadows_filter = usgs_shadows_filter
        self.mask_unlabelled_shadow_clear = bool(mask_unlabelled_shadow_clear)
        self.partial_unlabelled_shadow_clear = bool(partial_unlabelled_shadow_clear)
        self.partial_label_index = int(partial_label_index)
        if self.mask_unlabelled_shadow_clear and self.partial_unlabelled_shadow_clear:
            raise ValueError("Choose ignore or partial supervision for unlabelled raw ID 0")
        if (
            self.mask_unlabelled_shadow_clear or self.partial_unlabelled_shadow_clear
        ) and not self.target_to_source:
            raise ValueError("Partial shadow labels require target_to_source=True")
        if (
            (
                self.usgs_shadows_filter is not None
                or self.mask_unlabelled_shadow_clear
                or self.partial_unlabelled_shadow_clear
            )
            and self.usgs_shadow_status is None
        ):
            raise ValueError("USGS shadow metadata is required by the requested protocol")
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
        seen_selected_names = set()
        split_selected_names = set()
        eligible_selected_names = set()
        with self.manifest_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                shadow_status = (
                    self.usgs_shadow_status.get(row["scene"])
                    if self.usgs_shadow_status is not None else None
                )
                if self.usgs_shadow_status is not None and shadow_status is None:
                    raise RuntimeError(f"Missing USGS Shadows? value for {row['scene']}")
                if selected_names is not None and row["name"] in selected_names:
                    seen_selected_names.add(row["name"])
                    if row["new_split"] == self.manifest_split:
                        split_selected_names.add(row["name"])
                        if (
                            self.usgs_shadows_filter is None
                            or shadow_status == self.usgs_shadows_filter
                        ):
                            eligible_selected_names.add(row["name"])
                if (
                    row["new_split"] == self.manifest_split
                    and (selected_names is None or row["name"] in selected_names)
                    and (
                        self.usgs_shadows_filter is None
                        or shadow_status == self.usgs_shadows_filter
                    )
                ):
                    rows.append({**row, "usgs_shadows": shadow_status})
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
                item["label_map"] = l8_target_to_source_label_map(
                    row["usgs_shadows"],
                    self.mask_unlabelled_shadow_clear,
                    self.partial_unlabelled_shadow_clear,
                    self.partial_label_index,
                )
                if self.mask_unlabelled_shadow_clear and row["usgs_shadows"] == "no":
                    # A USGS Shadows?=no scene has no shadow truth mask. Its
                    # target ID 0 may mean clear, fill, or unlabelled shadow,
                    # so it cannot be used as a verified shadow negative.
                    # Thin/thick-cloud pixels remain valid supervision.
                    assert item["label_map"][0] == 255
                if self.partial_unlabelled_shadow_clear and row["usgs_shadows"] == "no":
                    # Keep raw target ID 0 distinguishable through the pipeline;
                    # the Phase 62 segmentor supervises it with {clear, shadow}.
                    assert item["label_map"][0] == self.partial_label_index
            data_list.append(item)
        if selected_names is not None:
            if seen_selected_names != selected_names:
                raise RuntimeError("Selection contains names absent from the manifest")
            if split_selected_names != selected_names:
                raise RuntimeError(
                    f"Selection contains names outside split {self.manifest_split!r}"
                )
            if len(data_list) != len(eligible_selected_names):
                raise RuntimeError(
                    f"Selection/filter expects {len(eligible_selected_names)} records "
                    f"but dataset found {len(data_list)}"
                )
        return data_list


@DATASETS.register_module()
class Phase50L8MappedDataset(Phase45L8ManifestDataset):
    """Expose Landsat labels in the source model's semantic class order."""

    METAINFO = dict(
        classes=("clear", "thick cloud", "thin cloud", "cloud shadow"),
        palette=[[0, 0, 0], [255, 255, 255], [170, 170, 170], [85, 85, 85]],
    )

    def __init__(self, **kwargs):
        super().__init__(target_to_source=True, **kwargs)
