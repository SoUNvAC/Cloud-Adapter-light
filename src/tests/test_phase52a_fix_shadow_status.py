from pathlib import Path

from cloud_adapter.datasets.manifest_seg import (
    l8_target_to_source_label_map,
    load_usgs_shadow_status,
)
from tools.fetch_l8_biome_usgs_metadata import parse_scene_metadata


def test_official_snapshot_counts():
    path = Path("research_plans/protocol_data/l8_biome_usgs_shadow_status.csv")
    metadata = load_usgs_shadow_status(path)
    assert len(metadata) == 96
    assert sum(value == "yes" for value in metadata.values()) == 32
    assert sum(value == "no" for value in metadata.values()) == 64


def test_partial_label_mapping():
    assert l8_target_to_source_label_map("yes", True) == {0: 0, 1: 3, 2: 2, 3: 1}
    assert l8_target_to_source_label_map("no", True) == {0: 255, 1: 3, 2: 2, 3: 1}
    assert l8_target_to_source_label_map("no", False) == {0: 0, 1: 3, 2: 2, 3: 1}


def test_html_parser_requires_one_status_per_scene():
    rows = []
    for index in range(96):
        scene = f"LC8{index:06d}2014001LGN00"
        rows.append(f"<tr><td>{scene}</td><td>{'yes' if index < 32 else 'no'}</td></tr>")
    metadata = parse_scene_metadata("<table>" + "".join(rows) + "</table>")
    assert len(metadata) == 96
    assert sum(value == "yes" for value in metadata.values()) == 32


if __name__ == "__main__":
    test_official_snapshot_counts()
    test_partial_label_mapping()
    test_html_parser_requires_one_status_per_scene()
    print("phase52a-fix shadow-status tests passed")
