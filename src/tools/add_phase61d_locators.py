"""Retrofit label-neutral target locators onto an existing Phase 61D packet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from prepare_phase61d_blind_review import locator_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet-root", default="work_dirs/phase61/blind_review")
    args = parser.parse_args()

    packet_root = Path(args.packet_root)
    records = json.loads((packet_root / "sealed_manifest.json").read_text(encoding="utf-8"))["records"]
    source_root = packet_root / "images"
    locator_root = packet_root / "images_locator"
    locator_root.mkdir(parents=True, exist_ok=True)

    off_center = 0
    for row in records:
        target_x = int(row.get("target_x_in_crop", row["center_x"] - row["crop_left"]))
        target_y = int(row.get("target_y_in_crop", row["center_y"] - row["crop_top"]))
        off_center += target_x != row["crop_size"] // 2 or target_y != row["crop_size"] // 2
        with Image.open(source_root / f"{row['tile_id']}.png") as image:
            locator_image(image.convert("RGB"), target_x, target_y).save(locator_root / f"{row['tile_id']}.png")

    (packet_root / "LOCATOR_README.md").write_text(
        "# Phase 61D locator images\n\n"
        "Review `images_locator/`, not the unmarked `images/` directory. Label the semantic class "
        "at the location enclosed by the yellow corner brackets; the surrounding crop is context only. "
        "The marker exposes neither the original label nor model output.\n",
        encoding="utf-8",
    )
    print(json.dumps({"records": len(records), "off_center": off_center, "output": str(locator_root)}))


if __name__ == "__main__":
    main()
