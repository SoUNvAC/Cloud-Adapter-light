"""Write non-destructive status markers for Phase63 random-point review sets."""

from __future__ import annotations

import argparse
import csv
import io
import json
import zipfile
from pathlib import Path

from phase56_protocol import sha256


EXPECTED_UI_ZIP = "3ea92af06157c405529e202768ce1b4e83919312dcd5f22f5b73ceb9912ff568"
EXPECTED_CONFIRM_ZIP = "dc16956f960c4d7e86329d41b06ddd98b35ce26311cc1359b3965fd8cdafab85"


def review_tables_blank(zip_path: Path, suffix: str) -> bool:
    with zipfile.ZipFile(zip_path) as archive:
        names = [name for name in archive.namelist() if name.endswith(suffix)]
        if not names:
            raise RuntimeError(f"No review tables ending {suffix} in {zip_path}")
        for name in names:
            rows = list(csv.DictReader(io.StringIO(archive.read(name).decode("utf-8"))))
            for row in rows:
                if any((value or "").strip() for key, value in row.items() if key != "tile_id"):
                    return False
    return True


def write_marker(path: Path, document: dict) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != document:
            raise RuntimeError(f"Existing status marker differs: {path}")
        return
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ui-root", default="work_dirs/phase63_selective_multigranularity/pixel_locator_ui_pilot"
    )
    parser.add_argument(
        "--confirmation-root",
        default="work_dirs/phase63_selective_multigranularity/human_confirmation_review",
    )
    args = parser.parse_args()
    ui_root, confirmation_root = Path(args.ui_root), Path(args.confirmation_root)
    ui_zip = ui_root / "phase63_pixel_locator_ui_pilot_packet.zip"
    confirmation_zip = confirmation_root / "phase63_confirmation_reviewer_packet.zip"
    if sha256(ui_zip) != EXPECTED_UI_ZIP or sha256(confirmation_zip) != EXPECTED_CONFIRM_ZIP:
        raise RuntimeError("Frozen packet hash mismatch; refusing to mark status")
    if not review_tables_blank(ui_zip, "_ui_pilot.csv"):
        raise RuntimeError("Random UI pilot contains nonblank review data; cannot mark before-review")
    if not review_tables_blank(confirmation_zip, "_confirmation.csv"):
        raise RuntimeError("Old confirmation packet contains nonblank confirmation data")
    write_marker(
        ui_root / "SUPERSEDED.json",
        {
            "status": "superseded_before_review",
            "packet_sha256": EXPECTED_UI_ZIP,
            "reason": "Semantic-unconditional random points do not test semantic coverage.",
            "may_be_used_for_results": False,
        },
    )
    write_marker(
        confirmation_root / "SUSPENDED.json",
        {
            "status": "suspended_not_formal_confirmation",
            "packet_sha256": EXPECTED_CONFIRM_ZIP,
            "reason": (
                "Semantic-unconditional hashed centers can leave Thin/Shadow denominators inadequate; "
                "the untouched packet no longer has automatic formal-confirmation status."
            ),
            "reviewer_packet_was_blank_at_suspension": True,
            "may_be_used_for_phase63c_gate": False,
        },
    )
    print(json.dumps({"ui": "superseded_before_review", "confirmation": "suspended"}, indent=2))


if __name__ == "__main__":
    main()
