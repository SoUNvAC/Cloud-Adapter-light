"""Create an immutable, provenance-tracked canonical Phase 61D2 review CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from score_phase61d2_calibrated_review import canonical, parse_label_set


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonicalize_rows(rows, blank_label):
    replacement = canonical(parse_label_set(blank_label))
    result, mapped = [], []
    for source in rows:
        row = dict(source)
        tile_id = row.get("tile_id", "").strip()
        if not tile_id:
            raise RuntimeError("Blank tile_id in review file")
        value = row.get("label_set", "").strip()
        if not value:
            row["label_set"] = replacement
            mapped.append(tile_id)
        else:
            row["label_set"] = canonical(parse_label_set(value))
        result.append(row)
    return result, mapped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--provenance", required=True)
    parser.add_argument("--blank-label", default="unobservable_nodata")
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    provenance_path = Path(args.provenance)
    if output_path.exists() or provenance_path.exists():
        raise RuntimeError("Refusing to overwrite canonical review or provenance")
    with input_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames
    if not rows or fieldnames is None or not {"tile_id", "label_set"}.issubset(fieldnames):
        raise RuntimeError("Invalid review CSV")
    ids = [row["tile_id"].strip() for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate review IDs")
    canonical_rows, mapped = canonicalize_rows(rows, args.blank_label)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(canonical_rows)
    document = {
        "phase": "61D2-review-canonicalization",
        "input": {"path": str(input_path), "sha256": sha256(input_path)},
        "output": {"path": str(output_path), "sha256": sha256(output_path)},
        "rows": len(rows),
        "blank_label_mapping": args.blank_label,
        "mapped_blank_units": len(mapped),
        "mapped_tile_ids": mapped,
        "reason": args.reason,
        "raw_input_preserved": True,
    }
    provenance_path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    print(json.dumps(document, indent=2))


if __name__ == "__main__":
    main()
