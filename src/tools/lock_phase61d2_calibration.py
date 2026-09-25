"""Lock the shared Phase 61D2 calibration before independent review begins."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from prepare_phase61d2_calibrated_review import sha256
from score_phase61d2_calibrated_review import parse_label_set


def read_completed(path, ids, column):
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not {"tile_id", column}.issubset(rows[0]):
        raise RuntimeError(f"Missing tile_id/{column} columns in {path}")
    found = [row["tile_id"].strip() for row in rows]
    if len(found) != len(set(found)) or set(found) != set(ids):
        raise RuntimeError(f"Calibration IDs do not match sealed calibration cohort: {path}")
    for row in rows:
        parse_label_set(row[column])
    return {"path": str(path), "sha256": sha256(path), "units": len(rows)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sealed", default="work_dirs/phase61d2_calibrated_review/sealed_manifest.json")
    parser.add_argument("--manual", default="research_plans/PHASE61D2_REVIEW_MANUAL.md")
    parser.add_argument("--reviewer-a-calibration", required=True)
    parser.add_argument("--reviewer-b-calibration", required=True)
    parser.add_argument("--consensus", required=True)
    parser.add_argument("--reviewer-a-id", required=True)
    parser.add_argument("--reviewer-b-id", required=True)
    parser.add_argument("--discussion-date", required=True, help="ISO date, e.g. 2026-09-26")
    parser.add_argument("--manual-amendment", action="append", default=[])
    parser.add_argument("--output", default="work_dirs/phase61d2_calibrated_review/calibration_lock.json")
    args = parser.parse_args()

    sealed = json.loads(Path(args.sealed).read_text(encoding="utf-8"))
    ids = [row["tile_id"] for row in sealed["records"] if row["cohort"] == "calibration"]
    if len(ids) != 40:
        raise RuntimeError(f"Expected 40 calibration units, found {len(ids)}")
    manual_hash = sha256(args.manual)
    if manual_hash != sealed["manual_sha256"]:
        raise RuntimeError("Operation manual changed after packet sealing; rebuild the packet")
    review_a = read_completed(args.reviewer_a_calibration, ids, "label_set")
    review_b = read_completed(args.reviewer_b_calibration, ids, "label_set")
    consensus = read_completed(args.consensus, ids, "agreed_label_set")
    if not args.manual_amendment:
        raise RuntimeError(
            "Record at least one calibration decision or explicit 'no amendment required' statement"
        )
    document = {
        "phase": "61D2-calibration-lock",
        "status": "locked_before_independent_review",
        "calibration_units": len(ids),
        "calibration_excluded_from_final_statistics": True,
        "reviewer_A_id": args.reviewer_a_id,
        "reviewer_B_id": args.reviewer_b_id,
        "discussion_date": args.discussion_date,
        "manual_sha256": manual_hash,
        "reviewer_A_calibration": review_a,
        "reviewer_B_calibration": review_b,
        "calibration_consensus": consensus,
        "manual_amendments_or_confirmations": args.manual_amendment,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2), encoding="utf-8")
    print(json.dumps(document, indent=2))


if __name__ == "__main__":
    main()
