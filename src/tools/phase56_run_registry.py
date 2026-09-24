"""Small append/update utility for the single-GPU Phase 56--59 registry."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import subprocess
from pathlib import Path

FIELDS = (
    "run_id", "phase", "route", "status", "commit", "config_sha256",
    "checkpoint_sha256", "manifest_sha256", "started_at", "ended_at",
    "command", "notes",
)


def git_commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def load(path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("register", "start", "finish", "fail", "show"))
    parser.add_argument("--registry", default="work_dirs/phase56_protocol/run_registry.csv")
    parser.add_argument("--run-id")
    parser.add_argument("--phase", default="56")
    parser.add_argument("--route", default="readout")
    parser.add_argument("--config-sha256", default="")
    parser.add_argument("--checkpoint-sha256", default="")
    parser.add_argument("--manifest-sha256", default="")
    parser.add_argument("--command", default="")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()
    path, now = Path(args.registry), dt.datetime.now(dt.timezone.utc).isoformat()
    rows = load(path)
    if args.action == "show":
        print(json.dumps(rows, indent=2))
        return
    if not args.run_id:
        parser.error("--run-id is required")
    matches = [row for row in rows if row["run_id"] == args.run_id]
    if args.action == "register":
        if matches:
            raise RuntimeError(f"Duplicate run id: {args.run_id}")
        rows.append(dict.fromkeys(FIELDS, ""))
        rows[-1].update(
            run_id=args.run_id, phase=args.phase, route=args.route,
            status="registered", commit=git_commit(),
            config_sha256=args.config_sha256,
            checkpoint_sha256=args.checkpoint_sha256,
            manifest_sha256=args.manifest_sha256,
            command=args.command, notes=args.notes,
        )
    else:
        if len(matches) != 1:
            raise RuntimeError(f"Run id must exist exactly once: {args.run_id}")
        row = matches[0]
        transitions = {
            "start": ("running", "started_at"),
            "finish": ("complete", "ended_at"),
            "fail": ("failed", "ended_at"),
        }
        row["status"], timestamp_field = transitions[args.action]
        row[timestamp_field] = now
        if args.notes:
            row["notes"] = args.notes
        if args.manifest_sha256:
            row["manifest_sha256"] = args.manifest_sha256
    save(path, rows)
    print(json.dumps([row for row in rows if row["run_id"] == args.run_id][0], indent=2))


if __name__ == "__main__":
    main()
