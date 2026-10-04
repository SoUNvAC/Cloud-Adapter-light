#!/usr/bin/env python3
"""Validate the frozen Phase 64 heterogeneous-label dataset registry.

The audit is intentionally dependency-free.  It checks ontology coverage,
split provenance, licensing metadata, and execution readiness without reading
dataset pixels or claiming that unavailable external data have been ingested.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXPECTED_PARENTS = ("surface_visible", "cloud", "shadow")
SURFACE_SUBTYPES = {"snow", "water", "flooded"}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_registry(registry: dict[str, Any]) -> dict[str, Any]:
    _require(registry.get("schema_version") == 1, "schema_version must be 1")
    hierarchy = registry.get("canonical_hierarchy", {})
    parents = tuple(hierarchy.get("parent_classes", []))
    _require(parents == EXPECTED_PARENTS, f"parent classes must be {EXPECTED_PARENTS}")
    _require("clear-sky land" in hierarchy.get("forbidden_shortcut", ""),
             "registry must forbid collapsing non-land surface labels into clear-sky land")

    datasets = registry.get("datasets", [])
    _require(len(datasets) >= 3, "at least three public datasets are required")
    ids = [row.get("id") for row in datasets]
    _require(len(ids) == len(set(ids)), "dataset ids must be unique")
    _require(all(row.get("public") is True for row in datasets),
             "all Day-1 registry entries must be public datasets")

    source_count = 0
    target_shadow_count = 0
    implemented = []
    blocked = []
    table = []
    for row in datasets:
        dataset_id = row.get("id")
        _require(isinstance(dataset_id, str) and dataset_id, "dataset id is required")
        role = row.get("role", "")
        source_count += int(role == "source")
        target_shadow_count += int(role.startswith("target") and row.get("has_shadow") is True)

        labels = row.get("native_labels", [])
        ignores = row.get("ignore_labels", [])
        _require(labels, f"{dataset_id}: native_labels must not be empty")
        native_ids = [str(item.get("id")) for item in labels]
        ignore_ids = [str(item.get("id")) for item in ignores]
        _require(len(native_ids) == len(set(native_ids)), f"{dataset_id}: duplicate native id")
        _require(not set(native_ids).intersection(ignore_ids),
                 f"{dataset_id}: an id cannot be both supervised and ignored")
        mapped_parents = {item.get("parent") for item in labels}
        _require(mapped_parents.issubset(EXPECTED_PARENTS),
                 f"{dataset_id}: unknown parent mapping {mapped_parents - set(EXPECTED_PARENTS)}")
        _require("cloud" in mapped_parents, f"{dataset_id}: cloud parent is missing")
        if row.get("has_shadow"):
            _require("shadow" in mapped_parents, f"{dataset_id}: shadow parent is missing")

        for item in labels:
            name = item.get("name")
            if name in SURFACE_SUBTYPES:
                _require(item.get("parent") == "surface_visible",
                         f"{dataset_id}: {name} must remain a surface_visible child")

        splits = row.get("splits", {})
        _require(all(key in splits for key in ("type", "train", "validation", "test", "unit")),
                 f"{dataset_id}: incomplete split provenance")
        license_info = row.get("license", {})
        _require(license_info.get("name") and license_info.get("status"),
                 f"{dataset_id}: license name and verification status are required")
        _require(len(row.get("references", [])) >= 2,
                 f"{dataset_id}: at least two primary/release references are required")

        repo = row.get("repository", {})
        if repo.get("implemented") is True:
            _require(repo.get("loader"), f"{dataset_id}: implemented loader path is required")
            implemented.append(dataset_id)
        else:
            blocked.append(dataset_id)

        table.append({
            "id": dataset_id,
            "role": role,
            "sensor": row.get("sensor"),
            "native_class_count": len(labels),
            "parents": sorted(mapped_parents),
            "thin_thick": row.get("thin_thick_granularity"),
            "split_type": splits.get("type"),
            "license": license_info.get("name"),
            "loader_implemented": bool(repo.get("implemented")),
            "data_presence": repo.get("data_presence"),
        })

    _require(source_count == 1, "exactly one Day-1 source dataset is required")
    _require(target_shadow_count >= 2,
             "at least two public target domains with native shadow labels are required")

    l8 = next((row for row in datasets if row.get("id") == "l8_biome"), None)
    _require(l8 is not None, "l8_biome entry is required")
    _require(l8.get("full_parent_supervision_filter") == "Shadows?=yes scenes only",
             "L8 full three-parent supervision must be restricted to Shadows?=yes scenes")

    ready_targets = [
        row["id"] for row in datasets
        if row.get("role", "").startswith("target")
        and row.get("repository", {}).get("implemented") is True
    ]
    return {
        "phase": registry.get("phase"),
        "status": "registry_valid",
        "dataset_count": len(datasets),
        "public_target_domains_with_shadow": target_shadow_count,
        "implemented_datasets": implemented,
        "unimplemented_datasets": blocked,
        "implemented_target_domains": ready_targets,
        "day2_two_target_execution_ready": len(ready_targets) >= 2,
        "rows": table,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("research_plans/protocol_data/phase64_dataset_registry.json"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    summary = validate_registry(registry)
    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
