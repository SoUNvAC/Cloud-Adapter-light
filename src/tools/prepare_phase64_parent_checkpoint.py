#!/usr/bin/env python3
"""Remove only the incompatible native classifier from a Phase 64 checkpoint.

The source Mask2Former checkpoint predicts four CloudSEN classes plus the
no-object class.  The parent-space screen predicts three parents plus
no-object.  PyTorch cannot non-strictly load tensors whose shapes differ, so
this tool makes the required omission explicit and auditable while retaining
every compatible backbone and decoder tensor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_within(path: Path, root: Path, *, must_exist: bool) -> Path:
    resolved_root = root.resolve()
    if must_exist:
        resolved = path.resolve(strict=True)
    else:
        resolved = path.parent.resolve(strict=True) / path.name
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"Path escapes repository root: {resolved}") from exc
    return resolved


def convert_checkpoint(
    checkpoint: dict[str, Any], source_classes: int = 4
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError("Checkpoint has no non-empty state_dict")

    expected_rows = source_classes + 1
    dropped = []
    retained = {}
    for key, value in state_dict.items():
        is_classifier = key in {
            "decode_head.cls_embed.weight",
            "decode_head.cls_embed.bias",
        }
        if is_classifier:
            if not isinstance(value, torch.Tensor) or value.ndim not in (1, 2):
                raise ValueError(f"Unexpected classifier tensor at {key}")
            if value.shape[0] != expected_rows:
                raise ValueError(
                    f"{key} has {value.shape[0]} rows, expected {expected_rows}"
                )
            dropped.append({"key": key, "shape": list(value.shape)})
        else:
            retained[key] = value

    expected_keys = {
        "decode_head.cls_embed.weight",
        "decode_head.cls_embed.bias",
    }
    if {row["key"] for row in dropped} != expected_keys:
        raise ValueError(
            "Did not find exactly the expected Mask2Former classifier tensors; "
            f"found {[row['key'] for row in dropped]}"
        )

    converted = dict(checkpoint)
    converted["state_dict"] = retained
    converted.pop("optimizer", None)
    converted.pop("param_schedulers", None)
    converted["phase64_parent_conversion"] = {
        "source_classes": source_classes,
        "target_parent_classes": 3,
        "dropped": dropped,
        "retained_tensor_count": len(retained),
    }
    return converted, dropped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--allowed-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    source = ensure_within(args.input, args.allowed_root, must_exist=True)
    destination = ensure_within(args.output, args.allowed_root, must_exist=False)
    summary_path = ensure_within(
        args.summary or destination.with_suffix(destination.suffix + ".json"),
        args.allowed_root,
        must_exist=False,
    )
    if (destination.exists() or summary_path.exists()) and not args.force:
        raise FileExistsError("Output exists; pass --force only after verifying target")

    checkpoint = torch.load(source, map_location="cpu")
    converted, dropped = convert_checkpoint(checkpoint)
    torch.save(converted, destination)
    summary = {
        "phase": "64-parent-checkpoint-conversion",
        "input": source.as_posix(),
        "input_sha256": sha256_file(source),
        "output": destination.as_posix(),
        "output_sha256": sha256_file(destination),
        "dropped": dropped,
        "retained_tensor_count": len(converted["state_dict"]),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
