#!/usr/bin/env python3
"""Build Phase 64 models and audit checkpoint compatibility/trainable scope."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import torch
from mmengine.config import Config
from mmseg.registry import MODELS
from mmseg.utils import register_all_modules


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))
import cloud_adapter  # noqa: E402,F401  register project modules


SOURCE_CONFIG = SRC_ROOT / "configs/protocol/phase64_source_parent_rgb.py"
TARGET_CONFIG = SRC_ROOT / "configs/protocol/phase64_target_parent_rgb.py"
METHODS = ("shared_parent", "full", "lora", "msre")


@contextmanager
def scoped_environment(values: dict[str, str]):
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parameter_report(model) -> tuple[int, int, list[str]]:
    model.train()
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return total, trainable, trainable_names


def load_report(model, state_dict) -> tuple[list[str], list[str]]:
    result = model.load_state_dict(state_dict, strict=False)
    return sorted(result.missing_keys), sorted(result.unexpected_keys)


def expected_missing(method: str, missing: list[str]) -> bool:
    classifier = {
        "decode_head.cls_embed.weight",
        "decode_head.cls_embed.bias",
    }
    values = set(missing)
    if not classifier.issubset(values):
        return False
    extras = values - classifier
    if method in {"source_parent", "shared_parent", "full"}:
        return not extras
    if method == "lora":
        return bool(extras) and all(
            name.endswith((".lora_a", ".lora_b")) for name in extras
        )
    if method == "msre":
        return bool(extras) and all(
            name.startswith(("backbone.target_msre.", "backbone.target_head_delta."))
            for name in extras
        )
    return False


def scope_gate(method: str, names: list[str], trainable: int) -> bool:
    if not names or trainable <= 0:
        return False
    if method in {"source_parent", "shared_parent"}:
        return all(not name.startswith("backbone.") for name in names)
    if method == "full":
        return (
            any(name.startswith("backbone.") for name in names)
            and any(name.startswith("decode_head.") for name in names)
        )
    if method == "lora":
        return all(name.endswith((".lora_a", ".lora_b")) for name in names)
    if method == "msre":
        return all(
            name.startswith(("backbone.target_msre.", "backbone.target_head_delta."))
            for name in names
        )
    return False


def inspect_model(config: Config, method: str, state_dict) -> dict:
    model = MODELS.build(config.model)
    missing, unexpected = load_report(model, state_dict)
    total, trainable, names = parameter_report(model)
    gates = {
        "checkpoint_no_unexpected_keys": not unexpected,
        "checkpoint_missing_only_declared_new_tensors": expected_missing(method, missing),
        "trainable_scope_exact": scope_gate(method, names, trainable),
    }
    if method == "msre":
        gates["msre_trainable_le_500k"] = trainable <= 500_000
    row = {
        "method": method,
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_fraction": trainable / total,
        "trainable_names": names,
        "checkpoint_missing_keys": missing,
        "checkpoint_unexpected_keys": unexpected,
        "gates": gates,
        "passed": all(gates.values()),
    }
    del model
    gc.collect()
    return row


def audit(checkpoint_path: Path) -> dict:
    register_all_modules(init_default_scope=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint has no state_dict")

    rows = []
    source_config = Config.fromfile(SOURCE_CONFIG)
    rows.append(inspect_model(source_config, "source_parent", state_dict))
    for method in METHODS:
        with scoped_environment(
            {
                "PHASE64_TARGET": "sparcs",
                "PHASE64_METHOD": method,
                "PHASE64_PARENT_CHECKPOINT": checkpoint_path.as_posix(),
            }
        ):
            config = Config.fromfile(TARGET_CONFIG)
        rows.append(inspect_model(config, method, state_dict))

    summary = {
        "phase": "64-day2-model-preflight",
        "checkpoint": checkpoint_path.as_posix(),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "models": rows,
    }
    summary["passed"] = all(row["passed"] for row in rows)
    summary["decision"] = (
        "proceed_to_data_preflight"
        if summary["passed"]
        else "stop_and_repair_model_matrix"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve(strict=True)
    try:
        checkpoint.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise ValueError("Checkpoint must remain inside the repository") from exc
    summary = audit(checkpoint)
    rendered = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output = args.output.resolve()
        try:
            output.relative_to(REPO_ROOT.resolve())
        except ValueError as exc:
            raise ValueError("Output must remain inside the repository") from exc
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
