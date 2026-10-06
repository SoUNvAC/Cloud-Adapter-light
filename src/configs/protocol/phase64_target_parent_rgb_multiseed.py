"""Phase 64 RGB MsRE repeat with the frozen seed-64 source checkpoint."""

_base_ = ["./phase64_target_parent_rgb.py"]

import os


target = os.environ.get("PHASE64_TARGET", "").strip().lower()
method = os.environ.get("PHASE64_METHOD", "").strip().lower()
seed_text = os.environ.get("PHASE64_SEED", "").strip()
if target not in {"l8", "sparcs"}:
    raise ValueError("PHASE64_TARGET must be l8 or sparcs")
if method != "msre":
    raise ValueError("Phase 64 RGB repeat is restricted to MsRE")
if seed_text not in {"65", "66"}:
    raise ValueError("PHASE64_SEED must be 65 or 66")

seed = int(seed_text)
randomness = dict(seed=seed, deterministic=True)
work_dir = f"./work_dirs/phase64_target_parent/{target}/msre/seed{seed}"
