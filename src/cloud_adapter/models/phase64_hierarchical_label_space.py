"""Phase 64 hierarchy-consistent heads for heterogeneous label spaces.

The parent distribution is shared across datasets.  Every dataset owns
conditional child heads, so native leaf probabilities are constructed as

    P_d(y | x) = P(parent(y) | x) P_d(y | parent(y), x).

The grouped leaf probability therefore equals the parent probability by
construction; no tunable consistency penalty is required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


PARENT_CLASSES = ("surface_visible", "cloud", "shadow")


@dataclass(frozen=True)
class DatasetHierarchySpec:
    dataset_id: str
    native_classes: tuple[str, ...]
    parent_by_native: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.dataset_id:
            raise ValueError("dataset_id must not be empty")
        if not self.native_classes:
            raise ValueError("native_classes must not be empty")
        if len(self.native_classes) != len(self.parent_by_native):
            raise ValueError("native_classes and parent_by_native lengths differ")
        if len(set(self.native_classes)) != len(self.native_classes):
            raise ValueError("native class names must be unique")
        if any(index not in range(len(PARENT_CLASSES)) for index in self.parent_by_native):
            raise ValueError("parent indices must be in [0, 2]")
        missing = set(range(len(PARENT_CLASSES))) - set(self.parent_by_native)
        if missing:
            raise ValueError(f"every dataset must observe all parents; missing={sorted(missing)}")


PHASE64_HIERARCHIES = {
    "cloudsen12_high": DatasetHierarchySpec(
        dataset_id="cloudsen12_high",
        native_classes=("clear", "thick_cloud", "thin_cloud", "cloud_shadow"),
        parent_by_native=(0, 1, 1, 2),
    ),
    "l8_biome": DatasetHierarchySpec(
        dataset_id="l8_biome",
        native_classes=("clear", "cloud_shadow", "thin_cloud", "cloud"),
        parent_by_native=(0, 2, 1, 1),
    ),
    "deep_fmask": DatasetHierarchySpec(
        dataset_id="deep_fmask",
        native_classes=("clear_sky_land", "cloud", "shadow", "snow", "water"),
        parent_by_native=(0, 1, 2, 0, 0),
    ),
    "sparcs": DatasetHierarchySpec(
        dataset_id="sparcs",
        native_classes=(
            "shadow", "shadow_over_water", "water", "snow", "land", "cloud", "flooded"
        ),
        parent_by_native=(2, 2, 0, 0, 0, 1, 0),
    ),
}


@dataclass
class HierarchicalLogProbabilities:
    parent_logits: Tensor
    parent_log_probs: Tensor
    child_logits: dict[str, Tensor]
    native_log_probs: Tensor


class HierarchicalLabelSpaceHead(nn.Module):
    """Shared three-parent head with dataset-specific conditional children."""

    def __init__(
        self,
        in_channels: int,
        specs: Mapping[str, DatasetHierarchySpec] | Sequence[DatasetHierarchySpec],
        *,
        kernel_size: int = 1,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        if isinstance(specs, Mapping):
            values = list(specs.values())
        else:
            values = list(specs)
        if not values:
            raise ValueError("at least one hierarchy spec is required")
        self.specs = {spec.dataset_id: spec for spec in values}
        if len(self.specs) != len(values):
            raise ValueError("dataset ids must be unique")
        padding = kernel_size // 2
        self.parent_head = nn.Conv2d(
            in_channels, len(PARENT_CLASSES), kernel_size, padding=padding
        )
        self.child_heads = nn.ModuleDict()
        for dataset_id, spec in self.specs.items():
            per_parent = nn.ModuleDict()
            for parent_index, parent_name in enumerate(PARENT_CLASSES):
                count = sum(index == parent_index for index in spec.parent_by_native)
                if count > 1:
                    per_parent[parent_name] = nn.Conv2d(
                        in_channels, count, kernel_size, padding=padding
                    )
            self.child_heads[dataset_id] = per_parent

    def forward(self, features: Tensor, dataset_id: str) -> HierarchicalLogProbabilities:
        if features.ndim != 4:
            raise ValueError("features must have shape (N, C, H, W)")
        if dataset_id not in self.specs:
            raise KeyError(f"Unknown dataset_id {dataset_id!r}")
        spec = self.specs[dataset_id]
        parent_logits = self.parent_head(features)
        parent_log_probs = F.log_softmax(parent_logits, dim=1)
        child_logits = {}
        native_logs: list[Tensor | None] = [None] * len(spec.native_classes)
        for parent_index, parent_name in enumerate(PARENT_CLASSES):
            native_indices = [
                index for index, value in enumerate(spec.parent_by_native)
                if value == parent_index
            ]
            if len(native_indices) == 1:
                conditional = torch.zeros_like(parent_log_probs[:, parent_index : parent_index + 1])
            else:
                logits = self.child_heads[dataset_id][parent_name](features)
                child_logits[parent_name] = logits
                conditional = F.log_softmax(logits, dim=1)
            for child_index, native_index in enumerate(native_indices):
                native_logs[native_index] = (
                    parent_log_probs[:, parent_index] + conditional[:, child_index]
                )
        if any(value is None for value in native_logs):  # pragma: no cover - constructor guards
            raise RuntimeError("Incomplete native probability construction")
        native_log_probs = torch.stack(native_logs, dim=1)  # type: ignore[arg-type]
        return HierarchicalLogProbabilities(
            parent_logits=parent_logits,
            parent_log_probs=parent_log_probs,
            child_logits=child_logits,
            native_log_probs=native_log_probs,
        )


def grouped_native_probabilities(native_log_probs: Tensor, spec: DatasetHierarchySpec) -> Tensor:
    if native_log_probs.ndim != 4 or native_log_probs.shape[1] != len(spec.native_classes):
        raise ValueError("native_log_probs shape does not match the hierarchy spec")
    native = native_log_probs.exp()
    grouped = native.new_zeros((native.shape[0], len(PARENT_CLASSES), *native.shape[2:]))
    for native_index, parent_index in enumerate(spec.parent_by_native):
        grouped[:, parent_index] += native[:, native_index]
    return grouped


def hierarchy_consistency_error(
    output: HierarchicalLogProbabilities, spec: DatasetHierarchySpec
) -> Tensor:
    grouped = grouped_native_probabilities(output.native_log_probs, spec)
    return (grouped - output.parent_log_probs.exp()).abs().amax()


def native_nll_loss(
    output: HierarchicalLogProbabilities,
    target: Tensor,
    *,
    ignore_index: int = 255,
) -> Tensor:
    return F.nll_loss(output.native_log_probs, target.long(), ignore_index=ignore_index)


def parent_cross_entropy_loss(
    output: HierarchicalLogProbabilities,
    target: Tensor,
    *,
    ignore_index: int = 255,
) -> Tensor:
    return F.nll_loss(output.parent_log_probs, target.long(), ignore_index=ignore_index)


def parent_distillation_loss(
    output: HierarchicalLogProbabilities,
    teacher_parent_probs: Tensor,
    *,
    valid_mask: Tensor | None = None,
) -> Tensor:
    """KL anchor for source retention using a frozen source model."""

    if teacher_parent_probs.shape != output.parent_log_probs.shape:
        raise ValueError("teacher_parent_probs must match parent_log_probs")
    if not torch.isfinite(teacher_parent_probs).all():
        raise ValueError("teacher_parent_probs must be finite")
    if torch.any(teacher_parent_probs < 0):
        raise ValueError("teacher_parent_probs must be non-negative")
    normalizer = teacher_parent_probs.sum(dim=1, keepdim=True)
    if torch.any(normalizer <= 0):
        raise ValueError("teacher probabilities must have positive mass")
    teacher = teacher_parent_probs / normalizer
    per_pixel = F.kl_div(output.parent_log_probs, teacher, reduction="none").sum(dim=1)
    if valid_mask is None:
        return per_pixel.mean()
    if valid_mask.shape != per_pixel.shape:
        raise ValueError("valid_mask shape must match the spatial KL map")
    valid = valid_mask.to(dtype=torch.bool)
    if not torch.any(valid):
        raise ValueError("valid_mask selects no pixels")
    return per_pixel[valid].mean()
