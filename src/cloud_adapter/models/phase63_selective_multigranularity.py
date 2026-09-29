"""Phase 63 prototype for selective, hierarchy-consistent prediction.

This module deliberately contains no learned risk estimator.  A caller supplies a
calibrated per-item or per-pixel risk score, so the prototype can be placed after
any four-class segmentation model (including, but not depending on, MsRE).
"""

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import Tensor, nn


# Frozen Phase22/52 model-output order. Phase50L8MappedDataset remaps target
# labels to this source order before supervision.
FINE_CLASSES = ("clear", "thick_cloud", "thin_cloud", "cloud_shadow")
THREE_CLASSES = ("clear", "cloud", "shadow")
BINARY_CLASSES = ("non_cloud", "cloud")

ROUTE_FINE = 0
ROUTE_COARSE = 1
ROUTE_ABSTAIN = 2


@dataclass(frozen=True)
class HierarchyProbabilities:
    """Consistent probabilities at the three Phase 63 granularities."""

    fine: Tensor
    three_class: Tensor
    binary: Tensor


@dataclass(frozen=True)
class SelectivePrediction:
    """Outputs retained by the prototype without collapsing their semantics."""

    probabilities: HierarchyProbabilities
    hierarchy_conflict: Tensor
    route: Tensor
    fine_prediction: Tensor
    coarse_prediction: Tensor


def _check_logits(logits: Tensor, classes: int, name: str) -> None:
    if logits.ndim < 2 or logits.shape[1] != classes:
        raise ValueError(
            f"{name} must have shape (N, {classes}, ...); got {tuple(logits.shape)}"
        )
    if not logits.is_floating_point():
        raise TypeError(f"{name} must be floating point")


def fine_logits_to_hierarchy(fine_logits: Tensor) -> HierarchyProbabilities:
    """Map source-order clear/thick/thin/shadow logits consistently.

    Shadow is non-cloud at the binary cloud-presence granularity.  Thus binary
    non-cloud is ``clear + shadow`` and binary cloud is ``thick + thin``.
    """

    _check_logits(fine_logits, 4, "fine_logits")
    fine = torch.softmax(fine_logits, dim=1)
    clear, thick, thin, shadow = fine.split(1, dim=1)
    cloud = thick + thin
    three_class = torch.cat((clear, cloud, shadow), dim=1)
    binary = torch.cat((clear + shadow, cloud), dim=1)
    return HierarchyProbabilities(
        fine=fine,
        three_class=three_class,
        binary=binary,
    )


def hierarchy_conflict_score(
    probabilities: HierarchyProbabilities,
    *,
    three_class_logits: Optional[Tensor] = None,
    binary_logits: Optional[Tensor] = None,
) -> Tensor:
    """Return per-location disagreement with optional independent coarse heads.

    The score is the largest absolute probability mismatch among all available
    hierarchy identities.  It is zero for probabilities obtained only by exact
    aggregation from the four-class prediction.
    """

    reference = probabilities.fine[:, 0]
    conflicts = [
        (probabilities.three_class[:, 1] - probabilities.binary[:, 1]).abs(),
        (
            probabilities.three_class[:, 0]
            + probabilities.three_class[:, 2]
            - probabilities.binary[:, 0]
        ).abs(),
    ]

    if three_class_logits is not None:
        _check_logits(three_class_logits, 3, "three_class_logits")
        if three_class_logits.shape[0] != reference.shape[0] or (
            three_class_logits.shape[2:] != reference.shape[1:]
        ):
            raise ValueError("three_class_logits must match fine_logits locations")
        three = torch.softmax(three_class_logits, dim=1)
        conflicts.extend(
            (three[:, index] - probabilities.three_class[:, index]).abs()
            for index in range(3)
        )

    if binary_logits is not None:
        _check_logits(binary_logits, 2, "binary_logits")
        if binary_logits.shape[0] != reference.shape[0] or (
            binary_logits.shape[2:] != reference.shape[1:]
        ):
            raise ValueError("binary_logits must match fine_logits locations")
        binary = torch.softmax(binary_logits, dim=1)
        conflicts.extend(
            (binary[:, index] - probabilities.binary[:, index]).abs()
            for index in range(2)
        )

    return torch.stack(conflicts, dim=0).amax(dim=0)


def route_from_calibrated_risk(
    risk_score: Tensor,
    *,
    fine_max_risk: float,
    coarse_max_risk: float,
) -> Tensor:
    """Route low/middle/high external risk to fine/coarse/abstain.

    Risk must be calibrated to [0, 1].  NaN or infinite values are treated as
    untrustworthy and routed to abstention rather than silently accepted.
    """

    if not 0.0 <= fine_max_risk <= coarse_max_risk <= 1.0:
        raise ValueError(
            "thresholds must satisfy 0 <= fine_max_risk <= coarse_max_risk <= 1"
        )
    if not risk_score.is_floating_point():
        raise TypeError("risk_score must be floating point")
    finite = torch.isfinite(risk_score)
    if torch.any(finite & ((risk_score < 0) | (risk_score > 1))):
        raise ValueError("finite risk_score values must lie in [0, 1]")

    route = torch.full_like(risk_score, ROUTE_ABSTAIN, dtype=torch.long)
    route = torch.where(
        finite & (risk_score <= coarse_max_risk),
        torch.full_like(route, ROUTE_COARSE),
        route,
    )
    route = torch.where(
        finite & (risk_score <= fine_max_risk),
        torch.full_like(route, ROUTE_FINE),
        route,
    )
    return route


def routing_coverage(route: Tensor, valid_mask: Optional[Tensor] = None) -> Dict[str, Tensor]:
    """Summarize fine, coarse and accepted coverage over valid locations."""

    if route.dtype != torch.long:
        raise TypeError("route must be a torch.long tensor")
    known = (route == ROUTE_FINE) | (route == ROUTE_COARSE) | (route == ROUTE_ABSTAIN)
    if not torch.all(known):
        raise ValueError("route contains an unknown route id")

    if valid_mask is None:
        valid = torch.ones_like(route, dtype=torch.bool)
    else:
        if valid_mask.shape != route.shape:
            raise ValueError("valid_mask must have the same shape as route")
        valid = valid_mask.to(dtype=torch.bool)
    denominator = valid.sum()
    if denominator.item() == 0:
        raise ValueError("coverage is undefined for an empty valid set")
    denominator = denominator.to(dtype=torch.float32)

    fine = ((route == ROUTE_FINE) & valid).sum().to(torch.float32) / denominator
    coarse = ((route == ROUTE_COARSE) & valid).sum().to(torch.float32) / denominator
    abstain = ((route == ROUTE_ABSTAIN) & valid).sum().to(torch.float32) / denominator
    return {
        "fine": fine,
        "coarse": coarse,
        "accepted": fine + coarse,
        "abstain": abstain,
    }


class SelectiveMultiGranularityHead(nn.Module):
    """Parameter-free Phase 63 head/router placed after any four-class model."""

    def __init__(self, fine_max_risk: float, coarse_max_risk: float) -> None:
        super().__init__()
        if not 0.0 <= fine_max_risk <= coarse_max_risk <= 1.0:
            raise ValueError(
                "thresholds must satisfy 0 <= fine_max_risk <= coarse_max_risk <= 1"
            )
        self.fine_max_risk = float(fine_max_risk)
        self.coarse_max_risk = float(coarse_max_risk)

    def forward(
        self,
        fine_logits: Tensor,
        risk_score: Tensor,
        *,
        three_class_logits: Optional[Tensor] = None,
        binary_logits: Optional[Tensor] = None,
    ) -> SelectivePrediction:
        probabilities = fine_logits_to_hierarchy(fine_logits)
        expected_risk_shape = (fine_logits.shape[0],) + tuple(fine_logits.shape[2:])
        if risk_score.ndim == fine_logits.ndim and risk_score.shape[1] == 1:
            risk_score = risk_score[:, 0]
        if tuple(risk_score.shape) != expected_risk_shape:
            raise ValueError(
                "risk_score must have shape (N, ...) or (N, 1, ...) matching logits"
            )
        route = route_from_calibrated_risk(
            risk_score,
            fine_max_risk=self.fine_max_risk,
            coarse_max_risk=self.coarse_max_risk,
        )
        conflict = hierarchy_conflict_score(
            probabilities,
            three_class_logits=three_class_logits,
            binary_logits=binary_logits,
        )
        return SelectivePrediction(
            probabilities=probabilities,
            hierarchy_conflict=conflict,
            route=route,
            fine_prediction=probabilities.fine.argmax(dim=1),
            coarse_prediction=probabilities.three_class.argmax(dim=1),
        )
