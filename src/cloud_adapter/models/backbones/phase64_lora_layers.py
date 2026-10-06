"""Minimal standard LoRA linear layer used by the Phase 64 baseline."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class LoRALinear(nn.Linear):
    """Drop-in ``nn.Linear`` with frozen base weight and low-rank residual.

    The base parameter names remain ``weight`` and ``bias`` so an ordinary
    DINOv2 checkpoint loads without key rewriting.  ``lora_b`` starts at zero,
    making replacement an exact identity at initialization.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rank: int,
        alpha: float,
        bias: bool,
        dropout: float = 0.0,
    ) -> None:
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        if alpha <= 0:
            raise ValueError("LoRA alpha must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0, 1)")
        super().__init__(in_features, out_features, bias=bias)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.enabled = True
        self.lora_dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.lora_a = nn.Parameter(torch.empty(self.rank, in_features))
        self.lora_b = nn.Parameter(torch.zeros(out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

    @classmethod
    def from_linear(
        cls, linear: nn.Linear, *, rank: int, alpha: float, dropout: float = 0.0
    ) -> "LoRALinear":
        result = cls(
            linear.in_features,
            linear.out_features,
            rank=rank,
            alpha=alpha,
            bias=linear.bias is not None,
            dropout=dropout,
        )
        with torch.no_grad():
            result.weight.copy_(linear.weight)
            if linear.bias is not None:
                result.bias.copy_(linear.bias)
        return result.to(device=linear.weight.device, dtype=linear.weight.dtype)

    def forward(self, inputs: Tensor) -> Tensor:
        base = F.linear(inputs, self.weight, self.bias)
        if not self.enabled:
            return base
        low_rank = F.linear(F.linear(self.lora_dropout(inputs), self.lora_a), self.lora_b)
        return base + self.scaling * low_rank


def lora_parameter_count(module: nn.Module) -> int:
    return sum(
        parameter.numel()
        for name, parameter in module.named_parameters()
        if "lora_a" in name or "lora_b" in name
    )
