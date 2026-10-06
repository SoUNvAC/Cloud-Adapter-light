"""Standard LoRA and full-fine-tuning DINOv2 baselines for Phase 64."""

from __future__ import annotations

from torch import nn

from mmseg.models.builder import BACKBONES

from .cloud_adapter_dinov2 import CloudAdapterDinoVisionTransformer
from .phase64_lora_layers import LoRALinear


@BACKBONES.register_module()
class Phase64LoRACloudAdapterDinoVisionTransformer(CloudAdapterDinoVisionTransformer):
    """Freeze the source path and add LoRA to DINO attention/MLP linears."""

    def __init__(
        self,
        lora_rank: int = 4,
        lora_alpha: float = 4.0,
        lora_dropout: float = 0.0,
        lora_targets=("qkv", "proj", "fc1", "fc2"),
        lora_enabled: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(lora_alpha)
        self.lora_targets = tuple(lora_targets)
        self.lora_enabled = bool(lora_enabled)
        allowed = {"qkv", "proj", "fc1", "fc2"}
        if not self.lora_targets or not set(self.lora_targets).issubset(allowed):
            raise ValueError(f"lora_targets must be a non-empty subset of {sorted(allowed)}")
        for block in self.blocks:
            if not hasattr(block, "attn") or not hasattr(block, "mlp"):
                continue
            for owner, name in (
                (block.attn, "qkv"),
                (block.attn, "proj"),
                (block.mlp, "fc1"),
                (block.mlp, "fc2"),
            ):
                if name in self.lora_targets:
                    linear = getattr(owner, name)
                    if not isinstance(linear, nn.Linear):
                        raise TypeError(f"Expected nn.Linear at {name}, got {type(linear)}")
                    setattr(
                        owner,
                        name,
                        LoRALinear.from_linear(
                            linear,
                            rank=self.lora_rank,
                            alpha=self.lora_alpha,
                            dropout=lora_dropout,
                        ),
                    )
        self.set_lora_enabled(self.lora_enabled)

    def set_lora_enabled(self, enabled: bool) -> None:
        """Enable the target low-rank residual without merging base weights."""
        self.lora_enabled = bool(enabled)
        for module in self.modules():
            if isinstance(module, LoRALinear):
                module.enabled = self.lora_enabled

    def train(self, mode: bool = True):
        nn.Module.train(self, mode)
        for parameter in self.parameters():
            parameter.requires_grad = False
        if mode:
            for module in self.modules():
                if isinstance(module, LoRALinear):
                    module.lora_a.requires_grad = True
                    module.lora_b.requires_grad = True
            self.cloud_adapter.eval()
        return self


@BACKBONES.register_module()
class Phase64FullFineTuneCloudAdapterDinoVisionTransformer(
    CloudAdapterDinoVisionTransformer
):
    """Explicitly unfreeze DINOv2 and Cloud-Adapter for the capacity baseline."""

    def train(self, mode: bool = True):
        nn.Module.train(self, mode)
        for parameter in self.parameters():
            parameter.requires_grad = bool(mode)
        return self
