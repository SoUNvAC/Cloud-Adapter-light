import torch
import torch.nn.functional as F
from torch import nn


CONDITIONS = (
    "rgb",
    "rgb_nir",
    "rgb_nir_swir",
    "rgb_nir_swir_solar",
    "rgb_nir_swir_solar_cloud",
)


def condition_mask(condition, *, device=None, dtype=None):
    """Return the fixed mask for NIR, SWIR1/2, four solar and cloud slots."""
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown Phase 54 condition: {condition}")
    active = CONDITIONS.index(condition)
    masks = (
        (0, 0, 0, 0, 0, 0, 0, 0),
        (1, 0, 0, 0, 0, 0, 0, 0),
        (1, 1, 1, 0, 0, 0, 0, 0),
        (1, 1, 1, 1, 1, 1, 1, 0),
        (1, 1, 1, 1, 1, 1, 1, 1),
    )
    return torch.tensor(masks[active], device=device, dtype=dtype)


def neighbouring_cloud_probability(cloud_probability, outer=15, inner=3):
    """Annular mean that excludes the local centre from cloud context."""
    if cloud_probability.ndim != 4 or cloud_probability.shape[1] != 1:
        raise ValueError("cloud_probability must have shape N,1,H,W")
    if outer <= inner or outer % 2 == 0 or inner % 2 == 0:
        raise ValueError("outer/inner must be odd and outer > inner")
    outer_sum = F.avg_pool2d(
        cloud_probability, outer, stride=1, padding=outer // 2,
        count_include_pad=True,
    ) * float(outer * outer)
    inner_sum = F.avg_pool2d(
        cloud_probability, inner, stride=1, padding=inner // 2,
        count_include_pad=True,
    ) * float(inner * inner)
    # Border windows contain fewer valid cells. Compute their exact counts.
    ones = torch.ones_like(cloud_probability)
    outer_count = F.avg_pool2d(
        ones, outer, stride=1, padding=outer // 2, count_include_pad=True,
    ) * float(outer * outer)
    inner_count = F.avg_pool2d(
        ones, inner, stride=1, padding=inner // 2, count_include_pad=True,
    ) * float(inner * inner)
    return (outer_sum - inner_sum) / (outer_count - inner_count).clamp_min(1.0)


def apply_shadow_residual(base_probabilities, residual, shadow_index=3):
    """Change shadow odds while retaining all non-shadow conditional ratios."""
    eps = torch.finfo(base_probabilities.dtype).eps
    shadow = base_probabilities[:, shadow_index : shadow_index + 1].clamp(eps, 1 - eps)
    shadow_logit = torch.logit(shadow) + residual
    new_shadow = torch.sigmoid(shadow_logit)
    nonshadow = base_probabilities.clone()
    nonshadow[:, shadow_index] = 0
    ratios = nonshadow / nonshadow.sum(dim=1, keepdim=True).clamp_min(eps)
    result = ratios * (1 - new_shadow)
    result[:, shadow_index : shadow_index + 1] = new_shadow
    return result


class FixedShadowProbe(nn.Module):
    """Equal-budget probe used by every Phase 54 information condition."""

    def __init__(self, feature_channels=384, auxiliary_channels=8, width=32):
        super().__init__()
        channels = feature_channels + auxiliary_channels
        self.input_projection = nn.Conv2d(channels, width, 1)
        self.context = nn.Conv2d(
            width, width, 3, padding=3, dilation=3, groups=width
        )
        self.output = nn.Conv2d(width, 1, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, feature, auxiliary, condition):
        if auxiliary.shape[1] != 8:
            raise ValueError("Phase 54 auxiliary tensor must have eight slots")
        mask = condition_mask(
            condition, device=auxiliary.device, dtype=auxiliary.dtype
        ).view(1, 8, 1, 1)
        inputs = torch.cat((feature, auxiliary * mask), dim=1)
        hidden = F.gelu(self.input_projection(inputs))
        hidden = F.gelu(self.context(hidden))
        return self.output(hidden)
