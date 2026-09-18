import torch
import torch.nn.functional as F


def label_boundaries(labels, valid, radius):
    if labels.ndim != 3 or valid.shape != labels.shape:
        raise ValueError("labels and valid must have matching BHW shapes")
    if radius < 0:
        raise ValueError("radius must be non-negative")
    boundary = torch.zeros_like(valid, dtype=torch.bool)

    horizontal = (
        (labels[:, :, 1:] != labels[:, :, :-1])
        & valid[:, :, 1:]
        & valid[:, :, :-1]
    )
    boundary[:, :, 1:] |= horizontal
    boundary[:, :, :-1] |= horizontal

    vertical = (
        (labels[:, 1:, :] != labels[:, :-1, :])
        & valid[:, 1:, :]
        & valid[:, :-1, :]
    )
    boundary[:, 1:, :] |= vertical
    boundary[:, :-1, :] |= vertical

    if radius:
        kernel = 2 * radius + 1
        boundary = F.max_pool2d(
            boundary.unsqueeze(1).float(),
            kernel_size=kernel,
            stride=1,
            padding=radius,
        ).squeeze(1).bool()
    return boundary & valid


def weighted_pixel_kl(
    pixel_kl,
    labels,
    valid,
    class_weights,
    boundary_weight,
    boundary_radius,
):
    if pixel_kl.shape != labels.shape or valid.shape != labels.shape:
        raise ValueError("pixel_kl, labels, and valid must have matching BHW shapes")
    if class_weights.ndim != 1:
        raise ValueError("class_weights must be one-dimensional")
    if not valid.any():
        return pixel_kl.mean() * 0.0

    safe_labels = labels.clamp(0, len(class_weights) - 1)
    weights = class_weights[safe_labels].to(
        device=pixel_kl.device, dtype=pixel_kl.dtype
    )
    boundaries = label_boundaries(labels, valid, boundary_radius)
    weights = weights * (1.0 + boundary_weight * boundaries.to(weights.dtype))
    weights = weights * valid.to(weights.dtype)
    return (pixel_kl * weights).sum() / weights.sum().clamp_min(1.0)
