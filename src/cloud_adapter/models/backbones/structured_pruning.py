def resolve_active_block_indices(
    num_blocks,
    active_block_indices,
    adapter_indices,
    output_indices,
):
    """Validate a static DINO block subset without importing PyTorch.

    Adapter and output locations must stay active because they define the four
    feature taps consumed by Cloud-Adapter and the segmentation decoder.
    """
    if num_blocks <= 0:
        raise ValueError("num_blocks must be positive")
    if active_block_indices is None:
        return tuple(range(num_blocks))

    active = tuple(int(index) for index in active_block_indices)
    if not active:
        raise ValueError("active_block_indices must not be empty")
    if tuple(sorted(set(active))) != active:
        raise ValueError(
            "active_block_indices must be strictly increasing and contain no duplicates"
        )
    invalid = [index for index in active if index < 0 or index >= num_blocks]
    if invalid:
        raise ValueError(
            f"active_block_indices contains indices outside [0, {num_blocks}): {invalid}"
        )

    required = set(int(index) for index in adapter_indices) | set(
        int(index) for index in output_indices
    )
    missing = sorted(required - set(active))
    if missing:
        raise ValueError(
            "Every adapter/output block must remain active; missing indices: "
            f"{missing}"
        )
    return active
