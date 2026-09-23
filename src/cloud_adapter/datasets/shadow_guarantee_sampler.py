import itertools
import json
from pathlib import Path
from typing import Iterator, Optional, Sized

import torch
from mmengine.dist import get_dist_info, sync_random_seed
from mmengine.registry import DATA_SAMPLERS
from torch.utils.data import Sampler


@DATA_SAMPLERS.register_module()
class ShadowGuaranteedInfiniteSampler(Sampler):
    """Yield one labeled shadow patch in every local training batch."""

    def __init__(
        self,
        dataset: Sized,
        diagnosis_path: str,
        batch_size: int = 4,
        seed: Optional[int] = None,
    ) -> None:
        self.dataset = dataset
        self.size = len(dataset)
        self.batch_size = int(batch_size)
        if self.batch_size < 2:
            raise ValueError("batch_size must be at least two")
        diagnosis = json.loads(Path(diagnosis_path).read_text(encoding="utf-8"))
        shadow_names = set(diagnosis["selected_1pct_labels"]["shadow_image_names"])
        self.shadow_indices = [
            index for index in range(self.size)
            if Path(dataset.get_data_info(index)["img_path"]).name in shadow_names
        ]
        matched_names = {
            Path(dataset.get_data_info(index)["img_path"]).name
            for index in self.shadow_indices
        }
        if matched_names != shadow_names:
            raise ValueError("Shadow diagnosis does not match selected dataset ordering")
        if not self.shadow_indices:
            raise ValueError("At least one shadow-containing patch is required")
        self.rank, self.world_size = get_dist_info()
        self.seed = sync_random_seed() if seed is None else int(seed)
        self.indices = self._infinite_indices()

    def _infinite_indices(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.rank)
        shadow = torch.tensor(self.shadow_indices, dtype=torch.long)
        while True:
            offset = torch.randint(len(shadow), (), generator=generator).item()
            yield int(shadow[offset])
            for _ in range(self.batch_size - 1):
                yield int(torch.randint(self.size, (), generator=generator).item())

    def __iter__(self) -> Iterator[int]:
        yield from self.indices

    def __len__(self) -> int:
        return self.size

    def set_epoch(self, epoch: int) -> None:
        del epoch
