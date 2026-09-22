import itertools
import json
from pathlib import Path
from typing import Iterator, Optional, Sized

import torch
from mmengine.dataset import DATA_SAMPLERS
from mmengine.dist import get_dist_info, sync_random_seed
from torch.utils.data import Sampler


@DATA_SAMPLERS.register_module()
class WeakPrevalenceInfiniteSampler(Sampler):
    """Infinite deterministic mixture of all samples and a weak-rich pool."""

    def __init__(
        self,
        dataset: Sized,
        manifest_path: str,
        rich_probability: float = 0.5,
        seed: Optional[int] = None,
    ) -> None:
        if not 0.0 <= rich_probability <= 1.0:
            raise ValueError("rich_probability must be in [0, 1]")
        self.dataset = dataset
        self.size = len(dataset)
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        if manifest.get("sample_count") != self.size:
            raise ValueError("Sampling manifest does not match dataset length")
        ordered_paths = manifest.get("ordered_paths", [])
        observed_paths = [
            Path(dataset.get_data_info(index)["seg_map_path"]).name
            for index in range(self.size)
        ]
        if ordered_paths != observed_paths:
            raise ValueError("Sampling manifest does not match dataset ordering")
        rich = [int(index) for index in manifest.get("rich_indices", [])]
        if not rich or min(rich) < 0 or max(rich) >= self.size:
            raise ValueError("Sampling manifest has invalid rich indices")
        if len(set(rich)) != len(rich):
            raise ValueError("Sampling manifest has duplicate rich indices")
        self.rich_indices = torch.tensor(rich, dtype=torch.long)
        self.rich_probability = float(rich_probability)
        self.rank, self.world_size = get_dist_info()
        self.seed = sync_random_seed() if seed is None else int(seed)
        self.indices = itertools.islice(
            self._infinite_indices(), self.rank, None, self.world_size
        )

    def _infinite_indices(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed)
        while True:
            choose_rich = torch.rand((), generator=generator).item()
            if choose_rich < self.rich_probability:
                offset = torch.randint(
                    len(self.rich_indices), (), generator=generator
                ).item()
                yield int(self.rich_indices[offset])
            else:
                yield int(torch.randint(self.size, (), generator=generator).item())

    def __iter__(self) -> Iterator[int]:
        yield from self.indices

    def __len__(self) -> int:
        return self.size

    def set_epoch(self, epoch: int) -> None:
        del epoch
