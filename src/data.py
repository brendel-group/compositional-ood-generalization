from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

from models import CompositionalFunction
from utils import get_digit_subscript


def sample_latents(
    dim_per_slot: List[int], mode: str = "random", n_samples: int = None, **kwargs
) -> torch.Tensor:
    if mode not in [
        "full_grid",
        "face_grid",
    ]:
        assert n_samples is not None, "Number of samples must be specified."

    if mode == "random":
        z = _sample_random(n_samples, dim_per_slot)
    elif mode == "orthogonal":
        z = _sample_orthogonal(n_samples, dim_per_slot)
    elif mode == "grid":
        z = _sample_grid(n_samples, dim_per_slot, **kwargs)
    elif mode == "full_grid":
        z = _get_grid(dim_per_slot, **kwargs)
    elif mode == "face_grid":
        z = _get_face_grids(dim_per_slot, **kwargs)
    else:
        raise ValueError(f"Unsupported sample mode '{mode}'.")

    return z


def _sample_random(n_samples: int, dim_per_slot: List[int]) -> torch.Tensor:
    """Sample randomly in unit cube."""
    z = torch.rand(n_samples, sum(dim_per_slot))
    return z


def _sample_orthogonal(n_samples: int, dim_per_slot: List[int]) -> torch.Tensor:
    """Sample randomly along each edge of the unit cube that originates in the origin."""
    total_dim = sum(dim_per_slot)

    # sample randomly within each slot
    _z = torch.rand(n_samples, total_dim)

    # for each final sample, randomly pick one slot, all other slots are 0
    slots = torch.randint(len(dim_per_slot), (n_samples, 1)).squeeze()

    z = torch.zeros(n_samples, total_dim)
    for i, slot in enumerate(slots):
        start = sum(dim_per_slot[:slot])
        stop = sum(dim_per_slot[: slot + 1])
        z[i, start:stop] = _z[i, start:stop]

    return z


def _get_grid(dim_per_slot: List[int], grid_size: int = 10) -> torch.Tensor:
    """Get all grid points in the unit cube."""
    total_dim = sum(dim_per_slot)
    z = torch.zeros(int(pow(grid_size, total_dim)), total_dim)
    for dim in range(total_dim):
        coords = torch.arange(grid_size) / (grid_size - 1)
        digit_repeats = int(pow(grid_size, dim))
        sequence_repeats = int(pow(grid_size, total_dim - dim - 1))
        z[:, dim] = coords.repeat(sequence_repeats).repeat_interleave(digit_repeats)

    return z


def _get_face_grids(
    dim_per_slot: List[int], grid_size: int = 10, dims: Tuple[int, int] = None
) -> torch.Tensor:
    """Get all grid points on each face of the unit cube that intersects the origin."""
    total_dim = sum(dim_per_slot)
    n_gridpoints = grid_size * grid_size

    # if a specific face is specified only return those points
    if dims is not None:
        assert dims[0] != dims[1], "Dimensions can't be the same."

        z = torch.zeros(n_gridpoints, total_dim)
        z[:, dims[0]] = torch.arange(grid_size).repeat(grid_size)
        z[:, dims[1]] = torch.arange(grid_size).repeat_interleave(grid_size)

        return z

    # otherwise return all faces
    n_faces = total_dim * (total_dim - 1) // 2
    z = torch.zeros(n_faces * n_gridpoints, total_dim)

    for dim1 in range(total_dim):
        for dim2 in range(dim1):
            face_idx = dim1 + dim2
            start = face_idx * n_gridpoints
            stop = start + n_gridpoints
            z[start:stop, dim1] = torch.arange(grid_size).repeat(grid_size)
            z[start:stop, dim2] = torch.arange(grid_size).repeat_interleave(grid_size)

    return z


def _sample_grid(
    n_samples: int, dim_per_slot: List[int], grid_size: int = 10
) -> torch.Tensor:
    """Sample on the grid in the unit cube"""
    z = torch.rand(n_samples, sum(dim_per_slot))
    z = (z * (grid_size + 1) - 0.5).round() / grid_size
    return z


class Dataset(torch.utils.data.TensorDataset):
    """Simple Dataset with fixed number of samples."""

    def __init__(self, generator: CompositionalFunction, **kwargs):
        z = sample_latents(generator.d_in, **kwargs)
        x = generator(z).detach()
        super().__init__(x, z)


class InfiniteDataset(torch.utils.data.IterableDataset):
    """Dataset that draws new samples every epoch."""

    def __init__(self, generator: CompositionalFunction, **kwargs):
        super().__init__()
        self.generator = generator
        self.kwargs = kwargs

    def __iter__(self):
        # can't handle multiple workers atm
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            raise NotImplementedError

        self.reset()
        return iter(zip(self.x, self.z))

    def reset(self):
        self.z = sample_latents(self.generator.d_in, **self.kwargs)
        self.x = self.generator(self.z).detach()


class BatchDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset: torch.utils.data.Dataset, batch_size: int):
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=None if isinstance(dataset, InfiniteDataset) else True,
        )


def get_dataloaders(
    generator: nn.Module,
    train_cfg: Dict[str, Any],
    eval_cfg: Dict[str, Any],
) -> Tuple[torch.utils.data.DataLoader, Dict[str, torch.utils.data.DataLoader]]:
    train_set = InfiniteDataset(generator, **train_cfg["sample"])
    train_ldr = BatchDataLoader(train_set, train_cfg["batch_size"])

    eval_set_cfgs = eval_cfg["sample"]
    if not isinstance(eval_set_cfgs, list):
        eval_set_cfgs = [eval_set_cfgs]

    eval_ldrs = {}
    for eval_set_cfg in eval_set_cfgs:
        eval_set = Dataset(generator, **eval_set_cfg)
        eval_ldr = BatchDataLoader(eval_set, eval_cfg["batch_size"])

        eval_ldrs[eval_set_cfg["name"]] = eval_ldr
    
    return train_ldr, eval_ldrs