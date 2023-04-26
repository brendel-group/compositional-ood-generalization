import warnings
from math import sqrt
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import torch
import torch.nn as nn

import models
from models import CompositionalFunction
from utils import all_equal


# TODO clean up the redundancy in this file, e.g. "gaps" duplicates the same behavior
# in multiple functions
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
    elif mode == "random_gap":
        z = _sample_random_with_gap(n_samples, dim_per_slot, **kwargs)
    elif mode == "orthogonal":
        z = _sample_orthogonal(n_samples, dim_per_slot)
    elif mode == "orthogonal_gap":
        z = _sample_orthogonal_with_gap(n_samples, dim_per_slot, **kwargs)
    elif mode == "diagonal":
        z = _sample_diagonal(n_samples, dim_per_slot, **kwargs)
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


def _sample_random_with_gap(
    n_samples: int, dim_per_slot: List[int], gaps: List[Tuple[int, float, float]]
) -> torch.Tensor:
    """Sample randomly in unit cube with gaps along any dimension(s) specified by (dim, start, stop)."""
    total_dim = sum(dim_per_slot)
    for dim, start, stop in gaps:
        assert dim in range(
            total_dim
        ), f"Gap dimension must be in range({total_dim}), but got {dim}."
        assert (
            start < stop
        ), f"Gap start must be smaller than stop, but got [{start}, {stop}] for dim {dim}."
        assert (
            start >= 0 and stop <= 1
        ), f"Gap edges must be in [0, 1], but got [{start}, {stop}] for dim {dim}."
        assert stop - start < 1, f"Gap can't span entire range [0, 1] for dim {dim}."

    z = torch.Tensor(0, total_dim)

    while z.shape[0] < n_samples:
        # first sample normal randomly
        _z = _sample_random(n_samples, dim_per_slot)

        # then reject points
        mask = torch.ones(n_samples)
        for dim, start, stop in gaps:
            _mask = torch.logical_or(_z[:, dim] < start, _z[:, dim] > stop)
            mask = torch.logical_and(mask, _mask)

        idx = mask.nonzero().squeeze(1)
        z = torch.cat([z, _z[idx]])

    return z[:n_samples]


def _sample_orthogonal(n_samples: int, dim_per_slot: List[int]) -> torch.Tensor:
    """Sample randomly within each individual slot while keeping all other slots 0."""
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


def _sample_orthogonal_with_gap(
    n_samples: int, dim_per_slot: List[int], gaps: List[Tuple[int, float, float]]
) -> torch.Tensor:
    """Sample Orthogonally with gaps along any dimension(s) specified by (dim, start, stop)."""
    total_dim = sum(dim_per_slot)
    for dim, start, stop in gaps:
        assert dim in range(
            total_dim
        ), f"Gap dimension must be in range({total_dim}), but got {dim}."
        assert (
            start < stop
        ), f"Gap start must be smaller than stop, but got [{start}, {stop}] for dim {dim}."
        assert (
            start >= 0 and stop <= 1
        ), f"Gap edges must be in [0, 1], but got [{start}, {stop}] for dim {dim}."
        assert stop - start < 1, f"Gap can't span entire range [0, 1] for dim {dim}."

    z = torch.Tensor(0, total_dim)

    while z.shape[0] < n_samples:
        # first sample normal orthogonal
        _z = _sample_orthogonal(n_samples, dim_per_slot)

        # then reject points
        mask = torch.ones(n_samples)
        for dim, start, stop in gaps:
            _mask = torch.logical_or(_z[:, dim] < start, _z[:, dim] > stop)
            mask = torch.logical_and(mask, _mask)

        idx = mask.nonzero().squeeze(1)
        z = torch.cat([z, _z[idx]])

    return z[:n_samples]


def _sample_diagonal(
    n_samples: int, dim_per_slot: List[int], delta: float
) -> torch.Tensor:
    total_dim = sum(dim_per_slot)
    max_delta = sqrt(2) / 2
    assert (
        delta > 0 and delta < max_delta
    ), f"Delta must be in [0, {max_delta}], but got {delta}."

    if not all_equal(dim_per_slot):
        raise NotImplementedError(
            "Diagonal sampling undefined for slots with different dimensions."
        )
    n_slots = len(dim_per_slot)
    dim_per_slot = dim_per_slot[0]

    z = torch.Tensor(0, total_dim)

    while z.shape[0] < n_samples:
        # sample randomly on diagonal
        z_in_slot = torch.rand(n_samples, dim_per_slot)
        z_on_diag = z_in_slot.unsqueeze(1).repeat(1, n_slots, 1)

        # sample noise from n_slots-ball
        noise = torch.randn(n_samples, n_slots + 2, dim_per_slot)
        noise = noise / torch.norm(noise, dim=1, keepdim=True)  # points on n-sphere
        noise = noise[:, :n_slots, :]  # remove two last points

        # project to hyperplane perpendicular to diagonal
        # this yields a random direction orthogonal to the diagonal
        ort_vec = noise - z_on_diag * (noise * z_on_diag).sum(
            axis=1, keepdim=True
        ) / z_on_diag.pow(2).sum(axis=1, keepdim=True)
        ort_vec /= torch.norm(ort_vec, p=2, dim=1, keepdim=True)

        # get the points within radius delta around the diagonal
        _z = (
            z_on_diag
            + ort_vec
            * torch.pow(torch.rand([n_samples, 1, dim_per_slot]), 1 / (n_slots - 1))
            * delta
        )

        # only keep samples inside the unit-cube
        mask = ((_z - 0.5).abs() <= 0.5).flatten(1).all(1)
        idx = mask.nonzero().squeeze(1)

        z = torch.cat([z, _z[idx].flatten(1)])

    return z[:n_samples]


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

    coords = torch.arange(grid_size) / (grid_size - 1)

    # if a specific face is specified only return those points
    if dims is not None:
        assert dims[0] != dims[1], "Dimensions can't be the same."

        z = torch.zeros(n_gridpoints, total_dim)
        z[:, dims[0]] = coords.repeat(grid_size)
        z[:, dims[1]] = coords.repeat_interleave(grid_size)

        return z

    # otherwise return all faces
    n_faces = total_dim * (total_dim - 1) // 2
    z = torch.zeros(n_faces * n_gridpoints, total_dim)

    for dim1 in range(total_dim):
        for dim2 in range(dim1):
            face_idx = dim1 + dim2
            start = face_idx * n_gridpoints
            stop = start + n_gridpoints
            z[start:stop, dim1] = coords.repeat(grid_size)
            z[start:stop, dim2] = coords.repeat_interleave(grid_size)

    return z


def _sample_grid(
    n_samples: int, dim_per_slot: List[int], grid_size: int = 10
) -> torch.Tensor:
    """Sample on the grid in the unit cube"""
    z = torch.rand(n_samples, sum(dim_per_slot))
    z = (z * grid_size - 0.5).round() / (grid_size - 1)
    return z


class Dataset(torch.utils.data.TensorDataset):
    """Simple Dataset with fixed number of samples."""

    def __init__(
        self,
        generator: CompositionalFunction,
        dev: torch.device,
        transform: Union[callable, str] = None,
        load: Path = None,
        **kwargs,
    ):
        if load is not None:
            warnings.warn("Loading pregenerated dataset, ignoring generator settings.")
            self.tensors = torch.load(load)

        else:
            z = sample_latents(generator.d_in, **kwargs).to(dev)

            if isinstance(transform, str):
                transform = getattr(models, transform)

            if transform is not None:
                z = transform(z)

            with torch.no_grad():
                x = generator(z)
            super().__init__(x, z)


class InfiniteDataset(torch.utils.data.IterableDataset):
    """Dataset that draws new samples every epoch."""

    def __init__(
        self,
        generator: CompositionalFunction,
        dev: torch.device,
        transform: Union[callable, str] = None,
        **kwargs,
    ):
        super().__init__()
        self.generator = generator
        self.kwargs = kwargs
        self.dev = dev
        if isinstance(transform, str):
            self.transform = getattr(models, transform)
        else:
            self.transform = transform

    def __iter__(self):
        # can't handle multiple workers atm
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            raise NotImplementedError

        self.reset()
        return iter(zip(self.x, self.z))

    def reset(self):
        self.z = sample_latents(self.generator.d_in, **self.kwargs).to(self.dev)

        if self.transform is not None:
            self.z = self.transform(self.z)

        with torch.no_grad():
            self.x = self.generator(self.z)


class BatchDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset: torch.utils.data.Dataset, batch_size: int):
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=None if isinstance(dataset, InfiniteDataset) else True,
        )


def get_dataloader(
    generator: nn.Module,
    dev: torch.device,
    resample: bool = False,
    batch_size: int = 10000,
    **kwargs,
) -> torch.utils.data.DataLoader:
    if resample:
        assert (
            "load" not in kwargs
        ), "Resampling is incompatible with pregenerated data."
        data_set = InfiniteDataset(generator, dev, **kwargs)
    else:
        data_set = Dataset(generator, dev, **kwargs)

    return BatchDataLoader(data_set, batch_size)


def get_dataloaders(
    generator: nn.Module,
    dev: torch.device,
    cfg: Dict[str, Any],
) -> Union[torch.utils.data.DataLoader, Dict[str, torch.utils.data.DataLoader]]:
    assert not generator.training, "Generator has to be in eval() mode!"

    if isinstance(cfg, dict):
        ldrs = {}
        for name, _cfg in cfg.items():
            ldrs[name] = get_dataloader(generator, dev, **_cfg)
        return ldrs
    else:
        return get_dataloader(generator, dev, **cfg)
