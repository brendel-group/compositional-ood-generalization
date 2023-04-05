from typing import List

import torch

from utils import get_digit_subscript


def sample_latents(
    n_samples: int, dim_per_slot: List[int], mode: str, **kwargs
) -> torch.Tensor:
    if mode == "random":
        z = _sample_random(n_samples, dim_per_slot)
    elif mode == "orthogonal":
        z = _sample_orthogonal(n_samples, dim_per_slot)
    else:
        raise ValueError(f"Unsupported sample mode '{mode}'.")

    return z


def _sample_random(n_samples: int, dim_per_slot: List[int]) -> torch.Tensor:
    z = torch.rand(n_samples, sum(dim_per_slot))
    return z


def _sample_orthogonal(n_samples: int, dim_per_slot: List[int]) -> torch.Tensor:
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
