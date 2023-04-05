from typing import List

import torch
import torch.nn as nn

from utils import all_equal


class InvertibleMLP(nn.Sequential):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        d_hidden: int = 10,
        n_layers: int = 2,
        nonlin: nn.Module = nn.Tanh(),
        dtype: torch.dtype = torch.float32,
        init_dict=None,
    ):
        """Function class for synthetic phi_k."""
        super().__init__(
            nn.Linear(d_in, d_hidden, dtype=dtype),
        )
        self.d_in = d_in
        self.d_out = d_out

        for _ in range(n_layers - 2):
            self.append(nonlin)
            self.append(nn.Linear(d_hidden, d_hidden, dtype=dtype))

        self.append(nonlin)
        self.append(nn.Linear(d_hidden, d_out, dtype=dtype))

        if init_dict is not None:
            self.init_weights(**init_dict)

    def init_weights(self, weight_init=None, bias_init=None):
        for m in self.children():
            if isinstance(m, nn.Linear):
                if weight_init is not None:
                    weight_init(m.weight)
                if bias_init is not None:
                    bias_init(m.bias)


class ParallelSlots(nn.Module):
    """Wrapper to collect multiple phi_k in a single phi."""

    def __init__(self, slot_functions: List[torch.nn.Module]):
        super().__init__()
        self.slot_functions = nn.ModuleList(slot_functions)
        self.d_in = [slot_func.d_in for slot_func in slot_functions]
        self.d_out = [slot_func.d_out for slot_func in slot_functions]

    def forward(self, x):
        # add a batch dimension for singleton inputs
        if x.ndim == 1:
            x = x.unsqueeze(0)

        # distribute inputs to each slot
        _x = x.split(self.d_in, dim=-1)
        return torch.cat(
            [slot_k(x_k) for slot_k, x_k in zip(self.slot_functions, _x)], dim=1
        )

    def __getitem__(self, item):
        return self.slot_functions[item]


class LinearComposition(nn.Module):
    """C for simple addition."""

    def __init__(self):
        super().__init__()

    def forward(self, x, d_hidden):
        assert all_equal(
            d_hidden
        ), f"The output dimension of each slot must be identical, but got {d_hidden}"

        # split outputs from each slot
        x = torch.stack(x.split(d_hidden, dim=-1), dim=1)
        return torch.sum(x, dim=1)


class CompositionalFunction(nn.Module):
    """Wrapper for combination function and slot functions"""

    def __init__(self, composition: nn.Module, slots: nn.Module):
        super().__init__()
        self.composition = composition
        self.slots = slots
        self.d_in = slots.d_in
        self.d_hidden = slots.d_out

    def forward(self, x, return_slot_outputs=False):
        _x = self.slots(x)
        out = self.composition(_x, self.d_hidden)

        if return_slot_outputs:
            return out, _x.split(self.d_hidden, dim=-1)
        return out

    def get_slot(self, idx: int):
        return self.slots[idx]
