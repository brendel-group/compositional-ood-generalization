from typing import List, Union

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


# TODO write base "Composition" class
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

    def get_d_out(self, slot_d_out: List[int]) -> int:
        return slot_d_out[0]


class OcclusionLinearComposition(nn.Module):
    def __init__(self, clamp: Union[bool, str] = True, soft_add: bool = False):
        super().__init__()
        self.clamp = clamp
        self.soft_add = soft_add

    def _add_behind(self, a: torch.Tensor, b: torch.Tensor):
        return torch.where(a < 0, b, a)

    def _soft_add_behind(self, a: torch.Tensor, b: torch.Tensor):
        # the normal add-behind is basically a·step(a) + b·step(-a)
        # soften the step function with a sigmoid here
        return a * nn.functional.sigmoid(a) + b * nn.functional.sigmoid(-a)

    def forward(self, x, d_hidden):
        assert all_equal(
            d_hidden
        ), f"The output dimension of each slot must be identical, but got {d_hidden}"

        # split outputs from each slot
        x = torch.stack(x.split(d_hidden, dim=-1), dim=1)
        out = x[:, 0, :]
        for slot in range(1, len(d_hidden)):
            if self.soft_add:
                out = self._soft_add_behind(out, x[:, slot, :])
            else:
                out = self._add_behind(out, x[:, slot, :])

        # the output can be restricted to positive numbers
        if isinstance(self.clamp, bool) and self.clamp:
            out = torch.where(out < 0, 0, out)
        elif self.clamp == "relu":
            out = nn.functional.relu(out)
        elif self.clamp == "sotfplus":
            out = nn.functional.softplus(out)

        return out

    def get_d_out(self, slot_d_out: List[int]) -> int:
        return slot_d_out[0]


class CompositionalFunction(nn.Module):
    """Wrapper for combination function and slot functions"""

    def __init__(self, composition: nn.Module, slots: nn.Module):
        super().__init__()
        self.composition = composition
        self.slots = slots
        self.d_in = slots.d_in
        self.d_hidden = slots.d_out
        self.d_out = composition.get_d_out(self.d_hidden)

    def forward(self, x, return_slot_outputs=False):
        _x = self.slots(x)
        out = self.composition(_x, self.d_hidden)

        if return_slot_outputs:
            return out, _x.split(self.d_hidden, dim=-1)
        return out

    def get_slot(self, idx: int):
        return self.slots[idx]
