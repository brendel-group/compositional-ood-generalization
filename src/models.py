from abc import ABC, abstractmethod
from math import prod
from typing import List, Union, Dict

import numpy as np
import torch
import torch.nn as nn
from spriteworld import renderers, sprite

from utils import all_equal

SPRITEWORLD_DEFAULT_RANGES = {
    "x": (0.1, 0.9),
    "y": (0.2, 0.8),
    "shape": ["triangle", "square", "circle"],
    "scale": (0.09, 0.22),
    "hue": (0.05, 0.95),
}


def scale_latents(
    z: torch.Tensor, ranges: Dict[str, List[float]] = SPRITEWORLD_DEFAULT_RANGES
) -> torch.Tensor:
    # unflatten slot dimension to scale all slots simultaneously
    z = z.view(z.shape[0], -1, len(ranges))
    for i, range in enumerate(ranges.values()):
        if isinstance(range, tuple):
            z[:, :, i] *= range[1] - range[0]
            z[:, :, i] += range[0]
        else:
            z[:, :, i] *= len(range) - 1
            z[:, :, i] = z[:, :, i].round()

    # flatten slot dimension again
    z = z.flatten(1)
    return z


class InvertibleMLP(nn.Sequential):
    def __init__(
        self,
        d_in: int,
        d_out: Union[int, List[int]],
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
        if isinstance(d_out, list):
            d_out = prod(d_out)
        self.append(nn.Linear(d_hidden, d_out, dtype=dtype))
        self.append(nn.Unflatten(dim=1, unflattened_size=self.d_out))

        if init_dict is not None:
            self.init_weights(**init_dict)

    def init_weights(self, weight_init=None, bias_init=None):
        for m in self.children():
            if isinstance(m, nn.Linear):
                if weight_init is not None:
                    weight_init(m.weight)
                if bias_init is not None:
                    bias_init(m.bias)


class SpriteworldRenderer(nn.Module):
    def __init__(self, d_in: int, d_out: List[int], **kwargs):
        super().__init__()
        img_h, img_w = d_out[:2]
        if d_out[2] != 3:
            raise NotImplementedError("Can only render to RGB.")

        self.d_in = d_in
        self.d_out = d_out

        self.shape_names = ["triangle", "square", "circle"]

        self.factor_names = ["x", "y", "shape", "scale", "c0"]
        self.renderer = renderers.PILRenderer(
            image_size=(img_h, img_w),
            anti_aliasing=1,  # can't do anti-aliasing without alpha channel
            color_to_rgb=renderers.color_maps.hsv_to_rgb,
        )

    def forward(self, x):
        out = []
        for _x in x:
            factors = dict(zip(self.factor_names, _x.tolist()))
            factors["shape"] = self.shape_names[int(factors["shape"])]
            factors.update(angle=0, c1=1, c2=1)
            _sprite = sprite.Sprite(**factors)
            out.append(self.renderer.render([_sprite]))
        # converting to numpy first is faster than direct conversion
        out = torch.Tensor(np.array(out))
        out /= 255
        return out


class ParallelSlots(nn.Module):
    """Wrapper to collect multiple phi_k in a single phi."""

    def __init__(self, slot_functions: List[torch.nn.Module]):
        super().__init__()
        self.slot_functions = nn.ModuleList(slot_functions)
        self.d_in = [slot_func.d_in for slot_func in slot_functions]
        self.d_out = [slot_func.d_out for slot_func in slot_functions]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # add a batch dimension for singleton inputs
        if x.ndim == 1:
            x = x.unsqueeze(0)

        # distribute inputs to each slot
        _x = x.split(self.d_in, dim=-1)
        # each slot returns a flattened tensor, so concatenation works even for slots
        # with different d_outs
        return [slot_k(x_k) for slot_k, x_k in zip(self.slot_functions, _x)]

    def __getitem__(self, item):
        return self.slot_functions[item]


# TODO clean up composition function signatures: are they input shape agnostic or not?
class Composition(nn.Module, ABC):
    """Abstract base class for all composition functions."""

    @abstractmethod
    def get_d_out(
        self, slot_d_out: Union[List[int], List[List[int]]]
    ) -> Union[int, List[int]]:
        pass


class LinearComposition(Composition):
    """C for simple addition."""

    def __init__(self):
        super().__init__()

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        try:
            x = torch.stack(x, dim=1)
        except RuntimeError:
            raise RuntimeError(
                f"Linear Composition expects slot with equal output size, \
                but got shapes {[_x.shape for _x in x]}."
            )

        return torch.sum(x, dim=1)

    def get_d_out(
        self, slot_d_out: Union[List[int], List[List[int]]]
    ) -> Union[int, List[int]]:
        return slot_d_out[0]


class OcclusionLinearComposition(Composition):
    def __init__(
        self,
        clamp: Union[bool, str] = True,
        add: str = "step",
        ste: bool = False,
        alpha: float = 1,
        color_channel: bool = True,
    ):
        super().__init__()
        self.clamp = clamp
        self.add = add
        self.ste = ste
        self.alpha = alpha
        self.color_channel = color_channel

    def _add_step(self, a: torch.Tensor, b: torch.Tensor):
        # basically a·step(a) + b·step(-a)
        if self.color_channel:
            mask = a.sum(-1).unsqueeze(-1) <= 0
        else:
            mask = a <= 0
        return torch.where(mask, b, a)

    def _add_sigmoid(self, a: torch.Tensor, b: torch.Tensor):
        # soften the step function with a sigmoid here
        if self.color_channel:
            mask = a.sum(-1).unsqueeze(-1)
        else:
            mask = a
        return a * nn.functional.sigmoid(mask * self.alpha) + b * nn.functional.sigmoid(
            -mask * self.alpha
        )

    def _add_hardsigmoid(self, a: torch.Tensor, b: torch.Tensor):
        if self.color_channel:
            mask = a.sum(-1).unsqueeze(-1)
        else:
            mask = a
        return a * nn.functional.hardsigmoid(
            mask * self.alpha
        ) + b * nn.functional.hardsigmoid(-mask * self.alpha)

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        try:
            x = torch.stack(x, dim=1)
        except RuntimeError:
            raise RuntimeError(
                f"Linear Composition expects slot with equal output size, \
                but got shapes {[_x.shape for _x in x]}."
            )

        # TODO either include an alpha channel in the individual slot outputs,
        # or do anti-aliasing here.

        # split outputs from each slot
        # we know that all slot outputs have the same dimension, so we can reshape here
        out = x[:, 0, ...]
        for slot in range(1, x.shape[1]):
            if self.add == "step":
                out_backward = self._add_step(out, x[:, slot, :])
            elif self.add == "sigmoid":
                out_backward = self._add_sigmoid(out, x[:, slot, :])
            elif self.add == "hardsigmoid":
                out_backward = self._add_hardsigmoid(out, x[:, slot, :])
            if self.ste:
                out_forward = self._add_step(out, x[:, slot, :])
                out = out_backward + (out_forward - out_backward).detach()
            else:
                out = out_backward

        # the output can be restricted to positive numbers
        if isinstance(self.clamp, bool) and self.clamp:
            out = torch.where(out < 0, 0, out)
        elif self.clamp == "relu":
            out = nn.functional.relu(out)
        elif self.clamp == "sotfplus":
            out = nn.functional.softplus(out)

        return out

    def get_d_out(
        self, slot_d_out: Union[List[int], List[List[int]]]
    ) -> Union[int, List[int]]:
        return slot_d_out[0]


class CompositionalFunction(nn.Module):
    """Wrapper for combination function and slot functions"""

    def __init__(self, composition: Composition, slots: nn.Module):
        super().__init__()
        self.composition = composition
        self.slots = slots
        self.d_in = slots.d_in
        self.d_slots_out = slots.d_out
        self.d_out = composition.get_d_out(self.d_slots_out)

    def forward(self, x: torch.Tensor, return_slot_outputs=False) -> torch.Tensor:
        x = self.slots(x)
        out = self.composition(x)

        if return_slot_outputs:
            return out, x

        return out

    def get_slot(self, idx: int):
        return self.slots[idx]
