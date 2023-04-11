from math import ceil, sqrt
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
import torch

from data import sample_latents
from models import CompositionalFunction
from utils import get_digit_subscript


def _rectify(t: torch.Tensor) -> torch.Tensor:
    "Finds the best rectangle representation for a 1d Tensor. Useful for plotting."
    area = t.shape[0]
    max_width = int(ceil(sqrt(area)))
    for w in reversed(range(1, max_width + 1)):
        if area % w == 0:
            return t.view(w, area // w)


def visualize_latents(latents: torch.Tensor, dim_per_slot: List[int], out: Path = None):
    cols = [
        f"z{get_digit_subscript(i+1)}{get_digit_subscript(j+1)}"
        for i, d in enumerate(dim_per_slot)
        for j in range(d)
    ]
    df = pd.DataFrame(latents.numpy(), columns=cols)
    sb.pairplot(df, corner=True)


def visualize_slots_and_output(
    output: torch.Tensor,
    slots: Tuple[torch.Tensor],
    plot_size: float = 3,
    title: str = "",
    out: Path = None,
):
    output = output.detach()
    slots = [slot.detach() for slot in slots]

    n_cols = output.shape[0]
    n_rows = len(slots) + 1

    fig = plt.figure(figsize=(n_cols * plot_size, n_rows * plot_size))
    fig.suptitle(title)

    for col in range(n_cols):
        # each column uses the same min/max for the colormap
        all_values = torch.cat([slot[col] for slot in slots] + [output[col]])
        vmin = min(all_values.min(), -all_values.max())
        vmax = max(all_values.max(), -all_values.min())

        # plot slots
        for row in range(n_rows - 1):
            ax = fig.add_subplot(n_rows, n_cols, row * n_cols + col + 1)
            ax.imshow(_rectify(slots[row][col]), cmap="RdBu", vmin=vmin, vmax=vmax)
            ax.axis("off")

        # plot output
        ax = fig.add_subplot(n_rows, n_cols, (n_rows - 1) * n_cols + col + 1)
        ax.imshow(_rectify(output[col]), cmap="RdBu", vmin=vmin, vmax=vmax)
        ax.axis("off")

    # add column titles
    for col, ax in enumerate(fig.axes[::n_rows]):
        ax.text(
            0.5,
            1.05,
            f"Sample {col + 1}",
            horizontalalignment="center",
            verticalalignment="bottom",
            transform=ax.transAxes,
        )

    # add row titles
    for row, ax in enumerate(fig.axes[: n_rows - 1]):
        ax.text(
            -0.05,
            0.5,
            f"Slot {row + 1}",
            rotation="vertical",
            horizontalalignment="right",
            verticalalignment="center",
            transform=ax.transAxes,
        )

    ax = fig.axes[n_rows - 1]
    ax.text(
        -0.05,
        0.5,
        "Output",
        rotation="vertical",
        horizontalalignment="right",
        verticalalignment="center",
        transform=ax.transAxes,
    )

    plt.show()


def visualize_output_grid(
    comp_func: CompositionalFunction,
    grid_size: int = 5,
    dims: Tuple[int, int] = None,
    plot_size: float = 3,
    title: str = "",
    out: Path = None,
):
    D = comp_func.d_in
    z = sample_latents(D, "face_grid", grid_size=grid_size, dims=dims)
    x = comp_func(z).detach()

    if dims is not None:
        fig = plt.figure(figsize=(grid_size * plot_size, grid_size * plot_size))
        fig.suptitle(title)

        vmin = min(x.min(), -x.max())
        vmax = max(x.max(), -x.min())

        for i, _x in enumerate(x):
            ax = fig.add_subplot(grid_size, grid_size, i + 1)
            ax.imshow(_rectify(_x), cmap="RdBu", vmin=vmin, vmax=vmax)
            ax.axis("off")

        plt.show()

    else:
        raise NotImplementedError()
