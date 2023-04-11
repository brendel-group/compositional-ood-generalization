from math import ceil, sqrt
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
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


def visualize_latents(
    latents: torch.Tensor, dim_per_slot: List[int], grid_size: int, out: Path = None
):
    cols = [
        f"z{get_digit_subscript(i+1)}{get_digit_subscript(j+1)}"
        for i, d in enumerate(dim_per_slot)
        for j in range(d)
    ]
    df = pd.DataFrame(latents.numpy(), columns=cols)

    bins = (np.arange(grid_size + 1) - 0.5) / (grid_size - 1)
    sb.pairplot(df, corner=True, diag_kws=dict(bins=bins))


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


def visualize_score_heatmaps(
    latents: torch.Tensor,
    scores: torch.Tensor,
    dim_per_slot: List[int],
    score_name: str = "score",
    plot_size: float = 3,
    title: str = "",
    out: Path = None,
):
    assert latents.shape[0] == scores.shape[0], "Requires a score for each latent."

    cols = [
        f"z{get_digit_subscript(i+1)}{get_digit_subscript(j+1)}"
        for i, d in enumerate(dim_per_slot)
        for j in range(d)
    ] + [score_name]

    combined_data = torch.cat([latents, scores.unsqueeze(-1)], dim=1)

    df = pd.DataFrame(combined_data.numpy(), columns=cols)

    n_dims = sum(dim_per_slot)

    fig = plt.figure(figsize=(n_dims * plot_size, n_dims * plot_size))
    fig.suptitle(title)

    # diagonal cells
    # find y limits first
    vmin, vmax = float("inf"), 0
    for i in range(n_dims):
        df_view = df.groupby(cols[i]).mean()[score_name]

        if df_view.min() < vmin:
            vmin = df_view.min()
        if df_view.max() > vmax:
            vmax = df_view.max()

    # plot barplot of mean error along this dimension
    for i in range(n_dims):
        ax = fig.add_subplot(n_dims, n_dims, i * (n_dims + 1) + 1)
        df_view = df.groupby(cols[i]).mean()[score_name]
        sb.lineplot(df_view)
        ax.set_ybound(vmin, vmax)

    # corner cells
    # find color range first
    vmin, vmax = float("inf"), 0
    for row in range(1, n_dims):
        for col in range(row):
            df_view = df.groupby([cols[row], cols[col]]).mean()[score_name]

            if df_view.min() < vmin:
                vmin = df_view.min()
            if df_view.max() > vmax:
                vmax = df_view.max()

    # plot heatmap of mean error over these two dimension
    for row in range(1, n_dims):
        for col in range(row):
            ax = fig.add_subplot(n_dims, n_dims, row * n_dims + col + 1)
            df_view = df.groupby([cols[row], cols[col]]).mean()[score_name]
            # transform into 2d data
            df_view = df_view.reset_index().pivot(
                columns=cols[col], index=cols[row], values=score_name
            )
            sb.heatmap(df_view, vmin=vmin, vmax=vmax, cmap="RdBu", cbar=False)

    fig.tight_layout()
    plt.show()
