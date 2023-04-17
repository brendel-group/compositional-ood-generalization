from math import ceil, sqrt
from pathlib import Path
from typing import List, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
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
    sb.set_theme("notebook", "whitegrid", rc={"axes.grid": False})

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
    sb.set_theme("notebook", "whitegrid")

    output = output.detach()
    slots = [slot.detach() for slot in slots]

    n_cols = output.shape[0]
    n_rows = len(slots) + 1

    fig = plt.figure(figsize=(n_cols * plot_size, n_rows * plot_size))
    fig.suptitle(title)

    # plot as image if there are color channels, otherwise as heatmap
    plot_rgb = output[0].dim() == 3

    for col in range(n_cols):
        if not plot_rgb:
            # each column uses the same min/max for the colormap
            all_values = torch.cat([slot[col] for slot in slots] + [output[col]])
            vmin = min(all_values.min(), -all_values.max())
            vmax = max(all_values.max(), -all_values.min())

        # plot slots
        for row in range(n_rows - 1):
            ax = fig.add_subplot(n_rows, n_cols, row * n_cols + col + 1)
            if plot_rgb:
                ax.imshow(slots[row][col])
            else:
                ax.imshow(slots[row][col], cmap="RdBu", vmin=vmin, vmax=vmax)
            ax.axis("off")

        # plot output
        ax = fig.add_subplot(n_rows, n_cols, (n_rows - 1) * n_cols + col + 1)
        if plot_rgb:
            ax.imshow(output[col])
        else:
            ax.imshow(output[col], cmap="RdBu", vmin=vmin, vmax=vmax)
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


def _plot_image_grid(axs, x):
    # plot as image if there are color channels, otherwise as heatmap
    plot_rgb = x.dim() == 4

    if not plot_rgb:
        vmin = min(x.min(), -x.max())
        vmax = max(x.max(), -x.min())

    for ax, _x in zip(axs, x):
        if plot_rgb:
            ax.imshow(_x)
        else:
            ax.imshow(_x, cmap="RdBu", vmin=vmin, vmax=vmax)
        ax.axis("off")


def _get_xy_between_plots(fig, axs):
    # get bounding boxes
    r = fig.canvas.get_renderer()
    get_bbox = lambda ax: ax.get_tightbbox(r).transformed(fig.transFigure.inverted())
    bboxes = np.array([get_bbox(ax) for ax in axs.flat], mtrans.Bbox).reshape(axs.shape)

    # get min/max extent and coordinates between them
    xmax = np.array([b.x1 for b in bboxes.flat]).reshape(axs.shape).max(axis=0)
    xmin = np.array([b.x0 for b in bboxes.flat]).reshape(axs.shape).min(axis=0)
    ymax = np.array([b.y1 for b in bboxes.flat]).reshape(axs.shape).max(axis=1)
    ymin = np.array([b.y0 for b in bboxes.flat]).reshape(axs.shape).min(axis=1)
    xs = np.c_[xmax[1:], xmin[:-1]].mean(axis=1)
    ys = np.c_[ymax[1:], ymin[:-1]].mean(axis=1)
    return xs, ys


def visualize_output_grid(
    comp_func: CompositionalFunction,
    latent_transform: callable = None,
    grid_size: int = 5,
    dims: Tuple[int, int] = None,
    plot_size: float = 3,
    title: str = "",
    out: Path = None,
):
    sb.set_theme("notebook", "whitegrid")

    D = comp_func.d_in

    if dims is not None:
        fig, axs = plt.subplots(
            grid_size, grid_size, figsize=(grid_size * plot_size, grid_size * plot_size)
        )
        fig.suptitle(title)

        z = sample_latents(D, "face_grid", grid_size=grid_size, dims=dims)
        if latent_transform is not None:
            z = latent_transform(z)
        x = comp_func(z).detach()

        _plot_image_grid(axs.flatten(), x)

    else:
        n_grids = sum(D) - 1
        fig, axs = plt.subplots(
            n_grids * grid_size,
            n_grids * grid_size,
            figsize=(n_grids * grid_size * plot_size, n_grids * grid_size * plot_size),
        )
        fig.suptitle(title)
        for ax in axs.flatten():
            ax.axis("off")

        for row in range(1, n_grids + 1):
            for col in range(row):
                z = sample_latents(D, "face_grid", grid_size=grid_size, dims=(row, col))
                if latent_transform is not None:
                    z = latent_transform(z)
                x = comp_func(z).detach()

                _axs = axs[
                    (row - 1) * grid_size : row * grid_size,
                    col * grid_size : (col + 1) * grid_size,
                ]
                _plot_image_grid(_axs.flatten(), x)

        # add background in a checkerboard pattern
        plt.tight_layout()
        xs, ys = _get_xy_between_plots(fig, axs)
        xs = [0] + list(xs[grid_size - 1 :: grid_size]) + [1]
        ys = [1] + list(ys[grid_size - 1 :: grid_size]) + [0]

        for row in range(n_grids):
            for col in range(row):
                # skip every 2nd cell
                if (row + col) % 2 == 0:
                    continue

                w = xs[col + 1] - xs[col]
                h = ys[row + 1] - ys[row]

                patch = patches.Rectangle(
                    (xs[col], ys[row]),
                    w,
                    h,
                    edgecolor=None,
                    facecolor="tab:blue",
                    transform=fig.transFigure,
                    zorder=0,
                )
                fig.add_artist(patch)

    plt.show()


def visualize_score_heatmaps(
    latents: torch.Tensor,
    scores: torch.Tensor,
    dim_per_slot: List[int],
    score_name: str = "score",
    plot_size: float = 3,
    title: str = "",
    out: Path = None,
    logging: bool = False,
):
    assert latents.shape[0] == scores.shape[0], "Requires a score for each latent."

    sb.set_theme("notebook", "whitegrid")

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
            sb.heatmap(df_view, vmin=vmin, vmax=vmax, cmap="mako", cbar=False)

    # add colorbar for heatmaps
    norm = plt.Normalize(vmin, vmax)
    sm = plt.cm.ScalarMappable(cmap="mako", norm=norm)
    sm.set_array([])

    ax = fig.add_subplot(n_dims, n_dims, n_dims)
    ax.figure.colorbar(sm)
    ax.axis("off")

    fig.tight_layout()
    if logging:
        return fig
    else:
        plt.show()
