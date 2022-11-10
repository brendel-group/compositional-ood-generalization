import argparse
import pickle as pk
import re
from pathlib import Path
from typing import List, Union, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb

from utils import StoreKwargsActions


def _load_dfs(paths: List[Union[str, Path]]):
    dfs = []
    for path in paths:
        if not isinstance(path, Path):
            path = Path(path)
        with path.open("rb") as f:
            dfs.append(pk.load(f))

    return pd.concat(dfs, ignore_index=True)


def r2_v_nsamples(
    df: pd.DataFrame,
    slices: Optional[List[callable]] = None,
    xscale: str = "log",
    list_max: bool = False,
):
    sb.set_style("whitegrid")
    if slices is not None:
        for slice in slices:
            df = slice(df)
    # df = df.loc[
    #     (
    #         df["model"].isin(
    #             ["MLP", "MLP compositional contrast λ=0.001 normalized", "Oracle"]
    #         )
    #     )
    #     & (df["metric"] == "test R²")
    #     & (df["l"] == 8)
    # ]
    df = df.loc[df["metric"] == "test R²"]

    fg = sb.relplot(
        data=df, x="n samples", y="val", hue="model", col="domain", kind="line"
    )

    fg.set(xscale=xscale)
    fg.set_ylabels("R²")
    fg.fig.suptitle("R² over # Observed Samples (1024 new samples / epoch)")
    fg.fig.subplots_adjust(top=0.85)

    # plot legend with max value in each plot
    if list_max:
        for ax in fg.fig.axes:
            vals = []
            line_artists = []
            for line in ax._children:
                # existing legends also add lines to the plot
                #   but only the original data lines are named '_child...'
                if line._label.startswith("_child"):
                    _, y = line.get_data()
                    vals.append(f"{y.max():0.4f}")
                    line_artists.append(line)
            ax.legend(line_artists, vals, title="max value")


def _save_fig(out_file: Union[str, Path]):
    if not isinstance(out_file, Path):
        out_file = Path(out_file)
    with out_file.open("w") as f:
        plt.savefig(out_file, bbox_inches="tight")


# TODO return a list of functions that can be used on a dataframe
def _slice(arg_str: str):
    """
    model=[MLP;MLP_compositional_contrast_λ=0.001_normalized;Oracle]
    metric=test_R²
    l=8

    1. split on the first instance of = < > !
    2. extract column name
    3. extract value as list or string
    4. convert comparison type to correct function
    """
    try:
        col_name, operator, value = re.search(
            r"([a-zA-Z]+)([=!<>])\[?([^\[\]\n]+)\]?", arg_str
        )[1:]
        value = value.replace("_", " ").split(";")
        if len(value) == 1:
            value = value[0]

        assert not (operator in ["<", ">"] and isinstance(value, list))

    except:
        raise argparse.ArgumentTypeError(
            f"Can't interpret {arg_str} as dataframe slice"
        )


if __name__ == "__main__":
    visualizations = {k: v for k, v in globals().items() if not k.startswith("_")}

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "type", type=str, help=f"type of visualization, one of {visualizations}"
    )
    parser.add_argument("out", type=Path, help="output file")
    parser.add_argument(
        "df_paths",
        type=Path,
        nargs="+",
        help="path to all pandas.DataFrame needed for visualization",
    )
    parser.add_argument(
        "-s",
        "--df-slices",
        type=_slice,
        nargs="+",
        help="column-value pairs to slice the dataframe before plotting",
    )
    parser.add_argument(
        "-p",
        "--plot-kwargs",
        nargs="+",
        default={},
        action=StoreKwargsActions,
        help="key-value pairs for plotting, given as `key=value`",
    )

    args = parser.parse_args()

    print("Loading data...")
    df = _load_dfs(args.df_paths)

    print("Visualizing...")
    visualizations[args.type](df, **args.plot_kwargs)

    print(f"Saving as {args.out} ...")
    _save_fig(args.out)
