import argparse
import pickle as pk
import re
from pathlib import Path
from typing import List, Union, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb

from utils import StoreKwargsActions


def _load_dfs(paths: List[Union[str, Path]], slices: Optional[List[callable]] = None):
    dfs = []
    for path in paths:
        if not isinstance(path, Path):
            path = Path(path)
        with path.open("rb") as f:
            dfs.append(pk.load(f))

    df = pd.concat(dfs, ignore_index=True)
    if slices is not None:
        for slice in slices:
            df = slice(df)

    return df


def r2_v_nsamples(
    df: pd.DataFrame,
    xscale: str = "log",
    list_max: bool = False,
):
    sb.set_style("whitegrid")

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


def _slice(arg_str: str):
    """
    Interpret key-operator-value strings as slicing functions for a pandas DataFrame.

    E.g. `metric=test_R²` will be interpreted as `df.loc[df['metric'] == 'test R²']`.

    In general, everything before an operator (:, =, :=, ==, !, !=, <, >, <=, >=) will
    be interpreted as the column name to slice; everything after will be interpreted
    as the value(s) to compare. Spaces can be replaced by `_` in the argument string
    and will be replaced.

    :, =, :=, == are all aliases for a `==` operation; !, != are aliases for `!=`.

    A list of values can be provided as `col_name=[value_1; value_2; ...]`, in which case
    the equal and not operators will use the `pandas.Series.isin()` function for
    comparison.
    """
    equal_ops = [":", "=", "==", ":="]
    not_ops = ["!", "!="]

    try:
        col_name, operator, value = re.search(
            r"([a-zA-Z]+)([:=!<>]{0,2})\[?([^\[\]\n]+)\]?", arg_str
        ).group(1, 2, 3)
        # TODO would be nice to actually use commas as the value separators
        #   in which case we would have to be able to escape commas in the
        #   value names
        value = value.replace("_", " ").split(";")

        if len(value) == 1:
            # normal comparisons
            value = value[0]
            if operator in equal_ops:
                return lambda x: x.loc[x[col_name] == value]
            elif operator == not_ops:
                return lambda x: x.loc[x[col_name] != value]
            elif operator == "<":
                return lambda x: x.loc[x[col_name] < value]
            elif operator == ">":
                return lambda x: x.loc[x[col_name] > value]
            elif operator == "<=":
                return lambda x: x.loc[x[col_name] <= value]
            elif operator == ">=":
                return lambda x: x.loc[x[col_name] >= value]
            else:
                raise argparse.ArgumentTypeError(
                    f"Unable to interpret operator {operator} in {arg_str}"
                )
        else:
            # list comparisons
            if operator in equal_ops:
                return lambda x: x.loc[x[col_name].isin(value)]
            elif operator == not_ops:
                return lambda x: x.loc[~x[col_name].isin(value)]
            elif operator in ["<", ">", "<=", ">="]:
                raise argparse.ArgumentTypeError(
                    "Can't compare against a list of values with < > operators."
                )
            else:
                raise argparse.ArgumentTypeError(
                    f"Unable to interpret operator {operator} in {arg_str}"
                )

    except:
        raise argparse.ArgumentTypeError(
            f"Can't interpret {arg_str} as dataframe slice"
        )


# TODO add a argparse type to interpret renaming commands and implement
#   renaming functionality in the load_df() function.


if __name__ == "__main__":
    # TODO this is bad as it also includes imports
    #   the better thing to do would probable be defining a global `visualizations` dict
    #   and adding functions to it with a custom @visualization decorator
    visualizations = {k: v for k, v in globals().items() if not k.startswith("_")}

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "type",
        type=str,
        choices=visualizations,
        help=f"type of visualization, one of {visualizations}",
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
    df = _load_dfs(args.df_paths, args.df_slices)

    print("Visualizing...")
    visualizations[args.type](df, **args.plot_kwargs)

    print(f"Saving as {args.out} ...")
    _save_fig(args.out)
