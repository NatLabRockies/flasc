"""Module for generating tabulated data from times series."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns

from flasc.analysis.expected_power_analysis_utilities import (
    _add_wd_ws_bins,
    _bin_and_group_dataframe_expected_power,
)


def _plot_binned_data_counts(df_grid: pd.DataFrame):
    """Plot the number of entries in each wind direction/wind speed bin.

    This is a supporting function used to display
    how complete one can derive a steady-state table (grid) from a timeseries of solutions, e.g.,
    from an LES timeseries.

    Args:
        df_grid (pd.DataFrame): DataFrame containing the binned data.

    Returns:
        matplotlib.axes.Axes: Axes object of the generated plot.
    """
    df_binned = df_grid
    df_N = (
        df_binned[["wd_bin", "ws_bin", "count"]]
        .set_index(["wd_bin", "ws_bin"])
        .unstack()
        .transpose()
    )
    df_N.index = [i[1] for i in df_N.index]

    fig, ax = plt.subplots(figsize=(24, 12))
    sns.heatmap(
        df_N,
        annot=True,
        annot_kws={"fontsize": 8},
        linewidth=0.5,
        cmap="rocket_r",
        ax=ax,
        fmt=".0f",
        cbar=False,
    )
    ax.set_ylabel("Wind speed bin (m/s)")
    ax.set_xlabel("Wind direction bin (deg)")
    ax.set_title("Number of entries per wd/ws bin in binning of timeseries data")
    plt.tight_layout()

    return ax


def bin_timeseries_to_grid(
    df_timeseries: pd.DataFrame,
    wd_step: float = 5.0,
    wd_min: float = 0.0,
    wd_max: float = 360.0,
    ws_step: float = 1.0,
    ws_min: float = 0.0,
    ws_max: float = 50.0,
    N_min: int = 3,
    plot: bool = False,
):
    """Convert a timeseries DataFrame into a gridded solution table.

    Table is based on wind direction and wind speed bins.

    Args:
        df_timeseries (pd.DataFrame): Dataframe with timeseries that you want to turn into a gridded
            solution table. Requires the columns 'wd', 'ws', and one power column
            (pow_000, pow_001, etc.) for every turbine in your wind farm.
        wd_step (float, optional): Step size for wind direction bins. Defaults to 5.0.
        wd_min (float, optional): Minimum wind direction for binning. Defaults to 0.0.
        wd_max (float, optional): Maximum wind direction for binning. Defaults to 360.0.
        ws_step (float, optional): Step size for wind speed bins. Defaults to 1.0.
        ws_min (float, optional): Minimum wind speed for binning. Defaults to 0.0.
        ws_max (float, optional): Maximum wind speed for binning. Defaults to 50.0.
        N_min (int, optional): Minimum number of samples within a bin for it to be considered valid.
            Defaults to 3.
        plot (bool, optional): Whether to generate a plot of the binned data counts.
            Defaults to False.

    Returns:
        pd.DataFrame: DataFrame containing the gridded solution table.
    """
    # Local copy of the timeseries that we can manipulate
    df_ = pl.from_pandas(df_timeseries.copy())
    df_ = df_.with_columns(pl.lit("name").alias("df_name"))

    # Add the columns ws_bin and wd_bin, based on the columns wd and ws in df_
    df_ = _add_wd_ws_bins(
        df_,
        wd_cols=["wd"],
        ws_cols=["ws"],
        wd_step=wd_step,
        wd_min=wd_min,
        wd_max=wd_max,
        ws_step=ws_step,
        ws_min=ws_min,
        ws_max=ws_max,
    )

    # Bin df_ into df_bin based on the columns bin_cols_without_df_name.
    # The output contains the mean and count of all test_cols for each bin, and also a total count
    # of samples in each bin (count).
    df_bin = _bin_and_group_dataframe_expected_power(
        df_=df_,
        test_cols=df_timeseries.columns,
        bin_cols_without_df_name=["wd_bin", "ws_bin"],
    )

    # Filter out bins with less than N_min samples.
    df_bin = df_bin.filter(pl.col("count") >= N_min)

    # Convert the dataframe back to Pandas for formatting and plotting
    df_grid = df_bin.to_pandas()

    # Generate a plot, if necessary, but removing unused columns from the dataframe
    if plot:
        _plot_binned_data_counts(df_grid)

    # Convert into a minimum dataframe with only necessary columns and rename
    df_grid = df_grid[
        [col for col in df_grid.columns if "pow_" in col] + ["wd_bin", "ws_bin", "ti_mean", "count"]
    ]
    df_grid = df_grid.rename(
        columns={
            "wd_bin": "wd",
            "ws_bin": "ws",
            **{col: col.replace("_mean", "") for col in df_grid.columns if "_mean" in col},
        }
    )

    # Make sure that all bins exists in df_grid, even if they have no data.
    for wd_value in np.arange(wd_min + 0.5 * wd_step, wd_max, wd_step):
        for ws_value in np.arange(ws_min + 0.5 * ws_step, ws_max + 0.5 * ws_step, ws_step):
            if not ((df_grid["wd"] == wd_value) & (df_grid["ws"] == ws_value)).any():
                new_row = pd.DataFrame({"wd": [wd_value], "ws": [ws_value]})
                df_grid = pd.concat([df_grid, new_row], ignore_index=True)

    # Sort table by wind directions and wind speeds
    df_grid = df_grid.sort_values(by=["ws", "wd"]).reset_index(drop=True)

    return df_grid
