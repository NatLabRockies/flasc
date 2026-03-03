import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from flasc.analysis.expected_power_analysis_utilities import _add_wd_ws_bins, _bin_and_group_dataframe_expected_power
import polars as pl


def _plot_binned_data_counts(df_grid):
    # Load df_binned from self
    df_binned = df_grid
    df_N = df_binned[["wd_bin", "ws_bin", "count"]].set_index(["wd_bin", "ws_bin"]).unstack().transpose()
    df_N.index = [i[1] for i in df_N.index]
        
    fig, ax = plt.subplots(figsize=(24, 12))
    sns.heatmap(df_N, annot=True, annot_kws={"fontsize": 8}, linewidth=.5, cmap="rocket_r", ax=ax, fmt=".0f", cbar=False)
    ax.set_ylabel("Wind speed bin (m/s)")
    ax.set_xlabel("Wind direction bin (deg)")
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
    N_min: int = 3,  # Minimum number of samples within a bin for it to be considered valid and used to generate an approximate table entry with
    plot: bool = False,
    verbose: bool = True
):
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
    # The output contains the mean and count of all test_cols for each bin, and also a total count of samples in each bin (count).
    df_bin = _bin_and_group_dataframe_expected_power(
        df_=df_,
        test_cols=df_timeseries.columns,
        bin_cols_without_df_name=["wd_bin", "ws_bin"],
    )

    # Filter out bins with less than 3 samples.
    df_bin = df_bin.filter(pl.col("count") >= N_min)

    # Convert the dataframe back to Pandas for formatting and plotting
    df_grid = df_bin.to_pandas()

    # Generate a plot, if necessary, but removing unused columns from the dataframe
    if plot:
        _plot_binned_data_counts(df_grid)

    # Convert into a minimum dataframe with only necessary columns and rename 
    df_grid = df_grid.drop(columns=['df_name', 'wd_mean', 'ws_mean', 'time_mean'] + [col for col in df_grid.columns if 'count' in col])
    df_grid = df_grid.rename(columns={'wd_bin': 'wd', 'ws_bin': 'ws', **{col: col.replace('_mean', '') for col in df_grid.columns if '_mean' in col}})

    # Make sure that all bins exists in df_grid, even if they have no data.
    for wd_value in np.arange(wd_min + 0.5 * wd_step, wd_max, wd_step):
        for ws_value in np.arange(ws_min + 0.5 * ws_step, ws_max + 0.5 * ws_step, ws_step):
            if not ((df_grid['wd'] == wd_value) & (df_grid['ws'] == ws_value)).any():
                new_row = pd.DataFrame({'wd': [wd_value], 'ws': [ws_value]})
                df_grid = pd.concat([df_grid, new_row], ignore_index=True)

    # Sort table by wind directions and wind speeds
    df_grid = df_grid.sort_values(by=['ws', 'wd']).reset_index(drop=True)

    return df_grid
