import numpy as np
import pandas as pd

from flasc.data_processing import dataframe_manipulations as dfm


# Introduce helper functions for calculating errors and generating plots/tables
def _err(old_val, new_val):
    return 100.0 * (new_val-old_val) / old_val


# Introduce helper functions for calculating wake loss (%)
def _wake_loss(cumprod_waked, cumprod_unwaked):
    return 100.0 * (cumprod_unwaked-cumprod_waked) / cumprod_unwaked


# A function to print table
def _print_pretty_table(table_dict, title):
    # Format title to be centered above the table, and print table in pretty format
    df_table = pd.DataFrame(table_dict)
    mrkdwn = df_table.to_markdown(headers='keys', tablefmt='psql', index=False, floatfmt=".2f")
    spc = int(np.floor(len(mrkdwn.split("\n")[0]) / 2 - len(title) / 2))
    mrkdwn = (" " * spc + title + "\n") + mrkdwn
    return print(mrkdwn)


def compare_cumulative_production_and_relative_wake_loss(
    df_list,  # Typical use case: 1st entry is SCADA, then remaining are models to compare. All models are compared against 1st entry.
    df_upstream,
    exclude_turbs=[],
    model_tags=None,
    print_to_console=True,
):
    # Check input dataframes are consistent in terms of number of turbines and timestamps
    if not all([all(df_list[0]["time"] == df["time"]) for df in df_list]):
        raise ValueError("Input dataframes have different timestamps. Please ensure all dataframes have the same timestamps.")

    n_turbs_list = [dfm.get_num_turbines(df) for df in df_list]
    if len(set(n_turbs_list)) > 1:
        raise ValueError(f"Input dataframes have different number of turbines: {n_turbs_list}. Please ensure all dataframes have the same number of turbines.")

    # Apply default model tags if not provided
    if model_tags is None:
        model_tags = [f"Model {ti}" for ti in range(len(df_list))]

    # Make local copies of dataframes that we can manipulate
    df_list = [df.copy() for df in df_list]

    # Apply simple NaN rule for every excluded turbine to avoid annoying edge cases in the analysis
    for ti in exclude_turbs:
        for df in df_list:
            df[f"pow_{ti:03d}"] = None

    # Helper variables
    n_turbs = dfm.get_num_turbines(df_list[0])
    pow_cols = [f"pow_{ti:03d}" for ti in range(n_turbs)]

    # First ensure consistent NaN mapping between modelled data and SCADA timeseries
    for ti in range(n_turbs):
        # For each individual turbine, identify all timestamps where any of the timeseries have NaN values, and create a combined mask of these
        ids_nan = np.zeros(len(df_list[0]), dtype=bool)
        for df in df_list:
            ids_nan = ids_nan | df[f"pow_{ti:03d}"].isna()
        ids_nan = np.where(ids_nan)[0]

        # Mirror NaNs across all timeseries
        for df in df_list:
            df.loc[ids_nan, f"pow_{ti:03d}"] = None

    # # Assert NaNs are identical between dataframes
    # n_nans_per_timeseries = [df[pow_cols].isna().sum().sum() for df in df_list]
    # print(f"NaNs in df_list power columns: {n_nans_per_timeseries}")

    for df in df_list:
        # Specify upstream power in the exact same way as with the SCADA data
        df = dfm.set_pow_ref_by_upstream_turbines(df, df_upstream, exclude_turbs=exclude_turbs)

    #######################################################################################################################
    ################## Compare cumulative production directly between SCADA and simulated data (LES) #####################
    #######################################################################################################################

    # Check if our dataset is a whole multiple of 8760 hours (1 year), if not, raise a warning that the AEP calculation is not
    # exact and that we are comparing cumulative energy rather than AEP. This is just a sanity check, not a strict requirement.
    t0 = df_list[0].iloc[0]["time"]
    t1 = df_list[0].iloc[-1]["time"]
    timespan = (t1 - t0)
    n_measurements_per_hour = np.timedelta64(1, 'h') / (df_list[0].iloc[1]["time"] - df_list[0].iloc[0]["time"])
    n_hours_total = timespan / np.timedelta64(1, 'h')
    n_weeks = n_hours_total / 168
    n_hours_in_year = 8760
    avg_no_years_in_data = round(n_hours_total / n_hours_in_year)
    avg_no_years_in_data = np.max([avg_no_years_in_data, 1])  # Minimum of 1 year needed to have cumulative production be comparable to AEP
    offset_prct = 100.0 * (n_hours_total - avg_no_years_in_data * n_hours_in_year) / n_hours_in_year # Offset percentage from whole number of years
    if np.abs(offset_prct) > 1.0:
        print(f"WARNING: Dataset spans {n_weeks:.1f} weeks and is not a whole multiple of 8760 hours (1 year). Note the " +
        f"reported numbers are CUMULATIVE ENERGY, not AEP. Offset from whole number of years: {offset_prct:+.2f}%.")

    # Absolute cumulative energy production for the entire farm and per turbine
    cumprod_turbine_list = [[(df[pc].sum() / n_measurements_per_hour) for pc in pow_cols] for df in df_list]
    cumprod_farm_list = [np.sum(cumprod_tm) for cumprod_tm in cumprod_turbine_list]

    table_absolute_cumprod_dict = {
        "Selection": ["Entire farm"] + [f"Turbine {ti:02d}" for ti in range(n_turbs)],
    }
    for mii, model_tag in enumerate(model_tags):
        table_absolute_cumprod_dict[f"{model_tag} (MWh)"] = [cumprod_farm_list[mii]] + cumprod_turbine_list[mii]
        if mii > 0:  # Compare against first entry (usually SCADA or LES) and calculate errors
            table_absolute_cumprod_dict[f"{model_tag} error (%)"] = (
                [_err(cumprod_farm_list[0], cumprod_farm_list[mii])] +
                [_err(x, y) for x, y in zip(cumprod_turbine_list[0], cumprod_turbine_list[mii])]
            )

    #######################################################################################################################
    ################ Now we do the same exercise, but with wake loss rather than cumulative production ####################
    #######################################################################################################################


    # Compare cumulative energy wake loss relative to most upstream turbines
    cumprod_turbine_waked_list = [np.array([None for _ in range(n_turbs)], dtype=float) for _ in df_list]
    cumprod_turbine_unwaked_list = [np.array([None for _ in range(n_turbs)], dtype=float) for _ in df_list]

    # For each timeseries and for every turbine, determine test and reference power productions
    for ti in range(n_turbs):
        for dii, df in enumerate(df_list):
            p_test = np.array(df[f"pow_{ti:03d}"], dtype=float, copy=True)
            p_ref = np.array(df["pow_ref"], dtype=float, copy=True)
            ids_non_nan = (~np.isnan(p_test)) & (~np.isnan(p_ref))
            cumprod_turbine_unwaked_list[dii][ti] = np.sum(p_ref[ids_non_nan]) / n_measurements_per_hour
            cumprod_turbine_waked_list[dii][ti] = np.sum(p_test[ids_non_nan]) / n_measurements_per_hour

    cumprod_farm_waked_list = [np.sum(cumprod_tm_waked) for cumprod_tm_waked in cumprod_turbine_waked_list]
    cumprod_farm_unwaked_list = [np.sum(cumprod_tm_unwaked) for cumprod_tm_unwaked in cumprod_turbine_unwaked_list]

    # Create placeholder dictionary to collect wake loss results
    table_wakeloss_cumprod_dict = {
        "Selection": ["Entire farm"] + [f"Turbine {ti:02d}" for ti in range(n_turbs)],
    }

    # Calculate wake losses for each model and store in list
    wake_losses_list = [
            np.hstack(
            [
                _wake_loss(cumprod_farm_waked_list[dii], cumprod_farm_unwaked_list[dii]),
                _wake_loss(cumprod_turbine_waked_list[dii], cumprod_turbine_unwaked_list[dii])
            ]
        )
        for dii in range(len(df_list))
    ]

    # Calculate wake loss errors between first and all remaining models, and store in list
    for mii, model_tag in enumerate(model_tags):
        table_wakeloss_cumprod_dict[f"{model_tag} (%)"] = wake_losses_list[mii]
        if mii > 0:  # Compare against first entry (usually SCADA or LES) and calculate error
            table_wakeloss_cumprod_dict[f"{model_tag} error (p.p.)"] = wake_losses_list[mii] - wake_losses_list[0]

    # Finally print
    if print_to_console:
        _print_pretty_table(table_absolute_cumprod_dict, title="Absolute cumulative energy (MWh)")
        print("\n")
        _print_pretty_table(table_wakeloss_cumprod_dict, title="Cumulative energy wake loss (%)")

    return table_absolute_cumprod_dict, table_wakeloss_cumprod_dict
