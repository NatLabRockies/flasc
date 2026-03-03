import numpy as np
import pandas as pd

from flasc.data_processing import dataframe_manipulations as dfm


# Introduce helper functions for calculating errors and generating plots/tables
def _err(old_val, new_val):
    return 100.0 * (new_val-old_val) / old_val


# Introduce helper functions for calculating wake loss (%)
def _wake_loss(aep_waked, aep_unwaked):
    return 100.0 * (aep_unwaked-aep_waked) / aep_unwaked


# A function to print table
def _print_pretty_table(table_dict, title):
    # Format title to be centered above the table, and print table in pretty format
    df_table = pd.DataFrame(table_dict)
    mrkdwn = df_table.to_markdown(headers='keys', tablefmt='psql', index=False, floatfmt=".2f")
    spc = int(np.floor(len(mrkdwn.split("\n")[0]) / 2 - len(title) / 2))
    mrkdwn = (" " * spc + title + "\n") + mrkdwn
    return print(mrkdwn)


def compare_absolute_aep_and_relative_wake_loss(
    df_list,  # Typical use case: 1st entry is SCADA, then remaining are models to compare. All models are compared against 1st entry.
    df_upstream,
    exclude_turbs=[],
    model_tags=None,
    print_to_console=True,
):
    # Input formatting options
    if isinstance(df_list, pd.DataFrame):
        df_list = [df_list]

    if model_tags is None:
        model_tags = [f"Model {ti}" for ti in range(len(df_list))]

    # Make local copies of dataframes that we can manipulate
    df_list = [df.copy() for df in df_list]

    # Apply simple NaN rule for every excluded turbine to avoid annoying edge cases in the analysis
    for ti in exclude_turbs:
        for df in df_list:
            df[f"pow_{ti:03d}"] = None
    
    # First ensure consistent NaN mapping between modelled data and SCADA timeseries
    n_turbs = dfm.get_num_turbines(df_list[0])
    for ti in range(n_turbs):
        # For each individual turbine, identify all timestamps where any of the timeseries have NaN values, and create a combined mask of these
        ids_nan = np.zeros(len(df_list[0]), dtype=bool)
        for df in df_list:
            ids_nan = ids_nan | df[f"pow_{ti:03d}"].isna()
        ids_nan = np.where(ids_nan)[0]

        # Mirror NaNs across all timeseries
        for df in df_list:
            df.loc[ids_nan, f"pow_{ti:03d}"] = None

    # Helper variable
    pow_cols = [f"pow_{ti:03d}" for ti in range(n_turbs)]

    # Assert NaNs are identical between dataframes
    n_nans_per_timeseries = [df[pow_cols].isna().sum().sum() for df in df_list]
    print(f"NaNs in df_list power columns: {n_nans_per_timeseries}")

    for df in df_list:
        # Specify upstream power in the exact same way as with the SCADA data
        df = dfm.set_pow_ref_by_upstream_turbines(df, df_upstream, exclude_turbs=exclude_turbs)

        # Determine number of turbines available at any given time, similar to SCADA
        df["n_turbs_available"] = (~df[pow_cols].isna()).sum(axis=1)

    # Assert that number of turbines available at each timestamp is identical between SCADA and LES simulation
    print(all(df_list[0]["n_turbs_available"] == df_list[1]["n_turbs_available"]))

    #######################################################################################################################
    ################## Compare absolute AEP directly between SCADA and simulated data (LES) ###############################
    #######################################################################################################################

    # Absolute AEP for the entire farm and per turbine
    aep_turbine_list = [[df[pc].sum() for pc in pow_cols] for df in df_list]
    aep_farm_list = [np.sum(aep_tm) for aep_tm in aep_turbine_list]

    table_absolute_aep_dict = {
        "Selection": ["Entire farm"] + [f"Turbine {ti:02d}" for ti in range(n_turbs)],
    }
    for mii, model_tag in enumerate(model_tags):
        table_absolute_aep_dict[f"{model_tag} (MWh)"] = [aep_farm_list[mii]] + aep_turbine_list[mii]
        if mii > 0:  # Compare against first entry (usually SCADA or LES) and calculate errors
            table_absolute_aep_dict[f"{model_tag} error (%)"] = (
                [_err(aep_farm_list[0], aep_farm_list[mii])] +
                [_err(x, y) for x, y in zip(aep_turbine_list[0], aep_turbine_list[mii])]
            )

    #######################################################################################################################
    ################ Now we do the same exercise, but with wake loss rather than absolute AEP #############################
    #######################################################################################################################


    # Compare AEP wake loss relative to most upstream turbines
    aep_turbine_waked_list = [np.array([None for _ in range(n_turbs)], dtype=float) for _ in df_list]
    aep_turbine_unwaked_list = [np.array([None for _ in range(n_turbs)], dtype=float) for _ in df_list]

    # For each timeseries and for every turbine, determine test and reference power productions
    for ti in range(n_turbs):
        for dii, df in enumerate(df_list):
            p_test = np.array(df[f"pow_{ti:03d}"], dtype=float, copy=True)
            p_ref = np.array(df["pow_ref"], dtype=float, copy=True)
            ids_non_nan = (~np.isnan(p_test)) & (~np.isnan(p_ref))
            aep_turbine_unwaked_list[dii][ti] = np.sum(p_ref[ids_non_nan])
            aep_turbine_waked_list[dii][ti] = np.sum(p_test[ids_non_nan])

    aep_farm_waked_list = [np.sum(aep_tm_waked) for aep_tm_waked in aep_turbine_waked_list]
    aep_farm_unwaked_list = [np.sum(aep_tm_unwaked) for aep_tm_unwaked in aep_turbine_unwaked_list]

    # Create placeholder dictionary to collect wake loss results
    table_wakeloss_aep_dict = {
        "Selection": ["Entire farm"] + [f"Turbine {ti:02d}" for ti in range(n_turbs)],
    }

    # Calculate wake losses for each model and store in list
    wake_losses_list = [
            np.hstack(
            [
                _wake_loss(aep_farm_waked_list[dii], aep_farm_unwaked_list[dii]),
                _wake_loss(aep_turbine_waked_list[dii], aep_turbine_unwaked_list[dii])
            ]
        )
        for dii in range(len(df_list))
    ]

    # Calculate wake loss errors between first and all remaining models, and store in list
    for mii, model_tag in enumerate(model_tags):
        table_wakeloss_aep_dict[f"{model_tag} (%)"] = wake_losses_models[mii]
        if mii > 0:  # Compare against first entry (usually SCADA or LES) and calculate error
            table_wakeloss_aep_dict[f"{model_tag} error (p.p.)"] = wake_losses_models[mii] - wake_losses_models[0]

    # Finally print
    if print_to_console:
        _print_pretty_table(table_absolute_aep_dict, title="Absolute cumulative energy (MWh)")
        print("\n")
        _print_pretty_table(table_wakeloss_aep_dict, title="Cumulative energy wake loss (%)")

    return table_absolute_aep_dict, table_wakeloss_aep_dict
