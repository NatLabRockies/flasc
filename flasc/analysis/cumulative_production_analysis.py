import numpy as np
import pandas as pd

from flasc.data_processing import dataframe_manipulations as dfm


def _err(old_val, new_val):
    """Calculate percentage error between old and new value. Note that this is not
    a symmetric error metric, and should be interpreted as the percentage change from
    old_val to new_val.
    """
    return 100.0 * (new_val-old_val) / old_val


def _wake_loss(cumprod_waked, cumprod_unwaked):
    """Calculate wake loss percentage between waked and unwaked cumulative production. Note that this is not
    a symmetric metric, and should be interpreted as the percentage loss from unwaked to waked production.
    """
    return 100.0 * (cumprod_unwaked-cumprod_waked) / cumprod_unwaked


def _print_pretty_table(table_dict, title):
    """Print the given table dictionary in a pretty format using markdown. The title is centered above the table.

    Args:
        table_dict (dict): Dictionary containing the table data.
        title (str): Title to be displayed above the table.

    Returns:
        None
    """
    df_table = pd.DataFrame(table_dict)
    mrkdwn = df_table.to_markdown(headers='keys', tablefmt='psql', index=False, floatfmt=".2f")
    spc = int(np.floor(len(mrkdwn.split("\n")[0]) / 2 - len(title) / 2))
    mrkdwn = (" " * spc + title + "\n") + mrkdwn
    return print(mrkdwn)


def compare_cumulative_production_and_relative_wake_loss(
    df_list,  # Typical use case: 1st entry is SCADA, then remaining are models to compare. All models are compared against 1st entry.
    df_upstream,
    exclude_turbs=[],
    ws_range=[0.0, 99.0],
    model_tags=None,
    print_to_console=True,
):
    """Calculate the cumulative energy production and the relative wake loss for a list of Pandas DataFrame timeseries. Then,
    calculate the error between the first timeseries in the list (typically SCADA or LES) and the remaining timeseries
    (typically LES and/or FLORIS models). 

    Args:
        df_list (list): List of Pandas DataFrame timeseries. The first entry is typically SCADA or LES, and the remaining entries are models to compare.
        df_upstream (Pandas DataFrame): Upstream data for reference, generated using 'ftools.get_upstream_turbs_floris()'
        exclude_turbs (list, optional): List of turbines to exclude from the analysis, i.e., because of poor performance or odd behavior. Defaults to [].
        ws_range (list, optional): Wind speed range for filtering the data. When inspecting wake losses, one may want to zoom into the relevant wind
        speed range, typically between 6 and 14 m/s. This also allows you to inspect the model performance for different wind speed regions. Defaults to [0.0, 99.0].
        model_tags (list, optional): List of string tags for the models. Defaults to None, which will generate tags as "Model 0", "Model 1", etc.
        print_to_console (bool, optional): Whether to print the results to the console. Defaults to True.

    Raises:
        ValueError: If input timeseries dataframes in df_list have different timestamps.
        ValueError: If input timeseries dataframes in df_list have different number of turbines.
        ValueError: If input timeseries dataframes in df_list already contain a 'pow_ref' column.

    Returns:
        table_absolute_cumprod_dict: Dictionary containing the absolute cumulative production numbers, including errors w.r.t. the first dataframe.
        table_wakeloss_cumprod_dict: Dictionary containing the relative wake loss numbers, including errors w.r.t. the first dataframe.
    """
    # Apply default model tags if not provided
    if model_tags is None:
        model_tags = [f"Model {ti}" for ti in range(len(df_list))]

    # Check input dataframes are consistent in terms of number of turbines and timestamps
    if not all([all(df_list[0]["time"] == df["time"]) for df in df_list]):
        raise ValueError("Input dataframes have different timestamps. Please ensure all dataframes have the same timestamps.")

    n_turbs_list = [dfm.get_num_turbines(df) for df in df_list]
    if len(set(n_turbs_list)) > 1:
        raise ValueError(f"Input dataframes have different number of turbines: {n_turbs_list}. Please ensure all dataframes have the same number of turbines.")

    for dfii, df in enumerate(df_list):
        if "pow_ref" in df.columns:
            raise ValueError(f"Input dataframe[{dfii}] for {model_tags[dfii]} may not contain 'pow_ref' column. This may only happen AFTER mirroring NaNs between dataframes and will be done automatically.")

    # Make local copies of dataframes that we can manipulate
    df_list = [df.copy() for df in df_list]

    # Apply wind speed filter
    N_b = (~df_list[0]["ws"].isna()).sum()  # Number of valid entries in first df before wind speed masking
    ids_within_ws_range = (df_list[0]["ws"] >= ws_range[0]) & (df_list[0]["ws"] <= ws_range[1])
    for dfii in range(len(df_list)):
        df_list[dfii] = df_list[dfii].loc[ids_within_ws_range].reset_index(drop=True)

    N_a = (~df_list[0]["ws"].isna()).sum()  # Number of valid entries in first df after wind speed masking
    if N_a < N_b:
        print(f"Masking down to wind speed range {ws_range[0]:.1f} m/s to {ws_range[1]:.1f} m/s. " +
        f"Number of measurements before: {N_b}, after: {N_a}. Percentage: {100.0 * N_a / N_b:.1f}%.")

    # Apply simple NaN rule for every excluded turbine to avoid annoying edge cases in the analysis
    for ti in exclude_turbs:
        for dfii in range(len(df_list)):
            df_list[dfii][f"pow_{ti:03d}"] = None

    # Helper variables
    n_turbs = dfm.get_num_turbines(df_list[0])
    pow_cols = [f"pow_{ti:03d}" for ti in range(n_turbs)]

    # Mirror NaNs across dataframes to ensure consistent NaN mapping between modelled data and SCADA timeseries.
    # This is important to ensure that we are comparing production and wake losses over the same timestamps,
    # and that we are not unfairly penalizing models for having values where SCADA has NaNs (e.g. due to turbine downtime).
    df_list = dfm.df_mirror_timeseries_nans(df_list, verbose=False)

    for dfii, df in enumerate(df_list):
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
    cumprod_turbine_list = [[(df[pc].sum() * 1.0e-6 / n_measurements_per_hour) for pc in pow_cols] for df in df_list]
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
            cumprod_turbine_unwaked_list[dii][ti] = np.sum(p_ref[ids_non_nan]) * 1.0e-6 / n_measurements_per_hour
            cumprod_turbine_waked_list[dii][ti] = np.sum(p_test[ids_non_nan]) * 1.0e-6 / n_measurements_per_hour

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
        c =  f"Data from {t0.strftime('%Y-%m-%d')} to {t1.strftime('%Y-%m-%d')}; Wind speeds " + f"{ws_range[0]:.1f} m/s to {ws_range[1]:.1f} m/s"
        print("\n")
        _print_pretty_table(table_absolute_cumprod_dict, title=f"Absolute cumulative energy (MWh); {c}")
        print("\n")
        _print_pretty_table(table_wakeloss_cumprod_dict, title=f"Cumulative energy wake loss (%); {c}")

    return table_absolute_cumprod_dict, table_wakeloss_cumprod_dict
