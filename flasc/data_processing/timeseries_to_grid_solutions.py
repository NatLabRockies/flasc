import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import circmean

class ConvertTimeseriesToSolutionGrid():
    def __init__(self, df_timeseries):
        self.df_timeseries = df_timeseries

    # Private functions: plotting
    def _plot_binned_data_counts_for_theoretical_ws_to_ti_curve(self):
        # Load df_binned from self
        df_binned = self.df_binned

        df = df_binned[["wd_bin", "ws_bin", "ti_bin", "N", "ti", "ti_theoretical_curve"]].set_index(["wd_bin", "ws_bin"]).unstack().transpose()
        df_N = df.loc["N"].astype(float)
        df_ti = df.loc["ti"].astype(float)
        df_ti_theoretical = df.loc["ti_theoretical_curve"].astype(float)
        ti_limits_list = [[a.left, a.right] for a in df.loc["ti_bin"].mode(axis=1)[0]]

        fig, ax = plt.subplots(figsize=(24, 12))
        sns.heatmap(df_N, annot=True, annot_kws={"fontsize": 8}, linewidth=.5, cmap="rocket_r", ax=ax, fmt=".0f", cbar=False)
        ax.set_ylabel("Wind speed bin (m/s), turbulence intensity interval (-)")
        ax.set_xlabel("Wind direction bin (deg)")
        ax.set_yticklabels([f"WS: {ws}, TI: [{ti[0]:.3f}, {ti[1]:.3f}]" for ws, ti in zip(df_N.index, ti_limits_list)])
        ax.set_title(f"Sample size for fitting turbulence intensity ranges")
        plt.tight_layout()

        fig, ax = plt.subplots(figsize=(24, 12))
        sns.heatmap(100.0*(df_ti - df_ti_theoretical), annot=True, annot_kws={"fontsize": 8}, linewidth=.5, cmap="vlag", ax=ax, fmt=".1f", cbar=False)
        ax.set_ylabel("Wind speed bin (m/s), turbulence intensity interval (-)")
        ax.set_xlabel("Wind direction bin (deg)")
        ax.set_yticklabels([f"WS: {ws}, TI: [{ti[0]:.3f}, {ti[1]:.3f}]" for ws, ti in zip(df_N.index, ti_limits_list)])
        ax.set_title(f"Turbulence intensity from ASPIRE minus desired theoretical curve")
        plt.tight_layout()

    def _plot_binned_data_counts_per_ti_bin(self):
        # Load df_binned from self
        df_binned = self.df_binned

        for ti_interval in df_binned["ti_bin"].unique():
            df_binned_ti = df_binned[df_binned["ti_bin"] == ti_interval]
            df_N = df_binned_ti[["wd_bin", "ws_bin", "N"]].set_index(["wd_bin", "ws_bin"]).unstack().transpose()
            df_N.index = [i[1] for i in df_N.index]
                
            fig, ax = plt.subplots(figsize=(24, 12))
            sns.heatmap(df_N, annot=True, annot_kws={"fontsize": 8}, linewidth=.5, cmap="rocket_r", ax=ax, fmt=".0f", cbar=False)
            ax.set_ylabel("Wind speed bin (m/s)")
            ax.set_xlabel("Wind direction bin (deg)")
            ax.set_title(f"Sample size for ambient turbulence intensity interval: {ti_interval}")
            plt.tight_layout()

    # Public functions
    def bin_timeseries(self, wd_step=5.0, ws_array=np.arange(3.5, 17.501, 1.0), ti_array=np.arange(0.0, 0.1501, 0.05), circular_cols=None, plot=False, verbose=True):
        # Import the timeseries
        df_timeseries = self.df_timeseries.copy()

        # Automatically identify circular columns if left unspecified
        if circular_cols is None:
            circular_cols = [c for c in df_timeseries.columns if c.startswith("wd_")]
            if verbose:
                print(f"No circular columns specified. Automatically identified: {circular_cols}.")
        
        # Manipulate the wind direction in the timeseries so that the center of the first wind direction bin is at 0 deg
        df_timeseries.loc[(df_timeseries["wd"] > 360.0 - 0.50 * wd_step), "wd"] += -360.0  # Wrap backwards past zero
        wd_array = np.arange(-0.50 * wd_step, 360.0001 - 0.5 * wd_step, wd_step, dtype=float)
    
        ws_bins = pd.cut(df_timeseries["ws"], ws_array).rename("ws_bin")
        wd_bins = pd.cut(df_timeseries["wd"], wd_array).rename("wd_bin")
        ti_bins = pd.cut(df_timeseries["ti"], ti_array).rename("ti_bin")

        # Expand the dataframe: add the bins and add a "count" column
        df = pd.concat(
            [wd_bins, ws_bins, ti_bins, df_timeseries, pd.DataFrame({"N": np.ones(len(wd_bins))})],
            axis=1
        )
        if "time" in df.columns:
            df = df.drop(columns=["time"])  # Drop the 'time' column

        # Now group by the bins and get bin averages and sums accordingly
        df_grouped = df.groupby(["wd_bin", "ws_bin", "ti_bin"], observed=True)

        # Apply a blank 'mean' for all variables
        if verbose:
            print(f"Applying regular averaging on entire dataframe ({df.shape[1]} columns).")
        df_binned = df_grouped.mean()

        # Now replace the directional measurements with a circular average
        if verbose:
            print(f"Applying circular averaging on {len(circular_cols)} columns.")
        for c in circular_cols:
            # Define circular average function to process
            def circular_mean(x):
                return circmean(x[c].values, low=0.0, high=360.0)
            df_binned[c] = df_grouped.apply(circular_mean)

        df_binned["N"] = np.array(df_grouped["N"].sum(), dtype=int)
        df_binned = df_binned.reset_index(drop=False)

        self.df_binned = df_binned

        if plot:
            self._plot_binned_data_counts_per_ti_bin()

        return df_binned

    def bin_timeseries_by_theoretical_ws_to_ti_curve(self, ws_to_ti_function, wd_step=5.0, ws_array=np.arange(3.5, 17.501, 1.0), ti_margin=0.01, plot=False):
        # Generate ws_bins array
        ws_intervals = [pd.Interval(l, r) for l, r in zip(ws_array[:-1], ws_array[1:])]
        df_binned_list = []
        for ws_interval in ws_intervals:
            # Find the fitting turbulence intensity range.
            ti_limits = [ws_to_ti_function(l) for l in [ws_interval.left, ws_interval.right]]
            ti_limits = np.array(np.sort(ti_limits)) + ti_margin * np.array([-1, 1])  # Add additional margin
            df_binned_ti = self.bin_timeseries(wd_step=wd_step, ws_array=[ws_interval.left, ws_interval.right], ti_array=ti_limits, plot=False, verbose=False)
            df_binned_ti["ti_theoretical_curve"] = ws_to_ti_function(df_binned_ti["ws"])  # Add theoretical TI value according to curve
            df_binned_list.append(df_binned_ti)

        df_binned = pd.concat(df_binned_list, axis=0, ignore_index=True).reset_index(drop=True)
        self.df_binned = df_binned
        
        if plot:
            self._plot_binned_data_counts_for_theoretical_ws_to_ti_curve()

        return df_binned

    def convert_binned_dataframe_to_df_fi_approx(self, N_min: int = 3):
        """Convert the binned dataframe to a df_fi_approx format, similar to the FLASC floris_tools definition.

        Returns:
            N_min (int): Minimal number of timestamps (samples) within this wind direction and wind speed bin for a
              them to be considered valid and useable to generate an approximate table entry with. Defaults to 3.
        """
        # Retrieve df_binned from self
        df_binned = self.df_binned.copy()

        # Define column names
        wd_bin_center = [i.mid for i in df_binned["wd_bin"]]
        ws_bin_center = [i.mid for i in df_binned["ws_bin"]]
        ti_bin_center = [i.mid for i in df_binned["ti_bin"]]
        pow_cols = [c for c in df_binned.columns if c.startswith("pow_") and len(c) == 7]
        mm_cols = [c for c in df_binned.columns if c.endswith("_metmast")]
        ws_cols = [c for c in df_binned.columns if c.startswith("ws_") and len(c) == 6 and not c == "ws_bin"]
        wd_cols = [c for c in df_binned.columns if c.startswith("wd_") and len(c) == 6 and not c == "wd_bin"]

        # Mark specific entries as NaN if we require more than 1 measurement for it to be valid
        if N_min > 1:
            ids_faulty = df_binned["N"] < N_min
            df_binned.loc[ids_faulty, pow_cols + ws_cols + wd_cols] = None
            print(f"Flagged an additional {np.sum(ids_faulty)} table entries as NaN because their sample pool is smaller than {N_min:d} (N_min).")

        # Construct sparse array
        df_sparse_list = [
            pd.DataFrame({"wd": wd_bin_center, "ws": ws_bin_center, "ti": ti_bin_center}),
            df_binned[mm_cols + wd_cols + ws_cols + pow_cols]
        ]
        df_approx_sparse = pd.concat(df_sparse_list, axis=1)

        # Create a full dataframe with all wind direction and wind speed combinations
        wd_grid, ws_grid = np.meshgrid(np.unique(df_approx_sparse["wd"]), np.unique(df_approx_sparse["ws"]), indexing="ij")
        conditions_to_add = []
        for wd, ws in zip(wd_grid.flatten(), ws_grid.flatten()):
            if not any((wd == df_approx_sparse["wd"]) & (ws == df_approx_sparse["ws"])):
                conditions_to_add.append([wd, ws])

        # Expand sparse dataframe
        df_approx = pd.concat([df_approx_sparse, pd.DataFrame(np.vstack(conditions_to_add), columns=["wd", "ws"])], axis=0)
        df_approx = df_approx.sort_values(by=["ws", "wd"]).reset_index(drop=True)

        # Format columns
        df_approx[pow_cols] = df_approx[pow_cols].astype(float)
        df_approx[ws_cols] = df_approx[ws_cols].astype(float)
        df_approx[wd_cols] = df_approx[wd_cols].astype(float)

        # Print number of valid and faulty entries
        no_tot_entries = df_approx.shape[0]
        no_faulty_entries = df_approx.isna().any(axis=1).sum()
        no_valid_entries = df_approx.shape[0] - no_faulty_entries
        print(f"Approximate table contains {no_valid_entries:d} valid entries ({100*no_valid_entries/no_tot_entries:.2f}%) and {no_faulty_entries:d} invalid entries ({100*no_faulty_entries/no_tot_entries:.2f}%).")

        return df_approx