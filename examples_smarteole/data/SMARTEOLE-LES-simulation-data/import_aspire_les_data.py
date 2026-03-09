import os
import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io
from time import perf_counter as timerpc

from datetime import timedelta as td

import tarfile 
import xarray as xr

from zenodo_get import download as zn_download
import zipfile

from floris.utilities import wrap_360
from flasc.data_processing.time_operations import (
    df_downsample,
    df_resample_by_interpolation
)
from flasc.visualization import plot_with_wrapping


class AspireTimeseriesReader():
    """This class is used to read the output .tar.gz files from Whiffle and export
    them in various formats.
    """
    def __init__(self, aspire_metmast_filelist=[], aspire_turbine_filelist=[], verbose=False):
        """Initialize the class.

        Args:
            aspire_filelist (list): List of strings, where each entry of the list defines the
            path to one of the ASPIRE nc files.
        """
        self.metmast_files = aspire_metmast_filelist
        self.turbine_files = aspire_turbine_filelist
        self.turbine_datasets = None
        self.metmast_datasets = None
        self.verbose = verbose

    # Private functions
    def _read_member_as_xarray(self, fn):
        """Read the contents of one of the files within the .tar.gz file as an xarray DataSet.

        Args:
            fn (str): Filename refering to a turbine ASPIRE datafile.

        Returns:
            dataset (xarray.Dataset): Contents of the HDF5 file imported as an xarray Dataset.
        """
        # Now read the turbine .nc (HDF5) data file as an xarray and convert to a Pandas DataFrame
        dataset = xr.open_dataset(fn).load()  # Load it into xarray format
        dataset.close()  # Close dataset after reading to avoid conflicts

        if self.verbose:
            print(f"Successfully imported the contents of {os.path.basename(fn)}.")

        return dataset

    def _concatenate_dataframes_and_remove_startup_time(self, df_list):
        """Typically, ASPIRE simulations are performed one day at a time, including several hours of
        simulation start-up. This means that each day runs for about 26 hours, and that means there
        is an overlap window of about 2 hours between simulations. The start-up period, usually the
        first two hours of the simulation, must be removed so that each datafile contains exactly
        24 hours of data. This script removes the startup periods and concatenates the Pandas
        DataFrames into a single file.

        Args:
            df_list (list): List where each entry is a Pandas DataFrame containing the simulation data
            for one day from ASPIRE. These datasets typically start about 2 hours before the actual
            day of simulation. These 2 hours will be removed so that it perfectly connects with the
            simulation data of the previous day.

        Returns:
            df_out (pd.DataFrame): Pandas DataFrame containing multiple days of simulation data,
            with the start-up periods and overlapping measurement times removed.
        """
        # Now stitch dataframes together while removing start-up periods
        for ii, df in enumerate(df_list):
            if ii == 0:
                # The dataset usually starts a couple hours before midnight in the day before the actual day
                # of the simulation. That is the start-up period. We must remove that period from the dataset.
                first_day_change = np.where(np.diff([t.day for t in df["time"]]) != 0)[0][0] + 2
                df = df.loc[first_day_change::]  # Remove start-up period from first file
            else:
                # For every file besides the first, we can see where the previous simulation ended and make sure
                # we remove measurement data of this simulation that happens *before* the latest simulation time
                # of the previous file. Namely, that period is considered the start-up period for this file and
                # should be removed.
                dt = df["time"].diff().median()  # Average duration between timesteps
                df = df.loc[df["time"] > t_end_prev_simulation + 0.5 * dt]  # Only keep timesteps at least 5 minutes past the last measurement from the previous dataset

            # Update the dataframe with the start-up measurements removed
            df_list[ii] = df
            t_end_prev_simulation = df.iloc[-1]["time"]

        # Collect all the outputs together into a single DataFrame and sort it chronologically
        df_out = pd.concat(df_list, axis=0).sort_values(by="time").reset_index(drop=True)
        return df_out

    def get_turbine_hub_heights(self):
        """Extract the turbine hub heights from the imported turbine data files.

        Returns:
            hub_heights (np.array): Array with length equal to the number of turbines, containing the hub height
            value for each wind turbine in the ASPIRE simulation.
        """
        if self.turbine_datasets is None:
            raise UserWarning("Cannot extract turbine hub heights. Please read the files first using get_turbine_data_as_xarrays().")

        # Extract hub heights from the first file
        hub_heights = np.array(self.turbine_datasets[0]["ztur"], dtype=float)
        return hub_heights

    def get_turbine_data_as_xarrays(self):
        """Import the turbine HDF5 file from each .tar.gz file and format them as xarray Datasets.

        Returns:
            turbine_datasets: List of xarray Datasets, one for each tarball file, containing
            the turbine simulation output data.
        """
        turbine_datasets = []
        for fn in self.turbine_files:
            # Now read the turbine .nc (HDF5) data file as an xarray
            turbine_dataset = self._read_member_as_xarray(fn)
            turbine_datasets.append(turbine_dataset)

        self.turbine_datasets = turbine_datasets
        return turbine_datasets

    def get_turbine_data_as_dataframe(self, variables=None):
        """Convert the turbine data that is formatted as xarray Datasets to a single Pandas DataFrame
        that is easy to investigate and do analysis with.

        Args:
            variables (list, optional): List of turbine measurement variables that should be exported
            from the simulation file. Defaults to ["ptur", "ufsf", "vfsf"].

        Returns:
            df_out (pd.DataFrame): Pandas DataFrame containing the turbine measurement data in a
            wide table format, where there is one column for each turbine and for each variable.
            Each rows depicts the timestamp of one set of measurements. The overlapping time windows
            of about 2 hours between each day of ASPIRE simulation has been removed so that the
            start-up periods are removed and the dataset is monotonically increasing with 10-minute
            timesteps.
        """
        # Load the turbine data files, if we haven't done that yet
        if self.turbine_datasets is None:
            self.get_turbine_data_as_xarrays()  # Get all turbine data as xarrays

        # Default options:
        if variables is None:
            if "ufsf" in list(self.turbine_datasets[0].data_vars):
                variables = ["ptur", "ufsf", "vfsf"]
                print(f"WARNING: Using legacy variable naming convention from GRASP, '{variables}'")
            elif "Mdfs" in list(self.turbine_datasets[0].data_vars):
                variables = ["ptur", "Mdfs", "cosangle", "sinangle"]
            else:
                raise UserWarning("Unfamiliar variable naming convention in GRASP datafiles.")
    
        # Convert files one by one to Pandas DataFrames
        df_list = []
        num_turbines = len(self.turbine_datasets[0].turbine)
        for turbine_dataset in self.turbine_datasets:
            # Convert the data from each file into a 'wide' Pandas DataFrame
            df_turbine = turbine_dataset[variables].to_dataframe().unstack()
            
            # Update column names in wide format following FLASC format
            columns = []
            for var in variables:
                columns = columns + [f"{var:s}_{ti:03d}" for ti in range(num_turbines)]
            df_turbine.columns = columns
            df_turbine = df_turbine.reset_index()
            
            # Finally, append it to a list that we will stitch together later
            df_list.append(df_turbine)

        # Concatenate all the individual dataframes and deal with overlap in timeseries entries
        df_out = self._concatenate_dataframes_and_remove_startup_time(df_list)
        return df_out

    def construct_flasc_timeseries(self):
        """

        """
        # First, get the turbine information, if we dont have those in memory yet
        self.get_turbine_data_as_dataframe()
        df_tot = self.get_turbine_data_as_dataframe()
        
        # Calculate turbine wind speed and wind direction, if variables available
        n_turbines = len([c for c in df_tot.columns if c.startswith("ptur_")])
        df_tot.columns = [c.replace("ptur_", "pow_") for c in df_tot.columns]

        dict_out = {}
        for ti in range(n_turbines):
            u = df_tot[f"cosangle_{ti:03d}"] * df_tot[f"Mdfs_{ti:03d}"]
            v = df_tot[f"sinangle_{ti:03d}"] * df_tot[f"Mdfs_{ti:03d}"]
            dict_out[f"ws_{ti:03d}"] = np.sqrt(u**2.0 + v**2.0)
            dict_out[f"wd_{ti:03d}"] = wrap_360(180.0 + np.rad2deg(np.arctan2(u, v)))
    
        # Append local wind speed and direction measurements to the dataframe
        df_tot = pd.concat([df_tot, pd.DataFrame(dict_out)], axis=1)
        df_tot = df_tot.drop(columns=[c for c in df_tot.columns if c.startswith("cosangle_")])
        df_tot = df_tot.drop(columns=[c for c in df_tot.columns if c.startswith("sinangle_")])
        df_tot = df_tot.drop(columns=[c for c in df_tot.columns if c.startswith("Mdfs_")])

        return df_tot


if __name__ == "__main__":
    # Download files from Zenodo
    root_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(root_path, "data")
    zn_download("10.5281/zenodo.18888663", output_dir=data_path)

    # Unzip the LES timeseries data
    path_to_zip_file = os.path.join(data_path, "les_output.zip")
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(data_path)

    aspire_turbine_files = glob.glob(os.path.join(data_path, "les_output", "2020", "*", "*", "00", "turbinesOut.les.nc"))
    aspire_turbine_files = np.sort(aspire_turbine_files)

    # Load them one at a time
    atr = AspireTimeseriesReader(
        aspire_turbine_filelist=aspire_turbine_files,
        verbose=True
    )
    df_turbine = atr.get_turbine_data_as_dataframe()
    df_les_timeseries = atr.construct_flasc_timeseries()

    # Resample to 10 min steps
    t0 = str(df_les_timeseries["time"].iloc[0])[0:17] + "00"  # Create time array rounded to nearest 10-min averages
    t1 = str(df_les_timeseries["time"].iloc[-1])[0:17] + "00"
    time_array = pd.date_range(start=t0, end=t1, freq="10min").tolist()
    df_les_timeseries = df_resample_by_interpolation(
        df=df_les_timeseries,
        time_array=time_array,  # Interpolate onto the same timeseries that df_metmast uses
        circular_cols=[c for c in df_les_timeseries.columns if c.startswith("wd_")],  # No variables in turbine measurement dataset that require circular averaging
        interp_method="linear",  # Use linear interpolation
        max_gap=td(minutes=20),  # Do not interpolate over gaps larger than 20 minutes between measurements
        verbose=False
    )

    # Save as .csv
    fout = os.path.join(root_path, "les_timeseries.csv")
    df_les_timeseries.to_csv(fout, index=False, float_format="%.3f")
    print("Converted LES timeseries data has been saved to: " + fout)
