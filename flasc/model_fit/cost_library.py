"""Library of cost functions for the optimization."""

from __future__ import annotations

from abc import ABCMeta, abstractmethod

from typing import List

import pandas as pd
from floris import FlorisModel

from flasc.data_processing.dataframe_manipulations import (
    _set_col_by_turbines,
    set_pow_ref_by_turbines,
)
from flasc.flasc_dataframe import FlascDataFrame


def total_wake_loss_error(
    df_scada: pd.DataFrame | FlascDataFrame,
    df_floris: pd.DataFrame | FlascDataFrame,
    fm_: FlorisModel,
    turbine_groupings: List = None,
):
    """Evaluate the overall wake loss from pow_ref to pow_test as percent reductions.

    Args:
        df_scada (pd.DataFrame): SCADA data
        df_floris (pd.DataFrame): FLORIS data
        fm_ (FlorisModel): FLORIS model (Unused but required for compatibility)
        turbine_groupings (List): List of turbine groupings.  Defaults to None.
            In None case, assumes pow_ref and pow_test are already identified (note
            this can be challenging to effect within FLORIS resimulation results).

    Returns:
        float: Overall wake losses squared error

    """
    # TODO: make this one work.
    if turbine_groupings is not None:
        # Set the reference turbines in both frames
        df_scada = set_pow_ref_by_turbines(df_scada, turbine_groupings["pow_ref"])
        df_floris = set_pow_ref_by_turbines(df_floris, turbine_groupings["pow_ref"])

        # Set the test turbines in both frames
        df_scada = _set_col_by_turbines(
            "pow_test", "pow", df_scada, turbine_groupings["pow_test"], False
        )
        df_floris = _set_col_by_turbines(
            "pow_test", "pow", df_floris, turbine_groupings["pow_test"], False
        )

    scada_wake_loss = df_scada["pow_ref"].values - df_scada["pow_test"].values
    floris_wake_loss = df_floris["pow_ref"].values - df_floris["pow_test"].values

    return ((scada_wake_loss - floris_wake_loss) ** 2).sum()


def farm_power_error(
    df_scada: pd.DataFrame | FlascDataFrame,
    df_floris: pd.DataFrame | FlascDataFrame,
):
    """Evaluate error with respect to farm power.

    Args:
        df_scada (pd.DataFrame): SCADA data
        df_floris (pd.DataFrame): FLORIS data

    Returns:
        float: Overall wake losses squared error

    """
    df_scada = set_pow_ref_by_turbines(df_scada, list(range(df_scada.n_turbines)))
    df_floris = set_pow_ref_by_turbines(df_floris, list(range(df_scada.n_turbines)))

    error = df_scada["pow_ref"].values - df_floris["pow_ref"].values
    error = error**2
    return error.sum()


def turbine_power_error_sq(
    df_scada: pd.DataFrame | FlascDataFrame,
    df_floris: pd.DataFrame | FlascDataFrame,
):
    """Evaluate error with respect to turbine power.

    Args:
        df_scada (pd.DataFrame): SCADA data
        df_floris (pd.DataFrame): FLORIS data
        fm_ (FlorisModel): FLORIS model (Unused but required for compatibility)
        turbine_groupings (List): List of turbine groupings.  Defaults to None.
            In None case, assumes pow_ref and pow_test are already identified (note
            this can be challenging to effect within FLORIS resimulation results).

    Returns:
        float: Overall wake losses squared error

    """
    turbine_columns = [c for c in df_scada.columns if c[:4] == "pow_" and c[4:].isdigit()]

    df_error = (df_scada[turbine_columns] - df_floris[turbine_columns]) ** 2

    return df_error.mean().mean()


def expected_turbine_power_error(
    df_scada: pd.DataFrame | FlascDataFrame,
    df_floris: pd.DataFrame | FlascDataFrame,
):
    """Evaluate error with respect to expected turbine power.

    Args:
        df_scada (pd.DataFrame): SCADA data
        df_floris (pd.DataFrame): FLORIS data

    Returns:
        float: Overall wake losses squared error

    """
    turbine_columns = [f"pow_{i:03d}" for i in range(df_scada.n_turbines)]

    df_error = (
        df_scada[turbine_columns].mean(axis=0) - df_floris[turbine_columns].mean(axis=0)
    ) ** 2

    return df_error.sum()

class CostFunctionBase(metaclass=ABCMeta):
    """Base class for cost functions."""

    def __init__(self, df_scada: FlascDataFrame | pd.DataFrame):
        """Initialize the cost function class.

        Args:
            df_scada (dataframe): The SCADA data to use in the cost function.
        """
        self.df_scada = FlascDataFrame(df_scada).convert_to_flasc_format()

    @abstractmethod
    def __call__(self, df_floris: pd.DataFrame | FlascDataFrame) -> float:
        """Call the instantiated object to evaluate the cost function.
        
        Abstract method to be implemented by subclasses.
        """
        pass


class TurbinePowerMeanAbsoluteError(CostFunctionBase):
    """Cost function for mean absolute error over all turbines and all times."""

    def __init__(
            self,
            df_scada: pd.DataFrame | FlascDataFrame,
            turbine_power_subset: list | None = None
        ):
        """Initialize the cost function class.

        Args:
            df_scada (dataframe): The SCADA data to use in the cost function.
            turbine_power_subset (list | None): List of turbine indices to use in the cost function.
                If None, all turbines will be used.
        """
        super().__init__(df_scada)

        self._turbine_subset = _process_turbine_powers_subset(self.df_scada, turbine_power_subset)

    def __call__(self, df_floris: pd.DataFrame | FlascDataFrame) -> float:
        """Evaluate the cost function.

        Args:
            df_floris (pd.DataFrame | FlascDataFrame): The FLORIS data to use in the cost function.

        Returns:
            float: The cost value.
        """
        df_error = (self.df_scada[self._turbine_subset] - df_floris[self._turbine_subset]).abs()

        return df_error.mean().mean()


def _process_turbine_powers_subset(df_scada, turbine_power_subset):
        if not isinstance(turbine_power_subset, list) and turbine_power_subset is not None:
            raise TypeError("turbine_power_subset must be a list or None.")

        if turbine_power_subset is None:
            turbine_power_subset = [
                "pow_{0:03d}".format(t) for t in range(df_scada.n_turbines)
            ]
        elif isinstance(turbine_power_subset[0], str):
            if not all([c[:4] == "pow_" and c[4:].isdigit() for c in turbine_power_subset]):
                turbine_power_subset = [
                    df_scada.channel_name_map[c] for c in turbine_power_subset
                ]
        elif isinstance(turbine_power_subset[0], int):
            turbine_power_subset = ["pow_{0:03d}".format(t) for t in turbine_power_subset]
        else:
            raise TypeError(
                "turbine_power_subset must be a list of strings or integers and must",
                " match the turbine names in df_scada."
            )

        return turbine_power_subset

