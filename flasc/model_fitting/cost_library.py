"""Library of cost functions for the model fitting optimization."""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import List

import pandas as pd

from flasc.data_processing.dataframe_manipulations import (
    set_col_by_turbines,
    set_pow_ref_by_turbines,
)
from flasc.flasc_dataframe import FlascDataFrame


class CostFunctionBase(metaclass=ABCMeta):
    """Base class for cost functions."""

    def __init__(self, df_scada: pd.DataFrame | FlascDataFrame | None = None):
        """Initialize the cost function class.

        Args:
            df_scada (dataframe): The SCADA data to use in the cost function.
        """
        self.assign_df_scada(df_scada)
        self._is_initialized_for_evaluation = False

    @property
    def df_scada(self) -> pd.DataFrame | FlascDataFrame | None:
        """Get the SCADA dataframe."""
        if self._df_scada is None:
            raise AttributeError("SCADA dataframe has not been assigned to cost object.")
        return self._df_scada

    def assign_df_scada(self, df_scada: pd.DataFrame | FlascDataFrame | None):
        """Assign the SCADA dataframe."""
        if (
            hasattr(self, "_df_scada")
            and self._df_scada is not None
            and not self._df_scada.equals(df_scada)
        ):
            print("Cost object already has df_scada assigned. Overwriting.")
        if df_scada is not None:
            self._df_scada = FlascDataFrame(df_scada).convert_to_flasc_format()
        else:
            self._df_scada = None

    @property
    def is_initialized_for_evaluation(self) -> bool:
        """Check if the cost function is ready for evaluation."""
        return self._is_initialized_for_evaluation

    @is_initialized_for_evaluation.setter
    def is_initialized_for_evaluation(self, value: bool):
        self._is_initialized_for_evaluation = value

    def initialize_for_evaluation(self):
        """Initialize the cost function for evaluation. Called before the first evaluation.

        This method will be called before evaluating the cost function for the first time, and
        should set the `initialized_for_evaluation` property to True.

        Subclasses may override this method to perform additional setup before evaluation.
        """
        self.is_initialized_for_evaluation = True

    def prepare_df_floris_for_evaluation(self, df_floris: pd.DataFrame | FlascDataFrame):
        """Prepare the cost function for evaluation. Called each time before evaluation."""
        return df_floris

    def __call__(self, df_floris: pd.DataFrame | FlascDataFrame) -> float:
        """Call the instantiated object to evaluate the cost function.

        Abstract method to be implemented by subclasses.
        """
        if not self.is_initialized_for_evaluation:
            self.initialize_for_evaluation()
        df_floris = self.prepare_df_floris_for_evaluation(df_floris)

        return self.cost(df_floris)

    @abstractmethod
    def cost(self, df_floris: pd.DataFrame | FlascDataFrame) -> float:
        """Evaluate the cost function.

        All subclasses must implement this method.

        Args:
            df_floris (pd.DataFrame | FlascDataFrame): The FLORIS data to use in the cost function.

        Returns:
            float: The cost value.
        """
        raise NotImplementedError(
            "Subclasses of CostFunctionBase must implement a cost() method. "
            "This method should take a dataframe (df_floris) as an input and return a float."
        )


class TurbinePowerErrorBase(CostFunctionBase):
    """Base class for cost functions based on the error between SCADA and FLORIS turbine powers."""

    def __init__(
        self,
        df_scada: pd.DataFrame | FlascDataFrame | None = None,
        turbine_power_subset: list | None = None,
    ):
        """Initialize the cost function class.

        Args:
            df_scada (dataframe): The SCADA data to use in the cost function.
            turbine_power_subset (list | None): List of turbine indices to use in the cost function.
                If None, all turbines will be used.
        """
        super().__init__(df_scada)

        # Save other parameters for now. These will be processed in the prepare method.
        self._turbine_power_subset = turbine_power_subset

    def initialize_for_evaluation(self):
        """Prepare the cost function for evaluation."""
        self._turbine_power_subset = self.process_turbine_powers_subset(
            self.df_scada, self._turbine_power_subset
        )

    def compute_errors(self, df_floris: pd.DataFrame | FlascDataFrame) -> pd.DataFrame:
        """Compute the errors between the SCADA and FLORIS turbine powers.

        Args:
            df_floris (pd.DataFrame | FlascDataFrame): The FLORIS data to use in the cost function.

        Returns:
            pd.DataFrame: DataFrame of errors between SCADA and FLORIS turbine powers.
        """
        return self.df_scada[self._turbine_power_subset] - df_floris[self._turbine_power_subset]

    @staticmethod
    def process_turbine_powers_subset(df_scada, turbine_power_subset):
        """Process the turbine_power_subset parameter."""
        if not isinstance(turbine_power_subset, list) and turbine_power_subset is not None:
            raise TypeError("turbine_power_subset must be a list or None.")

        if turbine_power_subset is None:
            turbine_power_subset = ["pow_{0:03d}".format(t) for t in range(df_scada.n_turbines)]
        elif isinstance(turbine_power_subset[0], str):
            if not all([c[:4] == "pow_" and c[4:].isdigit() for c in turbine_power_subset]):
                turbine_power_subset = [df_scada.channel_name_map[c] for c in turbine_power_subset]
        elif isinstance(turbine_power_subset[0], int):
            turbine_power_subset = ["pow_{0:03d}".format(t) for t in turbine_power_subset]
        else:
            raise TypeError(
                "turbine_power_subset must be a list of strings or integers and must",
                " match the turbine names in df_scada.",
            )

        return turbine_power_subset


class TurbinePowerMeanAbsoluteError(TurbinePowerErrorBase):
    """Cost function for mean absolute error over all turbines and all times."""

    def cost(self, df_floris: pd.DataFrame | FlascDataFrame) -> float:
        """Evaluate the mean absolute error of the turbine powers over all turbines and times.

        Args:
            df_floris (pd.DataFrame | FlascDataFrame): The FLORIS data to use in the cost function.

        Returns:
            float: The cost value.
        """
        df_error = self.compute_errors(df_floris)

        return df_error.abs().mean().mean()


class TurbinePowerRootMeanSquaredError(TurbinePowerErrorBase):
    """Cost function for root mean squared error over all turbines and all times."""

    def cost(self, df_floris: pd.DataFrame | FlascDataFrame) -> float:
        """Evaluate the mean squared error of the turbine powers over all turbines and times.

        Args:
            df_floris (pd.DataFrame | FlascDataFrame): The FLORIS data to use in the cost function.

        Returns:
            float: The cost value.
        """
        df_error = self.compute_errors(df_floris)

        return (df_error**2).mean().mean() ** 0.5


class FarmPowerErrorBase(CostFunctionBase):
    """Base class for cost functions based on the error between SCADA and FLORIS farm powers."""

    def __init__(self, df_scada: pd.DataFrame | FlascDataFrame | None = None):
        """Initialize the cost function class.

        Args:
            df_scada (dataframe): The SCADA data to use in the cost function.
        """
        super().__init__(df_scada)

    def compute_errors(self, df_floris: pd.DataFrame | FlascDataFrame) -> pd.Series:
        """Compute the errors between the SCADA and FLORIS farm powers.

        Args:
            df_floris (pd.DataFrame | FlascDataFrame): The FLORIS data to use in the cost function.

        Returns:
            pd.DataFrame: DataFrame of errors between SCADA and FLORIS farm powers.
        """
        pow_columns = ["pow_{0:03d}".format(t) for t in range(self.df_scada.n_turbines)]
        pow_farm_scada = self.df_scada[pow_columns].sum(axis=1)
        pow_farm_floris = df_floris[pow_columns].sum(axis=1)

        return pow_farm_scada - pow_farm_floris


class FarmPowerMeanAbsoluteError(FarmPowerErrorBase):
    """Cost function for mean absolute error of farm power over all times."""

    def cost(self, df_floris: pd.DataFrame | FlascDataFrame) -> float:
        """Evaluate cost function.

        Args:
            df_floris (pd.DataFrame | FlascDataFrame): The FLORIS data to use in the cost function.

        Returns:
            float: The cost value.
        """
        return self.compute_errors(df_floris).abs().mean()


class FarmPowerRootMeanSquaredError(FarmPowerErrorBase):
    """Cost function for root mean squared error of farm power over all times."""

    def cost(self, df_floris: pd.DataFrame | FlascDataFrame) -> float:
        """Evaluate cost function.

        Args:
            df_floris (pd.DataFrame | FlascDataFrame): The FLORIS data to use in the cost function.

        Returns:
            float: The cost value.
        """
        return (self.compute_errors(df_floris) ** 2).mean() ** 0.5


class WakeLossRootMeanSquaredError(CostFunctionBase):
    """Cost function for the overall wake loss RMSE between SCADA and FLORIS data."""

    def __init__(
        self,
        df_scada: pd.DataFrame | FlascDataFrame | None = None,
        reference_turbines: List[List[int]] | None = None,
        test_turbines: List[List[int]] | None = None,
    ):
        """Initialize the cost function class.

        Args:
            df_scada (dataframe): The SCADA data to use in the cost function.
            reference_turbines (List[List[int]] | None): List of lists of turbine indices to use as
                reference (free stream) turbines for wake loss calculations
            test_turbines (List[List[int]] | None): List of lists of turbine indices to use as test
                (waked) turbines for wake loss calculations.
        """
        super().__init__(df_scada)

        if reference_turbines is None or test_turbines is None:
            raise ValueError(
                "Both reference_turbines and test_turbines must be provided as lists of lists."
            )
        self.reference_turbines = reference_turbines
        self.test_turbines = test_turbines

    def initialize_for_evaluation(self):
        """Apply the reference and test turbines to the SCADA dataframe."""
        self.assign_df_scada(set_pow_ref_by_turbines(self.df_scada, self.reference_turbines))
        self.assign_df_scada(
            set_col_by_turbines("pow_test", "pow", self.df_scada, self.test_turbines, False)
        )

        self.is_initialized_for_evaluation = True

    def prepare_df_floris_evaluation(self, df_floris: pd.DataFrame | FlascDataFrame):
        """Apply the reference and test turbines to the FLORIS dataframe."""
        df_floris = set_pow_ref_by_turbines(df_floris, self.reference_turbines)
        df_floris = set_col_by_turbines("pow_test", "pow", df_floris, self.test_turbines, False)

        return df_floris

    def cost(self, df_floris: pd.DataFrame | FlascDataFrame) -> float:
        """Evaluate the overall wake loss error.

        Args:
            df_floris (pd.DataFrame | FlascDataFrame): The FLORIS data to use in the cost function.

        Returns:
            float: The overall wake loss error.
        """
        df_floris = self.prepare_df_floris_evaluation(df_floris)

        scada_wake_loss = self.df_scada["pow_ref"].values - self.df_scada["pow_test"].values
        floris_wake_loss = df_floris["pow_ref"].values - df_floris["pow_test"].values

        return ((scada_wake_loss - floris_wake_loss) ** 2).sum()
