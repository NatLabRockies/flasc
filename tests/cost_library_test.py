import numpy as np
import pandas as pd

from flasc import FlascDataFrame
from flasc.model_fitting.cost_library import (
    TurbinePowerMeanAbsoluteError,
    TurbinePowerRootMeanSquaredError,
)


def setup_data():
    # Create a simple dataframe for SCADA data
    df_scada = FlascDataFrame(
        pd.DataFrame(
            {
                "time": np.array([0, 1, 2]),
                "pow_000": np.array([1000.0, 1100.0, 1200.0]),
                "pow_001": np.array([900.0, 950.0, 1000.0]),
            }
        )
    )

    # Create a simple dataframe for FLORIS data
    df_floris = FlascDataFrame(
        pd.DataFrame(
            {
                "time": np.array([0, 1, 2]),
                "pow_000": np.array([1050.0, 1150.0, 1250.0]),
                "pow_001": np.array([950.0, 1000.0, 1050.0]),
            }
        )
    )

    return df_scada, df_floris


def test_turbine_power_error():
    df_scada, df_floris = setup_data()
    cf = TurbinePowerRootMeanSquaredError(df_scada)

    error = cf(df_floris)
    expected_error = np.sqrt((
        ((df_scada["pow_000"] - df_floris["pow_000"]) ** 2).mean()
        + ((df_scada["pow_001"] - df_floris["pow_001"]) ** 2).mean()
    ) / 2)

    assert error == expected_error


def test_turbine_power_error_abs():
    df_scada, df_floris = setup_data()
    cf = TurbinePowerMeanAbsoluteError(df_scada)

    error = cf(df_floris)
    expected_error = (
        (df_scada["pow_000"] - df_floris["pow_000"]).abs().mean()
        + (df_scada["pow_001"] - df_floris["pow_001"]).abs().mean()
    ) / 2

    assert error == expected_error
