import numpy as np
import pandas as pd

from flasc.model_fitting.cost_library import (
    CostFunctionBase,
    TurbinePowerMeanAbsoluteError,
)
from flasc.model_fitting.model_fit import ModelFit
from flasc.model_fitting.opt_library import opt_sweep
from flasc.utilities.utilities_examples import load_floris_artificial


def get_simple_inputs_gch():
    # TODO: share this between multiple test files?
    # Create a simple dataframe
    df = pd.DataFrame(
        {
            "time": np.array([0, 1, 2]),
            "pow_000": np.array([1000.0, 1100.0, 1200.0]),
            "ws_000": np.array([8.0, 9.0, 10.0]),
            "wd_000": np.array([270.0, 270.0, 270.0]),
            "pow_001": np.array([950.0, 1100.0, 1150.0]),
            "ws_001": np.array([7.5, 8.5, 9.5]),
            "wd_001": np.array([270.0, 270.0, 270.0]),
        }
    )

    # Assign ws_000 to ws and wd_000 to wd using the assign function
    df = df.assign(ws=df["ws_000"], wd=df["wd_000"])

    # Load floris and set to single turbine layout
    fm, _ = load_floris_artificial(wake_model="gch")
    fm.set(layout_x=[0.0, 1000.0], layout_y=[0.0, 0.0])

    # Define cost_function as a simple function
    class CostFunctionTest(CostFunctionBase):
        def cost(self, df_floris):
            return None

    cost_function = CostFunctionTest(df)

    # Define the parameters to tune the kA parameter of GCH
    parameter_list = [("wake", "wake_velocity_parameters", "gauss", "ka")]
    parameter_name_list = ["kA"]
    parameter_range_list = [(0.1, 0.5)]
    parameter_index_list = []

    return (
        df,
        fm,
        cost_function,
        parameter_list,
        parameter_name_list,
        parameter_range_list,
        parameter_index_list,
    )

def test_opt_optuna():
    pass

def test_opt_optuna_with_wd_std():
    pass

def test_opt_sweep():
    # Get simple inputs
    (
        df,
        fm,
        _,
        parameter_list,
        parameter_name_list,
        parameter_range_list,
        parameter_index_list,
    ) = get_simple_inputs_gch()

    # Single parameter
    mf = ModelFit(
        df,
        fm,
        TurbinePowerMeanAbsoluteError(),
        parameter_list,
        parameter_name_list,
        parameter_range_list,
        parameter_index_list,
    )

    results = opt_sweep(
        mf=mf,
        n_grid=5,
    )

    sweep_best = results["optimized_parameter_values"]
    test_best = results["all_parameter_combinations"][np.argmin(results["all_costs"])]

    assert np.allclose(sweep_best, test_best)

    # Multiple parameters
    parameter_list.append(("wake", "wake_velocity_parameters", "gauss", "kb"))
    parameter_name_list.append("kB")
    parameter_range_list.append((0.001, 0.005))

    mf = ModelFit(
        df,
        fm,
        TurbinePowerMeanAbsoluteError(),
        parameter_list,
        parameter_name_list,
        parameter_range_list,
        parameter_index_list,
    )

    results = opt_sweep(
        mf=mf,
        n_grid=[5, 4],
    )

    sweep_best = results["optimized_parameter_values"]
    test_best = results["all_parameter_combinations"][np.argmin(results["all_costs"])]

    assert np.allclose(sweep_best, test_best)
