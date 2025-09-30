import pickle

import matplotlib.pyplot as plt
from floris import ParFlorisModel, UncertainFlorisModel
from optuna.visualization.matplotlib import (
    plot_contour,
    plot_optimization_history,
    plot_slice,
)

from flasc.model_fitting.cost_library import TurbinePowerMeanAbsoluteError
from flasc.model_fitting.model_fit import ModelFit
from flasc.model_fitting.opt_library import opt_optuna_with_wd_std

""" Use ModelFit optimization to find the optimal wake expansion value and wind direction
standard deviation that best fits the uncertain data.

Demonstrate usage with parallelization via ParFlorisModel.  Note for this small case, this 
actually does not improve performance, but for larger cases, and on clusters can be very useful.s
"""

# Since ModelFit is always parallel this is important to include
if __name__ == "__main__":
    n_trials = 50

    # Load the data from previous example
    with open("two_turbine_data.pkl", "rb") as f:
        data = pickle.load(f)

    # Unpack - using df_u (uncertain data) instead of df
    df_u = data["df_u"]
    parameter = data["parameter"]
    we_value_original = data["we_value_original"]
    we_value_set = data["we_value_set"]
    wd_std_original = data["wd_std_original"]
    wd_std_set = data["wd_std_set"]

    # Declare parallel FLORIS model
    fm = data["fm_default"]
    fm_par = ParFlorisModel(fm)

    # Repeat settings from 00_generate_data.py
    wd_std_original = 3.0  # Standard deviation of wind direction in for uncertain model (default)
    ws_resolution = 0.25
    wd_resolution = 2.0
    ufm_par = UncertainFlorisModel(
        fm_par.copy(),
        wd_std=wd_std_original,
        ws_resolution=ws_resolution,
        wd_resolution=wd_resolution,
    )

    # Now pass the above cost function to the ModelFit class
    # Note: using ufm_default (UncertainFlorisModel) instead of fm_default
    mf = ModelFit(
        df_u,
        ufm_par,
        TurbinePowerMeanAbsoluteError(),
        parameter_list=[parameter],
        parameter_name_list=["wake expansion"],
        parameter_range_list=[(0.01, 0.07)],
        parameter_index_list=[],
    )

    # Compute the baseline cost
    print("Evaluating baseline cost")
    baseline_cost = mf.evaluate_floris()

    # Optimize using opt_optuna_with_wd_std which optimizes both the wake expansion
    # parameter and the wind direction standard deviation
    opt_result = opt_optuna_with_wd_std(mf, n_trials=n_trials)

    # Print results
    print("----------------------------")
    print(f"Default parameter (we_1): {we_value_original}")
    print(f"Set (Target) parameter (we_1): {we_value_set}")
    print(f"Calibrated parameter value (we_1):  {opt_result['optimized_parameter_values'][0]:.2f}")
    print()
    print(f"Default parameter (wd_std): {wd_std_original}")
    print(f"Set (Target) parameter (wd_std): {wd_std_set}")
    print(f"Calibrated wd_std value:     {opt_result['optimized_parameter_values'][1]:.2f}")
    print("----------------------------")

    # Show an optuna progress plot
    plot_optimization_history(opt_result["optuna_study"])

    # Show a slice plot
    axarr = plot_slice(opt_result["optuna_study"])

    # we_1 tuning
    ax = axarr[0]
    ax.axvline(we_value_original, color="k", linestyle="--", label="Original value")
    ax.axvline(we_value_set, color="r", linestyle="--", label="Set value")
    ax.legend()

    # wd_std tuning
    ax = axarr[1]
    ax.axvline(wd_std_original, color="k", linestyle="--", label="Original value")
    ax.axvline(wd_std_set, color="r", linestyle="--", label="Set value")
    ax.legend()

    # Show a contour plot of wake expansion vs wd_std
    ax = plot_contour(opt_result["optuna_study"])
    ax.axvline(we_value_original, color="k", linestyle="--", label="Original value")
    ax.axvline(we_value_set, color="r", linestyle="--", label="Set value")
    ax.axhline(wd_std_original, color="k", linestyle="--")
    ax.axhline(wd_std_set, color="r", linestyle="--")
    ax.legend()

    plt.tight_layout()
    plt.show()
