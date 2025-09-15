import pickle

import matplotlib.pyplot as plt

from flasc.model_fitting.cost_library import TurbinePowerMeanAbsoluteError
from flasc.model_fitting.model_fit import ModelFit
from flasc.model_fitting.opt_library import opt_sweep

""" Use ModelFit optimization to find the optimal wake expansion value that best fits the data.

In this example using the opt_sweep optimization routine from the opt_library.
"""

# Since ModelFit is always parallel this is important to include
if __name__ == "__main__":
    n_grid = 10

    # Load the data from previous example
    with open("two_turbine_data.pkl", "rb") as f:
        data = pickle.load(f)

    # Unpack
    df = data["df"]
    fm_default = data["fm_default"]
    parameter = data["parameter"]
    we_value_original = data["we_value_original"]
    we_value_set = data["we_value_set"]

    # Now pass the above cost function to the ModelFit class
    mf = ModelFit(
        df,
        fm_default,
        TurbinePowerMeanAbsoluteError(),
        parameter_list=[parameter],
        parameter_name_list=["wake expansion"],
        parameter_range_list=[(0.01, 0.07)],
        parameter_index_list=[],
    )

    # Compute the baseline cost
    print("Evaluating baseline cost")
    baseline_cost = mf.evaluate_floris()

    # Optimize
    opt_result = opt_sweep(mf, n_grid=n_grid)

    print(opt_result)

    # Print results
    print("----------------------------")
    print(f"Default parameter: {we_value_original}")
    print(f"Set parameter: {we_value_set}")
    print()
    print(f"Calibrated parameter value:  {opt_result['optimized_parameter_values'][0]:.2f}")
    print("----------------------------")

    # Plot the results
    plt.plot(opt_result["all_parameter_combinations"], opt_result["all_costs"])
    plt.axvline(we_value_original, color="k", linestyle="--", label="Original value")
    plt.axvline(we_value_set, color="r", linestyle="--", label="Set value")
    plt.xlabel("Wake expansion value")
    plt.ylabel("Cost value")
    plt.legend()
    plt.show()
