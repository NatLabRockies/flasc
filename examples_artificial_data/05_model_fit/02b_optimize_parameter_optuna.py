import pickle

import matplotlib.pyplot as plt
from optuna.visualization.matplotlib import (
    plot_optimization_history,
    plot_slice,
)

from flasc.model_fitting.cost_library import TurbinePowerMeanAbsoluteError
from flasc.model_fitting.model_fit import ModelFit
from flasc.model_fitting.opt_library import opt_optuna

""" Use ModelFit optimization to find the optimal wake expansion value that best fits the data.

In this example using the opt_optuna optimization routine from the opt_library.  Additionally, two
of optuna's provided visualization functions are used to assess the study: plot_optimization_history
and plot_slice.
"""

# Parameters
time_out = 5

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
opt_result = opt_optuna(mf, timeout=time_out, n_trials=None)

# Print results
print("----------------------------")
print(f"Default parameter: {we_value_original}")
print(f"Set parameter: {we_value_set}")
print()
print(f"Calibrated parameter value:  {opt_result['optimized_parameter_values'][0]:.2f}")
print("----------------------------")

# Show an optuna progress plot
plot_optimization_history(opt_result["optuna_study"])

# Show a slice plot
ax = plot_slice(opt_result["optuna_study"])
ax.axvline(we_value_original, color="k", linestyle="--", label="Original value")
ax.axvline(we_value_set, color="r", linestyle="--", label="Set value")
ax.legend()

plt.show()
