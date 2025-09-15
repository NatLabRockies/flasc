# Model Fitting with ModelFit

FLASC's ModelFit capability provides a modular framework for calibrating FLORIS engineering wake models to SCADA data. This automated calibration process optimizes model parameters to minimize differences between predicted and observed turbine performance, improving the accuracy of wake modeling for wind farm analysis.

## Overview

ModelFit implements an optimization-based approach to model calibration that links modular cost functions with optimization algorithms. The system is designed to:

- **Automatically calibrate FLORIS parameters** to match SCADA measurements
- **Support multiple cost functions** for different optimization objectives  
- **Provide flexible optimization algorithms** including grid search and Bayesian optimization
- **Handle various FLORIS model types** including standard (`FlorisModel`), parallel (`ParallelFlorisModel`), and uncertain models (`UncertainFlorisModel`)
- **Enable custom cost function development** through a base class interface

The ModelFit framework replaces and deprecates older calibration methods, providing a more robust and extensible approach to model tuning.

## ModelFit Class

The `ModelFit` class serves as the central component that coordinates FLORIS simulations, cost function evaluation, and parameter optimization. It manages the interface between SCADA data, FLORIS models, and optimization routines.

### Basic Usage

```python
from flasc.model_fitting.model_fit import ModelFit
from flasc.model_fitting.cost_library import TurbinePowerMeanAbsoluteError
from flasc.model_fitting.opt_library import opt_optuna

# Define parameters to optimize
parameter_list = [("wake", "wake_velocity_parameters", "jensen", "we")]
parameter_name_list = ["wake_expansion"] # Single string name
parameter_range_list = [(0.01, 0.07)]

# Create ModelFit instance
mf = ModelFit(
    df_scada,                           # SCADA data in FLASC format
    floris_model,                       # FLORIS model instance
    TurbinePowerMeanAbsoluteError(),    # Cost function
    parameter_list=parameter_list,
    parameter_name_list=parameter_name_list,
    parameter_range_list=parameter_range_list
)

# Run optimization
result = opt_optuna(mf, n_trials=100)
```

For multi-parameter optimization, you can tune multiple elements of parameter arrays simultaneously.  In this example code
the first array entries of the `wake_expansion_rates` array in the `empirical_gauss` model are calibrated together.

```python
# Define multiple parameters to optimize (first two wake expansion rates)
parameter_list = [
    ("wake", "wake_velocity_parameters", "empirical_gauss", "wake_expansion_rates"),
    ("wake", "wake_velocity_parameters", "empirical_gauss", "wake_expansion_rates")
]
parameter_name_list = ["we_1", "we_2"]  # Names for each parameter
parameter_range_list = [(0.0, 0.05), (0.0, 0.08)]  # Ranges for each parameter
parameter_index_list = [0, 1]  # Array indices to optimize

# Create ModelFit instance for multi-parameter optimization
mf_multi = ModelFit(
    df_scada,
    floris_model,
    TurbinePowerMeanAbsoluteError(),
    parameter_list=parameter_list,
    parameter_name_list=parameter_name_list,
    parameter_range_list=parameter_range_list,
    parameter_index_list=parameter_index_list
)

# Run optimization
result = opt_optuna(mf_multi, n_trials=200)
```

## Cost Functions and Cost Library

The cost library provides a number of pre-built cost functions and base classes for developing custom cost functions.

### Recommended Cost Function

While a number of cost functions are included, we recommend using **`TurbinePowerMeanAbsoluteError`** for most applications because it avoids several critical issues found in other cost functions:

- **Avoids skewing toward outliers**: Unlike squared error metrics, mean absolute error is not dominated by extreme values
- **Prevents turbine error cancellation**: Turbine-level errors are computed individually before averaging, preventing positive and negative errors from different turbines from canceling out
- **Eliminates time cancellation**: Absolute errors at each time step are preserved, preventing temporal error cancellation effects
- **Physical interpretability**: Errors are expressed in power units (kW) that are meaningful to engineers
- **Computational efficiency**: Simple calculation that scales well with large datasets

See also [Modeling and analysis of offshore wind farm wake effects on wind turbine components and power production](https://ntnuopen.ntnu.no/ntnu-xmlui/handle/11250/3178958) by Diederik van Binsbergen for further analysis on choice of cost function.

### Available Cost Functions

The library includes turbine-level, farm-level, and wake-specific cost functions, as well as base classes for custom development. For specialized applications, custom cost functions can be created by inheriting from `CostFunctionBase`:

```python
from flasc.model_fitting.cost_library import CostFunctionBase

class CustomCostFunction(CostFunctionBase):
    """Custom cost function example."""
    
    def cost(self, df_floris):
        """Implement custom cost calculation."""
        error = self.df_scada["pow_004"] - df_floris["pow_004"]
        return error.abs().mean()
```

## Optimization Functions and Optimization Library

The optimization library provides algorithms for parameter optimization.  It currently includes a simple grid search and a Bayesian optimization (optuna).

### Grid Search Optimization

Grid search evaluates all parameter combinations across a defined grid:

```python
from flasc.model_fitting.opt_library import opt_sweep

# Simple grid search
result = opt_sweep(mf, n_grid=10)

# Different grid sizes per parameter
result = opt_sweep(mf, n_grid=[10, 15, 8])
```


### Bayesian Optimization with Optuna

Optuna provides efficient Bayesian optimization for larger parameter spaces:

```python
from flasc.model_fitting.opt_library import opt_optuna

# Basic Optuna optimization
# n_trials limits the number of trials to 100
result = opt_optuna(mf, n_trials=100)

# With timeout and n_trials limited
result = opt_optuna(mf, n_trials=200, timeout=3600)
```

### Optuna Visualization and Analysis

Optuna provides analysis tools that can be used directly with ModelFit results:

```python
from optuna.visualization.matplotlib import plot_optimization_history, plot_slice, plot_contour

# Extract study object from results
study = result["optuna_study"]

# Visualization options
plot_optimization_history(study)  # Progress over trials
plot_slice(study)                 # Parameter sensitivity
plot_contour(study)              # Parameter interactions (2D)
```


### Optimizers with wd_std

Both implementations of optimizations included have versions that include optimization of the
standard deviation of wind direction (`wd_std`) within the `UncertainFlorisModel` as an additional
component of the optimization.

```python
from flasc.model_fitting.opt_library import opt_optuna_with_wd_std

# Optimize parameters + wind direction uncertainty
result = opt_optuna_with_wd_std(uncertain_model_fit, n_trials=100)
```

## Examples

### Artificial Data Examples

The artificial data examples in `examples_artificial_data/05_model_fit/` demonstrate fundamental ModelFit usage with synthetic two-turbine datasets:

- **`00_generate_data.py`**: Creates synthetic SCADA data using FLORIS simulations with known parameter values.

- **`01a_evaluate_costs.py`**: Shows how to evaluate different cost functions across parameter ranges.

- **`01b_evaluate_costs_uncertain.py`**: Demonstrates cost function evaluation using UncertainFlorisModel, showing how wind direction uncertainty affects parameter sensitivity and cost landscapes.

- **`02a_optimize_parameter_optsweep.py`**: Demonstrates grid search optimization to find optimal wake expansion parameters, ideal for understanding parameter spaces with few dimensions.

- **`02b_optimize_parameter_optuna.py`**: Illustrates Bayesian optimization using Optuna, including visualization tools like `plot_optimization_history` and `plot_slice` for analyzing optimization progress and parameter importance.

- **`02c_optimize_parameter_optuna_wd_std.py`**: Demonstrates optimization including wind direction standard deviation (`wd_std`) as an additional parameter using `opt_optuna_with_wd_std`, showing how to tune both model parameters and uncertainty parameters simultaneously.

### Real-World Application Example

The SMARTEOLE example in `examples_smarteole/11_model_tuning_with_model_fit.ipynb` demonstrates using ModelFit with real world data.

- **Sequential parameter tuning**: First calibrates wake expansion parameters using baseline data, then optimizes deflection gain using wake steering data.

- **Atmospheric condition analysis**: Demonstrates separate tuning for day and night conditions, revealing how optimal parameters vary with atmospheric stability.

- **Multi-parameter optimization**: Shows simultaneous optimization of multiple parameters and compares results with sequential tuning approaches.



## Deprecated Code

ModelFit replaces several older calibration methods that are now deprecated:

### Deprecated Modules
- **`flasc.model_fitting.floris_tuning`**: Original tuning implementation (deprecated v2.4)
- **Examples 07 and 08**: SMARTEOLE tuning examples using old methods

