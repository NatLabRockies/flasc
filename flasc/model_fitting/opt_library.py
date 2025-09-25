"""This module contains the optimization algorithms for the model fitting."""

from __future__ import annotations

import itertools
from typing import Dict, Tuple

import numpy as np
import optuna

from flasc.model_fitting.model_fit import ModelFit


def opt_optuna(
    mf: ModelFit,
    n_trials: int = 100,
    timeout: float | None = None,
    seed: int | None = None,
    verbose: bool = True,
) -> Dict:
    """Optimize the model parameters using Optuna.

    Args:
        mf (ModelFit): ModelFit object containing the model and parameters to optimize.
        n_trials (int): Number of trials to run. Defaults to 100.
        timeout (float | None): Timeout for the optimization in seconds.
            Defaults to None.
        seed (int | None): Seed for the random number generator. Defaults to None,
            in which case a random seed will be used.
        verbose (bool): Whether to print out the optimization process. Defaults to True which
            gives optuna INFO logging.

    Returns:
        Dict: Dictionary containing the optimal parameter values and
            the Optuna study object. All optimizers must contain keys "optimized_parameter_values"
            and "optimized_cost", and may optionally contain other optimizers-specific key-value
            pairs.
    """

    # Set up the objective function for optuna
    def objective(trial):
        parameter_values = []
        for p_idx in range(mf.n_parameters):
            parameter_name = mf.parameter_name_list[p_idx]
            parameter_range = mf.parameter_range_list[p_idx]
            parameter_values.append(
                trial.suggest_float(parameter_name, parameter_range[0], parameter_range[1])
            )

        return mf.set_parameter_and_evaluate(parameter_values)

    # Run the optimization
    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(seed=seed), study_name="ModelFit"
    )

    # If not verbose
    if not verbose:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Seed the initial value
    init_dict = {}
    for pname, pval in zip(mf.parameter_name_list, mf.get_parameter_values()):
        init_dict[pname] = pval
    study.enqueue_trial(init_dict)
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    # Make a list of the best parameter values
    best_params = []
    for parameter_name in mf.parameter_name_list:
        best_params.append(study.best_params[parameter_name])

    # Return results as dictionary
    result_dict = {
        "optimized_parameter_values": best_params,
        "optimized_cost": study.best_value,
        "optuna_study": study,
    }

    # Returns results
    return result_dict


def opt_optuna_with_wd_std(
    mf: ModelFit,
    n_trials: int = 100,
    timeout: float | None = None,
    verbose: bool = True,
) -> Tuple[Dict, optuna.Study]:
    """Optimize the model parameters using Optuna including wd_std.

    This version includes the wind direction standard deviation of the UncertainFlorisModel
    as a parameter to optimize.

    Args:
        mf (ModelFit): ModelFit object containing the model and parameters to optimize.
        n_trials (int): Number of trials to run. Defaults to 100.
        timeout (float | None): Timeout for the optimization in seconds.
            Defaults to None.
        verbose (bool): Whether to print out the optimization process. Defaults to True which
            gives optuna INFO logging.

    Returns:
        Dict: Dictionary containing the optimal parameter values and
            the Optuna study object. All optimizers must contain keys "optimized_parameter_values"
            and "optimized_cost", and may optionally contain other optimizers-specific key-value
            pairs.
    """

    # Set up the objective function for optuna
    def objective(trial):
        # Set wd_std
        mf.set_wd_std(wd_std=trial.suggest_float("wd_std", 0.1, 6.0))

        parameter_values = []
        for p_idx in range(mf.n_parameters):
            parameter_name = mf.parameter_name_list[p_idx]
            parameter_range = mf.parameter_range_list[p_idx]
            parameter_values.append(
                trial.suggest_float(parameter_name, parameter_range[0], parameter_range[1])
            )

        return mf.set_parameter_and_evaluate(parameter_values)

    # Run the optimization
    study = optuna.create_study()

    # If not verbose
    if not verbose:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Seed the initial value
    init_dict = {"wd_std": 3.0}
    for pname, pval in zip(mf.parameter_name_list, mf.get_parameter_values()):
        init_dict[pname] = pval
    study.enqueue_trial(init_dict)
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    # Make a list of the best parameter values
    best_params = []
    for parameter_name in mf.parameter_name_list + ["wd_std"]:
        best_params.append(study.best_params[parameter_name])

    # Return results as dictionary
    result_dict = {
        "optimized_parameter_values": best_params,
        "optimized_cost": study.best_value,
        "optuna_study": study,
    }

    # Returns results
    return result_dict


def extract_optuna_trial_data(study_obj, param_name):
    """Extract parameter values and costs from study trials."""
    param_values = [trial.params[param_name] for trial in study_obj.trials]
    cost_values = [trial.value for trial in study_obj.trials]

    # Sort both by parameter values
    param_values, cost_values = zip(*sorted(zip(param_values, cost_values)))

    # Get the best parameter value and cost
    best_param_value = study_obj.best_trial.params[param_name]
    best_cost = study_obj.best_trial.value

    # Normalize the cost values
    cost_values = np.array(cost_values) / np.min(cost_values)

    return param_values, cost_values, best_param_value, best_cost


def opt_sweep(
    mf: ModelFit,
    n_grid: int | list[int] = 10,
    verbose: bool = False,
) -> Tuple[Dict, optuna.Study]:
    """Optimize the model parameters using a grid sweep.

    Args:
        mf (ModelFit): ModelFit object containing the model and parameters to optimize.
        n_grid (int | list[int] | None): Number of grid points to use for each parameter.
            If an integer is provided, the same number of grid points will be used for
            each parameter. If a list is provided, it must have the same length as the
            number of parameters. Defaults to None, in which case 10 grid points will
            be used for each parameter.
        verbose (bool): Whether to print out the optimization process. Defaults to False.

    Returns:
        Dict: Dictionary containing the optimal parameter values. All optimizers must contain keys
            "optimized_parameter_values" and "optimized_cost", and may optionally contain other
            optimizers-specific key-value pairs.
    """
    # Handle n_grid parameter
    if isinstance(n_grid, int):
        n_grid = [n_grid] * mf.n_parameters
    elif len(n_grid) != mf.n_parameters:
        raise ValueError(
            f"Length of n_grid ({len(n_grid)}) must match number of parameters ({mf.n_parameters})"
        )

    # Create parameter arrays for each parameter
    parameter_arrays = []
    for p_idx in range(mf.n_parameters):
        param_array = np.linspace(
            mf.parameter_range_list[p_idx][0], mf.parameter_range_list[p_idx][1], n_grid[p_idx]
        )
        parameter_arrays.append(param_array)

    # Generate all combinations using itertools.product
    all_combinations = np.array(list(itertools.product(*parameter_arrays)))
    all_costs = np.zeros(all_combinations.shape[0])

    # Initialize tracking variables
    best_cost = float("inf")
    best_params = None

    # Evaluate each combination
    for i, param_combination in enumerate(all_combinations):
        if verbose:
            print(f"Evaluating combination {i + 1}/{len(all_combinations)}: {param_combination}")

        cost = mf.set_parameter_and_evaluate(param_combination)
        all_costs[i] = cost

        if cost < best_cost:
            best_cost = cost
            best_params = list(param_combination)

    results_dict = {
        "optimized_parameter_values": best_params,
        "optimized_cost": best_cost,
        "all_parameter_combinations": all_combinations,
        "all_costs": all_costs,
    }

    return results_dict


def opt_sweep_with_wd_std(
    mf: ModelFit,
    n_grid: int | list[int] = 10,
    wd_std_range: Tuple[float, float] = (0.1, 6.0),
    verbose: bool = False,
) -> Tuple[Dict, optuna.Study]:
    """Optimize the model parameters using a grid sweep including wd_std.

    This version includes the wind direction standard deviation of the UncertainFlorisModel
    as a parameter to optimize.

    Args:
        mf (ModelFit): ModelFit object containing the model and parameters to optimize.
        n_grid (int | list[int] | None): Number of grid points to use for each parameter.
            If an integer is provided, the same number of grid points will be used for
            each parameter. If a list is provided, it must have the same length as the
            number of parameters. Defaults to 10 (used for each parameter)
        wd_std_range (Tuple[float, float]): Range of wind direction standard deviation to sweep.
            Defaults to (0.1, 6.0).
        verbose (bool): Whether to print out the optimization process. Defaults to False.

    Returns:
        Dict: Dictionary containing the optimal parameter values. All optimizers must contain keys
            "optimized_parameter_values" and "optimized_cost", and may optionally contain other
            optimizers-specific key-value pairs.
    """
    # Handle n_grid parameter
    if isinstance(n_grid, int):
        n_grid = [n_grid] * (mf.n_parameters + 1)
    elif len(n_grid) != mf.n_parameters + 1:
        raise ValueError(
            f"Length of n_grid ({len(n_grid)}) must match number of parameters "
            f"({mf.n_parameters + 1})"
        )

    # Create parameter arrays for each parameter
    parameter_arrays = []
    for p_idx in range(mf.n_parameters):
        param_array = np.linspace(
            mf.parameter_range_list[p_idx][0], mf.parameter_range_list[p_idx][1], n_grid[p_idx]
        )
        parameter_arrays.append(param_array)

    # Add wd_std parameter array
    wd_std_array = np.linspace(wd_std_range[0], wd_std_range[1], n_grid[-1])
    parameter_arrays.append(wd_std_array)

    # Generate all combinations using itertools.product
    all_combinations = np.array(list(itertools.product(*parameter_arrays)))
    all_costs = np.zeros(all_combinations.shape[0])

    # Initialize tracking variables
    best_cost = float("inf")
    best_params = None

    # Evaluate each combination
    for i, param_combination in enumerate(all_combinations):
        if verbose:
            print(f"Evaluating combination {i + 1}/{len(all_combinations)}: {param_combination}")

        # Set wd_std
        mf.set_wd_std(wd_std=param_combination[-1])

        cost = mf.set_parameter_and_evaluate(param_combination[:-1])
        all_costs[i] = cost

        if cost < best_cost:
            best_cost = cost
            best_params = list(param_combination)

    results_dict = {
        "optimized_parameter_values": best_params,
        "optimized_cost": best_cost,
        "all_parameter_combinations": all_combinations,
        "all_costs": all_costs,
    }

    return results_dict
