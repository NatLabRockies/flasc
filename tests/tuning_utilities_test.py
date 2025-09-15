import numpy as np
import pandas as pd

from flasc.utilities.tuner_utilities import replicate_nan_values


def test_replicate_nan_values():
    # Sample dataframes
    data_1 = {"A": [1, 2, np.nan, 4], "B": [5, np.nan, 7, 8], "C": [np.nan, 1, 1, 1]}
    data_2 = {"A": [10, 20, 30, 40], "B": [50, 60, 70, 80]}
    df_1 = pd.DataFrame(data_1)
    df_2 = pd.DataFrame(data_2)

    # Call the function to replicate NaN values
    result_df = replicate_nan_values(df_1, df_2)

    # Expected output
    expected_df_1 = pd.DataFrame(
        {"A": [1, 2, np.nan, 4], "B": [5, np.nan, 7, 8], "C": [np.nan, 1, 1, 1]}
    )
    expected_df_2 = pd.DataFrame({"A": [10, 20, np.nan, 40], "B": [50, np.nan, 70, 80]})

    # Check if the result matches the expected output
    assert result_df.equals(expected_df_2)
    assert df_1.equals(expected_df_1)


def test_replicate_nan_values_with_time():
    # Sample dataframes
    data_1 = {"A": [1, 2, np.nan, 4], "B": [5, np.nan, 7, 8], "C": [np.nan, 1, 1, 1]}
    data_2 = {"A": [10, 20, 30, 40], "B": [50, 60, 70, 80]}
    df_1 = pd.DataFrame(data_1)
    df_2 = pd.DataFrame(data_2)

    df_1["time"] = pd.date_range("2021-01-01", periods=4)
    df_2["time"] = df_1["time"]

    # Call the function to replicate NaN values
    result_df = replicate_nan_values(df_1, df_2)

    print(result_df)

    # Expected output
    expected_df_1 = pd.DataFrame(
        {
            "A": [1, 2, np.nan, 4],
            "B": [5, np.nan, 7, 8],
            "C": [np.nan, 1, 1, 1],
            "time": pd.date_range("2021-01-01", periods=4),
        }
    )
    expected_df_2 = pd.DataFrame(
        {
            "A": [10, 20, np.nan, 40],
            "B": [50, np.nan, 70, 80],
            "time": pd.date_range("2021-01-01", periods=4),
        }
    )

    # Check if the result matches the expected output
    assert result_df.equals(expected_df_2)
    assert df_1.equals(expected_df_1)
