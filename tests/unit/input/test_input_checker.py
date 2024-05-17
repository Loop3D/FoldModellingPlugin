import pandas as pd
import pytest
from FoldOptLib.input.input_data_checker import CheckInputData


def test_check_input_data():
    check_input_data = CheckInputData()

    # Test if the function raises a TypeError when the input is not a pandas DataFrame.
    with pytest.raises(TypeError):
        check_input_data("not a dataframe")

    # Test if the function raises a ValueError when the required columns are not present in the DataFrame.
    df_missing_columns = pd.DataFrame(
        {
            "X": [1, 2, 3],
            "Y": [4, 5, 6],
            "Z": [7, 8, 9],
            "feature_name": ["s0", "s0", "s0"],
        }
    )
    with pytest.raises(ValueError):
        check_input_data(df_missing_columns)

    # Test if the function raises a ValueError when neither the strike and dip columns nor the gx, gy, and gz columns are present in the DataFrame.
    df_missing_strike_dip_gx_gy_gz = pd.DataFrame(
        {
            "X": [1, 2, 3],
            "Y": [4, 5, 6],
            "Z": [7, 8, 9],
            "feature_name": ["s0", "s0", "s0"],
            "strike": [0.0, 0.0, 0.0],
        }
    )
    with pytest.raises(ValueError):
        check_input_data(df_missing_strike_dip_gx_gy_gz)

    # Test if the function does not raise any error when the DataFrame is correctly formatted.
    df_correct = pd.DataFrame(
        {
            "X": [1, 2, 3],
            "Y": [4, 5, 6],
            "Z": [7, 8, 9],
            "feature_name": ["s0", "s0", "s0"],
            "strike": [0.0, 0.0, 0.0],
            "dip": [1.0, 1.0, 1.0],
        }
    )
    assert check_input_data(df_correct) is None
