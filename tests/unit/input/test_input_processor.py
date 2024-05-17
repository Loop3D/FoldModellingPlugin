import pytest
import pandas as pd
import numpy as np
from FoldOptLib.input.input_data_processor import InputDataProcessor
from FoldOptLib.input.data_storage import InputData
from FoldOptLib.datatypes import DataType
from LoopStructural import BoundingBox


def test_normalise():
    gradient = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    expected_output = gradient / np.linalg.norm(gradient, axis=1)[:, None]
    assert np.array_equal(InputDataProcessor.normalise(gradient), expected_output)


def test_post_init_strike_dip():
    data = pd.DataFrame(
        {
            "X": [0, 1, 2],
            "Y": [0, 1, 2],
            "Z": [0, 1, 2],
            "strike": [0, 90, 180],
            "dip": [0, 45, 90],
            "feature_name": ["feature", "feature", "feature"],
        }
    )
    input_data = InputData(data, BoundingBox([0, 0, 0], [1, 1, 1]))
    processor = InputDataProcessor(data=input_data)
    processed_data = processor.__post_init__()
    assert "gx" in processed_data.columns
    assert "gy" in processed_data.columns
    assert "gz" in processed_data.columns


def test_post_init_gx_gy_gz():
    data = pd.DataFrame(
        {
            "X": [0, 1, 2],
            "Y": [0, 1, 2],
            "Z": [0, 1, 2],
            "gx": [1, 2, 3],
            "gy": [4, 5, 6],
            "gz": [7, 8, 9],
            "feature_name": ["feature", "feature", "feature"],
        }
    )
    input_data = InputData(data, BoundingBox([0, 0, 0], [1, 1, 1]))
    processor = InputDataProcessor(data=input_data)
    processed_data = processor.__post_init__()
    assert "gx" in processed_data.columns
    assert "gy" in processed_data.columns
    assert "gz" in processed_data.columns
