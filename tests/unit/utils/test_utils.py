import numpy as np
import pandas as pd
import os
from FoldOptLib.utils.utils import (
    get_predicted_rotation_angle,
    fourier_series,
    fourier_series_x_intercepts,
    save_load_object,
    strike_dip_to_vectors,
    strike_dip_to_vector,
    rotate_vector,
    create_dict,
    create_gradient_dict,
    create_dataset,
)

# Common setup for the tests
fold_frame = np.array([1, 2])
fold_rotation = np.array([45, 90])
theta = np.array([0, 1, 1, 500])
fold_frame_coordinate = np.linspace(-10, 10, 100)


def test_get_predicted_rotation_angle():
    result = get_predicted_rotation_angle(theta, fold_frame_coordinate)
    assert isinstance(result, np.ndarray)


def test_fourier_series():
    popt = [1, 2, 3, 4]
    result = fourier_series(fold_frame, *popt)
    assert isinstance(result, (float, np.ndarray))


def test_fourier_series_x_intercepts():
    popt = [1.0, 2.0, 3.0, 4.0]
    x = np.array([1.0, 2.0, 3.0, 4.0])
    result = fourier_series_x_intercepts(x, popt)
    assert isinstance(result, np.ndarray)


def test_save_load_object():
    obj = {"key": "value"}
    file_path = "test_object.pkl"
    save_load_object(obj=obj, file_path=file_path, mode="save")
    loaded_obj = save_load_object(file_path=file_path, mode="load")
    assert obj == loaded_obj
    os.remove(file_path)


def test_strike_dip_to_vectors():
    strike = np.array([45, 90])
    dip = np.array([30, 60])
    result = strike_dip_to_vectors(strike, dip)
    assert isinstance(result, np.ndarray)


def test_strike_dip_to_vector():
    result = strike_dip_to_vector(45, 30)
    assert isinstance(result, np.ndarray)


def test_rotate_vector():
    v = np.array([1, 0])
    angle = np.pi / 4
    result = rotate_vector(v, angle)
    assert isinstance(result, np.ndarray)


def test_create_dict():
    result = create_dict(
        x=[1, 2],
        y=[1, 2],
        z=[1, 2],
        strike=[45, 90],
        dip=[30, 60],
        feature_name="test",
        coord=1,
        data_type="foliation",
    )
    assert isinstance(result, dict)


def test_create_gradient_dict():
    result = create_gradient_dict(
        x=[1, 2],
        y=[1, 2],
        z=[1, 2],
        nx=[0, 1],
        ny=[1, 0],
        nz=[0, 0],
        feature_name="test",
        coord=1,
    )
    assert isinstance(result, dict)


def test_make_dataset():
    vec = np.array([1, 0, 0])
    points = np.array([[1, 2, 3], [4, 5, 6]])
    result = create_dataset(vec, points)
    assert isinstance(result, pd.DataFrame)
