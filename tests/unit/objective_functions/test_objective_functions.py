import numpy as np
import pytest
from scipy.stats import vonmises
from FoldOptLib.objective_functions.objective_functions import ObjectiveFunction, ObjectiveType

def test_log_normal():
    # Test with valid inputs
    assert np.isclose(ObjectiveFunction.log_normal(1, 1, 1), 0.9189385332046727)
    # Test with sigma less than or equal to 0
    with pytest.raises(ValueError):
        ObjectiveFunction.log_normal(1, 1, 0)

def test_vonmises():
    # Test with valid inputs
    assert np.isclose(ObjectiveFunction.vonmises([0]), -vonmises(1e-10, 100).logpdf([0]))

def test_fourier_series():
    # This function returns another function, so we need to test the returned function
    # Here we assume that the function `get_predicted_rotation_angle` exists and works correctly
    rotation_angle = np.array([1, 2, 3])
    fold_frame_coordinate = np.array([4, 5, 6])
    objective_fourier_series = ObjectiveFunction.fourier_series(rotation_angle, fold_frame_coordinate)
    theta = np.array([0, 7, 8, 9])
    assert isinstance(objective_fourier_series(theta), float)

def test_angle_function():
    # Test with valid inputs
    v1 = np.array([[1, 0, 0], [0, 1, 0]])
    v2 = np.array([[1, 0, 0], [0, 1, 0]])
    assert np.isclose(ObjectiveFunction.angle_function(v1, v2), 0)
    # Test with v1 and v2 not being numpy arrays
    with pytest.raises(ValueError):
        ObjectiveFunction.angle_function([1, 0, 0], [1, 0, 0])
    # Test with v1 and v2 not having the same shape
    with pytest.raises(ValueError):
        ObjectiveFunction.angle_function(np.array([1, 0, 0]), np.array([[1, 0, 0], [0, 1, 0]]))

def test_getitem():
    # Test with valid inputs
    assert ObjectiveFunction[ObjectiveType.LOG_NORMAL] == ObjectiveFunction.log_normal
    assert ObjectiveFunction[ObjectiveType.VON_MISES] == ObjectiveFunction.vonmises
    assert ObjectiveFunction[ObjectiveType.FOURIER] == ObjectiveFunction.fourier_series
    assert ObjectiveFunction[ObjectiveType.ANGLE] == ObjectiveFunction.angle_function