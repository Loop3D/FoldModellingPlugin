from FoldOptLib.fold_modelling.engine import FoldModel
import pytest
import numpy as np
import pandas as pd
from FoldOptLib.datatypes import DataType, InputGeologicalKnowledge, CoordinateType
from FoldOptLib.input import InputData, OptData
from FoldOptLib.builders import FoldFrameBuilder, Builder
from LoopStructural import BoundingBox


# Sample data for tests
@pytest.fixture
def sample_data():
    data = {
        "gx": [0.0, 0.0, 0.0],
        "gy": [0.0, 0.5, 0.1],
        "gz": [0.5, 0.4, 0.5],
        "X": [10, 11, 12],
        "Y": [13, 14, 15],
        "Z": [16, 17, 18],
        "feature_name": ["feature", "feature", "feature"],
        "weight": [1, 1, 1],
    }
    dataset = pd.DataFrame(data)
    bounding_box = BoundingBox(np.array([0, 0, 0]), np.array([20, 20, 20]))
    input_data = InputData(dataset, bounding_box)

    return input_data


@pytest.fixture
def fold_model(sample_data):
    return FoldModel(sample_data)


def test_initialization(fold_model, sample_data):
    assert fold_model.raw_data is sample_data
    assert fold_model.bounding_box is sample_data[DataType.BOUNDING_BOX]
    assert fold_model.model is None
    # Ensure the columns are present
    assert (
        "weight" in sample_data[DataType.DATA].columns
    ), "'weight' column is missing from sample_data"
    np.testing.assert_array_equal(
        fold_model.gradient_data,
        sample_data[DataType.DATA][["gx", "gy", "gz"]].to_numpy(),
    )
    np.testing.assert_array_equal(
        fold_model.points, sample_data[DataType.DATA][["X", "Y", "Z"]].to_numpy()
    )
    assert fold_model.axial_surface is None
    assert fold_model.scaled_points is None


def test_set_data(fold_model, sample_data):

    fold_model.set_data(sample_data)
    np.testing.assert_array_equal(
        fold_model.data[["gx"]], sample_data[DataType.DATA][["gx"]]
    )


def test_initialise_model(fold_model):
    fold_model.initialise_model()
    assert fold_model.data is not None
    assert fold_model.scaled_points is not None
    assert fold_model.geological_knowledge is None


def test_process_axial_surface_proposition(fold_model, sample_data):
    fold_model.initialise_model()
    axial_normal = np.array([1.0, 0.0, 0.0], dtype=float)
    np.testing.assert_array_equal(
        fold_model.points, sample_data[DataType.DATA][["X", "Y", "Z"]].to_numpy()
    )
    result = fold_model.process_axial_surface_proposition(axial_normal)
    assert isinstance(result, OptData)


def test_build_fold_frame(fold_model, sample_data):
    fold_model.initialise_model()
    # Ensure the columns are present
    assert (
        "weight" in fold_model.raw_data[DataType.DATA].columns
    ), "'weight' column is missing from sample_data"
    axial_normal = np.array([1.0, 0.0, 0.0], dtype=float)
    np.testing.assert_array_equal(
        fold_model.points, sample_data[DataType.DATA][["X", "Y", "Z"]].to_numpy()
    )
    result = fold_model.process_axial_surface_proposition(axial_normal)
    assert isinstance(result, OptData)
    fold_model.build_fold_frame(axial_normal)
    assert isinstance(fold_model.axial_surface, FoldFrameBuilder)
    assert isinstance(
        fold_model.axial_surface[CoordinateType.AXIAL_FOLIATION_FIELD], Builder
    )
    assert isinstance(fold_model.axial_surface[CoordinateType.FOLD_AXIS_FIELD], Builder)


# def test_create_and_build_fold_event(fold_model):
#     fold_model.initialise_model()
#     axial_normal = np.array([1., 0., 0.], dtype=float)
#     fold_model.build_fold_frame(axial_normal)
#     fold_event = fold_model.create_and_build_fold_event()
#     assert fold_event is not None

# def test_calculate_svariogram(fold_model):
#     fold_frame = np.array([[1, 2, 3], [4, 5, 6]])
#     rotation_angles = np.array([0.1, 0.2])
#     result = fold_model.calculate_svariogram(fold_frame, rotation_angles)
#     assert isinstance(result, np.ndarray)

# def test_fit_fourier_series(fold_model):
#     fold_model.initialise_model()
#     fold_model.build_fold_frame(np.array([1., 0., 0.], dtype=float))
#     fold_frame_coordinate = np.array([1, 2, 3])
#     rotation_angle = np.array([0.1, 0.2, 0.3])
#     result = fold_model.fit_fourier_series(fold_frame_coordinate, rotation_angle)
#     assert isinstance(result, list)

# def test_calculate_folded_foliation_vectors(fold_model):
#     fold_model.initialise_model()
#     axial_normal = np.array([1, 0, 0])
#     fold_model.build_fold_frame(axial_normal)
#     fold_event = fold_model.create_and_build_fold_event()
#     result = fold_model.calculate_folded_foliation_vectors()
#     assert isinstance(result, np.ndarray)

# def test_get_predicted_foliation(fold_model):
#     axial_normal = np.array([1, 0, 0])
#     result = fold_model.get_predicted_foliation(axial_normal)
#     assert isinstance(result, np.ndarray)
