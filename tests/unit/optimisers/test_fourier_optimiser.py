import pytest
import numpy as np
from FoldOptLib.optimisers.fourier_optimiser import FourierSeriesOptimiser
from FoldOptLib.utils.utils import *


def test_fourier_series_optimiser_init():
    optimiser = FourierSeriesOptimiser(
        fold_frame_coordinate=np.empty(10),
        rotation_angle=np.linspace(-1, 1, 10),
        x=np.linspace(-1, 1, 100),
    )
    assert isinstance(optimiser, FourierSeriesOptimiser)


def test_generate_bounds_and_initial_guess():
    optimiser = FourierSeriesOptimiser(
        fold_frame_coordinate=np.empty(10),
        rotation_angle=np.linspace(-1, 1, 10),
        x=np.linspace(-1, 1, 100),
    )
    assert isinstance(optimiser, FourierSeriesOptimiser)
    optimiser.generate_bounds_and_initial_guess()
    assert isinstance(optimiser.bounds, np.ndarray) or optimiser.bounds is None
    assert isinstance(optimiser.guess, np.ndarray) or optimiser.guess is None


def test_setup_geological_knowledge():
    optimiser = FourierSeriesOptimiser(
        fold_frame_coordinate=np.array([1, 2, 3]),
        rotation_angle=np.array([1, 2, 3]),
        x=np.array([1, 2, 3]),
    )
    result = optimiser.setup_geological_knowledge(None)
    assert result is None


def test_build_optimisation_function():
    optimiser = FourierSeriesOptimiser(
        fold_frame_coordinate=np.array([1, 2, 3]),
        rotation_angle=np.array([1, 2, 3]),
        x=np.array([1, 2, 3]),
    )
    result = optimiser.build_optimisation_function(lambda x: x, lambda x: x)
    assert callable(result)


def test_setup_optimisation_method():
    optimiser = FourierSeriesOptimiser(
        fold_frame_coordinate=np.array([1, 2, 3]),
        rotation_angle=np.array([1, 2, 3]),
        x=np.array([1, 2, 3]),
    )
    optimiser.setup_optimisation_method()
    assert callable(
        optimiser.objective_function
    )  # or optimiser.objective_function is None


def test_setup_optimisation():
    optimiser = FourierSeriesOptimiser(
        fold_frame_coordinate=np.empty(10),
        rotation_angle=np.linspace(-1, 1, 10),
        x=np.linspace(-1, 1, 100),
    )
    assert isinstance(optimiser, FourierSeriesOptimiser)
    optimiser.setup_optimisation()
    assert callable(optimiser.objective_function)
    assert isinstance(optimiser.bounds, np.ndarray)
    assert isinstance(optimiser.guess, np.ndarray)
