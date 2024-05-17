import pytest
from LoopStructural import LoopInterpolator, BoundingBox
from FoldOptLib.datatypes import InterpolationConstraints
from FoldOptLib.builders import Builder
import numpy


@pytest.fixture
def bbox():
    bbox = BoundingBox(origin=[0, 0, 0], maximum=[1, 1, 1])

    return bbox


@pytest.fixture
def builder():
    # Create some mock data
    value_constraints = numpy.array([[1, 2, 3, 4, 5]], dtype=float)
    tangent_constraints = numpy.array([[1, 2, 3, 0.0, 0, 0.99, 7]], dtype=float)
    normal_constraints = numpy.array([[1, 2, 3, 0, 5, 6, 7]], dtype=float)
    gradient_constraints = numpy.array([[1, 2, 3, 4, 5, 6, 7]], dtype=float)

    # Create an instance of InterpolationConstraints
    constraints = InterpolationConstraints(
        value_constraints=value_constraints,
        tangent_constraints=tangent_constraints,
        normal_constraints=normal_constraints,
        gradient_constraints=gradient_constraints,
    )

    bbox = BoundingBox(origin=[0, 0, 0], maximum=[1, 1, 1])

    builder = Builder(bbox)
    builder.set_constraints(constraints)

    return builder


def test_init(builder):
    assert isinstance(builder.interpolator, LoopInterpolator)


def test_evaluate_scalar_value(builder):
    locations = numpy.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.1]], dtype=float)
    assert isinstance(builder.evaluate_scalar_value(locations), numpy.ndarray)


def test_evaluate_gradient(builder):
    locations = numpy.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.1]], dtype=float)
    builder.evaluate_gradient(locations)
    assert isinstance(builder.evaluate_gradient(locations), numpy.ndarray)


def test_min(builder):
    assert isinstance(builder.min(), float)


def test_max(builder):
    assert isinstance(builder.max(), float)
