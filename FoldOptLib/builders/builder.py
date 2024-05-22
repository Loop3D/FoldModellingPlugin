from .base_builder import BaseBuilder
from ..datatypes import InterpolationConstraints, ConstraintType
from LoopStructural import LoopInterpolator, BoundingBox
import numpy


class Builder(BaseBuilder):
    def __init__(self, boundingbox: BoundingBox):
        self.boundingbox = boundingbox

        self.interpolator = LoopInterpolator(
            self.boundingbox, 
            dimensions=self.boundingbox.dimensions, 
            nelements=1000
        )

    def set_constraints(self, constraints: InterpolationConstraints):
        self.interpolator.fit(
            values=constraints[ConstraintType.VALUE],
            tangent_vectors=constraints[ConstraintType.TANGENT],
            normal_vectors=constraints[ConstraintType.NORMAL],
        )

    def evaluate_scalar_value(self, locations: numpy.ndarray) -> numpy.ndarray:
        return self.interpolator.evaluate_scalar_value(locations)

    def evaluate_gradient(self, locations: numpy.ndarray) -> numpy.ndarray:
        gradient = self.interpolator.evaluate_gradient(locations)
        gradient /= numpy.linalg.norm(gradient, axis=1)[:, None]
        return gradient

    def min(self):
        """Calculate the min value of the fold frame
        in the model

        Returns
        -------
        minimum, float
            min value of the feature evaluated on a regular grid in the model domain
        """

        return numpy.nanmin(
            self.evaluate_scalar_value(self.boundingbox.regular_grid((10, 10, 10)))
        )

    def max(self):
        """Calculate the maximum value of the geological feature
        in the model

        Returns
        -------
        maximum, float
            max value of the feature evaluated on a regular grid in the model domain
        """

        return numpy.nanmax(
            self.evaluate_scalar_value(self.boundingbox.regular_grid((10, 10, 10)))
        )
