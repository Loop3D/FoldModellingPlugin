from base_builder import BaseBuilder
from ..datatypes import ConstraintType, InterpolationConstraints
from LoopStructural import LoopInterpolator, BoundingBox
import numpy


class StructuralFrameBuilder(BaseBuilder):

    def __init__(self, constraints: InterpolationConstraints, bounding_box: BoundingBox):
        self.constraints = constraints
        self.bounding_box = bounding_box
        self.interpolator = LoopInterpolator(
            self.bounding_box,
            dimensions=3,
            nelements=1000
        )

    def set_constraints(self):
        self.interpolator.fit(
            values=self.constraints[ConstraintType.VALUE],
            tangent_vectors=self.constraints[ConstraintType.TANGENT],
            normal_vectors=self.constraints[ConstraintType.NORMAL],
        )

    def evaluate_scalar_value(self, locations: numpy.ndarray) -> numpy.ndarray:
        return self.interpolator.evaluate_scalar_value(locations)

    def evaluate_gradient(self, locations: numpy.ndarray) -> numpy.ndarray:
        return self.interpolator.evaluate_gradient(locations)

