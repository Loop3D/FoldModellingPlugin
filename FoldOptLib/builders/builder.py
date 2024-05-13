from base_builder import BaseBuilder

from ..datatypes import InterpolationConstraints, ConstraintType 
from LoopStructural import LoopInterpolator, BoundingBox
import numpy
from typing import Union, Any


class Builder(BaseBuilder):

    def __init__(self, bounding_box: BoundingBox):
        self.bounding_box = bounding_box

        self.interpolator = LoopInterpolator(
            self.bounding_box,
            dimensions=3,
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

        return self.interpolator.evaluate_gradient(locations)

