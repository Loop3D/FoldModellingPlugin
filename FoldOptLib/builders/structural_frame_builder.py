from base_builder import BaseBuilder
from ..input import OptData
from ..datatypes import CoordinateType, InterpolationConstraints
from LoopStructural import LoopInterpolator, BoundingBox
import numpy
from typing import Union, Any


class Builder(BaseBuilder):

    def __init__(self, constraints: Union[InterpolationConstraints, OptData], bounding_box: BoundingBox):
        self.constraints = constraints
        self.bounding_box = bounding_box
        self.interpolator = LoopInterpolator(
            self.bounding_box,
            dimensions=3,
            nelements=1000
        )
    #TODO:Restart from here
    
    # def set_constraints(self, type: Union[CoordinateType, Any]):
    #     self.interpolator.fit(
    #         values=self.constraints[type.VALUE],
    #         tangent_vectors=self.constraints[type.TANGENT],
    #         normal_vectors=self.constraints[type.NORMAL],
    #     )

    def evaluate_scalar_value(self, locations: numpy.ndarray) -> numpy.ndarray:
        return self.interpolator.evaluate_scalar_value(locations)

    def evaluate_gradient(self, locations: numpy.ndarray) -> numpy.ndarray:
        return self.interpolator.evaluate_gradient(locations)

