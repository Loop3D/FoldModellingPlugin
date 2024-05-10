from .structural_frame_builder import Builder
from ..input import OptData
from ..datatypes import CoordinateType, InterpolationConstraints
from LoopStructural import LoopInterpolator, BoundingBox
import numpy


class FoldFrameBuilder(Builder):

    def __init__(self, constraints: OptData, bounding_box: BoundingBox):
        super().__init__(constraints, bounding_box)

    def build_axial_surface_field(self):
        
        self.set_constraints()

    def build_fold_axis_field(self):
        pass

    def build_x_axis_field(self):
        pass

    def build(self):
        pass
