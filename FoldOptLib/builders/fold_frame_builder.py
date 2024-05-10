from .structural_frame_builder import StructuralFrameBuilder
from ..datatypes import ConstraintType, InterpolationConstraints
from LoopStructural import LoopInterpolator, BoundingBox
import numpy


class FoldFrameBuilder(StructuralFrameBuilder):

    def __init__(self, constraints: InterpolationConstraints, bounding_box: BoundingBox):
        super().__init__(constraints, bounding_box)

    def build_axial_surface_field(self):
        pass

    def build_fold_axis_field(self):
        pass

    def build_x_axis_field(self):
        pass
