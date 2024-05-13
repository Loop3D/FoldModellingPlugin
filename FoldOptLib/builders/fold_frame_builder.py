from .builder import Builder
from ..input import OptData
from ..datatypes import CoordinateType, InterpolationConstraints
from LoopStructural import LoopInterpolator, BoundingBox
import numpy


class FoldFrameBuilder(Builder):
    
    """
    Class for building a fold frame.

    Attributes
    ----------
    constraints : OptData
        The optimization data.
    xyz : numpy.ndarray
        The coordinates used to evaluate to scalar and gradient fields.
    scalar_field : numpy.ndarray
        The scalar field.
    gradient : numpy.ndarray
        The gradient field.

    Methods
    -------
    build_axial_surface_field():
        Build the axial surface field.
    build_fold_axis_field():
        Build the fold axis field.
    build_x_axis_field():
        Build the x-axis field.
    build():
        Build the fold frame.
    """

    def __init__(self, constraints: OptData, bounding_box: BoundingBox):

        """
        Initialize the FoldFrameBuilder.

        Parameters
        ----------
        constraints : OptData
            The optimization data.
        bounding_box : BoundingBox
            The bounding box.
        """

        super().__init__(bounding_box)

        self.constraints = constraints
        self.xyz = self.constraints.data[['X', 'Y', 'Z']].to_numpy() 
        self.scalar_field = None
        self.gradient = None
        self.fold_frame = [None] * len(CoordinateType)

    def build_axial_surface_field(self):

        """
        Build the axial surface field.
        """        
        self.set_constraints(self.constraints[CoordinateType.AXIAL_FOLIATION_FIELD])
        self.scalar_field = self.evaluate_scalar_value(self.xyz)
        self.gradient = self.evaluate_gradient(self.xyz)

    def build_fold_axis_field(self):

        """
        Build the fold axis field.
        """
        
        self.set_constraints(self.constraints[CoordinateType.FOLD_AXIS_FIELD])
        self.scalar_field = self.evaluate_scalar_value(self.xyz)
        self.gradient = self.evaluate_gradient(self.xyz)

    def build_x_axis_field(self):

        """
         Build the x-axis field.
        """
            
        self.fold_frame[CoordinateType.X_AXIS] = self.set_constraints(
            self.constraints[CoordinateType.X_AXIS_FIELD]
            )
        self.scalar_field = self.evaluate_scalar_value(self.xyz)
        self.gradient = self.evaluate_gradient(self.xyz)

    def build(self):
            
        """ 
        Build the fold frame.
        """

        if self.constraints[CoordinateType.AXIAL_FOLIATION_FIELD] is None:
            
            raise ValueError('Axial surface field constraints not set.')

        elif self.constraints[CoordinateType.AXIAL_FOLIATION_FIELD] is not None:
            
            self.build_axial_surface_field()

        elif self.constraints[CoordinateType.FOLD_AXIS_FIELD] is not None:
            
            self.build_fold_axis_field()

        elif self.constraints[CoordinateType.X_AXIS_FIELD] is not None:
                
            self.build_x_axis_field()
        
