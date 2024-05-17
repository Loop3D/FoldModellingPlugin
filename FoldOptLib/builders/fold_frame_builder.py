from .builder import Builder
from ..input import OptData
from ..datatypes import CoordinateType
from LoopStructural import BoundingBox


class FoldFrameBuilder:
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

    def __init__(self, constraints: OptData, boundingbox: BoundingBox):
        """
        Initialize the FoldFrameBuilder.

        Parameters
        ----------
        constraints : OptData
            The optimization data.
        bounding_box : BoundingBox
            The bounding box.
        """

        self.constraints = constraints
        self.boundingbox = boundingbox
        self.xyz = self.constraints.data[["X", "Y", "Z"]].to_numpy()
        self.fold_frame = [None] * len(CoordinateType)

    def build_axial_surface_field(self):
        """
        Build the axial surface field.
        """
        builder = Builder(self.boundingbox)
        builder.set_constraints(self.constraints[CoordinateType.AXIAL_FOLIATION_FIELD])
        self.fold_frame[CoordinateType.AXIAL_FOLIATION_FIELD] = builder

    def build_fold_axis_field(self):
        """
        Build the fold axis field.
        """
        builder = Builder(self.boundingbox)
        builder.set_constraints(self.constraints[CoordinateType.FOLD_AXIS_FIELD])
        self.fold_frame[CoordinateType.FOLD_AXIS_FIELD] = builder

    def build_x_axis_field(self):
        """
        Build the x-axis field.
        """
        builder = Builder(self.boundingbox)
        builder.set_constraints(self.constraints[CoordinateType.X_AXIS])
        self.fold_frame[CoordinateType.X_AXIS] = builder

    def build(self):
        """
        Build the fold frame.
        """

        if self.constraints[CoordinateType.AXIAL_FOLIATION_FIELD] is None:
            raise ValueError("Axial surface field constraints not set.")

        if self.constraints[CoordinateType.AXIAL_FOLIATION_FIELD] is not None:
            self.build_axial_surface_field()

        if self.constraints[CoordinateType.FOLD_AXIS_FIELD] is not None:
            self.build_fold_axis_field()

        if self.constraints[CoordinateType.X_AXIS] is not None:
            self.build_x_axis_field()

    def __getitem__(self, coordinate_type: CoordinateType):
        """
        Get the fold frame coordinate.

        Parameters
        ----------
        coordinate_type : CoordinateType
            The type of fold frame coordinate.
        """

        return self.fold_frame[coordinate_type]
