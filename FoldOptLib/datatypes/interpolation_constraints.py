from dataclasses import dataclass
from typing import Union, Optional
from ..datatypes.enums import ConstraintType
import numpy
import beartype


@dataclass
class InterpolationConstraints:
    """
    Class to store interpolation constraints for a given problem.

    Attributes
    ----------
    value_constraints : Optional[Union[list, numpy.ndarray]]
        The value constraints for the interpolation, [x, y, z, value, weight].
    tangent_constraints : Optional[Union[list, numpy.ndarray]]
        The tangent constraints for the interpolation, [x, y, z, tx, ty, tz, weight].
    normal_constraints : Optional[Union[list, numpy.ndarray]]
        The normal constraints for the interpolation., [x, y, z, nx, ny, nz, weight].
    gradient_constraints : Optional[Union[list, numpy.ndarray]]
        The gradient constraints for the interpolation, [x, y, z, gx, gy, gz, weight].
    """

    value_constraints: Optional[Union[list, numpy.ndarray]] = None
    tangent_constraints: Optional[Union[list, numpy.ndarray]] = None
    normal_constraints: Optional[Union[list, numpy.ndarray]] = None
    gradient_constraints: Optional[Union[list, numpy.ndarray]] = None

    @beartype.beartype
    def __getitem__(self, constraint_type: ConstraintType):
        constraints = {
            ConstraintType.VALUE: self.value_constraints,
            ConstraintType.TANGENT: self.tangent_constraints,
            ConstraintType.NORMAL: self.normal_constraints,
            ConstraintType.GRADIENT: self.gradient_constraints,
        }
        return constraints[constraint_type]
