import numpy as np
import pytest
from FoldOptLib.datatypes.interpolation_constraints import InterpolationConstraints
from FoldOptLib.datatypes.enums import ConstraintType


def test_interpolation_constraints():
    # Initialize InterpolationConstraints object
    ic = InterpolationConstraints()

    # Test value_constraints
    ic.value_constraints = [1, 2, 3, 4, 5]
    assert ic[ConstraintType.VALUE] == [1, 2, 3, 4, 5]

    # Test tangent_constraints
    ic.tangent_constraints = [1, 2, 3, 4, 5, 6, 7]
    assert ic[ConstraintType.TANGENT] == [1, 2, 3, 4, 5, 6, 7]

    # Test normal_constraints
    ic.normal_constraints = [1, 2, 3, 4, 5, 6, 7]
    assert ic[ConstraintType.NORMAL] == [1, 2, 3, 4, 5, 6, 7]

    # Test gradient_constraints
    ic.gradient_constraints = [1, 2, 3, 4, 5, 6, 7]
    assert ic[ConstraintType.GRADIENT] == [1, 2, 3, 4, 5, 6, 7]

    # Test with numpy array
    ic.value_constraints = np.array([1, 2, 3, 4, 5])
    assert np.array_equal(ic[ConstraintType.VALUE], np.array([1, 2, 3, 4, 5]))
