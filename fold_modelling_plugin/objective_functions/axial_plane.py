import numpy as np


def is_axial_plane_compatible(v1, v2):
    """
    Calculate the angle difference between
    the predicted bedding and the observed one.

    """
    dot_product = np.einsum("ij,ij->i", v1, v2)

    angle_difference = np.arccos(dot_product)

    return angle_difference
