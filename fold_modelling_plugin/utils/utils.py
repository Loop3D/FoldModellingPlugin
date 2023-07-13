import numpy as np
from typing import Union


def calculate_semivariogram(fold_frame, fold_rotation, lag=None, nlag=60):
    svario = SVariogram(fold_frame, fold_rotation)
    svario.calc_semivariogram(lag=lag, nlag=nlag)
    wv = svario.find_wavelengths()
    theta = np.ones(4)
    theta[3] = wv[0]
    theta[0] = 0
    # py = wv[2]

    return theta, svario.lags, svario.variogram


def get_predicted_rotation_angle(theta, fold_frame_coordinate):
    y_pred = np.tan(np.deg2rad(fourier_series(
        fold_frame_coordinate, *theta)))

    return y_pred


def fourier_series(x, c0, c1, c2, w):
    """

    Parameters
    ----------
    x
    c0
    c1
    c2
    w

    Returns
    -------

    """
    v = np.array(x.astype(float))
    # v.fill(c0)
    v = c0 + c1 * np.cos(2 * np.pi / w * x) + c2 * np.sin(2 * np.pi / w * x)
    return np.rad2deg(np.arctan(v))
