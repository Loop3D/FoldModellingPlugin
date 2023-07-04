import numpy as np


def gaussian_log_likelihood(b, mu, sigma):
    """
    Calculate the log-likelihood of a Gaussian distribution.

    Parameters
    ----------
    b : Union[int, float]
        The observed value.
    mu : Union[int, float]
        The mean of the Gaussian distribution.
    sigma : Union[int, float]
        The standard deviation of the Gaussian distribution.

    Returns
    -------
    Union
        The calculated log-likelihood.
    """
    logpdf = -0.5 * np.log(2 * np.pi * sigma ** 2) - 0.5 * (b - mu) ** 2 / sigma ** 2

    return logpdf