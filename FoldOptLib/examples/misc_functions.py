import numpy as np


def sample_random_dataset(bounding_box, sample_size=2, seed=180):
    """
    Generate a random dataset of 3D coordinates within a specified model bounding box.

    Parameters
    ----------
    bounding_box : ndarray
        A 3x3 array where each row represents the minimum and maximum values of x and y coordinates.
    sample_size : int, optional
        The number of random samples to generate. Default is 2.
    seed : int, optional
        The seed for the random number generator for reproducibility. Default is 180.

    Returns
    -------
    random_xyz : ndarray
        A sample_size x 3 array of random 3D coordinates within the specified bounding box.
    """
    # Set the seed for the random number generator for reproducible results
    np.random.seed(seed)

    # Extract the maximum x and y coordinates from the bounding box
    xmax, ymax = bounding_box[1, 0], bounding_box[1, 1]

    # Define the maximum z coordinate (fixed at 0 to only sample model's surface)
    zmax = 0.

    # Generate 'sample_size' number of random x-coordinates
    xn = np.random.uniform(low=bounding_box[0, 0], high=xmax, size=sample_size)

    # Generate 'sample_size' number of random y-coordinates
    yn = np.random.uniform(low=bounding_box[0, 1], high=ymax, size=sample_size)

    # Create an array of z-coordinates, all set to 'zmax' (fixed z-coordinate for all points)
    zn = np.tile(zmax, sample_size)

    # Combine the x, y, and z coordinates into a single 2D array (shape: sample_size x 3)
    random_xyz = np.array([xn, yn, zn]).T

    return random_xyz
