import numpy as np
import pandas as pd
from ..from_loopstructural._svariogram import SVariogram
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


def fourier_series_x_intercepts(x, popt):
    v = fourier_series(x, *popt)

    foldrotm = np.ma.masked_where(v > 0, v)
    b = np.roll(foldrotm.mask, 1).astype(int) - foldrotm.mask.astype(int)
    c = np.roll(foldrotm.mask, -1).astype(int) - foldrotm.mask.astype(int)
    x_int = x[b != 0]
    x_int2 = x[c != 0]
    x_intr = x_int + x_int2
    x_intr /= 2
    return x_intr


def save_load_object(obj=None, file_path=None, mode='save'):
    """
    Saves or loads a python object to/from a file using the dill library.

    Parameters:
    obj (Any, optional): The python object to be saved.
    file_path (str, optional): The file path where the object should be saved or loaded from.
    mode (str, optional): The mode of operation. Must be either 'save' or 'load'. Defaults to 'save'.

    Returns:
    Any: The loaded python object, if `mode` is set to 'load'.
    None: Otherwise.

    Raises:
    ValueError: If `mode` is not set to either 'save' or 'load'.
    """
    if mode == 'save':
        with open(file_path, 'wb') as file:
            dill.dump(obj, file)
        print("Object saved to file:", file_path)
    elif mode == 'load':
        with open(file_path, 'rb') as file:
            loaded_obj = dill.load(file)
        print("Object loaded from file:", file_path)
        return loaded_obj
    else:
        raise ValueError("Invalid mode. Must be either 'save' or 'load'.")


def strike_dip_to_vectors(strike, dip):
    vec = np.zeros((len(strike), 3))
    s_r = np.deg2rad(strike)
    d_r = np.deg2rad((dip))
    vec[:, 0] = np.sin(d_r) * np.cos(s_r)
    vec[:, 1] = -np.sin(d_r) * np.sin(s_r)
    vec[:, 2] = np.cos(d_r)
    vec /= np.linalg.norm(vec, axis=1)[:, None]
    return vec


def strike_dip_to_vector(strike, dip):
    """
    Calculate the strike-dip vector given the strike and dip angles.

    Parameters:
    strike (float): The strike angle in degrees.
    dip (float): The dip angle in degrees.

    Returns:
    np.ndarray: The normalized strike-dip vector.
    """
    # Check if the inputs are of correct type
    if not isinstance(strike, (int, float)):
        raise TypeError(f"Expected strike to be a number, got {type(strike).__name__}")
    if not isinstance(dip, (int, float)):
        raise TypeError(f"Expected dip to be a number, got {type(dip).__name__}")

    # Convert degrees to radians
    s_r = np.deg2rad(strike)
    d_r = np.deg2rad(dip)

    # Calculate the components of the strike-dip vector
    nx = np.sin(d_r) * np.cos(s_r)
    ny = -np.sin(d_r) * np.sin(s_r)
    nz = np.cos(d_r)

    # Create the vector and normalize it
    vec = np.array([nx, ny, nz]).T
    vec /= np.linalg.norm(vec)

    return vec


def normal_vector_to_strike_and_dip(normal_vector):
    """
    Calculate the strike and dip angles given a normal vector.

    Parameters:
    normal_vector (np.ndarray): The normal vector.

    Returns:
    np.ndarray: The strike and dip angles in degrees.
    """
    # Check if the input is a numpy array
    if not isinstance(normal_vector, np.ndarray):
        raise TypeError("Normal vector must be a numpy array.")

    # Normalize the normal vector
    normal_vector /= np.linalg.norm(normal_vector, axis=1)[:, None]

    # Calculate the dip angle
    dip = np.degrees(np.arccos(normal_vector[:, 2]))

    # Calculate the strike angle
    strike = -np.rad2deg(np.arctan2(normal_vector[:, 1], normal_vector[:, 0]))

    return np.array([strike, dip]).T


def rotate_vector(v, angle, dimension=2):
    """
    Rotate a vector by a given angle around the origin using a rotation matrix.
    Args:
        v (ndarray): The vector to rotate.
        angle (float): The angle to rotate the vector by in radians.
        dimension (int): The dimension of the vector (2 or 3). Default is 2.
    Returns:
        ndarray: The rotated vector.
    """
    if dimension == 2:
        # Define the 2D rotation matrix
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle), np.cos(angle)]])
    elif dimension == 3:
        # Define the 3D rotation matrix
        R = np.array([[np.cos(angle), -np.sin(angle), 0],
                      [np.sin(angle), np.cos(angle), 0],
                      [0, 0, 1]])
    else:
        raise ValueError("Dimension must be either 2 or 3.")

    # Rotate the vector by multiplying with the rotation matrix
    v_rotated = np.dot(R, v)
    v_rotated /= np.linalg.norm(v_rotated)

    return v_rotated


def get_fold_curves(geological_feature, fold_frame=0):
    """
    Calculate the fold axis and limb rotation angle curves of a geological feature.

    Parameters:
    geological_feature: The geological feature to calculate the fold rotation angle curves for.
    fold_frame (optional): The fold frame coordinate to use, default 0 for fold limb rotation angle.
    for fold axis rotation angle use 1.
    If not provided, the function will use a default axis.

    Returns:
    tuple: The x values and the corresponding fold curve values.
    """
    # Check if the geological_feature has the required attributes
    if not hasattr(geological_feature, 'fold') or not hasattr(geological_feature.fold, 'foldframe') or not hasattr(
            geological_feature.fold, 'fold_axis_rotation') or not hasattr(geological_feature.fold,
                                                                          'fold_limb_rotation'):
        raise AttributeError(
            "Geological feature must have a 'fold' attribute with 'foldframe', "
            "'fold_axis_rotation', and 'fold_limb_rotation' attributes.")

    # Determine the axis to use for the calculation
    coordinate_to_use = fold_frame

    # Calculate the fold frame coordinate values x and the fold rotation angle curve
    x = np.linspace(geological_feature.fold.foldframe[coordinate_to_use].min(),
                    geological_feature.fold.foldframe[coordinate_to_use].max(), 200)
    curve = geological_feature.fold.fold_axis_rotation(
        x) if fold_frame is 1 else geological_feature.fold.fold_limb_rotation(x)

    return x, curve


def create_dict(x=None, y=None, z=None, strike=None, dip=None, feature_name=None,
                coord=None, data_type=None, **kwargs):
    if data_type == 'foliation':
        fn = np.empty(len(x)).astype(str)
        fn.fill(feature_name)
        c = np.empty((len(x))).astype(int)
        c.fill(coord)
        dictionary = {'X': x,
                      'Y': y,
                      'Z': z,
                      'strike': strike,
                      'dip': dip,
                      'feature_name': fn,
                      'coord': c}
        return dictionary

    if data_type == 'fold_axis':
        fn = np.empty(len(x)).astype(str)
        fn.fill(feature_name)
        c = np.empty((len(x))).astype(int)
        c.fill(coord)
        dictionary = {'X': x,
                      'Y': y,
                      'Z': z,
                      'plunge_dir': strike,
                      'plunge': dip,
                      'feature_name': fn,
                      'coord': c}
        return dictionary


def create_gradient_dict(x=None, y=None, z=None,
                         nx=None, ny=None, nz=None,
                         feature_name=None, coord=None,
                         **kwargs):
    fn = np.empty(len(x)).astype(str)
    fn.fill(feature_name)
    c = np.empty((len(x))).astype(int)
    c.fill(coord)
    dictionary = {'X': x,
                  'Y': y,
                  'Z': z,
                  'gx': nx,
                  'gy': ny,
                  'gz': nz,
                  'feature_name': fn,
                  'coord': c}
    return dictionary


def make_dataset(vec: np.ndarray, points: np.ndarray, name: str = 's0', coord: int = 0) -> pd.DataFrame:
    """

    Make a dataset from one unit vector and xyz points of the folded feature data.

    Parameters
    ----------
    vec : np.ndarray
        The unit vector to be used as the gradient.
    points : np.ndarray
        The xyz coordinates of the data points.
    name : str, optional
        The name of the feature, by default 's0'.
    coord : int, optional
        The coordinate, by default 0.

    Returns
    -------
    pd.DataFrame
        A DataFrame where each row represents a data point with its coordinates (X, Y, Z),
        gradient (gx, gy, gz), feature name, and coordinate.
    """
    g = np.tile(vec, (len(points), 1))
    dataset = pd.DataFrame()
    dataset['X'] = points[:, 0]
    dataset['Y'] = points[:, 1]
    dataset['Z'] = points[:, 2]
    dataset['gx'] = g[:, 0]
    dataset['gy'] = g[:, 1]
    dataset['gz'] = g[:, 2]
    dataset['feature_name'] = name
    dataset['coord'] = coord

    return dataset


def get_wavelength_guesses(guess, size):
    np.random.seed(180)
    mu, sigma = guess, guess / 3
    return np.random.normal(mu, abs(sigma), size)


def objective_wrapper(func1, func2):
    def objective_function(x):
        return func1(x) + func2(x)

    return objective_function
