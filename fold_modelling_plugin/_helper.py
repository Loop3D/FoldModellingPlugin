from modified_loopstructural.extra_utils import *
import numpy as np
import joblib as jb
from LoopStructural.modelling.features.fold import fourier_series
from uncertainty_quantification.fold_uncertainty import *
import dill
import mplstereonet


# def gaussian_func(b, mu, sigma):
#     return 0.5 * np.exp(- (b - mu) ** 2 / (
#             2 * sigma ** 2))

def gaussian_func(b, mu, sigma):
    return -0.5 * np.log(2 * np.pi * sigma ** 2) - 0.5 * (b - mu) ** 2 / sigma ** 2


def fourier_series_2(x, theta):
    c0 = theta[0]
    T = np.array([theta[1:4], theta[4:]])
    v = 0
    for i in range(len(T)):
        t = np.concatenate([[c0], T[i]])
        v += fourier_series(x, *t)
        # v = v.astype(float)
    # v1 = c0 + c1 * np.cos(2 * np.pi / w * x) + c2 * np.sin(2 * np.pi / w * x)
    return np.rad2deg(np.arctan(v))


def parallel(function, array, jobs=1):
    results = jb.Parallel(n_jobs=jobs, verbose=1, prefer='threads')(jb.delayed(function)(i) for i in array)
    results = np.asarray(results, dtype='object')
    return results


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


def axial_plane_stereonet(strike, dip):
    """
    Calculate the axial plane in a stereonet given the strike and dip angles.
    https://mplstereonet.readthedocs.io/en/latest/examples/axial_plane.html

    Parameters:
    strike (np.ndarray): The strike angles in degrees.
    dip (np.ndarray): The dip angles in degrees.

    Returns:
    tuple: The axial strike and dip angles in degrees.
    """
    # Check if the inputs are numpy arrays
    if not isinstance(strike, np.ndarray):
        raise TypeError(f"Expected strike to be a numpy array, got {type(strike).__name__}")
    if not isinstance(dip, np.ndarray):
        raise TypeError(f"Expected dip to be a numpy array, got {type(dip).__name__}")

    # Check if the inputs have the same shape
    if strike.shape != dip.shape:
        raise ValueError("Strike and dip arrays must have the same shape.")

    # Find the two modes
    centers = mplstereonet.kmeans(strike, dip, num=2, measurement='poles')

    # Fit a girdle to the two modes
    axis_s, axis_d = mplstereonet.fit_girdle(*zip(*centers), measurement='radians')

    # Find the midpoint
    mid, _ = mplstereonet.find_mean_vector(*zip(*centers), measurement='radians')
    midx, midy = mplstereonet.line(*mid)

    # Find the axial plane by fitting another girdle to the midpoint and the pole of the plunge axis
    xp, yp = mplstereonet.pole(axis_s, axis_d)
    x, y = [xp, midx], [yp, midy]
    axial_s, axial_dip = mplstereonet.fit_girdle(x, y, measurement='radians')

    return axial_s, axial_dip


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
