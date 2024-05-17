import numpy
import pandas
from ..from_loopstructural._svariogram import SVariogram
import mplstereonet
import dill


def calculate_semivariogram(fold_frame, fold_rotation, lag=None, nlag=None):
    svario = SVariogram(fold_frame, fold_rotation)
    svario.calc_semivariogram(lag=lag, nlag=nlag)
    wv = svario.find_wavelengths()
    theta = numpy.ones(4)
    theta[3] = wv[0]
    theta[0] = 0
    # py = wv[2]

    return theta, svario.lags, svario.variogram


def get_predicted_rotation_angle(theta, fold_frame_coordinate):
    # y_pred = numpy.tan(numpy.deg2rad(fourier_series(
    #     fold_frame_coordinate, *theta)))
    y_pred = fourier_series(fold_frame_coordinate, *theta)

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
    v = numpy.array(x.astype(float))
    # v.fill(c0)
    v = c0 + c1 * numpy.cos(2 * numpy.pi / w * x) + c2 * numpy.sin(2 * numpy.pi / w * x)
    return numpy.rad2deg(numpy.arctan(v))


def fourier_series_x_intercepts(x, popt):
    v = fourier_series(x, *popt)

    foldrotm = numpy.ma.masked_where(v > 0, v)
    b = numpy.roll(foldrotm.mask, 1).astype(int) - foldrotm.mask.astype(int)
    c = numpy.roll(foldrotm.mask, -1).astype(int) - foldrotm.mask.astype(int)
    x_int = x[b != 0]
    x_int2 = x[c != 0]
    x_intr = x_int + x_int2
    x_intr /= 2
    return x_intr


def save_load_object(obj=None, file_path=None, mode="save"):
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
    if mode == "save":
        with open(file_path, "wb") as file:
            dill.dump(obj, file)
        print("Object saved to file:", file_path)
    if mode == "load":
        with open(file_path, "rb") as file:
            loaded_obj = dill.load(file)
        print("Object loaded from file:", file_path)
        return loaded_obj
    else:
        raise ValueError("Invalid mode. Must be either 'save' or 'load'.")


def strike_dip_to_vectors(strike, dip):
    vec = numpy.zeros((len(strike), 3))
    s_r = numpy.deg2rad(strike)
    d_r = numpy.deg2rad((dip))
    vec[:, 0] = numpy.sin(d_r) * numpy.cos(s_r)
    vec[:, 1] = -numpy.sin(d_r) * numpy.sin(s_r)
    vec[:, 2] = numpy.cos(d_r)
    vec /= numpy.linalg.norm(vec, axis=1)[:, None]
    return vec


def strike_dip_to_vector(strike, dip):
    """
    Calculate the strike-dip vector given the strike and dip angles.

    Parameters:
    strike (float): The strike angle in degrees.
    dip (float): The dip angle in degrees.

    Returns:
    numpy.ndarray: The normalized strike-dip vector.
    """
    # Check if the inumpyuts are of correct type
    if not isinstance(strike, (int, float)):
        raise TypeError(f"Expected strike to be a number, got {type(strike).__name__}")
    if not isinstance(dip, (int, float)):
        raise TypeError(f"Expected dip to be a number, got {type(dip).__name__}")

    # Convert degrees to radians
    s_r = numpy.deg2rad(strike)
    d_r = numpy.deg2rad(dip)

    # Calculate the components of the strike-dip vector
    nx = numpy.sin(d_r) * numpy.cos(s_r)
    ny = -numpy.sin(d_r) * numpy.sin(s_r)
    nz = numpy.cos(d_r)

    # Create the vector and normalize it
    vec = numpy.array([nx, ny, nz]).T
    vec /= numpy.linalg.norm(vec)

    return vec


def normal_vector_to_strike_and_dip(normal_vector):
    """
    Calculate the strike and dip angles given a normal vector.

    Parameters:
    normal_vector (numpy.ndarray): The normal vector.

    Returns:
    numpy.ndarray: The strike and dip angles in degrees.
    """
    # Check if the inumpyut is a numpy array
    if not isinstance(normal_vector, numpy.ndarray):
        raise TypeError("Normal vector must be a numpy array.")

    # Normalize the normal vector
    normal_vector /= numpy.linalg.norm(normal_vector, axis=1)[:, None]

    # Calculate the dip angle
    dip = numpy.degrees(numpy.arccos(normal_vector[:, 2]))

    # Calculate the strike angle
    strike = -numpy.rad2deg(numpy.arctan2(normal_vector[:, 1], normal_vector[:, 0]))

    return numpy.array([strike, dip]).T


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
        R = numpy.array(
            [
                [numpy.cos(angle), -numpy.sin(angle)],
                [numpy.sin(angle), numpy.cos(angle)],
            ]
        )
    elif dimension == 3:
        # Define the 3D rotation matrix
        R = numpy.array(
            [
                [numpy.cos(angle), -numpy.sin(angle), 0],
                [numpy.sin(angle), numpy.cos(angle), 0],
                [0, 0, 1],
            ]
        )
    else:
        raise ValueError("Dimension must be either 2 or 3.")

    # Rotate the vector by multiplying with the rotation matrix
    v_rotated = numpy.dot(R, v)
    v_rotated /= numpy.linalg.norm(v_rotated)

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
    if (
        not hasattr(geological_feature, "fold")
        or not hasattr(geological_feature.fold, "foldframe")
        or not hasattr(geological_feature.fold, "fold_axis_rotation")
        or not hasattr(geological_feature.fold, "fold_limb_rotation")
    ):
        raise AttributeError(
            "Geological feature must have a 'fold' attribute with 'foldframe', "
            "'fold_axis_rotation', and 'fold_limb_rotation' attributes."
        )

    # Determine the axis to use for the calculation
    coordinate_to_use = fold_frame

    # Calculate the fold frame coordinate values x and the fold rotation angle curve
    x = numpy.linspace(
        geological_feature.fold.foldframe[coordinate_to_use].min(),
        geological_feature.fold.foldframe[coordinate_to_use].max(),
        200,
    )
    curve = (
        geological_feature.fold.fold_axis_rotation(x)
        if fold_frame == 1
        else geological_feature.fold.fold_limb_rotation(x)
    )

    return x, curve


def create_dict(
    x=None,
    y=None,
    z=None,
    strike=None,
    dip=None,
    feature_name=None,
    coord=None,
    **kwargs,
):
    fn = numpy.empty(len(x)).astype(str)
    fn.fill(feature_name)
    c = numpy.empty((len(x))).astype(int)
    c.fill(coord)
    dictionary = {
        "X": x,
        "Y": y,
        "Z": z,
        "strike": strike,
        "dip": dip,
        "feature_name": fn,
        "coord": c,
    }

    return dictionary


def create_gradient_dict(
    x=None,
    y=None,
    z=None,
    nx=None,
    ny=None,
    nz=None,
    feature_name=None,
    coord=None,
    **kwargs,
):
    fn = numpy.empty(len(x)).astype(str)
    fn.fill(feature_name)
    c = numpy.empty((len(x))).astype(int)
    c.fill(coord)
    dictionary = {
        "X": x,
        "Y": y,
        "Z": z,
        "gx": nx,
        "gy": ny,
        "gz": nz,
        "feature_name": fn,
        "coord": c,
    }
    return dictionary


def create_fold_frame_dataset(model, strike=0, dip=0):
    s1_ori = numpy.array([strike, dip])
    xyz = model.regular_grid(nsteps=[10, 10, 10])
    s1_orientation = numpy.tile(s1_ori, (len(xyz), 1))
    s1_dict = create_dict(
        x=xyz[:, 0][0:10:2],
        y=xyz[:, 1][0:10:2],
        z=xyz[:, 2][0:10:2],
        strike=s1_orientation[:, 0][0:10:2],
        dip=s1_orientation[:, 1][0:10:2],
        feature_name="s1",
        coord=0,
    )
    # Generate a dataset using s1 dictionary
    dataset = pandas.DataFrame(
        s1_dict, columns=["X", "Y", "Z", "strike", "dip", "feature_name", "coord"]
    )
    # Add y coordinate axis orientation. Y coordinate axis always perpendicular
    # to the axial surface and roughly parallel to the fold axis
    s2y = dataset.copy()
    s2s = s2y[["strike", "dip"]].to_numpy()
    s2s[:, 0] += 90
    s2s[:, 1] = dip
    s2y["strike"] = s2s[:, 0]
    s2y["dip"] = s2s[:, 1]
    s2y["coord"] = 1
    # Add y coordinate dictionary to s1 dataframe
    dataset = pandas.concat([dataset, s2y])

    return dataset, xyz


def create_dataset(
    vec: numpy.ndarray, points: numpy.ndarray, name: str = "s0", coord: int = 0
) -> pandas.DataFrame:
    """

    Make a dataset from one unit vector and xyz points of the folded feature data.

    Parameters
    ----------
    vec : numpy.ndarray
        The unit vector to be used as the gradient.
    points : numpy.ndarray
        The xyz coordinates of the data points.
    name : str, optional
        The name of the feature, by default 's0'.
    coord : int, optional
        The coordinate, by default 0.

    Returns
    -------
    pandas.DataFrame
        A DataFrame where each row represents a data point with its coordinates (X, Y, Z),
        gradient (gx, gy, gz), feature name, and coordinate.
    """
    g = numpy.tile(vec, (len(points), 1))
    dataset = pandas.DataFrame()
    dataset["X"] = points[:, 0]
    dataset["Y"] = points[:, 1]
    dataset["Z"] = points[:, 2]
    dataset["gx"] = g[:, 0]
    dataset["gy"] = g[:, 1]
    dataset["gz"] = g[:, 2]
    dataset["feature_name"] = name
    dataset["coord"] = coord

    return dataset


def get_wavelength_guesses(guess, size):
    rng = numpy.random.default_rng(1651465414615413541564580)

    mu, sigma = guess, guess / 3
    return rng.standard_normal(mu, abs(sigma), size)


def calculate_intersection_lineation(axial_surface, folded_foliation):
    """
    Calculate the intersection lineation of the axial surface and the folded foliation.

    Parameters:
    axial_surface (numpy.ndarray): The normal vector of the axial surface.
    folded_foliation (numpy.ndarray): The normal vector of the folded foliation.

    Returns:
    numpy.ndarray: The normalised intersection lineation vector.
    """
    # Check if the inumpyuts are numpy arrays
    if not isinstance(axial_surface, numpy.ndarray):
        raise TypeError("Axial surface vector must be a numpy array.")
    if not isinstance(folded_foliation, numpy.ndarray):
        raise TypeError("Folded foliation vector must be a numpy array.")

    # Check if the inumpyuts have the same shape
    if axial_surface.shape != folded_foliation.shape:
        raise ValueError(
            "Axial surface and folded foliation arrays must have the same shape."
        )

    # Calculate cross product of the axial surface and folded foliation normal vectors
    li = numpy.cross(axial_surface, folded_foliation)

    # Normalise the intersection lineation vector
    li /= numpy.linalg.norm(li, axis=1)[:, None]

    return li


def axial_plane_stereonet(strike, dip):
    """

    Calculate the axial plane in a stereonet given the strike and dip angles.
    credit: https://mplstereonet.readthedocs.io/en/latest/examples/axial_plane.html

    Parameters:
    strike (numpy.ndarray): The strike angles in degrees.
    dip (numpy.ndarray): The dip angles in degrees.

    Returns:
    tuple: The axial strike and dip angles in degrees.
    """
    # Check if the inumpyuts are numpy arrays
    if not isinstance(strike, numpy.ndarray):
        raise TypeError(
            f"Expected strike to be a numpy array, got {type(strike).__name__}"
        )
    if not isinstance(dip, numpy.ndarray):
        raise TypeError(f"Expected dip to be a numpy array, got {type(dip).__name__}")

    # Check if the inumpyuts have the same shape
    if strike.shape != dip.shape:
        raise ValueError("Strike and dip arrays must have the same shape.")

    # Find the two modes
    centers = mplstereonet.kmeans(strike, dip, num=2, measurement="poles")

    # Fit a girdle to the two modes
    axis_s, axis_d = mplstereonet.fit_girdle(*zip(*centers), measurement="radians")

    # Find the midpoint
    mid, _ = mplstereonet.find_mean_vector(*zip(*centers), measurement="radians")
    midx, midy = mplstereonet.line(*mid)

    # Find the axial plane by fitting another girdle to the midpoint and the pole of the plunge axis
    xp, yp = mplstereonet.pole(axis_s, axis_d)
    x, y = [xp, midx], [yp, midy]
    axial_s, axial_dip = mplstereonet.fit_girdle(x, y, measurement="radians")

    return axial_s, axial_dip


def clean_knowledge_dict(geological_knowledge):
    keys_to_delete = [key for key, value in geological_knowledge.items() if not value]
    for key in keys_to_delete:
        del geological_knowledge[key]

    return geological_knowledge


def objective_wrapper(func1, func2):
    def objective_function(x):
        return func1(x) + func2(x)

    return objective_function
