from scipy.optimize import minimize
# import knowledge_constraints
# importlib.reload(knowledge_constraints)
# from modified_loopstructural.modified_foldframe import FoldFrame
# from modified_loopstructural.extra_utils import *
from knowledge_constraints._helper import *
from knowledge_constraints.knowledge_constraints import GeologicalKnowledgeConstraints
from knowledge_constraints.splot_processor import SPlotProcessor
from knowledge_constraints.fourier_optimiser import FourierSeriesOptimiser
from LoopStructural import GeologicalModel
from LoopStructural.modelling.features.fold import FoldEvent
from LoopStructural.modelling.features.fold import FoldRotationAngle, SVariogram
from LoopStructural.modelling.features.fold import fourier_series
from LoopStructural.utils.helper import *
from geological_sampler.sampling_methods import *
from uncertainty_quantification.fold_uncertainty import *
import numpy as np
import pandas as pd
import mplstereonet
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize, differential_evolution
from scipy.stats import vonmises_fisher


def huber_loss(residuals, delta=0.5):
    s = np.zeros(len(residuals))

    for i, residual in enumerate(residuals):
        if abs(residual) <= delta:
            s[i] = 0.5 * residual ** 2
        else:
            s[i] = delta * abs(residual) - 0.5 * delta ** 2

    return s


def calculate_semivariogram(fold_frame, fold_rotation, lag=None, nlag=60):
    svario = SVariogram(fold_frame, fold_rotation)
    svario.calc_semivariogram(lag=lag, nlag=nlag)
    wv = svario.find_wavelengths()
    theta = np.ones(4)
    theta[3] = wv[0]
    theta[0] = 0
    # py = wv[2]

    return theta, svario.lags, svario.variogram


def make_dataset_3(vec, xyz, name=None):
    # dgx = np.tile(vec, (len(xyz), 1))

    s2_dict = create_gradient_dict(x=xyz[:, 0],
                                   y=xyz[:, 1],
                                   z=xyz[:, 2],
                                   nx=vec[:, 0],
                                   ny=vec[:, 1],
                                   nz=vec[:, 2],
                                   feature_name=name, coord=0,
                                   data_type='foliation')
    dataset = pd.DataFrame()
    dataset = dataset.append(pd.DataFrame
                             (s2_dict, columns=['X', 'Y', 'Z', 'gx', 'gy', 'gz', 'feature_name', 'coord']))
    return dataset


def make_dataset_2(xyz, vec, coord=0):
    # vec = np.array(vec)
    # xyz = self.points

    dgx = np.tile(vec, (len(xyz), 1))

    s1_dict = create_gradient_dict(x=xyz[:, 0],
                                   y=xyz[:, 1],
                                   z=xyz[:, 2],
                                   nx=dgx[:, 0],
                                   ny=dgx[:, 1],
                                   nz=dgx[:, 2],
                                   feature_name='s1', coord=coord,
                                   data_type='foliation')
    # s1_dict = make_constant(s1_dict)
    dataset = pd.DataFrame()
    dataset = dataset.append(pd.DataFrame
                             (s1_dict, columns=['X', 'Y', 'Z', 'gx', 'gy', 'gz', 'feature_name', 'coord']))
    return dataset


def calculate_intersection_lineation(axial_surface, folded_foliation):
    """
    Calculate the intersection lineation of the axial surface and the folded foliation.

    Parameters:
    axial_surface (np.ndarray): The normal vector of the axial surface.
    folded_foliation (np.ndarray): The normal vector of the folded foliation.

    Returns:
    np.ndarray: The normalized intersection lineation vector.
    """
    # Check if the inputs are numpy arrays
    if not isinstance(axial_surface, np.ndarray):
        raise TypeError("Axial surface vector must be a numpy array.")
    if not isinstance(folded_foliation, np.ndarray):
        raise TypeError("Folded foliation vector must be a numpy array.")

    # Check if the inputs have the same shape
    if axial_surface.shape != folded_foliation.shape:
        raise ValueError("Axial surface and folded foliation arrays must have the same shape.")

    # Calculate cross product of the axial surface and folded foliation normal vectors
    intesection_lineation = np.cross(axial_surface, folded_foliation)

    # Normalise the intersection lineation vector
    intesection_lineation /= np.linalg.norm(intesection_lineation, axis=1)[:, None]

    return intesection_lineation


def get_fold_curves(geological_feature, fold_axis=None):
    if fold_axis is not None:
        x = np.linspace(geological_feature.fold.foldframe[1].min(),
                        geological_feature.fold.foldframe[1].max(), 200)
        curve = geological_feature.fold.fold_axis_rotation(x)
        return x, curve

    if fold_axis is None:
        x = np.linspace(geological_feature.fold.foldframe[0].min(),
                        geological_feature.fold.foldframe[0].max(), 200)
        curve = geological_feature.fold.fold_limb_rotation(x)
        return x, curve





def scale(data):
    mms = StandardScaler()
    mms.fit(data)
    data_transformed = mms.transform(data)

    return data_transformed


# def logp(value: TensorVariable, mu: TensorVariable) -> TensorVariable:
#     return -(value - mu)**2


class AxialSurfaceOptimizer:
    def __init__(self, grid, points, folded_foliation_data, constraints, model,
                 axial_strike_dip=None,
                 wavelength_guess=None,
                 guess_method='plane_fit'):
        self.orientation_data = folded_foliation_data
        self.axial_strike_dip = axial_strike_dip
        self.initial_axial_guess = None
        self.wavelength_guess = wavelength_guess
        self.model = model  # self.model_initialisation()
        self.guess_method = guess_method
        self.grid = grid
        self.axial_surface = None
        self.optimised_axial_surface = None
        self.constraints = constraints
        self.points = self.model.rescale(self.model.data[['X', 'Y', 'Z']].to_numpy())
        self.fitted_params = None
        self.coords = self.model.data[['X', 'Y', 'Z']].to_numpy()
        self.fold = None

    def model_initialisation(self):
        # xmax = 1000
        # ymax = 1000
        # zmax = 1000
        # bounding_box = np.array([[0, 0, 0],
        #                          [xmax, ymax, zmax]])

        self.model = GeologicalModel(boundary_points[0, :],
                                     boundary_points[1, :])

    def initial_guess(self):
        if self.axial_strike_dip is not None:

            self.initial_axial_guess = self.axial_strike_dip

        # calculate an estimate of the initial guess of the axial surface
        else:
            if self.guess_method == 'plane_fit':
                strike_dip, self.initial_axial_guess = ax_plane(self.orientation_data)

            if self.guess_method == 'stereonet':
                strikex, dipx = mplstereonet.vector2pole(self.orientation_data[:, 0],
                                                         self.orientation_data[:, 1],
                                                         self.orientation_data[:, 2])

                sd = np.array([strikex, dipx]).T

                # strikedip = normal_vector_to_strike_and_dip(self.orientation_data)
                axial_s, axial_dip = axial_plane_stereonet(sd[:, 0], sd[:, 1])

                self.initial_axial_guess = m_strike_dip_vector(axial_s, axial_dip)[0]

    def process_data(self, axial_normal):

        # d = s1_data.loc[s1_data.feature_name == 'STSC'].copy()
        # self.points = self.model.rescale(d[['X', 'Y', 'Z']].to_numpy())
        # self.points = self.model.rescale(s1_data[['X', 'Y', 'Z']].to_numpy())
        # self.model_initialisation()
        # self.coords = self.model.data[['X', 'Y', 'Z']].to_numpy()
        assert len(self.points) == len(self.orientation_data), "coordinates must of the same length as data"

        data = make_dataset_3(self.orientation_data, self.points, name='s0')
        dataset = make_dataset_2(self.points, axial_normal, coord=0)
        y = rotate_vector(axial_normal, np.pi / 2, dimension=3)
        y_coord = make_dataset_2(self.points, y, coord=1)
        # self.model.data = self.model.data.append(dataset)
        # self.model.data = self.model.data.append(y_coord)
        data = data.append(dataset)
        data = data.append(y_coord)
        self.model.data = data

    # @partial(jit, static_argnums=0)
    def build_fold_frame(self, axial_normal):

        # self.model.data = data
        self.process_data(axial_normal)
        self.axial_surface = self.model.create_and_add_fold_frame('s1',
                                                                  buffer=0.3,
                                                                  solver='pyamg',
                                                                  nelements=2e4,
                                                                  damp=True)

        self.model.update(progressbar=False)

        # try:
        # self.axial_surface.model.set_model(self.model)
        # except:
        #     pass

        return self.axial_surface

    #     def jax_out_function(self, function):

    #         jaxpr, _, consts = make_jaxpr(function)
    #         function_nograd = lambda *args, **kwargs: eval_jaxpr(jaxpr, consts, *args, **kwargs)

    #         return function_nograd

    def calculate_fold_rotation_angle(self):

        # self.axial_surface = self.build_fold_frame()
        # calculate the fold limb rotation angles and the scalar field of the axial surface
        foldframe = FoldFrame('s1', self.axial_surface)
        # s0gg = ref_s0.evaluate_gradient(locations[i])
        s1g = self.axial_surface[0].evaluate_gradient(self.coords)
        s1g /= np.linalg.norm(s1g, axis=1)[:, None]
        # dot = np.einsum('ij,ij->i', s1g, self.orientation_data)
        # self.orientation_data[dot<0] *= -1

        flr, fld = foldframe.calculate_fold_limb_rotation(self.coords, self.orientation_data)

        return flr, fld

    def svariogram(self, fold_frame, rotation_angles):

        if self.wavelength_guess is not None:

            return [0, 1, 1, self.wavelength_guess]

        else:
            if len(rotation_angles) < 5:

                pdist = np.abs(fold_frame[:, None] - fold_frame[None, :])
                pdist[pdist == 0.] = np.nan
                lagx = np.nanmean(np.nanmin(pdist, axis=1))
                theta, lag, vario = calculate_semivariogram(fold_frame, rotation_angles,
                                                            lag=lagx,
                                                            nlag=60)

            else:

                theta, lag, vario = calculate_semivariogram(fold_frame, rotation_angles,
                                                            lag=None,
                                                            nlag=None)

        return theta

    # def setup_constraints(self, tightness, asymmetry, axial_trace, wavelength):
    #     # define the constraints based on the geological knowledge of the folded surface
    #     pass

    def calculate_rotation_angles(self):

        self.fold = self.model.create_and_add_folded_foliation('s0',
                                                               fold_frame=self.axial_surface,
                                                               axis_wl=500,
                                                               limb_wl=500,
                                                               buffer=0.3,
                                                               solver='fake',
                                                               nelements=100,
                                                               skip_variogram=True,
                                                               # svario=False,
                                                               damp=True)
        self.model.update(progressbar=False)
        self.fold.set_model(self.model)

        # flr = self.fold.fold.fold_limb_rotation.rotation_angle
        # fld = self.fold.fold.fold_limb_rotation.fold_frame_coordinate

        far = self.fold.fold.fold_axis_rotation.rotation_angle
        fad = self.fold.fold.fold_axis_rotation.fold_frame_coordinate

        fitted_far = self.fit_fourier_series(fad, far, constr_type='fold_axis')
        print(fitted_far.x)

        fold_axis_rotation = FoldRotationAngle(far, fad)
        fold_axis_rotation.set_function(lambda x: np.rad2deg(
            np.arctan(fourier_series(x, *fitted_far.x))))

        fold = FoldEvent(self.axial_surface,
                         fold_axis_rotation=fold_axis_rotation)

        # fold_axis = fold.fold_axis_rotation(self.points)
        flr, fld = self.fold.fold.foldframe.calculate_fold_limb_rotation(self.fold.builder,
                                                                         axis=fold.get_fold_axis_orientation)
        # flr *= -1
        fitted_flr = self.fit_fourier_series(fld, flr, constr_type='fold_limb')
        print(fitted_flr.x)
        fold_limb_rotation = FoldRotationAngle(flr, fld)
        fold_limb_rotation.set_function(lambda x: np.rad2deg(
            np.arctan(fourier_series(x, *fitted_flr.x))))

        fold.fold_limb_rotation = fold_limb_rotation

        return fold

    def fit_fourier_series(self, fold_frame, rotation_angle, constr_type='fold_limb'):
        # fit a fourier series to the rotation angles and the scalar field
        # flr, fld = self.calculate_fold_rotation_angle()
        guess = self.svariogram(fold_frame, rotation_angle)
        # print(guess[3])
        if constr_type == 'fold_limb':
            x = np.linspace(self.axial_surface[0].min(),
                            self.axial_surface[0].max(), 100)
        if constr_type == 'fold_axis':
            x = np.linspace(self.axial_surface[1].min(),
                            self.axial_surface[1].max(), 100)

        # print(x)
        fourier_opt = FourierSeriesOptimiser(fold_frame, rotation_angle,
                                             self.constraints[constr_type], x,
                                             at_constrain_only=None,
                                             coeff=4)
        # fourier_loglike = fourier_opt.objective_value
        # print(guess[3])
        opt = fourier_opt.fit_constrained_fourier_series(guess[3], x0=guess)

        return opt

    def get_predicted_bedding(self, fld, flr, fitted_params):

        # calculate the fold direction using the fourier parameters
        fold_limb_rotation = FoldRotationAngle(flr, fld)
        # fold_limb_rotation.fitted_params = theta
        fold_limb_rotation.set_function(lambda x: np.rad2deg(
            np.arctan(fourier_series(x, *fitted_params))))
        s1g = self.axial_surface[0].evaluate_gradient(self.coords)
        s1g /= np.linalg.norm(s1g, axis=1)[:, None]
        # print(len(s1g))
        fold_axis = calculate_intersection_lineation(s1g, self.orientation_data)
        mean_fold_axis = fold_axis.mean(0)
        mean_fold_axis /= np.linalg.norm(mean_fold_axis)
        # print(len(fold_axis))
        fold = FoldEvent(self.axial_surface,
                         fold_limb_rotation=fold_limb_rotation,
                         fold_axis=mean_fold_axis)
        fold_direction, fold_axis, zg = fold.get_deformed_orientation(self.coords)
        fold_direction /= np.linalg.norm(fold_direction, axis=1)[:, None]
        dot = np.einsum('ij,ij->i', s1g, fold_direction)
        fold_direction[dot < 0] *= -1
        predicted_bedding = np.cross(fold_axis, fold_direction)
        predicted_bedding /= np.linalg.norm(predicted_bedding, axis=1)[:, None]

        # free up memory
        del fold_direction, fold_axis, fold, dot, s1g, fold_limb_rotation
        gc.collect()

        return predicted_bedding

    def get_predicted_bedding_2(self, fold):

        #         # calculate the fold direction using the fourier parameters
        #         fold_limb_rotation = FoldRotationAngle(flr, fld)
        #         # fold_limb_rotation.fitted_params = theta
        #         fold_limb_rotation.set_function(lambda x: np.rad2deg(
        #                 np.arctan(fourier_series(x, *ffitted_params))))

        #         fold_axis_rotation = FoldRotationAngle(far, fad)
        #         fold_axis_rotation.set_function(lambda x: np.rad2deg(
        #                 np.arctan(fourier_series(x, *afitted_params))))

        s1g = self.axial_surface[0].evaluate_gradient(self.coords)
        s1g /= np.linalg.norm(s1g, axis=1)[:, None]
        # print(len(s1g))
        # print(len(fold_axis))
        # fold = FoldEvent(self.axial_surface,
        #                  fold_limb_rotation=fold_limb_rotation,
        #                  fold_axis_rotation=fold_axis_rotation)
        fold_direction, fold_axis = fold.get_deformed_orientation(self.coords)
        fold_direction /= np.linalg.norm(fold_direction, axis=1)[:, None]
        dot = np.einsum('ij,ij->i', s1g, fold_direction)
        fold_direction[dot < 0] *= -1
        predicted_bedding = np.cross(fold_axis, fold_direction)
        predicted_bedding /= np.linalg.norm(predicted_bedding, axis=1)[:, None]
        # predicted_bedding = rotate_vector(predicted_bedding, np.pi, dimension=3)
        print(predicted_bedding)

        return predicted_bedding

    def angle_difference(self, v1, v2):

        """
        Calculate the angle difference between
        the predicted bedding and the observed one.

        """

        # project the projected bedding and the observed one
        # onto the fold axis plane
        # projected_v1 = np.cross(fold_axis,
        #                         np.cross(v1, fold_axis, axisa=1, axisb=1),
        #                         axisa=1, axisb=1)
        # projected_v2 = np.cross(fold_axis,
        #                         np.cross(v2, fold_axis, axisa=1, axisb=1),
        #                         axisa=1, axisb=1)
        # projected_v1 /= np.linalg.norm(projected_v1, axis=1)[:, None]
        # projected_v2 /= np.linalg.norm(projected_v2, axis=1)[:, None]
        # # dot product between two unit vectors
        # dot_product = np.einsum("ij,ij->i", projected_v1, projected_v2)
        dot_product = np.einsum("ij,ij->i", v1, v2)

        return np.arccos(dot_product)

    def mle_objective(self, strike_dip):  # axial_normal):

        axial_normal = m_strike_dip_vector(strike_dip[0], strike_dip[1])[0]
        # init = self.initial_guess()
        axial_normal /= np.linalg.norm(axial_normal)
        self.build_fold_frame(axial_normal)
        flr, fld = self.calculate_fold_rotation_angle()
        # print('flr: ', flr)
        # print('fld: ', fld)
        # print('ith axial surface : ', axial_normal)
        opt = self.fit_fourier_series(fld, flr)
        self.fitted_params = opt.x
        # print('ith Fourier params :', self.fitted_params)

        predicted_bedding = self.get_predicted_bedding(fld, flr,
                                                       self.fitted_params)
        angle_difference = self.angle_difference(predicted_bedding,
                                                 self.orientation_data)
        del predicted_bedding, opt, flr, fld, self.axial_surface
        # angle_difference /= np.linalg.norm(angle_difference)
        # angle_difference = angle_difference.mean()
        # print('angle differences', angle_difference)

        # vm_logpdf = np.sum(-vM.logpdf(angle_difference, 10, loc=0))
        # mu = self.initial_axial_guess  # np.array([ 0.8 , 0.,  0.])
        # kappa = 5
        # vMF_logpdf = -vonmises_fisher_logp(axial_normal, mu, kappa)
        # log_likelihood = np.sum(-vonmises.logpdf(angle_difference, 0, 0.1))

        objective = angle_difference.sum()  # vm_logpdf
        del angle_difference
        gc.collect()
        # print('Objective fun:', objective)
        print(
            f"Axial surface optimisation...  \nAxial surface: {strike_dip} \nFourier params : {self.fitted_params} \nObjective fun : {objective}",
            end='\r', flush=True)
        # print('Axial surface: ', strike_dip, end='\r', flush=True)
        # print('Fourier params :', self.fitted_params, end='\r', flush=True)
        # print('Objective fun : ', objective, end='\r', flush=True)

        return objective

    def mle_objective_2(self, axial_normal):
        # init = self.initial_guess()
        axial_normal /= np.linalg.norm(axial_normal)
        # self.axial_surface = axial_normal
        initial_guess = self.initial_guess()
        fold_frame = self.build_fold_frame(axial_normal)
        fold = self.calculate_rotation_angles()
        print('ith axial surface : ', axial_normal)
        predicted_bedding = self.get_predicted_bedding_2(fold)
        angle_difference = self.angle_difference(predicted_bedding,
                                                 self.orientation_data)
        print('angle differences', angle_difference)

        # vm_logpdf = np.sum(-vM.logpdf(angle_difference, 10, loc=0))
        # mu = self.initial_axial_guess #np.array([ 0.8 , 0.,  0.])
        # kappa = 10
        # vMF_logpdf = -vonmises_fisher_logp(axial_normal, mu, kappa)
        objective = angle_difference.sum()

        print('Objective fun:', objective)

        return objective

    def find_axial_surface(self):
        init = self.initial_guess()
        x0 = self.initial_axial_guess
        # n = 30
        # V = generate_directional_training_dataset(x0,
        #                                           concentration=100,
        #                                           size=n)
        # init_pos = V['normals']
        # init_pos /= np.linalg.norm(init_pos, axis=1)[:, None]
        # del V
        # gc.collect()
        # min_nx = init_pos[:, 0].min()
        # max_nx = init_pos[:, 0].max()
        # min_ny = init_pos[:, 1].min()
        # max_ny = init_pos[:, 1].max()
        # min_nz = init_pos[:, 2].min()
        # max_nz = init_pos[:, 2].max()
        # bounds = np.array([(min_nx, max_nx),
        #                    (min_ny, max_ny),
        #                    (min_nz, max_nz)])
        # mu, sigma = 80, 10 # mean and standard deviation
        np.random.seed(180)
        dip = np.random.normal(80, 10, 20)
        strike = np.random.normal(110, 10, 20)
        mask = dip < 90
        dip = dip[mask]
        strike = strike[mask]
        init_pos = np.array([strike, dip]).T
        bounds = np.array([(0, 360), (0, 90)])  # np.array([(-1, 1), (-1, 1), (-1, 1)])
        best_pos = differential_evolution(self.mle_objective,
                                          bounds,
                                          # x0=[0.9999999, -0., 0.],
                                          init=init_pos,
                                          # init='sobol',
                                          strategy='best2exp',
                                          # mutation=(0.2, 0.45),
                                          polish=True,
                                          )
        x = best_pos.x
        x /= np.linalg.norm(x)

        self.optimised_axial_surface = x

        return self.optimised_axial_surface
