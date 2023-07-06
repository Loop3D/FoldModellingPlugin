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



class FoldModel:

    def __init__(self, bounding_box):

        self.bounding_box = bounding_box
        self.model = None
    def initialise_model(self):

        self.model = GeologicalModel(self.bounding_box[0, :],
                                     self.bounding_box[1, :])

    def process_data(self, axial_normal):

        # d = s1_data.loc[s1_data.feature_name == 'STSC'].copy()
        # self.points = self.model.rescale(d[['X', 'Y', 'Z']].to_numpy())
        # self.points = self.model.rescale(s1_data[['X', 'Y', 'Z']].to_numpy())
        # self.model_initialisation()
        # self.coords = self.model.data[['X', 'Y', 'Z']].to_numpy()
        assert len(self.points) == len(self.orientation_data), "coordinates must have the same length as data"

        data = make_dataset_3(self.orientation_data, self.points, name='s0')
        dataset = make_dataset_2(self.points, axial_normal, coord=0)
        y = rotate_vector(axial_normal, np.pi / 2, dimension=3)
        y_coord = make_dataset_2(self.points, y, coord=1)
        # self.model.data = self.model.data.append(dataset)
        # self.model.data = self.model.data.append(y_coord)
        data = data.append(dataset)
        data = data.append(y_coord)
        self.model.data = data

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

