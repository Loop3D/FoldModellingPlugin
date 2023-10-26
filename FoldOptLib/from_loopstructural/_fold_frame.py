# import logging
# from ....utils import getLogger
# logger = getLogger(__name__)

import numpy as np
from LoopStructural.modelling.features._structural_frame import StructuralFrame


class FoldFrame(StructuralFrame):
    def __init__(self, name, features, fold=None):
        """
        A structural frame that can calculate the fold axis/limb rotation angle
        Same constructor arguments as parent StructuralFrame

        Parameters
        ----------
        name
        features
        """
        super().__init__(name, features, fold)
        self.model = None

    def calculate_fold_axis_rotation(self, points, gradient, fold_axis=None):
        """
        Calculate the fold axis rotation angle by finding the angle between the
        intersection lineation and the gradient to the 1st coordinate of the
        fold frame
        Parameters
        ----------
        feature_builder - GeologicalFeatureBuilder
            - the builder for the geological feature that is folded

        Returns
        -------

        """
        # We need to ignore the fault when we are calculating the splot because it is done
        # in the restored space
        # self.features[0].faults_enabled = False
        # self.features[1].faults_enabled = False
        s1g = self.features[0].evaluate_gradient(points)
        s1g /= np.linalg.norm(s1g, axis=1)[:, None]
        s1gyg = self.features[1].evaluate_gradient(points)
        s1gyg /= np.linalg.norm(s1gyg, axis=1)[:, None]
        l1 = self.calculate_intersection_lineation(points, gradient)
        l1 /= np.linalg.norm(l1, axis=1)[:, None]
        if fold_axis is not None:
            if fold_axis.shape[0] > 0 and fold_axis.shape[1] == 6:
                l1 = np.vstack([l1, fold_axis])

        fad = self.features[1].evaluate_value(points)
        # project s0 onto axis plane B X A X B
        projected_l1 = np.cross(
            s1g, np.cross(l1, s1g, axisa=1, axisb=1), axisa=1, axisb=1
        )
        projected_s1gyg = np.cross(
            s1g, np.cross(s1gyg, s1g, axisa=1, axisb=1), axisa=1, axisb=1
        )

        # einsum dot product
        far = np.einsum("ij,ij->i", projected_l1, projected_s1gyg)
        far = np.rad2deg(np.arccos(far))
        # scalar triple product
        stp = np.einsum("ij,ij->i", np.cross(l1, s1gyg, axisa=1, axisb=1), s1g)
        # check bounds
        far -= 90
        # far[stp < 0] = 360.- far[stp < 0]
        far[far > 90] = far[far > 90] + -180
        far[far < -90] = far[far < -90] + 180

        return far, fad

    def calculate_fold_limb_rotation(self, points, gradient, axis=None):
        """
        Calculate the fold limb rotation angle using the axis specified and
        the normals to the folded foliation
        Parameters
        ----------
        feature_builder - GeologicalFeatureBuilder
            the feature interpolator for the folded feature that has the
            datapoints the fold limb rotation angle is
            going to be calculated for
        axis - GeologicalFeature
            Optional. Fold axis feature that when queried for location
            returns the fold axis

        Returns
        -------
        fold_limb_rotation, coordinate_0
        """
        # self.features[0].faults_enabled = False
        # self.features[1].faults_enabled = False
        # for f in feature_builder.faults:
        #     points[:,:3] = f.apply_to_points(points[:,:3])
        # get the normals from the points array
        s0g = gradient
        s0g /= np.linalg.norm(s0g, axis=1)[:, None]
        # calculate the gradient and value of the first coordinate of the
        # fold frame
        # for the locations and normalise
        s1g = self.features[0].evaluate_gradient(points)
        s1g /= np.linalg.norm(s1g, axis=1)[:, None]
        s1 = self.features[0].evaluate_value(points)
        # self.features[0].faults_enabled = True
        # self.features[1].faults_enabled = True

        if axis is None:
            # logger.info("Not using fold axis for fold limb rotation angle calculation")
            r2 = np.einsum("ij,ij->i", s1g, s0g)

            return np.rad2deg(np.arcsin(r2)), s1
        if axis is not None:
            fold_axis = axis(points)
            # project s0 onto axis plane B X A X B
            projected_s0 = np.cross(
                fold_axis, np.cross(s0g, fold_axis, axisa=1, axisb=1), axisa=1, axisb=1
            )
            projected_s1 = np.cross(
                fold_axis, np.cross(s1g, fold_axis, axisa=1, axisb=1), axisa=1, axisb=1
            )
            projected_s0 /= np.linalg.norm(projected_s0, axis=1)[:, None]
            projected_s1 /= np.linalg.norm(projected_s1, axis=1)[:, None]
            r2 = np.einsum("ij,ij->i", projected_s1, projected_s0)  #
            # adjust the fold rotation angle so that its always between -90
            # and 90
            vv = np.cross(s1g, s0g, axisa=1, axisb=1)
            ds = np.einsum("ij,ij->i", fold_axis, vv)
            flr = np.rad2deg(
                np.arcsin(r2)
            )  # np.where(ds > 0, np.rad2deg(np.arcsin(r2)),
            # (- )))
            flr[ds < 0] *= -1

            # flr = np.where(flr < -90, (180. + flr), flr)
            # flr = np.where(flr > 90, -(180. - flr), flr)
            return flr, s1

    def calculate_intersection_lineation(self, points, gradient):
        """
        Calculate the intersection lineation by finding the cross product
        between the first fold frame
        coordinate and the vector representing the normal to the folded
        foliation
        Parameters
        ----------
        feature_builder - GeologicalFeatureBuilder
            the feature builder that contains the data points that the
            intersection lineation is calculated for


        Returns Nx3 array of doubles
        -------

        """
        points = np.vstack(points)
        s1g = self.features[0].evaluate_gradient(points)
        s1g /= np.linalg.norm(points, axis=1)[:, None]
        gradient /= np.linalg.norm(gradient, axis=1)[:, None]
        l1 = np.cross(s1g, gradient)
        l1 /= np.linalg.norm(l1, axis=1)[:, None]
        return l1
