# TODO : Make a simplified fold frame builder. isolate codes from GeologicalModel
# class from get_interpolator() and create_and_add_fold_frame() methods.
# Use directly structural_frame_builder and FDI interpolator.

"""
Main entry point for creating a geological model
"""
from ...utils import getLogger, log_to_file

import numpy as np
import pandas as pd

try:
    from ...interpolators import DiscreteFoldInterpolator as DFI

    dfi = True
except ImportError:
    dfi = False
from ...interpolators import FiniteDifferenceInterpolator as FDI

try:
    from ...interpolators import PiecewiseLinearInterpolator as PLI

    pli = True
except ImportError:
    pli = False

# if LoopStructural.experimental:
from ...interpolators import P2Interpolator

try:
    from ...interpolators import SurfeRBFInterpolator as Surfe

    surfe = True

except ImportError:
    surfe = False

from ...interpolators import StructuredGrid
from ...interpolators import TetMesh
from ...modelling.features.fault import FaultSegment
from ...interpolators import DiscreteInterpolator

from .features.builders import (
    FaultBuilder,
    GeologicalFeatureBuilder,
    StructuralFrameBuilder,
    FoldedFeatureBuilder,
)
from ...modelling.features import (
    UnconformityFeature,
    StructuralFrame,
    GeologicalFeature,
    FeatureType,
)
from ...modelling.features.fold import (
    FoldEvent,
    FoldFrame,
)

from ...utils.exceptions import InterpolatorError
from ...utils.helper import (
    all_heading,
    gradient_vec_names,
    strike_dip_vector,
    get_vectors,
)

from ...modelling.intrusions import IntrusionBuilder

from ...modelling.intrusions import IntrusionFrameBuilder


class SimpleFoldFrameBuilder:

    def __init__(self, data, origin, maximum, rescale=True, reuse_supports=False, nsteps=1):

        self.bounding_box = None
        self.features = []
        self.feature_name_index = {}
        self._data = None
        self.data = data
        self.nsteps = nsteps

        # we want to rescale the model area so that the maximum length is
        # 1
        self.origin = np.array(origin).astype(float)
        originstr = f"Model origin: {self.origin[0]} {self.origin[1]} {self.origin[2]}"
        logger.info(originstr)
        self.maximum = np.array(maximum).astype(float)
        maximumstr = "Model maximum: {} {} {}".format(
            self.maximum[0], self.maximum[1], self.maximum[2]
        )
        logger.info(maximumstr)

        lengths = self.maximum - self.origin
        self.scale_factor = 1.0
        self.bounding_box = np.zeros((2, 3))
        self.bounding_box[1, :] = self.maximum - self.origin
        self.bounding_box[1, :] = self.maximum - self.origin
        if rescale:
            self.scale_factor = float(np.max(lengths))
            logger.info(
                "Rescaling model using scale factor {}".format(self.scale_factor)
            )

        self.bounding_box /= self.scale_factor
        self.support = {}
        self.reuse_supports = reuse_supports
        self.stratigraphic_column = None

        self.tol = 1e-10 * np.max(self.bounding_box[1, :] - self.bounding_box[0, :])
        self._dtm = None

    def set_interpolator(self, nelements, buffer=0.2, element_volume=None):
        """
        Set up an interpolator for a structured grid in 3D space.

        This function creates a structured grid interpolator based on the bounding box,
        total volume, and number of elements. It calculates the step vector for a regular
        cube and ensures that the number of steps in each direction is adequate for
        creating an interpolator.

        Parameters
        ----------
        bb : ndarray
            2x3 array representing the bounding box with min and max coordinates.
        nelements : int
            Number of elements in the structured grid.
        element_volume : float, optional
            Volume of a single element. If None, it is calculated from box_vol and nelements.

        Returns
        -------
        FDI
            An instance of the FDI class representing the structured grid interpolator.

        Raises
        ------
        ValueError
            If the number of steps in any direction is less than 3, making it impossible to
            create a valid interpolator.
        """
        bb = np.copy(self.bounding_box)
        # add a buffer to the interpolation domain, this is necessary for
        # faults but also generally a good
        # idea to avoid boundary problems
        # buffer = bb[1, :]
        buffer = (np.min(bb[1, :] - bb[0, :])) * buffer
        bb[0, :] -= buffer  # *(bb[1,:]-bb[0,:])
        bb[1, :] += buffer  # *(bb[1,:]-bb[0,:])
        box_vol = (bb[1, 0] - bb[0, 0]) * (bb[1, 1] - bb[0, 1]) * (bb[1, 2] - bb[0, 2])

        # Find the volume of one element
        if element_volume is None:
            element_volume = box_vol / nelements
        # Calculate the step vector of a regular cube
        step_vector = np.zeros(3)
        step_vector[:] = element_volume ** (1.0 / 3.0)
        # Number of steps is the length of the box / step vector
        nsteps = np.ceil((bb[1, :] - bb[0, :]) / step_vector).astype(int)
        if np.any(np.less(nsteps, 3)):
            axis_labels = ["x", "y", "z"]
            for i in range(3):
                if nsteps[i] < 3:
                    # logger.error(
                    #     f"Number of steps in direction {axis_labels[i]} is too small, try increasing nelements"
                    # )
                    raise ValueError("Number of steps too small cannot create interpolator")
        # Create a structured grid using the origin and number of steps
        if self.reuse_supports:
            grid_id = f"grid_{nelements}"
            grid = self.support.get(
                grid_id,
                StructuredGrid(origin=bb[0, :], nsteps=nsteps, step_vector=step_vector),
            )
            if grid_id not in self.support:
                self.support[grid_id] = grid
        else:
            grid = StructuredGrid(origin=bb[0, :], nsteps=nsteps, step_vector=step_vector)

        # logger.info(
        #     f"Creating regular grid with {grid.n_elements} elements \n"
        #     "for modelling using FDI"
        # )
        return FDI(grid)

    def create_and_add_fold_frame(self, foldframe_data, tol=None, **kwargs):
        """
        Parameters
        ----------
        foldframe_data : string
            unique string in feature_name column

        kwargs

        Returns
        -------
        fold_frame : FoldFrame
            the created fold frame
        """
        # if not self.check_inialisation():
        #     return False
        if tol is None:
            tol = self.tol

        # create fault frame
        interpolator = self.set_interpolator(**kwargs)
        #
        fold_frame_builder = StructuralFrameBuilder(
            interpolator, name=foldframe_data, frame=FoldFrame, **kwargs
        )
        # add data
        fold_frame_data = self.data[self.data["feature_name"] == foldframe_data]
        fold_frame_builder.add_data_from_data_frame(fold_frame_data)
        # self._add_faults(fold_frame_builder[0])
        # self._add_faults(fold_frame_builder[1])
        # self._add_faults(fold_frame_builder[2])
        kwargs["tol"] = tol
        fold_frame_builder.setup(**kwargs)
        fold_frame = fold_frame_builder.frame

        fold_frame.type = FeatureType.STRUCTURALFRAME
        fold_frame.builder = fold_frame_builder
        # self._add_feature(fold_frame)

        return fold_frame
