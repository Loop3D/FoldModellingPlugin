#TODO : Make a simplified fold frame builder. isolate codes from GeologicalModel
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

from ...modelling.features.builders import (
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

    def __init__(self):

        self.bounding_box = None
        # print('tet')
        if logfile:
            self.logfile = logfile
            log_to_file(logfile, level=loglevel)

        logger.info("Initialising geological model")
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
        if self.reuse_supports:
            logger.warning(
                "Supports are shared between geological features \n"
                "this may cause unexpected behaviour and should only\n"
                "be use by advanced users"
            )
        logger.info("Reusing interpolation supports: {}".format(self.reuse_supports))
        self.stratigraphic_column = None

        self.tol = 1e-10 * np.max(self.bounding_box[1, :] - self.bounding_box[0, :])
        self._dtm = None

    def get_interpolator(
        self,
        interpolatortype="FDI",
        nelements=1e4,
        buffer=0.2,
        element_volume=None,
        **kwargs,
    ):
        """
        Returns an interpolator given the arguments, also constructs a
        support for a discrete interpolator

        Parameters
        ----------
        interpolatortype : string
            define the interpolator type
        nelements : int
            number of elements in the interpolator
        buffer : double or numpy array 3x1
            value(s) between 0,1 specifying the buffer around the bounding box
        data_bb : bool
            whether to use the model boundary or the boundary around
        kwargs : no kwargs used, this just catches any additional arguments

        Returns
        -------
        interpolator : GeologicalInterpolator
            A geological interpolator

        Notes
        -----
        This method will create a geological interpolator for the bounding box of the model. A
        buffer area is added to the interpolation region to avoid boundaries and issues with faults.
        This function wil create a :class:`LoopStructural.interpolators.GeologicalInterpolator` which can either be:
        A discrete interpolator :class:`LoopStructural.interpolators.DiscreteInterpolator`

        - 'FDI' :class:`LoopStructural.interpolators.FiniteDifferenceInterpolator`
        - 'PLI' :class:`LoopStructural.interpolators.PiecewiseLinearInterpolator`
        - 'P1'  :class:`LoopStructural.interpolators.P1Interpolator`
        - 'DFI' :class:`LoopStructural.interpolators.DiscreteFoldInterpolator`
        - 'P2'  :class:`LoopStructural.interpolators.P2Interpolator`
        or

        - 'surfe'  :class:`LoopStructural.interpolators.SurfeRBFInterpolator`

        The discrete interpolators will require a support.

        - 'PLI','DFI','P1Interpolator','P2Interpolator' :class:`LoopStructural.interpolators.supports.TetMesh` or you can provide another
          mesh builder which returns :class:`LoopStructural.interpolators.support.UnStructuredTetMesh`

        - 'FDI' :class:`LoopStructural.interpolators.supports.StructuredGrid`
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
        if interpolatortype == "PLI" and pli:
            if element_volume is None:
                # nelements /= 5
                element_volume = box_vol / nelements
            # calculate the step vector of a regular cube
            step_vector = np.zeros(3)
            step_vector[:] = element_volume ** (1.0 / 3.0)
            # step_vector /= np.array([1,1,2])
            # number of steps is the length of the box / step vector
            nsteps = np.ceil((bb[1, :] - bb[0, :]) / step_vector).astype(int)
            if np.any(np.less(nsteps, 3)):
                axis_labels = ["x", "y", "z"]
                for i in range(3):
                    if nsteps[i] < 3:
                        logger.error(
                            f"Number of steps in direction {axis_labels[i]} is too small, try increasing nelements"
                        )
                logger.error("Cannot create interpolator: number of steps is too small")
                raise ValueError("Number of steps too small cannot create interpolator")
            # create a structured grid using the origin and number of steps
            if self.reuse_supports:
                mesh_id = f"mesh_{nelements}"
                mesh = self.support.get(
                    mesh_id,
                    TetMesh(origin=bb[0, :], nsteps=nsteps, step_vector=step_vector),
                )
                if mesh_id not in self.support:
                    self.support[mesh_id] = mesh
            else:
                if "meshbuilder" in kwargs:
                    mesh = kwargs["meshbuilder"](bb, nelements)
                else:
                    mesh = TetMesh(
                        origin=bb[0, :], nsteps=nsteps, step_vector=step_vector
                    )
            logger.info(
                "Creating regular tetrahedron mesh with %i elements \n"
                "for modelling using PLI" % (mesh.ntetra)
            )

            return PLI(mesh)
        if interpolatortype == "P2":
            if element_volume is None:
                # nelements /= 5
                element_volume = box_vol / nelements
            # calculate the step vector of a regular cube
            step_vector = np.zeros(3)
            step_vector[:] = element_volume ** (1.0 / 3.0)
            # step_vector /= np.array([1,1,2])
            # number of steps is the length of the box / step vector
            nsteps = np.ceil((bb[1, :] - bb[0, :]) / step_vector).astype(int)
            if "meshbuilder" in kwargs:
                mesh = kwargs["meshbuilder"](bb, nelements)
            else:
                raise NotImplementedError(
                    "Cannot use P2 interpolator without external mesh"
                )
            logger.info(
                "Creating regular tetrahedron mesh with %i elements \n"
                "for modelling using P2" % (mesh.ntetra)
            )
            return P2Interpolator(mesh)
        if interpolatortype == "FDI":

            # find the volume of one element
            if element_volume is None:
                element_volume = box_vol / nelements
            # calculate the step vector of a regular cube
            step_vector = np.zeros(3)
            step_vector[:] = element_volume ** (1.0 / 3.0)
            # number of steps is the length of the box / step vector
            nsteps = np.ceil((bb[1, :] - bb[0, :]) / step_vector).astype(int)
            if np.any(np.less(nsteps, 3)):
                logger.error("Cannot create interpolator: number of steps is too small")
                axis_labels = ["x", "y", "z"]
                for i in range(3):
                    if nsteps[i] < 3:
                        logger.error(
                            f"Number of steps in direction {axis_labels[i]} is too small, try increasing nelements"
                        )
                raise ValueError("Number of steps too small cannot create interpolator")
            # create a structured grid using the origin and number of steps
            if self.reuse_supports:
                grid_id = "grid_{}".format(nelements)
                grid = self.support.get(
                    grid_id,
                    StructuredGrid(
                        origin=bb[0, :], nsteps=nsteps, step_vector=step_vector
                    ),
                )
                if grid_id not in self.support:
                    self.support[grid_id] = grid
            else:
                grid = StructuredGrid(
                    origin=bb[0, :], nsteps=nsteps, step_vector=step_vector
                )
            logger.info(
                f"Creating regular grid with {grid.n_elements} elements \n"
                "for modelling using FDI"
            )
            return FDI(grid)

        if interpolatortype == "DFI" and dfi is True:
            if element_volume is None:
                nelements /= 5
                element_volume = box_vol / nelements
            # calculate the step vector of a regular cube
            step_vector = np.zeros(3)
            step_vector[:] = element_volume ** (1.0 / 3.0)
            # number of steps is the length of the box / step vector
            nsteps = np.ceil((bb[1, :] - bb[0, :]) / step_vector).astype(int)
            # create a structured grid using the origin and number of steps
            if "meshbuilder" in kwargs:
                mesh = kwargs["meshbuilder"].build(bb, nelements)
            else:
                mesh = kwargs.get(
                    "mesh",
                    TetMesh(origin=bb[0, :], nsteps=nsteps, step_vector=step_vector),
                )
            logger.info(
                f"Creating regular tetrahedron mesh with {mesh.ntetra} elements \n"
                "for modelling using DFI"
            )
            return DFI(mesh, kwargs["fold"])
        if interpolatortype == "Surfe" or interpolatortype == "surfe":
            # move import of surfe to where we actually try and use it
            if not surfe:
                logger.warning("Cannot import Surfe, try another interpolator")
                raise ImportError("Cannot import surfepy, try pip install surfe")
            method = kwargs.get("method", "single_surface")
            logger.info("Using surfe interpolator")
            return Surfe(method)
        logger.warning("No interpolator")
        raise InterpolatorError("Could not create interpolator")