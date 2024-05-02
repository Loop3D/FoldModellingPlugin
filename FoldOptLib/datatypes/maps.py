from ..objective_functions import ObjectiveFunction, LeastSquaresFunctions, GeologicalKnowledgeFunctions, \
    is_axial_plane_compatible
from .enums import ObjectiveType, OptimisationType, KnowledgeType, DataType
from enum import Enum


# TODO: finish this class
class OptimisationMethod(Enum):
    methods_map = {
        OptimisationType.MLE: ObjectiveFunction.log_normal,
        OptimisationType.VM_MLE: ObjectiveFunction.vector_loglikelihood,
        OptimisationType.FOURIER: ObjectiveFunction.fourier_loglikelihood,
        OptimisationType.ANGLE: is_axial_plane_compatible
    }

    def __call__(self, optimisation_type: OptimisationType):
        return self.methods_map[optimisation_type]
