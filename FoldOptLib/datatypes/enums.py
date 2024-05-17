from enum import IntEnum


class DataType(IntEnum):
    DATA = 0
    GEOLOGICAL_KNOWLEDGE = 1
    BOUNDING_BOX = 2


class KnowledgeType(IntEnum):
    ASYMMETRY = 0
    AXIAL_TRACE = 1
    WAVELENGTH = 2
    AXIS_WAVELENGTH = 3
    TIGHTNESS = 4
    HINGE_ANGLE = 5
    AXIAL_SURFACE = 6


class OptimisationType(IntEnum):
    ANGLE = 0
    MLE = 1
    VM_MLE = 2
    FOURIER = 3
    LEAST_SQUARES = 4
    PROBABILISTIC = 5

class OptimiserType(IntEnum):
    AXIAL_SURFACE = 0
    FOURIER = 1


class ObjectiveType(IntEnum):
    LOG_NORMAL = 0
    VON_MISES = 1
    FOURIER = 2
    ANGLE = 3


class SolverType(IntEnum):
    DIFFERENTIAL_EVOLUTION = 0
    CONSTRAINED_TRUST_REGION = 1
    UNCONSTRAINED_TRUST_REGION = 2
    PARTICLE_SWARM = 3


class FitType(IntEnum):
    LIMB = 0
    AXIS = 1
    AXIAL_SURFACE = 2


class ConstraintType(IntEnum):
    VALUE = 0
    TANGENT = 1
    NORMAL = 2
    GRADIENT = 3


class CoordinateType(IntEnum):
    AXIAL_FOLIATION_FIELD = 0
    FOLD_AXIS_FIELD = 1
    X_AXIS = 2
