from enum import IntEnum


class DataType(IntEnum):
    FoldedAxialSurface = 0
    FoldedFoliation = 1
    KNOWLEDGE = 2
    BOUNDINGBOX = 3


class KnowledgeType(IntEnum):
    ASYMMETRY = 0
    AXIALTRACE = 1
    FOLDWAVELENGTH = 2
    AXISWAVELENGTH = 3
    TIGHTNESS = 4
    HINGEANGLE = 5
    AXIALSURFACE = 6


class OptimisationType(IntEnum):
    ANGLE = 0
    MLE = 1
    LEASTSQUARES = 2
    PROBABILISTIC = 3


class LikelihoodType(IntEnum):
    LOGNORMAL = 0
    VMF = 1
    FOURIER = 2
