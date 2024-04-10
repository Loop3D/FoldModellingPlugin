from enum import IntEnum


class DataType(IntEnum):
    FoldedAxialSurface = 0
    FoldedFoliation = 0
    GeologicalKnowledge = 1
    BoundingBox = 2


class KnowledgeType(IntEnum):
    Asymmetry = 0
    AxialTrace = 1
    FoldWavelength = 2
    AxisWavelength = 3
    Tightness = 4
    HingeAngle = 5
    AxialSurface = 6


class OptimisationType(IntEnum):
    Angle = 0
    MaximumLikelihoodEstimation = 1
    LeastSquares = 2
    Probabilistic = 3


class LogLikelihoodType(IntEnum):
    Normal = 0
    VonMisesFisher = 1
    FourierSeries = 2
