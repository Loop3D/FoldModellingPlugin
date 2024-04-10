from dataclasses import dataclass
from typing import Union
import numpy
import beartype


@beartype.beartype
@dataclass
class Bounds:
    lower_bound: Union[float, list, numpy.ndarray]
    upper_bound: Union[float, list, numpy.ndarray]


@beartype.beartype
@dataclass
class NormalDistribution:
    bounds: Bounds
    mu: Union[float, list, numpy.ndarray]
    sigma: Union[float, list, numpy.ndarray]
    weight: Union[float, list, numpy.ndarray]


@beartype.beartype
@dataclass
class VonMisesFisherDistribution:
    bounds: Bounds
    mu: Union[float, list, numpy.ndarray]
    kappa: Union[float, list, numpy.ndarray]
    weight: Union[float, list, numpy.ndarray]
