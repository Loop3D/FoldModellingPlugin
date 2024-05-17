from dataclasses import dataclass
from typing import Union, Optional, List
import numpy
import beartype


@beartype.beartype
@dataclass
class Bounds:
    lower_bound: Optional[Union[int, float, List, numpy.ndarray]] = None
    upper_bound: Optional[Union[int, float, List, numpy.ndarray]] = None


@beartype.beartype
@dataclass
class NormalDistribution:
    mu: Union[int, float]
    sigma: Union[int, float]
    weight: Optional[Union[int, float]] = 1.0


@beartype.beartype
@dataclass
class VonMisesFisherDistribution:
    mu: Union[List, numpy.ndarray]
    kappa: Union[int, float]
    weight: Optional[Union[int, float]] = 1.0
