from dataclasses import dataclass
from typing import Union, Optional
import numpy
import beartype


@beartype.beartype
@dataclass
class Bounds:
    lower_bound: Optional[Union[int, float, list, numpy.ndarray]] = None
    upper_bound: Optional[Union[int, float, list, numpy.ndarray]] = None


@beartype.beartype
@dataclass
class NormalDistribution:
    mu: Union[int, float, list, numpy.ndarray]
    sigma: Union[int, float, list, numpy.ndarray]
    weight: Optional[Union[int, float, list, numpy.ndarray]] = 1.


@beartype.beartype
@dataclass
class VonMisesFisherDistribution:
    mu: Union[int, float, list, numpy.ndarray]
    kappa: Union[int, float, list, numpy.ndarray]
    weight: Optional[Union[int, float, list, numpy.ndarray]] = 1.
