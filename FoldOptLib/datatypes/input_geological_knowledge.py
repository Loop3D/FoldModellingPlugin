from dataclasses import dataclass
from ..utils import strike_dip_to_vector
from .enums import KnowledgeType
from .probability_distributions import NormalDistribution, VonMisesFisherDistribution
import beartype
from typing import Optional


@beartype.beartype
@dataclass
class InputGeologicalKnowledge:
    asymmetry: Optional[NormalDistribution] = None
    fold_wavelength: Optional[NormalDistribution] = None
    axis_wavelength: Optional[NormalDistribution] = None
    tightness: Optional[NormalDistribution] = None
    hinge_line: Optional[NormalDistribution] = None
    axial_trace: Optional[NormalDistribution] = None
    axial_surface: Optional[VonMisesFisherDistribution] = None

    def __post_init__(self):
        if self.axial_surface is not None:

            if len(self.axial_surface.mu) == 2:
                self.axial_surface.mu = strike_dip_to_vector(
                    self.axial_surface.mu[0], self.axial_surface.mu[1]
                )

    def __getitem__(self, input_knowledge: KnowledgeType):

        knowledge_map = {
            KnowledgeType.ASYMMETRY: self.asymmetry,
            KnowledgeType.AXIAL_TRACE: self.axial_trace,
            KnowledgeType.WAVELENGTH: self.fold_wavelength,
            KnowledgeType.AXIS_WAVELENGTH: self.axis_wavelength,
            KnowledgeType.AXIAL_SURFACE: self.axial_surface,
            KnowledgeType.TIGHTNESS: self.tightness,
            KnowledgeType.HINGE_ANGLE: self.hinge_line,
        }

        return knowledge_map[input_knowledge]
