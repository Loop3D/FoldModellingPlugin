from dataclasses import dataclass, field
from .enums import KnowledgeType
from .probability_distributions import NormalDistribution, VonMisesFisherDistribution
import beartype
from typing import Dict, Union


@beartype.beartype
@dataclass
class InputGeologicalKnowledge:
    asymmetry: NormalDistribution
    axial_trace: NormalDistribution
    fold_wavelength: NormalDistribution
    axis_wavelength: NormalDistribution
    axial_surface: VonMisesFisherDistribution
    tightness: NormalDistribution
    hinge_line: NormalDistribution
    knowledge_map: Dict[KnowledgeType, Union[NormalDistribution, VonMisesFisherDistribution]] = field(init=False)

    def __post_init__(self):
        self.knowledge_map = {
            KnowledgeType.Asymmetry: self.asymmetry,
            KnowledgeType.AxialTrace: self.axial_trace,
            KnowledgeType.FoldWavelength: self.fold_wavelength,
            KnowledgeType.AxisWavelength: self.axis_wavelength,
            KnowledgeType.AxialSurface: self.axial_surface,
            KnowledgeType.Tightness: self.tightness,
            KnowledgeType.HingeAngle: self.hinge_line,
        }

    @beartype.beartype
    def get(self, input_knowledge: KnowledgeType):
        return self.knowledge_map[input_knowledge]
