from dataclasses import dataclass, field
from .enums import KnowledgeType
from .probability_distributions import NormalDistribution, VonMisesFisherDistribution
import beartype
from typing import Dict, Union, Optional, List


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
    knowledge_map: Dict[KnowledgeType, Union[NormalDistribution, VonMisesFisherDistribution]] = field(init=False)
    filledflags: List[bool] = field(default_factory=lambda: [False] * len(KnowledgeType))

    def __post_init__(self):
        self.knowledge_map = {
            KnowledgeType.ASYMMETRY: self.asymmetry,
            KnowledgeType.AXIALTRACE: self.axial_trace,
            KnowledgeType.FOLDWAVELENGTH: self.fold_wavelength,
            KnowledgeType.AXISWAVELENGTH: self.axis_wavelength,
            KnowledgeType.AXIALSURFACE: self.axial_surface,
            KnowledgeType.TIGHTNESS: self.tightness,
            KnowledgeType.HINGEANGLE: self.hinge_line,
        }

        for knowledge_type in range(len(KnowledgeType)):

            # initialise dirtyflags and filled flags
            if (
                    self.knowledge_map[knowledge_type] is not None
                    and self.filledflags[knowledge_type] is False
            ):
                self.filledflags[knowledge_type] = True

    def __call__(self, input_knowledge: KnowledgeType):
        return self.knowledge_map[input_knowledge]
