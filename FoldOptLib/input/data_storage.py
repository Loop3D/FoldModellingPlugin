from dataclasses import dataclass
from ..datatypes import KnowledgeType, DataType, InputGeologicalKnowledge
from ..objective_functions import GeologicalKnowledgeFunctions
from LoopStructural import BoundingBox
import pandas


@dataclass
class InputData:
    folded_axial_surface: pandas.DataFrame
    folded_foliation: pandas.DataFrame
    bounding_box: BoundingBox
    geological_knowledge: InputGeologicalKnowledge = None
