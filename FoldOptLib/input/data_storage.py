from dataclasses import dataclass
from ..datatypes import KnowledgeType, DataType, InputGeologicalKnowledge
from ..objective_functions import GeologicalKnowledgeFunctions
from LoopStructural import BoundingBox
import pandas
import numpy
import beartype

@dataclass
class InputData:
    data: pandas.DataFrame
    bounding_box: BoundingBox
    geological_knowledge: InputGeologicalKnowledge = None


    def __post_init__(self):
        pass 

    def foliations(self): 
        return numpy.unique(self.data['feature_name'])
    
    def number_of_foliations(self): 
        return len(self.foliations())
    
    def get_foliations(self, feature_name): 
        return self.data[self.data['feature_name'] == feature_name]
    
    
    def __getitem__(self, data_type: DataType): 

        data_map = {
            DataType.DATA: self.data,
            DataType.GEOLOGICAL_KNOWLEDGE: self.geological_knowledge,
            DataType.BOUNDING_BOX: self.bounding_box
        }
        

        return data_map[data_type]

