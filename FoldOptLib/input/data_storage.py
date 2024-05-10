from dataclasses import dataclass, field
from ..datatypes import KnowledgeType, DataType, InputGeologicalKnowledge, CoordinateType, InterpolationConstraints, CoordinateType
from ..objective_functions import GeologicalKnowledgeFunctions
from LoopStructural import BoundingBox
import pandas
import numpy
import beartype
from typing import List

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
    
    @beartype.beartype
    def get_foliations(self, feature_name: str): 
        return self.data[self.data['feature_name'] == feature_name]
    
    @beartype.beartype
    def __getitem__(self, data_type: DataType): 

        data_map = {
            DataType.DATA: self.data,
            DataType.GEOLOGICAL_KNOWLEDGE: self.geological_knowledge,
            DataType.BOUNDING_BOX: self.bounding_box
        }
        

        return data_map[data_type]
    

@beartype.beartype
@dataclass
class OptData:
    """
    Class representing optimisation data.

    Attributes
    ----------
    data : pandas.DataFrame
        The input data.
    constraints : List[InterpolationConstraints]
        The interpolation constraints.

    Methods
    -------
    set_constraints(constraints: InterpolationConstraints, constraint_type: ConstraintType):
        Set the interpolation constraints for a given constraint type.
    axial_normals():
        Get the axial normals from the data.
    y_normals():
        Get the y normals from the data.
    set_axial_surface_field_constraints():
        Set the axial surface field constraints.
    set_fold_axis_field_constraints():
        Set the fold axis field constraints.
    _getitem__(constraint_type: ConstraintType):
        Get the interpolation constraints for a given constraint type.
    """
    data : pandas.DataFrame
    constraints: List[InterpolationConstraints] = field(default_factory=lambda: [None] * len(CoordinateType), init=False)


    def set_constraints(
            self, 
            constraints: InterpolationConstraints, 
            constraint_type: CoordinateType
            ): 
        """
        Set the interpolation constraints for a given constraint type.

        Parameters
        ----------
        constraints : InterpolationConstraints
            The interpolation constraints.
        constraint_type : ConstraintType
            The type of constraint.
        """
        self.constraints[constraint_type] = constraints


    def axial_normals(self): 
        """
        Get the axial normals from the data.

        Returns
        -------
        pandas.DataFrame
            The axial normals.
        """

        return self.data[self.data['feature_name'] == 'sn' and self.data['coord'] == CoordinateType.AXIAL_FOLIATION_FIELD]
    

    def y_normals(self): 
        """
        Get the y normals from the data.

        Returns
        -------
        pandas.DataFrame
            The y normals.
        """
            
        return self.data[self.data['feature_name'] == 'sn' and self.data['coord'] == CoordinateType.FOLD_AXIS_FIELD]


    def set_axial_surface_field_constraints(self): 
        """
        Set the axial surface field constraints.
        """
        try: 
            value_constraints = self.axial_normals[['X', 'Y', 'Z', 'value']].to_numpy()
        except:
            mean_x, mean_y, mean_z = self.axial_normals[['X', 'Y', 'Z']].mean()
            value_constraints = numpy.array([mean_x, mean_y, mean_z, 0.0])

        normal_constraints = self.axial_normals[['X', 'Y', 'Z', 'gx', 'gy', 'gz']].to_numpy()

        ic = InterpolationConstraints(
            value=value_constraints,
            normal_constraints=normal_constraints
            )
        
        self.set_constraints(ic, CoordinateType.AXIAL_SURFACE_FIELD)
      
    
    def set_fold_axis_field_constraints(self): 
        """
        Set the fold axis field constraints.
        """
        try: 
            value = self.y_normals[['X', 'Y', 'Z', 'value']].to_numpy()
        except:
            mean_x, mean_y, mean_z = self.axial_normals[['X', 'Y', 'Z']].mean()
            value = numpy.array([mean_x, mean_y, mean_z, 0.0])

        normal_constraints = self.y_normals[['X', 'Y', 'Z', 'gx', 'gy', 'gz']].to_numpy()

        ic = InterpolationConstraints(
            value=value,
            normal_constraints=normal_constraints
            )
        
        self.set_constraints(ic, CoordinateType.FOLD_AXIS_FIELD)
        

    
    def _getitem__(
            self, 
            constraint_type: CoordinateType
            ): 
        
        """
        Get the interpolation constraints for a given constraint type.

        Parameters
        ----------
        constraint_type : ConstraintType
            The type of constraint.

        Returns
        -------
        InterpolationConstraints
            The interpolation constraints.
        """
        if self.constraints[constraint_type] is None:
            
            if constraint_type is CoordinateType.AXIAL_SURFACE_FIELD:

                self.set_axial_surface_field_constraints()

                return self.constraints[constraint_type]
            
            if constraint_type is CoordinateType.FOLD_AXIS_FIELD:

                self.set_fold_axis_field_constraints()

                return self.constraints[constraint_type]
            
        if self.constraints[constraint_type] is not None:
        
            return self.constraints[constraint_type]
        