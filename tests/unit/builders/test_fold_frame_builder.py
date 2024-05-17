import pytest
import pandas as pd
import numpy as np
from FoldOptLib.builders.fold_frame_builder import FoldFrameBuilder
from FoldOptLib.input import OptData
from FoldOptLib.datatypes import CoordinateType
from LoopStructural import BoundingBox


@pytest.fixture
def fold_frame_builder(): 

    data = pd.DataFrame({
        'feature_name': ['sn', 'sn', 'sn', 'sn'],
        'coord': [
            CoordinateType.AXIAL_FOLIATION_FIELD, 
            CoordinateType.AXIAL_FOLIATION_FIELD,
            CoordinateType.FOLD_AXIS_FIELD, 
            CoordinateType.FOLD_AXIS_FIELD
                  ],
        'X': [0, 1, 2, 3],
        'Y': [0, 1, 2, 3],
        'Z': [0, 1, 2, 3],
        'value': [0, 1, 2, 3],
        'gx': [0, 1, 2, 3],
        'gy': [0, 1, 2, 3],
        'gz': [0, 1, 2, 3], 
        'weight': [1.0, 1.0, 1.0, 1.0]
    })
    constraints = OptData(data)
    bounding_box = BoundingBox(origin=[0, 0, 0], maximum=[10, 10, 10])
    fold_frame_builder = FoldFrameBuilder(constraints, bounding_box)

    return fold_frame_builder
    

def test_fold_frame_builder_initialization(fold_frame_builder):
    data = pd.DataFrame({
        'feature_name': ['sn', 'sn', 'sn', 'sn'],
        'coord': [
            CoordinateType.AXIAL_FOLIATION_FIELD, 
            CoordinateType.AXIAL_FOLIATION_FIELD,
            CoordinateType.FOLD_AXIS_FIELD, 
            CoordinateType.FOLD_AXIS_FIELD
                  ],
        'X': [0, 1, 2, 3],
        'Y': [0, 1, 2, 3],
        'Z': [0, 1, 2, 3],
        'value': [0, 1, 2, 3],
        'gx': [0, 1, 2, 3],
        'gy': [0, 1, 2, 3],
        'gz': [0, 1, 2, 3], 
        'weight': [1.0, 1.0, 1.0, 1.0]
    })
    constraints = OptData(data)

    assert isinstance(fold_frame_builder.constraints, OptData)
    assert np.array_equal(fold_frame_builder.xyz, constraints.data[['X', 'Y', 'Z']].to_numpy())
    assert fold_frame_builder.fold_frame == [None] * len(CoordinateType)
    assert fold_frame_builder.constraints[CoordinateType.AXIAL_FOLIATION_FIELD] is not None
    assert fold_frame_builder.constraints[CoordinateType.FOLD_AXIS_FIELD] is not None

def test_build_axial_surface_field(fold_frame_builder):
    
    fold_frame_builder.build_axial_surface_field()
    assert fold_frame_builder.fold_frame[CoordinateType.AXIAL_FOLIATION_FIELD] is not None
    

def test_build_fold_axis_field(fold_frame_builder):
   
    fold_frame_builder.build_fold_axis_field()
    assert fold_frame_builder.fold_frame[CoordinateType.FOLD_AXIS_FIELD] is not None
    

def test_build(fold_frame_builder):
    
    fold_frame_builder.build()
    assert fold_frame_builder.fold_frame[CoordinateType.AXIAL_FOLIATION_FIELD] is not None
    assert fold_frame_builder.fold_frame[CoordinateType.FOLD_AXIS_FIELD] is not None
    
    