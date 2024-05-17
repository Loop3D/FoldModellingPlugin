import pandas as pd
import numpy as np
from LoopStructural import BoundingBox
from FoldOptLib.datatypes import InputGeologicalKnowledge, DataType, CoordinateType, InterpolationConstraints
from FoldOptLib.input.data_storage import InputData, OptData

def test_InputData():
    data = pd.DataFrame({'feature_name': ['a', 'b', 'a', 'c', 'b', 'c', 'a']})
    bounding_box = BoundingBox([0, 0, 0], [1, 1, 1])
    input_data = InputData(data, bounding_box)

    assert np.array_equal(input_data.foliations(), np.array(['a', 'b', 'c']))
    assert input_data.number_of_foliations() == 3
    assert input_data.get_foliations('a').equals(data[data['feature_name'] == 'a'])
    assert input_data[DataType.DATA].equals(data)
    assert input_data[DataType.BOUNDING_BOX] == bounding_box

def test_OptData():
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
        'weight': [1, 1, 1, 1]
    })
    opt_data = OptData(data)
    
    assert opt_data[CoordinateType.AXIAL_FOLIATION_FIELD] is not None
    assert opt_data[CoordinateType.FOLD_AXIS_FIELD] is not None
    assert isinstance(opt_data[CoordinateType.AXIAL_FOLIATION_FIELD], InterpolationConstraints) 
    assert isinstance(opt_data[CoordinateType.FOLD_AXIS_FIELD], InterpolationConstraints)
