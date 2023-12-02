
Detailed Description of Data Formats for InputDataChecker
=========================================================

The `InputDataChecker` class in FoldOptLib requires specific data formats for foliation data, geological knowledge, and the bounding box. This document provides a detailed description of these formats.

1. Foliation Data Format
------------------------

- **Description**: Foliation data must be provided as a pandas DataFrame.

- **Required Columns**:
  - `X`, `Y`, `Z`: Numeric columns representing spatial coordinates.
  - `feature_name`: A column containing the names of geological features.
  - `strike`, `dip`, or `gx`, `gy`, `gz`: Columns representing geological measurements.

- **Example DataFrame**:

.. code-block:: python

    import pandas as pd

    data = {
        'X': [x1, x2, x3],  # Replace with actual coordinates
        'Y': [y1, y2, y3],
        'Z': [z1, z2, z3],
        'feature_name': ['feature1', 'feature2', 'feature3'],
        'strike': [s1, s2, s3],
        'dip': [d1, d2, d3]
    }
    foliation_data = pd.DataFrame(data)

2. Geological Knowledge Format
-------------------------------

- **Description**: Geological knowledge should be provided as a nested dictionary.

- **Constraints**:
  - Each key-value pair represents a specific geological feature and its attributes.
  - The format and the specific keys required are subject to the needs of the optimization model.

- **Example Dictionary**:

.. code-block:: python

    geological_knowledge = {
        'feature1': {
            'attribute1': value1,
            'attribute2': value2
        },
        'feature2': {
            'attribute1': value1,
            'attribute2': value2
        }
    }

3. Bounding Box Format
----------------------

- **Description**: The bounding box must be a numpy array.

- **Format**:
  - 2x3 array: `[[minX, minY, minZ], [maxX, maxY, maxZ]]`
  - Represents the minimum and maximum coordinates of the modeling area.

- **Example Array**:

.. code-block:: python

    import numpy as np

    bounding_box = np.array([[minX, minY, minZ], [maxX, maxY, maxZ]])
