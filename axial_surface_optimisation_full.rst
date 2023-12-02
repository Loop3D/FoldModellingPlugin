Fold modelling - Axial surface optimisation tutorial Part I



----

Overview

This tutorial demonstrates the use of a newly developed computational method for approximating fold axial surfaces. This section integrates algorithms proposed by Laurent et al. (2016) `Laurent et al. (2016) <https://linkinghub.elsevier.com/retrieve/pii/S0012821X16305209>`_ with a data-driven approach from `Grose et al. (2017) <https://gmd.copernicus.org/articles/14/3915/2021/>`_ and the geological likelihood functions of `Grose et al. (2019) <https://linkinghub.elsevier.com/retrieve/pii/S0191814118303638>`_ to address the challenge of estimating planar fold axial surfaces in poorly outcropping regions.


----

Solving the Axial Surface Optimization Problem

The axial surface optimisation problem is addressed through a multi-step computational algorithm that involves:

1. **Maximum Likelihood Estimation (MLE)**: The method uses MLE to estimate model parameters that define the axial surface: strike and dip or a 3D unit vector. MLE finds the most probable parameters (like the 3D unit vector representing the axial surface) by maximising a likelihood function based on the observed data and/or geological knowledge.

2. **Incorporating Geological Knowledge**: A key aspect of this approach is the inclusion of geological knowledge constraints, such as fold tightness, asymmetry, and axial traces. These constraints are used to guide the optimisation process, ensuring that the resulting model aligns with observations and Structural Geological concepts.

3. **Optimisation Techniques**: The method leverages optimisation techniques like the Differential Evolution algorithm, which helps in navigating complex parameter spaces and ensures a robust search for the optimal axial surface.

4. **Objective Function the relationship between the folded foliation and axial surface**: An objective function is used to evaluate the geometrical relationship between the axial surface and the folded surface, ensuring the geometrical compatibility between the folded foliation and its axial surface.

5. **Final Solution**: The result of this method is a data-driven and geologically informed model of the axial surface.

========

----

Pre-requisites: 
- **`LoopStructural-Fold Frame tutorial <https://github.com/Loop3D/LoopStructural/blob/master/examples/2_fold/plot_adding_folds_to_surfaces.py>`_**
- **`Fourier series optimisation tutorial <https://github.com/Loop3D/FoldOptLib/blob/main/FoldOptLib/examples/fourier_series_optimisation.ipynb>`_**


.. code-block:: python

    from misc_functions import sample_random_dataset
    from FoldOptLib.FoldOptLib.optimisers import AxialSurfaceOptimiser as ASO
    from FoldOptLib.FoldOptLib.optimisers import FourierSeriesOptimiser as FSO
    from FoldOptLib.FoldOptLib.helper.utils import create_fold_frame_dataset, create_dataset, clean_knowledge_dict
    from LoopStructural import GeologicalModel
    from LoopStructural.modelling.features.fold import FoldEvent
    # from LoopStructural.visualisation import LavaVuModelViewer
    # from custom_model_visualisation import LavaVuModelViewer
    from LoopStructural.visualisation import MapView
    from LoopStructural.visualisation import RotationAnglePlotter
    from LoopStructural.modelling.features.fold import FoldRotationAngle, SVariogram
    from LoopStructural.modelling.features import StructuralFrame, GeologicalFeature
    from LoopStructural.modelling.features.fold import fourier_series
    from LoopStructural.utils.helper import create_surface, create_box, get_data_bounding_box, normal_vector_to_strike_and_dip, plunge_and_plunge_dir_to_vector
    import os
    import time
    import numpy as np
    import pandas as pd
    from scipy.optimize import least_squares
    import seaborn as sns 
    import matplotlib
    import matplotlib.pyplot as plt
    %matplotlib inline 
    from sklearn.neighbors import NearestNeighbors
    from IPython.display import display, clear_output
    import ipywidgets as widgets
    from ipywidgets import IntProgress, interact, interactive


========
I. Creation of a synthetic model  

Like the Fourier series tutorial, we will start by creating a synthetic model. The synthetic model is the reference from within which we will sample data points that will be used to infer the axial surface.  

1. Define model bounding box

.. code-block:: python

    # define the maximum value of xyz model box boundaries
    xmin, ymin, zmin = 0, 0, 0
    xmax, ymax, zmax = 1000, 1000, 100
    
    bounding_box = np.array([[xmin, ymin, zmin],
                   [xmax, ymax, zmax]])

2. Initialise a geological model 

.. code-block:: python

    # initiliase geological model by initialising GeologicalModel class
    model = GeologicalModel(bounding_box[0, :], 
                            bounding_box[1, :])

3. Build a fold frame

.. code-block:: python

    # Create a dataset for s1 to build a fold frame 
    dataset, xyz = create_fold_frame_dataset(model, strike=0, dip=90)
    # add data to the initiliased geological model
    model.data = dataset

.. code-block:: python

    
    # build the s1 fold frame
    s1 = model.create_and_add_fold_frame('s1',
                                              buffer=0.3,
                                              solver='cg',
                                              nelements=360,
                                            damp=True)
    model.update()
    %timeit

.. code-block:: python

    # # 3D displaying the s1 fold frame
    # # import lavavu
    # viewer = LavaVuModelViewer(model, background='white')
    # viewer.add_isosurface(s1[0], colour='red')
    # viewer.add_isosurface(s1[1], colour='blue')
    # viewer.add_data(s1[0], disks=False, vectors=True, colour='red')
    # viewer.add_data(s1[1], disks=False, vectors=True, colour='blue')
    # # t = viewer.add_scalar_field(s1[0], cmap='prism')
    # # viewer.lv.colourbar(t, align=("bottom"))
    # viewer.lv.rotate([-63.015506744384766, -24.475210189819336, -8.501092910766602])
    # viewer.display()
    # viewer.interactive()

.. code-block:: python

    grid = model.regular_grid([10,10,10])
    s1g = s1[0].evaluate_gradient(grid)
    s1g /= np.linalg.norm(s1g, axis=1)[:, None]
    s1g

.. code-block:: python

    dataset

4. Define the fold limb rotation angle profile

.. code-block:: python

    def fold_limb_rotation_profile(c0, c1, c2, wl):
    
        theta = [c0, c1, c2, wl]
        x = np.linspace(s1[0].min(), s1[0].max(), 100)
        flr = np.rad2deg(np.arctan(fourier_series(x, *theta)))
        fold_limb_rotation = FoldRotationAngle(flr, x) 
        fold_limb_rotation.fitted_params = theta
        fold_limb_rotation.set_function(lambda x: np.rad2deg(
                        np.arctan(fourier_series(x, *theta))))
        plt.ylim(-90, 90)
        plt.xlabel('Fold Axial Surface Field')
        plt.ylabel('Fold Limb Rotation Angle')
        plt.title('Fold Limb S-Plot')
        plt.plot(x, flr)
        plt.show()
        
        return fold_limb_rotation
    
    def define_fold_axis_orientation(plunge_direction, plunge): 
        
        fold_axis = plunge_and_plunge_dir_to_vector(plunge, plunge_direction)
        
        return fold_axis

.. code-block:: python

    theta = [0, 2e-2, 2e-2, 500]
    fold_limb_rotation = fold_limb_rotation_profile(*theta)

**5. Calculate the normal vectors to the folded foliation**

The next lines of code define the orientation of the folded foliation within the fold frame. 
Let's break down the steps to understand the computations involved in this code snippet.

========
#
----

**- Defining Fold Axis Orientation**
Initially, the code defines a fold axis orientation based on a given plunge direction and plunge angle. The `define_fold_axis_orientation` function is used for this purpose, which returns the fold axis orientation vector.

========
#
----

**- Creating a Fold Event**
A `FoldEvent` object is created, representing a geological fold event. This object is initialised with the fold frame `s1` and the `fold_limb_rotation`. Later, the fold axis orientation vector `fold_axis` is assigned to this `FoldEvent` object.

========
#
----

**- Computing Deformed Orientation**
The `get_deformed_orientation` method is invoked on the `FoldEvent` object to compute the deformed orientation of the folded foliation. This method returns three values: the fold direction vectors, the fold axis vector, and the gradient of the scalar field of the X-axis of the fold frame.

========
#
----

**- Normalizing Vectors**
The gradient of S1, `dgx` and the fold direction vectors are normalized to ensure they have unit lengths.

========
#
----

**- Correct fold direction vectors with S1**
The dot product between `dgx` and the fold direction vectors is calculated to align any inverted fold direction vectors with the axial surface's direction to ensure consistency. If the dot product is negative, the fold direction vectors are inverted.

========
#
----

**- Computing Normal Vectors**
The normal vectors of the folded foliation are computed using the cross-product of the fold axis and the fold direction vectors. The cross-product yields a vector that is perpendicular to the plane defined by the fold axis and fold direction vectors.

.. code-block:: python

    plunge_direction = 0
    plunge = 0
    fold_axis = define_fold_axis_orientation(plunge_direction, plunge)
    fold = FoldEvent(s1, fold_limb_rotation=fold_limb_rotation)
    fold.fold_axis = fold_axis 
    fold_direction, fold_axis, dgz = fold.get_deformed_orientation(xyz)
    dgx = s1[0].evaluate_gradient(xyz)
    dgx /= np.linalg.norm(dgx, axis=1)[:, None]
    # make sure fold direction vectors are normalised
    fold_direction /= np.linalg.norm(fold_direction, axis=1)[:, None]
    # calculate the dot product of the s1 and the fold direction
    dot = np.einsum('ij,ij->i', dgx, fold_direction)
    # correct the orientation of the fold direction vectors to be consistent
    # with the direction of the axial surface 
    fold_direction[dot<0] *= -1
    # calculate the normal vectors of the folded foliation 
    # which are the cross product of the fold axis and 
    # the fold direction vectors
    s0n = np.cross(fold_axis, fold_direction)
    # normalise s0 normal vectors
    s0n /= np.linalg.norm(s0n, axis=1)[:, None]

6. Create a dataset for s0

.. code-block:: python

    dataset = pd.DataFrame()
    dataset['X'] = xyz[:, 0]
    dataset['Y'] = xyz[:, 1]
    dataset['Z'] = xyz[:, 2]
    dataset['gx'] = s0n[:, 0]
    dataset['gy'] = s0n[:, 1]
    dataset['gz'] = s0n[:, 2]
    dataset['feature_name'] = 's0'
    dataset['coord'] = 0

7. Build a 3D model of s0

.. code-block:: python

    fold_function = fold_limb_rotation.fold_rotation_function
    model.data = dataset.sample(frac=0.2)
    s0 = model.create_and_add_folded_foliation('s0',
                                               fold_frame=s1,
                                                # limb_wl=500,
                                                av_fold_axis=True,
                                                limb_function=fold_function,
                                                nelements=3e5,
                                                solver='cg',
                                                buffer=0.3,
                                                damp=True)
    model.update()
    # s0.fold.fold_limb_rotation.fitted_params = theta

- Check the fold limb rotation angle if it is correct

.. code-block:: python

    s0.set_model(model)
    plotter = RotationAnglePlotter(s0)
    plotter.default_titles()
    plotter.add_fold_limb_data()
    plotter.add_fold_limb_curve()
    # plotter.add_limb_svariogram()
    plt.show()

- Visualise the 3D model of s0

.. code-block:: python

    # viewer = LavaVuModelViewer(model, background='white')
    # viewer.nsteps = np.array([100, 100, 100])
    # t = viewer.add_scalar_field(s0, cmap='prism')
    # # viewer.lv.colourbar(t, align=("bottom"))
    # viewer.lv.rotate([-63.015506744384766, -24.475210189819336, -8.501092910766602])
    # viewer.display()
    # viewer.interactive()


========
II. Axial surface optimisation

- **write a brief intro of what will happen in this section**

##
----

**1. Sampling S<sub>0</sub> from the reference model**  

Now, we sample a random dataset from the reference model we just built. 


.. code-block:: python

    
    def update(sample_size):
        global points, s0g
        points = sample_random_dataset(xyz, sample_size=sample_size, seed=180)
        # Evaluate the gradient of the folded foliation using points 
        s0g = s0.evaluate_gradient(points) 
        # normalise the gradient 
        s0g /= np.linalg.norm(s0g, axis=1)[:, None] 
        
    # Create a slider for sample_size
    sample_size_slider = widgets.IntSlider(
        value=10,
        min=1,
        max=100,
        step=1,
        description='Sample Size:',
        continuous_update=True
    )
    
    # Use interact to create the interactive widget
    interactive_sampling = interactive(update, sample_size=sample_size_slider)
    
    
    display(interactive_sampling)
    

Now, we create a dataset called `test_data` using the random sample. The dataset should be a *Pandas Dataframe*.

.. code-block:: python

    # Create a dataframe
    test_data = pd.DataFrame()
    test_data['X'] = points[:, 0]
    test_data['Y'] = points[:, 1]
    test_data['Z'] = points[:, 2]
    test_data['gx'] = s0g[:, 0]
    test_data['gy'] = s0g[:, 1]
    test_data['gz'] = s0g[:, 2]
    test_data['feature_name'] = 's0'
    test_data['coord'] = 0

At this stage, we will use the dataset we built in the previous cell to find the orientation of the axial that is compatible with the folded foliation `s0`. To find the axial surface, we use the `AxialSurfaceOptimiser` class that contains the functions necessary to run an axial surface optimisation. The axial surface optimisation algorithm proceeds as follows:  
1. Define geological knowledge constraints if available
2. Build the fold frame
3. Calculate the fold rotation angles
4. Fit Fourier series parameters to the fold rotation angles 
5. Calculate the orientation of the predicted bedding 
6. Calculate the angle between the observed and the predicted bedding 
7. Optimise the normal log-likelihood function that finds the optimal axial surface

The `AxialSurfaceOptimiser` class is designed to optimise the axial surfaces based on the provided data, bounding box, and geological knowledge. To initialise the `AxialSurfaceOptimiser` we use the following parameters: 


---
-`data`: the input data for optimisation which should be a pd.DataFrame  
-`bounding_box`: The bounding box for the optimisation. Which should be a list or a NumPy array.   
-`geological_knowledge`: The geological knowledge used for optimisation, by default None. If used, the input should be a nested python dictionary. See `Fourier series optimisation tutorial <https://github.com/Loop3D/FoldOptLib/blob/main/FoldOptLib/examples/fourier_series_optimisation.ipynb>`_ for the structure of the dictionary.   
-`**kwargs`: Other optional parameters for optimisation.  
* `axial_surface_guess`: an estimate of the axial surface to provide to the algorithm to speed up optimisation. It should be a list in the following format: [strike, dip]  
* `av_fold_axis`: True for cylindrical folds and False for noncylindrical folds. 
It can include SciPy optimisation parameters for differential evolution and trust-constr methods.  
    `mode`: the optimisation mode to use, can be 'restricted' or 'unrestricted', by default 'unrestricted'. only unrestricted mode is supported for now.  
    `method`: the optimisation algorithm to use, can be 'differential_evolution' or 'trust-region',
    by default 'differential_evolution'.  


In the following example, we will use only data (test_data) to find the optimal axial surface. 

.. code-block:: python

    aso = ASO(test_data, 
              bounding_box, 
              method='differential_evolution', 
              axial_surface_guess=[90, 0], 
              av_fold_axis=True
             )

.. code-block:: python

    results = aso.optimise()

.. code-block:: python

    print('Axial surface optimisation results: ')
    print('strike: ', results.x[0])
    print('dip: ', results.x[1])

.. code-block:: python

    

