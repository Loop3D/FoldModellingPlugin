
AxialSurfaceOptimiser User Guide
================================

Introduction
------------
This guide provides detailed information and examples on how to use the `AxialSurfaceOptimiser` class from the FoldOptLib library. This class is designed for geological optimization tasks, focusing on axial surface optimization.

Installation
------------
Before using `AxialSurfaceOptimiser`, ensure that the FoldOptLib library is installed in your Python environment.

Step-by-Step Usage
------------------

1. **Importing the Class**

   .. code-block:: python

       from foldoptlib.axial_surface_optimiser import AxialSurfaceOptimiser

2. **Initializing the Optimizer**

   Create an instance of `AxialSurfaceOptimiser` with your data, a bounding box, and optional geological knowledge.

   .. code-block:: python

       optimiser = AxialSurfaceOptimiser(data, bounding_box, geological_knowledge=None)

3. **Generating Initial Guess**

   Generate an initial guess for the optimization process.

   .. code-block:: python

       initial_guess = optimiser.generate_initial_guess()

4. **Setting Up Optimisation**

   Set up the optimization parameters and objective functions.

   .. code-block:: python

       optimiser.setup_optimisation(geological_knowledge)

5. **Running the Optimisation**

   Execute the optimization process and retrieve the results.

   .. code-block:: python

       results = optimiser.optimise(geological_knowledge)

Example
-------
A simple example demonstrating how to use `AxialSurfaceOptimiser`.

.. code-block:: python

   # Sample data and parameters
   data = ...  # Your DataFrame with geological data
   bounding_box = [...]  # Define the bounding box
   geological_knowledge = {...}  # Optional geological knowledge

   # Initialize and set up the optimiser
   optimiser = AxialSurfaceOptimiser(data, bounding_box, geological_knowledge)
   optimiser.setup_optimisation(geological_knowledge)

   # Run optimisation and get results
   results = optimiser.optimise()

Conclusion
----------
`AxialSurfaceOptimiser` offers a robust and flexible way to perform geological optimization, especially useful in the field of geology and earth sciences.
