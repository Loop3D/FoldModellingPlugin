
FourierSeriesOptimiser User Guide
=================================

The `FourierSeriesOptimiser` class is designed for optimizing Fourier series in geological modeling. This guide 
provides detailed explanations of its methods, along with Python examples to demonstrate their usage.

__init__
--------

- **Purpose**: Initializes the Fourier Series Optimiser object with necessary attributes.
- **Parameters**:
    - `fold_frame_coordinate` (float): The fold frame coordinate for the optimiser.
    - `rotation_angle` (float): The rotation angle for the optimiser.
    - `geological_knowledge` (dict, optional): Knowledge constraints for the optimiser.
    - `x` (np.ndarray): Interpolated fold frame coordinate (z or y).
    - `**kwargs` (dict): Additional keyword arguments.
- **Example**:

  .. code-block:: python

    from fourier_optimiser import FourierSeriesOptimiser

    optimiser = FourierSeriesOptimiser(fold_frame_coordinate=0.5,
                                       rotation_angle=30,
                                       geological_knowledge={'key': 'value'},
                                       x=np.linspace(0, 10, 100))

generate_initial_guess
-----------------------

- **Purpose**: Generates an initial guess or bounds for the Fourier series optimization.
- **Returns**: np.ndarray or Any - Initial guess or bounds for the optimization.
- **Example**:

  .. code-block:: python

    initial_guess = optimiser.generate_initial_guess()

setup_optimisation
------------------

- **Purpose**: Sets up the optimization process for the Fourier series.
- **Returns**: tuple - Contains the objective function, geological knowledge, solver, and initial guess.
- **Example**:

  .. code-block:: python

    setup_data = optimiser.setup_optimisation()

optimise
--------

- **Purpose**: Executes the optimisation of the Fourier series.
- **Returns**: Dict[str, Any] - Result of the optimisation.
- **Example**:

  .. code-block:: python

    optimisation_result = optimiser.optimise()

Conclusion
----------

This guide provides a basic overview of the `FourierSeriesOptimiser` class and its methods. Users can utilize 
this guide to understand and implement Fourier series optimization in their geological modeling projects.
