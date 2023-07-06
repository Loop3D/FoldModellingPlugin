import numpy as np


class FoldOptimiser:
    """
    Base class for fold geometry optimisation
    """

    def __init__(self, folded_foliation_data, bounding_box,
                 opt_type,
                 knowledge_constraints=None):



        self.folded_foliation_data = folded_foliation_data
        self.bounding_box = bounding_box
        self.opt_type = opt_type
        self.knowledge_constraints = knowledge_constraints

    def setup_optimisation(self):
        """
        Setup the optimisation problem
        """
        pass

    def generate_initial_guess(self):
        """
        Generate an initial guess for the optimisation
        It generates a guess depending on the type of optimisation, if it's fourier series
        it will generate a guess of the wavelength, if it's axial surface it will generate a guess
        using the methods of the Differential Evolution algorithm (Storn and Price, 1997) or uses the
        Von Mises Fisher distribution (Fisher, 1953).
        """

        pass


    def optimise(self):
        """
        Run the optimisation
        """
        pass
