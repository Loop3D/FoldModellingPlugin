import numpy as np
from fold_modelling_plugin.input.input_data_checker import CheckInputData

class FoldOptimiser:
    """
    Base class for fold geometry optimisation
    """

    def __init__(self, folded_foliation_data, bounding_box,
                 opt_type,
                 knowledge_constraints=None):
        """
                Constructs all the necessary attributes for the FoldOptimiser object.

                Parameters
                ----------
                    folded_foliation_data : pd.DataFrame
                        The data related to folded foliation
                    bounding_box : nd.array
                        The bounding box data
                    opt_type : str
                        The type of optimisation to be performed, can be 'fourier' or 'axial_surface'
                    knowledge_constraints : dict, optional
                        The knowledge constraints data (default is None)
        """

        self.folded_foliation_data = folded_foliation_data
        self.bounding_box = bounding_box
        self.opt_type = opt_type
        self.knowledge_constraints = knowledge_constraints

    def prepare_knowledge_constraints(self):
        """
        Prepare the knowledge constraints data
        """
        pass

    def setup_fourier_optimisation(self):
        """
        Setup the optimisation problem for the fourier series
        """
        pass

    def setup_axial_surface_optimisation(self):
        """
        Setup the optimisation problem for the axial surface
        """
        pass

    def setup_optimisation(self):
        """
        Setup the optimisation problem
        """
        if self.opt_type == 'fourier':
            self.setup_fourier_optimisation()
        elif self.opt_type == 'axial_surface':
            self.setup_axial_surface_optimisation()
        else:
            raise ValueError('Optimisation type not recognised, '
                             'please choose between fourier and axial_surface')

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
