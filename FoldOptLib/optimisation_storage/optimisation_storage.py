import pickle


class OptimisationStorage:
    def __init__(self, optimisation_name, optimisation_type, optimisation_parameters, optimisation_results):
        self.optimisation_name = optimisation_name
        self.optimisation_type = optimisation_type
        self.optimisation_parameters = optimisation_parameters
        self.optimisation_results = optimisation_results

    def save_optimisation(self, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load_optimisation(file_path):
        with open(file_path, 'rb') as file:
            return pickle.load(file)