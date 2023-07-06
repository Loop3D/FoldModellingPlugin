
# create a class that checks the following things:
# Check Foliation Data format
# Check Knowledge dict format
# Check bounding box

class CheckInputData:

    """
    Check the input data for the optimisation.
    """

    def __init__(self, folded_foliation_data=None,
                 bounding_box=None,
                 knowledge_constraints=None):

        self.folded_foliation_data = folded_foliation_data
        self.bounding_box = bounding_box
        self.knowledge_constraints = knowledge_constraints

    def check_foliation_data(self):
        pass

    def check_knowledge_constraints(self):
        """"
        Check the knowledge constraints dictionary format
        """"

    def check_bounding_box(self):
        pass