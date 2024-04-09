from abc import ABC, abstractmethod
from base_builder import BaseBuilder


class FoldFrameBuilder(BaseBuilder):

    def set_value_constraints(self):
        pass

    def set_tangent_constraints(self):
        pass

    def set_normal_constraints(self):
        pass

    def set_gradient_constraints(self):
        pass

    def build(self):
        # This method is concrete and should have an implementation.
        # Since the diagram does not specify what it should do, it's just a pass here.
        pass

    def evaluate_scalar_value(self):
        pass

    def evaluate_gradient(self):
        pass
