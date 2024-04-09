from abc import ABC, abstractmethod


class BaseBuilder(ABC):

    @abstractmethod
    def set_value_constraints(self):
        pass

    @abstractmethod
    def set_tangent_constraints(self):
        pass

    @abstractmethod
    def set_normal_constraints(self):
        pass

    @abstractmethod
    def set_gradient_constraints(self):
        pass

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def evaluate_scalar_value(self):
        pass

    @abstractmethod
    def evaluate_gradient(self):
        pass