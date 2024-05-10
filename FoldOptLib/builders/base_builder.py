from abc import ABC, abstractmethod


class BaseBuilder(ABC):

    @abstractmethod
    def set_constraints(self):
        pass

    @abstractmethod
    def evaluate_scalar_value(self, *args):
        pass

    @abstractmethod
    def evaluate_gradient(self, *args):
        pass
