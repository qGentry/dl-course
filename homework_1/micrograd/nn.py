import numpy as np
from abc import ABC, abstractmethod
from engine import Value, Tensor


class Module(ABC):
    """
    Base class for every layer.
    """

    @abstractmethod
    def forward(self, *args, **kwargs):
        """Depends on functionality"""
        pass

    def __call__(self, *args, **kwargs):
        """For convenience we can use model(inp) to call forward pass"""
        return self.forward(*args, **kwargs)

    def parameters(self):
        """Return list of trainable parameters"""
        return []


class Linear(Module):
    def __init__(self, in_features, out_features, bias: bool = True):
        self.w = Tensor(np.random.uniform(-1, 1, size=[out_features, in_features]))
        self.bias = bias
        if self.bias:
            self.b = Tensor(np.random.uniform(-1, 1, size=[out_features]))
        """Initializing model"""
        # Create Linear Module

    def forward(self, inp):
        """Y = W * x + b"""
        result = self.w @ Tensor(inp)
        if self.bias:
            result = self.b + result
        return result

    def parameters(self):
        return [*self.w.parameters(), *self.b.parameters()]


class ReLU(Module):
    """The most simple and popular activation function"""

    def forward(self, inp):
        # Create ReLU Module
        return Tensor(np.where(inp.data > 0, inp.data, Value(0)))


class CrossEntropyLoss(Module):
    """Cross-entropy loss for multi-class classification"""

    def forward(self, inp, label):
        # Create CrossEntropy Loss Module
        return -inp[label] + inp.exp().sum().log()


