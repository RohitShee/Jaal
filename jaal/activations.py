import numpy as np

from .module import Module


class ReLU(Module):
    def __init__(self):
        super().__init__()
        self.x = None

    def forward(self, x):
        self.x = np.asarray(x)
        return np.maximum(self.x, 0)

    def backward(self, grad_output):
        if self.x is None:
            raise RuntimeError("Linear.backward called before forward")
        return grad_output * (self.x > 0)
