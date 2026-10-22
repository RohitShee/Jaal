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
            raise RuntimeError("ReLU.backward called before forward")
        return grad_output * (self.x > 0)

class Sigmoid(Module):
    def __init__(self):
        super().__init__()
        self.y = None

    def forward(self, x):
        x = np.asarray(x, dtype=float)
        self.y = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        return self.y

    def backward(self, grad_output):
        if self.y is None:
            raise RuntimeError("Sigmoid.backward called before forward")
        return grad_output * self.y * (1 - self.y)
