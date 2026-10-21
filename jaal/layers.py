import math

import numpy as np

from .module import Module, Parameter


class Linear(Module):
    def __init__(self, in_features, out_features, seed=42):
        super().__init__()
        rng = np.random.default_rng(seed)
        self.W = Parameter(
            rng.normal(0, math.sqrt(2 / in_features), (in_features, out_features))
        )
        self.b = Parameter(np.zeros(out_features))
        self.x = None

    def forward(self, x):
        self.x = np.asarray(x)
        return self.x @ self.W.data + self.b.data

    def backward(self, grad_output):
        if self.x is None:
            raise RuntimeError("Linear.backward called before forward")
        self.W.grad += self.x.T @ grad_output
        self.b.grad += grad_output.sum(axis=0)
        return  grad_output @self.W.data.T

    def parameters(self):
        return [self.W, self.b]
