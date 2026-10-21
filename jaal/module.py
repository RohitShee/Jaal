import numpy as np

class Parameter:
    def __init__(self,data):
        self.data = np.asarray(data,dtype=float)
        self.grad = np.zeros_like(self.data)


class Module:
    def __init__(self):
        self.training = True

    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError

    def __call__(self, x):
        return self.forward(x)

    def parameters(self):
        return []

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.grad.fill(0.0)
