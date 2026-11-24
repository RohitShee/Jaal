import math

import numpy as np

from .module import Module, Parameter


class Linear(Module):
    def __init__(self, in_features, out_features, seed=None):
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

class Dropout(Module):
    def __init__(self, p=0.5, seed=None):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        assert 0 <= p < 1
        self.p = p
        self.mask = None

    def forward(self, x):
        x = np.asarray(x)
        if self.training :
            self.mask = (self.rng.random(x.shape) > self.p) /(1-self.p)
            x = x * self.mask
        else :
            self.mask = np.ones(x.shape)
        return x

    def backward(self, grad_output):
        return grad_output * self.mask

class BatchNorm(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum          # weight on the NEW batch (PyTorch convention)
        self.gamma = Parameter(np.ones(num_features))
        self.beta = Parameter(np.zeros(num_features))
        # bookkeeping, NOT Parameters: the optimizer never touches these
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        self.x_hat = None
        self.den = None
        self._used_batch_stats = None

    def forward(self, x):
        x = np.asarray(x, dtype=float)
        if self.training:
            mean = x.mean(axis=0)
            var = x.var(axis=0)
            # update running stats in place (train mode only)
            self.running_mean *= (1 - self.momentum)
            self.running_mean += self.momentum * mean
            self.running_var *= (1 - self.momentum)
            self.running_var += self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var
        self._used_batch_stats = self.training
        self.den = np.sqrt(var + self.eps)
        self.x_hat = (x - mean) / self.den
        return self.gamma.data * self.x_hat + self.beta.data

    def backward(self, grad_output):
        if self.x_hat is None:
            raise RuntimeError("BatchNorm.backward called before forward")
        grad_output = np.asarray(grad_output, dtype=float)

        self.beta.grad += grad_output.sum(axis=0)
        self.gamma.grad += (grad_output * self.x_hat).sum(axis=0)

        dx_hat = grad_output * self.gamma.data
        if not self._used_batch_stats:
            # eval: mean and var are constants, so no extra paths
            return dx_hat / self.den

        # train: gradient also flows through the batch mean and variance
        N = grad_output.shape[0]
        return (N * dx_hat
                - dx_hat.sum(axis=0)
                - self.x_hat * (dx_hat * self.x_hat).sum(axis=0)) / (N * self.den)

    def parameters(self):
        return [self.gamma, self.beta]
