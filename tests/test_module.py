import numpy as np

from jaal.module import Module, Parameter


class Dummy(Module):
    def __init__(self):
        super().__init__()
        self.p = Parameter([[1, 2, 3], [4, 5, 6]])  # note: ints in a list

    def parameters(self):
        return [self.p]


m = Dummy()
assert m.p.data.dtype == float
assert m.p.grad.shape == (2, 3) and m.p.grad.dtype == float

m.p.grad += 5
m.zero_grad()
assert m.p.grad.shape == (2, 3)
assert m.p.grad.sum() == 0

m.eval()
assert m.training is False
