import numpy as np

from jaal.layers import Linear

layer = Linear(2, 3)
layer.W.data[:] = [[1, 0, -1], [0, 1, 1]]
out = layer.forward(np.array([[1, 2]]))  # must come first
dx = layer.backward(out)  # now self.x is set
print(out)
print(layer.W.grad)
print(layer.b.grad)
print(dx)

def test_linear_layers_differ_by_default():
    a, b = Linear(4, 4), Linear(4, 4)
    assert not np.array_equal(a.W.data, b.W.data)

def test_linear_seed_is_reproducible():
    a, b = Linear(4, 4, seed=0), Linear(4, 4, seed=0)
    assert np.array_equal(a.W.data, b.W.data)

test_linear_layers_differ_by_default()
test_linear_seed_is_reproducible()
