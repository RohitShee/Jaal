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
