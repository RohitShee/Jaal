import numpy as np

from jaal.activations import ReLU

x = np.array([[1.0, -2.0, 3.0], [-1.0, 0.5, -0.5]])
g = np.ones_like(x)

layer = ReLU()
print(layer(x))  # expect [[1, 0, 3], [0, 0.5, 0]]
print(layer.backward(g))  # expect [[1, 0, 1], [0, 1, 0]]
