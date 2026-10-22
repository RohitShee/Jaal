import numpy as np

from jaal.activations import ReLU,Sigmoid

x = np.array([[1.0, -2.0, 3.0], [-1.0, 0.5, -0.5]])
g = np.ones_like(x)

layer = ReLU()
print(layer(x))  # expect [[1, 0, 3], [0, 0.5, 0]]
print(layer.backward(g))  # expect [[1, 0, 1], [0, 1, 0]]

def test_sigmoid_by_hand():
    layer = Sigmoid()
    out = layer(np.array([[0.0, 2.0]]))
    assert np.allclose(out, [[0.5, 0.8808]], atol=1e-4)

    dx = layer.backward(np.array([[2.0, -3.0]]))   # mixed signs on purpose
    assert np.allclose(dx, [[0.5, -0.3150]], atol=1e-4)


def test_sigmoid_no_overflow():
    out = Sigmoid()(np.array([[-1000.0, 1000.0]]))
    assert np.allclose(out, [[0.0, 1.0]])

test_sigmoid_by_hand()
test_sigmoid_no_overflow()
