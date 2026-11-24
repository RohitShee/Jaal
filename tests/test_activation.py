import numpy as np

from jaal.activations import ReLU,Sigmoid


def test_relu_forward_by_hand():
    x = np.array([[1.0, -2.0, 3.0], [-1.0, 0.5, -0.5]])
    out = ReLU()(x)
    assert np.array_equal(out, [[1.0, 0.0, 3.0], [0.0, 0.5, 0.0]])


def test_relu_backward_by_hand_with_mixed_signs():
    # mixed-sign grad_output catches a gate that depends on grad_output
    # instead of the saved input (the precedence bug)
    x = np.array([[1.0, -2.0, 3.0], [-1.0, 0.5, -0.5]])
    g = np.array([[2.0, -3.0, -4.0], [5.0, -6.0, 7.0]])
    layer = ReLU()
    layer(x)
    dx = layer.backward(g)
    assert np.array_equal(dx, [[2.0, 0.0, -4.0], [0.0, -6.0, 0.0]])

def test_sigmoid_by_hand():
    layer = Sigmoid()
    out = layer(np.array([[0.0, 2.0]]))
    assert np.allclose(out, [[0.5, 0.8808]], atol=1e-4)

    dx = layer.backward(np.array([[2.0, -3.0]]))   # mixed signs on purpose
    assert np.allclose(dx, [[0.5, -0.3150]], atol=1e-4)


def test_sigmoid_no_overflow():
    out = Sigmoid()(np.array([[-1000.0, 1000.0]]))
    assert np.allclose(out, [[0.0, 1.0]])


