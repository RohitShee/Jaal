import numpy as np

from jaal.module import Parameter
from jaal.optim import SGD


def test_sgd_step_and_zero_grad():
    p = Parameter([1.0, 2.0])
    p.grad += np.array([0.5, -1.0])
    opt = SGD([p], lr=0.1)

    opt.step()
    assert np.allclose(p.data, [0.95, 2.1])
    assert np.allclose(p.grad, [0.5, -1.0])   # step must not touch grads

    opt.zero_grad()
    assert np.allclose(p.grad, [0.0, 0.0])
