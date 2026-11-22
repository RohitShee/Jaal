import numpy as np

from jaal.layers import Dropout
from jaal.containers import Sequential
from jaal.layers import Linear
from jaal.activations import ReLU

def test_dropout_forward():
    layer = Dropout(p=0.5, seed=0)
    layer.train()
    out = layer(np.ones((1000, 10)))
    assert set(np.unique(out)) <= {0.0, 2.0}
    assert 0.4 < (out == 0).mean() < 0.6
    assert abs(out.mean() - 1.0) < 0.1

    layer.eval()
    x = np.random.default_rng(1).normal(size=(4, 6))
    assert np.array_equal(layer(x), x)


def test_dropout_p_is_respected():
    layer = Dropout(p=0.2, seed=0)
    layer.train()
    out = layer(np.ones((1000, 10)))
    assert 0.15 < (out == 0).mean() < 0.25


def test_dropout_backward():
    layer = Dropout(p=0.5, seed=0)
    layer.train()
    out = layer(np.ones((5, 8)))
    dx = layer.backward(np.ones((5, 8)))
    assert np.array_equal(dx == 0, out == 0)
    assert np.allclose(dx, out)

    layer.eval()
    layer(np.ones((5, 8)))
    g = np.random.default_rng(2).normal(size=(5, 8))
    assert np.array_equal(layer.backward(g), g)

def test_mode_propagates_to_dropout():
    model = Sequential(Linear(4, 8), ReLU(), Dropout(0.5, seed=0), Linear(8, 1))
    x = np.random.default_rng(0).normal(size=(6, 4))

    model.eval()
    assert np.array_equal(model(x), model(x))       # deterministic in eval

    model.train()
    assert not np.array_equal(model(x), model(x))   # random in train

test_dropout_forward()
test_dropout_p_is_respected()
test_dropout_backward()
test_mode_propagates_to_dropout()
