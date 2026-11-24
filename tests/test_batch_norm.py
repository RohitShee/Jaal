import numpy as np
from jaal.layers import BatchNorm
from jaal.containers import Sequential
from jaal.layers import Linear
from jaal.activations import ReLU

def test_batchnorm_running_stats_update_in_train():
    layer = BatchNorm(2, momentum=0.1)
    x = np.array([[1., 100.], [2., 200.], [3., 300.], [4., 400.]])
    layer.train()
    layer(x)
    # one update from the initial mean 0: 0.1 * batch_mean
    assert np.allclose(layer.running_mean, 0.1 * x.mean(axis=0))


def test_batchnorm_eval_uses_running_stats():
    rng = np.random.default_rng(0)
    layer = BatchNorm(3, momentum=0.1)
    x = rng.normal(5.0, 2.0, size=(64, 3))

    layer.train()
    for _ in range(200):                  # let the running stats converge
        layer(x)

    layer.eval()
    out = layer(x)
    assert np.allclose(out.mean(axis=0), 0, atol=0.05)
    assert np.allclose(out.std(axis=0), 1, atol=0.05)

    single = layer(x[:1])                 # one sample works in eval
    assert np.isfinite(single).all()


def test_batchnorm_eval_does_not_update():
    layer = BatchNorm(2)
    x = np.random.default_rng(0).normal(size=(8, 2))
    layer.eval()
    before = layer.running_mean.copy()
    layer(x)
    assert np.array_equal(layer.running_mean, before)



