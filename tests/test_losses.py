import numpy as np

from jaal.activations import ReLU
from jaal.containers import Sequential
from jaal.layers import Linear
from jaal.losses import MSELoss,BCELoss


def test_mse_by_hand():
    loss_fn = MSELoss()
    pred = np.array([[2.0, 4.0]])
    target = np.array([[1.0, 1.0]])
    assert np.isclose(loss_fn(pred, target), 5.0)
    assert np.allclose(loss_fn.backward(), [[1.0, 3.0]])


def test_mse_batch_and_outputs():
    # batch > 1 and out > 1, so len(pred) and pred.size differ
    loss_fn = MSELoss()
    pred = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # N = 6
    target = np.zeros((3, 2))

    loss_fn(pred, target)
    assert np.allclose(loss_fn.backward(), (2 / 6) * pred)


def test_network_gradients_with_mse():
    rng = np.random.default_rng(0)
    model = Sequential(Linear(2, 3), ReLU(), Linear(3, 2))
    loss_fn = MSELoss()

    x = rng.normal(size=(5, 2))
    target = rng.normal(size=(5, 2))
    # analytic gradients first (the numeric loop below overwrites cached values)
    model.zero_grad()
    loss_fn(model(x), target)
    model.backward(loss_fn.backward())

    def loss():
        return loss_fn(model(x), target)

    eps = 1e-6
    for p in model.parameters():
        numeric = np.zeros_like(p.data)
        for i in np.ndindex(p.data.shape):
            old = p.data[i]
            p.data[i] = old + eps
            plus = loss()
            p.data[i] = old - eps
            minus = loss()
            p.data[i] = old
            numeric[i] = (plus - minus) / (2 * eps)
        assert np.allclose(p.grad, numeric, atol=1e-5)
def test_bce_by_hand():
    loss_fn = BCELoss()
    pred   = np.array([[0.9, 0.2]])
    target = np.array([[1.0, 0.0]])
    assert np.isclose(loss_fn(pred, target), 0.16425, atol=1e-4)
    assert np.allclose(loss_fn.backward(), [[-0.5556, 0.6250]], atol=1e-4)

def test_bce_confidently_wrong():
    loss = BCELoss()(np.array([[0.0, 1.0]]), np.array([[1.0, 0.0]]))
    assert np.isfinite(loss) and loss > 10


test_mse_by_hand()
test_mse_batch_and_outputs()
test_network_gradients_with_mse()
test_bce_by_hand()
test_bce_confidently_wrong()
