import numpy as np
from jaal.containers import Sequential
from jaal.layers import Linear
from jaal.activations import ReLU

model = Sequential(Linear(2, 3), ReLU(), Linear(3, 1))

# shapes
assert model(np.ones((5, 2))).shape == (5, 1)

# parameters
assert len(model.parameters()) == 4
assert sum(p.data.size for p in model.parameters()) == 13

# modes
model.eval()
assert model.training is False
assert all(layer.training is False for layer in model.layers)
model.train()
assert model.training is True
assert all(layer.training is True for layer in model.layers)
