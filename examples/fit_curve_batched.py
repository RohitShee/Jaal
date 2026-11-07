import numpy as np
from jaal.layers import Linear
from jaal.activations import ReLU
from jaal.containers import Sequential
from jaal.losses import MSELoss
from jaal.optim import Adam
from jaal.data import DataLoader
from jaal.train import fit

rng = np.random.default_rng(0)
x = rng.uniform(-2, 2, size=(1000, 1))
y = x ** 2

model = Sequential(Linear(1, 16), ReLU(), Linear(16, 1))
loader = DataLoader(x, y, batch_size=32, seed=0)

history = fit(model, MSELoss(), Adam(model.parameters(), lr=0.01), loader, epochs=100)
