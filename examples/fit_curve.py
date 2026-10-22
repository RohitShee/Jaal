import numpy as np
from jaal.layers import Linear
from jaal.containers import Sequential
from jaal.activations import ReLU
from jaal.losses import MSELoss
from jaal.optim import SGD

rng = np.random.default_rng(0)
x = rng.uniform(-2, 2, size=(200, 1))
y = x**2                    # target: learn y = x^2

model = Sequential(Linear(1, 16), ReLU(), Linear(16, 1))
loss_fn = MSELoss()
opt = SGD(model.parameters(), lr=0.1)

for epoch in range(2000):
    opt.zero_grad()
    pred = model.forward(x)
    loss = loss_fn(pred,y)
    model.backward(loss_fn.backward())
    opt.step()
    if epoch % 100 == 0:
        print(epoch, loss)

