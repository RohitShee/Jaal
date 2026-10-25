import numpy as np
from jaal.layers import Linear
from jaal.containers import Sequential
from jaal.activations import ReLU, Sigmoid
from jaal.losses import BCELoss
from jaal.optim import SGD

rng = np.random.default_rng(0)
x = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y = np.array([[0.], [1.], [1.], [0.]])

model = Sequential(Linear(2, 8), ReLU(), Linear(8, 1),Sigmoid())
loss_fn = BCELoss()
opt = SGD(model.parameters(), lr=0.5)

for epoch in range(2000):
    opt.zero_grad()
    pred = model.forward(x)
    loss = loss_fn(pred,y)
    model.backward(loss_fn.backward())
    opt.step()
    if epoch % 200 == 0:
        print(epoch, loss)

print(model(x))
