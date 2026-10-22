import numpy as np
from jaal.layers import Linear
from jaal.losses import MSELoss
from jaal.optim import SGD

rng = np.random.default_rng(0)
x = rng.normal(size=(100, 1))
y = 3 * x + 1                      # target: learn y = 3x + 1

model = Linear(1, 1)
loss_fn = MSELoss()
opt = SGD(model.parameters(), lr=0.1)

for epoch in range(100):
    opt.zero_grad()
    pred = model.forward(x)
    loss = loss_fn(pred,y)
    model.backward(loss_fn.backward())
    opt.step()
    if epoch % 10 == 0:
        print(epoch, loss)
print(model.W.data, model.b.data)
