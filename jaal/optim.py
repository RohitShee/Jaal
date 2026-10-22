class Optimizer:
    def __init__(self, params, lr):
        self.params = list(params)
        self.lr =lr

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        for parameter in self.params:
            parameter.grad.fill(0.0)


class SGD(Optimizer):
    def step(self):
        for parameter in self.params:
            parameter.data  -= self.lr * parameter.grad


