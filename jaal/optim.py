import numpy as np

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

class Adam(Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params, lr)
        self.eps = eps
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.t = 0
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        self.t+=1
        for p,m,v in zip(self.params,self.m,self.v) :
            m *= self.beta1
            m += (1-self.beta1)*p.grad
            v *= self.beta2
            v += (1-self.beta2)*p.grad**2
            m_hat = m/(1-self.beta1**self.t)
            v_hat = v/(1-self.beta2**self.t)
            p.data -= self.lr*m_hat/(np.sqrt(v_hat)+self.eps)

