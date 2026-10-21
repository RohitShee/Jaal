import numpy as np
class Loss:
    def forward(self, pred, target):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    def __call__(self, pred, target):
        return self.forward(pred,target)

class MSELoss(Loss):
    def forward(self, pred, target):
        pred = np.asarray(pred)
        target = np.asarray(target)
        self.pred = pred
        self.target = target
        mseloss = np.mean((pred-target)**2)
        return mseloss

    def backward(self):
        errors = self.pred-self.target
        return 2/self.pred.size*errors
